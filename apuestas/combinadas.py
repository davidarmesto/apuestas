"""Combinadas del mismo partido (crear apuesta / bet builder).

Las selecciones de una misma estadística se evalúan sobre su distribución
conjunta (p. ej. "más de 2,5 goles" y "ambos marcan" no son independientes).
Entre estadísticas distintas (goles, córners, tarjetas) se supone independencia.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from .mercados import Seleccion, mascaras, matriz_conjunta, mercados_estandar
from .modelo import Prediccion

AYUDA_PIERNA = (
    "formato 'estadistica tipo lado [linea]', p. ej.: 'goles total mas 2.5', 'goles btts si', "
    "'goles 1x2 1', 'goles doble 12', 'corners total mas 8.5', 'tarjetas total mas 3.5', "
    "'goles local mas 0.5', 'corners handicap local -0.5'"
)


def parsear_pierna(texto: str) -> Seleccion:
    partes = texto.replace(",", ".").split()
    if len(partes) not in (3, 4):
        raise ValueError(f"Pierna mal escrita: {texto!r} ({AYUDA_PIERNA})")
    linea = float(partes[3]) if len(partes) == 4 else None
    lado = partes[2].upper() if partes[1] in ("1x2", "doble") else partes[2]
    return Seleccion(partes[0], partes[1], lado, linea)


def probabilidad_combinada(preds: dict[str, Prediccion], piernas: list[Seleccion]) -> float:
    """Probabilidad de que ganen todas las piernas (sin nulos: usa líneas .5)."""
    por_est: dict[str, list[Seleccion]] = defaultdict(list)
    for s in piernas:
        if s.linea is not None and s.linea == int(s.linea) and s.tipo != "handicap":
            raise ValueError(f"Usa líneas .5 en combinadas (nulo ambiguo): {s}")
        por_est[s.estadistica].append(s)
    p = 1.0
    for est, sels in por_est.items():
        if est not in preds:
            raise ValueError(f"Sin datos de {est}")
        m = matriz_conjunta(preds[est])
        gana = np.ones(m.shape, dtype=bool)
        for s in sels:
            g, nulo = mascaras(s, m.shape)
            gana &= g & ~nulo
        p *= float(m[gana].sum())
    return p


def _compatibles(piernas: tuple[Seleccion, ...]) -> bool:
    claves = [(s.estadistica, s.tipo if s.tipo not in ("1x2", "doble") else "resultado") for s in piernas]
    return len(set(claves)) == len(claves)


def sugerir(preds: dict[str, Prediccion], cuota_objetivo: float, max_piernas: int = 4,
            estadisticas: list[str] | None = None, prob_min_pierna: float = 0.5,
            tolerancia: float = 0.08) -> list[tuple[tuple[Seleccion, ...], float]]:
    """Combinadas cuya cuota justa queda cerca del objetivo."""
    # Piernas que suelen ofrecer las casas en "crear apuesta" y que tengan sentido por sí solas
    candidatas = [
        s for s, p in mercados_estandar(preds)
        if prob_min_pierna <= p <= 0.95
        and s.estadistica in ("goles", "corners", "tarjetas", "remates_puerta")
        and (not estadisticas or s.estadistica in estadisticas)
    ]
    objetivo_p = 1 / cuota_objetivo
    salida = []
    for k in range(2, max_piernas + 1):
        for combo in itertools.combinations(candidatas, k):
            if not _compatibles(combo):
                continue
            p = probabilidad_combinada(preds, list(combo))
            if abs(p - objetivo_p) <= tolerancia * objetivo_p:
                salida.append((combo, p))
    # Primero las de menos piernas (cada pierna añade margen de la casa) y luego las más cercanas al objetivo
    salida.sort(key=lambda x: (len(x[0]), abs(1 / x[1] - cuota_objetivo)))
    return salida
