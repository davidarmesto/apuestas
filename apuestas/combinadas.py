"""Combinadas del mismo partido (crear apuesta / bet builder / MyMatch).

- Las selecciones de una misma estadística se evalúan juntas sobre su
  distribución conjunta exacta (p. ej. "más de 2,5 goles" y "ambos marcan").
- Entre estadísticas distintas (goles, córners, tarjetas, jugadores) se usa una
  cópula gaussiana con las correlaciones medidas en los datos históricos: por
  ejemplo, goles y remates a puerta del mismo equipo (+0,5) o goles y córners
  propios (−0,1).
- Un jugador se relaciona con su equipo según la parte de la estadística del
  equipo que suele acaparar (si hace el 40 % de los remates a puerta, carga √0,4).
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from .mercados import Seleccion, mascaras, matriz_conjunta, mercados_estandar
from .modelo import Prediccion
from .nombres import normalizar

AYUDA_PIERNA = (
    "formato 'estadistica tipo lado [linea]', p. ej.: 'goles total mas 2.5', 'goles btts si', "
    "'goles 1x2 1', 'goles doble 12', 'corners total mas 8.5', 'tarjetas total mas 3.5', "
    "'goles local mas 0.5', 'corners handicap local -0.5'; jugadores: "
    "'jugador local remates_puerta mas 1.5 Dembélé' (equipo local/visitante y el nombre al final)"
)

# Dirección de cada tipo de selección sobre (valor local, valor visitante)
_DIRECCION = {
    ("total", "mas"): (1, 1), ("total", "menos"): (-1, -1),
    ("local", "mas"): (1, 0), ("local", "menos"): (-1, 0),
    ("visitante", "mas"): (0, 1), ("visitante", "menos"): (0, -1),
    ("handicap", "local"): (1, -1), ("handicap", "visitante"): (-1, 1),
    ("1x2", "1"): (1, -1), ("1x2", "2"): (-1, 1), ("1x2", "X"): (0, 0),
    ("doble", "1X"): (0.5, -0.5), ("doble", "X2"): (-0.5, 0.5), ("doble", "12"): (0, 0),
    ("btts", "si"): (0.7, 0.7), ("btts", "no"): (-0.7, -0.7),
}


def parsear_pierna(texto: str) -> Seleccion:
    partes = texto.replace(",", ".").split()
    if partes and partes[0] == "jugador":
        if len(partes) < 6 or partes[1] not in ("local", "visitante"):
            raise ValueError(f"Pierna de jugador mal escrita: {texto!r} ({AYUDA_PIERNA})")
        return Seleccion(partes[2], "jugador", partes[3], float(partes[4]),
                         jugador=" ".join(partes[5:]), equipo=partes[1])
    if len(partes) not in (3, 4):
        raise ValueError(f"Pierna mal escrita: {texto!r} ({AYUDA_PIERNA})")
    linea = float(partes[3]) if len(partes) == 4 else None
    lado = partes[2].upper() if partes[1] in ("1x2", "doble") else partes[2]
    return Seleccion(partes[0], partes[1], lado, linea)


def buscar_jugador(nombre: str, jugadores: dict[tuple[str, str], float], estadistica: str) -> tuple[str, float] | None:
    """Encuentra al jugador en los mercados aunque se escriba solo el apellido o sin tildes."""
    n = normalizar(nombre)
    candidatos = [(j, lam) for (j, est), lam in jugadores.items()
                  if est == estadistica and n in normalizar(j)]
    return candidatos[0] if len(candidatos) == 1 else None


def _p_jugador(lam: float, s: Seleccion, lleva: int = 0) -> float:
    mas = float(stats.poisson.sf(math.floor(s.linea) - lleva, lam))
    if s.lado == "mas":
        return mas
    nulo = float(stats.poisson.pmf(s.linea - lleva, lam)) if s.linea == int(s.linea) else 0.0
    return 1 - mas - nulo


def probabilidad_combinada(preds: dict[str, Prediccion], piernas: list[Seleccion],
                           correlaciones: pd.DataFrame | None = None,
                           jugadores: dict[tuple[str, str], float] | None = None,
                           marcador: dict[str, tuple[int, int]] | None = None,
                           lleva: dict[tuple[str, str], int] | None = None) -> float:
    """Probabilidad de que ganen todas las piernas (sin nulos: usa líneas .5).

    En directo, `preds` y `jugadores` son lo que queda por jugarse, `marcador` lo que ya
    llevan los equipos en cada estadística y `lleva` lo que lleva cada jugador.
    """
    marcador, lleva = marcador or {}, lleva or {}
    por_est: dict[str, list[Seleccion]] = defaultdict(list)
    grupos: list[tuple[float, dict[str, float], float]] = []  # (probabilidad, cargas, ruido propio)
    for s in piernas:
        if s.linea is not None and s.linea == int(s.linea) and s.tipo not in ("handicap", "jugador"):
            raise ValueError(f"Usa líneas .5 en combinadas (nulo ambiguo): {s}")
        if s.tipo == "jugador":
            encontrado = buscar_jugador(s.jugador, jugadores or {}, s.estadistica)
            if encontrado is None:
                raise ValueError(f"No hay cuotas de {s.jugador} ({s.estadistica}); usa --anclar --jugadores")
            _, lam = encontrado
            lado = "l" if s.equipo == "local" else "v"
            equipo = preds.get(s.estadistica)
            media_equipo = (equipo.local if lado == "l" else equipo.visitante) if equipo else None
            carga = math.sqrt(min(0.8, lam / media_equipo)) if media_equipo else 0.0
            signo = 1 if s.lado == "mas" else -1
            ya = _lleva_jugador(lleva, s)
            grupos.append((_p_jugador(lam, s, ya), {f"{s.estadistica}_{lado}": signo * carga}, 1 - carga**2))
        else:
            por_est[s.estadistica].append(s)
    for est, sels in por_est.items():
        if est not in preds:
            raise ValueError(f"Sin datos de {est}")
        m = _desplazar(matriz_conjunta(preds[est]), marcador.get(est, (0, 0)))
        gana = np.ones(m.shape, dtype=bool)
        for s in sels:
            g, nulo = mascaras(s, m.shape)
            gana &= g & ~nulo
        dl = sum(_DIRECCION.get((s.tipo, s.lado), (0, 0))[0] for s in sels)
        dv = sum(_DIRECCION.get((s.tipo, s.lado), (0, 0))[1] for s in sels)
        grupos.append((float(m[gana].sum()), {f"{est}_l": dl, f"{est}_v": dv}, 0.0))

    # Las piernas ya ganadas no aportan nada (y romperían la cópula)
    grupos = [g for g in grupos if g[0] < 1 - 1e-12] or [(1.0, {}, 0.0)]
    probs = np.array([g[0] for g in grupos])
    if len(grupos) == 1 or correlaciones is None or np.any(probs <= 0):
        return float(np.prod(probs))
    return _copula(grupos, correlaciones)


def _desplazar(m: np.ndarray, actual: tuple[int, int]) -> np.ndarray:
    """Pasa de la distribución de lo que falta a la del total final sumando lo que ya ha pasado."""
    cl, cv = actual
    if not cl and not cv:
        return m
    salida = np.zeros((m.shape[0] + cl, m.shape[1] + cv))
    salida[cl:, cv:] = m
    return salida


def _lleva_jugador(lleva: dict[tuple[str, str], int], s: Seleccion) -> int:
    n = normalizar(s.jugador)
    return sum(v for (j, est), v in lleva.items() if est == s.estadistica and (n in normalizar(j) or normalizar(j) in n))


def _copula(grupos, correlaciones: pd.DataFrame) -> float:
    idx = {v: k for k, v in enumerate(correlaciones.columns)}
    c = correlaciones.to_numpy()
    cargas = np.zeros((len(grupos), len(idx)))
    ruido = np.array([g[2] for g in grupos])
    for g, (_, carga, _) in enumerate(grupos):
        for var, peso in carga.items():
            if var in idx:
                cargas[g, idx[var]] = peso
    cov = cargas @ c @ cargas.T + np.diag(ruido)
    var = np.diag(cov).copy()
    sin_relacion = var < 1e-9  # p. ej. solo "empate": no se relaciona con nada
    cov[sin_relacion, :] = 0
    cov[:, sin_relacion] = 0
    var[sin_relacion] = 1
    cov[np.diag_indices_from(cov)] = var
    d = np.sqrt(var)
    r = cov / np.outer(d, d) + np.eye(len(var)) * 1e-9
    probs = np.array([g[0] for g in grupos])
    return float(stats.multivariate_normal(mean=np.zeros(len(r)), cov=r).cdf(stats.norm.ppf(probs)))


def _compatibles(piernas: tuple[Seleccion, ...]) -> bool:
    claves = [(s.estadistica, s.tipo if s.tipo not in ("1x2", "doble") else "resultado") for s in piernas]
    return len(set(claves)) == len(claves)


def sugerir(preds: dict[str, Prediccion], cuota_objetivo: float, max_piernas: int = 4,
            estadisticas: list[str] | None = None, prob_min_pierna: float = 0.5,
            tolerancia: float = 0.08, correlaciones: pd.DataFrame | None = None
            ) -> list[tuple[tuple[Seleccion, ...], float]]:
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
            # Filtro rápido con independencia; la correlación rara vez mueve la probabilidad más de un 30 %
            p0 = probabilidad_combinada(preds, list(combo))
            if abs(p0 - objetivo_p) > 0.3 * objetivo_p:
                continue
            p = probabilidad_combinada(preds, list(combo), correlaciones)
            if abs(p - objetivo_p) <= tolerancia * objetivo_p:
                salida.append((combo, p))
    # Primero las de menos piernas (cada pierna añade margen de la casa) y luego las más cercanas al objetivo
    salida.sort(key=lambda x: (len(x[0]), abs(1 / x[1] - cuota_objetivo)))
    return salida
