"""Anclar el modelo al mercado.

Las casas fijan bien las líneas principales (ganador, total de goles, línea de
córners y de tarjetas), pero no tienen una forma coherente de calcular la
probabilidad de que pasen varias cosas a la vez. Aquí se ajustan las medias
del modelo para que reproduzcan las probabilidades sin margen del mercado, y
luego el modelo se usa para calcular combinadas de forma consistente.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from .mercados import Seleccion, matriz_conjunta, probabilidad
from .modelo import Prediccion
from .odds_api import Evento
from .valor import sin_margen


def _opuesta(s: Seleccion) -> Seleccion | None:
    if s.tipo in ("total", "local", "visitante"):
        return Seleccion(s.estadistica, s.tipo, "menos" if s.lado == "mas" else "mas", s.linea)
    if s.tipo == "btts":
        return Seleccion(s.estadistica, "btts", "no" if s.lado == "si" else "si")
    if s.tipo == "handicap":
        return Seleccion(s.estadistica, "handicap", "visitante" if s.lado == "local" else "local", -s.linea)
    return None


def _sin_nulo(s: Seleccion) -> bool:
    return s.linea is None or s.linea != int(s.linea)


def probabilidades_justas(ev: Evento) -> dict[Seleccion, tuple[float, int]]:
    """Probabilidad sin margen (mediana entre casas) y nº de casas, para cada selección."""
    por_casa: dict[str, dict[Seleccion, float]] = defaultdict(dict)
    for c in ev.cuotas:
        por_casa[c.casa][c.seleccion] = c.precio
    recogidas: dict[Seleccion, list[float]] = defaultdict(list)
    for precios in por_casa.values():
        tres = [Seleccion("goles", "1x2", x) for x in "1X2"]
        if all(t in precios for t in tres):
            for t, p in zip(tres, sin_margen([precios[t] for t in tres])):
                recogidas[t].append(p)
        for s, precio in precios.items():
            o = _opuesta(s)
            if o is not None and o in precios and _sin_nulo(s):
                recogidas[s].append(sin_margen([precio, precios[o]])[0])
    return {s: (float(np.median(v)), len(v)) for s, v in recogidas.items()}


def anclar(preds: dict[str, Prediccion], justas: dict[Seleccion, tuple[float, int]]) -> tuple[dict[str, Prediccion], list[str]]:
    """Ajusta cada estadística para que el modelo reproduzca el mercado. Devuelve las nuevas predicciones y un resumen."""
    nuevas, resumen = dict(preds), []
    for est, pred in preds.items():
        objetivos = [(s, p, n) for s, (p, n) in justas.items() if s.estadistica == est]
        if not objetivos:
            continue
        direccional = any(s.tipo in ("1x2", "handicap", "local", "visitante") for s, _, _ in objetivos)

        def ajustada(x) -> Prediccion:
            s, d = x[0], (x[1] if direccional else 0.0)
            return Prediccion(pred.local * math.exp(s + d), pred.visitante * math.exp(s - d), pred.dispersion)

        def error(x):
            m = matriz_conjunta(ajustada(x))
            # Las líneas con más casas pesan más (más fiables)
            return sum(math.sqrt(n) * (probabilidad(ajustada(x), s, m)[0] - p) ** 2 for s, p, n in objetivos)

        x0 = [0.0, 0.0] if direccional else [0.0]
        res = minimize(error, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10})
        nueva = ajustada(res.x)
        nuevas[est] = nueva
        rms = math.sqrt(error(res.x) / sum(math.sqrt(n) for _, _, n in objetivos))
        resumen.append(
            f"{est:<15} modelo {pred.local:.2f}-{pred.visitante:.2f} -> mercado {nueva.local:.2f}-{nueva.visitante:.2f}"
            f"  (total {pred.local + pred.visitante:.2f} -> {nueva.local + nueva.visitante:.2f}; "
            f"{len(objetivos)} líneas, error medio {rms:.1%})"
        )
    return nuevas, resumen
