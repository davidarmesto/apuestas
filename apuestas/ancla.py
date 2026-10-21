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
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from .mercados import Seleccion, matriz_conjunta, probabilidad
from .modelo import Prediccion
from .odds_api import Evento
from .valor import sin_margen


# Cuánto sube cada estadística (en escala logarítmica) por cada unidad que suben los goles esperados
# de un equipo: pendiente entre la fuerza de goles y la de remates a puerta de los equipos (≈0,6 en
# LaLiga, Premier, Bundesliga y Serie A).
MARGEN_MINIMO_JUGADOR = 0.05

ELASTICIDAD_GOLES = {"remates_puerta": 0.6, "remates": 0.4}


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
        dos = [Seleccion("goles", "ganador", x) for x in "12"]
        if all(d in precios for d in dos):
            # Ganador sin empate (baloncesto, tenis, hockey con prórroga...): mercado de dos
            for d, p in zip(dos, sin_margen([precios[d] for d in dos])):
                recogidas[d].append(p)
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
            return replace(pred, local=pred.local * math.exp(s + d), visitante=pred.visitante * math.exp(s - d))

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
    # Las estadísticas sin cuotas siguen a los goles: si el mercado espera más goles de un equipo,
    # también más remates a puerta
    if "goles" in nuevas and nuevas["goles"] is not preds["goles"]:
        g0, g1 = preds["goles"], nuevas["goles"]
        for est, beta in ELASTICIDAD_GOLES.items():
            if est in preds and not any(s.estadistica == est for s in justas):
                p = preds[est]
                nuevas[est] = replace(p, local=p.local * (g1.local / g0.local) ** beta,
                                      visitante=p.visitante * (g1.visitante / g0.visitante) ** beta)
                resumen.append(f"{est:<15} modelo {p.local:.2f}-{p.visitante:.2f} -> {nuevas[est].local:.2f}-"
                               f"{nuevas[est].visitante:.2f}  (sin cuotas: siguen a los goles del mercado)")
    return nuevas, resumen


def lambdas_jugadores(ev: Evento) -> dict[tuple[str, str], float]:
    """Media de cada estadística de jugador que reproduce las cuotas del mercado.

    Normalmente solo se publica el "más de", así que se ajusta a la vez la media
    (Poisson) y el margen de la casa. El margen no puede bajar del 5 %: en estos mercados de
    un solo lado las casas siempre se quedan al menos eso, y sin ese suelo el ajuste podía
    tomar como justa la probabilidad con margen.
    """
    from scipy import stats

    precios: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for c in ev.cuotas:
        s = c.seleccion
        if s.tipo == "jugador" and s.lado == "mas":
            precios[(s.jugador, s.estadistica)].append((s.linea, 1 / c.precio))
    salida = {}
    for clave, puntos in precios.items():
        lineas, implicitas = np.array([p[0] for p in puntos]), np.array([p[1] for p in puntos])

        def error(x):
            lam, margen = x
            p = stats.poisson.sf(np.floor(lineas), lam)
            return float(np.sum((np.minimum(0.995, p * (1 + margen)) - implicitas) ** 2))

        mejor = min(
            (minimize(error, [l0, 0.07], method="L-BFGS-B", bounds=[(0.01, 15), (MARGEN_MINIMO_JUGADOR, 0.15)])
             for l0 in (0.5, 1.5, 3.0)),
            key=lambda r: r.fun,
        )
        salida[clave] = float(mejor.x[0])
    return salida
