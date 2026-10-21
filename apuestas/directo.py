"""Apuestas en directo: lo que falta por jugarse de un partido ya empezado.

Las medias de cada estadística se reparten a lo largo del partido con una
intensidad que crece con el minuto (en la segunda parte hay más goles y, sobre
todo, más tarjetas). A la parte que queda se le suma lo que ya ha pasado.
"""

from __future__ import annotations

from dataclasses import replace

from .modelo import Prediccion

DURACION = 95  # minutos efectivos contando el añadido de las dos partes
# Cuánto más intenso es el final que el principio (0 = ritmo constante). Aproximado a partir de
# cómo se reparten estas estadísticas entre primera y segunda parte en las grandes ligas.
AUMENTO = {"goles": 0.35, "tarjetas": 1.2, "amarillas": 1.2, "corners": 0.15, "remates": 0.1, "remates_puerta": 0.1}


def _acumulado(t: float, a: float) -> float:
    """Parte de la estadística que se espera hasta el minuto efectivo t (de 0 a 1)."""
    t = min(max(t, 0.0), DURACION)
    return (t + a * t * t / (2 * DURACION)) / (DURACION + a * DURACION / 2)


def minuto_efectivo(minuto: float) -> float:
    # En la segunda parte ya se ha jugado el añadido de la primera (~2 min)
    return minuto + 2 if minuto > 45 else minuto


def fraccion_restante(estadistica: str, minuto: float) -> float:
    a = AUMENTO.get(estadistica, 0.2)
    return 1 - _acumulado(minuto_efectivo(minuto), a)


def restante(pred: Prediccion, fraccion: float) -> Prediccion:
    """Distribución de lo que queda por pasar.

    Con la binomial negativa (Poisson con media aleatoria del partido),
    var = μ + μ²/r, así que al quedar una fracción f la dispersión pasa a 1 + (φ − 1)·f.
    """
    return replace(pred, local=pred.local * fraccion, visitante=pred.visitante * fraccion,
                   dispersion=1 + (pred.dispersion - 1) * fraccion)


def predicciones_restantes(preds: dict[str, Prediccion], minuto: float) -> dict[str, Prediccion]:
    return {est: restante(p, fraccion_restante(est, minuto)) for est, p in preds.items()}


def valor_apuesta(p: float, cobro: float) -> float:
    """Lo que vale ahora mismo una apuesta que paga `cobro` si gana."""
    return p * cobro
