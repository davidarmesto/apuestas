"""Cálculo de valor esperado y tamaño de apuesta (Kelly fraccionado)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    ventaja_minima: float = 0.05  # valor esperado mínimo (5 %)
    cuota_maxima: float = 6.0  # por encima, el modelo es poco fiable
    probabilidad_minima: float = 0.15
    fraccion_kelly: float = 0.25
    apuesta_maxima: float = 0.03  # nunca más del 3 % del bankroll en una apuesta


def valor_esperado(p_ganar: float, cuota: float, p_nulo: float = 0.0) -> float:
    """Beneficio esperado por unidad apostada."""
    return p_ganar * cuota + p_nulo - 1


def kelly(p_ganar: float, cuota: float, p_nulo: float = 0.0) -> float:
    """Fracción de Kelly completa (0 si no hay valor). El nulo se trata como devolución."""
    b = cuota - 1
    p_perder = 1 - p_ganar - p_nulo
    if b <= 0:
        return 0.0
    return max(0.0, (p_ganar * b - p_perder) / b)


def stake(p_ganar: float, cuota: float, p_nulo: float, cfg: Config) -> float:
    return min(cfg.apuesta_maxima, cfg.fraccion_kelly * kelly(p_ganar, cuota, p_nulo))


def sin_margen(cuotas: list[float]) -> list[float]:
    """Probabilidades justas de un mercado quitando el margen de la casa (método proporcional)."""
    inversas = [1 / c for c in cuotas]
    total = sum(inversas)
    return [x / total for x in inversas]


def es_recomendable(p_ganar: float, cuota: float, p_nulo: float, cfg: Config) -> bool:
    return (
        cuota <= cfg.cuota_maxima
        and p_ganar >= cfg.probabilidad_minima
        and valor_esperado(p_ganar, cuota, p_nulo) >= cfg.ventaja_minima
    )
