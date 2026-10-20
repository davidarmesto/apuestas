"""Probabilidades de mercados a partir de las medias esperadas de cada estadística."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .modelo import Prediccion

NOMBRES = {
    "goles": "Goles",
    "remates": "Remates",
    "remates_puerta": "Remates a puerta",
    "corners": "Córners",
    "amarillas": "Tarjetas amarillas",
    "rojas": "Tarjetas rojas",
    "tarjetas": "Tarjetas",
    "faltas": "Faltas",
}


@dataclass(frozen=True)
class Seleccion:
    """Una apuesta concreta.

    tipo: '1x2' | 'doble' | 'btts' | 'total' | 'local' | 'visitante' | 'handicap'
    lado: '1','X','2' | '1X','12','X2' | 'si','no' | 'mas','menos' | 'local','visitante'
    linea: para totales y hándicaps (hándicap aplicado al equipo `lado`)
    """

    estadistica: str
    tipo: str
    lado: str
    linea: float | None = None

    def describir(self, local: str = "Local", visitante: str = "Visitante") -> str:
        est = NOMBRES.get(self.estadistica, self.estadistica)
        if self.tipo == "1x2":
            return {"1": f"Gana {local}", "X": "Empate", "2": f"Gana {visitante}"}[self.lado]
        if self.tipo == "doble":
            return {"1X": f"{local} o empate", "12": f"{local} o {visitante}", "X2": f"Empate o {visitante}"}[self.lado]
        if self.tipo == "btts":
            return "Ambos marcan: " + ("Sí" if self.lado == "si" else "No")
        palabra = "Más de" if self.lado == "mas" else "Menos de"
        if self.tipo == "total":
            return f"{est}: {palabra} {self.linea:g}"
        if self.tipo in ("local", "visitante"):
            equipo = local if self.tipo == "local" else visitante
            return f"{est} {equipo}: {palabra} {self.linea:g}"
        if self.tipo == "handicap":
            equipo = local if self.lado == "local" else visitante
            return f"{est}: {equipo} {self.linea:+g} (hándicap)"
        return f"{est} {self.tipo} {self.lado} {self.linea}"


def _pmf(media: float, dispersion: float, maximo: int) -> np.ndarray:
    k = np.arange(maximo + 1)
    if dispersion <= 1.02:
        p = stats.poisson.pmf(k, media)
    else:
        r = media / (dispersion - 1)
        p = stats.nbinom.pmf(k, r, r / (r + media))
    return p / p.sum()


def matriz_conjunta(pred: Prediccion) -> np.ndarray:
    """P[i, j] = P(local = i, visitante = j), suponiendo independencia."""
    sd = math.sqrt(max(pred.local, pred.visitante) * pred.dispersion)
    maximo = int(max(pred.local, pred.visitante) + 10 * sd + 10)
    return np.outer(_pmf(pred.local, pred.dispersion, maximo), _pmf(pred.visitante, pred.dispersion, maximo))


def _mas_menos(dist: np.ndarray, linea: float, lado: str) -> tuple[float, float]:
    """Devuelve (P(ganar), P(nulo)) para un más/menos sobre una distribución 1D."""
    k = np.arange(len(dist))
    nulo = float(dist[k == linea].sum())
    if lado == "mas":
        return float(dist[k > linea].sum()), nulo
    return float(dist[k < linea].sum()), nulo


def probabilidad(pred: Prediccion, sel: Seleccion, matriz: np.ndarray | None = None) -> tuple[float, float]:
    """Probabilidad de ganar y de nulo (devolución) de una selección."""
    m = matriz if matriz is not None else matriz_conjunta(pred)
    n = m.shape[0]
    i, j = np.indices(m.shape)
    if sel.tipo == "1x2":
        mascara = {"1": i > j, "X": i == j, "2": i < j}[sel.lado]
        return float(m[mascara].sum()), 0.0
    if sel.tipo == "doble":
        mascara = {"1X": i >= j, "12": i != j, "X2": i <= j}[sel.lado]
        return float(m[mascara].sum()), 0.0
    if sel.tipo == "btts":
        p = float(m[1:, 1:].sum())
        return (p if sel.lado == "si" else 1 - p), 0.0
    if sel.tipo == "total":
        total = np.bincount((i + j).ravel(), m.ravel(), minlength=2 * n)
        return _mas_menos(total, sel.linea, sel.lado)
    if sel.tipo == "local":
        return _mas_menos(m.sum(axis=1), sel.linea, sel.lado)
    if sel.tipo == "visitante":
        return _mas_menos(m.sum(axis=0), sel.linea, sel.lado)
    if sel.tipo == "handicap":
        dif = (i - j) if sel.lado == "local" else (j - i)
        margen = dif + sel.linea
        return float(m[margen > 0].sum()), float(m[margen == 0].sum())
    raise ValueError(f"Tipo de selección desconocido: {sel.tipo}")


def _lineas_cerca(media: float, n: int = 2) -> list[float]:
    centro = math.floor(media) + 0.5
    return [centro + d for d in range(-n, n + 1) if centro + d > 0]


def mercados_estandar(predicciones: dict[str, Prediccion]) -> list[tuple[Seleccion, float]]:
    """Genera las selecciones típicas con su probabilidad según el modelo."""
    salida: list[tuple[Seleccion, float]] = []
    for est, pred in predicciones.items():
        if est == "rojas":
            continue
        m = matriz_conjunta(pred)
        sels: list[Seleccion] = []
        if est == "goles":
            sels += [Seleccion(est, "1x2", s) for s in ("1", "X", "2")]
            sels += [Seleccion(est, "doble", s) for s in ("1X", "12", "X2")]
            sels += [Seleccion(est, "btts", s) for s in ("si", "no")]
            lineas = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            lineas = _lineas_cerca(pred.local + pred.visitante)
        for linea in lineas:
            sels += [Seleccion(est, "total", lado, linea) for lado in ("mas", "menos")]
        for tipo, media in (("local", pred.local), ("visitante", pred.visitante)):
            for linea in _lineas_cerca(media, 1):
                sels += [Seleccion(est, tipo, lado, linea) for lado in ("mas", "menos")]
        for sel in sels:
            salida.append((sel, probabilidad(pred, sel, m)[0]))
    return salida
