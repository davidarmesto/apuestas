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


def mascaras(sel: Seleccion, forma: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Casillas (goles local, goles visitante) en que la selección gana y en que es nula."""
    i, j = np.indices(forma)
    nada = np.zeros(forma, dtype=bool)
    if sel.tipo == "1x2":
        return {"1": i > j, "X": i == j, "2": i < j}[sel.lado], nada
    if sel.tipo == "doble":
        return {"1X": i >= j, "12": i != j, "X2": i <= j}[sel.lado], nada
    if sel.tipo == "btts":
        ambos = (i > 0) & (j > 0)
        return (ambos if sel.lado == "si" else ~ambos), nada
    if sel.tipo in ("total", "local", "visitante"):
        valor = {"total": i + j, "local": i, "visitante": j}[sel.tipo]
        gana = valor > sel.linea if sel.lado == "mas" else valor < sel.linea
        return gana, valor == sel.linea
    if sel.tipo == "handicap":
        margen = ((i - j) if sel.lado == "local" else (j - i)) + sel.linea
        return margen > 0, margen == 0
    raise ValueError(f"Tipo de selección desconocido: {sel.tipo}")


def probabilidad(pred: Prediccion, sel: Seleccion, matriz: np.ndarray | None = None) -> tuple[float, float]:
    """Probabilidad de ganar y de nulo (devolución) de una selección."""
    m = matriz if matriz is not None else matriz_conjunta(pred)
    gana, nulo = mascaras(sel, m.shape)
    return float(m[gana].sum()), float(m[nulo].sum())


def calibrar_goles(pred: Prediccion, c1: float, cx: float, c2: float) -> Prediccion:
    """Ajusta las medias de goles para reproducir las probabilidades 1X2 del mercado (sin margen).

    Se escalan ambas medias por e^(s±d): `d` mueve la superioridad y `s` el total de goles.
    """
    from scipy.optimize import minimize

    from .valor import sin_margen

    objetivo_p = np.array(sin_margen([c1, cx, c2]))

    def ajustada(x) -> Prediccion:
        s, d = x
        return Prediccion(pred.local * math.exp(s + d), pred.visitante * math.exp(s - d), pred.dispersion)

    def error(x):
        m = matriz_conjunta(ajustada(x))
        i, j = np.indices(m.shape)
        pm = np.array([m[i > j].sum(), m[i == j].sum(), m[i < j].sum()])
        return float(np.sum((pm - objetivo_p) ** 2))

    return ajustada(minimize(error, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-12}).x)


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
