"""Buscar casas que pagan más de lo justo según el propio mercado.

La probabilidad justa de cada selección es la mediana, entre todas las casas, de
su probabilidad sin margen. Una casa que paga por encima de esa probabilidad
ofrece valor frente al consenso: es la ventaja más fiable que hay, porque no
depende de que el modelo acierte (solo de que la mayoría de casas acierte).
"""

from __future__ import annotations

from dataclasses import dataclass

from .ancla import probabilidades_justas
from .mercados import Seleccion
from .odds_api import Evento


# Las bolsas de apuestas cobran comisión sobre la ganancia
COMISION = {"Betfair": 0.05, "Matchbook": 0.02, "Smarkets": 0.02}


def cuota_neta(cuota: float, casa: str) -> float:
    return 1 + (cuota - 1) * (1 - COMISION.get(casa, 0.0))


@dataclass
class Oportunidad:
    evento: Evento
    seleccion: Seleccion
    casa: str
    cuota: float
    prob_justa: float
    n_casas: int

    @property
    def valor(self) -> float:
        return self.prob_justa * cuota_neta(self.cuota, self.casa) - 1

    @property
    def cuota_justa(self) -> float:
        return 1 / self.prob_justa


def _jugadores_justos(ev: Evento, min_casas: int) -> dict[Seleccion, tuple[float, int]]:
    """Las cuotas de jugadores suelen ser solo 'más de': se quita el margen de cada casa
    con su propia media del mercado (la misma línea en todas las casas)."""
    from collections import defaultdict

    import numpy as np

    por_sel: dict[Seleccion, list[float]] = defaultdict(list)
    for c in ev.cuotas:
        if c.seleccion.tipo == "jugador":
            por_sel[c.seleccion].append(1 / c.precio)
    # Margen típico de estos mercados (~7 %): la probabilidad justa queda por debajo de la implícita
    return {s: (float(np.median(v)) / 1.07, len(v)) for s, v in por_sel.items() if len(v) >= min_casas}


def oportunidades(ev: Evento, min_casas: int = 5, casas: list[str] | None = None,
                  valor_minimo: float = 0.03, cuota_maxima: float = 8.0, jugadores: bool = False) -> list[Oportunidad]:
    justas = {s: v for s, v in probabilidades_justas(ev).items() if v[1] >= min_casas}
    if jugadores:
        justas.update(_jugadores_justos(ev, min_casas))
    salida = []
    for c in ev.cuotas:
        if c.seleccion not in justas or c.precio > cuota_maxima:
            continue
        if casas and not any(k.lower() in c.casa.lower() for k in casas):
            continue
        p, n = justas[c.seleccion]
        o = Oportunidad(ev, c.seleccion, c.casa, c.precio, p, n)
        if o.valor >= valor_minimo:
            salida.append(o)
    # Una por selección y casa (a veces la misma casa sale en varias regiones)
    unicas = {(o.seleccion, o.casa): o for o in sorted(salida, key=lambda o: o.valor)}
    return sorted(unicas.values(), key=lambda o: -o.valor)
