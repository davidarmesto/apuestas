"""Cliente de The Odds API (https://the-odds-api.com) con caché en disco.

Cada petición gasta créditos (mercados × regiones), así que las respuestas se
guardan en `datos/cache_odds/` y se reutilizan durante unas horas.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

from .datos import DIR_DATOS
from .mercados import Seleccion

URL = "https://api.the-odds-api.com/v4"
DIR_CACHE = DIR_DATOS / "cache_odds"

MERCADOS_PRINCIPALES = "h2h,totals"
# Mercados adicionales de fútbol; se piden partido a partido (endpoint de evento).
MERCADOS_EXTRA = "btts,double_chance,alternate_totals,team_totals,alternate_totals_corners,alternate_spreads_corners,alternate_totals_cards"
# Estadísticas de jugadores (sobre todo casas de EE. UU.: hay que pedir la región "us")
MERCADOS_JUGADOR = "player_shots_on_target,player_shots,player_goal_scorer_anytime"
_ESTADISTICA_JUGADOR = {"player_shots_on_target": "remates_puerta", "player_shots": "remates", "player_goal_scorer_anytime": "goles"}


@dataclass
class Cuota:
    seleccion: Seleccion
    precio: float
    casa: str


@dataclass
class Evento:
    id: str
    inicio: str
    local: str
    visitante: str
    cuotas: list[Cuota] = field(default_factory=list)

    def consenso(self) -> dict[Seleccion, float]:
        """Probabilidad implícita mediana entre todas las casas para cada selección (con margen)."""
        por_sel: dict[Seleccion, list[float]] = {}
        for c in self.cuotas:
            por_sel.setdefault(c.seleccion, []).append(1 / c.precio)
        return {s: float(np.median(v)) for s, v in por_sel.items()}

    def mejores_cuotas(self) -> list[Cuota]:
        """La mejor cuota disponible para cada selección."""
        mejores: dict[Seleccion, Cuota] = {}
        for c in self.cuotas:
            if c.seleccion not in mejores or c.precio > mejores[c.seleccion].precio:
                mejores[c.seleccion] = c
        return list(mejores.values())


def clave_api() -> str:
    clave = os.environ.get("ODDS_API_KEY")
    env = Path(__file__).resolve().parent.parent / ".env"
    if not clave and env.exists():
        for linea in env.read_text().splitlines():
            if linea.strip().startswith("ODDS_API_KEY="):
                clave = linea.split("=", 1)[1].strip().strip('"').strip("'")
    if not clave:
        raise RuntimeError(
            "Falta ODDS_API_KEY. Crea una clave gratis en https://the-odds-api.com y "
            "ponla en el archivo .env (ODDS_API_KEY=...)"
        )
    return clave


def _get(ruta: str, params: dict, max_horas: float) -> object:
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    firma = hashlib.sha1(json.dumps([ruta, params], sort_keys=True).encode()).hexdigest()[:16]
    cache = DIR_CACHE / f"{firma}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < max_horas:
        return json.loads(cache.read_text())
    resp = requests.get(f"{URL}{ruta}", params={**params, "apiKey": clave_api()}, timeout=30)
    if resp.status_code == 422:
        raise ValueError(f"The Odds API rechazó la petición: {resp.text}")
    resp.raise_for_status()
    restantes = resp.headers.get("x-requests-remaining")
    if restantes is not None:
        print(f"  (The Odds API: quedan {restantes} créditos este mes)")
    datos = resp.json()
    cache.write_text(json.dumps(datos))
    return datos


def eventos(sport: str, regiones: str = "eu", max_horas: float = 3) -> list[Evento]:
    datos = _get(
        f"/sports/{sport}/odds",
        {"regions": regiones, "markets": MERCADOS_PRINCIPALES, "oddsFormat": "decimal"},
        max_horas,
    )
    return [parsear_evento(e) for e in datos]


def cuotas_extra(sport: str, evento: Evento, regiones: str = "eu", mercados: str = MERCADOS_EXTRA, max_horas: float = 3) -> None:
    """Añade al evento las cuotas de mercados adicionales (córners, tarjetas, ambos marcan...)."""
    try:
        datos = _get(
            f"/sports/{sport}/events/{evento.id}/odds",
            {"regions": regiones, "markets": mercados, "oddsFormat": "decimal"},
            max_horas,
        )
    except ValueError as e:
        print(f"  Aviso: {e}")
        return
    evento.cuotas += parsear_evento(datos).cuotas


def parsear_evento(e: dict) -> Evento:
    ev = Evento(e["id"], e.get("commence_time", ""), e["home_team"], e["away_team"])
    for casa in e.get("bookmakers", []):
        for mercado in casa.get("markets", []):
            for o in mercado.get("outcomes", []):
                sel = _a_seleccion(mercado["key"], o, ev.local, ev.visitante)
                if sel is not None:
                    ev.cuotas.append(Cuota(sel, float(o["price"]), casa.get("title", casa.get("key", "?"))))
    return ev


def _linea_simple(punto) -> bool:
    """Líneas enteras o de medio gol; las asiáticas de cuarto (2.25, 2.75) no se modelan."""
    return punto is not None and float(punto) * 2 == int(float(punto) * 2)


def _lado_total(nombre: str) -> str | None:
    return {"over": "mas", "under": "menos"}.get(nombre.lower())


def _a_seleccion(mercado: str, o: dict, local: str, visitante: str) -> Seleccion | None:
    nombre, punto = o.get("name", ""), o.get("point")
    if mercado in _ESTADISTICA_JUGADOR and o.get("description"):
        if mercado == "player_goal_scorer_anytime":
            lado, punto = ("mas" if nombre.lower() == "yes" else None), 0.5
        else:
            lado = _lado_total(nombre)
        if lado and punto is not None:
            return Seleccion(_ESTADISTICA_JUGADOR[mercado], "jugador", lado, float(punto), jugador=o["description"])
        return None
    if mercado in ("h2h", "h2h_3_way"):
        lado = "1" if nombre == local else "2" if nombre == visitante else "X" if nombre.lower() == "draw" else None
        return Seleccion("goles", "1x2", lado) if lado else None
    if mercado == "double_chance":
        n = nombre.lower()
        tiene_l, tiene_v, tiene_x = local.lower() in n, visitante.lower() in n, "draw" in n
        lado = "1X" if tiene_l and tiene_x else "X2" if tiene_v and tiene_x else "12" if tiene_l and tiene_v else None
        return Seleccion("goles", "doble", lado) if lado else None
    if mercado == "btts":
        lado = {"yes": "si", "no": "no"}.get(nombre.lower())
        return Seleccion("goles", "btts", lado) if lado else None
    estadistica = {
        "totals": "goles",
        "alternate_totals": "goles",
        "alternate_totals_corners": "corners",
        "alternate_totals_cards": "tarjetas",
    }.get(mercado)
    if estadistica and _linea_simple(punto) and _lado_total(nombre):
        return Seleccion(estadistica, "total", _lado_total(nombre), float(punto))
    if mercado == "team_totals" and _linea_simple(punto) and _lado_total(nombre):
        equipo = o.get("description", "")
        tipo = "local" if equipo == local else "visitante" if equipo == visitante else None
        return Seleccion("goles", tipo, _lado_total(nombre), float(punto)) if tipo else None
    if mercado == "alternate_spreads_corners" and _linea_simple(punto):
        lado = "local" if nombre == local else "visitante" if nombre == visitante else None
        if lado:
            return Seleccion("corners", "handicap", lado, float(punto))
    return None
