"""Descarga y limpieza de resultados históricos de football-data.co.uk."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL_BASE = "https://www.football-data.co.uk/mmz4281/{temporada}/{liga}.csv"
DIR_DATOS = Path(__file__).resolve().parent.parent / "datos"

# Código de football-data -> (nombre legible, clave de The Odds API)
LIGAS = {
    "SP1": ("LaLiga", "soccer_spain_la_liga"),
    "SP2": ("LaLiga Hypermotion", "soccer_spain_segunda_division"),
    "E0": ("Premier League", "soccer_epl"),
    "E1": ("Championship", "soccer_efl_champ"),
    "I1": ("Serie A", "soccer_italy_serie_a"),
    "D1": ("Bundesliga", "soccer_germany_bundesliga"),
    "F1": ("Ligue 1", "soccer_france_ligue_one"),
    "N1": ("Eredivisie", "soccer_netherlands_eredivisie"),
    "P1": ("Primeira Liga", "soccer_portugal_primeira_liga"),
}

# Columnas de estadísticas por equipo: nombre interno -> (columna local, columna visitante)
ESTADISTICAS = {
    "goles": ("FTHG", "FTAG"),
    "remates": ("HS", "AS"),
    "remates_puerta": ("HST", "AST"),
    "corners": ("HC", "AC"),
    "amarillas": ("HY", "AY"),
    "rojas": ("HR", "AR"),
    "tarjetas": ("HCards", "ACards"),  # amarillas + rojas, como cuentan la mayoría de casas
    "faltas": ("HF", "AF"),
}


def temporada_actual(hoy: date | None = None) -> str:
    """Código de temporada de football-data, p. ej. '2627' para 2026/27."""
    hoy = hoy or date.today()
    inicio = hoy.year if hoy.month >= 7 else hoy.year - 1
    return f"{inicio % 100:02d}{(inicio + 1) % 100:02d}"


def temporadas_recientes(n: int, hoy: date | None = None) -> list[str]:
    actual = temporada_actual(hoy)
    inicio = 2000 + int(actual[:2])
    return [f"{(a) % 100:02d}{(a + 1) % 100:02d}" for a in range(inicio - n + 1, inicio + 1)]


def descargar(liga: str, temporada: str, forzar: bool = False, max_horas: float = 12) -> Path:
    """Descarga el CSV de una liga y temporada (con caché en disco)."""
    DIR_DATOS.mkdir(exist_ok=True)
    destino = DIR_DATOS / f"{liga}_{temporada}.csv"
    es_actual = temporada == temporada_actual()
    if destino.exists() and not forzar:
        antiguedad_h = (time.time() - destino.stat().st_mtime) / 3600
        # Las temporadas cerradas no cambian; la actual se refresca cada `max_horas`.
        if not es_actual or antiguedad_h < max_horas:
            return destino
    resp = requests.get(
        URL_BASE.format(temporada=temporada, liga=liga),
        headers={"User-Agent": "Mozilla/5.0 (recomendador-apuestas)"},
        timeout=30,
    )
    resp.raise_for_status()
    destino.write_bytes(resp.content)
    return destino


def cargar(liga: str, temporadas: list[str]) -> pd.DataFrame:
    """Carga y concatena varias temporadas en un DataFrame limpio y ordenado por fecha."""
    partes = []
    for temporada in temporadas:
        try:
            ruta = descargar(liga, temporada)
        except requests.HTTPError as e:
            print(f"Aviso: no se pudo descargar {liga} {temporada}: {e}")
            continue
        df = pd.read_csv(ruta, encoding="utf-8-sig", on_bad_lines="skip")
        df["Temporada"] = temporada
        partes.append(df)
    if not partes:
        raise RuntimeError(f"No hay datos para {liga}")
    df = pd.concat(partes, ignore_index=True)
    return limpiar(df)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Fecha"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha"])
    for local, visitante in ESTADISTICAS.values():
        for col in (local, visitante):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"HY", "AY", "HR", "AR"} <= set(df.columns):
        df["HCards"] = df["HY"] + df["HR"]
        df["ACards"] = df["AY"] + df["AR"]
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    return df.sort_values("Fecha").reset_index(drop=True)
