"""Emparejar nombres de equipos entre fuentes (football-data vs. casas de apuestas)."""

from __future__ import annotations

import difflib
import re
import unicodedata

# Nombre normalizado de la casa -> nombre normalizado en football-data
ALIAS = {
    "atletico madrid": "ath madrid",
    "athletic bilbao": "ath bilbao",
    "athletic club": "ath bilbao",
    "rayo vallecano": "vallecano",
    "celta vigo": "celta",
    "espanyol": "espanol",
    "deportivo alaves": "alaves",
    "deportivo la coruna": "la coruna",
    "sporting gijon": "sp gijon",
    "racing santander": "santander",
    "manchester united": "man united",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
    "nottingham forest": "nott m forest",
    "newcastle united": "newcastle",
    "brighton and hove albion": "brighton",
    "west ham united": "west ham",
    "leeds united": "leeds",
    "sheffield united": "sheffield united",
    "inter milan": "inter",
    "internazionale": "inter",
    "ac milan": "milan",
    "as roma": "roma",
    "bayern munich": "bayern munich",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "m'gladbach",
    "bayer leverkusen": "leverkusen",
    "eintracht frankfurt": "ein frankfurt",
    "paris saint germain": "paris sg",
    "psg": "paris sg",
    "olympique marseille": "marseille",
    "olympique lyonnais": "lyon",
}

_RUIDO = {"fc", "cf", "cd", "ud", "sd", "rc", "rcd", "sc", "afc", "ac", "club", "de", "real", "calcio", "1"}


def normalizar(nombre: str) -> str:
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sin_ruido(s: str) -> str:
    return " ".join(t for t in s.split() if t not in _RUIDO) or s


def emparejar(nombre: str, candidatos: list[str]) -> str | None:
    """Devuelve el candidato que corresponde a `nombre`, o None si no hay uno claro."""
    if nombre in candidatos:
        return nombre
    norm = {normalizar(c): c for c in candidatos}
    n = normalizar(nombre)
    n = ALIAS.get(n, n)
    if n in norm:
        return norm[n]
    limpios = {_sin_ruido(k): v for k, v in norm.items()}
    nl = _sin_ruido(n)
    if nl in limpios:
        return limpios[nl]
    # Un nombre contenido en el otro ("Betis" / "Real Betis Balompie")
    contenidos = [v for k, v in limpios.items() if nl in k.split() or k in nl.split() or k == nl]
    if len(contenidos) == 1:
        return contenidos[0]
    parecido = difflib.get_close_matches(nl, list(limpios), n=1, cutoff=0.75)
    return limpios[parecido[0]] if parecido else None
