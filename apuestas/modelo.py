"""Modelos de conteo (Poisson con sobredispersión) por estadística de equipo.

Para cada estadística (goles, remates, córners, tarjetas...) se ajusta:

    log λ_local     = μ + ventaja_local + ataque[local]     + defensa[visitante]
    log λ_visitante = μ                 + ataque[visitante] + defensa[local]

con pesos que decaen exponencialmente con la antigüedad del partido y una
penalización L2 que lleva a la media a los equipos con pocos partidos.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .datos import ESTADISTICAS


@dataclass
class Prediccion:
    """Media esperada de una estadística para cada equipo y su dispersión (var/media)."""

    local: float
    visitante: float
    dispersion: float = 1.0


class ModeloConteo:
    def __init__(self, vida_media_dias: float = 240, regularizacion: float = 2.0):
        self.xi = np.log(2) / vida_media_dias
        self.reg = regularizacion
        self.equipos: list[str] = []
        self.indice: dict[str, int] = {}
        self.mu = self.ventaja = 0.0
        self.ataque = self.defensa = np.zeros(0)
        self.dispersion = 1.0
        self.n_partidos = 0
        self.nivel_liga = np.zeros(1)
        self.liga_de: dict[str, int] = {}

    def ajustar(self, df: pd.DataFrame, col_local: str, col_visitante: str, fecha_ref=None) -> "ModeloConteo":
        df = df.dropna(subset=[col_local, col_visitante])
        if len(df) < 30:
            raise ValueError(f"Muy pocos partidos con {col_local}/{col_visitante} ({len(df)})")
        fecha_ref = pd.Timestamp(fecha_ref) if fecha_ref is not None else df["Fecha"].max()
        self.equipos = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.indice = {e: i for i, e in enumerate(self.equipos)}
        n = len(self.equipos)

        h = df["HomeTeam"].map(self.indice).to_numpy()
        a = df["AwayTeam"].map(self.indice).to_numpy()
        # Cada liga tiene su propio nivel medio (sin penalizar); la primera es la referencia.
        ligas = df["Liga"] if "Liga" in df.columns else pd.Series("", index=df.index)
        codigos = sorted(ligas.unique())
        lg = ligas.map({c: k for k, c in enumerate(codigos)}).to_numpy()
        n_ligas = len(codigos)
        self.liga_de = {}
        for equipo, k in list(zip(df["HomeTeam"], lg)) + list(zip(df["AwayTeam"], lg)):
            self.liga_de[equipo] = int(k)  # ordenado por fecha: se queda con la liga más reciente
        yh = df[col_local].to_numpy(float)
        ya = df[col_visitante].to_numpy(float)
        dias = (fecha_ref - df["Fecha"]).dt.days.clip(lower=0).to_numpy(float)
        w = np.exp(-self.xi * dias)

        def objetivo(theta):
            mu, ventaja = theta[0], theta[1]
            att, de = theta[2 : 2 + n], theta[2 + n : 2 + 2 * n]
            nivel = np.concatenate([[0.0], theta[2 + 2 * n :]])[lg]
            eta_h = mu + ventaja + att[h] + de[a] + nivel
            eta_a = mu + att[a] + de[h] + nivel
            lh, la = np.exp(eta_h), np.exp(eta_a)
            nll = np.sum(w * (lh - yh * eta_h)) + np.sum(w * (la - ya * eta_a))
            nll += self.reg * (att @ att + de @ de)
            # Gradiente analítico
            rh, ra = w * (lh - yh), w * (la - ya)
            g = np.zeros_like(theta)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g_att = np.bincount(h, rh, n) + np.bincount(a, ra, n) + 2 * self.reg * att
            g_def = np.bincount(a, rh, n) + np.bincount(h, ra, n) + 2 * self.reg * de
            g[2 : 2 + n], g[2 + n : 2 + 2 * n] = g_att, g_def
            g[2 + 2 * n :] = np.bincount(lg, rh + ra, n_ligas)[1:]
            return nll, g

        media = max(np.mean(np.concatenate([yh, ya])), 1e-3)
        theta0 = np.zeros(2 + 2 * n + n_ligas - 1)
        theta0[0] = np.log(media)
        res = minimize(objetivo, theta0, jac=True, method="L-BFGS-B")
        theta = res.x
        self.mu, self.ventaja = theta[0], theta[1]
        self.ataque, self.defensa = theta[2 : 2 + n], theta[2 + n : 2 + 2 * n]
        self.nivel_liga = np.concatenate([[0.0], theta[2 + 2 * n :]])

        # Sobredispersión de Pearson ponderada: var ≈ φ·media
        nivel = self.nivel_liga[lg]
        lh = np.exp(self.mu + self.ventaja + self.ataque[h] + self.defensa[a] + nivel)
        la = np.exp(self.mu + self.ataque[a] + self.defensa[h] + nivel)
        pearson = np.concatenate([(yh - lh) ** 2 / lh, (ya - la) ** 2 / la])
        pesos = np.concatenate([w, w])
        g_libertad = 1 - len(theta) / (2 * len(df))
        self.dispersion = float(max(1.0, np.sum(pesos * pearson) / np.sum(pesos) / max(g_libertad, 0.5)))
        self.n_partidos = len(df)
        return self

    def predecir(self, local: str, visitante: str) -> Prediccion:
        i, j = self.indice[local], self.indice[visitante]
        # Partido entre ligas distintas: se usa el nivel medio de ambas
        nivel = (self.nivel_liga[self.liga_de[local]] + self.nivel_liga[self.liga_de[visitante]]) / 2
        lh = np.exp(self.mu + self.ventaja + self.ataque[i] + self.defensa[j] + nivel)
        la = np.exp(self.mu + self.ataque[j] + self.defensa[i] + nivel)
        return Prediccion(float(lh), float(la), self.dispersion)


class ModeloPartido:
    """Agrupa un ModeloConteo por cada estadística disponible en los datos."""

    def __init__(self, vida_media_dias: float = 240, regularizacion: float = 2.0):
        self.vida_media_dias = vida_media_dias
        self.regularizacion = regularizacion
        self.modelos: dict[str, ModeloConteo] = {}

    def ajustar(self, df: pd.DataFrame, fecha_ref=None, estadisticas=None) -> "ModeloPartido":
        for nombre, (cl, cv) in ESTADISTICAS.items():
            if estadisticas and nombre not in estadisticas:
                continue
            if cl not in df.columns or df[cl].notna().sum() < 30:
                continue
            modelo = ModeloConteo(self.vida_media_dias, self.regularizacion)
            self.modelos[nombre] = modelo.ajustar(df, cl, cv, fecha_ref)
        if "goles" not in self.modelos:
            raise ValueError("No se pudo ajustar el modelo de goles")
        return self

    @property
    def equipos(self) -> list[str]:
        return self.modelos["goles"].equipos

    def resolver_equipo(self, nombre: str) -> str:
        """Encuentra el nombre exacto del equipo en los datos (tolera variantes)."""
        from .nombres import emparejar

        encontrado = emparejar(nombre, self.equipos)
        if encontrado is None:
            parecidos = difflib.get_close_matches(nombre, self.equipos, n=5, cutoff=0.3)
            raise KeyError(f"Equipo desconocido: {nombre!r}. ¿Quisiste decir: {', '.join(parecidos) or '-'}?")
        return encontrado

    def predecir(self, local: str, visitante: str) -> dict[str, Prediccion]:
        local, visitante = self.resolver_equipo(local), self.resolver_equipo(visitante)
        return {
            nombre: m.predecir(local, visitante)
            for nombre, m in self.modelos.items()
            if local in m.indice and visitante in m.indice
        }
