"""Backtest walk-forward: el modelo solo ve partidos anteriores a cada jornada.

Las apuestas se simulan con las cuotas de cierre históricas de football-data
(solo existen para 1X2 y más/menos 2,5 goles). Para córners, tarjetas y remates
no hay cuotas históricas gratuitas, así que se mide la calidad de la predicción
frente a una referencia ingenua (la media de la liga).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datos import ESTADISTICAS
from .mercados import Seleccion, matriz_conjunta, probabilidad
from .modelo import ModeloPartido
from .valor import Config, es_recomendable, sin_margen, stake

# Prefijos de columnas de cuotas de cierre en football-data
CASAS = {
    "media": ("AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5"),
    "max": ("MaxCH", "MaxCD", "MaxCA", "MaxC>2.5", "MaxC<2.5"),
    "b365": ("B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5"),
    "pinnacle": ("PSCH", "PSCD", "PSCA", "PC>2.5", "PC<2.5"),
}

SELECCIONES_CON_CUOTA = [
    (Seleccion("goles", "1x2", "1"), 0),
    (Seleccion("goles", "1x2", "X"), 1),
    (Seleccion("goles", "1x2", "2"), 2),
    (Seleccion("goles", "total", "mas", 2.5), 3),
    (Seleccion("goles", "total", "menos", 2.5), 4),
]


@dataclass
class Resultado:
    apuestas: list[dict] = field(default_factory=list)
    logloss_modelo: list[float] = field(default_factory=list)
    logloss_mercado: list[float] = field(default_factory=list)
    estadisticas: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)


def _gana(sel: Seleccion, fila) -> bool:
    gl, gv = fila["FTHG"], fila["FTAG"]
    if sel.tipo == "1x2":
        return {"1": gl > gv, "X": gl == gv, "2": gl < gv}[sel.lado]
    total = gl + gv
    return total > sel.linea if sel.lado == "mas" else total < sel.linea


def ejecutar(df: pd.DataFrame, temporada: str, casa: str = "media", cfg: Config | None = None,
             vida_media_dias: float = 240, estadisticas=("goles", "corners", "tarjetas", "remates_puerta")) -> Resultado:
    cfg = cfg or Config()
    cols = CASAS[casa]
    test = df[df["Temporada"] == temporada]
    if test.empty:
        raise ValueError(f"No hay partidos de la temporada {temporada}")
    res = Resultado(estadisticas={e: [] for e in estadisticas})
    semanas = test["Fecha"].dt.to_period("W")

    for _, jornada in test.groupby(semanas):
        inicio = jornada["Fecha"].min()
        entreno = df[(df["Fecha"] < inicio) & (df["Fecha"] >= inicio - pd.Timedelta(days=3 * 365))]
        try:
            modelo = ModeloPartido(vida_media_dias).ajustar(entreno, fecha_ref=inicio, estadisticas=estadisticas)
        except ValueError:
            continue
        for _, fila in jornada.iterrows():
            try:
                preds = modelo.predecir(fila["HomeTeam"], fila["AwayTeam"])
            except KeyError:
                continue  # equipo recién ascendido sin historial
            _evaluar_estadisticas(res, preds, fila, df_media=entreno)
            pred = preds["goles"]
            m = matriz_conjunta(pred)
            p_1x2 = [probabilidad(pred, s, m)[0] for s, _ in SELECCIONES_CON_CUOTA[:3]]

            cuotas = [pd.to_numeric(fila.get(c), errors="coerce") for c in cols]
            if all(np.isfinite(cuotas[:3])):
                justas = sin_margen(cuotas[:3])
                real = 0 if fila["FTHG"] > fila["FTAG"] else 1 if fila["FTHG"] == fila["FTAG"] else 2
                res.logloss_modelo.append(-np.log(max(p_1x2[real], 1e-9)))
                res.logloss_mercado.append(-np.log(justas[real]))

            for sel, k in SELECCIONES_CON_CUOTA:
                cuota = cuotas[k]
                if not np.isfinite(cuota):
                    continue
                p = probabilidad(pred, sel, m)[0]
                if es_recomendable(p, cuota, 0.0, cfg):
                    gana = _gana(sel, fila)
                    s = stake(p, cuota, 0.0, cfg)
                    res.apuestas.append({
                        "fecha": fila["Fecha"].date(), "partido": f"{fila['HomeTeam']} - {fila['AwayTeam']}",
                        "apuesta": sel.describir(fila["HomeTeam"], fila["AwayTeam"]), "prob": p, "cuota": cuota,
                        "stake": s, "beneficio_kelly": s * (cuota - 1) if gana else -s,
                        "beneficio_plano": (cuota - 1) if gana else -1.0, "gana": gana,
                    })
    return res


def _evaluar_estadisticas(res: Resultado, preds, fila, df_media: pd.DataFrame) -> None:
    for est, registros in res.estadisticas.items():
        cl, cv = ESTADISTICAS[est]
        if est not in preds or pd.isna(fila.get(cl)) or pd.isna(fila.get(cv)):
            continue
        p = preds[est]
        real = fila[cl] + fila[cv]
        ingenuo = (df_media[cl] + df_media[cv]).tail(380).mean()
        registros.append((p.local + p.visitante, ingenuo, real))


def informe(res: Resultado) -> str:
    lineas = []
    if res.logloss_modelo:
        lineas.append("Calidad 1X2 (log-loss, menor es mejor):")
        lineas.append(f"  modelo  {np.mean(res.logloss_modelo):.4f}")
        lineas.append(f"  mercado {np.mean(res.logloss_mercado):.4f}  (cuotas de cierre sin margen)")
    lineas.append("\nPredicción de totales (error absoluto medio, menor es mejor):")
    for est, regs in res.estadisticas.items():
        if not regs:
            continue
        r = np.array(regs)
        mae_m, mae_i = np.mean(np.abs(r[:, 0] - r[:, 2])), np.mean(np.abs(r[:, 1] - r[:, 2]))
        mejora = (1 - mae_m / mae_i) * 100
        lineas.append(f"  {est:<15} modelo {mae_m:5.2f}   media liga {mae_i:5.2f}   mejora {mejora:+.1f} %")

    ap = pd.DataFrame(res.apuestas)
    lineas.append("")
    if ap.empty:
        lineas.append("Ninguna apuesta superó el umbral de valor.")
        return "\n".join(lineas)
    n, aciertos = len(ap), ap["gana"].mean()
    roi_plano = ap["beneficio_plano"].sum() / n * 100
    roi_kelly = ap["beneficio_kelly"].sum() / ap["stake"].sum() * 100
    lineas.append(f"Apuestas simuladas: {n}  (acierto {aciertos:.1%}, cuota media {ap['cuota'].mean():.2f})")
    lineas.append(f"  ROI apuesta fija: {roi_plano:+.1f} %   ({ap['beneficio_plano'].sum():+.1f} unidades)")
    lineas.append(f"  ROI Kelly:        {roi_kelly:+.1f} %   (bankroll {ap['beneficio_kelly'].sum() * 100:+.1f} %)")
    por_mercado = ap.assign(mercado=ap["apuesta"].str.replace(r"Gana .*", "Gana equipo", regex=True))
    tabla = por_mercado.groupby("mercado").agg(n=("gana", "size"), acierto=("gana", "mean"), unidades=("beneficio_plano", "sum"))
    lineas.append("\nPor mercado (apuesta fija):")
    for mercado, fila in tabla.iterrows():
        lineas.append(f"  {mercado:<28} {int(fila.n):4d} apuestas  acierto {fila.acierto:5.1%}  {fila.unidades:+6.1f} u")
    return "\n".join(lineas)
