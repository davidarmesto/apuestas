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
from .mercados import NOMBRES, Seleccion, matriz_conjunta, probabilidad
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
    # Por estadística: (mercado, probabilidad del modelo, frecuencia de la liga, si salió)
    calibracion: dict[str, list[tuple[str, float, float, bool]]] = field(default_factory=dict)


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
    res = Resultado(estadisticas={e: [] for e in estadisticas}, calibracion={e: [] for e in estadisticas})
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

_MERCADO = {"total": "{}: total", "local": "{} del local", "visitante": "{} del visitante"}


def _evaluar_estadisticas(res: Resultado, preds, fila, df_media: pd.DataFrame) -> None:
    for est, registros in res.estadisticas.items():
        cl, cv = ESTADISTICAS[est]
        if est not in preds or pd.isna(fila.get(cl)) or pd.isna(fila.get(cv)):
            continue
        p = preds[est]
        real = fila[cl] + fila[cv]
        ingenuo = (df_media[cl] + df_media[cv]).tail(380).mean()
        registros.append((p.local + p.visitante, ingenuo, real))
        # Líneas habituales de las casas: el total y lo de cada equipo, alrededor de la media de la liga
        recientes = df_media[[cl, cv]].dropna().tail(380)
        m = matriz_conjunta(p)
        for tipo, valores, real_tipo in (
            ("total", recientes[cl] + recientes[cv], real),
            ("local", recientes[cl], fila[cl]),
            ("visitante", recientes[cv], fila[cv]),
        ):
            linea = float(np.floor(valores.median())) + 0.5
            sel = Seleccion(est, tipo, "mas", linea)
            res.calibracion[est].append(
                (_MERCADO[tipo].format(NOMBRES[est]), probabilidad(p, sel, m)[0], float((valores > linea).mean()),
                 bool(real_tipo > linea))
            )


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

    lineas += _informe_calibracion(res)
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


def _informe_calibracion(res: Resultado) -> list[str]:
    """¿Las probabilidades de córners, tarjetas y remates salen tantas veces como dice el modelo?

    Brier: error cuadrático de la probabilidad (menor es mejor), frente a usar siempre la
    frecuencia de la liga. Si el modelo no mejora a la frecuencia, no sabe nada de ese mercado
    más allá de la media.
    """
    lineas = ["\nCalibración de 'más de' en la línea habitual de la liga (Brier, menor es mejor; "
              "'liga' = usar siempre la frecuencia de la liga):"]
    for est, regs in res.calibracion.items():
        if not regs:
            continue
        df = pd.DataFrame(regs, columns=["mercado", "p", "liga", "sale"])
        for mercado, g in df.groupby("mercado", sort=False):
            b_m, b_l = ((g.p - g.sale) ** 2).mean(), ((g.liga - g.sale) ** 2).mean()
            # Cuando el modelo está más seguro (p ≥ 60 %), ¿cuántas veces sale de verdad?
            seguras = g[g.p >= 0.6]
            extra = f"   p≥60 %: dice {seguras.p.mean():.0%}, sale {seguras.sale.mean():.0%} (n={len(seguras)})" if len(seguras) >= 20 else ""
            lineas.append(f"  {mercado:<30} n={len(g):<4} modelo {b_m:.4f}  liga {b_l:.4f}  "
                          f"mejora {(1 - b_m / b_l) * 100:+5.1f} %{extra}")
    return lineas


# Combinadas típicas de "crear apuesta" para comprobar el cálculo conjunto con partidos reales
PLANTILLAS_COMBINADAS = {
    "Gana local + más de 2,5": ["goles 1x2 1", "goles total mas 2.5"],
    "Gana local + ambos marcan": ["goles 1x2 1", "goles btts si"],
    "Ambos marcan + más de 2,5": ["goles btts si", "goles total mas 2.5"],
    "Gana local + ambos marcan + más de 3,5": ["goles 1x2 1", "goles btts si", "goles total mas 3.5"],
    "Goles local >1,5 + a puerta local >4,5": ["goles local mas 1.5", "remates_puerta local mas 4.5"],
    "Goles local >1,5 + córners local >5,5": ["goles local mas 1.5", "corners local mas 5.5"],
    "Gana local + córners local >5,5": ["goles 1x2 1", "corners local mas 5.5"],
    "Córners local >5,5 + a puerta local >4,5": ["corners local mas 5.5", "remates_puerta local mas 4.5"],
    "Córners local >4,5 + córners visit. >3,5": ["corners local mas 4.5", "corners visitante mas 3.5"],
    "Tarjetas local >1,5 + visit. >1,5": ["tarjetas local mas 1.5", "tarjetas visitante mas 1.5"],
    "Más de 2,5 + córners >9,5": ["goles total mas 2.5", "corners total mas 9.5"],
    "Más de 2,5 + tarjetas >3,5": ["goles total mas 2.5", "tarjetas total mas 3.5"],
}


def _gana_seleccion(s: Seleccion, fila) -> bool:
    cl, cv = ESTADISTICAS[s.estadistica]
    l, v = fila[cl], fila[cv]
    if s.tipo == "1x2":
        return {"1": l > v, "X": l == v, "2": l < v}[s.lado]
    if s.tipo == "btts":
        return (l > 0 and v > 0) == (s.lado == "si")
    valor = {"total": l + v, "local": l, "visitante": v}[s.tipo]
    return valor > s.linea if s.lado == "mas" else valor < s.linea


def calibrar_combinadas(df: pd.DataFrame, temporada: str, vida_media_dias: float = 240) -> pd.DataFrame:
    """Para cada combinada típica: cuántas veces salió de verdad, qué decía el modelo con sus
    correlaciones y qué habría dicho multiplicando las probabilidades como si fueran independientes."""
    from .combinadas import parsear_pierna, probabilidad_combinada

    plantillas = {n: [parsear_pierna(t) for t in ps] for n, ps in PLANTILLAS_COMBINADAS.items()}
    test = df[df["Temporada"] == temporada]
    filas = []
    # Reajuste mensual: las correlaciones y los equipos cambian poco de una semana a otra
    for _, bloque in test.groupby(test["Fecha"].dt.to_period("M")):
        inicio = bloque["Fecha"].min()
        entreno = df[(df["Fecha"] < inicio) & (df["Fecha"] >= inicio - pd.Timedelta(days=3 * 365))]
        try:
            modelo = ModeloPartido(vida_media_dias).ajustar(
                entreno, fecha_ref=inicio, estadisticas=["goles", "corners", "tarjetas", "remates_puerta"])
        except ValueError:
            continue
        for _, fila in bloque.iterrows():
            try:
                preds = modelo.predecir(fila["HomeTeam"], fila["AwayTeam"])
            except KeyError:
                continue
            for nombre, piernas in plantillas.items():
                if any(s.estadistica not in preds or pd.isna(fila.get(c)) for s in piernas for c in ESTADISTICAS[s.estadistica]):
                    continue
                filas.append({
                    "combinada": nombre,
                    "sale": all(_gana_seleccion(s, fila) for s in piernas),
                    "modelo": probabilidad_combinada(preds, piernas, modelo.correlaciones),
                    "independientes": float(np.prod([probabilidad_combinada(preds, [s]) for s in piernas])),
                })
    r = pd.DataFrame(filas)
    if r.empty:
        raise ValueError(f"No hay partidos de la temporada {temporada}")
    return r.groupby("combinada", sort=False).agg(
        n=("sale", "size"), sale=("sale", "mean"), modelo=("modelo", "mean"), independientes=("independientes", "mean"))


def informe_combinadas(tabla: pd.DataFrame) -> str:
    lineas = ["Combinadas: % de veces que salieron frente a lo que decía el modelo", "",
              f"  {'':<42} {'partidos':>8} {'salió':>7} {'modelo':>7} {'si fueran indep.':>17}"]
    for nombre, f in tabla.iterrows():
        lineas.append(f"  {nombre:<42} {int(f.n):>8} {f.sale:>7.1%} {f.modelo:>7.1%} {f.independientes:>17.1%}")
    err_m = (tabla.modelo - tabla.sale).abs().mean()
    err_i = (tabla.independientes - tabla.sale).abs().mean()
    lineas.append(f"\n  Error medio: modelo {err_m:.1%}, independientes {err_i:.1%} (puntos de probabilidad)")
    return "\n".join(lineas)
