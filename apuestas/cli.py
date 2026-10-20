"""Línea de comandos: python -m apuestas <comando> ..."""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*")
import sys

import pandas as pd

from . import backtest, datos, odds_api
from .mercados import NOMBRES, calibrar_goles, Seleccion, matriz_conjunta, mercados_estandar, probabilidad
from .modelo import ModeloPartido
from .nombres import emparejar
from .valor import Config, es_recomendable, stake, valor_esperado


def _modelo(ligas: list[str], temporadas: int, vida_media: float) -> tuple[ModeloPartido, pd.DataFrame]:
    df = pd.concat([datos.cargar(liga, datos.temporadas_recientes(temporadas)) for liga in ligas], ignore_index=True)
    df = df.sort_values("Fecha").reset_index(drop=True)
    return ModeloPartido(vida_media).ajustar(df, fecha_ref=pd.Timestamp.today().normalize()), df


def _config(a) -> Config:
    return Config(ventaja_minima=a.ventaja, cuota_maxima=a.cuota_max, fraccion_kelly=a.kelly)


def _nombre_ligas(ligas: list[str]) -> str:
    return " + ".join(datos.LIGAS[l][0] for l in ligas)


def _predicciones(a, modelo: ModeloPartido, local: str, visitante: str):
    """Predicciones del modelo, calibradas con las cuotas 1X2 del mercado si se pasan."""
    preds = modelo.predecir(local, visitante)
    if a.mercado_1x2:
        original = preds["goles"]
        preds["goles"] = calibrar_goles(original, *a.mercado_1x2)
        print(f"Goles calibrados con el 1X2 del mercado {a.mercado_1x2}: "
              f"{original.local:.2f}-{original.visitante:.2f} -> {preds['goles'].local:.2f}-{preds['goles'].visitante:.2f}")
    elif len(a.liga) > 1:
        print("Aviso: equipos de ligas distintas; el modelo supone que las ligas tienen el mismo nivel. "
              "Usa --mercado-1x2 C1 CX C2 para calibrar los goles con las cuotas de la casa.")
    return preds


def _anclar_al_mercado(a, modelo: ModeloPartido, local: str, visitante: str, preds):
    """Descarga las cuotas del partido (todas las casas y mercados) y ajusta el modelo a ellas."""
    from .ancla import anclar, probabilidades_justas

    sport = a.competicion or datos.LIGAS[a.liga[0]][1]
    for ev in odds_api.eventos(sport, a.regiones):
        if (emparejar(ev.local, modelo.equipos), emparejar(ev.visitante, modelo.equipos)) == (local, visitante):
            odds_api.cuotas_extra(sport, ev, a.regiones)
            if getattr(a, "jugadores", False):
                odds_api.cuotas_extra(sport, ev, a.regiones, odds_api.MERCADOS_JUGADOR)
            justas = probabilidades_justas(ev)
            nuevas, resumen = anclar(preds, justas)
            print(f"Modelo anclado a {len({c.casa for c in ev.cuotas})} casas:")
            for linea in resumen:
                print("  " + linea)
            return nuevas, ev
    print(f"Aviso: no encuentro {local} - {visitante} en {sport}; uso el modelo sin anclar")
    return preds, None


def _preparar(a):
    modelo, _ = _modelo(a.liga, a.temporadas, a.vida_media)
    local, visitante = modelo.resolver_equipo(a.local), modelo.resolver_equipo(a.visitante)
    preds = modelo.predecir(local, visitante)
    ev, jugadores = None, {}
    if a.anclar:
        from .ancla import lambdas_jugadores

        preds, ev = _anclar_al_mercado(a, modelo, local, visitante, preds)
        if ev is not None:
            jugadores = lambdas_jugadores(ev)
            if jugadores:
                print(f"  {len(jugadores)} estadísticas de jugadores con cuotas")
    return local, visitante, preds, ev, modelo, jugadores


def cmd_combinada(a):
    from .combinadas import parsear_pierna, probabilidad_combinada

    local, visitante, preds, _, modelo, jugadores = _preparar(a)
    piernas = [parsear_pierna(t) for t in a.pierna]
    p = probabilidad_combinada(preds, piernas, modelo.correlaciones, jugadores)
    p_indep = probabilidad_combinada(preds, piernas, None, jugadores)
    print(f"\n{local} vs {visitante} — combinada de {len(piernas)} selecciones:")
    for s in piernas:
        print(f"  • {s.describir(local, visitante)}")
    print(f"\n  Probabilidad: {p:.1%}  (sale ~{p * 10:.1f} de cada 10 veces)")
    if abs(p - p_indep) > 0.005:
        print(f"  (si fueran independientes sería {p_indep:.1%}; la diferencia es por cómo se relacionan entre sí)")
    print(f"  Cuota justa:  {1 / p:.2f}")
    if a.cuota:
        ev = valor_esperado(p, a.cuota)
        print(f"  Cuota de la casa {a.cuota:.2f}: valor esperado {ev:+.1%} "
              + ("✅ por encima de lo justo" if ev > 0 else "❌ la casa paga menos de lo justo"))
        if a.importe:
            print(f"  {a.importe:.0f} € -> cobras {a.importe * a.cuota:.2f} € si acierta; "
                  f"a largo plazo ganarías/perderías {a.importe * ev:+.2f} € de media por apuesta")


def cmd_sugerir(a):
    from .combinadas import sugerir

    local, visitante, preds, ev, modelo, _ = _preparar(a)
    solo = a.solo
    if not solo and ev is not None:
        # Solo estadísticas que el mercado ha podido corregir; el resto del modelo no es fiable aquí
        from .ancla import probabilidades_justas

        solo = sorted({s.estadistica for s in probabilidades_justas(ev)})
    resultados = sugerir(preds, a.objetivo, a.max_piernas, solo, a.prob_pierna, correlaciones=modelo.correlaciones)
    if a.min_piernas:
        resultados = [r for r in resultados if len(r[0]) >= a.min_piernas]
    print(f"\nCombinadas con cuota justa ≈ {a.objetivo} ({len(resultados)} encontradas). Las {a.top} más probables:\n")
    for combo, p in resultados[: a.top]:
        print(f"  {p:5.1%}  cuota justa {1 / p:4.2f}  ·  " + "  +  ".join(s.describir(local, visitante) for s in combo))
    if not resultados:
        print("  (ninguna; prueba con otra cuota objetivo o baja --prob-pierna)")
    print("\nCompara cada cuota justa con la que te da la casa: solo hay valor si la casa paga MÁS.")


def cmd_equipos(a):
    modelo, df = _modelo(a.liga, 1, a.vida_media)
    print("\n".join(sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))))


def cmd_predecir(a):
    modelo, _ = _modelo(a.liga, a.temporadas, a.vida_media)
    local, visitante = modelo.resolver_equipo(a.local), modelo.resolver_equipo(a.visitante)
    print(f"\n{local} vs {visitante}  ({_nombre_ligas(a.liga)})\n")
    preds = _predicciones(a, modelo, local, visitante)
    print("Medias esperadas:")
    for est, p in preds.items():
        print(f"  {NOMBRES[est]:<20} {p.local:5.2f} - {p.visitante:5.2f}   total {p.local + p.visitante:5.2f}")
    filas = mercados_estandar(preds)
    est_actual = None
    for sel, p in filas:
        if a.solo and sel.estadistica not in a.solo:
            continue
        if sel.estadistica != est_actual:
            est_actual = sel.estadistica
            print(f"\n{NOMBRES[est_actual]}")
        print(f"  {sel.describir(local, visitante):<45} {p:6.1%}   cuota justa {1 / max(p, 1e-9):6.2f}")


def cmd_cuota(a):
    """Evalúa una cuota que el usuario ha visto en su casa de apuestas."""
    modelo, _ = _modelo(a.liga, a.temporadas, a.vida_media)
    local, visitante = modelo.resolver_equipo(a.local), modelo.resolver_equipo(a.visitante)
    preds = _predicciones(a, modelo, local, visitante)
    if a.estadistica not in preds:
        sys.exit(f"Sin datos de {a.estadistica} para esta liga")
    sel = Seleccion(a.estadistica, a.tipo, a.lado, a.linea)
    p, nulo = probabilidad(preds[a.estadistica], sel)
    ev = valor_esperado(p, a.precio, nulo)
    cfg = _config(a)
    print(f"\n{sel.describir(local, visitante)} @ {a.precio}")
    print(f"  Probabilidad modelo: {p:.1%}" + (f" (nulo {nulo:.1%})" if nulo else ""))
    print(f"  Cuota justa: {(1 - nulo) / max(p, 1e-9):.2f}")
    print(f"  Valor esperado: {ev:+.1%}")
    implicita = 1 / a.precio
    if abs(p - implicita) > 0.15:
        print(f"  ⚠️  La casa da un {implicita:.0%} y el modelo un {p:.0%}: con tanta diferencia lo más probable "
              "es que el modelo se equivoque (lesiones, árbitro, tipo de partido...)")
    if es_recomendable(p, a.precio, nulo, cfg):
        print(f"  ✅ Hay valor. Apuesta sugerida: {stake(p, a.precio, nulo, cfg):.1%} del bankroll")
    else:
        print("  ❌ No hay valor suficiente")


def cmd_valor(a):
    cfg = _config(a)
    sport = a.competicion or datos.LIGAS[a.liga[0]][1]
    modelo, _ = _modelo(a.liga, a.temporadas, a.vida_media)
    eventos = odds_api.eventos(sport, a.regiones)
    if a.dias:
        limite = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=a.dias)
        eventos = [e for e in eventos if pd.Timestamp(e.inicio) <= limite]
    if a.partido:
        equipo = modelo.resolver_equipo(a.partido)
        eventos = [e for e in eventos if equipo in (emparejar(e.local, modelo.equipos), emparejar(e.visitante, modelo.equipos))]
    print(f"{len(eventos)} partidos con cuotas en {sport}")
    recomendaciones = []
    for ev in eventos:
        local, visitante = emparejar(ev.local, modelo.equipos), emparejar(ev.visitante, modelo.equipos)
        if not local or not visitante:
            desconocidos = [n for n, e in ((ev.local, local), (ev.visitante, visitante)) if not e]
            print(f"  Sin datos de {', '.join(desconocidos)} ({ev.local} - {ev.visitante}): se omite")
            continue
        if a.extra:
            odds_api.cuotas_extra(sport, ev, a.regiones)
        preds = modelo.predecir(local, visitante)
        matrices = {e: matriz_conjunta(p) for e, p in preds.items()}
        consenso = ev.consenso()
        for c in ev.mejores_cuotas():
            est = c.seleccion.estadistica
            if est not in preds or c.seleccion.tipo == "jugador":
                continue
            p, nulo = probabilidad(preds[est], c.seleccion, matrices[est])
            if es_recomendable(p, c.precio, nulo, cfg):
                recomendaciones.append({
                    "inicio": pd.Timestamp(ev.inicio).tz_convert("Europe/Madrid").strftime("%d/%m %H:%M"),
                    "partido": f"{ev.local} - {ev.visitante}",
                    "apuesta": c.seleccion.describir(ev.local, ev.visitante),
                    "cuota": c.precio, "casa": c.casa, "prob": p,
                    "valor": valor_esperado(p, c.precio, nulo),
                    "stake": stake(p, c.precio, nulo, cfg),
                    # Si el modelo discrepa mucho de todas las casas, lo normal es que se equivoque el modelo
                    "discrepa": abs(p - consenso[c.seleccion]) > a.discrepancia,
                })
    if not recomendaciones:
        print("\nNo hay apuestas con valor según el modelo.")
        return
    todas = pd.DataFrame(recomendaciones).sort_values("valor", ascending=False)
    descartadas, tabla = todas[todas.discrepa], todas[~todas.discrepa].head(a.top)
    _registrar(tabla, "+".join(a.liga))
    print(f"\nApuestas con valor según el modelo (EV ≥ {cfg.ventaja_minima:.0%}):")
    print("  Ojo: en backtests el modelo NO bate al mercado en 1X2 ni en goles; trátalo como orientación.\n")
    for _, r in tabla.iterrows():
        print(f"  {r.inicio}  {r.partido}")
        print(f"      {r.apuesta:<42} @ {r.cuota:<5.2f} ({r.casa})  prob {r.prob:5.1%}  EV {r.valor:+5.1%}  stake {r.stake:.1%}")
    if tabla.empty:
        print("  (ninguna)")
    if not descartadas.empty:
        print(f"\nDescartadas: {len(descartadas)} en las que el modelo se aleja más de {a.discrepancia:.0%} "
              "de la media de las casas (probablemente es el modelo el que falla):")
        for _, r in descartadas.head(10).iterrows():
            print(f"  {r.partido}: {r.apuesta} @ {r.cuota:.2f}  (modelo {r.prob:.0%}, casas {1 / r.cuota:.0%})")


def _registrar(tabla: pd.DataFrame, liga: str) -> None:
    """Guarda las recomendaciones para poder medir después si el modelo acierta en mercados sin histórico de cuotas."""
    ruta = datos.DIR_DATOS / "registro.csv"
    nuevas = tabla.assign(liga=liga, registrado=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    # Se reescribe entero para que el CSV siga siendo válido aunque cambien las columnas
    if ruta.exists():
        nuevas = pd.concat([pd.read_csv(ruta), nuevas], ignore_index=True)
    nuevas.to_csv(ruta, index=False)


def cmd_backtest(a):
    cfg = _config(a)
    temporada = a.temporada or datos.temporadas_recientes(2)[0]
    inicio = 2000 + int(temporada[:2])
    previas = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(inicio - 3, inicio + 1)]
    df = datos.cargar(a.liga[0], previas)
    print(f"Backtest {datos.LIGAS[a.liga[0]][0]} {temporada} con cuotas de cierre '{a.casa}'...\n")
    res = backtest.ejecutar(df, temporada, a.casa, cfg, a.vida_media)
    print(backtest.informe(res))


def main(argv=None):
    p = argparse.ArgumentParser(prog="apuestas", description="Recomendador de apuestas de fútbol")
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--liga", nargs="+", default=["SP1"], choices=list(datos.LIGAS),
                       help="código(s) de liga (SP1 = LaLiga); varias para partidos europeos, p. ej. F1 SP1")
    comun.add_argument("--mercado-1x2", nargs=3, type=float, metavar=("C1", "CX", "C2"),
                       help="cuotas 1X2 de la casa para calibrar los goles (útil entre ligas distintas)")
    comun.add_argument("--temporadas", type=int, default=3, help="temporadas de historial para entrenar")
    comun.add_argument("--vida-media", type=float, default=240, help="días en que el peso de un partido se reduce a la mitad")
    comun.add_argument("--ventaja", type=float, default=0.05, help="valor esperado mínimo (0.05 = 5 %%)")
    comun.add_argument("--cuota-max", type=float, default=6.0)
    comun.add_argument("--kelly", type=float, default=0.25, help="fracción de Kelly")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("equipos", parents=[comun], help="lista los equipos de la liga").set_defaults(f=cmd_equipos)

    s = sub.add_parser("predecir", parents=[comun], help="probabilidades de todos los mercados de un partido")
    s.add_argument("local")
    s.add_argument("visitante")
    s.add_argument("--solo", nargs="+", choices=list(NOMBRES), help="mostrar solo estas estadísticas")
    s.set_defaults(f=cmd_predecir)

    s = sub.add_parser("cuota", parents=[comun], help="¿tiene valor esta cuota?")
    s.add_argument("local")
    s.add_argument("visitante")
    s.add_argument("estadistica", choices=list(NOMBRES))
    s.add_argument("tipo", choices=["1x2", "doble", "btts", "total", "local", "visitante", "handicap"])
    s.add_argument("lado", help="1/X/2, 1X/12/X2, si/no, mas/menos, local/visitante")
    s.add_argument("precio", type=float, help="cuota decimal")
    s.add_argument("--linea", type=float)
    s.set_defaults(f=cmd_cuota)

    s = sub.add_parser("valor", parents=[comun], help="busca apuestas con valor en las cuotas actuales (The Odds API)")
    s.add_argument("--competicion", choices=list(datos.COMPETICIONES.values()),
                   help="clave de The Odds API si no es una liga, p. ej. soccer_uefa_champs_league")
    s.add_argument("--extra", action="store_true", help="pide también córners, tarjetas, ambos marcan... (más créditos)")
    s.add_argument("--regiones", default="eu", help="regiones de casas: eu, uk, us, au")
    s.add_argument("--dias", type=float, default=7, help="solo partidos en los próximos N días")
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--discrepancia", type=float, default=0.10,
                   help="descarta selecciones donde el modelo se aleja más que esto del consenso de casas")
    s.add_argument("--partido", help="solo los partidos de este equipo (ahorra créditos con --extra)")
    s.set_defaults(f=cmd_valor)

    anclaje = argparse.ArgumentParser(add_help=False)
    anclaje.add_argument("local")
    anclaje.add_argument("visitante")
    anclaje.add_argument("--anclar", action="store_true", help="ajusta el modelo a las cuotas actuales (The Odds API)")
    anclaje.add_argument("--competicion", choices=list(datos.COMPETICIONES.values()))
    anclaje.add_argument("--regiones", default="eu", help="regiones de casas: eu, uk, us, au (separadas por comas)")
    anclaje.add_argument("--jugadores", action="store_true",
                         help="pide también remates y goleadores (casas de EE. UU.: añade 'us' a --regiones)")

    s = sub.add_parser("combinada", parents=[comun, anclaje], help="probabilidad de una combinada del mismo partido")
    s.add_argument("--pierna", action="append", required=True, help="una selección; repítelo por cada una")
    s.add_argument("--cuota", type=float, help="cuota que te ofrece la casa por la combinada")
    s.add_argument("--importe", type=float, help="euros a apostar")
    s.set_defaults(f=cmd_combinada)

    s = sub.add_parser("sugerir", parents=[comun, anclaje], help="busca combinadas con una cuota objetivo")
    s.add_argument("--objetivo", type=float, default=3.0, help="cuota buscada")
    s.add_argument("--max-piernas", type=int, default=4)
    s.add_argument("--min-piernas", type=int, default=2)
    s.add_argument("--prob-pierna", type=float, default=0.55, help="probabilidad mínima de cada selección")
    s.add_argument("--solo", nargs="+", help="estadísticas permitidas, p. ej. goles corners")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(f=cmd_sugerir)

    s = sub.add_parser("backtest", parents=[comun], help="simula el modelo en una temporada pasada")
    s.add_argument("--temporada", help="p. ej. 2526 (por defecto, la anterior)")
    s.add_argument("--casa", default="media", choices=list(backtest.CASAS))
    s.set_defaults(f=cmd_backtest)

    a = p.parse_args(argv)
    try:
        a.f(a)
    except (KeyError, RuntimeError, ValueError) as e:
        sys.exit(f"Error: {e.args[0] if e.args else e}")


if __name__ == "__main__":
    main()
