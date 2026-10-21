import numpy as np
import pandas as pd
import pytest

from apuestas.mercados import Seleccion, matriz_conjunta, probabilidad
from apuestas.modelo import ModeloConteo, Prediccion
from apuestas.nombres import emparejar
from apuestas.odds_api import parsear_evento
from apuestas.valor import kelly, sin_margen, valor_esperado


def partidos_simulados(n_equipos=10, vueltas=6, semilla=0):
    rng = np.random.default_rng(semilla)
    equipos = [f"E{i}" for i in range(n_equipos)]
    ataque = np.linspace(-0.4, 0.4, n_equipos)
    filas, fecha = [], pd.Timestamp("2024-01-01")
    for _ in range(vueltas):
        for i in range(n_equipos):
            for j in range(n_equipos):
                if i != j:
                    lh, la = np.exp(0.2 + 0.25 + ataque[i]), np.exp(0.2 + ataque[j])
                    filas.append({"HomeTeam": equipos[i], "AwayTeam": equipos[j], "Fecha": fecha,
                                  "FTHG": rng.poisson(lh), "FTAG": rng.poisson(la)})
        fecha += pd.Timedelta(days=7)
    return pd.DataFrame(filas), ataque


def test_modelo_recupera_parametros():
    df, ataque = partidos_simulados()
    m = ModeloConteo(vida_media_dias=10_000, regularizacion=0.1).ajustar(df, "FTHG", "FTAG")
    assert m.ventaja == pytest.approx(0.25, abs=0.1)
    assert np.corrcoef(m.ataque, ataque)[0, 1] > 0.9
    assert m.dispersion == pytest.approx(1.0, abs=0.15)


def test_probabilidades_suman_uno():
    pred = Prediccion(1.6, 1.1)
    m = matriz_conjunta(pred)
    assert m.sum() == pytest.approx(1)
    p1x2 = sum(probabilidad(pred, Seleccion("goles", "1x2", s), m)[0] for s in "1X2")
    assert p1x2 == pytest.approx(1)
    mas, _ = probabilidad(pred, Seleccion("goles", "total", "mas", 2.5), m)
    menos, _ = probabilidad(pred, Seleccion("goles", "total", "menos", 2.5), m)
    assert mas + menos == pytest.approx(1)


def test_linea_entera_tiene_nulo():
    pred = Prediccion(5.0, 4.5, dispersion=1.4)
    gana, nulo = probabilidad(pred, Seleccion("corners", "total", "mas", 10))
    pierde, _ = probabilidad(pred, Seleccion("corners", "total", "menos", 10))
    assert nulo > 0 and gana + nulo + pierde == pytest.approx(1)


def test_sobredispersion_ensancha_distribucion():
    estrecha = probabilidad(Prediccion(5, 5, 1.0), Seleccion("corners", "total", "mas", 13.5))[0]
    ancha = probabilidad(Prediccion(5, 5, 2.0), Seleccion("corners", "total", "mas", 13.5))[0]
    assert ancha > estrecha


def test_valor_y_kelly():
    assert valor_esperado(0.5, 2.2) == pytest.approx(0.1)
    assert kelly(0.5, 2.2) == pytest.approx(0.1 / 1.2)
    assert kelly(0.4, 2.0) == 0
    assert sum(sin_margen([1.9, 1.9])) == pytest.approx(1)


@pytest.mark.parametrize("casa,esperado", [
    ("Atlético Madrid", "Ath Madrid"), ("Athletic Bilbao", "Ath Bilbao"), ("Real Betis", "Betis"),
    ("Rayo Vallecano", "Vallecano"), ("Celta Vigo", "Celta"), ("Real Madrid", "Real Madrid"),
    ("Real Sociedad", "Sociedad"), ("RCD Espanyol", "Espanol"), ("CA Osasuna", "Osasuna"),
])
def test_emparejar_nombres(casa, esperado):
    equipos = ["Ath Madrid", "Ath Bilbao", "Betis", "Vallecano", "Celta", "Real Madrid", "Sociedad", "Espanol", "Osasuna"]
    assert emparejar(casa, equipos) == esperado


def test_parsear_evento_odds_api():
    ev = parsear_evento({
        "id": "x", "commence_time": "2026-10-25T19:00:00Z", "home_team": "Real Madrid", "away_team": "Barcelona",
        "bookmakers": [
            {"title": "A", "markets": [
                {"key": "h2h", "outcomes": [{"name": "Real Madrid", "price": 2.5}, {"name": "Draw", "price": 3.6},
                                            {"name": "Barcelona", "price": 2.7}]},
                {"key": "alternate_totals_corners", "outcomes": [{"name": "Over", "price": 1.9, "point": 9.5}]},
            ]},
            {"title": "B", "markets": [{"key": "h2h", "outcomes": [{"name": "Real Madrid", "price": 2.6}]}]},
        ],
    })
    mejores = {c.seleccion: c for c in ev.mejores_cuotas()}
    assert mejores[Seleccion("goles", "1x2", "1")].precio == 2.6
    assert mejores[Seleccion("goles", "1x2", "1")].casa == "B"
    assert Seleccion("corners", "total", "mas", 9.5) in mejores


def test_calibrar_goles_reproduce_mercado():
    from apuestas.mercados import calibrar_goles

    cal = calibrar_goles(Prediccion(1.8, 1.8), 2.10, 3.80, 3.40)
    m = matriz_conjunta(cal)
    p1 = probabilidad(cal, Seleccion("goles", "1x2", "1"), m)[0]
    assert p1 == pytest.approx(sin_margen([2.10, 3.80, 3.40])[0], abs=0.005)
    assert cal.local > cal.visitante


def test_lineas_de_cuarto_se_ignoran():
    ev = parsear_evento({
        "id": "x", "home_team": "A", "away_team": "B",
        "bookmakers": [{"title": "P", "markets": [{"key": "totals", "outcomes": [
            {"name": "Under", "price": 1.9, "point": 3.25}, {"name": "Under", "price": 2.1, "point": 3.0}]}]}],
    })
    assert [c.seleccion.linea for c in ev.cuotas] == [3.0]


def test_combinada_respeta_correlacion():
    from apuestas.combinadas import parsear_pierna, probabilidad_combinada

    preds = {"goles": Prediccion(1.8, 1.4), "corners": Prediccion(5, 4, 1.3)}
    btts, mas25 = parsear_pierna("goles btts si"), parsear_pierna("goles total mas 2.5")
    p_btts = probabilidad(preds["goles"], btts)[0]
    p_mas = probabilidad(preds["goles"], mas25)[0]
    juntas = probabilidad_combinada(preds, [btts, mas25])
    assert p_btts * p_mas < juntas <= min(p_btts, p_mas)  # correlación positiva
    corners = parsear_pierna("corners total mas 8.5")
    p_c = probabilidad(preds["corners"], corners)[0]
    assert probabilidad_combinada(preds, [btts, mas25, corners]) == pytest.approx(juntas * p_c)


def test_anclar_reproduce_mercado():
    from apuestas.ancla import anclar

    objetivo = {Seleccion("goles", "total", "mas", 2.5): (0.80, 10), Seleccion("goles", "1x2", "1"): (0.40, 10)}
    nuevas, _ = anclar({"goles": Prediccion(1.2, 1.2)}, objetivo)
    for s, (p, _) in objetivo.items():
        assert probabilidad(nuevas["goles"], s)[0] == pytest.approx(p, abs=0.01)


def test_correlacion_interna_mantiene_marginales():
    indep = matriz_conjunta(Prediccion(2.0, 2.2, 1.2, 0.0))
    corr = matriz_conjunta(Prediccion(2.0, 2.2, 1.2, 0.25))
    np.testing.assert_allclose(corr.sum(axis=1), indep.sum(axis=1), atol=1e-6)
    np.testing.assert_allclose(corr.sum(axis=0), indep.sum(axis=0), atol=1e-6)
    i, j = np.indices(corr.shape)
    cov = lambda m: (m * i * j).sum() - (m * i).sum() * (m * j).sum()
    assert cov(corr) > 0.1 and abs(cov(indep)) < 1e-6
    # Correlación positiva => más probabilidad en los extremos del total
    p_alto = lambda m: m[(i + j) > 6.5].sum()
    assert p_alto(corr) > p_alto(indep)


def test_copula_entre_estadisticas():
    from apuestas.combinadas import parsear_pierna, probabilidad_combinada

    preds = {"goles": Prediccion(1.6, 1.3), "remates_puerta": Prediccion(5.0, 4.0, 1.2)}
    cols = ["goles_l", "goles_v", "remates_puerta_l", "remates_puerta_v"]
    corr = pd.DataFrame(np.eye(4), index=cols, columns=cols)
    corr.loc["goles_l", "remates_puerta_l"] = corr.loc["remates_puerta_l", "goles_l"] = 0.5
    goles_local = parsear_pierna("goles local mas 1.5")
    sot_local, sot_visit = parsear_pierna("remates_puerta local mas 4.5"), parsear_pierna("remates_puerta visitante mas 3.5")
    indep = probabilidad_combinada(preds, [goles_local, sot_local])
    assert probabilidad_combinada(preds, [goles_local, sot_local], corr) > indep * 1.1
    # Sin correlación entre goles locales y remates visitantes => igual que independiente
    a = probabilidad_combinada(preds, [goles_local, sot_visit], corr)
    assert a == pytest.approx(probabilidad_combinada(preds, [goles_local, sot_visit]), rel=0.02)


def test_jugador_desde_cuotas():
    from apuestas.ancla import lambdas_jugadores
    from apuestas.combinadas import parsear_pierna, probabilidad_combinada
    from apuestas.odds_api import Cuota, Evento
    from scipy import stats as st

    ev = Evento("x", "", "A", "B")
    for linea in (0.5, 1.5, 2.5):
        p = st.poisson.sf(np.floor(linea), 1.6) * 1.06
        ev.cuotas.append(Cuota(Seleccion("remates_puerta", "jugador", "mas", linea, jugador="Ousmane Dembélé"), 1 / p, "X"))
    lams = lambdas_jugadores(ev)
    assert lams[("Ousmane Dembélé", "remates_puerta")] == pytest.approx(1.6, abs=0.15)
    pierna = parsear_pierna("jugador local remates_puerta mas 1.5 dembele")
    p = probabilidad_combinada({"goles": Prediccion(1.5, 1.2)}, [pierna], None, lams)
    assert p == pytest.approx(st.poisson.sf(1, 1.6), abs=0.04)


def test_directo_reparte_el_partido():
    from apuestas.directo import fraccion_restante, restante

    assert fraccion_restante("goles", 0) == pytest.approx(1)
    assert fraccion_restante("goles", 95) == pytest.approx(0)
    # Más goles y, sobre todo, más tarjetas en la segunda parte
    assert 0.52 < fraccion_restante("goles", 45) < 0.58
    assert fraccion_restante("tarjetas", 45) > fraccion_restante("goles", 45)
    r = restante(Prediccion(5.0, 4.0, 1.4), 0.5)
    assert (r.local, r.visitante, r.dispersion) == pytest.approx((2.5, 2.0, 1.2))


def test_directo_suma_lo_que_ya_ha_pasado():
    from apuestas.combinadas import parsear_pierna, probabilidad_combinada

    preds = {"corners": Prediccion(2.0, 1.5, 1.2), "goles": Prediccion(0.8, 0.6)}
    mas45 = parsear_pierna("corners local mas 4.5")
    sin_nada = probabilidad_combinada(preds, [mas45])
    con_tres = probabilidad_combinada(preds, [mas45], marcador={"corners": (3, 0)})
    # Con 3 córners hechos basta con 2 más
    assert con_tres == pytest.approx(probabilidad(preds["corners"], parsear_pierna("corners local mas 1.5"))[0])
    assert con_tres > sin_nada
    assert probabilidad_combinada(preds, [mas45], marcador={"corners": (5, 0)}) == pytest.approx(1)
    # Una pierna de jugador ya ganada no cambia la probabilidad de las demás
    jugador = parsear_pierna("jugador local remates_puerta mas 1.5 Ferran")
    lams = {("Ferran Torres", "remates_puerta"): 0.5}
    p = probabilidad_combinada(preds, [mas45, jugador], None, lams, {"corners": (3, 0)},
                               {("Ferran Torres", "remates_puerta"): 2})
    assert p == pytest.approx(con_tres)


def test_gana_seleccion_con_resultado_real():
    from apuestas.backtest import _gana_seleccion
    from apuestas.combinadas import parsear_pierna

    fila = {"FTHG": 2, "FTAG": 1, "HC": 6, "AC": 3}
    assert _gana_seleccion(parsear_pierna("goles 1x2 1"), fila)
    assert _gana_seleccion(parsear_pierna("goles btts si"), fila)
    assert not _gana_seleccion(parsear_pierna("goles total mas 3.5"), fila)
    assert _gana_seleccion(parsear_pierna("corners local mas 5.5"), fila)


def test_oportunidades_frente_al_consenso():
    from apuestas.mercado import oportunidades
    from apuestas.odds_api import Cuota, Evento

    ev = Evento("x", "", "A", "B")
    mas, menos = Seleccion("goles", "total", "mas", 2.5), Seleccion("goles", "total", "menos", 2.5)
    for k in range(6):
        ev.cuotas += [Cuota(mas, 1.90, f"C{k}"), Cuota(menos, 1.90, f"C{k}")]
    ev.cuotas += [Cuota(mas, 2.20, "Generosa"), Cuota(menos, 1.70, "Generosa")]
    ops = oportunidades(ev, min_casas=5)
    assert [(o.casa, o.seleccion) for o in ops] == [("Generosa", mas)]
    assert ops[0].valor == pytest.approx(0.5 * 2.20 - 1, abs=0.01)


def test_remates_siguen_a_los_goles_del_mercado():
    from apuestas.ancla import anclar

    preds = {"goles": Prediccion(1.5, 1.0), "remates_puerta": Prediccion(5.0, 3.5, 1.2)}
    objetivo = {Seleccion("goles", "total", "mas", 2.5): (0.75, 10), Seleccion("goles", "1x2", "1"): (0.60, 10)}
    nuevas, _ = anclar(preds, objetivo)
    sube = nuevas["goles"].local / preds["goles"].local
    assert sube > 1.2
    assert nuevas["remates_puerta"].local / 5.0 == pytest.approx(sube ** 0.6, rel=1e-6)
