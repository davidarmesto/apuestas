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
