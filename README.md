# Recomendador de apuestas de fútbol

Modelo estadístico que calcula probabilidades para muchos mercados de un partido
(1X2, doble oportunidad, ambos marcan, goles, **córners, tarjetas, remates, remates
a puerta, faltas**), las compara con las cuotas de las casas y sugiere las apuestas
con valor esperado positivo y su tamaño (Kelly fraccionado).

## Cómo funciona

- **Datos**: resultados y estadísticas de [football-data.co.uk](https://www.football-data.co.uk) (gratis, sin clave).
- **Modelo**: para cada estadística, regresión de Poisson con fuerza de ataque/defensa por
  equipo y ventaja de local, con más peso para los partidos recientes. Si los datos
  tienen más varianza que una Poisson (córners, tarjetas) se usa binomial negativa.
- **Cuotas en vivo**: [The Odds API](https://the-odds-api.com) (plan gratuito: 500 créditos/mes).
- **Valor**: `EV = probabilidad × cuota − 1`. Recomienda si EV ≥ 5 %, cuota ≤ 6 y apuesta
  ¼ de Kelly con un máximo del 3 % del bankroll.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # y pon tu ODDS_API_KEY
```

## Uso

```bash
# Probabilidades y cuotas justas de todos los mercados de un partido
.venv/bin/python -m apuestas predecir "Real Madrid" "Barcelona"
.venv/bin/python -m apuestas predecir "Real Madrid" "Barcelona" --solo corners tarjetas

# ¿Tiene valor una cuota que he visto? (córners más de 9,5 a 1,85)
.venv/bin/python -m apuestas cuota "Atletico Madrid" Betis corners total mas 1.85 --linea 9.5

# Buscar valor en las cuotas actuales (--extra añade córners, tarjetas, ambos marcan...)
.venv/bin/python -m apuestas valor --liga SP1 --extra

# Backtest en una temporada pasada
.venv/bin/python -m apuestas backtest --temporada 2526

# Partido de Champions: mezclar ligas y anclar el modelo a las cuotas actuales
.venv/bin/python -m apuestas predecir PSG Barcelona --liga F1 SP1 --mercado-1x2 2.40 4.40 2.375

# Combinadas (crear apuesta / MyMatch): probabilidad real y valor frente a la cuota de la casa
.venv/bin/python -m apuestas combinada PSG Barcelona --liga SP1 F1 --anclar --jugadores \
    --competicion soccer_uefa_champs_league --regiones eu,uk,us \
    --pierna "corners local mas 4.5" --pierna "jugador local remates_puerta mas 1.5 Dembele" --cuota 3.90

# Buscar combinadas con una cuota objetivo
.venv/bin/python -m apuestas sugerir PSG Barcelona --liga SP1 F1 --anclar \
    --competicion soccer_uefa_champs_league --regiones eu,uk,us --objetivo 3

# Equipos de la liga (nombres tal como los usa el modelo)
.venv/bin/python -m apuestas equipos --liga E0
```

Ligas: `SP1` LaLiga, `SP2` Segunda, `E0` Premier, `E1` Championship, `I1` Serie A,
`D1` Bundesliga, `F1` Ligue 1, `N1` Eredivisie, `P1` Portugal.

Cada ejecución de `valor` guarda sus recomendaciones en `datos/registro.csv` para
poder comprobar más adelante si el modelo acierta en córners y tarjetas.

## Combinadas

- `--anclar` ajusta goles, córners y tarjetas a las cuotas sin margen de todas las casas
  (The Odds API); el modelo deja de opinar sobre lo que el mercado ya sabe y se usa para
  calcular de forma coherente la probabilidad de que pasen varias cosas a la vez.
- Local y visitante se relacionan con una cópula de Frank (tarjetas +0,2, córners −0,1),
  medida en los datos y validada fuera de muestra.
- Entre estadísticas se usa una cópula gaussiana con las correlaciones históricas de los
  residuos (goles–remates a puerta del mismo equipo +0,5, goles–córners propios −0,1).
- `--jugadores` añade remates, remates a puerta y goleadores (casas de EE. UU.); la media
  de cada jugador se ajusta a sus cuotas y se relaciona con la de su equipo.

## Resultados del backtest (importante)

Comparado con las cuotas de cierre, **el modelo no bate al mercado en 1X2 ni en
más/menos 2,5 goles**: en LaLiga, Premier, Serie A y Bundesliga (temporadas 24/25
y 25/26, unas 2.900 partidos) el ROI simulado va de −5 % a −13 %, según el umbral
y si se coge la cuota media o la mejor. Las casas fijan esas líneas con mucha más
información (alineaciones, lesiones, dinero apostado).

Para córners, tarjetas y remates no hay cuotas históricas gratis, así que no se
puede simular el ROI; el modelo predice los totales solo un 1–3 % mejor que la
media de la liga. Son mercados menos eficientes, pero que tengan valor está
**sin demostrar**. Apuesta solo lo que estés dispuesto a perder.
