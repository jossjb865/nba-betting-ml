# NBA Betting ML — Sistema Predictivo (SportsDataIO + CatBoost + LSTM + Poisson Bivariada)

Sistema completo de predicción de mercados NBA **Moneyline** y **Totales (Over/Under)**
alimentado exclusivamente por datos reales de la API oficial de
[SportsDataIO](https://sportsdata.io/nba-api). Incluye capa de gestión de riesgo
(**Criterio de Kelly fraccionado + Valor Esperado**), validación walk-forward y
entrega diaria de pronósticos a **Telegram** mediante GitHub Actions.

> La documentación técnica completa (arquitectura, feature engineering,
> estrategia de ensamble, validación) está en
> [`docs/DOCUMENTO_TECNICO.md`](docs/DOCUMENTO_TECNICO.md).

## Arquitectura en una línea

```
SportsDataIO API → Ingestión (Parquet) → Feature Engineering →
  ├─ CatBoost (Moneyline + Totales)        ┐
  ├─ LSTM Momentum (secuencias por equipo) ├─→ Fusion MLP (stacking calibrado) →
  └─ Poisson Bivariada (marcadores)        ┘
→ Kelly fraccionado + EV → Telegram (vía GitHub Actions)
```

## Estructura

```
├── config/settings.yaml            # todos los parámetros operativos
├── src/
│   ├── sportsdataio/               # cliente API + registro de endpoints + esquemas
│   ├── ingestion/ingest.py         # descarga y persistencia (data/raw/*.parquet)
│   ├── features/                   # builder tabular + secuencias LSTM
│   ├── models/                     # catboost | lstm_momentum | bivariate_poisson | fusion_mlp
│   ├── staking/kelly.py            # Kelly fraccionado, EV, devig, filtros
│   ├── validation/walk_forward.py  # validación temporal + métricas + simulación ROI
│   ├── notify/telegram.py          # Bot API de Telegram
│   └── pipelines/                  # train.py | predict_daily.py
├── .github/workflows/
│   ├── daily_predictions.yml       # cron diario → predicciones → Telegram
│   └── weekly_retrain.yml          # cron semanal → reentrenamiento + artefactos
└── docs/                           # documento técnico + diagrama de arquitectura
```

## Puesta en marcha

1. **API key**: crea una cuenta en [sportsdata.io](https://sportsdata.io/nba-api) y
   obtén tu subscription key de NBA.
2. **Bot de Telegram**: crea un bot con [@BotFather](https://t.me/BotFather) y anota
   el token. Obtén tu `chat_id` (p. ej. escribiendo a [@userinfobot](https://t.me/userinfobot)).
3. **Clona el repo** y copia `.env.example` a `.env` con tus credenciales.
4. **Instala dependencias**: `pip install -r requirements.txt`
5. **Entrenamiento inicial** (backfill histórico + walk-forward, tarda según nº de temporadas):
   ```bash
   python -m src.pipelines.train
   ```
6. **Predicción diaria**:
   ```bash
   python -m src.pipelines.predict_daily            # hoy
   python -m src.pipelines.predict_daily --date 2026-01-15
   ```

## GitHub Actions

Configura en **Settings → Secrets and variables → Actions**:

| Secreto | Descripción |
|---|---|
| `SPORTSDATAIO_API_KEY` | Subscription key de SportsDataIO (NBA) |
| `TELEGRAM_BOT_TOKEN` | Token del bot creado con @BotFather |
| `TELEGRAM_CHAT_ID` | Chat/canal donde llegan los pronósticos |

- `weekly_retrain.yml` (domingos 06:00 UTC): backfill incremental, walk-forward,
  stacking y publicación de artefactos (`model-artifacts`).
- `daily_predictions.yml` (16:00 UTC diario, ~mediodía ET): descarga los artefactos,
  evalúa los partidos del día con odds y lesiones en vivo, aplica Kelly/EV y envía
  el informe a Telegram. También se puede lanzar manualmente con fecha objetivo.

## Reglas de datos

- **Prohibido el dato sintético**: todo feature, identificador (`GameID`, `TeamID`,
  `PlayerID`), cuota y métrica proviene de los endpoints reales documentados en
  `src/sportsdataio/endpoints.py`.
- Las features respetan el ciclo de vida del dato: **Pregame** (odds, lesiones,
  lineups), **Live** (deltas), **Postgame** (box scores finales → entrenamiento).
- Anti-leakage: medias móviles con `shift(1)` y validación walk-forward por fecha.

## Aviso

Proyecto con fines educativos y de investigación cuantitativa. No constituye
asesoramiento financiero ni garantiza beneficios; las apuestas conllevan riesgo
de pérdida. Consulta los términos de uso de SportsDataIO para el uso de su API.
