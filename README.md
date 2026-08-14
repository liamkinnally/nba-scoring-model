# NBA Scoring Model

Python project for predicting NBA player points, assists, and rebounds using recent performance, opponent history, team form, rest, and home/away status.

## What it does

- Pulls NBA schedules and box score data from ESPN JSON endpoints
- Stores games, teams, players, and player box scores in SQLite with SQLAlchemy
- Builds rolling player and matchup features
- Trains Gradient Boosting regression models for points, assists, and rebounds
- Uses chronological train/test splits so later games are not used to predict earlier games
- Reports MAE, RMSE, and R²
- Saves trained models with joblib

## Project structure

```text
nba-scoring-model/
├── nba_scoring_model/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── modeling/
│   ├── cli.py
│   └── config.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Data

The ingestion pipeline uses ESPN's NBA scoreboard endpoint to find games by date and the ESPN game summary endpoint to collect box score data.

```text
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary
```

These endpoints are not a documented public API, so their response format or availability can change.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Load historical data

For example, to load the 2025-26 regular season:

```bash
python -m nba_scoring_model.cli ingest-season --season 2025-26
```

You can test ingestion with only a few games first:

```bash
python -m nba_scoring_model.cli ingest-season --season 2025-26 --limit 10
```

Or limit the date range:

```bash
python -m nba_scoring_model.cli ingest-season \
  --season 2025-26 \
  --start 2025-10-21 \
  --end 2025-10-31
```

## Train the models

Once the database has historical data:

```bash
python -m nba_scoring_model.cli train \
  --start 2025-10-01 \
  --end 2026-04-15
```

The training pipeline holds out the most recent game dates for testing. Hyperparameter tuning uses expanding date-based folds instead of randomly mixing games from different points in the season.

## Features

Current features include:

- 5-game and 10-game averages for points, assists, and rebounds
- recent minutes
- recent shot/turnover activity proxy
- recent performance against the opponent
- team scoring and defensive form
- recent team win percentage
- rest days and back-to-back indicator
- home/away indicator
- optional betting total and spread fields

## Demo

There is also an offline demo that fills the database with generated data and runs the full training pipeline:

```bash
python -m nba_scoring_model.cli demo
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
