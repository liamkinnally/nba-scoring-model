# NBA Scoring Model

Python project for predicting NBA player points, assists, and rebounds using recent performance, opponent history, team form, rest, and home/away status.

## What it does

- Pulls NBA game and box score data from NBA-hosted endpoints
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

The project uses NBA-hosted JSON endpoints for live scoreboard and box score data. Historical game IDs are pulled from the NBA stats `leaguegamelog` endpoint and then matched to the NBA live-data box score endpoint.

The main endpoints used are:

```text
https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json
https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json
https://stats.nba.com/stats/leaguegamelog
```

These endpoints are hosted by NBA.com but are not a documented public API, so their response format or availability can change.

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

You can test the ingestion with only a few games first:

```bash
python -m nba_scoring_model.cli ingest-season --season 2025-26 --limit 10
```

To ingest today's games:

```bash
python -m nba_scoring_model.cli ingest-today --include-boxscores
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

This is mainly for checking that the pipeline runs without needing to download NBA data first.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
