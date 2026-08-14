import os
from pathlib import Path


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/nba_predictions.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
API_CALLS_PER_MINUTE = int(os.getenv("API_CALLS_PER_MINUTE", "60"))

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
