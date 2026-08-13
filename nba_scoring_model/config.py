import os
from pathlib import Path


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/nba_predictions.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
API_CALLS_PER_MINUTE = int(os.getenv("API_CALLS_PER_MINUTE", "20"))

NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_BOXSCORE_URL_TEMPLATE = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
NBA_LEAGUE_GAME_LOG_URL = "https://stats.nba.com/stats/leaguegamelog"
