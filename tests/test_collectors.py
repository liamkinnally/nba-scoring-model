from datetime import date

from nba_scoring_model.api.collectors import NBADataCollector
from nba_scoring_model.data.database import DatabaseManager


def test_extract_historical_game_ids(tmp_path):
    class FakeClient:
        def get_json(self, url, params=None):
            return {
                "events": [
                    {"id": "401000001", "season": {"type": 2}},
                    {"id": "401000002", "season": {"type": 2}},
                ]
            }

    db = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
    collector = NBADataCollector(FakeClient(), db, "scoreboard", "summary")
    game_ids = collector.fetch_historical_game_ids(
        "2025-26",
        date_from="10/21/2025",
        date_to="10/21/2025",
    )

    assert game_ids == ["401000001", "401000002"]


def test_parse_scoreboard_event():
    event = {
        "id": "401809234",
        "date": "2025-10-22T23:00Z",
        "competitions": [
            {
                "status": {"type": {"name": "STATUS_FINAL"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": "119",
                        "team": {"id": "18", "displayName": "New York Knicks", "abbreviation": "NY"},
                    },
                    {
                        "homeAway": "away",
                        "score": "111",
                        "team": {"id": "5", "displayName": "Cleveland Cavaliers", "abbreviation": "CLE"},
                    },
                ],
            }
        ],
    }

    parsed = NBADataCollector._parse_scoreboard_event(event)
    assert parsed["game_id"] == "401809234"
    assert parsed["home_team_id"] == "18"
    assert parsed["away_team_id"] == "5"
    assert parsed["game_status"] == "final"
    assert parsed["home_score"] == 119


def test_status_is_normalized():
    assert NBADataCollector._normalize_status("STATUS_FINAL") == "final"
    assert NBADataCollector._normalize_status("STATUS_SCHEDULED") == "scheduled"
    assert NBADataCollector._normalize_status("STATUS_IN_PROGRESS") == "live"
