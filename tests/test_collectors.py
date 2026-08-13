from nba_scoring_model.api.collectors import NBADataCollector


def test_extract_historical_game_ids(tmp_path):
    class FakeClient:
        def get_json(self, url, params=None):
            return {
                "resultSets": [
                    {
                        "name": "LeagueGameLog",
                        "headers": ["TEAM_ID", "GAME_ID", "GAME_DATE"],
                        "rowSet": [
                            [1, "0022500001", "2025-10-21"],
                            [2, "0022500001", "2025-10-21"],
                            [3, "0022500002", "2025-10-21"],
                            [4, "0022500002", "2025-10-21"],
                        ],
                    }
                ]
            }

    from nba_scoring_model.data.database import DatabaseManager

    db = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
    collector = NBADataCollector(FakeClient(), db, "score", "box/{game_id}", "games")
    game_ids = collector.fetch_historical_game_ids("2025-26")

    assert game_ids == ["0022500001", "0022500002"]


def test_status_is_normalized():
    assert NBADataCollector._normalize_status("Final") == "final"
    assert NBADataCollector._normalize_status(3) == "final"
    assert NBADataCollector._normalize_status(2) == "live"
