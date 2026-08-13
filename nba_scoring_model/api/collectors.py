from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.models import Game, Player, PlayerGameStats, Team

from .client import JSONAPIClient


class NBADataCollector:
    def __init__(
        self,
        client: JSONAPIClient,
        db_manager: DatabaseManager,
        scoreboard_url: str,
        boxscore_url_template: str,
        league_game_log_url: str,
    ):
        self.client = client
        self.db_manager = db_manager
        self.scoreboard_url = scoreboard_url
        self.boxscore_url_template = boxscore_url_template
        self.league_game_log_url = league_game_log_url

    def fetch_scoreboard(self) -> List[Dict[str, Any]]:
        payload = self.client.get_json(self.scoreboard_url)
        games = payload.get("scoreboard", {}).get("games", [])
        return [self._parse_scoreboard_game(game) for game in games]

    def fetch_boxscore(self, game_id: str) -> Dict[str, Any]:
        url = self.boxscore_url_template.format(game_id=game_id)
        return self.client.get_json(url)

    def fetch_historical_game_ids(
        self,
        season: str,
        season_type: str = "Regular Season",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[str]:
        params = {
            "Counter": 0,
            "DateFrom": date_from or "",
            "DateTo": date_to or "",
            "Direction": "ASC",
            "LeagueID": "00",
            "PlayerOrTeam": "T",
            "Season": season,
            "SeasonType": season_type,
            "Sorter": "DATE",
        }
        payload = self.client.get_json(self.league_game_log_url, params=params)
        headers, rows = self._extract_result_set(payload, "LeagueGameLog")

        if "GAME_ID" not in headers:
            raise ValueError("GAME_ID was not returned by leaguegamelog")

        game_id_index = headers.index("GAME_ID")
        game_ids = []
        seen = set()
        for row in rows:
            game_id = str(row[game_id_index])
            if game_id not in seen:
                seen.add(game_id)
                game_ids.append(game_id)
        return game_ids

    def ingest_scoreboard(self) -> List[str]:
        games = self.fetch_scoreboard()
        with self.db_manager.get_session() as session:
            for item in games:
                for prefix in ("home", "away"):
                    session.merge(
                        Team(
                            team_id=item[f"{prefix}_team_id"],
                            team_name=item[f"{prefix}_team_name"],
                            abbreviation=item[f"{prefix}_team_abbreviation"],
                        )
                    )

                session.merge(
                    Game(
                        game_id=item["game_id"],
                        date=item["date"],
                        home_team_id=item["home_team_id"],
                        away_team_id=item["away_team_id"],
                        game_status=item["game_status"],
                        home_score=item.get("home_score"),
                        away_score=item.get("away_score"),
                    )
                )
        return [item["game_id"] for item in games]

    def ingest_boxscore(self, game_id: str) -> int:
        payload = self.fetch_boxscore(game_id)
        game = payload.get("game", {})
        if not game:
            raise ValueError(f"No game data returned for {game_id}")

        game_id = str(game.get("gameId") or game_id)
        game_date = self._parse_datetime(game.get("gameTimeUTC") or game.get("gameEt"))
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})

        with self.db_manager.get_session() as session:
            for team_data in (home, away):
                team_id = str(team_data.get("teamId"))
                session.merge(
                    Team(
                        team_id=team_id,
                        team_name=team_data.get("teamName") or team_data.get("teamCity") or team_id,
                        abbreviation=team_data.get("teamTricode"),
                    )
                )

            session.merge(
                Game(
                    game_id=game_id,
                    date=game_date,
                    home_team_id=str(home.get("teamId")),
                    away_team_id=str(away.get("teamId")),
                    game_status=self._normalize_status(game.get("gameStatusText") or game.get("gameStatus")),
                    home_score=self._int_or_none(home.get("score")),
                    away_score=self._int_or_none(away.get("score")),
                )
            )

            count = 0
            for team_data in (home, away):
                team_id = str(team_data.get("teamId"))
                for player_data in team_data.get("players", []):
                    player_id = str(player_data.get("personId"))
                    if not player_id or player_id == "None":
                        continue

                    name = player_data.get("name") or player_data.get("nameI") or player_id
                    session.merge(
                        Player(
                            player_id=player_id,
                            name=name,
                            team_id=team_id,
                            position=player_data.get("position"),
                        )
                    )

                    stats = player_data.get("statistics") or {}
                    minutes = self._minutes_to_float(stats.get("minutesCalculated") or stats.get("minutes"))
                    fga = self._int_or_zero(stats.get("fieldGoalsAttempted"))
                    fta = self._int_or_zero(stats.get("freeThrowsAttempted"))
                    turnovers = self._int_or_zero(stats.get("turnovers"))

                    existing = session.scalar(
                        select(PlayerGameStats).where(
                            PlayerGameStats.game_id == game_id,
                            PlayerGameStats.player_id == player_id,
                        )
                    )
                    stat_row = existing or PlayerGameStats(
                        game_id=game_id,
                        player_id=player_id,
                        team_id=team_id,
                    )
                    stat_row.team_id = team_id
                    stat_row.minutes_played = minutes
                    stat_row.points = self._int_or_zero(stats.get("points"))
                    stat_row.assists = self._int_or_zero(stats.get("assists"))
                    stat_row.rebounds = self._int_or_zero(stats.get("reboundsTotal"))
                    stat_row.usage_rate = self._activity_proxy(fga, fta, turnovers, minutes)
                    stat_row.field_goal_attempts = fga
                    stat_row.free_throw_attempts = fta
                    stat_row.turnovers = turnovers
                    session.add(stat_row)
                    count += 1

            return count

    def ingest_season(
        self,
        season: str,
        season_type: str = "Regular Season",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        game_ids = self.fetch_historical_game_ids(season, season_type, date_from, date_to)
        if limit is not None:
            game_ids = game_ids[:limit]

        games_ingested = 0
        player_rows = 0
        failures = []

        for game_id in game_ids:
            try:
                player_rows += self.ingest_boxscore(game_id)
                games_ingested += 1
            except Exception as exc:
                failures.append({"game_id": game_id, "error": str(exc)})

        return {
            "games_found": len(game_ids),
            "games_ingested": games_ingested,
            "player_stat_rows_ingested": player_rows,
            "failures": failures,
        }

    @staticmethod
    def _extract_result_set(payload: Dict[str, Any], name: str):
        result_sets = payload.get("resultSets")
        if result_sets is None:
            result_sets = payload.get("resultSet")

        if isinstance(result_sets, dict):
            result_sets = [result_sets]
        if not isinstance(result_sets, list):
            raise ValueError("NBA stats response did not contain a result set")

        for result in result_sets:
            if result.get("name") == name or len(result_sets) == 1:
                return result.get("headers", []), result.get("rowSet", [])

        raise ValueError(f"Result set '{name}' was not returned")

    @staticmethod
    def _parse_scoreboard_game(game: Dict[str, Any]) -> Dict[str, Any]:
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        return {
            "game_id": str(game.get("gameId")),
            "date": NBADataCollector._parse_datetime(game.get("gameTimeUTC") or game.get("gameEt")),
            "game_status": NBADataCollector._normalize_status(game.get("gameStatusText") or game.get("gameStatus")),
            "home_team_id": str(home.get("teamId")),
            "home_team_name": home.get("teamName") or home.get("teamCity") or str(home.get("teamId")),
            "home_team_abbreviation": home.get("teamTricode"),
            "home_score": NBADataCollector._int_or_none(home.get("score")),
            "away_team_id": str(away.get("teamId")),
            "away_team_name": away.get("teamName") or away.get("teamCity") or str(away.get("teamId")),
            "away_team_abbreviation": away.get("teamTricode"),
            "away_score": NBADataCollector._int_or_none(away.get("score")),
        }

    @staticmethod
    def _normalize_status(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        if text in {"3", "final", "final/ot", "final/2ot", "final/3ot"} or text.startswith("final"):
            return "final"
        if text in {"1", "scheduled"}:
            return "scheduled"
        if text in {"2", "live", "in progress"}:
            return "live"
        return text

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if not value:
            raise ValueError("Game time was missing from NBA response")
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text)

    @staticmethod
    def _minutes_to_float(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        text = str(value)
        if text.startswith("PT") and text.endswith("M"):
            try:
                return float(text[2:-1])
            except ValueError:
                return 0.0
        if text.startswith("PT") and "M" in text:
            try:
                minute_part = text[2:].split("M", 1)[0]
                second_part = text.split("M", 1)[1].replace("S", "")
                return float(minute_part) + (float(second_part) / 60.0 if second_part else 0.0)
            except ValueError:
                return 0.0
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            try:
                return float(minutes) + float(seconds) / 60.0
            except ValueError:
                return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _int_or_none(value: Any):
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _activity_proxy(fga: int, fta: int, turnovers: int, minutes: float):
        if minutes <= 0:
            return None
        return round((fga + 0.44 * fta + turnovers) / minutes, 4)
