from datetime import date, datetime, timedelta
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
        summary_url: str,
    ):
        self.client = client
        self.db_manager = db_manager
        self.scoreboard_url = scoreboard_url
        self.summary_url = summary_url

    def fetch_scoreboard(self, game_date: Optional[date] = None) -> List[Dict[str, Any]]:
        params = None
        if game_date is not None:
            params = {"dates": game_date.strftime("%Y%m%d")}

        payload = self.client.get_json(self.scoreboard_url, params=params)
        return [self._parse_scoreboard_event(event) for event in payload.get("events", [])]

    def fetch_summary(self, game_id: str) -> Dict[str, Any]:
        return self.client.get_json(self.summary_url, params={"event": game_id})

    def fetch_historical_game_ids(
        self,
        season: str,
        season_type: str = "Regular Season",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        start, end = self._season_date_range(season, season_type, date_from, date_to)
        expected_type = self._espn_season_type(season_type)
        nba_team_ids = {str(i) for i in range(1, 31)}

        game_ids: List[str] = []
        seen = set()
        current = start

        while current <= end:
            payload = self.client.get_json(
                self.scoreboard_url,
                params={"dates": current.strftime("%Y%m%d")},
            )

            for event in payload.get("events", []):
                event_type = event.get("season", {}).get("type")
                if (
                    expected_type is not None
                    and event_type is not None
                    and event_type != expected_type
                ):
                    continue

                competitions = event.get("competitions") or []
                if not competitions:
                    continue

                competition = competitions[0]

                status = competition.get("status", {}).get("type", {})
                if not status.get("completed", False):
                    continue

                if season_type == "Regular Season":
                    competition_type = competition.get("type", {}).get("abbreviation")
                    if competition_type == "CC":
                        continue

                    competitors = competition.get("competitors") or []
                    team_ids = {
                        str(item.get("team", {}).get("id"))
                        for item in competitors
                        if item.get("team", {}).get("id") is not None
                    }

                    if len(team_ids) != 2 or not team_ids.issubset(nba_team_ids):
                        continue

                game_id = str(event.get("id") or "")
                if game_id and game_id not in seen:
                    seen.add(game_id)
                    game_ids.append(game_id)

                    if limit is not None and len(game_ids) >= limit:
                        return game_ids

            current += timedelta(days=1)

        return game_ids

    def ingest_scoreboard(self, game_date: Optional[date] = None) -> List[str]:
        games = self.fetch_scoreboard(game_date)
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
        payload = self.fetch_summary(game_id)
        competition = self._summary_competition(payload)
        competitors = competition.get("competitors", [])

        home = next((item for item in competitors if item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if item.get("homeAway") == "away"), None)
        if home is None or away is None:
            raise ValueError(f"Could not find home/away teams for {game_id}")

        game_date = self._parse_datetime(competition.get("date"))
        game_status = self._normalize_status(
            competition.get("status", {}).get("type", {}).get("name")
            or competition.get("status", {}).get("type", {}).get("description")
        )

        home_team = home.get("team", {})
        away_team = away.get("team", {})
        home_team_id = str(home_team.get("id"))
        away_team_id = str(away_team.get("id"))

        pickcenter = payload.get("pickcenter") or []
        odds = pickcenter[0] if pickcenter else {}
        vegas_total = self._float_or_none(odds.get("overUnder"))
        vegas_spread = self._float_or_none(odds.get("spread"))

        with self.db_manager.get_session() as session:
            for team_data in (home_team, away_team):
                team_id = str(team_data.get("id"))
                session.merge(
                    Team(
                        team_id=team_id,
                        team_name=team_data.get("displayName")
                        or team_data.get("shortDisplayName")
                        or team_data.get("name")
                        or team_id,
                        abbreviation=team_data.get("abbreviation"),
                    )
                )

            session.merge(
                Game(
                    game_id=str(game_id),
                    date=game_date,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    game_status=game_status,
                    home_score=self._int_or_none(home.get("score")),
                    away_score=self._int_or_none(away.get("score")),
                    vegas_total=vegas_total,
                    vegas_spread=vegas_spread,
                )
            )

            count = 0
            seen_players = set()
            for team_block in payload.get("boxscore", {}).get("players", []):
                team_data = team_block.get("team", {})
                team_id = str(team_data.get("id"))

                for stat_group in team_block.get("statistics", []):
                    keys = stat_group.get("keys") or stat_group.get("labels") or []

                    for athlete_row in stat_group.get("athletes", []):
                        if athlete_row.get("didNotPlay"):
                            continue

                        athlete = athlete_row.get("athlete", {})
                        player_id = str(athlete.get("id") or "")
                        if not player_id or player_id in seen_players:
                            continue

                        values = athlete_row.get("stats") or []
                        if not values:
                            continue

                        seen_players.add(player_id)
                        stat_map = {str(key): value for key, value in zip(keys, values)}

                        minutes = self._minutes_to_float(
                            self._stat_value(stat_map, "minutes", "MIN")
                        )
                        field_goals = self._stat_value(
                            stat_map,
                            "fieldGoalsMade-fieldGoalsAttempted",
                            "FG",
                        )
                        free_throws = self._stat_value(
                            stat_map,
                            "freeThrowsMade-freeThrowsAttempted",
                            "FT",
                        )
                        fga = self._attempts_from_made_attempted(field_goals)
                        fta = self._attempts_from_made_attempted(free_throws)
                        turnovers = self._int_or_zero(
                            self._stat_value(stat_map, "turnovers", "TO")
                        )

                        position = athlete.get("position") or {}
                        if isinstance(position, dict):
                            position = position.get("abbreviation") or position.get("name")

                        session.merge(
                            Player(
                                player_id=player_id,
                                name=athlete.get("displayName")
                                or athlete.get("shortName")
                                or player_id,
                                team_id=team_id,
                                position=position,
                            )
                        )

                        existing = session.scalar(
                            select(PlayerGameStats).where(
                                PlayerGameStats.game_id == str(game_id),
                                PlayerGameStats.player_id == player_id,
                            )
                        )
                        stat_row = existing or PlayerGameStats(
                            game_id=str(game_id),
                            player_id=player_id,
                            team_id=team_id,
                        )
                        stat_row.team_id = team_id
                        stat_row.minutes_played = minutes
                        stat_row.points = self._int_or_zero(
                            self._stat_value(stat_map, "points", "PTS")
                        )
                        stat_row.assists = self._int_or_zero(
                            self._stat_value(stat_map, "assists", "AST")
                        )
                        stat_row.rebounds = self._int_or_zero(
                            self._stat_value(stat_map, "rebounds", "REB")
                        )
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
        game_ids = self.fetch_historical_game_ids(
            season,
            season_type,
            date_from,
            date_to,
            limit=limit,
        )

        games_ingested = 0
        player_rows = 0
        failures = []

        for index, game_id in enumerate(game_ids, start=1):
            print(f"Fetching game {index}/{len(game_ids)}: {game_id}")
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
    def _summary_competition(payload: Dict[str, Any]) -> Dict[str, Any]:
        competitions = payload.get("header", {}).get("competitions", [])
        if not competitions:
            raise ValueError("ESPN summary did not contain competition data")
        return competitions[0]

    @staticmethod
    def _parse_scoreboard_event(event: Dict[str, Any]) -> Dict[str, Any]:
        competitions = event.get("competitions", [])
        if not competitions:
            raise ValueError("ESPN scoreboard event did not contain competition data")
        competition = competitions[0]
        competitors = competition.get("competitors", [])

        home = next((item for item in competitors if item.get("homeAway") == "home"), {})
        away = next((item for item in competitors if item.get("homeAway") == "away"), {})
        home_team = home.get("team", {})
        away_team = away.get("team", {})

        return {
            "game_id": str(event.get("id")),
            "date": NBADataCollector._parse_datetime(event.get("date") or competition.get("date")),
            "game_status": NBADataCollector._normalize_status(
                competition.get("status", {}).get("type", {}).get("name")
                or event.get("status", {}).get("type", {}).get("name")
            ),
            "home_team_id": str(home_team.get("id")),
            "home_team_name": home_team.get("displayName")
            or home_team.get("shortDisplayName")
            or home_team.get("name")
            or str(home_team.get("id")),
            "home_team_abbreviation": home_team.get("abbreviation"),
            "home_score": NBADataCollector._int_or_none(home.get("score")),
            "away_team_id": str(away_team.get("id")),
            "away_team_name": away_team.get("displayName")
            or away_team.get("shortDisplayName")
            or away_team.get("name")
            or str(away_team.get("id")),
            "away_team_abbreviation": away_team.get("abbreviation"),
            "away_score": NBADataCollector._int_or_none(away.get("score")),
        }

    @staticmethod
    def _season_date_range(
        season: str,
        season_type: str,
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> tuple[date, date]:
        start_year = int(season.split("-", 1)[0])
        end_year = start_year + 1
        season_type_lower = season_type.strip().lower()

        if season_type_lower in {"preseason", "pre-season"}:
            default_start = date(start_year, 9, 20)
            default_end = date(start_year, 10, 25)
        elif season_type_lower in {"postseason", "playoffs", "post-season"}:
            default_start = date(end_year, 4, 10)
            default_end = date(end_year, 6, 30)
        else:
            default_start = date(start_year, 10, 15)
            default_end = date(end_year, 4, 20)

        start = NBADataCollector._parse_input_date(date_from) if date_from else default_start
        end = NBADataCollector._parse_input_date(date_to) if date_to else default_end
        return start, end

    @staticmethod
    def _espn_season_type(season_type: str) -> Optional[int]:
        normalized = season_type.strip().lower()
        if normalized in {"preseason", "pre-season"}:
            return 1
        if normalized in {"regular season", "regular-season", "regular"}:
            return 2
        if normalized in {"postseason", "playoffs", "post-season"}:
            return 3
        return None

    @staticmethod
    def _parse_input_date(value: str) -> date:
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {value}")

    @staticmethod
    def _normalize_status(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        if "final" in text or text in {"3", "status_post"}:
            return "final"
        if "scheduled" in text or "pre" in text or text in {"1", "status_scheduled"}:
            return "scheduled"
        if "progress" in text or "live" in text or text in {"2", "status_in_progress"}:
            return "live"
        return text

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if not value:
            raise ValueError("Game time was missing from ESPN response")
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text)

    @staticmethod
    def _minutes_to_float(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        text = str(value)
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
    def _stat_value(stat_map: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in stat_map:
                return stat_map[key]
        return None

    @staticmethod
    def _attempts_from_made_attempted(value: Any) -> int:
        if value in (None, ""):
            return 0
        text = str(value)
        if "-" in text:
            try:
                return int(float(text.rsplit("-", 1)[1]))
            except ValueError:
                return 0
        return 0

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
    def _float_or_none(value: Any):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _activity_proxy(fga: int, fta: int, turnovers: int, minutes: float):
        if minutes <= 0:
            return None
        return round((fga + 0.44 * fta + turnovers) / minutes, 4)
