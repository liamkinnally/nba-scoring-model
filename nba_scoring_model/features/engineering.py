
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.models import Game, PlayerGameStats


class FeatureEngineer:
    """Build leakage-safe player features using only games before the target game."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def generate_player_features(
        self,
        player_id: str,
        game_id: str,
        *,
        team_id: Optional[str] = None,
    ) -> Dict[str, float]:
        with self.db_manager.get_session() as session:
            game = session.get(Game, game_id)
            if game is None:
                raise ValueError(f"Unknown game_id: {game_id}")

            if team_id is None:
                target_stat = session.scalar(
                    select(PlayerGameStats).where(
                        PlayerGameStats.game_id == game_id,
                        PlayerGameStats.player_id == player_id,
                    )
                )
                if target_stat is None:
                    raise ValueError("team_id is required when the target game has no stored stat row")
                team_id = target_stat.team_id

            if team_id not in (game.home_team_id, game.away_team_id):
                raise ValueError(f"Team {team_id} does not participate in game {game_id}")

            opponent_id = game.away_team_id if team_id == game.home_team_id else game.home_team_id
            features: Dict[str, float] = {
                "is_home_game": float(team_id == game.home_team_id),
                "vegas_total": float(game.vegas_total) if game.vegas_total is not None else np.nan,
                "vegas_spread": float(game.vegas_spread) if game.vegas_spread is not None else np.nan,
            }
            features.update(self._player_form(session, player_id, game.date))
            features.update(self._opponent_history(session, player_id, opponent_id, game.date))
            features.update(self._team_form(session, team_id, game.date))
            features.update(self._schedule_features(session, player_id, game.date))
            return features

    @staticmethod
    def _prior_player_rows(session: Session, player_id: str, as_of: datetime):
        return list(
            session.execute(
                select(PlayerGameStats, Game)
                .join(Game, PlayerGameStats.game_id == Game.game_id)
                .where(PlayerGameStats.player_id == player_id, PlayerGameStats.minutes_played > 0, Game.date < as_of)
                .order_by(Game.date.desc())
            ).all()
        )

    def _player_form(self, session: Session, player_id: str, as_of: datetime) -> Dict[str, float]:
        rows = self._prior_player_rows(session, player_id, as_of)

        def avg(field: str, n: int) -> float:
            vals = [getattr(stat, field) for stat, _ in rows[:n] if getattr(stat, field) is not None]
            return float(np.mean(vals)) if vals else np.nan

        return {
            "points_avg_5": avg("points", 5),
            "points_avg_10": avg("points", 10),
            "assists_avg_5": avg("assists", 5),
            "assists_avg_10": avg("assists", 10),
            "rebounds_avg_5": avg("rebounds", 5),
            "rebounds_avg_10": avg("rebounds", 10),
            "minutes_avg_5": avg("minutes_played", 5),
            "activity_proxy_avg_5": avg("usage_rate", 5),
        }

    def _opponent_history(
        self,
        session: Session,
        player_id: str,
        opponent_id: str,
        as_of: datetime,
    ) -> Dict[str, float]:
        rows = list(
            session.execute(
                select(PlayerGameStats, Game)
                .join(Game, PlayerGameStats.game_id == Game.game_id)
                .where(
                    PlayerGameStats.player_id == player_id, PlayerGameStats.minutes_played > 0,
                    Game.date < as_of,
                    or_(Game.home_team_id == opponent_id, Game.away_team_id == opponent_id),
                )
                .order_by(Game.date.desc())
                .limit(5)
            ).all()
        )

        def avg(field: str) -> float:
            vals = [getattr(stat, field) for stat, _ in rows]
            return float(np.mean(vals)) if vals else np.nan

        return {
            "vs_opponent_points_avg_5": avg("points"),
            "vs_opponent_assists_avg_5": avg("assists"),
            "vs_opponent_rebounds_avg_5": avg("rebounds"),
        }

    def _team_form(self, session: Session, team_id: str, as_of: datetime) -> Dict[str, float]:
        games = list(
            session.scalars(
                select(Game)
                .where(
                    Game.date < as_of,
                    Game.game_status == "final",
                    or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                )
                .order_by(Game.date.desc())
                .limit(10)
            )
        )
        if not games:
            return {
                "team_points_for_avg_10": np.nan,
                "team_points_against_avg_10": np.nan,
                "team_recent_win_pct_10": np.nan,
                "team_pace_proxy_10": np.nan,
            }

        points_for = []
        points_against = []
        wins = []
        totals = []
        for game in games:
            if game.home_score is None or game.away_score is None:
                continue
            is_home = game.home_team_id == team_id
            pf = game.home_score if is_home else game.away_score
            pa = game.away_score if is_home else game.home_score
            points_for.append(pf)
            points_against.append(pa)
            wins.append(float(pf > pa))
            totals.append(pf + pa)

        return {
            "team_points_for_avg_10": float(np.mean(points_for)) if points_for else np.nan,
            "team_points_against_avg_10": float(np.mean(points_against)) if points_against else np.nan,
            "team_recent_win_pct_10": float(np.mean(wins)) if wins else np.nan,
            "team_pace_proxy_10": float(np.mean(totals) / 2.0) if totals else np.nan,
        }

    def _schedule_features(self, session: Session, player_id: str, as_of: datetime) -> Dict[str, float]:
        prev_date = session.scalar(
            select(Game.date)
            .join(PlayerGameStats, PlayerGameStats.game_id == Game.game_id)
            .where(PlayerGameStats.player_id == player_id, PlayerGameStats.minutes_played > 0, Game.date < as_of)
            .order_by(Game.date.desc())
            .limit(1)
        )
        if prev_date is None:
            return {"days_rest": np.nan, "is_back_to_back": 0.0}
        days_rest = max((as_of.date() - prev_date.date()).days, 0)
        return {"days_rest": float(days_rest), "is_back_to_back": float(days_rest == 1)}
