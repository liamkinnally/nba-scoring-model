
from datetime import datetime
from typing import Sequence

import pandas as pd
from sqlalchemy import select

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.models import Game, PlayerGameStats
from nba_scoring_model.features.engineering import FeatureEngineer

from .baselines import TARGETS


def build_feature_frame(
    db_manager: DatabaseManager,
    feature_engineer: FeatureEngineer,
    start_date: datetime,
    end_date: datetime,
    *,
    targets: Sequence[str] = TARGETS,
) -> pd.DataFrame:
    """Build the leakage-safe evaluation frame for the given date range.

    One row per (game, player) with minutes played, containing the requested
    target columns plus every engineered feature. Training uses this with a
    single target; reporting uses it with all three so features are only
    generated once.
    """
    with db_manager.get_session() as session:
        rows = list(
            session.execute(
                select(PlayerGameStats, Game)
                .join(Game, PlayerGameStats.game_id == Game.game_id)
                .where(
                    Game.date >= start_date,
                    Game.date <= end_date,
                    PlayerGameStats.minutes_played > 0,
                )
                .order_by(Game.date.asc(), Game.game_id.asc(), PlayerGameStats.player_id.asc())
            ).all()
        )

    records = []
    for stat, game in rows:
        features = feature_engineer.generate_player_features(
            stat.player_id,
            game.game_id,
            team_id=stat.team_id,
        )
        record = {
            "date": game.date,
            "game_id": game.game_id,
            "player_id": stat.player_id,
        }
        for target in targets:
            record[target] = float(getattr(stat, target))
        record.update(features)
        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No rows found for the requested date range")
    return frame.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)
