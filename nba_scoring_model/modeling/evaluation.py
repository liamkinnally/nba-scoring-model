from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import select

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.models import Game, PlayerGameStats
from nba_scoring_model.modeling.base import PlayerStatModel


def evaluate_recent_average_baselines(
    db_manager: DatabaseManager,
    start_date: datetime,
    end_date: datetime,
) -> Dict[str, object]:
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
                .order_by(
                    Game.date.asc(),
                    Game.game_id.asc(),
                    PlayerGameStats.player_id.asc(),
                )
            ).all()
        )

    records = [
        {
            "date": game.date,
            "game_id": game.game_id,
            "player_id": stat.player_id,
            "points": float(stat.points),
            "assists": float(stat.assists),
            "rebounds": float(stat.rebounds),
        }
        for stat, game in rows
    ]

    frame = pd.DataFrame(records)

    if frame.empty:
        raise ValueError("No rows found for the requested date range")

    frame = frame.sort_values(
        ["date", "game_id", "player_id"]
    ).reset_index(drop=True)

    for target in ("points", "assists", "rebounds"):
        for window in (5, 10):
            frame[f"{target}_avg_{window}"] = (
                frame.groupby("player_id")[target]
                .transform(
                    lambda values: values.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )

    _, test_frame = PlayerStatModel.chronological_holdout(frame)

    output: Dict[str, object] = {
        "rows": len(frame),
        "test_rows": len(test_frame),
        "cutoff_date": str(pd.to_datetime(test_frame["date"]).min().date()),
        "targets": {},
    }

    for target in ("points", "assists", "rebounds"):
        target_results = {}

        for window in (5, 10):
            prediction_column = f"{target}_avg_{window}"
            valid = test_frame[[target, prediction_column]].dropna()

            actual = valid[target]
            predicted = valid[prediction_column]

            target_results[f"last_{window}"] = {
                "rows": len(valid),
                "mae": round(float(mean_absolute_error(actual, predicted)), 3),
                "rmse": round(
                    float(np.sqrt(mean_squared_error(actual, predicted))),
                    3,
                ),
                "r2": round(float(r2_score(actual, predicted)), 3),
            }

        output["targets"][target] = target_results

    return output
