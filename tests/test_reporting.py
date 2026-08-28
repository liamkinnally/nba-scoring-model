from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.demo import seed_demo_database
from nba_scoring_model.modeling.baselines import add_rolling_baseline_columns
from nba_scoring_model.modeling.trainer import ModelTrainer
from nba_scoring_model.reporting import generate_report
from nba_scoring_model.reporting.data import (
    compute_rolling_mae,
    resolve_evaluation_window,
)


def test_rolling_baseline_is_shifted_and_leakage_safe():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-03", "2026-01-05"]
            ),
            "game_id": ["g1", "g1", "g2", "g2", "g3"],
            "player_id": ["a", "b", "a", "b", "a"],
            "points": [10.0, 4.0, 20.0, 6.0, 30.0],
            "assists": [1.0, 2.0, 3.0, 4.0, 5.0],
            "rebounds": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    out = add_rolling_baseline_columns(frame, windows=(5,))
    player_a = out[out["player_id"] == "a"]["baseline_points_last_5"].tolist()
    player_b = out[out["player_id"] == "b"]["baseline_points_last_5"].tolist()

    assert np.isnan(player_a[0])
    assert player_a[1] == 10.0
    assert player_a[2] == 15.0
    assert np.isnan(player_b[0])
    assert player_b[1] == 4.0


def test_rolling_mae_is_row_weighted():
    rows = []
    for target in ("points", "assists", "rebounds"):
        rows.append(
            {
                "date": pd.Timestamp("2026-03-11"),
                "target": target,
                "actual": 10.0,
                "model_pred": 0.0,
                "baseline_last10_pred": -10.0,
            }
        )
        rows.extend(
            {
                "date": pd.Timestamp("2026-03-12"),
                "target": target,
                "actual": 0.0,
                "model_pred": 0.0,
                "baseline_last10_pred": 0.0,
            }
            for _ in range(9)
        )
        rows.append(
            {
                "date": pd.Timestamp("2026-03-17"),
                "target": target,
                "actual": 0.0,
                "model_pred": 0.0,
                "baseline_last10_pred": 0.0,
            }
        )

    rolling = compute_rolling_mae(pd.DataFrame(rows))
    points = rolling[(rolling["target"] == "points") & rolling["window_complete"]]
    final = points.iloc[-1]

    assert final["n_rows"] == 11
    assert final["model_mae"] == pytest.approx(10.0 / 11.0)
    assert final["baseline_mae"] == pytest.approx(20.0 / 11.0)


def test_stored_window_is_authoritative():
    metadata = {
        target: {
            "frame_start": "2025-10-21T19:00:00",
            "frame_end": "2026-04-14T19:00:00",
            "holdout_cutoff": "2026-03-11",
        }
        for target in ("points", "assists", "rebounds")
    }

    start, end, cutoff, source = resolve_evaluation_window(metadata, None, None)
    assert source == "model metadata"
    assert start == datetime(2025, 10, 21, 19)
    assert end == datetime(2026, 4, 14, 19)
    assert cutoff == pd.Timestamp("2026-03-11")

    with pytest.raises(ValueError, match="already contain their evaluation window"):
        resolve_evaluation_window(
            metadata,
            datetime(2025, 10, 21),
            datetime(2026, 4, 14),
        )


def test_mixed_window_metadata_is_rejected():
    metadata = {
        "points": {
            "frame_start": "2025-10-21T19:00:00",
            "frame_end": "2026-04-14T19:00:00",
            "holdout_cutoff": "2026-03-11",
        },
        "assists": {},
        "rebounds": {},
    }

    with pytest.raises(ValueError, match="mix old and new"):
        resolve_evaluation_window(
            metadata,
            datetime(2025, 10, 21),
            datetime(2026, 4, 14),
        )


def test_legacy_models_require_explicit_window():
    metadata = {target: {} for target in ("points", "assists", "rebounds")}
    start = datetime(2025, 10, 21)
    end = datetime(2026, 4, 14)

    resolved_start, resolved_end, cutoff, source = resolve_evaluation_window(
        metadata, start, end
    )
    assert (resolved_start, resolved_end) == (start, end)
    assert cutoff is None
    assert source.startswith("cli")

    with pytest.raises(ValueError, match="predate evaluation-window metadata"):
        resolve_evaluation_window(metadata, None, None)


def test_report_generates_outputs(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path / 'demo.db'}")
    seed_demo_database(db, game_days=18)

    trainer = ModelTrainer(db, tmp_path / "models")
    trainer.train_all(datetime(2025, 10, 1), datetime(2026, 1, 31), tune=False)

    output_dir = tmp_path / "reports"
    summary = generate_report(
        db, model_dir=tmp_path / "models", output_dir=output_dir
    )

    assert summary["window_source"] == "model metadata"
    for name in (
        "holdout_predictions.csv",
        "rolling_mae.csv",
        "permutation_importance.csv",
        "tier_analysis.csv",
    ):
        assert (output_dir / "data" / name).exists()
    for name in (
        "prediction_quality.png",
        "residual_diagnostics.png",
        "rolling_mae.png",
        "permutation_importance_points.png",
        "permutation_importance_assists.png",
        "permutation_importance_rebounds.png",
    ):
        assert (output_dir / "figures" / name).exists()

    summary_again = generate_report(
        db, model_dir=tmp_path / "models", output_dir=output_dir
    )
    assert summary_again["frame_source"].startswith("cache")
