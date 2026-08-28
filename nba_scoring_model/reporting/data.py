import hashlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.features.engineering import FeatureEngineer
from nba_scoring_model.modeling.base import PlayerStatModel
from nba_scoring_model.modeling.baselines import (
    TARGETS,
    add_rolling_baseline_columns,
    baseline_column,
)
from nba_scoring_model.modeling.frames import build_feature_frame

CACHE_VERSION = 1
ROLLING_WINDOW_DAYS = 7
PERMUTATION_REPEATS = 10
BINARY_FEATURES = {"is_home_game", "is_back_to_back"}
TIER_BINS = [0.0, 8.0, 15.0, 22.0, np.inf]
TIER_LABELS = ["<8 PPG", "8-15 PPG", "15-22 PPG", "22+ PPG"]
TIER_FEATURE = "points_avg_10"

_WINDOW_KEYS = ("frame_start", "frame_end", "holdout_cutoff")


def load_models(
    db_manager: DatabaseManager,
    feature_engineer: FeatureEngineer,
    model_dir: str | Path,
) -> Tuple[Dict[str, PlayerStatModel], Dict[str, dict]]:
    """Load the three saved model artifacts and their metadata."""
    models: Dict[str, PlayerStatModel] = {}
    metadata: Dict[str, dict] = {}
    for target in TARGETS:
        model = PlayerStatModel(target, db_manager, feature_engineer, model_dir)
        try:
            metadata[target] = model.load()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No saved model for '{target}' in {model_dir}. "
                "Run `python -m nba_scoring_model.cli train ...` first."
            ) from exc
        models[target] = model

    columns = {tuple(m.feature_columns) for m in models.values()}
    if len(columns) != 1:
        raise ValueError(
            "Saved models disagree on feature columns; retrain all three targets "
            "together before generating a report."
        )
    return models, metadata


def resolve_evaluation_window(
    metadata: Dict[str, dict],
    cli_start: Optional[datetime],
    cli_end: Optional[datetime],
) -> Tuple[datetime, datetime, Optional[pd.Timestamp], str]:
    """Resolve the evaluation range without silently changing model provenance.

    Newer artifacts store the frame range and exact holdout cutoff used during
    training. Those values are authoritative. Explicit dates are supported only
    for older artifacts that predate this metadata.
    """
    if (cli_start is None) != (cli_end is None):
        raise ValueError("Provide both --start and --end, or neither.")

    stored = [
        {key: meta.get(key) for key in _WINDOW_KEYS} for meta in metadata.values()
    ]
    populated_counts = [
        sum(value is not None for value in entry.values()) for entry in stored
    ]

    if any(0 < count < len(_WINDOW_KEYS) for count in populated_counts):
        raise ValueError(
            "Saved model evaluation metadata is incomplete; retrain all three "
            "targets before generating a report."
        )

    has_window = [count == len(_WINDOW_KEYS) for count in populated_counts]
    if any(has_window) and not all(has_window):
        raise ValueError(
            "Saved models mix old and new evaluation metadata; retrain all three "
            "targets together before generating a report."
        )

    if not any(has_window):
        if cli_start is not None and cli_end is not None:
            return cli_start, cli_end, None, "cli (legacy model metadata)"
        raise ValueError(
            "Saved models predate evaluation-window metadata. Pass --start and "
            "--end matching the original training range, or retrain the models."
        )

    if any(entry != stored[0] for entry in stored[1:]):
        raise ValueError(
            "Saved models were trained on different evaluation windows; retrain "
            "all three targets together before generating a report."
        )

    if cli_start is not None and cli_end is not None:
        raise ValueError(
            "Saved models already contain their evaluation window. Omit "
            "--start/--end so the report uses the stored range and holdout cutoff."
        )

    start = datetime.fromisoformat(stored[0]["frame_start"])
    end = datetime.fromisoformat(stored[0]["frame_end"])
    cutoff = pd.Timestamp(stored[0]["holdout_cutoff"]).normalize()
    return start, end, cutoff, "model metadata"


def _feature_code_fingerprint() -> str:
    """Short hash used to invalidate cached frames after feature-code changes."""
    source = inspect.getsource(FeatureEngineer) + inspect.getsource(build_feature_frame)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def get_evaluation_frame(
    db_manager: DatabaseManager,
    feature_engineer: FeatureEngineer,
    start: datetime,
    end: datetime,
    *,
    cache_dir: Path,
    expected_features: List[str],
    refresh_cache: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Return the evaluation frame, reusing a cache only when it still matches.

    Cache metadata includes the date range, database URL, cache format version,
    and a fingerprint of the feature-building code. ``--refresh-cache`` remains
    available when the underlying database contents change in place.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"evaluation_{start:%Y-%m-%d}_{end:%Y-%m-%d}_v{CACHE_VERSION}.joblib"
    )
    database_url = str(db_manager.engine.url)
    feature_code = _feature_code_fingerprint()

    if cache_path.exists() and not refresh_cache:
        payload = joblib.load(cache_path)
        frame = payload.get("frame")
        valid = (
            payload.get("cache_version") == CACHE_VERSION
            and payload.get("start") == str(start)
            and payload.get("end") == str(end)
            and payload.get("database_url") == database_url
            and payload.get("feature_code") == feature_code
            and frame is not None
            and set(expected_features).issubset(frame.columns)
            and set(TARGETS).issubset(frame.columns)
        )
        if valid:
            return frame, f"cache ({cache_path.name})"

    frame = build_feature_frame(
        db_manager, feature_engineer, start, end, targets=TARGETS
    )
    joblib.dump(
        {
            "cache_version": CACHE_VERSION,
            "start": str(start),
            "end": str(end),
            "database_url": database_url,
            "feature_code": feature_code,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frame": frame,
        },
        cache_path,
    )
    return frame, "rebuilt"


def split_holdout(
    frame: pd.DataFrame, cutoff: Optional[pd.Timestamp]
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Slice the holdout rows, honoring a stored cutoff when one exists."""
    normalized = pd.to_datetime(frame["date"]).dt.normalize()
    if cutoff is None:
        _, holdout = PlayerStatModel.chronological_holdout(frame)
        cutoff = pd.to_datetime(holdout["date"]).min().normalize()
        return holdout.copy(), cutoff
    holdout = frame[normalized >= cutoff].copy()
    if holdout.empty:
        raise ValueError(f"No holdout rows on or after stored cutoff {cutoff.date()}")
    return holdout, cutoff


def build_holdout_predictions(
    models: Dict[str, PlayerStatModel], holdout: pd.DataFrame
) -> pd.DataFrame:
    """Long-format holdout predictions: one row per (game, player, target)."""
    pieces = []
    for target, model in models.items():
        X = holdout[model.feature_columns]
        pieces.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(holdout["date"]).dt.normalize(),
                    "game_id": holdout["game_id"].to_numpy(),
                    "player_id": holdout["player_id"].to_numpy(),
                    "target": target,
                    "actual": holdout[target].to_numpy(),
                    "model_pred": model.pipeline.predict(X),
                    "baseline_last10_pred": holdout[
                        baseline_column(target, 10)
                    ].to_numpy(),
                    "prior_points_avg_10": holdout[TIER_FEATURE].to_numpy(),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def compute_rolling_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Row-weighted trailing 7-day MAE, model vs last-10, on common rows only.

    Computed as rolling(sum of absolute errors) / rolling(row count) — not a
    mean of daily MAEs — so thin two-game slates cannot swing the line.
    """
    records = []
    for target in TARGETS:
        sub = predictions[
            (predictions["target"] == target)
            & predictions["baseline_last10_pred"].notna()
        ]
        if sub.empty:
            raise ValueError(
                f"No common model/baseline rows are available for target '{target}'."
            )
        daily = (
            sub.assign(
                model_abs_err=(sub["actual"] - sub["model_pred"]).abs(),
                baseline_abs_err=(sub["actual"] - sub["baseline_last10_pred"]).abs(),
            )
            .groupby("date")
            .agg(
                model_abs_err=("model_abs_err", "sum"),
                baseline_abs_err=("baseline_abs_err", "sum"),
                n_rows=("actual", "size"),
            )
        )
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range, fill_value=0.0)
        rolled = daily.rolling(ROLLING_WINDOW_DAYS, min_periods=1).sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            model_mae = rolled["model_abs_err"] / rolled["n_rows"]
            baseline_mae = rolled["baseline_abs_err"] / rolled["n_rows"]
        records.append(
            pd.DataFrame(
                {
                    "date": full_range,
                    "target": target,
                    "model_mae": model_mae.to_numpy(),
                    "baseline_mae": baseline_mae.to_numpy(),
                    "n_rows": rolled["n_rows"].to_numpy().astype(int),
                    "window_complete": np.arange(len(full_range))
                    >= ROLLING_WINDOW_DAYS - 1,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def compute_permutation_importance(
    models: Dict[str, PlayerStatModel], holdout: pd.DataFrame
) -> pd.DataFrame:
    """Holdout permutation importance in MAE units.

    With ``scoring="neg_mean_absolute_error"`` the reported mean is exactly the
    increase in holdout MAE when the feature is shuffled.
    """
    records = []
    for target, model in models.items():
        result = permutation_importance(
            model.pipeline,
            holdout[model.feature_columns],
            holdout[target],
            scoring="neg_mean_absolute_error",
            n_repeats=PERMUTATION_REPEATS,
            random_state=42,
        )
        for feature, mean, std in zip(
            model.feature_columns, result.importances_mean, result.importances_std
        ):
            records.append(
                {
                    "target": target,
                    "feature": feature,
                    "mae_increase_mean": float(mean),
                    "mae_increase_std": float(std),
                }
            )
    frame = pd.DataFrame(records)
    return frame.sort_values(
        ["target", "mae_increase_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def compute_tier_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Model vs baseline MAE by prior scoring tier, on common rows only."""
    sub = predictions[
        predictions["baseline_last10_pred"].notna()
        & predictions["prior_points_avg_10"].notna()
    ].copy()
    sub["tier"] = pd.cut(
        sub["prior_points_avg_10"],
        bins=TIER_BINS,
        labels=TIER_LABELS,
        right=False,
        include_lowest=True,
    )
    records = []
    for (target, tier), group in sub.groupby(["target", "tier"], observed=True):
        if group.empty:
            continue
        model_mae = float((group["actual"] - group["model_pred"]).abs().mean())
        baseline_mae = float(
            (group["actual"] - group["baseline_last10_pred"]).abs().mean()
        )
        records.append(
            {
                "target": target,
                "tier": str(tier),
                "n_rows": len(group),
                "model_mae": round(model_mae, 3),
                "baseline_mae": round(baseline_mae, 3),
                "delta_mae": round(model_mae - baseline_mae, 3),
                "delta_pct": round(
                    100.0 * (model_mae - baseline_mae) / baseline_mae, 2
                )
                if baseline_mae
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def select_pdp_features(importance: pd.DataFrame, target: str = "points") -> List[str]:
    """Top two continuous features by permutation importance for the target.

    Binary indicators are excluded — a partial dependence curve of a 0/1
    feature is just two points and tells the reader nothing.
    """
    sub = importance[
        (importance["target"] == target)
        & ~importance["feature"].isin(BINARY_FEATURES)
        & (importance["mae_increase_mean"] > 0)
    ]
    return sub.nlargest(2, "mae_increase_mean")["feature"].tolist()


def holdout_metrics(predictions: pd.DataFrame) -> Dict[str, dict]:
    """Recompute headline metrics as a sanity check against the README table."""
    out: Dict[str, dict] = {}
    for target in TARGETS:
        sub = predictions[predictions["target"] == target]
        common = sub[sub["baseline_last10_pred"].notna()]
        out[target] = {
            "holdout_rows": int(len(sub)),
            "common_rows": int(len(common)),
            "model_mae": round(float(mean_absolute_error(sub["actual"], sub["model_pred"])), 3),
            "model_rmse": round(
                float(np.sqrt(mean_squared_error(sub["actual"], sub["model_pred"]))), 3
            ),
            "model_r2": round(float(r2_score(sub["actual"], sub["model_pred"])), 3),
            "baseline_last10_mae_common": round(
                float(
                    mean_absolute_error(common["actual"], common["baseline_last10_pred"])
                ),
                3,
            ),
            "model_mae_common": round(
                float(mean_absolute_error(common["actual"], common["model_pred"])), 3
            ),
        }
    return out


def generate_report(
    db_manager: DatabaseManager,
    *,
    model_dir: str | Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    output_dir: str | Path = "reports",
    refresh_cache: bool = False,
    dpi: int = 170,
) -> dict:
    """Produce every reporting CSV and figure from the saved models.

    The full flow is: load models -> resolve the evaluation window -> build or
    load the cached feature frame -> attach baselines -> slice the holdout ->
    write CSVs -> render figures. Deterministic given the same database and
    saved models.
    """
    from nba_scoring_model.reporting import plots

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    cache_dir = output_dir / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_engineer = FeatureEngineer(db_manager)
    models, metadata = load_models(db_manager, feature_engineer, model_dir)
    feature_columns = models["points"].feature_columns

    start, end, cutoff, window_source = resolve_evaluation_window(metadata, start, end)
    frame, frame_source = get_evaluation_frame(
        db_manager,
        feature_engineer,
        start,
        end,
        cache_dir=cache_dir,
        expected_features=feature_columns,
        refresh_cache=refresh_cache,
    )
    frame = add_rolling_baseline_columns(frame, windows=(10,))
    holdout, cutoff = split_holdout(frame, cutoff)

    predictions = build_holdout_predictions(models, holdout)
    rolling = compute_rolling_mae(predictions)
    importance = compute_permutation_importance(models, holdout)
    tiers = compute_tier_analysis(predictions)
    pdp_features = select_pdp_features(importance, target="points")

    predictions.to_csv(data_dir / "holdout_predictions.csv", index=False)
    rolling.to_csv(data_dir / "rolling_mae.csv", index=False)
    importance.to_csv(data_dir / "permutation_importance.csv", index=False)
    tiers.to_csv(data_dir / "tier_analysis.csv", index=False)

    figure_paths = [
        plots.plot_prediction_quality(
            predictions, figures_dir / "prediction_quality.png", dpi=dpi
        ),
        plots.plot_residual_diagnostics(
            predictions, figures_dir / "residual_diagnostics.png", dpi=dpi
        ),
        plots.plot_rolling_mae(rolling, figures_dir / "rolling_mae.png", dpi=dpi),
    ]
    for target in TARGETS:
        figure_paths.append(
            plots.plot_permutation_importance(
                importance,
                target,
                figures_dir / f"permutation_importance_{target}.png",
                dpi=dpi,
            )
        )
    if pdp_features:
        figure_paths.append(
            plots.plot_partial_dependence(
                models["points"].pipeline,
                holdout[feature_columns],
                pdp_features,
                figures_dir / "partial_dependence_points.png",
                dpi=dpi,
            )
        )

    return {
        "window_source": window_source,
        "frame_source": frame_source,
        "start": str(start),
        "end": str(end),
        "holdout_cutoff": str(cutoff.date()),
        "rows": int(len(frame)),
        "holdout_rows": int(len(holdout)),
        "metrics": holdout_metrics(predictions),
        "pdp_features": pdp_features,
        "data_files": sorted(str(p) for p in data_dir.glob("*.csv")),
        "figures": [str(p) for p in figure_paths],
    }
