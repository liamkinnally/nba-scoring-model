
from typing import Iterable, Sequence

import pandas as pd

TARGETS: Sequence[str] = ("points", "assists", "rebounds")


def baseline_column(target: str, window: int) -> str:
    """Canonical name for a rolling-average baseline column.

    Deliberately distinct from the model feature names (e.g. ``points_avg_10``)
    so the two can coexist in a single evaluation frame without collisions.
    """
    return f"baseline_{target}_last_{window}"


def add_rolling_baseline_columns(
    frame: pd.DataFrame,
    *,
    targets: Sequence[str] = TARGETS,
    windows: Iterable[int] = (5, 10),
) -> pd.DataFrame:
    """Attach leakage-safe rolling-average baseline predictions.

    For each target and window this adds ``baseline_{target}_last_{window}``:
    the mean of the player's previous ``window`` games. The ``shift(1)`` is the
    leakage-critical step — it guarantees the game being predicted is never
    included in its own baseline. This is the single authoritative
    implementation; both baseline evaluation and report generation must use it.

    Returns a chronologically sorted copy of ``frame`` with the new columns.
    Rows with no prior games receive NaN (no baseline prediction is possible).
    """
    missing = [t for t in targets if t not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing target columns: {missing}")

    out = (
        frame.sort_values(["date", "game_id", "player_id"])
        .reset_index(drop=True)
        .copy()
    )
    grouped = out.groupby("player_id", sort=False)
    for target in targets:
        for window in windows:
            out[baseline_column(target, window)] = grouped[target].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )
    return out
