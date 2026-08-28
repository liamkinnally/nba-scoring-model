from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # save-only module; must work headless (CI, servers)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from nba_scoring_model.modeling.baselines import TARGETS

# Colorblind-safe pair used consistently: blue = model, orange = last-10 baseline.
MODEL_COLOR = "#0173B2"
BASELINE_COLOR = "#DE8F05"
REFERENCE_GREY = "#666666"

STYLE = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
}


def _save(fig, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction_quality(
    predictions: pd.DataFrame, path: Path, *, dpi: int = 170
) -> Path:
    """Three-panel actual-vs-predicted hexbin parity plot with metrics."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1))
        for ax, target in zip(axes, TARGETS):
            sub = predictions[predictions["target"] == target]
            actual = sub["actual"].to_numpy()
            pred = sub["model_pred"].to_numpy()

            lo = min(0.0, actual.min(), pred.min())
            hi = max(actual.max(), pred.max()) * 1.03
            hb = ax.hexbin(
                pred,
                actual,
                gridsize=42,
                cmap="viridis",
                mincnt=1,
                extent=(lo, hi, lo, hi),
                linewidths=0.2,
            )
            ax.plot(
                [lo, hi], [lo, hi],
                linestyle="--", color=REFERENCE_GREY, linewidth=1.0, zorder=3,
            )
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            ax.text(
                0.04, 0.96,
                f"MAE  {mae:.3f}\nRMSE {rmse:.3f}\nR$^2$   {r2:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
            )
            ax.set_title(target.capitalize())
            ax.set_xlabel("Predicted")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(False)
            cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7)
        axes[0].set_ylabel("Actual")
        fig.suptitle("Holdout prediction quality: actual vs. predicted", y=1.0)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_residual_diagnostics(
    predictions: pd.DataFrame, path: Path, *, dpi: int = 170
) -> Path:
    """Three-panel residual-vs-predicted plot with a zero reference line."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
        for ax, target in zip(axes, TARGETS):
            sub = predictions[predictions["target"] == target]
            pred = sub["model_pred"].to_numpy()
            residual = sub["actual"].to_numpy() - pred
            ax.scatter(
                pred, residual,
                s=7, alpha=0.12, color=MODEL_COLOR,
                edgecolors="none", rasterized=True,
            )
            ax.axhline(0.0, linestyle="--", color=REFERENCE_GREY, linewidth=1.0)
            limit = np.abs(residual).max() * 1.05
            ax.set_ylim(-limit, limit)
            ax.set_title(target.capitalize())
            ax.set_xlabel("Predicted")
        axes[0].set_ylabel("Residual (actual - predicted)")
        fig.suptitle("Holdout residual diagnostics", y=1.02)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_rolling_mae(rolling: pd.DataFrame, path: Path, *, dpi: int = 170) -> Path:
    """Stacked panels of trailing 7-day MAE, model vs last-10, common rows.

    Only complete 7-day windows are drawn; partial windows at the start of the
    holdout period are computed in the CSV but excluded here to avoid a
    misleadingly volatile opening stretch.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
        for ax, target in zip(axes, TARGETS):
            sub = rolling[
                (rolling["target"] == target) & rolling["window_complete"]
            ]
            dates = pd.to_datetime(sub["date"])
            ax.plot(dates, sub["model_mae"], color=MODEL_COLOR,
                    label="Gradient Boosting model")
            ax.plot(dates, sub["baseline_mae"], color=BASELINE_COLOR,
                    label="Last-10 average")
            ax.set_ylabel("MAE")
            ax.set_title(target.capitalize(), loc="left")
        axes[0].legend(frameon=False, ncol=2, loc="upper right")
        locator = mdates.AutoDateLocator()
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.suptitle(
            "Trailing 7-day holdout MAE (rows where both predictions exist)",
            y=1.0,
        )
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_permutation_importance(
    importance: pd.DataFrame, target: str, path: Path, *, dpi: int = 170
) -> Path:
    """Horizontal-bar holdout permutation importance with repeat error bars."""
    sub = importance[importance["target"] == target].sort_values("mae_increase_mean")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.barh(
            sub["feature"], sub["mae_increase_mean"],
            xerr=sub["mae_increase_std"],
            color=MODEL_COLOR,
            error_kw=dict(ecolor="#444444", lw=1.0, capsize=2),
        )
        ax.set_xlabel("Increase in holdout MAE when feature is shuffled")
        ax.set_title(f"Permutation importance - {target} model (holdout)")
        ax.grid(axis="x")
        ax.grid(False, axis="y")
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_partial_dependence(
    pipeline: Pipeline,
    X: pd.DataFrame,
    features: Sequence[str],
    path: Path,
    *,
    dpi: int = 170,
) -> Path:
    """Partial dependence panels for the selected top continuous features."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(features), figsize=(9.0, 3.8), sharey=True)
        axes_list: List = list(np.atleast_1d(axes))
        PartialDependenceDisplay.from_estimator(
            pipeline,
            X,
            features=list(features),
            kind="average",
            ax=axes_list,
            line_kw={"color": MODEL_COLOR},
        )
        for ax in axes_list:
            ax.set_title("")
        fig.suptitle("Partial dependence - points model", y=1.0)
        fig.tight_layout()
        return _save(fig, path, dpi)
