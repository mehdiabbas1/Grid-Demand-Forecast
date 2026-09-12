"""The baselines and the metrics.

Demand is strongly weekly, so "same half hour last week" is already decent and
it's the number to beat. An MAE with nothing next to it doesn't mean anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.features import PERIODS_PER_DAY, PERIODS_PER_WEEK


def seasonal_naive(series: pd.Series, season_periods: int) -> pd.Series:
    """Predict the value one full season ago."""
    return series.shift(season_periods)


def persistence(series: pd.Series, horizon_periods: int) -> pd.Series:
    """Predict the last value known at the forecast origin."""
    return series.shift(horizon_periods)


def build_baselines(
    frame: pd.DataFrame,
    target: str,
    horizon_hours: float,
    published_forecast: str | None = None,
) -> pd.DataFrame:
    """Return a frame of baseline predictions aligned to the target column."""
    horizon_periods = int(round(horizon_hours * 2))
    out = pd.DataFrame(index=frame.index)
    out["timestamp"] = frame["timestamp"]
    out["actual"] = frame[target]

    out["naive_last_week"] = seasonal_naive(frame[target], PERIODS_PER_WEEK)
    out["naive_yesterday"] = seasonal_naive(frame[target], PERIODS_PER_DAY)
    out["persistence"] = persistence(frame[target], horizon_periods)

    # Average of the last 4 matching weekdays - common rule of thumb, harder to beat.
    weekly = pd.concat(
        [frame[target].shift(PERIODS_PER_WEEK * k) for k in (1, 2, 3, 4)],
        axis=1,
    )
    out["naive_4week_mean"] = weekly.mean(axis=1)

    if published_forecast and published_forecast in frame.columns:
        out["published_forecast"] = frame[published_forecast]

    return out


def mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    return float(np.mean(np.abs(a[mask] - p[mask]))) if mask.any() else float("nan")


def mape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """MAPE, skipping near-zero actuals.

    Wind can sit near zero and the percentage blows up there.
    """
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p)) & (np.abs(a) > 1e-6)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2))) if mask.any() else float("nan")


def bias(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Mean signed error. Positive means the forecast runs high."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    return float(np.mean(p[mask] - a[mask])) if mask.any() else float("nan")


def score(actual, predicted) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "bias": bias(actual, predicted),
    }


def score_table(frame: pd.DataFrame, actual_col: str = "actual") -> pd.DataFrame:
    """Score every prediction column in a frame against the actuals."""
    rows = []
    for column in frame.columns:
        if column in {actual_col, "timestamp"}:
            continue
        rows.append({"model": column, **score(frame[actual_col], frame[column])})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
