"""Builds the training table.

Standing at time t, predict t+h. So every feature has to be something you'd
actually have at t. Easy to get wrong and the metrics won't tell you - see
tests/test_no_leakage.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_pkg
except ImportError:  # pragma: no cover
    holidays_pkg = None

# Half-hourly because that's the settlement period here. Demand/wind/gen come
# at 15 min, SNSP at 30, so 30 is the common denominator.
RESOLUTION = "30min"
PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = PERIODS_PER_DAY * 7


def to_half_hourly(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Put every series on a regular half-hourly index."""
    out = frame.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.set_index(timestamp_col).sort_index()
    numeric = out.select_dtypes(include="number")
    resampled = numeric.resample(RESOLUTION).mean()
    return resampled.reset_index()


def add_calendar_features(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Calendar features, on the target time.

    Not leakage - you always know what hour next Tuesday is.
    """
    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col])

    minutes = ts.dt.hour * 60 + ts.dt.minute
    day_fraction = minutes / (24 * 60)
    out["tod_sin"] = np.sin(2 * np.pi * day_fraction)
    out["tod_cos"] = np.cos(2 * np.pi * day_fraction)

    doy_fraction = ts.dt.dayofyear / 365.25
    out["doy_sin"] = np.sin(2 * np.pi * doy_fraction)
    out["doy_cos"] = np.cos(2 * np.pi * doy_fraction)

    out["dayofweek"] = ts.dt.dayofweek
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["month"] = ts.dt.month
    out["hour"] = ts.dt.hour

    out["is_holiday"] = _irish_holiday_flag(ts)
    # The day either side of a public holiday behaves unlike a normal weekday.
    out["is_holiday_adjacent"] = (
        _irish_holiday_flag(ts + pd.Timedelta(days=1)) | _irish_holiday_flag(ts - pd.Timedelta(days=1))
    ) & (1 - out["is_holiday"])
    return out


def _irish_holiday_flag(ts: pd.Series) -> pd.Series:
    if holidays_pkg is None:
        return pd.Series(0, index=ts.index, dtype=int)
    years = range(int(ts.dt.year.min()), int(ts.dt.year.max()) + 1)
    calendar = holidays_pkg.Ireland(years=list(years))
    return ts.dt.date.map(lambda d: int(d in calendar)).astype(int)


def add_origin_lags(
    frame: pd.DataFrame,
    column: str,
    horizon_periods: int,
    lags_periods: tuple[int, ...] = (0, 1, 2, PERIODS_PER_DAY, PERIODS_PER_WEEK),
    rolling_windows: tuple[int, ...] = (PERIODS_PER_DAY, PERIODS_PER_WEEK),
) -> pd.DataFrame:
    """Lags of `column` as known at the origin.

    Row i is the target at t_i, origin is t_i - horizon. So lag L sits at
    t_i - horizon - L. Shifting by horizon+lag, not just lag, is the whole point.
    """
    out = frame.copy()
    for lag in lags_periods:
        out[f"{column}_lag{lag}"] = out[column].shift(horizon_periods + lag)
    for window in rolling_windows:
        shifted = out[column].shift(horizon_periods)
        out[f"{column}_roll{window}_mean"] = shifted.rolling(window, min_periods=window // 2).mean()
        out[f"{column}_roll{window}_std"] = shifted.rolling(window, min_periods=window // 2).std()
    return out


def build_features(
    frame: pd.DataFrame,
    target: str,
    horizon_hours: float,
    lag_columns: tuple[str, ...] | None = None,
    known_future_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the table for one horizon.

    known_future_columns is for things you really do have in advance, like
    EirGrid's own published wind forecast. Those get used at target time.
    Has to be opted into by name so nothing slips in by accident.
    """
    horizon_periods = int(round(horizon_hours * 2))
    if horizon_periods < 1:
        raise ValueError("horizon_hours must be at least half an hour")

    out = add_calendar_features(frame)
    for column in lag_columns or (target,):
        if column in out.columns:
            out = add_origin_lags(out, column, horizon_periods)

    for column in known_future_columns:
        if column in out.columns:
            out[f"{column}_known"] = out[column]

    out["horizon_hours"] = horizon_hours
    out["y"] = out[target]
    return out


def feature_columns(frame: pd.DataFrame, target: str, raw_series: tuple[str, ...]) -> list[str]:
    """Feature columns, minus anything that would leak.

    Raw series get excluded by name - at the origin you don't know what
    demand or wind will be at the target time.
    """
    banned = {"timestamp", "y", target, *raw_series}
    return [c for c in frame.columns if c not in banned and pd.api.types.is_numeric_dtype(frame[c])]


def time_split(
    frame: pd.DataFrame,
    valid_start: str,
    test_start: str,
    timestamp_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """Chronological split. Don't shuffle time series."""
    ts = pd.to_datetime(frame[timestamp_col])
    return {
        "train": frame[ts < valid_start].copy(),
        "valid": frame[(ts >= valid_start) & (ts < test_start)].copy(),
        "test": frame[ts >= test_start].copy(),
    }
