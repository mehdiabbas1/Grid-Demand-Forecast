"""Leakage tests.

If a feature accidentally contains data from after the forecast was made, the
model scores great and is useless, and nothing in the metrics shows it. These
catch that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast import baselines
from gridcast.features import (
    PERIODS_PER_DAY,
    PERIODS_PER_WEEK,
    build_features,
    feature_columns,
    time_split,
    to_half_hourly,
)


@pytest.fixture
def series() -> pd.DataFrame:
    """Series where every value is unique.

    Means I can trace any feature value back to the timestamp it came from.
    """
    index = pd.date_range("2024-01-01", periods=PERIODS_PER_WEEK * 8, freq="30min")
    return pd.DataFrame({"timestamp": index, "demand": np.arange(len(index), dtype=float)})


@pytest.mark.parametrize("horizon_hours", [0.5, 1.0, 6.0, 24.0, 48.0])
def test_every_lag_predates_the_forecast_origin(series, horizon_hours):
    """No lag feature may come from after target_time - horizon."""
    table = build_features(series, target="demand", horizon_hours=horizon_hours)
    lookup = dict(zip(series["timestamp"], series["demand"]))
    horizon = pd.Timedelta(hours=horizon_hours)

    row = table.iloc[-1]
    target_time = row["timestamp"]
    origin = target_time - horizon

    for column in table.columns:
        if not column.startswith("demand_lag"):
            continue
        value = row[column]
        if np.isnan(value):
            continue
        # value == position in the series, so it identifies its own timestamp
        source_time = series.loc[series["demand"] == value, "timestamp"].iloc[0]
        assert source_time <= origin, (
            f"{column} at horizon {horizon_hours}h used {source_time}, "
            f"which is after the forecast origin {origin}"
        )
        assert lookup[source_time] == value


@pytest.mark.parametrize("horizon_hours", [1.0, 24.0])
def test_rolling_windows_predate_the_origin(series, horizon_hours):
    """Rolling window must not reach into the target period."""
    table = build_features(series, target="demand", horizon_hours=horizon_hours)
    horizon_periods = int(horizon_hours * 2)

    row_index = len(table) - 1
    value = table[f"demand_roll{PERIODS_PER_DAY}_mean"].iloc[row_index]
    window_end = row_index - horizon_periods
    expected = series["demand"].iloc[window_end - PERIODS_PER_DAY + 1 : window_end + 1].mean()
    assert value == pytest.approx(expected)


def test_raw_target_is_not_a_feature(series):
    table = build_features(series, target="demand", horizon_hours=24.0)
    features = feature_columns(table, target="demand", raw_series=("demand",))
    assert "demand" not in features
    assert "y" not in features
    assert features, "expected at least one usable feature"


def test_contemporaneous_series_are_excluded(series):
    """A second series measured at target time must not get in."""
    frame = series.copy()
    frame["wind"] = frame["demand"] * 0.5
    table = build_features(frame, target="demand", horizon_hours=24.0, lag_columns=("demand", "wind"))
    features = feature_columns(table, target="demand", raw_series=("demand", "wind"))
    assert "wind" not in features
    assert "wind_lag0" in features


def test_known_future_column_is_kept_deliberately(series):
    """Published forecasts are allowed at target time, but only by name."""
    frame = series.copy()
    frame["wind_forecast"] = frame["demand"] * 0.3
    table = build_features(
        frame,
        target="demand",
        horizon_hours=24.0,
        lag_columns=("demand",),
        known_future_columns=("wind_forecast",),
    )
    features = feature_columns(table, target="demand", raw_series=("demand", "wind_forecast"))
    assert "wind_forecast_known" in features, "opt-in future column should be available"
    assert "wind_forecast" not in features, "the raw column must still be excluded"


@pytest.mark.parametrize("horizon_hours", [1.0, 24.0])
def test_baselines_are_available_at_the_origin(series, horizon_hours):
    """Baselines go in as features so they follow the same rule."""
    base = baselines.build_baselines(series, "demand", horizon_hours)
    horizon = pd.Timedelta(hours=horizon_hours)
    row = base.iloc[-1]
    target_time = row["timestamp"]
    origin = target_time - horizon

    offsets = {
        "naive_last_week": pd.Timedelta(days=7),
        "naive_yesterday": pd.Timedelta(days=1),
        "persistence": horizon,
    }
    for column, offset in offsets.items():
        if np.isnan(row[column]):
            continue
        assert target_time - offset <= origin, (
            f"{column} at horizon {horizon_hours}h reads {target_time - offset}, after origin {origin}"
        )


def test_splits_are_chronological_and_disjoint(series):
    table = build_features(series, target="demand", horizon_hours=24.0)
    splits = time_split(table, valid_start="2024-02-01", test_start="2024-02-15")
    assert splits["train"]["timestamp"].max() < splits["valid"]["timestamp"].min()
    assert splits["valid"]["timestamp"].max() < splits["test"]["timestamp"].min()
    total = sum(len(part) for part in splits.values())
    assert total == len(table), "splits must partition the data with no overlap or loss"


def test_resampling_does_not_invent_data():
    index = pd.date_range("2024-01-01", periods=8, freq="15min")
    frame = pd.DataFrame({"timestamp": index, "demand": [1.0, 2, 3, 4, np.nan, np.nan, 7, 8]})
    out = to_half_hourly(frame)
    assert len(out) == 4
    assert out["demand"].iloc[0] == pytest.approx(1.5)
    assert np.isnan(out["demand"].iloc[2]), "a fully missing half hour must stay missing"


def test_mape_ignores_near_zero_actuals():
    actual = np.array([0.0, 100.0])
    predicted = np.array([50.0, 110.0])
    assert baselines.mape(actual, predicted) == pytest.approx(10.0)


def test_bias_sign_is_forecast_minus_actual():
    assert baselines.bias([100.0], [110.0]) == pytest.approx(10.0)
    assert baselines.bias([100.0], [90.0]) == pytest.approx(-10.0)
