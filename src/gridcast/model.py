"""LightGBM, one model per horizon.

Separate models rather than one with horizon as a feature. At 1h the last
observed value is everything, at 24h it's nearly useless and the calendar does
the work. Tried it as one model first and it was worse at both ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast import baselines
from gridcast.features import build_features, feature_columns, time_split

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "l1",  # same as the metric I report, so no mismatch
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}


@dataclass
class HorizonResult:
    horizon_hours: float
    scores: dict[str, dict[str, float]]
    best_iteration: int
    n_train: int
    n_test: int
    predictions: pd.DataFrame = field(repr=False)
    importance: pd.DataFrame = field(repr=False)


def train_one_horizon(
    frame: pd.DataFrame,
    target: str,
    horizon_hours: float,
    raw_series: tuple[str, ...],
    valid_start: str,
    test_start: str,
    lag_columns: tuple[str, ...] | None = None,
    known_future_columns: tuple[str, ...] = (),
    published_forecast: str | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 3000,
    early_stopping: int = 100,
) -> HorizonResult:
    table = build_features(
        frame,
        target=target,
        horizon_hours=horizon_hours,
        lag_columns=lag_columns,
        known_future_columns=known_future_columns,
    )
    base = baselines.build_baselines(frame, target, horizon_hours, published_forecast=published_forecast)
    table = table.merge(base.drop(columns=["actual"]), on="timestamp", how="left")

    features = feature_columns(table, target=target, raw_series=raw_series)
    table = table.dropna(subset=["y"]).reset_index(drop=True)

    splits = time_split(table, valid_start=valid_start, test_start=test_start)
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            f"empty split at horizon {horizon_hours}h "
            f"(train={len(train)}, valid={len(valid)}, test={len(test)})"
        )

    train_set = lgb.Dataset(train[features], label=train["y"], free_raw_data=False)
    valid_set = lgb.Dataset(valid[features], label=valid["y"], reference=train_set, free_raw_data=False)

    booster = lgb.train(
        {**DEFAULT_PARAMS, **(params or {})},
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
    )

    predictions = test[["timestamp", "y"]].rename(columns={"y": "actual"}).copy()
    predictions["model"] = booster.predict(test[features], num_iteration=booster.best_iteration)
    for column in ("naive_last_week", "naive_yesterday", "naive_4week_mean", "persistence", "published_forecast"):
        if column in test.columns:
            predictions[column] = test[column].to_numpy()

    scores = {
        column: baselines.score(predictions["actual"], predictions[column])
        for column in predictions.columns
        if column not in {"timestamp", "actual"}
    }

    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    return HorizonResult(
        horizon_hours=horizon_hours,
        scores=scores,
        best_iteration=int(booster.best_iteration or 0),
        n_train=len(train),
        n_test=len(test),
        predictions=predictions,
        importance=importance,
    )


def run_horizons(
    frame: pd.DataFrame,
    target: str,
    horizons: tuple[float, ...],
    raw_series: tuple[str, ...],
    valid_start: str,
    test_start: str,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[float, HorizonResult]]:
    """Train all horizons, return scores + results."""
    rows = []
    results: dict[float, HorizonResult] = {}
    for horizon in horizons:
        result = train_one_horizon(
            frame,
            target=target,
            horizon_hours=horizon,
            raw_series=raw_series,
            valid_start=valid_start,
            test_start=test_start,
            **kwargs,
        )
        results[horizon] = result
        for model_name, metrics in result.scores.items():
            rows.append({"horizon_hours": horizon, "model": model_name, **metrics})
    return pd.DataFrame(rows), results


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """How much of the baseline's error we removed. 0 = no better, 1 = perfect."""
    if not np.isfinite(baseline_mae) or baseline_mae == 0:
        return float("nan")
    return 1.0 - (model_mae / baseline_mae)
