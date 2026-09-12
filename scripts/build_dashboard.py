"""Builds dashboard.html from results/.

Data gets embedded as JSON so it's one file with no dependencies, which is what
GitHub Pages needs.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
OUTPUT = ROOT / "dashboard.html"

HEADLINE_HORIZON = 24.0


def build_payload() -> dict:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(RESULTS / "scores.csv")

    horizons = sorted(scores["horizon_hours"].unique())
    by_horizon = {
        "horizons": horizons,
        "model": [_mae(scores, h, "model") for h in horizons],
        "naive_last_week": [_mae(scores, h, "naive_last_week") for h in horizons],
        "naive_4week_mean": [_mae(scores, h, "naive_4week_mean") for h in horizons],
        "mape_model": [_metric(scores, h, "model", "mape") for h in horizons],
        "skill": [
            round(100 * (1 - _mae(scores, h, "model") / _mae(scores, h, "naive_last_week")), 1)
            for h in horizons
        ],
    }

    predictions = pd.read_parquet(RESULTS / f"predictions_{HEADLINE_HORIZON:g}h.parquet")
    predictions = predictions.dropna(subset=["actual", "model"]).reset_index(drop=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    error = predictions["model"] - predictions["actual"]
    predictions["abs_error"] = error.abs()
    predictions["signed_error"] = error

    by_hour = (
        predictions.assign(hour=predictions["timestamp"].dt.hour)
        .groupby("hour")
        .agg(mae=("abs_error", "mean"), bias=("signed_error", "mean"))
        .reindex(range(24))
        .round(1)
    )

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_weekday = (
        predictions.assign(dow=predictions["timestamp"].dt.dayofweek)
        .groupby("dow")
        .agg(mae=("abs_error", "mean"))
        .reindex(range(7))
        .round(1)
    )

    worst = (
        predictions.nlargest(8, "abs_error")[["timestamp", "actual", "model", "signed_error"]]
        .assign(timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%d %H:%M"))
        .round(0)
    )

    series = {
        "t": predictions["timestamp"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
        "actual": predictions["actual"].round(0).tolist(),
        "model": predictions["model"].round(0).tolist(),
        "naive": predictions["naive_last_week"].round(0).tolist()
        if "naive_last_week" in predictions
        else [],
    }

    headline = summary["by_horizon"][str(HEADLINE_HORIZON)]

    # generate the caption from the data - wrote it by hand first and it was
    # already wrong after one retrain
    hour_mae = by_hour["mae"]
    peak_hour = int(hour_mae.idxmax())
    quiet_hour = int(hour_mae.idxmin())
    hard_hours = sorted(hour_mae[hour_mae >= hour_mae.quantile(0.75)].index.tolist())
    hour_caption = (
        f"Mean absolute error at {HEADLINE_HORIZON:g} hours ahead, by hour of day. "
        f"Error peaks at {peak_hour:02d}:00 ({hour_mae.max():.0f} MW) and is lowest at "
        f"{quiet_hour:02d}:00 ({hour_mae.min():.0f} MW) &mdash; a spread of "
        f"{hour_mae.max() / max(hour_mae.min(), 1e-9):.1f}&times;. "
        f"The hardest quarter of the day is {_describe_hours(hard_hours)}."
    )

    return {
        "hourCaption": hour_caption,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": summary,
        "headline": {
            "horizon": HEADLINE_HORIZON,
            "mae": round(headline["model"]["mae"], 1),
            "mape": round(headline["model"]["mape"], 2),
            "skill": round(headline["skill_vs_last_week"] * 100, 1),
            "baseline_mae": round(headline["naive_last_week"]["mae"], 1),
            "bias": round(headline["model"]["bias"], 1),
        },
        "byHorizon": by_horizon,
        "byHour": {
            "hours": list(range(24)),
            "mae": by_hour["mae"].fillna(0).tolist(),
            "bias": by_hour["bias"].fillna(0).tolist(),
        },
        "byWeekday": {"names": weekday_names, "mae": by_weekday["mae"].fillna(0).tolist()},
        "worst": worst.to_dict("records"),
        "series": series,
    }


def _describe_hours(hours: list[int]) -> str:
    """[10,11,12,13,16] -> '10:00-13:00 and 16:00'."""
    if not hours:
        return "spread across the day"
    runs: list[list[int]] = [[hours[0]]]
    for hour in hours[1:]:
        if hour == runs[-1][-1] + 1:
            runs[-1].append(hour)
        else:
            runs.append([hour])
    parts = [
        f"{run[0]:02d}:00" if len(run) == 1 else f"{run[0]:02d}:00&ndash;{run[-1]:02d}:00"
        for run in runs
    ]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _mae(scores: pd.DataFrame, horizon: float, model: str) -> float:
    return _metric(scores, horizon, model, "mae")


def _metric(scores: pd.DataFrame, horizon: float, model: str, metric: str) -> float:
    row = scores[(scores["horizon_hours"] == horizon) & (scores["model"] == model)]
    return round(float(row[metric].iloc[0]), 2) if len(row) else float("nan")


def main() -> None:
    payload = build_payload()
    template = (ROOT / "scripts" / "dashboard_template.html").read_text(encoding="utf-8")
    html = template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT} ({size_kb:.0f} KB, {len(payload['series']['t'])} test points)")


if __name__ == "__main__":
    main()
