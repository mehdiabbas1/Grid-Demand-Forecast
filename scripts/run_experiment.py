"""Trains every horizon and writes results/.

    python scripts/run_experiment.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from gridcast.features import to_half_hourly  # noqa: E402
from gridcast.model import run_horizons, skill_score  # noqa: E402

HORIZONS = (1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
RESULTS = Path("results")


def load_series(name: str, region: str = "ALL") -> pd.DataFrame | None:
    files = sorted(glob.glob(f"data/raw/{region}/{name}/*.parquet"))
    if not files:
        return None
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def longest_complete_span(frame: pd.DataFrame, column: str, min_coverage: float = 0.5):
    """Longest run of months that are mostly there.

    The download always has holes because of the rate limiting. Training across
    a hole quietly breaks every lag feature that spans it, so find the longest
    clean stretch and use that.
    """
    monthly = frame.set_index("timestamp")[column].resample("MS").apply(lambda s: s.notna().mean())
    runs, current = [], []
    for month, coverage in monthly.items():
        if coverage >= min_coverage:
            current.append(month)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        raise SystemExit(f"no usable months for {column}")
    best = max(runs, key=len)
    return best[0], best[-1] + pd.offsets.MonthBegin(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="demand")
    parser.add_argument("--region", default="ALL")
    parser.add_argument("--valid-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    args = parser.parse_args()

    raw = load_series(args.target, args.region)
    if raw is None:
        raise SystemExit(f"no cached data for {args.target}; run fetch_history.py first")

    frame = to_half_hourly(raw)
    start, end = longest_complete_span(frame, args.target)
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].reset_index(drop=True)

    test_start = end - pd.DateOffset(months=args.test_months)
    valid_start = test_start - pd.DateOffset(months=args.valid_months)
    print(f"span {start.date()} .. {end.date()}  ({len(frame)} half-hours)")
    print(f"train < {valid_start.date()} | valid < {test_start.date()} | test < {end.date()}")

    # only bring wind in if enough of it downloaded to cover the span
    lag_columns = [args.target]
    wind = load_series("wind", args.region)
    if wind is not None:
        wind_hh = to_half_hourly(wind)
        merged = frame.merge(wind_hh, on="timestamp", how="left")
        coverage = merged["wind"].notna().mean()
        if coverage > 0.9:
            frame = merged
            lag_columns.append("wind")
            print(f"including wind as an input ({coverage:.0%} covered)")
        else:
            print(f"skipping wind - only {coverage:.0%} of the span is downloaded")

    scores, results = run_horizons(
        frame,
        target=args.target,
        horizons=HORIZONS,
        raw_series=tuple(lag_columns),
        valid_start=str(valid_start.date()),
        test_start=str(test_start.date()),
        lag_columns=tuple(lag_columns),
    )

    RESULTS.mkdir(exist_ok=True)
    scores.to_csv(RESULTS / "scores.csv", index=False)

    summary = {
        "target": args.target,
        "region": args.region,
        "span_start": str(start.date()),
        "span_end": str(end.date()),
        "valid_start": str(valid_start.date()),
        "test_start": str(test_start.date()),
        "inputs": lag_columns,
        "horizons": list(HORIZONS),
        "n_train": results[HORIZONS[0]].n_train,
        "n_test": results[HORIZONS[0]].n_test,
        "by_horizon": {},
    }

    importances = []
    for horizon, result in results.items():
        result.predictions.to_parquet(RESULTS / f"predictions_{horizon:g}h.parquet", index=False)
        imp = result.importance.copy()
        imp["horizon_hours"] = horizon
        importances.append(imp)

        model_mae = result.scores["model"]["mae"]
        base_mae = result.scores["naive_last_week"]["mae"]
        summary["by_horizon"][str(horizon)] = {
            "model": result.scores["model"],
            "naive_last_week": result.scores["naive_last_week"],
            "naive_4week_mean": result.scores["naive_4week_mean"],
            "skill_vs_last_week": skill_score(model_mae, base_mae),
            "best_iteration": result.best_iteration,
        }

    pd.concat(importances, ignore_index=True).to_csv(RESULTS / "importance.csv", index=False)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(scores.pivot(index="model", columns="horizon_hours", values="mae").round(1).to_string())
    print()
    for horizon in HORIZONS:
        entry = summary["by_horizon"][str(horizon)]
        print(
            f"  {horizon:>5g}h  MAE {entry['model']['mae']:6.1f} MW  "
            f"MAPE {entry['model']['mape']:5.2f}%  "
            f"skill vs last-week {entry['skill_vs_last_week'] * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
