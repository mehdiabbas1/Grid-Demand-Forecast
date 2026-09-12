"""CSVs for Power BI.

Power BI does read Parquet but the types come through wrong and you end up
fixing them by hand, so write CSVs instead. Split into a star schema rather
than one flat sheet.

    fact_predictions.csv   one row per forecast time per horizon
    dim_date.csv           date table, Power BI needs one for time intelligence
    dim_horizon.csv        one row per horizon + its scores
    scores_long.csv        model x horizon x metric
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
OUT = ROOT / "powerbi"

PRETTY = {
    "model": "Model",
    "naive_last_week": "Naive: same time last week",
    "naive_4week_mean": "Naive: 4-week average",
    "naive_yesterday": "Naive: same time yesterday",
    "persistence": "Persistence",
    "published_forecast": "EirGrid published forecast",
}


def build_fact() -> pd.DataFrame:
    """Long form: one row per forecast time per horizon."""
    frames = []
    for path in sorted(RESULTS.glob("predictions_*h.parquet")):
        horizon = float(path.stem.replace("predictions_", "").replace("h", ""))
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.dropna(subset=["actual", "model"])

        out = pd.DataFrame(
            {
                "Timestamp": frame["timestamp"],
                "Date": frame["timestamp"].dt.normalize(),
                "HorizonHours": horizon,
                "ActualMW": frame["actual"].round(1),
                "ForecastMW": frame["model"].round(1),
            }
        )
        if "naive_last_week" in frame:
            out["NaiveMW"] = frame["naive_last_week"].round(1)

        out["ErrorMW"] = (out["ForecastMW"] - out["ActualMW"]).round(1)
        out["AbsErrorMW"] = out["ErrorMW"].abs()
        out["AbsPctError"] = (out["AbsErrorMW"] / out["ActualMW"].abs() * 100).round(3)
        if "NaiveMW" in out:
            out["NaiveAbsErrorMW"] = (out["NaiveMW"] - out["ActualMW"]).abs().round(1)

        frames.append(out)

    fact = pd.concat(frames, ignore_index=True)
    return fact.sort_values(["HorizonHours", "Timestamp"]).reset_index(drop=True)


def build_dim_date(fact: pd.DataFrame) -> pd.DataFrame:
    """Date table. Has to be contiguous or Power BI time intelligence breaks."""
    dates = pd.date_range(fact["Date"].min(), fact["Date"].max(), freq="D")
    dim = pd.DataFrame({"Date": dates})
    dim["Year"] = dim["Date"].dt.year
    dim["Month"] = dim["Date"].dt.month
    dim["MonthName"] = dim["Date"].dt.strftime("%b")
    dim["MonthSort"] = dim["Date"].dt.year * 100 + dim["Date"].dt.month
    dim["Day"] = dim["Date"].dt.day
    dim["DayOfWeek"] = dim["Date"].dt.dayofweek + 1          # 1 = Monday, for sorting
    dim["DayName"] = dim["Date"].dt.strftime("%a")
    dim["IsWeekend"] = dim["Date"].dt.dayofweek >= 5
    dim["WeekStarting"] = (dim["Date"] - pd.to_timedelta(dim["Date"].dt.dayofweek, unit="D")).dt.date

    try:
        import holidays as holidays_pkg

        calendar = holidays_pkg.Ireland(years=sorted(dim["Year"].unique().tolist()))
        dim["IsHoliday"] = dim["Date"].dt.date.map(lambda d: d in calendar)
        dim["HolidayName"] = dim["Date"].dt.date.map(lambda d: calendar.get(d) or "")
    except ImportError:
        dim["IsHoliday"] = False
        dim["HolidayName"] = ""

    return dim


def build_dim_horizon() -> pd.DataFrame:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for horizon, entry in summary["by_horizon"].items():
        rows.append(
            {
                "HorizonHours": float(horizon),
                "HorizonLabel": f"{float(horizon):g} h ahead",
                "ModelMAE": round(entry["model"]["mae"], 2),
                "ModelMAPE": round(entry["model"]["mape"], 3),
                "ModelBias": round(entry["model"]["bias"], 2),
                "NaiveMAE": round(entry["naive_last_week"]["mae"], 2),
                "SkillPct": round(entry["skill_vs_last_week"] * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("HorizonHours").reset_index(drop=True)


def build_scores_long() -> pd.DataFrame:
    scores = pd.read_csv(RESULTS / "scores.csv")
    scores["ModelName"] = scores["model"].map(lambda m: PRETTY.get(m, m))
    scores = scores.rename(
        columns={"horizon_hours": "HorizonHours", "mae": "MAE", "rmse": "RMSE",
                 "mape": "MAPE", "bias": "Bias"}
    )
    scores["IsBaseline"] = scores["model"] != "model"
    return scores[["HorizonHours", "ModelName", "IsBaseline", "MAE", "RMSE", "MAPE", "Bias"]].round(3)


def main() -> None:
    if not (RESULTS / "summary.json").exists():
        raise SystemExit("No results yet - run scripts/run_experiment.py first.")

    OUT.mkdir(exist_ok=True)
    fact = build_fact()

    fact.to_csv(OUT / "fact_predictions.csv", index=False)
    build_dim_date(fact).to_csv(OUT / "dim_date.csv", index=False)
    build_dim_horizon().to_csv(OUT / "dim_horizon.csv", index=False)
    build_scores_long().to_csv(OUT / "scores_long.csv", index=False)

    print(f"fact_predictions.csv  {len(fact):>6} rows  "
          f"({fact['HorizonHours'].nunique()} horizons x {fact['Timestamp'].nunique()} timestamps)")
    for name in ("dim_date", "dim_horizon", "scores_long"):
        rows = len(pd.read_csv(OUT / f"{name}.csv"))
        print(f"{name}.csv{' ' * (22 - len(name) - 4)}{rows:>6} rows")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
