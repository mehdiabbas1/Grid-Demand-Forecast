"""Bulk download. Resumable.

Sweeps the (series, month) list repeatedly because of the throttling. Cached
months get skipped instantly so each sweep only retries what's missing, with a
longer pause each time. Kill it and restart whenever, it picks up where it was.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from gridcast.eirgrid import SERIES, EirGridError, FetchConfig, fetch_month, month_range  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("fetch.log"), logging.StreamHandler()],
)
LOG = logging.getLogger("fetch")

START = dt.date(2023, 1, 1)
END = dt.date(2026, 9, 1)
CACHE = Path("data/raw")
REGION = "ALL"
SWEEPS = 8


def cache_path(series: str, year: int, month: int) -> Path:
    return CACHE / REGION / series / f"{year}-{month:02d}.parquet"


def main() -> None:
    tasks = [
        (series, year, month)
        for series in SERIES
        for year, month in month_range(START, END)
    ]
    LOG.info("%d (series, month) tasks total", len(tasks))

    config = FetchConfig(max_retries=3, base_backoff=6.0, polite_delay=2.0)

    for sweep in range(1, SWEEPS + 1):
        pending = [t for t in tasks if not cache_path(*t).exists()]
        LOG.info("sweep %d/%d - %d tasks pending", sweep, SWEEPS, len(pending))
        if not pending:
            break

        for series, year, month in pending:
            try:
                fetch_month(series, year, month, region=REGION, cache_dir=CACHE, config=config)
            except EirGridError as exc:
                LOG.warning("deferred %s %04d-%02d (%s)", series, year, month, str(exc)[:70])
            time.sleep(config.polite_delay)

        still = [t for t in tasks if not cache_path(*t).exists()]
        if not still:
            break
        pause = 60 * sweep
        LOG.info("sweep %d done, %d still missing; pausing %ds", sweep, len(still), pause)
        time.sleep(pause)

    assemble()


def assemble() -> None:
    """Join all the cached months into one table."""
    merged: pd.DataFrame | None = None
    for series in SERIES:
        files = sorted((CACHE / REGION / series).glob("*.parquet"))
        if not files:
            LOG.warning("no cached months for %s", series)
            continue
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
        LOG.info("%s: %d months, %d rows", series, len(files), len(frame))
        merged = frame if merged is None else merged.merge(frame, on="timestamp", how="outer")

    if merged is None:
        LOG.error("nothing cached; aborting")
        return

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    merged.to_parquet("data/processed/grid_raw.parquet", index=False)
    LOG.info(
        "DONE rows=%d cols=%s range=%s..%s",
        len(merged), list(merged.columns), merged.timestamp.min(), merged.timestamp.max(),
    )


if __name__ == "__main__":
    main()
