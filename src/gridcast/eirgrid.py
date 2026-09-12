"""EirGrid Smart Grid Dashboard client.

No docs for this API, found it by watching the network tab on their dashboard.
It rate limits hard - after a few quick requests you get an HTML 503 page back
instead of JSON, which is why everything below retries and caches.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

BASE_URL = "https://www.smartgriddashboard.com/DashboardService.svc/data"

# Areas that actually work. Tried co2intensity and interconnection too,
# both 503 every time, so either they're gone or the name is different.
SERIES: dict[str, dict[str, str]] = {
    "demand": {"area": "demandactual", "field": "SYSTEM_DEMAND", "unit": "MW"},
    "wind": {"area": "windactual", "field": "WIND_ACTUAL", "unit": "MW"},
    "wind_forecast": {"area": "windforecast", "field": "WIND_FCAST", "unit": "MW"},
    "generation": {"area": "generationactual", "field": "GEN_EXP", "unit": "MW"},
    "snsp": {"area": "SnspALL", "field": "SNSP_ALL", "unit": "%"},
}

REGIONS = ("ALL", "ROI", "NI")

# Only accepts this exact date format, anything else 503s.
_DATE_FMT = "%d-%b-%Y+00%%3A00"


class EirGridError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchConfig:
    max_retries: int = 6
    base_backoff: float = 4.0
    polite_delay: float = 2.5
    timeout: int = 90


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end


def _fmt(value: date) -> str:
    return value.strftime(_DATE_FMT)


def fetch_window(
    series: str,
    start: date,
    end: date,
    region: str = "ALL",
    config: FetchConfig | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """One series, one window. Retries on the 503s."""
    if series not in SERIES:
        raise ValueError(f"Unknown series {series!r}; known: {sorted(SERIES)}")
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; known: {REGIONS}")

    config = config or FetchConfig()
    session = session or requests.Session()
    area = SERIES[series]["area"]
    url = f"{BASE_URL}?area={area}&region={region}&datefrom={_fmt(start)}&dateto={_fmt(end)}"

    last_error = ""
    for attempt in range(1, config.max_retries + 1):
        try:
            response = session.get(url, timeout=config.timeout, headers={"Accept": "application/json"})
            payload = response.json()
        except Exception as exc:  # broad on purpose, it returns HTML not an error code
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if payload.get("ErrorMessage"):
                last_error = str(payload["ErrorMessage"])
            else:
                rows = payload.get("Rows") or []
                LOG.info("%s %s %s..%s -> %d rows", series, region, start, end, len(rows))
                return _rows_to_frame(rows, series)

        wait = config.base_backoff * (2 ** (attempt - 1))
        LOG.warning(
            "%s %s %s..%s attempt %d/%d failed (%s); retrying in %.0fs",
            series, region, start, end, attempt, config.max_retries, last_error[:80], wait,
        )
        time.sleep(wait)

    raise EirGridError(
        f"{series} {region} {start}..{end} failed after {config.max_retries} attempts: {last_error[:200]}"
    )


def _rows_to_frame(rows: list[dict], series: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", series])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["EffectiveTime"], format="%d-%b-%Y %H:%M:%S")
    frame = frame.rename(columns={"Value": series})[["timestamp", series]]
    return frame.sort_values("timestamp").reset_index(drop=True)


def fetch_month(
    series: str,
    year: int,
    month: int,
    region: str = "ALL",
    cache_dir: Path | None = None,
    config: FetchConfig | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """One calendar month, cached to disk.

    Don't cache the current month, it's still filling up.
    """
    start, end = _month_bounds(year, month)
    today = datetime.now().date()
    is_current = start <= today < end

    cache_path = None
    if cache_dir is not None and not is_current:
        cache_path = Path(cache_dir) / region / series / f"{year}-{month:02d}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    frame = fetch_window(series, start, end, region=region, config=config, session=session)

    if cache_path is not None and not frame.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    cursor = date(start.year, start.month, 1)
    while cursor < end:
        months.append((cursor.year, cursor.month))
        cursor = date(cursor.year + 1, 1, 1) if cursor.month == 12 else date(cursor.year, cursor.month + 1, 1)
    return months


def fetch_series(
    series: str,
    start: date,
    end: date,
    region: str = "ALL",
    cache_dir: Path | None = None,
    config: FetchConfig | None = None,
    skip_errors: bool = False,
) -> pd.DataFrame:
    """One series across a date range, a month at a time.

    skip_errors=True means a month that won't download gets logged and skipped
    instead of killing the run. Cached months mean a re-run only fills gaps.
    """
    config = config or FetchConfig()
    session = requests.Session()
    frames = []
    for year, month in month_range(start, end):
        try:
            frames.append(
                fetch_month(series, year, month, region=region, cache_dir=cache_dir, config=config, session=session)
            )
        except EirGridError as exc:
            if not skip_errors:
                raise
            LOG.error("skipping %s %04d-%02d: %s", series, year, month, str(exc)[:120])
        time.sleep(config.polite_delay)
    if not frames:
        return pd.DataFrame(columns=["timestamp", series])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return combined[(combined["timestamp"] >= pd.Timestamp(start)) & (combined["timestamp"] < pd.Timestamp(end))]


def fetch_all(
    start: date,
    end: date,
    series_names: list[str] | None = None,
    region: str = "ALL",
    cache_dir: Path | None = None,
    config: FetchConfig | None = None,
) -> pd.DataFrame:
    """Fetch several series and join them on timestamp."""
    names = series_names or ["demand", "wind", "wind_forecast", "generation", "snsp"]
    merged: pd.DataFrame | None = None
    for name in names:
        frame = fetch_series(name, start, end, region=region, cache_dir=cache_dir, config=config)
        merged = frame if merged is None else merged.merge(frame, on="timestamp", how="outer")
    result = (merged if merged is not None else pd.DataFrame()).sort_values("timestamp")
    return result.reset_index(drop=True)
