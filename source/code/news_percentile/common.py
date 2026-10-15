from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
RESULTS = ROOT / "results" / "news_percentile_v2"
VOL_RESULTS = ROOT / "results" / "vol_score"
DATA = ROOT / "data"
SOURCE = Path(r"C:\Users\Boxuan\Documents\GitHub\ReutersNewstTraining")
RAW_REUTERS = SOURCE / "Reuters"
PROCESSED = SOURCE / "processed"

CRSP = DATA / "macro" / "exp_crsp_daily_compact.parquet"
RIC_MAP = PROCESSED / "ric_to_ticker_map.csv"
RIC_NAMES = PROCESSED / "ric_names_corpus.parquet"
CRSP_NAMES = DATA / "gn" / "gn_crsp_names.parquet"

RIC_RE = re.compile(r"(?:^|\|)R:([^|]+)")


def load_config(path: str | Path | None = None) -> dict:
    target = Path(path) if path else Path(__file__).with_name("config.json")
    return json.loads(target.read_text(encoding="utf-8"))


def ensure_dirs() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for child in ["manifests", "scores", "prompts"]:
        (RESULTS / child).mkdir(exist_ok=True)
    return RESULTS


def stable_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cutoff_utc(dates: pd.DatetimeIndex, hour_et: int) -> pd.DatetimeIndex:
    naive = pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=hour_et)
    return naive.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def assign_decision_date(
    timestamps: pd.Series,
    trading_dates: pd.DatetimeIndex,
    hour_et: int,
) -> pd.Series:
    cutoffs = cutoff_utc(trading_dates, hour_et)
    values = pd.to_datetime(timestamps, utc=True, errors="coerce")
    positions = np.searchsorted(cutoffs.asi8, values.astype("int64"), side="left")
    output = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = values.notna().to_numpy() & (positions < len(cutoffs))
    output[valid] = trading_dates.values[positions[valid]]
    return pd.Series(pd.to_datetime(output), index=timestamps.index)


def experiment_dates(cfg: dict, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    spans = [
        (cfg["reflection_start"], cfg["reflection_end"]),
        (cfg["selection_start"], cfg["selection_end"]),
        (cfg["test_start"], cfg["test_end"]),
    ]
    selected: list[pd.Timestamp] = []
    for lo, hi in spans:
        selected.extend(calendar[(calendar >= pd.Timestamp(lo)) & (calendar <= pd.Timestamp(hi))])
    return pd.DatetimeIndex(sorted(set(selected)))


def period_bounds(cfg: dict, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    names = {
        "reflection": ("reflection_start", "reflection_end"),
        "selection": ("selection_start", "selection_end"),
        "test": ("test_start", "test_end"),
    }
    if period not in names:
        raise KeyError(f"Unknown period {period}")
    lo, hi = names[period]
    return pd.Timestamp(cfg[lo]), pd.Timestamp(cfg[hi])


def empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        raise ValueError("Empty percentile reference")
    val = np.asarray(values, dtype=float)
    return (np.searchsorted(ref, val, side="right") + 0.5) / (len(ref) + 1.0)


def empirical_quantile(percentiles: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        raise ValueError("Empty quantile reference")
    p = np.clip(np.asarray(percentiles, dtype=float), 0.005, 0.995)
    grid = np.linspace(0.0, 1.0, len(ref))
    return np.interp(p, grid, ref)
