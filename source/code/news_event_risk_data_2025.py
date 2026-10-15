"""CPU-only numerical inputs and inspectable, point-in-time Reuters evidence.

Public APIs: prepare_numerical(out) returns cases and writes labels_sealed.parquet;
prepare_evidence(out, cases) returns {case_id: packet}. Neither returns labels in
inputs. Event calendars are supplied separately by the coordinating runner.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

import numpy as np
import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/news_event_risk_2025"
TICKERS = ("NVDA", "MSFT", "CAT", "XPO", "BBY", "AMD", "CFLT", "APLD", "UPS", "WMT")
TRAIN = pd.date_range("2025-08-01", "2025-10-03", freq="W-FRI")
VALID = pd.date_range("2025-10-17", "2025-11-07", freq="W-FRI")
TEST = pd.date_range("2025-11-21", "2025-12-19", freq="W-FRI")
MONTHS = tuple(f"2025-{m:02d}" for m in range(7, 13))
SOURCE = Path(r"C:\Users\Boxuan\Documents\GitHub\ReutersNewstTraining")
RAW_NEWS = SOURCE / "Reuters"
UNIVERSE = ROOT / "results/thematic_universe_2025/universe.csv"
ANCHOR_LABELS = ROOT / "results/thematic_weekly_learning_2025_8w/labels_and_benchmarks_sealed.parquet"
FROZEN_ANCHOR = ROOT / "results/news_agent_batched_2025/anchor.json"
TRADING = ROOT / "data/macro/cal_trading_days.parquet"
MAX_PACKET_CHARS = 12000
EXPOSED_HISTORY_SESSIONS = 22
SCHEMA_VERSION = 1
MACRO_TAGS = {"N2:MCE", "N2:ECI", "N2:ECON", "N2:CEN", "N2:FED", "N2:INFL", "N2:TRD"}
FINANCE_TAGS = {"N2:STX", "N2:FRX", "N2:DBT", "N2:GVD", "N2:INT", "N2:USDEBT"}
ALIASES = {"NVDA": ("Nvidia",), "MSFT": ("Microsoft",), "CAT": ("Caterpillar",),
           "XPO": ("XPO",), "BBY": ("Best Buy",), "AMD": ("Advanced Micro Devices", "AMD"),
           "CFLT": ("Confluent",), "APLD": ("Applied Digital",),
           "UPS": ("United Parcel Service", "UPS"), "WMT": ("Walmart", "Wal-Mart")}
EVENT_RE = re.compile(r"\b(?:earnings|results|guidance|forecast|tariffs?|export|antitrust|"
                      r"lawsuit|court|acquisition|merger|takeover|financing|debt|strike|"
                      r"outage|shutdown|launch|contract|inflation|payrolls?|rates?|FOMC)\b", re.I)
POLICY = {
    "version": "label-free-relevance-v2", "cutoff": "strictly_before_16_ET_including_half_days",
    "availability": "max(firstCreated,versionCreated,recorded_timestamps)",
    "version_rule": "latest_known_version_before_relevance_and_status_filters",
    "own": "date_valid_native_RIC_tag_only",
    "common": "US_macro_or_finance_headline_classifier_or_native_topics_without_company_tags",
    "neighbors": False, "lookback_days": 31, "max_packet_chars": MAX_PACKET_CHARS,
    "ranking": ["recent_8_calendar_days", "firm_or_US_macro_focus", "unrepresented_family",
                "available_at_desc", "financial_event_terms", "source_id"],
    "source_quality": "raw_verified_verbatim_no_generated_judgments_shared_by_both_arms",
}
NON_SUBSTANTIVE = re.compile(r"^(?:DIARY|TIMELINE|Newscasts|COMMENT-|BUZZ-.*?FX Markets Open)|\b(?:fashion|racing|sponsorship|mentorship|brand ambassador)\b", re.I)
US_MACRO = re.compile(r"\b(?:Fed(?:eral Reserve)?|FOMC|U\.?S\.?|Wall Street|Treasur(?:y|ies)|S&P 500|Nasdaq|Trump|American)\b", re.I)
ECONOMIC = re.compile(r"\b(?:inflation|CPI|PPI|payrolls?|jobs?|employment|GDP|rates?|yields?|tariffs?|shutdown|budget|debt|trade|stocks?|markets?|dollar|lending|credit|bonds?|policy|monetary|economy|growth|recession)\b", re.I)
FIRM_MATERIAL = re.compile(r"\b(?:earnings|results|guidance|revenue|profits?|forecasts?|investments?|capacity|data cent(?:er|re)s?|cloud|AI|chips?|semiconductors?|exports?|contracts?|acquisition|merger|regulator|antitrust|outage|lawsuit|strike|demand|sales|costs?|financing|debt|leases?)\b", re.I)
_IMPORT_LOCK = threading.RLock()


def serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(serialized(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_stamp(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def provenance(path):
    return {**file_stamp(path), "sha256": file_hash(path)}


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
        with temp.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temp:
        temp.write_text(serialized(value) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def isolated_module(name, path, bindings=None):
    """Bind legacy absolute imports in this module only; leave sys.modules intact."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if bindings:
        def bound_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in bindings:
                return bindings[name]
            return builtins.__import__(name, globals, locals, fromlist, level)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=bound_import)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def numerical_helpers():
    folder = ROOT / "code/midcap_standard_2025"
    config = isolated_module("_event_risk_benchmark_config", folder / "config.py")
    benchmark = isolated_module("_event_risk_benchmark", folder / "benchmarks.py", {"config": config})
    prepare = isolated_module("_event_risk_prepare", ROOT / "code/midcap_tools/prepare.py")
    return prepare, benchmark


@lru_cache(maxsize=1)
def news_helpers():
    prepare, _ = numerical_helpers()
    with _IMPORT_LOCK:
        parser = prepare.news_helpers()
    return prepare, parser


def cutoff(date):
    return (pd.Timestamp(date).normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")


def horizon(calendar, date):
    date = pd.Timestamp(date)
    if date not in calendar:
        raise ValueError("Origin is not an observed trading session")
    future = calendar[calendar > date][:6]
    if len(future) != 6:
        raise ValueError("Five outcome sessions plus the next maturity session are required")
    matured = (future[5] + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York").tz_convert("UTC")
    return [d.date().isoformat() for d in future[:5]], matured.isoformat()


def exact_target(future):
    values = np.asarray(future, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Target requires exactly five finite decimal total returns")
    variance = float(np.mean(values ** 2))
    if variance <= 0:
        raise ValueError("Exact log target is undefined for an all-zero window")
    return float(np.log(variance))


def frozen_anchor():
    prior = pd.read_parquet(ANCHOR_LABELS, filters=[("date", "<", "2025-08-01"),
                                                 ("ticker", "in", list(TICKERS))],
                            columns=["case_id", "ticker", "date", "available_at", "y", "garch11"])
    if (len(prior) != 290 or prior.case_id.duplicated().any()
            or prior.groupby("ticker").size().to_dict() != {t: 29 for t in TICKERS}
            or not np.isfinite(prior[["y", "garch11"]]).all().all()
            or not (pd.to_datetime(prior.available_at, utc=True) < pd.Timestamp("2025-08-01", tz="UTC")).all()):
        raise ValueError("Frozen 290-row, fully matured pre-August anchor differs")
    coeff = np.linalg.lstsq(np.c_[np.ones(len(prior)), prior.garch11], prior.y, rcond=None)[0]
    saved = read_json(FROZEN_ANCHOR)
    if saved["n"] != 290 or not np.allclose(coeff, saved["coefficients"], rtol=0, atol=1e-12):
        raise ValueError("Pre-August anchor coefficients changed")
    return {**saved, "coefficients": list(saved["coefficients"]),
            "definition": "Frozen pre-August OLS prediction of log mean squared decimal total returns on intercept and GARCH log variance; not a no-news counterfactual.",
            "training_rows_sha256": digest(prior.to_dict("records"))}


def union_returns(raw, compact, expected):
    """Prefer raw precision, fill absent rows from compact, never fill missing returns."""
    keys = ["permno", "date"]
    for frame in (raw, compact):
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate CRSP security-session rows")
    a, b = raw.set_index(keys).sort_index(), compact.set_index(keys).sort_index()
    overlap = a.index.intersection(b.index)
    ar, br = a.loc[overlap, "ret"].to_numpy(float), b.loc[overlap, "ret"].to_numpy(float)
    finite = np.isfinite(ar) & np.isfinite(br)
    if not np.isclose(ar[finite], br[finite], rtol=1e-6, atol=1e-8).all():
        raise ValueError("Raw and compact total returns disagree")
    result = pd.concat([a, b.loc[~b.index.isin(a.index)]]).sort_index().reset_index()
    calendar = pd.DatetimeIndex(sorted(result.date.unique()))
    if not calendar.equals(expected):
        missing = expected.difference(calendar).strftime("%Y-%m-%d").tolist()
        extra = calendar.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Return calendar differs through Dec31: missing={missing}, extra={extra}")
    return result, {"overlap_rows": len(overlap), "finite_overlap_rows": int(finite.sum()),
                    "raw_rows": len(raw), "compact_rows": len(compact),
                    "compact_only_rows": int((~b.index.isin(a.index)).sum()),
                    "missingness_disagreements": int((np.isfinite(ar) != np.isfinite(br)).sum()),
                    "first_session": str(calendar[0].date()), "last_session": str(calendar[-1].date())}


def numerical_case(firm, panel, calendar, date, split, anchor, benchmark):
    pos = calendar.get_loc(date)
    history = panel.iloc[:pos + 1].tail(benchmark.TRAILING_SESSIONS)
    returns = history.ret.to_numpy(float)
    variance, status = benchmark.garch_forecast(returns, False)
    if status != "ok" or not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"GARCH failed for {firm.ticker} {date}: {status}")
    b = float(np.log(variance))
    level = float(anchor["coefficients"][0] + anchor["coefficients"][1] * b)
    state = {"garch_fit_observations": int(np.isfinite(returns).sum()), "garch_fit_status": status}
    for n in (1, 5, 22, 63):
        values = returns[-n:]
        rv = float(np.mean(values ** 2)) if len(values) == n and np.isfinite(values).all() else None
        state[f"mean_squared_ret_{n}"] = rv
        state[f"log_ret_rv{n}"] = float(np.log(max(rv, benchmark.RISK_FLOOR))) if rv is not None else None
    sessions, _ = horizon(calendar, date)
    records = {"dates": [], "total_return_decimal": [], "price_usd": []}
    for row in history.tail(EXPOSED_HISTORY_SESSIONS).itertuples():
        ret, price = float(row.ret), float(row.prc)
        records["dates"].append(row.Index.date().isoformat())
        records["total_return_decimal"].append(ret if np.isfinite(ret) else None)
        records["price_usd"].append(abs(price) if np.isfinite(price) else None)
    state["numerical_aids"] = {"garch_mean_daily_squared_return": variance,
                              "garch_daily_rms_decimal": float(np.sqrt(variance)),
                              "exp_anchor": float(np.exp(level)),
                              "anchor_to_garch_variance_ratio": float(np.exp(level - b)),
                              "variance_multipliers": {str(d): float(np.exp(d)) for d in (-1, -.5, 0, .5, 1)}}
    return {"case_id": f"{firm.ticker}:{date:%Y-%m-%d}", "date": str(date.date()),
            "asof": cutoff(date).isoformat(), "ticker": firm.ticker, "company": firm.company,
            "industry": firm.theme, "permno": int(firm.permno), "split": split,
            "b": b, "anchor": level, "next_sessions": sessions, "state": state,
            "stock_history": {"asof": cutoff(date).isoformat(),
                              "exposed_window_sessions": EXPOSED_HISTORY_SESSIONS,
                              "garch_fit_window_sessions": benchmark.TRAILING_SESSIONS, **records}}


def _valid_cache(out, manifest_name, key):
    path = out / manifest_name
    if not path.exists():
        return False
    manifest = read_json(path)
    return (manifest.get("key") == key
            and all((out / name).is_file() and file_hash(out / name) == sha
                    for name, sha in manifest.get("outputs", {}).items())
            and bool(manifest.get("outputs")))


def prepare_numerical(out=OUT):
    """Return 190 input-only cases; write exact targets to labels_sealed.parquet."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".numerical.lock"), timeout=0):
        p, benchmark = numerical_helpers()
        inputs = [UNIVERSE, ANCHOR_LABELS, FROZEN_ANCHOR, TRADING, p.COMPACT,
                  p.PROCESSED / "crsp_daily.parquet", p.PROCESSED / "crsp_daily_2025.parquet",
                  Path(__file__), ROOT / "code/midcap_tools/prepare.py",
                  ROOT / "code/midcap_standard_2025/benchmarks.py",
                  ROOT / "code/midcap_standard_2025/config.py"]
        sources = [provenance(path) for path in inputs]
        dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "arch", "scipy", "pyarrow")}
        key = digest([sources, dependencies, list(TICKERS), SCHEMA_VERSION])
        if _valid_cache(out, "numerical_manifest.json", key):
            return read_json(out / "cases.json")
        universe = pd.read_csv(UNIVERSE).set_index("ticker").loc[list(TICKERS)].reset_index()
        if universe.permno.nunique() != 10 or len(universe) != 10:
            raise ValueError("The ten-firm universe differs")
        anchor = frozen_anchor()
        start, end = "2023-01-01", "2025-12-31"
        raw = p.read_raw(start, end, universe.permno)
        compact = p.read_frame(p.COMPACT, ["permno", "date", "ret", "prc"], start, end, universe.permno)
        expected = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(TRADING).date))
        expected = expected[(expected >= start) & (expected <= end)]
        if expected.has_duplicates or not expected.is_monotonic_increasing or expected[-1] != pd.Timestamp(end):
            raise ValueError("Stored trading calendar is incomplete through Dec31")
        daily, audit = union_returns(raw, compact, expected)
        cases, labels, histories = [], [], []
        for firm in universe.itertuples():
            panel = daily[daily.permno.eq(firm.permno)].set_index("date").reindex(expected)
            for split, dates in (("train", TRAIN), ("valid", VALID), ("test", TEST)):
                for date in dates:
                    case = numerical_case(firm, panel, expected, date, split, anchor, benchmark)
                    history = panel.loc[:date].tail(benchmark.TRAILING_SESSIONS)
                    histories.append({"case_id": case["case_id"], "asof": case["asof"],
                                      "dates": history.index.strftime("%Y-%m-%d").tolist(),
                                      "total_return_decimal": history.ret.to_numpy(float).tolist()})
                    sessions, available = horizon(expected, date)
                    y = exact_target(panel.loc[pd.to_datetime(sessions), "ret"].to_numpy(float))
                    labels.append({k: case[k] for k in ("case_id", "date", "ticker", "permno", "split", "b", "anchor")}
                                  | {"y": y, "u": y - case["anchor"], "available_at": available,
                                     "matured_at": available, "next_sessions": sessions})
                    cases.append(case)
            print("Numerical", firm.ticker, "19 stable GARCH fits", flush=True)
        cases.sort(key=lambda c: (c["date"], TICKERS.index(c["ticker"])))
        labels = pd.DataFrame(labels).set_index("case_id").loc[[c["case_id"] for c in cases]].reset_index()
        if len(cases) != 190 or Counter(c["split"] for c in cases) != {"train": 100, "valid": 40, "test": 50}:
            raise ValueError("Case schedule differs")
        write_json(out / "cases.json", cases)
        with atomic_path(out / "labels_sealed.parquet") as temp:
            labels.to_parquet(temp, index=False, compression="zstd")
        with atomic_path(out / "garch_history.parquet") as temp:
            pd.DataFrame(histories).to_parquet(temp, index=False, compression="zstd")
        write_json(out / "anchor.json", anchor)
        write_json(out / "numerical_manifest.json", {
            "key": key, "sources": sources, "dependencies": dependencies, "calendar": audit,
            "case_counts": dict(Counter(c["split"] for c in cases)), "provider_calls": 0,
            "target": "log(mean(next_five_decimal_total_returns ** 2)); no target floor",
            "maturity": "09:30 America/New_York on next observed trading day after fifth session",
            "information_cutoff": "16:00 America/New_York, including half-days; next session forecast",
            "exposed_history_sessions": EXPOSED_HISTORY_SESSIONS,
            "history_audit": "garch_history.parquet contains the exact dated as-of fit arrays",
            "return_vintages_certified": False, "raw_precision_preferred": True,
            "outputs": {n: file_hash(out / n) for n in ("cases.json", "labels_sealed.parquet", "anchor.json", "garch_history.parquet")}})
        return cases


def _stamp(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Naive Reuters timestamp")
    return stamp.astimezone(timezone.utc)


def normalize_article(obj, parser, mapper):
    """Parse with existing tag/classifier APIs; do not add alias/neighbor membership."""
    data = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(data, dict):
        return None
    dates = {k: _stamp(data[k]) for k in ("firstCreated", "versionCreated") if data.get(k)}
    if not dates:
        raise ValueError("Missing Reuters publication time")
    recorded = [_stamp(t["timestamp"]) for t in obj.get("timestamps", [])
                if t.get("name") == "recorded" and t.get("timestamp")]
    at = max([*dates.values(), *recorded])
    version = dates.get("versionCreated", dates.get("firstCreated"))
    lookup = mapper.for_date(pd.Timestamp(at).tz_convert("America/New_York").tz_localize(None).normalize())
    subjects = data.get("subjects") or []
    topics = sorted(set(re.findall(r"N2:[A-Za-z0-9_]+", "|".join(subjects) if isinstance(subjects, list) else str(subjects))))
    ids = sorted({int(lookup[t]) for t in parser.parse_tags(subjects) if t in lookup})
    headline, body = str(data.get("headline") or ""), str(data.get("body") or "")
    kind, family, _ = parser.common_category(headline, body, "")
    if "N2:US" in topics and MACRO_TAGS.intersection(topics):
        kind = "macro"
    elif "N2:US" in topics and FINANCE_TAGS.intersection(topics):
        kind = "finance"
    story = str(data.get("altId") or data.get("id") or obj.get("guid") or "")
    if not story:
        raise ValueError("Missing Reuters story identifier")
    status = data.get("pubStatus") or "stat:usable"
    provider = data.get("provider") or "NS:RTRS"
    usable = (status == "stat:usable" and data.get("language") in (None, "en") and provider == "NS:RTRS")
    return {"story_id": story, "available_at": at.isoformat(timespec="microseconds"),
            "version_created": version.isoformat(timespec="microseconds"),
            "first_created": dates.get("firstCreated", version).isoformat(timespec="microseconds"),
            "take_sequence": int(data.get("takeSequence") or 0), "usable": usable,
            "pub_status": status, "provider": provider, "ids": ids, "topics": topics,
            "kind": kind or "other", "family": family or parser.substantive_category(headline, body),
            "headline": headline}


def build_month_index(out, month, mapper, parser, dependencies):
    """Stream the entire raw month, indexing all versions including tombstones."""
    source = RAW_NEWS / f"STORY.RTRS.{month}.REC.JSON.txt"
    path = Path(out) / "reuters_index" / f"{month}.sqlite"
    manifest_path = path.with_suffix(".json")
    before = file_stamp(source)
    index_policy = digest([inspect.getsource(normalize_article), inspect.getsource(_stamp), SCHEMA_VERSION])
    key = digest([before, dependencies, index_policy])
    if path.exists() and manifest_path.exists():
        saved = read_json(manifest_path)
        if (saved.get("key") == key and saved.get("complete") is True
                and saved["index_sha256"] == file_hash(path)):
            return saved
    count = Counter()
    source_hash = hashlib.sha256()
    lo, hi = None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temp:
        connection = sqlite3.connect(temp)
        try:
            connection.execute("CREATE TABLE versions (story TEXT, available TEXT, version TEXT, take INTEGER, source_line INTEGER PRIMARY KEY, metadata TEXT)")
            batch = []
            with source.open("rb") as stream:
                offset = 0
                for line_number, raw in enumerate(stream, 1):
                    source_hash.update(raw)
                    obj = parser.parse_line(raw)
                    try:
                        row = normalize_article(obj, parser, mapper)
                    except (ValueError, TypeError, OverflowError) as exc:
                        raise ValueError(f"Unindexable Reuters record {month}:{line_number}") from exc
                    count["raw_lines"] += 1
                    if row is not None:
                        row.update({"source_month": month, "source_line": line_number,
                                    "byte_offset": offset, "byte_length": len(raw),
                                    "raw_sha256": hashlib.sha256(raw).hexdigest()})
                        row["source_id"] = f"RTRS:{month}:{line_number}:{row['raw_sha256'][:12]}"
                        batch.append((row["story_id"], row["available_at"], row["version_created"],
                                      row["take_sequence"], line_number, serialized(row)))
                        count["versions"] += 1
                        count["unusable_versions"] += int(not row["usable"])
                        lo = min(lo, row["available_at"]) if lo else row["available_at"]
                        hi = max(hi, row["available_at"]) if hi else row["available_at"]
                    elif b'"guid"' in raw:
                        raise ValueError(f"Unparsed Reuters item {month}:{line_number}")
                    offset += len(raw)
                    if len(batch) >= 5000:
                        connection.executemany("INSERT INTO versions VALUES (?,?,?,?,?,?)", batch)
                        batch.clear()
                connection.executemany("INSERT INTO versions VALUES (?,?,?,?,?,?)", batch)
            connection.execute("CREATE INDEX availability_idx ON versions(available)")
            connection.execute("CREATE INDEX story_idx ON versions(story, version)")
            connection.commit()
        finally:
            connection.close()
        if before != file_stamp(source):
            raise ValueError("Raw Reuters source changed during indexing")
        begin = pd.Timestamp(month + "-01", tz="UTC")
        end = begin + pd.offsets.MonthBegin(1)
        if not lo or pd.Timestamp(lo) > begin + pd.Timedelta(days=1) or pd.Timestamp(hi) < end - pd.Timedelta(days=1):
            raise ValueError("Raw Reuters month lacks full timestamp coverage: " + month)
    saved = {"key": key, "complete": True, "index_policy_sha256": index_policy,
             "dependencies": dependencies, "source": {**before, "sha256": source_hash.hexdigest()},
             "index": str(path.resolve()), "index_sha256": file_hash(path), "counts": dict(count),
             "first_available_at": lo, "last_available_at": hi}
    write_json(manifest_path, saved)
    print("Reuters index", month, dict(count), flush=True)
    return saved


def latest_versions(indexes, asof):
    """Resolve across months before filtering topics, entity tags or usability."""
    latest = {}
    cutoff_text = _stamp(asof).isoformat(timespec="microseconds")
    lower = (_stamp(asof) - pd.Timedelta(days=POLICY["lookback_days"])).isoformat(timespec="microseconds")
    inspected = 0
    for index in indexes:
        with closing(sqlite3.connect(Path(index["index"]).as_uri() + "?mode=ro", uri=True)) as db:
            for (metadata,) in db.execute("SELECT metadata FROM versions WHERE available >= ? AND available < ?", (lower, cutoff_text)):
                row = json.loads(metadata)
                key = (row["version_created"], row["take_sequence"], row["available_at"], row["source_id"])
                old = latest.get(row["story_id"])
                if old is None or key > old[0]:
                    latest[row["story_id"]] = (key, row)
                inspected += 1
    return [item[1] for item in latest.values()], inspected


def read_article(row, parser):
    path = RAW_NEWS / f"STORY.RTRS.{row['source_month']}.REC.JSON.txt"
    with path.open("rb") as stream:
        stream.seek(row["byte_offset"])
        raw = stream.read(row["byte_length"])
    if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
        raise ValueError("Raw Reuters source verification failed")
    obj = parser.parse_line(raw)
    if not obj or str(obj["data"].get("headline") or "") != row["headline"]:
        raise ValueError("Indexed headline differs from raw Reuters")
    return str(obj["data"].get("body") or "")


def _firm_pattern(ticker):
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(s) for s in ALIASES[ticker]) + r")(?![A-Za-z0-9])", re.I)


def _future_pattern(sessions):
    patterns = []
    for session in sessions:
        d = pd.Timestamp(session)
        patterns.extend((re.escape(session), rf"\b{d:%b}(?:{d.strftime('%B')[3:]})?\.?\s+0?{d.day}\b",
                         rf"\b0?{d.day}\s+{d:%b}(?:{d.strftime('%B')[3:]})?\b"))
    return re.compile("|".join(patterns), re.I)


def source_channel(row, case, parser=None):
    if not row["usable"]:
        return None
    if case["permno"] in row["ids"]:
        return "own"
    if row["kind"] not in {"macro", "finance"}:
        return None
    if row["ids"]:
        if parser is None:
            _, parser = news_helpers()
        kind, _, _ = parser.common_category(row["headline"], "", "")
        if kind not in {"macro", "finance"}:
            return None
        return kind
    return row["kind"]


def relevance_profile(row, case, channel):
    """Label-free primary-subject and recency checks, never outcome-based selection."""
    title = row["headline"]
    if NON_SUBSTANTIVE.search(title):
        return None
    primary_firm = bool(_firm_pattern(case["ticker"]).search(title))
    if channel == "own":
        focus = 3 if primary_firm and FIRM_MATERIAL.search(title) else 2 if primary_firm else 0
    else:
        if not US_MACRO.search(title) or not ECONOMIC.search(title):
            return None
        # Other companies' native corporate stories are not general macro evidence.
        if row["ids"] and not re.match(r"(?:UPDATE \d+-|WRAPUP \d+-)?(?:US STOCKS|U\.S\. STOCKS|GLOBAL MARKETS|GLOBAL MARKETS-)", title, re.I):
            return None
        focus = 2
    age = (_stamp(case["asof"])-_stamp(row["available_at"])).total_seconds()/86400
    return {"focus": focus, "recent_8_days": age <= 8, "age_days": age,
            "financial_headline_terms": len(set(m.group().lower() for m in FIRM_MATERIAL.finditer(title)))}


def passage_spans(body, firm_pattern, future_pattern, limit=1050):
    if not body:
        return []
    candidates = [(0, min(len(body), 350))]
    for pattern in (firm_pattern, future_pattern, EVENT_RE):
        matches = list(pattern.finditer(body))[:30]
        matches.sort(key=lambda m: (-len(future_pattern.findall(body[max(0, m.start()-150):m.end()+350])), m.start()))
        for match in matches[:2]:
            candidates.append((max(0, match.start() - 140), min(len(body), match.end() + 400)))
    selected, remaining = [], limit
    for start, end in candidates:
        for a, b in sorted(selected):
            if a <= start < b:
                start = b
            if start < a < end:
                end = a
        end = min(end, start + remaining)
        if end > start:
            selected.append((start, end))
            remaining -= end - start
        if remaining == 0:
            break
    return [{"field": "body", "start": a, "end": b, "text": body[a:b]} for a, b in sorted(selected)]


def select_packet(case, rows, inspected, parser, index_by_month, body_cache=None, feature_cache=None):
    at = _stamp(case["asof"])
    if pd.Timestamp(at) != cutoff(case["date"]):
        raise ValueError("Evidence case must use the 16 ET cutoff")
    firm, future = _firm_pattern(case["ticker"]), _future_pattern(case["next_sessions"])
    candidates = []
    counts = Counter()
    for row in rows:
        if _stamp(row["available_at"]) >= at:
            raise ValueError("Future Reuters version reached selection")
        channel = source_channel(row, case, parser)
        if channel is None:
            continue
        relevance = relevance_profile(row, case, channel)
        if relevance is None:
            continue
        if body_cache is not None and row["source_id"] in body_cache:
            body = body_cache[row["source_id"]]
        else:
            body = read_article(row, parser)
            if body_cache is not None:
                body_cache[row["source_id"]] = body
        feature_key = (row["source_id"], tuple(case["next_sessions"]))
        features = feature_cache.get(feature_key) if feature_cache is not None else None
        if features is None:
            text = row["headline"] + "\n" + body
            features = {"next_session_date_mentions": len(future.findall(text)),
                        "event_terms": len(set(m.group().lower() for m in EVENT_RE.finditer(text)))}
            if feature_cache is not None:
                feature_cache[feature_key] = features
        features = {**features, **relevance, "firm_headline_mentions": len(firm.findall(row["headline"]))}
        candidates.append((row, body, channel, features))
        counts[channel] += 1
    packet = {"case_id": case["case_id"], "asof": case["asof"], "policy_sha256": digest(POLICY),
              "sources": [], "passages": [], "coverage": {"versions_inspected": inspected, "latest_stories": len(rows),
                  "own_candidates": counts["own"], "macro_candidates": counts["macro"],
                  "finance_candidates": counts["finance"], "selected_sources": 0,
                  "selected_own": 0, "selected_common": 0, "snippet_chars": 0},
              "tool_traces": [{"tool": "latest_versions", "lookback_days": POLICY["lookback_days"],
                               "strict_before": case["asof"]},
                              {"tool": "lexical_select", "own_reserved_chars": 7800,
                               "packet_limit_chars": MAX_PACKET_CHARS, "family_diversification": 1}]}
    seen_families, used = set(), set()
    # Reserve common evidence space while taking distinct own-company event families.
    for wanted, budget in (("own", 7800), ("common", MAX_PACKET_CHARS), ("own", MAX_PACKET_CHARS)):
        pool = [item for item in candidates if (item[2] == "own") == (wanted == "own")]
        rerank = True
        packet_size = len(serialized(packet))
        while pool:
            if rerank:
                pool.sort(key=lambda item: (not item[3]["recent_8_days"], -item[3]["focus"],
                          item[0]["family"] in seen_families, -_stamp(item[0]["available_at"]).timestamp(),
                          -item[3]["financial_headline_terms"], item[0]["source_id"]),
                          reverse=True)
                rerank = False
            row, body, channel, features = pool.pop()
            if row["story_id"] in used:
                continue
            source = {k: row[k] for k in ("source_id", "story_id", "headline", "available_at",
                      "version_created", "first_created", "pub_status", "provider", "family")}
            source.update({"channel": channel, "tagged_permnos": row["ids"], "selection_features": features,
                           "citation": {k: row[k] for k in ("source_month", "source_line", "byte_offset", "byte_length", "raw_sha256")}})
            source["citation"]["archive_sha256"] = index_by_month[row["source_month"]]["source"]["sha256"]
            lead_text = body[:350] if body else row["headline"]
            lead = {"field": "body" if body else "headline", "start": 0, "end": len(lead_text),
                    "text": lead_text, "id": row["source_id"], "time": row["available_at"],
                    "headline": row["headline"], "scope": channel}
            # Every excerpt contains the lead. Reject impossible fits before scanning the body.
            minimum_size = (packet_size + len(serialized(source)) + len(serialized(lead))
                            + bool(packet["sources"]) + bool(packet["passages"]))
            if minimum_size > budget - 200:
                continue
            spans = passage_spans(body, firm, future)
            if not spans:
                spans = [{k: lead[k] for k in ("field", "start", "end", "text")}]
            passages = [{**span, "id": row["source_id"], "time": row["available_at"],
                         "headline": row["headline"], "scope": channel} for span in spans]
            packet["sources"].append(source)
            packet["passages"].extend(passages)
            if len(serialized(packet)) > budget - 200:
                packet["sources"].pop()
                del packet["passages"][-len(passages):]
                continue
            seen_families.add(row["family"])
            used.add(row["story_id"])
            packet_size = len(serialized(packet))
            rerank = True
    coverage = packet["coverage"]
    coverage.update({"selected_sources": len(packet["sources"]),
                     "selected_own": sum(s["channel"] == "own" for s in packet["sources"]),
                     "selected_common": sum(s["channel"] != "own" for s in packet["sources"]),
                     "snippet_chars": sum(len(p["text"]) for p in packet["passages"])})
    if len(serialized(packet)) > MAX_PACKET_CHARS:
        raise ValueError("Evidence packet exceeds frozen character budget")
    return packet


def prepare_evidence(out, cases):
    """Return source-only packets; no provider calls, generated judgments or labels."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate input cases")
    with FileLock(str(out / ".evidence.lock"), timeout=0):
        p, parser = news_helpers()
        dependencies = [provenance(path) for path in (Path(__file__), p.NAMES, p.RIC,
                        ROOT / "code/midcap_tools/prepare.py", ROOT / "code/news_percentile/build.py",
                        ROOT / "code/news_percentile/common.py")]
        mapper = p.NewsMapper(pd.read_parquet(p.NAMES), pd.read_csv(p.RIC))
        index_dependencies = [entry for entry in dependencies if Path(entry["path"]) != Path(__file__).resolve()]
        indexes = [build_month_index(out, month, mapper, parser, index_dependencies) for month in MONTHS]
        index_by_month = dict(zip(MONTHS, indexes))
        # Whitelist inputs so sealed labels or downstream calendar additions cannot enter packets.
        bindings = [{k: c[k] for k in ("case_id", "date", "asof", "ticker", "permno", "next_sessions")} for c in cases]
        key = digest([indexes, bindings, POLICY, dependencies])
        if _valid_cache(out, "evidence_manifest.json", key):
            return read_json(out / "evidence.json")
        packets = {}
        for date in sorted({c["date"] for c in bindings}):
            selected = [c for c in bindings if c["date"] == date]
            rows, inspected = latest_versions(indexes, cutoff(date).isoformat())
            body_cache, feature_cache = {}, {}
            for case in selected:
                packets[case["case_id"]] = select_packet(case, rows, inspected, parser, index_by_month,
                                                       body_cache, feature_cache)
            print("Evidence", date, len(selected), "cases", len(rows), "latest stories", flush=True)
        write_json(out / "evidence.json", packets)
        write_json(out / "evidence_manifest.json", {"key": key, "policy": POLICY,
            "policy_sha256": digest(POLICY), "dependencies": dependencies, "archives": indexes,
            "case_count": len(packets), "provider_calls": 0,
            "existing_archive_reuse": "Older filtered archives lack full-month recorded-time and raw-offset certification; complete raw indexes used.",
            "mapping_vintages_certified": False,
            "outputs": {"evidence.json": file_hash(out / "evidence.json")}})
        return packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stage", choices=("numerical", "evidence", "all"), default="all")
    args = parser.parse_args()
    cases = prepare_numerical(args.out) if args.stage != "evidence" else read_json(args.out / "cases.json")
    if args.stage != "numerical":
        prepare_evidence(args.out, cases)


if __name__ == "__main__":
    main()
