"""CPU-only, point-in-time Reuters earnings diary calendar.

Public APIs:
    build_calendar(output_dir, cases) -> {case_id: feature} (writes audit artifacts)
    feature_per_case(case, receipts) -> dict (pure; receipts from the build)
    scan_plan(cases, raw_dir=RAW_NEWS) -> list[dict] (metadata only)

Cases supply case_id, ticker, permno, date, timezone-aware asof and next_sessions
(the next five XNYS dates). Subsets of the locked 190-case design are supported.
The information cutoff is the supplied asof (16:00 ET even on half-days).
Actual exchange closes determine event-to-return session mapping only.
Status concerns the first close-to-close return exposed to a scheduled release:
AMC maps to the following session; BMO maps to that day's session. Unknown times
remain ambiguous at window boundaries. A past planned date is not proof of an
actual release. Missing retrieval never becomes a negative event indicator.

Builds scan January through the last cutoff month once each, including months
before the first origin so earlier announcements are not silently dropped. They
use the standard preparation code's bounded-repair Reuters JSON parser, with a
hard record-size bound. Compact raw locators are indexed in bounded memory so
even an out-of-order, bodyless withdrawal can supersede a diary. After the single
streaming pass, bounded direct reads materialize matching histories as receipts.
Existing filtered archives are not reused.

Call scan_plan and coordinate the multi-GB scan before calling build_calendar.
Importing this module and running its tests do not scan production Reuters.
There are no model, network, return-data or outcome-label calls. Use the repo's
.venv_dspy interpreter (exchange_calendars and standard-helper dependencies).

Limits: explicit table rows only; no prose/name-only inference, inferred fiscal
dates, or prior-year news. Reuters schedules are evidence, not issuer guarantees.
Receipts preserve the full decoded diary object and exact raw-line hash/location;
non-diary replacements preserve metadata and a raw locator. Malformed records
and missing months are reported as coverage gaps. No claim of exhaustive event
coverage is made, even when all source files were read successfully.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import hashlib
import ast
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results/news_event_risk_2025/calendar"
RAW_NEWS = Path(r"C:\Users\Boxuan\Documents\GitHub\ReutersNewstTraining\Reuters")
TICKERS = ("NVDA", "MSFT", "CAT", "XPO", "BBY", "AMD", "CFLT", "APLD", "UPS", "WMT")
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_INDEX_BYTES = 2 * 1024 * 1024 * 1024
SCHEMA_VERSION = 1
ARTIFACTS = ("receipts.jsonl", "features.jsonl", "manifest.json")
ORIGINS = {
    name: tuple((date.fromisoformat(start) + timedelta(weeks=i)).isoformat()
                for i in range(count))
    for name, start, count in (("training", "2025-08-01", 10),
                               ("validation", "2025-10-17", 4),
                               ("evaluation", "2025-11-21", 5))
}
_SPLITS = {day: split for split, days in ORIGINS.items() for day in days}
_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}-[A-Za-z]{3}-\d{4})"
_ZONE = r"(?:EDT|EST|ET|UTC|GMT)"
_ROW = re.compile(
    rf"^\s*\*?\s*(?P<date>{_DATE})\s+"
    rf"(?:(?P<time>\d{{1,2}}:\d{{2}}(?:\s*{_ZONE})?|AMC|BMO|DBH|NTS|TBA|TBD|--)\s+)?"
    r"(?P<ric><?[A-Z][A-Z0-9]*(?:\.[A-Z]+)?>?)\s+(?P<event>.+?)\s*$", re.S)
_ROW_START = re.compile(rf"^\s*\*?\s*{_DATE}\s+")
_DIARY = re.compile(r"\bDIARY\b", re.I)
_RELEASE = re.compile(r"\b(?:earnings\s+release|(?:financial\s+|quarterly\s+|full.year\s+)results\s+release)\b", re.I)
_PERIOD = re.compile(r"\b(Q[1-4]|H[12]|FY|Full Year)\s+(20\d{2})\b", re.I)
_CANCEL = re.compile(r"\b(?:cancelled|canceled|withdrawn|postponed|deleted)\b", re.I)
_ZONE_NOTE = re.compile(
    rf"(?:all\s+)?times[^\n]{{0,100}}?\b(?:in\s+)?(?:U\.?S\.?\s+)?({_ZONE})\b", re.I)
_CANCELED_STATUSES = {"canceled", "cancelled", "withdrawn"}


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)


def _stamp(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return stamp.astimezone(UTC)


def _iso(stamp):
    return stamp.astimezone(UTC).isoformat(timespec="microseconds")


@lru_cache(maxsize=1)
def _sessions():
    import exchange_calendars as xcals

    cal = xcals.get_calendar("XNYS", start="2024-01-01", end="2026-12-31")
    # Iterating schedule rows avoids old calendar methods' pandas unit assumptions.
    return tuple((s.date().isoformat(), r["open"].to_pydatetime(),
                  r["close"].to_pydatetime()) for s, r in cal.schedule.iterrows())


def _case(case):
    required = {"case_id", "ticker", "permno", "asof", "date", "next_sessions"}
    if not required <= case.keys():
        raise ValueError("Case requires " + ", ".join(sorted(required)))
    if case["ticker"] not in TICKERS or case["date"] not in _SPLITS:
        raise ValueError("Case is outside the locked firms/origins")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("Nonempty case_id required")
    if isinstance(case["permno"], bool) or not isinstance(case["permno"], int) or case["permno"] <= 0:
        raise ValueError("Positive integer permno required")
    stamp = _stamp(case["asof"])
    if stamp.astimezone(NY).date().isoformat() != case["date"]:
        raise ValueError("Case date and New York asof date disagree")
    sessions = _sessions()
    days = [s[0] for s in sessions]
    pos = bisect_left(days, case["date"])
    if pos == len(days) or days[pos] != case["date"]:
        raise ValueError("Origin is not an XNYS session")
    future = days[pos + 1:pos + 6]
    if list(case["next_sessions"]) != future:
        raise ValueError("next_sessions must be the next five XNYS sessions")
    return {key: case[key] for key in required} | {
        "next_sessions": future, "split": _SPLITS[case["date"]],
        "cutoff": _iso(stamp),
    }


def _cases(cases):
    rows = [_case(c) for c in cases]
    if not rows:
        raise ValueError("At least one case required")
    if len({c["case_id"] for c in rows}) != len(rows):
        raise ValueError("Duplicate case_id")
    if len({(c["ticker"], c["date"]) for c in rows}) != len(rows):
        raise ValueError("Duplicate ticker/origin")
    mapping = defaultdict(set)
    for c in rows:
        mapping[c["ticker"]].add(c["permno"])
    if any(len(ids) != 1 for ids in mapping.values()):
        raise ValueError("Inconsistent frozen ticker/permno mapping")
    ids = [next(iter(values)) for values in mapping.values()]
    if len(ids) != len(set(ids)):
        raise ValueError("A permno maps to more than one ticker")
    return sorted(rows, key=lambda c: (c["date"], c["ticker"], c["case_id"]))


def scan_plan(cases, raw_dir=RAW_NEWS):
    """Return necessary raw filenames and sizes without opening their contents."""
    rows = _cases(cases)
    last_month = max(_stamp(c["cutoff"]).month for c in rows)
    paths = [Path(raw_dir) / f"STORY.RTRS.2025-{m:02d}.REC.JSON.txt"
             for m in range(1, last_month + 1)]
    return [{"month": f"2025-{i:02d}", "path": str(p.resolve()),
             "exists": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else None}
            for i, p in enumerate(paths, 1)]


@lru_cache(maxsize=1)
def _raw_parser():
    # Reuse the exact helper function without importing its unrelated sklearn pipeline.
    path = ROOT / "code/news_percentile/build.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_line")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_line"]


def parse_diary_rows(body):
    """Extract exact dated RIC cells, with verbatim spans; never infer from names."""
    zone_matches = list(_ZONE_NOTE.finditer(body))
    zones = {m.group(1).upper() for m in zone_matches}
    default_zone = next(iter(zones)) if len(zones) == 1 else None
    rows = []
    for start, end in _table_spans(body):
        quote = body[start:end]
        match = _ROW.fullmatch(quote)
        if match:
            ric = match["ric"].strip("<>")
            ticker, _, suffix = ric.partition(".")
            if ticker in TICKERS and suffix in {"", "O", "OQ", "N"} and _RELEASE.search(match["event"]):
                try:
                    day = (date.fromisoformat(match["date"]) if match["date"][4:5] == "-"
                           else datetime.strptime(match["date"], "%d-%b-%Y").date())
                except ValueError:
                    continue
                period = _PERIOD.search(match["event"])
                period = " ".join(period.groups()).upper() if period else None
                timing = match["time"] or "unknown"
                explicit_zone = re.search(rf"\b({_ZONE})$", timing)
                zone = explicit_zone[1] if explicit_zone else default_zone
                rows.append({"ticker": ticker, "ric": ric, "event_date": day.isoformat(),
                             "period": period, "time_text": timing, "timezone": zone,
                             "timezone_note": [m.group(0) for m in zone_matches],
                             "action": "cancelled" if _CANCEL.search(match["event"]) else "scheduled",
                             "quote": quote, "span": [start, end]})
    return rows


def _table_spans(body):
    """Join only indented continuations, stopping before every next dated row."""
    offset, start, end, lines = 0, None, 0, 0
    for line in body.splitlines(keepends=True):
        dated = bool(_ROW_START.match(line))
        continuation = bool(re.match(r"^ {16,}\S", line))
        if dated or not continuation:
            if start is not None and lines <= 8 and end - start <= 2048:
                yield start, end
            start, lines = (offset, 0) if dated else (None, 0)
        if start is not None:
            lines += 1
            end = offset + len(line.rstrip("\r\n"))
        offset += len(line)
    if start is not None and lines <= 8 and end - start <= 2048:
        yield start, end


def _container_header(raw, month):
    """Recognize the feed envelope structurally; arbitrary bad JSON is a gap."""
    value = raw.strip()
    if not value.endswith(b"["):
        return False
    try:
        header = json.loads(value + b"]}")
    except (ValueError, TypeError):
        return False
    return (header.get("RIC") == "MRN_STORY" and header.get("Items") == []
            and str(header.get("Start", "")).startswith(month)
            and str(header.get("End", "")).startswith(month))


def _version(obj, locator):
    data = obj["data"]
    alt = data.get("altId") or data.get("id") or obj.get("guid")
    if not alt:
        return None
    times = [data[k] for k in ("versionCreated", "firstCreated", "ts") if data.get(k)]
    times.extend(t["timestamp"] for t in obj.get("timestamps", [])
                 if t.get("name") == "recorded" and t.get("timestamp"))
    try:
        available = _iso(max(_stamp(t) for t in times))
        version_created = _iso(_stamp(data.get("versionCreated") or data.get("firstCreated")))
    except (ValueError, TypeError, OverflowError):
        available, version_created = None, None
    headline, body = str(data.get("headline") or ""), str(data.get("body") or "")
    diary = (bool(_DIARY.search(headline)) and data.get("language") in (None, "en")
             and data.get("provider", "NS:RTRS") == "NS:RTRS")
    status = str(data.get("pubStatus") or "stat:usable").lower().split(":")[-1]
    try:
        sequence = int(data.get("takeSequence") or data.get("version", obj.get("version", 0)))
    except (ValueError, TypeError):
        sequence = 0
    metadata = {k: data[k] for k in ("altId", "id", "versionCreated", "firstCreated", "ts",
                                    "pubStatus", "version", "takeSequence", "language", "headline") if k in data}
    return {"receipt_id": locator["raw_sha256"], "story_id": str(alt), "available_at": available,
            "version_created": version_created, "recorded_timestamps": obj.get("timestamps", []),
            "version_sequence": sequence, "cancelled": status in _CANCELED_STATUSES,
            "usable": status == "usable",
            "is_diary": diary, "headline": headline, "source": locator,
            "raw_object": obj if diary else None, "metadata": metadata,
            "rows": parse_diary_rows(body) if diary else []}


def _event_key(row):
    return row["ticker"], row["period"] or ("date:" + row["event_date"])


def _observations(receipts, ticker, cutoff):
    histories = defaultdict(list)
    for receipt in receipts:
        if receipt["available_at"] is None or _stamp(receipt["available_at"]) <= cutoff:
            histories[receipt["story_id"]].append(receipt)
    observations, audit, uncertain = [], [], False
    for history in histories.values():
        touched = [r for v in history for r in v["rows"] if r["ticker"] == ticker]
        if not touched:
            continue
        keys = {_event_key(r) for r in touched}
        if any(v["available_at"] is None for v in history):
            uncertain = True
            audit.extend({"receipt_id": v["receipt_id"], "state": "undated_version"}
                         for v in history if v["available_at"] is None)
            continue
        rank = max((v["version_created"], v["version_sequence"]) for v in history)
        latest = [v for v in history if (v["version_created"], v["version_sequence"]) == rank]
        # At equal version rank, contradictory contents are unresolved, never hash-ordered.
        signatures = {_json([v["usable"], v["is_diary"], v["rows"]]) for v in latest}
        ambiguous = len(signatures) > 1
        current = min(latest, key=lambda v: (v["available_at"], v["receipt_id"]))
        audit.extend({"receipt_id": v["receipt_id"], "state": "ambiguous_version" if ambiguous
                      else "withdrawn" if v["cancelled"] else "latest"} for v in latest)
        active = [r for r in current["rows"] if r["ticker"] == ticker]
        if ambiguous or not current["usable"] or not current["is_diary"]:
            active = []
        active_keys = {_event_key(r) for r in active}
        for row in active:
            observations.append({**row, "receipt_id": current["receipt_id"],
                                 "article_id": current["story_id"], "source_text": row["quote"],
                                 "version_created": current["version_created"],
                                 "available_at": current["available_at"]})
        for key in keys - active_keys:
            old = next(r for r in touched if _event_key(r) == key)
            observations.append({**old, "action": "withdrawn", "receipt_id": current["receipt_id"],
                                 "article_id": current["story_id"], "source_text": current["headline"],
                                 "version_created": current["version_created"],
                                 "available_at": current["available_at"]})
    return observations, audit, uncertain


def _timing(row):
    day = date.fromisoformat(row["event_date"])
    sessions = _sessions()
    days = [s[0] for s in sessions]
    pos = bisect_left(days, day.isoformat())
    if pos >= len(days) - 1 or day.isoformat() < days[0]:
        return None
    is_session = days[pos] == day.isoformat()
    opening, closing = sessions[pos][1:]
    start = datetime.combine(day, time.min, NY).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time.min, NY).astimezone(UTC) - timedelta(microseconds=1)
    timing = row["time_text"]
    if timing == "AMC" and is_session:
        return closing + timedelta(microseconds=1), end
    if timing == "BMO" and is_session:
        return start, opening - timedelta(microseconds=1)
    clock = re.fullmatch(rf"(\d{{1,2}}):(\d{{2}})(?:\s*{_ZONE})?", timing)
    if clock and row["timezone"]:
        zones = {"ET": NY, "EDT": timezone(timedelta(hours=-4)),
                 "EST": timezone(timedelta(hours=-5)), "UTC": UTC, "GMT": UTC}
        try:
            exact = datetime.combine(day, time(int(clock[1]), int(clock[2])),
                                     zones[row["timezone"]]).astimezone(UTC)
            return exact, exact
        except (ValueError, KeyError):
            return None
    # DBH, blank and unzoned clock times cannot safely locate a close boundary.
    return start, end


def _classify(row, case):
    interval = _timing(row)
    if interval is None:
        return {"date_status": "unknown", "reason": "unsupported_timing", "reaction_sessions": []}
    low, high = interval
    origin_close = next(s[2] for s in _sessions() if s[0] == case["date"])
    if high <= origin_close:
        return {"date_status": "unknown", "reason": "planned_time_passed", "reaction_sessions": []}
    closes = [s[2] for s in _sessions()]
    left, right = bisect_left(closes, low), bisect_left(closes, high)
    possible = [s[0] for s in _sessions()[left:right + 1]]
    inside = [s in case["next_sessions"] for s in possible]
    if low <= origin_close or (any(inside) and not all(inside)):
        status, reason = "unknown", "timing_straddles_boundary"
    elif inside and all(inside):
        status, reason = "confirmed_in_window", "source_backed_release"
    else:
        status, reason = "known_outside_window", "source_backed_release"
    return {"date_status": status, "reason": reason, "reaction_sessions": possible,
            "time_lower_utc": _iso(low), "time_upper_utc": _iso(high)}


def feature_per_case(case, receipts):
    """Pure per-case features from decoded receipts; null means unknown, never zero.

    `earnings_in_window` is True for a supported in-window release and None
    otherwise. `known_outside_window` describes observed schedules only; it is
    deliberately not evidence that there is no other in-window release.
    """
    case = _case(case)
    observations, audit, uncertain = _observations(receipts, case["ticker"], _stamp(case["cutoff"]))
    grouped = defaultdict(list)
    for row in observations:
        grouped[_event_key(row)].append(row)
    events = []
    for rows in grouped.values():
        newest = max(r["version_created"] for r in rows)
        current = [r for r in rows if r["version_created"] == newest]
        signatures = {(r["event_date"], r["time_text"], r["timezone"], r["action"]) for r in current}
        row = current[0]
        event = {k: row[k] for k in ("ticker", "period", "event_date", "time_text", "timezone", "action")}
        event.update(available_at=row["available_at"], published_at=row["available_at"],
                     article_id=row["article_id"], source_text=row["source_text"],
                     receipt_ids=sorted({r["receipt_id"] for r in current}))
        if len(signatures) > 1 or row["action"] != "scheduled":
            event.update(date_status="unknown", reason="conflicting_schedules" if len(signatures) > 1
                         else "schedule_withdrawn", reaction_sessions=[])
        else:
            event.update(_classify(row, case))
        event["in_window"] = True if event["date_status"] == "confirmed_in_window" else None
        indices = [case["next_sessions"].index(s) + 1 for s in event["reaction_sessions"]
                   if s in case["next_sessions"]]
        event["session_index"] = indices[0] if len(indices) == 1 and event["in_window"] else None
        events.append(event)
    events.sort(key=lambda e: (e["event_date"], e["period"] or ""))
    live = [e for e in events if e["reason"] != "planned_time_passed"]
    if uncertain:
        status, reason = "unknown", "undated_version"
        for event in events:
            event.update(date_status="unknown", in_window=None, session_index=None,
                         reason="undated_version")
    elif any(e["date_status"] == "confirmed_in_window" for e in live):
        status, reason = "confirmed_in_window", "source_backed_release"
    elif any(e["date_status"] == "unknown" for e in live):
        status, reason = "unknown", "unresolved_schedule"
    elif live:
        status, reason = "known_outside_window", "observed_schedules_outside_window"
    else:
        status, reason = "unknown", "no_supported_future_release"
    return {**case, "status": status, "date_status": status, "reason": reason,
            "earnings_in_window": True if status == "confirmed_in_window" else None,
            "confirmed_release_count": sum(e["date_status"] == "confirmed_in_window" for e in live)
            if status == "confirmed_in_window" else None,
            "events": events, "source_versions": audit,
            "coverage_note": "Observed Reuters schedules only; absence is not evidence of no event."}


def _scan_month(db, item, upper):
    path = Path(item["path"])
    counts = Counter()
    if not item["exists"]:
        return {**item, "status": "missing", "counts": {}}
    parser, digest = _raw_parser(), hashlib.sha256()
    before = path.stat()
    import time as clock
    started = clock.monotonic()
    print(f"Calendar scanning {item['month']}: {before.st_size / 1e9:.2f} GB", flush=True)
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline(MAX_RECORD_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError(f"Reuters record exceeds {MAX_RECORD_BYTES} bytes: {path}:{counts['lines'] + 1}")
            digest.update(raw)
            counts["lines"] += 1
            obj = parser(raw)
            if not isinstance(obj, dict) or not isinstance(obj.get("data"), dict):
                if counts["lines"] == 1 and _container_header(raw, item["month"]):
                    counts["container_headers"] += 1
                else:
                    counts["unparsed_lines"] += 1
                continue
            locator = {"path": str(path), "month": item["month"], "line": counts["lines"],
                       "byte_offset": offset, "byte_length": len(raw),
                       "raw_sha256": hashlib.sha256(raw).hexdigest()}
            receipt = _version(obj, locator)
            if receipt is None:
                counts["missing_story_id"] += 1
                continue
            if receipt["available_at"] is None:
                counts["invalid_timestamp"] += 1
            elif receipt["available_at"] > upper:
                counts["after_last_cutoff"] += 1
                continue
            counts["indexed_versions"] += 1
            db.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (receipt["story_id"], receipt["available_at"], int(receipt["is_diary"]),
                        item["month"], counts["lines"], offset, len(raw), bytes.fromhex(locator["raw_sha256"])))
            if counts["indexed_versions"] % 5000 == 0:
                db.commit()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Reuters source changed during scan: " + str(path))
    db.commit()
    elapsed = clock.monotonic() - started
    print(f"Calendar finished {item['month']}: {counts['lines']} lines in {elapsed:.1f}s", flush=True)
    return {**item, "status": "scanned", "sha256": digest.hexdigest(),
            "mtime_ns": after.st_mtime_ns, "counts": dict(counts), "elapsed_seconds": elapsed}


def _read_receipts(db, sources):
    """Read only matching histories by audited offset, never rescan a raw month."""
    paths = {s["month"]: Path(s["path"]) for s in sources}
    rows = db.execute(
        "SELECT month, line, offset, length, hash FROM versions WHERE story IN "
        "(SELECT DISTINCT story FROM versions WHERE relevant = 1) ORDER BY month, line")
    receipts, opened = [], {}
    with ExitStack() as stack:
        for month, line, offset, length, digest in rows:
            if month not in opened:
                opened[month] = stack.enter_context(paths[month].open("rb"))
            stream = opened[month]
            stream.seek(offset)
            raw = stream.read(length)
            if hashlib.sha256(raw).digest() != digest:
                raise ValueError(f"Raw receipt changed: {month}:{line}")
            locator = {"path": str(paths[month]), "month": month, "line": line,
                       "byte_offset": offset, "byte_length": length, "raw_sha256": digest.hex()}
            receipts.append(_version(_raw_parser()(raw), locator))
    return receipts


def _preserve_build(output):
    lock = output / ".build.lock"
    with lock.open("x", encoding="ascii"):
        pass
    try:
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = output / "versions" / version
        destination.mkdir(parents=True, exist_ok=False)
        for name in ARTIFACTS:
            source = output / name
            if source.exists():
                source.rename(destination / name)
        print("Preserved previous calendar: " + str(destination), flush=True)
    finally:
        lock.unlink()


def build_calendar(output_dir, cases, *, raw_dir=RAW_NEWS, rebuild=False):
    """Scan raw months once and publish immutable receipts/features/manifest.

    Returns {case_id: feature}. Output must be OUTPUT_ROOT or a descendant.
    Complete builds with identical inputs can be reused after hash validation;
    existing named artifacts are never replaced. Set rebuild=True to move an
    incompatible/incomplete build into versions/<UTC timestamp> before a fresh
    scan. Raw histories are receipts.jsonl.
    `raw_dir` permits small local fixtures; there is intentionally no archive or
    retrieval-result input that could silently substitute incomplete coverage.
    """
    cases = _cases(cases)
    output = Path(output_dir).resolve()
    allowed = OUTPUT_ROOT.resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("Calendar artifacts must stay under " + str(allowed))
    plan = scan_plan(cases, raw_dir)
    fingerprints = {
        "schema_version": SCHEMA_VERSION,
        "cases_sha256": hashlib.sha256(_json(cases).encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parser_sha256": hashlib.sha256((ROOT / "code/news_percentile/build.py").read_bytes()).hexdigest(),
    }
    if any((output / name).exists() for name in ARTIFACTS):
        if all((output / name).is_file() for name in ARTIFACTS):
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            same = manifest.get("source_files_complete") and all(manifest.get(k) == v for k, v in fingerprints.items())
            for current, saved in zip(plan, manifest.get("sources", [])):
                same = same and current == {k: saved.get(k) for k in current}
                same = same and current["exists"] and Path(current["path"]).stat().st_mtime_ns == saved.get("mtime_ns")
            same = same and len(plan) == len(manifest.get("sources", []))
            if same:
                for name in ARTIFACTS[:2]:
                    with (output / name).open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["artifact_sha256"][name]:
                            raise ValueError("Calendar artifact checksum mismatch: " + name)
                with (output / "features.jsonl").open(encoding="utf-8") as stream:
                    features = [json.loads(line) for line in stream]
                return {f["case_id"]: f for f in features}
        if not rebuild:
            raise FileExistsError("Existing calendar is incomplete or inputs changed; use rebuild=True to preserve and rebuild")
        _preserve_build(output)
    upper = max(c["cutoff"] for c in cases)
    output.mkdir(parents=True, exist_ok=True)
    # The lock is exclusive, and the manifest is published last as the completion marker.
    lock = output / ".build.lock"
    with lock.open("x", encoding="ascii"):
        pass
    try:
        with tempfile.TemporaryDirectory(prefix=".build-", dir=output) as temporary:
            stage = Path(temporary)
            db = sqlite3.connect(":memory:")
            try:
                db.execute("PRAGMA temp_store=MEMORY")
                page_size = db.execute("PRAGMA page_size").fetchone()[0]
                db.execute(f"PRAGMA max_page_count={MAX_INDEX_BYTES // page_size}")
                db.execute("CREATE TABLE versions (story TEXT, available TEXT, relevant INTEGER, "
                           "month TEXT, line INTEGER, offset INTEGER, length INTEGER, hash BLOB)")
                db.execute("CREATE INDEX story_versions ON versions(story)")
                sources = [_scan_month(db, item, upper) for item in plan]
                receipts = _read_receipts(db, sources)
                index_bytes = db.execute("PRAGMA page_count").fetchone()[0] * page_size
            except sqlite3.OperationalError as exc:
                if "full" in str(exc).lower():
                    raise RuntimeError("Calendar locator index exceeded its memory bound") from exc
                raise
            finally:
                db.close()
            features = [feature_per_case(c, receipts) for c in cases]
            complete = all(s["status"] == "scanned" and not any(s["counts"].get(k, 0) for k in
                           ("unparsed_lines", "missing_story_id", "invalid_timestamp")) for s in sources)
            for feature in features:
                feature["source_files_complete"] = complete
            for name, rows in (("receipts.jsonl", receipts), ("features.jsonl", features)):
                with (stage / name).open("w", encoding="utf-8", newline="\n") as saved:
                    for row in rows:
                        saved.write(_json(row) + "\n")
            import importlib.metadata
            manifest = {**fingerprints, "origins": ORIGINS, "tickers": TICKERS,
                        "case_count": len(cases), "receipt_count": len(receipts),
                        "source_files_complete": complete, "sources": sources,
                        "source_history_start": "2025-01-01", "last_cutoff": upper,
                        "archive_reuse": False, "no_event_inferred_from_absence": False,
                        "max_record_bytes": MAX_RECORD_BYTES, "max_index_bytes": MAX_INDEX_BYTES,
                        "index_bytes": index_bytes, "receipt_direct_reads": len(receipts),
                        "exchange_calendars_version": importlib.metadata.version("exchange-calendars"),
                        "parser": "news_percentile.build.parse_line (same helper as midcap_tools.prepare)",
                        "artifact_sha256": {}}
            for name in ARTIFACTS[:2]:
                with (stage / name).open("rb") as stream:
                    manifest["artifact_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
            (stage / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
            for name in ARTIFACTS:
                # Exclusive creation also protects against unrelated concurrent writers.
                with (output / name).open("xb") as target, (stage / name).open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
    finally:
        lock.unlink()
    return {f["case_id"]: f for f in features}
