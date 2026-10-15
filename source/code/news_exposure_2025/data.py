"""Read-only source ingestion; isolated experiment outputs and sealed labels."""
from collections import Counter
from contextlib import closing
import argparse
import gzip
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

import news_event_risk_data_2025 as legacy
from news_agent_batched_runtime_2025 import digest, read, write, locked_write
from .config import OUT, OLD, ROOT, FIRMS, SPLITS, TEST, VERSION
from .graph import build_graph, freeze_comentions


def cutoff(day):
    return (pd.Timestamp(day) + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York").tz_convert("UTC")


def schedule():
    import news_event_calendar_2025 as calendar
    sessions = calendar._sessions()
    days = [r[0] for r in sessions]
    result = {}
    for day, split in SPLITS.items():
        at = days.index(day)
        result[day] = {"split": split, "next_sessions": days[at+1:at+6],
                       "available_at": sessions[at+6][1].isoformat()}
    test_windows = [s for d in TEST for s in result[d]["next_sessions"]]
    if len(test_windows) != len(set(test_windows)):
        raise ValueError("Evaluation outcomes overlap")
    for a, b in (("train", "valid"), ("valid", "test")):
        if max(pd.Timestamp(x["available_at"]) for x in result.values() if x["split"] == a) >= min(cutoff(d) for d, x in result.items() if x["split"] == b):
            raise ValueError("Outcome maturity overlaps the following split")
    return result


def numerical(out):
    if (out / "numerical_manifest.json").exists():
        saved = read(out / "numerical_manifest.json")
        if saved["schedule"] != schedule():
            raise ValueError("Cached numerical schedule changed")
        for name, sha in saved["outputs"].items():
            if legacy.file_hash(out / name) != sha:
                raise ValueError("Numerical artifact changed: " + name)
        return read(out / "cases.json")
    p, benchmark = legacy.numerical_helpers()
    universe = pd.read_csv(legacy.UNIVERSE).set_index("ticker").loc[list(FIRMS)].reset_index()
    anchor = legacy.frozen_anchor()
    raw = p.read_raw("2023-01-01", "2025-12-31", universe.permno)
    compact = p.read_frame(p.COMPACT, ["permno", "date", "ret", "prc"], "2023-01-01", "2025-12-31", universe.permno)
    expected = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(legacy.TRADING).date))
    expected = expected[(expected >= "2023-01-01") & (expected <= "2025-12-31")]
    daily, audit = legacy.union_returns(raw, compact, expected)
    full_calendar = expected.append(pd.DatetimeIndex(["2026-01-02", "2026-01-05"]))
    timing = schedule()
    cases, labels, histories = [], [], []
    for firm in universe.itertuples():
        panel = daily[daily.permno.eq(firm.permno)].set_index("date").reindex(full_calendar)
        for day, item in timing.items():
            date = pd.Timestamp(day)
            case = legacy.numerical_case(firm, panel, full_calendar, date, item["split"], anchor, benchmark)
            case["asof"] = cutoff(day).isoformat()
            if case["next_sessions"] != item["next_sessions"]:
                raise ValueError("Return/exchange calendar mismatch")
            y = legacy.exact_target(panel.loc[pd.to_datetime(item["next_sessions"]), "ret"].to_numpy(float))
            cases.append(case)
            labels.append({k: case[k] for k in ("case_id", "date", "ticker", "split", "b", "anchor")} |
                          {"y": y, "available_at": item["available_at"]})
            history = panel.loc[:date].tail(benchmark.TRAILING_SESSIONS)
            histories.append({"case_id": case["case_id"], "dates": history.index.strftime("%Y-%m-%d").tolist(),
                              "returns": history.ret.to_numpy(float).tolist()})
        print("NUMERICAL", firm.ticker, len(timing), flush=True)
    cases.sort(key=lambda c: (c["date"], c["ticker"]))
    write(out / "cases.json", cases)
    pd.DataFrame(labels).to_parquet(out / "labels_sealed.parquet", index=False)
    pd.DataFrame(histories).to_parquet(out / "garch_history.parquet", index=False)
    write(out / "anchor.json", anchor)
    sources = [legacy.provenance(x) for x in (legacy.UNIVERSE, legacy.TRADING, legacy.FROZEN_ANCHOR,
               legacy.ANCHOR_LABELS, p.COMPACT, p.PROCESSED / "crsp_daily.parquet", p.PROCESSED / "crsp_daily_2025.parquet")]
    write(out / "numerical_manifest.json", {"version": VERSION, "schedule": timing, "calendar": audit,
          "sources": sources, "target": "log(mean(next five squared decimal total returns))",
          "cutoff": "16:15 America/New_York, all origin dates including early closes",
          "garch": "zero-mean normal GARCH(1,1), 504-session rolling fit, analytic five-session mean variance",
          "vintages_certified": False, "outputs": {n: legacy.file_hash(out/n) for n in
          ("cases.json", "labels_sealed.parquet", "garch_history.parquet", "anchor.json")}})
    return cases


def load_indexes(out):
    indexes = []
    mapper = parser = None
    for path in sorted((OLD / "reuters_index").glob("*.json")):
        item = read(path)
        src = item["source"]
        # Verify bytes, not just mtime, once before the full preparation.
        if legacy.file_hash(Path(src["path"])) != src["sha256"]:
            raise ValueError("Reuters archive checksum mismatch")
        target = out / "reuters_index" / (path.stem + ".jsonl.gz")
        meta = target.with_suffix(".manifest.json")
        if not target.exists() or not meta.exists():
            if mapper is None:
                p, parser = legacy.news_helpers()
                mapper = p.NewsMapper(pd.read_parquet(p.NAMES), pd.read_csv(p.RIC))
            import hashlib
            target.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with legacy.atomic_path(target) as temporary:
                with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as output, Path(src["path"]).open("rb") as raw_file:
                    offset = 0
                    for number, raw in enumerate(raw_file, 1):
                        obj = parser.parse_line(raw)
                        row = legacy.normalize_article(obj, parser, mapper)
                        if row is not None:
                            sha = hashlib.sha256(raw).hexdigest()
                            row.update(source_month=path.stem, source_line=number, byte_offset=offset,
                                       byte_length=len(raw), raw_sha256=sha, source_id=f"RTRS:{path.stem}:{number}:{sha[:12]}")
                            output.write(json.dumps(row, separators=(",", ":"))+"\n")
                            count += 1
                        offset += len(raw)
            if count != item["counts"]["versions"]:
                raise ValueError("Compressed index omitted records")
            write(meta, {"sha256": legacy.file_hash(target), "source_sha256": src["sha256"], "records": count})
            print("COMPRESSED INDEX", path.stem, count, flush=True)
        saved = read(meta)
        if saved["source_sha256"] != src["sha256"] or legacy.file_hash(target) != saved["sha256"]:
            raise ValueError("Compressed index provenance mismatch")
        item = {**item, "index": str(target), "index_sha256": saved["sha256"], "format": "jsonl.gz"}
        indexes.append(item)
    if len(indexes) != 6:
        raise ValueError("Six complete July-December Reuters indexes required")
    return indexes


def resolve_history(indexes, asof, eligible=None):
    """All versions since July; superseding deletions resolved before filtering."""
    latest = {}
    at = pd.Timestamp(asof).isoformat(timespec="microseconds")
    for item in indexes:
        if item["first_available_at"] >= at:
            continue
        with gzip.open(item["index"], "rt", encoding="utf-8") as db:
            for text in db:
                row = json.loads(text)
                if row["available_at"] >= at:
                    continue
                key = (row["version_created"], row["take_sequence"], row["available_at"], row["source_id"])
                old = latest.get(row["story_id"])
                if old is None or key > old[0]:
                    keep = eligible is None or bool(set(row["ids"]) & eligible) or relevant_common(row)
                    # Keep version keys for irrelevant updates too: dropping the
                    # full body metadata must never resurrect an older tagged take.
                    latest[row["story_id"]] = (key, row if keep else None)
    return [r for _, r in latest.values() if r is not None]


def calendar_features(out, cases):
    module = legacy.isolated_module("_exposure_calendar", ROOT / "code/news_event_calendar_2025.py")
    module.OUTPUT_ROOT = out / "calendar"
    module._SPLITS = SPLITS
    module.ORIGINS = {s: tuple(d for d in SPLITS if SPLITS[d] == s) for s in ("train", "valid", "test")}
    return module.build_calendar(out / "calendar", cases)


def family(headline):
    # Deduplicate update-labelled copies only; related but different reports survive.
    return re.sub(r"^(?:(?:UPDATE|WRAPUP)\s*\d*|BRIEF|CORRECTED|REFILE)\s*[-:]\s*", "", headline, flags=re.I).strip().lower()


def relevant_common(row):
    if row["kind"] in {"macro", "finance"}:
        return True
    topics = set(row["topics"])
    if not row["ids"]:
        return bool(topics & (legacy.MACRO_TAGS | legacy.FINANCE_TAGS))
    return bool(topics & legacy.MACRO_TAGS) and bool(re.search(
        r"\b(?:tariffs?|sanctions?|export (?:controls?|curbs?)|central bank|Fed|FOMC|inflation|payrolls?)\b",
        row["headline"], re.I))


def prepare(out=OUT, refresh=False):
    out.mkdir(parents=True, exist_ok=True)
    cases = numerical(out)
    features = calendar_features(out, cases)
    if (out / "data_manifest.json").exists() and not refresh:
        manifest = read(out / "data_manifest.json")
        for name, sha in manifest["outputs"].items():
            if legacy.file_hash(out / name) != sha:
                raise ValueError("Prepared artifact changed: " + name)
        return cases
    if refresh and (out / "RUN_MANIFEST.json").exists():
        raise ValueError("Cannot refresh a frozen provider experiment")
    if refresh and (out / "data_manifest.json").exists():
        old = read(out / "data_manifest.json")
        locked_write(out / "preparation_history" / (digest(old)+".json"), old)
    indexes = load_indexes(out)
    universe = pd.read_csv(legacy.UNIVERSE).set_index("ticker").loc[list(FIRMS)].reset_index()
    graph = build_graph(out, universe)
    july = resolve_history(indexes[:1], "2025-08-01T00:00:00+00:00")
    freeze_comentions(graph, july, out)
    relations, eligible = {}, set(int(p) for p in universe.permno)
    for firm in universe.itertuples():
        paths = graph.paths(firm.ticker, 2)
        cm = graph.data["comention"].get(str(firm.permno), [])
        relations[firm.ticker] = paths
        eligible.update(p for row in paths for p in row["permnos"])
        eligible.update(x["permno"] for x in cm)
    write(out / "relation_candidates.json", relations)
    _, parser = legacy.news_helpers()
    outputs = ["cases.json", "labels_sealed.parquet", "graph.json", "relation_candidates.json"]
    controls, corpus_audit = [], []
    # Raw body cache shared across dates and firms, never a model-generated summary.
    bodies = {}
    for day in sorted(SPLITS):
        rows = resolve_history(indexes, cutoff(day).isoformat(), eligible)
        selected, group = [], {}
        at = cutoff(day)
        for row in rows:
            if not row["usable"] or not (set(row["ids"]) & eligible or relevant_common(row)):
                continue
            age = (at-pd.Timestamp(row["available_at"])).total_seconds()/86400
            own_target = bool(set(row["ids"]) & set(universe.permno.astype(int)))
            if age > 31 and not own_target:
                continue
            if legacy.NON_SUBSTANTIVE.search(row["headline"]):
                continue
            key = (tuple(row["ids"]), family(row["headline"]), str(pd.Timestamp(row["available_at"]).date()))
            if key not in group or row["available_at"] > group[key]["available_at"]:
                group[key] = row
        for row in group.values():
            sid = row["source_id"]
            if sid not in bodies:
                bodies[sid] = legacy.read_article(row, parser)
            selected.append({**row, "body": bodies[sid], "event_key": digest([row["story_id"], family(row["headline"])])[:20],
                             "grouping": "latest story version plus exact normalized same-day headline; not gold event clustering"})
        selected.sort(key=lambda r: (r["available_at"], r["source_id"]))
        path = out / "corpus" / (day + ".json.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with legacy.atomic_path(path) as temporary:
            with gzip.open(temporary, "wt", encoding="utf-8") as f:
                json.dump(selected, f, ensure_ascii=True, allow_nan=False)
        outputs.append(str(path.relative_to(out)))
        for case in (c for c in cases if c["date"] == day):
            cal = features[case["case_id"]]
            recent_count = sum(case["permno"] in r["ids"] and (at-pd.Timestamp(r["available_at"])).days < 7 for r in selected)
            controls.append({"case_id": case["case_id"], "date": day, "ticker": case["ticker"], "b": case["b"],
                             "earnings_confirmed": int(cal["status"] == "confirmed_in_window"),
                             "calendar_unknown": int(cal["status"] == "unknown"),
                             **{f"rv{n}": case["state"][f"log_ret_rv{n}"] for n in (1, 5, 22)},
                             "log_news_count": float(np.log1p(recent_count))})
        corpus_audit.append({"date": day, "resolved_eligible_stories": len(rows), "eligible_events": len(selected)})
        print("CORPUS", day, len(selected), flush=True)
    pd.DataFrame(controls).to_parquet(out / "controls.parquet", index=False)
    write(out / "calendar_features.json", features)
    outputs += ["controls.parquet", "calendar_features.json"]
    write(out / "data_manifest.json", {"version": VERSION, "indexes": indexes, "corpus": corpus_audit,
          "graph_limit": "All direct links; up to 1500 two-hop paths balanced by topology and intermediate; missing identities remain explicit",
          "news_history": "July onward for persistent own events, 31 days for common/related events",
          "outputs": {p: legacy.file_hash(out/p) for p in outputs}, "provider_calls": 0})
    return cases


def embed(out=OUT):
    """Locally cached sentence embeddings; no external embedding API."""
    import torch
    from sentence_transformers import SentenceTransformer
    from midcap_tools.retrieval import _sentence_spans
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    cache, metadata = {}, {}
    global_keys = []
    if (out / "embedding_manifest.json").exists():
        previous = read(out / "embedding_manifest.json")
        for name, sha in previous["outputs"].items():
            if legacy.file_hash(out/name) != sha:
                raise ValueError("Existing embedding cache changed")
        metadata = read(out / "corpus/chunks.json")
        with np.load(out / "corpus/embeddings.npz", allow_pickle=False) as z:
            global_keys = z["keys"].tolist()
            cache = dict(zip(global_keys, z["vectors"]))
    for path in sorted((out / "corpus").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)
        fresh, texts = [], []
        for row in rows:
            sid = row["source_id"]
            if sid in metadata:
                continue
            body = row["body"]
            spans, begin, end = [], None, None
            for a, b in _sentence_spans(body):
                if begin is not None and b-begin > 1400:
                    spans.append((begin, end))
                    begin = None
                begin = a if begin is None else begin
                end = b
            if begin is not None:
                spans.append((begin, end))
            if not spans:
                spans = [(0, len(body))]
            metadata[sid] = spans
            for n, (a, b) in enumerate(spans):
                fresh.append(f"{sid}#{n}")
                texts.append(row["headline"] + "\n" + body[a:b])
        if texts:
            vectors = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
            cache.update(zip(fresh, vectors.astype(np.float16)))
            global_keys.extend(fresh)
        keys = [f"{r['source_id']}#{i}" for r in rows for i in range(len(metadata[r["source_id"]]))]
        day = path.name[:10]
        print("EMBED", day, len(keys), "chunks", flush=True)
    # Store shared chunks once, not once per overlapping origin.
    np.savez_compressed(out / "corpus/embeddings.npz", keys=np.asarray(global_keys), vectors=np.asarray([cache[k] for k in global_keys]))
    write(out / "corpus/chunks.json", metadata)
    artifacts = [out / "corpus/embeddings.npz", out / "corpus/chunks.json"]
    write(out / "embedding_manifest.json", {"model": "sentence-transformers/all-MiniLM-L6-v2", "local_only": True,
          "device": str(model.device), "outputs": {str(p.relative_to(out)): legacy.file_hash(p) for p in artifacts}})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prepare", "embed"), default="prepare")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--refresh", action="store_true", help="Rebuild pre-provider corpus, preserving prior preparation manifest")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.refresh)
    else:
        embed(args.out)
