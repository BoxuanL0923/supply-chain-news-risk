"""CPU/local-embedding preparation. Sealed labels never enter agent capabilities."""
from collections import Counter
import argparse
import gzip
import json
import re
from types import SimpleNamespace
import numpy as np
import pandas as pd

import news_event_risk_data_2025 as legacy
from news_agent_batched_runtime_2025 import read, write, digest
from news_exposure_2025 import data as prior
from news_exposure_2025.graph import active, build_graph, freeze_comentions
from .config import ROOT, OUT, OLD, DATES, SPLITS, BASKETS, VERSION


def schedule():
    import news_event_calendar_2025 as calendar
    sessions = calendar._sessions()
    days = [s[0] for s in sessions]
    result = {}
    for day in DATES:
        i = days.index(day)
        result[day] = {"asof": sessions[i][2].isoformat(), "next_sessions": days[i+1:i+6],
                       "available_at": sessions[i+6][1].isoformat(), "split": SPLITS[day]}
    windows = [s for r in result.values() for s in r["next_sessions"]]
    assert len(windows) == len(set(windows)) == 60
    for a, b in (("train", "valid"), ("valid", "test")):
        assert max(pd.Timestamp(r["available_at"]) for r in result.values() if r["split"] == a) < min(
            pd.Timestamp(r["asof"]) for r in result.values() if r["split"] == b)
    return result


def indexes():
    result = []
    for month in range(7, 13):
        stem = f"2025-{month:02d}"
        meta = read(ROOT / "results/news_event_risk_2025/reuters_index" / (stem + ".json"))
        path = OLD / "reuters_index" / (stem + ".jsonl.gz")
        saved = read(path.with_suffix(".manifest.json"))
        assert legacy.file_hash(path) == saved["sha256"]
        assert legacy.file_hash(meta["source"]["path"]) == meta["source"]["sha256"]
        result.append({**meta, "index": str(path), "index_sha256": saved["sha256"]})
    return result


def universe(out, july):
    path = out / "universe.csv"
    if path.exists():
        return pd.read_csv(path)
    p, _ = legacy.numerical_helpers()
    names = p.active_names(pd.read_parquet(p.NAMES), pd.Timestamp("2025-07-31"))
    names = names[names.ticker.isin(sum(BASKETS.values(), []))].copy()
    names = names[names.groupby("ticker").permno.transform("nunique").eq(1)]
    raw = p.read_raw("2025-05-01", "2025-07-31", names.permno)
    raw["dollars"] = raw.prc.abs()*raw.vol
    raw["size"] = raw.prc.abs()*raw.shrout*1000
    stats = raw.groupby("permno").agg(adv=("dollars", "mean"), median_dollars=("dollars", "median"),
                                     observations=("ret", "count"))
    stats["size"] = raw.sort_values("date").groupby("permno").tail(1).set_index("permno")["size"]
    mapping = active(pd.read_csv(legacy.SOURCE / "processed/factset_to_permno.csv"), "link_bdate", "link_edate")
    counts = Counter(p for r in july if r["usable"] for p in set(r["ids"]))
    candidates = names.merge(stats, on="permno")
    candidates["july_stories"] = candidates.permno.map(counts).fillna(0).astype(int)
    candidates["eligible"] = (candidates.adv.ge(10e6) & candidates.median_dollars.ge(5e6)
                              & candidates.observations.ge(55) & candidates["size"].ge(2e9)
                              & candidates.july_stories.ge(2) & candidates.permno.isin(mapping.PERMNO))
    selected = []
    for theme, tickers in BASKETS.items():
        group = candidates[candidates.eligible & candidates.ticker.isin(tickers)].sort_values(
            ["median_dollars", "ticker"], ascending=[False, True])
        mid = group[group["size"].lt(30e9)].head(5)
        large = group[group["size"].ge(30e9)].head(5)
        take = pd.concat([mid, large])
        take = pd.concat([take, group[~group.permno.isin(take.permno)].head(10-len(take))])
        if len(take) != 10:
            raise ValueError(f"Only {len(take)} eligible firms in {theme}; do not substitute using outcomes")
        take = take.assign(company=take.comnam, theme=theme)
        selected.append(take)
    result = pd.concat(selected).sort_values(["theme", "ticker"])
    assert len(result) == result.permno.nunique() == 50
    result.to_csv(path, index=False)
    candidates.to_csv(out / "universe_candidates.csv", index=False)
    write(out / "universe_manifest.json", {"formed_at": "2025-07-31", "baskets": BASKETS,
          "selection": "10 per basket: up to five liquid $2-30bn and five >=$30bn; fill on pre-period liquidity",
          "inputs": [legacy.provenance(x) for x in (p.NAMES, p.PROCESSED / "crsp_daily_2025.parquet")],
          "outcomes_used": False, "universe_sha256": legacy.file_hash(path)})
    return result


def numerical(out, firms):
    if (out / "numerical_manifest.json").exists():
        for name, sha in read(out / "numerical_manifest.json")["outputs"].items():
            assert legacy.file_hash(out/name) == sha
        return read(out / "cases.json")
    p, benchmark = legacy.numerical_helpers()
    original = benchmark
    def stable_garch(returns, asymmetric):
        value, status = original.garch_forecast(returns, asymmetric)
        if status == "ok":
            return value, status
        # Same likelihood/model, standardized input for deterministic numerical retry.
        from arch import arch_model
        recent = np.asarray(returns[-original.TRAILING_SESSIONS:], float)
        recent = recent[np.isfinite(recent)]
        scale = float(np.std(recent))
        if len(recent) < original.MIN_OBS or scale <= 0:
            return value, status
        fit = arch_model(recent/scale, mean="Zero", vol="GARCH", p=1, o=int(asymmetric), q=1,
                         dist="normal", rescale=False).fit(disp="off", options={"maxiter": 3000, "ftol": 1e-9})
        if fit.convergence_flag:
            return value, status
        variance = float(fit.forecast(horizon=5).variance.iloc[-1].mean())*scale**2
        print("GARCH NUMERICAL RETRY", status, flush=True)
        return max(variance, original.RISK_FLOOR), "ok"
    benchmark = SimpleNamespace(TRAILING_SESSIONS=original.TRAILING_SESSIONS,
                               RISK_FLOOR=original.RISK_FLOOR, garch_forecast=stable_garch)
    raw = p.read_raw("2023-01-01", "2025-12-31", firms.permno)
    compact = p.read_frame(p.COMPACT, ["permno", "date", "ret", "prc"], "2023-01-01", "2025-12-31", firms.permno)
    expected = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(legacy.TRADING).date))
    expected = expected[(expected >= "2023-01-01") & (expected <= "2025-12-31")]
    daily, audit = legacy.union_returns(raw, compact, expected)
    days = expected.append(pd.DatetimeIndex(["2026-01-02", "2026-01-05"]))
    cases, labels = [], []
    timing = schedule()
    for firm in firms.itertuples():
        panel = daily[daily.permno.eq(firm.permno)].set_index("date").reindex(days)
        for day, item in timing.items():
            case = legacy.numerical_case(firm, panel, days, pd.Timestamp(day), item["split"],
                                         {"coefficients": [0., 1.]}, benchmark)
            case["asof"] = item["asof"]
            case["stock_history"]["asof"] = item["asof"]
            assert case["next_sessions"] == item["next_sessions"]
            cases.append(case)
            labels.append({k: case[k] for k in ("case_id", "ticker", "date", "split", "b")} | {
                "y": legacy.exact_target(panel.loc[pd.to_datetime(item["next_sessions"]), "ret"]),
                "available_at": item["available_at"]})
        print("NUMERICAL", firm.ticker, flush=True)
    cases.sort(key=lambda c: (c["date"], c["ticker"]))
    write(out / "cases.json", cases)
    # Physically separate files: the optimizer never opens a test-label parquet.
    for split in ("train", "valid", "test"):
        pd.DataFrame([r for r in labels if r["split"] == split]).to_parquet(out / f"labels_{split}.parquet", index=False)
    files = ["cases.json"] + [f"labels_{s}.parquet" for s in ("train", "valid", "test")]
    write(out / "numerical_manifest.json", {"schedule": timing, "audit": audit,
        "target": "log mean next five squared decimal total returns; not intraday RV",
        "anchor": "raw log zero-mean Gaussian GARCH(1,1), rolling 504 sessions, analytic five-session mean variance",
        "outputs": {f: legacy.file_hash(out/f) for f in files}})
    return cases


def calendars(out, cases):
    module = legacy.isolated_module("_network50_calendar", ROOT / "code/news_event_calendar_2025.py")
    module.TICKERS = tuple(sorted({c["ticker"] for c in cases}))
    module._SPLITS = SPLITS
    receipts = []
    source = OLD / "calendar/receipts.jsonl"
    manifest = read(OLD / "calendar/manifest.json")
    assert manifest["source_files_complete"]
    assert legacy.file_hash(source) == manifest["artifact_sha256"]["receipts.jsonl"]
    # The old cache contains full raw diary objects for ALL firms, not only old parsed rows.
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("raw_object"):
                row = module._version(row["raw_object"], row["source"])
            receipts.append(row)
    result = {}
    for case in cases:
        eligible = [r for r in receipts if r["available_at"] is None or
                    pd.Timestamp(r["available_at"]) < pd.Timestamp(case["asof"])]
        result[case["case_id"]] = module.feature_per_case(module._case(case), eligible)
    write(out / "calendar_features.json", result)
    write(out / "calendar_provenance.json", {"source": legacy.provenance(source),
          "reparsed_all_firms": True, "strict_cutoff": True, "source_manifest": manifest,
          "absence_means": "unknown, never no earnings"})
    return result


def diversified(rows, scores, limit):
    ranked = sorted(zip(rows, scores), key=lambda x: (-float(x[1]), x[0]["source_id"]))
    chosen, seen = [], set()
    for row, _ in ranked:
        headline = re.sub(r"[^a-z0-9]+", " ", prior.family(row["headline"]).replace("u.s.", "us")).strip()
        key = (headline, str(pd.Timestamp(row["available_at"]).date()))
        if key not in seen:
            chosen.append(row)
            seen.add(key)
        if len(chosen) >= limit:
            break
    return chosen


def common_packet(rows, scores, ages):
    """Recent US events diversified by topic; generic rate news cannot fill every slot."""
    categories = (r"\bFed\b|FOMC|interest rate|monetary", r"payroll|jobs|CPI|PPI|inflation|employment|GDP",
                  r"shutdown|budget|debt ceiling|government spending", r"tariff|trade|sanction|export",
                  r"GLOBAL MARKETS|Wall Street|stocks|Treasury|yield|credit|bond|dollar")
    eligible = [i for i, r in enumerate(rows) if prior.relevant_common(r) and legacy.US_MACRO.search(r["headline"])]
    selected = {}
    for number, category in enumerate(categories):
        pool = [i for i in eligible if re.search(category, rows[i]["headline"], re.I)]
        if number in (2, 3):
            # Reserve these slots for policy developments, not another rates/FX reaction.
            pool = [i for i in pool if not re.search(r"^(?:TREASURIES|FOREX|GLOBAL MARKETS|LIVE MARKETS)|\bFed(?:'s)?\b",
                                                     rows[i]["headline"], re.I)]
        recent = [i for i in pool if ages[i] <= 8]
        pool = recent or pool
        # Fresh observations outrank old commentary; semantic relevance breaks ties.
        ranking = scores[pool] + 1.5*np.exp(-ages[pool]/2)
        for row in diversified([rows[i] for i in pool], ranking, 2):
            selected[row["source_id"]] = row
    return list(selected.values())


def substantive_neighbor(headline):
    # Exclude mechanical alerts, not reported analyst reasoning or operational news.
    routine = (r"<[^>]+>:\s*.*(?:TARGET PRICE|PRICE TARGET|RAISES TO|CUTS TO)|"
               r"\bFORMS? 8\.[35]\b|\bAS PER TRADING PLAN\b|"
               r"\b(?:PRICE MONITORING EXTENSION|AUCTION EXTENSION|ORDER IMBALANCE|OPENING INDICATION)\b")
    return re.search(routine, headline, re.I) is None


def three_hop_candidates(rows, scores, net_idx, relation):
    """Reserve 8/6/6 places for shortest 1/2/3-hop typed news, then fill to 20."""
    typed = {h: {p for r in relation if r.get("type") != "comention" and r["hops"] == h
                 for p in r["permnos"]} for h in (1, 2, 3)}
    tiers = {h: [] for h in (1, 2, 3)}
    for i in net_idx:
        hop = next((h for h in (1, 2, 3) if set(rows[i]["ids"]) & typed[h]), None)
        if hop is not None:
            tiers[hop].append(i)
    chosen = []
    for hop, quota in ((1, 8), (2, 6), (3, 6)):
        idx = tiers[hop]
        chosen += diversified([rows[i] for i in idx], scores[idx], quota)
    picked = {row["source_id"] for row in chosen}
    remaining = [i for i in net_idx if rows[i]["source_id"] not in picked]
    if len(chosen) < 20:
        chosen += diversified([rows[i] for i in remaining], scores[remaining], 20-len(chosen))
    return chosen


def graph_control_scope(relation, rows, net_idx, max_hops):
    """Use direct typed neighborhood for fixed controls in the 3-hop study."""
    neighbors = {p for r in relation if max_hops != 3 or
                 (r.get("type") != "comention" and r["hops"] == 1)
                 for p in r["permnos"]}
    indices = ([i for i in net_idx if set(rows[i]["ids"]) & neighbors]
               if max_hops == 3 else net_idx)
    return neighbors, indices


def prepare(out=OUT, max_hops=2):
    if max_hops not in (2, 3):
        raise ValueError("Preparation supports two or three hops; use a new output directory for each")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data_manifest.json").exists():
        saved = read(out / "data_manifest.json")
        if saved.get("max_hops", 2) != max_hops:
            raise ValueError("Prepared pool radius differs; use a separate output directory")
        for name, sha in saved["outputs"].items():
            assert legacy.file_hash(out/name) == sha, name
        return
    idx = indexes()
    july = prior.resolve_history(idx[:1], "2025-08-01T00:00:00+00:00")
    firms = universe(out, july)
    cases = numerical(out, firms)
    calendar = calendars(out, cases)
    graph = build_graph(out, firms)
    freeze_comentions(graph, july, out)
    paths, eligible = {}, set(firms.permno.astype(int))
    for firm in firms.itertuples():
        values = graph.paths(firm.ticker, 2) if max_hops == 2 else graph.pool_paths(firm.ticker, 3)
        for cm in graph.data["comention"].get(str(firm.permno), []):
            values.append({"type": "comention", "permnos": [cm["permno"]], "hops": 1,
                           "path": [], "steps": [], "mechanism_verified": False, "vintage": "2025-07-31"})
        paths[firm.ticker] = values
        eligible.update(p for r in values for p in r["permnos"])
    write(out / "relation_candidates.json", paths)
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    queries = ["earnings guidance demand production outage financing regulation tariff uncertainty",
               "US Federal Reserve inflation rates Treasury credit tariffs macroeconomic market risk",
               "supplier customer competitor disruption demand bottleneck capacity substitution earnings"]
    q = model.encode(queries, normalize_embeddings=True)
    _, parser = legacy.news_helpers()
    bodies, controls, outputs, audits = {}, [], [], []
    for day in DATES:
        dayfile = out / "pools" / (day + ".json")
        if dayfile.exists():
            cached = read(dayfile)
            controls.extend(cached["controls"])
            outputs.append(str(dayfile.relative_to(out)))
            continue
        cutoff = pd.Timestamp(schedule()[day]["asof"])
        allrows = prior.resolve_history(idx, cutoff.isoformat(), eligible)
        rows = [r for r in allrows if r["usable"] and not legacy.NON_SUBSTANTIVE.search(r["headline"])
                and (cutoff-pd.Timestamp(r["available_at"])).total_seconds() <= 31*86400]
        vectors = model.encode([r["headline"] for r in rows], batch_size=256, normalize_embeddings=True)
        scores = vectors @ q.T
        ages = np.array([(cutoff-pd.Timestamp(r["available_at"])).total_seconds()/86400 for r in rows])
        scores += .12*np.exp(-ages[:, None]/7)
        common = common_packet(rows, scores[:, 1], ages)
        pools, chosen_ids, daycontrols = {}, set(), []
        for case in (c for c in cases if c["date"] == day):
            own_idx = [i for i, r in enumerate(rows) if case["permno"] in r["ids"]]
            own = diversified([rows[i] for i in own_idx], scores[own_idx, 0], 10)
            # Preserve known pending states beyond the current news window without inferred resolution.
            history = [r for r in allrows if r["usable"] and case["permno"] in r["ids"] and
                       re.search(r"acquisition|merger|outage|shutdown|earnings|results", r["headline"], re.I)]
            history = sorted(history, key=lambda r: r["available_at"], reverse=True)[:12]
            relation = paths[case["ticker"]]
            neighbors = {p for r in relation for p in r["permnos"]}
            net_idx = [i for i, r in enumerate(rows) if case["permno"] not in r["ids"] and
                       not prior.relevant_common(r) and set(r["ids"]) & neighbors and substantive_neighbor(r["headline"])]
            if max_hops == 2:
                direct = {p for r in relation if r["hops"] == 1 for p in r["permnos"]}
                direct_idx = [i for i in net_idx if set(rows[i]["ids"]) & direct]
                higher_idx = [i for i in net_idx if not set(rows[i]["ids"]) & direct]
                net = diversified([rows[i] for i in direct_idx], scores[direct_idx, 2], 12)
                net += diversified([rows[i] for i in higher_idx], scores[higher_idx, 2], 8)
            else:
                # Fixed, outcome-free quota gives each exact-hop tier a chance
                # before the common 20-document cap fills with nearby news.
                net = three_hop_candidates(rows, scores[:, 2], net_idx, relation)
            picked = {r["source_id"] for r in net}
            remaining = [i for i in net_idx if rows[i]["source_id"] not in picked]
            if max_hops == 2 and len(net) < 20:
                net += diversified([rows[i] for i in remaining], scores[remaining, 2], 20-len(net))
            base = {r["source_id"]: r for r in own+common+history}
            base_ids = sorted(base)
            net_ids = [r["source_id"] for r in net if r["source_id"] not in base]
            chosen_ids.update(base_ids+net_ids)
            cal = calendar[case["case_id"]]
            earnings = int(cal["status"] == "confirmed_in_window")
            recent = [rows[i] for i in own_idx if ages[i] < 7]
            # These are transparent mechanical text-state controls, not a hindsight event dummy.
            post = int(any(re.search(r"(?:reports?|posts?) .*(?:quarter|results|earnings)|quarter.*(?:profit|revenue)",
                                      r["headline"], re.I) for r in recent))
            pending = int(any(re.search(r"(?:acquir|buy|takeover|acquisition|merger).*\b(?:cash|agreement|agrees)\b|all-cash",
                                         r["headline"], re.I) for r in history))
            # In a three-hop ablation, keep conventional graph controls fixed
            # and direct-only so H1 cannot see a summary of H2/H3 candidates.
            control_neighbors, control_net_idx = graph_control_scope(relation, rows, net_idx, max_hops)
            control = {"case_id": case["case_id"], "date": day, "ticker": case["ticker"], "b": case["b"],
                       **{f"rv{n}": case["state"][f"log_ret_rv{n}"] for n in (1, 5, 22)},
                       "earnings_confirmed": earnings, "calendar_unknown": int(cal["status"] == "unknown"),
                       "post_earnings": post, "pending_cash_deal": pending,
                       "log_news_count": float(np.log1p(len(recent))),
                       "log_neighbor_count": float(np.log1p(sum(ages[i] < 7 for i in control_net_idx))),
                       "log_degree": float(np.log1p(len(control_neighbors)))}
            daycontrols.append(control)
            pools[case["case_id"]] = {"case": case, "calendar": cal, "controls": control,
                "own_ids": base_ids, "common_ids": [r["source_id"] for r in common],
                "network_ids": net_ids, "history_ids": [r["source_id"] for r in history],
                "relations": relation, "coverage": {"own_31d_count": len(own_idx), "network_31d_count": len(net_idx),
                    "unmapped_paths": sum(not r["permnos"] for r in relation),
                    "mapping_note": "US RIC-to-PERMNO only; unmapped private/foreign sources explicitly missing",
                    "network_candidates": len(net_ids), "network_cap": 20}}
        byid = {r["source_id"]: r for r in allrows}
        docs = {}
        for sid in sorted(chosen_ids):
            row = byid[sid]
            if sid not in bodies:
                bodies[sid] = legacy.read_article(row, parser)
            docs[sid] = {k: row[k] for k in ("source_id", "headline", "available_at", "ids", "story_id", "raw_sha256")}
            docs[sid]["body"] = bodies[sid] or row["headline"]
        for pool in pools.values():
            pool["own_packet_hash"] = digest([docs[s] for s in pool["own_ids"]])
        write(dayfile, {"pools": pools, "documents": docs, "controls": daycontrols})
        outputs.append(str(dayfile.relative_to(out)))
        controls.extend(daycontrols)
        audits.append({"date": day, "eligible_31d": len(rows), "stored_documents": len(docs)})
        print("POOLS", day, len(docs), flush=True)
    pd.DataFrame(controls).to_parquet(out / "controls.parquet", index=False)
    outputs += ["universe.csv", "cases.json", "controls.parquet", "calendar_features.json", "graph.json",
                "relation_candidates.json", "labels_train.parquet", "labels_valid.parquet", "labels_test.parquet"]
    write(out / "data_manifest.json", {"version": VERSION, "sources": idx, "audit": audits,
        "retrieval": "local frozen MiniLM headline ranking, recency tie-break; native lexical passage search within scoped pool",
        "grouping": "latest version; exact normalized same-day headline, not gold event clustering",
        "max_hops": max_hops,
        "graph_assumption": graph.data["audit"], "outcomes_exposed_to_tools": False,
        "outputs": {name: legacy.file_hash(out/name) for name in outputs}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=__import__('pathlib').Path, default=OUT)
    parser.add_argument("--max-hops", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    prepare(args.out, args.max_hops)
