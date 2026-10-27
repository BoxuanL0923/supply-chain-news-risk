"""Prepare the full development panel (all original train/valid cases) from read-only local caches (v2, 2026-10-26).

Never opens a mixed sealed-label artifact. Labels are derived only from bounded
August raw-return queries. Sources and exact inspected path receipts stay on D:.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .graph_skills import Snapshot, graph_fields
from .prompt_gd import MODEL, digest, read, write, validate_cases
from .toolkit import Toolkit


def prepare(source, out, returns):
    out.mkdir(parents=True, exist_ok=True)
    if (out/"development_manifest.json").exists():
        raise ValueError("Prepared data already exists; do not overwrite")
    # cases.json is a capability/clock file, not an outcome table.
    metadata = read(source/"cases.json")
    seal = min(c["date"] for c in metadata if c["split"] == "test")
    selected = [c for c in metadata if c["split"] in ("train", "valid")]
    if any(c["date"] >= seal for c in selected):
        raise ValueError("Development origin at or after the first evaluation origin")
    data = read(source/"graph.json")
    universe = sorted(n for n, v in data["nodes"].items() if v.get("permnos"))
    data["monthly_universe"] = {m: universe for m in sorted({c["date"][:7] for c in selected})}
    graph = Snapshot(data, "2025-08-01T20:15:00+00:00")
    supplies = {(e["source"], e["target"]) for e in data["edges"] if e["type"] == "supplies"}
    # Store graph once, not in every case or prompt. No revenue weights.
    compact = {"nodes": list(data["nodes"]), "edges": [{k: e[k] for k in
               ("source", "target", "type", "valid_from", "vintage")} for e in data["edges"]],
               "vintage": graph.vintage, "monthly_universe": data["monthly_universe"]}
    write(out/"graph_snapshot.json", compact)
    output, audits = [], []
    for day in sorted({c["date"] for c in selected}):
        with gzip.open(source/"corpus"/(day+".json.gz"), "rt", encoding="utf-8") as handle:
            docs = json.load(handle)
        docs = sorted([d for d in docs if d["usable"]], key=lambda d: (d["available_at"], d["source_id"]), reverse=True)
        by_permno = {}
        for d in docs:
            if d["available_at"][:10] < day[:8]+"01" and day != "2025-08-01":
                continue
            for p in d["ids"]:
                by_permno.setdefault(p, []).append(d)
        for original in [c for c in selected if c["date"] == day]:
            case = {k: original[k] for k in ("case_id", "date", "asof", "ticker", "anchor", "next_sessions")}
            case["split"] = original["split"]
            target = data["targets"][case["ticker"]][0]
            graph.cutoff = case["asof"]  # same frozen June graph, universe held fixed
            used, candidates, relations, inspected = set(), [], [], []
            own = [d for d in by_permno.get(original["permno"], []) if d["available_at"] < case["asof"]][:2]
            for d in own:
                used.add(d["source_id"])
                candidates.append({"source_id": d["source_id"], "available_at": d["available_at"],
                                   "text": d["body"], "ids": d["ids"], "hop": 0, "graph": {}, "path": []})
            # Two narrow, staged chains. Every prefix is backed by an actual source read.
            kit = Toolkit({"case": case, "network": True, "documents": {}, "relations": relations,
                           "calendar": {}, "controls": {}, "coverage": {}})
            for branch in range(2):
                chain = [target]
                for depth in range(1, 4):
                    choices = []
                    for n in graph.adj[chain[-1]]:
                        if n in chain:
                            continue
                        for permno in data["nodes"][n].get("permnos", []):
                            for d in by_permno.get(permno, [])[:5]:
                                if d["source_id"] not in used and original["permno"] not in d["ids"] and d["available_at"] < case["asof"]:
                                    choices.append((d["available_at"], d["source_id"], n, d))
                    if not choices:
                        break
                    _, sid, n, d = max(choices, key=lambda a: a[:3])
                    chain = chain+[n]
                    used.add(sid)
                    steps = []
                    for a, b in zip(chain[:-1], chain[1:]):
                        role = "customer" if (a, b) in supplies else "supplier" if (b, a) in supplies else "competitor"
                        steps.append({"neighbor_role": role})
                    relation = {"entity": n, "permnos": data["nodes"][n]["permnos"], "hops": depth,
                                "path": chain, "steps": steps, "vintage": graph.vintage}
                    relations.append(relation)
                    kit.docs[sid] = d
                    receipt = kit.inspect_event(sid)
                    # Entire source, not a generated excerpt. Prefix receipt precedes next hop.
                    inspected = [list(p) for p in kit.inspected_paths]
                    degree = graph.degree_result(target, depth, inspected)
                    triangle = graph.triangle_result(target, n, depth, inspected)
                    stats = graph_fields(degree, triangle)
                    candidates.append({"source_id": sid, "available_at": d["available_at"], "text": d["body"],
                                       "ids": d["ids"], "hop": depth, "path": chain, "graph": stats, "verified_paths": [relation]})
                    audits.append({"case_id": case["case_id"], "source_id": sid, "path": chain,
                                   "source_sha256": digest(d["body"]), "inspection": receipt["citation_eligible"]})
            case.update(candidates=candidates, inspected_paths=inspected, relations=relations, graph_firm_id=target)
            output.append(case)
            print(MODEL, case["case_id"], "hops", [e["hop"] for e in candidates], flush=True)
    validate_cases(output)
    write(out/"cases.json", {"model": MODEL, "cases": output})
    write(out/"inspection_audit.json", {"model": MODEL, "receipts": audits})
    # Read only the date/ID projection of raw daily returns needed for these cases.
    dates = sorted({d for c in output for d in c["next_sessions"]})
    if dates[-1] >= seal:
        raise ValueError("Sealed target window")
    ids = sorted({c["permno"] for c in selected})
    frame = pd.read_parquet(returns, columns=["permno", "date", "ret"],
        filters=[("date", ">=", dates[0]), ("date", "<=", dates[-1]), ("permno", "in", ids)])
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate raw returns")
    labels = {"train": {}, "valid": {}}
    for c in output:
        permno = next(r["permno"] for r in selected if r["case_id"] == c["case_id"])
        values = frame[frame.permno.eq(permno)].set_index("date").loc[pd.to_datetime(c["next_sessions"]), "ret"].to_numpy(float)
        if len(values) != 5 or not np.isfinite(values).all() or np.mean(values**2) <= 0:
            raise ValueError("Five finite nonzero-variance returns required")
        labels[c["split"]][c["case_id"]] = float(np.log(np.mean(values**2)))
    for split, rows in labels.items(): write(out/f"labels_{split}.json", rows)
    write(out/"development_manifest.json", {"model": MODEL, "latest_origin": max(c["date"] for c in output),
          "latest_target_session": dates[-1], "splits": {s: list(r) for s, r in labels.items()},
          "label_hashes": {s: digest(r) for s, r in labels.items()}, "cases_hash": digest(output),
          "source": str(source), "returns_source": str(returns), "original_split": "original train/valid labels kept",
          "development_split": "all original train (Aug-Sep 2025) and valid (Oct 2025) cases; evaluation from " + seal + " sealed",
          "seal_date": seal,
          "anchor": "existing pre-August OLS intercept + GARCH (not PR2 expanded 11-control refit)",
          "universe": "June mapped entities, held fixed through October; not a certified monthly membership vintage",
          "evaluation_labels_opened": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("C:/OptResearch/results/news_exposure_2025"))
    p.add_argument("--out", type=Path, default=Path("D:/OptResearch_data/agent_q1/data_v2"))
    p.add_argument("--returns", type=Path, default=Path("C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/crsp_daily_2025.parquet"))
    a = p.parse_args(); prepare(a.source, a.out, a.returns)
