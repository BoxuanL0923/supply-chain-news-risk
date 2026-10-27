"""Standalone stratified degree-preserving switches and audited placebo test.

Generate graph draws, then let the host recompute/refit the SAME feature on
those graphs. Evaluation accepts their aligned held-out SSE summaries. No
arbitrary expression execution, model calls, or outcome files are hidden here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Finite numeric input required")
    return float(value)


def graph_state(graph):
    if type(graph["directed"]) is not bool:
        raise ValueError("directed must be boolean")
    nodes = graph["nodes"]
    if not nodes or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in nodes.items()):
        raise ValueError("nodes must map string IDs to nonmissing industry strata")
    occupied, degrees, mixing = set(), Counter(), Counter()
    for edge in graph["edges"]:
        a, b, period = edge["source"], edge["target"], edge["period"]
        if a not in nodes or b not in nodes or a == b:
            raise ValueError("Known endpoints and no self loops required")
        parsed = date.fromisoformat(period + "-01")
        if period != parsed.strftime("%Y-%m") or parsed.year not in range(2015, 2022):
            raise ValueError("Period must be a calendar training month YYYY-MM")
        pair = (a, b) if graph["directed"] else tuple(sorted((a, b)))
        key = (period, *pair)
        if key in occupied:
            raise ValueError("Duplicate graph edge")
        occupied.add(key)
        if graph["directed"]:
            degrees[period, a, "out", nodes[b]] += 1
            degrees[period, b, "in", nodes[a]] += 1
            mixing[period, nodes[a], nodes[b]] += 1
        else:
            degrees[period, a, nodes[b]] += 1
            degrees[period, b, nodes[a]] += 1
            mixing[(period, *sorted((nodes[a], nodes[b])))] += 1
    return occupied, degrees, mixing


def verify_rewire(original, draw):
    old, degrees, mixing = graph_state(original)
    new, other_degrees, other_mixing = graph_state(draw)
    if original["nodes"] != draw["nodes"] or original["directed"] != draw["directed"]:
        raise ValueError("Draw changed node strata or directionality")
    if degrees != other_degrees or mixing != other_mixing or len(old) != len(new):
        raise ValueError("Draw failed degree/industry/period preservation")
    # Slot attributes travel with their original edge, including old-edge labels.
    attrs = lambda e: {k: v for k, v in e.items() if k not in ("source", "target")}
    if [attrs(e) for e in original["edges"]] != [attrs(e) for e in draw["edges"]]:
        raise ValueError("Draw changed edge-slot attributes")
    return dict(degrees_preserved=True, industry_degrees_preserved=True, strata="calendar month x endpoint industries",
                changed_fraction=len(old - new) / len(old) if old else 0., graph_sha256=digest(draw))


def rewire(graph, seed=20261026, sweeps=20):
    occupied, _, _ = graph_state(graph)
    if type(seed) is not int or type(sweeps) is not int or sweeps < 1:
        raise ValueError("Integer seed and positive integer sweeps required")
    edges = [dict(e) for e in graph["edges"]]
    nodes = graph["nodes"]
    rng, swaps = random.Random(seed), 0
    directed = graph["directed"]
    key = lambda p, a, b: (p, a, b) if directed else (p, *sorted((a, b)))
    blocks = {}
    for i, e in enumerate(edges):
        pair = (nodes[e["source"]], nodes[e["target"]])
        if not directed:
            pair = tuple(sorted(pair))
        blocks.setdefault((e["period"], *pair), []).append(i)
    for _, ids in sorted(blocks.items()):
        if len(ids) < 2:
            continue
        for _ in range(sweeps * len(ids)):
            i, j = rng.sample(ids, 2)
            e, f = edges[i], edges[j]
            a, b, c, d = e["source"], e["target"], f["source"], f["target"]
            if not directed and rng.random() < .5:
                c, d = d, c
            if nodes[a] != nodes[c] or nodes[b] != nodes[d] or len({a, b, c, d}) < 4:
                continue
            period = e["period"]
            u, v = key(period, a, d), key(period, c, b)
            if u in occupied or v in occupied:
                continue
            occupied.remove(key(period, e["source"], e["target"]))
            occupied.remove(key(period, f["source"], f["target"]))
            occupied.update((u, v))
            e.update(source=a, target=d); f.update(source=c, target=b)
            swaps += 1
    result = dict(graph, edges=edges)
    audit = verify_rewire(graph, result)
    audit.update(seed=seed, sweeps=sweeps, swaps=swaps)
    return result, audit


def loss_score(rows, profile):
    blocks, ids, signature, parities = {}, set(), [], set()
    total_n = old = new = 0.
    for row in rows:
        identity = str(row["row_id"])
        if not identity or identity in ids:
            raise ValueError("Unique row_id required")
        ids.add(identity)
        day = date.fromisoformat(str(row["date"]))
        iso = day.isocalendar()
        fit = finite(row["fit_parity"])
        if day.year not in range(2015, 2022) or fit not in (0, 1) or fit == iso.week % 2:
            raise ValueError("Training years and alternate-week fit provenance required")
        if profile != "loop" and iso.week % 2 != 1:
            raise ValueError("agent-v4/o2 use odd validation weeks")
        parities.add(iso.week % 2)
        n, s0, s1 = (finite(row[k]) for k in ("n", "sse_incumbent", "sse_candidate"))
        if n <= 0 or n != int(n) or min(s0, s1) < 0:
            raise ValueError("Positive counts and nonnegative SSE required")
        total_n += n; old += s0; new += s1
        blocks[(iso.year, iso.week)] = True
        signature.append((identity, day.isoformat(), n, s0, fit))
    if len(blocks) < 2 or old <= 0 or (profile == "loop" and parities != {0, 1}):
        raise ValueError("Two clusters, positive incumbent SSE and complete folds required")
    statistic = 100 * (1 - new / old) if profile == "o2" else (old - new) / total_n
    return statistic, digest(sorted(signature))


def evaluate(bundle, profile="agent-v4", threshold=.05, draws=100, two_sided=False):
    if profile not in ("agent-v4", "loop", "o2"):
        raise ValueError("Unknown profile")
    if not 0 < finite(threshold) <= .05 or type(draws) is not int or draws < 1:
        raise ValueError("threshold must be in (0, .05]; draws must be a positive integer")
    if type(two_sided) is not bool:
        raise ValueError("two_sided must be boolean")
    if two_sided and (profile != "agent-v4" or not bundle.get("ambiguity_rationale", "").strip()):
        raise ValueError("Two-sided direction-free gain requires agent-v4 and an ex ante ambiguity rationale")
    if not bundle.get("operator", "").strip() or bundle.get("same_operator_refit") is not True:
        raise ValueError("Host must attest identical feature construction and refitting for every draw")
    if len(bundle["draws"]) != draws:
        raise ValueError("Draw count does not equal the declared draw count")
    real, sample = loss_score(bundle["observed_rows"], profile)
    gains, audits, seeds = [], [], set()
    for item in bundle["draws"]:
        if type(item["seed"]) is not int or item["seed"] in seeds:
            raise ValueError("Each draw needs a distinct integer seed")
        seeds.add(item["seed"])
        # Reproduce draws rather than trusting an unaudited topology receipt.
        expected, audit = rewire(bundle["graph"], item["seed"], item.get("sweeps", 20))
        if expected != item["graph"]:
            raise ValueError("Supplied graph differs from the seeded stratified switch result")
        gain, other = loss_score(item["rows"], profile)
        if sample != other:
            raise ValueError("Observed and null sample/count/baseline/fold mismatch")
        gains.append(gain); audits.append(audit)
    exceed = sum(g >= real - 1e-12 for g in gains)
    correction = 0 if profile == "o2" else 1
    p = (correction + exceed) / (draws + correction)
    return dict(gate="rewire_placebo", profile=profile, passed=p < threshold,
                p=p, threshold=threshold, draws=draws, exceedances=exceed,
                formula="count/B" if profile == "o2" else "(1+count)/(B+1)", tie_tolerance=1e-12,
                sidedness="two-sided direction-free validation gain" if two_sided else "one-sided upper-tail validation gain",
                exploratory=two_sided or draws != 100, guideline_two_sided_threshold=.025,
                statistic_units="relative SSE reduction percent" if profile == "o2" else "MSE reduction",
                observed_gain=real, null_gains=gains, sample_sha256=sample, rewires=audits,
                input_sha256=digest(bundle),
                verification="Seeded rewires reproduced; month/industry degrees and sample alignment verified; host operator/refit attested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--input", required=True, help="Graph JSON")
    generate.add_argument("--draws", type=int, default=100)
    generate.add_argument("--seed", type=int, default=20261026)
    generate.add_argument("--sweeps", type=int, default=20)
    gate = sub.add_parser("evaluate")
    gate.add_argument("--input", required=True, help="Bundle JSON")
    gate.add_argument("--profile", choices=("agent-v4", "loop", "o2"), default="agent-v4")
    gate.add_argument("--threshold", type=float, default=.05)
    gate.add_argument("--draws", type=int, default=100)
    gate.add_argument("--two-sided", action="store_true")
    args = parser.parse_args()
    try:
        obj = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if args.command == "generate":
            if args.draws < 1:
                raise ValueError("Positive draw count required")
            result = []
            for i in range(args.draws):
                graph, audit = rewire(obj, args.seed + i, args.sweeps)
                result.append(dict(graph=graph, seed=args.seed + i, sweeps=args.sweeps, audit=audit))
        else:
            result = evaluate(obj, args.profile, args.threshold, args.draws, args.two_sided)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status="ERROR", error=str(error))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if args.command == "generate" or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
