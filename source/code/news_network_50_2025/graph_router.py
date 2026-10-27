"""Deterministic graph-tool routing before a selector/model call.

Standalone entry point only: no active workflow, toolkit or ablation imports
this module. Rules may be serialized for later training-only GEPA/prompt GD.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .graph_skills import Snapshot, utc


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RouterRules:
    min_candidate_neighbours: int = 2
    max_hops: int = 3
    thresh_hub: float = 90.
    single_neighbour_weight: float = .5
    thresh_ra: float = 75.
    thresh_clust: float = .5
    path_limit: int = 200

    def __post_init__(self):
        if type(self.min_candidate_neighbours) is not int or self.min_candidate_neighbours < 2:
            raise ValueError("Triangle trigger needs at least two candidate neighbours")
        if type(self.max_hops) is not int or not 1 <= self.max_hops <= 3:
            raise ValueError("Hard hop cap: max_hops must be 1..3")
        if type(self.path_limit) is not int or not 1 <= self.path_limit <= 2000:
            raise ValueError("path_limit must be 1..2000")
        for key, lo, hi in (("thresh_hub", 0, 100), ("thresh_ra", 0, 100),
                            ("thresh_clust", 0, 1), ("single_neighbour_weight", 0, 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid {key}")
        if self.single_neighbour_weight >= 1:
            raise ValueError("Hub single-neighbour weight must down-weight evidence (<1)")


def _inspected(case, graph, snapshot_hash):
    prefixes, receipts = set(), []
    for item in case.get("inspections", []):
        path = item["path"]
        if (not isinstance(path, list) or not 2 <= len(path) <= 3 or len(set(path)) != len(path)
                or path[0] != case["firm_id"]):
            raise ValueError("Inspected prefixes must be simple target-rooted 1- or 2-hop paths")
        if any(b not in graph.adj.get(a, ()) for a, b in zip(path, path[1:])):
            raise ValueError("Inspected prefix is absent from active dated graph")
        if item["case_id"] != case["case_id"] or item["snapshot_sha256"] != snapshot_hash or item["cutoff"] != case["cutoff"]:
            raise ValueError("Inspection receipt belongs to another case, snapshot or cutoff")
        if (item.get("inspected") is not True or not isinstance(item.get("source_ids"), list)
                or not item["source_ids"] or any(not isinstance(s, str) or not s.strip() for s in item["source_ids"])):
            raise ValueError("Host-inspected source receipts required for deeper routing")
        if utc(item["available_at"]) >= utc(case["cutoff"]):
            raise ValueError("Inspected source must be strictly available before cutoff")
        if tuple(path) in prefixes:
            raise ValueError("Duplicate inspection prefix")
        prefixes.add(tuple(path)); receipts.append(item)
    for path in prefixes:
        if len(path) == 3 and path[:2] not in prefixes:
            raise ValueError("Two-hop inspection also requires its exact one-hop prefix receipt")
    return prefixes, sorted(receipts, key=lambda item: item["path"])


def route_case(snapshot, case, rules=None):
    rules = rules or RouterRules()
    if not isinstance(rules, RouterRules):
        raise ValueError("rules must be RouterRules")
    target, cutoff = case["firm_id"], case["cutoff"]
    if not isinstance(case["case_id"], str) or not case["case_id"].strip():
        raise ValueError("Nonempty case_id required")
    graph = Snapshot(snapshot, cutoff)
    if target not in graph.universe:
        raise ValueError("Target must be in supplied monthly universe")
    candidates = case["candidate_neighbours"]
    if (not isinstance(candidates, list) or any(not isinstance(c, str) for c in candidates)
            or len(set(candidates)) != len(candidates) or set(candidates) - graph.adj[target]):
        raise ValueError("Candidate neighbours must be distinct active adjacent graph IDs")
    candidates = sorted(candidates)
    snapshot_hash = digest(snapshot)
    prefixes, inspections = _inspected(case, graph, snapshot_hash)
    calls, skipped = [], []
    degree = graph.degree_result(target, thresh_hub=rules.thresh_hub)
    calls.append(dict(tool="/sc-graph-degree", reason="Always measure target degree before routing motifs", result=degree))
    triangles = {}
    if len(candidates) >= rules.min_candidate_neighbours:
        for neighbour in candidates:
            value = graph.triangle_result(target, neighbour, thresh_ra=rules.thresh_ra, thresh_clust=rules.thresh_clust)
            triangles[neighbour] = value
            calls.append(dict(tool="/sc-graph-triangles", neighbour=neighbour,
                              reason="Candidate-neighbour count meets motif trigger", result=value))
    else:
        skipped.append(dict(tool="/sc-graph-triangles", reason="Fewer candidate neighbours than min_candidate_neighbours"))
    paths, truncated = [], False
    for depth in range(1, rules.max_hops + 1):
        eligible = depth == 1 or any(len(p) == depth for p in prefixes)
        if not eligible:
            skipped.append(dict(tool="path_enumeration", max_hops=depth,
                                reason=f"No inspected exact {depth-1}-hop prefix"))
            continue
        values, incomplete = graph.paths(target, depth, prefixes, rules.path_limit)
        calls.append(dict(tool="path_enumeration", max_hops=depth,
                          reason="Direct adjacency" if depth == 1 else "Host inspection receipts permit this depth",
                          paths=values, truncated=incomplete))
        paths = values
        truncated = truncated or incomplete
        if incomplete:
            skipped.append(dict(tool="path_enumeration", reason="Path budget reached; deeper expansion stopped"))
            break
    hub_single = degree["hub"] and len(candidates) == 1
    body = dict(schema_version=1, router="deterministic-pre-model", case_id=case["case_id"], firm_id=target,
                cutoff=cutoff, vintage=graph.vintage, snapshot_sha256=snapshot_hash, case_sha256=digest(case),
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                graph_implementation_sha256=hashlib.sha256(Path(__file__).with_name("graph_skills.py").read_bytes()).hexdigest(),
                rules=asdict(rules), candidate_neighbours=candidates, candidate_count=len(candidates),
                calls=calls, skipped=skipped, inspected_prefixes=[list(p) for p in sorted(prefixes)],
                inspection_receipts=inspections, paths=paths, truncated=truncated or degree["truncated"] or
                any(t["truncated"] for t in triangles.values()),
                evidence_weight=rules.single_neighbour_weight if hub_single else 1.,
                weight_reason="Hub with single candidate neighbour" if hub_single else "Default evidence weight",
                model_calls=0, verification="Strict pre-cutoff graph; exact host-inspected prefixes; hard three-hop cap",
                limitations="Source inspection is host-attested. Topology does not establish product continuity. Truncation means incomplete coverage.")
    return dict(body, receipt_sha256=digest(body))


def verify_receipt(receipt):
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if digest(body) != receipt["receipt_sha256"]:
        raise ValueError("Routing receipt hash mismatch")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--rules", help="Optional JSON fields for RouterRules; omitted fields retain defaults")
    args = parser.parse_args()
    try:
        read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
        result = route_case(read(args.snapshot), read(args.case), RouterRules(**read(args.rules)) if args.rules else None)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status="ERROR", error=str(error))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
