"""Dated simple-graph statistics; no labels, network, or vendor weights.

Degrees count distinct directed partners (reciprocal partners count twice in
total degree). Triangles/RA use the simple undirected projection. Percentiles
are weak empirical CDFs in an explicitly supplied monthly universe.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import math

MODEL = "qwen3.8-27b"


def utc(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


class Snapshot:
    def __init__(self, data, cutoff, universe=None):
        self.cutoff = cutoff
        at = utc(cutoff)
        self.vintage = data.get("vintage", data.get("audit", {}).get("graph_date"))
        if not self.vintage or utc(self.vintage) >= at:
            raise ValueError("Graph vintage must precede cutoff")
        self.incoming = {n: set() for n in data["nodes"]}
        self.outgoing = {n: set() for n in data["nodes"]}
        for e in data["edges"]:
            if e.get("type", "supplies") not in {"supplies", "competitor"}:
                raise ValueError("Unknown dated edge type")
            if not e.get("vintage") or not e.get("valid_from"):
                raise ValueError("Every edge needs vintage and valid_from")
            if utc(e["vintage"]) >= at or utc(e["valid_from"]) >= at:
                raise ValueError("At/after-cutoff edge rejected")
            if e.get("valid_to") and utc(e["valid_to"]) <= at:
                continue
            a, b = e["source"], e["target"]
            if a not in self.incoming or b not in self.incoming:
                raise ValueError("Edge endpoint missing")
            if a != b:
                self.outgoing[a].add(b)
                self.incoming[b].add(a)
        self.adj = {n: self.incoming[n] | self.outgoing[n] for n in self.incoming}
        month = cutoff[:7]
        self.universe = list(universe if universe is not None else data.get("monthly_universe", {}).get(month, []))
        if not self.universe or len(set(self.universe)) != len(self.universe) or set(self.universe) - self.adj.keys():
            raise ValueError("Explicit nonempty monthly universe of unique graph IDs required")
        self.distribution = sorted(self.degree(n)["total_degree"] for n in self.universe)

    def degree(self, node):
        return {"in_degree": len(self.incoming[node]), "out_degree": len(self.outgoing[node]),
                "total_degree": len(self.incoming[node]) + len(self.outgoing[node])}

    def paths(self, target, max_hops=1, inspected_prefixes=(), limit=200):
        if type(max_hops) is not int or not 1 <= max_hops <= 3:
            raise ValueError("max_hops must be 1..3")
        if target not in self.adj or target not in self.universe:
            raise ValueError("Target missing from monthly universe")
        if type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError("Path limit must be 1..2000")
        inspected = {tuple(p) for p in inspected_prefixes}
        queue, result = [[target]], []
        for depth in range(1, max_hops + 1):
            following = []
            for chain in queue:
                if depth > 1 and tuple(chain) not in inspected:
                    continue
                for node in sorted(self.adj[chain[-1]]):
                    if node in chain:
                        continue
                    path = chain + [node]
                    result.append({"path": path, "hops": depth})
                    following.append(path)
                    if len(result) >= limit:
                        return result, True
            queue = following
        return result, False

    def degree_result(self, target, max_hops=1, inspected_prefixes=(), thresh_hub=90):
        if not math.isfinite(thresh_hub) or not 0 <= thresh_hub <= 100:
            raise ValueError("THRESH_HUB must be 0..100")
        paths, truncated = self.paths(target, max_hops, inspected_prefixes)
        d = self.degree(target)
        percentile = 100 * bisect_right(self.distribution, d["total_degree"]) / len(self.distribution)
        nodes = sorted({n for p in paths for n in p["path"][1:]})
        return {"model": MODEL, "tool": "sc_graph_degree", "firm_id": target,
                "cutoff": self.cutoff, "vintage": self.vintage, "month": self.cutoff[:7],
                "universe_size": len(self.universe), **d, "degree_percentile": percentile,
                "hub": percentile >= thresh_hub, "THRESH_HUB": thresh_hub,
                "neighbour_degrees": {n: self.degree(n) for n in nodes},
                "paths": paths, "truncated": truncated, "max_hops": max_hops}

    def ra(self, target, other):
        shared = self.adj[target] & self.adj[other]
        return len(shared), sum(1 / len(self.adj[k]) for k in shared)

    def triangle_result(self, target, neighbour=None, max_hops=1, inspected_prefixes=(),
                        thresh_ra=75, thresh_clust=0.5):
        if not 0 <= thresh_ra <= 100 or not 0 <= thresh_clust <= 1:
            raise ValueError("RA percentile 0..100 and clustering threshold 0..1 required")
        paths, truncated = self.paths(target, max_hops, inspected_prefixes)
        if neighbour is not None and neighbour not in {p["path"][-1] for p in paths}:
            # Bounded listings must not hide an explicitly inspected valid branch.
            prefixes = {tuple(p) for p in inspected_prefixes}
            prefixes.add((target,))
            witnesses = [list(p)+[neighbour] for p in prefixes if p and p[0] == target
                         and len(p) <= max_hops and neighbour not in p
                         and all(b in self.adj.get(a, ()) for a, b in zip(p, p[1:]))
                         and neighbour in self.adj.get(p[-1], ())
                         and all(p[:h+1] in prefixes for h in range(1, len(p)))]
            if not witnesses:
                raise ValueError("Pair endpoint requires an accessible inspected-prefix path")
            witness = min(witnesses, key=lambda p: (len(p), p))
            paths.append({"path": witness, "hops": len(witness)-1})
        neighbours = self.adj[target]
        triangles = sum(len(neighbours & self.adj[n]) for n in neighbours) // 2
        degree = len(neighbours)
        clustering = 2 * triangles / (degree * (degree - 1)) if degree > 1 else 0.0
        # Fixed target-to-month-universe reference; not a selected-path percentile.
        distribution = sorted(self.ra(target, n)[1] for n in self.universe if n != target)
        if distribution:
            index = (len(distribution) - 1) * thresh_ra / 100
            lo, hi = math.floor(index), math.ceil(index)
            threshold = distribution[lo] + (index - lo) * (distribution[hi] - distribution[lo])
        else:
            threshold = 0.0
        count, ra = self.ra(target, neighbour) if neighbour is not None else (None, None)
        percentile = 100 * bisect_right(distribution, ra) / len(distribution) if ra is not None and distribution else None
        return {"model": MODEL, "tool": "sc_graph_triangles", "firm_id": target,
                "cutoff": self.cutoff, "vintage": self.vintage, "triangle_count": triangles,
                "clustering": clustering, "neighbour": neighbour, "shared_neighbour_count": count,
                "pair_triangle_count": count if neighbour in neighbours else 0,
                "ra": ra, "ra_percentile": percentile, "THRESH_RA": thresh_ra,
                "ra_value_threshold": threshold, "specialized_shared_exposure": ra is not None and ra > 0 and ra >= threshold,
                "THRESH_CLUST": thresh_clust, "dense_cluster": clustering >= thresh_clust,
                "paths": paths, "max_hops": max_hops, "truncated": truncated}


def graph_fields(degree, triangles):
    """Allowlisted host-computed scalar fields for the selector."""
    return {**{k: degree[k] for k in ("in_degree", "out_degree", "total_degree", "degree_percentile", "hub")},
            **{k: triangles[k] for k in ("triangle_count", "clustering", "shared_neighbour_count", "pair_triangle_count",
                                         "ra", "ra_percentile", "specialized_shared_exposure", "dense_cluster")}}


def cli(kind):
    import argparse
    import json
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--firm-id", required=True)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--max-hops", type=int, default=1)
    p.add_argument("--inspected-prefixes", type=Path)
    p.add_argument("--thresh-hub", type=float, default=90)
    p.add_argument("--thresh-ra", type=float, default=75)
    p.add_argument("--thresh-clust", type=float, default=.5)
    p.add_argument("--neighbour")
    a = p.parse_args()
    graph = Snapshot(json.loads(a.snapshot.read_text(encoding="utf-8")), a.cutoff)
    prefixes = json.loads(a.inspected_prefixes.read_text()) if a.inspected_prefixes else []
    result = (graph.degree_result(a.firm_id, a.max_hops, prefixes, a.thresh_hub) if kind == "degree" else
              graph.triangle_result(a.firm_id, a.neighbour, a.max_hops, prefixes, a.thresh_ra, a.thresh_clust))
    print(json.dumps(result, allow_nan=False))
