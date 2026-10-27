"""Entity graph at the user-authorized June 6 snapshot; no return-label access."""
from collections import defaultdict, Counter, deque
import math
import pandas as pd

from news_agent_batched_runtime_2025 import write, read
from news_event_risk_data_2025 import provenance
from .config import FACTSET, SOURCE, GRAPH_DATE, FIRMS


def active(frame, start, end, when=GRAPH_DATE):
    a = pd.to_datetime(frame[start], errors="coerce")
    b = pd.to_datetime(frame[end], errors="coerce")
    at = pd.Timestamp(when) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    # Never merge disjoint intervals or take a maximum weight from future rows.
    return frame[a.notna() & (a <= at) & (b.isna() | (b > at))].copy()


def build_graph(out, universe):
    if (out / "graph.json").exists():
        return Graph(read(out / "graph.json"))
    sc_path, rel_path = FACTSET / "ent_scr_supply_chain.txt", FACTSET / "ent_scr_relationships.txt"
    mapping_path = SOURCE / "processed/factset_to_permno.csv"
    mapping = active(pd.read_csv(mapping_path), "link_bdate", "link_edate")
    names_path = FACTSET / "mapping.csv"
    names = pd.read_csv(names_path, usecols=["FACTSET_ENTITY_ID", "ENTITY_PROPER_NAME"], low_memory=False)
    name_map = names.dropna().drop_duplicates("FACTSET_ENTITY_ID").set_index("FACTSET_ENTITY_ID").ENTITY_PROPER_NAME.to_dict()
    nodes, targets, edges = {}, {}, []
    for row in mapping.itertuples():
        node = nodes.setdefault(row.FACTSET_ENTITY_ID, {"name": name_map.get(row.FACTSET_ENTITY_ID), "permnos": []})
        node["permnos"].append(int(row.PERMNO))
    for firm in universe.itertuples():
        found = sorted(mapping.loc[mapping.PERMNO.eq(firm.permno), "FACTSET_ENTITY_ID"].unique())
        if not found:
            raise ValueError("No dated FactSet mapping for " + firm.ticker)
        targets[firm.ticker] = found
        for entity in found:
            nodes[entity]["name"] = firm.company
    sc = pd.read_csv(sc_path, sep="|", low_memory=False)
    sc_max = str(pd.to_datetime(sc.START_DATE, errors="coerce").max())
    for row in active(sc, "START_DATE", "END_DATE").itertuples():
        edges.append({"id": str(row.ID), "source": row.SUPPLIER_FACTSET_ENTITY_ID,
                      "target": row.CUSTOMER_FACTSET_ENTITY_ID, "type": "supplies",
                      "valid_from": str(row.START_DATE), "vintage": GRAPH_DATE,
                      "revenue_pct": None if pd.isna(row.REVENUE_PCT) else float(row.REVENUE_PCT)})
    rel = pd.read_csv(rel_path, sep="|", low_memory=False)
    rel_max = str(pd.to_datetime(rel.START_DATE, errors="coerce").max())
    for row in active(rel[rel.REL_TYPE.eq("COMPETITOR")], "START_DATE", "END_DATE").itertuples():
        edges.append({"id": str(row.ID), "source": row.SOURCE_FACTSET_ENTITY_ID,
                      "target": row.TARGET_FACTSET_ENTITY_ID, "type": "competitor",
                      "valid_from": str(row.START_DATE), "vintage": GRAPH_DATE, "revenue_pct": None})
    for e in edges:
        for entity in (e["source"], e["target"]):
            nodes.setdefault(entity, {"name": name_map.get(entity), "permnos": []})
    for n in nodes.values():
        n["permnos"] = sorted(set(n["permnos"]))
    value = {"nodes": nodes, "edges": edges, "targets": targets, "comention": {},
             "audit": {"graph_date": GRAPH_DATE, "held_constant_through": "2025-12-31",
                       "point_in_time_certified": False,
                       "assumption": "Effective-date reconstruction of current raw files, held fixed by user assumption; archived acquisition vintage unavailable. Names are identity aids, not dated business profiles.",
                       "sources": [provenance(p) for p in (sc_path, rel_path, mapping_path, names_path)],
                       "latest_raw_start": {"supply": sc_max, "relationships": rel_max},
                       "nodes_without_stock_mapping": sum(not n["permnos"] for n in nodes.values()),
                       "no_future_revenue_aggregation": True}}
    write(out / "graph.json", value)
    return Graph(value)


def freeze_comentions(graph, july_rows, out):
    """One pre-development month, one count per resolved story; no future ranking."""
    counts, degree = Counter(), Counter()
    for row in july_rows:
        ids = sorted(set(row["ids"]))
        if not row["usable"] or not 2 <= len(ids) <= 20:
            continue
        for p in ids:
            degree[p] += 1
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                counts[a, b] += 1
    values = defaultdict(list)
    for (a, b), n in counts.items():
        weight = n / math.sqrt(degree[a]*degree[b])
        for u, v in ((a, b), (b, a)):
            values[str(u)].append({"permno": v, "count": n, "weight": weight})
    frozen = {k: sorted(v, key=lambda r: (-r["weight"], -r["count"], r["permno"]))[:20] for k, v in values.items()}
    if graph.data.get("comention") == frozen and graph.data.get("audit", {}).get("comention"):
        return
    graph.data["comention"] = frozen
    graph.data["audit"]["comention"] = {"window": "2025-07-01 to 2025-07-31", "frozen_before": "2025-08-01",
                                           "normalization": "pair stories / sqrt(entity story counts)",
                                           "max_tags_per_story": 20, "interpretation": "retrieval lead only"}
    write(out / "graph.json", graph.data)


class Graph:
    def __init__(self, data):
        self.data = data
        self.adj = defaultdict(list)
        for e in data["edges"]:
            for u, v, role in ((e["source"], e["target"], "customer"), (e["target"], e["source"], "supplier")):
                self.adj[u].append((v, "competitor" if e["type"] == "competitor" else role, e))
        for items in self.adj.values():
            items.sort(key=lambda x: (x[1], x[0], x[2]["id"]))

    def pool_paths(self, ticker, hops=2, limit=1500):
        """Outcome-free offline enumeration; runtime tools still require inspected via."""
        return self.paths(ticker, hops, limit=limit, _pool=True)

    def paths(self, ticker, hops=1, via=None, limit=1500, *, _pool=False):
        if hops not in (1, 2, 3) or (hops == 3 and not (via or _pool)):
            raise ValueError("Three-hop search requires a specific inspected intermediate")
        roots = self.data["targets"][ticker]
        def expand(node, chain, steps, role, edges):
            seen = set()
            for neighbor, _, edge in edges:
                if neighbor in chain or neighbor in roots or neighbor in seen:
                    continue
                seen.add(neighbor)
                new_chain = chain + [neighbor]
                new_steps = steps + [{"from": node, "to": neighbor, "neighbor_role": role,
                                      "edge_id": edge["id"], "revenue_pct": edge["revenue_pct"],
                                      **{k: edge[k] for k in ("source", "target", "type", "vintage", "valid_from", "valid_to") if k in edge}}]
                yield (neighbor, new_chain, new_steps)

        queue, result = [(r, [r], []) for r in roots], []
        for depth in range(1, hops+1):
            groups = defaultdict(deque)
            for node, chain, steps in queue:
                by_role = defaultdict(list)
                for entry in self.adj[node]:
                    by_role[entry[1]].append(entry)
                for role, edges in by_role.items():
                    # Within a relation, recorded materiality precedes identity order.
                    edges.sort(key=lambda x: (-(x[2]["revenue_pct"] or 0), x[0], x[2]["id"]))
                    pattern = tuple(s["neighbor_role"] for s in steps) + (role,)
                    groups[pattern].append(expand(node, chain, steps, role, edges))
            following = []
            patterns = deque(sorted(groups))
            # All direct links survive. Higher hops share capacity across path types
            # and intermediaries, not the first alphabetical branch alone.
            while patterns and (depth == 1 or len(following) < limit):
                pattern = patterns.popleft()
                branches = groups[pattern]
                found = None
                while branches and found is None:
                    branch = branches.popleft()
                    found = next(branch, None)
                    if found is not None:
                        branches.append(branch)
                if branches:
                    patterns.append(pattern)
                if found is None:
                    continue
                neighbor, chain, steps = found
                if via and depth > 1 and via not in chain:
                    continue
                following.append(found)
                if not via or via in chain:
                    from news_network_50_2025.graph_projection import role_motif
                    result.append({"entity": neighbor, **self.data["nodes"][neighbor], "hops": len(steps),
                                   "path": chain, "steps": steps, "degree": len(self.adj[neighbor]),
                                   "role_motif": role_motif(s["neighbor_role"] for s in steps),
                                   "vintage": GRAPH_DATE, "mechanism_verified": False})
            queue = following
        return result
