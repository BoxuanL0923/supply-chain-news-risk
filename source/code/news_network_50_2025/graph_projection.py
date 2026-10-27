"""Typed retrieval topology and endpoint binding; no exposure weights or labels."""
from __future__ import annotations

from .graph_skills import utc


def indexed_snapshot(snapshot):
    index = {}
    for edge in snapshot["edges"]:
        index.setdefault(str(edge.get("id")), []).append(edge)
    return {"_edge_index": index}


def role_motif(roles):
    roles = tuple(roles)
    if roles == ("supplier", "customer"):
        return "shared_supplier_fork"
    if roles and all(r == "supplier" for r in roles):
        return "all_upstream"
    if roles and all(r == "customer" for r in roles):
        return "all_downstream"
    if roles == ("competitor",):
        return "direct_competitor"
    return "competitor_mixed" if "competitor" in roles else "direction_change"


def project_path(path, cutoff, snapshot=None):
    """Check host paths, including the two *outgoing* edges of i <- s -> j.

    Legacy role-only paths can be classified but cannot certify edge clocks.
    The enabled runner supplies a snapshot and requires exact edge bindings.
    """
    if not path.get("vintage") or utc(path["vintage"]) >= utc(cutoff):
        raise ValueError("Graph path vintage must precede cutoff")
    if path.get("type") == "comention":
        if path.get("hops") != 1 or path.get("steps") != []:
            raise ValueError("Invalid co-mention lead")
        return dict(path, role_motif="co_mention_only")
    steps, chain, hops = path.get("steps", []), path.get("path", []), path.get("hops")
    if type(hops) is not int or hops not in (1, 2, 3) or len(steps) != hops:
        raise ValueError("Invalid graph path length")
    roles = [s.get("neighbor_role") for s in steps]
    if any(r not in {"supplier", "customer", "competitor"} for r in roles):
        raise ValueError("Invalid graph role")
    if snapshot is not None or chain:
        if len(chain) != hops + 1 or len(set(chain)) != len(chain) or path.get("entity", chain[-1]) != chain[-1]:
            raise ValueError("Invalid path endpoint")
    for n, step in enumerate(steps):
        edge = step
        if snapshot is not None:
            edges = (snapshot["_edge_index"].get(str(step.get("edge_id")), [])
                     if "_edge_index" in snapshot else snapshot["edges"])
            matches = [e for e in edges if str(e.get("id")) == str(step.get("edge_id"))
                       and {e["source"], e["target"]} == set(chain[n:n+2])]
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous edge binding")
            edge = matches[0]
        if "source" in edge or snapshot is not None:
            a, b = chain[n:n+2]
            if {edge.get("source"), edge.get("target")} != {a, b}:
                raise ValueError("Edge endpoints disagree with path")
            expected = ("competitor" if edge.get("type", edge.get("edge_type")) == "competitor"
                        else "customer" if edge["source"] == a else "supplier")
            if roles[n] != expected:
                raise ValueError("Supply edge direction disagrees with path")
            for clock in ("vintage", "valid_from"):
                if not edge.get(clock) or utc(edge[clock]) >= utc(cutoff):
                    raise ValueError("Edge clock must precede cutoff")
            if edge.get("valid_to") and utc(edge["valid_to"]) <= utc(cutoff):
                raise ValueError("Expired edge")
    motif = role_motif(roles)
    value = dict(path, role_motif=motif)
    if motif == "shared_supplier_fork" and chain:
        value["fork"] = dict(focal=chain[0], supplier=chain[1], peer=chain[2],
                             input_edges=[[chain[1], chain[0]], [chain[1], chain[2]]])
    return value


def bound_paths(relations, document, cutoff, snapshot=None):
    """News tags bind to the path endpoint, never to an intermediate supplier.

    Tags establish identity for retrieval only; they do not prove that the
    article reports an event at that entity or its affected product.
    """
    if utc(document["available_at"]) >= utc(cutoff):
        raise ValueError("News must be available strictly before cutoff")
    paths = []
    for path in relations:
        if set(path["permnos"]) & set(document["ids"]):
            projected = project_path(path, cutoff, snapshot)
            projected["news_binding"] = ("peer_endpoint" if projected["role_motif"] == "shared_supplier_fork"
                                         else "path_endpoint")
            paths.append(projected)
    return paths
