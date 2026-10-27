"""Conservative finance and graph features for point-in-time context ranking.

These are *hypotheses for retrieval*, not exposure weights, causal effects, or
realized return signals. Only allowlisted categories leave this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from .graph_projection import role_motif, project_path


ROLES = {"supplier", "customer", "competitor"}
EVENT_STATES = {"pending", "ongoing", "resolved"}
CHANNEL_HINTS = {
    ("supplier", "supplier disruption"): "input_shock_hypothesis",
    ("customer", "customer demand"): "output_demand_hypothesis",
    ("competitor", "competitor substitution"): "substitution_hypothesis",
}


def _role_motif(pattern: tuple[str, ...]) -> str:
    """Classify the ordered path without assuming input or sales transmission."""
    return role_motif(pattern)


def _finance_hypothesis(motif: str, category: str) -> str:
    """Map aligned topology and event class to a *testable* risk hypothesis."""
    if motif == "shared_supplier_fork":
        return "common_input_possible_dated_supplier_disruption"
    if motif == "all_upstream" and category == "supplier disruption":
        return "upstream_input_risk"
    if motif == "all_downstream" and category == "customer demand":
        return "downstream_demand_risk"
    if motif == "direct_competitor" and category == "competitor substitution":
        return "competitive_substitution_risk"
    return "unaligned_or_ambiguous"


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid graph timestamp") from exc


def finance_graph_signal(row: dict, asof: str, *, event_category: str | None = None) -> dict:
    result = _finance_graph_signal(row, asof, event_category=event_category)
    weight = row.get("router_evidence_weight")
    if weight is not None and (isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1):
        raise ValueError("Invalid router evidence weight")
    result["router_evidence_weight"] = weight
    stats = row.get("graph_statistics")
    fields = ("in_degree", "out_degree", "total_degree", "degree_percentile", "hub",
              "triangle_count", "clustering", "shared_neighbour_count", "pair_triangle_count",
              "ra", "ra_percentile", "specialized_shared_exposure", "dense_cluster")
    result.update({k: None for k in fields})
    if stats is not None:
        import math
        if _utc(stats.get("cutoff")) != _utc(asof) or _utc(stats.get("vintage")) >= _utc(asof):
            raise ValueError("Invalid graph-statistics clock")
        for k in fields:
            value = stats.get(k)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid numerical graph statistic")
            if value is not None and k in {"clustering", "hub", "specialized_shared_exposure", "dense_cluster"} and value > 1:
                raise ValueError("Invalid bounded graph statistic")
            if value is not None and k in {"degree_percentile", "ra_percentile"} and value > 100:
                raise ValueError("Invalid graph percentile")
            result[k] = value
    return result


def _finance_graph_signal(row: dict, asof: str, *, event_category: str | None = None) -> dict:
    """Extract topology and finance hypotheses without disclosing graph IDs/text.

    `verified_paths` are host-joined retrieval paths. This function never
    promotes them to a verified product channel, even if a candidate contains
    a model-written `channel_receipt` or a FactSet revenue percentage.
    """
    cutoff = _utc(asof)
    state = str(row.get("status") or "").strip().lower()
    state = state if state in EVENT_STATES else "unknown"
    scope = row.get("_scope", "own_common")
    if scope not in {"own_common", "network_supplement"}:
        raise ValueError("Invalid context scope")
    if scope == "own_common":
        return {"scope": "own_common", "graph_role": "none", "hops": 0,
                "path_pattern": "none", "path_count_bin": "none",
                "channel_status": "not_applicable", "mechanism_hint": "own_or_common",
                "event_state": state, "role_motif": "not_applicable",
                "finance_hypothesis": "not_applicable", "evidence_gap": "not_applicable",
                "alternative_route": "not_applicable"}

    paths = row.get("verified_paths", [])
    if not isinstance(paths, list):
        raise ValueError("Invalid graph paths")
    typed, comention_count = [], 0
    for path in paths:
        if not isinstance(path, dict) or _utc(path.get("vintage")) >= cutoff:
            raise ValueError("Missing or post-cutoff graph vintage")
        if path.get("type") == "comention":
            if path.get("hops") != 1 or path.get("steps") != []:
                raise ValueError("Invalid co-mention lead")
            comention_count += 1
            continue
        if path.get("type") not in (None, "typed"):
            raise ValueError("Invalid graph path type")
        hops, steps = path.get("hops"), path.get("steps")
        if type(hops) is not int or hops not in (1, 2, 3) or not isinstance(steps, list) or len(steps) != hops:
            raise ValueError("Invalid graph path length")
        if any(not isinstance(step, dict) or step.get("neighbor_role") not in ROLES for step in steps):
            raise ValueError("Invalid graph role")
        project_path(path, asof)
        typed.append(tuple(step["neighbor_role"] for step in steps))

    if not typed:
        lead = bool(comention_count)
        return {"scope": "network_supplement", "graph_role": "co_mention" if lead else "unknown",
                "hops": 1 if lead else 0, "path_pattern": "co_mention" if lead else "none",
                "path_count_bin": "one" if comention_count == 1 else "multiple" if lead else "none",
                "channel_status": "retrieval_lead" if lead else "unlinked",
                "mechanism_hint": "retrieval_only" if lead else "no_graph_path",
                "event_state": state, "role_motif": "co_mention_only" if lead else "no_typed_path",
                "finance_hypothesis": "unaligned_or_ambiguous",
                "evidence_gap": "typed_edge_and_exposure" if lead else "no_typed_path",
                "alternative_route": "none_observed"}

    shortest = min(len(pattern) for pattern in typed)
    shortest_patterns = [pattern for pattern in typed if len(pattern) == shortest]
    unique_patterns = set(shortest_patterns)
    role = next(iter(unique_patterns))[0] if len(unique_patterns) == 1 else "mixed_or_unknown"
    pattern = ("mixed_shortest_paths" if len(unique_patterns) > 1 else
               "direct_" + role if shortest == 1 else "_to_".join(next(iter(unique_patterns))))
    category = event_category if event_category is not None else row.get("event_type")
    hint = ("ambiguous_path_hypothesis" if len(unique_patterns) > 1 else
            "intermediate_product_unverified" if shortest > 1 else
            CHANNEL_HINTS.get((role, category), "no_aligned_finance_channel"))
    motif = ("conflicting_shortest" if len(unique_patterns) > 1 else
             _role_motif(next(iter(unique_patterns))))
    hypothesis = _finance_hypothesis(motif, category)
    gap = ("path_and_product_ambiguity" if len(unique_patterns) > 1 else
           "intermediate_product_continuity" if shortest > 1 else
           "direct_product_exposure")
    if motif == "shared_supplier_fork":
        pattern = "shared_supplier_fork"
        role = "shared_supplier_peer"
        hint = "common_input_retrieval_hypothesis"
        gap = "supplier_event_and_focal_product_exposure"
    alternative = ("longer_typed_path_observed" if any(len(p) > shortest for p in typed) else
                   "different_shortest_motifs" if len(unique_patterns) > 1 else
                   "none_observed")
    return {"scope": "network_supplement", "graph_role": role, "hops": shortest,
            "path_pattern": pattern, "path_count_bin": "one" if len(shortest_patterns) == 1 else "multiple",
            "channel_status": "path_only", "mechanism_hint": hint, "event_state": state,
            "role_motif": motif, "finance_hypothesis": hypothesis,
            "news_binding": "peer_endpoint" if motif == "shared_supplier_fork" else "path_endpoint",
            "evidence_gap": gap, "alternative_route": alternative}
