---
name: sc-graph-router
description: Route dated degree, triangle/RA and inspected-prefix path tools deterministically for one case before any model call.
---

Invoke explicitly:
`/sc-graph-router route graph tools for one case; inputs: dated snapshot with monthly_universe, case ID, target, cutoff, candidate_neighbours and inspection receipts; script: skills/sc-graph-router/scripts/graph_router.py; constraints: MIN_CANDIDATE_NEIGHBOURS=2, THRESH_HUB=90, hub single-neighbour weight=0.5, THRESH_RA=75, THRESH_CLUST=0.5, MAX_HOPS=3, PATH_LIMIT=200; deliverable: ordered deterministic routing receipt; verify cutoff, exact prefixes and hash; stop after receipt or coverage/error.`

Before execution state the required output, verification, and stop above.
Run the repository-local script, sharing the new router module and existing
numerical `graph_skills.py`; no model or running ablation is imported:
`python skills/sc-graph-router/scripts/graph_router.py --snapshot snapshot.json --case case.json --rules rules.json`
Omit `--rules` for the documented defaults. Rules JSON is a subset of
`RouterRules` fields in [the schema](../../docs/SKILLS_AND_ROUTER.md).

Always run `/sc-graph-degree` first. Run `/sc-graph-triangles` (including pair RA)
for each candidate only if at least two distinct active candidate neighbours
exist. Enumerate 1-hop paths; enumerate 2 hops only with an exact inspected
1-hop prefix, and 3 hops only with inspected exact 2-hop AND 1-hop prefixes.
Never exceed 3, including after tuning. Exclude repeated-node paths. At target
degree percentile >=90, a case with only one candidate neighbour receives
weight 0.5. The numerical weight is a new routing default, not a fitted effect.

The host supplies dated inspection receipts bound to the same case, snapshot
hash and cutoff, with `inspected:true`, nonempty `source_ids`, source
`available_at < cutoff`, and target-rooted `path`. Routing never marks its own
new paths inspected. Reinvoke after host inspection for further expansion.
Graph/edge vintage and valid_from must be strictly before cutoff; expired
edges are excluded. Use the supplied monthly universe, not retrieved firms.

Required output: ordered calls and skips/reasons, thresholds, weights, paths,
truncation, bound inspection receipts, input/code hashes, receipt SHA256 and
`model_calls:0`. Verify with `verify_receipt`; recompute with `route_case` for
full replay. Stop at path budget and mark incomplete coverage. A receipt hash
is tamper evidence, not proof of source inspection or product continuity.
Runner/Toolkit integrate this contract behind `enable_graph_router` (default
off for legacy snapshot compatibility). Persist Skill/Python traces and require
the source-bound prefix receipts before expanding. See
`docs/PR5_IMPLEMENTATION.md`; the frozen Qwen run remains untouched.
