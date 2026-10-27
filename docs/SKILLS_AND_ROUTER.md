# Numerical gate skills and pre-model graph routing

This extension adds deterministic screening adapters and graph-tool routing.
Every implementation file is new. No active Qwen/DeepSeek ablation module,
toolkit, workflow, selector or model transport is modified. All examples are
invented sufficient statistics and toy graphs. No private data, news text,
credentials, fitted research outputs or remote calls are needed.

## Invocation contract

Use `/skill-name` (or `$skill-name` in Codex). Each SKILL.md states its task,
inputs, script, numerical constraints, required output, verification and stop
condition before execution. Skills have explicit invocation policies.

| Skill | One repeatable job | Default screen |
|---|---|---|
| `/sc-gate-validation` | Compare held-out candidate/incumbent SSE | Positive gain >1 week-clustered SE |
| `/sc-gate-sign-stability` | Test declared direction, then training-year stability | Right odd-week sign in >=4/7 calendar years |
| `/sc-gate-industry-peer` | Compare aligned graph/peer SSE | Graph strictly beats indpeer |
| `/sc-gate-rewire-placebo` | Generate seeded graphs or evaluate real/null gains | 100 draws; p <0.05 |
| `/sc-acceptance-ledger` | Run gates, stop on failure, append one decision | a -> c -> d -> e -> b |
| `/sc-graph-router` | Execute ordered graph tools before model use | Degree -> conditional motifs -> guarded paths |

Standalone numerical scripts live within each gate skill and use the standard
library. CSV is dependency-free; Parquet optionally uses pandas/pyarrow. The
ledger composes the four sibling skill scripts. The router entrypoint uses
the new `news_network_50_2025.graph_router` and existing `graph_skills` modules.
None import `sc_risk` or use paths to private source/data at runtime.

## Gate policies and provenance

Implementation semantics were inspected in the local reference files
`C:/OptResearch/code/sc_risk/llm_mech_agent.py` (`heldout_gain`,
`HostEngine.evaluate`), `llm_mech_loop.py` (`validation`, `supported`,
`industry_benchmark`, `placebo_test`), and `o2_acceptance_gate.py` (`score`,
`null_scores`, `report`), together with section 8.3 of the project guideline.
They implement different policies; choose a profile before examining outcomes.

| Rule | `agent-v4` (default) | `loop` | `o2` |
|---|---|---|---|
| Fit/validation ISO weeks | even -> odd | both opposite-parity directions | even -> odd |
| Training coefficient sample | odd weeks | pooled | pooled |
| Stability, complete 2015-2021 calendar years | >=4/7 right sign | >=4/7 right sign AND signed day/week t >1.96 | >=5/7 right sign |
| Peer rule | graph gain > indpeer gain | graph-vs-peer gain >1 week SE | graph beats BOTH FF49 and NAICS4 |
| Null statistic | MSE gain | MSE gain | relative SSE gain (%) |
| Placebo p, ties included within 1e-12 statistic units | (1+exceedances)/101 | (1+exceedances)/101 | exceedances/100 |
| Positive gain with zero cluster SE | passes | fails finite-t requirement | fails finite-t requirement |

The guideline describes a different calendar (4/6 sign-stable years), a
week-block bootstrap SE and lists gates a,b,c,d,e. These adapters reproduce
the implemented ISO-week cluster-score formula, not a bootstrap. The current
host executes a,c,d,e,b to avoid expensive rewires after a cheaper failure.
The ledger follows that actual order for all named profiles; O2's original
report computes all criteria rather than short-circuiting. Missing later
checks in the ledger mean **not run**, never pass.

Two-sided declarations are supported by v4: declare the ambiguity before
testing, use a nonzero pooled odd-week estimate's direction for stability,
and test the upper tail of direction-free predictive gain. Do not double p
or take absolute validation gains. The user's October 26 decision sets the
default two-sided threshold to **0.05**, overriding guideline 8.3's **0.025**.
Use `--threshold 0.025` (ledger `placebo_threshold:0.025`) to choose that stricter
screen. Thresholds are strict and may not be increased above 0.05. One-sided
tests also default to 0.05. Two-sided and non-100-draw receipts are exploratory.
The legacy/O2 profiles do not invent two-sided semantics absent from their code.

These adapters verify numerical screens on host-supplied sufficient statistics.
They do not refit private models or certify production eligibility. The host
must attest fixed incumbent/feature identity, upstream fit provenance, source
availability, preregistration, complete monthly calendar, intact matched sets,
sample selection and common controls. Upstream v4 also requires all 84 months
and its authorized sample fraction; this generic ledger is labelled
`screening_only:true`, and does not promote/freeze a production model.
Removal's separate nonworsening-only exception is deliberately outside this
addition/replacement contract. No post-hoc sign reversal is silently accepted.

## Generic loss and coefficient inputs

`validation.csv` or Parquet has one record per disjoint held-out block:

```csv
row_id,date,n,sse_incumbent,sse_candidate,fit_parity
toy-1,2015-01-15,10,10,5,0
toy-2,2015-01-29,10,10,6,0
```

`date` is the calendar observation date. The program derives ISO year/week
for clustering and parity, and never pools equal week numbers across years.
Use unique row IDs, positive integer counts, finite nonnegative SSEs and
calendar dates in 2015-2021. Opposite `fit_parity` is mandatory. For predictions,
sum `(outcome-prediction)**2` for each model over the identical row IDs and
use the row count as n. Upstream cross-products can produce the same SSEs as
`y'y - 2*b'X'y + b'X'X*b`. Aggregate sign-specific SSEs and counts together for
the loss screen; coefficient summaries must retain the project's balanced
up/down slope definition. Do not average group MSEs with unequal group sizes.

For total N and per-week loss difference D_w/count n_w:

```
gain = sum_w(D_w) / N
week_se = sqrt(sum_w((D_w - gain*n_w)**2)) / N
pass = gain > 0 and gain > se_multiplier*week_se
```

Default `se_multiplier=1`. Require at least two week clusters. Loop requires
both validation parities and a finite t (nonzero SE). The scripts reject
NaN/infinity, booleans as numeric values, negative SSEs and duplicate block IDs.
Aggregated tables cannot prove row-level sample identity; the host is
responsible for constructing disjoint blocks from the same outcomes.

Sign JSON contains `declared_at`, `evaluated_at` (timezone-aware chronological
timestamps), `predicted_sign` (`+`/`-`), `estimate_split` (`odd`/`pooled`),
`pooled:{estimate}`, and `years:{"2015":{estimate},...,"2021":{estimate}}`.
Loop yearly statistics additionally require `t_day` and `t_week`.
v4 optionally accepts `two_sided:true` and `ambiguity_rationale`; it rejects
two-sided declarations containing `post_hoc_of`. A timestamp supplied today
cannot by itself prove historical preregistration; retain a trusted prior hash.
`--required-years` and `--t-cutoff` expose alternate declared screens.

Peer CSV/Parquet contains `row_id,date,n,fit_parity,sse_graph,sse_indpeer`;
O2 substitutes `sse_ff49,sse_naics4`. Losses must share the fixed incumbent,
outcomes and BOTH graph/peer missingness controls. Lower graph SSE is then
equivalent to larger incremental gain. The separate benchmark JSON is:

```json
{"operator":"neighbour mean","industries":["indpeer"],"size_factor":2,
 "same_operator":true,"exclude_self_and_supply_links":true,
 "shared_missingness_controls":true,"point_in_time":true}
```

Choose same-industry nonlinked peers within a factor of two in lagged size,
nearest log-size distance, deterministic ID ties, up to the original degree.
Exclude self and direct supply links, use symmetric union for undirected
operators and keep peer shortages. Apply the identical operator/normalization.
These are host attestations, not properties inferable from SSE alone.

## Rewire contract

Graph JSON maps IDs to industry labels: `nodes:{"A":"industry1",...}`,
`directed:true|false`, `edges:[{source,target,period:"2015-01",...}]`.
Preserve all extra edge-slot attributes. The generator attempts 20 sweeps
per month/industry-pair block with default seed 20261026. It preserves node
in/out and industry-specific degrees for directed graphs, or undirected
degree/industry mixing, rejecting self loops and duplicates. Frozen blocks
are retained and report zero movement. A failed mixing diagnostic is not
fixed by deleting edges. Pure Python seeds are reproducible in this adapter;
they do not reproduce upstream Numba RNG streams byte for byte.

For each generated graph the host recomputes the same network feature,
refits on the same even/odd training folds and supplies held-out loss rows.
Bundle JSON is:

```text
graph: original graph
operator: fixed operator description
same_operator_refit: true
observed_rows: validation-schema rows for the real graph
draws: [{seed, sweeps:20, graph:generated graph, rows:validation-schema rows}, ...]
ambiguity_rationale: required only for a declared two-sided test
```

Evaluation reproduces every seeded draw and checks identical row IDs, dates,
counts, fit parity and incumbent SSEs between observed/null tables. It
checks per-month industry-specific degrees and slot attributes. It cannot
infer whether supplied candidate SSEs came from the claimed operator/refit.
Source implementations use a common 10% intact-event placebo sample: O2
samples within year/month/announcer FF49; the loop additionally uses event
date/relation/sign sampling strata. Keep these event strata fixed upstream.
Graph switches are stratified by month and both endpoint industries here.
Use the real **sampled** score, not a full-sample score, with sampled nulls.

## Reproduce the synthetic examples

PowerShell, repository root, no network:

```powershell
$py = 'C:/OptResearch/.venv_dspy/Scripts/python.exe'
$env:PYTHONPATH = 'source/code'
$env:PYTHONDONTWRITEBYTECODE = '1'
$demo = 'C:/OptResearch/results/pr2_gate_demo_new' # choose a fresh output folder
& $py examples/gate_skills_synthetic.py --out $demo
& $py skills/sc-gate-validation/scripts/gate_validation.py --input "$demo/validation.csv"
& $py skills/sc-gate-sign-stability/scripts/gate_sign_stability.py --input "$demo/sign.json"
& $py skills/sc-gate-industry-peer/scripts/gate_industry_peer.py --input "$demo/peers.csv" --contract "$demo/peer_contract.json"
& $py skills/sc-gate-rewire-placebo/scripts/gate_rewire_placebo.py evaluate --input "$demo/placebo.json" --threshold 0.05 --draws 100
& $py skills/sc-acceptance-ledger/scripts/acceptance_ledger.py --config "$demo/acceptance.json" --ledger "$demo/decisions.jsonl"
& $py skills/sc-graph-router/scripts/graph_router.py --snapshot "$demo/snapshot.json" --case "$demo/router_case.json"
```

The gate fixture passes v4 with gain 0.5 and p=1/101. Its SSEs are invented
arithmetic fixtures, not output from fitting on the toy graph. Known failures,
unequal losses, strict thresholds, policy differences and corrupt receipts are
covered in `examples/test_gate_skills.py`. Scripts exit 0=pass, 1=failed screen,
2=invalid input/error. Generated files stay outside the repository.

The ledger uses canonical sorted JSON, SHA256, a zero genesis hash, sequential
indices and an exclusive sidecar lock. It appends/flushes/fsyncs exactly one
ACCEPT/REJECT/ERROR row per unique attempt ID and retains all earlier bytes.
It never repairs a corrupt/partial chain or removes a foreign lock. On a
failed gate later files are not opened and later checks remain null. Save
the returned head separately and pass `--expected-head` to detect suffix
truncation/rollback; an unanchored hash chain cannot detect that attack.

## Router schema and rules

`route_case(snapshot, case, RouterRules(...))` executes the numerical tools
and returns a receipt before any model invocation. Case schema:

```json
{"case_id":"toy-router","firm_id":"A","cutoff":"2020-02-01",
 "candidate_neighbours":["B","C"],"inspections":[]}
```

Snapshot schema is the existing degree/triangle schema: nodes, dated directed
edges, vintage, and a monthly universe. Candidates must be unique active
direct neighbours in the undirected projection. Supplier/competitor edges
follow the existing numerical tools' semantics. No news or labels are read.

| RouterRules field | Default | Effect |
|---|---:|---|
| min_candidate_neighbours | 2 | Run per-candidate triangles/RA at or above this count; hard minimum 2 |
| thresh_hub | 90 | Target monthly degree percentile at/above this is a hub |
| single_neighbour_weight | 0.5 | Evidence weight for a hub case with exactly one candidate neighbour |
| thresh_ra | 75 | Existing monthly RA reference percentile |
| thresh_clust | 0.5 | Existing dense-cluster flag threshold |
| max_hops | 3 | Hard cap 3; 2/3-hop expansion requires exact inspected prefixes |
| path_limit | 200 | Cumulative displayed-path budget per depth; hard maximum 2000 |

The 90/75/0.5 topology thresholds come from existing graph skills. The routing
trigger (2), hub evidence weight (0.5) and path budget (200) are declared
engineering defaults, not fitted economic coefficients. Serialize a subset
of these fields to `--rules rules.json`. GEPA or prompt_gd can later propose
this small parameter set using training data, with validation/holdout rules
and hard hop/prefix/date guards retained. No optimizer is connected here.

Order: degree first; if eligible, triangles/RA in sorted candidate-ID order;
then paths of increasing allowed depth. If no inspected one-hop prefix
exists, skip two hops; if no inspected two-hop prefix exists, skip three.
An inspected two-hop prefix also requires its exact one-hop prefix receipt.
Each inspection has `path`, `case_id`, `cutoff`, `snapshot_sha256`,
`inspected:true`, `source_ids:[...]`, and `available_at < cutoff`.
The host creates inspection receipts; routing never converts a newly
enumerated path into an inspected path. Receipts cannot transfer between
cases/snapshots/cutoffs. Cycles, invalid dates and unknown endpoints fail.
Edge/snapshot vintage and valid_from must strictly precede cutoff; valid_to
at/before cutoff expires the edge. The supplied monthly universe controls
percentiles. Reinvoke the router after new host inspections to expand.

The receipt records ordered calls/results, skipped calls/reasons, all rules,
weight, candidate count, exact inspection receipts, graph/case/code hashes,
paths, truncation and model_calls=0. `verify_receipt` checks its hash;
`route_case` replay checks computation. Hashes provide integrity, not proof
that a host really inspected a source. A truncated result means incomplete
coverage. Graph paths and RA do not establish a product channel.

Future integration belongs immediately before the selector/model call:
construct the case from host candidates, call route_case, persist its receipt,
and pass returned statistics/weights to selection. Add an explicit flag such
as `enable_graph_router=False` before hooking toolkit/context_selection/
workflow/domain_signals. Do that only after
`D:/OptResearch_data/agent_q1/hop3_run/QWEN_RUN_DONE_v2.json` exists. No hook or
change to an imported ablation module is included in this extension.

## Offline checks and manifest review

Run every `examples/test*.py` suite in its own `python -m unittest` process,
with `PYTHONPATH=source/code` and the specified venv. Set
`Q1_SEAL_DATE=2025-11-03` only for test_q1_identify, test_q1_identify_v2 and
test_q1_design; remove it for test_q1_graph_prompt_gd and other suites.
Also run `python -m unittest news_network_50_2025.test_design` and
`python -m unittest discover -s skills/supply-chain-news-risk/scripts -p test_validate_dictionary.py`.
The six new skills also pass skill-creator's quick_validate.py.

Review run: **194/194 tests pass** across 14 isolated example/design suites
(25 new, 169 existing), plus **3/3 validator unit tests**. A combined-process
run exposed existing DeepSeek/Qwen global-model interference (two failures);
both transport suites pass in isolation. No in-use file was edited to hide it.

The unchanged PR #1 forecasting validator reports **BLOCKED** in the current
working tree: manifest mismatches in docs/Q1_GRAPH_SKILLS_AND_PROMPT_GD.md and
source/code/news_network_50_2025/{data,prompt_gd,runtime}.py. Those mismatches
predate this extension. New files also need manifest entries: the current
validator checks listed hashes but does not enumerate unlisted additions.
Neither the validator nor historical source hashes are weakened/refreshed here.

The exact recommended commit allowlist is
`docs/PR2_EXTENSION_COMMIT_FILES.txt`: 25 new files and one operator-refreshed
manifest. To prepare a reviewable patch and clean candidate snapshot:

```powershell
& $py examples/prepare_review_manifest.py --files docs/PR2_EXTENSION_COMMIT_FILES.txt --out C:/OptResearch/results/pr2_gate_router_review/final/file_manifest.review.json --patch C:/OptResearch/results/pr2_gate_router_review/final/manifest_refresh.patch --snapshot C:/OptResearch/results/pr2_gate_router_review/final/candidate_snapshot
```

Use fresh output names if these task review artifacts already exist. This
preserves clean tracked files' checkout bytes, uses HEAD with checkout filters
for excluded dirty files, and adds every selected new file from the explicit
allowlist. It excludes unrelated dirty ablation changes. Manifest hashes are
byte-specific: line-ending conversion on another checkout requires review and
regeneration there. Inspect the exact manifest diff, then the operator can run:

```powershell
git apply --check C:/OptResearch/results/pr2_gate_router_review/final/manifest_refresh.patch
git apply C:/OptResearch/results/pr2_gate_router_review/final/manifest_refresh.patch
& $py skills/supply-chain-news-risk/scripts/validate_dictionary.py --repo . --mode forecasting
```

Current uncommitted data.py/runtime.py remain excluded from this extension;
their working-tree hashes will still differ after applying a manifest for
the proposed commit tree. Validate the clean candidate snapshot separately,
or validate the intended commit in an isolated checkout. Do not certify those
active changes merely to turn the current working-tree check green.
The prepared snapshot validates as **QUALIFIED** with no package mismatches;
existing warnings about historical source differences, rank deficiency and
unrecomputed counts/GARCH remain.

Do not commit the pre-existing .bak file, untracked hop3/product_continuity
modules/tests, dirty data.py/runtime.py, caches, generated ledger/draws,
review snapshots/logs/patches, licensed text or research data/results. The
external review directory records the complete file inventory and scan.
