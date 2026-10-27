# Shared suppliers, runner receipts and cached hop reporting

PR5 implements the code requests in “Research plan after the two merged implementation PRs” in the collaborator's October 26 report. This is an offline engineering change, not an empirical acceptance or a new financial claim. The completed Qwen cache, its frozen prompts/predictions and the separate incomplete DeepSeek run were not modified. No network request, model call, process termination or Git write was performed.

## Shared-supplier retrieval

`graph_projection.py` and the original `Graph.paths` projection classify `i <- s -> j` as `shared_supplier_fork`. Both input edges originate at s. The selector receives `path_pattern=shared_supplier_fork`, `finance_hypothesis=common_input_possible_dated_supplier_disruption` and `evidence_gap=supplier_event_and_focal_product_exposure`. This means **common input / possible dated disruption at s**, a retrieval hypothesis.

The host binds a source's mapped identifiers to the path endpoint. A j source is `peer_endpoint`; it is never rebound to intermediate s because of an event label, free text or a model-written receipt. Supplier news is bound to the direct s endpoint. Identity tags are retrieval metadata, not proof of the article's event subject. Separate inspected supplier-event, affected-product and focal-exposure facts remain necessary; no new code promotes a fork above `path_only`. A serial product-continuity receipt also cannot certify a fork.

Enabled routing validates each edge ID, endpoints, direction, vintage, effective date and expiry against the supplied graph. Path and source clocks are strictly before cutoff. The selector enforces news availability and counts a resolved story once across source IDs; the evidence validator also collapses repeated events/pages from one story. `path_count_bin` and `alternative_route` remain descriptions of topology, with no procurement-dependence, substitution or redundancy interpretation. Tests vary path multiplicity without changing the hypothesis or channel status.

## Gate 4: actual runner and training hooks

Use `Runner(..., enable_graph_router=True)` or the runner CLI's `--enable-graph-router`. Direct Runner/Toolkit callers can use `SC_ENABLE_GRAPH_ROUTER=1`; explicit constructor/capability flags take precedence. **The default is OFF.** Compatibility checking with the flag forced on exposes legacy graph fixtures/capabilities without snapshots or dated edge metadata. The enabled contract fails closed on these inputs; silently inventing vintages would invalidate the receipts. Default-off legacy retrieval remains compatible. The selector's dated finance projection remains strict in either mode.

`Runner.capability/evidence` calls the deterministic router before native retrieval. Toolkit calls it again after actual inspections unlock exact prefixes. Its bounded paths constrain search and article access, and the host independently recomputes the final route before accepting evidence. Model-written graph statistics and weights are discarded. A finite, allowlisted `router_evidence_weight` reaches selector metadata and the scoring packet; it is a retrieval policy, not an estimated exposure coefficient.

Each enabled prediction persists a `case_traces/*.json` trace and embeds it in the prediction. Events contain Skill reads with byte hashes and Python entrypoints, numerical graph calls, router decisions/skips and receipt hashes, native tool calls/results, the actual host `select_context` call/receipt and acceptance status. An optional DSPy program still performs its existing required Skill/tool episode; the host rechecks its selection. Without a program, the host reads the Skill and uses stable evidence order through the same Python tool, including empty pools. This fallback is not an optimized DSPy ranking. Duplicate/future-source exclusions appear in the selection receipt. Retrieval and prediction cache keys include the enabled capability/trace so legacy cache entries cannot substitute for receipts.

The GEPA selection path calls `choose_with_gates` before replacing its seed. A heuristic loss win requires `Runner.accept_candidate`, which invokes the **existing** acceptance-ledger and four gate scripts. Missing inputs return `NOT_RUN`, not acceptance. Rejected/error receipts retain the seed. The wrapper binds policy hash, development case IDs, input byte hashes, script hashes and the ledger row hash; changing inputs cannot reuse a passing row. Inputs are frozen at preflight and Runner initialization. A legacy optimization cache without gate-4 provenance is refused.

On resume, a cached selected policy rechecks the gate-input/ledger binding and restores its acceptance receipts into the fresh Runner before scoring. The cached decision cannot substitute for those checks.

Supply `--acceptance-bundles` as a JSON object mapping policy hashes to local bundle filenames. Each bundle has exactly `policy_sha256`, `case_ids` and `config` (the existing ledger configuration). Referenced aggregate files must be regular local CSV/JSON files. The adapters retain their original 2015–2021 alternate-week profiles and fitting/construction attestations. They do not refit 2025 Q1 data or automatically supply the missing peer/rewire comparisons. No gate acceptance is asserted for the completed Qwen run. Fixed/seed and provisional training predictions explicitly record gates as not run.

## Label-access verification

`label_firewall.static_verify` checks executable label access and dynamic code in the router/retrieval/selection boundary, the context optimizer's early train/valid guard, the optimizer's literal development-loader splits, and the enabled runner's return before final evaluation. Inference recursively rejects nested label/outcome fields. Runtime boundaries block label-file reads and arbitrary subprocess escape; the existing restricted native provider can use only the checked MCP Toolkit capability, whose Python process installs its own boundary. Native Parquet loading uses an explicit `development_labels` split allowlist before calling pandas/Arrow; test/evaluation splits and symlink aliases are rejected. Training permits train/valid labels only. The enabled run ends at frozen test predictions and does not enter the separate final evaluator.

Enabled preflight carries the preparation manifest's declared evaluation seals without opening or hashing evaluation-label bytes. Tests cover both this preflight path and complete fake-client execution through sealed predictions; the evaluation loader is never called.

Verify persisted traces without opening labels or contacting a model:

```powershell
$env:PYTHONPATH = 'source/code'
& $py -m news_network_50_2025.runner_receipts --run $env:PR5_RUN_ROOT
```

The verifier refuses an empty trace directory, altered Skill bytes, hash tampering and cross-case/cutoff/policy receipts. Canary tests attempt evaluation/label reads, nested label payloads, evaluation training rows and injected source access. This is a scoped host-boundary verification, with host source-inspection and gate-fitting attestations kept explicit; a hash is not independent evidence of a source's economic meaning or a proof about an opaque provider implementation.

## Gate 5: frozen-cache reporting

`hop3_analysis.cached_hop_report` joins the completed Qwen input index, sealed prediction list, prediction files, response receipts and existing matched development CSV. It checks byte hashes, full cell alignment, prompts, source IDs, availability and CSV/frozen-score equality, then computes exact-hop evidence/missing shares, raw and nested training-fitted losses, and costs on identical cases. It imports no provider/runner transport. See [HOP3_ABLATION.md](HOP3_ABLATION.md) for the resulting tables.

```powershell
& $py -m news_network_50_2025.hop3_analysis --cached-qwen $env:HOP3_QWEN_CACHE --out $env:PR5_COST_JSON --markdown $env:PR5_COST_MARKDOWN
```

Output files must be new. Frozen run code hashes remain historical; this reporting command does not resume/rewrite that run. Its JSON records hashes for every consumed cache artifact. Cost counts unique scoring requests per arm, separately reports additional requests versus the preceding arm, and deduplicates all 1,778 physical requests in the overall total. Tokens are 11,620,303 prompt plus 726,109 completion tokens; recorded model service time is 7,916.83 seconds. Article counts are retained scoring source IDs, including OWN; they exclude host-only support reads. Retrieval/end-to-end seconds were not cached and remain null. Costs across wider arms are not matched compute, and continuity missingness is not evidence of absent economic propagation.

## Offline verification and operator handoff

Run every `examples/*.py` as an isolated `python -m unittest examples.<stem>` process with `PYTHONPATH=source/code`. Set `Q1_SEAL_DATE=2025-11-03` only for `test_q1_identify`, `test_q1_identify_v2` and `test_q1_design`; the graph/prompt suite retains its September 1 default. Also run `news_network_50_2025.test_design` and the PR1 validator unit tests. The four helper examples import successfully and have no unittest cases (exit code 5). Test logs, cached-report JSON, synthetic trace demonstration, exact-file scan and manifest patch/snapshot are external review artifacts.

Final results: **216 passing tests** (188 example tests, 25 design tests and 3 validator tests), plus four successful helper imports. Each suite ran in its own process using the requested interpreter, with network/model transports blocked. The forced-on compatibility run failed five legacy suites (`test_hop3_v2`, `test_hop_ablation`, `test_q1_graph_prompt_gd`, `test_three_hop_toolkit`, and `test_design`), establishing the default-off decision. The enabled fixture/runner tests pass. A persisted synthetic case verifies 43 Skill/Python/tool/decision events, including all four passing arithmetic gate receipts; this is not a real-data acceptance.

The exact 29-path candidate scan found no secret material, licensed news/copyright payloads, workstation paths, binaries or files over 1 MiB. Heuristic scans were complemented by reviewing the changed source/Skill/report text. The pre-existing backup was preserved and excluded. The old manifest-helper unit assertion was corrected because PR4 now tracks `hop3_qwen.py`; the validator itself was not changed.

The validator is unchanged. The working-tree manifest requires refresh; the exact candidate snapshot must qualify independently. `docs/PR5_COMMIT_FILES.txt` is the staging allowlist, including the operator-applied manifest patch. The pre-existing untracked backup is excluded. Regenerate/review manifest hashes after any byte or line-ending conversion. Historical source mismatches, rank deficiency and unrecomputed counts/GARCH remain qualifications rather than package-hash errors. Do not stage caches, raw news, review artifacts or the backup.

Prepare a fresh review patch and candidate snapshot, then check both without changing Git metadata:

```powershell
& $py examples/prepare_review_manifest.py --files docs/PR5_COMMIT_FILES.txt --out "$env:PR5_REVIEW_ROOT/file_manifest.review.json" --patch "$env:PR5_REVIEW_ROOT/manifest_refresh.patch" --snapshot "$env:PR5_REVIEW_ROOT/candidate_snapshot"
git apply --check "$env:PR5_REVIEW_ROOT/manifest_refresh.patch"
& $py skills/supply-chain-news-risk/scripts/validate_dictionary.py --repo "$env:PR5_REVIEW_ROOT/candidate_snapshot" --mode forecasting
# Operator, after reviewing the exact patch:
git apply "$env:PR5_REVIEW_ROOT/manifest_refresh.patch"
```
