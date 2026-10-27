# Real-data cumulative hop ablation

**Engineering results only. Qwen3.8-27B is not look-ahead clean for these historical dates. There are only six date clusters.** The completed Qwen run does not establish incremental third-hop predictive value, practical equivalence, or causal transmission. DeepSeek is an incomplete, separate second model and contributes no scores to the results below.

## Implementation and commit scope

The experiment rebuilds the network evidence pool through three typed graph hops. Full-graph BFS supplies exact shortest distances, avoiding bias from capped path enumeration. One deterministic shortest witness per mapped entity and every required prefix survive; mapped intermediates, fewer competitor steps and stable IDs break ties without changing distance. Target-entity stock aliases, own news and co-mentions are excluded from network candidates. Frozen local MiniLM headline similarity plus recency fills 8/6/6 reserves to 20 candidates. Host-only dated intermediate articles support exact-prefix inspection and never consume extra primary scoring slots.

The five cumulative arms are OWN, H1, H2, H3 and H3_GRAPH. Each has an all-paths and a continuity-only panel. The latter retains direct paths without calling direct product exposure verified; deeper paths need dated inspected verbatim edge-product receipts, explicit intermediate input/output links, and an affected-item endpoint receipt. Topology, revenue percentages, keyword overlap, sector or name similarity and undated profiles cannot establish continuity. Missing receipts mean unverified continuity, not absence of economic transmission.

All paths below are relative to the repository; module names in this table are under `source/code/news_network_50_2025/`.

| File | Change and necessity | Compatibility |
|---|---|---|
| `data.py` | Adds the opt-in `--development-only` CLI switch and lazily dispatches it to `hop3_prepare.prepare(out, max_hops)`. Needed for the dedicated label-free pool entry point. | Existing `prepare` API and CLI defaults remain unchanged when the flag is absent. The new path requires `--max-hops 3` and an explicit fresh output root. |
| `runtime.py` | Adds `QwenClient`: local score transport/cache and deterministic `native()` host Toolkit inspection with exact-prefix gates and trace hashes. Needed by `hop3_v2` input preparation; its legacy scorer uses 700 output tokens. The completed run instead uses `hop3_qwen.QwenClient` with 7,000. | Existing `NativeClient` and provider defaults are unchanged. New dependencies are imported lazily. No automatic router or Qwen activation. |
| `hop3_prepare.py` | Rebuilds six date pools, outcome-free source/label hash manifests, shortest relations, 8/6/6 candidates, common controls and host-only prefix support. Runs one date per child process with local-only embeddings. | New opt-in module. Per-user package-path injection removed; dependencies must already be installed in the selected interpreter. |
| `hop3_shortest.py` | Exact full-graph BFS and independent selected-source radius audit. | New module; does not change the existing capped enumerator. |
| `product_continuity.py` | Conservative source-bound two/three-hop product-flow validator, including date, edge, quote, direction and endpoint checks. | New module; returns `path_only` when required evidence is missing. |
| `hop3_continuity_sources.py` | Audits dated vendor keyword fields and exclusions without promoting search leads into continuity facts. | New module; vendor input stays external. No vendor export or guide text is packaged. |
| `hop3_guard.py` | Six-date case/window checks, read-only live inputs, sealed/mixed artifact blocks, explicit Arrow label gates, complete hashed prediction freeze and Windows resource checks. | New opt-in module. Configured roots are guarded even with nonstandard directory names; existing generic cache-name blocks remain. |
| `hop3_v2.py` | Shared host evidence/prompt preparation, graph degree/triangle receipts, continuity and scope audits; retains the original legacy scorer/report for provenance. | New opt-in module. Its 700-token legacy prediction path is not the completed Qwen result. Server-owner PID is configurable for that legacy path; no process management. |
| `hop3_deepseek.py` | Separate DeepSeek transport, input conversion/archive, pilot, estimated-cost reservations, durable requests/responses, deduplication, finite-score/citation checks and full prediction freeze. | New opt-in module; reads the API key only from the environment. Import it in its own process: it sets the shared guard model identity. |
| `hop3_deepseek_report.py` | Post-freeze verification/analysis and explicit partial, budget and account-block reports without fabricated statistics. | New module. Partial runs cannot open labels for a statistical readout. |
| `hop3_qwen.py` | Sequential local-only 7,000-token scoring, raw-response persistence, unknown-outcome stop, full response/score verification, label gating, report and final completion signal. Supports fresh Qwen preparation without requiring a partial DeepSeek run. | New opt-in module; archived historical prompts still take precedence when present. Code hashes prevent resuming a frozen run with modified code. |
| `hop3_analysis.py` | Raw MSE, nested train-fitted validation regressions, pooled exploratory HC3/CR1 90% intervals, effective-rank handling, g3 sensitivity bounds and power grid. | New model-independent module. Redundant terms are explicitly unidentified. |
| `examples/test_hop3_v2.py` | Synthetic continuity, case/freeze/file guards, configured-root protection, regression recovery/rank and shortest/prefix tests. | Offline tests; no real source text or data. |
| `examples/test_hop3_deepseek.py` | Mocked transport, cost/cache/reservation and parsing tests. | Offline; credential-shaped input is a clearly synthetic placeholder. |
| `examples/test_hop3_qwen.py` | Mocked local transport, failure/caching/context/model/citation/label checks and a complete synthetic fresh-input preparation check. | Offline; no model request. |
| `docs/HOP3_ABLATION.md` | Implementation inventory, execution protocol, engineering readout and review instructions. | New documentation; supersedes older pending hop-run status statements. |
| `docs/HOP3_COMMIT_FILES.txt` | Exact 18-file allowlist for one operator commit, including the refreshed package manifest. | New documentation; stage only these paths. |
| `metadata/file_manifest.json` | Operator-applied hash/size refresh for the exact candidate tree, including every new selected file. | Same manifest format and unchanged validator. Historical source provenance is not rewritten. |

Preparation code uses environment-configured local roots instead of workstation paths. Generated data, licensed articles/vendor exports, credentials, caches, SQLite embeddings, predictions, reports from a run, logs, backups, review snapshots and patches are excluded from the commit. Aggregate engineering results in this document are the requested exception for the readout. Existing `hop_ablation.py`, graph skills, toolkit, graph router and earlier Q1 documentation/tests are dependencies already on the branch, not additional files in this commit.

## Protocol

- Exactly 50 firms on each date: 200 train cases on October 1/8/15/22, 2025; 100 validation cases on October 30 and November 6. Every five-session target window ends before the November 14 evaluation seal. Q1 identification/design suites have their separate November 3 seal; they must not change this experiment's seal.
- Open only those six development pools. Reject evaluation pools/cases, mixed cases/controls/calendar/anchor artifacts and evaluation labels. Native Arrow reads occur only behind explicit path, freeze and source-hash checks.
- Use the frozen outcome-free raw GARCH anchor `b`, not an OLS anchor fitted before scoring. All 11 conventional controls enter the post-freeze regression. Identical source content, controls and per-case excerpt caps remain common across cumulative arms.
- Freeze and hash all 300 cases x five arms x two panels (3,000 predictions), request identities and response bytes; reverify complete coverage and finite scores before either train or validation labels open. Exact request bodies share cached responses. Unknown outcomes stop for inspection rather than automatic duplicate inference. Labels are never used to select, omit or rescore a case for citation quality.
- Use deterministic host Toolkit retrieval, including full-page inspection and every exact prefix; models score leading excerpts. This is not autonomous Qwen or DeepSeek tool use. Wider arms can change order/attention and are not matched-compute comparisons.
- Fit nested coefficients on the four train dates and apply unchanged to validation. Pooled train+valid inference is exploratory and is distinct from the train-fitted validation comparison. No evaluation result is reported.

## Running with external local inputs

The repository does not distribute the licensed Reuters/FactSet inputs or a model. Before running, supply the already prepared live six-date pools, graph, numerical manifest, exact train/valid label files, Reuters index metadata and compressed indexes, locally available article files, vendor keyword export and cached `sentence-transformers/all-MiniLM-L6-v2`. Use a Windows Python environment with the repository dependencies plus CPU torch/sentence-transformers already available. No dependency installation or model download is part of this workflow.

Set all overrides **before starting Python**. Use a new writable run root outside the checkout; do not point preparation at the completed historical run.

| Environment variable | Meaning / default |
|---|---|
| `HOP3_PYTHON` | Interpreter selected by the operator; used as `$py` below. |
| `HOP3_PROJECT_ROOT` | Local research project holding `code`, `data` and `results`; default is this package's `source` directory. |
| `HOP3_LIVE_ROOT` | Read-only network cache; default `<project>/results/news_network_50_2025`. |
| `HOP3_EXPOSURE_ROOT` | Read-only exposure cache/indexes; default `<project>/results/news_exposure_2025`. |
| `HOP3_REUTERS_INDEX_ROOT` | Reuters index metadata; default `<project>/results/news_event_risk_2025/reuters_index`. |
| `HOP3_FACTSET_ROOT` | External vendor export directory; default `<project>/data/factset`. |
| `HOP3_RUN_ROOT` | New experiment root; default `<project>/results/hop3_run`. |
| `HOP3_REPORT_PATH` | Published report destination; default `<run>/reports/HOP3_ABLATION_REPORT.md`. Use a distinct destination per model/run. |
| `HOP3_QWEN_SERVER_LOG` | Qwen server log; default `<project>/data/qwen/server.log`. |
| `HOP3_SERVER_OWNER_PID` | Positive owner PID, used only by the legacy `hop3_v2 --stage predict` wait gate. Not used by the recommended scorer below. |

Shared **offline preparation**, from the repository root in PowerShell:

```powershell
$py = $env:HOP3_PYTHON
if (-not (Test-Path -LiteralPath $py)) { throw 'Set HOP3_PYTHON to the local interpreter' }
if (-not $env:HOP3_RUN_ROOT) { throw 'Set HOP3_RUN_ROOT to a fresh external directory' }
$env:PYTHONPATH = 'source/code'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $py -m news_network_50_2025.data --development-only --max-hops 3 --out $env:HOP3_RUN_ROOT
if ($LASTEXITCODE) { throw 'Pool preparation failed' }
& $py -m news_network_50_2025.hop3_shortest
if ($LASTEXITCODE) { throw 'Independent radius audit failed' }
& $py -m news_network_50_2025.hop3_v2 --stage inputs
if ($LASTEXITCODE) { throw 'Host input preparation failed' }
```

The preparation guard prohibits external network access. Local MiniLM must load from cache. Metadata/code/source hashes, not input files, may be reviewed; none belong in the proposed commit. `--out` and `HOP3_RUN_ROOT` must refer to the same directory.

Run the offline checks below after preparation. Preserve their actual log and write `TESTS_PASSED.json` only after successful substantive suites (exit code zero). For Qwen place it in `<run>/hop_ablation_v2/qwen/`; for DeepSeek in `<run>/hop_ablation_v2/`. The scorer checks `exit_code: 0`. This host attestation is not a replacement for inspecting test results.

Qwen, with an already running operator-managed model at the fixed loopback endpoint `127.0.0.1:8081`:

```powershell
& $py -m news_network_50_2025.hop3_qwen --stage prepare
if ($LASTEXITCODE) { throw 'Qwen preparation failed' }
Copy-Item -LiteralPath (Join-Path $env:HOP3_RUN_ROOT 'TESTS_PASSED.json') -Destination (Join-Path $env:HOP3_RUN_ROOT 'hop_ablation_v2/qwen/TESTS_PASSED.json')
& $py -m news_network_50_2025.hop3_qwen --stage run
```

`run` scores, freezes, verifies and reports, then writes `QWEN_RUN_DONE_v2.json` last. For a fresh run it uses the original Qwen input preparation directly; a historical DeepSeek handoff uses `handoff_archive_qwen3.8-27b` when present. It preserves prior DeepSeek artifacts by hash without mixing scores. Each request uses temperature zero, seed 20261026, thinking disabled, a 7,000-token output allowance, a conservative byte-plus-framing prompt bound below 24,000 and total context below 32,768. It neither starts nor stops a server. An existing completion signal stops an automatic rerun; inspect the saved status first. Protocol code hashes intentionally reject changed code: this PR is not permission to rebaseline old frozen artifacts or restart the completed experiment.

DeepSeek uses a **separate fresh run root**, repeating the shared preparation first. Only the explicitly invoked scoring stages use the external endpoint; no such stages were run during this offline PR preparation.

```powershell
& $py -m news_network_50_2025.hop3_deepseek --stage prepare
if ($LASTEXITCODE) { throw 'DeepSeek preparation failed' }
Copy-Item -LiteralPath (Join-Path $env:HOP3_RUN_ROOT 'TESTS_PASSED.json') -Destination (Join-Path $env:HOP3_RUN_ROOT 'hop_ablation_v2/TESTS_PASSED.json')
# Supply DEEPSEEK_API_KEY in the process environment, never a file or command literal.
& $py -m news_network_50_2025.hop3_deepseek --stage pilot
if ($LASTEXITCODE) { throw 'Inspect the pilot/block receipt before proceeding' }
& $py -m news_network_50_2025.hop3_deepseek --stage predict --workers 8
if ($LASTEXITCODE) { throw 'Inspect the durable request state before resuming' }
& $py -m news_network_50_2025.hop3_deepseek --stage report
```

DeepSeek archives the unscored Qwen preparation, converts model identity uniformly, validates the exact case/input set, and runs a deterministic 20-request pilot before full scoring. Only the authorized DeepSeek HTTPS host is allowed; proxies and redirects are disabled. Its hard-coded reference rates and $15 reservation threshold are historical **estimated costs**, not a reliable billing ceiling: observed pricing was approximately four times the logged estimate. Reconcile current operator-supplied rates/budget before any new paid run. The preserved historical DeepSeek run stopped for budget and remains partial (880/1,778 at the last checkpoint, 898 response files, 1,186 cells in an older partial freeze). It must not be reported as a completed comparison or combined with Qwen.

## Engineering readout copied from the completed report

Source: `HOP3_ABLATION_REPORT.md`, completed October 27, 2026; SHA256 `c596a1ec62ce24804a80599dcf0edda10407daca76bc97b73cee725b93254322`. The report and local `QWEN_RUN_DONE_v2.json` agree on success, 1,778 unique responses and 3,000 predictions. These are report transcriptions, not a new model execution. Frozen-prediction SHA256: `e35818011d4c305217442341a97ace4080683f9db35537669e9e3f14a9ae6d95`. Freeze time: 2026-10-27 14:55:20.451883 UTC; pre-label verification: 14:55:48.096578 UTC.

Raw residual MSE is `mean((y - raw GARCH anchor - model score)^2)`. Zero-score baselines: **train 1.30514, validation 0.854961**. Every raw arm remains above its corresponding baseline.

| Arm | Train MSE, all paths | Validation MSE, all paths | Train MSE, continuity only | Validation MSE, continuity only |
|---|---:|---:|---:|---:|
| Zero score | 1.30514 | 0.854961 | 1.30514 | 0.854961 |
| OWN | 1.37708 | 0.956171 | 1.37708 | 0.956171 |
| H1 | 1.36048 | 0.954624 | 1.36048 | 0.954624 |
| H2 | 1.3664 | 0.952103 | 1.36048 | 0.954624 |
| H3 | 1.3653 | 0.962951 | 1.36048 | 0.954624 |
| H3_GRAPH | 1.35429 | 0.921325 | 1.34872 | 0.914943 |

The nested equation is `y = intercept + X b + g_own OWN + g1 H1 + g2(H2-H1) + g3(H3-H2) + e`, with the same 11 controls. The `b` vector here denotes regression coefficients; the raw GARCH control is also named `b` in the data schema.

| Panel | Controls + OWN validation MSE | + H1 | + (H2-H1) | + (H3-H2) |
|---|---:|---:|---:|---:|
| All paths | 0.862878 | 0.935706 | 0.941227 | 0.944518 |
| Continuity only | 0.862878 | 0.935706 | 0.935706 | 0.935706 |

These fits use train coefficients unchanged on validation. Adding H3 raises raw validation MSE from 0.952103 to 0.962951 (+0.010849 from unrounded values) and nested validation MSE from 0.941227 to 0.944518 (+0.003291). The best raw all-paths arm, H3_GRAPH, still exceeds the zero-score baseline.

Pooled **exploratory** estimates use all six train+validation dates:

| Panel | Term | Estimate | HC3 90% CI | Date-cluster 90% CI |
|---|---|---:|---|---|
| All paths | g1 / H1 | 3.02316 | [0.325362, 5.72096] | [-0.0548322, 6.10115] |
| All paths | g2 / H2-H1 | 0.407809 | [-4.89886, 5.71448] | [-5.69597, 6.51159] |
| All paths | g3 / H3-H2 | 1.49165 | [-1.88887, 4.87216] | [-2.3195, 5.3028] |
| Continuity only | g1 / H1 | 2.87963 | [0.750698, 5.00856] | [-0.0190824, 5.77834] |
| Continuity only | g2 / H2-H1 | unidentified | unidentified | unidentified |
| Continuity only | g3 / H3-H2 | unidentified | unidentified | unidentified |

All three all-paths date-cluster intervals include zero; the HC3 g1 interval does not. HC3 uses residual effective-rank degrees of freedom. CR1 uses `G/(G-1) * (n-1)/(n-rank)` and t intervals with five date degrees of freedom. Redundant controls and zero contrasts are unidentified; pseudoinverse fitted predictions do not make those coefficients interpretable. With only six dates, cluster intervals are fragile.

| Panel | Smallest symmetric g3 bound, HC3 | Date-cluster bound | Partial SD of H3-H2 |
|---|---:|---:|---:|
| All paths | 4.87216 | 5.3028 | 0.0261171 |
| Continuity only | unidentified | unidentified | 0 |

The bound is `max(abs(lower 90% CI), abs(upper 90% CI))`, in log variance per unit increment score. A strictly larger symmetric margin satisfies the interval criterion. **No practical equivalence margin was prespecified.** This is observed design sensitivity, not evidence of practical equivalence.

| Panel | Assumed g3 | RMS log-risk effect | Cases for 80% power, HC3 | Cases for 80% power, new date clusters |
|---|---:|---:|---:|---:|
| All paths | 0.1 | 0.00261171 | 778365 | 663500 |
| All paths | 0.25 | 0.00652929 | 124539 | 106200 |
| All paths | 0.5 | 0.0130586 | 31135 | 26550 |
| All paths | 1.0 | 0.0261171 | 7784 | 6650 |
| Continuity only | 0.1 | 0 | unidentified | unidentified |
| Continuity only | 0.25 | 0 | unidentified | unidentified |
| Continuity only | 0.5 | 0 | unidentified | unidentified |
| Continuity only | 1.0 | 0 | unidentified | unidentified |

Power uses the local normal approximation `n_new = n*((z_.95 + z_.80)*SE/effect)^2`, two-sided alpha .10, assuming stable signal, design and noise. The cluster projection rounds to 50 cases per **new independent date**; more firms on the same dates do not add independent time variation. These exploratory extrapolations are not finite-sample guarantees.

**Continuity-only result:** H1 = H2 = H3 exactly because no complete multi-hop chain passed the receipts. Consequently g2/g3, their bounds and their power sample sizes are unidentified. H3_GRAPH can differ through graph instructions and direct-pair numerical receipts; that does not establish multi-hop product continuity.

| Split | Exact H1 candidates | Exact H2 | Exact H3 | Any H3 candidate | Accessible H3 sources | Cases with accessible H3 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1583 | 1200 | 1217 | 100.0% | 657 | 190 |
| Validation | 800 | 600 | 600 | 100.0% | 330 | 95 |

The independent corrected-pool audit found zero radius mismatches. The earlier capped-enumeration preflight had 930 false H3 rows and two own aliases, corrected before scoring. Graph skills used repository degree/triangle calculations at 90/75/0.5 thresholds. The June effective-date graph and mapped membership stayed fixed; archived monthly acquisition vintages were not certified. Reuters pages were inspected for retrieval but not exhaustively converted to structured flow facts, so conservative continuity coverage can miss real chains.

The report records 77 passing pre-run checks, 45 citation-flagged responses, five exact unique ID alias resolutions and one response with unresolved IDs; all finite scores were retained. Every response identified the Qwen model and ended with `finish_reason=stop`. Maximum actual prompt length was 8,961 tokens; maximum prompt plus the 7,000-token allowance was 15,961 of 32,768. One sequential request and fixed temperature/seed do not imply bitwise backend determinism.

## Cached exact-hop missingness and cost

Derived offline from frozen development artifacts; zero new model calls. Evaluation labels remain sealed.

| Mode | Split | Contrast | n | Any exact-hop evidence | Missing | Raw MSE | Raw gain | Nested MSE | Nested gain | Calls (additional unique) | Cumulative tokens | Cumulative articles | Model seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_paths | train | H1 | 200 | 100.0% | 0.0% | 1.360484 | 0.016600 | 0.880489 | 0.046663 | 200 (200) | 1,283,727 | 6254 | 755.09 |
| all_paths | train | H2-H1 | 200 | 98.0% | 2.0% | 1.366398 | -0.005915 | 0.877868 | 0.002621 | 200 (196) | 1,430,414 | 6968 | 720.35 |
| all_paths | train | H3-H2 | 200 | 95.0% | 5.0% | 1.365296 | 0.001103 | 0.877839 | 0.000029 | 200 (190) | 1,567,765 | 7625 | 716.59 |
| all_paths | valid | H1 | 100 | 100.0% | 0.0% | 0.954624 | 0.001546 | 0.935706 | -0.072828 | 100 (100) | 634,054 | 3307 | 388.76 |
| all_paths | valid | H2-H1 | 100 | 100.0% | 0.0% | 0.952103 | 0.002522 | 0.941227 | -0.005521 | 100 (100) | 711,602 | 3711 | 365.88 |
| all_paths | valid | H3-H2 | 100 | 95.0% | 5.0% | 0.962951 | -0.010849 | 0.944518 | -0.003291 | 100 (95) | 776,495 | 4041 | 358.95 |
| continuity_only | train | H1 | 200 | 100.0% | 0.0% | 1.360484 | 0.016600 | 0.880489 | 0.046663 | 200 (200) | 1,283,727 | 6254 | 755.09 |
| continuity_only | train | H2-H1 | 200 | 0.0% | 100.0% | 1.360484 | 0.000000 | 0.880489 | 0.000000 | 200 (0) | 1,283,727 | 6254 | 755.09 |
| continuity_only | train | H3-H2 | 200 | 0.0% | 100.0% | 1.360484 | 0.000000 | 0.880489 | 0.000000 | 200 (0) | 1,283,727 | 6254 | 755.09 |
| continuity_only | valid | H1 | 100 | 100.0% | 0.0% | 0.954624 | 0.001546 | 0.935706 | -0.072828 | 100 (100) | 634,054 | 3307 | 388.76 |
| continuity_only | valid | H2-H1 | 100 | 0.0% | 100.0% | 0.954624 | 0.000000 | 0.935706 | 0.000000 | 100 (0) | 634,054 | 3307 | 388.76 |
| continuity_only | valid | H3-H2 | 100 | 0.0% | 100.0% | 0.954624 | 0.000000 | 0.935706 | 0.000000 | 100 (0) | 634,054 | 3307 | 388.76 |

| Mode | Split | Contrast | Additional-request tokens | Additional-request seconds | Article change | Token change | Model-second change |
|---|---|---|---:|---:|---:|---:|---:|
| all_paths | train | H1 | 1,283,727 | 755.09 | +1583 | +324,851 | -88.48 |
| all_paths | train | H2-H1 | 1,398,609 | 692.55 | +714 | +146,687 | -34.74 |
| all_paths | train | H3-H2 | 1,488,430 | 668.49 | +657 | +137,351 | -3.76 |
| all_paths | valid | H1 | 634,054 | 388.76 | +800 | +156,081 | -51.55 |
| all_paths | valid | H2-H1 | 711,602 | 365.88 | +404 | +77,548 | -22.88 |
| all_paths | valid | H3-H2 | 737,858 | 337.53 | +330 | +64,893 | -6.93 |
| continuity_only | train | H1 | 1,283,727 | 755.09 | +1583 | +324,851 | -88.48 |
| continuity_only | train | H2-H1 | 0 | 0.00 | +0 | +0 | +0.00 |
| continuity_only | train | H3-H2 | 0 | 0.00 | +0 | +0 | +0.00 |
| continuity_only | valid | H1 | 634,054 | 388.76 | +800 | +156,081 | -51.55 |
| continuity_only | valid | H2-H1 | 0 | 0.00 | +0 | +0 | +0.00 |
| continuity_only | valid | H3-H2 | 0 | 0.00 | +0 | +0 | +0.00 |

No retained scoring article at this exact hop; not a zero forecast increment or absence of economic exposure.
Retained source IDs per case in cumulative scoring packets, including OWN; repeated across cases count separately. Host-only supporting reads are excluded.
Unique cached scoring requests per arm/split/mode. Additional requests are set difference from previous arm; signed cost changes are separate. Shared requests are counted once in unique_run_cost.
Recorded model response service seconds. Retrieval and end-to-end timing were not cached and remain null; no timing is imputed.
Raw residual MSE and nested OLS MSE with coefficients fitted on training cases only. Positive gain means lower loss. H1 is compared with OWN.

Cumulative costs describe standalone arms. They are not matched compute. The JSON also reports signed cost changes and additional-request cost.

The source index, freeze and existing development CSV are hashed in the external derived-report JSON. Across all arms/panels there are 1,778 unique cached scoring requests, 12,346,412 total tokens and 7,916.83 recorded model seconds. No live label file is opened by this reporting command. The H1 comparison is against OWN; H2-H1 and H3-H2 retain identical cases within each split.

## Router decision

PR5 wires the deterministic router into Runner and Toolkit, with actual host-inspection prefix receipts, persisted Skill/Python traces, provider-safe weights and training acceptance-gate receipts. The flag defaults **OFF** because legacy fixtures/capabilities do not meet the enabled dated-snapshot contract. Enabled execution ends before evaluation labels open. See [PR5_IMPLEMENTATION.md](PR5_IMPLEMENTATION.md). This integration does not resume or modify the frozen Qwen run.

## Offline checks and operator manifest patch

The following paragraphs retain the original PR4 preparation record. PR5's final check is **216 passing tests**, four helper imports and a clean 29-path scan; its allowlist is `docs/PR5_COMMIT_FILES.txt`. Use [PR5_IMPLEMENTATION.md](PR5_IMPLEMENTATION.md) for the current manifest, trace and default-off compatibility instructions.

Run every `examples/*.py` file in its own `python -m unittest examples.<stem>` process with `PYTHONPATH=source/code`. Set `Q1_SEAL_DATE=2025-11-03` only for `test_q1_identify`, `test_q1_identify_v2` and `test_q1_design`; remove it for other suites (the graph/prompt suite uses its September 1 default). Also run:

```powershell
& $py -m unittest news_network_50_2025.test_design
& $py -m unittest discover -s skills/supply-chain-news-risk/scripts -p test_validate_dictionary.py
& $py skills/supply-chain-news-risk/scripts/validate_dictionary.py --repo . --mode forecasting
```

For a fresh experiment, this isolated-process loop writes a receipt only after the substantive suites pass. Keep the log outside the checkout; do not reuse a receipt after editing code:

```powershell
$testLog = Join-Path $env:HOP3_RUN_ROOT 'offline_unittest.log'
if (Test-Path -LiteralPath $testLog) { throw 'Choose a fresh run root/log' }
foreach ($file in Get-ChildItem examples -Filter '*.py' | Sort-Object Name) {
    Remove-Item Env:Q1_SEAL_DATE -ErrorAction SilentlyContinue
    if ($file.BaseName -in @('test_q1_identify','test_q1_identify_v2','test_q1_design')) {
        $env:Q1_SEAL_DATE = '2025-11-03'
    }
    & $py -m unittest ('examples.' + $file.BaseName) 2>&1 | Tee-Object -FilePath $testLog -Append
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not ($code -eq 5 -and $file.BaseName -notlike 'test*')) {
        throw ('Failed suite: ' + $file.Name)
    }
}
Remove-Item Env:Q1_SEAL_DATE -ErrorAction SilentlyContinue
& $py -m unittest news_network_50_2025.test_design 2>&1 | Tee-Object -FilePath $testLog -Append
if ($LASTEXITCODE) { throw 'Design tests failed' }
& $py -m unittest discover -s skills/supply-chain-news-risk/scripts -p test_validate_dictionary.py 2>&1 | Tee-Object -FilePath $testLog -Append
if ($LASTEXITCODE) { throw 'Validator tests failed' }
$receipt = @{ exit_code=0; log_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $testLog).Hash; checked_at=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $env:HOP3_RUN_ROOT 'TESTS_PASSED.json'), $receipt)
```

PR preparation verification: all 13 substantive example suites passed (171 tests), design passed (25), and validator unit tests passed (3): **199 passing tests**. The four helper examples contain no unittest cases and return Python's exit code 5 for zero tests; they imported successfully. The offline harness blocked actual network I/O, allowing only the Windows asyncio socket-pair self-pipe. Its initially overbroad self-pipe block caused a harness-only design failure; that suite passed after narrowing the block, without changing repository tests.

The unchanged PR #1 validator is **BLOCKED** on the unrefreshed working-tree manifest (the two modified tracked modules). Generate the review patch/snapshot from the explicit allowlist using the existing helper; choose fresh external paths:

```powershell
& $py examples/prepare_review_manifest.py --files docs/HOP3_COMMIT_FILES.txt --out $env:HOP3_REVIEW_MANIFEST --patch $env:HOP3_MANIFEST_PATCH --snapshot $env:HOP3_CANDIDATE_SNAPSHOT
git apply --check $env:HOP3_MANIFEST_PATCH
& $py skills/supply-chain-news-risk/scripts/validate_dictionary.py --repo $env:HOP3_CANDIDATE_SNAPSHOT --mode forecasting
# Operator, after reviewing the patch:
git apply $env:HOP3_MANIFEST_PATCH
& $py skills/supply-chain-news-risk/scripts/validate_dictionary.py --repo . --mode forecasting
```

The prepared candidate snapshot is **QUALIFIED**, with no package-manifest mismatches. Existing warnings remain: historical source differences, rank deficiency, and counts/GARCH not recomputed. This is package qualification, not clean historical provenance or predictive validation. The validator and historical source manifest are unchanged. Every selected new file is explicitly covered, even though the old validator itself checks only listed entries. Hashes reflect exact checkout bytes; regenerate and review after any line-ending conversion or file edit. The operator applies the patch and stages exactly the paths listed in `docs/HOP3_COMMIT_FILES.txt`; `.git` was not written during preparation.

The exact candidate set was reviewed for secret/key material, credential assignments, workstation paths, licensed-source payloads, generated data/binaries and files over 1 MiB. No such payload was selected. Environment-variable names and synthetic mocked credential values are not real credentials. The external audit records file sizes and SHA256 values; it and the manifest patch/snapshot/logs are excluded from the commit.
