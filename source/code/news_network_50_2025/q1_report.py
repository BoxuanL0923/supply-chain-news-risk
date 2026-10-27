"""Render Q1 aggregate results; never loads outcome tables or licensed evidence."""
from pathlib import Path
import shutil
import numpy as np
from .prompt_gd import MODEL, NAMES, read, convergence_report, smooth_score


class Unavailable(float):
    """Display an unrun metric explicitly; never serialize it as a result."""
    def __new__(cls): return super().__new__(cls, float("nan"))
    def __format__(self, spec): return "not run"


def median_metric(rows, key):
    values = [r[key] for r in rows]
    return float(np.median(values)) if np.isfinite(values).all() else Unavailable()


def render(root=Path("D:/OptResearch_data/agent_q1")):
    repo = Path(__file__).resolve().parents[3]
    destination = Path("C:/OptResearch/results/sc_risk/agent_q1")
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"figures").mkdir(exist_ok=True)
    run = root/"run"
    a = read(run/"gd.json")
    b_complete = (run/"spsa.json").exists()
    if b_complete:
        b = read(run/"spsa.json")
    else:
        b = read(run/"spsa.partial.json")
        spread = np.ptp([r["theta"] for r in b["runs"]], axis=0)
        b.update(theta_range=spread.tolist(), same_theta=bool(max(spread) <= .05),
                 generalizes=None, grid=[], validation_baseline_mse=Unavailable())
        for r in b["runs"]:
            r.update(validation_mse=Unavailable(), gap=Unavailable())
    convergence_report(run, {"gd": a, "spsa": b})
    smoke = (read(root/"smoke/summary.json") if (root/"smoke/summary.json").exists() else
             {"status": "BLOCKED (local endpoint unavailable)", "validation_n": 0})
    smoke_complete = smoke["status"] == "PASS"
    status = read(root/"run_status.json")
    if (root/"run_status_initial.json").exists():
        initial = read(root/"run_status_initial.json")
        resumed = [] if status == initial else [{**s, "stage": "resume/"+s["stage"]} for s in status["completed"]]
        status = {"completed": [{**s, "stage": "initial/"+s["stage"]} for s in initial["completed"]] + resumed}
    manifest = read(root/"data/development_manifest.json")
    figures = repo/"docs/figures"
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run/"convergence.png", figures/"q1_convergence.png")
    shutil.copyfile(run/"convergence.png", destination/"figures/q1_convergence.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (name, report) in zip(axes, (("A: analytic GD", a), ("B: SPSA", b))):
        for item in report["runs"]:
            axis.plot(range(7), item["theta"], marker="o", alpha=.6)
        axis.set_xticks(range(7), NAMES, rotation=35, ha="right")
        axis.set_ylim(-.03, 1.03); axis.set_ylabel("Normalized parameter")
        axis.set_title(name+" — five starts, two schedules")
    fig.suptitle(MODEL+" — final parameter dispersion (engineering only)")
    fig.tight_layout(); fig.savefig(figures/"q1_parameters.png", dpi=150); plt.close(fig)
    shutil.copyfile(figures/"q1_parameters.png", destination/"figures/q1_parameters.png")
    best_a = min(a["runs"], key=lambda r: r["train_mse"])
    best_b = min(b["runs"], key=lambda r: r["train_mse"])
    features = {r["case_id"]: r["features"] for r in read(run/"features.json")["cases"]}
    training_features = [features[cid] for cid in manifest["splits"]["train"]]
    jacobian = np.array([smooth_score(best_a["theta"], f)[1] for f in training_features])
    nonzero = {h: sum(e["relevance"]*abs(e["direction"])*e["materiality"] > 0
                     for f in training_features for e in f if e["hop"] == h) for h in range(4)}
    lines = ["# Q1: graph skills and prompt-parameter convergence", "",
        f"Model: **{MODEL}**, local `http://127.0.0.1:8081/v1`, temperature 0, seed 20261026, thinking disabled. "
        "Engineering only: the specified model was released after these historical dates and is not look-ahead-clean.", "",
        "## Verdict", "",
        f"Scheme A: same-parameter convergence **{a['same_theta']}**; validation improvement **{a['generalizes']}**. "
        f"Scheme B: same-parameter convergence **{b['same_theta']}**; validation improvement **{b['generalizes'] if b_complete else 'NOT ASSESSED'}**. "
        f"The Part-3 pipeline smoke is **{smoke['status']}**, with {smoke['validation_n']} matched validation cases. "
        "These are bounded engineering diagnostics, not evidence of predictive or causal supply-chain effects.", "",
        ("All requested numerical stages completed." if b_complete and smoke_complete else
         "**Incomplete execution:** the local HTTP connection reset after all ten SPSA starts and the endpoint then refused connections. "
         "SPSA grid completion/validation and the real-Qwen Part-3 smoke remain blocked. No missing score or validation result is fabricated. "
         "The user was asked about endpoint availability; no server process was stopped or restarted."), "",
        "## Implementation and skill invocations", "",
        "`graph_skills.py` supplies standard-library directed degree, monthly degree percentiles, undirected triangle/clustering statistics, "
        "pair shared-partner counts and resource allocation. Standalone repository CLI scripts bootstrap the local numerical module; no package installation or network is required. "
        "`Toolkit`, MCP tool registration and `TOOL_NAMES` expose both skills. `Runner(graph_snapshot=...)` can attach a dated host snapshot; "
        "the evidence prompt explicitly invokes the skills. `validate_evidence` discards model-written graph statistics and recomputes host fields. "
        "`finance_graph_signal` passes only allowlisted numerical values to the existing context selector; missing graph coverage is null. "
        "The context-selection Skill documents invocation, thresholds, required outputs, verification and stop conditions.", "",
        "```text",
        "/sc-graph-degree compute target degree; inputs: firm_id, cutoff, dated directed snapshot with monthly_universe, max_hops, inspected_prefixes; script: skills/sc-graph-degree/scripts/graph_degree.py; constraints: strictly pre-cutoff edges, max_hops <= 3, inspected exact prefixes, THRESH_HUB=90; deliverable: verified JSON statistics then stop.",
        "/sc-graph-triangles compute target/pair motifs; inputs: firm_id, neighbour, cutoff, dated directed snapshot with monthly_universe, max_hops, inspected_prefixes; script: skills/sc-graph-triangles/scripts/graph_triangles.py; constraints: strictly pre-cutoff edges, max_hops <= 3, inspected exact prefixes, THRESH_RA=75, THRESH_CLUST=0.5; deliverable: verified JSON statistics then stop.",
        "/supply-chain-context-selection rank inspected candidates; inputs: dated events and host graph statistics; script: source/code/news_network_50_2025/context_selection.py; constraints: cutoff and event budget; deliverable: selected IDs and Python receipt.", "```", "",
        "Degree counts unique directed partners, excluding self loops and duplicate edges; a reciprocal partner contributes to both in- and out-degree. "
        "The monthly percentile is the weak empirical CDF (ties share the upper rank). Triangles and RA use the simple undirected projection, "
        "including the supplied supplier/customer and competitor links. Pair triangles are zero for nonadjacent endpoints, while shared-neighbour count and RA remain meaningful. "
        "RA is the sum of inverse undirected degrees over shared partners. Its default threshold is the linearly interpolated 75th percentile "
        "across target-to-other-month-universe pairs, with an additional positive-RA guard to avoid declaring zero exposure specialized. "
        "Targets with percentile >=90 down-weight single-neighbour news. Clustering >=0.5 denotes possible common exposure, not independent corroboration. "
        "Triangle/clustering effects are not assumed robust. No graph statistic validates product continuity.", "",
        "Graph snapshots and each edge's vintage/valid_from must precede cutoff. Expired edges are excluded. "
        "Every deeper path needs inspected exact prefixes; an entity visited on a different branch does not unlock the path. "
        "Listings are bounded to 200 paths and disclose truncation. An explicitly requested pair can supply a verified branch beyond the display cap. "
        "The preparation host actually reads the prefix sources before each next hop.", "",
        "## Data discipline and local provenance", "",
        "The live 50-firm config assigns its train/valid cases to October-November 2025. Those labels conflict with the stricter sealed September-December window, "
        "so none were opened. The mixed `news_exposure_2025/labels_sealed.parquet` file was also never opened. "
        "Instead, nine original **August training** cases were selected by ticker/date alone: AMD, CAT and MSFT on August 1/8/22. "
        "Six August 1/8 cases train Scheme A; three August 22 cases validate. Scheme B uses the first two training cases (AMD/CAT, August 1) "
        "and two validation cases (AMD/CAT, August 22). No outcome-based selection of cases or sources occurred.", "",
        "MSE comparisons are within each scheme's panel. Scheme A and B use different case counts, so their raw MSE magnitudes do not rank the two optimization methods.", "",
        "Read-only source caches: `C:/OptResearch/results/news_exposure_2025/cases.json`, the three dated August corpus gzip files, "
        "and `graph.json`; the live code copies supplied path/configuration information only. The graph is the pre-existing June 6 effective-date reconstruction. "
        "Its June-mapped entities define the explicit August reference universe; this is a held-fixed membership assumption, not a certified monthly vendor vintage. "
        "The anchor is the cached pre-August intercept-plus-GARCH OLS, not a refit of PR #2's expanded eleven-control anchor. "
        "The residual-MSE contract is identical, but this engineering panel is not a replication of the 50-firm experiment.", "",
        "August-only labels were newly derived from a date/ID-filtered projection of `crsp_daily_2025.parquet`: "
        "exactly five finite decimal total returns, y=log(mean(r²)), no target floor. All target sessions end by August 29. "
        "Validation is separated from training by the unused August 15 origin. No September-December labels, predictions, or evaluation reports were read. "
        "The new label files have an explicit development manifest, split allowlists and hashes; loaders validate metadata before opening them. "
        "Smoke predictions are durably frozen before the smoke's validation-label loader runs. These validation labels had already served the earlier optimization diagnostics; the smoke is not a fresh statistical holdout.", "",
        "Large artifacts and source text stay under `D:/OptResearch_data/agent_q1/`; only aggregate figures/results and code enter the branch.", "",
        "## Parameterization and predeclared convergence tests", "",
        "| Parameter | Bounds | Default | Meaning |", "|---|---:|---:|---|",
        "| lambda2 | [0,1] | 0.5 | Two-hop discount |", "| lambda3 | [0,1] | 0.05 | Three-hop discount |",
        "| THRESH_HUB | [50,100] percentile | 90 | Target diversification threshold |",
        "| THRESH_RA | [50,100] percentile | 75 | Positive shared-exposure threshold |",
        "| THRESH_CLUST | [0,1] | 0.5 | Dense/common-exposure threshold |",
        "| materiality | [0,1] | 0.5 | Neighbour-news materiality gate |",
        "| k | fixed at 8 | 8 | Maximum evidence budget |", "| w | [0,1] | 0.5 | score = w × raw score |", "",
        "Optimization uses seven normalized coordinates x in [0,1], with native theta=LOW+(HIGH-LOW)x. "
        "Five seeded starts, two schedules, tolerance epsilon=1e-5, relative change delta=1e-7 for M=5 iterations, "
        "and maximum caps were declared in code/PROTOCOL before execution. Same-theta success requires maximum coordinate range <=0.05 "
        "across all starts/schedules. Generalization requires the training-selected solution to beat validation's zero-score anchor baseline; "
        "validation does not select the reported optimum. Near-flat Gauss-Newton directions use eigenvalues <1e-6. "
        "All attempted starts and schedules are reported, including failures to meet stopping tolerances.", "",
        "### Scheme A: exact gradients of a cached-feature surrogate", "",
        "Qwen was called once per case (nine requests) to extract relevance, signed variance-risk direction and materiality for every candidate. "
        "Host code joins hop and graph statistics; the model cannot overwrite them. Retrieval, source order and the eight-event budget are frozen. "
        "Let sigmoid be s, H=s((degree_percentile-THRESH_HUB)/10), R=s((RA_percentile-THRESH_RA)/10) for positive RA (else 0), "
        "C=s((clustering-THRESH_CLUST)/0.1), and M=s((materiality_feature-materiality_threshold)/0.1). "
        "For network candidates the contribution is relevance × direction × materiality_feature × M × (1-0.5H) × (1+0.25R) × (1-0.5C) × hop_discount / 8. "
        "Own candidates omit the threshold gate M and the graph/hop factors. Score is w times the sum. This is a differentiable engineering surrogate, not a calibrated economic model. "
        "The loss is mean((y-anchor-score)²), differentiated analytically including normalized-coordinate scaling. "
        "Projected GD uses step 0.5 or 0.5/(1+t/10)^0.602 for 400 iterations. Both raw and projected gradient norms are saved; "
        "the projected norm is used at box constraints. A full 3^7=2,187-point grid uses normalized levels 0.1/0.5/0.9.", "",
        "| Schedule | Start | Iterations | Stop | Initial loss | Train MSE | Validation MSE | Gap | Initial gradient | Final gradient | Final projected gradient |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in a["runs"]:
        t = r["trajectory"]
        lines.append(f"| {r['schedule']} | {r['start']} | {len(t)-1} | {r['stop']} | {t[0]['loss']:.6f} | {r['train_mse']:.6f} | {r['validation_mse']:.6f} | {r['gap']:.6f} | {t[0]['gradient_norm']:.6g} | {t[-1]['gradient_norm']:.6g} | {t[-1]['projected_gradient_norm']:.6g} |")
    lines += ["", f"Training-selected A: {best_a['schedule']}, start {best_a['start']}; train MSE {best_a['train_mse']:.6f}, "
              f"validation {best_a['validation_mse']:.6f}, validation baseline {a['validation_baseline_mse']:.6f}. "
              f"Coarse grid train MSE {a['grid']['train_mse']:.6f}, validation {a['grid']['validation_mse']:.6f}. "
              f"Gauss-Newton near-flat directions: {a['flat_directions']}/7. Eigenvalues: `{a['gauss_newton_eigenvalues']}`. "
              "This is a local sensitivity diagnostic, not a proof about the full nonlinear Hessian. Six training observations cannot identify seven unconstrained directions.", "",
        f"The cached training evidence has nonzero signed feature counts by hop {nonzero}. "
        f"The selected-point score Jacobian has numerical rank {np.linalg.matrix_rank(jacobian)}; "
        f"its column norms are `{dict(zip(NAMES, np.linalg.norm(jacobian, axis=0).tolist()))}`. "
        "In particular, no nonzero two-hop training feature can identify lambda2 in this cache. This is a data/support limitation, not evidence that the economic two-hop effect is zero.", "",
        "### Scheme B: thresholds inside the local-model prompt", "",
        "SPSA uses frozen retrieved text, the same seed for both perturbations, temperature zero, and host-applied w shrinkage. "
        "The model reads all numeric parameters and emits bounded raw residual scores in [-3,3]. Perturbations use independent Rademacher coordinates, "
        "c_t=0.1/(t+1)^0.101, and a_t=0.12/(t+1)^alpha for alpha in {0.602,0.8}. "
        "The perturbation center is moved inward so both probes stay in the box; each update is projected to the box. "
        "Each run has at most 12 iterations; only relative-loss stability or the cap can stop it, since one zero noisy gradient is not proof of stationarity. "
        "An additional four-point coarse grid varies materiality and w over {0.2,0.8}, holding the other five coordinates at defaults. "
        "This is a restricted two-dimensional grid comparison, not an exhaustive seven-dimensional prompt search. "
        "Response caching uses the full request hash; both the declared and server-returned model IDs are recorded. Requests are strictly serial.", "",
        "| Schedule | Start | Iterations | Stop | Train MSE | Validation MSE | Gap | First SPSA norm | Last SPSA norm |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in b["runs"]:
        t = r["trajectory"]
        lines.append(f"| {r['schedule']} | {r['start']} | {len(t)} | {r['stop']} | {r['train_mse']:.6f} | {r['validation_mse']:.6f} | {r['gap']:.6f} | {t[0]['gradient_norm']:.6g} | {t[-1]['gradient_norm']:.6g} |")
    lines += ["", f"Training-selected B: {best_b['schedule']}, start {best_b['start']}; train MSE {best_b['train_mse']:.6f}, "
              f"validation {best_b['validation_mse']:.6f}, validation baseline {b['validation_baseline_mse']:.6f}.", "",
              "| Grid materiality | Grid w | Train MSE | Validation MSE |", "|---:|---:|---:|---:|"]
    for r in b["grid"]:
        lines.append(f"| {r['theta'][5]:.1f} | {r['theta'][6]:.1f} | {r['train_mse']:.6f} | {r['validation_mse']:.6f} |")
    if not b_complete:
        lines += ["", "Scheme B grid/validation metrics are not available: the connection failed during the grid. "
                  "The ten completed optimization trajectories above are genuine local-Qwen results. Resume replays their cached responses before issuing only missing requests."]
    lines += ["", "### Dispersion and schedules", "",
        "| Coordinate | A normalized final range | B normalized final range | A selected native value | B selected native value |",
        "|---|---:|---:|---:|---:|"]
    for i, name in enumerate(NAMES):
        lines.append(f"| {name} | {a['theta_range'][i]:.6f} | {b['theta_range'][i]:.6f} | {best_a['parameters'][name]:.6f} | {best_b['parameters'][name]:.6f} |")
    lines += ["", "| Scheme/schedule | Median train MSE | Median validation MSE | Maximum normalized within-schedule range |", "|---|---:|---:|---:|"]
    for name, report in (("A", a), ("B", b)):
        for schedule in sorted({r["schedule"] for r in report["runs"]}):
            rows = [r for r in report["runs"] if r["schedule"] == schedule]
            lines.append(f"| {name}/{schedule} | {median_metric(rows, 'train_mse'):.6f} | {median_metric(rows, 'validation_mse'):.6f} | {np.ptp([r['theta'] for r in rows], axis=0).max():.6f} |")
    lines += ["", "![Loss trajectories and gradient norms](figures/q1_convergence.png)", "",
        "![Final normalized parameters](figures/q1_parameters.png)", "",
        "Parameters are not established as uniquely identified: inspect the dispersion, A's flat directions, "
        "and the prompt response's finite precision/discontinuities. SPSA norms are noisy directional estimates, not exact gradients; "
        "a small last estimate alone is not convergence. The small validation panels do not support inferential claims.", "",
        "## Part-3 smoke", "",
        f"Status: **{smoke['status']}**. Arms: OWN, H1, H2, H3 and H3_GRAPH. " +
        ("All five real-model arms completed. " if smoke_complete else
         "These five arms are implemented and covered by the offline smoke test, but the real-model smoke has not run. ") +
        "The host reuses `scope_network_pool`, exact-hop auditing, `Toolkit.inspect_event`, both graph tools and `matched_scores`. "
        "Article eligibility follows the shortest accessible typed path in each scope, including multi-tag sources initially discovered through longer chains. "
        "Own evidence, parameters and model remain fixed across arms; graph statistics are supplied only in H3_GRAPH. "
        "The host freezes every arm before loading validation labels, and checks the saved freeze hash. "
        "No regression or empirical performance claim is made for this two-case smoke. " +
        ("Artifacts: " if smoke_complete else "Required artifacts, not yet produced: ") +
        "`D:/OptResearch_data/agent_q1/smoke/matched_validation.csv`, `summary.json`, `prediction_freeze.json`, and `PROTOCOL.json`.", "",
        "## Execution, resources, and reproduction", "",
        "| Stage | Exit code | Seconds | Sampled process peak working set (MB) |", "|---|---:|---:|---:|"]
    for s in status["completed"]:
        lines.append(f"| {s['stage']} | {s['exit_code']} | {s['seconds']:.1f} | {s['peak_working_set_mb']:.1f} |")
    lines += ["", "The runner samples each child's OS peak working set every two seconds. Preparation was the largest process. "
        "The GD row is the initial monitored pass; the corrected rerun is recorded separately in `gd_correction.json` and `gd_corrected.log`, with exit code zero and no new model requests. "
        "Small buffers, one worker, and single-threaded BLAS keep this workload below 4 GB per process; the existing model server is external and was not managed. "
        "No process was killed, no server stopped/restarted, and no paid provider or non-loopback network endpoint was used. "
        "The HTTP client rejects non-authorized endpoints, bypasses proxies and refuses redirects.", "",
        "```powershell", "$env:PYTHONPATH='D:/OptResearch_data/external/supply-chain-news-risk/source/code'",
        "$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'",
        "python -m news_network_50_2025.q1_prepare  # fresh output only",
        "python -m news_network_50_2025.prompt_gd cache-features",
        "python -m news_network_50_2025.prompt_gd gd",
        "python -m news_network_50_2025.prompt_gd spsa",
        "python -m news_network_50_2025.hop_ablation --q1-validation-data D:/OptResearch_data/agent_q1/data --out D:/OptResearch_data/agent_q1/smoke",
        "python -m news_network_50_2025.prompt_gd convergence-report",
        "python -m news_network_50_2025.q1_report", "```", "",
        "`q1_run` executes these computational stages sequentially with background logs/status files. "
        "Resume uses content-addressed model responses. Do not run the historical unqualified `hop_ablation` entry point for Q1: it opens test labels and uses the old provider.", "",
        "Once the existing local endpoint is available, resume with `python -m news_network_50_2025.q1_run`, "
        "then regenerate this report with `python -m news_network_50_2025.q1_report`. The runner does not manage llama-server. "
        "The requested local commit message is prepared at `D:/OptResearch_data/agent_q1/COMMIT_MESSAGE.txt`; committing still requires write access to this checkout's `.git`.", "",
        "## Possible errors and checks performed", "",
        "- Test suites: 57 example tests (including 13 Q1 tests), 25 PR #2 design tests, and 3 dictionary-validator tests passed. Both new skill frontmatter validators passed. Logs are under `D:/OptResearch_data/agent_q1/`.",
        "- The forecasting dictionary validator is QUALIFIED with zero errors and zero package-manifest mismatches. Existing qualifications concern unrecomputed counts/GARCH, a rank-deficient historical training design, and source differences from the recorded historical run. The package manifest was refreshed for this Windows checkout and the new artifacts.",
        "- Known-value graph tests cover directed degree, triangle/clustering, RA=1/3, isolates, duplicates, self loops, expired/future edges, monthly universes and three-hop prefixes. A real-data display-cap failure was corrected and regression-tested.",
        "- All 108 prepared typed path steps were checked against the dated graph's supplier/customer/competitor directions, with zero mismatches. Each selected target has exactly one graph entity. Preparation now selects direction from actual supply edges, avoiding competitor-direction ambiguity in future inputs.",
        "- Analytic derivatives agree with central finite-difference checks to eight decimal places; a five-start quadratic verifies the optimizer and stopping condition. Zero shrinkage remains differentiable.",
        "- Guards reject evaluation splits/dates before label I/O, target sessions in September, future source passages, duplicate source IDs and invalid feature values. Cached hop/graph fields must match their host-owned inputs.",
        "- An offline smoke test asserts that all five prediction arms are on disk before the validation label loader is called. Existing staged-path and context-selector tests remain intact; native MCP exposes eight tools.",
        "- The smoke integration test includes a multi-tag H3-discovered article with a direct eligible tag and verifies that it enters H1. This corrected an initial selected-path filter before any real smoke execution.",
        "- August preparation initially encountered a string-versus-timestamp Parquet filter mismatch. The bounded string-date filter was corrected; no label file was opened to work around it. Test coverage also exposed mixed timezone notation, now normalized to UTC.",
        "- A specification audit caught the first Scheme A implementation applying the neighbour-materiality threshold to own news. It was corrected and regression-tested; GD was rerun using the same nine cached responses, starts and schedules. The superseded result is preserved as `run/gd_pre_neighbor_gate_fix.json`. Scheme B already applied that threshold only to neighbours and was unaffected. Reported Scheme A numbers use the corrected function.",
        "- After all ten SPSA starts, the local HTTP connection reset and the endpoint refused connections. The runner stopped on the error. Completed request-hashed responses and the original failure log are preserved; resume can reuse them without restarting or stopping the model server.",
        "- All local model responses identify qwen3.8-27b. One extraction response per case is reused for every Scheme A start, schedule and grid point; labels never enter model request bodies.",
        "- Source article reads support retrieval only. Reuters tags, a graph edge, and a hop sequence do not prove event materiality or product continuity. The narrow recency-selected chains are not exhaustive or independent evidence.",
        "- Limitations: tiny samples, revised/non-certified vendor vintage, fixed June membership, a reduced anchor, smooth-gate design choices, flat directions, bounded optimization caps, a restricted Scheme B grid, and a model released after the observation dates. No claim of economic generalization follows.", "",
        "## Local commit status", "",
        "The requested branch is `pr2-local-extensions`. The managed sandbox marks this checkout's `.git` read-only. "
        "`git add` failed with `Unable to create .../.git/index.lock: Permission denied`; consequently no new local commit can be created in this session. "
        "No push was attempted. The intended commit must end with:", "",
        "```text", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "```", ""]
    body = "\n".join(lines)
    (repo/"docs/Q1_GRAPH_SKILLS_AND_PROMPT_GD.md").write_text(body, encoding="utf-8", newline="\n")
    (destination/"Q1_REPORT.md").write_text(body, encoding="utf-8", newline="\n")
    print(MODEL, "report written", flush=True)


if __name__ == "__main__": render()
