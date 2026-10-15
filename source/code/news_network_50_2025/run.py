"""Resumable bounded experiment. No test-label access before all policies/predictions freeze."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import traceback
import importlib.metadata

from filelock import FileLock
import pandas as pd
from news_agent_batched_runtime_2025 import read, write, locked_write, digest, BudgetStop, ProviderStop
from news_event_risk_data_2025 import file_hash
from news_agent_r2_2025.config import CLAUDE
from .config import OUT, ROOT, VERSION, MODEL, LIMITS, SEED, CONTROLS, ARMS, WORKERS, SPLITS
from .runtime import NativeClient
from .workflow import Runner
from .metrics import anchor, matched, summarize


def progress(out, **values):
    path = out / "progress.json"
    old = read(path) if path.exists() else {}
    write(path, {**old, **values, "updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
    current = read(path)
    usage = read(out / "usage.json") if (out / "usage.json").exists() else {}
    text = ["# Live Run Status", "", f"Status: {current.get('status')}; stage: {current.get('stage')}.",
            f"Updated: {current['updated_at']}. PID: {current['pid']}.", "",
            "50 firms, 12 nonoverlapping five-session periods. OWN/NET fixed and selected prompts. No placebo.",
            f"Cached score records (including development candidates/repeats): {current.get('cached_predictions', 0)}.",
            f"CLI attempts: {usage.get('attempts', 0)}. List-price equivalent: ${usage.get('list_price_usd', 0):.2f}.",
            "List-price equivalent is not subscription billing. Failed-call reservations may be conservative.", "",
            "Final test outcomes remain sealed until prompts and all four arms' predictions are frozen.",
            "A retained seed is not described as an optimization improvement."]
    if current.get("error"):
        text += ["", "Last error: " + current["error"]]
    (out / "LIVE.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def preflight(out):
    data = read(out / "data_manifest.json")
    for name, sha in data["outputs"].items():
        if file_hash(out/name) != sha:
            raise ValueError("Prepared artifact changed: " + name)
    cases = read(out / "cases.json")
    if len(cases) != 600 or len({c["case_id"] for c in cases}) != 600:
        raise ValueError("Expected 600 unique firm-periods")
    plan = {"version": VERSION, "model": MODEL, "schedule": SPLITS, "firms": sorted({c["ticker"] for c in cases}),
            "arms": ARMS, "placebo": False, "controls": CONTROLS, "limits": LIMITS,
            "tool_use": {"turn_cap": None, "call_cap": None, "native_request_timeout_seconds": 900,
                         "note": "No hard per-agent turn/call cap; retain global budget and source restrictions"},
            "selection": "train 200 / validation 100 / final test 300; time holdout primary; no weekly updates",
            "optimization": "Individual-case GEPA/Pareto, residual-MSE, at most two candidates per OWN/NET; frozen evidence",
            "inference": "Exploratory: only six test dates, historical periods previously researched, graph vintage assumption",
            "claim_limit": "NET adds evidence and compute. Without placebo, not an isolated graph-topology effect.",
            "primary": "Pooled explanatory delta R2; separate development-fitted regression forecast R2",
            "target": "Five-session mean squared close-to-close returns, log units; not intraday RV or IV",
            "revision_note": "Frozen seed retained when validation does not justify a challenger; no guaranteed improvement"}
    plan = json.loads(json.dumps(plan))
    locked_write(out / "PROTOCOL.json", plan)
    sources = list((ROOT / "code/news_network_50_2025").glob("*.py"))
    sources += [ROOT / ("code/"+p) for p in ("news_agent_batched_runtime_2025.py", "news_event_risk_data_2025.py",
                "news_event_calendar_2025.py", "news_exposure_2025/graph.py", "news_exposure_2025/data.py",
                "news_exposure_2025/workflow.py", "news_exposure_2025/metrics.py", "midcap_standard_2025/benchmarks.py")]
    manifest = {"protocol": digest(plan), "data_manifest": file_hash(out/"data_manifest.json"),
                "workers": WORKERS,
                "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "dependencies": {p: importlib.metadata.version(p) for p in ("dspy", "gepa", "mcp", "numpy", "pandas", "pyarrow")}}
    return cases, manifest


def acceptance(out, cases):
    """Outcome-free invariants; real-source relevance review is a separate signed gate."""
    from .workflow import pool_file
    from .toolkit import Toolkit
    reviewed = []
    # Stratified development-only checks over all four training origins and all baskets.
    selected = [c for c in cases if c["split"] == "train"][::3][:60]
    for case in selected:
        day = pool_file(str(out), case["date"])
        pool = day["pools"][case["case_id"]]
        ids = pool["own_ids"]+pool["network_ids"]
        for sid in ids:
            if pd.Timestamp(day["documents"][sid]["available_at"]) >= pd.Timestamp(case["asof"]):
                raise ValueError("Post-cutoff article")
        if set(pool["own_ids"]) & set(pool["network_ids"]):
            raise ValueError("Network supplement overlaps base packet")
        if len(pool["network_ids"]) > 20:
            raise ValueError("Network candidate cap exceeded")
        expected = digest([day["documents"][s] for s in pool["own_ids"]])
        if expected != pool["own_packet_hash"]:
            raise ValueError("Base evidence identity changed")
        reviewed.append({"case_id": case["case_id"], "own_documents": len(pool["own_ids"]),
                         "network_documents": len(pool["network_ids"]), "own_packet_hash": expected})
    write(out / "acceptance.json", {"passed": True, "cases": reviewed, "provider_calls": 0,
          "scope": "Timing, disjointness, hashes, caps and case integrity; not human relevance ground truth"})


def resume_manifest(out, current):
    path = out / "RUN_MANIFEST.json"
    if not path.exists():
        locked_write(path, current)
        return current
    initial = read(path)
    if initial == current:
        return initial
    if {k:v for k,v in initial.items() if k != "sources"} != {k:v for k,v in current.items() if k != "sources"}:
        raise ValueError("Resume cannot change protocol, data, dependencies or worker configuration")
    approval = read(out / "RESUME_APPROVAL.json")
    changed = sorted(k for k in set(initial["sources"]) | set(current["sources"])
                     if initial["sources"].get(k) != current["sources"].get(k))
    if (approval.get("from_manifest") != digest(initial) or approval.get("to_manifest") != digest(current)
            or approval.get("changed_sources") != changed or not approval.get("retain_validated_cache")):
        raise ValueError("Unapproved implementation revision; do not silently relabel old predictions")
    locked_write(out / "engineering_revisions" / (digest(current)+".json"), current)
    return initial


def resource_limits(out):
    limits = dict(LIMITS)
    path = out / "RESOURCE_EXTENSION.json"
    if not path.exists():
        return limits
    extension = read(path)
    changes = extension.get("limits", {})
    if (not extension.get("authorization") or set(changes) != {"input_token_stop"}
            or extension.get("initial_manifest") != digest(read(out / "RUN_MANIFEST.json"))):
        raise ValueError("Unrecognized resource-only continuation")
    cap = changes["input_token_stop"]
    if isinstance(cap, bool) or not isinstance(cap, int) or not limits["input_token_stop"] <= cap <= 300_000_000:
        raise ValueError("Input-token continuation exceeds its authorized ceiling")
    limits.update(changes)
    return limits


def execute(out=OUT, smoke=False):
    with FileLock(str(out / ".run.lock"), timeout=0):
        cases, manifest = preflight(out)
        acceptance(out, cases)
        if not smoke:
            review = read(out / "toolkit_review.json")
            if not review.get("approved") or review.get("manifest") != digest(manifest):
                raise ValueError("Native tool/evidence review not approved for these source bytes")
            cache_manifest = resume_manifest(out, manifest)
        else:
            cache_manifest = manifest
        anchors = anchor(out, cases)
        client = NativeClient(out, resource_limits(out), CLAUDE)
        client.semaphore = threading.BoundedSemaphore(WORKERS)
        client.budget.start()
        runner = Runner(out, client, cases, anchors, digest(cache_manifest), digest(manifest))
        progress(out, status="running", stage="smoke" if smoke else "development", active_job=True, error=None, traceback=None,
                 implementation_manifest=digest(manifest), runtime_limits=client.limits,
                 resource_extension=(file_hash(out / "RESOURCE_EXTENSION.json")
                                     if (out / "RESOURCE_EXTENSION.json").exists() else None),
                 score_tasks_this_process=0, last_case=None, last_arm=None)
        count = 0
        lock = threading.Lock()

        def panel(batch, policy, network=False, nonce=""):
            nonlocal count
            def task(case):
                nonlocal count
                value = runner.score(case, policy, network, nonce)
                with lock:
                    count += 1
                    progress(out, last_case=case["case_id"], last_arm="NET" if network else "OWN",
                             score_tasks_this_process=count, cached_predictions=len(list((out/"predictions").glob("*.json"))))
                return value
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                return list(pool.map(task, batch))

        try:
            if smoke:
                # Source review only: no numerical outcome is returned or used for acceptance.
                smoke_cases = [next(c for c in cases if c["ticker"] == ticker and c["split"] == "train")
                               for ticker in ("AMD", "CAT", "WMT") if any(c["ticker"] == ticker for c in cases)]
                if len(smoke_cases) < 3:
                    smoke_cases = [c for c in cases if c["split"] == "train"][::17][:3]
                rows = {arm: panel(smoke_cases, SEED, network) for arm, network in (("OWN_FIXED", False), ("NET_FIXED", True))}
                write(out / "smoke.json", {"manifest": digest(manifest), "records": rows,
                      "test_labels_accessed": False, "source_review_pending": True})
                progress(out, status="awaiting_toolkit_review", stage="smoke_complete", active_job=False)
                return
            development = [c for c in cases if c["split"] != "test"]
            records = {a: [] for a in ARMS}
            records["OWN_FIXED"] = panel(development, SEED)
            records["NET_FIXED"] = panel(development, SEED, True)
            # Repeated seeds on a preselected development subset measure generation noise, not tune on it.
            repeat_cases = [c for c in development if c["split"] == "train"][::17][:12]
            repeats = {arm: panel(repeat_cases, SEED, net, "repeat-1") for arm, net in (("OWN", False), ("NET", True))}
            write(out / "repeatability.json", repeats)
            progress(out, stage="optimization")
            from .optimize import optimize
            policies = {}
            for arm, net in (("OWN_OPT", False), ("NET_OPT", True)):
                # Optional search cannot consume the reserve needed to finish the fixed comparisons.
                native_times = [read(p)["seconds"] for p in (out/"native_responses").glob("*.json")]
                score_times = [r["seconds"] for p in (out/"responses").glob("*.json")
                               if (r := read(p))["purpose"] == "score"]
                import numpy as np
                native_mean = float(np.mean(native_times)) if native_times else 90.
                score_mean = float(np.mean(score_times)) if score_times else 25.
                completion_seconds = 1.15*(600*native_mean + 1200*score_mean)/WORKERS + 300
                candidate_seconds = 450*score_mean/WORKERS + 180
                reserve = max(7200, completion_seconds+candidate_seconds)
                write(out/"completion_reserve.json", {"arm": arm, "seconds": reserve,
                    "remaining": client.budget.remaining_seconds(), "native_mean_seconds": native_mean,
                    "score_mean_seconds": score_mean, "purpose": "Protect fixed comparisons and selected-policy test scoring"})
                if client.budget.remaining_seconds() < reserve:
                    decision = {"policy": SEED, "selection": {"changed": False},
                                "reason": "completion reserve: optimization not attempted, not an optimized improvement"}
                    write(out / "optimization" / ("NET" if net else "OWN") / "decision.json", decision)
                else:
                    decision = optimize(runner, cases, net, panel)
                policies[arm] = decision["policy"]
                records[arm] = panel(development, policies[arm], net)
            locked_write(out / "frozen_policies.json", policies)
            # Fit all readouts before opening test labels. Raw scores still enter one-step OLS, no beta preprocessing.
            train = matched(out, cases, records, "train")
            from .metrics import fit
            readouts = {arm: {name: fit(train, controls, arm) for name, controls in
                         (("garch", ("b",)), ("expanded", CONTROLS))} for arm in ARMS}
            locked_write(out / "frozen_readouts.json", readouts)
            progress(out, stage="sealed_test_prediction")
            test_cases = [c for c in cases if c["split"] == "test"]
            for arm, net, policy in (("OWN_FIXED", False, SEED), ("NET_FIXED", True, SEED),
                                    ("OWN_OPT", False, policies["OWN_OPT"]), ("NET_OPT", True, policies["NET_OPT"])):
                records[arm] += panel(test_cases, policy, net)
            locked_write(out / "all_prediction_keys.json", {a: [r["key"] for r in rs] for a, rs in records.items()})
            progress(out, stage="final_evaluation")
            test = matched(out, cases, records, "test")
            summary = summarize(train, test)
            write(out / "summary.json", summary)
            test.to_csv(out / "matched_test.csv", index=False)
            lines = ["# 50-Firm Network Experiment", "", "Complete. No placebo arm.", "",
                     "50 firms; 4 training, 2 validation and 6 test origins. Five-session log variance.", "",
                     "| Arm | Pooled increment over GARCH (pp) | Beyond expanded controls (pp) | Score coefficient | Frozen forecast R2 vs expanded baseline |",
                     "|---|---:|---:|---:|---:|"]
            for arm, value in summary["arms"].items():
                lines.append(f"| {arm} | {value['garch']['increment_pp']:.3f} | {value['expanded']['increment_pp']:.3f} | {value['expanded']['score_coefficient']:.3f} | {value['expanded']['frozen_incremental_r2']:.4f} |")
            lines += ["", summary["inference"], "", summary["placebo"], "",
                      "Pooled R2 is an explanatory OLS refit on test observations, not forecast R2. Frozen readouts use training data only.",
                      "Negative score coefficients contradict the intended risk direction even when refitted R2 rises. Raw residual MSE and uncalibrated score forecast R2 are reported separately in summary.json.",
                      "Optimization minimizes training-OLS residual MSE on individual cases; both validation periods must support replacing the seed.",
                      "See optimization/*/decision.json for whether either optimized arm actually changed. No improved arm is claimed when the seed was retained.",
                      "Known calendar and cash-deal controls include transparent text proxies, not a perfect corporate-event database."]
            (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
            progress(out, status="complete", stage="complete", active_job=False, test_labels_accessed=True)
        except (BudgetStop, ProviderStop) as exc:
            progress(out, status="stopped", active_job=False, error=str(exc))
            raise
        except Exception as exc:
            progress(out, status="failed", active_job=False, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    execute(args.out, args.smoke)
