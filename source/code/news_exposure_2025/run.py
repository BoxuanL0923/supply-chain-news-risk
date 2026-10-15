"""Run preflight, acceptance and the three frozen arms; no implicit launch."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time
import traceback

import pandas as pd
from filelock import FileLock

from news_agent_batched_runtime_2025 import Client, BudgetStop, ProviderStop, read, write, locked_write, digest
from news_event_risk_data_2025 import file_hash
from news_agent_r2_2025.config import CLAUDE
from .config import *
from .graph import Graph
from .metrics import matched_frame, summarize
from .workflow import ForecastRunner
from .optimize import optimize


class GuardedClient(Client):
    def call(self, *args, **kwargs):
        for attempt in range(2):
            if shutil.disk_usage(self.root).free < 500_000_000:
                raise BudgetStop("Disk reserve below 500 MB; no additional provider calls")
            try:
                return super().call(*args, **kwargs)
            except (ProviderStop, subprocess.TimeoutExpired) as exc:
                message = str(exc).lower()
                fatal = any(x in message for x in ("usage limit", "hit your limit", "not logged", "model mismatch", "session limit", "authentication"))
                transient = isinstance(exc, subprocess.TimeoutExpired) or any(x in message for x in ("overloaded", "rate_limit_error", "529", "temporarily unavailable"))
                if attempt or fatal or not transient:
                    raise
                time.sleep(5)
                kwargs = {**kwargs, "retry": True}


def protocol():
    return {"version": VERSION, "model": MODEL, "firms": list(FIRMS), "splits": SPLITS, "arms": list(ARMS),
            "controls": list(CONTROLS), "policy_seed": SEED, "contract": CONTRACT, "network_policy": NETWORK_RULES,
            "demonstrations": DEMONSTRATIONS, "graph_date": GRAPH_DATE, "limits": LIMITS, "max_action_rounds": MAX_ACTION_ROUNDS,
            "max_tool_calls_per_round": MAX_TOOL_CALLS, "forecast_workers": WORKERS,
            "optimization": "At most six GEPA revisions, training-only reflection; at most two nominees plus seed evaluated once on validation; one-SE seed preference; no weekly updates",
            "target": "log(mean(next 5 squared decimal total returns)); one raw score relative to frozen OLS anchor",
            "evaluation": "Pooled evaluation-sample OLS incremental R2; descriptive, not frozen-coefficient prediction R2",
            "limitations": ["Previously examined history, exploratory only", "Eight time clusters, four validation origins",
                            "FactSet effective-date reconstruction, not certified acquisition vintage", "Return revisions not vintage-certified",
                            "July-only frozen co-mentions", "Entity name/news mapping incomplete for private and foreign intermediaries",
                            "Source logbook begins July; calendar scan begins January", "Post-release evaluation does not certify model training contents"],
            "preparation": "Local models/CPU only; no provider call before acceptance and frozen manifest"}


def preflight(out):
    review = read(out / "retrieval_review.json")
    if review.get("approved") is not True or review.get("acceptance_sha256") != file_hash(out / "retrieval_acceptance.json"):
        raise ValueError("Real-source retrieval review required before provider usage")
    for p in ("data_manifest.json", "embedding_manifest.json", "numerical_manifest.json"):
        obj = read(out/p)
        for filename, sha in obj["outputs"].items():
            if file_hash(out/filename) != sha:
                raise ValueError("Prepared artifact altered: " + filename)
    cases = read(out / "cases.json")
    if len(cases) != 200 or len({c["case_id"] for c in cases}) != 200:
        raise ValueError("Expected 200 unique cases")
    controls = pd.read_parquet(out / "controls.parquet")
    if len(controls) != 200 or not controls[list(CONTROLS)].notna().all().all():
        raise ValueError("Missing declared controls")
    if not Path(CLAUDE).is_file():
        raise ValueError("Pinned Claude CLI unavailable")
    source = list(Path(__file__).parent.glob("*.py")) + [ROOT/"code"/x for x in (
        "news_agent_batched_runtime_2025.py", "news_source_clock_2025.py", "news_event_calendar_2025.py",
        "news_event_risk_data_2025.py", "midcap_tools/retrieval.py", "midcap_standard_2025/benchmarks.py")]
    manifest = {"protocol_hash": digest(protocol()), "sources": {str(p): file_hash(p) for p in source},
                "python": sys.version, "cli_sha256": file_hash(Path(CLAUDE)),
                "dependencies": {p: importlib.metadata.version(p) for p in ("gepa", "dspy", "numpy", "pandas", "scipy", "pyarrow", "filelock")},
                "artifacts": {p: file_hash(out/p) for p in ("data_manifest.json", "embedding_manifest.json", "numerical_manifest.json", "calendar_features.json", "retrieval_review.json")}}
    locked_write(out / "PLAN.json", protocol())
    locked_write(out / "RUN_MANIFEST.json", manifest)
    return cases, controls, manifest


def publish(out, frame, complete):
    if frame.date.nunique() < 3:
        write(out / "live_evaluation/status.json", {"rows": len(frame), "origins": int(frame.date.nunique()),
              "note": "No regressions until at least three complete matched origins; no interim significance."})
        return
    result = summarize(frame, complete)
    write(out / "live_evaluation/metrics.json", result)
    frame.to_csv(out / "live_evaluation/regression_rows.csv", index=False)
    lines = ["# Frozen Evidence and Network Risk Experiment", "",
             f"Completed {frame.date.nunique()}/8 matched origins; {len(frame)}/80 firm-origins. Complete: {complete}.", "",
             "Pinned Sonnet 4.5; model-directed local tools; ten firms; no weekly prompt updates.",
             "Target: log mean squared decimal total return over the next five sessions.",
             "R2 below refits OLS on evaluation rows: explanatory, not frozen-coefficient forecasting.", "",
             "| Arm | Beyond GARCH (pp) | Beyond expanded controls (pp) | Adjusted expanded increment (pp) | LLM slope |",
             "|---|---:|---:|---:|---:|"]
    for arm, item in result["arms"].items():
        lines.append(f"| {arm} | {item['garch']['increment_pp']:+.3f} | {item['expanded']['increment_pp']:+.3f} | {item['expanded']['adjusted_increment_pp']:+.3f} | {item['expanded']['full']['coefficient'][-1]:+.3f} |")
    lines += ["", "Expanded controls: log GARCH, confirmed earnings indicator, unknown-calendar indicator, log RV1/RV5/RV22 and log(1+own event count).",
              "Both arms receive the same conventional information. The anchor is pre-August frozen OLS, not another learned LLM weight.", ""]
    if (out / "optimization/decision.json").exists():
        decision = read(out / "optimization/decision.json")
        lines += [f"Prompt selection: {decision['selection']['reason']}; changed: {decision['selection']['changed']}; distinct training candidates including seed: {decision['distinct_candidates']}.",
                  "A retained seed makes OWN_OPT identical to OWN_FIXED by construction; it is not evidence that an altered prompt helped.",
                  "Raw nested OLS R2 increments cannot be negative. A negative LLM slope contradicts the score's intended risk ordering and is reported explicitly.", ""]
    for name, item in result["contrasts"].items():
        lines.append(f"{name}: `{json.dumps(item)}`")
    lines += ["", "Full leave-one-firm/date diagnostics and regression coefficients are in metrics.json.",
              "Exploratory reused historical sample. Eight time clusters do not support strong confirmation. Bootstrap uncertainty conditions on generated scores and does not rerun prompt selection.",
              "The network uses the user-authorized June 6 effective-date snapshot assumption; acquisition vintage is not certified.",
              "This experiment does not separately isolate tools versus no tools."]
    (out / "live_evaluation/REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(out=OUT, acceptance_only=False):
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / "run.lock"), timeout=0):
        status = {"status": "preflight", "pid": os.getpid(), "stage": "preflight", "completed_origins": 0, "active_job": True}
        write(out / "progress.json", status)
        try:
            cases, controls, manifest = preflight(out)
            calendars, graph = read(out / "calendar_features.json"), Graph(read(out / "graph.json"))
            client = GuardedClient(out, LIMITS, CLAUDE)
            client.semaphore = threading.BoundedSemaphore(WORKERS)
            client.budget.start()
            runner = ForecastRunner(client, out, cases, calendars, graph, digest(manifest))
            train, valid, test = [[c for c in cases if c["split"] == s] for s in ("train", "valid", "test")]
            # Labels are not loaded until the source/tool acceptance has passed.
            status.update(stage="live_acceptance", status="running")
            write(out / "progress.json", status)
            acceptance_cases = [next(c for c in train if c["ticker"] == t and c["date"] == "2025-09-19") for t in ("NVDA", "CAT", "BBY")]
            acceptance = []
            for i, case in enumerate(acceptance_cases):
                row = runner.forecast(case, SEED, network=(i==0))
                if not row["tool_trace"] or not row["inspected_sources"]:
                    raise ValueError("Acceptance requires actual tool calls and source inspection")
                acceptance.append(row)
            write(out / "live_acceptance.json", {"pass": True, "labels_used": False, "records": acceptance})
            if acceptance_only:
                status.update(status="acceptance_complete", stage="ready")
                return
            development_labels = pd.read_parquet(out / "labels_sealed.parquet", filters=[("split", "!=", "test")])
            status.update(stage="optimization")
            write(out / "progress.json", status)
            decision = optimize(runner, train, valid, controls, development_labels)
            status.update(selected_changed=decision["selection"]["changed"])
            write(out / "progress.json", status)
            frames = []
            for day in TEST:
                status.update(stage="evaluation", origin=day)
                write(out / "progress.json", status)
                these = [c for c in test if c["date"] == day]
                arm_rows = {}
                for arm in ARMS:
                    policy = decision["policy"] if arm == "OWN_OPT" else SEED
                    arm_rows[arm] = runner.panel(these, policy, network=(arm == "NET_FIXED"))
                    locked_write(out / "evaluation" / day / (arm+".json"), arm_rows[arm])
                # Evaluator alone loads outcomes after all matched scores are saved.
                labels = pd.read_parquet(out / "labels_sealed.parquet", filters=[("date", "==", day)])
                frame = matched_frame(arm_rows["OWN_FIXED"], these, controls, labels).rename(columns={"score": "OWN_FIXED"})
                for arm in ("OWN_OPT", "NET_FIXED"):
                    mapping = {r["case_id"]: r["prediction"]["score"] for r in arm_rows[arm]}
                    frame[arm] = frame.case_id.map(mapping)
                event_keys = {c["case_id"]: sorted({key for arm in ARMS for r in arm_rows[arm]
                              if r["case_id"] == c["case_id"] for key in r["cited_event_keys"]}) for c in these}
                frame["event_keys"] = frame.case_id.map(event_keys)
                frames.append(frame)
                status["completed_origins"] = len(frames)
                write(out / "progress.json", status)
                publish(out, pd.concat(frames, ignore_index=True), complete=len(frames)==len(TEST))
            status.update(status="complete", stage="complete")
        except (BudgetStop, ProviderStop) as exc:
            status.update(status="stopped", error=str(exc))
            raise
        except Exception as exc:
            status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            status.update(active_job=False, updated_at=time.time())
            write(out / "progress.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--acceptance-only", action="store_true")
    args = parser.parse_args()
    run(args.out, args.acceptance_only)
