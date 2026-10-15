"""Development-only GEPA panel optimization with one-time validation."""
import json
import re
import numpy as np
import pandas as pd

from news_agent_batched_runtime_2025 import digest, read, write, locked_write
from .config import SEED, FIRMS, LIMITS, SEED_NUMBER, MAX_ACTION_ROUNDS
from .metrics import matched_frame, metric, sensitivity


def inspected_passages(value, cited):
    """Keep real evidence from search, direct reads and nested state-tool results."""
    result = {}
    def visit(item):
        if isinstance(item, dict):
            if item.get("source_id") in cited and isinstance(item.get("text"), str) and item.get("citation_eligible", True):
                result[(item["source_id"], item.get("chunk", 0))] = item
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return list(result.values())


def valid_policy(policy):
    if not isinstance(policy, dict) or set(policy) != set(SEED):
        return False
    for value in policy.values():
        if not isinstance(value, str) or not value.strip() or len(value.split()) > 1800:
            return False
        if re.search(r"\b(?:"+"|".join(FIRMS)+r")\b|\b2025[-/]\d|RTRS:", value):
            return False
    return True


def choose_validation(evaluated):
    """One-SE preference for the seed; precision is a heuristic, not a p-value."""
    seed = evaluated[0]
    best = min(evaluated, key=lambda x: x["metric"]["profiled_mse"])
    if best is seed:
        return seed, {"reason": "seed_best", "changed": False}
    errors_seed = np.asarray(seed["metric"]["full"]["errors"])**2
    errors_best = np.asarray(best["metric"]["full"]["errors"])**2
    dates = np.asarray(best["dates"])
    gains = np.asarray([np.mean((errors_seed-errors_best)[dates == d]) for d in sorted(set(dates))])
    se = float(np.std(gains, ddof=1)/np.sqrt(len(gains))) if len(gains)>1 else float("inf")
    gain = float(np.mean(gains))
    changed = gain > se
    return (best if changed else seed), {"reason": "one_SE_preference_for_seed", "changed": changed,
            "paired_date_mean_mse_gain": gain, "date_standard_error": se,
            "warning": "Only four validation origins; this is regularized selection, not significance."}


def optimize(runner, train_cases, valid_cases, controls, development_labels):
    from gepa import optimize as gepa_optimize
    from gepa.core.adapter import EvaluationBatch
    from gepa.utils.stop_condition import MaxCandidateProposalsStopper, NoImprovementStopper
    out = runner.out / "optimization"
    if (out / "decision.json").exists():
        return read(out / "decision.json")
    if set(development_labels.split) - {"train", "valid"}:
        raise ValueError("Evaluation labels supplied to optimization")
    train_labels = development_labels[development_labels.split.eq("train")]
    valid_labels = development_labels[development_labels.split.eq("valid")]
    if pd.to_datetime(train_labels.available_at, utc=True).max() >= min(pd.Timestamp(c["asof"]) for c in valid_cases):
        raise ValueError("Training outcomes immature at validation")
    locked_write(out / "split.json", {"train": [c["case_id"] for c in train_cases], "valid": [c["case_id"] for c in valid_cases],
                 "objective": "negative pooled expanded-controls-plus-raw-score OLS MSE", "evaluation_labels_sent": False,
                 "validation_access": "once after training-only nominations; never sent to reflection"})
    completed = {p.stem: read(p) for p in (out / "training").glob("*.json")}

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            key = digest(candidate)
            if key not in completed:
                if not valid_policy(candidate):
                    result = {"policy": candidate, "eligible": False, "error": "Invalid contract, identity-specific rule, or oversized module"}
                else:
                    records = runner.panel(train_cases, candidate)
                    frame = matched_frame(records, train_cases, controls, train_labels)
                    result = {"policy": candidate, "eligible": True, "metric": metric(frame),
                              "sensitivity": sensitivity(frame), "records": records}
                completed[key] = result
                write(out / "training" / (key+".json"), result)
            rec = completed[key]
            score = -rec["metric"]["profiled_mse"] if rec["eligible"] else -1e12
            return EvaluationBatch(outputs=[{"policy_hash": key} for _ in batch], scores=[score for _ in batch],
                                   trajectories=[{"policy_hash": key} for _ in batch] if capture_traces else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            rec = completed[digest(candidate)]
            if not rec["eligible"]:
                return {k: [{"Feedback": rec["error"]}] for k in components_to_update}
            frame = matched_frame(rec["records"], train_cases, controls, train_labels)
            errors = np.asarray(rec["metric"]["full"]["errors"])
            # Include ordinary and difficult cases, both error signs and all dates.
            indices = sorted(set(int(i) for i in np.linspace(0, len(frame)-1, 8).astype(int)) |
                             {int(np.argmax(errors)), int(np.argmin(errors))})
            by_id = {r["case_id"]: r for r in rec["records"]}
            cases_by_id = {c["case_id"]: c for c in train_cases}
            examples = []
            for i in indices:
                row = frame.iloc[i]
                record = by_id[row.case_id]
                cited = {r["source_id"] for r in record["prediction"]["evidence"]}
                traces = inspected_passages(record["tool_trace"], cited)
                examples.append({"Inputs": {"case": cases_by_id[row.case_id], "inspected_evidence": traces[:4]},
                    "Generated Outputs": record["prediction"],
                    "Feedback": {"observed_log_risk": float(row.y), "profiled_regression_error": float(errors[i]),
                                 "factual_audit": "Validate source clocks, pending/resolved status and quoted qualifiers; outcome alone is not a factual error.",
                                 "uncertainty": "One realization is noisy. Require cross-event support and counterexamples."}})
            history = [{"hash": k, "eligible": v["eligible"], "metric": v.get("metric", {}).get("profiled_mse"),
                        "policy": v["policy"]} for k, v in completed.items()]
            summary = {"profiled_mse": rec["metric"]["profiled_mse"], "increment_pp": rec["metric"]["increment_pp"],
                       "sensitivity": rec["sensitivity"], "prior_candidates_including_rejections": history}
            return {k: [{"TrainingPanelSummary": summary}]+examples for k in components_to_update}

    def reflection(messages):
        prompt = ("Improve ONE reusable module using GEPA's requested output format. Preserve valid existing rules. "
                  "Candidate selection targets pooled R2 beyond logGARCH, calendar, lagged RV and news-count controls. "
                  "A global score offset or rescaling cannot help. Changes must be grounded in inspected sources and "
                  "generalize across firms, quiet periods and event episodes. No firm names, dates or memorized outcomes "
                  "in the policy. Do not equate a large forecast error with a predictable event. The tools, schema, "
                  "allowed information, arithmetic and clocks are fixed. Diagnose before adding instructions; avoid "
                  "duplicating rejected proposals. Explain factual versus statistical failures in feedback internally, "
                  "but return only the requested revised module. All labels below are training labels.\n" + json.dumps(messages, default=str))
        return runner.client.call(prompt, purpose="reflection")["answer"]

    def stop_for_reserve(state):
        usage = read(runner.client.budget.path)
        durations = [read(p)["seconds"] for p in (runner.out / "responses").glob("*.json")]
        per = max(15., float(np.mean(durations))*1.25) if durations else 60.
        requests = [len(r["requests"]) for c in completed.values() if c["eligible"] for r in c["records"]]
        per_case = min(MAX_ACTION_ROUNDS+2, max(2., 1.15*float(np.mean(requests)))) if requests else MAX_ACTION_ROUNDS+1
        # Reserve all comparisons and a complete next candidate using observed
        # agent length. The ledger separately enforces hard, not estimated, caps.
        proposals = sum(c["eligible"] and c["policy"] != SEED for c in completed.values())
        nominees = min(3, proposals+2)
        cases = nominees*len(valid_cases)+3*80+6*nominees+len(train_cases)
        calls = int(np.ceil(cases*per_case))+20
        future_seconds = calls * per / 4 + 180
        mean_cost = max(.02, usage["list_price_usd"]/max(1, usage["attempts"])*1.25)
        # Do not double-count call-length margins when reserving tokens. Network
        # cases receive extra room because their evidence contexts are longer.
        token_samples = []
        for c in completed.values():
            if not c["eligible"]:
                continue
            for record in c["records"]:
                responses = [read(runner.out/"responses"/(key+".json")) for key in record["requests"]
                             if (runner.out/"responses"/(key+".json")).exists()]
                if responses:
                    token_samples.append((sum(sum(r["envelope"]["usage"].get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")) for r in responses),
                                          sum(r["envelope"]["usage"].get("output_tokens", 0) for r in responses)))
        reserve_input = reserve_output = 0
        if token_samples:
            average_input, average_output = np.mean(token_samples, axis=0)
            reserve_input = int((cases+40)*average_input*1.10+50000)
            reserve_output = int((cases+20)*average_output*1.10+10000)
        stop = (usage["attempts"]+calls > LIMITS["hard_total_provider_attempts"] or
                runner.client.budget.remaining_seconds() < future_seconds or
                usage["list_price_usd"] + calls*mean_cost > LIMITS["provider_list_price_stop_usd"] or
                usage.get("input_tokens", 0)+reserve_input > LIMITS["input_token_stop"] or
                usage.get("output_tokens", 0)+reserve_output > LIMITS["output_token_stop"])
        write(out / "completion_reserve.json", {"stop_search": stop, "reserved_attempts": calls,
              "estimated_seconds": future_seconds, "cost_estimate_usd": calls*mean_cost,
              "reserved_input_tokens": reserve_input, "reserved_output_tokens": reserve_output,
              "observed_calls_per_case_with_margin": per_case,
              "note": "Observed length/cost plus margins protect completion; hard runtime ledger remains authoritative."})
        return stop

    already = len([r for r in completed.values() if r["policy"] != SEED])
    result = gepa_optimize(seed_candidate=SEED, trainset=[{"panel": "training"}], valset=[{"panel": "training"}],
        adapter=Adapter(), reflection_lm=reflection, candidate_selection_strategy="current_best",
        reflection_minibatch_size=1, skip_perfect_score=False, use_merge=False, max_metric_calls=36,
        reflection_prompt_template=("Current module:\n```\n<curr_param>\n```\nTraining evidence:\n<side_info>\n"
            "Make a targeted reusable improvement. No firm/date rules, retrospective facts or output changes. "
            "Return the complete revised module inside one triple-backtick block."),
        stop_callbacks=[MaxCandidateProposalsStopper(max(0, 6-already)), NoImprovementStopper(3), stop_for_reserve],
        run_dir=str(out / "gepa"), seed=SEED_NUMBER, raise_on_exception=True, cache_evaluation=True)
    write(out / "gepa_result.json", result.to_dict())
    eligible = [r for r in completed.values() if r["eligible"] and r["policy"] != SEED]
    nominees = [SEED]+[r["policy"] for r in sorted(eligible, key=lambda r: r["metric"]["profiled_mse"])[:2]]
    locked_write(out / "nominations.json", {"policies": nominees, "validation_feedback_used": False})
    repeat_cases = [c for c in train_cases if c["date"] in ("2025-08-08", "2025-09-19") and c["ticker"] in ("NVDA", "CAT", "BBY")]
    repeatability = []
    for policy in nominees:
        first = runner.panel(repeat_cases, policy)
        second = runner.panel(repeat_cases, policy, nonce="development-repeat-1")
        repeatability.append({"policy_hash": digest(policy), "pairs": [{"case_id": a["case_id"],
            "first": a["prediction"]["score"], "second": b["prediction"]["score"]} for a, b in zip(first, second)]})
    write(runner.out / "repeatability.json", {"stage": "before_validation", "selection_metric_unchanged": True,
          "note": "Fixed training-only subset; variability disclosed, no outcome-selected averaging or redraws.", "records": repeatability})
    evaluated = []
    for policy in nominees:
        records = runner.panel(valid_cases, policy)
        frame = matched_frame(records, valid_cases, controls, valid_labels)
        evaluated.append({"policy": policy, "metric": metric(frame), "dates": frame.date.tolist(),
                          "sensitivity": sensitivity(frame), "records": records})
    locked_write(out / "validation_once.json", evaluated)
    chosen, selection = choose_validation(evaluated)
    decision = {"policy": chosen["policy"], "policy_hash": digest(chosen["policy"]), "selection": selection,
                "distinct_candidates": len(completed), "validation_origin_count": len(set(chosen["dates"])),
                "evaluation_labels_used": False, "frozen": True}
    locked_write(out / "decision.json", decision)
    return decision
