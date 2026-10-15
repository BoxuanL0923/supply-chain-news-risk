"""Individual-case GEPA residual-MSE search; distinct reflection and validation data."""
import json
import re
import numpy as np
import pandas as pd
from news_agent_batched_runtime_2025 import digest, read, write, locked_write
from .config import SEED


def eligible_policy(text, firms):
    return (isinstance(text, str) and 20 <= len(text.split()) <= 650 and
            not re.search(r"\b202[0-9][- /]\d|RTRS:|(?:always|automatically).{0,30}(?:resolved|priced in)", text, re.I) and
            not any(re.search(r"\b"+re.escape(t)+r"\b", text) for t in firms))


def choose(candidates, residuals, cases):
    ids = [c["case_id"] for c in cases]
    y = np.array([residuals[i] for i in ids])
    dates = np.array([c["date"] for c in cases])
    seed = candidates[0]
    seed_error = (y-np.array([seed["scores"][i] for i in ids]))**2
    audits, accepted = [], []
    for candidate in candidates[1:]:
        gain = seed_error-(y-np.array([candidate["scores"][i] for i in ids]))**2
        bydate = [float(gain[dates == d].mean()) for d in sorted(set(dates))]
        date_se = float(np.std(bydate, ddof=1)/np.sqrt(len(bydate)))
        changed = min(bydate) > 0 and float(gain.mean()) > date_se
        audits.append({"policy_hash": digest(candidate["policy"]), "gain": float(gain.mean()),
                       "by_date": bydate, "date_se": date_se, "accepted": changed})
        if changed:
            accepted.append((float(gain.mean()), candidate))
    selected = max(accepted, key=lambda x: x[0])[1] if accepted else seed
    return selected, {"audits": audits, "changed": selected is not seed,
                      "rule": "Positive paired residual-MSE gain on both validation origins and above one date-SE; heuristic, not a significance test"}


def reflection_context(runner, case, network):
    fields = ("source_id", "headline", "fact", "interpretation", "qualifications", "event_type", "status")
    context = {"sector": case["industry"], "state": case["state"],
               "anchor": runner.anchors[case["case_id"]], "evidence": {}}
    for name, enabled in (("own_common", False), ("network_supplement", True)):
        if enabled and not network:
            continue
        evidence = runner.evidence(case, enabled)
        context["evidence"][name] = {"events": [{k: e[k] for k in fields} for e in evidence["events"]],
                                    "missing": evidence["missing"], "coverage": evidence["coverage"]}
    return context


def optimize(runner, cases, network, panel):
    from gepa import optimize as gepa_optimize
    from gepa.core.adapter import EvaluationBatch
    from gepa.utils.stop_condition import MaxCandidateProposalsStopper
    name = "NET" if network else "OWN"
    out = runner.out / "optimization" / name
    if (out / "decision.json").exists():
        return read(out / "decision.json")
    train_cases = [c for c in cases if c["split"] == "train"]
    valid_cases = [c for c in cases if c["split"] == "valid"]
    labels = pd.concat([pd.read_parquet(runner.out/f"labels_{s}.parquet") for s in ("train", "valid")])
    residuals = {r.case_id: float(r.y-runner.anchors[r.case_id]) for r in labels.itertuples()}
    lookup = {c["case_id"]: c for c in cases if c["split"] != "test"}
    firms = sorted({c["ticker"] for c in cases})
    train = [{"case_id": c["case_id"]} for c in train_cases]
    valid = [{"case_id": c["case_id"]} for c in valid_cases]
    locked_write(out / "split.json", {"train": train, "valid": valid, "objective": "negative mean (training-OLS residual - raw LLM score)^2",
          "individual_cases": True, "pareto_selection": True, "reflection_minibatch": 16, "max_proposals": 2,
          "test_labels_accessed": False, "retrieval": "frozen; only interpretation instructions optimized"})
    policies = {digest(SEED): SEED}
    for path in sorted((out / "candidates").glob("*.json")):
        policy = read(path)["policy"]
        policies[digest(policy)] = policy
    # Reflection sees panel-wide seed diagnostics, not just the small mutation minibatch.
    seed_training = panel(train_cases, SEED, network)
    seed_scores = {r["case_id"]: r["score"] for r in seed_training}
    training_audit = {"n": len(train_cases), "grouped": {}}
    for field in ("date", "industry"):
        training_audit["grouped"][field] = {}
        for value in sorted({c[field] for c in train_cases}):
            ids = [c["case_id"] for c in train_cases if c[field] == value]
            target = np.asarray([residuals[i] for i in ids])
            scores = np.asarray([seed_scores[i] for i in ids])
            training_audit["grouped"][field][value] = {"n": len(ids), "seed_residual_mse": float(np.mean((target-scores)**2)),
                "zero_score_mse": float(np.mean(target**2)), "mean_error": float(np.mean(target-scores))}
    write(out / "training_audit.json", training_audit)

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            policy = candidate["interpretation"]
            good = eligible_policy(policy, firms) or policy == SEED
            if good:
                policies[digest(policy)] = policy
                write(out / "candidates" / (digest(policy)+".json"), {"policy": policy})
                rows = panel([lookup[b["case_id"]] for b in batch], policy, network)
                outputs = [{"case_id": r["case_id"], "score": r["score"], "prediction": r["prediction"]} for r in rows]
                scores = [-(residuals[r["case_id"]]-r["score"])**2 for r in rows]
            else:
                outputs = [{"case_id": b["case_id"], "error": "Invalid policy contract or identity-specific rule"} for b in batch]
                scores = [-1e9]*len(batch)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=outputs if capture_traces else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            examples = []
            for row in eval_batch.outputs:
                case = lookup[row["case_id"]]
                if case["split"] != "train":
                    raise ValueError("Validation/test feedback reached reflector")
                # Anonymize identity/date in feedback metadata; text remains historical evidence.
                examples.append({"Input": reflection_context(runner, case, network),
                    "Generated Output": row.get("prediction", row), "Feedback": {
                        "target_residual": residuals[row["case_id"]], "squared_error": -eval_batch.scores[len(examples)],
                        "rule": "One realized variance is noisy. Diagnose only patterns supported by several cases and counterexamples."}})
            return {key: [{"All_training_seed_diagnostics": training_audit}]+examples for key in components_to_update}

    def reflection(messages):
        return runner.client.call(
            "Revise the interpretation policy with at most two small, reusable changes supported by multiple cases. "
            "Keep the numerical target, OLS-anchor meaning and evidence contract. No ticker/company/date-specific "
            "rules, no numeric bands fitted to outcomes, no universal priced-in/resolution rule. "
            "Do not optimize a constant offset from one noisy week. Preserve valid starting instructions. "
            "All numerical feedback is training-only. Follow GEPA's requested fenced output format.\n"
            + json.dumps(messages, default=str), purpose="reflection")["answer"]

    class StratifiedSampler:
        def __init__(self):
            self.calls = 0

        def next_minibatch_ids(self, loader, state):
            rng = np.random.default_rng(20251001+self.calls)
            ids = list(loader.all_ids())
            entries = dict(zip(ids, loader.fetch(ids)))
            selected, used = [], set()
            dates = sorted({lookup[r["case_id"]]["date"] for r in entries.values()})
            for day in dates:
                available = [i for i in ids if lookup[entries[i]["case_id"]]["date"] == day]
                sectors = sorted({lookup[entries[i]["case_id"]]["industry"] for i in available})
                sectors = sectors[self.calls % len(sectors):] + sectors[:self.calls % len(sectors)]
                for sector in sectors[:4]:
                    choices = [i for i in available if lookup[entries[i]["case_id"]]["industry"] == sector
                               and lookup[entries[i]["case_id"]]["ticker"] not in used]
                    i = choices[int(rng.integers(len(choices)))]
                    selected.append(i)
                    used.add(lookup[entries[i]["case_id"]]["ticker"])
            self.calls += 1
            return selected

    result = gepa_optimize(seed_candidate={"interpretation": SEED}, trainset=train, valset=valid,
        adapter=Adapter(), reflection_lm=reflection, candidate_selection_strategy="pareto",
        batch_sampler=StratifiedSampler(), skip_perfect_score=False, use_merge=False,
        max_metric_calls=400, stop_callbacks=[MaxCandidateProposalsStopper(2)], seed=20251001,
        run_dir=str(out / "gepa"), cache_evaluation=True, raise_on_exception=True)
    write(out / "gepa_result.json", result.to_dict())
    # Ensure all proposed candidates are paired on exactly the same validation rows.
    candidates = []
    for policy in policies.values():
        rows = panel(valid_cases, policy, network)
        candidates.append({"policy": policy, "scores": {r["case_id"]: r["score"] for r in rows}})
    chosen, selection = choose(candidates, residuals, valid_cases)
    decision = {"policy": chosen["policy"], "selection": selection, "candidate_count": len(candidates),
                "test_labels_accessed": False, "frozen": True}
    locked_write(out / "decision.json", decision)
    return decision
