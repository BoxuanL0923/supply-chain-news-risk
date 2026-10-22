"""DSPy-ranked evidence, enforced by a deterministic point-in-time Python tool."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import dspy


SKILL_PATH = Path(__file__).resolve().parents[3] / "skills" / "supply-chain-context-selection" / "SKILL.md"
SKILL_NAME = "supply-chain-context-selection"
CATEGORIES = {"supplier disruption", "customer demand", "shared bottleneck",
              "competitor substitution", "routine", "macro", "other"}


def _category(value) -> str:
    text = str(value or "").lower()
    if text.strip() in CATEGORIES:
        return text.strip()
    for pattern, category in (
        (r"\bsupplier|\bshortage|\bplant (?:closure|outage)|\bproduction halt", "supplier disruption"),
        (r"\bcustomer|\bdemand|\border cancellation", "customer demand"),
        (r"\bbottleneck|\bport|\bpower outage|\bshipping delay", "shared bottleneck"),
        (r"\bcompetitor|\bsubstitut", "competitor substitution"),
        (r"\broutine|\bfiling", "routine"),
        (r"\bmacro|\binflation|\brate decision", "macro"),
    ):
        if re.search(pattern, text):
            return category
    return "unknown"


def skill_instructions(path: Path = SKILL_PATH) -> tuple[str, str]:
    """Load a reviewed local skill; reject a different skill or broken header."""
    raw = Path(path).read_text(encoding="utf-8")
    if not raw.startswith("---\n") or f"name: {SKILL_NAME}\n" not in raw.split("---", 2)[1]:
        raise ValueError("Invalid context-selection skill")
    if f"Use the {SKILL_NAME} skill" not in raw:
        raise ValueError("Missing explicit skill invocation")
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo is None else parsed.astimezone(timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid evidence timestamp") from exc


def _model_metadata(candidates: list[dict], asof: str):
    """No raw headline, free text, source ID, or exact timestamp reaches a model."""
    cutoff = _time(asof)
    metadata, alias_to_id = [], {}
    for row in candidates:
        available = _time(row.get("available_at"))
        if available >= cutoff:
            continue
        alias = f"C{len(metadata) + 1}"
        metadata.append({"source_id": alias, "event_type": _category(row.get("event_type")),
                         "age_days": (cutoff - available).days})
        alias_to_id[alias] = row["source_id"]
    if not metadata:
        raise ValueError("No eligible context candidates")
    return metadata, alias_to_id


def _resolve_aliases(ids, alias_to_id):
    if not isinstance(ids, list) or any(sid not in alias_to_id for sid in ids):
        raise ValueError("Unknown model source alias")
    return [alias_to_id[sid] for sid in ids]


def select_context(candidates: list[dict], ranked_ids: list[str], *, asof: str, max_events: int):
    """The only route from model ranking to forecast evidence."""
    cutoff = _time(asof)
    if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 1:
        raise ValueError("Invalid context budget")
    if not isinstance(ranked_ids, list) or not ranked_ids or any(not isinstance(s, str) or not s for s in ranked_ids):
        raise ValueError("Missing ranked IDs")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Duplicate ranked IDs")
    by_id = {}
    for event in candidates:
        if not isinstance(event, dict) or not isinstance(event.get("source_id"), str) or not event["source_id"]:
            raise ValueError("Invalid candidate")
        sid = event["source_id"]
        if sid in by_id:
            raise ValueError("Duplicate candidate ID")
        _time(event.get("available_at"))
        by_id[sid] = event
    if set(ranked_ids) - by_id.keys():
        raise ValueError("Unknown ranked ID")
    future = [sid for sid in ranked_ids if _time(by_id[sid]["available_at"]) >= cutoff]
    selected = [by_id[sid] for sid in ranked_ids if sid not in future][:max_events]
    receipt = {"tool": "select_context", "tool_called": True,
               "selected_ids": [e["source_id"] for e in selected],
               "excluded_future": future, "candidate_count": len(candidates), "max_events": max_events}
    return selected, receipt


def openai_required_tool_call(responses, *, model: str, candidates: list[dict], asof: str,
                              max_events: int, ranking_instruction: str = ""):
    """Optional Responses adapter: force one function call, then execute Python locally.

    Only local aliases, age in days, and allowlisted event categories enter this request.
    The caller must ensure these fields are approved for its provider.
    """
    skill_text, skill_hash = skill_instructions()
    metadata, alias_to_id = _model_metadata(candidates, asof)
    response = responses.create(
        model=model,
        instructions=f"Use the {SKILL_NAME} skill.\n{skill_text}\n{ranking_instruction}\n"
                     "Call select_context exactly once with ranked_ids. No final answer before the tool call.",
        input=json.dumps({"candidates": metadata, "max_events": max_events}),
        tools=[{"type": "function", "name": "select_context",
                "description": "Rank eligible source IDs for the local Python context-selection tool",
                "strict": True,
                "parameters": {"type": "object", "properties": {
                    "ranked_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["ranked_ids"], "additionalProperties": False}}],
        tool_choice={"type": "function", "name": "select_context"},
        parallel_tool_calls=False,
    )
    calls = [item for item in response.output if item.type == "function_call"]
    if len(calls) != 1 or calls[0].name != "select_context":
        raise ValueError("Required Python tool call missing")
    try:
        args = json.loads(calls[0].arguments)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid tool arguments") from exc
    if set(args) != {"ranked_ids"}:
        raise ValueError("Unexpected tool arguments")
    ranked_ids = _resolve_aliases(args["ranked_ids"], alias_to_id)
    events, receipt = select_context(candidates, ranked_ids, asof=asof, max_events=max_events)
    return events, {**receipt, "skill_sha256": skill_hash, "tool_choice": "forced_function"}


class RankEvidence(dspy.Signature):
    """Rank source IDs for five-session supply-chain risk; never invent an ID."""

    task: str = dspy.InputField(desc="Explicit skill invocation and fixed instructions")
    candidates: str = dspy.InputField(desc="Eligible inspected event metadata and IDs")
    ranked_ids: str = dspy.OutputField(desc="Comma-separated source IDs, highest relevance first")


class ContextProgram(dspy.Module):
    """A DSPy-optimizable ranker whose output must pass the Python tool."""

    def __init__(self, max_events: int, skill_path: Path = SKILL_PATH):
        super().__init__()
        self.max_events = max_events
        self.skill_text, self.skill_hash = skill_instructions(skill_path)
        self.rank = dspy.Predict(RankEvidence)

    def forward(self, candidates: list[dict], asof: str):
        metadata, alias_to_id = _model_metadata(candidates, asof)
        prediction = self.rank(task=f"Use the {SKILL_NAME} skill.\n{self.skill_text}",
                               candidates=json.dumps(metadata, ensure_ascii=True))
        aliases = [part.strip() for part in prediction.ranked_ids.split(",")]
        ids = _resolve_aliases(aliases, alias_to_id)
        events, receipt = select_context(candidates, ids, asof=asof, max_events=self.max_events)
        return dspy.Prediction(events=events, receipt={**receipt, "skill_sha256": self.skill_hash},
                               ranked_ids=ids)


def compile_context_program(rows, forecast_score, *, max_events: int,
                            instruction_proposer=None, max_metric_calls: int = 30):
    """Optimize actual selected context against a fixed forecast scorer.

    ``rows`` contains training and validation cases only. ``forecast_score`` is
    a callback ``(case_id, selected_events) -> residual forecast``. The caller
    configures the DSPy LM and supplies point-in-time inspected events.
    """
    if not rows or any(row.get("split") not in {"train", "valid"} for row in rows):
        raise ValueError("Context search accepts only train/validation rows")
    if any(set(row) != {"case_id", "split", "candidates", "asof", "y"} for row in rows):
        raise ValueError("Invalid context optimization row")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate optimization case")
    train = [dspy.Example(**row).with_inputs("candidates", "asof")
             for row in rows if row["split"] == "train"]
    valid = [dspy.Example(**row).with_inputs("candidates", "asof")
             for row in rows if row["split"] == "valid"]
    if not train or not valid:
        raise ValueError("Train and validation rows required")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        forecast = float(forecast_score(gold.case_id, pred.events))
        loss = (float(gold.y) - forecast) ** 2
        return dspy.Prediction(score=-loss, feedback=f"Squared forecast error: {loss:.6f}")

    def default_proposer(*, candidate, reflective_dataset, components_to_update):
        # No raw example, label, or licensed text is sent to a reflection model.
        del reflective_dataset
        categories = ("supplier disruption", "customer demand", "shared bottleneck",
                      "competitor substitution")
        return {name: candidate[name] + " Prioritize " + next(
            (category for category in categories if "Prioritize " + category not in candidate[name]),
            categories[0]) + " over routine news."
                for name in components_to_update}

    seed = ContextProgram(max_events)
    def mean_score(program, examples):
        return sum(float(metric(row, program(**row.inputs())).score) for row in examples) / len(examples)

    seed_validation = mean_score(seed, valid)
    compiled = dspy.GEPA(
        metric=metric, instruction_proposer=instruction_proposer or default_proposer,
        max_metric_calls=max_metric_calls, reflection_minibatch_size=min(2, len(train)),
        candidate_selection_strategy="pareto", use_merge=False,
        skip_perfect_score=False, seed=7,
    ).compile(seed, trainset=train, valset=valid)
    optimized_validation = mean_score(compiled, valid)
    selected = compiled if optimized_validation > seed_validation else seed
    report = {"train_cases": len(train), "validation_cases": len(valid),
              "seed_validation_negative_mse": seed_validation,
              "candidate_validation_negative_mse": optimized_validation,
              "selected_candidate": selected is compiled,
              "test_labels_accessed": False, "max_metric_calls": max_metric_calls}
    return selected, report


def runner_forecast_scorer(runner, *, policy: str, network: bool, max_events: int):
    """Adapt the existing Runner's fixed scorer to the DSPy context metric."""
    case_by_id = {case["case_id"]: case for case in runner.cases}

    def score(case_id: str, selected_events: list[dict]) -> float:
        selected_ids = [event["source_id"] for event in selected_events]

        class FixedContext:
            def __init__(self):
                self.max_events = max_events

            def __call__(self, *, candidates, asof):
                events, receipt = select_context(candidates, selected_ids, asof=asof,
                                                 max_events=max_events)
                return dspy.Prediction(events=events, receipt=receipt, ranked_ids=selected_ids)

        return float(runner.score(case_by_id[case_id], policy, network,
                                  context_program=FixedContext())["score"])

    return score
