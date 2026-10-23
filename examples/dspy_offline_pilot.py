"""Offline DSPy/GEPA feasibility check using synthetic risk signals only.

Run with DSPy 3.4.0 and GEPA 0.1.4. No provider, MCP server, licensed data,
historical outcome, or repository production runner is accessed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import dspy
from dspy.lm15 import Message, Response, Usage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "code"))
from news_agent_batched_runtime_2025 import Engine  # noqa: E402


def synthetic_answer(prompt: str) -> str:
    flag = re.search(r"event_flag=(0|1)", prompt)
    source = re.search(r"source_id=(S[0-9]+)", prompt)
    if flag is None or source is None:
        raise AssertionError("Synthetic input was not passed to the DSPy module")
    use_signal = "Use the synthetic event flag" in prompt
    score = 0.5 if use_signal and flag.group(1) == "1" else 0.0
    # DSPy 3.4.0 ChatAdapter expects spaces inside these field markers.
    return (
        f"[[ ## score ## ]]\n{score}\n"
        f"[[ ## source_id ## ]]\n{source.group(1)}\n"
        "[[ ## completed ## ]]"
    )


class SyntheticRisk(dspy.Signature):
    """Return a residual log-risk score using the supplied synthetic signal."""

    signal: str = dspy.InputField(desc="Synthetic event flag and source ID")
    score: float = dspy.OutputField(desc="Residual log-risk score")
    source_id: str = dspy.OutputField(desc="ID of the synthetic source used")


class LocalEngine:
    """Deterministic stand-in for the model transport; never makes network calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.tool_requests = 0

    def complete(self, request):
        self.calls += 1
        if request.tools:
            self.tool_requests += 1
            raise AssertionError("This scoring-only pilot must not request tools")
        prompt = "\n".join(
            [str(request.system or "")]
            + [message.text or "" for message in request.messages]
        )
        return Response(
            id=None,
            model="custom/synthetic",
            message=Message.assistant(synthetic_answer(prompt)),
            finish_reason="stop",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


class LocalCliStub:
    """Exercises the repository's existing Engine without starting Claude CLI."""

    def __init__(self) -> None:
        self.calls = 0

    def call(self, prompt, purpose="score", nonce="", retry=False):
        del purpose, nonce, retry
        self.calls += 1
        return {
            "request_sha256": "synthetic-only",
            "answer": synthetic_answer(prompt),
            "envelope": {"uuid": "synthetic-only", "usage": {"input_tokens": 0, "output_tokens": 0}},
        }


class RiskScorer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(SyntheticRisk)

    def forward(self, signal: str):
        return self.predict(signal=signal)


def examples(start: int, flags: tuple[int, ...]):
    return [
        dspy.Example(
            signal=f"event_flag={flag}; source_id=S{start + index}",
            y=0.5 if flag else 0.0,
            expected_source=f"S{start + index}",
        ).with_inputs("signal")
        for index, flag in enumerate(flags)
    ]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.source_id != gold.expected_source:
        return dspy.Prediction(score=0.0, feedback="Source ID mismatch")
    loss = (gold.y - float(pred.score)) ** 2
    return dspy.Prediction(score=1.0 - loss, feedback=f"Squared error: {loss:.3f}")


def propose(*, candidate, reflective_dataset, components_to_update):
    # A fixed local proposal tests GEPA integration; it is not an LLM reflection.
    del reflective_dataset
    return {
        name: candidate[name] + " Use the synthetic event flag to set the score."
        for name in components_to_update
    }


def main() -> None:
    engine = LocalEngine()
    lm = dspy.LM("custom/synthetic", engine=engine, cache=False, num_retries=0)
    dspy.configure(lm=lm)
    train = examples(1, (0, 1, 0, 1))
    valid = examples(5, (1, 0))
    test = examples(7, (0, 1))
    seed = RiskScorer()

    def mean_score(program, rows):
        return sum(float(metric(row, program(signal=row.signal)).score) for row in rows) / len(rows)

    seed_valid = mean_score(seed, valid)
    optimizer = dspy.GEPA(
        metric=metric,
        instruction_proposer=propose,
        max_metric_calls=30,
        reflection_minibatch_size=2,
        candidate_selection_strategy="pareto",
        use_merge=False,
        skip_perfect_score=False,
        seed=7,
    )
    compiled = optimizer.compile(seed, trainset=train, valset=valid)
    compiled_valid = mean_score(compiled, valid)
    compiled_test = mean_score(compiled, test)
    cli_stub = LocalCliStub()
    cli_lm = dspy.LM("custom/synthetic", engine=Engine(cli_stub), cache=False, num_retries=0)
    with dspy.context(lm=cli_lm):
        bridge_prediction = compiled(signal="event_flag=1; source_id=S99")
    bridge_passed = float(bridge_prediction.score) == 0.5 and bridge_prediction.source_id == "S99"
    output = {
        "dspy_version": dspy.__version__,
        "train_cases": len(train),
        "validation_cases": len(valid),
        "heldout_cases": len(test),
        "seed_validation_score": round(seed_valid, 6),
        "compiled_validation_score": round(compiled_valid, 6),
        "compiled_heldout_score": round(compiled_test, 6),
        "instruction_changed": compiled.predict.signature.instructions != seed.predict.signature.instructions,
        "existing_engine_bridge_passed": bridge_passed,
        "cli_stub_calls": cli_stub.calls,
        "local_model_calls": engine.calls,
        "tool_requests": engine.tool_requests,
        "provider_calls": 0,
    }
    assert output["instruction_changed"], output
    assert compiled_valid > seed_valid, output
    assert compiled_test == 1.0, output
    assert bridge_passed and cli_stub.calls == 1, output
    assert engine.tool_requests == 0, output
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
