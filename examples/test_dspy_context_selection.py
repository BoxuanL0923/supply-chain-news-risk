"""Synthetic, provider-free contract tests for the DSPy context path."""

import sys
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import dspy
from dspy.lm15 import Message, Response, Usage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "code"))
from news_network_50_2025.context_selection import (  # noqa: E402
    ContextProgram, compile_context_program, openai_required_tool_call,
    runner_forecast_scorer, select_context, skill_instructions,
)
from news_network_50_2025.workflow import apply_context_program  # noqa: E402


def toy_events():
    return [
        {"source_id": "NOISE", "available_at": "2025-01-02T09:00:00", "headline": "Routine item", "event_type": "routine"},
        {"source_id": "RISK", "available_at": "2025-01-02T09:00:00", "headline": "Supplier plant closure", "event_type": "supplier disruption"},
        {"source_id": "FUTURE", "available_at": "2025-01-04T09:00:00", "headline": "Later disclosure", "event_type": "supplier disruption"},
    ]


class ContextSelectionTests(unittest.TestCase):
    def test_runner_scorer_uses_selected_events_and_fixed_policy(self):
        class FakeRunner:
            out = "toy-out"
            client = object()
            cases = [{"case_id": "CASE", "asof": "2025-01-03T12:00:00"}]
            anchors = {"CASE": 0.0}
            manifest = "toy-manifest"
            implementation_manifest = "toy-manifest"
            def score(self, case, policy, network, *, context_program=None):
                self.seen = (case["case_id"], policy, network)
                prediction = context_program(candidates=toy_events()[:2], asof=case["asof"])
                return {"score": float(prediction.events[0]["source_id"] == "RISK")}

        fake = FakeRunner()
        scorer = runner_forecast_scorer(fake, policy="fixed-policy", network=False, max_events=1)
        self.assertEqual(scorer("CASE", [toy_events()[1]]), 1.0)
        self.assertEqual(fake.seen, ("CASE", "fixed-policy", False))

    def test_compile_api_rejects_test_labels_before_search(self):
        rows = [{"case_id": "A", "split": "train", "candidates": toy_events(),
                 "asof": "2025-01-03T12:00:00", "y": 1.0},
                {"case_id": "B", "split": "test", "candidates": toy_events(),
                 "asof": "2025-01-03T12:00:00", "y": 1.0}]
        with self.assertRaises(ValueError):
            compile_context_program(rows, lambda case_id, events: 0.0, max_events=1)

    def test_openai_adapter_forces_and_executes_python_function(self):
        class FakeResponses:
            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(output=[SimpleNamespace(type="function_call", name="select_context",
                    arguments='{"ranked_ids":["C2","C1"]}')])
        fake = FakeResponses()
        events, receipt = openai_required_tool_call(fake, model="test-model", candidates=toy_events()[:2],
            asof="2025-01-03T12:00:00", max_events=1)
        self.assertEqual(fake.request["tool_choice"], {"type": "function", "name": "select_context"})
        self.assertFalse(fake.request["parallel_tool_calls"])
        self.assertIn("Use the supply-chain-context-selection skill", fake.request["instructions"])
        self.assertEqual(events[0]["source_id"], "RISK")
        self.assertTrue(receipt["tool_called"])

    def test_scoring_packet_uses_only_tool_selected_events(self):
        class StubProgram:
            max_events = 1
            def __call__(self, *, candidates, asof):
                events, receipt = select_context(candidates, ["RISK", "NOISE"],
                                                 asof=asof, max_events=1)
                return dspy.Prediction(events=events, receipt=receipt, ranked_ids=["RISK", "NOISE"])

        packet = {"own_common": {"events": toy_events()[:2], "missing": ["Supplier identity unknown"],
                                 "coverage": {"status": "partial"}}}
        filtered, receipt = apply_context_program(packet, StubProgram(), "2025-01-03T12:00:00")
        self.assertEqual([e["source_id"] for e in filtered["own_common"]["events"]], ["RISK"])
        self.assertEqual(filtered["own_common"]["missing"], packet["own_common"]["missing"])
        self.assertEqual(receipt["selected_ids"], ["RISK"])

    def test_scoring_packet_preserves_optimized_rank_order(self):
        class StubProgram:
            max_events = 2
            def __call__(self, *, candidates, asof):
                ids = ["RISK", "NOISE"]
                events, receipt = select_context(candidates, ids, asof=asof, max_events=2)
                return dspy.Prediction(events=events, receipt=receipt, ranked_ids=ids)

        packet = {"own_common": {"events": toy_events()[:2], "missing": [], "coverage": {}}}
        filtered, _ = apply_context_program(packet, StubProgram(), "2025-01-03T12:00:00")
        self.assertEqual([e["source_id"] for e in filtered["own_common"]["events"]], ["RISK", "NOISE"])

    def test_python_tool_filters_future_and_enforces_budget(self):
        selected, receipt = select_context(toy_events(), ["FUTURE", "RISK", "NOISE"],
                                           asof="2025-01-03T12:00:00", max_events=1)
        self.assertEqual([e["source_id"] for e in selected], ["RISK"])
        self.assertEqual(receipt["tool"], "select_context")
        self.assertEqual(receipt["excluded_future"], ["FUTURE"])

    def test_python_tool_fails_closed_on_unknown_and_duplicate_ids(self):
        for ranked in (["MADE_UP"], ["RISK", "RISK"], []):
            with self.subTest(ranked=ranked), self.assertRaises(ValueError):
                select_context(toy_events(), ranked, asof="2025-01-03T12:00:00", max_events=1)

    def test_skill_is_loaded_and_explicitly_invoked(self):
        instructions, digest = skill_instructions(ROOT / "skills" / "supply-chain-context-selection" / "SKILL.md")
        self.assertIn("Use the supply-chain-context-selection skill", instructions)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            skill_instructions(ROOT / "skills" / "supply-chain-news-risk" / "SKILL.md")

    def test_program_must_call_python_tool(self):
        class StubRanker:
            def __call__(self, **kwargs):
                self.task = kwargs["task"]
                self.candidates = kwargs["candidates"]
                return dspy.Prediction(ranked_ids="C2,C1")

        program = ContextProgram(max_events=1)
        program.rank = StubRanker()
        result = program(candidates=toy_events(), asof="2025-01-03T12:00:00")
        self.assertEqual([e["source_id"] for e in result.events], ["RISK"])
        self.assertTrue(result.receipt["tool_called"])
        self.assertIn("Use the supply-chain-context-selection skill", program.rank.task)
        self.assertNotIn("Supplier plant closure", program.rank.candidates)
        self.assertNotIn("FUTURE", program.rank.candidates)
        sensitive = toy_events()[:2]
        sensitive[1]["event_type"] = "Private customer contract 98765"
        program(candidates=sensitive, asof="2025-01-03T12:00:00")
        self.assertNotIn("Private customer", program.rank.candidates)
        sensitive[1]["event_type"] = "Confidential supplier shortage at private site"
        program(candidates=sensitive, asof="2025-01-03T12:00:00")
        self.assertIn("supplier disruption", program.rank.candidates)
        self.assertNotIn("Confidential", program.rank.candidates)

    def test_gepa_changes_selected_context_on_sealed_toy_split(self):
        class LocalEngine:
            def __init__(self):
                self.calls = 0
            def complete(self, request):
                self.calls += 1
                prompt = "\n".join([str(request.system or "")] + [m.text or "" for m in request.messages])
                ids = re.findall(r'"source_id": "(C[0-9]+)"', prompt)
                if "Prioritize supplier disruption" in prompt and "C2" in ids:
                    ids = ["C2"] + [sid for sid in ids if sid != "C2"]
                return Response(id=None, model="custom/synthetic", message=Message.assistant(
                    "[[ ## ranked_ids ## ]]\n" + ",".join(ids) + "\n[[ ## completed ## ]]"),
                    finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0))

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            # Frozen toy scorer: a selected supplier disruption predicts unit risk.
            forecast = float(pred.events[0]["event_type"] == "supplier disruption")
            loss = (gold.y - forecast) ** 2
            return dspy.Prediction(score=1.0 - loss, feedback=f"Toy forecast squared error: {loss:.1f}")

        rows = [dspy.Example(candidates=toy_events(), asof="2025-01-03T12:00:00", y=1.0)
                .with_inputs("candidates", "asof") for _ in range(8)]
        engine = LocalEngine()
        with dspy.context(lm=dspy.LM("custom/synthetic", engine=engine, cache=False, num_retries=0)):
            seed = ContextProgram(max_events=1)
            seed_score = metric(rows[4], seed(**rows[4].inputs())).score
            search_rows = [{"case_id": f"S{i}", "split": "train" if i < 4 else "valid",
                            "candidates": toy_events(), "asof": "2025-01-03T12:00:00", "y": 1.0}
                           for i in range(6)]
            compiled, report = compile_context_program(search_rows,
                lambda case_id, events: float(events[0]["event_type"] == "supplier disruption"),
                max_events=1, max_metric_calls=30)
            heldout = [metric(row, compiled(**row.inputs())).score for row in rows[6:]]
        self.assertEqual(seed_score, 0.0)
        self.assertEqual(heldout, [1.0, 1.0])
        self.assertNotEqual(compiled.rank.signature.instructions, seed.rank.signature.instructions)
        self.assertTrue(report["selected_candidate"])
        self.assertEqual(report["seed_validation_negative_mse"], -1.0)
        self.assertEqual(report["candidate_validation_negative_mse"], 0.0)
        self.assertFalse(report["test_labels_accessed"])
        self.assertGreater(engine.calls, 0)


if __name__ == "__main__":
    unittest.main()
