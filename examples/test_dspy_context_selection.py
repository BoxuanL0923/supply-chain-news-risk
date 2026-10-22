"""Synthetic, provider-free contract tests for the DSPy context path."""

import sys
import re
import json
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
from news_network_50_2025.domain_signals import finance_graph_signal  # noqa: E402
from news_network_50_2025.workflow import apply_context_program  # noqa: E402


def toy_events():
    return [
        {"source_id": "NOISE", "available_at": "2025-01-02T09:00:00", "headline": "Routine item", "event_type": "routine"},
        {"source_id": "RISK", "available_at": "2025-01-02T09:00:00", "headline": "Supplier plant closure", "event_type": "supplier disruption"},
        {"source_id": "FUTURE", "available_at": "2025-01-04T09:00:00", "headline": "Later disclosure", "event_type": "supplier disruption"},
    ]


class StubToolAgent:
    def __init__(self, program, aliases="C1"):
        self.program, self.aliases = program, aliases

    def __call__(self, **kwargs):
        self.metadata = kwargs["candidates"]
        self.task = kwargs["task"]
        self.program.use_supply_chain_context_skill()
        self.program.select_context_tool(self.aliases.split(","))
        return dspy.Prediction(ranked_ids=self.aliases)


class ContextSelectionTests(unittest.TestCase):
    def test_dspy_agent_requires_skill_call_before_python_selection(self):
        class NoToolAgent:
            def __call__(self, **kwargs):
                return dspy.Prediction(ranked_ids="C1")

        program = ContextProgram(max_events=1)
        program.rank = NoToolAgent()
        with self.assertRaisesRegex(ValueError, "Skill.*select_context"):
            program(candidates=toy_events()[:2], asof="2025-01-03T12:00:00")

        class WrongOrderAgent:
            def __call__(self, **kwargs):
                program.select_context_tool(["C2"])

        program.rank = WrongOrderAgent()
        with self.assertRaisesRegex(ValueError, "Skill must be invoked"):
            program(candidates=toy_events()[:2], asof="2025-01-03T12:00:00")

        class RecoveredWrongOrderAgent:
            def __call__(self, **kwargs):
                try:
                    program.select_context_tool(["C2"])
                except ValueError:
                    pass
                program.use_supply_chain_context_skill()
                program.select_context_tool(["C2"])
                return dspy.Prediction(ranked_ids="C2")

        program.rank = RecoveredWrongOrderAgent()
        with self.assertRaisesRegex(ValueError, "Skill and select_context"):
            program(candidates=toy_events()[:2], asof="2025-01-03T12:00:00")

        class ToolAgent:
            def __call__(self, **kwargs):
                self.skill = program.use_supply_chain_context_skill()
                self.result = program.select_context_tool(["C2", "C1"])
                return dspy.Prediction(ranked_ids="C2,C1")

        program.rank = ToolAgent()
        result = program(candidates=toy_events()[:2], asof="2025-01-03T12:00:00")
        self.assertEqual([e["source_id"] for e in result.events], ["RISK"])
        self.assertEqual(result.receipt["tool_sequence"],
                         ["use_supply_chain_context_skill", "select_context"])
        self.assertEqual(result.receipt["skill_sha256"], skill_instructions()[1])
        self.assertEqual(result.receipt["tool_choice"], "dspy_react")

        class FalseTraceAgent(ToolAgent):
            def __call__(self, **kwargs):
                super().__call__(**kwargs)
                return dspy.Prediction(trajectory={"tool_name_0": "finish"})

        program.rank = FalseTraceAgent()
        with self.assertRaisesRegex(ValueError, "Unexpected selector tool sequence"):
            program(candidates=toy_events()[:2], asof="2025-01-03T12:00:00")

    def test_direct_supplier_shock_is_a_hypothesis_not_a_verified_channel(self):
        row = {"event_type": "supplier disruption", "status": "ongoing",
               "_scope": "network_supplement", "verified_paths": [{"hops": 1,
               "vintage": "2025-06-06", "steps": [{"neighbor_role": "supplier",
               "edge_id": "PRIVATE-EDGE", "revenue_pct": 72.5}]}]}
        signal = finance_graph_signal(row, "2025-10-03T12:00:00")
        self.assertEqual(signal["path_pattern"], "direct_supplier")
        self.assertEqual(signal["mechanism_hint"], "input_shock_hypothesis")
        self.assertEqual(signal["event_state"], "ongoing")
        self.assertEqual(signal["channel_status"], "path_only")
        self.assertEqual(signal["path_count_bin"], "one")
        self.assertNotIn("PRIVATE-EDGE", str(signal))
        self.assertNotIn("72.5", str(signal))

    def test_two_hop_paths_preserve_role_sequence_and_mechanism_gap(self):
        row = {"event_type": "supplier disruption", "status": "resolved",
               "_scope": "network_supplement", "verified_paths": [{"hops": 2,
               "vintage": "2025-06-06", "steps": [{"neighbor_role": "supplier"},
               {"neighbor_role": "supplier"}]}]}
        signal = finance_graph_signal(row, "2025-10-03T12:00:00")
        self.assertEqual(signal["path_pattern"], "supplier_to_supplier")
        self.assertEqual(signal["mechanism_hint"], "intermediate_product_unverified")
        self.assertEqual(signal["event_state"], "resolved")

    def test_three_hop_motif_reaches_ranker_as_unverified_chain(self):
        row = {"source_id": "THIRD", "available_at": "2025-10-02T09:00:00",
               "event_type": "supplier disruption", "_scope": "network_supplement",
               "verified_paths": [{"hops": 3, "vintage": "2025-06-06",
               "steps": [{"neighbor_role": "supplier", "edge_id": "E1"},
                         {"neighbor_role": "supplier", "edge_id": "E2"},
                         {"neighbor_role": "supplier", "edge_id": "E3"}]}]}
        program = ContextProgram(max_events=1)
        program.rank = StubToolAgent(program)
        program(candidates=[row], asof="2025-10-03T12:00:00")
        self.assertIn('"hops": 3', program.rank.metadata)
        self.assertIn('"path_pattern": "supplier_to_supplier_to_supplier"', program.rank.metadata)
        self.assertIn('"mechanism_hint": "intermediate_product_unverified"', program.rank.metadata)
        self.assertIn('"finance_hypothesis": "upstream_input_risk"', program.rank.metadata)
        self.assertIn('"role_motif": "all_upstream"', program.rank.metadata)
        self.assertIn('"evidence_gap": "intermediate_product_continuity"', program.rank.metadata)
        self.assertIn('"channel_status": "path_only"', program.rank.metadata)
        self.assertNotIn("E3", program.rank.metadata)

    def test_mixed_shortest_paths_and_co_mentions_cannot_become_channels(self):
        base = {"event_type": "customer demand", "_scope": "network_supplement"}
        direct = lambda role: {"hops": 1, "vintage": "2025-06-06",
                               "steps": [{"neighbor_role": role}]}
        mixed = finance_graph_signal({**base, "verified_paths": [direct("supplier"),
                direct("customer")]}, "2025-10-03T12:00:00")
        self.assertEqual(mixed["path_pattern"], "mixed_shortest_paths")
        self.assertEqual(mixed["mechanism_hint"], "ambiguous_path_hypothesis")
        self.assertEqual(mixed["path_count_bin"], "multiple")
        comention = finance_graph_signal({**base, "verified_paths": [{"type": "comention",
                "vintage": "2025-07-31", "hops": 1, "steps": []}]}, "2025-10-03T12:00:00")
        self.assertEqual(comention["mechanism_hint"], "retrieval_only")
        self.assertEqual(comention["channel_status"], "retrieval_lead")

    def test_unknown_finance_status_fails_closed_and_model_receives_only_bins(self):
        row = {"source_id": "SENSITIVE", "available_at": "2025-10-02T09:00:00",
               "event_type": "customer demand", "status": "Private counterparty 123",
               "_scope": "network_supplement", "verified_paths": [{"hops": 1,
               "vintage": "2025-06-06", "steps": [{"neighbor_role": "customer"}]}]}
        program = ContextProgram(max_events=1)
        program.rank = StubToolAgent(program)
        program(candidates=[row], asof="2025-10-03T12:00:00")
        self.assertIn('"mechanism_hint": "output_demand_hypothesis"', program.rank.metadata)
        self.assertIn('"event_state": "unknown"', program.rank.metadata)
        self.assertNotIn("Private counterparty", program.rank.metadata)
        self.assertNotIn("SENSITIVE", program.rank.metadata)

    def test_graph_signal_rejects_malformed_or_future_topology(self):
        base = {"event_type": "supplier disruption", "_scope": "network_supplement"}
        for path in ({"hops": 1, "vintage": "2025-12-01",
                      "steps": [{"neighbor_role": "supplier"}]},
                     {"hops": 2, "vintage": "2025-06-06",
                      "steps": [{"neighbor_role": "supplier"}]},
                     {"hops": 1, "vintage": "2025-06-06",
                      "steps": [{"neighbor_role": "invented"}]},
                     {"type": "invented", "hops": 1, "vintage": "2025-06-06",
                      "steps": [{"neighbor_role": "supplier"}]},
                     {"hops": 4, "vintage": "2025-06-06",
                      "steps": [{"neighbor_role": "supplier"}] * 4}):
            with self.subTest(path=path), self.assertRaises(ValueError):
                finance_graph_signal({**base, "verified_paths": [path]},
                                     "2025-10-03T12:00:00")
        with self.assertRaisesRegex(ValueError, "scope"):
            finance_graph_signal({"_scope": "unexpected"}, "2025-10-03T12:00:00")

    def test_multihop_finance_motifs_and_evidence_gaps(self):
        def path(*roles):
            return {"hops": len(roles), "vintage": "2025-06-06",
                    "steps": [{"neighbor_role": role} for role in roles]}

        base = {"_scope": "network_supplement", "status": "ongoing"}
        supplier = finance_graph_signal({**base, "event_type": "supplier disruption",
            "verified_paths": [path("supplier", "supplier", "supplier")]},
            "2025-10-03T12:00:00")
        self.assertEqual(supplier["role_motif"], "all_upstream")
        self.assertEqual(supplier["finance_hypothesis"], "upstream_input_risk")
        self.assertEqual(supplier["evidence_gap"], "intermediate_product_continuity")
        self.assertEqual(supplier["alternative_route"], "none_observed")
        self.assertEqual(supplier["channel_status"], "path_only")

        customer = finance_graph_signal({**base, "event_type": "customer demand",
            "verified_paths": [path("customer", "customer")]},
            "2025-10-03T12:00:00")
        self.assertEqual(customer["role_motif"], "all_downstream")
        self.assertEqual(customer["finance_hypothesis"], "downstream_demand_risk")

        mixed = finance_graph_signal({**base, "event_type": "supplier disruption",
            "verified_paths": [path("supplier", "customer")]},
            "2025-10-03T12:00:00")
        self.assertEqual(mixed["role_motif"], "direction_change")
        self.assertEqual(mixed["finance_hypothesis"], "unaligned_or_ambiguous")

        competitor = finance_graph_signal({**base, "event_type": "competitor substitution",
            "verified_paths": [path("competitor", "supplier")]},
            "2025-10-03T12:00:00")
        self.assertEqual(competitor["role_motif"], "competitor_mixed")
        self.assertEqual(competitor["finance_hypothesis"], "unaligned_or_ambiguous")

    def test_alternative_paths_are_descriptive_not_redundancy(self):
        def path(*roles):
            return {"hops": len(roles), "vintage": "2025-06-06",
                    "steps": [{"neighbor_role": role} for role in roles]}
        row = {"_scope": "network_supplement", "event_type": "supplier disruption",
               "channel_receipt": {"status": "conditional_channel"},
               "verified_paths": [path("supplier", "supplier"),
                                  path("supplier", "supplier"),
                                  path("supplier", "supplier", "supplier")]}
        signal = finance_graph_signal(row, "2025-10-03T12:00:00")
        self.assertEqual(signal["path_count_bin"], "multiple")
        self.assertEqual(signal["alternative_route"], "longer_typed_path_observed")
        self.assertEqual(signal["finance_hypothesis"], "upstream_input_risk")
        self.assertEqual(signal["channel_status"], "path_only")
        self.assertNotIn("redundancy", str(signal))
        self.assertNotIn("conditional_channel", str(signal))

    def test_conflicting_shortest_paths_and_unaligned_events_fail_closed(self):
        def path(*roles):
            return {"hops": len(roles), "vintage": "2025-06-06",
                    "steps": [{"neighbor_role": role} for role in roles]}
        base = {"_scope": "network_supplement", "event_type": "supplier disruption"}
        conflicting = finance_graph_signal({**base, "verified_paths": [
            path("supplier", "supplier"), path("supplier", "customer")]},
            "2025-10-03T12:00:00")
        self.assertEqual(conflicting["role_motif"], "conflicting_shortest")
        self.assertEqual(conflicting["finance_hypothesis"], "unaligned_or_ambiguous")
        self.assertEqual(conflicting["evidence_gap"], "path_and_product_ambiguity")
        unaligned = finance_graph_signal({**base, "event_type": "routine",
            "verified_paths": [path("supplier")]}, "2025-10-03T12:00:00")
        self.assertEqual(unaligned["finance_hypothesis"], "unaligned_or_ambiguous")
        self.assertEqual(unaligned["evidence_gap"], "direct_product_exposure")

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
            def __init__(self):
                self.requests = []
            def create(self, **kwargs):
                self.requests.append(kwargs)
                if len(self.requests) == 1:
                    return SimpleNamespace(id="skill-response", output=[SimpleNamespace(
                        type="function_call", name="use_supply_chain_context_skill",
                        arguments="{}", call_id="skill-call")])
                return SimpleNamespace(output=[SimpleNamespace(type="function_call", name="select_context",
                    arguments='{"ranked_ids":["C2","C1"]}', call_id="selection-call")])
        fake = FakeResponses()
        events, receipt = openai_required_tool_call(fake, model="test-model", candidates=toy_events()[:2],
            asof="2025-01-03T12:00:00", max_events=1)
        self.assertEqual([request["tool_choice"]["name"] for request in fake.requests],
                         ["use_supply_chain_context_skill", "select_context"])
        self.assertEqual(fake.requests[1]["previous_response_id"], "skill-response")
        self.assertEqual(fake.requests[1]["input"][0]["type"], "function_call_output")
        self.assertEqual(fake.requests[1]["input"][0]["call_id"], "skill-call")
        self.assertIn("Use the supply-chain-context-selection skill", fake.requests[1]["input"][0]["output"])
        self.assertFalse(any(request["parallel_tool_calls"] for request in fake.requests))
        self.assertEqual(events[0]["source_id"], "RISK")
        self.assertTrue(receipt["tool_called"])
        self.assertEqual(receipt["tool_sequence"],
                         ["use_supply_chain_context_skill", "select_context"])

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

    def test_network_scope_is_added_by_host_not_model(self):
        class StubProgram:
            max_events = 1
            def __call__(self, *, candidates, asof):
                self.candidates = candidates
                ids = ["NETWORK"]
                events, receipt = select_context(candidates, ids, asof=asof, max_events=1)
                return dspy.Prediction(events=events, receipt=receipt, ranked_ids=ids)

        event = {"source_id": "NETWORK", "available_at": "2025-01-02T09:00:00",
                 "event_type": "supplier outage", "verified_paths": [{"hops": 1,
                 "steps": [{"neighbor_role": "supplier", "edge_id": "E1"}]}]}
        packet = {"own_common": {"events": [], "missing": [], "coverage": {}},
                  "network_supplement": {"events": [event], "missing": [], "coverage": {}}}
        program = StubProgram()
        filtered, _ = apply_context_program(packet, program, "2025-01-03T12:00:00")
        self.assertEqual(program.candidates[0]["_scope"], "network_supplement")
        self.assertNotIn("_scope", filtered["network_supplement"]["events"][0])

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
        self.assertIn("exposure_paths", instructions)
        self.assertIn("check_network_channel", instructions)
        self.assertIn("five-session", instructions)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            skill_instructions(ROOT / "skills" / "supply-chain-news-risk" / "SKILL.md")

    def test_program_must_call_python_tool(self):
        program = ContextProgram(max_events=1)
        program.rank = StubToolAgent(program, "C2,C1")
        result = program(candidates=toy_events(), asof="2025-01-03T12:00:00")
        self.assertEqual([e["source_id"] for e in result.events], ["RISK"])
        self.assertTrue(result.receipt["tool_called"])
        self.assertIn("Use the supply-chain-context-selection skill", program.rank.task)
        self.assertNotIn("Supplier plant closure", program.rank.metadata)
        self.assertNotIn("FUTURE", program.rank.metadata)
        sensitive = toy_events()[:2]
        sensitive[1]["event_type"] = "Private customer contract 98765"
        program(candidates=sensitive, asof="2025-01-03T12:00:00")
        self.assertNotIn("Private customer", program.rank.metadata)
        sensitive[1]["event_type"] = "Confidential supplier shortage at private site"
        program(candidates=sensitive, asof="2025-01-03T12:00:00")
        self.assertIn("supplier disruption", program.rank.metadata)
        self.assertNotIn("Confidential", program.rank.metadata)

    def test_graph_path_direction_reaches_ranker_as_typed_lead(self):
        event = {"source_id": "NEIGHBOR", "available_at": "2025-10-02T09:00:00",
                 "event_type": "supplier plant outage", "_scope": "network_supplement",
                 "verified_paths": [{"hops": 1, "vintage": "2025-06-06",
                                     "steps": [{"neighbor_role": "supplier", "edge_id": "E1"}]}]}
        program = ContextProgram(max_events=1)
        program.rank = StubToolAgent(program)
        program(candidates=[event], asof="2025-10-03T12:00:00")
        self.assertIn('"scope": "network_supplement"', program.rank.metadata)
        self.assertIn('"graph_role": "supplier"', program.rank.metadata)
        self.assertIn('"hops": 1', program.rank.metadata)
        self.assertIn('"channel_status": "path_only"', program.rank.metadata)
        self.assertNotIn("E1", program.rank.metadata)
        with self.assertRaises(ValueError):
            program(candidates=[{**event, "verified_paths": [{**event["verified_paths"][0],
                "vintage": "2025-12-01"}]}], asof="2025-10-03T12:00:00")
        comention = {**event, "verified_paths": [{"type": "comention", "hops": 1,
                                                   "steps": [], "vintage": "2025-07-31"}]}
        program(candidates=[comention], asof="2025-10-03T12:00:00")
        self.assertIn('"graph_role": "co_mention"', program.rank.metadata)
        self.assertIn('"channel_status": "retrieval_lead"', program.rank.metadata)

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
                system = str(request.system or "")
                phase = (3 if "next_tool_name" not in system else
                         2 if "observation_1" in prompt else
                         1 if "observation_0" in prompt else 0)
                if phase == 0:
                    output = {"next_thought": "Read the reviewed Skill.",
                              "next_tool_name": "use_supply_chain_context_skill", "next_tool_args": {}}
                elif phase == 1:
                    output = {"next_thought": "Select the ranked aliases with Python.",
                              "next_tool_name": "select_context", "next_tool_args": {"ranked_ids": ids}}
                elif phase == 2:
                    output = {"next_thought": "Tool receipt is available.",
                              "next_tool_name": "finish", "next_tool_args": {}}
                else:
                    output = {"ranked_ids": ",".join(ids)}
                if phase < 3:
                    rendered = ("[[ ## next_thought ## ]]\n" + output["next_thought"] +
                        "\n[[ ## next_tool_name ## ]]\n" + output["next_tool_name"] +
                        "\n[[ ## next_tool_args ## ]]\n" + json.dumps(output["next_tool_args"]) +
                        "\n[[ ## completed ## ]]")
                else:
                    rendered = ("[[ ## reasoning ## ]]\nSelection tool completed.\n"
                                "[[ ## ranked_ids ## ]]\n" + output["ranked_ids"] +
                                "\n[[ ## completed ## ]]")
                return Response(id=None, model="custom/synthetic", message=Message.assistant(rendered),
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
            compiled_receipt = compiled(**rows[6].inputs()).receipt
        self.assertEqual(seed_score, 0.0)
        self.assertEqual(heldout, [1.0, 1.0])
        self.assertNotEqual(compiled.rank.react.signature.instructions, seed.rank.react.signature.instructions)
        self.assertIn("direct_supplier", compiled.rank.react.signature.instructions)
        self.assertIn("path_only", compiled.rank.react.signature.instructions)
        self.assertTrue(report["selected_candidate"])
        self.assertEqual(report["seed_validation_negative_mse"], -1.0)
        self.assertEqual(report["candidate_validation_negative_mse"], 0.0)
        self.assertFalse(report["test_labels_accessed"])
        self.assertGreater(engine.calls, 0)
        self.assertEqual(compiled_receipt["tool_sequence"],
                         ["use_supply_chain_context_skill", "select_context"])


if __name__ == "__main__":
    unittest.main()
