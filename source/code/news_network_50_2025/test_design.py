"""Deterministic safeguards and an optional small native-transport acceptance call."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from .data import schedule, diversified, common_packet, substantive_neighbor
from .toolkit import Toolkit
from .workflow import validate_evidence, context_rule
from .optimize import choose, eligible_policy, reflection_context
from .metrics import fit
from .config import SEED


def fixture(network=False):
    a = {"source_id": "a", "body": "The factory stopped production. A backup facility is available.",
         "headline": "Factory outage with backup", "available_at": "2025-10-01T19:59:00+00:00", "ids": [2]}
    return {"case": {"asof": "2025-10-01T20:00:00+00:00"}, "documents": {"a": a},
            "network": network, "calendar": {"status": "unknown"}, "controls": {}, "coverage": {},
            "relations": [{"entity": "partner", "permnos": [2], "hops": 1, "path": ["target", "partner"],
                           "steps": [{"neighbor_role": "supplier"}]}] if network else []}


class DesignTests(unittest.TestCase):
    def test_own_context_does_not_exclude_own_articles(self):
        self.assertIn("All listed sources are available", context_rule(False))
        self.assertNotIn("Only NEW", context_rule(False))
        self.assertIn("Only NEW", context_rule(True))

    def test_approved_resume_preserves_original_cache_manifest(self):
        from .run import resume_manifest
        from news_agent_batched_runtime_2025 import write, digest
        initial = {"protocol": "frozen", "sources": {"workflow.py": "old"}}
        current = {"protocol": "frozen", "sources": {"workflow.py": "new"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/"RUN_MANIFEST.json", initial)
            write(root/"RESUME_APPROVAL.json", {"from_manifest": digest(initial), "to_manifest": digest(current),
                "changed_sources": ["workflow.py"], "retain_validated_cache": True})
            self.assertEqual(resume_manifest(root, current), initial)
            with self.assertRaises(ValueError):
                resume_manifest(root, {**current, "protocol": "changed"})

    def test_missing_inspection_retries_native_tools_not_text_repair(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from .workflow import Runner
        cap = fixture()
        cap["own_packet_hash"] = "unchanged"
        kit = Toolkit(cap)
        kit.call("inspect_event", source_id="a")
        event = {"source_id": "a", "page": 0, "fact": "outage", "interpretation": "risk",
                 "qualifications": "backup", "event_type": "operations", "status": "ongoing"}
        responses = [{"request_sha256": "unread", "answer": '{"events": []}',
                      "trace": {"calls": [], "reads": {}}},
                     {"request_sha256": "read", "answer": json.dumps({"events": [event]}),
                      "trace": {"calls": kit.trace, "reads": kit.reads}}]
        client = SimpleNamespace(native=Mock(side_effect=responses),
                                 call=Mock(side_effect=AssertionError("Must not repair missing reads with text")))
        case = {"case_id": "test", "ticker": "TEST", "company": "Test Firm", "asof": cap["case"]["asof"],
                "next_sessions": ["2025-10-02"]}
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(Path(tmp), client, [case], {}, "initial", "patched")
            runner.capability = lambda case, network: cap
            result = runner.evidence(case)
        self.assertEqual(client.native.call_count, 2)
        client.call.assert_not_called()
        self.assertEqual(result["retrieval_recovery_from"], "unread")
        self.assertEqual(result["implementation_manifest"], "patched")

    def test_nonoverlap_and_maturity(self):
        dates = schedule()
        self.assertEqual(len({x for r in dates.values() for x in r["next_sessions"]}), 60)
        self.assertEqual([r["split"] for r in dates.values()].count("test"), 6)

    def test_close_strict(self):
        data = fixture()
        Toolkit(data)
        data["documents"]["a"]["available_at"] = data["case"]["asof"]
        with self.assertRaises(ValueError):
            Toolkit(data)

    def test_no_outcomes(self):
        data = fixture()
        data["y"] = 7
        with self.assertRaises(ValueError):
            Toolkit(data)

    def test_read_scoping_and_logged_error(self):
        kit = Toolkit(fixture())
        result = kit.call("inspect_event", source_id="not-this-case")
        self.assertIn("error", result)
        self.assertFalse(kit.trace[-1]["ok"])
        self.assertFalse(kit.reads)

    def test_read_not_search_makes_citable(self):
        kit = Toolkit(fixture())
        kit.call("search_events", query="factory")
        self.assertFalse(kit.reads)
        kit.call("inspect_event", source_id="a")
        self.assertIn("a", kit.reads)

    def test_no_total_tool_call_cap(self):
        kit = Toolkit(fixture())
        for _ in range(30):
            self.assertNotIn("error", kit.call("inspect_event", source_id="a"))
        self.assertEqual(len(kit.trace), 30)
        self.assertTrue(all(item["ok"] for item in kit.trace))

    def test_two_hop_requires_inspected_intermediate(self):
        data = fixture(True)
        data["documents"]["b"] = {**data["documents"]["a"], "source_id": "b", "ids": [3]}
        data["relations"].append({"entity": "upstream", "permnos": [3], "hops": 2,
                                  "path": ["target", "partner", "upstream"], "steps": []})
        kit = Toolkit(data)
        self.assertIn("error", kit.call("inspect_event", source_id="b"))
        kit.call("inspect_event", source_id="a")
        self.assertNotIn("error", kit.call("inspect_event", source_id="b"))

    def test_own_has_no_graph(self):
        self.assertIn("error", Toolkit(fixture()).call("exposure_paths"))

    def test_target_lookup_does_not_filter_out_neighbors(self):
        cap = fixture(True)
        cap["case"]["ticker"] = "WMT"
        cap["case"]["company"] = "Walmart Inc"
        cap["relations"][0]["name"] = "Acme Communications"
        kit = Toolkit(cap)
        result = kit.call("exposure_paths", query="WMT")
        self.assertEqual(result["total"], 1)
        self.assertTrue(result["target_query_interpreted_as_root_lookup"])
        absent = kit.call("exposure_paths", query="CAT")
        self.assertEqual(absent["total"], 0)
        self.assertEqual(absent["available_before_query"], 1)

    def test_search_does_not_return_zero_relevance_hits(self):
        result = Toolkit(fixture()).call("search_events", query="unrelatedxyz")
        self.assertEqual(result["results"], [])
        self.assertIn("not no news", result["search_scope_note"])

    def test_verbatim_quote_required(self):
        kit = Toolkit(fixture())
        kit.call("inspect_event", source_id="a")
        event = {"source_id": "a", "quote": "The factory stopped production.", "fact": "outage",
                 "interpretation": "possible risk", "qualifications": "backup exists", "event_type": "operation", "status": "ongoing"}
        trace = {"calls": kit.trace, "reads": kit.reads}
        validate_evidence({"events": [event]}, trace, fixture())
        event["quote"] = "Factory production certainly stops permanently."
        with self.assertRaises(ValueError):
            validate_evidence({"events": [event]}, trace, fixture())

    def test_both_validation_periods_required(self):
        cases = [{"case_id": str(i), "date": str(i//2)} for i in range(4)]
        residuals = {str(i): 1. for i in range(4)}
        seed = {"policy": "seed", "scores": {str(i): 0. for i in range(4)}}
        overfit = {"policy": "overfit", "scores": {"0": 1., "1": 1., "2": -.1, "3": -.1}}
        chosen, result = choose([seed, overfit], residuals, cases)
        self.assertIs(chosen, seed)
        self.assertFalse(result["changed"])

    def test_consistent_challenger(self):
        cases = [{"case_id": str(i), "date": str(i//2)} for i in range(4)]
        residuals = {str(i): 1. for i in range(4)}
        seed = {"policy": "seed", "scores": {str(i): 0. for i in range(4)}}
        better = {"policy": "better", "scores": {str(i): .5 for i in range(4)}}
        self.assertIs(choose([seed, better], residuals, cases)[0], better)

    def test_resource_extension_cannot_change_price_or_science(self):
        from .run import resource_limits
        from .config import LIMITS
        from news_agent_batched_runtime_2025 import write, digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resource_limits(root), LIMITS)
            manifest = {"test": "initial"}
            write(root / "RUN_MANIFEST.json", manifest)
            ext = {"authorization": "overnight request", "initial_manifest": digest(manifest),
                   "limits": {"input_token_stop": 300_000_000}}
            write(root / "RESOURCE_EXTENSION.json", ext)
            revised = resource_limits(root)
            self.assertEqual(revised["input_token_stop"], 300_000_000)
            self.assertEqual({k:v for k,v in revised.items() if k != "input_token_stop"},
                             {k:v for k,v in LIMITS.items() if k != "input_token_stop"})
            ext["limits"]["provider_list_price_stop_usd"] = 999
            write(root / "RESOURCE_EXTENSION.json", ext)
            with self.assertRaises(ValueError):
                resource_limits(root)

    def test_same_sample_mse_r2_identity(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"b": rng.normal(size=100), "score": rng.normal(size=100)})
        frame["y"] = frame.b + frame.score*.3 + rng.normal(size=100)
        base, full = fit(frame, ("b",)), fit(frame, ("b",), "score")
        self.assertAlmostEqual(full["r2"]-base["r2"], len(frame)*(base["mse"]-full["mse"])/base["tss"])

    def test_policy_identity_memorization_rejected(self):
        self.assertTrue(eligible_policy(SEED, ["AMD", "MSFT"]))
        self.assertFalse(eligible_policy(SEED+" Always give AMD more risk.", ["AMD", "MSFT"]))

    def test_network_reflection_keeps_own_context(self):
        from types import SimpleNamespace
        event = {k: k for k in ("source_id", "headline", "fact", "interpretation", "qualifications", "event_type", "status")}
        runner = SimpleNamespace(anchors={"a": -8.}, evidence=lambda case, net:
            {"events": [] if net else [event]*8, "missing": [], "coverage": {}})
        case = {"case_id": "a", "industry": "retail", "state": {}}
        value = reflection_context(runner, case, True)
        self.assertEqual(len(value["evidence"]["own_common"]["events"]), 8)
        self.assertEqual(value["evidence"]["network_supplement"]["events"], [])
        self.assertEqual(value["anchor"], -8.)

    def test_missing_is_not_empty_calendar(self):
        self.assertEqual(Toolkit(fixture()).event_logbook()["calendar"]["status"], "unknown")

    def test_native_server_initializes_without_llm(self):
        import asyncio
        import os
        import sys
        from mcp import ClientSession, StdioServerParameters, stdio_client
        from news_agent_batched_runtime_2025 import write
        from .config import ROOT
        async def exercise():
            with tempfile.TemporaryDirectory() as tmp:
                cap, trace = Path(tmp)/"cap.json", Path(tmp)/"trace.json"
                write(cap, fixture())
                params = StdioServerParameters(command=sys.executable, args=["-m", "news_network_50_2025.toolkit",
                    str(cap), str(trace)], env={**os.environ, "PYTHONPATH": str(ROOT/"code")})
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        listed = await session.list_tools()
                        self.assertEqual(len(listed.tools), 6)
                        result = await session.call_tool("inspect_event", {"source_id": "a", "page": 0})
                        self.assertFalse(result.is_error)
                        self.assertTrue(trace.exists())
        asyncio.run(exercise())

    def test_batch_inspection_and_host_quotation(self):
        cap = fixture()
        kit = Toolkit(cap)
        result = kit.call("inspect_events", requests=[{"source_id": "a", "page": 0}, {"source_id": "foreign"}])
        self.assertIn("error", result["results"][1])
        event = {"source_id": "a", "page": 0, "fact": "outage", "interpretation": "risk",
                 "qualifications": "backup", "event_type": "operation", "status": "ongoing"}
        value = validate_evidence({"events": [event]}, {"calls": kit.trace, "reads": kit.reads}, cap)
        self.assertEqual(value["events"][0]["quote"], cap["documents"]["a"]["body"])

    def test_neighbor_filter_preserves_operations(self):
        self.assertFalse(substantive_neighbor("IRIDIUM <IRDM.O>: BARCLAYS CUTS TARGET PRICE TO $25"))
        self.assertFalse(substantive_neighbor("Qualcomm Form 8.3 Aqua Acquisition"))
        self.assertTrue(substantive_neighbor("Supplier announces factory outage and backup capacity"))

    def test_common_topics_not_only_rates(self):
        rows = [{"source_id": str(i), "headline": h, "available_at": "2025-10-01T19:00:00+00:00",
                 "kind": "macro", "topics": [], "ids": []} for i, h in enumerate([
                     "Fed discusses interest rates", "US government shutdown starts", "Australian central bank holds rates"])]
        selected = common_packet(rows, np.array([.9, .1, 1.]), np.zeros(3))
        self.assertIn("1", {r["source_id"] for r in selected})
        self.assertNotIn("2", {r["source_id"] for r in selected})


if __name__ == "__main__":
    unittest.main()
