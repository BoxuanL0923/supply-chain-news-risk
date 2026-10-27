"""Offline routing decisions, temporal boundaries, prefix guards and receipts."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from news_network_50_2025.graph_router import RouterRules, digest, route_case, verify_receipt


def snapshot():
    return dict(nodes=list("ABCDEF"), vintage="2020-01-01", monthly_universe={"2020-02": list("ABCDEF")},
                edges=[dict(source=a, target=b, vintage="2020-01-01", valid_from="2020-01-01")
                       for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"))])


def case(neighbours=None):
    return dict(case_id="synthetic-router", firm_id="A", cutoff="2020-02-01",
                candidate_neighbours=["B", "C"] if neighbours is None else neighbours, inspections=[])


def inspection(graph, value, path):
    return dict(path=path, case_id=value["case_id"], cutoff=value["cutoff"], snapshot_sha256=digest(graph),
                inspected=True, source_ids=["synthetic-source"], available_at="2020-01-31")


class RouterTests(unittest.TestCase):
    def test_default_order_and_triangle_trigger(self):
        result = route_case(snapshot(), case())
        tools = [c["tool"] for c in result["calls"]]
        self.assertEqual(tools, ["/sc-graph-degree", "/sc-graph-triangles", "/sc-graph-triangles", "path_enumeration"])
        self.assertEqual(result["calls"][1]["result"]["triangle_count"], 1)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(any(c["tool"] == "/sc-graph-triangles" for c in route_case(snapshot(), case(["B"]))["calls"]))
        self.assertTrue(any(s.get("max_hops") == 3 for s in result["skipped"]))

    def test_exact_inspected_prefix_and_three_hop_cap(self):
        graph, value = snapshot(), case()
        value["inspections"] = [inspection(graph, value, ["A", "C"])]
        result = route_case(graph, value)
        self.assertEqual(max(p["hops"] for p in result["paths"]), 2)
        self.assertFalse(any(c.get("max_hops") == 3 for c in result["calls"]))
        value["inspections"].append(inspection(graph, value, ["A", "C", "D"]))
        result = route_case(graph, value)
        self.assertIn(dict(path=["A", "C", "D", "E"], hops=3), result["paths"])
        self.assertFalse(any("F" == p["path"][-1] for p in result["paths"]))
        with self.assertRaises(ValueError): RouterRules(max_hops=4)
        with self.assertRaises(ValueError): RouterRules(max_hops=True)
        result = route_case(graph, value, RouterRules(max_hops=1))
        self.assertEqual(max(p["hops"] for p in result["paths"]), 1)

    def test_prefixes_cannot_self_certify_or_transfer(self):
        graph, value = snapshot(), case()
        value["inspections"] = [inspection(graph, value, ["A", "C", "D"])]
        with self.assertRaises(ValueError): route_case(graph, value)
        for key, bad in (("inspected", False), ("source_ids", []), ("available_at", value["cutoff"]),
                         ("snapshot_sha256", "wrong"), ("case_id", "other"), ("path", ["A", "D"])):
            value["inspections"] = [inspection(graph, value, ["A", "C"])]
            value["inspections"][0][key] = bad
            with self.assertRaises(ValueError): route_case(graph, value)

    def test_cutoff_expiry_and_monthly_universe(self):
        for key in ("vintage", "valid_from"):
            graph = snapshot(); graph["edges"][0][key] = "2020-02-01"
            with self.assertRaises(ValueError): route_case(graph, case())
        graph = snapshot(); graph["vintage"] = "2020-02-01"
        with self.assertRaises(ValueError): route_case(graph, case())
        graph = snapshot(); graph["edges"][0]["valid_to"] = "2020-02-01"
        with self.assertRaises(ValueError): route_case(graph, case())  # candidate B expired
        result = route_case(graph, case(["C"]))
        self.assertEqual(result["calls"][0]["result"]["total_degree"], 1)
        graph = snapshot(); graph["monthly_universe"] = {}
        with self.assertRaises(ValueError): route_case(graph, case())

    def test_hub_boundary_and_configurable_routing(self):
        graph = snapshot()
        result = route_case(graph, case(["C"]), RouterRules(thresh_hub=0, single_neighbour_weight=.25))
        self.assertEqual(result["evidence_weight"], .25)
        percentile = result["calls"][0]["result"]["degree_percentile"]
        self.assertEqual(route_case(graph, case(["C"]), RouterRules(thresh_hub=percentile))["evidence_weight"], .5)
        self.assertEqual(route_case(graph, case(), RouterRules(thresh_hub=0))["evidence_weight"], 1.)
        result = route_case(graph, case(), RouterRules(min_candidate_neighbours=3))
        self.assertFalse(any(c["tool"] == "/sc-graph-triangles" for c in result["calls"]))
        for kwargs in (dict(thresh_hub=float("nan")), dict(min_candidate_neighbours=1), dict(single_neighbour_weight=1)):
            with self.assertRaises(ValueError): RouterRules(**kwargs)

    def test_receipt_replay_tamper_and_truncation(self):
        graph, value = snapshot(), case()
        result = route_case(graph, value)
        self.assertEqual(result, route_case(graph, value))
        self.assertTrue(verify_receipt(result))
        bad = copy.deepcopy(result); bad["evidence_weight"] = .01
        with self.assertRaises(ValueError): verify_receipt(bad)
        result = route_case(graph, value, RouterRules(path_limit=1))
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["paths"]), 1)

    def test_invalid_candidates_and_standalone_cli(self):
        for neighbours in (["B", "B"], ["D"], ["missing"]):
            with self.assertRaises(ValueError): route_case(snapshot(), case(neighbours))
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for name, value in (("snapshot", snapshot()), ("case", case())):
                (Path(directory)/f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            process = subprocess.run([sys.executable, str(repo/"skills/sc-graph-router/scripts/graph_router.py"),
                "--snapshot", "snapshot.json", "--case", "case.json"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(verify_receipt(json.loads(process.stdout)))


if __name__ == "__main__":
    unittest.main()
