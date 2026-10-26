"""Offline Q1 graph, analytic-gradient, split and transport invariants."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/code"))
from news_network_50_2025.graph_skills import Snapshot
from news_network_50_2025.prompt_gd import (LocalQwen, smooth_score, descend, validate_cases,
    load_labels, cache_features, digest, write, MODEL, objective)
from news_network_50_2025.toolkit import Toolkit
from news_network_50_2025.domain_signals import finance_graph_signal


def snapshot():
    return {"nodes": list("ABCDEF"), "vintage": "2025-06-01", "monthly_universe": {"2025-08": list("ABCDEF")},
            "edges": [{"source": a, "target": b, "valid_from": "2025-05-01", "vintage": "2025-06-01"}
                      for a, b in (("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"))]}


def case():
    return {"case_id": "toy", "ticker": "TOY", "date": "2025-08-01", "asof": "2025-08-01T20:00:00Z",
            "next_sessions": [f"2025-08-{d:02}" for d in range(4, 9)], "split": "train", "anchor": 0.,
            "candidates": [{"source_id": "s1", "available_at": "2025-07-31", "hop": 0, "text": "Synthetic event", "graph": {}}]}


class GraphTests(unittest.TestCase):
    def test_degrees_triangle_ra_known_values(self):
        g = Snapshot(snapshot(), "2025-08-01")
        d = g.degree_result("A")
        self.assertEqual((d["in_degree"], d["out_degree"], d["total_degree"]), (1, 1, 2))
        self.assertAlmostEqual(d["degree_percentile"], 100*5/6)
        t = g.triangle_result("A", "B")
        self.assertEqual(t["triangle_count"], 1)
        self.assertEqual(t["clustering"], 1.)
        self.assertEqual(t["pair_triangle_count"], 1)
        self.assertAlmostEqual(t["ra"], 1/3)
        self.assertFalse(g.degree_result("A")["hub"])

    def test_prefix_guard_and_three_hops(self):
        g = Snapshot(snapshot(), "2025-08-01")
        self.assertEqual(max(p["hops"] for p in g.paths("A", 3)[0]), 1)
        prefixes = [["A", "C"], ["A", "C", "D"]]
        t = g.triangle_result("A", "E", 3, prefixes)
        self.assertIn({"path": ["A", "C", "D", "E"], "hops": 3}, t["paths"])
        self.assertEqual(t["pair_triangle_count"], 0)
        with self.assertRaises(ValueError): g.triangle_result("A", "E", 3, [["A", "B", "D"]])
        with self.assertRaises(ValueError): g.paths("A", 4)
        with self.assertRaises(ValueError): g.paths("A", True)

    def test_dates_duplicates_isolates_and_universe(self):
        s = snapshot(); s["edges"] += [dict(s["edges"][0])]
        s["edges"].append({**s["edges"][0], "source": "A", "target": "A"})
        g = Snapshot(s, "2025-08-01")
        self.assertEqual(g.degree("A")["total_degree"], 2)
        self.assertEqual(g.triangle_result("F")["clustering"], 0)
        for field in ("vintage", "valid_from"):
            s = snapshot(); s["edges"][0][field] = "2025-08-01"
            with self.assertRaises(ValueError): Snapshot(s, "2025-08-01")
        s = snapshot(); s["edges"][0]["valid_to"] = "2025-07-01"
        self.assertEqual(Snapshot(s, "2025-08-01").degree("A")["total_degree"], 1)
        with self.assertRaises(ValueError): Snapshot(snapshot(), "2025-07-01")

    def test_toolkit_and_selector_fields(self):
        kit = Toolkit({"case": {"asof": "2025-08-01"}, "documents": {}, "network": True,
                       "graph_snapshot": snapshot(), "graph_firm_id": "A"})
        d = kit.call("sc_graph_degree")
        self.assertEqual(d["total_degree"], 2)
        self.assertIn("error", kit.call("sc_graph_triangles", neighbour="E", max_hops=3))
        kit.inspected_paths.update({("A", "C"), ("A", "C", "D")})
        self.assertNotIn("error", kit.call("sc_graph_triangles", neighbour="E", max_hops=3))
        row = {"graph_statistics": {"cutoff": "2025-08-01", "vintage": "2025-06-01", "degree_percentile": 90, "ra": .5}}
        self.assertEqual(finance_graph_signal(row, "2025-08-01")["ra"], .5)
        row["graph_statistics"]["vintage"] = "2025-09-01"
        with self.assertRaises(ValueError): finance_graph_signal(row, "2025-08-01")

    def test_explicit_pair_beyond_display_cap(self):
        s = snapshot()
        for i in range(220):
            node = f"B{i:03}"
            s["nodes"].append(node)
            s["edges"].append({**s["edges"][0], "target": node})
        g = Snapshot(s, "2025-08-01")
        self.assertTrue(g.triangle_result("A", "C")["truncated"])

    def test_standalone_skill_clis(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.json"; write(path, snapshot())
            for kind, expected in (("degree", "total_degree"), ("triangles", "triangle_count")):
                result = subprocess.run([sys.executable, str(repo/f"skills/sc-graph-{kind}/scripts/graph_{kind}.py"),
                    "--snapshot", str(path), "--firm-id", "A", "--cutoff", "2025-08-01"],
                    check=True, capture_output=True, text=True)
                value = json.loads(result.stdout)
                self.assertIn(expected, value); self.assertEqual(value["model"], MODEL)

    def test_evidence_host_replaces_model_graph_statistics(self):
        from news_network_50_2025.workflow import validate_evidence
        cap = {"case": {"asof": "2025-08-01"}, "network": True, "calendar": {}, "controls": {}, "coverage": {},
               "documents": {"s": {"source_id": "s", "headline": "Synthetic outage", "body": "A synthetic factory outage occurred.",
                                    "available_at": "2025-07-31", "ids": [1]}},
               "relations": [{"entity": "B", "permnos": [1], "hops": 1, "path": ["A", "B"],
                              "steps": [{"neighbor_role": "customer"}], "vintage": "2025-06-01"}],
               "graph_snapshot": snapshot(), "graph_firm_id": "A"}
        kit = Toolkit(cap)
        kit.call("exposure_paths"); kit.call("search_events", query="outage"); kit.call("inspect_event", source_id="s")
        event = {"source_id": "s", "page": 0, "fact": "outage", "interpretation": "risk", "qualifications": "unknown",
                 "event_type": "supplier disruption", "status": "ongoing", "graph_statistics": {"total_degree": 999}}
        receipt = validate_evidence({"events": [event]}, {"calls": kit.trace, "reads": kit.reads,
                                    "inspected_paths": [list(p) for p in kit.inspected_paths]}, cap)
        stats = receipt["events"][0]["graph_statistics"]
        self.assertEqual(stats["total_degree"], 2)
        self.assertAlmostEqual(stats["ra"], 1/3)


class OptimizerTests(unittest.TestCase):
    def test_analytic_gradient_including_boundaries(self):
        features = [{"relevance": .8, "direction": (-1)**h, "materiality": .7, "hop": h,
                     "graph": {"degree_percentile": 95., "ra_percentile": 80., "ra": .3, "clustering": .6}}
                    for h in range(4)]
        x = np.array([.3, .2, .5, .4, .6, .7, .8])
        value, jac = smooth_score(x, features)
        for i in range(7):
            dx = np.eye(7)[i]*1e-6
            numeric = (smooth_score(x+dx, features)[0]-smooth_score(x-dx, features)[0])/2e-6
            self.assertAlmostEqual(jac[i], numeric, places=8)
        x[6] = 0
        self.assertEqual(smooth_score(x, features)[0], 0.)
        self.assertTrue(np.isfinite(smooth_score(x, features)[1]).all())

    def test_quadratic_convergence_and_stop(self):
        target = np.full(7, .3)
        f = lambda x: (float(np.sum((x-target)**2)), 2*(x-target))
        for start in np.random.default_rng(1).uniform(0, 1, (5, 7)):
            result = descend(f, start, "constant")
            np.testing.assert_allclose(result["theta"], target, atol=1e-6)
            self.assertEqual(result["stop"], "projected_gradient")

    def test_residual_objective_and_gradient(self):
        c = case(); c["anchor"] = 2.
        features = [{"relevance": .8, "direction": -.4, "materiality": .7, "hop": 0, "graph": {}}]
        x = np.full(7, .5)
        score, derivative = smooth_score(x, features)
        loss, gradient = objective(x, [c], {"toy": features}, {"toy": 2.3})
        self.assertAlmostEqual(loss, (.3-score)**2)
        np.testing.assert_allclose(gradient, 2*(score-.3)*derivative)
        self.assertEqual(derivative[5], 0., "Neighbour materiality threshold must not gate own news")

    def test_no_sealed_labels_opened_and_no_remote_endpoint(self):
        c = case(); c["date"] = "2025-09-01"
        with patch("pathlib.Path.read_text", side_effect=AssertionError("file opened")):
            with self.assertRaises(ValueError): load_labels("missing", [c], "valid")
        with self.assertRaises(ValueError): LocalQwen("cache", endpoint="https://example.com/v1")
        c = case(); c["next_sessions"][-1] = "2025-09-01"
        with self.assertRaises(ValueError): validate_cases([c])
        c = case(); c["y"] = 1
        with self.assertRaises(ValueError): validate_cases([c])
        c = case(); c["candidates"].append(copy.deepcopy(c["candidates"][0]))
        with self.assertRaises(ValueError): validate_cases([c])

    def test_one_feature_request_per_case_and_host_fields(self):
        class Fake:
            calls = 0
            def call(self, task, payload):
                self.calls += 1
                self.payload = payload
                return {"features": [{"source_id": "s1", "relevance": .5, "direction": -1, "materiality": .3, "hop": 99}]}
        fake = Fake()
        with tempfile.TemporaryDirectory() as directory:
            result = cache_features([case()], directory, fake)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result["cases"][0]["features"][0]["hop"], 0)
        self.assertNotIn("y", fake.payload)

    def test_smoke_freezes_all_arms_before_validation_labels(self):
        from news_network_50_2025.hop_ablation import validation_smoke
        from news_network_50_2025.prompt_gd import load_labels as original_loader
        c = case(); c["split"] = "valid"; c["graph_firm_id"] = "A"
        c["candidates"][0]["ids"] = [0]
        c["relations"], c["inspected_paths"] = [], []
        chain = list("ACDE")
        for h in (1, 2, 3):
            path = chain[:h+1]
            relation = {"hops": h, "permnos": [h], "path": path, "entity": path[-1],
                        "steps": [{"neighbor_role": "supplier"}]*h, "vintage": "2025-06-01"}
            c["relations"].append(relation); c["inspected_paths"].append(path)
            c["candidates"].append({"source_id": f"h{h}", "available_at": "2025-07-31",
                                   "hop": h, "path": path, "text": "Inspected synthetic outage.",
                                   "ids": [h], "graph": {}, "verified_paths": [relation]})
        class Fake:
            def call(self, *args, **kwargs): return {"raw_score": .2}
        # A source found through H3 also tags the direct entity and must enter H1.
        c["candidates"][-1]["ids"] = [1, 3]
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)/"data"; out = Path(directory)/"smoke"
            write(data/"cases.json", {"model": MODEL, "cases": [c]})
            write(data/"graph_snapshot.json", snapshot())
            labels = {"toy": .1}
            write(data/"labels_valid.json", labels)
            write(data/"development_manifest.json", {"latest_origin": "2025-08-01", "latest_target_session": "2025-08-08",
                "splits": {"valid": ["toy"]}, "label_hashes": {"valid": digest(labels)}})
            def check_freeze(directory, cases, split):
                freeze = json.loads((out/"prediction_freeze.json").read_text())
                self.assertEqual(set(freeze["predictions"]), {"OWN", "H1", "H2", "H3", "H3_GRAPH"})
                self.assertEqual(split, "valid")
                return original_loader(directory, cases, split)
            with patch("news_network_50_2025.prompt_gd.load_labels", side_effect=check_freeze):
                result = validation_smoke(data, out, Fake())
            self.assertEqual(result["status"], "PASS")
            self.assertTrue((out/"matched_validation.csv").exists())
            h1 = next(a for a in result["scope_audit"] if a["arm"] == "H1")
            self.assertEqual(h1["counts"][1], 2)


if __name__ == "__main__": unittest.main()
