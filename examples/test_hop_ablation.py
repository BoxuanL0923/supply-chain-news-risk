"""Synthetic, provider-free tests for cumulative supply-chain radius comparisons."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "code"))
from news_network_50_2025.hop_ablation import scope_network_pool, regression_report, matched_scores, exact_hop_counts
from news_exposure_2025.graph import Graph
from news_network_50_2025.data import three_hop_candidates, graph_control_scope
from news_network_50_2025.workflow import Runner
from unittest.mock import patch


class HopAblationTests(unittest.TestCase):
    def setUp(self):
        self.relations = [{"hops": h, "type": "supplies", "permnos": [h],
                           "path": list(range(h + 1)), "steps": [{}] * h}
                          for h in (1, 2, 3)]
        self.relations.append({"hops": 1, "type": "comention", "permnos": [4],
                               "path": [], "steps": []})
        self.docs = {str(h): {"ids": [h]} for h in range(1, 5)}
        self.pool = {"network_ids": list(self.docs), "relations": self.relations,
                     "coverage": {"network_candidates": 4, "network_31d_count": 44,
                                  "unmapped_paths": 7}}

    def test_cumulative_scopes_exclude_deeper_hops_and_comentions(self):
        for radius, expected in ((1, {"1"}), (2, {"1", "2"}), (3, {"1", "2", "3"})):
            ids, relations, coverage = scope_network_pool(self.pool, self.docs, radius)
            self.assertEqual(set(ids), expected)
            self.assertEqual({p["hops"] for p in relations}, set(range(1, radius + 1)))
            self.assertEqual(coverage["max_hops"], radius)
            self.assertNotIn("network_31d_count", coverage)
            self.assertEqual(coverage["unmapped_paths"], 0)
        self.assertEqual(self.pool["network_ids"], list(self.docs))

    def test_multi_tag_story_is_kept_at_shortest_available_hop(self):
        self.docs["3"]["ids"] = [1, 3]
        ids, relations, _ = scope_network_pool(self.pool, self.docs, 1)
        self.assertIn("3", ids)
        self.assertEqual([p["hops"] for p in relations], [1])

    def test_runner_capability_filters_deeper_paths_before_tool_prompt(self):
        saved = {"pools": {"A": {**self.pool, "case": {"asof": "2025-10-03"},
                  "calendar": {}, "controls": {}, "own_packet_hash": "same",
                  "own_ids": [] , "history_ids": []}}, "documents": self.docs}
        runner = Runner(Path("/unused"), None, [], {}, "manifest", max_hops=2)
        with patch("news_network_50_2025.workflow.pool_file", return_value=saved):
            capability = runner.capability({"date": "2025-10-03", "case_id": "A"}, True)
        self.assertEqual(set(capability["documents"]), {"1", "2"})
        self.assertEqual([p["hops"] for p in capability["relations"]], [1, 2])

    def test_three_hop_quota_survives_direct_news_crowding(self):
        rows = [{"source_id": str(i), "ids": [1 if i < 20 else 3],
                 "headline": f"Synthetic outage {i}",
                 "available_at": "2025-10-02T09:00:00+00:00"} for i in range(25)]
        relation = [self.relations[0], self.relations[2]]
        selected = three_hop_candidates(rows, np.arange(25, 0, -1), list(range(25)), relation)
        self.assertLessEqual(len(selected), 20)
        self.assertTrue(any(row["ids"] == [3] for row in selected))

    def test_exact_hop_audit_uses_shortest_typed_path(self):
        items = [{"ids": [1, 3]}, {"ids": [2]}, {"ids": [4]}]
        self.assertEqual(exact_hop_counts(items, self.relations, document_mode=True),
                         {1: 1, 2: 1, 3: 0})
        events = [{"verified_paths": [self.relations[2]]},
                  {"verified_paths": [self.relations[3]]}]
        self.assertEqual(exact_hop_counts(events, []), {1: 0, 2: 0, 3: 1})

    def test_three_hop_controls_use_direct_neighbors_only(self):
        rows = [{"ids": [1]}, {"ids": [2]}, {"ids": [3]}]
        neighbors, indices = graph_control_scope(self.relations[:3], rows, [0, 1, 2], 3)
        self.assertEqual(neighbors, {1})
        self.assertEqual(indices, [0])

    def test_fail_closed_on_missing_third_hop_pool(self):
        with self.assertRaises(ValueError):
            scope_network_pool({**self.pool, "relations": self.relations[:2]}, self.docs, 3,
                               require_radius_three=True)

    def test_offline_pool_enumerates_three_hops_without_bypassing_tool_gate(self):
        nodes = {key: {"name": key, "permnos": [i]} for i, key in enumerate("abcd")}
        edges = [{"id": str(i), "source": downstream, "target": upstream,
                  "type": "supplies", "revenue_pct": None}
                 for i, (downstream, upstream) in enumerate(zip("abc", "bcd"))]
        graph = Graph({"targets": {"TOY": ["a"]}, "nodes": nodes, "edges": edges})
        with self.assertRaises(ValueError):
            graph.paths("TOY", 3)
        self.assertEqual({p["hops"] for p in graph.pool_paths("TOY", 3)}, {1, 2, 3})

    def test_regression_uses_incremental_hop_scores_and_same_rows(self):
        n = 40
        x = np.arange(n, dtype=float)
        frame = pd.DataFrame({"case_id": [str(i) for i in range(n)],
            "y": 1 + .02*x + np.sin(x)*.2,
            "c": np.cos(x/3), "H1": np.sin(x/4),
            "H2": np.sin(x/4) + np.cos(x/5)*.1,
            "H3": np.sin(x/4) + np.cos(x/5)*.1 + np.sin(x/7)*.1})
        report = regression_report(frame, controls=("c",))
        self.assertEqual(report["n"], n)
        self.assertEqual(report["terms"], ["H1", "H2_minus_H1", "H3_minus_H2"])
        self.assertEqual(report["nested"]["through_h3"]["rank"], 5)
        self.assertTrue(np.isfinite(report["nested"]["through_h3"]["r2"]))
        frozen = regression_report(frame.iloc[20:], controls=("c",), train=frame.iloc[:20])
        self.assertTrue(np.isfinite(frozen["nested"]["through_h3"]["frozen_mse"]))
        unidentified = regression_report(frame.assign(H3=frame.H2), controls=("c",))
        self.assertEqual(unidentified["nested"]["through_h3"]["rank_increment"], 0)
        self.assertIsNone(unidentified["nested"]["through_h3"]["coefficient"])
        with self.assertRaises(ValueError):
            regression_report(frame.drop(index=0).assign(H3=np.nan), controls=("c",))

    def test_matched_scores_reject_missing_or_duplicated_radius(self):
        cases = [{"case_id": "A"}, {"case_id": "B"}]
        controls = pd.DataFrame({"case_id": ["A", "B"], "c": [0., 1.]})
        labels = pd.DataFrame({"case_id": ["A", "B"], "y": [1., 2.]})
        scores = {h: [{"case_id": cid, "score": float(h + j)}
                      for j, cid in enumerate(("A", "B"))] for h in (1, 2, 3)}
        own = [{"case_id": cid, "score": float(j)} for j, cid in enumerate(("A", "B"))]
        frame = matched_scores(cases, controls, labels, scores, own)
        self.assertEqual(frame[["H1", "H2", "H3"]].iloc[0].tolist(), [1., 2., 3.])
        self.assertEqual(frame.OWN.tolist(), [0., 1.])
        with self.assertRaises(ValueError):
            matched_scores(cases, controls, labels, {**scores, 3: scores[3][:1]})


if __name__ == "__main__":
    unittest.main()
