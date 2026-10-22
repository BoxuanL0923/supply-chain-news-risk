"""Synthetic contract tests for staged third-hop graph discovery."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "code"))
from news_network_50_2025.toolkit import Toolkit  # noqa: E402
from news_network_50_2025.workflow import validate_evidence  # noqa: E402


def capability():
    asof = "2025-10-03T12:00:00+00:00"
    path = ["target", "mid1", "mid2", "upstream"]
    roles = ["supplier", "supplier", "supplier"]
    relations = [
        {"entity": node, "name": node, "permnos": [idx + 2],
         "hops": idx + 1, "path": path[:idx + 2],
         "steps": [{"neighbor_role": role} for role in roles[:idx + 1]],
         "vintage": "2025-06-06"}
        for idx, node in enumerate(path[1:])
    ]
    docs = {
        key: {"source_id": key, "headline": "Synthetic production outage",
              "body": "Synthetic production outage affected the input.",
              "available_at": "2025-10-02T09:00:00+00:00", "ids": [idx + 2]}
        for idx, key in enumerate(("direct", "second", "third"))
    }
    return {"case": {"asof": asof, "ticker": "TOY", "company": "Toy Firm"},
            "documents": docs, "network": True, "calendar": {}, "controls": {},
            "coverage": {}, "relations": relations}


class ThirdHopToolkitTests(unittest.TestCase):
    def test_third_hop_requires_two_inspected_intermediates(self):
        kit = Toolkit(capability())
        self.assertIn("error", kit.call("exposure_paths", hops=3, via="mid2"))
        self.assertIn("error", kit.call("search_events", query="outage", hops=3, via="mid2"))
        self.assertIn("error", kit.call("inspect_event", source_id="third"))
        self.assertNotIn("error", kit.call("inspect_event", source_id="direct"))
        self.assertIn("error", kit.call("inspect_event", source_id="third"))
        self.assertNotIn("error", kit.call("exposure_paths", hops=2, via="mid1"))
        self.assertNotIn("error", kit.call("inspect_event", source_id="second"))
        paths = kit.call("exposure_paths", hops=3, via="mid2")
        self.assertEqual([p["hops"] for p in paths["paths"]], [1, 2, 3])
        hits = kit.call("search_events", query="outage", hops=3, via="mid2")
        self.assertIn("third", [hit["source_id"] for hit in hits["results"]])
        self.assertNotIn("error", kit.call("inspect_event", source_id="third"))
        answer = {"events": [{"source_id": "third", "page": 0,
                   "fact": "synthetic outage", "interpretation": "possible input risk",
                   "qualifications": "unknown", "event_type": "supplier disruption",
                   "status": "ongoing"}], "missing": []}
        trace = {"calls": kit.trace, "reads": kit.reads,
                 "inspected_paths": [list(path) for path in kit.inspected_paths]}
        verified = validate_evidence(answer, trace, kit.data)
        self.assertEqual(verified["events"][0]["verified_paths"][0]["hops"], 3)

    def test_wrong_depth_via_cannot_unlock_third_hop(self):
        kit = Toolkit(capability())
        kit.call("inspect_event", source_id="direct")
        self.assertIn("error", kit.call("exposure_paths", hops=3, via="mid1"))
        self.assertIn("error", kit.call("search_events", query="outage", hops=3, via="mid1"))

    def test_validation_excludes_uninspected_three_hop_path_on_multi_tag_story(self):
        data = capability()
        data["relations"][2]["permnos"].append(2)
        kit = Toolkit(data)
        kit.call("exposure_paths")
        kit.call("search_events", query="outage")
        kit.call("inspect_event", source_id="direct")
        answer = {"events": [{"source_id": "direct", "page": 0,
                   "fact": "synthetic outage", "interpretation": "possible input risk",
                   "qualifications": "unknown", "event_type": "supplier disruption",
                   "status": "ongoing"}], "missing": []}
        trace = {"calls": kit.trace, "reads": kit.reads,
                 "inspected_entities": sorted(kit.inspected_entities)}
        result = validate_evidence(answer, trace, data)
        self.assertEqual([p["hops"] for p in result["events"][0]["verified_paths"]], [1])

    def test_co_mentions_remain_retrieval_leads(self):
        data = capability()
        data["relations"].append({"type": "comention", "permnos": [5],
                                  "hops": 1, "path": [], "steps": [],
                                  "vintage": "2025-07-31"})
        data["documents"]["co"] = {**data["documents"]["direct"],
            "source_id": "co", "ids": [5]}
        kit = Toolkit(data)
        paths = kit.call("exposure_paths")
        self.assertIn("comention", [p.get("type") for p in paths["paths"]])
        self.assertNotIn("error", kit.call("inspect_event", source_id="co"))

    def test_same_intermediate_entity_on_wrong_branch_does_not_unlock_hop_three(self):
        data = capability()
        # mid2 is reachable through either mid1 or other; the inspected path
        # through other cannot certify the uninspected prefix through mid1.
        data["relations"][1]["permnos"] = [99]
        data["relations"].append({"entity": "other", "permnos": [5],
            "hops": 1, "path": ["target", "other"],
            "steps": [{"neighbor_role": "supplier"}], "vintage": "2025-06-06"})
        data["relations"].append({"entity": "mid2", "permnos": [3],
            "hops": 2, "path": ["target", "other", "mid2"],
            "steps": [{"neighbor_role": "supplier"}] * 2, "vintage": "2025-06-06"})
        data["documents"]["other"] = {**data["documents"]["direct"],
            "source_id": "other", "ids": [5]}
        kit = Toolkit(data)
        kit.call("inspect_event", source_id="direct")
        kit.call("inspect_event", source_id="other")
        kit.call("inspect_event", source_id="second")
        self.assertIn("error", kit.call("inspect_event", source_id="third"))


if __name__ == "__main__":
    unittest.main()
