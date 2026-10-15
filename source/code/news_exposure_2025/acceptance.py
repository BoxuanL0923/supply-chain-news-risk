"""Outcome-blind real-source retrieval checks; does not load risk labels."""
import argparse
from pathlib import Path
from news_agent_batched_runtime_2025 import read, write
from .config import OUT
from .graph import Graph
from .tools import ToolSession


def check(out=OUT):
    cases = read(out / "cases.json")
    calendar, graph = read(out / "calendar_features.json"), Graph(read(out / "graph.json"))
    results = []
    for ticker, query in (("NVDA", "export restrictions China chips demand"),
                          ("CAT", "tariffs manufacturing orders demand"),
                          ("BBY", "earnings guidance tariffs consumer demand"),
                          ("MSFT", "cloud capital expenditure capacity outage recovery")):
        case = next(c for c in cases if c["ticker"] == ticker and c["date"] == "2025-09-19")
        session = ToolSession(case, calendar[case["case_id"]], graph, out, True)
        own = session.call("search_events", {"query": query, "channel": "own", "limit": 2})
        common = session.call("search_events", {"query": "U.S. Federal Reserve September interest rate decision inflation employment", "channel": "common", "limit": 2})
        related = session.call("search_related_events", {"query": query, "limit": 2})
        state = session.call("get_event_state", {"query": "pending acquisition earnings capacity disruption recovery", "limit": 2})
        if not own["results"] or not common["results"]:
            raise ValueError("Expected source coverage unavailable for " + ticker)
        results.append({"case_id": case["case_id"], "query": query, "own": own, "common": common,
                        "related": related, "state": state})
        results[-1]["inspected"] = [session.call("read_evidence", {"source_id": r["source_id"], "chunk": r["chunk"]})
                                    for r in own["results"]+related["results"]]
        if ticker == "NVDA":
            results[-1]["focused_common"] = session.call("search_events", {"query": "Fed cuts interest rates", "channel": "common", "days_back": 7, "limit": 4})
            results[-1]["named_exposure"] = session.call("lookup_exposure", {"query": "Taiwan", "hops": 1})
    value = {"mechanical_pass": True, "review_status": "requires source relevance review before model launch",
             "outcome_labels_loaded": False, "examples": results}
    write(out / "retrieval_acceptance.json", value)
    print("Retrieved", len(results), "review cases; no model calls or outcome labels.", flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    check(args.out)
