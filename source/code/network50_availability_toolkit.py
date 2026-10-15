"""Outcome-free retrieval follow-up, separate from the frozen experiment toolkit."""
from collections import Counter
from datetime import datetime
import argparse
from pathlib import Path

from news_agent_batched_runtime_2025 import read, write, digest
from news_network_50_2025.config import OUT
from news_network_50_2025.toolkit import Toolkit


class AvailabilityToolkit(Toolkit):
    def inventory(self, group: str = "all", offset: int = 0, limit: int = 10):
        """List available source IDs, typed retrieval leads and reading status; no forecast labels."""
        groups = {"all", "own_common", "direct_economic", "comention_without_direct", "higher_hop_only", "unlinked"}
        if group not in groups or not isinstance(offset, int) or offset < 0 or not 1 <= limit <= 20:
            raise ValueError("Choose a documented source group, nonnegative offset and limit 1..20")
        cutoff = datetime.fromisoformat(self.data["case"]["asof"])
        rows = []
        for sid, doc in self.docs.items():
            paths = self.paths_for(doc) if self.data["network"] else []
            direct = [p for p in paths if p["hops"] == 1 and p.get("type") != "comention"]
            co = [p for p in paths if p.get("type") == "comention"]
            kind = ("own_common" if not self.data["network"] else "direct_economic" if direct else
                    "comention_without_direct" if co else "higher_hop_only" if paths else "unlinked")
            eligible = (not self.data["network"] or any(p["hops"] == 1 or
                        any(v in self.inspected_entities for v in p.get("path", [])[1:-1]) for p in paths))
            leads = [{k:p.get(k) for k in ("entity", "name", "hops", "type", "steps")} for p in direct[:3]]
            rows.append({"source_id": sid, "headline": doc["headline"], "available_at": doc["available_at"],
                         "news_role": "related_company" if self.data["network"] else
                             "own_tagged" if self.data["case"].get("permno") in doc.get("ids", []) else "common_or_sector",
                         "age_calendar_days": round((cutoff-datetime.fromisoformat(doc["available_at"])).total_seconds()/86400, 2),
                         "group": kind, "inspectable_now": eligible,
                         "inspected_pages": sorted(int(p) for p in self.reads.get(sid, {})),
                         "page_count": len(self.paragraphs(doc)), "direct_economic_leads": leads,
                         "direct_economic_path_count": len(direct), "comention_path_count": len(co),
                         "blocked_intermediates": sorted({v for p in paths if p["hops"] > 1
                             for v in p.get("path", [])[1:-1] if v not in self.inspected_entities})[:10]})
        priority = {"own_common":0, "direct_economic":0, "comention_without_direct":1, "higher_hop_only":2, "unlinked":3}
        rows.sort(key=lambda r:(priority[r["group"]], r["age_calendar_days"], r["source_id"]))
        selected = [r for r in rows if group == "all" or r["group"] == group]
        return {"sources": selected[offset:offset+limit], "counts": dict(Counter(r["group"] for r in rows)),
                "news_role_counts": dict(Counter(r["news_role"] for r in rows)),
                "total_in_group": len(selected), "next_offset": offset+limit if len(selected) > offset+limit else None,
                "scope": "Only the preselected case/arm packet, not the full Reuters archive or all partner news.",
                "rules": "Metadata graph paths are retrieval leads, not verified material exposure. Unread is not unavailable. No source in this packet is not evidence of no news. Co-mention is not a supplier/customer relation.",
                "missing_mapping": self.data.get("coverage", {}).get("unmapped_paths")}

    def call(self, tool, **args):
        if tool != "inventory":
            return super().call(tool, **args)
        try:
            result = self.inventory(**args)
            ok = True
        except (ValueError, KeyError, TypeError) as exc:
            result, ok = {"error": str(exc)}, False
        self.trace.append({"tool": tool, "args": args, "ok": ok, "result": result})
        if self.trace_path:
            write(self.trace_path, {"capability_hash": digest(self.data), "calls": self.trace,
                                   "reads": self.reads, "inspected_entities": sorted(self.inspected_entities)})
        return result


def audit():
    from types import SimpleNamespace
    from news_network_50_2025.workflow import Runner
    selected = {"DE:2025-10-15", "AMD:2025-10-01", "CAT:2025-10-15", "WMT:2025-10-01", "ORCL:2025-10-15"}
    records = []
    cases = read(OUT / "cases.json")
    def no_provider(*args, **kwargs):
        raise ValueError("Acceptance audit may use only the active experiment's cached evidence")
    runner = Runner(OUT, SimpleNamespace(native=no_provider, call=no_provider), cases,
                    read(OUT / "anchor.json")["anchors"], digest(read(OUT / "RUN_MANIFEST.json")))
    for case in cases:
        if case["case_id"] not in selected:
            continue
        evidence = runner.evidence(case, True)
        cap = read(OUT / "capabilities" / (evidence["key"]+".json"))
        if cap["case"]["split"] != "train":
            raise ValueError("Source-acceptance cases must be development only")
        kit = AvailabilityToolkit(cap)
        inventory = kit.call("inventory", limit=20)
        if len(inventory["sources"]) != len(cap["documents"]):
            raise ValueError("Small acceptance packet must enumerate every source")
        records.append({"case_id":evidence["case_id"], "old_tool_calls":len(evidence["tool_calls"]),
                        "old_accepted_events":len(evidence["events"]), "inventory":inventory})
    if selected != {r["case_id"] for r in records}:
        raise ValueError("Missing planned development acceptance case")
    write(OUT / "coverage_followup/availability_acceptance.json", {
        "llm_calls":0, "labels_read":False, "original_experiment_changed":False,
        "source_inventory_correct":True, "records":records,
        "limitation":"CPU acceptance only; fewer actual agent calls or improved forecasting have not been established."})
    print([{k:r[k] for k in ("case_id","old_tool_calls","old_accepted_events")} |
           {"source_groups":r["inventory"]["counts"]} for r in records], flush=True)


def serve(capfile, tracefile):
    from mcp.server.mcpserver import MCPServer
    kit = AvailabilityToolkit(read(capfile), tracefile)
    server = MCPServer("news")

    @server.tool()
    def inventory(group: str = "all", offset: int = 0, limit: int = 10) -> dict:
        """List actually available source IDs, graph lead types and unread/blocked status. No source in this packet is not proof of no news."""
        return kit.call("inventory", group=group, offset=offset, limit=limit)

    @server.tool()
    def market_state() -> dict:
        """Read pre-cutoff financial state and calendar coverage, not labels."""
        return kit.call("market_state")

    @server.tool()
    def search_events(query: str, limit: int = 5, hops: int = 1, via: str = "") -> dict:
        """Search the inventory's scoped pool. Inspect sources before citation."""
        return kit.call("search_events", query=query, limit=limit, hops=hops, via=via)

    @server.tool()
    def inspect_event(source_id: str, page: int = 0) -> dict:
        """Read an available article page verbatim; two-hop-only reads need an inspected intermediate."""
        return kit.call("inspect_event", source_id=source_id, page=page)

    @server.tool()
    def inspect_events(requests: list[dict]) -> dict:
        """Batch up to six source-page reads, with the same evidence restrictions."""
        return kit.call("inspect_events", requests=requests)

    @server.tool()
    def event_logbook(query: str = "pending earnings acquisition disruption", limit: int = 5) -> dict:
        """Read dated source history and calendar, never old model forecasts as facts."""
        return kit.call("event_logbook", query=query, limit=limit)

    if kit.data["network"]:
        @server.tool()
        def exposure_paths(hops: int = 1, via: str = "", query: str = "", offset: int = 0) -> dict:
            """Look up target-scoped graph paths. Inventory identifies which sources are actually available."""
            return kit.call("exposure_paths", hops=hops, via=via, query=query, offset=offset)

    server.run(transport="stdio")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true")
    p.add_argument("capability", nargs="?")
    p.add_argument("trace", nargs="?")
    args = p.parse_args()
    if args.audit:
        audit()
    elif args.capability and args.trace:
        serve(args.capability, args.trace)
    else:
        p.error("Provide --audit or capability and trace paths")
