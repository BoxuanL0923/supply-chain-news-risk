"""Read-only, single-case capability. No label, filesystem, URL or arbitrary query tool."""
from datetime import datetime
import math
import re
from news_agent_batched_runtime_2025 import read, write, digest


class Toolkit:
    def __init__(self, capability, trace_path=None):
        self.data = capability
        self.trace_path = trace_path
        self.trace = []
        self.reads = {}
        self.inspected_entities = set()
        self.docs = capability["documents"]
        cutoff = datetime.fromisoformat(capability["case"]["asof"])
        if any(datetime.fromisoformat(r["available_at"]) >= cutoff for r in self.docs.values()):
            raise ValueError("At/after-cutoff document in tool capability")
        if any(k in capability for k in ("y", "outcome", "residual", "labels")):
            raise ValueError("Forbidden outcome field")

    def call(self, tool, **args):
        try:
            if tool not in {"market_state", "search_events", "inspect_event", "inspect_events", "exposure_paths", "event_logbook"}:
                raise ValueError("Unknown tool")
            result = getattr(self, tool)(**args)
            item = {"tool": tool, "args": args, "ok": True, "result": result}
        except (ValueError, KeyError, TypeError) as exc:
            result = {"error": str(exc), "instruction": "Correct arguments or report the coverage gap; do not invent evidence"}
            item = {"tool": tool, "args": args, "ok": False, "result": result}
        self.trace.append(item)
        if self.trace_path:
            write(self.trace_path, {"capability_hash": digest(self.data), "calls": self.trace,
                                   "reads": self.reads, "inspected_entities": sorted(self.inspected_entities)})
        return result

    def market_state(self):
        c = self.data["case"]
        return {"case": c, "calendar": self.data["calendar"], "controls": self.data["controls"],
                "ratios_to_garch": {str(n): math.exp(c["state"][f"log_ret_rv{n}"]-c["b"]) for n in (1, 5, 22)},
                "coverage": self.data["coverage"],
                "anchor_definition": "Raw GARCH; calendar and news are NOT already included",
                "asof_note": "Historical close snapshot, not an executable trade exactly at that close"}

    def search_events(self, query: str, limit: int = 5, hops: int = 1, via: str = ""):
        if not isinstance(query, str) or not 0 < len(query) <= 800 or not 1 <= limit <= 10:
            raise ValueError("Provide a focused query and limit 1..10")
        if hops not in (1, 2):
            raise ValueError("Only one/two-hop evidence allowed")
        if self.data["network"] and hops == 2 and via not in self.inspected_entities:
            raise ValueError("First inspect direct-neighbor evidence; supply its entity id as via")
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        hits = []
        for sid, doc in self.docs.items():
            paths = self.paths_for(doc)
            if self.data["network"]:
                paths = [p for p in paths if p["hops"] <= hops and (p["hops"] == 1 or via in p["path"])]
                if not paths:
                    continue
            paragraphs = self.paragraphs(doc)
            scored = [(len(terms & set(re.findall(r"[a-z0-9]+", text.lower()))), i)
                      for i, text in enumerate(paragraphs)]
            score, page = max(scored, default=(0, 0))
            score += 2*len(terms & set(re.findall(r"[a-z0-9]+", doc["headline"].lower())))
            if score == 0:
                continue
            hits.append((score, sid, page, paths))
        hits.sort(key=lambda v: (-v[0], -datetime.fromisoformat(self.docs[v[1]]["available_at"]).timestamp(), v[1]))
        return {"results": [{"source_id": sid, "headline": self.docs[sid]["headline"],
                            "available_at": self.docs[sid]["available_at"], "page": page,
                            "paths": paths[:3], "citation_eligible": False} for _, sid, page, paths in hits[:limit]],
                "eligible_candidates": len(hits), "backend": "lexical paragraph search in locally MiniLM-ranked scoped pool",
                "coverage": self.data["coverage"],
                "search_scope_note": "An empty search means no lexical match in this scoped candidate pool, not no news or exposure"}

    @staticmethod
    def paragraphs(doc):
        # Entire paragraphs, split only at sentence boundaries when extremely long.
        parts = [p.strip() for p in re.split(r"\n\s*\n|\r?\n", doc["body"]) if p.strip()]
        result, current = [], ""
        for part in parts:
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", part) if len(part) > 5000 else [part]:
                if current and len(current)+len(sentence) > 5000:
                    result.append(current)
                    current = ""
                current += ("\n" if current else "") + sentence
        if current:
            result.append(current)
        return result or [doc["headline"]]

    def paths_for(self, doc):
        return [p for p in self.data.get("relations", []) if set(p["permnos"]) & set(doc["ids"])]

    def inspect_event(self, source_id: str, page: int = 0):
        if source_id not in self.docs:
            raise ValueError("Source outside this case/arm/as-of capability")
        doc = self.docs[source_id]
        paragraphs = self.paragraphs(doc)
        if isinstance(page, bool) or not isinstance(page, int) or not 0 <= page < len(paragraphs):
            raise ValueError("Page outside article; request page 0 or search to find a relevant page")
        paths = self.paths_for(doc)
        if self.data["network"] and not any(p["hops"] == 1 or
                any(v in self.inspected_entities for v in p.get("path", [])[1:-1]) for p in paths):
            raise ValueError("Inspect a direct intermediate before reading a two-hop-only source")
        self.inspected_entities.update(p.get("entity") for p in paths if p["hops"] == 1 and p.get("entity"))
        self.reads.setdefault(source_id, {})[str(page)] = paragraphs[page]
        return {"source_id": source_id, "headline": doc["headline"], "available_at": doc["available_at"],
                "page": page, "page_count": len(paragraphs), "text": paragraphs[page],
                "paths": paths[:5], "citation_eligible": True,
                "qualification_check": "Read adjoining pages for update, negation and mitigation before claiming a mechanism"}

    def exposure_paths(self, hops: int = 1, via: str = "", query: str = "", offset: int = 0):
        if not self.data["network"]:
            raise ValueError("Graph unavailable in OWN arm")
        if hops not in (1, 2) or (hops == 2 and via not in self.inspected_entities):
            raise ValueError("Two hops require an entity from inspected direct-neighbor evidence")
        if offset < 0 or len(query) > 200:
            raise ValueError("Invalid query or offset")
        normalized = " ".join(query.lower().split())
        target_aliases = {" ".join(str(self.data["case"].get(k, "")).lower().split())
                          for k in ("ticker", "company")}
        target_query = bool(normalized) and normalized in target_aliases
        terms = set(re.findall(r"[a-z0-9]+", normalized)) if not target_query else set()
        available = [p for p in self.data["relations"] if p["hops"] <= hops and
                     (p["hops"] == 1 or via in p["path"])]
        paths = [p for p in available if terms <= set(re.findall(r"[a-z0-9]+",
                 " ".join(str(p.get(k, "")) for k in ("name", "entity", "permnos")).lower()))]
        return {"paths": paths[offset:offset+15], "next_offset": offset+15 if len(paths) > offset+15 else None,
                "total": len(paths), "graph_date": "2025-06-06",
                "available_before_query": len(available), "target_query_interpreted_as_root_lookup": target_query,
                "query_note": "This tool is already scoped to the target. Empty query lists its neighbors. No filtered match is not no network. Use entity IDs from paths for two-hop via.",
                "assumption": "Held constant through 2025; effective-date reconstruction, not certified archived vintage",
                "rules": "Paths are leads, not proof of propagation. Shared supplier is not competition. Unknown revenue denominator. No headquarters exposure inference."}

    def inspect_events(self, requests: list[dict]):
        if not isinstance(requests, list) or not 1 <= len(requests) <= 6:
            raise ValueError("Batch one to six source_id/page requests")
        results = []
        for request in requests:
            try:
                results.append(self.inspect_event(request["source_id"], request.get("page", 0)))
            except (ValueError, KeyError, TypeError) as exc:
                results.append({"request": request, "error": str(exc)})
        return {"results": results}

    def event_logbook(self, query: str = "pending earnings acquisition disruption", limit: int = 5):
        result = self.search_events(query, limit=limit)
        return {"history": result, "calendar": self.data["calendar"],
                "state_rule": "Only dated source facts. Check updates before assigning pending/resolved. No previous forecast is a fact."}


def serve(capability_path, trace_path):
    from mcp.server.mcpserver import MCPServer
    kit = Toolkit(read(capability_path), trace_path)
    server = MCPServer("news")

    @server.tool()
    def market_state() -> dict:
        """Read exact pre-cutoff GARCH, prices, lagged variance and earnings-calendar coverage."""
        return kit.call("market_state")

    @server.tool()
    def search_events(query: str, limit: int = 5, hops: int = 1, via: str = "") -> dict:
        """Search the scoped Reuters pool. Inspect result pages before citing; two hops need an inspected intermediate."""
        return kit.call("search_events", query=query, limit=limit, hops=hops, via=via)

    @server.tool()
    def inspect_event(source_id: str, page: int = 0) -> dict:
        """Read an article page verbatim, with date and typed paths. Subsequent pages preserve qualifications."""
        return kit.call("inspect_event", source_id=source_id, page=page)

    @server.tool()
    def exposure_paths(hops: int = 1, via: str = "", query: str = "", offset: int = 0) -> dict:
        """Target is preselected. Empty query lists neighbors; query filters partner names. For two hops, via is an inspected entity ID. OWN has no graph."""
        return kit.call("exposure_paths", hops=hops, via=via, query=query, offset=offset)

    @server.tool()
    def inspect_events(requests: list[dict]) -> dict:
        """Efficiently read 1..6 article pages at once: requests=[{source_id, page}]. Same validation as inspect_event."""
        return kit.call("inspect_events", requests=requests)

    @server.tool()
    def event_logbook(query: str = "pending earnings acquisition disruption", limit: int = 5) -> dict:
        """Retrieve source-backed event history and calendar, not remembered model predictions or outcomes."""
        return kit.call("event_logbook", query=query, limit=limit)

    server.run(transport="stdio")


if __name__ == "__main__":
    import sys
    serve(sys.argv[1], sys.argv[2])
