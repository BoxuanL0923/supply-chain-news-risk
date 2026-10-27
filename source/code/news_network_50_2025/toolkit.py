"""Read-only, single-case capability. No label, filesystem, URL or arbitrary query tool."""
from datetime import datetime
import math
import re
import os
from news_agent_batched_runtime_2025 import read, write, digest
from .graph_skills import utc
from .graph_projection import bound_paths, indexed_snapshot, project_path
from .label_firewall import boundary, check_payload
from contextlib import nullcontext


class Toolkit:
    def __init__(self, capability, trace_path=None):
        if 'enable_graph_router' not in capability and os.environ.get('SC_ENABLE_GRAPH_ROUTER', '0') == '1':
            capability = dict(capability, enable_graph_router=True)
        self.data = capability
        self.trace_path = trace_path
        self.trace = []
        self.reads = {}
        self.inspected_entities = set()
        self.inspected_paths = set()
        self.docs = capability["documents"]
        check_payload(capability)
        self.router_events, self.router_receipts = [], []
        self.router_paths = None
        self.projection = indexed_snapshot(capability["graph_snapshot"]) if capability.get("enable_graph_router") and capability.get("graph_snapshot") else None
        cutoff = utc(capability["case"]["asof"])
        if any(utc(r["available_at"]) >= cutoff for r in self.docs.values()):
            raise ValueError("At/after-cutoff document in tool capability")
        if capability.get("enable_graph_router") and capability["network"]:
            self.refresh_router()

    def refresh_router(self):
        from .runner_receipts import route_capability
        from .graph_router import RouterRules
        route, events = route_capability(self.data, self.reads, inspected_paths=self.inspected_paths,
            rules=RouterRules(max_hops=self.data.get("router_max_hops", 3)))
        self.router_paths = {tuple(p["path"]) for p in route["paths"]}
        self.router_receipts.append(route)
        self.router_events.extend(events)

    def call(self, tool, **args):
        with boundary() if self.data.get("enable_graph_router") else nullcontext():
            return self._call(tool, **args)

    def _call(self, tool, **args):
        try:
            if tool not in {"market_state", "search_events", "inspect_event", "inspect_events", "exposure_paths", "event_logbook", "sc_graph_degree", "sc_graph_triangles"}:
                raise ValueError("Unknown tool")
            result = getattr(self, tool)(**args)
            item = {"tool": tool, "args": args, "ok": True, "result": result}
        except (ValueError, KeyError, TypeError) as exc:
            result = {"error": str(exc), "instruction": "Correct arguments or report the coverage gap; do not invent evidence"}
            item = {"tool": tool, "args": args, "ok": False, "result": result}
        self.trace.append(item)
        if self.data.get("enable_graph_router"):
            self.router_events.append(dict(event="tool_call", receipt=item))
        if self.trace_path:
            write(self.trace_path, {"capability_hash": digest(self.data), "calls": self.trace,
                                   "reads": self.reads, "inspected_entities": sorted(self.inspected_entities),
                                   "router_events": self.router_events, "router_receipts": self.router_receipts,
                                   "inspected_paths": [list(path) for path in sorted(self.inspected_paths)]})
        return result

    def market_state(self):
        c = self.data["case"]
        return {"case": c, "calendar": self.data["calendar"], "controls": self.data["controls"],
                "ratios_to_garch": {str(n): math.exp(c["state"][f"log_ret_rv{n}"]-c["b"]) for n in (1, 5, 22)},
                "coverage": self.data["coverage"],
                "anchor_definition": "Raw GARCH; calendar and news are NOT already included",
                "asof_note": "Historical close snapshot, not an executable trade exactly at that close"}

    def _graph(self):
        from .graph_skills import Snapshot
        if not self.data["network"] or "graph_snapshot" not in self.data:
            raise ValueError("Dated graph snapshot and monthly universe required in network capability")
        if not hasattr(self, "_snapshot"):
            self._snapshot = Snapshot(self.data["graph_snapshot"], self.data["case"]["asof"])
        return self._snapshot

    def sc_graph_degree(self, max_hops=1, thresh_hub=90):
        return self._graph().degree_result(self.data["graph_firm_id"], max_hops,
                                          self.inspected_paths, thresh_hub)

    def sc_graph_triangles(self, neighbour=None, max_hops=1, thresh_ra=75, thresh_clust=.5):
        return self._graph().triangle_result(self.data["graph_firm_id"], neighbour, max_hops,
                                            self.inspected_paths, thresh_ra, thresh_clust)

    def search_events(self, query: str, limit: int = 5, hops: int = 1, via: str = ""):
        if not isinstance(query, str) or not 0 < len(query) <= 800 or not 1 <= limit <= 10:
            raise ValueError("Provide a focused query and limit 1..10")
        if type(hops) is not int or hops not in (1, 2, 3):
            raise ValueError("Only one/two/three-hop evidence allowed")
        if self.data["network"] and not self._valid_via(hops, via):
            raise ValueError("Inspect each preceding intermediate before searching a deeper hop")
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        hits = []
        for sid, doc in self.docs.items():
            paths = self.paths_for(doc)
            if self.data["network"]:
                paths = [p for p in paths if self._accessible_path(p, hops, via)]
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
        if self.data.get('enable_graph_router'):
            return bound_paths(self.data.get("relations", []), doc, self.data["case"]["asof"], self.projection)
        # Preserve legacy staged-retrieval fixtures/caches when the new gate is
        # disabled. The selector still enforces its own dated path contract.
        return [p for p in self.data.get('relations', []) if set(p['permnos']) & set(doc['ids'])]

    def _valid_via(self, hops, via):
        if hops == 1:
            return True
        return any(
            p.get("hops") == hops - 1 and p.get("entity") == via
            and tuple(p.get("path", [])) in self.inspected_paths
            for p in self.data.get("relations", []))

    def _accessible_path(self, path, max_hops, via=None):
        hops = path.get("hops")
        chain = path.get("path", [])
        if path.get("type") == "comention":
            return hops == 1 and chain == []
        if type(hops) is not int or not 1 <= hops <= max_hops or len(chain) != hops + 1:
            return False
        if self.router_paths is not None and tuple(chain) not in self.router_paths:
            return False
        if any(tuple(chain[:depth + 1]) not in self.inspected_paths
               for depth in range(1, hops)):
            return False
        return hops < max_hops or hops == 1 or via is None or chain[-2] == via

    def inspect_event(self, source_id: str, page: int = 0):
        if source_id not in self.docs:
            raise ValueError("Source outside this case/arm/as-of capability")
        doc = self.docs[source_id]
        paragraphs = self.paragraphs(doc)
        if isinstance(page, bool) or not isinstance(page, int) or not 0 <= page < len(paragraphs):
            raise ValueError("Page outside article; request page 0 or search to find a relevant page")
        paths = self.paths_for(doc)
        if self.data["network"]:
            paths = [p for p in paths if self._accessible_path(p, 3)]
            if not paths:
                raise ValueError("Inspect every preceding intermediate before reading a deeper-hop source")
        before = set(self.inspected_paths)
        self.inspected_entities.update(p["entity"] for p in paths if p.get("entity"))
        self.inspected_paths.update(tuple(p["path"]) for p in paths if p.get("path"))
        self.reads.setdefault(source_id, {})[str(page)] = paragraphs[page]
        if self.data.get("enable_graph_router") and self.data["network"] and before != self.inspected_paths:
            self.refresh_router()
        return {"source_id": source_id, "headline": doc["headline"], "available_at": doc["available_at"],
                "page": page, "page_count": len(paragraphs), "text": paragraphs[page],
                "paths": paths[:5], "citation_eligible": True,
                "qualification_check": "Read adjoining pages for update, negation and mitigation before claiming a mechanism"}

    def exposure_paths(self, hops: int = 1, via: str = "", query: str = "", offset: int = 0):
        if not self.data["network"]:
            raise ValueError("Graph unavailable in OWN arm")
        if type(hops) is not int or hops not in (1, 2, 3) or not self._valid_via(hops, via):
            raise ValueError("Deeper hops require an inspected preceding intermediate")
        if offset < 0 or len(query) > 200:
            raise ValueError("Invalid query or offset")
        normalized = " ".join(query.lower().split())
        target_aliases = {" ".join(str(self.data["case"].get(k, "")).lower().split())
                          for k in ("ticker", "company")}
        target_query = bool(normalized) and normalized in target_aliases
        terms = set(re.findall(r"[a-z0-9]+", normalized)) if not target_query else set()
        available = [p for p in self.data["relations"] if self._accessible_path(p, hops, via)]
        if self.data.get('enable_graph_router'):
            available = [project_path(p, self.data["case"]["asof"], self.projection) for p in available]
        paths = [p for p in available if terms <= set(re.findall(r"[a-z0-9]+",
                 " ".join(str(p.get(k, "")) for k in ("name", "entity", "permnos")).lower()))]
        return {"paths": paths[offset:offset+15], "next_offset": offset+15 if len(paths) > offset+15 else None,
                "total": len(paths), "graph_date": "2025-06-06",
                "available_before_query": len(available), "target_query_interpreted_as_root_lookup": target_query,
                "query_note": "This tool is scoped to the target. Empty query lists accessible paths. No filtered match is not no network. Inspect each intermediate before requesting the next hop via its entity ID.",
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
    def sc_graph_degree(max_hops: int = 1, thresh_hub: float = 90) -> dict:
        """/sc-graph-degree: dated degree statistics; inspected prefixes required beyond one hop."""
        return kit.call("sc_graph_degree", max_hops=max_hops, thresh_hub=thresh_hub)

    @server.tool()
    def sc_graph_triangles(neighbour: str = None, max_hops: int = 1,
                           thresh_ra: float = 75, thresh_clust: float = .5) -> dict:
        """/sc-graph-triangles: triangles, clustering and RA; graph paths are leads only."""
        return kit.call("sc_graph_triangles", neighbour=neighbour, max_hops=max_hops,
                        thresh_ra=thresh_ra, thresh_clust=thresh_clust)

    @server.tool()
    def market_state() -> dict:
        """Read exact pre-cutoff GARCH, prices, lagged variance and earnings-calendar coverage."""
        return kit.call("market_state")

    @server.tool()
    def search_events(query: str, limit: int = 5, hops: int = 1, via: str = "") -> dict:
        """Search the scoped Reuters pool. Inspect result pages; hops two/three need inspected intermediates."""
        return kit.call("search_events", query=query, limit=limit, hops=hops, via=via)

    @server.tool()
    def inspect_event(source_id: str, page: int = 0) -> dict:
        """Read an article page verbatim, with date and typed paths. Subsequent pages preserve qualifications."""
        return kit.call("inspect_event", source_id=source_id, page=page)

    @server.tool()
    def exposure_paths(hops: int = 1, via: str = "", query: str = "", offset: int = 0) -> dict:
        """Target is preselected. Empty query lists neighbors; query filters names. Hops two/three need inspected via entities. OWN has no graph."""
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
