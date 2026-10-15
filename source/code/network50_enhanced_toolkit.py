"""Development-only combined toolkit; never substituted into frozen main forecasts."""
import argparse
from news_agent_batched_runtime_2025 import read, write, digest
from news_network_50_2025.config import OUT
from network50_availability_toolkit import AvailabilityToolkit
from network50_archive_tools import load
from network50_cached_followups import risk_arithmetic


class EnhancedToolkit(AvailabilityToolkit):
    def __init__(self, capability, archive, anchor, trace_path=None):
        super().__init__(capability, trace_path)
        if archive.case["asof"] != capability["case"]["asof"] or archive.case["permno"] != capability["case"]["permno"]:
            raise ValueError("Archive and packet must share target and cutoff")
        if archive.network != capability["network"]:
            raise ValueError("Archive and packet must share arm")
        self.archive, self.anchor = archive, anchor
        self.submissions = []
        self.archive.reads = self.reads
        self.archive.inspected_entities = self.inspected_entities

    def market_state(self):
        result = super().market_state()
        result["anchor_definition"] = "b is raw conditional GARCH. OLS_anchor is a separate training-fitted conventional predictor including the listed controls."
        result["OLS_anchor"] = self.anchor
        result["computed_comparisons"] = risk_arithmetic(self.data["case"],self.anchor,0.)
        return result

    def search_archive(self, query, scope="own", limit=8):
        return self.archive.search(query,scope,limit)

    def inspect_archive(self, source_id, page=0):
        result = self.archive.inspect(source_id,page)
        self.reads.setdefault(source_id,{})[str(page)] = result["text"]
        return result

    def risk_check(self, score):
        if isinstance(score,bool):
            raise ValueError("A numerical log-risk residual, not a boolean, is required")
        return risk_arithmetic(self.data["case"],self.anchor,score)

    def submit_forecast(self, forecast):
        if not self.data.get('diagnostic_submission'):
            raise ValueError('Forecast submission unavailable in this capability')
        from network50_oneweek_diagnostic import validate
        audit=validate(forecast,{'calls':self.trace,'reads':self.reads},self.data,self.anchor,
                       self.data.get('diagnostic_own'))
        receipt=digest({'case_id':self.data['case']['case_id'],'forecast':forecast,'audit':audit})
        self.submissions.append({'receipt':receipt,'forecast':forecast,'audit':audit})
        return {'accepted':True,'receipt':receipt,'score':forecast['score'],'computed':audit['computed'],
                'instruction':'This checked forecast is recorded. Return the receipt; do not restate or change its score in final prose.'}

    def save_trace(self):
        if self.trace_path:
            write(self.trace_path,{"capability_hash":digest(self.data),"calls":self.trace,"reads":self.reads,
                "inspected_entities":sorted(self.inspected_entities),"labels_accessible":False,
                'submissions':self.submissions})

    def call(self, tool, **args):
        if tool not in {"search_archive","inspect_archive","risk_check","submit_forecast"}:
            result=super().call(tool,**args)
            self.save_trace()
            return result
        try:
            result = getattr(self,tool)(**args)
            ok = True
        except (ValueError,KeyError,TypeError,OverflowError) as exc:
            result,ok = {"error":str(exc)},False
        self.trace.append({"tool":tool,"args":args,"ok":ok,"result":result})
        self.save_trace()
        return result


def serve(capfile, tracefile):
    from mcp.server.mcpserver import MCPServer
    cap = read(capfile)
    archive,_ = load(cap["case"]["case_id"],cap["network"])
    anchor = read(OUT / "anchor.json")["anchors"][cap["case"]["case_id"]]
    kit = EnhancedToolkit(cap,archive,anchor,tracefile)
    server = MCPServer("news")

    @server.tool()
    def inventory(group: str = "all", offset: int = 0, limit: int = 10) -> dict:
        """List shortlist source IDs, freshness, available pages and typed graph leads; empty shortlist is not no news."""
        return kit.call("inventory",group=group,offset=offset,limit=limit)

    @server.tool()
    def market_state() -> dict:
        """Read cutoff-safe financial state and explicit distinctions between raw GARCH and the conventional OLS anchor."""
        return kit.call("market_state")

    @server.tool()
    def search_events(query: str, limit: int = 5, hops: int = 1, via: str = "") -> dict:
        """Search shortlisted paragraphs. If facts are missing, use search_archive rather than repeating broad queries."""
        return kit.call("search_events",query=query,limit=limit,hops=hops,via=via)

    @server.tool()
    def inspect_event(source_id: str, page: int = 0) -> dict:
        """Read a shortlisted source page with qualifications and typed paths."""
        return kit.call("inspect_event",source_id=source_id,page=page)

    @server.tool()
    def inspect_events(requests: list[dict]) -> dict:
        """Batch up to six shortlisted source-page reads, retaining provenance."""
        return kit.call("inspect_events",requests=requests)

    @server.tool()
    def event_logbook(query: str = "pending earnings acquisition disruption", limit: int = 5) -> dict:
        """Read source history and calendar coverage; unknown is not absent."""
        return kit.call("event_logbook",query=query,limit=limit)

    @server.tool()
    def search_archive(query: str, scope: str = "own", limit: int = 8) -> dict:
        """Search beyond shortlist: own/common, or network when enabled. Exact headline words are conjunctive. Use focused terms."""
        return kit.call("search_archive",query=query,scope=scope,limit=limit)

    @server.tool()
    def inspect_archive(source_id: str, page: int = 0) -> dict:
        """Read a verified raw Reuters archive page. Read qualifications before assigning event status or transmission."""
        return kit.call("inspect_archive",source_id=source_id,page=page)

    @server.tool()
    def risk_check(score: float) -> dict:
        """Translate a proposed log-risk residual into variance ratios and geometric RMS. Check narrative direction, units and exact five exchange sessions. No outcome access."""
        return kit.call("risk_check",score=score)

    if cap["network"]:
        @server.tool()
        def exposure_paths(hops: int = 1, via: str = "", query: str = "", offset: int = 0) -> dict:
            """Look up typed graph paths; inspect direct evidence before two-hop traversal. A link is not materiality."""
            return kit.call("exposure_paths",hops=hops,via=via,query=query,offset=offset)

    if cap.get('diagnostic_submission'):
        @server.tool()
        def submit_forecast(forecast: dict) -> dict:
            """Validate the actual final score, horizon and inspected evidence. Resolve returned errors in this tool loop. An accepted receipt, not later free-text revisions, defines the forecast."""
            return kit.call('submit_forecast',forecast=forecast)

    try:
        server.run(transport="stdio")
    finally:
        archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capability")
    parser.add_argument("trace")
    args = parser.parse_args()
    serve(args.capability,args.trace)
