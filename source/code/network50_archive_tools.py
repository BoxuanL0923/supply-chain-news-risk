"""Read-only archive fallback, independent of the frozen short-list experiment."""
import argparse
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timedelta
import json
import re
import sqlite3
import threading
from types import SimpleNamespace

from news_agent_batched_runtime_2025 import read, write
from news_network_50_2025.config import OUT, OLD, ROOT
from news_network_50_2025.toolkit import Toolkit
from news_exposure_2025 import data as prior
import news_event_risk_data_2025 as legacy


def parse_record(raw):
    # Reuters record framing precedes JSON and can have trailing array punctuation.
    start = raw.find(b"{")
    if start < 0:
        return None
    value,_ = json.JSONDecoder().raw_decode(raw[start:].decode("utf-8"))
    return value


def source_reader(row):
    return legacy.read_article(row, SimpleNamespace(parse_line=parse_record))


class ArchiveSearch:
    def __init__(self, latest_rows, case, relations, network=False, reader=source_reader):
        self.case, self.reader, self.records = case, reader, {}
        self.network, self.relations = network, deepcopy(relations) if network else []
        self.paths_by_permno = {}
        for path in self.relations:
            for step in path.get("steps", []):
                step.pop("revenue_pct", None)
            path["materiality"] = "unknown; unverified revenue-denominator weights withheld"
            for permno in path["permnos"]:
                self.paths_by_permno.setdefault(permno, []).append(path)
        self.body_cache = {}
        self.inspected_entities, self.opened_network = set(), set()
        cutoff = datetime.fromisoformat(case["asof"])
        neighbors = {n for p in self.relations for n in p["permnos"]}
        self.db_lock = threading.Lock()
        self.db = sqlite3.connect(":memory:",check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE headlines USING fts5(source_id UNINDEXED, headline, scope UNINDEXED)")
        for row in latest_rows:
            at = datetime.fromisoformat(row["available_at"])
            if not row["usable"] or not cutoff-timedelta(days=31) <= at < cutoff:
                continue
            ids = set(row["ids"])
            scope = ("own" if case["permno"] in ids else "common" if prior.relevant_common(row)
                     and legacy.US_MACRO.search(row["headline"]) else "network" if network and ids & neighbors else None)
            if scope is None:
                continue
            self.records[row["source_id"]] = {**row,"scope":scope}
            self.db.execute("INSERT INTO headlines VALUES(?,?,?)",(row["source_id"],row["headline"],scope))
        self.db.commit()

    def close(self):
        self.db.close()

    def paths_for(self,row):
        paths = [p for permno in row["ids"] for p in self.paths_by_permno.get(permno, [])]
        return list({id(p): p for p in paths}.values())

    def target_topics(self):
        generic = {"update", "brief", "refile", "corrected", "stocks", "stock", "shares",
                   "share", "company", "companies", "reports", "reported", "quarter",
                   "earnings", "revenue", "profit", "profits", "sales", "forecast", "outlook",
                   "results", "market", "markets", "after", "before", "amid", "over",
                   "deal", "deals", "acquisition", "acquisitions", "merger", "mergers",
                   "buys", "buy", "bought", "purchase", "purchases", "takeover",
                   "from", "with", "their", "more", "than", "this", "that", "will",
                   "says", "said", "year", "years", "first", "second", "third", "fourth",
                   "and", "the", "for", "last", "buzz", "rpt", "analyst", "premarket",
                   "price", "target", "targets", "billion", "billions", "trillion",
                   "raises", "falls", "rise", "move", "holdings", "weekly", "day",
                   "2025", "2026", "2027", "2028", "2029", "2030"}
        own = [row["headline"] for row in self.records.values() if row["scope"] == "own"]
        company_words = set(re.findall(r"[a-z]{3,}", self.case["company"].lower()))
        company_words.update({"inc", "corp", "plc", "com", self.case["ticker"].lower()})
        counts = Counter(word for headline in own for word in re.findall(r"[a-z]{3,}", headline.lower())
                         if word not in generic and word not in company_words)
        return {word for word, _ in counts.most_common(40)}

    def prioritized(self, query="", relation_permnos=None, offset=0, limit=12):
        """Rank recent direct-economic event leads without reading outcomes or article bodies."""
        if not self.network:
            raise ValueError("Related-news search is available only in NET")
        if not isinstance(query, str) or len(query) > 150 or not isinstance(offset, int) or offset < 0 or not 1 <= limit <= 20:
            raise ValueError("Use a short query, nonnegative offset and limit 1..20")
        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower())
                 if term not in {"a", "an", "and", "for", "in", "of", "the", "to"}]
        cutoff = datetime.fromisoformat(self.case["asof"])
        topics = self.target_topics()
        patterns = (
            ("operational", r"\b(?:shutdown|shut down|outage|factory fire|plant fire|production halt|strike|recall|disruption)\b", 5),
            ("distress", r"\b(?:bankrupt|insolven|default|restructur)\w*\b", 4),
            ("deal", r"\b(?:acquir|merger|takeover|buyout)\w*\b", 3),
            ("guidance", r"\b(?:earnings|revenue|profit|forecast|guidance|outlook|results)\b", 2),
            ("policy", r"\b(?:tariff\w*|sanction\w*|export\w*|licen[cs]\w*|regulat\w*|"
             r"commerce department|trade restrict\w*)\b", 1),
        )
        candidates = []
        for row in self.records.values():
            if row["scope"] != "network":
                continue
            age = (cutoff-datetime.fromisoformat(row["available_at"])).total_seconds()/86400
            if age > 14 or (relation_permnos is not None and not set(row["ids"]) & set(relation_permnos)):
                continue
            headline = row["headline"]
            query_hits = [term for term in terms if re.search(
                r"\b"+re.escape(term[:-1] if term.endswith('s') and len(term)>4 else term)+r"\w*\b",
                headline, re.I)]
            if terms and not query_hits:
                continue
            if re.search(r"\b(?:BUZZ|price target|target price|premarket|last down|"
                         r"stock (?:gains?|falls?|slides?)|shares? (?:rise|fall|jump|soar))\b", headline, re.I):
                continue
            if re.search(r"\b(?:deals of the day|mergers and acquisitions roundup|factbox)\b", headline, re.I):
                continue
            direct = [p for p in self.paths_for(row) if p["hops"] == 1 and
                      p.get("type") != "comention" and any(
                          step.get("neighbor_role") in {"supplier", "customer", "competitor"}
                          for step in p.get("steps", []))]
            if not direct:
                continue
            match = next(((name, weight) for name, pattern, weight in patterns
                          if re.search(pattern, headline, re.I)), None)
            if match is None:
                continue
            role_bonus = 2 if any(any(step.get("neighbor_role") in {"supplier", "customer"}
                                     for step in p.get("steps", [])) for p in direct) else 0
            headline_words = set(re.findall(r"[a-z]{3,}", headline.lower()))
            topic_hits = sorted(topic for topic in topics if topic in headline_words or
                                topic+'s' in headline_words)
            brief = bool(re.match(r'^(?:RPT-|BRIEF-|BUZZ-|[A-Z][A-Z ]{6,}:)', headline))
            candidates.append((-len(query_hits), -len(topic_hits), brief,
                               -(match[1]+role_bonus+min(3, len(topic_hits))),
                               age, row["source_id"], row, direct, match[0], topic_hits, query_hits))
        candidates.sort(key=lambda item: item[:6])
        seen, selected = set(), []
        for _, _, _, _, age, _, row, paths, event_type, topic_hits, query_hits in candidates:
            story = row.get("story_id", row["source_id"])
            if story in seen:
                continue
            seen.add(story)
            selected.append({"source_id": row["source_id"], "headline": row["headline"],
                             "available_at": row["available_at"], "age_days": round(age, 2),
                             "event_type_lead": event_type, "target_topic_hits": topic_hits,
                             "query_term_hits": query_hits,
                             "path_leads": [{
                                 "entity": p.get("entity"), "name": p.get("name"),
                                 "role": p["steps"][-1].get("neighbor_role")}
                                 for p in paths[:3]], "citation_eligible": False})
        if not terms:
            buckets = {name: deque(row for row in selected if row["event_type_lead"] == name)
                       for name in ("operational", "distress", "policy", "deal", "guidance")}
            selected = []
            while any(buckets.values()):
                for bucket in buckets.values():
                    if bucket:
                        selected.append(bucket.popleft())
        return {"results": selected[offset:offset+limit], "matched": len(selected),
                "next_offset": offset+limit if len(selected) > offset+limit else None,
                "scope": "Direct economic paths and articles from the prior 14 days; target topics come only from pre-cutoff own-news headlines",
                "qualification": "Empty query gives event-type-balanced discovery. Focused query uses any term with overlap ranking, not semantic entailment. Topic overlap is a search aid, not transmission evidence; inspect and verify the target channel."}

    def network_readable(self,row):
        return any(p["hops"]==1 or any(v in self.inspected_entities for v in p.get("path",[])[1:-1])
                   for p in self.paths_for(row))

    def search(self, query, scope="own", limit=8):
        if scope not in {"own","common","network"} or scope == "network" and not self.network:
            raise ValueError("Scope unavailable in this arm")
        if not isinstance(query,str) or not 1 <= len(query) <= 400 or not 1 <= limit <= 20:
            raise ValueError("Focused query and limit 1..20 required")
        terms = re.findall(r"[a-zA-Z0-9]+",query)
        if not terms:
            raise ValueError("Query requires searchable words")
        expression = " AND ".join('"'+t+'"' for t in terms)
        with self.db_lock:
            hits = list(self.db.execute("SELECT source_id,bm25(headlines) FROM headlines WHERE headlines MATCH ? AND scope=?",
                                        (expression,scope)))
        if scope == "network":
            hits = [h for h in hits if self.network_readable(self.records[h[0]])]
        # Within a focused exact-word match, prioritize fuller reporting over price recaps.
        hits.sort(key=lambda h:(bool(re.match(r"(?:BUZZ|BRIEF|[A-Z .]+<)",self.records[h[0]]["headline"])),
                                -datetime.fromisoformat(self.records[h[0]]["available_at"]).timestamp(),h[1],h[0]))
        return {"results":[{k:self.records[s][k] for k in ("source_id","headline","available_at","scope")}
                           for s,_ in hits[:limit]], "matched":len(hits),
                "eligible_in_scope":sum(r["scope"]==scope for r in self.records.values()),
                "backend":"SQLite FTS5 headline search over latest pre-cutoff versions in the full eligible 31-day catalogue, not the initial short list",
                "limitation":"Focused terms are conjunctive, not a semantic body search. Try a shorter entity/event query if empty. Mapped-US network coverage remains incomplete; use prioritized direct leads for discovery."}

    def inspect(self, source_id, page=0):
        if source_id not in self.records:
            raise ValueError("Source outside this target/arm/as-of archive capability")
        row = self.records[source_id]
        if row["scope"] == "network":
            if not self.network_readable(row):
                raise ValueError("Inspect a direct intermediate before a two-hop-only source")
        if source_id not in self.body_cache:
            self.body_cache[source_id] = self.reader(row)
        pages = Toolkit.paragraphs({"body":self.body_cache[source_id],"headline":row["headline"]})
        if isinstance(page,bool) or not isinstance(page,int) or not 0 <= page < len(pages):
            raise ValueError("Page outside article")
        paths = self.paths_for(row)
        if row["scope"] == "network":
            self.opened_network.add(source_id)
            self.inspected_entities.update(p["entity"] for p in paths if p["hops"]==1 and p.get("entity"))
        return {"source_id":source_id,"headline":row["headline"],"available_at":row["available_at"],
                "scope":row["scope"],"page":page,"page_count":len(pages),"text":pages[page],
                "raw_sha256":row["raw_sha256"],"paths":paths[:5],
                "qualification":"Read qualifications and later pages before assigning event state. Paths are retrieval leads, not proof of material transmission."}


def load(case_id, network=False):
    case = next(c for c in read(OUT / "cases.json") if c["case_id"]==case_id)
    pool = read(OUT / "pools" / (case["date"]+".json"))["pools"][case_id]
    indexes = []
    for month in range(7,13):
        stem = f"2025-{month:02d}"
        meta = read(OLD.parent / "news_event_risk_2025/reuters_index" / (stem+".json"))
        indexes.append({**meta,"index":str(OLD / "reuters_index" / (stem+".jsonl.gz"))})
    eligible = {case["permno"]}
    if network:
        eligible |= {n for p in pool["relations"] for n in p["permnos"]}
    # Resolve superseding updates/deletions BEFORE restricting the retrieval catalogue.
    rows = prior.resolve_history(indexes,case["asof"],eligible)
    return ArchiveSearch(rows,case,pool["relations"],network),pool


def audit():
    kit,pool = load("CFLT:2025-12-08")
    result = kit.search("IBM Confluent")
    first = result["results"][0]
    pages = [kit.inspect(first["source_id"],0)]
    pages += [kit.inspect(first["source_id"],i) for i in range(1,pages[0]["page_count"])]
    body = "\n".join(p["text"] for p in pages)
    checks = {"recovered_previously_omitted_source":first["source_id"] not in pool["own_ids"],
              "cash_funding_recovered":bool(re.search(r"fund.{0,100}cash on hand",body,re.I|re.S)),
              "closing_timeline_recovered":bool(re.search(r"expected to close.{0,80}2026",body,re.I|re.S)),
              "strictly_pre_cutoff":first["available_at"]<kit.case["asof"]}
    if not all(checks.values()):
        raise ValueError("Archive acceptance failed: "+str(checks))
    write(OUT / "coverage_followup/archive_acceptance.json",{
        "llm_calls":0,"labels_read":False,"checks":checks,"search":result,"pages":pages,
        "warning":"This case was selected after test-error review. Retrieval correctness only, not an out-of-sample forecasting gain. Original experiment unchanged."})
    print(checks)
    kit.close()


def serve(case_id, network, trace):
    from mcp.server.mcpserver import MCPServer
    kit,_ = load(case_id,network)
    server = MCPServer("archive")
    calls = []
    def invoke(name, **args):
        try:
            result = getattr(kit,name)(**args)
            ok = True
        except (ValueError,KeyError,TypeError) as exc:
            result,ok = {"error":str(exc)},False
        calls.append({"tool":name,"args":args,"ok":ok,"result":result})
        if trace:
            write(trace,{"case_id":case_id,"network":network,"calls":calls,"labels_accessible":False})
        return result

    @server.tool()
    def search_archive(query: str, scope: str = "own", limit: int = 8) -> dict:
        """Search beyond the original short list, with focused entity/event headline terms. Same historical cutoff; no web or labels."""
        return invoke("search",query=query,scope=scope,limit=limit)

    @server.tool()
    def inspect_archive(source_id: str, page: int = 0) -> dict:
        """Read an eligible raw Reuters page with hash verification. Further pages retain qualifications and transaction terms."""
        return invoke("inspect",source_id=source_id,page=page)

    try:
        server.run(transport="stdio")
    finally:
        kit.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case_id",nargs="?")
    p.add_argument("--network",action="store_true")
    p.add_argument("--trace")
    p.add_argument("--audit",action="store_true")
    args = p.parse_args()
    if args.audit:
        audit()
    elif args.case_id:
        serve(args.case_id,args.network,args.trace)
    else:
        p.error("Provide --audit or a case ID")
