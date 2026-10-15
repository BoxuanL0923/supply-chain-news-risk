"""Model-requested tools, strictly bound to one target and historical cutoff."""
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import json
import re
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd

from news_agent_batched_runtime_2025 import read, write, digest
from news_source_clock_2025 import publication_clock
from midcap_tools.retrieval import _query_vector, _Corpus, WORD
from .config import OUT, FIRMS
from .data import relevant_common
from .graph import Graph

SCHEMAS = {
    "get_market_state": "{}; exact dated own-firm prices/returns, computed risks and controls",
    "search_events": "{query:string, channel:own|common|all, days_back:1..31 (default 31), limit:1..6, offset:integer}; discovery snippets; read_evidence required before citing",
    "read_evidence": "{source_id:string, chunk:integer}; a complete parent passage and source clock",
    "get_event_state": "{query:string, limit:1..6}; historical own-event evidence plus earnings calendar, not outcome memory",
    "lookup_exposure": "{hops:1|2|3, query:optional entity-name substring, via:entity_id|null, offset:integer, limit:1..20}; typed dated paths; third hop requires an inspected intermediate",
    "search_related_events": "{query:string, hops:1|2 (default 1), days_back:1..31 (default 31), limit:1..6, offset:integer}; top 20 graph/attention candidates, with typed paths and passages",
}
_CORPUS_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def corpus(root, day):
    from pathlib import Path
    out = Path(root)
    with gzip.open(out / "corpus" / (day + ".json.gz"), "rt", encoding="utf-8") as f:
        rows = json.load(f)
    all_chunks, all_keys, all_vectors = embeddings(root)
    ids = {r["source_id"] for r in rows}
    indices = [i for i, k in enumerate(all_keys) if k.rsplit("#", 1)[0] in ids]
    by_id, keys = {r["source_id"]: r for r in rows}, [all_keys[i] for i in indices]
    postings, lengths = defaultdict(list), []
    for i, key in enumerate(keys):
        sid, chunk = key.rsplit("#", 1)
        a, b = all_chunks[sid][int(chunk)]
        row = by_id[sid]
        counts = Counter(WORD.findall((row["headline"]+" "+row["headline"]+" "+row["body"][a:b]).lower()))
        lengths.append(sum(counts.values()))
        for term, count in counts.items():
            postings[term].append((i, count))
    sparse = SimpleNamespace(chunks=keys, lengths=np.asarray(lengths, dtype=np.float32),
        postings={term: (np.asarray([v[0] for v in values], dtype=np.int64), np.asarray([v[1] for v in values], dtype=np.float32))
                  for term, values in postings.items()})
    return by_id, {s: all_chunks[s] for s in ids}, keys, all_vectors[indices].astype(np.float32), sparse


@lru_cache(maxsize=1)
def embeddings(root):
    from pathlib import Path
    out = Path(root)
    chunks = read(out / "corpus/chunks.json")
    with np.load(out / "corpus/embeddings.npz", allow_pickle=False) as z:
        return chunks, z["keys"].tolist(), z["vectors"]


def count_arg(value, default=4, maximum=6):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError("Integer tool limit outside declared range")
    return value


class ToolSession:
    def __init__(self, case, calendar, graph, root=OUT, network=False):
        self.case, self.calendar, self.graph, self.root, self.network = case, calendar, graph, root, network
        self.calendar = {k: calendar.get(k) for k in ("status", "reason", "earnings_in_window", "coverage_note")}
        self.calendar["status"] = calendar.get("status") or "unknown"
        self.calendar["interpretation"] = ("Schedule coverage is UNKNOWN. Do not assert that no earnings are scheduled or that the calendar is clean."
                                           if self.calendar["status"] == "unknown" else "Use only the dated scheduled events supplied here.")
        self.calendar["events"] = [{k: e.get(k) for k in ("event_date", "time_text", "action", "date_status", "in_window", "session_index", "available_at", "source_text", "receipt_ids")}
                                   for e in calendar.get("events", []) if e.get("reason") != "planned_time_passed"]
        with _CORPUS_LOCK:
            self.rows, self.chunks, self.keys, self.vectors, self.sparse_index = corpus(str(root), case["date"])
        self.read_ids, self.seen_entities, self.trace = set(), set(), []
        self.paths = graph.paths(case["ticker"], 2) if network else []
        self.relations = {}
        for path in self.paths:
            for p in path["permnos"]:
                choices = self.relations.setdefault(p, [])
                pattern = tuple(s["neighbor_role"] for s in path["steps"])
                if not any(tuple(s["neighbor_role"] for s in old["steps"]) == pattern for old in choices):
                    choices.append(path)
        if network:
            for row in graph.data.get("comention", {}).get(str(case["permno"]), []):
                self.relations.setdefault(row["permno"], [{"type": "comention", "permnos": [row["permno"]],
                    "hops": 1, "weight": row["weight"], "path": [], "steps": [], "mechanism_verified": False,
                    "vintage": "2025-07-31", "degree": 1}])
        self.allowed = {sid: self.channel(row) for sid, row in self.rows.items()}
        self.allowed = {sid: kind for sid, kind in self.allowed.items() if kind is not None}

    def channel(self, row):
        if not row["usable"] or pd.Timestamp(row["available_at"]) >= pd.Timestamp(self.case["asof"]):
            return None
        if self.case["permno"] in row["ids"]:
            return "own"
        if relevant_common(row):
            return "common"
        if self.network and any(p in self.relations for p in row["ids"]):
            return "network"
        return None

    def paths_for(self, row):
        paths = [path for p in row["ids"] for path in self.relations.get(p, [])]
        return sorted(paths, key=lambda p: (p["hops"], p.get("type") == "comention", p["degree"]))

    def market(self):
        result = {k: self.case[k] for k in ("case_id", "ticker", "company", "industry", "date", "asof", "next_sessions",
                                           "anchor", "b", "state", "stock_history")}
        comparisons = {}
        for n in (1, 5, 22):
            rv = self.case["state"].get(f"log_ret_rv{n}")
            if rv is not None:
                comparisons[f"rv{n}_versus_garch"] = {"log_difference": rv-self.case["b"],
                    "variance_ratio": float(np.exp(rv-self.case["b"])),
                    "ordering": "lower" if rv < self.case["b"] else "higher" if rv > self.case["b"] else "equal"}
        result["host_computed_comparisons"] = comparisons
        result["risk_measure_note"] = "Squared close-to-close returns, not intraday realized variance; a past price reaction does not establish complete resolution of future uncertainty."
        return result

    def inspect(self, sid, chunk=0, max_hops=None, discovery=False):
        if sid not in self.allowed:
            raise ValueError("Source outside this arm, entity set or as-of cutoff")
        row = self.rows[sid]
        spans = self.chunks[sid]
        if isinstance(chunk, bool) or not isinstance(chunk, int) or not 0 <= chunk < len(spans):
            raise ValueError("Invalid source chunk")
        # One preceding and following sentence chunk provides qualifications/updates.
        a, b = spans[chunk] if discovery else (spans[max(0, chunk-1)][0], spans[min(len(spans)-1, chunk+1)][1])
        if not discovery:
            self.read_ids.add(sid)
        paths = self.paths_for(row) if self.allowed[sid] == "network" else []
        if max_hops is not None:
            paths = [p for p in paths if p["hops"] <= max_hops]
        for p in paths:
            self.seen_entities.update(p.get("path", []))
        return {"source_id": sid, "story_id": row["story_id"], "event_key": row["event_key"],
                "channel": self.allowed[sid], "headline": row["headline"], "available_at": row["available_at"],
                "family": row.get("family", "unknown"),
                "clock": publication_clock(row["available_at"], self.case["asof"], self.case["next_sessions"]),
                "chunk": chunk, "chunk_count": len(spans), "span": [a, b], "text": row["body"][a:b] or row["headline"],
                "text_origin": "body_passage" if row["body"] else "headline_only_alert",
                "citation_eligible": not discovery,
                "context_rule": "Discovery only: use read_evidence on this source/chunk for full surrounding context before citing" if discovery else "Full parent passage inspected",
                "paths": paths[:6], "available_path_count": len(paths),
                "provenance": {k: row[k] for k in ("source_month", "source_line", "raw_sha256")}}

    def search(self, query, channel="all", limit=4, offset=0, hops=1, historical=False, days_back=31):
        if not isinstance(query, str) or len(query) > 1200:
            raise ValueError("Invalid search query")
        if isinstance(hops, bool) or hops not in (1, 2):
            raise ValueError("Related-news search supports only one or two hops")
        if isinstance(days_back, bool) or not isinstance(days_back, int) or not 1 <= days_back <= 31:
            raise ValueError("News lookback must be between one and 31 calendar days")
        limit, offset = count_arg(limit), count_arg(offset, 0, 10000)
        if channel not in ("own", "common", "all", "network") or (channel == "network" and not self.network):
            raise ValueError("Unavailable channel")
        ids = {s for s, c in self.allowed.items() if (c in ("own", "common") if channel == "all" else c == channel)}
        if not historical:
            ids = {s for s in ids if (pd.Timestamp(self.case["asof"])-pd.Timestamp(self.rows[s]["available_at"])).total_seconds() <= days_back*86400}
        if channel == "network":
            ids = {s for s in ids if any(p["hops"] <= hops for p in self.paths_for(self.rows[s]))}
        if not ids:
            return {"results": [], "candidate_count": 0, "backend": "hybrid", "coverage": "no eligible indexed events; not proof of no risk"}
        vector, route = _query_vector(self.root, query or "material financial events uncertainty")
        mask = np.asarray([k.rsplit("#", 1)[0] in ids for k in self.keys])
        # Reuse the existing BM25 implementation, including arm-eligible DF and
        # length normalization. Rank fusion avoids arbitrary dense/sparse scales.
        lexical = _Corpus.sparse(self.sparse_index, query, mask)
        dense = self.vectors @ vector
        scores, best_dense, best_lexical = {}, {}, {}
        lexical_scale = max(float(lexical.max()), 1e-8)
        for j, key in enumerate(self.keys):
            sid, n = key.rsplit("#", 1)
            if sid not in ids:
                continue
            best_dense[sid] = max(best_dense.get(sid, -1.), float(dense[j]))
            best_lexical[sid] = max(best_lexical.get(sid, 0.), float(lexical[j]))
            passage_score = .65*float(dense[j])+.35*float(lexical[j])/lexical_scale
            if sid not in scores or passage_score > scores[sid][0]:
                scores[sid] = (passage_score, int(n))
        dense_rank = {s: i+1 for i, s in enumerate(sorted(ids, key=lambda s: (-best_dense[s], s)))}
        lexical_rank = {s: i+1 for i, s in enumerate(sorted(ids, key=lambda s: (-best_lexical[s], s)))}
        scored = {}
        for sid in ids:
            row = self.rows[sid]
            score = 30.5*(1/(60+dense_rank[sid])+1/(60+lexical_rank[sid]))
            if not historical:
                age = (pd.Timestamp(self.case["asof"])-pd.Timestamp(row["available_at"])).total_seconds()/86400
                score += .05*np.exp(-age/7)
            if channel == "network":
                path = min((p for p in self.paths_for(row) if p["hops"] <= hops), key=lambda p: (p.get("type") == "comention", p["hops"], p["degree"]))
                score /= 1+.1*(path["hops"]-1)+.025*np.log1p(path["degree"])
                if path.get("type") == "comention":
                    score *= .75
            scored[sid] = (float(score), scores[sid][1])
        ranked = sorted(scored, key=lambda s: (-scored[s][0], s))
        unique, seen_passages = [], set()
        for sid in ranked:
            chunk = scored[sid][1]
            spans = self.chunks[sid]
            a, b = spans[max(0, chunk-1)][0], spans[min(len(spans)-1, chunk+1)][1]
            passage = self.rows[sid]["body"][a:b] or self.rows[sid]["headline"]
            fingerprint = digest(" ".join(passage.split()))
            if fingerprint not in seen_passages:
                seen_passages.add(fingerprint)
                unique.append(sid)
        ranked = unique
        if channel == "network":
            ranked = ranked[:20]
        results = [{**self.inspect(s, scored[s][1], max_hops=hops if channel == "network" else None, discovery=True), "retrieval_score": scored[s][0]} for s in ranked[offset:offset+limit]]
        return {"results": results, "candidate_count": len(scored), "eligible_after_cap": len(ranked),
                "offset": offset, "next_offset": offset+len(results) if offset+len(results) < len(ranked) else None,
                "identical_passage_duplicates_removed": len(scored)-len(unique),
                "backend": "MiniLM + BM25 reciprocal-rank fusion", "query_embedding": "local-MiniLM", "network_candidate_cap": 20 if channel == "network" else None}

    def event_state(self, query="", limit=4):
        found = self.search(query or "acquisition agreement pending regulatory approval earnings outage recovery disruption financing", "own", limit, historical=True)
        recent = sorted((s for s, c in self.allowed.items() if c == "own"), key=lambda s: self.rows[s]["available_at"], reverse=True)[:8]
        return {"calendar": self.calendar, "historical_evidence": found,
                "recent_update_headlines": [{"source_id": s, "headline": self.rows[s]["headline"], "available_at": self.rows[s]["available_at"]} for s in recent],
                "state_rule": "Source-backed history since July. Status/persistence must be checked against available updates; no previous predictions or outcomes stored."}

    def exposures(self, hops=1, via=None, offset=0, limit=10, query=""):
        if not self.network:
            raise ValueError("Network disabled for own arm")
        if hops == 3 and via not in self.seen_entities:
            raise ValueError("Inspect the intermediate before requesting three hops")
        paths = self.graph.paths(self.case["ticker"], hops, via=via)
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Invalid entity query")
        if query:
            terms = re.findall(r"[a-z0-9]+", query.lower())
            paths = [p for p in paths if all(term in (str(p.get("name") or "")+" "+p["entity"]).lower() for term in terms)]
        offset, limit = count_arg(offset, 0, 100000), count_arg(limit, 10, 20)
        chosen = paths[offset:offset+limit]
        self.seen_entities.update(n for p in chosen for n in p["path"])
        return {"paths": chosen, "path_count": len(paths), "next_offset": offset+len(chosen) if offset+len(chosen) < len(paths) else None,
                "point_in_time_certified": False, "snapshot_assumption": self.graph.data["audit"]["assumption"],
                "unmapped_entities": "Entities without reliable news identifiers remain in paths; missing news is not no exposure."}

    def call(self, name, args):
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        if name == "get_market_state":
            if args:
                raise ValueError("Market state is bound to target")
            result = self.market()
        elif name == "search_events":
            result = self.search(**args)
        elif name == "read_evidence":
            result = self.inspect(args["source_id"], args.get("chunk", 0))
        elif name == "get_event_state":
            result = self.event_state(**args)
        elif name == "lookup_exposure":
            result = self.exposures(**args)
        elif name == "search_related_events":
            result = self.search(channel="network", **args)
        else:
            raise ValueError("Unknown local tool")
        self.trace.append({"tool": name, "args": args, "result": result})
        return result

    def intro(self):
        own = sorted((s for s, c in self.allowed.items() if c == "own" and "ORDER IMBALANCE" not in self.rows[s]["headline"].upper()), key=lambda s: self.rows[s]["available_at"], reverse=True)
        common = sorted((s for s, c in self.allowed.items() if c == "common"), key=lambda s: self.rows[s]["available_at"], reverse=True)
        return {"target_anchor": self.case["anchor"], "market": self.market(), "calendar": self.calendar,
                "headlines": [{"source_id": s, "headline": self.rows[s]["headline"], "channel": self.allowed[s],
                               "available_at": self.rows[s]["available_at"]} for s in own[:8]+common[:6]],
                "headline_note": "Discovery only, not citeable until inspected. Search covers all eligible indexed events.",
                "tools": {k: v for k, v in SCHEMAS.items() if self.network or k not in ("lookup_exposure", "search_related_events")}}
