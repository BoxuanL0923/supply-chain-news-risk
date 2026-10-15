"""Bounded, input-only Reuters retrieval for the mid-cap pilot.

Public interface: EvidenceStore(case).state, search_news(), inspect_event(),
get_connections(), fixed_packet(), headlines(), counts(), and audit_log(). ``related`` is
the canonical network channel; ``network`` is accepted as a search alias.
Cards carry source-body offsets and a *heuristic*, not gold, duplicate group.
The original body is retained intact in the shared corpus, never in a card.

Build with system Python: python code/midcap_tools/retrieval.py --build-index
This caches every passage, plus fixed-packet queries from cases.jsonl. Serving
imports no torch. Uncached queries use a local-only system-Python subprocess;
set MIDCAP_EMBED_PYTHON if the venv's base Python is not the torch installation.
Missing/incompatible caches or an unavailable encoder produce an explicit
warning and lexical-only backend status in both cards and the backend audit.
The main runtime must apply MAX_EVENTS/MAX_EVIDENCE_CHARS to accumulated cards.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date as Date, datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import warnings
import zlib

import numpy as np

if __package__:
    from .config import EMBED_MODEL, MAX_EVENTS, MAX_EVIDENCE_CHARS, OUT
else:
    from config import EMBED_MODEL, MAX_EVENTS, MAX_EVIDENCE_CHARS, OUT


INDEX_VERSION = 2
DIMENSION = 384
CHILD_CHARS = 1000
PARENT_CHARS = 1500
DUPLICATE_SECONDS = 6 * 60 * 60
DUPLICATE_NOTE = "time-bounded text duplicate heuristic, not true event gold"
CHANNEL_CAPS = {"common": 2, "own": 3, "related": 3}
WORD = re.compile(r"[a-z0-9]+", re.I)
SENTENCE_END = re.compile(r'''[.!?]+["'\u2019\u201d)\]]*(?=\s+|$)''')
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "inc", "corp", "co", "ltd",
                 "vs", "st", "no", "fig", "u.s", "u.k", "e.g", "i.e"}
STATE_FIELDS = {
    "case_id", "permno", "ticker", "company", "industry", "date", "asof",
    "q_idio", "q_total", "p0_idio", "p0_total", "baseline_idio", "baseline_total",
    "historical_mean_idio", "historical_mean_total",
}
_QUERY_LOCK = threading.RLock()
_QUERY_MEMORY: dict[tuple, np.ndarray] = {}
_QUERY_WORKER = None
_ARTICLES = {"a", "an", "the"}
_NUMERIC_FACT = re.compile(r"(?<!\w)[$\u00a3\u20ac+\-\u2212]?\d[\d,.]*(?:\s*%)?")


class RetrievalWarning(RuntimeWarning):
    """Dense retrieval is unavailable; the audit records the precise fallback."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _date(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Case/corpus date must be YYYY-MM-DD")
    Date.fromisoformat(value)
    return value


def _timestamp(value, *, date_allowed=False) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Source timestamp must be an ISO timestamp with timezone")
    if date_allowed and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return datetime.combine(Date.fromisoformat(value), datetime.min.time(), timezone.utc)
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid source timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Naive source timestamps are not allowed")
    return stamp.astimezone(timezone.utc)


def _permno(value) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[1-9]\d*", str(value)):
        raise ValueError("permno must be a positive integer")
    return int(value)


def _limit(value, maximum) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("limit must be a nonnegative integer")
    return min(value, maximum)


def _file_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _sentence_spans(body: str) -> tuple[tuple[int, int], ...]:
    """Keep whole prose sentences or bulletin paragraphs and original offsets."""
    spans, boundaries, start = [], set(), 0
    for match in SENTENCE_END.finditer(body):
        if match.group().startswith(".."):
            continue
        prefix = body[max(0, match.start() - 40):match.start()]
        token = re.search(r"([A-Za-z.]+)$", prefix)
        word = token.group(1).lower() if token else ""
        if match.group().startswith(".") and (
                word in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)*[a-z]", word)):
            continue
        boundaries.add(match.end())
    # Reuters BRIEF/BUZZ bullet blocks often have no sentence-ending punctuation.
    boundaries.update(match.start() for match in re.finditer(r"\n[ \t]*\n", body))
    for boundary in sorted(boundaries):
        end = boundary
        while end > start and body[end - 1].isspace():
            end -= 1
        while start < end and body[start].isspace():
            start += 1
        if start < end:
            spans.append((start, end))
        start = boundary
    while start < len(body) and body[start].isspace():
        start += 1
    end = len(body.rstrip())
    if start < end:
        spans.append((start, end))
    return tuple(spans)


def _chunk_spans(body: str) -> tuple[tuple[int, int], ...]:
    sentences = _sentence_spans(body)
    chunks = []
    start = end = None
    for left, right in sentences:
        if start is not None and right - start > CHILD_CHARS:
            chunks.append((start, end))
            start = None
        if start is None:
            start = left
        end = right
    if start is not None:
        chunks.append((start, end))
    return tuple(chunks) or ((0, 0),)


@dataclass(frozen=True)
class _Event:
    event_id: str
    ts: str
    stamp: datetime
    alt: str
    headline: str
    body: str
    ids: frozenset[int]
    kind: str
    family: str
    urgency: int


@dataclass(frozen=True)
class _Chunk:
    event: int
    start: int
    end: int


class _Corpus:
    def __init__(self, date, fingerprint, rows):
        self.date, self.fingerprint = date, fingerprint
        events, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each event must be an object")
            for field in ("date", "corpus_date"):
                if field in row and row[field] != date:
                    raise ValueError("Event belongs to a different case-date corpus")
            eid = row.get("event_id")
            if not isinstance(eid, str) or not eid or eid in seen:
                raise ValueError("Every corpus event needs a unique, nonempty event_id")
            seen.add(eid)
            if not isinstance(row.get("ids"), list):
                raise ValueError("Event ids must be a list of permnos")
            for field in ("ts", "headline", "body"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"Event {field} must be a string")
            events.append(_Event(eid, row["ts"], _timestamp(row["ts"]),
                                 str(row.get("alt") or ""), row["headline"], row["body"],
                                 frozenset(_permno(p) for p in row["ids"]),
                                 str(row.get("kind") or "").lower(),
                                 str(row.get("family") or ""), int(row.get("urgency") or 3)))
        self.events = tuple(events)
        self.duplicate_features = {}
        self.by_id = {event.event_id: i for i, event in enumerate(events)}
        self.chunks = tuple(_Chunk(i, start, end) for i, event in enumerate(events)
                            for start, end in _chunk_spans(event.body))
        self.event_chunks = defaultdict(list)
        postings = defaultdict(list)
        lengths = []
        for i, chunk in enumerate(self.chunks):
            self.event_chunks[chunk.event].append(i)
            counts = Counter(WORD.findall(self.chunk_text(i).lower()))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                postings[term].append((i, count))
        self.lengths = np.asarray(lengths, dtype=np.float32)
        self.postings = {term: (np.asarray([x[0] for x in values], dtype=np.int64),
                                np.asarray([x[1] for x in values], dtype=np.float32))
                         for term, values in postings.items()}
        self.layout_sha = _digest([(events[c.event].event_id, c.start, c.end) for c in self.chunks])

    def chunk_text(self, index):
        chunk = self.chunks[index]
        event = self.events[chunk.event]
        return event.headline + "\n" + event.body[chunk.start:chunk.end]

    def sparse(self, query, eligible):
        # DF/length normalization use only this target's pre-cutoff eligible passages.
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        n = int(eligible.sum())
        mean = float(self.lengths[eligible].mean()) if n else 1.0
        for term in set(WORD.findall(query.lower())):
            if term not in self.postings:
                continue
            indexes, counts = self.postings[term]
            keep = eligible[indexes]
            indexes, counts = indexes[keep], counts[keep]
            df = len(indexes)
            if not df:
                continue
            idf = math.log(1 + (n - df + .5) / (df + .5))
            norm = 1.2 * (.25 + .75 * self.lengths[indexes] / max(mean, 1))
            scores[indexes] += idf * counts * 2.2 / (counts + norm)
        return scores


@lru_cache(maxsize=4)
def _load_corpus(root: str, date: str, stamp: tuple) -> _Corpus:
    path = Path(root) / "events" / f"{date}.json.gz"
    with path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("Case-date corpus must contain a list of full events")
    if _file_stamp(path) != stamp:
        raise RuntimeError("Case-date corpus changed while loading; retry after the build")
    return _Corpus(date, fingerprint, rows)


def _manifest(corpus):
    return {"version": INDEX_VERSION, "model": EMBED_MODEL, "dimension": DIMENSION,
            "date": corpus.date, "corpus_sha256": corpus.fingerprint,
            "layout_sha256": corpus.layout_sha, "chunks": len(corpus.chunks)}


def _index_paths(root, corpus):
    key = _digest(_manifest(corpus))[:24]
    base = Path(root) / "indexes" / f"{corpus.date}-{key}"
    return base.with_suffix(".json"), base.with_suffix(".npy")


@lru_cache(maxsize=4)
def _read_vectors(path: str, stamp: tuple, count: int):
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (count, DIMENSION) or vectors.dtype != np.float32:
        raise ValueError("Embedding shape/dtype does not match corpus")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite corpus embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if count and not np.allclose(norms, 1, atol=.002):
        raise ValueError("Corpus embeddings must be normalized")
    if _file_stamp(Path(path)) != stamp:
        raise ValueError("Embedding index changed while loading")
    return vectors


def _dense_index(root, corpus):
    meta, data = _index_paths(root, corpus)
    if not meta.exists() or not data.exists():
        return None, "missing_index: run system Python retrieval.py --build-index"
    try:
        if json.loads(meta.read_text(encoding="utf-8")) != _manifest(corpus):
            raise ValueError("Index manifest does not match bound corpus")
        return _read_vectors(str(data), _file_stamp(data), len(corpus.chunks)), "hybrid"
    except (OSError, ValueError, EOFError) as exc:
        return None, f"invalid_index: {type(exc).__name__}: {exc}"


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=True, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_npy(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _query_path(root, query):
    return Path(root) / "indexes" / "queries" / _digest([EMBED_MODEL, INDEX_VERSION])[:16] / (
        _digest(query) + ".npy")


def _valid_query(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (DIMENSION,) or not np.isfinite(vector).all():
        raise ValueError("Invalid query embedding")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero query embedding")
    return vector / norm


class _QueryWorker:
    """One lazy local encoder process, serialized by _QUERY_LOCK and reused."""

    def __init__(self):
        executable = os.environ.get("MIDCAP_EMBED_PYTHON") or getattr(
            sys, "_base_executable", sys.executable)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}
        self.process = subprocess.Popen(
            [executable, "-B", str(Path(__file__).resolve()), "--query-server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.responses = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        atexit.register(self.close)

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def encode(self, query):
        if self.process.poll() is not None:
            raise RuntimeError("Local MiniLM worker exited; check MIDCAP_EMBED_PYTHON and CUDA")
        self.process.stdin.write(json.dumps({"query": query}) + "\n")
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=60)
        except queue.Empty as exc:
            raise RuntimeError("Local MiniLM worker timed out") from exc
        if response is None:
            raise RuntimeError("Local MiniLM worker exited before returning an embedding")
        value = json.loads(response)
        if "error" in value:
            raise RuntimeError(value["error"])
        return _valid_query(value["vector"])

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin:
            self.process.stdin.close()
        self.reader.join(timeout=5)
        if not self.reader.is_alive() and self.process.stdout:
            self.process.stdout.close()


def _query_vector(root, query):
    global _QUERY_WORKER
    path = _query_path(root, query)
    key = (str(path),)
    with _QUERY_LOCK:
        if key in _QUERY_MEMORY:
            return _QUERY_MEMORY[key], "memory"
        if path.exists():
            vector = _valid_query(np.load(path, allow_pickle=False))
            route = "disk"
        else:
            try:
                if _QUERY_WORKER is None:
                    _QUERY_WORKER = _QueryWorker()
                vector = _QUERY_WORKER.encode(query)
            except Exception:
                if _QUERY_WORKER is not None:
                    _QUERY_WORKER.close()
                    _QUERY_WORKER = None
                raise
            _atomic_npy(path, vector)
            route = "local_worker"
        if len(_QUERY_MEMORY) >= 2048:
            _QUERY_MEMORY.pop(next(iter(_QUERY_MEMORY)))
        _QUERY_MEMORY[key] = vector
        return vector, route


def _duplicate_tokens(body):
    # Discard only recognizable terminal wire credits, never new article sentences.
    text = re.split(r"(?i)(?:\n|(?<=\.)\s+)\(?\s*(?:reporting|editing) by\b", body)[0]
    return tuple(WORD.findall(text.lower()))


def _shingles(tokens):
    if len(tokens) < 5:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + 5]) for i in range(len(tokens) - 4)}


def _minhash(shingles):
    hashed = np.asarray([zlib.crc32(s.encode()) for s in shingles], dtype=np.uint64)
    # Fixed affine permutations make grouping deterministic across Python processes.
    multipliers = np.arange(1, 65, 2, dtype=np.uint64)
    offsets = np.arange(32, dtype=np.uint64) * np.uint64(2654435761)
    signature = np.full(32, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(hashed), 2048):
        block = (hashed[start:start + 2048, None] * multipliers + offsets) % np.uint64(4294967311)
        signature = np.minimum(signature, block.min(axis=0))
    return tuple(int(x) for x in signature)


def _duplicate_groups(corpus, eligible):
    groups, headlines, bands, features = {}, defaultdict(list), defaultdict(list), {}
    for index in sorted(eligible, key=lambda i: (corpus.events[i].stamp, corpus.events[i].event_id)):
        event = corpus.events[index]
        if index not in corpus.duplicate_features:
            tokens = tuple(t for t in _duplicate_tokens(event.body) if t not in _ARTICLES)
            shingles = _shingles(tokens)
            signature = _minhash(shingles)
            title = " ".join(WORD.findall(event.headline.lower()))
            numbers = tuple(x.rstrip(".,").replace(" ", "") for x in
                            _NUMERIC_FACT.findall(event.headline + "\n" + event.body))
            corpus.duplicate_features[index] = tokens, shingles, signature, title, numbers
        tokens, shingles, signature, title, numbers = corpus.duplicate_features[index]
        scope = (tuple(sorted(event.ids)), event.kind, event.family)
        keys = [(scope, band, signature[band * 4:band * 4 + 4]) for band in range(8)]
        candidates = set(headlines[(scope, title)])
        for key in keys:
            candidates.update(bands[key])
        representative = index
        for other in sorted(candidates):
            previous = corpus.events[other]
            if (event.stamp - previous.stamp).total_seconds() > DUPLICATE_SECONDS:
                continue
            old_tokens, old_shingles, old_title, old_numbers = features[other]
            if not tokens or not old_tokens:
                continue
            similarity = len(shingles & old_shingles) / max(1, len(shingles | old_shingles))
            if tokens != old_tokens and similarity < .92:
                continue
            # Near-verbatim can still change a key figure or negate an announcement.
            # Only whitespace, punctuation, credits and articles may differ in bodies.
            if tokens != old_tokens or numbers != old_numbers:
                continue
            if title != old_title:
                title_tokens, old_title_tokens = title.split(), old_title.split()
                if tuple(t for t in title_tokens if t not in _ARTICLES) != tuple(
                        t for t in old_title_tokens if t not in _ARTICLES):
                    continue
            representative = other
            break
        groups[index] = representative
        if representative == index:
            headlines[(scope, title)].append(index)
            for key in keys:
                bands[key].append(index)
            features[index] = (tokens, shingles, title, numbers)
    return groups


def _fixed_query(state):
    return (f"{state['ticker']} {state['company']} {state.get('industry', '')} "
            "earnings guidance demand supply financing liquidity uncertainty risk "
            "US market interest rates credit inflation policy")


def evidence_chars(cards) -> int:
    """Conservative shared-runtime budget: full JSON cards, including metadata."""
    return len(json.dumps(cards, ensure_ascii=True))


class EvidenceStore:
    """Bind one target and cutoff; tool calls cannot supply either binding."""

    def __init__(self, case: dict):
        if not isinstance(case, dict):
            raise TypeError("case must be a dict")
        self._date = _date(case["date"])
        self._permno = _permno(case["permno"])
        self._cutoff = _timestamp(case["asof"])
        if self._cutoff.date().isoformat() != self._date:
            raise ValueError("Case asof must fall on its case date in UTC")
        for field in ("ticker", "company"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Case {field} is required")
        self._state = deepcopy({k: v for k, v in case.items() if k in STATE_FIELDS})
        self._state.update(permno=self._permno, date=self._date, asof=self._cutoff.isoformat())
        self._root = Path(OUT).resolve()
        path = self._root / "events" / f"{self._date}.json.gz"
        self._corpus = _load_corpus(str(self._root), self._date, _file_stamp(path))
        self._audit, self._focus, self._warned = [], {}, set()
        self._connections = self._load_connections()
        neighbors = {row["permno"] for row in self._connections}
        self._channels = {}
        rejected_time = 0
        for i, event in enumerate(self._corpus.events):
            if event.stamp >= self._cutoff:
                rejected_time += 1
                continue
            if self._permno in event.ids:
                self._channels[i] = "own"
            elif neighbors & event.ids:
                self._channels[i] = "related"
            elif event.kind in {"macro", "finance"}:
                self._channels[i] = "common"
        self._groups = _duplicate_groups(self._corpus, self._channels)
        mentions = Counter(p for i in set(self._groups.values())
                           for p in self._corpus.events[i].ids if p in neighbors)
        self._connections = tuple(sorted(
            ({**row, "eligible_event_count": mentions[row["permno"]]} for row in self._connections),
            key=lambda row: (not bool(row["eligible_event_count"]),
                             row["role"].lower() not in {"customer", "supplier"},
                             -row["eligible_event_count"], -(row["rev_pct"] or 0),
                             row["permno"], row["role"], row["asof"])))
        self._vectors, self._backend = _dense_index(self._root, self._corpus)
        if self._vectors is None and self._channels:
            self._warn(self._backend)
        self._record("bind", [], eligible_events=len(self._channels),
                     rejected_at_or_after_cutoff=rejected_time,
                     rejected_relationships=self._rejected_relationships,
                     duplicate_method=DUPLICATE_NOTE)

    @property
    def state(self) -> dict:
        return deepcopy(self._state)

    def _warn(self, reason):
        if reason not in self._warned:
            warnings.warn("Midcap lexical-only retrieval: " + reason, RetrievalWarning, stacklevel=3)
            self._warned.add(reason)

    def _load_connections(self):
        path = self._root / "relations" / f"{self._date}.json"
        with path.open(encoding="utf-8") as stream:
            mappings = json.load(stream)
        if not isinstance(mappings, dict):
            raise ValueError("Relations must map focal permno to metadata lists")
        rows = mappings.get(str(self._permno), [])
        if not isinstance(rows, list):
            raise ValueError("Focal relations must be a list")
        connections, seen = [], set()
        self._rejected_relationships = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Relation must be an object")
            for field in ("date", "corpus_date"):
                if field in row and row[field] != self._date:
                    raise ValueError("Relation belongs to a different case-date corpus")
            if _timestamp(row["asof"], date_allowed=True) >= self._cutoff:
                self._rejected_relationships += 1
                continue
            item = {field: deepcopy(row.get(field)) for field in
                    ("permno", "ticker", "company", "role", "source", "rev_pct", "asof")}
            item["permno"] = _permno(item["permno"])
            if item["permno"] == self._permno:
                continue
            for field in ("ticker", "company"):
                if not isinstance(item[field], str):
                    raise ValueError(f"Relation {field} must be a string (possibly empty)")
            for field in ("role", "source"):
                if not isinstance(item[field], str) or not item[field].strip():
                    raise ValueError(f"Relation {field} must be a nonempty string")
            if item["rev_pct"] is not None and (
                    isinstance(item["rev_pct"], bool) or not isinstance(item["rev_pct"], (int, float))
                    or not math.isfinite(item["rev_pct"])):
                raise ValueError("Relation rev_pct must be a finite number or null")
            key = _digest(item)
            if key not in seen:
                connections.append(item)
                seen.add(key)
        return tuple(sorted(connections, key=lambda r: (r["permno"], r["role"], r["asof"])))

    def _record(self, action, cards, **details):
        # Only backend-generated input metadata enters this log; no arbitrary case fields.
        self._audit.append({"action": action, "permno": self._permno, "date": self._date,
                            "asof": self._cutoff.isoformat(), "corpus_sha256": self._corpus.fingerprint,
                            "backend": self._backend, "event_ids": [c["event_id"] for c in cards],
                            "sources": [{k: c[k] for k in ("event_id", "ts", "channel",
                                         "duplicate_group", "span") if k in c} for c in cards],
                            **details})

    def audit_log(self) -> list[dict]:
        """Return a copy of input-only backend records for the runtime to persist."""
        return deepcopy(self._audit)

    def _card(self, index, chunk_index, *, parent=False, backend=None):
        event = self._corpus.events[index]
        if index not in self._channels or event.stamp >= self._cutoff:
            raise ValueError("Event is not eligible in this bound case-date corpus")
        chunk = self._corpus.chunks[chunk_index]
        if chunk.event != index:
            raise ValueError("Passage is not owned by the requested source")
        start, end = chunk.start, chunk.end
        if parent:
            sentences = _sentence_spans(event.body)
            left = next((i for i, span in enumerate(sentences) if span[0] == start), 0)
            right = next((i for i, span in enumerate(sentences) if span[1] == end), left)
            while True:
                if left > 0 and end - sentences[left - 1][0] <= PARENT_CHARS:
                    left -= 1
                    start = sentences[left][0]
                elif right + 1 < len(sentences) and sentences[right + 1][1] - start <= PARENT_CHARS:
                    right += 1
                    end = sentences[right][1]
                else:
                    break
        channel = self._channels[index]
        links = [row for row in self._connections if row["permno"] in event.ids]
        if channel == "own":
            relation = "Target mentioned: " + self._state["ticker"]
        elif channel == "common":
            relation = "US macro/finance; source-filtered common context"
        else:
            relation = "; ".join(
                f"{r['ticker'] or ('permno ' + str(r['permno']))}: {r['role']} relative to {self._state['ticker']}; "
                f"source={r['source']}; rev_pct(supplier revenue)={r['rev_pct']}; asof={r['asof']}" for r in links)
        representative = self._corpus.events[self._groups[index]].event_id
        return {"event_id": event.event_id, "source_id": event.alt or event.event_id,
                "ts": event.ts, "headline": event.headline, "channel": channel,
                "relationship": relation, "passage": event.body[start:end] or event.headline,
                "span": [start, end], "parent_id": event.event_id, "corpus_date": self._date,
                "duplicate_group": _digest([self._date, representative])[:20],
                "duplicate_heuristic": True, "retrieval_backend": backend or self._backend}

    def search_news(self, query: str, channel: str = "all", limit: int = 3) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1-2000 characters")
        channel = "related" if channel == "network" else channel
        if channel not in {"all", "common", "own", "related"}:
            raise ValueError("channel must be all, own, related/network, or common")
        limit = _limit(limit, MAX_EVENTS)
        eligible = np.asarray([c.event in self._channels and (
            channel == "all" or self._channels[c.event] == channel) for c in self._corpus.chunks], dtype=bool)
        indexes = np.flatnonzero(eligible)
        cards, query_route, backend = [], "unused", self._backend
        if limit and len(indexes):
            sparse = self._corpus.sparse(query, eligible)
            lexical = indexes[sparse[indexes] > 0]
            lexical = lexical[np.argsort(-sparse[lexical], kind="stable")]
            fused = np.zeros(len(self._corpus.chunks), dtype=np.float64)
            fused[lexical] = 1 / (60 + np.arange(1, len(lexical) + 1))
            if self._vectors is not None:
                try:
                    vector, query_route = _query_vector(self._root, query)
                    dense_scores = self._vectors[indexes] @ vector
                    dense_order = indexes[np.argsort(-dense_scores, kind="stable")]
                    fused[dense_order] += 1 / (60 + np.arange(1, len(dense_order) + 1))
                    backend = "hybrid"
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                    backend = "query_unavailable: " + str(exc)
                    self._warn(backend)
                    query_route = "failed"
            seen = set()
            for chunk_index in indexes[np.argsort(-fused[indexes], kind="stable")]:
                if fused[chunk_index] <= 0:
                    break
                event_index = self._corpus.chunks[chunk_index].event
                group = self._groups[event_index]
                if group in seen:
                    continue
                seen.add(group)
                # Stable IDs let the runtime deduplicate cards across different queries.
                chunk_index = max(self._corpus.event_chunks[group], key=lambda i: fused[i])
                self._focus[group] = int(chunk_index)
                cards.append(self._card(group, chunk_index, backend=backend))
                if len(cards) == limit:
                    break
        self._record("search_news", cards, query=query, channel=channel, limit=limit,
                     query_vector=query_route, backend=backend, eligible_passages=int(len(indexes)))
        return cards

    def inspect_event(self, event_id: str) -> dict:
        if not isinstance(event_id, str):
            raise ValueError("event_id must be a string")
        index = self._corpus.by_id.get(event_id)
        if index not in self._channels:
            self._record("inspect_event_rejected", [], requested_event_id=event_id)
            raise ValueError("Event is not eligible in this bound case-date corpus")
        chunk_index = self._focus.get(index, self._corpus.event_chunks[index][0])
        card = self._card(index, chunk_index, parent=True)
        self._record("inspect_event", [card])
        return card

    def get_connections(self, limit: int = 10) -> list[dict]:
        limit = _limit(limit, 10)
        result = deepcopy(list(self._connections[:limit]))
        self._record("get_connections", [], limit=limit, connections=result)
        return result

    def counts(self) -> dict[str, int]:
        """Input-only event counts after the same conservative duplicate heuristic."""
        counts = {"common": 0, "own": 0, "network": 0}
        for index in set(self._groups.values()):
            channel = self._channels[index]
            counts["network" if channel == "related" else channel] += 1
        self._record("counts", [], counts=counts.copy())
        return counts

    def headlines(self) -> list[dict]:
        pools = {channel: [] for channel in CHANNEL_CAPS}
        for index in sorted(set(self._groups.values()), key=lambda i: (self._corpus.events[i].stamp,
                                                                      self._corpus.events[i].event_id), reverse=True):
            event = self._corpus.events[index]
            channel = self._channels[index]
            pools[channel].append({"event_id": event.event_id, "headline": event.headline,
                                   "channel": channel})
        # The parent shows only the first twenty: reserve early positions for firm news.
        result, positions = [], Counter()
        while any(positions[c] < len(pool) for c, pool in pools.items()):
            for channel in ("own", "related", "own", "related", "common"):
                position = positions[channel]
                if position < len(pools[channel]):
                    result.append(pools[channel][position])
                    positions[channel] += 1
        self._record("headlines", result)
        return result

    def fixed_packet(self) -> list[dict]:
        query = _fixed_query(self._state)
        pools = {channel: self.search_news(query, channel, cap) for channel, cap in CHANNEL_CAPS.items()}
        cards, seen = [], set()
        for rank in range(3):
            for channel in ("own", "related", "common"):
                if rank >= len(pools[channel]):
                    continue
                card = pools[channel][rank]
                if card["duplicate_group"] in seen:
                    continue
                if len(cards) < MAX_EVENTS and evidence_chars(cards + [card]) <= MAX_EVIDENCE_CHARS:
                    cards.append(card)
                    seen.add(card["duplicate_group"])
        self._record("fixed_packet", cards, evidence_chars=evidence_chars(cards),
                     caps=CHANNEL_CAPS.copy())
        return cards


def _embedder(device="cuda"):
    # This is the sole torch-dependent path, used only by the offline CLI/worker.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, local_files_only=True, device=device)


def build_all(*, device: str = "cuda") -> list[dict]:
    """Build date indexes and fixed-query caches using only input cases/corpora."""
    root = Path(OUT).resolve()
    paths = sorted((root / "events").glob("*.json.gz"))
    if not paths:
        raise FileNotFoundError("No date corpora in " + str(root / "events"))
    model, built = None, []
    for path in paths:
        date = _date(path.name.removesuffix(".json.gz"))
        corpus = _load_corpus(str(root), date, _file_stamp(path))
        vectors, _ = _dense_index(root, corpus)
        if vectors is None:
            if corpus.chunks:
                if model is None:
                    model = _embedder(device)
                vectors = model.encode([corpus.chunk_text(i) for i in range(len(corpus.chunks))],
                                       batch_size=128, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=True)
            else:
                vectors = np.empty((0, DIMENSION), dtype=np.float32)
            meta, data = _index_paths(root, corpus)
            _atomic_npy(data, vectors)
            _atomic_json(meta, _manifest(corpus))
        built.append({"date": date, "events": len(corpus.events), "passages": len(corpus.chunks)})
        print(json.dumps(built[-1]), flush=True)
    cases = root / "cases.jsonl"
    if cases.exists():
        queries = set()
        with cases.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    queries.add(_fixed_query(json.loads(line)))
        queries = sorted(query for query in queries if not _query_path(root, query).exists())
        if queries:
            if model is None:
                model = _embedder(device)
            vectors = model.encode(queries, batch_size=64, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=True)
            for query, vector in zip(queries, vectors):
                _atomic_npy(_query_path(root, query), _valid_query(vector))
    return built


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-index", action="store_true")
    action.add_argument("--encode-query", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--query-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.build_index:
        build_all(device=args.device)
    elif args.query_server:
        from contextlib import redirect_stdout
        model = None
        for line in sys.stdin:
            try:
                query = json.loads(line)["query"]
                if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                    raise ValueError("Invalid query")
                with redirect_stdout(sys.stderr):
                    if model is None:
                        model = _embedder(args.device)
                    vector = model.encode([query], normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False)[0]
                reply = {"vector": _valid_query(vector).tolist()}
            except Exception as exc:
                reply = {"error": "Local MiniLM encoder failed (" + type(exc).__name__ +
                         "); check MIDCAP_EMBED_PYTHON, CUDA and the local model cache"}
            print(json.dumps(reply), flush=True)
    else:
        from contextlib import redirect_stdout
        request = json.load(sys.stdin)
        query = request["query"]
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Invalid query")
        with redirect_stdout(sys.stderr):
            vector = _embedder(args.device).encode([query], normalize_embeddings=True,
                                                   convert_to_numpy=True, show_progress_bar=False)[0]
        print(json.dumps({"vector": _valid_query(vector).tolist()}))


if __name__ == "__main__":
    _main()
