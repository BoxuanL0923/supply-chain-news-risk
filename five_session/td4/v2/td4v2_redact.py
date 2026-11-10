"""TD4 v2 span-level redaction of the masker-v2 inputs (researcher decision 2026-11-07: option 2 of
WITHHOLDING_DECISION_NOTE_v1; replaces the whole-input detector-flag rule as the withholding rule; masker v2 and its
builds are unchanged).

Pre-registered rule (PREREGISTRATION_v2 'redaction'), applied before any provider call and before any outcome is read,
identically for every arm and reference selector (one text per firm-week is sent to every provider; a withheld row
takes each arm's fallback):

Transform. A separate, deterministic post-processing step on the masker-v2 INPUTS. Every string matched by an included
detector class (the classes of td4v2_flags; same exclusions, same article-part scope) is replaced, in article text only
(the headline and text of each PARTNER NEWS article), by a placeholder:
  [REDACTED]  name residues: names, name parts, tickers, RIC parts, initialisms, codes, glued compounds, story IDs,
              URL residues and mis-tag notice wording;
  [DATE]      date, year and story-time residues;
  [TERM]      give-back slots: the one role-token occurrence that a give-back detector names (the token stands for an
              ordinary or index word the masker hid), replaced as a whole token.
Never touched: every other pseudonym token, the frozen header (instructions, risk menu, characteristics, graph), the
article labels and the field labels. A located span is trimmed to article text and never covers a bracket token; a
[TERM] span is exactly one role token inside article text.
Location. Each detector's own output: the matched string and its context window are re-found by rebuilding the
detector's context string at every candidate position (exact match); for the classes whose output holds keys only
(s41_targeted, s41b_nasdaq, ma2_s12_dates_v2) and for the date detectors of s11, s14, s15 and s16, the detector's own
regular expressions (read from the run copy) are re-run on the input and every match in article text is replaced;
GB give-back groups are re-scanned at every occurrence (frozen groups, below); MA2 single-word give-backs are located by
re-running MA2's own raw-versus-masked word alignment. A flagged class of an input that yields no located span (no text
span at all, as the M4 firm-level classes NONAME and NONAME_AFTER_2022, or a match outside article text) withholds the
input. The reviewers' adjudicated 2021-2022 key sets are withheld.
Rounds. Round r = 1, 2, 3 redacts the matches of the detector run on the round r-1 inputs (round 0 = the masker-v2
inputs and their detector-flag run) and then re-runs every detector on the round r inputs (td4v2_flags copies with only
the INPUTS path changed; the population, metadata and article store are those of the masker-v2 build). An input still
flagged after the round-3 re-run is withheld.
Still flagged: the re-run's outputs, extracted by td4v2_flags.extract (same classes and exclusions), contain the key.
Population-level detectors are frozen across rounds: an s18 GB group (permno, word, side, neighbour) kept in any round
so far stays a group in every later round, and its slots are re-scanned in the round's text (an input with a remaining
slot of a frozen group is still flagged), so a group that falls below s18's ratio after redaction cannot release its
remaining slots. Per-class hit caps (s10 and s12 12, s14-s16 8, s17b 30 per class and input) only bound what one round
locates from the outputs: an input with a further match is listed again by the re-run and is still flagged.
Ledger. Every replacement is recorded (private): key, article, field, offset and length in the masker-v2 text, classes,
placeholder, SHA256 of the original string, round. No plaintext is stored in a public file.

Outputs (private): OUT/inputs_m2r/<period>/round_<r>/INPUTS.jsonl, LEDGER.jsonl, ROUND.json, RESIDUAL_<UTC>.json;
OUT/inputs_m2r/<period>/REDACTION_FINAL_<UTC>.json. Detector re-runs: OUT/detector_flags/<period>_rd<r>_<UTC>/.
Reads no label, return or provider output.
"""
import ast
import bisect
import difflib
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import td4_spec as T
import td4_env as E

# PUBLIC COPY: the two CRSP PERMNOs of the firms whose names begin with these index words are kept in the private
# frozen code (see five_session/PUBLIC_COPY_NOTES.md); with this empty tuple the MA2:G3 class never fires here.
INDEX_NAME_PERMNOS = ()

ROUNDS = 3
ROOT = T.OUT / 'inputs_redacted'          # (OUT/inputs_m2r holds a stopped first round-1 attempt; record only, not read)
PH = {'name': '[REDACTED]', 'date': '[DATE]', 'term': '[TERM]'}
PLACEHOLDERS = tuple(PH.values())
TOKEN = re.compile(r'\[[A-Z0-9_/]+\]')                     # any bracket token (role tokens and placeholders)
ROLE = re.compile(r'\[(?:FOCAL|LINK_\d+|(?:SUPPLIER|CUSTOMER)(?:_OF_(?:SUPPLIER|CUSTOMER))*_\d+)\]')
ART10 = re.compile(r'^Article (\d+) \| firm: ((?:\[[A-Z0-9_]+\](?:, )?)+) \| [^\n]*\nHeadline: ([^\n]*)\nText: (.*?)'
                   r'(?=\n\nArticle \d+ \||\Z)', re.S | re.M)      # s10's article parser (text up to the next article)
B11 = re.compile(r"\s*([A-Z][\w&.'\- ]{2,80}?) (?:reports?|posts?) ")
POSS = re.compile(r"['’]s$")
FIRM_LEVEL = ('M4:NONAME', 'M4:NONAME_AFTER_2022')         # flags without a text span (a firm in the input)
NAME_DATE_CLASSES = ('MA1_dates:id_reuters_story', 'MA1_dates:x_url', 'MA1_dates:m1_MISTAG', 'MA1_dates:v2_MISTAG_RX',
                     'MA1_dates:x_mistag_any')
TERM_CLASSES = ('M2:surface', 'M2:weekday_token_giveback', 'M3:P5_price_slot', 'MA1:IDX', 'MA1:GB_kept',
                'MA1:GB_frozen', 'MA1:nasdaq', 'MA2:G1', 'MA2:G2', 'MA2:G3')


def kind_of(cls):
    """Placeholder kind of a detector class (name, date or term)."""
    if cls in TERM_CLASSES or cls.startswith('M4:GB_'):
        return 'term'
    if cls in NAME_DATE_CLASSES:
        return 'name'
    if cls.startswith(('MA1_dates:', 'MA2:D:', 'MA2:D12:', 'M1:D_', 'M2:year2', 'M2:month_day')) or \
            cls in ('M3:P10_story_time_remnant', 'M4:DATE_MD', 'M4:YEAR4', 'M4:STORY_REM'):
        return 'date'
    return 'name'


def sha_s(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def tokmap(meta):
    t = {'[FOCAL]': int(meta['focal_permno'])}
    for s in meta['shown']:
        t[s['token']] = int(s['permno'])
    for x in meta['links']:
        t[x['token']] = int(x['permno'])
    return t


# ------------------------------------------------------------------------------------------------ derived strings
class Src:
    """A string derived from the input text (copied pieces, inserted separators, tokens replaced) with a map back to
    input-text positions; inserted characters map to nothing."""

    def __init__(self):
        self.buf, self.segs, self.n = [], [], 0

    def copy(self, text, a, b):
        if b > a:
            self.segs.append((self.n, self.n + b - a, a))
            self.buf.append(text[a:b])
            self.n += b - a
        return self

    def lit(self, s):
        self.buf.append(s)
        self.n += len(s)
        return self

    def copy_sub(self, text, a, b, rx, rep):
        i = a
        for m in rx.finditer(text, a, b):
            self.copy(text, i, m.start())
            self.lit(rep)
            i = m.end()
        return self.copy(text, i, b)

    def done(self):
        self.s = ''.join(self.buf)
        self.starts = [x[0] for x in self.segs]
        del self.buf
        return self

    def to_text(self, a, b):
        out = []
        k = max(0, bisect.bisect_right(self.starts, a) - 1)
        while k < len(self.segs) and self.segs[k][0] < b:
            ss, se, ts = self.segs[k]
            lo, hi = max(a, ss), min(b, se)
            if lo < hi:
                out.append((ts + lo - ss, ts + hi - ss))
            k += 1
        return out


def find_exact(s, needle, ctx, wl, wr, rep=None):
    """Positions where the detector's context window s[max(0, i - wl):i + len(needle) + wr] (newlines replaced by rep)
    equals the recorded context."""
    out = []
    if not needle:
        return out
    i = s.find(needle)
    while i >= 0:
        e = i + len(needle)
        c = s[max(0, i - wl):e + wr]
        if rep is not None:
            c = c.replace('\n', rep)
        if c == ctx:
            out.append((i, e))
        i = s.find(needle, i + 1)
    return out


def find_prefix(s, prefix, ctx, wl, wr, rep=None, reach=200):
    """As find_exact for a recorded string that is a truncated or shortened form of the match (end searched)."""
    out = []
    i = s.find(prefix) if prefix else -1
    while i >= 0:
        for e in range(i + len(prefix), min(len(s), i + len(prefix) + reach) + 1):
            c = s[max(0, i - wl):e + wr]
            if rep is not None:
                c = c.replace('\n', rep)
            if c == ctx:
                out.append((i, e))
                break
        i = s.find(prefix, i + 1)
    return out


def word_rx(w, flags=0, left=r'(?<![A-Za-z0-9])', right=r'(?![A-Za-z0-9])'):
    return re.compile(left + re.escape(w) + right, flags)


# --------------------------------------------------------------------------------------------------- one input
class Doc:
    """One input in the round's text: article content segments, protected bracket tokens, role-token occurrences, and
    the replacement requests of the round."""

    def __init__(self, key, text, meta):
        self.key, self.text = key, text
        self.t2p = tokmap(meta)
        self.meta = meta
        self.arts = []                            # (article no, own tokens, (hs, he), (ts, te))
        self.raw_end = []                         # s10's untrimmed text end per article
        for m in ART10.finditer(text):
            te = m.end(4)
            while te > m.start(4) and text[te - 1] in ' \t\r\n':
                te -= 1
            self.arts.append((int(m.group(1)), TOKEN.findall(m.group(2)), (m.start(3), m.end(3)), (m.start(4), te)))
            self.raw_end.append(m.end(4))
        self.segs = []
        for i, (no, toks, h, t) in enumerate(self.arts):
            self.segs.append((h[0], h[1], i, 'H'))
            self.segs.append((t[0], t[1], i, 'T'))
        self.segs.sort()
        self.seg_starts = [s[0] for s in self.segs]
        self.prot = [(m.start(), m.end()) for m in TOKEN.finditer(text)]
        self.prot_starts = [p[0] for p in self.prot]
        self.roles = [(m.start(), m.end(), m.group(0)) for m in ROLE.finditer(text)]
        self.news_at = text.find('\nPARTNER NEWS\n')
        self.req = []                             # (start, end, kind, class)
        self.found = Counter()                    # class -> located spans
        self._cache = {}

    def seg_of(self, a, b):
        k = bisect.bisect_right(self.seg_starts, a) - 1
        if k >= 0 and self.segs[k][0] <= a and b <= self.segs[k][1]:
            return self.segs[k]
        return None

    def first_line(self, i):
        """End of the first line of the article's text field (s17b and s11 fields are single lines)."""
        ts, te = self.arts[i][3]
        j = self.text.find('\n', ts, te)
        return te if j < 0 else j

    def fields(self):
        """(article index, field, start, end) of the s17b/s11 single-line fields: headline line and first text line."""
        out = []
        for i, (no, toks, h, t) in enumerate(self.arts):
            out.append((i, 'H', h[0], h[1]))
            out.append((i, 'T', t[0], self.first_line(i)))
        return out

    def add(self, a, b, cls, kind=None):
        """Request a [REDACTED]/[DATE] replacement of text[a:b]: cut to article content, minus bracket tokens,
        whitespace trimmed. Returns the number of spans added."""
        kind = kind or kind_of(cls)
        if kind == 'term':
            return self.add_term(a, cls)
        n = 0
        for s0, e0, _, _ in self.segs:
            lo, hi = max(a, s0), min(b, e0)
            if lo >= hi:
                continue
            k = bisect.bisect_right(self.prot_starts, hi) - 1
            j = max(0, bisect.bisect_right(self.prot_starts, lo) - 1)
            cut = [p for p in self.prot[j:k + 1] if p[0] < hi and p[1] > lo]
            pieces, i = [], lo
            for ps, pe in sorted(cut):
                if ps > i:
                    pieces.append((i, ps))
                i = max(i, pe)
            if i < hi:
                pieces.append((i, hi))
            for x, y in pieces:
                while x < y and self.text[x].isspace():
                    x += 1
                while y > x and self.text[y - 1].isspace():
                    y -= 1
                if x < y:
                    self.req.append((x, y, kind, cls))
                    n += 1
        if n:
            self.found[cls] += n
        return n

    def add_term(self, tok_start, cls):
        """Request a [TERM] replacement of the role token starting at tok_start (article content only)."""
        for s, e, tok in self.roles:
            if s == tok_start:
                if self.seg_of(s, e) is None:
                    return 0
                self.req.append((s, e, 'term', cls))
                self.found[cls] += 1
                return 1
        return 0

    def add_tokens_in(self, a, b, tok, cls):
        n = 0
        for s, e, t in self.roles:
            if a <= s and e <= b and t == tok:
                n += self.add_term(s, cls)
        return n

    def content_matches(self, rx, a=None, b=None):
        """Regex matches over the whole input text whose span lies inside one article content segment."""
        for m in rx.finditer(self.text, a or 0, len(self.text) if b is None else b):
            if m.end() > m.start() and self.seg_of(m.start(), m.end()) is not None:
                yield m

    def final_spans(self):
        """Non-overlapping replacement spans: [TERM] tokens as they are; [REDACTED]/[DATE] requests merged when they
        overlap or touch ([REDACTED] wins over [DATE])."""
        terms = {}
        for s, e, k, c in self.req:
            if k == 'term':
                terms.setdefault((s, e), set()).add(c)
        other = sorted((s, e, k, c) for s, e, k, c in self.req if k != 'term')
        merged = []
        for s, e, k, c in other:
            if merged and s <= merged[-1][1] and self.seg_of(merged[-1][0], max(e, merged[-1][1])) is not None:
                ms, me, mk, mc = merged[-1]
                merged[-1] = (ms, max(me, e), 'name' if 'name' in (mk, k) else 'date', mc | {c})
            else:
                merged.append((s, e, k, {c}))
        out = [(s, e, 'term', cs) for (s, e), cs in terms.items()] + merged
        out.sort()
        for (a1, b1, _, _), (a2, b2, _, _) in zip(out, out[1:]):
            if a2 < b1:
                raise ValueError(f'overlapping replacement spans in {self.key}')
        return out


# ------------------------------------------------------------------------------------- reviewer constants (run copy)
def consts(path, names, extra=None):
    """Top-level assignments `names` of a reviewer script (its own regular expressions and tables), evaluated in order
    without running the script."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    ns = {'re': re, **(extra or {})}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
            exec(compile(ast.Module([node], []), str(path), 'exec'), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise ValueError(f'{path}: constants not found {missing}')
    return ns


def _jsonl(p):
    p = Path(p)
    if not p.exists():
        return
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Hits:
    """The detector outputs of one run folder, indexed by key (only the flagged keys)."""

    def __init__(self, run, flagged, period, store, kept_groups):
        import td4v2_flags as FL
        r1, r2 = Path(run) / 'ma1', Path(run) / 'ma2'
        p2 = r2 / 'population_private2'
        want = set(flagged)
        self.run = Path(run)
        self.m1 = {r['key']: r['hits'] for r in _jsonl(r1 / 's10_m1_matches.jsonl') if r['key'] in want}
        self.m2 = defaultdict(list)
        for t in _jsonl(r1 / 's11_triples.jsonl'):
            if t['class'] in FL.EXCLUDED['s11']:
                continue
            for k in t['keys']:
                if k in want:
                    self.m2[k].append((t['class'], t['permno'], t['form']))
        self.header = defaultdict(list)
        for t in _jsonl(r1 / 's11_header_candidates.jsonl'):
            for k in t['keys']:
                if k in want:
                    self.header[k].append((t['header'], t['permno']))
        self.surface = defaultdict(list)
        if (r1 / 's11_surface_keys.json').exists():
            for lab, ks in json.loads((r1 / 's11_surface_keys.json').read_text(encoding='utf-8')).items():
                for k in ks:
                    if k in want:
                        self.surface[k].append(lab)
        self.m3 = {r['key']: r['hits'] for r in _jsonl(r1 / 's12_m3_matches.jsonl') if r['key'] in want}
        self.m4 = defaultdict(list)
        for h in _jsonl(r1 / 's13_m4_hits.jsonl'):
            if h['key'] in want:
                self.m4[h['key']].append(h)
        self.s17 = {r['key']: r['hits'] for r in _jsonl(r1 / 's17b_hits.jsonl') if r['key'] in want}
        self.targeted = defaultdict(list)
        if (r1 / 's41_targeted.json').exists():
            for lab, ks in json.loads((r1 / 's41_targeted.json').read_text(encoding='utf-8'))['keys'].items():
                for k in ks:
                    if k in want:
                        self.targeted[k].append(lab)
        self.ma2 = {}
        acr, g2w = FL._ma2_rule(Path(run))
        vocab = json.loads((p2 / 'vocab.json').read_text(encoding='utf-8'))
        self.need_alt = set()
        for r in _jsonl(p2 / 'pop_items.jsonl'):
            k = r['key']
            if k not in want or not (r['sw'] or r['glued'] or r['date']):
                continue
            g = []
            for x in r['sw']:
                wu, case, p, tcls, own, why, fld, alt, prev3, next3, tok = x
                if tcls == 'link':
                    continue
                c = None
                if 'ticker' in why and (vocab.get(wu.lower(), 0) >= 3 or wu in acr):
                    c = 'MA2:G1'
                elif wu in g2w and 'ticker' not in why:
                    c = 'MA2:G2'
                elif wu in ('NASDAQ', 'DOW') and p in INDEX_NAME_PERMNOS:
                    c = 'MA2:G3'
                if c:
                    g.append((c, wu, fld, alt, prev3, next3, tok))
                    self.need_alt.add(alt)
            glued = [x for x in r['glued'] if vocab.get(x[0].lower(), 0) == 0]
            dates = [x for x in r['date'] if x[0] not in FL.EXCLUDED['ma2_date']]
            self.ma2[k] = (g, glued, dates)
        del vocab
        self.kept = kept_groups
        c14 = r1 / 's14_dates_d8_mistag.py'
        ns14 = consts(c14, ['APOS', 'MON_ANY', 'MON2', 'RX', 'YEAR_UNITS'])
        self.rx14, self.year_units = ns14['RX'], ns14['YEAR_UNITS']
        self.rx15 = consts(r1 / 's15_dates_extra.py', ['APOS', 'MON', 'RX'])['RX']
        self.rx16 = consts(r1 / 's16_dates_extra2.py', ['MON', 'RX'])['RX']
        ns11 = consts(r1 / 's11_m2_pop.py', ['APOS', 'MON', 'RX', 'HEADER', 'FIRMS'])
        self.rx11, self.header_rx, self.firms = ns11['RX'], ns11['HEADER'], ns11['FIRMS']
        self.pat41 = consts(r1 / 's41_targeted.py', ['PAT'])['PAT']
        src41b = (r1 / 's41b_nasdaq.py').read_text(encoding='utf-8')
        self.rx41b = consts(r1 / 's41b_nasdaq.py', ['RX'])['RX']
        self.p41b = int(re.search(r'if p != (\d+):', src41b).group(1))
        self.pats12 = FL._s12_patterns(Path(run))
        self.store = store
        self.raw = {}

    def load_raw(self):
        import pyarrow.parquet as pq
        if not self.need_alt:
            return
        pf = pq.ParquetFile(self.store)
        for g in range(pf.metadata.num_row_groups):
            d = pf.read_row_group(g, columns=['alt_id', 'headline', 'body_short']).to_pydict()
            for i, a in enumerate(d['alt_id']):
                if a in self.need_alt:
                    self.raw[a] = (d['headline'][i] or '', (d['body_short'][i] or '')[:2500])
            del d


# ----------------------------------------------------------------------------------------------- locators
def _m1_sources(doc):
    """s10's body ('\\n'.join(headline + '\\n' + text)) and per-article strings, mapped to the input."""
    body = Src()
    arts = []
    for i, (no, toks, h, t) in enumerate(doc.arts):
        te = doc.raw_end[i]
        if i:
            body.lit('\n')
        body.copy(doc.text, h[0], h[1]).lit('\n').copy(doc.text, t[0], te)
        arts.append((toks, Src().copy(doc.text, h[0], h[1]).lit('\n').copy(doc.text, t[0], te).done()))
    return body.done(), arts


def loc_m1(doc, hits):
    if 'm1' not in doc._cache:
        doc._cache['m1'] = _m1_sources(doc)
    body, arts = doc._cache['m1']
    for cls, hs in hits.items():
        c = 'M1:' + cls
        for h in hs:
            m, ctx, tok = h['m'], h['ctx'], h.get('tok') or ''
            spans = []
            if cls == 'B13' and '-' in m:
                word = m.split('-', 1)[1]
                for mm in doc.content_matches(re.compile(re.escape(tok) + r'-(' + re.escape(word) + r')\b')):
                    doc.add(mm.start(1), mm.end(1), c)
                continue
            if cls == 'B11':
                for i, (no, toks, hd, t) in enumerate(doc.arts):
                    if tok in toks:
                        mm = B11.match(doc.text, hd[0], hd[1])
                        if mm and mm.group(1)[:50] == m:
                            doc.add(mm.start(1), mm.end(1), c)
                continue
            if '(own' in m:
                base = m.split('(', 1)[0]
                rx = word_rx(base, re.I if base.isalpha() else 0, left=r'(?<![A-Za-z0-9])', right=r'(?![A-Za-z0-9])')
                for i, (no, toks, hd, t) in enumerate(doc.arts):
                    if not tok or tok in toks:
                        for a, b in (hd, t):
                            for mm in doc.content_matches(rx, a, b):
                                doc.add(mm.start(), mm.end(), c)
                continue
            for src in [body] + [s for _, s in arts]:
                spans = find_exact(src.s, m, ctx, 70, 70, ' | ')
                if not spans:
                    spans = find_prefix(src.s, m, ctx, 70, 70, ' | ')
                if spans:
                    for a, b in spans:
                        for x, y in src.to_text(a, b):
                            doc.add(x, y, c)
                    break


def _plain11(doc, i):
    """s11's article string: headline + ' || ' + text line, every bracket token replaced by a space."""
    no, toks, h, t = doc.arts[i]
    s = Src().copy_sub(doc.text, h[0], h[1], re.compile(r'\[[A-Z0-9_]+\]'), ' ').lit(' || ')
    return s.copy_sub(doc.text, t[0], doc.first_line(i), re.compile(r'\[[A-Z0-9_]+\]'), ' ').done()


def _trim11(cls, form, a, b):
    """The name part of an s11 form (the verb of english_short_name, the bullet of ticker_brief_bullet, 'BRIEF-' and a
    possessive are not name residues)."""
    if cls == 'english_short_name':
        mm = re.match(r'[A-Za-z]+', form)
        return (a, a + mm.end()) if mm else None
    if cls == 'ticker_brief_bullet':
        mm = re.search(r'\*\s*([A-Za-z0-9]+)\s*[-:–]', form)
        return (a + mm.start(1), a + mm.end(1)) if mm else None
    if cls == 'ticker_title_brief_headline' and form.startswith('BRIEF-'):
        return a + 6, b
    if cls in ('initialism2', 'ticker_title_case', 'ticker_upper_nonenglish'):
        mm = POSS.search(form)
        return (a, b - len(mm.group(0))) if mm else (a, b)
    return a, b


def loc_m2(doc, triples, rx11):
    plains = None
    for cls, permno, form in triples:
        c = 'M2:' + cls
        if cls in rx11:                                     # s11's own date and RIC-suffix expressions (full text)
            for mm in rx11[cls].finditer(doc.text):
                if cls == 'ric_suffix_after_token':
                    tm = TOKEN.match(mm.group(0))
                    if tm:
                        doc.add(mm.start() + tm.end(), mm.end(), c)
                else:
                    doc.add(mm.start(), mm.end(), c)
            continue
        if cls == 'weekday_token_giveback' or not form:
            continue                                         # no located span: withheld
        if plains is None:
            plains = [_plain11(doc, i) for i in range(len(doc.arts))]
        right = cls != 'english_short_name' and len(form) < 40     # s11's eng_rx verb has no right boundary; [:40]
        for p in plains:
            i = p.s.find(form)
            while i >= 0:
                e = i + len(form)
                ok = (not form[0].isalnum() or i == 0 or not p.s[i - 1].isalnum()) and \
                     (not right or not form[-1].isalnum() or e == len(p.s) or not p.s[e].isalnum())
                if ok:
                    t = _trim11(cls, form, i, e)
                    if t:
                        for x, y in p.to_text(*t):
                            doc.add(x, y, c)
                i = p.s.find(form, i + 1)


def loc_header(doc, headers, header_rx):
    want = {h for h, _ in headers}
    for i in range(len(doc.arts)):
        ts = doc.arts[i][3][0]
        mm = header_rx.match(doc.text[ts:doc.first_line(i)])     # '^' anchors at the field start (s11: group(3))
        if mm and mm.group(1).strip() in want:
            doc.add(ts + mm.start(1), ts + mm.end(1), 'M2:header')


def loc_surface(doc, labels, firms, pm_firm):
    for tok, p in doc.t2p.items():
        lab = pm_firm.get(p)
        if lab in labels:
            rx = re.compile(firms[lab].replace('{T}', re.escape(tok)))
            for mm in rx.finditer(doc.text):
                doc.add_tokens_in(mm.start(), mm.end(), tok, 'M2:surface')


def loc_m3(doc, hits, M3):
    news0 = doc.text.find('PARTNER NEWS')
    own = defaultdict(list)                                  # token -> content segments of its own articles
    for i, (no, toks, h, t) in enumerate(doc.arts):
        for tk in toks:
            own[tk] += [h, t]
    allseg = [(s, e) for s, e, _, _ in doc.segs]
    for name, hs in hits.items():
        c = 'M3:' + name
        if name.startswith('X_'):
            continue
        for cls, tok, permno, snip in hs:
            mm = re.match(r'<([^>|]+)(?:\|[^>]*)?> ', snip)
            w = mm.group(1) if mm else None
            if name in ('P1_lowercase_name', 'P3_short_name_glued', 'P7_prefix_of_one_word_name'):
                rx, segs = word_rx(w), allseg
            elif name == 'P9_digit_name_part':
                rx, segs = word_rx(w, re.I, left=r'(?<![A-Za-z0-9.])'), allseg
            elif name == 'P2_titlecase_ticker':
                rx, segs = word_rx(w), own.get(tok, [])
            elif name == 'P4_short_name_in_own_brief':
                rx = re.compile(r'(?:\*|' + re.escape(tok) + r'\s*[-–:])\s*(' + re.escape(w) + r')\b(?!\s*\[)')
                for s, e in own.get(tok, []):
                    for m in doc.content_matches(rx, s, e):
                        doc.add(m.start(1), m.end(1), c)
                continue
            elif name == 'P6_code_subsequence':
                code = snip.split(' key ')[0]
                rx, segs = word_rx(code, left=r'(?<![A-Za-z0-9\-])', right=r'(?!\d)'), own.get(tok, [])
            elif name == 'P8_name_part_next_to_token':
                for m in M3.ADJ.finditer(doc.text, max(0, news0)):
                    t = m.group(1) or m.group(4) or m.group(5)
                    g = 2 if m.group(2) else (3 if m.group(3) else 6)
                    if t == tok and m.group(g).upper() == w:
                        doc.add(m.start(g), m.end(g), c)
                continue
            elif name == 'P5_price_slot':
                for rx in (M3.PRICE_SLOT, M3.YEAR_SLOT):
                    for m in rx.finditer(doc.text, max(0, news0)):
                        if m.group(1) == tok:
                            doc.add_term(m.start(1), c)
                continue
            elif name == 'P11_bang_after_token':
                for m in M3.BANG.finditer(doc.text, max(0, news0)):
                    if m.group(1) == tok:
                        doc.add(m.end(1), m.end(), c, 'name')
                continue
            elif name == 'P10_story_time_remnant':
                for m in doc.content_matches(M3.STORY_REMNANT, max(0, news0)):
                    doc.add(m.start(), m.end(), c)
                continue
            else:
                continue                                     # unknown class: no located span (withheld)
            if not w and name != 'P6_code_subsequence':
                continue
            for s, e in segs:
                for m in doc.content_matches(rx, s, e):
                    doc.add(m.start(), m.end(), c)


def loc_m4(doc, hits):
    for h in hits:
        c = 'M4:' + h['class']
        if c in FIRM_LEVEL:
            continue                                         # firm-level flag: no text span (withheld)
        for a, b in find_exact(doc.text, h['match'], h['ctx'], 70, 70, ' '):
            if c.startswith('M4:GB_'):
                doc.add_tokens_in(a, b, h['token'], c)
            else:
                doc.add(a, b, c)


def loc_dates(doc, classes, H):
    """MA1 s14/s15/s16 classes: the scripts' own expressions re-run; every match in article text (s14: after the
    PARTNER NEWS line, a 4-digit year followed by a unit excluded as in the flag rule)."""
    import td4v2_flags as FL
    for c in classes:
        cls = c.split(':', 1)[1]
        if cls in H.rx14:
            if cls in FL.EXCLUDED['s14']:
                continue
            for m in doc.content_matches(H.rx14[cls]):
                if m.start() <= doc.news_at:
                    continue
                if cls.endswith('YEAR4') and H.year_units.match(doc.text, m.end()):
                    continue
                doc.add(m.start(), m.end(), c)
        for rx in (H.rx15.get(cls), H.rx16.get(cls)):
            if rx is not None:
                for m in doc.content_matches(rx):
                    doc.add(m.start(), m.end(), c)


def _field_token_ctx(doc, fs, fe, s, e, tok):
    return doc.text[max(fs, s - 70):s] + tok + doc.text[e:min(fe, e + 70)]


def loc_s17(doc, hits):
    flds = doc.fields()
    plains = {}

    def plain(i, f, fs, fe):
        if (i, f) not in plains:
            plains[(i, f)] = Src().copy_sub(doc.text, fs, fe, ROLE, ' ').done()
        return plains[(i, f)]
    for det, hs in hits.items():
        if det == 'GB':
            continue                                         # GB: kept groups only (loc_gb)
        c = 'MA1:' + det
        for h in hs:
            if det in ('IDX', 'REM'):
                p, ctx = h[0], h[-1]
                for i, f, fs, fe in flds:
                    for s, e, tok in doc.roles:
                        if fs <= s and e <= fe and doc.t2p.get(tok) == p and _field_token_ctx(doc, fs, fe, s, e, tok) == ctx:
                            if det == 'IDX':
                                doc.add_term(s, c)
                            else:
                                mm = re.match(r'[ \t]*-?[ \t]*(' + re.escape(h[1]) + r')\b', doc.text[e:min(fe, e + 40)])
                                if mm:
                                    doc.add(e + mm.start(1), e + mm.end(1), c)
                continue
            if det == 'EMB':
                p, s_, ctx = h
                if 'emb' not in plains:
                    a = doc.news_at + len('\nPARTNER NEWS\n') if doc.news_at >= 0 else 0
                    plains['emb'] = Src().copy_sub(doc.text, a, len(doc.text), ROLE, ' ').done()
                src = plains['emb']
                for a, b in find_exact(src.s, s_, ctx, 70, 70):
                    for x, y in src.to_text(a, b):
                        doc.add(x, y, c)
                continue
            if det == 'OWN':
                s_, ctx = h[2], h[3]
            elif det == 'CO':
                s_, ctx = h[1], h[2]
            elif det == 'PFX':
                s_, ctx = h[2], h[3]
            else:
                continue
            for i, f, fs, fe in flds:
                src = plain(i, f, fs, fe)
                if ctx and ctx[:20] not in src.s:
                    continue
                for a, b in find_exact(src.s, s_, ctx, 70, 70):
                    if det == 'CO':
                        mm = POSS.search(s_)
                        b -= len(mm.group(0)) if mm else 0
                    for x, y in src.to_text(a, b):
                        doc.add(x, y, c)


_WORD17 = re.compile(r"[A-Za-z][A-Za-z0-9&'’]*")
_ANYTOK = re.compile(r'\[[A-Z0-9_/]+\]')


def _words17(s):
    return [w.lower().replace('’', "'").rstrip("'") for w in _WORD17.findall(s)]


def loc_gb(doc, groups, cls='MA1:GB_kept'):
    """Every slot of a frozen GB group (permno, word, side, neighbour) in the input: a role token of the firm whose
    adjacent readable word in the same field is the group's neighbour (s17b's neighbour rule)."""
    by_p = defaultdict(set)
    for p, w, side, nb in groups:
        by_p[p].add((side, nb))
    if not by_p:
        return 0
    n = 0
    for i, f, fs, fe in doc.fields():
        for s, e, tok in doc.roles:
            if not (fs <= s and e <= fe):
                continue
            p = doc.t2p.get(tok)
            if p not in by_p:
                continue
            left, right = doc.text[max(fs, s - 40):s], doc.text[e:min(fe, e + 40)]
            lseg, rseg = _ANYTOK.split(left)[-1], _ANYTOK.split(right)[0]
            lw, rw = _words17(lseg), _words17(rseg)
            prev = lw[-1] if lw and re.search(r"[A-Za-z0-9'’-]\s*$", lseg) else None
            nxt = rw[0] if rw and re.match(r"^[\s-]*[A-Za-z]", rseg) else None
            if ('L', prev) in by_p[p] or ('R', nxt) in by_p[p]:
                n += doc.add_term(s, cls)
    return n


def loc_targeted(doc, labels, pat41):
    news_at = doc.news_at + len('\nPARTNER NEWS\n') if doc.news_at >= 0 else 0
    for tok, p in doc.t2p.items():
        for spec in pat41.get(p, ()):
            if spec[0] not in labels:
                continue
            if spec[1] is not None:
                for mm in re.finditer(spec[1].replace('{T}', re.escape(tok)), doc.text[news_at:]):
                    doc.add_tokens_in(news_at + mm.start(), news_at + mm.end(), tok, 'MA1:targeted')
            else:
                for mm in re.finditer(spec[2], doc.text[news_at:]):
                    doc.add(news_at + mm.start(), news_at + mm.end(), 'MA1:targeted', 'name')


def loc_nasdaq(doc, rx, permno):
    news_at = doc.news_at + len('\nPARTNER NEWS\n') if doc.news_at >= 0 else 0
    news = doc.text[news_at:]
    for tok, p in doc.t2p.items():
        if p != permno:
            continue
        for m in re.finditer(rx.replace('{T}', re.escape(tok)), news):
            left, right = news[max(0, m.start() - 30):m.start()], news[m.end():m.end() + 80]
            if re.search(r'(?i)short[ \t]+interest[ \t]+on[ \t]*$', left) or re.search(r'(?i)the[ \t]+exchange[ \t]+said', right):
                continue
            doc.add_tokens_in(news_at + m.start(), news_at + m.end(), tok, 'MA1:nasdaq')


# MA2 single-word alignment (ma2_s7b_population, replicated with positions)
_PREFIX2 = re.compile(r"^(BRIEF-|UPDATE\s?\d*-|ANALYSIS-|ROUNDUP-|EXCLUSIVE-)", re.I)
_TOK2 = re.compile(r"\[[A-Z_0-9]+\]")
_SKIP2 = ("[DATE]", "[YEAR]", "[NEWS_AGENCY]", "[EMAIL]", "[URL]")


def _core2(seg):
    s = _PREFIX2.sub("", seg)
    s = re.sub(r"(['\u2019]S|['\u2019]s|['\u2019])$", "", s.strip(",.;:()\"<>!?"))
    return s.strip(",.;:()\"<>!?-")


def _align2(a, b_text, b_off):
    """ma2_s7b spans(): one-word raw spans replaced by a masked span with exactly one role-like token; returns
    (raw word, token, prev3, next3, token position in the input)."""
    aw = a.split()
    bm = list(re.finditer(r'\S+', b_text))
    bw = [m.group(0) for m in bm]
    sm = difflib.SequenceMatcher(None, aw, bw, autojunk=False)
    res = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and (i2 - i1) == 1:
            mseg = " ".join(bw[j1:j2])
            toks = [t for t in _TOK2.findall(mseg) if t not in _SKIP2]
            if len(toks) == 1:
                pos = None
                for j in range(j1, j2):
                    k = bw[j].find(toks[0])
                    if k >= 0:
                        pos = b_off + bm[j].start() + k
                        break
                res.append((aw[i1], toks[0], " ".join(aw[max(0, i1 - 3):i1]), " ".join(aw[i2:i2 + 3]), pos))
    return res


def loc_ma2(doc, item, raw):
    g, glued, dates = item
    for x in glued:                                          # B1: the glued compound (ctx: full text, 60, newline ' ')
        for a, b in find_exact(doc.text, x[0], x[5], 60, 60, ' '):
            doc.add(a, b, 'MA2:B1')
    pos = doc.text.find('PARTNER NEWS')
    body = doc.text[pos:] if pos >= 0 else doc.text
    off = pos if pos >= 0 else 0
    for nm, mt, ctx in dates:                                # D: the date match (ctx: from PARTNER NEWS, 60)
        for a, b in find_exact(body, mt, ctx, 60, 60, ' '):
            doc.add(off + a, off + b, 'MA2:D:' + nm)
    if not g:
        return
    blocks = re.split(r"\n(?=Article \d+ \| firm: )", doc.text)
    starts, p = [], 0
    for b in blocks:
        starts.append(p)
        p += len(b) + 1
    arts = list(zip(blocks[1:], starts[1:]))
    alt_k = {a['alt_id']: k for k, a in enumerate(doc.meta['articles'])}
    want = defaultdict(list)
    for c, wu, fld, alt, prev3, next3, tok in g:
        want[(alt, fld)].append((c, wu, prev3, next3, tok))
    for (alt, fld), ws in want.items():
        k = alt_k.get(alt)
        if k is None or k >= len(arts) or alt not in raw:
            continue
        blk, b0 = arts[k]
        rh, rb = raw[alt]
        if fld == 'H':
            mm = re.search(r"\nHeadline: (.*)", blk)
            if not mm:
                continue
            al = _align2(rh, mm.group(1), b0 + mm.start(1))
        else:
            mm = re.search(r"\nText: (.*)", blk)
            if not mm:
                continue
            mt = mm.group(1)
            lead = len(mt) - len(mt.lstrip())
            mts = mt.strip()
            cut = len(mts.split()) + 15
            al = _align2(" ".join(rb.split()[:cut]), mts, b0 + mm.start(1) + lead)
        for c, wu, prev3, next3, tok in ws:
            for rawseg, t, p3, n3, ps in al:
                w = _core2(rawseg)
                if t == tok and w.upper() == wu and p3 == prev3 and n3 == next3 and ps is not None:
                    doc.add_term(ps, c)


# ------------------------------------------------------------------------------------------------ round driver
_ROOT = {'root': ROOT}                         # a development run points this at OUT/inputs_m2r_dev/<UTC>


def period_root(period):
    return _ROOT['root'] / str(period)


def round_dir(period, rnd):
    return period_root(period) / f'round_{rnd}'


def _pm_firm(firms):
    """s11's PERMNO -> surface-pattern firm map (CRSP comnam prefix, namedt <= 2024-12-31)."""
    import csv
    out = {}
    for fn in T.CRSP_NAME_FILES:
        with Path(fn).open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if (r.get('namedt') or '')[:10] > '2024-12-31' or not r.get('comnam'):
                    continue
                for k in firms:
                    if r['comnam'].startswith(k):
                        try:
                            out[int(float(r['permno']))] = k
                        except (TypeError, ValueError):
                            pass
    return out


def _m3_module():
    import importlib.util
    import td4v2_flags as FL
    path = FL.ROUND1_MODULES[2]
    spec = importlib.util.spec_from_file_location('td4v2_redact_m3_copy', path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    return mod


def locate(doc, classes, H, ctx):
    """All replacement requests of one input for its flagged classes; returns {class: located spans}."""
    k = doc.key
    if k in H.m1:
        loc_m1(doc, H.m1[k])
    if k in H.m2:
        loc_m2(doc, H.m2[k], H.rx11)
    if k in H.header:
        loc_header(doc, H.header[k], H.header_rx)
    if k in H.surface:
        loc_surface(doc, set(H.surface[k]), H.firms, ctx['pm_firm'])
    if k in H.m3:
        loc_m3(doc, H.m3[k], ctx['M3'])
    if k in H.m4:
        loc_m4(doc, H.m4[k])
    dc = [c for c in classes if c.startswith('MA1_dates:')]
    if dc:
        loc_dates(doc, dc, H)
    if k in H.s17:
        loc_s17(doc, H.s17[k])
    if H.kept:
        loc_gb(doc, H.kept)
    if k in H.targeted:
        loc_targeted(doc, set(H.targeted[k]), H.pat41)
    if 'MA1:nasdaq' in classes:
        loc_nasdaq(doc, H.rx41b, H.p41b)
    if k in H.ma2:
        loc_ma2(doc, H.ma2[k], H.raw)
    d12 = [c for c in classes if c.startswith('MA2:D12:')]
    if d12:
        pos = doc.text.find('\nPARTNER NEWS\n')
        for c in d12:
            rx = H.pats12[c.split(':', 2)[2]]
            for m in doc.content_matches(rx, max(0, pos)):
                doc.add(m.start(), m.end(), c)
    return dict(doc.found)


def _ledger_index(period, upto):
    """Cumulative ledger of rounds 1..upto: key -> sorted [(v2 start, v2 end, placeholder)]."""
    idx = defaultdict(list)
    for r in range(1, upto + 1):
        for e in _jsonl(round_dir(period, r) / 'LEDGER.jsonl'):
            idx[e['key']].append((e['offset'], e['offset'] + e['length'], e['placeholder']))
    for k in idx:
        idx[k].sort()
    return idx


def to_v2(prior, a, b):
    """Map a span of the round's text to masker-v2 coordinates given the earlier replacements (v2 spans with
    placeholders); the span must not touch an earlier placeholder."""
    delta = 0
    for s, e, ph in prior:
        ts = s + delta
        te = ts + len(ph)
        if te <= a:
            delta += len(ph) - (e - s)
            continue
        if ts >= b:
            break
        raise ValueError('span overlaps an earlier placeholder')
    return a - delta, b - delta


def flagged_classes(period, d_inputs, run):
    """key -> included classes flagged by the detector run (td4v2_flags.extract), adjudicated sets apart."""
    import td4v2_flags as FL
    cls, union, n_call, adjud = FL.extract(str(period), d_inputs, Path(run))
    by_key = defaultdict(set)
    adj = set()
    for c, ks in cls.items():
        if c in FL.REPORT_ONLY:
            continue
        if c.startswith('ADJ_'):
            adj |= ks
            continue
        for k in ks:
            by_key[k].add(c)
    return by_key, adj, n_call, {c: len(v) for c, v in cls.items()}


def kept_groups(runs):
    out = set()
    for run in runs:
        p = Path(run) / 'ma1' / 's18_gb_kept.json'
        if p.exists():
            for g in json.loads(p.read_text(encoding='utf-8')):
                out.add((g['permno'], g['w'], g['side'], g['nb']))
    return out


def _round0_run(period):
    import td4v2_flags as FL
    d = FL._dir(period)
    return Path(json.loads(FL.flags_path(d).read_text(encoding='utf-8'))['run_dir'])


def _residual_file(period, rnd):
    runs = sorted(round_dir(period, rnd).glob('RESIDUAL_*.json'))
    if not runs:
        raise PermissionError(f'round {rnd} of {period}: detectors not re-run (residual missing)')
    return runs[-1]


def redact_round(period, rnd, dev_limit=None):
    """Round rnd: locate the matches of the detector run on the round rnd-1 inputs and write the round rnd inputs and
    ledger (private). Population-check-withheld inputs are copied unchanged (never sent). dev_limit (development
    only): the first N records into OUT/inputs_m2r_dev/<UTC>/ (an engineering check, never bound)."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    t0 = time.time()
    period = str(period)
    rnd = int(rnd)
    if not 1 <= rnd <= ROUNDS:
        raise ValueError('rounds 1-3')
    if dev_limit:
        if rnd != 1:
            raise ValueError('development runs: round 1 only')
        _ROOT['root'] = T.OUT / 'inputs_m2r_dev' / R.utc().replace(':', '')
    d = FL._dir(period)
    out = round_dir(period, rnd)
    if out.exists():
        raise FileExistsError(str(out))
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    if rnd == 1:
        run = _round0_run(period)
        src = Path(d) / 'INPUTS.jsonl'
        by_key, adj, n_call, cls_counts = flagged_classes(period, d, run)
        runs_so_far = [run]
        gb_frozen = set()
    else:
        res = json.loads(_residual_file(period, rnd - 1).read_text(encoding='utf-8'))
        run = Path(res['run_dir'])
        src = round_dir(period, rnd - 1) / 'INPUTS.jsonl'
        by_key = defaultdict(set)
        for c, ks in res['by_class_keys'].items():
            for k in ks:
                by_key[k].add(c)
        adj, n_call, cls_counts = set(), res['callable'], res['by_class_inputs']
        runs_so_far = res['runs_so_far']
    groups = kept_groups(runs_so_far)
    H = Hits(run, set(by_key), period, T.OUT / 'inputs' / FL.STORES[period], groups)
    H.load_raw()
    ctx = {'pm_firm': _pm_firm(H.firms), 'M3': _m3_module()}
    prior = _ledger_index(period, rnd - 1)
    out.mkdir(parents=True, exist_ok=False)
    stats = Counter()
    by_class_spans, by_class_inputs, unlocated = Counter(), Counter(), defaultdict(list)
    n_red_inputs = 0
    chars_red = 0
    chars_art = 0
    per_input = []
    with Path(src).open(encoding='utf-8') as fi, (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as fm, \
            (out / 'INPUTS.jsonl').open('x', encoding='utf-8', newline='\n') as fo, \
            (out / 'LEDGER.jsonl').open('x', encoding='utf-8', newline='\n') as fl:
        for li, lm in zip(fi, fm):
            if dev_limit and stats['records'] >= dev_limit:
                break
            r = json.loads(li)
            m = json.loads(lm)
            if r['key'] != m['key']:
                raise ValueError('INPUTS and INPUT_META_COMPACT misaligned at ' + r['key'])
            k = r['key']
            text = r['input_text']
            if rnd == 1:
                r['v2_input_sha256'] = r['input_sha256']
            stats['records'] += 1
            classes = by_key.get(k, set())
            if k in pop:
                stats['population_withheld_copied'] += 1
                fo.write(json.dumps(r, ensure_ascii=False) + '\n')
                continue
            doc = Doc(k, text, m)
            chars_art += sum(e - s for s, e, _, _ in doc.segs)
            located = {}
            if classes or groups:
                located = locate(doc, classes, H, ctx)
                if 'MA1:GB_frozen' in classes:
                    located['MA1:GB_frozen'] = located.get('MA1:GB_kept', 0)
            for c in classes:
                if located.get(c, 0) == 0:
                    unlocated[k].append(c)
            spans = doc.final_spans()
            if spans:
                n_red_inputs += 1
                pr = prior.get(k, [])
                pieces, i = [], 0
                for s, e, kind, cs in spans:
                    ph = PH[kind]
                    va, vb = to_v2(pr, s, e)
                    seg = doc.seg_of(s, e)
                    fl.write(json.dumps({'key': k, 'article': doc.arts[seg[2]][0], 'field': seg[3], 'offset': va,
                                         'length': vb - va, 'placeholder': ph, 'classes': sorted(cs),
                                         'original_sha256': sha_s(text[s:e]), 'round': rnd}) + '\n')
                    pieces.append(text[i:s])
                    pieces.append(ph)
                    i = e
                    chars_red += e - s
                    for c in cs:
                        by_class_spans[c] += 1
                    stats['spans_' + kind] += 1
                for c in {c for _, _, _, cs in spans for c in cs}:
                    by_class_inputs[c] += 1
                pieces.append(text[i:])
                text = ''.join(pieces)
                per_input.append(len(spans))
            r['input_text'] = text
            r['input_sha256'] = sha_s(text)
            r['input_bytes'] = len(text.encode('utf-8'))
            r['redaction_round'] = rnd
            fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    flagged = set(by_key) - pop
    res = {'period': period, 'round': rnd, 'source_inputs': str(src), 'source_inputs_sha256': R.sha(src),
           'detector_run': str(run), 'runs_so_far': [str(x) for x in runs_so_far],
           'inputs': str(out / 'INPUTS.jsonl'), 'inputs_sha256': R.sha(out / 'INPUTS.jsonl'),
           'ledger': str(out / 'LEDGER.jsonl'), 'ledger_sha256': R.sha(out / 'LEDGER.jsonl'),
           'records': stats['records'], 'callable': n_call, 'flagged_inputs_entering': len(flagged),
           'flag_classes_entering': cls_counts, 'gb_frozen_groups': len(groups),
           'adjudicated_withheld': sorted(adj), 'inputs_with_new_redaction': n_red_inputs,
           'new_spans': sum(per_input), 'spans_by_kind': {k: v for k, v in stats.items() if k.startswith('spans_')},
           'spans_by_class': dict(sorted(by_class_spans.items())),
           'inputs_by_class': dict(sorted(by_class_inputs.items())),
           'chars_replaced': chars_red, 'article_chars_callable': chars_art,
           'unlocated_inputs': len(unlocated), 'unlocated': {k: sorted(v) for k, v in sorted(unlocated.items())},
           'unlocated_by_class': dict(Counter(c for v in unlocated.values() for c in v)),
           'firm_level_classes': list(FIRM_LEVEL), 'raw_articles_loaded': len(H.raw),
           'code_sha256': R.sha(Path(__file__)), 'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(out / 'ROUND.json', res)
    return {k: v for k, v in res.items() if k not in ('unlocated', 'adjudicated_withheld')}


def prepare_rerun(period, rnd):
    """Detector re-run on the round rnd inputs: td4v2_flags copies with the INPUTS path changed (population, metadata,
    article store and withheld set of the masker-v2 build); writes RUN_DETECTORS.sh."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    period = str(period)
    d = FL._dir(period)
    inp = (round_dir(period, rnd) / 'INPUTS.jsonl').as_posix()
    if not (round_dir(period, rnd) / 'ROUND.json').exists():
        raise PermissionError('round not written')
    wf = [IV.withheld_file(d)]
    store = T.OUT / 'inputs' / FL.STORES[period]
    run = FL.ROOT / f'{period}_rd{rnd}_{R.utc().replace(":", "")}'
    r1, r2 = run / 'ma1', run / 'ma2'
    r1.mkdir(parents=True, exist_ok=False)
    r2.mkdir(parents=True, exist_ok=False)
    dq, wh = Path(d).as_posix(), [Path(x).as_posix() for x in wf]
    c = (FL.MA1 / 'ma1_common.py').read_text(encoding='utf-8')
    c = FL._sub(c, [("D = Path('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3')", f"D = Path('{dq}')"),
                    ("OUT = Path('D:/OptResearch_data/five_session_review/td4m2/ma1')", f"OUT = Path('{r1.as_posix()}')"),
                    ("json.loads((D / fn).read_text(encoding='utf-8'))", "json.loads(Path(fn).read_text(encoding='utf-8'))"),
                    ("(D / 'INPUTS.jsonl')", f"Path('{inp}')")])
    c = re.sub(r"WITHHELD_FILES = \([^)]*\)", lambda m: 'WITHHELD_FILES = ' + repr(tuple(wh)), c)
    FL._write(r1 / 'ma1_common.py', c)
    for fn in FL.MA1_RUN:
        s = (FL.MA1 / fn).read_text(encoding='utf-8')
        if fn == 's02_fsb1_names.py':
            s = FL._sub(s, [("D = Path('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3')", f"D = Path('{dq}')"),
                            ("OUT = Path('D:/OptResearch_data/five_session_review/td4m2/ma1')",
                             f"OUT = Path('{r1.as_posix()}')")])
        FL._write(r1 / fn, s)
    bad = set()
    for f in wf:
        bad |= set(json.loads(Path(f).read_text(encoding='utf-8'))['withheld_keys'])
    keys = []
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            k = json.loads(line)['key']
            if k not in bad:
                keys.append(k)
    FL._write(r2 / 'MA2_CALLABLE_KEYS.json', json.dumps(keys))
    for fn in FL.MA2_RUN + FL.MA2_READ:
        s = (FL.MA2 / fn).read_text(encoding='utf-8')
        s = s.replace('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3/', dq + '/')
        s = s.replace('"D:/OptResearch_data/five_session_review/td4m2/ma2/"', '"' + r2.as_posix() + '/"')
        s = s.replace('sys.path.insert(0, "D:/OptResearch_data/five_session_review/td4m2/ma2")',
                      'sys.path.insert(0, "' + r2.as_posix() + '")')
        s = s.replace('D:/OptResearch_data/five_session/td4/inputs/article_store_select_2122.parquet', store.as_posix())
        if fn in ('ma2_s7b_population.py', 'ma2_s12_dates_v2.py'):
            s = FL._sub(s, [('open(D + "INPUTS.jsonl"', f'open("{inp}"')])
        FL._write(r2 / fn, s)
    sh = FL.run_plan(run)
    return {'run_dir': str(run), 'callable': len(keys), 'run_script': str(sh),
            'next': f'bash {sh.as_posix()} ; then run_td4v2.py redact-residual --period {period} --which {run.name}'}


def residual(period, rnd, run_name):
    """Still-flagged inputs after the round rnd re-run: td4v2_flags.extract on the re-run (same classes, exclusions,
    article scope; adjudicated sets apart) plus the remaining slots of every frozen GB group in the round's text."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    period, rnd = str(period), int(rnd)
    run = FL.ROOT / run_name
    if not run.name.startswith(f'{period}_rd{rnd}_'):
        raise ValueError('run folder of another period or round')
    t0 = time.time()
    status = FL.run_status(run)
    rd = round_dir(period, rnd)
    by_key, adj, n_call, cls_counts = flagged_classes(period, rd, run)
    prev = json.loads((rd / 'ROUND.json').read_text(encoding='utf-8'))['runs_so_far']
    runs_so_far = [str(x) for x in prev] + [str(run)]
    groups = kept_groups(runs_so_far)
    d = FL._dir(period)
    callable_keys = set(json.loads((run / 'ma2' / 'MA2_CALLABLE_KEYS.json').read_text(encoding='utf-8')))
    gbf = set()
    with (rd / 'INPUTS.jsonl').open(encoding='utf-8') as fi, (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as fm:
        for li, lm in zip(fi, fm):
            r = json.loads(li)
            if r['key'] not in callable_keys:
                continue
            doc = Doc(r['key'], r['input_text'], json.loads(lm))
            if loc_gb(doc, groups, 'MA1:GB_frozen'):
                gbf.add(r['key'])
    for k in gbf:
        by_key[k].add('MA1:GB_frozen')
    by_class = defaultdict(list)
    for k, cs in by_key.items():
        for c in cs:
            by_class[c].append(k)
    res = {'period': period, 'round': rnd, 'run_dir': str(run), 'runs_so_far': runs_so_far,
           'round_inputs_sha256': R.sha(rd / 'INPUTS.jsonl'), 'callable': n_call,
           'still_flagged_inputs': len(by_key), 'still_flagged_share': round(len(by_key) / n_call, 5),
           'by_class_inputs': {c: len(v) for c, v in sorted(by_class.items())},
           'gb_frozen_groups': len(groups), 'gb_frozen_inputs': len(gbf),
           'by_class_keys': {c: sorted(v) for c, v in sorted(by_class.items())},
           'scripts_status': status, 'run_copies_sha256': {str(p): FL.sha(p) for p in sorted(run.rglob('*.py'))},
           'code_sha256': R.sha(Path(__file__)), 'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(rd / f'RESIDUAL_{R.utc().replace(":", "")}.json', res)
    return {k: v for k, v in res.items() if k not in ('by_class_keys', 'run_copies_sha256')}


# ------------------------------------------------------------------------------------------- invariants (checks)
_LABELS = re.compile(r'^Article \d+ \| [^\n]*$|^Headline: |^Text: ', re.M)


def check_record(v2_text, red_text, entries):
    """Independent check of one redacted input against its masker-v2 text and its ledger entries (v2 coordinates:
    (offset, offset + length, placeholder, original SHA256)). Returns the list of problems (empty = clean):
    outside the ledger spans the texts are byte-identical; each span reads as its placeholder; the original string's
    SHA256 matches; a [TERM] span is exactly one role token; a [REDACTED]/[DATE] span holds no bracket token; every
    span lies in one article content field of the v2 text; the header (instructions, menu, characteristics, graph),
    the article and field labels and every bracket token outside [TERM] spans are byte-identical and in order."""
    probs = []
    ents = sorted(entries)
    segs = Doc('check', v2_text, {'focal_permno': 0, 'shown': [], 'links': []})
    i = j = 0
    terms = set()
    for s, e, ph, osha in ents:
        if s < i or e <= s:
            probs.append('overlap_or_empty')
            break
        n = s - i
        if v2_text[i:s] != red_text[j:j + n]:
            probs.append('outside_span_differs')
            break
        j += n
        if ph not in PLACEHOLDERS or red_text[j:j + len(ph)] != ph:
            probs.append('placeholder_mismatch')
            break
        orig = v2_text[s:e]
        if sha_s(orig) != osha:
            probs.append('original_sha256')
        if ph == PH['term']:
            if not ROLE.fullmatch(orig):
                probs.append('term_not_one_role_token')
            terms.add(s)
        elif TOKEN.search(orig):
            probs.append('bracket_token_inside_span')
        if segs.seg_of(s, e) is None:
            probs.append('span_outside_article_text')
        j += len(ph)
        i = e
    else:
        if v2_text[i:] != red_text[j:]:
            probs.append('tail_differs')
    na = v2_text.find('\nPARTNER NEWS\n')
    if na < 0 or red_text[:na + 14] != v2_text[:na + 14]:
        probs.append('header_differs')
    if _LABELS.findall(v2_text) != _LABELS.findall(red_text):
        probs.append('labels_differ')
    keep = [m.group(0) for m in TOKEN.finditer(v2_text) if m.start() not in terms]
    added = Counter(ph for _, _, ph, _ in ents)
    got = Counter(TOKEN.findall(red_text))
    for ph, c in added.items():
        got[ph] -= c
    if +got != Counter(keep):
        probs.append('bracket_tokens_differ')
    if ROLE.findall(red_text) != [t for t in keep if ROLE.fullmatch(t)]:
        probs.append('role_tokens_differ')
    return probs


def render(v2_text, entries):
    """The redacted text from the masker-v2 text and the ledger entries (offset, end, placeholder)."""
    out, i = [], 0
    for s, e, ph in sorted(entries):
        out.append(v2_text[i:s])
        out.append(ph)
        i = e
    out.append(v2_text[i:])
    return ''.join(out)


def verify_round(period, rnd, idem_n=2000):
    """Invariant checks of round rnd against the masker-v2 inputs and the cumulative ledger (every record), and the
    idempotency check: the round's own locator re-run with the same detector outputs on the first idem_n redacted
    callable inputs finds no further span."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    period, rnd = str(period), int(rnd)
    t0 = time.time()
    d = FL._dir(period)
    rd = round_dir(period, rnd)
    led = defaultdict(list)
    for r in range(1, rnd + 1):
        for e in _jsonl(round_dir(period, r) / 'LEDGER.jsonl'):
            led[e['key']].append((e['offset'], e['offset'] + e['length'], e['placeholder'], e['original_sha256']))
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    probs = Counter()
    bad_keys = []
    n = n_red = sha_ok = 0
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fv, (rd / 'INPUTS.jsonl').open(encoding='utf-8') as fr:
        for lv, lr in zip(fv, fr):
            v, r = json.loads(lv), json.loads(lr)
            if v['key'] != r['key']:
                raise ValueError('misaligned round file')
            n += 1
            ents = led.get(v['key'], [])
            p = check_record(v['input_text'], r['input_text'], ents)
            if v['key'] in pop and ents:
                p.append('population_withheld_redacted')
            if render(v['input_text'], [x[:3] for x in ents]) != r['input_text']:
                p.append('render_differs')
            if sha_s(r['input_text']) == r['input_sha256'] and len(r['input_text'].encode('utf-8')) == r['input_bytes']:
                sha_ok += 1
            else:
                p.append('record_hash')
            if r.get('v2_input_sha256') != v['input_sha256']:
                p.append('v2_hash_link')
            for x in p:
                probs[x] += 1
            if p and len(bad_keys) < 50:
                bad_keys.append(v['key'])
            n_red += bool(ents)
    if n != sum(1 for _ in (rd / 'INPUTS.jsonl').open(encoding='utf-8')):
        probs['record_count'] += 1
    idem = idempotency(period, rnd, idem_n)
    res = {'period': period, 'round': rnd, 'records': n, 'records_with_ledger_entries': n_red,
           'ledger_entries': sum(len(v) for v in led.values()), 'problems': dict(probs), 'problem_keys': bad_keys,
           'record_hash_ok': sha_ok, 'clean': not probs and sha_ok == n and idem['new_spans'] == 0,
           'idempotency': idem, 'code_sha256': R.sha(Path(__file__)), 'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(rd / f'VERIFY_{R.utc().replace(":", "")}.json', res)
    return res


def idempotency(period, rnd, n_max):
    """The round's locator applied again, with the same detector outputs and frozen groups, to the round's output text
    of the first n_max redacted callable inputs (file order): the number of further spans found (0 = idempotent)."""
    import td4v2_flags as FL
    import td4v2_inputs as IV
    d = FL._dir(period)
    rd = round_dir(period, rnd)
    info = json.loads((rd / 'ROUND.json').read_text(encoding='utf-8'))
    run = Path(info['detector_run'])
    if rnd == 1:
        by_key = flagged_classes(period, d, run)[0]
    else:
        res = json.loads(_residual_file(period, rnd - 1).read_text(encoding='utf-8'))
        by_key = defaultdict(set)
        for c, ks in res['by_class_keys'].items():
            for k in ks:
                by_key[k].add(c)
    redacted = []
    for e in _jsonl(rd / 'LEDGER.jsonl'):
        if not redacted or redacted[-1] != e['key']:
            redacted.append(e['key'])
        if len(redacted) > n_max:
            break
    redacted = set(redacted[:n_max])
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    groups = kept_groups(info['runs_so_far'])
    H = Hits(run, set(by_key) & redacted, period, T.OUT / 'inputs' / FL.STORES[period], groups)
    H.load_raw()
    ctx = {'pm_firm': _pm_firm(H.firms), 'M3': _m3_module()}
    checked = new = 0
    examples = []
    with (rd / 'INPUTS.jsonl').open(encoding='utf-8') as fi, (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as fm:
        for li, lm in zip(fi, fm):
            r = json.loads(li)
            k = r['key']
            if k not in redacted or k in pop:
                continue
            doc = Doc(k, r['input_text'], json.loads(lm))
            locate(doc, by_key.get(k, set()), H, ctx)
            sp = doc.final_spans()
            new += len(sp)
            if sp and len(examples) < 10:
                examples.append([k, sorted({c for x in sp for c in x[3]})])
            checked += 1
    return {'checked_inputs': checked, 'new_spans': new, 'examples_keys_classes': examples}
