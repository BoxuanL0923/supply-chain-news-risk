"""Independent second residual-name scanner, v3 (TD3C2; reviews TD3C C1 item 8 and TD3BC D1 item 5). v4 (TD3D):
unchanged except this docstring; the v4 adjacency sweep with hidden raw words is built by td3_build (it needs the raw
article text, which this scanner never reads).

v5 (TD3E; review TD3C4 D1, D2), four more checks, written without the masker's code-prefix and near-name rules:
- 'firm_code': a code <letters>[-]<digits>[letters] (letters in upper or Title case) whose letters abbreviate a masked
  firm in the article's input: a ticker or RIC root, a subsequence of a ticker that keeps its first letter (QOVX ->
  QVX), the initials of the first k >= 2 core tokens of a name (also with camel-case parts of mixed-case FactSet names
  split), or the first k >= 2 letters of a name's first core token;
- 'near_prefix': an upper-case word that is a firm's ticker, collapsed initials or first core token minus its last one
  or two letters (>= 2 left), followed by a capitalised word ("QR Logistics" for QRT);
- 'glued_token': a role token glued to a letter or digit;
- 'house_number': a 1-6 digit number right before a role token.

v6 (TD3F; review TD3C5 D1), one more check: 'initialism_unit': an upper- or Title-case word equal to a masked firm's
ticker or RIC root (letters only) or to the initials of the first k >= 2 core tokens of one of its names (camel-case
parts of mixed-case names split), followed by a corporate-unit word or a legal suffix (_SCAN_UNITS, the scanner's own
list).

Written separately from td3_mask and deliberately different from its alias rules:
- name sources, loaded here: every raw CRSP comnam and ticker row (both CRSP files, namedt <= 2022-12-31) of each
  masked firm and of its issuer siblings (PERMNOs sharing a CRSP issuer CUSIP6); every FactSet company name and
  ticker (FSB1 provisional test data, start_ <= 2022-12-31) of the FactSet ids linked to the firm's issuer CUSIP6
  (rows with a blank CUSIP included); RIC roots (first month <= 2022-12) of the firm and its siblings;
- method: article text and names are reduced to normalized token sequences (case-folded, accents and apostrophes
  dropped, '&' read as 'and', connectors of/the/and dropped, runs of single capitals joined: 'U.S.' -> 'us').
  v2 checks: name token n-grams, rare single name tokens (rarity = document frequency over the whole CRSP universe
  with its CUSIP-linked FactSet names), concatenated tokens, portmanteaus of name-token prefixes, initials of name
  tokens and tickers (upper case, or title case when the word never occurs in lower case in the scanned articles).
  v3 fuzzy checks: edit distance <= 1 (<= 2 for >= 9 letters) to a distinctive name token ('fuzzy_token'); a
  distinctive name token, ticker or acronym inside a longer token ('embedded'); rapidfuzz partial_ratio >= 90 of a
  multi-token name core against the article ('fuzzy_phrase'); one-token names that are English words in Title or
  UPPER case ('dict_single'); RIC-like strings whose root is a masked firm's ticker or RIC root, or that occur in the
  article's archive rics field ('ric_residual'). An adjacency report lists the words right before and after every
  role token for the reader's review.
Hits are candidates for adjudication, not masking rules; nothing here changes an input. Reads no label or return.
"""
import csv
import datetime as dt
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

LAST_DAY = '2022-12-31'
LAST_MONTH = 202212
TOKEN_DF_MAX = 10       # a single name token is distinctive when at most 10 CRSP firms carry it in any name
NGRAM_DF_MAX = 25       # a name n-gram is distinctive when at most 25 CRSP firms carry it
PHRASE_SCORE = 90       # rapidfuzz partial_ratio cutoff for the fuzzy phrase check
CONNECTORS = {'of', 'the', 'and', 'de', 'du', 'des', 'la', 'le', 'del'}
LEGALISH = {'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'cos', 'companies', 'ltd', 'limited',
            'plc', 'holdings', 'holding', 'hldgs', 'hldg', 'group', 'grp', 'nv', 'sa', 'ag', 'se', 'llc', 'lp',
            'trust', 'tr', 'new', 'old', 'adr', 'ads', 'cl', 'intl', 'international', 'partners', 'enterprises',
            'industries', 'technologies', 'technology', 'systems', 'bancorp', 'sab', 'cv', 'kgaa', 'spa', 'asa',
            'ab', 'as', 'oyj', 'pbc', 'na', 'bv', 'reit', 'fund', 'etf', 'class', 'shares', 'common', 'stock',
            'ordinary', 'preferred', 'units', 'unit', 'warrants', 'rights', 'acquisition', 'corp.', 'llp', 'del',
            'de', 'md', 'pa', 'nj', 'ny', 'oh', 'va', 'mn', 'tx', 'ca', 'fl', 'ma', 'ind', 'que', 'ont', 'alta', 'cda'}
SCAN_STOP = {'US', 'USA', 'UK', 'EU', 'UN', 'CEO', 'CFO', 'COO', 'CTO', 'GDP', 'IPO', 'FDA', 'SEC', 'FED', 'FTC',
             'DOJ', 'EPA', 'AI', 'IT', 'TV', 'PC', 'EV', 'EVS', 'AM', 'PM', 'ET', 'EST', 'EDT', 'GMT', 'UTC', 'EPS',
             'ESG', 'LNG', 'OPEC', 'ETF', 'IMF', 'WHO', 'NATO', 'NYSE', 'HK', 'EUR', 'USD', 'GBP', 'JPY', 'CNY',
             'BRIEF', 'UPDATE', 'RPT', 'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'YOY', 'CPI', 'PPI', 'PMI', 'API', 'COVID',
             'NEWS', 'AGENCY', 'URL', 'EMAIL', 'DATE', 'YEAR', 'FOCAL', 'LINK', 'SUPPLIER', 'CUSTOMER', 'INC',
             'CORP', 'CO', 'LTD', 'LLC', 'LP', 'PLC', 'THE', 'AND', 'FOR', 'ARE', 'ALL', 'ONE', 'NEW', 'NOW', 'CAN',
             'HAS', 'WAS', 'NOT', 'BUT', 'ITS', 'OUT', 'SAYS', 'SEES', 'NET', 'PER', 'PCT', 'MLN', 'BLN', 'SHR',
             'QTRLY', 'GAAP', 'NON', 'ADJ', 'OTC', 'TSX', 'ASX', 'NASDAQ', 'AMEX', 'JAN', 'FEB', 'MAR', 'APR',
             'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'SEPT', 'OCT', 'NOV', 'DEC', 'MON', 'TUE', 'WED', 'THU', 'FRI',
             'SAT', 'SUN', 'II', 'III', 'IV', 'NA', 'SA', 'AG', 'NV', 'SE', 'DE', 'OK', 'NO', 'SO', 'GO', 'DO', 'BE',
             'AT', 'BY', 'IN', 'OF', 'OR', 'TO', 'UP', 'IS', 'AS', 'AN', 'WE', 'ON', 'IF', 'MY', 'HE', 'ME', 'A', 'I'}
_ROLE = re.compile(r'\[[A-Z0-9_]+\]')
_TOK = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*|&")
_RICLIKE = re.compile(r'(?<![A-Za-z0-9.])([A-Z0-9]{1,8}(?:_?[a-z]{1,2})?)\.([A-Z]{1,3}[a-z]?)(?![A-Za-z0-9.])|'
                      r'<\s*([^<>\s]{1,20})\s*>')
_SUFFIXES = (('ies', 'y'), ('es', ''), ('s', ''), ('ed', ''), ('ed', 'e'), ('ing', ''), ('ing', 'e'), ('ers', ''),
             ('er', ''), ('ly', ''))
# v5 (TD3C4): code, glued-token and house-number patterns of the scanner
_CODE = re.compile(r'(?<![A-Za-z0-9])([A-Za-z]{2,8})[-–]?\d{1,8}[A-Za-z]{0,3}(?:[-–/]\d{1,8}[A-Za-z]{0,3})*'
                   r'(?![A-Za-z0-9])(?![.,]\d)')
_GLUED_BEFORE = re.compile(r'[^\W_]+(?P<tok>\[[A-Z0-9_]+\])')
_GLUED_AFTER = re.compile(r'(?P<tok>\[[A-Z0-9_]+\])[^\W_]+')
_HOUSE = re.compile(r'(?<![A-Za-z0-9.,$#\-])\d{1,6}[A-Za-z]?[ \t]+(?P<tok>\[[A-Z0-9_]+\])')
# v6: legal forms dropped from the end of a name before its unit initials are formed
_LEGAL_FORMS = frozenset('inc incorporated corp corporation co company cos ltd limited plc llc lp nv sa ag se new old '
                         'del de cl adr ads the'.split())
# v6 (TD3C5 D1): the scanner's own corporate-unit words and legal suffixes (lower case)
_SCAN_UNITS = frozenset('''ventures venture capital holdings holding partners partner investments investment midstream
energy financial technologies technology solutions solution labs lab bio pharma pharmaceuticals therapeutics properties
property realty services service systems system group international global industries industry fund funds trust bank
securities advisors management resources networks media ventures inc corp co llc ltd plc lp'''.split())


def _norm(word):
    return re.sub(r"['’]", '', word).lower()


def fold(text):
    """Accents removed (NFKD, combining marks dropped): 'ZELWÏCK' -> 'ZELWICK'."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text or '') if not unicodedata.combining(c))


def english_word(w, english):
    w = w.lower()
    return w in english or any(w.endswith(a) and len(w) - len(a) >= 3 and w[:-len(a)] + b in english
                               for a, b in _SUFFIXES)


def name_tokens(name):
    """Normalized token list of a firm name (connectors kept; dropped later)."""
    raw = [m.group(0) for m in _TOK.finditer(fold(name))]
    out, run = [], []
    for w in raw:
        if w == '&':
            if run:
                out.append(''.join(run).lower())
                run = []
            out.append('and')
        elif len(w) == 1 and w.isalnum():
            run.append(w)
        else:
            if run:
                out.append(''.join(run).lower())
                run = []
            out.append(_norm(w))
    if run:
        out.append(''.join(run).lower())
    return out


def text_tokens(text):
    """Article tokens: dicts with norm, alt (possessive stripped), orig, start, end. Role tokens become breaks (None).
    Runs of single capital letters or digits separated only by '.', space, '/' or '-' are joined."""
    text = _ROLE.sub('  ', fold(text))
    toks = []
    for m in re.finditer("[A-Za-z0-9]+(?:['’][A-Za-z]+)*|&|", text):
        w = m.group(0)
        if w == '':
            toks.append(None)
            continue
        toks.append({'orig': w, 'start': m.start(), 'end': m.end(),
                     'norm': 'and' if w == '&' else _norm(w),
                     'alt': _norm(re.sub(r"['’][sS]$", '', w)) if re.search(r"['’][sS]$", w) else None,
                     'poss': bool(re.search(r"['’][sS]$", w))})
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t is not None and len(t['orig']) == 1 and (t['orig'].isupper() or t['orig'].isdigit()):
            j = i
            while j + 1 < len(toks) and toks[j + 1] is not None and len(toks[j + 1]['orig']) == 1 and \
                    (toks[j + 1]['orig'].isupper() or toks[j + 1]['orig'].isdigit()) and \
                    re.fullmatch(r'[.\s/\-]*', text[toks[j]['end']:toks[j + 1]['start']]):
                j += 1
            if j > i:
                orig = text[t['start']:toks[j]['end']]
                out.append({'orig': orig, 'start': t['start'], 'end': toks[j]['end'],
                            'norm': ''.join(x['orig'] for x in toks[i:j + 1]).lower(), 'alt': None, 'poss': False})
                i = j + 1
                continue
        out.append(t)
        i += 1
    return out, text


def _drop_connectors(seq):
    return [x for x in seq if x is None or x['norm'] not in CONNECTORS]


def ngrams(tokens, nmax=4):
    toks = [t for t in tokens if t not in CONNECTORS]
    out = set()
    for n in range(2, nmax + 1):
        for i in range(len(toks) - n + 1):
            out.add(' '.join(toks[i:i + n]))
    return out


def load_crsp(paths):
    """{permno: {'names': set, 'tickers': set, 'cusips': set}} from every row with namedt <= 2022-12-31."""
    firms = defaultdict(lambda: {'names': set(), 'tickers': set(), 'cusips': set()})
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                day = (r.get('namedt') or '')[:10]
                if not day or day > LAST_DAY:
                    continue
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                if r.get('comnam'):
                    firms[p]['names'].add(r['comnam'].strip())
                if r.get('ticker'):
                    firms[p]['tickers'].add(r['ticker'].strip())
                for c in (r.get('ncusip'), r.get('cusip')):
                    if c and c.strip():
                        firms[p]['cusips'].add(c.strip()[:8])
    return dict(firms)


def load_factset_by_cusip(path, cusip_to_permnos):
    """{permno: set of FactSet names} by 8-digit CUSIP over the whole CRSP universe (document frequencies only)."""
    out = defaultdict(set)
    if path is None or not Path(path).exists():
        return dict(out), False
    pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=65536)
    value_set = pa.array(sorted(cusip_to_permnos), type=pa.string())
    seal = pa.scalar(dt.date(2022, 12, 31), type=pa.date32())
    for b in pf.iter_batches(batch_size=65536, columns=['cusip', 'name', 'start_'], use_threads=False):
        keep = pc.and_(pc.less_equal(b.column('start_'), seal), pc.is_in(b.column('cusip'), value_set=value_set))
        t = b.filter(keep)
        for c, n in zip(t.column('cusip').to_pylist(), t.column('name').to_pylist()):
            for p in cusip_to_permnos.get(c, ()):
                out[p].add(n)
    return dict(out), True


def load_factset_by_issuer(path, cusip6_by_permno):
    """v3: FactSet ids whose rows (start_ <= 2022-12-31) carry a firm's issuer CUSIP6, then every name and ticker of
    those ids (start_ <= 2022-12-31). Returns ({permno: names}, {permno: tickers})."""
    names, tickers = defaultdict(set), defaultdict(set)
    if path is None or not Path(path).exists() or not cusip6_by_permno:
        return dict(names), dict(tickers)
    by6 = defaultdict(set)
    for p, cs in cusip6_by_permno.items():
        for c in cs:
            by6[c].add(p)
    if not by6:
        return dict(names), dict(tickers)
    pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=65536)
    seal = pa.scalar(dt.date(2022, 12, 31), type=pa.date32())
    v6 = pa.array(sorted(by6), type=pa.string())
    ids = defaultdict(set)
    for b in pf.iter_batches(batch_size=65536, columns=['id', 'cusip', 'start_'], use_threads=False):
        b = b.filter(pc.less_equal(b.column('start_'), seal))
        t = b.filter(pc.is_in(pc.utf8_slice_codeunits(b.column('cusip'), 0, 6), value_set=v6))
        for i, c in zip(t.column('id').to_pylist(), t.column('cusip').to_pylist()):
            ids[i] |= by6.get(c[:6], set())
    if not ids:
        return dict(names), dict(tickers)
    vid = pa.array(sorted(ids), type=pa.string())
    for b in pf.iter_batches(batch_size=65536, columns=['id', 'name', 'ticker', 'start_'], use_threads=False):
        b = b.filter(pc.less_equal(b.column('start_'), seal))
        t = b.filter(pc.is_in(b.column('id'), value_set=vid))
        for i, n, tk in zip(*(t.column(k).to_pylist() for k in ('id', 'name', 'ticker'))):
            for p in ids[i]:
                if n and n.strip():
                    names[p].add(n.strip())
                tk = (tk or '').strip()
                if re.fullmatch(r'[A-Z]{2,5}', tk):
                    tickers[p].add(tk)
    return dict(names), dict(tickers)


def document_frequencies(crsp, factset):
    """Token and n-gram document frequency over every CRSP firm (CRSP names plus its CUSIP-linked FactSet names)."""
    df_tok, df_ng = Counter(), Counter()
    for p, f in crsp.items():
        toks, grams = set(), set()
        for n in f['names'] | factset.get(p, set()):
            t = name_tokens(n)
            toks.update(x for x in t if x not in CONNECTORS)
            grams |= ngrams(t)
        df_tok.update(toks)
        df_ng.update(grams)
    return df_tok, df_ng


def ric_roots(ricmap, permnos):
    out = defaultdict(set)
    for ric, (p, m) in ricmap.items():
        if int(p) in permnos and int(m) <= LAST_MONTH:
            root = ric.split('.')[0]
            if root:
                out[int(p)].add(root)
    return out


def _core(tokens):
    core = [x for x in tokens if x not in CONNECTORS]
    while core and core[-1] in LEGALISH:
        core.pop()
    while core and core[0] == 'the':
        core.pop(0)
    return core


class FirmTerms:
    """Distinctive terms of one firm for the scan."""

    def __init__(self, names, tickers, df_tok, df_ng, english=frozenset()):
        self.ngrams, self.tokens, self.concat, self.portmanteaus, self.acronyms = {}, {}, {}, {}, {}
        self.fuzzy, self.embed, self.phrases, self.dict_single = {}, {}, {}, {}
        for name in sorted(names):
            t = name_tokens(name)
            for g in ngrams(t):
                if df_ng.get(g, 0) <= NGRAM_DF_MAX:
                    self.ngrams.setdefault(g, name)
            core = [x for x in t if x not in CONNECTORS]
            for x in core:
                if len(x) >= 4 and x not in LEGALISH and not x.isdigit() and df_tok.get(x, 0) <= TOKEN_DF_MAX:
                    self.tokens.setdefault(x, name)
                if x.isalpha() and x not in LEGALISH and not english_word(x, english):
                    if len(x) >= 5:
                        self.fuzzy.setdefault(x, name)
                    if len(x) >= 4:
                        self.embed.setdefault(x, name)
            for pos, (a, b) in enumerate(zip(core, core[1:])):
                if df_ng.get(a + ' ' + b, 0) > NGRAM_DF_MAX:
                    continue
                self.concat.setdefault(a + b, name)
                if pos < 2 and a.isalpha() and b.isalpha():
                    for i in range(3, len(a) + 1):
                        for j in range(2, len(b) + 1):
                            w = a[:i] + b[:j]
                            if len(w) >= 5 and w not in core:
                                self.portmanteaus.setdefault(w, f'{a}+{b} ({name})')
            sig = list(core)
            while sig and sig[-1] in LEGALISH:     # legal words dropped from the end only ('<name> bank ...')
                sig.pop()
            for k in range(2, len(sig) + 1):
                acro = ''.join(x[0] for x in sig[:k]).upper()
                if acro not in SCAN_STOP:
                    self.acronyms.setdefault(acro, name)
            c = _core(t)
            if len(c) >= 2 and len(' '.join(c)) >= 10:
                self.phrases.setdefault(' '.join(c), name)
            if len(c) == 1 and len(c[0]) >= 3 and c[0].isalpha() and english_word(c[0], english):
                self.dict_single.setdefault(c[0], name)
        for w in list(self.portmanteaus):
            if w in self.tokens or w in self.concat:
                del self.portmanteaus[w]
        self.tickers = {x for x in tickers if x and x.upper() not in SCAN_STOP and len(x) >= 2}
        for tk in self.tickers:
            if len(tk) >= 3 and tk.isalpha() and not english_word(tk, english):
                self.embed.setdefault(tk.lower(), 'ticker')
        for a in self.acronyms:
            if len(a) >= 3 and a.isalpha() and not english_word(a, english):
                self.embed.setdefault(a.lower(), 'initials')
        self.ticker_roots = {x.upper() for x in tickers if x}
        # v5 (TD3C4 D2): code prefixes and near prefixes, derived here independently of the masker
        self.code_terms, self.near_terms = {}, {}
        self.unit_terms = set()                  # v6 (TD3C5 D1): tickers and name initials
        for tk in {x.upper() for x in tickers if x and x.isalpha() and len(x) >= 2}:
            self.code_terms.setdefault(tk, 'ticker')
            self.unit_terms.add(tk)
            for r in range(2, len(tk)):
                for rest in combinations(tk[1:], r - 1):
                    self.code_terms.setdefault(tk[0] + ''.join(rest), 'ticker_subsequence')
            for k in (1, 2):
                if len(tk) - k >= 2:
                    self.near_terms.setdefault(tk[:-k], 'ticker ' + tk)
        for name in sorted(names):
            core = [x for x in name_tokens(name) if x not in CONNECTORS]
            while core and core[-1] in LEGALISH:
                core.pop()
            if core and core[0] == 'the':
                core = core[1:]
            camel = [w.lower() for w in re.findall(r'[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+|\d+',
                                                    re.sub(r"['’]", '', fold(name)))]
            camel = [x for x in camel if x not in CONNECTORS]
            while camel and camel[-1] in LEGALISH:
                camel.pop()
            for seq_ in (core, camel):
                for k in range(2, len(seq_) + 1):
                    ini = ''.join(x[0] for x in seq_[:k]).upper()
                    if ini.isalpha():
                        self.code_terms.setdefault(ini, 'initials')
                        self.unit_terms.add(ini)
            if core and core[0].isalpha() and len(core[0]) >= 3:
                first = core[0].upper()
                for k in range(2, len(first) + 1):
                    self.code_terms.setdefault(first[:k], 'leading_letters')
                for k in (1, 2):
                    if len(first) - k >= 2:
                        self.near_terms.setdefault(first[:-k], 'first token ' + first)
        for k in list(self.near_terms):
            if k in SCAN_STOP or (len(k) >= 3 and english_word(k, english)):
                del self.near_terms[k]
        for k in list(self.code_terms):
            if k in SCAN_STOP:
                del self.code_terms[k]
        # v6: unit initials keep descriptive name words (industries, systems, ...) and drop only legal forms
        for name in names:
            for seq_ in ([x for x in name_tokens(name) if x not in CONNECTORS],
                         [w.lower() for w in re.findall(r'[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+|\d+',
                                                        re.sub(r"['’]", '', fold(name))) if w.lower() not in CONNECTORS]):
                while seq_ and seq_[-1] in _LEGAL_FORMS:
                    seq_.pop()
                for k in range(2, len(seq_) + 1):
                    ini = ''.join(x[0] for x in seq_[:k]).upper()
                    if ini.isalpha():
                        self.unit_terms.add(ini)
        self.unit_terms -= SCAN_STOP


def _not_lower(s):
    return s != s.lower()


def _embedded_at(word, term):
    """Position check of a term inside a longer token: token start or end, or a lower-to-upper boundary."""
    low = word.lower()
    i = low.find(term)
    while i >= 0:
        j = i + len(term)
        if i == 0 or j == len(word) or (word[i - 1].islower() and word[i].isupper()) or \
                (j < len(word) and word[j - 1].islower() and word[j].isupper()):
            return True
        i = low.find(term, i + 1)
    return False


def scan_article(text, firm_terms, lower_words, english=frozenset(), archive_rics=(), chosen=()):
    """firm_terms: {token: FirmTerms}; chosen: tokens of the firms the article was chosen for. Returns hit dicts (no
    masking)."""
    toks, ntext = text_tokens(text)
    seq = _drop_connectors(toks)
    grams, forms, joined, upper, bare_map = (defaultdict(list) for _ in range(5))
    for i, t in enumerate(seq):
        if t is None:
            continue
        for n in range(2, 5):
            win = seq[i:i + n]
            if len(win) < n or any(w is None for w in win):
                break
            grams[' '.join(w['norm'] for w in win)].append((win[0]['start'], win[-1]['end']))
            if n <= 3:
                joined[''.join(w['norm'] for w in win)].append(
                    (win[0]['start'], win[-1]['end'], any(_not_lower(w['orig']) for w in win)))
        for f in {t['norm']} | ({t['alt']} if t['alt'] else set()):
            forms[f].append(i)
        bare = re.sub(r"['’][sS]$", '', t['orig'])
        letters = re.sub(r'[.\s/\-]', '', bare)
        if bare.isupper() and len(letters) >= 2:
            upper[letters].append(i)
        bare_map[bare].append(i)
    hits = []

    def add(token, check, term, a, b, source):
        hits.append({'token': token, 'check': check, 'term': term, 'matched': ntext[a:b], 'start': a, 'end': b,
                     'source': source,
                     'context': ntext[max(0, a - 70):b + 70].replace('', '[TOKEN]').replace('\n', ' ')})
    words = [(i, t) for i, t in enumerate(seq) if t is not None and re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)?",
                                                                                  t['orig']) and _not_lower(t['orig'])]
    flat = ' '.join('|' if t is None else t['norm'] for t in seq)
    for token, ft in firm_terms.items():
        for g, name in ft.ngrams.items():
            for a, b in grams.get(g, ()):
                add(token, 'ngram', g, a, b, name)
        for f, name in ft.tokens.items():
            for i in forms.get(f, ()):
                if _not_lower(seq[i]['orig']):
                    add(token, 'rare_token', f, seq[i]['start'], seq[i]['end'], name)
        for f, name in ft.concat.items():
            if f in ft.tokens:
                continue
            for i in forms.get(f, ()):
                if _not_lower(seq[i]['orig']):
                    add(token, 'concatenated_name_tokens', f, seq[i]['start'], seq[i]['end'], name)
        for f in set(ft.tokens) | set(ft.concat):
            for a, b, nl in joined.get(f, ()):
                if nl:
                    add(token, 'joined_text_tokens', f, a, b, ft.tokens.get(f) or ft.concat.get(f))
        for f, name in ft.portmanteaus.items():
            if f in lower_words:     # an ordinary word (seen in lower case in the scanned articles), not a coinage
                continue
            for i in forms.get(f, ()):
                if _not_lower(seq[i]['orig']):
                    add(token, 'portmanteau', f, seq[i]['start'], seq[i]['end'], name)
        for acro, name in ft.acronyms.items():
            for i in upper.get(acro, ()):
                t = seq[i]
                nxt = seq[i + 1] if i + 1 < len(seq) else None
                if len(acro) >= 3 or t['poss'] or (nxt is not None and nxt['norm'] in LEGALISH):
                    add(token, 'acronym', acro, t['start'], t['end'], name)
        for tk in ft.tickers:
            for i in bare_map.get(tk, ()):
                if tk.isupper():
                    add(token, 'ticker_upper', tk, seq[i]['start'], seq[i]['end'], 'ticker')
            title = tk[:1].upper() + tk[1:].lower()
            if len(tk) >= 3 and tk.isalpha() and title != tk and tk.lower() not in lower_words:
                for i in bare_map.get(title, ()):
                    add(token, 'ticker_title', tk, seq[i]['start'], seq[i]['end'], 'ticker')
        # v3 fuzzy checks
        for i, t in words:
            w = re.sub(r"['’][sS]$", '', t['orig'])
            lw = w.lower()
            if len(lw) >= 5 and not english_word(lw, english):
                for f, name in ft.fuzzy.items():
                    if lw == f or abs(len(lw) - len(f)) > 2:
                        continue
                    limit = 2 if len(f) >= 9 else 1
                    if Levenshtein.distance(lw, f, score_cutoff=limit) <= limit:
                        add(token, 'fuzzy_token', f, t['start'], t['end'], name)
                        break
                for f, name in ft.embed.items():
                    if len(f) < len(lw) and f in lw and _embedded_at(w, f):
                        add(token, 'embedded', f, t['start'], t['end'], name)
                        break
            if lw in ft.dict_single:
                add(token, 'dict_single', lw, t['start'], t['end'], ft.dict_single[lw])
        for ph, name in ft.phrases.items():
            al = fuzz.partial_ratio_alignment(ph, flat, score_cutoff=PHRASE_SCORE)
            if al is not None:
                sub = flat[al.dest_start:al.dest_end]
                hits.append({'token': token, 'check': 'fuzzy_phrase', 'term': ph, 'matched': sub,
                             'start': al.dest_start, 'end': al.dest_end, 'source': name,
                             'context': f'score={al.score:.1f}: ' + flat[max(0, al.dest_start - 60):al.dest_end + 60]})
        for m in _RICLIKE.finditer(ntext):
            ric = m.group(0).strip('<> ')
            root = (m.group(1) or ric.split('.')[0]).upper().rstrip('_')
            if root in ft.ticker_roots or \
                    any(len(r) >= 3 and root.startswith(r) and len(root) - len(r) <= 3 for r in ft.ticker_roots):
                add(token, 'ric_residual', ric, m.start(), m.end(), 'ric root of this firm')
            elif ric in archive_rics and token in chosen:
                add(token, 'ric_residual', ric, m.start(), m.end(), 'ric in the archive rics field of the article')
    # v5 (TD3C4 D2): firm codes and near prefixes
    for m in _CODE.finditer(ntext):
        pre = m.group(1)
        if not (pre.isupper() or (pre[0].isupper() and pre[1:].islower())):
            continue
        for token, ft in firm_terms.items():
            rule = ft.code_terms.get(pre.upper())
            if rule:
                add(token, 'firm_code', pre.upper(), m.start(), m.end(), rule)
    for i, t in enumerate(seq):
        if t is None or not re.fullmatch(r'[A-Z]{2,6}', t['orig']) or i + 1 >= len(seq) or seq[i + 1] is None:
            continue
        nxt = seq[i + 1]['orig']
        if not (nxt[:1].isupper() and nxt.isalpha() and len(nxt) >= 3) or seq[i + 1]['norm'] in CONNECTORS:
            continue
        for token, ft in firm_terms.items():
            src = ft.near_terms.get(t['orig'])
            if src:
                add(token, 'near_prefix', t['orig'], t['start'], seq[i + 1]['end'], src)
    # v6 (TD3C5 D1): a ticker or name initialism followed by a corporate-unit word or legal suffix
    for i, t in enumerate(seq):
        if t is None or i + 1 >= len(seq) or seq[i + 1] is None or \
                not re.fullmatch(r'[A-Z]{2,6}|[A-Z][a-z]{1,5}', t['orig']):
            continue
        if seq[i + 1]['norm'] not in _SCAN_UNITS or seq[i + 1]['orig'] == seq[i + 1]['orig'].lower():
            continue
        for token, ft in firm_terms.items():
            if t['orig'].upper() in ft.unit_terms:
                add(token, 'initialism_unit', t['orig'].upper(), t['start'], seq[i + 1]['end'], 'ticker or initials')
    # v5 (TD3C4 D1): glued tokens and house numbers before a token (positions in the masked text itself)
    for rx, check in ((_GLUED_BEFORE, 'glued_token'), (_GLUED_AFTER, 'glued_token'), (_HOUSE, 'house_number')):
        for m in rx.finditer(text):
            tok = m.group('tok')
            if tok in firm_terms:
                hits.append({'token': tok, 'check': check, 'term': m.group(0), 'matched': m.group(0),
                             'start': 10 ** 6 + m.start(), 'end': 10 ** 6 + m.end(), 'source': 'masked text',
                             'context': text[max(0, m.start() - 70):m.end() + 70].replace('\n', ' ')})
    # v3 (TD3C2 manual read): a visible name word of a firm in the same text as that firm's role token can let the
    # reader read the masked word back ("Jane Zelwick sued ... [X] said she"); each such pair is reported for review
    present = {m.group(0) for m in _ROLE.finditer(text)}
    for h in list(hits):
        if h['check'] in ('rare_token', 'dict_single', 'fuzzy_token', 'ngram') and h['token'] in present:
            hits.append({**h, 'check': 'name_echo', 'source': h['check'] + ': ' + str(h['source'])})
    return hits


def adjacency(text):
    """Words right before and after every role token of a text: [(token, before, after)] ('' at a boundary)."""
    out = []
    for m in _ROLE.finditer(text):
        before = re.search(r"([A-Za-z0-9&'’.\-]+)[\s,;:*]*$", _ROLE.sub(' ', text[max(0, m.start() - 40):m.start()]))
        after = re.match(r"[\s,;:*]*([A-Za-z0-9&'’.\-]+)", _ROLE.sub(' ', text[m.end():m.end() + 40]))
        out.append((m.group(0), before.group(1) if before else '', after.group(1) if after else ''))
    return out


def hit_id(rank, alt_id, field, hit):
    return hashlib.sha256(f"{rank}|{alt_id}|{field}|{hit['token']}|{hit['check']}|{hit['term']}|{hit['start']}"
                          .encode()).hexdigest()[:20]


def lower_vocabulary(texts):
    words = set()
    for t in texts:
        words.update(m.group(0) for m in re.finditer(r'\b[a-z]{3,}\b', _ROLE.sub(' ', t)))
    return words


def scan_build(metas, crsp_paths, factset_path, ricmap, english=frozenset()):
    """Scan every masked article of a build. Returns (hits, summary, adjacency rows)."""
    crsp = load_crsp(crsp_paths)
    cusip_to_permnos, by6 = defaultdict(set), defaultdict(set)
    for p, f in crsp.items():
        for c in f['cusips']:
            cusip_to_permnos[c].add(p)
            if len(c) >= 6:
                by6[c[:6]].add(p)
    factset_df, factset_used = load_factset_by_cusip(factset_path, cusip_to_permnos)
    df_tok, df_ng = document_frequencies(crsp, factset_df)
    masked = {a['permno'] for m in metas for a in m['aliases'].values()}
    siblings = {p: sorted({q for c in crsp.get(p, {}).get('cusips', ()) if len(c) >= 6 for q in by6[c[:6]]} - {p})
                for p in masked}
    c6 = {p: {c[:6] for c in crsp.get(p, {}).get('cusips', ()) if len(c) >= 6} for p in masked}
    fs_names, fs_tickers = load_factset_by_issuer(factset_path, c6)
    rics = ric_roots(ricmap, masked | {q for v in siblings.values() for q in v})
    texts = [x['headline'] + '\n' + x['body'] for m in metas for x in m['articles']]
    lower_words = lower_vocabulary(texts)
    cache, hits, adj = {}, [], []
    for m in metas:
        role = {}
        for tok, a in m['aliases'].items():
            p = a['permno']
            if p not in cache:
                f = crsp.get(p, {'names': set(), 'tickers': set()})
                names = set(f['names']) | fs_names.get(p, set())
                tickers = set(f['tickers']) | fs_tickers.get(p, set()) | rics.get(p, set())
                for q in siblings.get(p, ()):
                    names |= crsp.get(q, {}).get('names', set())
                    tickers |= crsp.get(q, {}).get('tickers', set()) | rics.get(q, set())
                cache[p] = FirmTerms(names, tickers, df_tok, df_ng, english)
            role[tok] = cache[p]
        for x in m['articles']:
            for field in ('headline', 'body'):
                for h in scan_article(x[field], role, lower_words, english, set(x.get('archive_rics', ())),
                                      set(x.get('tokens', ()))):
                    cls = 'focal' if h['token'] == '[FOCAL]' else 'link' if h['token'].startswith('[LINK_') else \
                        'shown'
                    h.update(rank=m['sample_rank'], key=m['key'], alt_id=x['alt_id'], field=field, firm_class=cls,
                             permno=m['aliases'][h['token']]['permno'])
                    h['hit_id'] = hit_id(m['sample_rank'], x['alt_id'], field, h)
                    hits.append(h)
                for tok, before, after in adjacency(x[field]):
                    if tok in m['aliases']:
                        cls = 'focal' if tok == '[FOCAL]' else 'link' if tok.startswith('[LINK_') else 'shown'
                        adj.append({'rank': m['sample_rank'], 'alt_id': x['alt_id'], 'field': field, 'token': tok,
                                    'firm_class': cls, 'before': before, 'after': after})
    summary = {'articles_scanned': sum(len(m['articles']) for m in metas), 'inputs_scanned': len(metas),
               'masked_firms': len(masked), 'firms_with_crsp_names': sum(1 for p in masked if crsp.get(p)),
               'firms_with_factset_names': sum(1 for p in masked if fs_names.get(p)), 'factset_used': factset_used,
               'firms_with_issuer_siblings': sum(1 for p in masked if siblings.get(p)),
               'token_df_max': TOKEN_DF_MAX, 'ngram_df_max': NGRAM_DF_MAX, 'phrase_score': PHRASE_SCORE,
               'hits': len(hits), 'hits_by_class': dict(Counter(h['firm_class'] for h in hits)),
               'hits_by_check': dict(Counter(h['check'] for h in hits)),
               'adjacency_rows': len(adj),
               'adjacency_word_types': len({(r['before'].upper(), r['after'].upper()) for r in adj})}
    return hits, summary, adj
