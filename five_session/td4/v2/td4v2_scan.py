"""Independent v2 scanner: one deterministic detector per masking-review class, run over the texts actually sent.

It does not use the masker's alias lists: the terms come from the CRSP name files (namedt <= the period seal), the
TD4 compact metadata (token -> PERMNO) and the frozen common-word table. Hits are candidates for the reviewer; classes
whose detector is exact (dates, story-field stubs, mis-tag notices, hyphen and model remnants next to a token, RIC
suffixes after a token) withhold the input for all providers alike (td4v2_verify2). Executive names next to a token are
counted only (the disclosed TD3 limitation, M2 section 5). Examples in this file are invented.
"""
import csv
import re
from collections import Counter
from pathlib import Path

import td4_spec as T

# withheld for every arm alike: the exact detectors and the narrow give-back and own-article code detectors
WITHHOLD_CLASSES = ('S10_date', 'S11_story_field', 'S12_mistag_notice', 'S8_hyphen_name_remnant',
                    'S13_model_after_token', 'S14_ric_suffix', 'S7_code_subsequence', 'S9_giveback_collocation',
                    'S9_giveback_index_list', 'S9_giveback_slug', 'S9_giveback_name_remainder')
_ROLE = r'\[(?:FOCAL|LINK_\d+|(?:SUPPLIER|CUSTOMER)(?:_OF_(?:SUPPLIER|CUSTOMER))*_\d+)\]'
_TOKEN = re.compile(_ROLE)
_ARTICLE = re.compile(r'^Article \d+ \| firm: (?P<toks>[^|]+) \| published[^\n]*\nHeadline: (?P<h>[^\n]*)\n'
                      r'Text: (?P<b>[^\n]*)', re.M)
_LEGALW = {'INC', 'CORP', 'CO', 'LTD', 'PLC', 'LLC', 'LP', 'SA', 'AG', 'NV', 'SE', 'HOLDINGS', 'HLDGS', 'GROUP',
           'COMPANY', 'CORPORATION', 'INCORPORATED', 'LIMITED', 'THE', 'OF', 'AND', 'NEW', 'DEL', 'TRUST', 'INTL',
           'INTERNATIONAL', 'CL', 'A', 'B', 'COM', 'ADR', 'SPON', 'HLDG', 'COS', 'BANCORP', 'TR', 'OLD', 'DE'}
_MONTH = (r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|'
          r'Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)')
DATE_RX = (
    ('year4', re.compile(r'(?<![\d.,$])(?:19|20)\d\d(?![\d%])')),
    ('numeric_date', re.compile(r'(?<![\d/.])\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})(?![\d/])')),
    ('month_day', re.compile(r'(?i)\b' + _MONTH + r'\.?,?[ \t]+\d{1,2}(?:st|nd|rd|th)?(?![\d%])(?![.,]\d)')),
    ('day_month', re.compile(r'(?i)\b\d{1,2}(?:st|nd|rd|th)?[ \t]+(?:of[ \t]+)?' + _MONTH + r'\b')),
    ('glued_day', re.compile(r'\b' + _MONTH + r'[ \t]+\d{1,2}[A-Z][a-z]')),
    ('two_digit_year', re.compile("\\bCY\\d{2}\\b|\\bMY\\d{2}\\b|(?<![\\w'’])['’]\\d{2}s?\\b")),
)
_YEAR_UNITS = re.compile(r'\s*(?:%|percent|mln|bln|million|billion|shares|units|employees|people|stores|tons|barrels|'
                         r'bpd|MW|GW|km|miles|feet|points?)\b')
STORY_RX = re.compile(r'STORY_|(?<![A-Za-z])(?:ORY|RY|Y)_(?:NUMBER|DATE|TIME)|(?<![A-Za-z_])(?:IME|ME)[ \t]*:[ \t]*\d|'
                      r'(?<![\d:])\d{1,2}:\d{2}:\d{2}(?![\d:])')
MISTAG_RX = re.compile(r'(?i)\b(?:inadvertently|incorrectly|wrongly|mistakenly|erroneously)\s+(?:tagged|issued|'
                       r'published|sent|attached)\b|\btagged\s+(?:in\s+error|incorrectly|wrongly)\b')
EXEC_RX = re.compile(r'(?:' + _ROLE + r")(?:'s)?[ \t]+(?:CEO|Chief Executive(?: Officer)?|CFO|Chief Financial Officer|"
                     r'President|Chairman|Chairwoman|Chair|founder|co-founder|Founder)[ \t,]+[A-Z][a-z]+(?:[ \t]+[A-Z]'
                     r'[a-z\-]+)+')
GIVEBACK_RX = (
    ('weekday_slot', re.compile(r'\b(?:on|said|signed|announced|from|until|by|last|next|this)[ \t]+(?:' + _ROLE + r')'
                                r'(?=[ \t]*(?:[,.]|evening|morning|night|afternoon))')),
    ('collocation', re.compile(r'(?i)(?:' + _ROLE + r')[ \t]+(?:roundup|and[ \t]+development|thermal[ \t]+units?)\b|'
                               r'\b(?:purchase|offering|closing|exercise|strike)[ \t]+(?:' + _ROLE + r')[ \t]+(?:of|per|at)\b|'
                               r'\bfor[ \t]+(?:' + _ROLE + r')[ \t]+years?\b|\b(?:Logone|Taco|Oval)[ \t]+(?:' + _ROLE + r')|'
                               r'\bRay-Ban[ \t]+(?:' + _ROLE + r')|\bCURRENTLY[ \t]+(?:' + _ROLE + r')[ \t]+IN\b')),
    ('index_list', re.compile(r'(?:Indexes:|indexes)[^\n]{0,20}(?:' + _ROLE + r')[ \t]+(?:up|down)|(?:' + _ROLE + r')[ \t]+'
                              r'(?:up|down)[ \t]+[\d.]+%[ \t]*,[ \t]*(?:S&P|Nasdaq|Dow)')),
    ('slug', re.compile(r'(?:^|UPDATE[ \t]*\d+-)(?:' + _ROLE + r')-(?=[A-Z])')),
    ('name_remainder', re.compile(r'(?:' + _ROLE + r')[ \t]+(?:City|Tree|General|Chemical|66)\b(?=[ \t]+[a-z]|[ \t]*[.,;:])|'
                                  r'\bFamily[ \t]+(?:' + _ROLE + r')')),
)
MODEL_RX = re.compile(r'(?:' + _ROLE + r')[ \t]+(?:\d{3}-\d{1,4}|\d{3}[ \t]?MAX|7[0-8]7s?\b|\d{3}[ \t]+(?:regional[ \t]+)?'
                      r'jets?)')
RIC_SUFFIX_RX = re.compile(r'(?:' + _ROLE + r')\.(?:[A-Z]{1,3}|[A-Z]{1,2}[a-z])(?![A-Za-z0-9])')
_SUBJ = re.compile(r'(?:^|\n|\*[ \t]*|BRIEF-|BUZZ-|\][ \t]*[-–:][ \t]*)$')
_VERB = re.compile(r"(?:[ \t]*:|[ \t]+-[ \t]|['’][sS]\b|[ \t]+[A-Z]{3,}(?:S|ED|ING)\b|[ \t]+(?:said|says|reported|"
                   r'announced|Inc|Corp))')
_CTXR = re.compile(r"(?:['’]s|[ \t]+(?:said|says|reported|announced|shares|Inc|Corp|Co|CEO))\b")
_INDEX = {'NASDAQ', 'DOW', 'RUSSELL', 'FTSE', 'NIKKEI', 'STOXX'}
_INDEX_CTX = re.compile(r'(?:S&P|Nasdaq|NASDAQ|Dow|DOW|Composite|[ \t]+(?:up|down|rose|fell|gained|lost|at|Composite|'
                        r'100|futures|index)\b|%)')
_CODE = re.compile(r'(?<![A-Za-z0-9])([A-Za-z]{2,5})[-–]?(\d{1,6})[A-Za-z]{0,3}(?![A-Za-z0-9])')


def _clean(name):
    name = re.sub(r"['’]", '', name or '')
    name = re.sub(r'[,.;:()"/]', ' ', name)
    return ' '.join(name.upper().replace('&', ' & ').split())


def _alt(words, title=False):
    forms = set(words)
    if title:
        forms |= {w[0] + w[1:].lower() for w in words if len(w) > 1}
    return '|'.join(re.escape(w) for w in sorted(forms, key=len, reverse=True))


def load_names(permnos, last_day):
    """{permno: {'comnams': set, 'tickers': set}} from both CRSP name files (namedt <= last_day)."""
    want = {int(p) for p in permnos}
    out = {p: {'comnams': set(), 'tickers': set()} for p in want}
    for path in T.CRSP_NAME_FILES:
        with Path(path).open(encoding='utf-8-sig', newline='') as fh:
            for r in csv.DictReader(fh):
                d = (r.get('namedt') or '')[:10]
                if not d or d > last_day:
                    continue
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                if p in out:
                    if r.get('comnam'):
                        out[p]['comnams'].add(r['comnam'].strip())
                    if r.get('ticker'):
                        out[p]['tickers'].add(r['ticker'].strip().upper())
    return out


def _eng(w, english, common):
    lw = w.lower()
    return lw in english or lw in common or (lw.endswith('s') and (lw[:-1] in english or lw[:-1] in common))


class Terms:
    """Scanner terms of one firm, derived only from its CRSP names and tickers (compiled once)."""

    def __init__(self, comnams, tickers, english, common):
        self.english, self.common = english, common
        self.tickers = {t for t in tickers if re.fullmatch(r'[A-Z]{2,6}', t)}
        words, firsts, inits, digit, onew = set(), set(), set(), set(), set()
        for c in comnams:
            ws = [w for w in _clean(c).split() if w != '&']
            if ws and ws[0] == 'THE':
                ws = ws[1:]
            core = [w for w in ws if w not in _LEGALW]
            for w in ws:
                if re.search(r'\d', w) and re.search(r'[A-Z]', w) and len(w) >= 2:
                    digit.add(w)
            if not core:
                continue
            firsts.add(core[0])
            if len(core) == 1 and core[0].isalpha() and len(core[0]) >= 6:
                onew.add(core[0])
            for w in core:
                if w.isalpha() and len(w) >= 4:
                    words.add(w)
            letters = [w[0] for w in core if w[:1].isalpha()]
            for k in range(2, min(len(letters), 5) + 1):
                inits.add(''.join(letters[:k]))
        self.words, self.firsts, self.onew = words, firsts, onew
        self.distinct = {w for w in words if not _eng(w, english, common)}
        self.inits = {i for i in inits if i not in self.tickers and not _eng(i, english, common)}
        strong = {t for t in self.tickers if len(t) >= 3 and not _eng(t, english, common)}
        self.rx_tick = re.compile(r'(?<![A-Za-z0-9&.$])(' + _alt(self.tickers) + r')(?![A-Za-z0-9&]|\.[A-Za-z0-9])') \
            if self.tickers else None
        self.strong = strong
        titles = {t[0] + t[1:].lower() for t in self.tickers if len(t) >= 3 and t.lower() not in common}
        self.rx_title = re.compile(r'(?<![A-Za-z0-9.$])(' + _alt(titles) + r')(?![A-Za-z0-9])') if titles else None
        self.rx_init = re.compile(r"(?<![A-Za-z0-9&.])(" + _alt(self.inits) + r")(?:['’]s)?(?![A-Za-z0-9&]|"
                                  r'\.[A-Za-z0-9])') if self.inits else None
        self.rx_dist = re.compile(r"(?i)(?<![A-Za-z0-9])(" + _alt(self.distinct) + r")(?:['’]s)?(?![A-Za-z0-9])") \
            if self.distinct else None
        dw = words - self.distinct
        self.rx_ctx = re.compile(r'(?<![A-Za-z0-9])(' + _alt(dw, title=True) + r')(?![A-Za-z0-9])') if dw else None
        cw = {w for w in self.distinct if len(w) >= 4} | {t for t in strong}
        self.rx_cpd = re.compile(r'(?i)(?<![A-Za-z0-9])(?:[A-Za-z]{2,}(?:' + _alt(cw) + r')|(?:' + _alt(cw) +
                                 r')[A-Za-z]{2,})(?![A-Za-z0-9])') if cw else None
        self.rx_digit = re.compile(r'(?i)(?<![A-Za-z0-9])(' + _alt(digit) + r')(?![A-Za-z0-9])') if digit else None
        sp = set()
        for w in onew:
            for i in range(2, len(w) - 2):
                sp.add(re.escape(w[:i]) + r'[ \t\-]+' + re.escape(w[i:]))
        self.rx_split = re.compile(r'(?i)(?<![A-Za-z0-9])(?:' + '|'.join(sorted(sp)) + r')(?![A-Za-z0-9])') if sp \
            else None
        self.hw = sorted(w for w in words if len(w) >= 3)

    def subseq_ok(self, prefix):
        p = prefix.upper()
        for src in list(self.tickers) + list(self.firsts):
            if len(src) >= 3 and p[0] == src[0] and p != src:
                it = iter(src[1:])
                if all(c in it for c in p[1:]):
                    return True
        return False


def scan_text(text, tok_permno, terms, art_tokens=()):
    """Hits [(class, term, context)] of one article field. tok_permno: token -> PERMNO of the input; art_tokens: the
    tokens the article was chosen for."""
    hits = []

    def hit(cls, term, m):
        a, b = m.start(), m.end()
        hits.append((cls, term, text[max(0, a - 50):b + 50].replace('\n', ' ')))
    present = set(_TOKEN.findall(text))
    for tok, p in tok_permno.items():
        t = terms.get(p)
        if t is None:
            continue
        own = tok in art_tokens
        if t.rx_tick is not None:
            for m in t.rx_tick.finditer(text):
                tk = m.group(1)
                if re.match(r'\.?[ \t]*\d', text[m.end():]):
                    continue
                if tk in t.strong or (_SUBJ.search(text[:m.start()]) and _VERB.match(text[m.end():])):
                    hit('S1_ticker', tk, m)
        if own and t.rx_title is not None:
            for m in t.rx_title.finditer(text):
                hit('S1_ticker_title', m.group(1), m)
        if t.rx_init is not None and (tok in present or own):
            for m in t.rx_init.finditer(text):
                if not re.match(r'\.?[ \t]*\d', text[m.end():]):
                    hit('S2_initialism', m.group(1), m)
        if t.rx_dist is not None:
            for m in t.rx_dist.finditer(text):
                w = m.group(1).upper()
                if w in _INDEX and not own and (_INDEX_CTX.search(text[max(0, m.start() - 30):m.start()]) or
                                                _INDEX_CTX.match(text[m.end():m.end() + 30])):
                    continue                     # an index name in a market list, outside the firm's own article
                hit('S3_name_word', w, m)
        if t.rx_ctx is not None:
            for m in t.rx_ctx.finditer(text):
                if _CTXR.match(text[m.end():]) and not re.search(r'[A-Z][A-Za-z]+[ \t]+$', text[:m.start()]):
                    hit('S3_name_word_company_context', m.group(1).upper(), m)
        if t.rx_cpd is not None:
            for m in t.rx_cpd.finditer(text):
                if not _eng(m.group(0), t.english, t.common):
                    hit('S4_compound', m.group(0), m)
        if t.rx_digit is not None:
            for m in t.rx_digit.finditer(text):
                hit('S5_digit_name', m.group(1).upper(), m)
        if t.rx_split is not None:
            for m in t.rx_split.finditer(text):
                hit('S6_split_name', m.group(0), m)
        if own:
            for m in _CODE.finditer(text):
                pre = m.group(1)
                if (pre.isupper() or (pre[0].isupper() and pre[1:].islower())) and t.subseq_ok(pre) and \
                        not _eng(pre, t.english, t.common):
                    hit('S7_code_subsequence', pre, m)
        if tok in present and t.hw:
            alt = '|'.join(re.escape(w) for w in t.hw)
            for m in re.finditer(re.escape(tok) + r'(?:-[A-Za-z])?-(?:' + alt + r')\b|\b(?:' + alt + r')-(?:[A-Za-z]-)?' +
                                 re.escape(tok), text, re.I):
                hit('S8_hyphen_name_remnant', m.group(0), m)
    for name, rx in GIVEBACK_RX:
        for m in rx.finditer(text):
            hit('S9_giveback_' + name, name, m)
    for name, rx in DATE_RX:
        for m in rx.finditer(text):
            if name == 'year4' and _YEAR_UNITS.match(text[m.end():]):
                continue
            hit('S10_date', name, m)
    for m in STORY_RX.finditer(text):
        hit('S11_story_field', 'stub', m)
    for m in MISTAG_RX.finditer(text):
        hit('S12_mistag_notice', 'notice', m)
    for m in MODEL_RX.finditer(text):
        hit('S13_model_after_token', 'model', m)
    for m in RIC_SUFFIX_RX.finditer(text):
        hit('S14_ric_suffix', 'suffix', m)
    for m in EXEC_RX.finditer(text):
        hit('S15_executive_next_to_token', 'executive', m)
    return hits


def articles(input_text):
    """[(tokens, headline, text)] of the PARTNER NEWS section."""
    part = input_text.split('\nPARTNER NEWS\n', 1)
    if len(part) < 2:
        return []
    return [([x.strip() for x in m.group('toks').split(',')], m.group('h'), m.group('b'))
            for m in _ARTICLE.finditer(part[1])]


def tok_map(meta):
    out = {'[FOCAL]': int(meta['focal_permno'])}
    for s in meta['shown']:
        out[s['token']] = int(s['permno'])
    for x in meta['links']:
        out[x['token']] = int(x['permno'])
    return out


def scan_input(text, meta, terms):
    hits = []
    tm = tok_map(meta)
    for toks, h, b in articles(text):
        for field, s in (('headline', h), ('body', b)):
            for cls, term, ctx in scan_text(s, tm, terms, set(toks)):
                hits.append({'class': cls, 'term': term, 'field': field, 'context': ctx})
    return hits


def build_terms(permnos, last_day, english, common, renames=None):
    names = load_names(permnos, last_day)
    if renames:
        for p, e in renames.items():
            if int(p) in names and e['dated'][:4] <= last_day[:4]:
                names[int(p)]['comnams'] |= set(e['names'])
    return {p: Terms(v['comnams'], v['tickers'], english, common) for p, v in names.items()}


def class_counts(hits_by_key):
    c = Counter()
    for hs in hits_by_key.values():
        for k in {h['class'] for h in hs}:
            c[k] += 1
    return dict(sorted(c.items()))
