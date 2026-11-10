"""TD4 v2.2 span-level redaction (fix round after the TD4M3 redaction audit FAIL of 2026-11-07; redaction only).

Same rule A.4 as v2/v2.1: at most three redact-and-re-run rounds from the masker-v2 inputs (inputs_m2/*_r3, unchanged);
inputs still flagged or unlocatable after round 3 are withheld for every arm alike; the 2021-2022 adjudicated key sets
are withheld; MASKER_NO_SOURCE (v2.1) is unchanged. Changes (F1-F3):

F1 word boundaries. A "word" is a maximal run of characters c with c.isalnum() (Unicode letters and digits), where one
   connector character from CONNECT (hyphens, en dash, apostrophes, ampersand, period, slash) between two such characters
   joins the run; a leading news slug ('BRIEF-', 'UPDATE 2-', ... SLUG_WORD) is not part of the word. Every
   [REDACTED]/[DATE] span is widened to whole words (the words of its first and last characters), inside one article
   field and never over a bracket token; a word or word part glued to any placeholder ([REDACTED], [DATE], [TERM], also
   masker-v2's own [DATE]) is replaced by [REDACTED], so no ledger placeholder is next to a letter or digit (P6 = 0). A
   month abbreviation cut out of a longer word ('markets', 'decision') or a date form inside a code ('1/2a') becomes the
   whole word, with the placeholder of the class that matched.
F2 builder locators, run in every round with the reviewers' detectors and repeated on the provisional text of the round
   until they find nothing more (at most ITER passes; an input still finding spans in the last pass is not converged
   and stays flagged), from the receipt-bound name tables of td4v2_nosource (CRSP own and issuer-sibling rows <= the
   build's last day, FSB1 names and US tickers, CIZ 2025) of the input's focal and shown firms. Lists built at run time:
   distinctive words dw (>= 4 letters, not LEGAL, not GENERIC, not an English word or inflection: is_eng), core words cw
   (>= 5 letters, not GENERIC2, English allowed), initialisms (first letters of the first 2-4 significant words), tickers.
   (a) own names, whole word: dw in any case; every multi-word core name and every two-word part of it that is not two
       English/generic words, in any case; initialisms and tickers in upper case, and in Title case when not an English
       word. Guard: an initialism or ticker that is an English word or a short common word (COMMON_SHORT) is matched
       only in upper case and only in a mixed-case line (more than 30% lower-case letters within 60 characters on the
       same line); generic acronyms (GENERIC_ACR) are never matched. The firm's first significant name word when it is
       an English word (>= 4 letters, not GENERIC2) is matched in any case inside the firm's own article(s).
   (b) hyphenated name parts (>= 3 letters, next to a hyphen) and first-letter-cut forms (first core word without its
       first letter + the second core word; a one-word distinctive name without its first letter).
   (c) compounds, the whole word redacted: a longer non-English word that contains a dw (>= 5 letters anywhere, else as
       prefix or suffix) or a cw; a longer word containing a name word's prefix of length max(5, n-2)..n-1 (anywhere) or
       a dw's suffix of that length (at the end); either part of a dw that is a closed compound of two English words
       (prefix/suffix); a word starting with an own initialism, ticker or a 2-3 letter prefix of the first name word or
       ticker followed by a digit (codes such as 'ABC-101'), and any other word carrying the same >= 3 digit code number;
       the long form of a short dw ('Tri Vo...' style: 2-3 capitalised words starting with consecutive parts of the dw,
       first part >= 3 letters). English-derived matches (cw, prefix fragments, compound parts) skip a lower-case word in
       mixed-case text and a hyphenated word whose parts all read as English (eng_part).
   (d) after a role token (space or hyphen): a word with a digit and a letter; an upper-case word of >= 2 letters that is
       not an English word or English compound; any upper-case word in a mixed-case line; a 3-5 digit number not
       followed by a unit word; continued over the following code-like words (digits, short capitals, 'A1, A2' lists);
       stop words, Reuters abbreviations and period codes (D_SKIP: Q1, H2, FY23, 3Q, CY24, COVID-19, ...) excepted.
       Also an acronym defined in parentheses after a role token or placeholder and the words before it ('[X] UNIT I
       (UXI)'), with every whole-word use of that acronym; a company name before a legal suffix right after a role
       token ('[X] SOME RIVER GAS COMPANY, L.L.C.').
   (e) give-back phrases: every M4 GB_* pattern (s13's own expressions) with any bracket token in its slot: the
       pattern's literal words are redacted and a role-token slot becomes [TERM] (F3 context words); the neighbour word
       of a frozen MA1 GB group slot; both neighbour words of a [REDACTED] slot that hides a generic acronym inside a
       phrase; a role token of a firm named like a period code (PCODE_NAME) in a period slot becomes [TERM].
   (f) dates: a two-to-four-digit year next to [YEAR] ('[YEAR]-22', '22/[YEAR]'); a month name (and day) next to a
       [DATE]; the day number after 'Month [DATE]' (the masker-v2 swallow case).
   Over-redaction is identical for all arms; td4v2_check22 measures it (share of article word tokens redacted).
F3 no placeholder waiver for M4 GB_* hits: a GB_* hit still present after round 3 is still flagged; the only waiver is
   a match that exists only because of a placeholder's own characters (an M4 word hit equal to the word inside
   '[TERM]', '[DATE]' or '[REDACTED]'), listed per class in REDACTION_FINAL_v2_2.
Storage: rounds 1-2 are ledger-only (LEDGER.jsonl, HASHES.jsonl, ROUND.json); round 3 also writes INPUTS.jsonl. Any
round is rebuilt with td4v2_render.render(masker-v2 text, cumulative ledger); the detector re-runs read rounds 1-2
through td4v2_render.open_round. Name, phrase and word lists are built at run time from private tables (nothing
private is in this file). Outputs (private): OUT/inputs_redacted_v2_2/<period>/round_<r>/; detector re-runs
OUT/detector_flags/<period>_v22rd<r>_<UTC>/; <build>/REDACTION_FINAL_v2_2_<UTC>.json. Reads no label, return or
provider output.
"""
import bisect
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_redact as RD
import td4v2_render as RR

ROUNDS = 3
ITER = 6
RESCAN_N = 3000
TCACHE = 16                                       # compiled Terms kept (most firm sets occur once; bounds memory)
ROOT = T.OUT / 'inputs_redacted_v2_2'
PH = RD.PH
PLACEHOLDERS = RD.PLACEHOLDERS
CONNECT = set("-\u2010\u2011\u2012\u2013'\u2019&./")
ROLE_RX = r'\[(?:FOCAL|LINK_\d+|(?:SUPPLIER|CUSTOMER)(?:_OF_(?:SUPPLIER|CUSTOMER))*_\d+)\]'
ENGLISH_PATH = Path('C:/Users/Boxuan/AppData/Roaming/nltk_data/corpora/words/en')
MON = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|'
       r'Nov(?:ember)?|Dec(?:ember)?)')
EXTRA_ENG = set('online internet website websites app apps smartphone smartphones email emails ecommerce startup '
                'startups biotech fintech blockchain crypto cryptocurrency covid vaccine vaccines streaming broadband '
                'wireless pharma healthcare semiconductor semiconductors software hardware chipmaker chipmakers '
                'automaker automakers airline airlines biopharma telecom telecoms cybersecurity datacenter rideshare '
                'esports offshore onshore downstream upstream midstream'.split())
LEGAL = set('INC INCORPORATED CORP CORPORATION CO COMPANY COS LTD LIMITED PLC LLC LP L P NV N V SA S A AG SE AB ASA OYJ '
            'SPA BV THE NEW DEL DE HLDGS HOLDINGS HOLDING GROUP GRP OF AND & TRUST ADR ADS CL CLASS INTL INTERNATIONAL '
            'SPON SPONSORED COM SHS'.split())
GENERIC = set('american national united general first global international energy financial bank bancorp capital group '
              'holdings technologies technology systems services solutions industries pharmaceuticals therapeutics '
              'communications resources partners properties realty trust health healthcare medical brands foods '
              'products enterprises networks software semiconductor electronics motors airlines entertainment media '
              'insurance investment investments acquisition corp company inc ltd north south east west new data '
              'digital power oil gas petroleum mining minerals steel chemical chemicals materials biosciences bio '
              'labs laboratories sciences pharma devices instruments retail stores restaurants hotels resorts '
              'transportation logistics marine air gold silver copper water electric utilities'.split())
GENERIC2 = GENERIC | set('pharmaceutical biopharmaceutical biopharmaceuticals biotechnology biologics genetics genomics '
                          'diagnostics therapeutic medicine medicines system service solution industry industrial product '
                          'network enterprise communication telecommunications resource partner property brand food '
                          'electronic motor airline chemical material bioscience laboratory science device instrument '
                          'store restaurant hotel resort utility manufacturing aerospace defense automotive apparel '
                          'hospitality interactive innovations innovation environmental engineering construction '
                          'infrastructure development management equipment components packaging paper transport shipping '
                          'tankers drilling exploration production pipeline royalty royalties residential commercial '
                          'mortgage lending credit payments wireless broadcasting publishing education hospitals clinical '
                          'surgical dental vision animal nutrition beverage beverages tobacco brewing distillers cosmetics '
                          'household consumer wholesale distribution freight trucking railroad railway aviation america '
                          'americas corporation company holdings bancshares bancorporation reinsurance insurance '
                          'semiconductors technologies'.split())
GENERIC_ACR = set('US USA UK EU UN AI IT TV PC CEO CFO COO CTO IPO ETF GDP FDA SEC FTC DOJ EPS NAV FX OK LA NY DC EV AM '
                  'PM FY YOY LLC INC CO LTD PLC LP SA AG NV SE AS OY BV NA II III IV VI GMT EST EDT UTC ID HQ RE TO OF '
                  'IN AT BY OR AN NO SO UP GO DO BE WE HE ME MY IF IS QTR OTC API ESG IR PR HR OP VP SVP EVP GM PE VC AUM '
                  'NII FFO EBIT EBITDA GAAP IBES LSEG FCC FAA EPA IRS NATO OPEC NYSE ALL ARE CAN NOW ON ONE SEE KEY NET '
                  'BIG LOW HAS ANY FOR RUN DR MR MS ET PT VS SG AC CC QQ YY MD CA TX FL NJ PA OH IL WA MA GA NC MI VA '
                  'AZ MN WI MO TN SC AL KY CT IA AR UT KS NM NE WV HI NH RI MT SD ND AK VT WY THE AND NEW Q1 Q2 Q3 Q4 H1 '
                  'H2 ADR ADS REIT SPAC IPO ETF BRIEF UPDATE LIVE PREVIEW SAYS SAID SEES CUTS RAISES NAMES'.split())
STOP_AFTER = set('INC CO CORP LTD LLC PLC LP SA AG NV SE AB AND OR TO OF IN ON AT BY FOR THE IS HAS SAYS SAID SEES CEO '
                 'CFO COO CTO US USA UK EU AS IT WILL WAS ARE BE CAN NOT NO NEW Q1 Q2 Q3 Q4 FY H1 H2 SHARES STOCK UNIT '
                 'UNITS GROUP NAMES CUTS RAISES TO SET GETS GOT WINS WON BUY SELL AN A IPO ETF EPS REIT SPAC ESG AI IR '
                 'PR HR VP SVP EVP GM GMT EST EDT NYSE NASDAQ SEC FDA FTC DOJ EPA IRS FAA OTC ADR ADS LIVE BRIEF UPDATE '
                 'PREVIEW'.split())
COMMON_SHORT = set('a an as at be by do go he if in is it me my no of on or so to up us we am id ok ox ad ah eh hi ho '
                   'oh'.split())
PERIOD_CODE = re.compile(r'(?i)^(?:Q[1-4]|H[12]|[1-4]Q\d{0,4}|FY\d{0,4}|CY\d{0,4}|COVID-?19|G\d{1,2}|P\d|M\d|'
                         r'Q[1-4]-?\d{2,4}|H[12]-?\d{2,4}|9M|6M|3M|12M|\d{1,2}[QH])$')
D_SKIP = re.compile(r'(?i)^(?:Q[1-4]|H[12]|[1-4]Q\d{0,4}|FY\d{0,4}|CY\d{0,4}|Q[1-4]-?\d{2,4}|H[12]-?\d{2,4}|9M|6M|3M|12M|'
                    r'\d{1,2}[QH]|COVID-?19)$')                    # period codes kept after a role token (d)
REUTERS_ABBR = set('QTRLY QTR QUARTERLY MLN BLN MILLION BILLION SHR SHRS REV REVS ADJ ANNUAL DIV PCT YR YRS VS EST NOTES '
                   'BOARD DIRECTOR DIRECTORS'.split())
UNIT_AFTER = re.compile(r'\s*(?:%|percent\b|per\b|million\b|billion\b|mln\b|bln\b|thousand\b|people\b|jobs\b|workers\b|'
                        r'employees\b|stores\b|shares\b|units\b|barrels\b|tons\b|tonnes\b|vehicles\b|cars\b|patients\b|'
                        r'customers\b|homes\b|miles\b|days\b|weeks\b|months\b|years\b|hours\b|basis\b|bps\b|points\b|'
                        r'cents\b|dollars\b|euros\b|locations\b|sites\b|branches\b|restaurants\b)', re.I)
PREFIXES = ('re', 'pre', 'un', 'non', 'co', 'sub', 'multi', 'inter', 'over', 'under', 'out', 'mid', 'post', 'anti',
            'semi', 'self', 'cross', 'ex', 'de', 'dis', 'mis', 'bi', 'tri', 'next', 'well', 'long', 'short', 'high', 'low')
SLUG_WORD = re.compile(r'(?:BRIEF|EXCLUSIVE|ANALYSIS|INSIGHT|FACTBOX|COLUMN|MEDIA|RPT|CORRECTED|REFILE|BREAKINGVIEWS|'
                       r'DIGEST|BUZZ|EXPLAINER|FOCUS|GRAPHIC|POLL|INTERVIEW|TIMELINE|PREVIEW|WRAPUP|UPDATE|CORRECTION|'
                       r'ADVISORY|DEALTALK|FEATURE|TABLE|TEXT|REPORT|STOCKS|MARKETS|POLITICS)-')
FUNCTION = set('a an the and or of for with from to in on at by as is are was were be been has have had its their his '
               'her our your this that these those which who will would can could may might shall should said says say '
               'also not no than then but if so up out over into onto via per vs its'.split())
EXCH = set('NYSE NASDAQ LSE TSX ASX OTC AMEX HKEX SGX JPX BSE NSE CBOE CME ICE'.split())
_ENG = {}


def english():
    if 'w' not in _ENG:
        _ENG['w'] = frozenset(x.strip().lower() for x in ENGLISH_PATH.open(encoding='utf-8', errors='replace')
                              if x.strip())
    return _ENG['w']


def is_eng(w, eng=None):
    """English word or inflection (the refined test of the redaction audit)."""
    eng = english() if eng is None else eng
    lw = w.lower().replace('\u2019', "'")
    if lw.endswith("'s"):
        lw = lw[:-2]
    if lw in eng or lw in EXTRA_ENG:
        return True
    c = []
    if lw.endswith('ies'):
        c.append(lw[:-3] + 'y')
    if lw.endswith('es'):
        c.append(lw[:-2])
    if lw.endswith('s'):
        c.append(lw[:-1])
    if lw.endswith('ed'):
        c += [lw[:-2], lw[:-1]]
    if lw.endswith('ing'):
        c += [lw[:-3], lw[:-3] + 'e']
    if lw.endswith('ers'):
        c += [lw[:-3], lw[:-2]]
    if lw.endswith('er'):
        c += [lw[:-2], lw[:-1]]
    if lw.endswith('ly'):
        c.append(lw[:-2])
    return any(x in eng or x in EXTRA_ENG for x in c if len(x) >= 3)


def eng_part(x, eng):
    """A hyphen part that reads as English: an English word or inflection, a number, a common prefix, or a common
    prefix plus an English word."""
    x = x.lower()
    if not x or x.isdigit() or x in PREFIXES or is_eng(x, eng):
        return True
    return any(x.startswith(q) and len(x) - len(q) >= 3 and is_eng(x[len(q):], eng) for q in PREFIXES)


def eng_compound(w, eng):
    """An English word, or a closed compound of two English words (each >= 3 letters), or hyphen parts that all read
    as English."""
    parts = [x for x in re.split(r'[\-\u2010\u2011/]', w) if x]
    if len(parts) > 1:
        return all(eng_part(x, eng) for x in parts)
    lw = re.sub(r'[^a-z]', '', w.lower())
    if not lw or is_eng(lw, eng):
        return True
    return any(is_eng(lw[:i], eng) and is_eng(lw[i:], eng) for i in range(3, len(lw) - 2))


# ------------------------------------------------------------------------------------------------------- words (F1)
def word_span(text, a, b, lo=0, hi=None):
    """The whole words that contain text[a:b] (widened inside [lo, hi)); see the module docstring for 'word'."""
    a0 = a
    a, b = _word_span(text, a, b, lo, hi)
    w = text[a:b]
    m = SLUG_WORD.match(w)
    if m is None and re.match(r'\d{1,2}-', w) and re.search(r'(?:UPDATE|WRAPUP) $', text[max(lo, a - 8):a]):
        m = re.match(r'\d{1,2}-', w)
    if m is not None and a + m.end() <= a0 and a + m.end() < b:
        a += m.end()                                   # a news slug ('BRIEF-', 'UPDATE 2-') is not part of the word
    return a, b


def _word_span(text, a, b, lo=0, hi=None):
    hi = len(text) if hi is None else hi
    while a > lo:
        c = text[a - 1]
        if c.isalnum():
            a -= 1
        elif c in CONNECT and a - 1 > lo and text[a - 2].isalnum() and a < hi and text[a].isalnum():
            a -= 1
        else:
            break
    while b < hi:
        c = text[b]
        if c.isalnum():
            b += 1
        elif c in CONNECT and b + 1 < hi and text[b + 1].isalnum() and b > lo and text[b - 1].isalnum():
            b += 1
        else:
            break
    return a, b


def words(text, lo, hi):
    """(start, end) of every word in text[lo:hi]."""
    out, i = [], lo
    while i < hi:
        if text[i].isalnum():
            a, b = word_span(text, i, i + 1, lo, hi)
            out.append((a, b))
            i = b
        else:
            i += 1
    return out


def widen(doc):
    """F1: widen every [REDACTED]/[DATE] request of a Doc to whole words inside its field."""
    out = []
    for s, e, kind, cls in doc.req:
        if kind != 'term':
            seg = doc.seg_of(s, e)
            if seg is not None:
                s, e = word_span(doc.text, s, e, seg[0], seg[1])
        out.append((s, e, kind, cls))
    doc.req = out


def glue_fix(doc, spans):
    """F1: a word or word part glued to a placeholder span (or to an existing placeholder) is redacted."""
    n = 0
    t = doc.text
    marks = [(s, e) for s, e, _, _ in spans] + [(m.start(), m.end()) for m in re.finditer(r'\[(?:REDACTED|DATE|TERM)\]', t)]
    for s, e in marks:
        seg = doc.seg_of(s, e)
        if seg is None:
            continue
        if s > seg[0] and t[s - 1].isalnum():
            a, b = word_span(t, s - 1, s, seg[0], s)
            n += doc.add(a, b, 'V22:glue', 'name')
        if e < seg[1] and t[e].isalnum():
            a, b = word_span(t, e, e + 1, e, seg[1])
            n += doc.add(a, b, 'V22:glue', 'name')
    return n


# --------------------------------------------------------------------------------------------- firm profiles (F2)
def _words_upper(s):
    return [w for w in re.split(r"[^A-Za-z0-9&]+", s.upper()) if w]


def firm_profile(names, tickers, eng):
    """Name terms of one firm from its table names and tickers."""
    dw, inits, phrases, hyph, cut, tks = set(), set(), set(), set(), set(), set()
    cw, frag, pref, pcode, own = set(), set(), set(), False, set()
    for n in names:
        nu = ' '.join(n.upper().split())
        for hw in re.findall(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", nu):
            for part in hw.split('-'):
                if len(part) >= 3 and part.isalpha() and part not in LEGAL:
                    hyph.add(part)
        ws = _words_upper(nu)
        sig = [w for w in ws if w not in LEGAL]
        for w in sig:
            if len(w) >= 4 and w.isalpha() and w.lower() not in GENERIC and not is_eng(w, eng):
                dw.add(w)
            if len(w) >= 5 and w.isalpha() and w.lower() not in GENERIC2:
                cw.add(w)
        if sig and PCODE_NAME.match(sig[0]):
            pcode = True
        if sig and sig[0].isalpha() and len(sig[0]) >= 4 and sig[0].lower() not in GENERIC2 and is_eng(sig[0], eng):
            own.add(sig[0])                            # an English first name word, read in the firm's own article
        if sig and sig[0].isalpha() and len(sig[0]) >= 4:
            pref.update(sig[0][:k] for k in (2, 3))
        core = list(ws)
        while core and core[0] == 'THE':
            core = core[1:]
        while core and core[-1] in LEGAL:
            core = core[:-1]
        if len(core) >= 2 and any(len(w) >= 3 for w in core):
            phrases.add(tuple(core))
            for i in range(len(core) - 1):
                a2, b2 = core[i], core[i + 1]
                if a2 not in LEGAL and b2 not in LEGAL and len(a2 + b2) >= 6 and not \
                        ((is_eng(a2, eng) or a2.lower() in GENERIC) and (is_eng(b2, eng) or b2.lower() in GENERIC)):
                    phrases.add((a2, b2))
        ini = [w for w in sig if w[0].isalpha()]
        for k in (2, 3, 4):
            if len(ini) >= k:
                s = ''.join(w[0] for w in ini[:k])
                if s not in GENERIC_ACR:
                    inits.add(s)
        if len(sig) >= 2 and len(sig[0]) >= 5 and sig[0].isalpha():
            cut.add((sig[0][1:], sig[1]))
        if len(sig) == 1 and len(sig[0]) >= 6 and sig[0].isalpha() and not is_eng(sig[0][1:], eng):
            cut.add((sig[0][1:],))
    for t in tickers:
        t = (t or '').strip().upper()
        if len(t) >= 2 and re.fullmatch(r'[A-Z][A-Z0-9]{1,5}', t) and t not in GENERIC_ACR:
            tks.add(t)
            if t.isalpha():
                pref.update(t[:k] for k in range(2, len(t)))
        if PCODE_NAME.match(t or ''):
            pcode = True
    for w in dw | cw:
        if len(w) >= 7 and w.isalpha():
            for k in range(max(5, len(w) - 2), len(w)):
                frag.add(('A', w[:k]))                 # a name word's prefix anywhere inside a longer word
                if w in dw:
                    frag.add(('S', w[-k:]))            # a distinctive word's suffix at the end of a longer word
    for w in dw:
        if w.isalpha():
            for i in range(4, len(w) - 3):
                if is_eng(w[:i], eng) and is_eng(w[i:], eng):
                    frag.add(('P', w[:i]))             # closed compound of two English words: either part
                    frag.add(('T', w[i:]))
    port = set()
    for w in dw:
        if w.isalpha() and 5 <= len(w) <= 10:
            for i in range(3, len(w) - 1):
                port.add((w[:i], w[i:]))
                for j in range(i + 2, len(w) - 1):
                    port.add((w[:i], w[i:j], w[j:]))
    pref = {x for x in pref if not PERIOD_CODE.match(x)}
    return {'dw': dw, 'cw': cw, 'frag': frag, 'port': port, 'own': own, 'ini': inits - GENERIC_ACR, 'tk': tks, 'pref': pref,
            'phrases': phrases, 'hyph': hyph, 'cut': cut, 'period_code_name': pcode}


def build_profiles(period, permnos):
    """{permno: profile} from the build's receipt-bound name tables (td4v2_nosource.tables)."""
    import td4v2_nosource as NS
    tb = NS.tables(period, permnos)
    eng = english()
    last = tb['last_day']
    out = {}
    for p in permnos:
        names, ticks = set(), set()
        for q in [p] + [int(x) for x in tb['siblings'].get(p, [])]:
            for d, c, t in tb['names'].get(q, []):
                if d <= last:
                    if c:
                        names.add(c)
                    if t:
                        ticks.add(t)
        names |= set(tb['fsnames'].get(p, ()))
        ticks |= set(tb['fstickers'].get(p, ()))
        for d, n, t in tb['ciz'].get(p, ()):
            if d <= last:
                if n:
                    names.add(n)
                if t:
                    ticks.add(t)
        out[p] = firm_profile(names, ticks, eng)
        out[p]['permno'] = p
    return out, tb['files_sha256']


def _alt(xs):
    return '|'.join(re.escape(x) for x in sorted(set(xs), key=len, reverse=True))


class Terms:
    """Compiled locator expressions of one set of firms (focal and shown firms of an input)."""

    def __init__(self, profs, eng):
        dw = set().union(*[p['dw'] for p in profs]) if profs else set()
        codes = set().union(*[p['ini'] | p['tk'] for p in profs]) if profs else set()
        self.dw = dw
        self.pcode_permnos = {p.get('permno') for p in profs if p.get('period_code_name')}
        self.dw_rx = re.compile('(?i)(' + _alt(dw) + ')') if dw else None
        ph = set().union(*[p['phrases'] for p in profs]) if profs else set()
        self.ph_rx = re.compile(r'(?i)(?<!\w)(?:' + '|'.join(r'[\s\-\u2010\u2011]+'.join(re.escape(w) for w in p)
                                                         for p in sorted(ph, key=lambda x: -len(' '.join(x)))) +
                                r')(?!\w)') if ph else None
        up = {c for c in codes if c.isalpha() or re.search(r'[A-Z]', c)}
        plain = {c for c in up if not (is_eng(c, eng) or c.lower() in COMMON_SHORT)}
        engc = up - plain
        self.up_rx = re.compile(r'(?<!\w)(' + _alt(plain) + r')(?!\w)') if plain else None
        title = {c[0] + c[1:].lower() for c in plain if c.isalpha()}
        self.title_rx = re.compile(r'(?<!\w)(' + _alt(title) + r')(?!\w)') if title else None
        self.eng_rx = re.compile(r'(?<!\w)(' + _alt(engc) + r')(?!\w)') if engc else None
        self.code_rx = re.compile(r'(?<!\w)(' + _alt(up) + r')(?=[\-\u2010\u2011]?\d|[\-\u2010\u2011][A-Za-z0-9])') \
            if up else None
        pre = set().union(*[p['pref'] for p in profs]) if profs else set()
        self.pref_rx = re.compile(r'(?<!\w)(' + _alt(pre) + r')(?=[\-\u2010\u2011]?\d)') if pre else None
        cw = (set().union(*[p['cw'] for p in profs]) if profs else set()) - dw
        self.cw_rx = re.compile('(?i)(' + _alt(cw) + ')') if cw else None
        fr = set().union(*[p['frag'] for p in profs]) if profs else set()
        fa, fp, fs, ft = ({f for k, f in fr if k == x} for x in 'APST')
        self.frag_a = re.compile(r'(?i)(' + _alt(fa) + r')') if fa else None
        self.frag_p = re.compile(r'(?i)(?<!\w)(' + _alt(fp) + r')(?=\w)') if fp else None
        self.frag_s = re.compile(r'(?i)(?<=\w)(' + _alt(fs) + r')(?!\w)') if fs else None
        self.frag_t = re.compile(r'(?i)(?<=\w)(' + _alt(ft) + r')(?!\w)') if ft else None
        po = set().union(*[p['port'] for p in profs]) if profs else set()
        self.port_rx = re.compile(r'(?i)(?<!\w)(?:' + '|'.join(r'[a-z]*[ \-\u2010\u2011]+'.join(re.escape(x) for x in q) +
                                                         r'[a-z]*' for q in sorted(po, key=lambda q: (-len(''.join(q)), q))) +
                                  r')(?!\w)') if po else None
        self._own = {p.get('permno'): p['own'] for p in profs if p.get('own')}
        self._own_rx = {}
        hy = set().union(*[p['hyph'] for p in profs]) if profs else set()
        self.hy_rx = re.compile(r'(?i)(?:(?<=[\-\u2010\u2011])(' + _alt(hy) + r')(?!\w)|(?<!\w)(' + _alt(hy) +
                                r')(?=[\-\u2010\u2011]))') if hy else None
        cu = set().union(*[p['cut'] for p in profs]) if profs else set()
        self.cut_rx = re.compile(r'(?i)(?<![A-Za-z])(' + '|'.join(r'\s+'.join(re.escape(w) for w in c)
                                                                  for c in sorted(cu, key=lambda x: -len(' '.join(x)))) +
                                 r')(?!\w)') if cu else None


def _own_rx(terms, permno):
    if permno not in terms._own:
        return None
    if permno not in terms._own_rx:
        terms._own_rx[permno] = re.compile(r"(?i)(?<!\w)(" + _alt(terms._own[permno]) + r")(?:['\u2019]s)?(?!\w)")
    return terms._own_rx[permno]


def mixed_line(text, a, b):
    """mixed() limited to the line of the match (an all-caps BRIEF line next to a Title-case headline is not mixed)."""
    lo = text.rfind('\n', 0, a) + 1
    hi = text.find('\n', b)
    hi = len(text) if hi < 0 else hi
    s = text[max(lo, a - 60):min(hi, b + 60)]
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.islower() for c in letters) / len(letters) > 0.3


def mixed(text, a, b):
    s = text[max(0, a - 60):b + 60]
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.islower() for c in letters) / len(letters) > 0.3


PCODE_NAME = re.compile(r'(?i)^(?:Q[1-4]|H[12]|[1-4]Q|FY|CY)$')    # firm names that read as a period code (e2)
_PSLOT_L = re.compile(r'(?i)(?:\bsees|\bsee|\bin|\bfor|\bfiscal|\bits|\bthe|\bexpects|\bend of)\s+$')
_PSLOT_R = re.compile(r'(?i)\s+(?:total|revenue|revenues|earnings|results|sales|eps|net|adjusted|guidance|outlook|profit|loss|'
                      r'fy|ended|\[YEAR\]|\d)')
_PNEAR_L = re.compile(r'(?<![A-Za-z0-9])(?:Q[1-4]|H[12]|[1-4]Q)\s*(?:,|and|&|-|\u2013|/|to|vs\.?|versus)?\s*$', re.I)
_PNEAR_R = re.compile(r'\s*(?:,|and|&|-|\u2013|/|to|vs\.?|versus)?\s*(?:Q[1-4]|H[12]|[1-4]Q)(?![A-Za-z0-9])', re.I)
_LEGAL_SFX = (r'(?:INC|Inc|CO|Co|COMPANY|Company|CORP|Corp|CORPORATION|Corporation|LLC|L\.L\.C|LP|L\.P|LTD|Ltd|'
              r'LIMITED|Limited|PLC|Plc|AG|SA|NV|GMBH|GmbH)\.?')
_DEFACR = re.compile(r'(?:' + ROLE_RX + r'|\[REDACTED\]|\[TERM\])(?:[ \-\u2010\u2011][A-Za-z0-9&\'\u2019]+){0,6}'
                     r'(?:,? ' + _LEGAL_SFX + r')? ?\((?P<acr>[A-Z][A-Za-z0-9&\-]{1,9})\)')
_UNIT = re.compile(ROLE_RX + r' (?P<unit>(?:[A-Z][A-Za-z0-9&\'\u2019]*[ \-]){0,5}[A-Z][A-Za-z0-9&\'\u2019]*),? '
                   + _LEGAL_SFX + r'(?![A-Za-z])')
_GB = {}
_AFTER = re.compile(ROLE_RX + r'(?:[ \-\u2010\u2011])(?=\S)')
_YEAR = [re.compile(r'\[YEAR\]\s?[-\u2013/]\s?(\d{2,4})(?!\d)'), re.compile(r'(?<![\d.,$])(\d{2,4})\s?[-\u2013/]\s?\[YEAR\]')]
_MON_BEFORE = re.compile(r'(?<![A-Za-z])(' + MON + r'\.?)(?:\s{1,2}\d{1,2}(?:st|nd|rd|th)?)?,?\s{0,2}(?:-|\u2013|to|and|through|&)?\s{0,2}\[DATE\]', re.I)
_MON_AFTER = re.compile(r'\[DATE\],?\s{0,2}(?:-|\u2013|to|and|through|&)?\s{0,2}(' + MON + r')(?![A-Za-z])', re.I)
_SWALLOW = re.compile(r'(?<![A-Za-z])' + MON + r'\.?\s\[DATE\],?\s(\d{1,2})(?![\d,.%])', re.I)


def gb_patterns(run):
    """The M4 GB_* expressions of s13 (read from the run copy, not run)."""
    key = str(run)
    if key not in _GB:
        _GB[key] = RD.consts(Path(run) / 'ma1' / 's13_m4_pop.py', ['GB'])['GB']
    return _GB[key]


def locate_v22(doc, terms, gb, eng):
    """F2 locators (a)-(f) on one Doc (current text); requests are added to doc.req."""
    t = doc.text
    if terms.dw_rx is not None:
        for m in doc.content_matches(terms.dw_rx):
            seg = doc.seg_of(m.start(), m.end())
            a, b = word_span(t, m.start(), m.end(), seg[0], seg[1])
            w = t[a:b]
            wu = re.sub(r'[^A-Za-z0-9]', '', w.upper())
            hit = m.group(1).upper()
            if wu == hit or (w.upper().endswith(("'S", "\u2019S")) and wu[:-1] == hit):
                doc.add(a, b, 'V22:a_name')
            elif len(hit) >= 5 or wu.startswith(hit) or wu.endswith(hit):
                if not is_eng(w, eng):
                    doc.add(a, b, 'V22:c_compound')
    for rx, english_part in ((terms.cw_rx, True), (terms.frag_a, True), (terms.frag_p, True), (terms.frag_t, True),
                             (terms.frag_s, False)):              # (c) compounds of name words / name fragments
        if rx is None:
            continue
        for m in doc.content_matches(rx):
            seg = doc.seg_of(m.start(), m.end())
            a, b = word_span(t, m.start(), m.end(), seg[0], seg[1])
            w = t[a:b]
            if b - a > m.end() - m.start() and len(re.sub(r'[^A-Za-z]', '', w)) >= 6 and not is_eng(w, eng) and \
                    not any(is_eng(x, eng) and len(x) >= len(w) - 1 for x in re.split(r'[\-\u2010\u2011/]', w)) and \
                    not (len(re.split(r'[\-\u2010\u2011/]', w)) > 1 and eng_compound(w, eng)) and \
                    not (english_part and w[:1].islower() and mixed(t, a, b)):
                doc.add(a, b, 'V22:c_compound')
    if terms.port_rx is not None:                                  # (c) long form of a short name ('Tri Volta')
        for m in doc.content_matches(terms.port_rx):
            ws = re.split(r'[ \-\u2010\u2011]+', m.group(0))
            if all(x[:1].isupper() for x in ws):
                doc.add(m.start(), m.end(), 'V22:c_long_form')
    for no, toks, hs, tx in doc.arts:                              # (a) English name words in the firm's own article
        for tok in set(toks):
            rx = _own_rx(terms, doc.t2p.get(tok))
            if rx is None:
                continue
            for lo, hi in (hs, tx):
                for m in rx.finditer(t, lo, hi):
                    if doc.seg_of(m.start(), m.end()) is not None:
                        doc.add(m.start(), m.end(), 'V22:a_own_article')
    if terms.ph_rx is not None:
        for m in doc.content_matches(terms.ph_rx):
            doc.add(m.start(), m.end(), 'V22:a_name')
    for rx in (terms.up_rx, terms.title_rx):
        if rx is not None:
            for m in doc.content_matches(rx):
                doc.add(m.start(), m.end(), 'V22:a_initialism')
    if terms.eng_rx is not None:
        for m in doc.content_matches(terms.eng_rx):
            if mixed_line(t, m.start(), m.end()):
                doc.add(m.start(), m.end(), 'V22:a_initialism')
    nums = set()
    for rx in (terms.code_rx, terms.pref_rx):                     # (c) codes built from a ticker, initialism or prefix
        if rx is None:
            continue
        for m in doc.content_matches(rx):
            seg = doc.seg_of(m.start(), m.end())
            a, b = word_span(t, m.start(), m.end(), seg[0], seg[1])
            if PERIOD_CODE.match(t[a:b]):
                continue
            doc.add(a, b, 'V22:c_compound')
            nums.update(x for x in re.findall(r'\d+', t[a:b]) if len(x) >= 3)
    if nums:                                                       # the same code number under another prefix
        for m in doc.content_matches(re.compile(r'(?<![\w\]])[A-Za-z]{1,10}[\-\u2010\u2011]?(' + _alt(nums) + r')(?!\d)')):
            seg = doc.seg_of(m.start(), m.end())
            a, b = word_span(t, m.start(), m.end(), seg[0], seg[1])
            if not PERIOD_CODE.match(t[a:b]):
                doc.add(a, b, 'V22:c_compound')
    if terms.hy_rx is not None:
        for m in doc.content_matches(terms.hy_rx):
            doc.add(m.start(), m.end(), 'V22:b_hyphen')
    if terms.cut_rx is not None:
        for m in doc.content_matches(terms.cut_rx):
            doc.add(m.start(), m.end(), 'V22:b_cut')
    for m in _AFTER.finditer(t):                                   # (d) word after a role token
        i = m.end()
        seg = doc.seg_of(i, i + 1)
        if seg is None or not t[i].isalnum():
            continue
        a, b = word_span(t, i, i + 1, seg[0], seg[1])
        first = True
        while True:
            w = t[a:b]
            core = re.sub(r"['\u2019]s$", '', w)
            has_d, has_l = any(c.isdigit() for c in core), any(c.isalpha() for c in core)
            caps = len(core) >= 2 and core.isalpha() and core.isupper()
            cu = core.upper()
            ok = not (D_SKIP.match(core) or cu in STOP_AFTER or cu in GENERIC_ACR or cu in REUTERS_ABBR) and \
                ((has_d and has_l) or (caps and not eng_compound(core, eng)) or
                 (first and caps and mixed_line(t, a, b)) or
                 (first and core.isdigit() and 3 <= len(core) <= 5 and not UNIT_AFTER.match(t, b)) or
                 (not first and ((core.isdigit() and len(core) <= 5) or (caps and len(core) <= 3))))
            if not ok:
                break
            doc.add(a, b, 'V22:d_after_token')
            first = False
            j = b
            if j + 2 < seg[1] and t[j] == ',' and t[j + 1] == ' ' and t[j + 2].isalnum():
                a2, b2 = word_span(t, j + 2, j + 3, seg[0], seg[1])
                w2 = t[a2:b2]
                if any(c.isdigit() for c in w2) and any(c.isalpha() for c in w2):
                    a, b = a2, b2                          # a list of model codes ('G6, G7')
                    continue
                break
            if j + 1 >= seg[1] or t[j] not in ' -\u2010\u2011/' or not t[j + 1].isalnum():
                break
            a, b = word_span(t, j + 1, j + 2, seg[0], seg[1])
    for m in _UNIT.finditer(t):                                    # (d) a unit's company name after a role token
        ws = m.group('unit').split()
        if any(x.lower().strip('.') in FUNCTION for x in ws) or doc.seg_of(m.start('unit'), m.end('unit')) is None:
            continue
        doc.add(m.start('unit'), m.end('unit'), 'V22:d_unit_name')
    for m in _DEFACR.finditer(t):                                  # (d) an acronym defined after a hidden name
        acr = m.group('acr')
        if acr in GENERIC_ACR or acr in STOP_AFTER or acr in EXCH or sum(c.isupper() for c in acr) < 2 or \
                doc.seg_of(m.start('acr'), m.end('acr')) is None:
            continue
        for mm in doc.content_matches(re.compile(r'(?<![\w\[])' + re.escape(acr) + r'(?![\w\]])')):
            doc.add(mm.start(), mm.end(), 'V22:d_defined_acronym')
    for nm, rx in gb.items():                                      # (e) give-back phrases, any bracket token in the slot
        for m in rx.finditer(t):
            seg = doc.seg_of(m.start(), m.end())
            if seg is None:
                continue
            for a, b in words(t, m.start(), m.end()):
                if not _in_token(t, a):
                    doc.add(a, b, 'V22:e_giveback_' + nm, 'name')
            for s, e, tok in doc.roles:
                if m.start() <= s and e <= m.end():
                    doc.add_term(s, 'V22:e_giveback_' + nm)
    for s0, e0, tok in doc.roles:                                  # (e2) a firm named like a period code (Q2...)
        if doc.t2p.get(tok) in terms.pcode_permnos and doc.seg_of(s0, e0) is not None and \
                (_PSLOT_L.search(t[max(0, s0 - 14):s0]) or _PSLOT_R.match(t, e0) or
                 _PNEAR_L.search(t[max(0, s0 - 16):s0]) or _PNEAR_R.match(t, e0)):
            doc.add_term(s0, 'V22:e_period_code_name')
    for rx in _YEAR:                                               # (f) dates
        for m in rx.finditer(t):
            if doc.seg_of(m.start(1), m.end(1)) is not None:
                doc.add(m.start(1), m.end(1), 'V22:f_date', 'date')
    for rx in (_MON_BEFORE, _MON_AFTER):
        for m in rx.finditer(t):
            if doc.seg_of(m.start(1), m.end(1)) is not None:
                doc.add(m.start(1), m.end(1), 'V22:f_date', 'date')
    for m in _SWALLOW.finditer(t):
        if doc.seg_of(m.start(), m.end()) is not None:
            doc.add(m.start(), m.end(), 'V22:f_date', 'date')


def _in_token(t, a):
    j = t.rfind('[', 0, a + 1)
    return j >= 0 and t.find(']', j) >= a and not re.search(r'\s', t[j:a + 1])


def gb_context(doc, groups):
    """F3: the neighbour word of a frozen GB group slot (s17b's neighbour rule, as RD.loc_gb) is redacted with the
    slot, so the phrase no longer gives the hidden word back."""
    by_p = defaultdict(set)
    for p, w, side, nb in groups:
        by_p[p].add((side, nb))
    n = 0
    t = doc.text
    for i, f, fs, fe in doc.fields():
        for s, e, tok in doc.roles:
            if not (fs <= s and e <= fe):
                continue
            p = doc.t2p.get(tok)
            if p not in by_p:
                continue
            left, right = t[max(fs, s - 40):s], t[e:min(fe, e + 40)]
            lseg, rseg = RD._ANYTOK.split(left)[-1], RD._ANYTOK.split(right)[0]
            lw, rw = RD._words17(lseg), RD._words17(rseg)
            prev = lw[-1] if lw and re.search(r"[A-Za-z0-9'\u2019-]\s*$", lseg) else None
            nxt = rw[0] if rw and re.match(r"^[\s-]*[A-Za-z]", rseg) else None
            if ('L', prev) in by_p[p]:
                m = list(RD._WORD17.finditer(lseg))[-1]
                a = s - len(lseg) + m.start()
                n += doc.add(a, a + len(m.group(0)), 'V22:e_gb_context', 'name')
            if ('R', nxt) in by_p[p]:
                m = RD._WORD17.search(rseg)
                a = e + m.start()
                n += doc.add(a, a + len(m.group(0)), 'V22:e_gb_context', 'name')
    return n


def slot_context(doc, spans, eng):
    """(e) A [REDACTED] slot that hides a generic acronym (GENERIC_ACR, e.g. a ticker that is a common term such as
    'AI' or 'EV') inside a phrase (a content word right before and right after it, joined by one space or hyphen) is
    given back by the phrase; both neighbour words are redacted too."""
    t = doc.text
    n = 0
    for s, e, kind, cs in spans:
        if kind != 'name':
            continue
        w = t[s:e]
        if not (w.isalpha() and len(w) <= 4 and w.upper() in GENERIC_ACR):
            continue
        seg = doc.seg_of(s, e)
        if seg is None or s - 2 < seg[0] or e + 2 > seg[1]:
            continue
        if t[s - 1] not in ' -' or t[e] not in ' -' or not t[s - 2].isalpha() or not t[e + 1].isalpha():
            continue
        la, lb = word_span(t, s - 2, s - 1, seg[0], s - 1)
        ra, rb = word_span(t, e + 1, e + 2, e + 1, seg[1])
        lw_, rw_ = t[la:lb], t[ra:rb]
        if lw_.lower() in FUNCTION or rw_.lower() in FUNCTION or '[' in t[la - 1:la] or ']' in t[rb:rb + 1]:
            continue
        n += doc.add(la, lb, 'V22:e_slot_context', 'name') + doc.add(ra, rb, 'V22:e_slot_context', 'name')
    return n


# ----------------------------------------------------------------------------------------------------- one input
def process(key, v2_text, meta, prior, H, ctx, classes, terms, gb, eng):
    """All replacement entries of one input for this round, in masker-v2 coordinates. prior: earlier rounds' entries
    (offset, end, placeholder). Pass 0 locates the detector hits (on the round's text) and the F2 locators; passes
    1..ITER-1 repeat the F2 locators and the glue fix on the provisional text until nothing new is found. Returns
    (new entries [(s, e, ph, classes, orig_sha)], located {class: n}, passes)."""
    cur = sorted(prior)
    new = []
    located = {}
    passes = 0
    for it in range(ITER):
        text = RR.render(v2_text, cur)
        doc = RD.Doc(key, text, meta)
        if it == 0:
            if classes or (H is not None and H.kept):
                located = RD.locate(doc, classes, H, ctx)
                if 'MA1:GB_frozen' in classes:
                    located['MA1:GB_frozen'] = located.get('MA1:GB_kept', 0)
            if H is not None and H.kept:
                gb_context(doc, H.kept)
        locate_v22(doc, terms, gb, eng)
        widen(doc)
        spans = doc.final_spans()
        slot_context(doc, spans, eng)
        widen(doc)
        spans = doc.final_spans()
        glue_fix(doc, spans)
        widen(doc)
        spans = doc.final_spans()
        passes = it + 1
        if not spans:
            break
        if it == ITER - 1:
            passes = ITER + 1                             # not converged: the last pass still found spans
        base = list(cur)                                  # the pass's text = render(v2, base)
        for s, e, kind, cs in spans:
            ph = PH[kind]
            va, vb = RD.to_v2(base, s, e)
            new.append((va, vb, ph, sorted(cs), RD.sha_s(text[s:e])))
            cur.append((va, vb, ph))
        cur.sort()
    return new, located, passes


# ------------------------------------------------------------------------------------------------ round driver
def slim_meta(m):
    """The INPUT_META_COMPACT fields the locators read (token map and article alt_ids): the full records of a period
    take about 1 GB as Python objects, the slim ones a small fraction."""
    return {'key': m['key'], 'focal_permno': m['focal_permno'],
            'shown': [{'token': x['token'], 'permno': x['permno']} for x in m['shown']],
            'links': [{'token': x['token'], 'permno': x['permno']} for x in m['links']],
            'articles': [{'alt_id': x['alt_id']} for x in m['articles']]}


def meta_permnos(d, accept):
    """Focal and shown PERMNOs of the inputs whose key passes accept(key), streaming INPUT_META_COMPACT."""
    out = set()
    with (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            if accept(m['key']):
                out.add(int(m['focal_permno']))
                out.update(int(x['permno']) for x in m['shown'])
    return sorted(out)


def round_dir(period, rnd):
    return ROOT / str(period) / f'round_{rnd}'


def ledgers_upto(period, rnd):
    return [round_dir(period, r) / 'LEDGER.jsonl' for r in range(1, rnd + 1)]


def _residual_file(period, rnd):
    runs = sorted(round_dir(period, rnd).glob('RESIDUAL_*.json'))
    if not runs:
        raise PermissionError(f'v2.2 round {rnd} of {period}: detectors not re-run (residual missing)')
    return runs[-1]


def _round_records(period, rnd):
    """Records of the round's input text: masker-v2 (round 1) or rendered from the ledgers of rounds 1..rnd-1."""
    import td4v2_flags as FL
    d = FL._dir(period)
    return RR.iter_round(Path(d) / 'INPUTS.jsonl', ledgers_upto(period, rnd - 1), rnd - 1)


def redact_round(period, rnd, keys=None, out_root=None):
    """v2.2 round rnd. keys / out_root: development runs on a subset (written under out_root, never bound)."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    t0 = time.time()
    period, rnd = str(period), int(rnd)
    if not 1 <= rnd <= ROUNDS:
        raise ValueError('rounds 1-3')
    d = FL._dir(period)
    root = Path(out_root) if out_root else ROOT
    out = root / period / f'round_{rnd}'
    if out.exists():
        raise FileExistsError(str(out))
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    if rnd == 1:
        run = RD._round0_run(period)
        by_key, adj, n_call, cls_counts = RD.flagged_classes(period, d, run)
        runs_so_far = [str(run)]
    else:
        res = json.loads(_residual_file(period, rnd - 1).read_text(encoding='utf-8'))
        run = Path(res['run_dir'])
        by_key = defaultdict(set)
        for c, ks in res['by_class_keys'].items():
            for k in ks:
                by_key[k].add(c)
        adj, n_call, cls_counts = set(), res['callable'], res['by_class_inputs']
        runs_so_far = res['runs_so_far']
    groups = RD.kept_groups(runs_so_far)
    want = set(by_key) if keys is None else set(by_key) & set(keys)
    H = RD.Hits(run, want, period, T.OUT / 'inputs' / FL.STORES[period], groups)
    H.load_raw()
    ctx = {'pm_firm': RD._pm_firm(H.firms), 'M3': RD._m3_module()}
    gb = gb_patterns(run)
    eng = english()
    permnos = meta_permnos(d, lambda k: k not in pop and (keys is None or k in keys))
    profs, table_sha = build_profiles(period, permnos)
    tcache = {}
    prior = RR.ledger_index(ledgers_upto(period, rnd - 1)) if out_root is None else \
        RR.ledger_index([root / period / f'round_{r}' / 'LEDGER.jsonl' for r in range(1, rnd)])
    out.mkdir(parents=True, exist_ok=False)
    stats = Counter()
    by_class_spans, unlocated, per_input, passes_c = Counter(), defaultdict(list), [], Counter()
    not_conv = []
    fo = (out / 'INPUTS.jsonl').open('x', encoding='utf-8', newline='\n') if rnd == ROUNDS else None
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fv, \
            (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as fm, \
            (out / 'LEDGER.jsonl').open('x', encoding='utf-8', newline='\n') as fl, \
            (out / 'HASHES.jsonl').open('x', encoding='utf-8', newline='\n') as fh:
        for line, lm in zip(fv, fm):
            rec = json.loads(line)
            k = rec['key']
            if keys is not None and k not in keys:
                continue
            stats['records'] += 1
            ents = prior.get(k, [])
            if k in pop:
                rr = RR.render_record(rec, ents, rnd)
                fh.write(json.dumps({'key': k, 'input_sha256': rr['input_sha256'], 'input_bytes': rr['input_bytes'],
                                     'entries': len(ents)}) + '\n')
                if fo is not None:
                    fo.write(json.dumps(rr, ensure_ascii=False) + '\n')
                continue
            m = slim_meta(json.loads(lm))
            if m['key'] != k:
                raise ValueError('INPUTS and INPUT_META_COMPACT misaligned')
            fs = frozenset([int(m['focal_permno'])] + [int(s['permno']) for s in m['shown']])
            if fs not in tcache:
                if len(tcache) > TCACHE:
                    tcache.clear()
                    re.purge()
                tcache[fs] = Terms([profs[p] for p in fs if p in profs], eng)
            classes = by_key.get(k, set())
            if rnd > 1 and k not in by_key:
                # not flagged by the round rnd-1 re-run (reviewers' detectors, frozen GB groups, F2 residual): its
                # text is round rnd-1's output, on which process() reached its fixpoint with the same name tables,
                # GB patterns and locators, so process() would add nothing; skipped (checked by idempotency22).
                new, located, passes = [], {}, 0
                stats['skipped_unflagged'] += 1
            else:
                new, located, passes = process(k, rec['input_text'], m, ents, H if k in want or groups else None,
                                               ctx, classes, tcache[fs], gb, eng)
            passes_c[passes] += 1
            if passes > ITER:
                not_conv.append(k)
            for c in classes:
                if c.startswith('V22:'):          # F2 residual of the previous round: re-located by process() itself,
                    continue                      # checked again by this round's residual (still flagged after round 3)
                if located.get(c, 0) == 0:
                    unlocated[k].append(c)
            if new:
                v2t = rec['input_text']
                segs = RD.Doc(k, v2t, m)
                for s, e, ph, cs, osha in new:
                    seg = segs.seg_of(s, e)
                    fl.write(json.dumps({'key': k, 'article': segs.arts[seg[2]][0] if seg else None,
                                         'field': seg[3] if seg else None, 'offset': s, 'length': e - s,
                                         'placeholder': ph, 'classes': cs, 'original_sha256': RD.sha_s(v2t[s:e]),
                                         'round': rnd}) + '\n')
                    stats['spans_' + ph] += 1
                    for c in cs:
                        by_class_spans[c] += 1
                per_input.append(len(new))
                stats['inputs_with_new_redaction'] += 1
            rr = RR.render_record(rec, ents + [(s, e, ph) for s, e, ph, _, _ in new], rnd)
            fh.write(json.dumps({'key': k, 'input_sha256': rr['input_sha256'], 'input_bytes': rr['input_bytes'],
                                 'entries': len(ents) + len(new)}) + '\n')
            if fo is not None:
                fo.write(json.dumps(rr, ensure_ascii=False) + '\n')
    if fo is not None:
        fo.close()
    res = {'period': period, 'round': rnd, 'version': 'v2.2', 'detector_run': str(run), 'runs_so_far': runs_so_far,
           'records': stats['records'], 'callable': n_call, 'flagged_inputs_entering': len(set(by_key) - pop),
           'flag_classes_entering': cls_counts, 'gb_frozen_groups': len(groups),
           'adjudicated_withheld': sorted(adj), 'inputs_with_new_redaction': stats['inputs_with_new_redaction'],
           'skipped_unflagged': stats['skipped_unflagged'],
           'new_spans': sum(per_input), 'spans_by_placeholder': {k[6:]: v for k, v in stats.items() if k.startswith('spans_')},
           'spans_by_class': dict(sorted(by_class_spans.items())), 'passes': dict(passes_c),
           'not_converged_keys': sorted(not_conv),
           'unlocated_inputs': len(unlocated), 'unlocated': {k: sorted(v) for k, v in sorted(unlocated.items())},
           'unlocated_by_class': dict(Counter(c for v in unlocated.values() for c in v)),
           'ledger': str(out / 'LEDGER.jsonl'), 'ledger_sha256': R.sha(out / 'LEDGER.jsonl'),
           'hashes': str(out / 'HASHES.jsonl'), 'hashes_sha256': R.sha(out / 'HASHES.jsonl'),
           'inputs': str(out / 'INPUTS.jsonl') if rnd == ROUNDS else None,
           'inputs_sha256': R.sha(out / 'INPUTS.jsonl') if rnd == ROUNDS else None,
           'storage': 'ledger and full INPUTS' if rnd == ROUNDS else 'ledger only (render with td4v2_render)',
           'name_tables_sha256': table_sha, 'dev_subset': None if keys is None else len(keys),
           'code_sha256': {'td4v2_redact22.py': R.sha(Path(__file__)), 'td4v2_redact.py': R.sha(Path(RD.__file__)),
                           'td4v2_render.py': R.sha(Path(RR.__file__))},
           'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(out / 'ROUND.json', res)
    return {k: v for k, v in res.items() if k not in ('unlocated', 'adjudicated_withheld')}


# ------------------------------------------------------------------------------------- detector re-runs and residuals
_EXTRACT = {}


def extract_records(period, source, run):
    """td4v2_flags.extract with its one INPUTS read replaced by `source` (a file-like reader of the round's records,
    e.g. td4v2_render.open_round for a ledger-only round); the extraction logic is FL.extract's own source."""
    import inspect
    import td4v2_flags as FL
    if 'f' not in _EXTRACT:
        src = inspect.getsource(FL.extract)
        old = "with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fh:"
        if src.count(old) != 1:
            raise RuntimeError('td4v2_flags.extract changed')
        ns = dict(vars(FL))
        exec(compile(src.replace(old, 'with _SOURCE(d) as fh:').replace('def extract(', 'def extract_v22(', 1),
                     FL.__file__, 'exec'), ns)
        _EXTRACT['f'] = ns
    ns = _EXTRACT['f']
    ns['_SOURCE'] = lambda d: source
    return ns['extract_v22'](str(period), None, Path(run))


def round_source(period, rnd, mode='r'):
    """Reader of the round-rnd text: the stored INPUTS of the final round, else rendered from the ledgers."""
    import td4v2_flags as FL
    if rnd == ROUNDS and (round_dir(period, rnd) / 'INPUTS.jsonl').exists():
        if 'b' in mode:
            return (round_dir(period, rnd) / 'INPUTS.jsonl').open('rb')
        return (round_dir(period, rnd) / 'INPUTS.jsonl').open('r', encoding='utf-8')
    return RR.open_round(Path(FL._dir(period)) / 'INPUTS.jsonl', ledgers_upto(period, rnd), mode, rnd)


def prepare_rerun(period, rnd):
    """Detector re-run on the v2.2 round-rnd text: td4v2_flags copies whose INPUTS open is replaced by a reader of
    the round (ledger rendering for rounds 1-2, the stored file for round 3); population, metadata, article store
    and withheld set of the masker-v2 build."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    period, rnd = str(period), int(rnd)
    d = FL._dir(period)
    if not (round_dir(period, rnd) / 'ROUND.json').exists():
        raise PermissionError('round not written')
    v2in = (Path(d) / 'INPUTS.jsonl').as_posix()
    leds = [p.as_posix() for p in ledgers_upto(period, rnd)]
    final = (round_dir(period, rnd) / 'INPUTS.jsonl').as_posix()
    code = Path(__file__).resolve().parent.as_posix()
    if rnd == ROUNDS:
        rd_b = "open(r'" + final + "', 'rb')"
        rd_t = 'open(r"' + final + '", encoding="utf-8")'
    else:
        rd_b = "_RR.open_round(r'" + v2in + "', " + repr(leds) + ", 'rb')"
        rd_t = '_RR.open_round(r"' + v2in + '", ' + repr(leds) + ', "r")'
    imp = "import sys as _sys\n_sys.path.insert(0, r'" + code + "')\nimport td4v2_render as _RR\n"
    wf = [IV.withheld_file(d)]
    store = T.OUT / 'inputs' / FL.STORES[period]
    run = FL.ROOT / f'{period}_v22rd{rnd}_{R.utc().replace(":", "")}'
    r1, r2 = run / 'ma1', run / 'ma2'
    r1.mkdir(parents=True, exist_ok=False)
    r2.mkdir(parents=True, exist_ok=False)
    dq, wh = Path(d).as_posix(), [Path(x).as_posix() for x in wf]
    c = (FL.MA1 / 'ma1_common.py').read_text(encoding='utf-8')
    c = FL._sub(c, [("D = Path('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3')", f"D = Path('{dq}')"),
                    ("OUT = Path('D:/OptResearch_data/five_session_review/td4m2/ma1')", f"OUT = Path('{r1.as_posix()}')"),
                    ("json.loads((D / fn).read_text(encoding='utf-8'))", "json.loads(Path(fn).read_text(encoding='utf-8'))"),
                    ("(D / 'INPUTS.jsonl').open('rb')", rd_b),
                    ("from pathlib import Path\n", "from pathlib import Path\n" + imp)])
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
            i = s.index('\nimport ')
            s = s[:i + 1] + imp + s[i + 1:]
            s = FL._sub(s, [('open(D + "INPUTS.jsonl", encoding="utf-8")', rd_t)])
        FL._write(r2 / fn, s)
    sh = FL.run_plan(run)
    return {'run_dir': str(run), 'callable': len(keys), 'run_script': str(sh),
            'reader': 'stored INPUTS' if rnd == ROUNDS else 'td4v2_render.open_round (ledger-only round)',
            'next': f'bash {sh.as_posix()} ; then run_td4v2.py redact22-residual --period {period} --which {run.name}'}


def literal_placeholder_hit(run_dir, key, cls):
    """F3: the only waiver. True when every hit of an M4 class for the key is the word inside a placeholder
    ('TERM' of '[TERM]', 'DATE' of '[DATE]', 'REDACTED' of '[REDACTED]'); GB_* hits are never waived."""
    import td4v2_redact_final as RF
    if not cls.startswith('M4:') or cls.startswith('M4:GB_') or cls in RD.FIRM_LEVEL:
        return False
    run_dir = str(run_dir)
    if run_dir not in RF._M4:
        idx = {}
        for h in RD._jsonl(Path(run_dir) / 'ma1' / 's13_m4_hits.jsonl'):
            idx.setdefault(h['key'], []).append(h)
        RF._M4[run_dir] = idx
    hs = [h for h in RF._M4[run_dir].get(key, []) if 'M4:' + h['class'] == cls]
    return bool(hs) and all(('[' + (h.get('match') or '') + ']') in PLACEHOLDERS and
                            ('[' + (h.get('match') or '') + ']') in h.get('ctx', '') for h in hs)


def v22_scan(doc, terms, gb, eng):
    """Classes of the F2 locators that would still add a span to the doc's text (0 = nothing left)."""
    locate_v22(doc, terms, gb, eng)
    widen(doc)
    sp = doc.final_spans()
    glue_fix(doc, sp)
    widen(doc)
    return {c for _s, _e, _k, cs in doc.final_spans() for c in cs}


def residual(period, rnd, run_name):
    """Still-flagged inputs after the v2.2 round-rnd re-run: td4v2_flags.extract on the re-run (read through the
    round reader), the frozen GB groups, and the F2 locators on the round's text (any span found = still flagged)."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    period, rnd = str(period), int(rnd)
    run = FL.ROOT / run_name
    if not run.name.startswith(f'{period}_v22rd{rnd}_'):
        raise ValueError('run folder of another period or round')
    t0 = time.time()
    status = FL.run_status(run)
    with round_source(period, rnd, 'r') as src:
        cls, union, n_call, adjud = extract_records(period, src, run)
    by_key = defaultdict(set)
    for c, ks in cls.items():
        if c in FL.REPORT_ONLY or c.startswith('ADJ_'):
            continue
        for k in ks:
            by_key[k].add(c)
    rd = round_dir(period, rnd)
    prev = json.loads((rd / 'ROUND.json').read_text(encoding='utf-8'))['runs_so_far']
    runs_so_far = [str(x) for x in prev] + [str(run)]
    groups = RD.kept_groups(runs_so_far)
    d = FL._dir(period)
    callable_keys = set(json.loads((run / 'ma2' / 'MA2_CALLABLE_KEYS.json').read_text(encoding='utf-8')))
    profs, _ = build_profiles(period, meta_permnos(d, lambda k: k in callable_keys))
    gb = gb_patterns(run)
    eng = english()
    gbf, v22 = set(), defaultdict(set)
    # F2 locators: process() ran them to a fixpoint on this round's text, so their residual is the inputs whose
    # fixpoint was not reached within ITER passes (ROUND.json not_converged_keys); a rescan of the first
    # RESCAN_N callable inputs (file order) checks that they find nothing more.
    for k in json.loads((rd / 'ROUND.json').read_text(encoding='utf-8')).get('not_converged_keys', []):
        v22[k].add('V22:not_converged')
    rescan = Counter()
    fm = (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8')
    for r, lm in zip(RR.iter_round(Path(d) / 'INPUTS.jsonl', ledgers_upto(period, rnd), rnd), fm):
        k = r['key']
        if k not in callable_keys:
            continue
        m = slim_meta(json.loads(lm))
        if m['key'] != k:
            raise ValueError('INPUTS and INPUT_META_COMPACT misaligned')
        doc = RD.Doc(k, r['input_text'], m)
        if RD.loc_gb(doc, groups, 'MA1:GB_frozen'):
            gbf.add(k)
        if rescan['inputs'] < RESCAN_N:
            fs = [int(m['focal_permno'])] + [int(s['permno']) for s in m['shown']]
            rescan['inputs'] += 1
            for c in v22_scan(RD.Doc(k, r['input_text'], m), Terms([profs[p] for p in fs if p in profs], eng), gb, eng):
                v22[k].add(c)
                rescan['found'] += 1
    fm.close()
    for k in gbf:
        by_key[k].add('MA1:GB_frozen')
    for k, cs in v22.items():
        by_key[k] |= cs
    by_class = defaultdict(list)
    for k, cs in by_key.items():
        for c in cs:
            by_class[c].append(k)
    res = {'period': period, 'round': rnd, 'version': 'v2.2', 'run_dir': str(run), 'runs_so_far': runs_so_far,
           'callable': n_call, 'still_flagged_inputs': len(by_key),
           'still_flagged_share': round(len(by_key) / n_call, 5) if n_call else None,
           'by_class_inputs': {c: len(v) for c, v in sorted(by_class.items())},
           'gb_frozen_groups': len(groups), 'gb_frozen_inputs': len(gbf), 'v22_locator_inputs': len(v22),
           'v22_rescan': dict(rescan),
           'by_class_keys': {c: sorted(v) for c, v in sorted(by_class.items())},
           'scripts_status': status, 'run_copies_sha256': {str(p): FL.sha(p) for p in sorted(run.rglob('*.py'))},
           'code_sha256': R.sha(Path(__file__)), 'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(rd / f'RESIDUAL_{R.utc().replace(":", "")}.json', res)
    return {k: v for k, v in res.items() if k not in ('by_class_keys', 'run_copies_sha256')}


# ------------------------------------------------------------------------------------------- verification and final
def _ledger_full(period, rnd):
    led = defaultdict(list)
    for r in range(1, rnd + 1):
        for e in RD._jsonl(round_dir(period, r) / 'LEDGER.jsonl'):
            led[e['key']].append((e['offset'], e['offset'] + e['length'], e['placeholder'], e['original_sha256']))
    return led


def p6_entries(text_v2, entries):
    """Ledger entries whose placeholder has a letter or digit right before or after it in the rendered text."""
    n, shift, out = 0, 0, RR.render(text_v2, entries)
    for s, e, ph, *_ in sorted(entries):
        q = s + shift
        lft = out[q - 1] if q > 0 else ''
        rgt = out[q + len(ph)] if q + len(ph) < len(out) else ''
        n += lft.isalnum() or rgt.isalnum()
        shift += len(ph) - (e - s)
    return n


def verify_round22(period, rnd, idem_n=2000):
    """Every record of round rnd: the ledger-only rendering reproduces the round's HASHES (and, for round 3, the
    stored INPUTS text byte for byte); RD.check_record invariants on the cumulative ledger; no ledger entry for a
    population-withheld input; P6 (placeholder next to a letter or digit) per entry; and the idempotency check (the
    round's own process() on the first idem_n redacted callable inputs finds nothing more)."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    period, rnd = str(period), int(rnd)
    t0 = time.time()
    d = FL._dir(period)
    rd = round_dir(period, rnd)
    led = _ledger_full(period, rnd)
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    hashes = {}
    for h in RD._jsonl(rd / 'HASHES.jsonl'):
        hashes[h['key']] = (h['input_sha256'], h['input_bytes'])
    probs, p6 = Counter(), Counter()
    bad = []
    n = n_red = 0
    stored = (rd / 'INPUTS.jsonl').open(encoding='utf-8') if rnd == ROUNDS else None
    try:
        with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fv:
            for lv in fv:
                v = json.loads(lv)
                k = v['key']
                n += 1
                ents = led.get(k, [])
                text = RR.render(v['input_text'], ents)
                p = RD.check_record(v['input_text'], text, ents)
                if k in pop and ents:
                    p.append('population_withheld_redacted')
                hb = text.encode('utf-8')
                if hashes.get(k) != (hashlib.sha256(hb).hexdigest(), len(hb)):
                    p.append('hash_differs')
                if stored is not None:
                    r = json.loads(stored.readline())
                    if r['key'] != k or r['input_text'] != text or r.get('v2_input_sha256') != v['input_sha256'] or \
                            r['input_sha256'] != hashlib.sha256(hb).hexdigest():
                        p.append('stored_round_differs')
                g = p6_entries(v['input_text'], ents) if ents else 0
                if g and k not in pop:
                    p6['entries'] += g
                    p6['inputs'] += 1
                for x in p:
                    probs[x] += 1
                if p and len(bad) < 50:
                    bad.append(k)
                n_red += bool(ents)
        if stored is not None and stored.readline():
            probs['record_count'] += 1
    finally:
        if stored is not None:
            stored.close()
    if n != len(hashes):
        probs['hash_count'] += 1
    idem = idempotency22(period, rnd, idem_n)
    res = {'period': period, 'round': rnd, 'version': 'v2.2', 'records': n, 'records_with_ledger_entries': n_red,
           'ledger_entries': sum(len(v) for v in led.values()), 'problems': dict(probs), 'problem_keys': bad,
           'p6_callable': dict(p6), 'clean': not probs and idem['new_spans'] == 0 and not p6,
           'storage': 'stored INPUTS checked' if rnd == ROUNDS else 'ledger-only round rendered and hash-checked',
           'idempotency': idem, 'code_sha256': R.sha(Path(__file__)), 'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(rd / f'VERIFY22_{R.utc().replace(":", "")}.json', res)
    return res


def idempotency22(period, rnd, n_max):
    """process() of round rnd applied again (same detector outputs, frozen groups, F2 locators) to the round's output
    text of the first n_max redacted callable inputs (ledger order): further spans found (0 = idempotent)."""
    import td4v2_flags as FL
    import td4v2_inputs as IV
    d = FL._dir(period)
    rd = round_dir(period, rnd)
    info = json.loads((rd / 'ROUND.json').read_text(encoding='utf-8'))
    run = Path(info['detector_run'])
    if rnd == 1:
        by_key = RD.flagged_classes(period, d, run)[0]
    else:
        res = json.loads(_residual_file(period, rnd - 1).read_text(encoding='utf-8'))
        by_key = defaultdict(set)
        for c, ks in res['by_class_keys'].items():
            for k in ks:
                by_key[k].add(c)
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    order = []
    for e in RD._jsonl(rd / 'LEDGER.jsonl'):
        if (not order or order[-1] != e['key']) and e['key'] not in pop:
            order.append(e['key'])
        if len(order) > n_max:
            break
    keys = set(order[:n_max])
    if rnd > 1:                    # rounds 2-3 skip unflagged inputs: check the first n_max/2 of them as well
        extra = []
        with (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as f:
            for line in f:
                k = json.loads(line)['key']
                if k not in pop and k not in by_key and k not in keys:
                    extra.append(k)
                    if len(extra) >= n_max // 2:
                        break
        keys |= set(extra)
    led = _ledger_full(period, rnd)
    groups = RD.kept_groups(info['runs_so_far'])
    H = RD.Hits(run, set(by_key) & keys, period, T.OUT / 'inputs' / FL.STORES[period], groups)
    H.load_raw()
    ctx = {'pm_firm': RD._pm_firm(H.firms), 'M3': RD._m3_module()}
    gb = gb_patterns(run)
    eng = english()
    metas = {}
    with (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            if m['key'] in keys:
                metas[m['key']] = slim_meta(m)
    permnos = sorted({p for m in metas.values() for p in [int(m['focal_permno'])] + [int(s['permno']) for s in m['shown']]})
    profs, _ = build_profiles(period, permnos)
    checked = new = 0
    examples = []
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fv:
        for line in fv:
            rec = json.loads(line)
            k = rec['key']
            if k not in keys:
                continue
            m = metas[k]
            fs = [int(m['focal_permno'])] + [int(s['permno']) for s in m['shown']]
            terms = Terms([profs[p] for p in fs if p in profs], eng)
            prior = [x[:3] for x in led.get(k, [])]
            nw, _, _ = process(k, rec['input_text'], m, prior, H, ctx, by_key.get(k, set()), terms, gb, eng)
            new += len(nw)
            if nw and len(examples) < 10:
                examples.append([k, sorted({c for x in nw for c in x[3]})])
            checked += 1
    return {'checked_inputs': checked, 'new_spans': new, 'examples_keys_classes': examples}


def _waived(run_dir, k, c):
    return literal_placeholder_hit(run_dir, k, c)


def finalize22(period):
    """REDACTION_FINAL_v2_2: withheld (every arm alike, before any provider call) = the adjudicated 2021-2022 key sets
    | inputs with a flagged class that could not be located in any round | inputs still flagged after the round-3
    re-run (reviewers' detectors, frozen GB groups, the F2 locators) | MASKER_NO_SOURCE (2023-2025, as v2.1). For
    2023-2025 the M4 firm-level classes NONAME / NONAME_AFTER_2022 do not withhold (rule v2.1); for 2021-2022 they do
    (rule v2, unchanged). No placeholder waiver for M4 GB_* hits; the only waiver is a hit that exists only because of
    a placeholder's own characters (literal_placeholder_hit), listed per class with its count."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    import td4v2_nosource as NS
    import td4v2_spec as S
    period = str(period)
    d = Path(FL._dir(period))
    rounds = [json.loads((round_dir(period, r) / 'ROUND.json').read_text(encoding='utf-8')) for r in range(1, ROUNDS + 1)]
    resid_f = [_residual_file(period, r) for r in range(1, ROUNDS + 1)]
    resid = [json.loads(f.read_text(encoding='utf-8')) for f in resid_f]
    ver = []
    for r in range(1, ROUNDS + 1):
        vs = sorted(round_dir(period, r).glob('VERIFY22_*.json'))
        if not vs or not json.loads(vs[-1].read_text(encoding='utf-8'))['clean']:
            raise PermissionError(f'round {r} verification missing or not clean')
        ver.append(vs[-1])
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    callable_keys = []
    with (d / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            k = json.loads(line)['key']
            if k not in pop:
                callable_keys.append(k)
    cset = set(callable_keys)
    n_call = len(callable_keys)
    sel = period == str(S.SELECT)
    counts_firm_level = sel                              # rule v2 for 2021-2022, rule v2.1 for 2023-2025
    adj = set(rounds[0]['adjudicated_withheld']) & cset
    waived = Counter()
    unloc = set()
    for i, r in enumerate(rounds):
        for k, cs in r['unlocated'].items():
            keep = []
            for c in cs:
                if c in RD.FIRM_LEVEL and not counts_firm_level:
                    continue
                if _waived(r['detector_run'], k, c):
                    waived[f'round {i + 1} unlocated {c}'] += 1
                    continue
                keep.append(c)
            if keep:
                unloc.add(k)
    still3, still_cls = set(), Counter()
    for c, ks in resid[-1]['by_class_keys'].items():
        for k in ks:
            if c in RD.FIRM_LEVEL and not counts_firm_level:
                continue
            if _waived(resid[-1]['run_dir'], k, c):
                waived[f'round 3 still flagged {c}'] += 1
                continue
            still3.add(k)
            still_cls[c] += 1
    unloc &= cset
    still3 &= cset
    ns_keys = set()
    nsp = None
    if not sel:
        nsp = NS.latest(d)
        ns_keys = set(json.loads(nsp.read_text(encoding='utf-8'))['flagged_keys']) & cset
    withheld = adj | unloc | still3 | ns_keys
    per_key, kinds, per_round = Counter(), Counter(), {}
    for r in range(1, ROUNDS + 1):
        n_r = 0
        for e in RD._jsonl(round_dir(period, r) / 'LEDGER.jsonl'):
            if e['key'] in cset:
                per_key[e['key']] += 1
                kinds[e['placeholder']] += 1
                n_r += 1
        per_round[str(r)] = n_r
    counts_all = [per_key.get(k, 0) for k in callable_keys]
    counts_red = [c for c in counts_all if c]
    sub = None
    if sel:
        keep = IV.subsample(S.SELECT)
        sc = [k for k in callable_keys if k in keep]
        sub = {'subsample_callable': len(sc), 'withheld_in_subsample': len(withheld & set(sc)),
               'withheld_share_subsample': round(len(withheld & set(sc)) / max(1, len(sc)), 5),
               'sent_in_subsample': len(set(sc) - withheld)}
    final_inputs = round_dir(period, ROUNDS) / 'INPUTS.jsonl'
    res = {'period': period, 'rule_version': 'v2.2',
           'rule': 'span-level redaction v2.2 (td4v2_redact22: F1 word boundaries, F2 builder locators, F3 no GB '
                   'waiver), three rounds, rule A.4; 2021-2022 rule v2 withholding classes, 2023-2025 rule v2.1 '
                   '(MASKER_NO_SOURCE replaces the M4 firm-level classes) (PREREGISTRATION_v2_2)',
           'build_dir': str(d), 'population_check_file': str(IV.withheld_file(d)),
           'final_inputs': str(final_inputs), 'final_inputs_sha256': R.sha(final_inputs),
           'callable': n_call, 'inputs_with_redaction': len(counts_red),
           'inputs_with_redaction_share': round(len(counts_red) / n_call, 5),
           'redactions_per_input_all': {'mean': round(sum(counts_all) / n_call, 3), 'p50': RF_q(counts_all, 0.5),
                                        'p95': RF_q(counts_all, 0.95)},
           'replacements_by_placeholder': dict(kinds), 'ledger_entries_by_round': per_round,
           'flagged_before_round_1': rounds[0]['flagged_inputs_entering'],
           'residual_after_round': {str(r['round']): r['still_flagged_inputs'] for r in resid},
           'residual_share_after_round': {str(r['round']): r['still_flagged_share'] for r in resid},
           'residual_by_class_after_round_3': resid[-1]['by_class_inputs'],
           'unlocated_by_round': {str(r['round']): r['unlocated_inputs'] for r in rounds},
           'unlocated_by_class_by_round': {str(r['round']): r['unlocated_by_class'] for r in rounds},
           'withheld_reasons': {'adjudicated': len(adj), 'unlocatable': len(unloc),
                                'still_flagged_after_round_3': len(still3), 'masker_no_source': len(ns_keys)},
           'still_flagged_after_round_3_by_class': dict(sorted(still_cls.items())),
           'firm_level_classes_withhold': counts_firm_level,
           'waived_literal_placeholder_hits': dict(sorted(waived.items())),
           'gb_waiver': 'none (F3): every M4 GB_* hit after round 3 is still flagged',
           'withheld_keys': sorted(withheld), 'withheld_inputs': len(withheld),
           'withheld_share': round(len(withheld) / n_call, 5), 'subsample': sub,
           'masker_no_source_file': str(nsp) if nsp else None, 'masker_no_source_sha256': R.sha(nsp) if nsp else None,
           'rounds': [{k: v for k, v in r.items() if k not in ('unlocated', 'adjudicated_withheld')} for r in rounds],
           'residual_files': [str(f) for f in resid_f], 'residual_files_sha256': [R.sha(f) for f in resid_f],
           'verify_files': [str(f) for f in ver], 'verify_files_sha256': [R.sha(f) for f in ver],
           'detector_reruns': [r['run_dir'] for r in resid],
           'ledgers_sha256': {str(round_dir(period, r) / 'LEDGER.jsonl'): R.sha(round_dir(period, r) / 'LEDGER.jsonl')
                              for r in range(1, ROUNDS + 1)},
           'code_sha256': {'td4v2_redact22.py': R.sha(Path(__file__)), 'td4v2_redact.py': R.sha(Path(RD.__file__)),
                           'td4v2_render.py': R.sha(Path(RR.__file__))},
           'note': 'withheld for every arm alike before any provider call; no outcome or provider output is read',
           **R.peak()}
    out = d / f'REDACTION_FINAL_v2_2_{R.utc().replace(":", "")}.json'
    R.new_json(out, res)
    return {k: v for k, v in res.items() if k not in ('withheld_keys', 'rounds')}


def RF_q(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]
