"""TD4 v2.2 redaction checks on the sent texts (counts only; private outputs; no text is printed).

Reproduces, from their written definitions, the classes of the TD4M3 redaction review's population scan (s04 / s04b)
and invariant P6 (s02), and measures the over-redaction:

  D1   '[YEAR]' followed by a two-digit year ('[YEAR]-22', '[YEAR]/23');
  D2   month name, '[DATE]', day number ('March [DATE] 25');
  N1   '[REDACTED]' or '[TERM]' glued to letters on both sides; N1b one side (a possessive "'s" excepted);
  N7   a shown firm's name with its first letter cut ('ORVEX DYNAMICS'): first significant CRSP name word (>= 5 letters)
       without its first letter, a space, the second significant word; upper case as in the tables;
  N4er a word of >= 7 letters that contains a distinctive own name word (>= 5 letters, not an English word or
       inflection, not generic) and is neither that word nor an English word;
  P6   a ledger placeholder with a letter or digit right before or right after it in the sent text.
Name tables as in the review: the focal and shown firms' own CRSP stocknames/names rows dated on or before the build's
last day, plus CIZ 2025 for 2025 (read through td4v2_nosource.tables, the receipt-bound copies). The scanned part of
each text is everything after the PARTNER NEWS line, as in the review.

Over-redaction: article word tokens (runs of ASCII letters and digits in headline and text fields of the masker-v2
text, bracket tokens excluded) that a ledger span overlaps, as a share of all article word tokens of the sent inputs;
computed for v2.1 (rounds 1-3 of td4v2_redact) and v2.2 (rounds 1-3 of td4v2_redact22), each on its own sent set and
on the inputs sent in both. Reads no label, return or provider output.
"""
import bisect
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import td4_env as E
import td4v2_redact as RD
import td4v2_render as RR

MON = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|'
       r'Nov(?:ember)?|Dec(?:ember)?)')
RX = {'D1': re.compile(r'\[YEAR\]\s?[-–/]\s?\d{2}(?!\d)'),
      'D2': re.compile(r'(?<![A-Za-z])' + MON + r'\.?\s\[DATE\],?\s\d{1,2}(?![\d,.%])'),
      'N1': re.compile(r'[A-Za-z]\[(?:REDACTED|TERM)\][A-Za-z]'),
      'N1b': re.compile(r"[A-Za-z]\[(?:REDACTED|TERM)\]|\[(?:REDACTED|TERM)\](?!['’]s\b)[A-Za-z]")}
WORD7 = re.compile(r'(?<![A-Za-z])[A-Za-z]{7,}(?![A-Za-z])')
TOKEN = re.compile(r'\[[A-Z0-9_/]+\]|[A-Za-z0-9]+')
LEGAL = set('INC CORP CORPORATION CO COMPANY COS LTD LIMITED PLC LLC LP L P NV N V SA S A AG SE AB ASA OYJ SPA BV THE NEW '
            'DEL DE HLDGS HOLDINGS HOLDING GROUP GRP OF AND & TRUST ADR CL CLASS INTL INTERNATIONAL'.split())
EXTRA_ENG = set('online internet website websites app apps smartphone smartphones email emails ecommerce startup '
                'startups biotech fintech blockchain crypto cryptocurrency covid vaccine vaccines streaming broadband '
                'wireless pharma healthcare semiconductor semiconductors software hardware chipmaker chipmakers '
                'automaker automakers airline airlines biopharma telecom telecoms cybersecurity datacenter rideshare '
                'ecommerce esports offshore onshore downstream upstream midstream'.split())
GENERIC = set('american national united general first global international energy financial bank bancorp capital group '
              'holdings technologies technology systems services solutions industries pharmaceuticals therapeutics '
              'communications resources partners properties realty trust health healthcare medical brands foods '
              'products enterprises networks software semiconductor electronics motors airlines entertainment media '
              'insurance investment investments acquisition corp company inc ltd north south east west new data '
              'digital power oil gas petroleum mining minerals steel chemical chemicals materials biosciences bio '
              'labs laboratories sciences pharma devices instruments retail stores restaurants hotels resorts '
              'transportation logistics marine air gold silver copper water electric utilities'.split())


def is_eng(w, eng):
    lw = w.lower()
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


def reviewer_profiles(period, permnos, eng):
    """{permno: {'dw', 'cut'}} with the review's s04/s04b rules on the own-PERMNO CRSP rows (<= last day) and CIZ 2025."""
    import td4v2_nosource as NS
    tb = NS.tables(period, permnos)
    last = tb['last_day']
    out = {}
    for p in permnos:
        names = {c.strip().upper() for d, c, t in tb['names'].get(p, []) if d <= last and c}
        if str(period) == '2025':
            names |= {' '.join(n.upper().split()) for d, n, t in tb['ciz'].get(p, ()) if n}
        dw, cut = set(), set()
        for n in names:
            ws = [w for w in re.split(r"[^A-Za-z0-9&]+", n.upper()) if w]
            sig = [w for w in ws if w not in LEGAL]
            for w in sig:
                if len(w) >= 4 and w.isalpha() and w.lower() not in eng and w.lower() not in GENERIC:
                    dw.add(w)
            if len(sig) >= 2 and len(sig[0]) >= 5 and sig[0].isalpha():
                cut.add(sig[0][1:] + ' ' + sig[1])
        out[p] = {'dw': {w for w in dw if len(w) >= 5 and not is_eng(w, eng)}, 'cut': cut}
    return out, tb['files_sha256']


def _alt(ws):
    return '|'.join(re.escape(w) for w in sorted(ws, key=len, reverse=True))


def render_pos(v2, entries):
    """render() plus the start of each placeholder in the rendered text (entries sorted by offset)."""
    ents = sorted(entries)
    out, pos, i, n = [], [], 0, 0
    for s, e, ph in ents:
        out.append(v2[i:s])
        n += s - i
        pos.append(n)
        out.append(ph)
        n += len(ph)
        i = e
    out.append(v2[i:])
    return ''.join(out), ents, pos


def scan_version(period, ledgers, sent, label, keys=None):
    """Counts of the review classes, P6, placeholders and the redacted share of article word tokens on the sent inputs
    of one version (texts rendered from the masker-v2 INPUTS and the version's cumulative ledgers)."""
    import td4v2_flags as FL
    import td4v2_redact22 as R22
    t0 = time.time()
    d = FL._dir(period)
    eng = R22.english()
    led = RR.ledger_index(ledgers)
    want = (lambda k: k in sent) if keys is None else (lambda k: k in sent and k in keys)
    prof, tsha = reviewer_profiles(period, R22.meta_permnos(d, want), eng)
    rxc, memo = {}, {}

    def prx(p):
        if p not in rxc:
            pr = prof.get(p, {'dw': set(), 'cut': set()})
            rxc[p] = (re.compile(r'(?<![A-Za-z])(' + _alt(pr['cut']) + r')\b') if pr['cut'] else None, pr['dw'])
        return rxc[p]

    def engc(w):
        if w not in memo:
            memo[w] = is_eng(w, eng)
        return memo[w]

    inputs_by, hits, ph_entries, ph_text = defaultdict(set), Counter(), Counter(), Counter()
    p6 = Counter()
    tok_all = tok_red = 0
    per_key_tok = {}
    n = 0
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fv, \
            (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as fm:
        for line, lm in zip(fv, fm):
            r = json.loads(line)
            k = r['key']
            if not want(k):
                continue
            m = R22.slim_meta(json.loads(lm))
            if m['key'] != k:
                raise ValueError('INPUTS and INPUT_META_COMPACT misaligned')
            n += 1
            v2 = r['input_text']
            text, ents, pos = render_pos(v2, led.get(k, []))
            for (s, e, ph), q in zip(ents, pos):
                ph_entries[ph] += 1
                lft = text[q - 1] if q > 0 else ''
                rgt = text[q + len(ph)] if q + len(ph) < len(text) else ''
                if lft.isalnum() or rgt.isalnum():
                    p6['entries'] += 1
                    inputs_by['P6'].add(k)
            i = text.find('\nPARTNER NEWS\n')
            body = text[i + 14:] if i >= 0 else ''
            for ph in R22.PLACEHOLDERS:
                ph_text[ph] += body.count(ph)
            for c, rx in RX.items():
                ms = rx.findall(body)
                if ms:
                    hits[c] += len(ms)
                    inputs_by[c].add(k)
            longw = None
            for p in [int(m['focal_permno'])] + [int(s['permno']) for s in m['shown']]:
                cut_rx, dw = prx(p)
                if cut_rx is not None:
                    ms = cut_rx.findall(body)
                    if ms:
                        hits['N7'] += len(ms)
                        inputs_by['N7'].add(k)
                if dw:
                    if longw is None:
                        longw = [mm.group(0).upper() for mm in WORD7.finditer(body)]
                    for wu in longw:
                        if wu not in dw and any(x in wu for x in dw) and not engc(wu):
                            hits['N4er'] += 1
                            inputs_by['N4er'].add(k)
            doc = RD.Doc(k, v2, m)                         # article word tokens of the masker-v2 text
            starts = [s for s, e, ph in ents]
            ta = tr = 0
            for s0, e0, _, _ in doc.segs:
                for mm in TOKEN.finditer(v2, s0, e0):
                    if mm.group(0).startswith('['):
                        continue
                    ta += 1
                    j = bisect.bisect_right(starts, mm.start()) - 1
                    hit = (j >= 0 and ents[j][1] > mm.start()) or \
                          (j + 1 < len(ents) and ents[j + 1][0] < mm.end())
                    tr += hit
            tok_all += ta
            tok_red += tr
            per_key_tok[k] = (ta, tr)
    res = {'version': label, 'period': str(period), 'sent_inputs_scanned': n,
           'inputs_by_class': {c: len(inputs_by.get(c, ())) for c in ('D1', 'D2', 'N1', 'N1b', 'N7', 'N4er', 'P6')},
           'hits_by_class': dict(sorted(hits.items())), 'p6_entries': p6['entries'],
           'ledger_entries_by_placeholder': dict(ph_entries), 'placeholders_in_article_part': dict(ph_text),
           'article_word_tokens': tok_all, 'article_word_tokens_redacted': tok_red,
           'article_word_token_share_redacted': round(tok_red / tok_all, 6) if tok_all else None,
           'name_tables_sha256': tsha, 'ledgers': [str(x) for x in ledgers], 'elapsed_s': round(time.time() - t0)}
    return res, {c: sorted(v) for c, v in inputs_by.items()}, per_key_tok


def _sent(period, final_file):
    import td4v2_flags as FL
    import td4v2_inputs as IV
    d = FL._dir(period)
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    wh = set(json.loads(Path(final_file).read_text(encoding='utf-8'))['withheld_keys'])
    keep = IV.subsample(period)
    out = set()
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            k = json.loads(line)['key']
            if k not in pop and k not in wh and (keep is None or k in keep):
                out.add(k)
    return out


def run(period):
    """Both versions on their sent sets (v2.1: REDACTION_FINAL_v2_1; v2.2: REDACTION_FINAL_v2_2) and the over-redaction
    comparison on the inputs sent in both. Writes <OUT>/inputs_redacted_v2_2/<period>/CHECK22_<UTC>.json (counts) and
    CHECK22_KEYS_<UTC>.json (keys per class; private)."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_redact22 as R22
    period = str(period)
    d = Path(FL._dir(period))
    f21 = sorted(d.glob('REDACTION_FINAL_v2_1_*.json'))[-1]
    f22 = sorted(d.glob('REDACTION_FINAL_v2_2_*.json'))[-1]
    s21, s22 = _sent(period, f21), _sent(period, f22)
    l21 = [RD.round_dir(period, r) / 'LEDGER.jsonl' for r in range(1, RD.ROUNDS + 1)]
    l22 = R22.ledgers_upto(period, R22.ROUNDS)
    r22, k22, t22 = scan_version(period, l22, s22, 'v2.2')
    r21, k21, t21 = scan_version(period, l21, s21, 'v2.1')
    both = set(t21) & set(t22)
    a21 = sum(t21[k][0] for k in both)
    res = {'period': period, 'v2_1_final': str(f21), 'v2_1_final_sha256': R.sha(f21), 'v2_2_final': str(f22),
           'v2_2_final_sha256': R.sha(f22), 'v2_2': r22, 'v2_1': r21,
           'sent_in_both': len(both),
           'article_word_token_share_redacted_sent_in_both': {
               'v2.1': round(sum(t21[k][1] for k in both) / a21, 6) if a21 else None,
               'v2.2': round(sum(t22[k][1] for k in both) / a21, 6) if a21 else None},
           'code_sha256': R.sha(Path(__file__)), **R.peak()}
    out = R22.ROOT / period / f'CHECK22_{R.utc().replace(":", "")}.json'
    R.new_json(out, res)
    R.new_json(out.with_name(out.name.replace('CHECK22_', 'CHECK22_KEYS_')), {'v2.2': k22, 'v2.1': k21})
    return res
