"""Detector-flag withholding rule for the masker-v2 inputs (researcher decision 2026-11-06: masker v3 is not used; the
masker-v2 inputs of every period are sent minus the inputs that the masking reviewers' blocking-class detectors flag).

Pre-registered (td4v2_spec.PREREGISTRATION inputs.detector_flags; manifest() is bound in PREREGISTRATION_v2), applied
before any provider call and before any outcome is read, identically for the three selectors (a withheld row takes each
arm's fallback). No adjudication: an input with at least one match of an included detector class is withheld.

Detectors (the reviewers' scripts, copied to the private run folder with only the input folder, the article store,
the withheld set and the output folder changed; the review folders are only read):
  round 1 (td4m M1-M4) through the round-2 adapters of MA1, which import the round-1 modules read-only:
    s10_m1_pop (M1 B1-B13, A1 and its date checks), s11_m2_pop (M2 classes, header and surface patterns),
    s12_m3_pop (M3 P1-P11), s13_m4_pop (M4 A1-A5, STORY, RENAME);
  round 2 MA1 (classes C1-C8, D1-D4): s14_dates_d8_mistag, s15_dates_extra, s16_dates_extra2 (date, D8 and mis-tag
    detectors incl. the round-1 regexes), s17b_new_classes (EMB, OWN, IDX, PFX, CO, REM; GB through the s18_gb_rescore
    precision filter that defines the GB detector), s41_targeted, s41b_nasdaq; s02_fsb1_names supplies FactSet names;
  round 2 MA2 (classes G1-G3, B1, D): ma2_s7b_population (single-word masks, glued compounds, dates), classified with
    the frozen G1/G2/G3 rule of ma2_s11_contexts; a glued compound counts when its string is never written in lower
    case in the raw corpus (the lower_count field of ma2_s8_aggregate; 'median' or 'statement' after a firm word 'MEDIA'
    or 'STATE' are ordinary words, reported as MA2:B1_any only); ma2_s12_dates_v2 (date patterns, article part).
Scope: date detectors and the MA2 glued-compound detector count only matches in the article part of the input (the
PARTNER NEWS section); the frozen header text is not article text.
Excluded (disclosed, not blocking classes): executive-name detectors (s11 exec_apposition, s12 X_*), clock times
(s14 x_clock_gmt), a 4-digit number followed by a unit (s14 YEAR4 year_unit_after), a month name cut by the truncation
ellipsis (MA2 D6_month_then_ellipsis).
2021-2022 also withholds the reviewers' adjudicated blocking key sets (MA1 s40_classes.json without '_borderline';
MA2 population_private2/class_keys.json).
Output: <build dir>/DETECTOR_FLAGS_M2.json (withheld keys, per-class counts and shares, script SHA256, run folder) and
the private run folder OUT/detector_flags/<period>/ (detector outputs). Reads no label, return or provider output.
"""
import ast
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import td4_spec as T
import td4_env as E

# PUBLIC COPY: the two CRSP PERMNOs of the firms whose names begin with these index words are kept in the private
# frozen code (see five_session/PUBLIC_COPY_NOTES.md); with this empty tuple the MA2:G3 class never fires here.
INDEX_NAME_PERMNOS = ()

REVIEW = Path('D:/OptResearch_data/five_session_review')
MA1 = REVIEW / 'td4m2' / 'ma1'
MA2 = REVIEW / 'td4m2' / 'ma2'
ROUND1_MODULES = (REVIEW / 'td4m' / 'm1' / 'm1_06c_population.py', REVIEW / 'td4m' / 'm1' / 'm1_08b_supplement.py',
                  REVIEW / 'td4m' / 'm3' / 's5c_population_v3.py')
MA1_RUN = ('s02_fsb1_names.py', 's10_m1_pop.py', 's11_m2_pop.py', 's12_m3_pop.py', 's13_m4_pop.py',
           's14_dates_d8_mistag.py', 's15_dates_extra.py', 's16_dates_extra2.py', 's17b_new_classes.py',
           's18_gb_rescore.py', 's41_targeted.py', 's41b_nasdaq.py')
MA2_RUN = ('ma2_s7b_population.py', 'ma2_s8_aggregate.py', 'ma2_s12_dates_v2.py')
MA2_READ = ('ma2_s11_contexts.py', 'ma2_memprobe.py')
REPORT_ONLY = ('MA2:B1_any',)      # counted for the record, not a withholding class
ADJUDICATED = (MA1 / 's40_classes.json', MA2 / 'population_private2' / 'class_keys.json')
EXCLUDED = {'s11': ('exec_apposition',), 's12': ('X_',), 's14': ('x_clock_gmt',), 'ma2_date': ('D6_month_then_ellipsis',)}
ROOT = T.OUT / 'detector_flags'
STORES = {'2122': 'article_store_select_2122.parquet', '2023': 'article_store_2023.parquet',
          '2024': 'article_store_2024.parquet', '2025': 'article_store_2025.parquet'}
PY = sys.executable


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flags_path(d):
    """The latest detector-flag file of a build (each run writes DETECTOR_FLAGS_M2_<UTC>.json; earlier files stay as
    the record)."""
    runs = sorted(Path(d).glob('DETECTOR_FLAGS_M2_*.json'))
    return runs[-1] if runs else Path(d) / 'DETECTOR_FLAGS_M2.json'


def manifest():
    """Included and excluded detector classes and the SHA256 of every reviewer script used (pure function of the
    files); bound in PREREGISTRATION_v2."""
    files = [MA1 / 'ma1_common.py'] + [MA1 / f for f in MA1_RUN] + [MA2 / f for f in MA2_RUN + MA2_READ] + \
        list(ROUND1_MODULES)
    return {'rule': 'withhold an input with >= 1 match of an included detector class (no adjudication)',
            'detectors_round1_via_ma1_adapters': ['s10_m1_pop', 's11_m2_pop', 's12_m3_pop', 's13_m4_pop'],
            'detectors_ma1': ['s14_dates_d8_mistag', 's15_dates_extra', 's16_dates_extra2', 's17b_new_classes '
                              '(EMB, OWN, IDX, PFX, CO, REM)', 's18_gb_rescore (GB kept groups)', 's41_targeted',
                              's41b_nasdaq'],
            'detectors_ma2': ['ma2_s7b_population (G1/G2/G3 by the ma2_s11_contexts rule; B1 glued compounds whose '
                              'string is never written in lower case in the raw corpus, the ma2_s8 lower_count field; '
                              'D dates)',
                              'ma2_s12_dates_v2 (patterns)'],
            'article_part_only': ['s14', 's15', 's16', 'ma2_s12 patterns', 'ma2 glued compounds'],
            'excluded_classes': {'s11': 'exec_apposition (executive names, disclosed)',
                                 's12': 'X_* (executive-name proxies, disclosed)',
                                 's14': 'x_clock_gmt (clock times, disclosed); YEAR4 matches followed by a unit',
                                 'ma2_date': 'D6_month_then_ellipsis (no day, no year; not blocking)'},
            'adjudicated_2122': [str(p) for p in ADJUDICATED] + ['MA1 classes except _borderline; every MA2 list'],
            'script_sha256': {str(p): sha(p) for p in files if p.exists()},
            'flags_code_sha256': sha(Path(__file__)),
            'run_copies': 'copies differ from the originals only in the input folder, article store, withheld set and '
                          'output folder; their SHA256 are written per period in DETECTOR_FLAGS_M2.json'}


def _write(p, text):
    with Path(p).open('x', encoding='utf-8') as f:                  # exclusive creation (TD3 runtime contract)
        f.write(text)


def _sub(text, pairs):
    for a, b in pairs:
        if a not in text:
            raise ValueError('path to replace not found: ' + a)
        text = text.replace(a, b)
    return text


def prepare(period, d, store, withheld_files):
    """Run folder with the copied detectors (exclusive: a period is run once per folder name)."""
    import td3_runtime as R
    run = ROOT / f'{period}_{R.utc().replace(":", "")}'
    r1, r2 = run / 'ma1', run / 'ma2'
    r1.mkdir(parents=True, exist_ok=False)
    r2.mkdir(parents=True, exist_ok=False)
    dq, wh = Path(d).as_posix(), [Path(x).as_posix() for x in withheld_files]
    c = (MA1 / 'ma1_common.py').read_text(encoding='utf-8')
    c = _sub(c, [("D = Path('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3')", f"D = Path('{dq}')"),
                 ("OUT = Path('D:/OptResearch_data/five_session_review/td4m2/ma1')", f"OUT = Path('{r1.as_posix()}')"),
                 ("json.loads((D / fn).read_text(encoding='utf-8'))", "json.loads(Path(fn).read_text(encoding='utf-8'))")])
    c = re.sub(r"WITHHELD_FILES = \([^)]*\)", lambda m: 'WITHHELD_FILES = ' + repr(tuple(wh)), c)
    _write(r1 / 'ma1_common.py', c)
    for fn in MA1_RUN:
        s = (MA1 / fn).read_text(encoding='utf-8')
        if fn == 's02_fsb1_names.py':
            s = _sub(s, [("D = Path('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3')", f"D = Path('{dq}')"),
                         ("OUT = Path('D:/OptResearch_data/five_session_review/td4m2/ma1')",
                          f"OUT = Path('{r1.as_posix()}')")])
        _write(r1 / fn, s)
    bad = set()
    for f in withheld_files:
        bad |= set(json.loads(Path(f).read_text(encoding='utf-8'))['withheld_keys'])
    keys = []
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            k = json.loads(line)['key']
            if k not in bad:
                keys.append(k)
    _write(r2 / 'MA2_CALLABLE_KEYS.json', json.dumps(keys))
    for fn in MA2_RUN + MA2_READ:
        s = (MA2 / fn).read_text(encoding='utf-8')
        s = s.replace('D:/OptResearch_data/five_session/td4/inputs_m2/select_2122_r3/', dq + '/')
        s = s.replace('"D:/OptResearch_data/five_session_review/td4m2/ma2/"', '"' + r2.as_posix() + '/"')
        s = s.replace('sys.path.insert(0, "D:/OptResearch_data/five_session_review/td4m2/ma2")',
                      'sys.path.insert(0, "' + r2.as_posix() + '")')
        s = s.replace('D:/OptResearch_data/five_session/td4/inputs/article_store_select_2122.parquet',
                      Path(store).as_posix())
        _write(r2 / fn, s)
    return run, len(keys)


def run_plan(run):
    """The detector runs, one process at a time, outside the TD3 runtime (which permits no subprocess): a shell
    script written to the run folder (RUN_DETECTORS.sh). Each line is 'cd <folder> && python -B <script>', its exit
    code and wall time appended to run.log."""
    lines = ['#!/bin/bash', 'set -u', 'export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1',
             f'LOG="{(run / "run.log").as_posix()}"', 'date -u > "$LOG"']
    for sub, names in (('ma1', MA1_RUN), ('ma2', MA2_RUN)):
        for fn in names:
            lines += [f'echo "== {fn} $(date -u +%H:%M:%S)" >> "$LOG"',
                      f'(cd "{(run / sub).as_posix()}" && "{Path(PY).as_posix()}" -B {fn}) >> "$LOG" 2>&1; '
                      f'echo "RC {fn} $?" >> "$LOG"']
    lines += ['date -u >> "$LOG"', 'echo DONE >> "$LOG"']
    _write(run / 'RUN_DETECTORS.sh', '\n'.join(lines) + '\n')
    return run / 'RUN_DETECTORS.sh'


def run_status(run):
    """Exit codes of the detector runs from run.log (every script must have run and returned 0)."""
    log = (run / 'run.log').read_text(encoding='utf-8', errors='replace')
    rc = dict(re.findall(r'^RC (\S+) (\d+)$', log, re.M))
    want = list(MA1_RUN) + list(MA2_RUN)
    bad = [f for f in want if rc.get(f) != '0']
    if bad or 'DONE' not in log:
        raise RuntimeError(f'detector runs incomplete or failed in {run}: {bad}')
    return {f: int(rc[f]) for f in want}


def _jsonl(p):
    if not Path(p).exists():
        return
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _ma2_rule(run):
    """The frozen G1/G2/G3 candidate rule of ma2_s11_contexts (constants read from the reviewer's file)."""
    tree = ast.parse((run / 'ma2' / 'ma2_s11_contexts.py').read_text(encoding='utf-8'))
    const = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', '') in ('ACR', 'G2W'):
            const[node.targets[0].id] = ast.literal_eval(node.value)
    return const['ACR'], const['G2W']


def _s12_patterns(run):
    src = (run / 'ma2' / 'ma2_s12_dates_v2.py').read_text(encoding='utf-8')
    start, end = src.index('M = r"'), src.index('grp = collections')
    ns = {'re': re}
    exec(src[start:end], ns)                                     # the reviewer's pattern definitions only
    return ns['PATS']


def extract(period, d, run):
    """Per-class key sets from the detector outputs; union; adjudicated sets for 2021-2022."""
    r1, r2 = run / 'ma1', run / 'ma2' / 'population_private2'
    callable_keys = set(json.loads((run / 'ma2' / 'MA2_CALLABLE_KEYS.json').read_text(encoding='utf-8')))
    pending = defaultdict(list)          # key -> [(class, detector context, newline replacement)]: article-part checks
    cls = defaultdict(set)

    def add(c, k):
        if k in callable_keys:
            cls[c].add(k)
    for r in _jsonl(r1 / 's10_m1_matches.jsonl'):
        for c in r['hits']:
            add('M1:' + c, r['key'])
    for t in _jsonl(r1 / 's11_triples.jsonl'):
        if t['class'] not in EXCLUDED['s11']:
            for k in t['keys']:
                add('M2:' + t['class'], k)
    for t in _jsonl(r1 / 's11_header_candidates.jsonl'):
        for k in t['keys']:
            add('M2:header', k)
    if (r1 / 's11_surface_keys.json').exists():
        for lab, ks in json.loads((r1 / 's11_surface_keys.json').read_text(encoding='utf-8')).items():
            for k in ks:
                add('M2:surface', k)
    for r in _jsonl(r1 / 's12_m3_matches.jsonl'):
        for c in r['hits']:
            if not c.startswith(EXCLUDED['s12']):
                add('M3:' + c, r['key'])
    for h in _jsonl(r1 / 's13_m4_hits.jsonl'):
        add('M4:' + h['class'], h['key'])
    for h in _jsonl(r1 / 's14_hits.jsonl'):
        if h['in_news'] and not h['year_unit_after'] and h['class'] not in EXCLUDED['s14']:
            add('MA1_dates:' + h['class'], h['key'])
    for fn in ('s15_hits.jsonl', 's16_hits.jsonl'):
        for h in _jsonl(r1 / fn):
            pending[h['key']].append(('MA1_dates:' + h['class'], h['ctx'], ' | '))
    for r in _jsonl(r1 / 's17b_hits.jsonl'):
        for det in r['hits']:
            if det != 'GB':
                add('MA1:' + det, r['key'])
    if (r1 / 's18_gb_kept.json').exists():
        for g in json.loads((r1 / 's18_gb_kept.json').read_text(encoding='utf-8')):
            for k in g['keys']:
                add('MA1:GB_kept', k)
    if (r1 / 's41_targeted.json').exists():
        for lab, ks in json.loads((r1 / 's41_targeted.json').read_text(encoding='utf-8'))['keys'].items():
            for k in ks:
                add('MA1:targeted', k)
    if (r1 / 's41b_nasdaq.json').exists():
        for k in json.loads((r1 / 's41b_nasdaq.json').read_text(encoding='utf-8'))['keys']:
            add('MA1:nasdaq', k)
    acr, g2w = _ma2_rule(run)
    vocab = json.loads((r2 / 'vocab.json').read_text(encoding='utf-8'))
    for r in _jsonl(r2 / 'pop_items.jsonl'):
        k = r['key']
        for wu, case, p, tcls, own, why, fld, alt, prev3, next3, tok in r['sw']:
            if tcls == 'link':
                continue
            if 'ticker' in why and (vocab.get(wu.lower(), 0) >= 3 or wu in acr):
                add('MA2:G1', k)
            elif wu in g2w and 'ticker' not in why:
                add('MA2:G2', k)
            elif wu in ('NASDAQ', 'DOW') and p in INDEX_NAME_PERMNOS:
                add('MA2:G3', k)
        for g in r['glued']:
            pending[k].append(('MA2:B1_any', g[5], ' '))  # reported only (ordinary words such as 'median')
            if vocab.get(g[0].lower(), 0) == 0:          # ma2_s8 lower_count field: never written in lower case
                pending[k].append(('MA2:B1', g[5], ' '))
        for nm, mt, ctx in r['date']:
            if nm not in EXCLUDED['ma2_date']:
                add('MA2:D:' + nm, k)
    pats = _s12_patterns(run)
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as fh:   # one streaming pass over the sent texts
        for line in fh:
            r = json.loads(line)
            k = r['key']
            if k not in callable_keys:
                continue
            t = r['input_text']
            i = t.find('\nPARTNER NEWS\n')
            body = t[i:] if i >= 0 else ''
            for nm, rx in pats.items():                   # ma2_s12 patterns on the article part
                if rx.search(body):
                    add('MA2:D12:' + nm, k)
            forms = {}
            for c, ctx, sep in pending.pop(k, ()):        # the match and its context lie in the article part
                if sep not in forms:
                    forms[sep] = body.replace('\n', sep)
                if ctx in forms[sep]:
                    add(c, k)
    adjud = {}
    if str(period) == '2122':
        a1 = json.loads(ADJUDICATED[0].read_text(encoding='utf-8'))
        for c, ks in a1.items():
            if c != '_borderline':
                for k in ks:
                    add('ADJ_MA1:' + c, k)
        a2 = json.loads(ADJUDICATED[1].read_text(encoding='utf-8'))
        for c, v in a2.items():
            for lst in (v.values() if isinstance(v, dict) else [v]):
                for k in lst:
                    add('ADJ_MA2:' + c, k)
        adjud = {str(p): sha(p) for p in ADJUDICATED}
    union = set().union(*[v for c, v in cls.items() if c not in REPORT_ONLY]) if cls else set()
    return cls, union, len(callable_keys), adjud


def _dir(period):
    import td4v2_build2 as B2
    key = '2122' if str(period) == '2122' else int(period)
    return Path(B2.PERIOD_DIR[key])


def prepare_period(period):
    """Step 1 (in the runtime): the run folder with the copied detectors and RUN_DETECTORS.sh."""
    E.paths()
    import td4v2_inputs as IV
    d = _dir(period)
    wf = [IV.withheld_file(d)]
    rdir, n = prepare(str(period), d, T.OUT / 'inputs' / STORES[str(period)], wf)
    sh = run_plan(rdir)
    return {'run_dir': str(rdir), 'callable': n, 'run_script': str(sh),
            'next': f'bash {sh.as_posix()} ; then run_td4v2.py flags --period {period} --which {rdir.name}'}


def run(period, run_name):
    """Step 3 (in the runtime, after RUN_DETECTORS.sh): detector flags for one period's masker-v2 build."""
    E.paths()
    import td3_runtime as R
    import td4v2_inputs as IV
    d = _dir(period)
    out = Path(d) / f'DETECTOR_FLAGS_M2_{R.utc().replace(":", "")}.json'
    wf = [IV.withheld_file(d)]
    rdir = ROOT / run_name
    if not rdir.name.startswith(str(period) + '_'):
        raise ValueError('run folder of another period')
    t0 = time.time()
    status = run_status(rdir)
    cls, union, n_call, adjud = extract(str(period), d, rdir)
    copies = {str(p): sha(p) for p in sorted(rdir.rglob('*.py'))}
    det_union = set().union(*[v for c, v in cls.items() if not c.startswith('ADJ_') and c not in REPORT_ONLY])         if cls else set()
    res = {'period': str(period), 'build_dir': str(d), 'run_dir': str(rdir), 'callable_before_flags': n_call,
           'population_check_file': str(wf[0]), 'population_check_sha256': sha(wf[0]),
           'withheld_keys': sorted(union), 'withheld_inputs': len(union),
           'withheld_share': round(len(union) / n_call, 5) if n_call else None,
           'detector_union_inputs': len(det_union), 'adjudicated_only_inputs': len(union - det_union),
           'by_class_inputs': {c: len(v) for c, v in sorted(cls.items())},
           'by_class_share': {c: round(len(v) / n_call, 5) for c, v in sorted(cls.items())} if n_call else {},
           'by_family_inputs': {f: len(set().union(*[v for c, v in cls.items() if c.split(':')[0] == f and
                                                     c not in REPORT_ONLY]))
                                for f in sorted({c.split(':')[0] for c in cls})},
           'report_only_classes': list(REPORT_ONLY),
           'scripts_status': status, 'run_copies_sha256': copies, 'manifest_sha256':
               hashlib.sha256(json.dumps(manifest(), sort_keys=True).encode()).hexdigest(),
           'adjudicated_sets_sha256': adjud, 'extract_seconds': round(time.time() - t0),
           'over_10pct': bool(n_call and len(union) / n_call > 0.10),
           'note': 'withheld for every arm alike before any provider call; no outcome or provider output is read',
           **R.peak()}
    R.new_json(out, res)
    return {k: v for k, v in res.items() if k not in ('withheld_keys', 'run_copies_sha256', 'by_class_share')}


def stand_in(d):
    """Rehearsal only: an empty detector-flag file for a sandbox build under OUT/rehearsal_v2_<id>/ (the reviewers'
    detectors are not run on development data; the rehearsal is an engineering check, not a result)."""
    import td3_runtime as R
    if 'rehearsal_v2_' not in str(d):
        raise PermissionError('stand-in flags only in a rehearsal sandbox')
    return R.new_json(flags_path(d), {'withheld_keys': [], 'withheld_inputs': 0, 'rehearsal_stand_in': True})
