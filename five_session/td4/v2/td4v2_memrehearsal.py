"""v2.3 fix brief F6: full-size memory rehearsal of the once-only evaluation (sandbox; no 2023-2025 label or return).

The complete v2.3 td4v2_evaluate.run runs on the REAL 2023-2025 provider outputs, ledgers and sent inputs, the real
TD2-style features, the real TD1 b and the real TD2 heads, under the real runtime contract (td3_runtime job memory
cap td4_spec.JOB_MEMORY_CAP = 1,400,000,000 bytes, offline audit hook, exclusive writes), with these replacements:
- td4v2_outcomes.test_labels is a synthetic-label stub: the same token, test-year and seal checks, the same read
  windows (the year, plus 1-15 January of the next year only when that year is <= 2025), the same row keys (the
  evaluator's own keys) and the same frame transformations and label rule (the unchanged td4_labels.label with the
  2025-12-31 bound) applied to a synthetic CRSP-shaped return panel: deterministic pseudo-returns from SHA256-seeded
  generators, date32/int32/float32/int8 columns as in the CRSP files, 5,800 synthetic PERMNOs (about 1.5 million rows a
  year, more than the 1,314,536 rows of the 2022 CRSP file, so the label step is not lighter than the real one);
- td4_labels._crsp, td4_labels.compute, td4_labels.development and td4v2_outcomes.selection_labels raise if called;
- td4v2_gate.admit_b is a sandbox gate: the once-only file check, the real ledger-digest recomputation
  (td4v2_select.combined_test_digest) and the model-file hashes; it reads no clearance file and hashes no CRSP file;
- the latch, the primary result, the result and this receipt are written under OUT/rehearsal_mem_v2_3_<id>/; the
  selection freeze read is selection_v2/SELECTION_FREEZE_v2.json (read only; the v2.3 select reproduces it);
- a file-open trace is installed first, before any data module is imported: an audit hook sees every open (and
  refuses every write outside the sandbox), and every parquet/CSV reader is wrapped. Every data file opened must be on
  the allowlist below; anything else (any CRSP file, any manifest other than the 2025 feature columns, the clearance
  folder) is refused and recorded, and the rehearsal fails.
The numbers the sandboxed evaluator writes rest on pseudo-labels: engineering checks, never results. The receipt keeps
no P1/P2 number. Run in a fresh process: run_td4v2.py mem-rehearsal.
"""
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_spec as S

N_PERMNO = 5800
CRSP_2022_ROWS = 1314536          # parquet metadata of the 2022 CRSP file (development year; read before, not here)
SYN_BASE = 900001
LABEL_COLUMNS = {'y', 'label_end', 'label_observed', 'fit_included', 'selection_included', 'refit_included'}
CODE_EXT = {'.py', '.pyc', '.pyd', '.dll', '.so', '.pth', '.zip', '.typed', '.pem', '.cfg'}
MUST_RUN_FRESH = ('td4v2_gate', 'td4v2_runner', 'td4v2_select', 'td4v2_evaluate', 'td4v2_inputs', 'td4v2_slim',
                  'td4v2_outcomes', 'td4_labels', 'td4_models', 'td4_score', 'td4_evaluate', 'pyarrow', 'pandas')


def _norm(p):
    return os.path.normcase(os.path.abspath(os.fsdecode(p)))


class Trace:
    """File-open trace: every open and reader call, classified; forbidden or unlisted data files are refused."""

    def __init__(self, box):
        self.box = _norm(box)
        self.allow_files, self.allow_dirs, self.feature_only = set(), [], set()
        import site
        roots = {sys.base_prefix, sys.prefix, os.path.dirname(sys.executable)}
        try:
            roots.add(site.getusersitepackages())
        except Exception:
            pass
        self.code_roots = [_norm(r) + os.sep for r in roots if r]
        self.code_roots.append(_norm(T.FIVE) + os.sep)          # the public code tree (.py only, see classify)
        self.data, self.refused, self.code_opens, self.scans = {}, [], Counter(), Counter()
        self.writes = Counter()
        self.events = []

    def allow(self, path=None, directory=None, feature_columns_only=False):
        if path is not None:
            self.allow_files.add(_norm(path))
            if feature_columns_only:
                self.feature_only.add(_norm(path))
        if directory is not None:
            self.allow_dirs.append(_norm(directory) + os.sep)

    def classify(self, path):
        n = _norm(path)
        low = os.path.basename(n).lower()
        ext = os.path.splitext(low)[1]
        if n == self.box or n.startswith(self.box + os.sep):
            return 'sandbox'
        if 'crsp' in low:
            return 'forbidden: CRSP file'
        if n.startswith(_norm(T.CLEARANCE_DIR) + os.sep):
            return 'forbidden: clearance folder'
        if n in self.allow_files or any(n.startswith(d) for d in self.allow_dirs):
            return 'allowed'
        in_code_root = any(n.startswith(r) for r in self.code_roots)
        if ext in CODE_EXT and (in_code_root or ext != '.py'):
            return 'code'
        if in_code_root and not n.startswith(_norm(T.FIVE) + os.sep):
            return 'code'                                        # Python install and site-packages files
        return 'forbidden: data file not on the allowlist'

    def _record(self, path, via, columns=None):
        n = _norm(path)
        kind = self.classify(n)
        if kind == 'code':
            self.code_opens[via] += 1
            return
        if kind.startswith('forbidden'):
            self.refused.append({'path': n, 'via': via, 'reason': kind})
            raise PermissionError('rehearsal trace: ' + kind + ': ' + n)
        if n in self.feature_only:
            cols = list(columns) if columns is not None else None
            if cols is None or LABEL_COLUMNS & set(cols):
                self.refused.append({'path': n, 'via': via, 'reason': 'feature-only file read without a column '
                                                                     'projection or with a label column'})
                raise PermissionError('rehearsal trace: feature-only file: ' + n)
        d = self.data.setdefault(n, {'opens': 0, 'via': [], 'columns': [], 'kind': kind})
        d['opens'] += 1
        if via not in d['via']:
            d['via'].append(via)
        if columns is not None:
            c = sorted(map(str, columns))
            if c not in d['columns']:
                d['columns'].append(c)

    def hook(self, event, args):
        if event == 'open':
            path, mode, flags = args
            if path is None or isinstance(path, int):
                return
            flags = flags or 0
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or \
                bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            if writing:
                n = _norm(path)
                if not (n == self.box or n.startswith(self.box + os.sep)):
                    self.refused.append({'path': n, 'via': 'open-write', 'reason': 'write outside the sandbox'})
                    raise PermissionError('rehearsal trace: write outside the sandbox: ' + n)
                self.writes[n] += 1
                return
            self._record(path, 'open')
        elif event == 'os.mkdir':
            n = _norm(args[0])
            if os.path.isdir(n):                     # Path.mkdir(exist_ok=True) on an existing folder creates nothing
                self.scans['mkdir-existing:' + n] += 1
                return
            if not (n == self.box or n.startswith(self.box + os.sep)):
                self.refused.append({'path': n, 'via': 'mkdir', 'reason': 'directory outside the sandbox'})
                raise PermissionError('rehearsal trace: mkdir outside the sandbox: ' + n)
        elif event in ('os.scandir', 'os.listdir'):
            if args and isinstance(args[0], (str, bytes, os.PathLike)):
                self.scans[_norm(args[0])] += 1

    def reader(self, src, via, columns=None):
        if isinstance(src, (str, bytes, os.PathLike)):
            self._record(src, via, columns)
        elif isinstance(src, (list, tuple)):
            for s in src:
                self.reader(s, via, columns)
        else:
            self.code_opens['reader-object:' + type(src).__name__] += 1


def install_reader_wrappers(trace):
    """Wrap every parquet, dataset and CSV entry point that takes a path (module attributes, looked up at call). The
    outermost call is checked with its own column projection; the reader's internal nested calls on the same source
    (read_table -> ParquetDataset -> dataset) are not checked again. Python-level opens are always checked."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.csv as pcsv
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    depth = [0]

    def wrap(mod, name):
        orig = getattr(mod, name)

        def w(*a, **k):
            if depth[0] == 0:
                src = a[0] if a else k.get('source', k.get('path', k.get('where', k.get('filepath_or_buffer'))))
                trace.reader(src, f'{mod.__name__}.{name}', k.get('columns', k.get('usecols')))
            depth[0] += 1
            try:
                return orig(*a, **k)
            finally:
                depth[0] -= 1
        w.__wrapped__ = orig
        setattr(mod, name, w)
    for name in ('read_table', 'read_schema', 'read_metadata', 'read_pandas'):
        wrap(pq, name)
    wrap(ds, 'dataset')
    for name in ('read_parquet', 'read_csv', 'read_json', 'read_pickle', 'read_feather'):
        wrap(pd, name)
    wrap(pcsv, 'read_csv')
    for name in ('memory_map', 'input_stream'):
        wrap(pa, name)
    for cls_name in ('ParquetFile', 'ParquetDataset'):
        base = getattr(pq, cls_name)

        def make(base, cls_name):
            class Logged(base):
                def __init__(self, *a, **k):
                    if depth[0] == 0:
                        src = a[0] if a else k.get('source', k.get('path_or_paths'))
                        trace.reader(src, f'pyarrow.parquet.{cls_name}', k.get('columns'))
                    depth[0] += 1
                    try:
                        super().__init__(*a, **k)
                    finally:
                        depth[0] -= 1
            Logged.__name__ = cls_name
            return Logged
        setattr(pq, cls_name, make(base, cls_name))


def synthetic_crsp(year, lo, hi, panel_permnos):
    """A CRSP-shaped frame (date32, int32, float32, int8 -> pandas, as td4_labels._crsp returns) of N_PERMNO synthetic
    PERMNOs on the weekdays of [lo, hi]; pseudo-returns from a SHA256-seeded generator; the panel PERMNOs always
    have share_type 0."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    days = pd.bdate_range(lo, hi)
    perm = np.arange(SYN_BASE, SYN_BASE + N_PERMNO, dtype=np.int32)
    seed = int.from_bytes(hashlib.sha256(f'TD4-v2.3-memrehearsal|{year}|{lo}|{hi}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    n = len(days) * len(perm)
    ret = (rng.standard_t(5, n) * 0.02).astype(np.float32)
    ret[rng.random(n) < 0.01] = np.nan
    st = rng.choice(np.array([0, 2, 1], dtype=np.int8), size=n, p=[0.85, 0.10, 0.05])
    permno = np.tile(perm, len(days))
    st[np.isin(permno, panel_permnos)] = 0
    date = np.repeat(days.values.astype('datetime64[D]'), len(perm))
    t = pa.table({'date': pa.array(date, pa.date32()), 'permno': pa.array(permno, pa.int32()),
                  'ret': pa.array(ret, pa.float32()), 'share_type': pa.array(st, pa.int8())})
    del ret, st, permno, date
    return t.to_pandas()


def make_stub_labels(log):
    def stub_test_labels(year, keys, token):
        """td4v2_outcomes.test_labels with the CRSP read replaced by synthetic_crsp (pseudo-labels)."""
        import numpy as np
        import pandas as pd
        import td4_labels as L
        import td4v2_gate as G
        if not G.valid(token, ('evaluate',)):
            raise PermissionError('Test label admission required')
        if year not in S.TEST_YEARS:
            raise PermissionError('Not a test year: ' + str(year))
        bound = S.LABEL_END_BOUND
        if bound > T.HARD_SEAL:
            raise PermissionError('Label seal')
        E.guard()
        firms = sorted(set(keys.firm_id))
        perm_of = {f: SYN_BASE + i for i, f in enumerate(firms)}
        panel = np.array(sorted(perm_of.values()), dtype=np.int32)
        windows = [(year, f'{year}-01-01', min(f'{year}-12-31', bound))]
        nxt_hi = min(f'{year + 1}-01-15', bound)
        if year + 1 <= 2025 and nxt_hi >= f'{year + 1}-01-01':
            windows.append((year + 1, f'{year + 1}-01-01', nxt_hi))
        parts = [synthetic_crsp(yy, lo, hi, panel) for yy, lo, hi in windows]
        d = pd.concat(parts, ignore_index=True)
        syn_rows = int(len(d))
        d = d[d.share_type.isin([0, 2])].copy()                     # from here: the test_labels body
        d['date'] = pd.to_datetime(d.date)
        if d.duplicated(['permno', 'date']).any():
            raise ValueError('Duplicate return session')
        dates = pd.DatetimeIndex(sorted(d.date.unique()))
        wide = d.pivot(index='date', columns='permno', values='ret').reindex(dates)
        rows = []
        for firm, g in keys.groupby('firm_id', sort=True):
            p = perm_of.get(firm)
            r = wide[p].to_numpy(float) if p is not None else None
            for c, o in zip(g.cutoff, g.origin_session):
                t = dates.get_indexer([pd.Timestamp(o)])[0] if r is not None else -1
                if t < 0:
                    rows.append((firm, c, np.nan, pd.NaT, False))
                    continue
                y, end, ok = L.label(dates, r, t, bound)
                rows.append((firm, c, y, end, ok))
        out = pd.DataFrame(rows, columns=['firm_id', 'cutoff', 'y', 'label_end', 'label_observed'])
        obs = out.label_observed.astype(bool)
        log.append({'year': int(year), 'windows': windows, 'synthetic_crsp_rows': syn_rows,
                    'synthetic_permnos': N_PERMNO, 'panel_firms': len(firms), 'keys': int(len(keys)),
                    'pseudo_observed': int(obs.sum()),
                    'max_observed_label_end': str(pd.to_datetime(out.label_end[obs]).max()) if obs.any() else None,
                    'observed_after_bound': int((pd.to_datetime(out.label_end[obs]) > pd.Timestamp(bound)).sum())})
        return out
    return stub_test_labels


def job_peak():
    import ctypes
    from ctypes import wintypes as w
    import td3_runtime as R

    class BASIC(ctypes.Structure):
        _fields_ = [('a', ctypes.c_int64), ('b', ctypes.c_int64), ('flags', w.DWORD), ('min', ctypes.c_size_t),
                    ('max', ctypes.c_size_t), ('active', w.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', w.DWORD), ('scheduling', w.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(x, ctypes.c_uint64) for x in ['r', 'w', 'o', 'rb', 'wb', 'ob']]

    class EXT(ctypes.Structure):
        _fields_ = [('basic', BASIC), ('io', IO), ('process', ctypes.c_size_t), ('job', ctypes.c_size_t),
                    ('peakprocess', ctypes.c_size_t), ('peakjob', ctypes.c_size_t)]
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
    e = EXT()
    if not k.QueryInformationJobObject(R._state['job'], 9, ctypes.byref(e), ctypes.sizeof(e), None):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'process_memory_limit_bytes': e.process, 'job_memory_limit_bytes': e.job,
            'peak_process_memory_used_bytes': e.peakprocess, 'peak_job_memory_used_bytes': e.peakjob}


def _first(events, pred):
    return next((i for i, e in enumerate(events) if pred(e)), None)


def run():
    loaded = [m for m in MUST_RUN_FRESH if m in sys.modules]
    if loaded:
        raise RuntimeError('mem-rehearsal must run in a fresh process (already imported: ' + ', '.join(loaded) + ')')
    rid = uuid.uuid4().hex[:8]
    box = T.OUT / f'rehearsal_mem_v2_3_{rid}'
    trace = Trace(box)
    sys.addaudithook(trace.hook)                       # 1. the trace, before any data module is imported
    numpy_before = 'numpy' in sys.modules
    R = E.install('offline')                           # 2. the real runtime contract (job cap 1.4e9 bytes; single-
    E.paths()                                          #    thread BLAS set before numpy loads, as in run_td4v2.main)
    install_reader_wrappers(trace)                     # 3. parquet / dataset / CSV readers (numpy loads here)
    import td4_inputs as I
    import td4_manifest as M
    import td4_labels as L
    import td4v2_build2 as B2
    import td4v2_outcomes as O
    labels_log = []

    def forbidden(name):
        def f(*a, **k):
            trace.refused.append({'call': name, 'reason': 'label or CRSP source called in the sandbox'})
            raise PermissionError('rehearsal: ' + name + ' is never called here')
        return f
    O.test_labels = make_stub_labels(labels_log)       # 4. the stub, before any CRSP or label source can be opened
    for mod, name in ((L, '_crsp'), (L, 'compute'), (L, 'development'), (O, 'selection_labels')):
        setattr(mod, name, forbidden(f'{mod.__name__}.{name}'))
    for p in S.RUN_ORDER:                              # 5. the allowlist
        trace.allow(directory=S.RUNS / f'real_{p}')
        trace.allow(directory=B2.PERIOD_DIR[S.SELECT if p == S.SELECT else int(p)])
    for y in S.TEST_YEARS:
        trace.allow(I.feature_path(y))
        trace.allow(I.evidence_path(y))
        pred = T.TD1 / f'predictions_v2/year_{y}.parquet'
        if pred.exists():
            trace.allow(pred)
        else:
            trace.allow(M.manifest_path(y), feature_columns_only=True)
    for p in (T.TD1 / 'model_frozen_v2.pkl', T.TD2 / 'MODEL_v2.json', T.TD2 / 'risk_menu.csv',
              S.SELECTION_V2_RECORD['selection_freeze_v2'], S.PREREG):
        trace.allow(p)
    import td4v2_redact_final as RF
    for p in S.RUN_ORDER:                              # the round-3 INPUTS named by each build's redaction result
        d = B2.PERIOD_DIR[S.SELECT if p == S.SELECT else int(p)]
        trace.allow(json.loads(RF.final_path(d).read_text(encoding='utf-8'))['final_inputs'])
    real_paths = {'latch': S.EVAL_LATCH, 'primary': S.EVAL_PRIMARY, 'result': S.EVAL_RESULT}
    real_before = {k: Path(v).exists() for k, v in real_paths.items()}
    S.EV = box / 'evaluation_v2'                       # 6. sandbox outputs
    S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT = (S.EV / 'EVALUATION_STARTED_v2.json',
                                                   S.EV / 'EVALUATION_PRIMARY_v2.json',
                                                   S.EV / 'EVALUATION_RESULT_v2.json')
    S.SELECTION_FREEZE = S.SELECTION_V2_RECORD['selection_freeze_v2']
    S.CLEARANCE_A, S.CLEARANCE_B = box / 'clearance' / 'A_never_written', box / 'clearance' / 'B_never_written'
    import td4v2_gate as G
    import td4v2_select as SL
    import td4v2_slim as SL3
    import td4v2_evaluate as EVV
    import td4_evaluate as EV1
    events = trace.events

    def stub_admit_b():
        events.append({'event': 'admit_b'})
        present = G.once_only_files()
        if present:
            raise PermissionError('STOP: once-only evaluation latch exists; no replay (sandbox)')
        digest, by_year = SL.combined_test_digest()
        models = {str(p): R.sha(p) for p in SL.models_files()}
        tok = G.Token('evaluate', 'sandbox-clearance-b-' + rid)
        G._tokens[id(tok)] = tok
        return tok, {}, {'sandbox': True, 'ledger_digest_test': digest, 'models_files_sha256': models}
    G.admit_b = stub_admit_b

    def logged(mod, name, tag):
        orig = getattr(mod, name)

        def w(*a, **k):
            arg = a[0] if a and isinstance(a[0], (str, int)) else None
            events.append({'event': tag + ':start', 'arg': arg, **SL3.memory()})
            out = orig(*a, **k)
            events.append({'event': tag + ':end', 'arg': arg, **SL3.memory()})
            return out
        setattr(mod, name, w)
    logged(EVV, 'actual_costs', 'actual_costs')
    logged(SL3, 'collect_slim', 'collect_slim')
    logged(O, 'test_labels', 'test_labels')
    logged(EV1, 'evaluate_core', 'evaluate_core')
    logged(EVV, 'per_year_part', 'per_year')
    logged(EVV, 'year_2025_halves', 'halves_2025')
    logged(EVV, 'secondary_s1', 'secondary_s1')
    logged(EV1, 'product_continuity', 'product_continuity')
    new_json = R.new_json

    def logged_json(path, value, *a, **k):
        events.append({'event': 'write', 'file': Path(str(path)).name})
        return new_json(path, value, *a, **k)
    R.new_json = logged_json
    t0 = time.perf_counter()
    error = None
    try:
        ret = EVV.run()                                # 7. the complete v2.3 evaluate path
    except BaseException as e:                         # recorded in the receipt (MemoryError included)
        ret, error = None, f'{type(e).__name__}: {str(e)[:400]}'
    elapsed = time.perf_counter() - t0
    R.new_json = new_json
    peaks = R.peak()
    jp = job_peak()
    target = S.EVAL_MEMORY_TARGET_BYTES

    def idx(pred):
        return _first(events, pred)
    i_admit = idx(lambda e: e['event'] == 'admit_b')
    i_cost_end = idx(lambda e: e['event'] == 'actual_costs:end')
    i_collect = [i for i, e in enumerate(events) if e['event'] == 'collect_slim:start']
    i_latch = idx(lambda e: e.get('file') == 'EVALUATION_STARTED_v2.json')
    i_label = idx(lambda e: e['event'] == 'test_labels:start')
    i_primary = idx(lambda e: e.get('file') == 'EVALUATION_PRIMARY_v2.json')
    i_desc = idx(lambda e: e['event'] in ('per_year:start', 'halves_2025:start', 'secondary_s1:start',
                                          'product_continuity:start'))
    i_result = idx(lambda e: e.get('file') == 'EVALUATION_RESULT_v2.json')
    ends = [i for i, e in enumerate(events) if e['event'] in ('per_year:end', 'halves_2025:end', 'secondary_s1:end',
                                                                 'product_continuity:end')]
    i_desc_last = max(ends) if ends else None
    order = {'admit_b_first': i_admit == 0,
             'costs_before_collect': None not in (i_cost_end,) and bool(i_collect) and i_cost_end < min(i_collect),
             'collect_before_latch': bool(i_collect) and i_latch is not None and max(i_collect) < i_latch,
             'latch_before_first_label': None not in (i_latch, i_label) and i_latch < i_label,
             'primary_before_descriptives': None not in (i_primary, i_desc) and i_primary < i_desc,
             'result_after_every_descriptive_part': None not in (i_result, i_desc_last) and i_result > i_desc_last}
    files = {k: Path(v).exists() for k, v in (('latch', S.EVAL_LATCH), ('primary', S.EVAL_PRIMARY),
                                              ('result', S.EVAL_RESULT))}
    same_primary = None
    if files['primary'] and files['result']:
        pr = json.loads(Path(S.EVAL_PRIMARY).read_text(encoding='utf-8'))
        rs = json.loads(Path(S.EVAL_RESULT).read_text(encoding='utf-8'))
        skip = {'schema', 'written_utc', 'note'}
        same_primary = all(rs.get(k) == v for k, v in pr.items() if k not in skip) and \
            rs.get('primary_result_sha256') == R.sha(S.EVAL_PRIMARY)
        desc_errors = {k: (v.get('error') if isinstance(v, dict) else None) for k, v in
                       (('per_year', rs.get('per_year_descriptive')), ('halves', rs.get('year_2025_halves_descriptive')),
                        ('s1', rs.get('secondary_S1_descriptive')))}
        per_year_errors = {y: v.get('error') for y, v in (rs.get('per_year_descriptive') or {}).items()
                           if isinstance(v, dict) and v.get('error')}
        continuity_errors = {y: v.get('error') for y, v in (rs.get('product_continuity_evidence') or {}).items()
                             if isinstance(v, dict) and v.get('error')}
        structure = {'P1_keys': sorted(pr['P1']), 'P2_keys': sorted(pr['P2']), 'rows_info': pr.get('rows_info'),
                     'P1_n': pr['P1'].get('n'), 'memory_phases': rs.get('memory_phases'),
                     'descriptive_errors': {**{k: v for k, v in desc_errors.items() if v},
                                            'per_year': per_year_errors, 'product_continuity': continuity_errors},
                     'result_write_error': (ret or {}).get('result_write_error')}
        del pr, rs
    else:
        structure = {}
    real_after = {k: Path(v).exists() for k, v in real_paths.items()}
    data = {p: v for p, v in sorted(trace.data.items())}
    crsp_opened = [p for p in data if 'crsp' in os.path.basename(p).lower()]
    label_cols = [p for p, v in data.items() if any(LABEL_COLUMNS & set(c) for c in v['columns'])]
    clean = not trace.refused and not crsp_opened and not label_cols
    within = all(v is not None and v <= target for v in (peaks['peak_working_set_bytes'], peaks['peak_private_bytes'],
                                                         jp['peak_process_memory_used_bytes']))
    ok = (error is None and clean and within and not numpy_before and all(order.values()) and all(files.values())
          and same_primary is True and not any(real_after.values()) and not any(real_before.values())
          and not structure.get('result_write_error') and not structure.get('descriptive_errors', {}).get('per_year')
          and len(labels_log) == len(S.TEST_YEARS) and all(x['observed_after_bound'] == 0 for x in labels_log))
    import td4v2_gate as G2
    out = {'schema': 'TD4_MEMREHEARSAL_v2_3', 'code_version': S.CODE_VERSION, 'sandbox': str(box),
           'label': 'Full-size memory rehearsal of the v2.3 evaluate step on real provider outputs and features with '
                    'synthetic pseudo-labels; an engineering check, not a result (no P1/P2 number kept).',
           'pass': ok, 'error': error, 'elapsed_seconds': elapsed,
           'target_bytes': target, 'job_memory_cap': T.JOB_MEMORY_CAP, **peaks, 'job_peak': jp,
           'within_target': within, 'numpy_loaded_before_runtime': numpy_before,
           'blas_threads_env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
           'order': order, 'sandbox_files': files, 'primary_repeated_in_result': same_primary,
           'real_evaluation_files_before': real_before, 'real_evaluation_files_after': real_after,
           'stub_labels': labels_log, 'synthetic_crsp': {'permnos': N_PERMNO, 'crsp_2022_rows_reference': CRSP_2022_ROWS},
           'structure': structure,
           'trace': {'clean': clean, 'refused': trace.refused, 'crsp_files_opened': crsp_opened,
                     'files_read_with_label_columns': label_cols, 'data_files_opened': data,
                     'feature_only_files': sorted(trace.feature_only),
                     'allowlist_files': sorted(trace.allow_files), 'allowlist_dirs': sorted(trace.allow_dirs),
                     'code_and_library_opens': dict(trace.code_opens), 'sandbox_writes': dict(trace.writes),
                     'directory_scans': dict(trace.scans)},
           'events': events, 'code_sha256': G2.code_inventory()}
    p = new_json(box / 'MEMREHEARSAL_v2_3.json', out)
    return {'receipt': str(p), 'sha256': R.sha(p), 'pass': ok, 'error': error,
            'peak_working_set_bytes': peaks['peak_working_set_bytes'], 'peak_private_bytes': peaks['peak_private_bytes'],
            'job_peak': jp, 'trace_clean': clean, 'refused': trace.refused[:5], 'order': order,
            'elapsed_seconds': round(elapsed, 1)}
