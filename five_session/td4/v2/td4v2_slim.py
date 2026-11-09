"""v2.3 (fix brief F2): compact main-pass provider outputs for the once-only evaluation, and their parity check.

collect_slim(period) streams the period's responses.jsonl files in numeric session order (td4v2_runner.sessions, N4)
and applies, for the MAIN pass only, exactly the td4v2_runner.collect_root rule: an output counts (http_ok) only for
HTTP 200, a returned model inside the allowlist and a stored input SHA256 equal to the sent input's SHA256 (N5); per
(key, selector) the first such output is kept, otherwise the last attempt. Per (key, selector) it keeps only what
td4_score.provider_outputs reads: http_ok, status, and the arm's one field (ordering for deepseek_order, choice for
jev_choice, yes_scores for jev_noul), the field only when http_ok and status == 'valid' (provider_outputs reads it in
no other case). Raw responses, re-ask rows, rows of keys that were not sent, probabilities, ordering_raw, issues,
usage, returned model, cost and hash fields are dropped while each line is streamed; equal strings are shared.
rows() yields the minimal rows that the unchanged td4_score.provider_outputs consumes, so the arm values are built by
the same function as in v2.2.

parity(period) compares, on one period's feature rows (no label is read), the arm values of the v2.2 path
(td4v2_runner.collect + td4v2_inputs.eligible_map + td4_score.provider_outputs) with the v2.3 path, element by element
and type by type, and writes the receipt OUT/v2_3/SLIM_PARITY_<period>_<tag>.json.
memory(tag) reads the process's current and peak working set and commit (psapi), for phase logs.
"""
import gc
import hashlib
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

import td4_spec as T
import td4v2_spec as S

ARM_FIELD = {'deepseek_order': 'ordering', 'jev_choice': 'choice', 'jev_noul': 'yes_scores'}
_SEL = {a: a for a in ARM_FIELD}


def memory(tag=None):
    import ctypes
    from ctypes import wintypes as w

    class PMC(ctypes.Structure):
        _fields_ = [('cb', w.DWORD), ('faults', w.DWORD), ('peak_ws', ctypes.c_size_t), ('ws', ctypes.c_size_t),
                    ('qpp', ctypes.c_size_t), ('qp', ctypes.c_size_t), ('qpnp', ctypes.c_size_t),
                    ('qnp', ctypes.c_size_t), ('commit', ctypes.c_size_t), ('peak_commit', ctypes.c_size_t)]
    p = PMC()
    p.cb = ctypes.sizeof(p)
    k = ctypes.WinDLL('kernel32')
    k.GetCurrentProcess.restype = w.HANDLE
    fn = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
    fn(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
    out = {'working_set_bytes': p.ws, 'commit_bytes': p.commit, 'peak_working_set_bytes': p.peak_ws,
           'peak_commit_bytes': p.peak_commit}
    return {'phase': tag, **out} if tag is not None else out


class SlimOutputs:
    """Compact main-pass outputs of one period: store[(key, selector)] = (http_ok, status, field or None)."""
    __slots__ = ('period', 'store', 'eligible_map', 'counts')

    def __init__(self, period, store, eligible_map, counts):
        self.period, self.store, self.eligible_map, self.counts = period, store, eligible_map, counts

    def __len__(self):
        return len(self.store)

    def rows(self):
        """Minimal td4v2_runner.collect rows (pass 'main') for td4_score.provider_outputs; generated on the fly."""
        for (key, sel), (ok, status, value) in self.store.items():
            row = {'key': key, 'selector': sel, 'pass': 'main', 'http_ok': ok, 'status': status,
                   'ordering': None, 'choice': None, 'yes_scores': None}
            row[ARM_FIELD[sel]] = value
            yield row


def _canon(table, x):
    return table.setdefault(x, x) if isinstance(x, str) else x


def _compact(table, sel, value):
    if value is None:
        return None
    if sel == 'deepseek_order' and isinstance(value, list):
        return [_canon(table, v) for v in value]
    if sel == 'jev_noul' and isinstance(value, dict):
        return {_canon(table, k): v for k, v in value.items()}
    return _canon(table, value)


def collect_slim_root(root, records, period=None):
    """records: {key: record with 'input_sha256'} of the inputs sent (td4v2_inputs.callable_inputs)."""
    import td4_runner as RU
    import td4v2_runner as RV
    table, best, counts = {}, {}, Counter()
    for p in RV.sessions(root, 'responses.jsonl'):
        for r in RU.iter_jsonl(p):
            counts['response_rows'] += 1
            if r['pass'] != 'main':
                counts['dropped_not_main_pass'] += 1
                continue
            sel = _SEL.get(r['selector'])
            if sel is None:
                counts['dropped_other_selector'] += 1
                continue
            rec = records.get(r['key'])
            if rec is None:                           # never looked up: eligible_map holds sent keys only
                counts['dropped_key_not_sent'] += 1
                continue
            pr = r['parsed']
            ok = r['http_status'] == 200 and pr.get('status') != 'model_not_allowed'
            if rec['input_sha256'] != r.get('input_sha256'):
                ok = False
            k = (r['key'], sel)
            prev = best.get(k)
            if prev is None or (ok and not prev[0]) or (ok == prev[0] and not prev[0]):
                status = pr.get('status')
                value = pr.get(ARM_FIELD[sel]) if (ok and status == 'valid') else None
                best[(_canon(table, r['key']), sel)] = (ok, _canon(table, status), _compact(table, sel, value))
            del r, pr
    counts['entries'] = len(best)
    return SlimOutputs(period, best, None, dict(counts)), table


def collect_slim(period, mode='real'):
    """Compact main-pass outputs of a period plus its eligible map (firm_id, cutoff) -> sent key."""
    import td4v2_inputs as IV
    import td4v2_runner as RV
    records = IV.callable_inputs(period)
    emap = IV.eligible_map(period, keys=set(records))       # = IV.eligible_map(period): keys default to these
    slim, table = collect_slim_root(RV.period_root(period, mode), records, period)
    del records
    slim.eligible_map = {k: _canon(table, v) for k, v in emap.items()}
    del emap, table
    gc.collect()
    return slim


# ------------------------------------------------------------------------------------------------------ parity
def compare_outputs(a, b):
    """Element-by-element and type-by-type differences of two td4_score.provider_outputs results (empty: equal)."""
    diffs = Counter()
    for arm in T.PROVIDER_ARMS:
        for k in ('ordering', 'choice', 'yes_scores', 'status'):
            x, y = a[arm][k], b[arm][k]
            if len(x) != len(y):
                diffs[f'{arm}.{k}.length'] += 1
                continue
            for u, v in zip(x, y):                    # JSON text: values, nested types and key order (NaN-safe)
                if type(u) is not type(v) or json.dumps(u) != json.dumps(v):
                    diffs[f'{arm}.{k}'] += 1
        va, vb = a[arm]['valid'], b[arm]['valid']
        if va.dtype != vb.dtype or va.shape != vb.shape or not np.array_equal(va, vb):
            diffs[f'{arm}.valid'] += 1
    return dict(diffs)


def digest_outputs(o):
    h = hashlib.sha256()
    for arm in T.PROVIDER_ARMS:
        part = {k: o[arm][k] for k in ('ordering', 'choice', 'yes_scores', 'status')}
        part['valid'] = [bool(v) for v in o[arm]['valid']]
        part['valid_dtype'] = str(o[arm]['valid'].dtype)
        h.update(json.dumps({arm: part}, allow_nan=True).encode())
    return h.hexdigest()


def status_counts(o):
    return {arm: dict(Counter(o[arm]['status'])) for arm in T.PROVIDER_ARMS}


def parity_root(root, records, rows, emap, period=None, measure=False):
    """v2.2 path versus v2.3 path on one run root; rows: frame with firm_id and cutoff (no label). measure=True also
    reports what each kept structure holds: the compact store and its eligible map in Python heap bytes (tracemalloc,
    built first), then the growth of the process commit while the v2.2 collect rows are built (tracemalloc off)."""
    import tracemalloc
    import td4_score as SC1
    import td4v2_runner as RV
    if measure:
        gc.collect()
        tracemalloc.start()
    b0 = tracemalloc.get_traced_memory()[0] if measure else 0
    slim, table = collect_slim_root(root, records, period)
    slim.eligible_map = {k: _canon(table, v) for k, v in emap.items()}
    del table
    gc.collect()
    b1 = tracemalloc.get_traced_memory()[0] if measure else 0
    if measure:
        tracemalloc.stop()
        gc.collect()
    m0 = memory()['commit_bytes']
    old_rows = RV.collect_root(root, records)
    gc.collect()
    m1 = memory()['commit_bytes']
    old = SC1.provider_outputs(None, rows, old_rows, emap)
    n_old = len(old_rows)
    del old_rows
    gc.collect()
    new = SC1.provider_outputs(None, rows, slim.rows(), slim.eligible_map)
    diffs = compare_outputs(old, new)
    d_old, d_new = digest_outputs(old), digest_outputs(new)
    out = {'rows': int(len(rows)), 'v2_2_collect_rows': n_old, 'v2_3_entries': len(slim), 'v2_3_counts': slim.counts,
           'differences': diffs, 'digest_v2_2': d_old, 'digest_v2_3': d_new,
           'equal': (not diffs) and d_old == d_new, 'status_counts': status_counts(new)}
    if measure:
        out['kept_memory_bytes'] = {
            'v2_3_compact_store_and_eligible_map_python_heap': b1 - b0,
            'v2_2_collect_rows_commit_growth': m1 - m0,
            'method': 'compact store: tracemalloc traced bytes after gc while it is alive (built first); v2.2 rows: '
                      'growth of the process commit (psapi PagefileUsage) while td4v2_runner.collect_root builds '
                      'them after gc, tracemalloc off (a lower bound when freed memory is reused)'}
    return out


def _rows(period):
    """Rows for the parity check, no label: the year's TD2-style primary feature rows (test periods) or the
    selection subsample's ELIGIBLE rows (2122)."""
    import td4_models as MD
    import td4v2_inputs as IV
    if str(period) == str(S.SELECT):
        sub = IV.subsample_frame()
        return sub[['firm_id', 'cutoff']].reset_index(drop=True)
    return MD.feature_frame(int(period))[['firm_id', 'cutoff']].reset_index(drop=True)


def parity(period, write=True):
    import td3_runtime as R
    import td4v2_gate as G
    import td4v2_inputs as IV
    import td4v2_runner as RV
    t0 = time.perf_counter()
    rows = _rows(period)
    records = IV.callable_inputs(period)
    emap_v22 = IV.eligible_map(period)                       # the v2.2 call, which re-reads the inputs
    emap_v23 = IV.eligible_map(period, keys=set(records))    # the v2.3 call
    res = parity_root(RV.period_root(period), records, rows, emap_v22, period, measure=True)
    res['eligible_map_equal'] = emap_v22 == emap_v23
    res['equal'] = res['equal'] and res['eligible_map_equal']
    out = {'schema': 'TD4_SLIM_PARITY_v2_3', 'code_version': S.CODE_VERSION, 'period': str(period),
           'rows_source': ('selection subsample ELIGIBLE rows' if str(period) == str(S.SELECT) else
                           'TD2-style primary feature rows of the year (td4_models.feature_frame)'),
           'label_read': False, **res, 'code_sha256': G.code_inventory(),
           'elapsed_seconds': time.perf_counter() - t0, **R.peak()}
    if write:
        tag = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()[:8]
        p = R.new_json(S.V23 / f'SLIM_PARITY_{period}_{tag}.json', out)
        return {'path': str(p), 'sha256': R.sha(p), 'equal': out['equal'], 'rows': out['rows'],
                'differences': out['differences'], 'kept_memory_bytes': out['kept_memory_bytes'],
                'peak_working_set_bytes': out['peak_working_set_bytes'], 'peak_private_bytes': out['peak_private_bytes']}
    return out


def latest_parity(period):
    files = sorted(S.V23.glob(f'SLIM_PARITY_{period}_*.json'), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None
