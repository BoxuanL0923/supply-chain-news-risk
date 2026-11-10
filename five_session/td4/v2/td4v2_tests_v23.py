"""v2.3 tests (fix brief F8): costs before the collection and the latch (F1); compact outputs identical to the v2.2
path on invented fixtures and on the builder's rehearsal fixtures (F2); the primary result written before every
descriptive part, and a descriptive failure contained without touching the primary result (F4); wording (F5); call and
reask refused, the new paths, the once-only files checked first, the provider-output record (F7).
Every firm, key and value here is invented. Fixtures under D:/OptResearch_data/five_session/td4/fixtures/<uuid>/."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import td4_spec as T
import td4_env as E
import td4v2_spec as S
from td4v2_tests import fixture_dir, _R

REHEARSAL_FIXTURES = T.OUT / 'rehearsal_v2_5e8e8422'       # the builder's v2 rehearsal sandbox (mock providers)
NAMES = list(T.NAMES)


def _write_jsonl(R, path, rows):
    w = R.JsonlWriter(path, R.Redactor())
    for r in rows:
        w.write(r)
    w.close()


def _resp(key, sel, status=None, http=200, sha='h', pas='main', **parsed):
    p = {} if status is None else {'status': status}
    p.update(parsed)
    return {'key': key, 'selector': sel, 'pass': pas, 'http_status': http, 'input_sha256': sha,
            'response_raw': 'invented raw body', 'usd_estimate': 0.001, 'parsed': p}


def _vals(i):
    order = NAMES[i % 5:] + NAMES[:i % 5]
    return {'deepseek_order': {'ordering': order, 'ordering_raw': 'x'},
            'jev_choice': {'choice': NAMES[i % 5], 'probabilities': {n: 0.2 for n in NAMES}},
            'jev_noul': {'yes_scores': {n: round(0.1 * ((i + j) % 10), 3) for j, n in enumerate(NAMES)}}}


def _all(key, status='valid', i=0, **kw):
    return [_resp(key, sel, status, **{**_vals(i)[sel], **kw}) for sel in T.PROVIDER_ARMS]


class TestV23SlimOutputs(unittest.TestCase):
    """F2: td4v2_slim gives the same arm values as td4v2_runner.collect_root + td4_score.provider_outputs."""

    def _compare(self, root, records, rows, emap):
        import td4_score as SC1
        import td4v2_runner as RV
        import td4v2_slim as SL3
        old = SC1.provider_outputs(None, rows, RV.collect_root(root, records), emap)
        slim, _ = SL3.collect_slim_root(root, records)
        new = SC1.provider_outputs(None, rows, slim.rows(), emap)
        self.assertEqual(SL3.compare_outputs(old, new), {})
        self.assertEqual(SL3.digest_outputs(old), SL3.digest_outputs(new))
        return old, new, slim

    def test_identical_on_invented_edge_cases(self):
        R = _R()
        root = fixture_dir()
        s1 = _all('k1', i=1) + [_resp('k1', 'deepseek_order', 'valid', pas='reask1', ordering=NAMES[::-1])]
        s1 += [_resp('k2', sel, None, http=500) for sel in T.PROVIDER_ARMS]                     # failed, then valid
        s1 += _all('k3', i=3) + _all('k4', status='invalid', i=4) + _all('k5', i=5, sha='other')  # N5 mismatch
        s1 += [_resp('k6', sel, 'model_not_allowed') for sel in T.PROVIDER_ARMS]
        s1 += [_resp('k7', sel, 'valid', pas='reask1', **_vals(7)[sel]) for sel in T.PROVIDER_ARMS]   # re-ask only
        s1 += _all('k8', i=8)                                                                  # key not sent
        s1 += [_resp('k10', sel, None, http=500) for sel in T.PROVIDER_ARMS]
        s2 = _all('k2', i=2) + _all('k3', i=31) + _all('k9', i=9)                              # first valid kept
        s2 += [_resp('k10', sel, 'transport', http=502) for sel in T.PROVIDER_ARMS]            # last failure kept
        s10 = _all('k9', i=12)                                                                 # session_10 after 2
        for name, rows in (('session_1', s1), ('session_2', s2), ('session_10', s10)):
            _write_jsonl(R, root / name / 'responses.jsonl', rows)
        sent = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k9', 'k10']
        records = {k: {'input_sha256': 'h'} for k in sent}
        c = pd.Timestamp('2024-03-09 04:59:59')
        firms = [f'f{i}' for i in range(1, 12)]
        rows = pd.DataFrame({'firm_id': firms, 'cutoff': [c] * len(firms)})
        emap = {(f'f{i}', c): f'k{i}' for i in range(1, 11) if f'k{i}' in records}
        old, new, slim = self._compare(root, records, rows, emap)
        st = new['deepseek_order']['status']
        self.assertEqual(st[0], 'valid')                                   # k1
        self.assertEqual(st[1], 'valid')                                   # k2: the later HTTP 200 output
        self.assertEqual(new['deepseek_order']['ordering'][2], _vals(3)['deepseek_order']['ordering'])   # k3: first
        self.assertEqual(st[3], 'invalid')                                 # k4
        self.assertEqual(st[4], 'no_output')                               # k5: input hash mismatch
        self.assertEqual(st[5], 'no_output')                               # k6: model not allowed
        self.assertEqual(st[6], 'no_output')                               # k7: re-ask rows never count
        self.assertEqual(st[7], 'not_called')                              # f8: key not sent
        self.assertEqual(new['jev_choice']['choice'][8], _vals(9)['jev_choice']['choice'])   # k9: session_2 first
        self.assertNotEqual(_vals(9)['jev_choice']['choice'], _vals(12)['jev_choice']['choice'])
        self.assertNotEqual(_vals(3)['deepseek_order']['ordering'], _vals(31)['deepseek_order']['ordering'])
        self.assertEqual(st[9], 'no_output')                               # k10: last failed attempt
        self.assertEqual(st[10], 'not_called')                             # f11: no input
        self.assertIsNone(new['jev_noul']['yes_scores'][3])                # invalid: no field kept
        self.assertGreater(slim.counts['dropped_not_main_pass'], 0)
        self.assertGreater(slim.counts['dropped_key_not_sent'], 0)
        for (_, sel), (ok, status, value) in slim.store.items():           # compact: one field, valid only
            self.assertTrue(value is None or (ok and status == 'valid'))

    def test_identical_on_rehearsal_fixtures(self):
        import pyarrow.parquet as pq
        import td4_inputs as I
        self.assertTrue(REHEARSAL_FIXTURES.exists(), 'builder rehearsal fixtures missing')
        for run, build in (('real_2122', 'select_2122'), ('real_2022', 'period_2022')):
            d = REHEARSAL_FIXTURES / 'inputs_m2' / build
            records = I.load_inputs(None, d / 'INPUTS.jsonl')
            e = pq.read_table(d / 'ELIGIBLE.parquet').to_pandas()
            rows = e[['firm_id', 'cutoff']].reset_index(drop=True)
            b = e[(e.status == 'input_built') & e.key.isin(set(records))]
            emap = {(a, pd.Timestamp(c)): k for a, c, k in zip(b.firm_id, b.cutoff, b.key)}
            old, new, slim = self._compare(REHEARSAL_FIXTURES / 'runs_v2' / run, records, rows, emap)
            self.assertGreater(sum(new['jev_noul']['valid']), 0, run)
            self.assertGreater(slim.counts['dropped_not_main_pass'], 0, run)     # the re-ask session is dropped


class EvalSandbox:
    """td4v2_evaluate.run on invented frames: admit_b, completion, the compact collection, features, b, labels and
    product continuity are stubbed; the real td4_score.provider_outputs, td4v2_score._join, td4_evaluate.evaluate_core,
    selection code and writers run. Records the event order and whether the primary file exists at each event."""

    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.events = []

    def ev(self, name, arg=None):
        self.events.append({'event': name, 'arg': arg, 'primary_exists': Path(S.EVAL_PRIMARY).exists(),
                            'latch_exists': Path(S.EVAL_LATCH).exists()})

    def run(self):
        import td4_tests as V1
        import td4_select as SL
        import td4_evaluate as EV1
        import td4_models as MD
        import td4v2_evaluate as EV
        import td4v2_gate as G
        import td4v2_runner as RV
        import td4v2_slim as SL3
        import td4v2_score as SC2
        import td4v2_outcomes as O
        R = _R()
        base = V1.TestSelectEvaluate()
        base.setUp()
        self.base = base
        d = fixture_dir()
        n = len(base.f)
        i = np.arange(n)
        frames, bfr, labs, slims = {}, {}, {}, {}
        for y in S.TEST_YEARS:
            f = base.f.copy()
            f['cutoff'] = pd.Timestamp(f'{y}-01-04 04:59:59') + pd.to_timedelta((i // 40) * 35, unit='D')
            f['origin_session'] = f.cutoff - pd.Timedelta(hours=5)
            frames[y] = f.drop(columns=['y', 'b'])
            bfr[y] = f[['firm_id', 'cutoff', 'b']].copy()
            labs[y] = pd.DataFrame({'firm_id': f.firm_id, 'cutoff': f.cutoff, 'y': f.y,
                                    'label_end': f.origin_session + pd.Timedelta(days=7), 'label_observed': True})
            store, emap = {}, {}
            for j in range(n):
                if not f.has_news.iloc[j]:
                    continue
                key = f'k{y}_{j}'
                emap[(f.firm_id.iloc[j], pd.Timestamp(f.cutoff.iloc[j]))] = key
                store[(key, 'deepseek_order')] = (True, 'valid', list(base.outs['deepseek_order']['ordering'][j]))
                store[(key, 'jev_choice')] = (True, 'valid', base.outs['jev_choice']['choice'][j])
                store[(key, 'jev_noul')] = (True, 'valid', dict(base.outs['jev_noul']['yes_scores'][j]))
            slims[y] = SL3.SlimOutputs(str(y), store, emap, {'entries': len(store)})
        sel = {**SL.select_core(base.f, base.outs, base.heads), 'training_beta_order_frozen': NAMES}
        self.costs_stub = {'usd_by_provider': {'deepseek': 1.5, 'jev': 2.5}, 'jev_requests': 7}
        tok = G.Token('evaluate', 'fixture-clearance-b')
        G._tokens[id(tok)] = tok
        new_json = R.new_json

        def logged_json(path, value, *a, **k):
            self.ev('write', Path(str(path)).name)
            return new_json(path, value, *a, **k)

        def labels(year, keys, token):
            self.ev('test_labels', int(year))
            if not G.valid(token, ('evaluate',)):
                raise PermissionError('token')
            return labs[int(year)].copy()

        def wrap(fn, name):
            def w(*a, **k):
                self.ev(name + ':start')
                out = fn(*a, **k)
                self.ev(name + ':end')
                return out
            return w
        patches = [(G, 'admit_b', lambda: (self.ev('admit_b'), (tok, {}, {}))[1]),
                   (RV, 'completion', lambda p, records=None, mode='real': (self.ev('completion', p),
                                                                             dict(self.costs_stub))[1]),
                   (SL3, 'collect_slim', lambda period, mode='real': (self.ev('collect', str(period)),
                                                                      slims[int(period)])[1]),
                   (MD, 'feature_frame', lambda year, *a, **k: frames[int(year)].copy()),
                   (SC2, 'test_b', lambda year: bfr[int(year)].copy()),
                   (SC2, 'test_heads', lambda: base.heads),
                   (O, 'test_labels', labels),
                   (EV1, 'product_continuity', lambda year: (self.ev('product_continuity', int(year)),
                                                             {'H1|unavailable|physical_channel_verified=False': 3})[1]),
                   (R, 'new_json', logged_json)]
        for name in ('per_year_part', 'year_2025_halves', 'secondary_s1'):
            patches.append((EV, name, wrap(self.overrides.get(name, getattr(EV, name)), name)))
        for key, fn in self.overrides.items():
            if isinstance(key, tuple):
                patches.append((key[0], key[1], fn))
        paths = ('EV', 'EVAL_LATCH', 'EVAL_PRIMARY', 'EVAL_RESULT', 'SELECTION_FREEZE')
        saved_paths = {k: getattr(S, k) for k in paths}
        saved = [(m, n_, getattr(m, n_)) for m, n_, _ in patches]
        try:
            S.EV = d / 'evaluation_v2'
            S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT = (S.EV / 'EVALUATION_STARTED_v2.json',
                                                           S.EV / 'EVALUATION_PRIMARY_v2.json',
                                                           S.EV / 'EVALUATION_RESULT_v2.json')
            S.SELECTION_FREEZE = new_json(d / 'SELECTION_FREEZE_fixture.json', sel)
            for m, n_, fn in patches:
                setattr(m, n_, fn)
            self.ret = EV.run()
            self.primary = json.loads(Path(S.EVAL_PRIMARY).read_text(encoding='utf-8'))
            self.result = json.loads(Path(S.EVAL_RESULT).read_text(encoding='utf-8'))
            self.result_text = Path(S.EVAL_RESULT).read_text(encoding='utf-8')
            self.primary_text = Path(S.EVAL_PRIMARY).read_text(encoding='utf-8')
            self.primary_sha = R.sha(S.EVAL_PRIMARY)
            pooled = pd.concat([frames[y].merge(bfr[y], on=['firm_id', 'cutoff']).merge(
                labs[y][['firm_id', 'cutoff', 'y']], on=['firm_id', 'cutoff']) for y in S.TEST_YEARS],
                ignore_index=True)
            self.pooled_rows = len(pooled)
        finally:
            for m, n_, fn in saved:
                setattr(m, n_, fn)
            for k, v in saved_paths.items():
                setattr(S, k, v)
        return self

    def index(self, pred):
        return next((i for i, e in enumerate(self.events) if pred(e)), None)


_RUNS = {}


def normal_run():
    if 'normal' not in _RUNS:
        _RUNS['normal'] = EvalSandbox().run()
    return _RUNS['normal']


class TestV23Evaluator(unittest.TestCase):
    def test_f1_costs_before_collection_and_latch(self):
        sb = normal_run()
        ev = [e['event'] for e in sb.events]
        self.assertEqual(ev[0], 'admit_b')
        comp = [i for i, e in enumerate(sb.events) if e['event'] == 'completion']
        coll = [i for i, e in enumerate(sb.events) if e['event'] == 'collect']
        latch = sb.index(lambda e: e.get('arg') == 'EVALUATION_STARTED_v2.json')
        label = sb.index(lambda e: e['event'] == 'test_labels')
        self.assertEqual([sb.events[i]['arg'] for i in comp], S.RUN_ORDER)       # every period's ledgers
        self.assertEqual([sb.events[i]['arg'] for i in coll], [str(y) for y in S.TEST_YEARS])
        self.assertLess(max(comp), min(coll))                                  # costs before any output
        self.assertLess(max(coll), latch)
        self.assertLess(latch, label)
        self.assertFalse(any(sb.events[i]['latch_exists'] for i in comp))
        expected = {p: {'usd_by_provider_including_retries_and_reasks': sb.costs_stub['usd_by_provider'],
                        'jev_requests': sb.costs_stub['jev_requests']} for p in S.RUN_ORDER}
        self.assertEqual(sb.primary['cost'], expected)                         # the small result is what is kept

    def test_f4_primary_before_descriptives(self):
        sb = normal_run()
        prim = sb.index(lambda e: e.get('arg') == 'EVALUATION_PRIMARY_v2.json')
        desc = [i for i, e in enumerate(sb.events) if e['event'] in ('per_year_part:start', 'year_2025_halves:start',
                                                                     'secondary_s1:start', 'product_continuity')]
        res = sb.index(lambda e: e.get('arg') == 'EVALUATION_RESULT_v2.json')
        self.assertEqual(len(desc), 3 + 1 + 1 + 3)
        self.assertLess(prim, min(desc))
        self.assertTrue(all(sb.events[i]['primary_exists'] for i in desc))
        self.assertGreater(res, max(desc))
        for k in ('P1', 'P2', 'stop_rule_readings', 'arms', 'references', 'hop_increments', 'cost', 'rows_info'):
            self.assertIn(k, sb.primary)
        for k in ('per_year_descriptive', 'secondary_S1_descriptive', 'year_2025_halves_descriptive',
                  'product_continuity_evidence'):
            self.assertNotIn(k, sb.primary)
            self.assertIn(k, sb.result)
        skip = {'schema', 'written_utc', 'note'}
        self.assertTrue(all(sb.result[k] == v for k, v in sb.primary.items() if k not in skip))
        self.assertEqual(sb.result['primary_result_sha256'], sb.primary_sha)
        self.assertEqual(sb.primary['P1']['n'], sb.pooled_rows)
        self.assertIsNone(sb.ret['result_write_error'])
        r = sb.result                                                          # the clean run has no error
        self.assertTrue(all('error' not in v for v in r['per_year_descriptive'].values()))
        self.assertNotIn('error', r['secondary_S1_descriptive'])
        self.assertIn('DESCRIPTIVE', r['secondary_S1_descriptive']['label'])
        self.assertEqual(len(r['year_2025_halves_descriptive']), 2)
        self.assertTrue(all('error' not in v for v in r['year_2025_halves_descriptive'].values()))
        self.assertTrue(all('error' not in v for v in r['product_continuity_evidence'].values()))
        self.assertTrue(r['memory_phases'])

    def test_f4_descriptive_failure_contained(self):
        import td4_evaluate as EV1
        import td4v2_evaluate as EV
        calls = {'n': 0}
        real_per_year = EV.per_year_part

        def per_year_fail(f, o, heads, sel):
            calls['n'] += 1
            if calls['n'] == 2:
                raise MemoryError('invented per-year failure')
            return real_per_year(f, o, heads, sel)

        def s1_fail(*a):
            raise MemoryError('invented S1 failure')

        def continuity_fail(year):
            if int(year) == 2023:
                raise RuntimeError('invented continuity failure')
            return {'H1|unavailable|physical_channel_verified=False': 3}
        sb = EvalSandbox({'per_year_part': per_year_fail, 'secondary_s1': s1_fail,
                          'year_2025_halves': lambda *a: {'x': float('nan')},          # refused by the strict writer
                          (EV1, 'product_continuity'): continuity_fail}).run()
        ok = normal_run()
        r = sb.result
        self.assertIn('MemoryError', r['per_year_descriptive']['2024']['error'])
        self.assertNotIn('error', r['per_year_descriptive']['2023'])
        self.assertNotIn('error', r['per_year_descriptive']['2025'])
        self.assertIn('MemoryError', r['secondary_S1_descriptive']['error'])
        self.assertIn('ValueError', r['year_2025_halves_descriptive']['error'])
        self.assertIn('RuntimeError', r['product_continuity_evidence']['2023']['error'])
        self.assertNotIn('error', r['product_continuity_evidence']['2024'])
        skip = {'schema', 'written_utc', 'note'}
        self.assertTrue(all(r[k] == v for k, v in sb.primary.items() if k not in skip))   # primary repeated
        self.assertEqual(r['primary_result_sha256'], sb.primary_sha)
        for k in ('P1', 'P2', 'stop_rule_readings', 'arms', 'references', 'hop_increments', 'hop_ablations',
                  'descriptive_grid_test', 'coverage', 'cost', 'rows_info'):
            self.assertEqual(sb.primary[k], ok.primary[k], k)                  # identical to the clean run

    def test_f5_wording_only(self):
        import td4_evaluate as EV1
        import td4v2_evaluate as EV
        sb = normal_run()
        for text in (sb.primary_text, sb.result_text):
            self.assertNotIn('chosen on 2024', text)
            self.assertNotIn('interval excludes zero', text)
        self.assertEqual(sb.primary['P1']['comparison'], 'best pipeline arm (chosen on 2021-2022) vs firm-only MLP')
        self.assertEqual(sb.primary['P2']['comparison'],
                         'better Jev arm (chosen on 2021-2022) vs matched-input DeepSeek, same head')
        for k in ('P1', 'P2'):
            self.assertIn('interval above zero', sb.primary[k]['confirmation_meaning'])
            self.assertIn('exploratory historical reproduction of 2023-2025', sb.primary[k]['confirmation_meaning'])
            self.assertIs(sb.primary[k]['exploratory'], True)
        self.assertIn('exploratory historical reproduction', sb.primary['label'])
        self.assertIn('interval above zero', sb.primary['label'])
        import td4_select as SL
        sel = json.loads(json.dumps({**SL.select_core(sb.base.f, sb.base.outs, sb.base.heads),
                                     'training_beta_order_frozen': NAMES}, default=str))
        pooled, pooled_outs = _pooled(sb)
        raw = EV1.evaluate_core(pooled, pooled_outs, sb.base.heads, sel)
        for k in ('P1', 'P2'):                                                 # numbers and flags unchanged
            for f in ('mean', 'se', 'normal_95_interval', 'confirmed', 'n', 'uncertainty_status'):
                self.assertEqual(sb.primary[k][f], json.loads(json.dumps(raw[k][f])), (k, f))
        self.assertEqual(sb.primary['stop_rule_readings'], EV.readings(raw))
        self.assertTrue(raw['P1']['exploratory'])                              # the TD2 marker, kept


def _pooled(sb):
    """The pooled frame and outputs the evaluator built, rebuilt from the sandbox's own inputs."""
    import td4_score as SC1
    import td4v2_score as SC2
    import td4v2_slim as SL3
    base = sb.base
    n = len(base.f)
    i = np.arange(n)
    frames, outs = [], []
    for y in S.TEST_YEARS:
        f = base.f.copy()
        f['cutoff'] = pd.Timestamp(f'{y}-01-04 04:59:59') + pd.to_timedelta((i // 40) * 35, unit='D')
        f['origin_session'] = f.cutoff - pd.Timedelta(hours=5)
        f['label_end'] = f.origin_session + pd.Timedelta(days=7)
        f['label_observed'] = True
        store, emap = {}, {}
        for j in range(n):
            if not f.has_news.iloc[j]:
                continue
            key = f'k{y}_{j}'
            emap[(f.firm_id.iloc[j], pd.Timestamp(f.cutoff.iloc[j]))] = key
            store[(key, 'deepseek_order')] = (True, 'valid', list(base.outs['deepseek_order']['ordering'][j]))
            store[(key, 'jev_choice')] = (True, 'valid', base.outs['jev_choice']['choice'][j])
            store[(key, 'jev_noul')] = (True, 'valid', dict(base.outs['jev_noul']['yes_scores'][j]))
        f['test_year'] = y
        frames.append(f)
        outs.append(SC1.provider_outputs(None, f, SL3.SlimOutputs(str(y), store, emap, {}).rows(), emap))
    return pd.concat(frames, ignore_index=True), SC2.concat_outputs(outs)


class TestV23Gate(unittest.TestCase):
    def test_f7_call_and_reask_refused(self):
        import td4v2_gate as G
        import td4v2_runner as RV
        R = _R()
        d = fixture_dir()
        R.new_json(d / 'f.json', {'code_sha256': {}})
        R.new_json(d / 'c.json', {'schema': S.CLEARANCE_A_SCHEMA, 'status': 'CLEARED', 'freeze_sha256': G.sha(d / 'f.json')})
        for step in ('call', 'reask'):
            with self.assertRaises(PermissionError) as cm:
                G.admit(step, d / 'c.json', d / 'f.json', inventory={})
            self.assertIn('closed in v2.3', str(cm.exception))
        for reask in (False, True):
            with self.assertRaises(PermissionError) as cm:
                RV.run_real('2023', reask=reask)
            self.assertIn('closed in v2.3', str(cm.exception))
        import run_td4v2
        for cmd in ('call', 'reask'):
            with self.assertRaises(PermissionError) as cm:
                run_td4v2.main([cmd, '--period', '2023'])
            self.assertIn('closed in v2.3', str(cm.exception))
        import td4v2_rehearsal as RH
        with self.assertRaises(PermissionError):
            RH.run()                                                           # superseded dev-year rehearsal
        self.assertIn('closed', G.status()['provider_calls'])

    def test_f7_new_paths(self):
        import td4v2_gate as G
        import td4v2_select as SL
        self.assertEqual(S.CODE_VERSION, 'v2.3')
        self.assertEqual((S.FREEZE_A.name, S.REQUEST_A.name),
                         ('FREEZE_TD4_A_v2_3.json', 'CLEARANCE_REQUEST_TD4_A_v2_3.json'))
        self.assertEqual((S.CLEARANCE_A.name, S.CLEARANCE_B.name),
                         ('TD4_CLEARANCE_A_v2_3.json', 'TD4_CLEARANCE_B_v2_3.json'))
        self.assertEqual((S.CLEARANCE_A_SCHEMA, S.CLEARANCE_B_SCHEMA), ('TD4_CLEARANCE_A_v2_3', 'TD4_CLEARANCE_B_v2_3'))
        self.assertEqual(S.SEL.name, 'selection_v2_3')
        self.assertEqual((S.SELECTION_FREEZE.parent, S.REQUEST_B.parent), (S.SEL, S.SEL))
        self.assertEqual((S.SELECTION_FREEZE.name, S.REQUEST_B.name),
                         ('SELECTION_FREEZE_v2_3.json', 'CLEARANCE_REQUEST_TD4_B_v2_3.json'))
        for p in S.SELECTION_V2_RECORD.values():
            self.assertEqual(p.parent.name, 'selection_v2')
            self.assertNotEqual(p.parent, S.SEL)
        self.assertNotIn('selection_v2/', Path(SL.__file__).read_text(encoding='utf-8'))
        self.assertEqual(S.PREREG.name, 'PREREGISTRATION_v2_2.json')               # unchanged and bound as it is
        f22 = json.loads((T.OUT / 'FREEZE_TD4_A_v2_2.json').read_text(encoding='utf-8'))
        self.assertEqual(S.preregistration_sha256(), f22['preregistration_sha256'])
        self.assertEqual(G.sha(S.PREREG), f22['preregistration_sha256'])
        self.assertEqual(S.EVAL_PRIMARY.parent, S.EV)
        self.assertIn('provider_outputs_record', G.A_BINDINGS)
        self.assertEqual(S.SUPERSEDED_V2['freeze_v2_2'].name, 'FREEZE_TD4_A_v2_2.json')
        self.assertLessEqual(S.EVAL_MEMORY_TARGET_BYTES, T.JOB_MEMORY_CAP)
        self.assertEqual(T.JOB_MEMORY_CAP, 1_400_000_000)                          # the cap is not raised

    def test_f7_once_only_files_checked_first(self):
        import td4v2_gate as G
        R = _R()
        saved = (S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT)
        try:
            for which in range(3):
                d = fixture_dir()
                S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT = d / 'latch.json', d / 'primary.json', d / 'result.json'
                R.new_json((S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT)[which], {'x': 1})
                with self.assertRaises(PermissionError) as cm:
                    G.admit_b()
                self.assertIn('once-only evaluation latch exists', str(cm.exception))
        finally:
            S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT = saved

    def test_f7_provider_output_record(self):
        import td4v2_gate as G
        R = _R()
        runs = fixture_dir()
        for i, p in enumerate(S.RUN_ORDER):
            _write_jsonl(R, runs / f'real_{p}' / 'session_1' / 'ledger.jsonl',
                         [{'kind': 'reserve', 'req_key': f'r{i}', 'attempt': 1},
                          {'kind': 'settle', 'req_key': f'r{i}', 'attempt': 1, 'http_status': 200,
                           'response_sha256': 'ab' * 32}])
        rec = {'periods': G.runs_digests(runs=runs)}
        self.assertEqual(G.check_runs_record(rec, runs=runs), [])
        self.assertEqual(rec['periods']['2023']['settle_rows'], 1)
        _write_jsonl(R, runs / 'real_2024' / 'session_2' / 'ledger.jsonl',
                     [{'kind': 'settle', 'req_key': 'new', 'attempt': 1, 'http_status': 200, 'response_sha256': 'cd'}])
        reasons = G.check_runs_record(rec, runs=runs)
        self.assertTrue(any('ledger digest of 2024' in r for r in reasons))
        self.assertTrue(any('run files of 2024' in r for r in reasons))
        self.assertFalse(any('2023' in r for r in reasons))
        self.assertTrue(G.check_runs_record(None))


def tests():
    return [TestV23SlimOutputs, TestV23Evaluator, TestV23Gate]
