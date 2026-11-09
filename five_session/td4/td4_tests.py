"""TD4 offline tests: synthetic fixtures and mocked providers only (no network, no label of 2023-2025, no provider).

Fixture files are written under D:/OptResearch_data/five_session/td4/fixtures/<uuid>/ (never deleted). Examples use
invented names. qualify() writes QUALIFICATION_TD4_<uuid>.json with the counts and the code inventory.
"""
import datetime as dt
import json
import re
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import td4_spec as T
import td4_env as E

FIX = T.OUT / 'fixtures'


def fixture_dir():
    d = FIX / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    return d


def _R():
    E.paths()
    import td3_runtime as R
    return R


# ------------------------------------------------------------------------------------------------------- spec/prereg
class TestSpec(unittest.TestCase):
    def test_prereg_hash_and_content(self):
        t = T.preregistration_text()
        self.assertEqual(T.preregistration_text(), t)
        self.assertIn('P1', T.PREREGISTRATION['primary_comparisons'])
        self.assertIn('P2', T.PREREGISTRATION['primary_comparisons'])
        self.assertIn('report no validated total gain', t)
        self.assertIn('report no Jev advantage', t)
        self.assertIn('trained after 2025', t)
        self.assertEqual(len(T.preregistration_sha256()), 64)

    def test_periods_and_seal(self):
        self.assertEqual(T.PERIODS, (2024, 2025))
        self.assertEqual(T.HARD_SEAL, '2025-12-31')
        self.assertEqual(T.TRAIN_END, {2024: 2023, 2025: 2024})
        import td4_inputs as I
        with self.assertRaises(PermissionError):
            I.set_period((2026,))
        p = I.period_params((2025,))
        self.assertEqual((p['LAST_DAY'], p['NEWS_YEARS'], p['ARTICLE_LO']), ('2025-12-31', (2024, 2025), '2024-12-20'))
        I.set_period((2024,))

    def test_csv_and_parquet_seal(self):
        import td4_sources as S
        with self.assertRaises(PermissionError):
            S.csv_dated(T.QUARTERLY, ['gvkey'], hi='2026-01-01')
        with self.assertRaises(PermissionError):
            S.parquet_dated(T.CRSP_DAILY, ['permno'], hi='2026-01-02')


# ------------------------------------------------------------------------------------------------------------ gates
class TestGates(unittest.TestCase):
    def setUp(self):
        import td4_gates as Q
        self.Q = Q
        self.order = ['cash', 'size', 'leverage', 'inventory', 'cash_flow']
        self.av = np.ones((1, 5), bool)

    def idx(self, q):
        return [T.NAMES[j] for j in np.flatnonzero(q[0])]

    def test_deepseek_topk(self):
        q, v = self.Q.deepseek_gate([self.order], self.av, 1, 'none')
        self.assertEqual(self.idx(q), ['cash'])
        q, _ = self.Q.deepseek_gate([self.order], self.av, 2, 'none')
        self.assertEqual(sorted(self.idx(q)), ['cash', 'size'])
        av = self.av.copy()
        av[0, T.NAMES.index('cash')] = False       # missing characteristic does not consume the budget
        q, _ = self.Q.deepseek_gate([self.order], av, 1, 'none')
        self.assertEqual(self.idx(q), ['size'])
        q, _ = self.Q.deepseek_gate([self.order], self.av, 0, 'all')
        self.assertEqual(q.sum(), 0)
        q, _ = self.Q.deepseek_gate([self.order], av, 'all', 'none')
        self.assertEqual(q.sum(), 4)

    def test_deepseek_invalid_fallback_no_backfill(self):
        for bad in (None, self.order[:4], self.order[:4] + ['cash']):
            q, v = self.Q.deepseek_gate([bad], self.av, 1, 'none')
            self.assertFalse(v[0])
            self.assertEqual(q.sum(), 0)
            q, _ = self.Q.deepseek_gate([bad], self.av, 1, 'all')
            self.assertEqual(q.sum(), 5)

    def test_choice(self):
        q, _ = self.Q.choice_gate(['leverage'], [True], self.av, 'all')
        self.assertEqual(self.idx(q), ['leverage'])
        q, _ = self.Q.choice_gate(['none'], [True], self.av, 'all')
        self.assertEqual(q.sum(), 0)                                    # 'none' activates nothing (no fallback)
        av = self.av.copy()
        av[0, T.NAMES.index('leverage')] = False
        q, _ = self.Q.choice_gate(['leverage'], [True], av, 'all')
        self.assertEqual(q.sum(), 0)
        q, _ = self.Q.choice_gate([None], [False], self.av, 'all')      # invalid -> fallback
        self.assertEqual(q.sum(), 5)

    def test_noul_topk_and_threshold(self):
        s = {'size': 0.9, 'cash': 0.4, 'cash_flow': 0.4, 'leverage': 0.1, 'inventory': 0.6}
        q, _ = self.Q.noul_topk_gate([s], self.av, 2, 'none')
        self.assertEqual(sorted(self.idx(q)), ['inventory', 'size'])
        q, _ = self.Q.noul_topk_gate([s], self.av, 3, 'none')
        self.assertIn('cash', self.idx(q))                               # tie 0.4: menu order, cash before cash_flow
        self.assertNotIn('cash_flow', self.idx(q))
        q, _ = self.Q.noul_threshold_gate([s], self.av, 0.5, 'none')
        self.assertEqual(sorted(self.idx(q)), ['inventory', 'size'])
        q, _ = self.Q.noul_threshold_gate([None], self.av, 0.5, 'all')
        self.assertEqual(q.sum(), 5)

    def test_gate_to_feature_mapping(self):
        b = np.array([0.5, -1.0])
        z = np.array([[1., 2., 3., 4., 5.], [0., 0., 0., 0., 0.]])
        beta = [0.1, 0.2, 0.3, 0.4, 0.5]
        q = np.array([[0, 1, 0, 0, 1.], [1, 1, 1, 1, 1.]])
        p = self.Q.predict(b, z, q, beta)
        self.assertAlmostEqual(p[0], 0.5 + 2 * 0.2 + 5 * 0.5)
        self.assertEqual(p[1], -1.0)                                     # no news: Z = 0, prediction = b
        self.assertTrue(np.array_equal(self.Q.predict(b, z, np.zeros((2, 5)), beta), b))

    def test_config_grids(self):
        self.assertEqual(len(self.Q.configs('deepseek_order')), 12)
        self.assertEqual(len(self.Q.configs('jev_noul')), 18)
        self.assertEqual(len(self.Q.configs('jev_choice')), 2)
        self.assertEqual(self.Q.configs('deepseek_order')[0], ('top_k', 0, 'none'))


# ------------------------------------------------------------------------------------------------- labels / leakage
class _Poison:
    def __getitem__(self, k):
        raise AssertionError('future return field touched')

    def __len__(self):
        return 10


class TestLabels(unittest.TestCase):
    def test_token_required(self):
        import td4_labels as L
        import td4_gate as G
        keys = pd.DataFrame({'firm_id': ['x'], 'cutoff': [pd.Timestamp('2025-01-04')],
                             'origin_session': [pd.Timestamp('2025-01-03')]})
        with self.assertRaises(PermissionError):
            L.compute(2025, keys, '2025-12-31', None)
        fake = G.Token('evaluate', 'x')                    # not issued by admit: refused
        with self.assertRaises(PermissionError):
            L.compute(2025, keys, '2025-12-31', fake)

    def test_window_after_allowed_end_not_opened(self):
        import td4_labels as L
        dates = pd.date_range('2025-12-20', periods=10, freq='B')
        y, end, ok = L.label(dates, _Poison(), 3, '2025-12-24')
        self.assertTrue(np.isnan(y) and not ok)
        y, end, ok = L.label(pd.date_range('2025-12-24', periods=10, freq='B'), _Poison(), 1, '2026-06-30')
        self.assertFalse(ok)                                            # clamp at the 2025 seal

    def test_label_value(self):
        import td4_labels as L
        dates = pd.date_range('2025-03-03', periods=8, freq='B')
        r = np.array([0, 0.01, -0.02, 0.03, 0.0, 0.01, 0.5, 0.5])
        y, end, ok = L.label(dates, r, 0, '2025-12-31')
        self.assertTrue(ok)
        self.assertAlmostEqual(y, np.log(np.mean(r[1:6] ** 2)))
        self.assertEqual(end, dates[5])

    def test_feature_files_have_no_outcome_columns(self):
        import pyarrow.parquet as pq
        E.paths()
        from spec_v2 import forbidden_columns
        for p in (T.OUT / 'features/year_2025.parquet', T.OUT / 'firm_week_manifest/year_2025.parquet'):
            if p.exists():
                cols = pq.read_schema(p).names
                bad = [c for c in forbidden_columns(cols) if not c.startswith('ret') and not c.endswith('__available_at')
                       and not c.startswith('ret') and c not in ('ret1', 'ret5', 'ret22')]
                self.assertFalse({'y', 'label_end', 'label_observed'} & set(cols), p)
                self.assertEqual([c for c in bad if 'label' in c or c == 'y'], [])

    def test_score_frame_rejects_window_crossing(self):
        import td4_score as SC
        import td4_models as MD
        orig = MD.feature_frame
        try:
            MD.feature_frame = lambda year: pd.DataFrame({'firm_id': ['a'], 'cutoff': [pd.Timestamp('2025-12-20')],
                                                          'origin_session': [pd.Timestamp('2025-12-19')],
                                                          'has_news': [True]})
            labels = pd.DataFrame({'firm_id': ['a'], 'cutoff': [pd.Timestamp('2025-12-20')], 'y': [-7.0],
                                   'label_end': [pd.Timestamp('2026-01-05')], 'label_observed': [True]})
            b = pd.DataFrame({'firm_id': ['a'], 'cutoff': [pd.Timestamp('2025-12-20')], 'b': [-7.2]})
            with self.assertRaises(ValueError):
                SC.frame(2025, labels, b)
        finally:
            MD.feature_frame = orig


# ------------------------------------------------------------------------------------------- copied TD1/TD2/TD3 code
class TestCopies(unittest.TestCase):
    def graph(self):
        g = pd.DataFrame({'id': ['CUST-1', 'CUST-1', 'CUST-2', 'SUPP-3', 'CUST-4'],
                          'start': ['2024-01-02', '2025-06-05', '2025-02-01', '2023-05-01', '2025-11-30'],
                          'end': ['2025-06-05', None, None, '2025-03-01', None],
                          'status': ['valid', 'missing', 'missing', 'valid', 'missing'],
                          'typ': ['CUSTOMER', 'CUSTOMER', 'CUSTOMER', 'SUPPLIER', 'CUSTOMER'],
                          'src': ['E1', 'E1', 'E2', 'E3', 'E4'], 'tgt': ['E2', 'E2', 'E3', 'E1', 'E1'],
                          'rownum': [1, 2, 3, 4, 5]})
        links = pd.DataFrame({'entity': ['E1', 'E2', 'E3', 'E4'], 'permno': [11, 22, 33, 44],
                              'start': ['2000-01-01'] * 4, 'end': ['2025-12-31'] * 4})
        return g, links

    def test_mapped_edges_seal_future_and_ended(self):
        import td4_features as F
        g, links = self.graph()
        with self.assertRaises(ValueError):
            F.mapped_edges(g, links, pd.Timestamp('2026-01-02'))
        e = {x['id']: x for x in F.mapped_edges(g, links, pd.Timestamp('2025-04-01'))}
        self.assertEqual(sorted(e), ['CUST-1', 'CUST-2'])                 # SUPP-3 ended, CUST-4 starts later
        self.assertEqual(e['CUST-1']['start'], '2024-01-02')
        e = {x['id']: x for x in F.mapped_edges(g, links, pd.Timestamp('2025-12-01'))}
        self.assertEqual(e['CUST-1']['start'], '2025-06-05')              # latest version before the origin
        self.assertIn('CUST-4', e)
        for x in e.values():
            self.assertLess(pd.Timestamp(x['start']), pd.Timestamp('2025-12-01'))
        E.paths()
        import fast_builder
        self.assertEqual(F.mapped_edges(g, links, pd.Timestamp('2024-06-03'), seal=T.TD2_SEAL),
                         fast_builder.mapped_edges(g, links, pd.Timestamp('2024-06-03')))

    def test_graph_memory_filter_exact(self):
        import td4_features as F
        import pyarrow as pa
        g, links = self.graph()
        t = pa.Table.from_pandas(g.rename(columns={'start': 'date'}), preserve_index=False)
        for min_day in ('2025-02-15', '2025-04-01', '2025-07-01'):
            kept = F.alive_filter(t, min_day).to_pandas().rename(columns={'date': 'start'})
            self.assertEqual('SUPP-3' in set(kept.id), min_day < '2025-03-01')     # ends 2025-03-01
            for o in pd.date_range(min_day, '2025-12-19', freq='21D'):
                self.assertEqual(F.mapped_edges(kept, links, o), F.mapped_edges(g, links, o))

    def test_eligible_period_and_missing(self):
        import td4_inputs as I
        I.set_period((2025,))
        f = pd.DataFrame({'firm_id': ['a', 'b', 'c', 'd', 'e'],
                          'origin_session': pd.to_datetime(['2025-01-03', '2024-12-27', '2025-02-07', '2025-03-07',
                                                            '2025-04-04']),
                          'cutoff': pd.to_datetime(['2025-01-04 04:59', '2024-12-28 04:59', '2025-02-08 04:59',
                                                    '2025-03-08 04:59', '2025-04-05 03:59']),
                          'has_news': [True, True, True, False, True], 'mapped_focal': [True, True, True, True, False]})
        for n in T.NAMES:
            f['missing_' + n] = [False, False, True, False, False]
        e = I.eligible(f, (2025,))
        self.assertEqual(list(e.firm_id), ['a'])                         # 2024 row, all-missing, no news, unmapped out
        I.set_period((2024,))

    def test_verify_article_future_news(self):
        E.paths()
        import td3_build as B
        o, c = pd.Timestamp('2025-03-07'), pd.Timestamp('2025-03-08 04:59:59')
        news = {'ok': {'date': pd.Timestamp('2025-03-05'), 'available_at': pd.Timestamp('2025-03-06 05:00'),
                       'firms': [7], 'first_months': [202501]},
                'origin_day': {'date': o, 'available_at': pd.Timestamp('2025-03-08 05:00'), 'firms': [7],
                               'first_months': [202501]},
                'late': {'date': pd.Timestamp('2025-03-05'), 'available_at': pd.Timestamp('2025-03-09'),
                         'firms': [7], 'first_months': [202501]},
                'new_map': {'date': pd.Timestamp('2025-03-05'), 'available_at': pd.Timestamp('2025-03-06 05:00'),
                            'firms': [7], 'first_months': [202503]}}
        self.assertIsNone(B.verify_article('ok', 7, o, c, news))
        self.assertEqual(B.verify_article('origin_day', 7, o, c, news), 'outside_window_or_origin_day')
        self.assertEqual(B.verify_article('late', 7, o, c, news), 'available_not_before_cutoff')
        self.assertEqual(B.verify_article('new_map', 7, o, c, news), 'partner_not_tagged_before_origin_month')

    def test_story_field_repair_rule(self):
        import td4_repair as P
        E.paths()
        import td3_mask
        for s in ('Corp said. STORY_TIME: 11:02:33 PM GMT ((Reporting by Invented Person))',
                  'STORY_TIME:4:15 PM GMT (Editing)', 'STORY_NUMBER: 123456\nNext line'):
            t, n = P.repair_text(s)
            self.assertEqual(len(t), len(s))                    # truncation windows unchanged
            self.assertEqual(n, 1)
            self.assertNotIn('STORY_', t)
            self.assertNotIn('story_fields', td3_mask.date_residuals(t))
        t, n = P.repair_text('STORY_TIME:[11:02:33] ((x))')       # bracketed: left to TD3's own rule
        self.assertEqual(n, 0)
        self.assertEqual(P.repair_text(None), (None, 0))

    def test_year_origins_drop_crossing(self):
        import td4_sources as S
        cal = pd.bdate_range('2025-12-01', '2025-12-31').append(pd.bdate_range('2026-01-02', '2026-01-09'))
        cal = cal[cal != pd.Timestamp('2025-12-25')]
        o = S.year_origins(cal, 2025)
        self.assertEqual(o[-1], pd.Timestamp('2025-12-19'))              # 12-26 would need 2026 sessions
        self.assertNotIn(pd.Timestamp('2025-12-26'), o)

    def test_graph_extract_synthetic(self):
        import td4_sources as S
        d = fixture_dir()
        p = d / 'rel.txt'
        lines = ['"ID"|"REL_TYPE"|"SOURCE_FACTSET_ENTITY_ID"|"TARGET_FACTSET_ENTITY_ID"|"START_DATE"|"END_DATE"|"REVENUE_PCT"',
                 '"COMP-9"|"COMPETITOR"|"E1"|"E2"|2020-01-01 14:00:00||',
                 '"CUST-5"|"CUSTOMER"|"E1"|"E2"|2024-03-01 14:00:00|2026-02-01 14:00:00|',
                 '"CUST-5"|"CUSTOMER"|"E1"|"E2"|2023-03-01 14:00:00|2024-03-01 14:00:00|1.5',
                 '"SUPP-2"|"SUPPLIER"|"E3"|"E1"|2025-07-01 08:00:00||',
                 '"SUPP-1"|"SUPPLIER"|"E3"|"E1"|2026-01-03 08:00:00||']
        with p.open('x', encoding='latin-1', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        t, ex = S.graph_extract(seal='2025-12-31', path=p)
        g = t.to_pandas()
        self.assertEqual(list(g.id), ['CUST-5', 'CUST-5', 'SUPP-2'])
        self.assertEqual(list(g.rownum), [3, 2, 4])                       # 1-based data rows, sorted by id/start
        self.assertEqual(list(g.status), ['valid', 'censored_after_cutoff', 'missing'])
        self.assertTrue(g.end.iloc[1] is None)                             # end after the seal never stored
        self.assertEqual(ex['start_after_seal'], 1)
        self.assertEqual(ex['other_type'], 1)


# ----------------------------------------------------------------------------------------------------------- runner
def _records(n=3):
    out = {}
    for i in range(n):
        text = f'FROZEN INPUT (test)\nInvented context {i} for [FOCAL].\n'
        out[f'k{i:03d}'] = {'key': f'k{i:03d}', 'sample_rank': i + 1, 'input_text': text,
                            'input_sha256': _R().sha_text(text)}
    return out


class Recorder:
    def __init__(self, inner, script=None):
        self.inner, self.script, self.calls = inner, script or (lambda url, n: None), []

    def post(self, url, headers, body):
        self.calls.append((url, headers.get('Authorization', '')))
        alt = self.script(url, len([c for c in self.calls if c[0] == url]))
        if alt is not None:
            return alt
        return self.inner.post(url, headers, body)


class TestRunner(unittest.TestCase):
    def run_session(self, records, transport, balance=62.57, clock=None, caps=None, root=None, keyring=None):
        import td4_runner as RU
        R = _R()
        kr = keyring or RU.Keyring(R.Redactor(), {'deepseek': 'sk-test-deepseek-AAAA', 'jev': 'test-jev-key-BBBB'})
        caps = caps or {'deepseek_max_usd': 5.0, 'jev_max_usd': 5.0, 'jev_max_requests': 1000}
        root = root or fixture_dir() / 'period'
        d, s = RU.run_session(2024, records, transport, kr, (lambda: balance) if not callable(balance) else balance,
                              clock or RU.SimClock('2026-11-07T12:00:00'), caps, 'test', {'test': True}, root=root,
                              concurrency={'deepseek': 2, 'jev': 2})
        return d, s, root

    def mock(self):
        E.paths()
        import td3_providers as P
        return P.MockTransport()

    def test_complete_then_resume_sends_nothing(self):
        import td4_runner as RU
        rec = _records(4)
        tr = Recorder(self.mock())
        d, s, root = self.run_session(rec, tr)
        self.assertIsNone(s['halt'])
        self.assertEqual(len(tr.calls), 12)
        d2, s2, _ = self.run_session(rec, tr, root=root)
        self.assertEqual(len(tr.calls), 12)                               # idempotent: no re-billing
        self.assertEqual(s2['skipped_done_before'], 12)
        out = RU.collect(2024, root=root)
        self.assertEqual(len(out), 12)
        self.assertTrue(all(r['http_ok'] for r in out))

    def test_unsettled_reserve_counted_and_resent(self):
        import td4_runner as RU
        R = _R()
        rec = _records(1)
        root = fixture_dir() / 'period'
        (root / 'session_1').mkdir(parents=True)
        w = R.JsonlWriter(root / 'session_1' / 'ledger.jsonl', R.Redactor())
        rk = RU.make_plan(2024, rec, *self.qs())[0]['req_key']
        w.write({'kind': 'reserve', 'req_key': rk, 'attempt': 1, 'provider': 'deepseek', 'usd_upper': 0.01})
        w.close()
        done, usd, jev = RU.prior_state(root)
        self.assertNotIn(rk, done)
        self.assertAlmostEqual(usd['deepseek'], 0.01)
        tr = Recorder(self.mock())
        self.run_session(rec, tr, root=root)
        self.assertEqual(len(tr.calls), 3)

    def test_response_written_before_settle(self):
        import td4_runner as RU
        R = _R()
        rec = _records(2)
        tr = Recorder(self.mock())
        d, s, root = self.run_session(rec, tr)
        resp = [r['req_key'] for r in R.read_jsonl(d / 'responses.jsonl')]
        settles = [r['req_key'] for r in R.read_jsonl(d / 'ledger.jsonl') if r['kind'] == 'settle']
        self.assertEqual(sorted(resp), sorted(settles))
        out = RU.collect(2024, root=root)
        self.assertTrue(all('response_raw' not in r for r in out))

    def test_reask_requires_main_and_uses_new_keys(self):
        import td4_runner as RU
        rec = _records(3)
        root = fixture_dir() / 'period'
        tr = Recorder(self.mock())
        with self.assertRaises(PermissionError):
            self.run_session_reask(rec, tr, root)
        self.run_session(rec, tr, root=root)
        n = len(tr.calls)
        self.run_session_reask(rec, tr, root)
        self.assertEqual(len(tr.calls) - n, 3 * min(3, T.REASK_N))
        out = [r for r in RU.collect(2024, root=root) if r['pass'] == 'reask1']
        self.assertEqual(len(out), 3 * min(3, T.REASK_N))

    def run_session_reask(self, rec, tr, root):
        import td4_runner as RU
        R = _R()
        kr = RU.Keyring(R.Redactor(), {'deepseek': 'sk-test-deepseek-AAAA', 'jev': 'test-jev-key-BBBB'})
        return RU.run_session(2024, rec, tr, kr, lambda: 62.57, RU.SimClock('2026-11-07T12:00:00'),
                              {'deepseek_max_usd': 5.0, 'jev_max_usd': 5.0, 'jev_max_requests': 1000}, 'test',
                              {'test': True}, reask=True, root=root, concurrency={'deepseek': 2, 'jev': 2})

    def qs(self):
        E.paths()
        import td3_spec as S3
        menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
        return S3.question_bundle(menu)[1], {'deepseek_order': S3.DEEPSEEK_MODEL, 'jev_choice': S3.JEV_MODEL,
                                              'jev_noul': S3.JEV_MODEL}

    def test_balance_floor_stops_before_floor(self):
        tr = Recorder(self.mock())
        d, s, root = self.run_session(_records(3), tr, balance=8.0001)
        self.assertTrue(s['halt'].startswith('DEEPSEEK_BALANCE_FLOOR'))
        self.assertFalse(any('deepseek' in u for u, _ in tr.calls))

    def test_balance_reread_lag_is_conservative(self):
        import td4_runner as RU
        b = RU.Budget({'deepseek_max_usd': 100, 'jev_max_usd': 1, 'jev_max_requests': 10}, {}, 0,
                      lambda: 20.0, RU.SimClock())
        b.reserve('deepseek', 5.0)
        b.settle('deepseek', 5.0, 5.0)
        b.calls_since_read = 10 ** 6
        b.maybe_reread()                                                  # stale read: still 20.0
        self.assertAlmostEqual(b.projected_balance(), 15.0)
        with self.assertRaises(RU.CapStop):
            b.reserve('deepseek', 7.5)

    def test_balance_read_failure_halts(self):
        calls = {'n': 0}

        def bal():
            calls['n'] += 1
            if calls['n'] > 1:
                raise OSError('network down')
            return 62.57
        saved = T.DEEPSEEK_BALANCE_READ_EVERY_CALLS
        T.DEEPSEEK_BALANCE_READ_EVERY_CALLS = 1
        try:
            d, s, root = self.run_session(_records(3), Recorder(self.mock()), balance=bal)
        finally:
            T.DEEPSEEK_BALANCE_READ_EVERY_CALLS = saved
        self.assertEqual(s['halt'], 'deepseek_balance_read_failed_OSError')

    def test_off_peak_only(self):
        import td4_runner as RU
        R = _R()
        clock = RU.SimClock('2026-11-09T00:59:00')                         # Monday, one minute before peak
        tr = Recorder(self.mock())
        d, s, root = self.run_session(_records(3), tr, clock=clock)
        self.assertIsNone(s['halt'])
        for r in R.read_jsonl(d / 'ledger.jsonl'):
            if r['kind'] == 'reserve' and r['provider'] == 'deepseek':
                t = dt.datetime.strptime(r['utc'], '%Y-%m-%dT%H:%M:%SZ')
                self.assertFalse(RU.in_peak(t), r['utc'])
                self.assertGreaterEqual(t.hour, 4)
        self.assertGreater(s['counts'].get('deepseek_peak_waits', 0), 0)

    def test_jev_429_backoff(self):
        E.paths()
        import td3_spec as S3
        tr = Recorder(self.mock(), script=lambda url, n: (429, b'{}') if url == S3.JEV_URL and n <= 2 else None)
        d, s, root = self.run_session(_records(1), tr)
        self.assertIsNone(s['halt'])
        self.assertEqual(sum(1 for u, _ in tr.calls if u == S3.JEV_URL), 4)

    def test_http_401_halts(self):
        E.paths()
        import td3_spec as S3
        tr = Recorder(self.mock(), script=lambda url, n: (401, b'{}') if url == S3.DEEPSEEK_URL else None)
        d, s, root = self.run_session(_records(2), tr)
        self.assertEqual(s['halt'], 'http_401_deepseek')

    def test_model_violation_halts(self):
        E.paths()
        import td3_spec as S3
        import td3_providers as P

        def script(url, n):
            if url == S3.JEV_URL:
                st, body = P.mock_response(url, {}, json.dumps({'state': 'x', 'questions': S3.jev_choice_question(
                    S3.menu_rows(T.TD2 / 'risk_menu.csv'))}).encode())
                d = json.loads(body)
                d['model'] = 'jev-1.13.0'
                return 200, json.dumps(d).encode()
        tr = Recorder(self.mock(), script=script)
        d, s, root = self.run_session(_records(2), tr)
        self.assertTrue(s['halt'].startswith('model_not_allowed'))

    def test_invalid_output_logged_not_backfilled(self):
        E.paths()
        import td3_spec as S3
        import td4_runner as RU

        def script(url, n):
            if url == S3.DEEPSEEK_URL:
                body = {'model': 'deepseek-flash', 'choices': [{'message': {'content': json.dumps(
                    {'ordering': ['size', 'size', 'cash', 'leverage', 'inventory']})}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
                return 200, json.dumps(body).encode()
        tr = Recorder(self.mock(), script=script)
        d, s, root = self.run_session(_records(2), tr)
        self.assertIsNone(s['halt'])
        out = [r for r in RU.collect(2024, root=root) if r['selector'] == 'deepseek_order']
        self.assertTrue(all(r['status'] == 'invalid' and r['ordering'] is None for r in out))
        self.assertTrue(all('duplicate_names' in r['issues'] for r in out))

    def test_keys_never_written(self):
        E.paths()
        import td3_spec as S3
        import td4_runner as RU
        R = _R()
        red = R.Redactor()
        kr = RU.Keyring(red, {'deepseek': 'sk-SENTINEL-DEEPSEEK-123456', 'jev': 'SENTINEL-JEV-654321'})

        def script(url, n):
            return 500, ('echo ' + ('Bearer sk-SENTINEL-DEEPSEEK-123456' if url == S3.DEEPSEEK_URL
                                    else 'Bearer SENTINEL-JEV-654321')).encode()
        tr = Recorder(self.mock(), script=script)
        d, s, root = self.run_session(_records(1), tr, keyring=kr)
        for p in Path(root).rglob('*'):
            if p.is_file():
                txt = p.read_text(encoding='utf-8', errors='replace')
                self.assertNotIn('SENTINEL', txt, p)

    def test_jev_request_cap(self):
        tr = Recorder(self.mock())
        d, s, root = self.run_session(_records(3), tr, caps={'deepseek_max_usd': 5, 'jev_max_usd': 5,
                                                             'jev_max_requests': 2})
        self.assertTrue(s['halt'].startswith('JEV_REQUEST_CAP'))

    def test_completion_and_order_rule(self):
        import td4_runner as RU
        self.assertTrue(RU.deepseek_window_ok(dt.datetime(2026, 11, 7, 2, 0)))       # Saturday
        self.assertFalse(RU.deepseek_window_ok(dt.datetime(2026, 11, 9, 5, 59)))     # Monday, 06:00 within guard
        self.assertTrue(RU.deepseek_window_ok(dt.datetime(2026, 11, 9, 10, 0)))

    def test_balance_parse(self):
        import td4_runner as RU
        self.assertEqual(RU.parse_balance({'is_available': True, 'balance_infos': [
            {'currency': 'USD', 'total_balance': '62.57'}]}), 62.57)
        with self.assertRaises(ValueError):
            RU.parse_balance({'balance_infos': [{'currency': 'CNY', 'total_balance': '1'}]})


# -------------------------------------------------------------------------------------------------------------- gate
class TestGate(unittest.TestCase):
    def test_clearance_absent_and_request_only(self):
        import td4_gate as G
        d = fixture_dir()
        ok, reasons, _ = G.check_a(d / 'none.json', d / 'freeze.json', T.CODE)
        self.assertFalse(ok)
        R = _R()
        R.new_json(d / 'freeze.json', {'version': T.VERSION, 'code_sha256': G.code_inventory(),
                                       'artifact_sha256': {}, 'external_sha256': {},
                                       'preregistration_sha256': T.preregistration_sha256()})
        R.new_json(d / 'req.json', {'schema': T.CLEARANCE_A_SCHEMA, 'status': 'CLEARED', 'request_only': True,
                                    'freeze_sha256': G.sha(d / 'freeze.json')})
        ok, reasons, _ = G.check_a(d / 'req.json', d / 'freeze.json', T.CODE)
        self.assertFalse(ok)
        self.assertTrue(any('request is never a clearance' in r for r in reasons))
        with self.assertRaises(PermissionError):
            G.admit('call', d / 'req.json', d / 'freeze.json', T.CODE)

    def test_code_inventory_change_refused(self):
        import td4_gate as G
        R = _R()
        d = fixture_dir()
        code = d / 'code'
        code.mkdir()
        with (code / 'a.py').open('x', encoding='utf-8') as f:
            f.write('x = 1\n')
        R.new_json(d / 'freeze.json', {'code_sha256': G.code_inventory(code)})
        with (code / 'b.py').open('x', encoding='utf-8') as f:
            f.write('y = 2\n')
        R.new_json(d / 'c.json', {'schema': T.CLEARANCE_A_SCHEMA, 'status': 'CLEARED',
                                  'freeze_sha256': G.sha(d / 'freeze.json'), 'code_sha256': G.code_inventory(code)})
        ok, reasons, _ = G.check_a(d / 'c.json', d / 'freeze.json', code)
        self.assertFalse(ok)
        self.assertTrue(any('code inventory' in r for r in reasons))

    def test_real_steps_refuse_without_clearance(self):
        import td4_gate as G
        if T.CLEARANCE_A.exists():
            self.skipTest('a clearance exists')
        for step in G.STEPS_A:
            with self.assertRaises(PermissionError):
                G.admit(step)
        with self.assertRaises(PermissionError):
            G.admit_b()


# ---------------------------------------------------------------------------------------------- selection/evaluation
def synth_frame(n=400, seed=5):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({'firm_id': [f'f{i % 40}' for i in range(n)],
                      'cutoff': pd.to_datetime('2025-01-04') + pd.to_timedelta((np.arange(n) // 40) * 7, unit='D'),
                      'has_news': rng.random(n) < 0.9})
    f['origin_session'] = f.cutoff - pd.Timedelta(hours=5)
    for n_ in T.NAMES:
        f['missing_' + n_] = rng.random(n) < 0.1
    for arm in ['Z_', 'Z1_', 'Z2_', 'Z3_']:
        for n_ in T.NAMES:
            f[arm + n_] = np.where(f.has_news & ~f['missing_' + n_], rng.normal(size=n), 0.0)
    f['b'] = rng.normal(-7, 0.5, n)
    for h in T.HOPS:
        for c in ('graph_partners', 'news_partners', 'selected_partners'):
            f[f'H{h}_{c}'] = rng.integers(0, 5, n)
    return f


class TestSelectEvaluate(unittest.TestCase):
    def setUp(self):
        self.f = synth_frame()
        n = len(self.f)
        self.heads = {'beta': [0.0, 0.0, 0.0, 1.0, 0.0], 'training_beta_order': T.NAMES,
                      'heads': {a: {'beta': [0.0, 0.0, 0.0, 0.5, 0.0]} for a in ('1r', '2r', '3r')}}
        rng = np.random.default_rng(1)
        # truth: only leverage interaction matters
        self.f['y'] = self.f.b + self.f['Z_leverage'] * 1.0 + rng.normal(0, 0.05, n)
        good = ['leverage', 'size', 'cash', 'cash_flow', 'inventory']
        self.outs = {'deepseek_order': {'ordering': [good] * n, 'choice': [None] * n, 'yes_scores': [None] * n,
                                        'valid': np.ones(n, bool), 'status': ['valid'] * n},
                     'jev_choice': {'ordering': [None] * n, 'choice': ['size'] * n, 'yes_scores': [None] * n,
                                    'valid': np.ones(n, bool), 'status': ['valid'] * n},
                     'jev_noul': {'ordering': [None] * n, 'choice': [None] * n,
                                  'yes_scores': [{'size': .9, 'cash': .2, 'cash_flow': .2, 'leverage': .8,
                                                  'inventory': .1}] * n,
                                  'valid': np.ones(n, bool), 'status': ['valid'] * n}}

    def test_select_core(self):
        import td4_select as SL
        r = SL.select_core(self.f, self.outs, self.heads)
        self.assertEqual(r['chosen_provider_configs']['deepseek_order']['param'], 1)
        self.assertEqual(r['best_pipeline_arm'], 'deepseek_order')
        self.assertEqual(r['better_jev_arm'], 'jev_noul')
        self.assertEqual(r['chosen_provider_configs']['jev_choice']['fallback'], 'none')   # tie: none before all

    def test_evaluate_core_confirmed_and_not(self):
        import td4_select as SL
        import td4_evaluate as EV
        sel = SL.select_core(self.f, self.outs, self.heads)
        sel['training_beta_order_frozen'] = T.NAMES
        r = EV.evaluate_core(self.f, self.outs, self.heads, sel)
        self.assertTrue(r['P1']['confirmed'])
        self.assertEqual(r['stop_rule_readings']['P1'], T.PREREGISTRATION['stop_rule_readings']['P1_confirmed'])
        self.assertEqual(r['P2']['jev_arm'], 'jev_noul')
        f2 = self.f.copy()
        f2['y'] = f2.b + np.random.default_rng(3).normal(0, 0.5, len(f2))  # no signal
        r2 = EV.evaluate_core(f2, self.outs, self.heads, sel)
        if not r2['P1']['confirmed']:
            self.assertEqual(r2['stop_rule_readings']['P1'],
                             T.PREREGISTRATION['stop_rule_readings']['P1_not_confirmed'])
        self.assertEqual(len(r['hop_increments']), 4 * 4)


# ------------------------------------------------------------------------------------------------------ diagnostics
class TestDiagnostics(unittest.TestCase):
    def test_ndcg_topone_calibration(self):
        import td4_diagnostics as D
        rel = {'size': 3, 'cash': 0, 'cash_flow': 1, 'leverage': 2, 'inventory': 0}
        self.assertAlmostEqual(D.ndcg_at(['size', 'leverage', 'cash_flow', 'cash', 'inventory'], rel, 3), 1.0)
        self.assertLess(D.ndcg_at(['cash', 'inventory', 'size', 'leverage', 'cash_flow'], rel, 2), 0.5)
        labels = {'L1': {'rel': rel, 'top': 'size'}}
        outs = {'L1': {'deepseek_order': {'ordering': ['size', 'leverage', 'cash_flow', 'cash', 'inventory']},
                       'jev_choice': {'choice': 'size', 'probabilities': {'size': .7, 'cash': .05, 'cash_flow': .05,
                                                                         'leverage': .1, 'inventory': .05, 'none': .05}},
                       'jev_noul': {'yes_scores': {'size': .9, 'cash': .1, 'cash_flow': .3, 'leverage': .8,
                                                   'inventory': .1}}}}
        s = D.score_core(labels, outs)
        self.assertEqual(s['top_one_accuracy']['jev_choice'], 1.0)
        self.assertAlmostEqual(s['ndcg_deepseek_order']['@3'], 1.0)
        self.assertIsNotNone(s['jev_choice_calibration']['multiclass_brier'])

    def test_stability(self):
        import td4_diagnostics as D
        o = ['size', 'cash', 'cash_flow', 'leverage', 'inventory']
        rows = [{'key': 'k', 'selector': 'deepseek_order', 'pass': 'main', 'ordering': o},
                {'key': 'k', 'selector': 'deepseek_order', 'pass': 'reask1', 'ordering': o[::-1]}]
        s = D.stability(2024, rows)
        self.assertEqual(s['deepseek_order']['identical_ordering'], 0.0)
        self.assertAlmostEqual(s['deepseek_order']['mean_kendall_tau'], -1.0)


# --------------------------------------------------------------------------------------------------- public folder
class TestPublicFolder(unittest.TestCase):
    def test_only_code_and_docs(self):
        bad = [p.name for p in T.CODE.iterdir() if p.is_file() and p.suffix not in ('.py', '.md')]
        self.assertEqual(bad, [])
        self.assertFalse((T.CODE / '__pycache__').exists())

    def test_no_sample_text_or_names(self):
        R = _R()
        code = '\n'.join(p.read_text(encoding='utf-8') for p in sorted(T.CODE.glob('*.py')) + sorted(T.CODE.glob('*.md')))
        code_low = code.lower()
        for y in T.PERIODS:
            p = T.OUT / 'inputs' / f'period_{y}' / 'AUDIT_META.jsonl'
            if not p.exists():
                continue
            names, windows = set(), set()
            for m in R.read_jsonl(p)[:150]:
                for a in m['aliases'].values():
                    names.update(c for c in a.get('comnams', []) if len(c) >= 8)
                for x in m['articles']:
                    w = re.findall(r'[A-Za-z]+', x['body_raw'])
                    windows.update(' '.join(w[i:i + 8]).lower() for i in range(0, max(0, len(w) - 8), 3))
            self.assertEqual([n for n in names if n.lower() in code_low], [])
            flat = ' '.join(re.findall(r'[A-Za-z]+', code)).lower()
            self.assertEqual([w for w in windows if w in flat], [])


def qualify():
    R = E.install('offline')
    import td4_gate as G
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for cls in (TestSpec, TestGates, TestLabels, TestCopies, TestRunner, TestGate, TestSelectEvaluate,
                TestDiagnostics, TestPublicFolder):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    res = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    out = {'version': T.VERSION, 'tests': res.testsRun, 'failures': len(res.failures), 'errors': len(res.errors),
           'skipped': len(res.skipped), 'failed_tests': [str(t) for t, _ in res.failures + res.errors],
           'code_sha256': G.code_inventory(), **R.peak()}
    p = R.new_json(T.OUT / f'QUALIFICATION_TD4_{uuid.uuid4().hex}.json', out)
    print('QUALIFICATION', p, json.dumps({k: out[k] for k in ('tests', 'failures', 'errors', 'skipped')}), flush=True)
    return 0 if not (res.failures or res.errors) else 1
