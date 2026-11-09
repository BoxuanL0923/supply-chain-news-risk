"""TD2 v2 synthetic fixture tests. Invented values only: no private payload, label or return is read.

Fixture files are written under D:/OptResearch_data/five_session/td2/fixtures_v2/<uuid>/ (private root).
"""
import ast
import collections
import hashlib
import inspect
import json
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import engine
import selectors
import decoder_v3 as dec
import io_v2 as io
import gates_v2 as gv
import build_v2
import head_v2
import cost_v2
import evaluation_v3
import freeze_v2
from runtime_v3 import OUT, CODE, sha, new_json
from spec_v2 import NAMES, K_GRID, HOPS, SELECTORS, VERSION, feature_columns, forbidden_columns

ORIGIN = pd.Timestamp('2020-06-12'); CUTOFF = pd.Timestamp('2020-06-13 03:59:59')
V2_FILES = ['spec_v2.py', 'io_v2.py', 'gates_v2.py', 'build_v2.py', 'head_v2.py', 'cost_v2.py', 'freeze_v2.py',
            'evaluation_v3.py', 'tests_v2.py', 'qualify_v2.py', 'run_td2_v3.py']

def fid(p):
    return hashlib.sha256(f'TD1-firm:{p}'.encode()).hexdigest()[:24]

def fixture_root():
    p = OUT / 'fixtures_v2' / uuid.uuid4().hex
    p.mkdir(parents=True)
    return p

def links_frame():
    return pd.DataFrame([(f'e{i}', 100 + i, '2010-01-01', '2024-12-31') for i in range(1, 7)],
                        columns=['entity', 'permno', 'start', 'end'])

def graph_frame():
    rows = [('r12', '2019-01-01', None, 'valid', 'CUSTOMER', 'e1', 'e2', 1),
            ('r23', '2019-01-01', None, 'valid', 'SUPPLIER', 'e2', 'e3', 2),
            ('r34', '2019-01-01', None, 'valid', 'CUSTOMER', 'e3', 'e4', 3),
            ('rorigin', '2020-06-12', None, 'valid', 'CUSTOMER', 'e1', 'e5', 4),   # starts on origin day
            ('rfuture', '2020-07-01', None, 'valid', 'SUPPLIER', 'e1', 'e6', 5),   # future-dated edge
            ('rended', '2018-01-01', '2020-06-01', 'valid', 'CUSTOMER', 'e5', 'e6', 6)]
    return pd.DataFrame(rows, columns=['id', 'start', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'])

def manifest_rows(poison=0.0):
    rows = []
    for p in (101, 999):
        r = {'firm_id': fid(p), 'cutoff': CUTOFF, 'origin_session': ORIGIN, 'eligible_features': True,
             'y': poison, 'label_end': ORIGIN + pd.Timedelta(days=7)}
        for i, n in enumerate(NAMES):
            r[n] = float(i + 1); r[n + '__available_at'] = pd.Timestamp('2020-05-01 04:00:00')
        rows.append(r)
    rows[0]['cash'] = np.nan; rows[0]['cash__available_at'] = pd.NaT
    return pd.DataFrame(rows)

def news_frame():
    def rec(aid, day, firms, months, text):
        return dict(date=pd.Timestamp(day), alt_id=aid, firms=firms, first_months=months,
                    available_at=engine.day_clock(day), **engine.score(text))
    return pd.DataFrame([rec('a1', '2020-06-10', [102], [201901], 'debt shortage'),
                         rec('a2', '2020-06-11', [103, 104], [201901, 201901], 'inventory supply production'),
                         rec('a3', '2020-06-12', [102], [201901], 'bankruptcy default credit'),   # origin day
                         rec('a4', '2020-06-20', [102], [201901], 'liquidity payment'),           # future story
                         rec('a5', '2020-06-11', [105], [201901], 'demand'),                      # via origin-day edge only
                         rec('a6', '2020-06-11', [102], [202006], 'cash operating'),              # mapping month = origin month
                         rec('a0', '2020-06-04', [102], [201901], 'orders')])                     # outside 7 days

def build(poison=0.0, suspects=frozenset(), stories=None):
    stories = dec.week_stories(news_frame(), ORIGIN) if stories is None else stories
    links = links_frame()
    return build_v2.origin_rows(ORIGIN, CUTOFF, manifest_rows(poison), NAMES, stories, graph_frame(), links,
                                set(suspects), build_v2.firm_index(links))

class BuildFixture(unittest.TestCase):
    def frame(self, **kw):
        f, e = build(**kw)
        return pd.DataFrame(f), pd.DataFrame(e)
    def row(self, f, firm=101, variant='primary'):
        return f[(f.firm_id == fid(firm)) & (f.variant == variant)].iloc[0]
    def test_story_filter_excludes_origin_day_future_window_and_month(self):
        self.assertEqual(sorted(s['alt_id'] for s in dec.week_stories(news_frame(), ORIGIN)), ['a1', 'a2', 'a5'])
    def test_exact_hops_and_future_edges_excluded(self):
        f, e = self.frame(); r = self.row(f)
        self.assertEqual([int(r[f'H{h}_graph_partners']) for h in HOPS], [1, 1, 1])
        self.assertEqual([int(r[f'H{h}_news_partners']) for h in HOPS], [1, 1, 1])
        self.assertNotIn(105, set(e.partner)); self.assertNotIn(106, set(e.partner))
    def test_E_and_Z_values(self):
        f, _ = self.frame(); r = self.row(f); s1 = engine.score('debt shortage'); s2 = engine.score('inventory supply production')
        for i, n in enumerate(NAMES):
            c = float(i + 1)
            for h, s in [(1, s1), (2, s2), (3, s2)]:
                self.assertAlmostEqual(r[f'E{h}_{n}'], s[n] / 5)
                self.assertAlmostEqual(r[f'Z{h}_{n}'], 0.0 if n == 'cash' else c * s[n] / 5)
            self.assertAlmostEqual(r['Z_' + n], sum(r[f'Z{h}_{n}'] for h in HOPS) / 3)
            self.assertAlmostEqual(r['E_' + n], sum(r[f'E{h}_{n}'] for h in HOPS) / 3)
    def test_missing_characteristic_recorded_and_zero(self):
        f, _ = self.frame(); r = self.row(f)
        self.assertTrue(r['missing_cash']); self.assertTrue(np.isnan(r['C_cash']))
        self.assertTrue(all(r[f'Z{h}_cash'] == 0 for h in HOPS)); self.assertFalse(r['missing_size'])
    def test_unmapped_focal_has_no_exposure(self):
        f, _ = self.frame(); r = self.row(f, 999)
        self.assertFalse(r['mapped_focal']); self.assertFalse(r['has_news'])
        self.assertEqual(sum(r['Z_' + n] for n in NAMES), 0)
    def test_intermediate_evidence_and_no_physical_claim(self):
        _, e = self.frame(); w = e[(e.variant == 'primary') & (e.hop == 3)].iloc[0]
        self.assertEqual(list(w.intermediate_nodes), [102, 103]); self.assertEqual(list(w.relationship_ids), ['r12', 'r23', 'r34'])
        self.assertEqual(list(w.reporting_roles), ['CUSTOMER', 'SUPPLIER', 'CUSTOMER'])
        self.assertFalse(e.physical_channel_verified.any())
        self.assertTrue((e.intermediate_product_evidence == 'unavailable_in_sources').all())
        self.assertTrue((e.max_edge_available_at < CUTOFF).all() and (e.max_news_available_at < CUTOFF).all())
    def test_fsb5_sensitivity_drops_suspect_identity(self):
        f, _ = self.frame(suspects={('e2', 102)})
        self.assertTrue(self.row(f)['has_news']); self.assertFalse(self.row(f, variant='fsb5_sensitivity')['has_news'])
    def test_label_columns_in_manifest_are_ignored(self):
        a, ea = self.frame(poison=0.0); b, eb = self.frame(poison=1e9)
        pd.testing.assert_frame_equal(a, b); pd.testing.assert_frame_equal(ea, eb)
    def test_injected_future_news_rejected(self):
        stories = dec.week_stories(news_frame(), ORIGIN)
        bad = dict(stories[0], date=ORIGIN, available_at=engine.day_clock(ORIGIN))
        with self.assertRaises(ValueError): build(stories=stories + [bad])
        late = dict(stories[0], alt_id='late', available_at=CUTOFF)
        with self.assertRaises(ValueError): build(stories=stories + [late])
    def test_schema_roundtrip_and_no_outcome_columns(self):
        feats, ev = build(); t = build_v2.to_table(feats, build_v2.feature_schema())
        self.assertEqual(t.schema.names, feature_columns()); self.assertEqual(forbidden_columns(t.schema.names), [])
        self.assertEqual(build_v2.to_table(ev, build_v2.EVIDENCE_SCHEMA).num_rows, len(ev))
        params = inspect.signature(build_v2.origin_rows).parameters
        self.assertFalse([p for p in params if 'ret' in p or 'label' in p or p == 'y'])
    def test_coverage_counts(self):
        f, _ = self.frame(); cov = {c['variant']: c for c in build_v2.coverage(f, NAMES)}
        self.assertEqual(cov['primary']['firm_weeks_with_news'], 1); self.assertEqual(cov['primary']['missing_characteristics']['cash'], 1)
        self.assertEqual(cov['primary']['hop_coverage']['3']['rows_with_selected_partner'], 1)

class Quarantine(unittest.TestCase):
    def db(self, rows, added=91869):
        path = fixture_root() / 'graph.sqlite'; con = sqlite3.connect(str(path))
        con.execute('CREATE TABLE o (id TEXT, start TEXT, "end" TEXT, status TEXT, typ TEXT, src TEXT, tgt TEXT, rownum INT)')
        con.execute('CREATE TABLE classes (id TEXT, category TEXT)')
        con.executemany('INSERT INTO classes VALUES (?,?)', [('q1', 'added')] + [(f'z{i}', 'added') for i in range(added - 1)])
        con.executemany('INSERT INTO o VALUES (?,?,?,?,?,?,?,?)', rows); con.commit(); con.close()
        return path
    def test_quarantine_and_date_seal(self):
        rows = [('q1', '2019-01-01', None, 'valid', 'CUSTOMER', 'a', 'b', 1), ('ok1', '2019-01-01', None, 'valid', 'SUPPLIER', 'a', 'b', 2),
                ('unp', '2019-01-01', None, 'unparseable', 'CUSTOMER', 'a', 'b', 3), ('late', '2025-01-05', None, 'valid', 'CUSTOMER', 'a', 'b', 4),
                ('other', '2019-01-01', None, 'valid', 'COMPETITOR', 'a', 'b', 5)]
        with patch.object(dec, 'GRAPH', self.db(rows)):
            graph, q = dec.graph_rows()
        self.assertEqual(list(graph.id), ['ok1']); self.assertEqual(len(q), 91869)
    def test_unsealed_end_rejected(self):
        with patch.object(dec, 'GRAPH', self.db([('ok1', '2019-01-01', '2025-03-01', 'valid', 'CUSTOMER', 'a', 'b', 1)])):
            with self.assertRaises(ValueError): dec.graph_rows()
    def test_quarantine_count_enforced(self):
        with patch.object(dec, 'GRAPH', self.db([('ok1', '2019-01-01', None, 'valid', 'CUSTOMER', 'a', 'b', 1)], added=5)):
            with self.assertRaises(ValueError): dec.graph_rows()

class Gates(unittest.TestCase):
    def test_gate_matrix_parity_with_v1(self):
        rng = np.random.default_rng(11)
        for _ in range(300):
            order = list(rng.permutation(NAMES)); avail = [n for n in NAMES if rng.random() < .7]
            ranks = np.array([[order.index(n) for n in NAMES]]); amask = np.array([[n in avail for n in NAMES]])
            for k in K_GRID:
                np.testing.assert_array_equal(gv.gate_matrix(ranks, amask, k)[0], selectors.gate(NAMES, order, k, avail))
    def test_random_rank_parity_with_v1_ordering(self):
        frame = pd.DataFrame({'firm_id': [fid(i) for i in range(30)],
                              'cutoff': pd.date_range('2021-01-09 04:59:59', periods=30, freq='7D')})
        for seed in (72101, 72103, 72109):
            ranks = gv.rank_matrix(frame, NAMES, 'random', seed)
            for i, r in enumerate(frame.itertuples()):
                expected = selectors.ordering(NAMES, 'random', f'{r.firm_id}|{r.cutoff}', seed)
                self.assertEqual([NAMES[j] for j in np.argsort(ranks[i])], expected)
    def test_beta_order_uses_standardized_coefficients(self):
        std = [0.1, 0.5, 0.05, 0.3, 0.2]; raw = [s / sc for s, sc in zip(std, [10., 0.1, 0.001, 0.1, 0.1])]
        self.assertEqual(gv.beta_order(NAMES, std), ['cash', 'leverage', 'inventory', 'size', 'cash_flow'])
        self.assertNotEqual(gv.beta_order(NAMES, std), gv.beta_order(NAMES, raw))
    def test_fallback_only_rows_without_news(self):
        ranks = np.tile(np.arange(5), (2, 1)); avail = np.ones((2, 5), bool); news = np.array([True, False])
        self.assertEqual(gv.gates(ranks, avail, news, 2, 'none').sum(axis=1).tolist(), [2, 0])
        self.assertEqual(gv.gates(ranks, avail, news, 2, 'all').sum(axis=1).tolist(), [2, 5])
    def test_k_zero_is_baseline_and_none(self):
        q = gv.gate_matrix(np.tile(np.arange(5), (3, 1)), np.ones((3, 5), bool), 0)
        np.testing.assert_array_equal(gv.predict([1., 2., 3.], np.ones((3, 5)), q, np.ones(5)), [1., 2., 3.])
        self.assertEqual(selectors.choice_gate(NAMES, 'none', NAMES).sum(), 0)
    def test_missing_items_do_not_consume_budget(self):
        avail = np.array([[False, True, False, True, True]])
        self.assertEqual(gv.gate_matrix(np.array([[0, 1, 2, 3, 4]]), avail, 1)[0].tolist(), [0, 1, 0, 0, 0])

class Head(unittest.TestCase):
    def data(self, n=4000, seed=3):
        rng = np.random.default_rng(seed); z = rng.normal(size=(n, 5)) * np.array([1., 100., .01, 1., 1.])
        beta = np.array([.3, -.002, 20., 0., .1]); bdev = rng.normal(size=n)
        y = bdev + z @ beta + rng.normal(scale=.05, size=n); idx = np.arange(n)
        return z, y, bdev, idx < 2400, (idx >= 2400) & (idx < 3200), idx < 3200, idx >= 3200, beta
    def test_fit_arm_recovers_and_uses_only_its_partitions(self):
        z, y, bdev, tr, va, ref, other, beta = self.data()
        r = head_v2.fit_arm(z, y, bdev, bdev, tr, va, ref)
        scale = np.array(r['scale_training_rms']); x = z / scale
        np.testing.assert_allclose(np.array(r['training_beta']) * scale, beta * scale, atol=0.02)
        best = min(r['ridge_grid'], key=lambda t: t['selection_mse']); self.assertEqual(best['penalty'], r['penalty'])
        np.testing.assert_allclose(r['training_coef_std'], engine.ridge(x[tr], (y - bdev)[tr], r['penalty']))
        np.testing.assert_allclose(r['final_coef_std'], engine.ridge(x[ref], (y - bdev)[ref], r['penalty']))
        y2 = y.copy(); y2[other] = 1e9
        self.assertEqual(head_v2.fit_arm(z, y2, bdev, bdev, tr, va, ref), r)
    def test_scale_fixed_on_training_rows(self):
        z, y, bdev, tr, va, ref, other, _ = self.data(); z2 = z.copy(); z2[~tr] *= 50
        r = head_v2.fit_arm(z2, y, bdev, bdev, tr, va, ref)
        np.testing.assert_allclose(r['scale_training_rms'], np.sqrt(np.mean(z[tr] ** 2, axis=0)))
    def test_masks_refuse_boundary_crossing(self):
        f = pd.DataFrame({'fit_included': [True, False], 'selection_included': [False, True], 'refit_included': [True, True],
                          'y': [1., 2.], 'label_end': ['2021-01-04', '2021-06-01'], 'origin_session': ['2020-12-28', '2021-05-21']})
        with self.assertRaises(ValueError): head_v2.masks(f)
    def test_development_loader_refuses_evaluation_years_before_reading(self):
        with patch.object(io.dec, 'td1_model', side_effect=AssertionError('payload read')):
            with self.assertRaises(PermissionError): head_v2.load_development(NAMES, years=[2022, 2023])
    def test_selection_sweep_picks_first_minimum_per_key(self):
        rng = np.random.default_rng(7); n = 300
        v = pd.DataFrame({'firm_id': [fid(i % 40) for i in range(n)], 'cutoff': pd.Timestamp('2021-03-06 04:59:59') + pd.to_timedelta(np.arange(n) // 40 * 7, 'D'),
                          'has_news': rng.random(n) < .6, 'y': rng.normal(size=n), 'b_development': rng.normal(size=n)})
        for n_ in NAMES:
            v['missing_' + n_] = rng.random(n) < .2
            for p in ['Z_'] + [f'Z{h}_' for h in HOPS]:
                v[p + n_] = np.where(v.has_news, rng.normal(size=n), 0.)
        heads = {a: {'training_beta': list(rng.normal(size=5))} for a in ['main', '1r', '2r', '3r']}
        grid, chosen, ablation = head_v2.selection_sweep(v, NAMES, heads, NAMES)
        for key, c in chosen.items():
            sub = [g for g in grid if f"{g['selector']}:{g['seed']}" == key]
            self.assertEqual(c, min(sub, key=lambda g: g['selection_mse']))  # min() keeps the first minimum
        self.assertEqual(len(ablation), 7 * len(SELECTORS))
    def test_checked_join_refuses_duplicates(self):
        f = pd.DataFrame([dict(firm_id='a', cutoff=1)])
        with self.assertRaises(pd.errors.MergeError): head_v2.checked_join(f, pd.concat([f, f]))

class Manifest(unittest.TestCase):
    def test_feature_projection_requests_no_label_columns(self):
        seen = []
        def capture(path, columns, datecol, lo, hi):
            seen.extend(columns); return pd.DataFrame(columns=columns)
        with patch.object(dec, 'verify_td1', return_value={'manifest_sha256': collections.defaultdict(lambda: 'h')}), \
             patch.object(dec, 'sha', return_value='h'), patch.object(dec, 'read_frame', side_effect=capture):
            io.manifest_features(2023)
        self.assertTrue(seen); self.assertEqual(forbidden_columns(seen), [])
    def test_2023_labels_refused_before_any_read(self):
        with patch.object(dec, 'verify_td1', side_effect=AssertionError('read')), patch.object(dec, 'read_frame', side_effect=AssertionError('read')):
            with self.assertRaises(PermissionError): io.manifest_labels(2023)

class Stream(unittest.TestCase):
    def test_exclusive_stream_and_partial_on_failure(self):
        import pyarrow as pa
        root = fixture_root(); schema = pa.schema([('a', pa.int64())]); final = root / 'x.parquet'
        with io.parquet_stream(final, schema) as w:
            w.write_table(pa.table({'a': [1, 2]}, schema=schema))
        self.assertEqual(pd.read_parquet(final).a.tolist(), [1, 2])
        with self.assertRaises(FileExistsError):
            with io.parquet_stream(final, schema) as w: pass
        with self.assertRaises(RuntimeError):
            with io.parquet_stream(root / 'y.parquet', schema) as w: raise RuntimeError('interrupted')
        self.assertFalse((root / 'y.parquet').exists()); self.assertEqual(len(list(root.glob('y.parquet.partial_*'))), 1)

class Cost(unittest.TestCase):
    def test_masked_prompt_contains_only_masks_for_articles(self):
        hops = {1: {'graph': 3, 'news': 1, 'partners': [('P1.1', ['CUSTOMER'], 0)]}}
        p = cost_v2.masked_prompt(dict.fromkeys(NAMES, 1.0), hops, [('YYYY-MM-DD', 'P1.1', 50), ('YYYY-MM-DD', 'P1.1', 7)])
        lines = p.split('Articles (masked):\n')[1].strip().split('\n')
        self.assertEqual([l.split(': ', 1)[1] for l in lines], ['x' * 50, 'x' * 7])
        self.assertEqual(len(p.encode('utf-8')), len(p))
    def test_dollar_formula(self):
        price = cost_v2.PRICES['off_peak']
        self.assertAlmostEqual(cost_v2.dollars(10, 1000, 20, 0, price), 10 * (1000 * .15 + 20 * .6) / 1e6)
        self.assertAlmostEqual(cost_v2.dollars(10, 1000, 20, 400, price), 10 * (600 * .15 + 400 * .003 + 20 * .6) / 1e6)
    def test_estimate_table_scenarios(self):
        rows = cost_v2.estimate_table(100, 4000., 800., {'a': .24, 'b': .3}, {'x': 20., 'y': 1700.})
        self.assertEqual(len(rows), 16); self.assertTrue(all(r['usd'] > 0 for r in rows))
    def test_missing_ledger_is_reported(self):
        self.assertEqual(cost_v2.ledger_calibration(Path('Z:/no/such/ledger.csv')), {'available': False})
    def test_v2_sources_have_no_network_or_provider_imports(self):
        forbidden = {'requests', 'httpx', 'urllib', 'socket', 'http', 'ssl', 'wrds', 'openai', 'anthropic', 'aiohttp'}
        for name in V2_FILES:
            tree = ast.parse((CODE / name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(forbidden.intersection(m.split('.')[0] for m in mods), name)

def synthetic_binding():
    f = {'schema': 'TD2_FREEZE_v2', 'version': VERSION, 'evaluation_years': [2023, 2024], 'maximum_label_end': '2024-12-31',
         **{k: {'synthetic': 'hash'} for k in evaluation_v3.BINDINGS if k != 'version'}}
    c = {k: f[k] for k in evaluation_v3.BINDINGS}
    c.update(schema='TD2_CLEARANCE_v1', status='CLEARED', freeze_sha256='F', evaluation_years=[2023, 2024], maximum_label_end='2024-12-31')
    return c, f

class Evaluation(unittest.TestCase):
    def test_absent_clearance_before_any_read(self):
        root = fixture_root(); paths = evaluation_v3.Paths(root, root, CODE, root / 'NONE.json')
        with patch.object(io, 'admitted_returns', side_effect=AssertionError('read')), patch.object(io, 'read_features', side_effect=AssertionError('read')), \
             patch.object(io, 'baseline_frame', side_effect=AssertionError('read')):
            with self.assertRaises(PermissionError): evaluation_v3.run(paths)
        self.assertFalse((root / 'EVALUATION_STARTED_v2.json').exists())
    def test_real_gate_path(self):
        self.assertEqual(str(evaluation_v3.REAL.clearance).replace('\\', '/'), 'D:/OptResearch_data/five_session/clearance/TD2_CLEARANCE_v1.json')
    def test_bindings_request_and_scope(self):
        c, f = synthetic_binding(); evaluation_v3.validate_clearance(c, f, 'F')
        for field in evaluation_v3.BINDINGS + ['freeze_sha256']:
            c2 = dict(c); c2[field] = 'tampered'
            with self.assertRaises(PermissionError): evaluation_v3.validate_clearance(c2, f, 'F')
        for change in [{'status': 'REQUESTED'}, {'request_only': True}, {'evaluation_years': [2023, 2024, 2025]}, {'maximum_label_end': '2025-01-10'}]:
            with self.assertRaises(PermissionError): evaluation_v3.validate_clearance(dict(c, **change), f, 'F')
    def test_label_requires_admission_and_skips_clipped_window(self):
        with self.assertRaises(PermissionError): evaluation_v3.label(None, None, 0, None)
        class Poison:
            def __getitem__(self, key): raise AssertionError('future returns touched')
        previous = io._evaluation_token; token = object(); io._evaluation_token = token
        try:
            self.assertEqual(evaluation_v3.label(pd.bdate_range('2024-12-24', periods=7), Poison(), 1, token), (None, None))
            dates = pd.bdate_range('2020-01-01', periods=6); r = np.array([9., .01, .02, .03, .04, .05])
            v, end = evaluation_v3.label(dates, r, 0, token)
            self.assertAlmostEqual(v, np.log(np.mean(r[1:] ** 2))); self.assertEqual(end, dates[5])
        finally:
            io._evaluation_token = previous
    def test_returns_reader_refuses_without_token(self):
        with self.assertRaises(PermissionError): io.admitted_returns(OUT, 2023, None)
    def test_end_to_end_synthetic_gate_labels_and_latch(self):
        root = fixture_root(); out = root / 'out'; td1 = root / 'td1'; rng = np.random.default_rng(5)
        dates = pd.bdate_range('2023-01-02', '2024-12-31'); permnos = [201, 202, 203]
        crsp = pd.DataFrame([(d, p, rng.normal(0, .02), 0) for d in dates for p in permnos], columns=['date', 'permno', 'ret', 'share_type'])
        crsp['ret'] = crsp.ret.astype('float32').astype(float)
        for y in (2023, 2024):
            part = crsp[crsp.date.dt.year == y].reset_index(drop=True)
            # Mirror the real TD1 input schema: date32, int32 permno, float32 ret, int8 share_type;
            # one share_type=1 row per year must be filtered out by the runner.
            part = pd.concat([part, pd.DataFrame([(part.date.iloc[3], 201, 9.0, 1)], columns=part.columns)], ignore_index=True)
            part = part.assign(date=part.date.dt.date, permno=part.permno.astype('int32'), ret=part.ret.astype('float32'),
                               share_type=part.share_type.astype('int8'))
            io.write_frame(td1 / f'inputs/crsp_{y}.parquet', part)
        origins = [d for d in dates if d.weekday() == 4]; feats = []; preds = []
        for o in origins:
            cut = o + pd.Timedelta(hours=28, minutes=59, seconds=59)
            for p in (201, 202, 'zz'):
                firm = fid(p); news = bool(rng.random() < .7); miss = {n: bool(rng.random() < .2) for n in NAMES}
                preds.append({'firm_id': firm, 'cutoff': cut, 'origin_session': o, 'b': float(rng.normal(-8, .5))})
                zs = {k: (rng.normal(size=5) if news else np.zeros(5)) for k in ['Z_', 'Z1_', 'Z2_', 'Z3_']}
                for variant in ('primary', 'fsb5_sensitivity'):
                    r = {'firm_id': firm, 'cutoff': cut, 'origin_session': o, 'variant': variant, 'mapped_focal': p != 'zz', 'has_news': news}
                    for i, n in enumerate(NAMES):
                        r.update({'C_' + n: np.nan if miss[n] else 1., 'available_' + n: pd.NaT, 'missing_' + n: miss[n],
                                  'Z_' + n: 0. if miss[n] else zs['Z_'][i], 'E_' + n: 0.})
                        for h in HOPS:
                            r[f'Z{h}_{n}'] = 0. if miss[n] else zs[f'Z{h}_'][i]; r[f'E{h}_{n}'] = 0.
                    for h in HOPS:
                        r.update({f'H{h}_graph_partners': 1, f'H{h}_news_partners': int(news), f'H{h}_selected_partners': int(news), f'H{h}_unique_articles': int(news)})
                    feats.append(r)
        feats = pd.DataFrame(feats); preds = pd.DataFrame(preds)
        for y in (2023, 2024):
            io.write_frame(out / f'features_v2/year_{y}.parquet', build_v2.to_table(feats[feats.origin_session.dt.year == y].to_dict('records'), build_v2.feature_schema()).to_pandas())
            io.write_frame(td1 / f'predictions_v2/year_{y}.parquet', preds[preds.origin_session.dt.year == y].reset_index(drop=True))
        beta = [.2, -.1, .05, .3, -.2]
        model = {'names': NAMES, 'beta': beta, 'heads': {a: {'beta': list(rng.normal(size=5))} for a in ['1r', '2r', '3r']},
                 'training_beta_order': NAMES, 'selected_local_gates': {f'{k}:{s}': {'K': 2, 'fallback': 'none'} for k, s in SELECTORS}}
        new_json(out / 'MODEL_v2.json', model, timed=False); new_json(out / 'QUALIFICATION_fixture.json', {'tests': 1}, timed=False)
        arts = [out / 'MODEL_v2.json'] + [out / f'features_v2/year_{y}.parquet' for y in (2023, 2024)]
        t1 = [td1 / f'{d}/{s}_{y}.parquet' for y in (2023, 2024) for d, s in [('inputs', 'crsp'), ('predictions_v2', 'year')]]
        freeze = {'schema': 'TD2_FREEZE_v2', 'version': VERSION, 'code_sha256': {str(p): sha(p) for p in sorted(CODE.glob('*.py'))},
                  'artifact_sha256': {str(p): sha(p) for p in arts}, 'td1_sha256': {str(p): sha(p) for p in t1},
                  'model_sha256': sha(out / 'MODEL_v2.json'), 'qualification_sha256': {str(out / 'QUALIFICATION_fixture.json'): sha(out / 'QUALIFICATION_fixture.json')},
                  'evaluation_years': [2023, 2024], 'maximum_label_end': '2024-12-31'}
        new_json(out / 'FREEZE_v2.json', freeze, timed=False)
        clearance = {**{k: freeze[k] for k in evaluation_v3.BINDINGS}, 'schema': 'TD2_CLEARANCE_v1', 'status': 'CLEARED',
                     'freeze_sha256': sha(out / 'FREEZE_v2.json'), 'evaluation_years': [2023, 2024], 'maximum_label_end': '2024-12-31'}
        paths = evaluation_v3.Paths(out, td1, CODE, td1 / 'clearance/TD2_CLEARANCE_v1.json')
        new_json(paths.clearance, clearance, timed=False)
        evaluation_v3.run(paths)
        result = json.loads((out / 'EVALUATION_RESULT_v2.json').read_text(encoding='utf-8'))
        # Independent recomputation for primary, training_beta, K=all, main arm.
        wide = crsp.pivot(index='date', columns='permno', values='ret'); gains = []
        f = feats[feats.variant == 'primary'].merge(preds, on=['firm_id', 'cutoff', 'origin_session'])
        for r in f.itertuples():
            if r.firm_id not in (fid(201), fid(202)): continue
            pos = dates.get_loc(r.origin_session)
            if pos + 5 >= len(dates): continue
            w = wide[201 if r.firm_id == fid(201) else 202].to_numpy()[pos + 1:pos + 6]; yv = np.log(np.mean(w ** 2))
            pred = r.b + (sum(getattr(r, 'Z_' + n) * beta[i] for i, n in enumerate(NAMES) if not getattr(r, 'missing_' + n)) if r.has_news else 0.)
            gains.append((yv - r.b) ** 2 - (yv - pred) ** 2)
        arm = [a for a in result['arms'] if a['variant'] == 'primary' and a['selector'] == 'training_beta' and a['K'] == 'all' and a['arm'] == 'main'][0]
        self.assertEqual(arm['n'], len(gains)); self.assertAlmostEqual(arm['mean'], float(np.mean(gains)), places=12)
        k0 = [a for a in result['arms'] if a['K'] == 0][0]; self.assertEqual(k0['mean'], 0.0)
        cov = {(c['year'], c['variant']): c for c in result['coverage']}
        self.assertGreater(cov[(2024, 'primary')]['rows_without_label'], 0)
        self.assertEqual(len(result['arms']), 2 * len(SELECTORS) * len(K_GRID) * 7)
        self.assertEqual(len(list((out / 'evaluation_v2').glob('squared_errors_*.parquet'))), 2 * len(SELECTORS))
        with self.assertRaises(PermissionError): evaluation_v3.run(paths)

class FreezeRequest(unittest.TestCase):
    def test_request_is_never_a_clearance_and_paths_exclude_partials(self):
        c, f = synthetic_binding(); req = dict(c, status='REQUESTED', request_only=True)
        with self.assertRaises(PermissionError): evaluation_v3.validate_clearance(req, f, 'F')
        root = fixture_root(); (root / 'features_v2').mkdir()
        new_json(root / 'features_v2/year_2015.parquet.partial_abc', {'x': 1}, timed=False)
        self.assertFalse([p for p in freeze_v2.artifact_paths(root) if 'partial' in p.name])
        self.assertEqual(str(freeze_v2.CLEARANCE).replace('\\', '/'), 'D:/OptResearch_data/five_session/clearance/TD2_CLEARANCE_v1.json')

SUITES = [BuildFixture, Quarantine, Gates, Head, Manifest, Stream, Cost, Evaluation, FreezeRequest]

def run_v2():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in SUITES)
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    return {'tests': r.testsRun, 'passed': r.testsRun - len(r.errors) - len(r.failures), 'errors': len(r.errors),
            'failures': len(r.failures), 'synthetic_only': True}
