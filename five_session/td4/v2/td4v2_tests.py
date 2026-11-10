"""TD4 v2 offline tests: the v1 suite (it tests the v1 modules v2 reuses unchanged) plus v2-specific tests.
Synthetic fixtures under D:/OptResearch_data/five_session/td4/fixtures/<uuid>/ (never deleted); invented examples."""
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
import td4v2_spec as S


def fixture_dir():
    d = T.OUT / 'fixtures' / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    return d


def _R():
    E.paths()
    import td3_runtime as R
    return R


class TestV2Spec(unittest.TestCase):
    def test_prereg_v2(self):
        t = S.preregistration_text()
        self.assertEqual(t, S.preregistration_text())
        for s in ('P1', 'P2', 'report no validated total gain', 'report no Jev advantage', 'historical reproduction',
                  'OpenAI', 'October 2025'):
            self.assertIn(s, t)
        self.assertEqual(S.RUN_ORDER, ['2122', '2023', '2024', '2025'])
        self.assertEqual(S.LABEL_END_BOUND, '2025-12-31')
        self.assertNotEqual(S.preregistration_sha256(), T.preregistration_sha256())

    def test_record_key_and_rank(self):
        import td4v2_build as BV
        E.paths()
        import td3_build as B
        for f, c in (('aa11', '2021-03-06T04:59:59'), ('bb22', '2022-11-12T04:59:59')):
            self.assertEqual(BV.record_key(f, pd.Timestamp(c)), B.record_key(f, pd.Timestamp(c)))
        k = BV.record_key('aa11', pd.Timestamp('2021-03-06T04:59:59'))
        self.assertEqual(BV.select_rank(k), BV.select_rank(k))
        e = pd.DataFrame({'firm_id': [f'f{i}' for i in range(50)],
                          'cutoff': [pd.Timestamp('2021-01-09 04:59:59')] * 50})
        o = BV._order(e)
        ranks = [BV.select_rank(BV.record_key(a, b)) for a, b in zip(o.firm_id, o.cutoff)]
        self.assertEqual(ranks, sorted(ranks))


def _write_build(R, d, recs, statuses, withheld):
    d.mkdir(parents=True, exist_ok=True)
    w = R.JsonlWriter(d / 'INPUTS.jsonl', R.Redactor())
    for i, (k, text) in enumerate(recs, 1):
        w.write({'key': k, 'sample_rank': i, 'input_text': text, 'input_sha256': R.sha_text(text),
                 'input_bytes': len(text.encode())})
    w.close()
    R.new_json(d / 'BUILD.json', {'counts': {}})
    R.new_json(d / 'WITHHELD_POPULATION_v2.json', {'withheld_keys': sorted(withheld)})
    E.write_parquet(d / 'ELIGIBLE.parquet', pd.DataFrame(statuses, columns=['firm_id', 'cutoff', 'status', 'key']))


class TestV2Inputs(unittest.TestCase):
    def test_merge_withhold_rank_offset(self):
        import td4v2_build as BV
        import td4v2_inputs as IV
        R = _R()
        root = fixture_dir()
        main, rep = root / 'main', root / 'rep'
        c = pd.Timestamp('2023-03-04 04:59:59')
        _write_build(R, main, [('k1', 'FROZEN INPUT (x)\na'), ('k2', 'FROZEN INPUT (x)\nb')],
                     [('f1', c, 'input_built', 'k1'), ('f2', c, 'input_built', 'k2'),
                      ('f3', c, 'withheld_date_residual', None)], {'k2'})
        _write_build(R, rep, [('k3', 'FROZEN INPUT (x)\nc')], [('f3', c, 'input_built', 'k3')], set())
        saved = (dict(BV.YEAR_DIR), dict(BV.REPAIR_DIR))
        try:
            BV.YEAR_DIR[2023], BV.REPAIR_DIR[2023] = main, rep
            recs = IV.callable_inputs('2023')
            self.assertEqual(sorted(recs), ['k1', 'k3'])                       # k2 population-withheld
            self.assertEqual(recs['k3']['sample_rank'], IV.REPAIR_RANK_OFFSET + 1)
            self.assertEqual(recs['k3']['input_text'], 'FROZEN INPUT (x)\nc')
            m = IV.eligible_map('2023')
            self.assertEqual(m, {('f1', c): 'k1', ('f3', c): 'k3'})
        finally:
            BV.YEAR_DIR.clear(); BV.YEAR_DIR.update(saved[0])
            BV.REPAIR_DIR.clear(); BV.REPAIR_DIR.update(saved[1])

    def test_missing_population_check_refused(self):
        import td4v2_inputs as IV
        d = fixture_dir()
        with self.assertRaises(PermissionError):
            IV.withheld_file(d)


class TestV2Gate(unittest.TestCase):
    def test_absent_request_only_inventory(self):
        import td4v2_gate as G
        R = _R()
        d = fixture_dir()
        ok, reasons, _ = G.check_a(d / 'none.json', d / 'f.json')
        self.assertFalse(ok)
        inv = {'a.py': 'x'}
        R.new_json(d / 'f.json', {'code_sha256': inv})
        R.new_json(d / 'c.json', {'schema': S.CLEARANCE_A_SCHEMA, 'status': 'CLEARED', 'request_only': True,
                                  'freeze_sha256': G.sha(d / 'f.json'), 'code_sha256': inv})
        ok, reasons, _ = G.check_a(d / 'c.json', d / 'f.json', inventory={'a.py': 'y'})
        self.assertFalse(ok)
        self.assertTrue(any('request is never' in r for r in reasons))
        self.assertTrue(any('code inventory' in r for r in reasons))
        if not S.CLEARANCE_A.exists():
            for step in G.STEPS_A:
                with self.assertRaises(PermissionError):
                    G.admit(step)
            with self.assertRaises(PermissionError):
                G.admit_b()

    def test_outcome_tokens(self):
        import td4v2_outcomes as O
        import td4v2_gate as G
        keys = pd.DataFrame({'firm_id': ['x'], 'cutoff': [pd.Timestamp('2024-01-06')],
                             'origin_session': [pd.Timestamp('2024-01-05')]})
        with self.assertRaises(PermissionError):
            O.selection_labels((2021,), None)
        with self.assertRaises(PermissionError):
            O.test_labels(2024, keys, G.Token('evaluate', 'fake'))            # not issued by admit_b
        tok = G.Token('evaluate', 'x')
        G._tokens[id(tok)] = tok
        try:
            with self.assertRaises(PermissionError):
                O.test_labels(2022, keys, tok)                                 # not a test year
        finally:
            G._tokens.pop(id(tok))

    def test_order_rule(self):
        import td4v2_runner as RV
        state = {'2122': True, '2023': False}
        fn = lambda p: {'main_complete': state.get(p, False)}
        self.assertTrue(RV.check_order('2023', fn))
        with self.assertRaises(PermissionError):
            RV.check_order('2024', fn)
        self.assertTrue(RV.check_order('2122', fn))


class TestV2Models(unittest.TestCase):
    def test_heads_follow_td2_protocol(self):
        import td4v2_score as SC2
        m = SC2.td2_model()
        sh, th = SC2.selection_heads(), SC2.test_heads()
        self.assertEqual(sh['beta'], m['heads']['main']['training_beta'])
        self.assertEqual(th['beta'], m['beta'])
        self.assertNotEqual(sh['beta'], th['beta'])
        self.assertEqual(sh['training_beta_order'], m['training_beta_order'])
        for a in ('1r', '2r', '3r'):
            self.assertEqual(sh['heads'][a]['beta'], m['heads'][a]['training_beta'])
            self.assertEqual(th['heads'][a]['beta'], m['heads'][a]['beta'])

    def test_concat_outputs_and_readings(self):
        import td4v2_score as SC2
        import td4v2_evaluate as EV
        one = {a: {'ordering': [None], 'choice': [None], 'yes_scores': [None], 'status': ['no_output'],
                   'valid': np.array([False])} for a in T.PROVIDER_ARMS}
        out = SC2.concat_outputs([one, one])
        self.assertEqual(len(out['deepseek_order']['status']), 2)
        self.assertEqual(out['jev_noul']['valid'].dtype, bool)
        r = EV.readings({'P1': {'confirmed': False}, 'P2': {'confirmed': False}})
        self.assertEqual(r['P1'], S.PREREGISTRATION['stop_rule_readings']['P1_not_confirmed'])
        self.assertEqual(r['P2'], S.PREREGISTRATION['stop_rule_readings']['P2_not_confirmed'])


class TestV2Labels(unittest.TestCase):
    def batch(self):
        import td4v2_labels as LB
        ids = ['C0000000001', 'C0000000002']
        return {'contexts': [{'context_id': i, 'context': 'x'} for i in ids], 'output_schema': LB.output_schema(ids)}

    def good(self):
        rel = {'size': 3, 'cash': 0, 'cash_flow': 1, 'leverage': 2, 'inventory': 0}
        return {'labels': {'C0000000001': {'relevance': dict(rel), 'top_choice': 'size'},
                           'C0000000002': {'relevance': dict(rel), 'top_choice': 'none'}}}

    def test_validation(self):
        import td4v2_labels as LB
        b = self.batch()
        self.assertEqual(LB.validate_batch(b, self.good()), [])
        bad = self.good(); bad['labels'].pop('C0000000002')
        self.assertTrue(LB.validate_batch(b, bad))
        bad = self.good(); bad['labels']['C0000000001']['relevance']['size'] = 4
        self.assertTrue(LB.validate_batch(b, bad))
        bad = self.good(); bad['labels']['C0000000001']['relevance']['size'] = True
        self.assertTrue(LB.validate_batch(b, bad))
        bad = self.good(); bad['labels']['C0000000001']['extra'] = 1
        self.assertTrue(LB.validate_batch(b, bad))
        bad = self.good(); bad['labels']['C0000000001']['top_choice'] = 'beta'
        self.assertTrue(LB.validate_batch(b, bad))
        bad = self.good(); bad['labels']['C0000000001']['relevance'].pop('cash')
        self.assertTrue(LB.validate_batch(b, bad))
        bad = self.good(); bad['notes'] = 'x'
        self.assertTrue(LB.validate_batch(b, bad))

    def test_schema_and_note(self):
        import td4v2_labels as LB
        sch = LB.output_schema(['C1'])
        self.assertFalse(sch['properties']['labels']['additionalProperties'])
        self.assertEqual(sch['properties']['labels']['required'], ['C1'])
        self.assertIn('OpenAI', S.LLM_LABEL_NOTE)
        self.assertIn('not human ground truth', S.LLM_LABEL_NOTE)
        self.assertIn('Do not try to identify any firm, date or outcome', LB.INSTRUCTIONS)


class TestV2PublicFolder(unittest.TestCase):
    def test_v2_folder_clean(self):
        bad = [p.name for p in S.CODE_V2.iterdir() if p.is_file() and p.suffix not in ('.py', '.md')]
        self.assertEqual(bad, [])
        self.assertFalse((S.CODE_V2 / '__pycache__').exists())
        R = _R()
        code = '\n'.join(p.read_text(encoding='utf-8') for p in sorted(S.CODE_V2.glob('*.py')) +
                         sorted(S.CODE_V2.glob('*.md')))
        low = code.lower()
        import td4v2_build2 as B2
        for d in list(B2.PERIOD_DIR.values()):
            p = Path(d) / 'AUDIT_META.jsonl'
            if not p.exists():
                continue
            names, windows = set(), set()
            with p.open(encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    m = json.loads(line)
                    for a in m['aliases'].values():
                        names.update(c for c in a.get('comnams', []) if len(c) >= 8)
                    for x in m['articles']:
                        w = re.findall(r'[A-Za-z]+', x['body_raw'])
                        windows.update(' '.join(w[j:j + 8]).lower() for j in range(0, max(0, len(w) - 8), 3))
            self.assertEqual([n for n in names if n.lower() in low], [])
            flat = ' '.join(re.findall(r'[A-Za-z]+', code)).lower()
            self.assertEqual([w for w in windows if w in flat], [])


def qualify():
    R = E.install('offline')
    import td4_tests as V1
    import td4v2_gate as G
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    v1 = (V1.TestSpec, V1.TestGates, V1.TestLabels, V1.TestCopies, V1.TestRunner, V1.TestGate, V1.TestSelectEvaluate,
          V1.TestDiagnostics, V1.TestPublicFolder)
    import td4v2_tests_m2 as TM
    import td4v2_tests_redact as TRD
    import td4v2_tests_redact22 as TRD22
    import td4v2_tests_v23 as TV23
    v2 = (TestV2Spec, TestV2Gate, TestV2Models, TestV2Labels, TestV2PublicFolder) + TM.CLASSES + tuple(TRD.tests()) + \
        tuple(TRD22.tests()) + tuple(TV23.tests())
    for cls in v1 + v2:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    res = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    n_v2 = sum(loader.loadTestsFromTestCase(c).countTestCases() for c in v2)
    n_v23 = sum(loader.loadTestsFromTestCase(c).countTestCases() for c in TV23.tests())
    out = {'version': S.VERSION, 'code_version': S.CODE_VERSION, 'tests': res.testsRun, 'v1_tests': res.testsRun - n_v2,
           'v2_tests': n_v2, 'v2_3_tests': n_v23,
           'failures': len(res.failures), 'errors': len(res.errors), 'skipped': len(res.skipped),
           'failed_tests': [str(t) for t, _ in res.failures + res.errors], 'code_sha256': G.code_inventory(),
           **R.peak()}
    p = R.new_json(T.OUT / f'QUALIFICATION_TD4v2_{uuid.uuid4().hex}.json', out)
    print('QUALIFICATION_V2', p, json.dumps({k: out[k] for k in ('tests', 'v1_tests', 'v2_tests', 'failures',
                                                                 'errors', 'skipped')}), flush=True)
    return 0 if not (res.failures or res.errors) else 1
