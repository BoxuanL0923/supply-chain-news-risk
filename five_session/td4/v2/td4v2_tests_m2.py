"""Offline tests of masker v2 (td4v2_mask), the independent v2 scanner (td4v2_scan), the masker-v2 inputs gate
(td4v2_inputs), the runner fixes N1/N2/N4/N5 (td4v2_runner) and the clearance-B hardening R1 (td4v2_gate).
Every firm, ticker and text here is invented. Fixtures under D:/OptResearch_data/five_session/td4/fixtures/<uuid>/."""
import json
import unittest
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_spec as S
from td4v2_tests import fixture_dir, _R

_ENV = {}
TEST_TABLE = {
    'schema': 'TD4_MASK_TABLES_v2', 'written_utc': 'test',
    'common_names': {'ZORBEX MOTORS': {'aliases': ['Zorbex Dynamo Unit'], 'source': 'test'}},
    'owner_aliases': {'QUILLMARK UNITED': {'aliases': ['United'], 'source': 'test'}},
    'always_acronyms': {'ZORBEX MOTORS': {'acronyms': ['ZWS'], 'source': 'test'}},
    'renames': {'999001': {'names': ['GLIMRA HOLDINGS INC'], 'tickers': [], 'dated': '2025-03-01',
                           'source': 'test'}},
}


def env(last_day='2025-12-31'):
    E.paths()
    import td3_build as B
    import td4v2_mask as MV
    import td4v2_words as W
    if 'english' not in _ENV:
        _ENV['english'] = B.load_english(Path('C:/Users/Boxuan/AppData/Roaming/nltk_data/corpora/words/en'))
        _ENV['common'] = W.load_common()[0] if W.COMMON_FILE.exists() else frozenset({'live', 'price', 'now'})
    MV.install(_ENV['common'], {}, MV.validate_tables_v2(json.loads(json.dumps(TEST_TABLE))), last_day)
    return MV, _ENV['english']


def firm(comnams, tickers, fs=(), permno=None):
    import td3_mask as M
    MV, english = env()
    variants = set()
    for c in comnams:
        variants |= M.name_variants(c)
    al = {'names': sorted(variants), 'common': sorted(M.common_aliases(comnams)), 'tickers': list(tickers),
          'tickers_extra': [], 'rics': []}
    al['embedded'] = [list(x) for x in M.embedded_terms(al, english)]
    al['fuzzy'] = M.fuzzy_words(al, english)
    al['products'] = M.firm_products(comnams)
    codes = M.code_prefix_terms(comnams, list(fs), tickers, english)
    al['code_prefixes'] = sorted(codes)
    al['unit_prefixes'] = M.unit_prefixes(codes)
    al['near'] = []
    return MV.add_v2(al, comnams, list(fs), [], english, permno=permno)


def mask(entries, text, own=()):
    MV, english = env()
    mk = MV.MaskerV2(entries, english, own=own)
    _, body, *_ = mk.mask_article('', text)
    return body, mk


class TestMaskV2Names(unittest.TestCase):
    def test_b1_plural_ticker_and_initialism_never_kept(self):
        e = [('[FOCAL]', firm(['ZAP SYSTEMS INC'], ['ZAPS']))]
        out, mk = mask(e, '* PARTNERS SAY ZAPS SHARES ROSE ON THE NEWS TODAY')
        self.assertNotIn('ZAPS', out)
        self.assertEqual(mk.residual(out), [])

    def test_two_letter_initials_and_initialism_rest(self):
        e = [('[SUPPLIER_1]', firm(['ZORBEX QUILLON ENTERPRISE CO'], ['ZQE']))]
        out, mk = mask(e, 'ZQ owed money; ZQ Enterprise said it would appeal.')
        self.assertEqual(out.count('[SUPPLIER_1]'), 2)
        self.assertNotIn('ZQ', out)
        self.assertTrue(any(h['kind'] == 'init2' for h in mk.residual('ZQ owed money')))

    def test_digit_name_lower_case(self):
        e = [('[FOCAL]', firm(['8ZORB INC'], ['EZRB']))]
        out, mk = mask(e, 'BRIEF-8zorb raises cash; 8ZORB and 8Zorb too')
        self.assertNotIn('zorb', out.lower())
        self.assertTrue(any(h['kind'] == 'digit_name' for h in mk.residual('shares of 8zorb fell')))

    def test_us_word_and_single_letter_next(self):
        e = [('[FOCAL]', firm(['UNITED STATES ZORBALT CORP'], ['USZ'])),
             ('[LINK_1]', firm(['Q QUILLABS THERAPEUTICS INC'], ['QQLB']))]
        out, _ = mask(e, 'U.S. Zorbalt and US Zorbalt said; Q-Quillabs rose.')
        self.assertNotIn('Zorbalt', out)
        self.assertNotIn('Quillabs', out)

    def test_split_needs_separator(self):
        e = [('[FOCAL]', firm(['GLIMMERQUILL INC'], ['GMQL'], fs=['GlimmerQuill Inc.']))]
        out, _ = mask(e, 'Glimmer Quill said; Glimmer-Quill too.')
        self.assertNotIn('Quill', out)

    def test_leading_part_and_nonfirst_word(self):
        e = [('[FOCAL]', firm(['ZORBEXMOTORS CORP'], ['ZBXM'])),
             ('[LINK_1]', firm(['QUILL ZORVANDER INDUSTRIES INC'], ['QZV']))]
        out, _ = mask(e, 'Zorbex said; founder Dan Zorvander spoke.')
        self.assertNotIn('Zorbex', out)
        self.assertNotIn('Zorvander', out)

    def test_compound_camel_lead_suffix(self):
        e = [('[FOCAL]', firm(['QX SEMICONDUCTOR CORP'], ['QX'])),
             ('[LINK_1]', firm(['GLIMMER PAYMENTS INC'], ['GLMP'])),
             ('[LINK_2]', firm(['ZORBEXIA CORP'], ['QZX']))]
        out, _ = mask(e, 'Qxsemi and QXSEMI grew; GlimmerNet launched; Zorbexiaweapons and Betqzx too.')
        for w in ('semi', 'GlimmerNet', 'weapons', 'Betqzx'):
            self.assertNotIn(w, out)

    def test_weekday_never_alias(self):
        e = [('[FOCAL]', firm(['MONDAY ZORB LTD'], ['MZRB']))]
        out, _ = mask(e, 'The deal, announced on Monday, closed.')
        self.assertIn('on Monday', out)

    def test_rename_table_by_period(self):
        MV, english = env('2025-12-31')
        a = firm(['ZORBQUILL CORP'], ['ZBQ'], permno=999001)
        self.assertIn('GLIMRA HOLDINGS INC', a['v2']['rename_names'])
        MV.install(_ENV['common'], {}, json.loads(json.dumps(TEST_TABLE)), '2024-12-31')
        import td3_mask as M
        b = MV.add_v2({'names': sorted(M.name_variants('ZORBQUILL CORP')), 'common': [], 'tickers': ['ZBQ'],
                       'tickers_extra': []}, ['ZORBQUILL CORP'], [], [], english, permno=999001)
        self.assertEqual(b['v2']['rename_names'], [])


class TestMaskV2Context(unittest.TestCase):
    def test_collocations_and_slug_and_index_kept(self):
        e = [('[FOCAL]', firm(['PRICE ZORB INC'], ['PZRB'])), ('[LINK_1]', firm(['INSIGHT QUILLON CORP'], ['IQLN'])),
             ('[LINK_2]', firm(['NASDAQ ZORB INC'], ['NZRB']))]
        out, _ = mask(e, 'The purchase Price of $5 was set. Indexes: Nasdaq down 0.5%, Dow up.')
        self.assertIn('purchase Price of', out)
        self.assertIn('Nasdaq down', out)
        head, _ = mask(e, 'UPDATE 1-INSIGHT-How talks went')                # a slug at the start of a headline
        self.assertIn('INSIGHT-How', head)

    def test_aux_upper_case_kept_company_context_masked(self):
        e = [('[FOCAL]', firm(['LIVE ZORB INC'], ['LVZR']))]
        out, _ = mask(e, '* APP: BUY FOR ME IS CURRENTLY LIVE IN APP')
        self.assertIn('CURRENTLY LIVE IN', out)

    def test_adjacent_capitalised_word_absorbed(self):
        e = [('[FOCAL]', firm(['ZORBEX CORP'], ['ZRBX']))]
        out, _ = mask(e, 'fans at rival Zorbex Arena cheered, then Tamsin Zorbex spoke.')
        self.assertNotIn('Arena', out)
        self.assertNotIn('Tamsin', out)
        self.assertIn('[FOCAL] cheered', out)

    def test_own_article_month_ticker_and_title_ticker(self):
        e = [('[FOCAL]', firm(['NOV ZORB INC'], ['NOV'])), ('[LINK_1]', firm(['ZORPLEX CORP'], ['ZORP']))]
        out, _ = mask(e, '* NOV ANNOUNCES LEADERSHIP CHANGES * NOV 5 MEETING. BRIEF-Zorp Announces Deal',
                      own=('[FOCAL]', '[LINK_1]'))
        self.assertIn('[FOCAL] ANNOUNCES', out)
        self.assertIn('[DATE] MEETING', out)
        self.assertIn('BRIEF-[LINK_1]', out)

    def test_own_codes_by_length(self):
        e = [('[FOCAL]', firm(['ZORBEX PHARMA CORP'], ['ZRBX']))]
        own, _ = mask(e, 'trial of ZB-12 and ZBX-801', own=('[FOCAL]',))
        other, _ = mask(e, 'trial of ZB-12 and ZBX-801')
        self.assertNotIn('ZB-12', own)
        self.assertIn('ZB-12', other)                      # 2-letter subsequence: own article only
        self.assertNotIn('ZBX-801', other)                 # 3 letters: wherever the firm is a token

    def test_hyphen_ric_model_unit(self):
        e = [('[FOCAL]', firm(['GLIM A ZORB WORKSHOP INC'], ['GAZW'])),
             ('[LINK_1]', firm(['ZEPHAIRO AIRCRAFT CORP'], ['ZPHR']))]
        out, mk = mask(e, 'can sue Glim-A-Zorb now; Zephairo.MI fell; the Zephairo 737-800 jet; '
                          'Zephairo Financial Services Software Ltd said.')
        self.assertNotIn('Zorb', out)
        self.assertNotIn('.MI', out)
        self.assertNotIn('737', out)
        self.assertNotIn('Software', out)
        kinds = {h['kind'] for h in mk.residual('[FOCAL]-Zorb and [LINK_1].MI and [LINK_1] 737-800')}
        self.assertTrue({'v2_hyphen_own_word', 'v2_ric_suffix', 'v2_model_after_token'} <= kinds)

    def test_template_subject_and_always_acronym(self):
        e = [('[FOCAL]', firm(['ZORBEX MOTORS CORP'], ['ZBXM']))]
        out, _ = mask(e, 'Zorbex Holdco Group Ltd reports results for the quarter ended - Earnings Summary',
                      own=('[FOCAL]',))
        self.assertTrue(out.startswith('[FOCAL] reports results'))
        out2, _ = mask(e, 'A partner collaborates with ZWS and the Zorbex Dynamo Unit. BRIEF-ZWS Says More')
        self.assertNotIn('ZWS', out2)
        self.assertNotIn('Dynamo', out2)
        e2 = [('[CUSTOMER_1]', firm(['QUILLMARK UNITED PLC'], ['QMKU']))]
        out3, _ = mask(e2, 'BRIEF-Quillmark United Confirms Coach Of United, Tam Doe To Leave', own=('[CUSTOMER_1]',))
        self.assertNotIn('United', out3)


class TestMaskV2DatesStory(unittest.TestCase):
    def test_dates(self):
        MV, _ = env()
        t = MV.mask_dates_v2("NEW YORK, July 1Zorbex rose. AS OF MARCH, 31, 2025. TUESDAY, APRIL 29TH. Exiting CY26 "
                             "into '27.")
        for s in ('July 1', 'MARCH, 31', '29TH', 'CY26', "'27", '2025'):
            self.assertNotIn(s, t)
        self.assertIn('[DATE] Zorbex', t)
        self.assertEqual(MV.date_residuals_v2(t), [])
        self.assertIn('v2_month_comma_day', MV.date_residuals_v2('AS OF MARCH, 31'))

    def test_story_fields_equal_length_and_no_backtracking(self):
        MV, _ = env()
        s = 'x STORY_NUMBER: nZRB1234Q STORY_DATE: 07/07/2025 STORY_TIME: 01:59:26 PM GMT tail STORY_TIME: [kept]'
        out, n = MV.story_fields_v2(s)
        self.assertEqual(len(out), len(s))
        self.assertEqual(n, 3)
        self.assertNotIn('TIME: 01', out)
        self.assertIn('STORY_TIME: [kept]', out)
        self.assertTrue(MV.STORY_STUB.search('click IME:07:30 PM GMT'))
        self.assertTrue(MV.CLOCK_SECONDS.search('at 01:59:26 GMT'))

    def test_mistag(self):
        MV, _ = env()
        self.assertTrue(MV.mistag('ADVISORY-Alert withdrawn', 'An alert was inadvertently tagged to Zorbex.'))
        self.assertFalse(MV.mistag('Zorbex raises outlook', 'Shares rose.'))


class TestScanV2(unittest.TestCase):
    def test_detectors(self):
        import td4v2_scan as SC
        _, english = env()
        t = SC.Terms(['ZORBEX QUILL CORP', '8ZORB INC'], ['ZRBQ'], english, frozenset())
        terms = {1: t}
        hits = SC.scan_text('ZRBQ fell; Zorbex said; ZQ rose; 8zorb up; [FOCAL]-Quill; on [FOCAL], ok. '
                            'July 1Zorbex; STORY_TIME: 1; [FOCAL].MI', {'[FOCAL]': 1}, terms, {'[FOCAL]'})
        cls = {h[0] for h in hits}
        for c in ('S1_ticker', 'S3_name_word', 'S2_initialism', 'S5_digit_name', 'S8_hyphen_name_remnant',
                  'S10_date', 'S11_story_field', 'S14_ric_suffix'):
            self.assertIn(c, cls)
        self.assertTrue(set(SC.WITHHOLD_CLASSES) >= {'S10_date', 'S11_story_field', 'S12_mistag_notice'})


class TestInputsM2(unittest.TestCase):
    def test_masker_v2_receipt_population_check_and_redaction_required(self):
        import td4v2_build2 as B2
        import td4v2_inputs as IV
        R = _R()
        d = fixture_dir()
        w = R.JsonlWriter(d / 'INPUTS.jsonl', R.Redactor())
        for i, k in enumerate(('k1', 'k2', 'k3'), 1):
            txt = 'FROZEN INPUT (x)\n' + k
            w.write({'key': k, 'sample_rank': i, 'input_text': txt, 'input_sha256': R.sha_text(txt),
                     'input_bytes': len(txt)})
        w.close()
        R.new_json(d / 'BUILD.json', {'counts': {}})
        saved = dict(B2.PERIOD_DIR)
        try:
            B2.PERIOD_DIR[2023] = d
            with self.assertRaises(PermissionError):
                IV.callable_inputs('2023')                                   # no masker-v2 receipt
            R.new_json(d / 'MASK_V2_RECEIPT.json', {'v1_masker_inputs_used': False})
            with self.assertRaises(PermissionError):
                IV.callable_inputs('2023')                                   # no population check
            R.new_json(d / 'WITHHELD_POPULATION_M2.json', {'withheld_keys': ['k2']})
            with self.assertRaises(PermissionError):
                IV.callable_inputs('2023')                                   # no redaction result
            R.new_json(d / 'DETECTOR_FLAGS_M2.json', {'withheld_keys': ['k1']})   # superseded: no longer withholds
            R.new_json(d / 'REDACTION_FINAL_2026-01-01T000000Z.json',
                       {'final_inputs': str(d / 'INPUTS.jsonl'), 'final_inputs_sha256': R.sha(d / 'INPUTS.jsonl'),
                        'withheld_keys': ['k3']})
            self.assertEqual(sorted(IV.callable_inputs('2023')), ['k1'])     # population and redaction withheld
        finally:
            B2.PERIOD_DIR.clear()
            B2.PERIOD_DIR.update(saved)

    def test_no_v2_path_imports_masker_v3(self):
        import td4v2_spec as SP
        v3 = 'td4' + 'v3'
        for p in sorted(SP.CODE_V2.glob('*.py')):
            if p.name.startswith(v3 + '_'):
                continue                                                     # masker v3: record only
            self.assertNotIn(v3, p.read_text(encoding='utf-8'), p.name)

    def test_detector_flag_manifest_and_run_status(self):
        import td4v2_flags as FL
        R = _R()
        m = FL.manifest()
        self.assertIn('excluded_classes', m)
        self.assertTrue(all(len(h) == 64 for h in m['script_sha256'].values()))
        self.assertEqual(len(m['script_sha256']), 1 + len(FL.MA1_RUN) + len(FL.MA2_RUN) + len(FL.MA2_READ) +
                         len(FL.ROUND1_MODULES))
        run = fixture_dir()
        lines = [f'RC {f} 0' for f in FL.MA1_RUN + FL.MA2_RUN]
        R.new_text(run / 'run.log', '\n'.join(lines[:-1] + [f'RC {FL.MA2_RUN[-1]} 1', 'DONE']) + '\n')
        with self.assertRaises(RuntimeError):
            FL.run_status(run)                                               # a failed detector run is refused
        run2 = fixture_dir()
        R.new_text(run2 / 'run.log', '\n'.join(lines + ['DONE']) + '\n')
        self.assertEqual(set(FL.run_status(run2).values()), {0})


class TestRunnerFixes(unittest.TestCase):
    def _ledger(self, R, root, session, rows):
        d = root / session
        w = R.JsonlWriter(d / 'ledger.jsonl', R.Redactor())
        for r in rows:
            w.write(r)
        w.close()

    def test_terminal_units_and_numeric_order_and_hash(self):
        import td4v2_runner as RV
        R = _R()
        root = fixture_dir()
        bad = {'kind': 'settle', 'req_key': 'u1', 'attempt': 1, 'http_status': 400, 'provider': 'deepseek'}
        self._ledger(R, root, 'session_1', [bad])
        self.assertEqual(RV.terminal_units(root), {})                         # one session only
        self._ledger(R, root, 'session_2', [bad])
        self.assertEqual(RV.terminal_units(root), {'u1': 'deepseek'})
        self.assertEqual(RV.terminal_cap(100), 3)
        self.assertEqual(RV.terminal_cap(10000), 20)
        for n in (10, 2):
            w = R.JsonlWriter(root / f'session_{n}' / 'responses.jsonl', R.Redactor())
            w.write({'key': 'k', 'selector': 'jev_choice', 'pass': 'main', 'http_status': 200,
                     'input_sha256': 'h', 'parsed': {'status': 'valid', 'choice': f's{n}'}})
            w.close()
        rows = RV.collect_root(root, {'k': {'input_sha256': 'h'}})
        self.assertEqual(rows[0]['choice'], 's2')                             # session_2 before session_10
        rows = RV.collect_root(root, {'k': {'input_sha256': 'other'}})
        self.assertFalse(rows[0]['http_ok'])                                  # input hash mismatch -> no output

    def test_model_halt_needs_override(self):
        import td4v2_runner as RV
        R = _R()
        saved = (S.RUNS, S.OVERRIDES)
        try:
            S.RUNS, S.OVERRIDES = fixture_dir(), fixture_dir()
            R.new_json(RV.period_root('2023') / 'session_1' / 'SESSION_SUMMARY.json',
                       {'halt': 'model_not_allowed_jev_choice:x'})
            h = RV.model_halts('2023')
            self.assertFalse(h[0]['reviewed'])
            R.new_json(S.OVERRIDES / 'MODEL_OVERRIDE_2023_session_1.json', {'status': 'REVIEWED'})
            self.assertTrue(RV.model_halts('2023')[0]['reviewed'])
        finally:
            S.RUNS, S.OVERRIDES = saved


class TestGateR1(unittest.TestCase):
    def test_admit_b_checks_before_latch(self):
        import td4v2_gate as G
        import td4v2_evaluate as EV
        saved = (S.CLEARANCE_A, S.CLEARANCE_B, S.EVAL_LATCH, S.EVAL_RESULT, S.FREEZE_A, S.EVAL_PRIMARY)
        try:
            d = fixture_dir()
            S.CLEARANCE_A, S.CLEARANCE_B = d / 'A.json', d / 'B.json'
            S.EVAL_LATCH, S.EVAL_RESULT, S.FREEZE_A = d / 'latch.json', d / 'res.json', d / 'F.json'
            S.EVAL_PRIMARY = d / 'primary.json'
            with self.assertRaises(PermissionError):
                EV.run()
            self.assertFalse(S.EVAL_LATCH.exists())                            # a refusal never burns the latch
            self.assertIn('models_files_sha256', G.B_BINDINGS)
        finally:
            S.CLEARANCE_A, S.CLEARANCE_B, S.EVAL_LATCH, S.EVAL_RESULT, S.FREEZE_A, S.EVAL_PRIMARY = saved

    def test_ciz_2025_source(self):
        import td4v2_names2025 as N25
        rows, h = N25.load_ciz()
        self.assertEqual(h, N25.CIZ_SHA256)
        self.assertTrue(all('2025-01-01' <= d <= '2025-12-31' for v in rows.values() for d, _, _ in v))


class TestSecondaryS1(unittest.TestCase):
    def test_s1_and_oracle_and_halves(self):
        import numpy as np
        import pandas as pd
        import td4_tests as V1
        import td4_select as SL
        import td4v2_evaluate as EV
        base = V1.TestSelectEvaluate()
        base.setUp()
        frames, outs = [], []
        for y in (2023, 2024, 2025):
            f = base.f.copy()
            f['origin_session'] = pd.Timestamp(f'{y}-03-06') + pd.to_timedelta((np.arange(len(f)) % 40) * 7, 'D')
            frames.append(f)
            outs.append(base.outs)
        sel = {**SL.select_core(frames[0], base.outs, base.heads), 'training_beta_order_frozen': T.NAMES}
        saved = list(S.TEST_YEARS)
        try:
            S.TEST_YEARS[:] = [2023, 2024, 2025]
            s1 = EV.secondary_s1(frames, outs, base.heads, sel)
            self.assertIn('DESCRIPTIVE', s1['label'])
            self.assertTrue(s1['a_same_configuration_2122_vs_2023']['deepseek_order']['same_config'])
            self.assertIn('not a valid forecast', s1['b_in_year_oracle']['2024']['label'])
            self.assertNotIn('confirmed', s1['S1_P1_arm_vs_mlp_2024_2025'])
            h = EV.year_2025_halves(frames, outs, base.heads, sel)
            self.assertEqual(sum(v['n'] for v in h.values()), len(frames[2]))
        finally:
            S.TEST_YEARS[:] = saved


CLASSES = (TestSecondaryS1, TestMaskV2Names, TestMaskV2Context, TestMaskV2DatesStory, TestScanV2, TestInputsM2,
           TestRunnerFixes, TestGateR1)
