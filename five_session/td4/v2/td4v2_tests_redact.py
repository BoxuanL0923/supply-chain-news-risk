"""Offline tests of the span-level redaction (td4v2_redact, td4v2_redact_final) and of the request construction from
redacted text. Every firm, token map and text here is invented. Fixtures under OUT/fixtures/<uuid>/."""
import hashlib
import json
import re
import unittest
from pathlib import Path

import td4_spec as T
import td4_env as E
from td4v2_tests import fixture_dir, _R

HEADER = ('FROZEN INPUT (TD3-input-v7)\nInstructions line.\n\nRISK MENU (firm characteristics)\n1. size: Log dollar market '
          'equity.\n\nFOCAL FIRM [FOCAL]: CHARACTERISTICS\n- size: 1.00\n\nSUPPLY-CHAIN GRAPH STATE\nExact hop 1: Shown: '
          '[SUPPLIER_1] (supplier of [FOCAL]; link age 12 days).\n\nPARTNER NEWS\nArticles: 2.\n')
META = {'focal_permno': 900001, 'shown': [{'token': '[SUPPLIER_1]', 'permno': 900002}], 'links': [],
        'articles': [{'alt_id': 'nA1', 'tokens': ['[SUPPLIER_1]']}, {'alt_id': 'nA2', 'tokens': ['[FOCAL]']}]}


def text_fixture(h1='BRIEF-[SUPPLIER_1] Says Zorvex Unit Wins Deal On Dec.27',
                 t1='[DATE] [SUPPLIER_1] : * [SUPPLIER_1] SAYS ZORVEX UNIT WINS DEAL * SEES Q4 9.5. GROWTH',
                 h2='[FOCAL] shares rose with the [FOCAL] Composite', t2='Shares of [FOCAL] rose 2% on 5/12.'):
    return (HEADER + '\nArticle 1 | firm: [SUPPLIER_1] | published 1 day before the forecast date\nHeadline: ' + h1 +
            '\nText: ' + t1 + '\n\nArticle 2 | firm: [FOCAL] | published 2 days before the forecast date\nHeadline: ' +
            h2 + '\nText: ' + t2)


def apply(doc):
    """Render a Doc's requests the way redact_round does; returns (text, ledger entries in v2 coordinates)."""
    import td4v2_redact as RD
    spans = doc.final_spans()
    out, i, ents = [], 0, []
    for s, e, kind, cs in spans:
        ph = RD.PH[kind]
        out += [doc.text[i:s], ph]
        ents.append((s, e, ph, RD.sha_s(doc.text[s:e])))
        i = e
    out.append(doc.text[i:])
    return ''.join(out), ents


class TestRedaction(unittest.TestCase):
    def setUp(self):
        E.paths()

    def test_placeholder_kinds(self):
        import td4v2_redact as RD
        self.assertEqual(RD.kind_of('MA1_dates:d_dmy_periods'), 'date')
        self.assertEqual(RD.kind_of('MA2:D12:Y4_any'), 'date')
        self.assertEqual(RD.kind_of('M2:month_day_left'), 'date')
        self.assertEqual(RD.kind_of('MA1_dates:id_reuters_story'), 'name')
        self.assertEqual(RD.kind_of('MA2:G1'), 'term')
        self.assertEqual(RD.kind_of('MA1:GB_kept'), 'term')
        self.assertEqual(RD.kind_of('M4:GB_ROUNDUP'), 'term')
        self.assertEqual(RD.kind_of('MA1:EMB'), 'name')
        self.assertEqual(RD.kind_of('M2:header'), 'name')
        self.assertEqual(set(RD.PLACEHOLDERS), {'[REDACTED]', '[DATE]', '[TERM]'})

    def test_spans_cut_to_article_text_and_around_tokens(self):
        import td4v2_redact as RD
        t = text_fixture()
        doc = RD.Doc('k', t, META)
        self.assertEqual(len(doc.arts), 2)
        i = t.index('* [SUPPLIER_1] SAYS')
        self.assertEqual(doc.add(i, i + 19, 'MA1:EMB'), 2)              # '* [SUPPLIER_1] SAYS' minus the token
        got = [t[s:e] for s, e, _, _ in doc.req]
        self.assertEqual(got, ['*', 'SAYS'])
        i = t.index('Zorvex Unit')
        self.assertEqual(doc.add(i - 13, i + 6, 'MA1:EMB'), 1)          # starts inside a token: cut at its end
        self.assertEqual(t[doc.req[-1][0]:doc.req[-1][1]], 'Says Zorvex')
        j = t.index('Log dollar')
        self.assertEqual(doc.add(j, j + 10, 'MA1:EMB'), 0)              # risk menu: never touched
        k = t.index('published 1 day')
        self.assertEqual(doc.add(k, k + 9, 'MA1:EMB'), 0)               # article label: never touched
        lab = t.index('firm: [SUPPLIER_1]') + 6
        self.assertEqual(doc.add_term(lab, 'MA2:G1'), 0)                # a token in an article label stays
        tok = t.index('[FOCAL] Composite')
        self.assertEqual(doc.add_term(tok, 'MA1:IDX'), 1)               # a give-back slot in article text
        self.assertEqual(doc.add_term(tok + 1, 'MA1:IDX'), 0)           # not a token start

    def test_invariants_hold_and_tampering_is_caught(self):
        import td4v2_redact as RD
        t = text_fixture()
        doc = RD.Doc('k', t, META)
        a = t.index('Zorvex')
        doc.add(a, a + 6, 'MA1:EMB')
        b = t.index('Dec.27')
        doc.add(b, b + 6, 'MA1_dates:d_month_period_day')
        doc.add_term(t.index('[FOCAL] Composite'), 'MA1:IDX')
        c = t.index('5/12')
        doc.add(c, c + 4, 'MA1_dates:x_month_slash_day')
        red, ents = apply(doc)
        self.assertEqual(RD.check_record(t, red, ents), [])
        self.assertEqual(RD.render(t, [x[:3] for x in ents]), red)
        self.assertIn('[TERM] Composite', red)
        self.assertIn('Says [REDACTED] Unit', red)
        self.assertEqual(red[:red.index('\nPARTNER NEWS\n')], t[:t.index('\nPARTNER NEWS\n')])
        self.assertEqual(RD.ROLE.findall(red).count('[FOCAL]'), RD.ROLE.findall(t).count('[FOCAL]') - 1)
        bad = red.replace('Shares of', 'Shares in')                       # a change outside the ledger
        self.assertIn('outside_span_differs', RD.check_record(t, bad, ents) + RD.check_record(t, bad, ents))
        bad2 = red.replace('[DATE] [SUPPLIER_1] :', '[DATE] [SUPPLIER_2] :')
        self.assertTrue(RD.check_record(t, bad2, ents))                   # a pseudonym token changed
        bad3 = red.replace('size: Log dollar', 'size: Log [REDACTED]')
        self.assertTrue(RD.check_record(t, bad3, ents))                   # header changed
        wrong = [(s, e, '[REDACTED]' if ph == '[TERM]' else ph, h) for s, e, ph, h in ents]
        self.assertTrue(RD.check_record(t, RD.render(t, [x[:3] for x in wrong]), wrong))  # token under [REDACTED]

    def test_idempotent(self):
        import td4v2_redact as RD
        t = text_fixture()
        rx = re.compile(r'(?<![\d.])(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?:\d{2}|\d{4})?(?![\d])')

        class H:
            rx14, rx15, rx16 = {}, {'d_dmy_periods': rx}, {}
            year_units = re.compile(r'x^')
        doc = RD.Doc('k', t, META)
        RD.loc_dates(doc, ['MA1_dates:d_dmy_periods'], H)
        i = t.index('Zorvex')
        ctx = t[max(0, i - 70):i + 6 + 70].replace('\n', ' ')
        hit = {'class': 'COMPOUND', 'token': '[SUPPLIER_1]', 'match': 'Zorvex', 'ctx': ctx}
        RD.loc_m4(doc, [hit])
        red, ents = apply(doc)
        self.assertEqual(len(ents), 2)
        doc2 = RD.Doc('k', red, META)                                    # the same detector outputs, applied again
        RD.loc_dates(doc2, ['MA1_dates:d_dmy_periods'], H)
        RD.loc_m4(doc2, [hit])
        self.assertEqual(doc2.final_spans(), [])
        self.assertEqual(apply(doc2)[0], red)

    def test_exact_context_location(self):
        import td4v2_redact as RD
        s = 'aa Zorvex bb Zorvex cc'
        self.assertEqual(RD.find_exact(s, 'Zorvex', s[max(0, 13 - 3):19 + 3], 3, 3), [(13, 19)])
        self.assertEqual(RD.find_prefix(s, 'Zorv', s[10:22], 3, 3), [(13, 19)])

    def test_round_mapping_to_v2_offsets(self):
        import td4v2_redact as RD
        prior = [(10, 16, '[REDACTED]'), (30, 34, '[DATE]')]               # v2 spans of earlier rounds
        self.assertEqual(RD.to_v2(prior, 5, 8), (5, 8))
        self.assertEqual(RD.to_v2(prior, 20 + 4, 22 + 4), (20, 22))       # after '[REDACTED]' (+4 characters)
        self.assertEqual(RD.to_v2(prior, 40 + 6, 41 + 6), (40, 41))       # after both (+4, +2)
        with self.assertRaises(ValueError):
            RD.to_v2(prior, 12, 14)                                       # inside an earlier placeholder

    def test_ma2_alignment_positions(self):
        import td4v2_redact as RD
        raw = 'Shares of ALL rose as ALL employees left'
        masked = 'Shares of [FOCAL] rose as [FOCAL] employees left'
        al = RD._align2(raw, masked, 100)
        self.assertEqual([(w, t, p) for w, t, _, _, p in al], [('ALL', '[FOCAL]', 110), ('ALL', '[FOCAL]', 126)])
        self.assertEqual(al[1][2:4], ('ALL rose as', 'employees left'))     # ma2's three raw words either side

    def test_gb_frozen_group_rescan(self):
        import td4v2_redact as RD
        t = text_fixture(t2='The cloud unit of parent [FOCAL] said; [FOCAL] fell.')
        doc = RD.Doc('k', t, META)
        n = RD.loc_gb(doc, {(900001, 'zorvex', 'L', 'parent')})
        self.assertEqual(n, 1)
        self.assertEqual(t[doc.req[0][0]:doc.req[0][1]], '[FOCAL]')
        self.assertEqual(t[doc.req[0][0] - 7:doc.req[0][0]], 'parent ')

    def test_placeholder_dependent_flags(self):
        import td4v2_redact_final as RF
        R = _R()
        run = fixture_dir()
        rows = [{'key': 'k1', 'class': 'GB_LOGONE', 'token': '[REDACTED]', 'match': 'Logone [REDACTED]', 'ctx': 'x'},
                {'key': 'k2', 'class': 'GB_LOGONE', 'token': '[REDACTED]', 'match': 'Logone [REDACTED]', 'ctx': 'x'},
                {'key': 'k2', 'class': 'GB_LOGONE', 'token': '[FOCAL]', 'match': 'Logone [FOCAL]', 'ctx': 'x'},
                {'key': 'k3', 'class': 'T_ANY', 'token': '[FOCAL]', 'match': 'TERM', 'ctx': 'a [TERM] b'},
                {'key': 'k4', 'class': 'NONAME', 'token': '[FOCAL]', 'match': '', 'ctx': ''}]
        R.new_text(run / 'ma1' / 's13_m4_hits.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
        self.assertTrue(RF.placeholder_dependent(run, None, 'k1', 'M4:GB_LOGONE'))     # slot holds a placeholder
        self.assertFalse(RF.placeholder_dependent(run, None, 'k2', 'M4:GB_LOGONE'))    # one real token slot
        self.assertTrue(RF.placeholder_dependent(run, None, 'k3', 'M4:T_ANY'))         # the word inside '[TERM]'
        self.assertFalse(RF.placeholder_dependent(run, None, 'k4', 'M4:NONAME'))       # firm-level: never
        self.assertFalse(RF.placeholder_dependent(run, None, 'k1', 'MA1:EMB'))         # other families: never

    def test_reviewer_constants_without_running_the_script(self):
        import td4v2_redact as RD
        p = fixture_dir() / 'script.py'
        _R().new_text(p, "import re\nA = 'x'\nRX = {'a': re.compile(A + 'y')}\nraise SystemExit('ran')\n")
        ns = RD.consts(p, ['A', 'RX'])
        self.assertTrue(ns['RX']['a'].search('xy'))


class TestMaskerNoSource(unittest.TestCase):
    """Rule v2.1 MASKER_NO_SOURCE on invented name tables."""
    NAMES = {900001: [('2019-03-01', 'ZORVEX DYNAMICS INC', 'ZRVX')],           # own CRSP row
             900002: [('2026-02-01', 'QUILLMARK HOLDINGS', 'QLMK')],            # only a row after the last day
             900003: [],                                                         # no own row
             900004: [('2018-01-01', 'BRINWOOD PARENT CORP', '')]}             # sibling of 900003
    SIB = {900003: [900004]}

    def test_positive_unsourced_firm_flags_its_inputs(self):
        import td4v2_nosource as NS
        self.assertFalse(NS.sourced(900002, self.NAMES, {}, {}, {}, {}, '2025-12-31'))
        unsourced = {900002}
        metas = [{'key': 'k1', 'focal_permno': 900001, 'shown': [{'permno': 900002}]},
                 {'key': 'k2', 'focal_permno': 900002, 'shown': []},
                 {'key': 'k3', 'focal_permno': 900001, 'shown': [{'permno': 900003}]}]
        self.assertEqual(NS.flag_inputs(metas, unsourced), {'k1': [900002], 'k2': [900002]})

    def test_negative_sourced_by_crsp_sibling_or_factset(self):
        import td4v2_nosource as NS
        self.assertTrue(NS.sourced(900001, self.NAMES, {}, {}, {}, {}, '2023-12-31'))      # own CRSP row
        self.assertTrue(NS.sourced(900003, self.NAMES, self.SIB, {}, {}, {}, '2023-12-31'))  # issuer sibling's row
        self.assertTrue(NS.sourced(900003, self.NAMES, {}, {900003: {'Brinwood Ventures'}}, {}, {}, '2023-12-31'))
        self.assertTrue(NS.sourced(900003, self.NAMES, {}, {}, {900003: {'BRWV'}}, {}, '2023-12-31'))
        self.assertFalse(NS.sourced(900003, self.NAMES, {}, {}, {}, {}, '2023-12-31'))
        self.assertFalse(NS.sourced(900001, self.NAMES, {}, {}, {}, {}, '2018-12-31'))     # row after the last day

    def test_2025_only_ciz_firm_counts_as_sourced(self):
        import td4v2_nosource as NS
        ciz = {900003: [('2025-04-15', 'GLIMRA HOLDINGS INC', 'GLMR')]}
        self.assertTrue(NS.sourced(900003, self.NAMES, {}, {}, {}, ciz, '2025-12-31'))      # 2025 build: CIZ counts
        self.assertFalse(NS.sourced(900003, self.NAMES, {}, {}, {}, ciz, '2024-12-31'))     # not before its date
        self.assertEqual(NS.flag_inputs([{'key': 'k', 'focal_permno': 900003, 'shown': []}], set()), {})

    def test_v2_1_redaction_result_is_preferred(self):
        import td4v2_redact_final as RF
        R = _R()
        d = fixture_dir()
        R.new_json(d / 'REDACTION_FINAL_2026-01-01T000000Z.json', {'withheld_keys': ['a']})
        self.assertEqual(RF.final_path(d).name, 'REDACTION_FINAL_2026-01-01T000000Z.json')
        R.new_json(d / 'REDACTION_FINAL_v2_1_2026-01-01T000001Z.json', {'withheld_keys': []})
        R.new_json(d / 'REDACTION_FINAL_2026-01-02T000000Z.json', {'withheld_keys': ['a']})
        self.assertEqual(RF.final_path(d).name, 'REDACTION_FINAL_v2_1_2026-01-01T000001Z.json')
        self.assertEqual(RF.v2_path(d).name, 'REDACTION_FINAL_2026-01-02T000000Z.json')


class TestRedactedInputsAndConstruction(unittest.TestCase):
    def test_inputs_require_redaction_result_and_hash(self):
        import td4v2_build2 as B2
        import td4v2_inputs as IV
        R = _R()
        d = fixture_dir()
        rd = fixture_dir()
        w = R.JsonlWriter(d / 'INPUTS.jsonl', R.Redactor())
        w2 = R.JsonlWriter(rd / 'INPUTS.jsonl', R.Redactor())
        for i, k in enumerate(('k1', 'k2', 'k3'), 1):
            txt = 'FROZEN INPUT (x)\n' + k + ' Zorvex'
            w.write({'key': k, 'sample_rank': i, 'input_text': txt, 'input_sha256': R.sha_text(txt),
                     'input_bytes': len(txt)})
            txt2 = txt.replace('Zorvex', '[REDACTED]')
            w2.write({'key': k, 'sample_rank': i, 'input_text': txt2, 'input_sha256': R.sha_text(txt2),
                      'input_bytes': len(txt2)})
        w.close()
        w2.close()
        R.new_json(d / 'BUILD.json', {'counts': {}})
        R.new_json(d / 'MASK_V2_RECEIPT.json', {'v1_masker_inputs_used': False})
        R.new_json(d / 'WITHHELD_POPULATION_M2.json', {'withheld_keys': ['k2']})
        saved = dict(B2.PERIOD_DIR)
        try:
            B2.PERIOD_DIR[2023] = d
            with self.assertRaises(PermissionError):
                IV.callable_inputs('2023')                                   # no redaction result
            R.new_json(d / 'REDACTION_FINAL_2026-01-01T000000Z.json',
                       {'final_inputs': str(rd / 'INPUTS.jsonl'), 'final_inputs_sha256': '0' * 64,
                        'withheld_keys': ['k3']})
            with self.assertRaises(PermissionError):
                IV.callable_inputs('2023')                                   # redacted file hash differs
            R.new_json(d / 'REDACTION_FINAL_2026-01-02T000000Z.json',
                       {'final_inputs': str(rd / 'INPUTS.jsonl'), 'final_inputs_sha256': R.sha(rd / 'INPUTS.jsonl'),
                        'withheld_keys': ['k3']})
            recs = IV.callable_inputs('2023')
            self.assertEqual(sorted(recs), ['k1'])                           # population and redaction withheld
            self.assertIn('[REDACTED]', recs['k1']['input_text'])            # the redacted text is what is sent
        finally:
            B2.PERIOD_DIR.clear()
            B2.PERIOD_DIR.update(saved)

    def test_construction_parity_600_on_fixed_text(self):
        """200 fixed texts x 3 selectors: the unchanged TD3 selector classes (as td4_runner calls them) build exactly
        the documented request body, with the input text verbatim."""
        E.paths()
        import td3_providers as P
        import td3_spec as S3
        menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
        sels = P.selectors(menu)
        self.assertEqual(tuple(sels), S3.SELECTORS)
        ok = 0
        for i in range(200):
            txt = text_fixture(t2=f'Shares of [FOCAL] rose {i}% on [DATE]; [REDACTED] and [TERM] stay as written.')
            for name, sel in sels.items():
                body = sel.body(txt)
                if name == 'deepseek_order':
                    want = {'model': S3.DEEPSEEK_MODEL,
                            'messages': [{'role': 'system', 'content': S3.SYSTEM_DEEPSEEK},
                                         {'role': 'user', 'content': S3.deepseek_question(menu) + '\n\n' + txt}],
                            'thinking': dict(S3.DEEPSEEK_THINKING), 'temperature': S3.DEEPSEEK_TEMPERATURE,
                            'max_tokens': S3.DEEPSEEK_MAX_TOKENS, 'response_format': {'type': 'json_object'},
                            'stream': False}
                    got_text = json.loads(body)['messages'][1]['content'][-len(txt):]
                else:
                    qs = S3.jev_choice_question(menu) if name == 'jev_choice' else S3.jev_noul_questions(menu)
                    want = {'state': txt, 'model': S3.JEV_MODEL, 'questions': qs}
                    got_text = json.loads(body)['state']
                exp = json.dumps(want, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                ok += body == exp and got_text == txt
        self.assertEqual(ok, 600)


def tests():
    return [TestRedaction, TestRedactedInputsAndConstruction, TestMaskerNoSource]
