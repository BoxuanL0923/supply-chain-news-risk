"""Offline tests of the v2.2 redaction fix round (td4v2_redact22, td4v2_render, td4v2_check22): F1 word boundaries,
each F2 locator class, F3 (no GB waiver), ledger-only rounds rendering exactly. Every firm, name, code and text here
is invented. Fixtures under OUT/fixtures/<uuid>/."""
import hashlib
import json
import re
import unittest

import td4_env as E
from td4v2_tests import fixture_dir, _R
from td4v2_tests_redact import HEADER

META = {'focal_permno': 900001, 'shown': [{'token': '[SUPPLIER_1]', 'permno': 900002},
                                          {'token': '[CUSTOMER_1]', 'permno': 900003}], 'links': []}
NAMES = {900001: (['ZORVEX DYNAMICS INC'], ['ZRVX']),
         900002: (['QUILLMARK HOLDINGS', 'BRINWOOD-KESTREL CORP', 'TRIVO BANCORP'], ['QLMK', 'ARC']),
         900003: (['NEXUS GENERATION CORP', 'ARC LUMEN SYSTEMS', 'WREN BLUEJAY INC', 'H2 VELORA INC'], ['NXGN'])}


def doc_text(*arts):
    """HEADER plus articles (firm token, headline, text)."""
    out = [HEADER.replace('Articles: 2.', f'Articles: {len(arts)}.')]
    for i, (tok, h, t) in enumerate(arts, 1):
        out.append(f'\nArticle {i} | firm: {tok} | published 1 day before the forecast date\nHeadline: {h}\nText: {t}\n')
    return ''.join(out).rstrip('\n')


def terms():
    import td4v2_redact22 as R22
    eng = R22.english()
    profs = []
    for p, (names, ticks) in NAMES.items():
        pr = R22.firm_profile(set(names), set(ticks), eng)
        pr['permno'] = p
        profs.append(pr)
    return R22.Terms(profs, eng), eng


def run(text, gb=None, H=None, classes=()):
    """process() on an invented masker-v2 text; returns (rendered text, entries)."""
    import td4v2_redact22 as R22
    import td4v2_render as RR
    tm, eng = terms()
    new, _, _ = R22.process('k', text, META, [], H, None, set(classes), tm, gb or {}, eng)
    ents = [(s, e, ph) for s, e, ph, _, _ in new]
    return RR.render(text, ents), new


class TestWordsF1(unittest.TestCase):
    def setUp(self):
        E.paths()

    def test_word_definition(self):
        import td4v2_redact22 as R22
        t = 'x ZorvexPro-2 and Zürichwerk, l’Orvex A&B 1/2a BRIEF-Zorvex UPDATE 2-Zorvex x'
        def w(sub, k=0):
            i = t.index(sub) + k
            a, b = R22.word_span(t, i, i + 1)
            return t[a:b]
        self.assertEqual(w('Pro'), 'ZorvexPro-2')                    # internal hyphen and digit join the word
        self.assertEqual(w('werk'), 'Zürichwerk')                    # non-ASCII letters are letters
        self.assertEqual(w('Orvex'), 'l’Orvex')                 # apostrophe between letters joins
        self.assertEqual(w('&B'), 'A&B')
        self.assertEqual(w('/2a', 1), '1/2a')
        self.assertEqual(w('Zorvex UPDATE'), 'Zorvex')               # a news slug is not part of the word
        self.assertEqual(w('Zorvex x'), 'Zorvex')                    # 'UPDATE 2-' slug
        self.assertEqual(w('and'), 'and')

    def test_month_cut_and_code_cut_become_whole_words(self):
        import td4v2_redact as RD
        import td4v2_redact22 as R22
        t = doc_text(('[FOCAL]', 'Global markets rally', 'The decision on Phase 1/2a data came on [DATE].'))
        doc = RD.Doc('k', t, META)
        i = t.index('markets')
        doc.add(i, i + 3, 'MA1_dates:d_month')                       # 'mar' cut out of 'markets'
        j = t.index('decision') + 2
        doc.add(j, j + 3, 'MA1_dates:d_month')                       # 'dec' inside 'decision'
        k = t.index('1/2a')
        doc.add(k, k + 3, 'MA1_dates:x_month_slash_day')             # '1/2' inside a trial code
        R22.widen(doc)
        got = sorted(t[s:e] for s, e, _, _ in doc.final_spans())
        self.assertEqual(got, ['1/2a', 'decision', 'markets'])

    def test_glued_words_and_p6(self):
        import td4v2_check22 as C
        import td4v2_redact22 as R22
        t = doc_text(('[FOCAL]', 'Zorvexian unit grows', 'Sales rose on [DATE]th and at Zorvex-led sites.'))
        red, new = run(t)
        self.assertNotIn('Zorvex', red)
        self.assertIn('[REDACTED] unit grows', red)                  # the whole word, not 'ZORVEX' + 'ian'
        self.assertIn('on [DATE][REDACTED] and', red)              # a word part glued to a masker placeholder
        self.assertEqual(R22.p6_entries(t, [(s, e, ph) for s, e, ph, _, _ in new]), 0)
        for c in ('N1', 'N1b'):
            self.assertIsNone(C.RX[c].search(red))


class TestLocatorsF2(unittest.TestCase):
    def setUp(self):
        E.paths()

    def test_a_own_names_any_case_and_initialism_guard(self):
        t = doc_text(('[FOCAL]', 'zorvex and ZORVEX and Zorvex gain',
                      'The arc lumen systems deal: ALS shares, Als unit; an Arc Lumen lamp.'),
                     ('[SUPPLIER_1]', 'Analysts see ARC beating rivals in the arc of growth', 'MORE ARC NEWS TODAY'),
                     ('[CUSTOMER_1]', 'Bluejay pricing', 'Founded in [YEAR], wren has grown.'))
        red, _ = run(t)
        self.assertNotRegex(red, r'(?i)zorvex')
        self.assertNotIn('arc lumen systems', red)                   # multi-word core name, any case
        self.assertIn('an Arc Lumen lamp', red)                      # two English words alone are not the name
        self.assertNotIn('ALS', red)
        self.assertNotIn('Als unit', red)                            # Title case of a non-English initialism
        self.assertIn('see [REDACTED] beating', red)                 # English-word ticker: upper case, mixed text
        self.assertIn('the arc of growth', red)                      # ... never lower case
        self.assertIn('MORE ARC NEWS TODAY', red)                    # ... nor in an all-caps line
        self.assertIn('[YEAR], [REDACTED] has grown', red)           # English first name word, own article

    def test_b_hyphen_part_and_first_letter_cut(self):
        t = doc_text(('[SUPPLIER_1]', 'Kestrel-backed fund buys stake', 'RINWOOD KESTREL REPORTS SALES'))
        red, _ = run(t)
        self.assertNotIn('Kestrel', red)
        self.assertNotIn('RINWOOD', red)

    def test_c_compounds_codes_fragments_long_forms(self):
        t = doc_text(('[SUPPLIER_1]', 'ZORVEXPRO launch; Quillfive resin approved',
                      'Trial ZRVX-101 and the Neoquillmaroil venture; Tri Volta Bank hires; a bookmark; '
                      'next-generation chips; the NEXT-GENERATION line'))
        red, new = run(t)
        for s in ('ZORVEXPRO', 'Quillfive', 'ZRVX-101', 'Neoquillmaroil', 'Tri Volta'):
            self.assertNotIn(s, red)
        self.assertIn('a bookmark', red)                             # English compound in lower case: kept
        self.assertIn('next-generation chips', red)                  # all hyphen parts English: kept
        self.assertIn('NEXT-GENERATION line', red)
        cls = {c for x in new for c in x[3]}
        self.assertTrue({'V22:c_compound', 'V22:c_long_form'} <= cls)

    def test_d_codes_after_role_tokens(self):
        t = doc_text(('[FOCAL]', '[FOCAL] KX9 QZR 4070 TI cards; [FOCAL] MAX outlasts rivals',
                      '* [FOCAL] 2680 WAS WELL TOLERATED * [FOCAL] Q2 RESULTS * [FOCAL] 500 JOBS CUT * '
                      '[FOCAL] G6, G7, ONE, AND ONE+ MONITORS * [FOCAL] PRODUCER I (ZPI) CONTRACT; ZPI RENEWED * '
                      '[FOCAL] DRUG WINS APPROVAL'),
                     ('[SUPPLIER_1]', 'Pipeline notice', '[SUPPLIER_1] BRINDLE RIVER GAS COMPANY, L.L.C. (BRGC): '
                                                         '* BRGC DECLARES FORCE MAJEURE'))
        red, _ = run(t)
        for s in ('KX9', 'QZR', '4070', ' TI ', 'MAX', '2680', 'G6', 'G7', 'ZPI', 'BRINDLE', 'BRGC'):
            self.assertNotIn(s, red)
        for s in ('[FOCAL] Q2 RESULTS', '[FOCAL] 500 JOBS', 'ONE, AND ONE+', '[FOCAL] DRUG WINS'):
            self.assertIn(s, red)

    def test_e_giveback_phrases_context_words_and_period_code_names(self):
        import td4v2_redact as RD
        import td4v2_redact22 as R22
        gb = {'GB_TEST': re.compile(r'Quorra \[[A-Z_0-9]+\]')}
        t = doc_text(('[FOCAL]', 'Clash in Quorra [FOCAL] province', 'Rebels left Quorra [REDACTED] on [DATE].'),
                     ('[CUSTOMER_1]', 'SEES [CUSTOMER_1] TOTAL REVENUE', 'Sees [CUSTOMER_1] revenue; [CUSTOMER_1] '
                                                                         'shares up; [FOCAL] 500 JOBS'))
        red, new = run(t, gb=gb)
        self.assertNotIn('Quorra', red)                              # the phrase's trigger word goes too (F3)
        self.assertIn('[REDACTED] [TERM] province', red)
        self.assertIn('SEES [TERM] TOTAL', red)                      # a firm named like a period code
        self.assertIn('Sees [TERM] revenue', red)
        self.assertIn('[CUSTOMER_1] shares up', red)
        self.assertIn('[FOCAL] 500 JOBS', red)                       # another firm's token is not a period slot
        # frozen GB group: the neighbour word is redacted with the slot
        tm, eng = terms()
        t2 = doc_text(('[SUPPLIER_1]', 'Supplier explores sale of Family [FOCAL] - paper', 'Nothing else.'))
        doc = RD.Doc('k', t2, META)
        R22.gb_context(doc, {(900001, 'zorvex', 'L', 'family')})
        self.assertEqual([t2[s:e] for s, e, _, _ in doc.final_spans()], ['Family'])
        # a short ordinary hidden word inside a phrase: both neighbours
        t3 = doc_text(('[CUSTOMER_1]', 'Orders', '[CUSTOMER_1] - ENTERPRISE AI APPLICATIONS ORDERED'))
        doc = RD.Doc('k', t3, META)
        i = t3.index('AI APP')
        doc.add(i, i + 2, 'MA1:EMB')
        R22.slot_context(doc, doc.final_spans(), eng)
        self.assertEqual(sorted(t3[s:e] for s, e, _, _ in doc.final_spans()), ['AI', 'APPLICATIONS', 'ENTERPRISE'])

    def test_f_dates(self):
        import td4v2_check22 as C
        t = doc_text(('[FOCAL]', 'The [YEAR]-22 season and 21/[YEAR] budget',
                      'Units: March [DATE] 25 PJM; outages from [DATE] to April; [YEAR]/23 plan'))
        red, _ = run(t)
        for c in ('D1', 'D2'):
            self.assertIsNone(C.RX[c].search(red))
        self.assertNotIn('March', red)
        self.assertNotIn('April', red)
        self.assertNotIn(' 25 PJM', red)
        self.assertIn('[YEAR]-[DATE] season', red)


class TestNoWaiverAndRoundsF3(unittest.TestCase):
    def setUp(self):
        E.paths()

    def test_no_gb_waiver_only_literal_placeholder_words(self):
        import td4v2_redact22 as R22
        R = _R()
        run_dir = fixture_dir()
        rows = [{'key': 'k1', 'class': 'GB_LOGONE', 'token': '[REDACTED]', 'match': 'Quorra [REDACTED]', 'ctx': 'x'},
                {'key': 'k3', 'class': 'T_ANY', 'token': '[FOCAL]', 'match': 'TERM', 'ctx': 'a [TERM] b'},
                {'key': 'k4', 'class': 'NONAME', 'token': '[FOCAL]', 'match': '', 'ctx': ''},
                {'key': 'k5', 'class': 'T_ANY', 'token': '[FOCAL]', 'match': 'Zorvex', 'ctx': 'a Zorvex b'}]
        R.new_text(run_dir / 'ma1' / 's13_m4_hits.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
        self.assertFalse(R22.literal_placeholder_hit(run_dir, 'k1', 'M4:GB_LOGONE'))   # GB_*: never waived (F3)
        self.assertTrue(R22.literal_placeholder_hit(run_dir, 'k3', 'M4:T_ANY'))        # the word inside '[TERM]'
        self.assertFalse(R22.literal_placeholder_hit(run_dir, 'k4', 'M4:NONAME'))
        self.assertFalse(R22.literal_placeholder_hit(run_dir, 'k5', 'M4:T_ANY'))       # a real word

    def test_fixpoint_idempotent_and_invariants(self):
        import td4v2_check22 as C
        import td4v2_redact as RD
        import td4v2_redact22 as R22
        import td4v2_render as RR
        t = doc_text(('[FOCAL]', 'Zorvexian ZRVX-7 launch in the [YEAR]-23 season', 'March [DATE] 4 note; zorvex up'),
                     ('[SUPPLIER_1]', 'Tri Volta Bank and Kestrel-led fund', '[SUPPLIER_1] KX9 ships; Neoquillmaroil'))
        tm, eng = terms()
        new, _, passes = R22.process('k', t, META, [], None, None, set(), tm, {}, eng)
        ents = [(s, e, ph, h) for s, e, ph, _, h in new]
        red = RR.render(t, ents)
        self.assertEqual(RD.check_record(t, red, ents), [])
        self.assertEqual(R22.p6_entries(t, ents), 0)
        for c in ('D1', 'D2', 'N1', 'N1b'):
            self.assertIsNone(C.RX[c].search(red))
        again, _, _ = R22.process('k', t, META, [x[:3] for x in ents], None, None, set(), tm, {}, eng)
        self.assertEqual(again, [])                                  # nothing more on the round's output
        self.assertLess(passes, R22.ITER)
        doc = RD.Doc('k', red, META)
        self.assertEqual(R22.v22_scan(doc, tm, {}, eng), set())      # the residual scan finds nothing

    def test_ledger_only_rounds_render_exactly(self):
        import td4v2_check22 as C
        import td4v2_render as RR
        R = _R()
        d = fixture_dir()
        recs = []
        for i, k in enumerate(('k1', 'k2', 'k3')):
            txt = doc_text(('[FOCAL]', f'Zorvex unit {i}', f'Sales of Zorvex rose {i}% on [DATE].'))
            recs.append({'key': k, 'sample_rank': i, 'input_text': txt,
                         'input_sha256': hashlib.sha256(txt.encode()).hexdigest(), 'input_bytes': len(txt.encode())})
        R.new_text(d / 'INPUTS.jsonl', ''.join(json.dumps(r) + '\n' for r in recs))
        t0 = recs[0]['input_text']
        a = t0.index('Zorvex unit')
        b = t0.index('Zorvex rose')
        l1 = [{'key': 'k1', 'offset': a, 'length': 6, 'placeholder': '[REDACTED]'}]
        l2 = [{'key': 'k1', 'offset': b, 'length': 6, 'placeholder': '[REDACTED]'},
              {'key': 'k3', 'offset': a, 'length': 6, 'placeholder': '[REDACTED]'}]
        R.new_text(d / 'L1.jsonl', ''.join(json.dumps(x) + '\n' for x in l1))
        R.new_text(d / 'L2.jsonl', ''.join(json.dumps(x) + '\n' for x in l2))
        with RR.open_round(d / 'INPUTS.jsonl', [d / 'L1.jsonl', d / 'L2.jsonl'], 'rb', 2) as f:
            lines = [json.loads(x) for x in f]
        self.assertEqual([r['key'] for r in lines], ['k1', 'k2', 'k3'])
        self.assertEqual(lines[0]['input_text'], t0.replace('Zorvex', '[REDACTED]'))
        self.assertEqual(lines[1]['input_text'], recs[1]['input_text'])
        for r, v in zip(lines, recs):
            self.assertEqual(r['input_sha256'], hashlib.sha256(r['input_text'].encode()).hexdigest())
            self.assertEqual(r['v2_input_sha256'], v['input_sha256'])
            self.assertEqual(r['redaction_round'], 2)
        f = RR.open_round(d / 'INPUTS.jsonl', [d / 'L1.jsonl'], 'r', 1)
        self.assertEqual(json.loads(f.readline())['input_text'].count('[REDACTED]'), 1)
        f.close()
        text, ents, pos = C.render_pos(t0, [(a, a + 6, '[REDACTED]'), (b, b + 6, '[REDACTED]')])
        self.assertEqual(text, RR.render(t0, ents))
        self.assertTrue(all(text[p:p + 10] == '[REDACTED]' for p in pos))
        with self.assertRaises(ValueError):
            RR.render(t0, [(a, a + 6, '[REDACTED]'), (a + 2, a + 8, '[DATE]')])

    def test_v2_2_redaction_result_is_preferred(self):
        import td4v2_redact_final as RF
        R = _R()
        d = fixture_dir()
        R.new_json(d / 'REDACTION_FINAL_v2_1_2026-01-01T000001Z.json', {'withheld_keys': []})
        self.assertEqual(RF.final_path(d).name, 'REDACTION_FINAL_v2_1_2026-01-01T000001Z.json')
        R.new_json(d / 'REDACTION_FINAL_v2_2_2026-01-01T000000Z.json', {'withheld_keys': ['a']})
        R.new_json(d / 'REDACTION_FINAL_v2_1_2026-01-02T000000Z.json', {'withheld_keys': []})
        self.assertEqual(RF.final_path(d).name, 'REDACTION_FINAL_v2_2_2026-01-01T000000Z.json')


def tests():
    return [TestWordsF1, TestLocatorsF2, TestNoWaiverAndRoundsF3]
