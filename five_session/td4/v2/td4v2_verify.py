"""Population checks and audit samples for the v2 builds (selection subsample, 2023, and their repairs).

population() repeats the v1 td4_verify.population_checks text checks (same regexes and rule) for any build folder and
writes the withheld key list as WITHHELD_POPULATION_v2.json (v1 writes the list only for its own period folders).
audit() calls the v1 td4_inputs.audit_sample unchanged (re-mask from raw text, strict residual, TD3 scanner).
"""
import json
import re
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_build as BV

TARGETS = {'select': (lambda: BV.SELECT_DIR, 2022, 'select_2122'),
           'select-repair': (lambda: BV.SELECT_REPAIR_DIR, 2022, 'select_2122_repair'),
           '2023': (lambda: BV.YEAR_DIR[2023], 2023, '2023'),
           '2023-repair': (lambda: BV.REPAIR_DIR[2023], 2023, 'repair_2023')}


def population(d):
    E.paths()
    import td3_runtime as R
    import td3_build as B
    import td3_mask
    import td3_spec as S3
    import td4_verify as V1
    d = Path(d)
    role = B.ROLE_TOKEN.pattern
    glued = re.compile(r'[^\W_]' + role + '|' + role + r'[^\W_]')
    remnant = re.compile(role + r'[\-–]\d')
    meta = {m['key']: (m['firm_id'], len(m['articles'])) for m in R.read_jsonl(d / 'INPUT_META_COMPACT.jsonl')}
    problems, bad, n, articles = Counter(), set(), 0, 0
    with (d / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            text = r['input_text']
            if not text.startswith(f'FROZEN INPUT ({S3.INPUT_VERSION})'):
                problems['header'] += 1
                bad.add(r['key'])
            fid, n_art = meta.get(r['key'], (None, None))
            if fid is None or fid in text:
                problems['no_compact_meta' if fid is None else 'firm_id_in_input'] += 1
                bad.add(r['key'])
            body = '\n'.join(x for x in text.split('\n') if x.startswith('Headline: ') or x.startswith('Text: '))
            k = text.count('\nArticle ')
            articles += k
            if n_art is not None and k != n_art:
                problems['article_count_differs_from_meta'] += 1
                bad.add(r['key'])
            for name, rx, src in (('year_in_article_text', V1.YEAR, body), ('year_glued_to_time', V1.YEAR_TIME, body),
                                  ('url_in_article_text', td3_mask.URL_RESIDUAL, body),
                                  ('numeric_date_in_input', V1.NUM_DATE, text),
                                  ('story_fields_in_input', V1.STORY, text),
                                  ('glued_role_token', glued, body), ('token_hyphen_digit', remnant, body)):
                if rx.search(src):
                    problems[name] += 1
                    bad.add(r['key'])
    out = {'dir': str(d), 'inputs': n, 'articles': articles, 'problems': dict(problems), 'withheld_inputs': len(bad),
           'rule': 'an input with any problem is withheld from all three providers alike and takes the fallback'}
    R.new_json(d / 'POPULATION_CHECK_v2b.json', out)
    R.new_json(d / 'WITHHELD_POPULATION_v2.json', {'dir': str(d), 'withheld_keys': sorted(bad),
                                                   'problems': dict(problems)})
    return out


def run(which):
    d, _, _ = TARGETS[which]
    return population(d())


def audit(which, max_n=300):
    import td4_inputs as I
    d, year, tag = TARGETS[which]
    return I.audit_sample(year, period_dir=d(), store_tag=tag, max_n=max_n)
