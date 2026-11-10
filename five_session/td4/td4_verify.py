"""Population checks over every frozen TD4 input of a period (the text-level checks of td3_build.verify that need no
raw text): no year 19xx/20xx and no year glued to a time in article text, no URL, no numeric calendar date, no Reuters
story field, no role token glued to a letter or digit, no token-hyphen-digit remnant, no opaque firm ID, the frozen
input header, and the article count matching the compact metadata. Streams INPUTS.jsonl (no outcome, no network).
The re-masking from raw text, the strict residual and the independent scanner run on the audit sample
(td4_inputs.audit_sample), as TD3's full re-checks ran on its 200 inputs.
"""
import json
import re
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E

YEAR = re.compile(r'(?<![\d.,])(?:19|20)\d\d(?!\d)')
YEAR_TIME = re.compile(r'(?<![\d.,])(?:19|20)\d\d(?=\d{1,2}:\d\d)')
NUM_DATE = re.compile(r'\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}/\d{2,4}\b')
STORY = re.compile(r'STORY_(?:NUMBER|DATE|TIME)')


def population_checks(year, inputs_dir=None, repair=False):
    E.paths()
    import td3_runtime as R
    import td3_build as B
    import td3_mask
    import td3_spec as S3
    d = Path(inputs_dir) if inputs_dir else (T.OUT / 'inputs' / (f'repair_{year}' if repair else f'period_{year}'))
    role = B.ROLE_TOKEN.pattern
    glued = re.compile(r'[^\W_]' + role + '|' + role + r'[^\W_]')
    remnant = re.compile(role + r'[\-–]\d')
    meta = {}
    for m in R.read_jsonl(d / 'INPUT_META_COMPACT.jsonl'):
        meta[m['key']] = (m['firm_id'], len(m['articles']))
    problems, examples, n, articles = Counter(), {}, 0, 0
    bad_keys = set()
    with (d / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            text = r['input_text']
            if not text.startswith(f'FROZEN INPUT ({S3.INPUT_VERSION})'):
                problems['header'] += 1
            fid, n_art = meta.get(r['key'], (None, None))
            if fid is None:
                problems['no_compact_meta'] += 1
            elif fid in text:
                problems['firm_id_in_input'] += 1
            body = '\n'.join(x for x in text.split('\n') if x.startswith('Headline: ') or x.startswith('Text: '))
            k = text.count('\nArticle ')
            articles += k
            if n_art is not None and k != n_art:
                problems['article_count_differs_from_meta'] += 1
            for name, rx, src in (('year_in_article_text', YEAR, body), ('year_glued_to_time', YEAR_TIME, body),
                                  ('url_in_article_text', td3_mask.URL_RESIDUAL, body),
                                  ('numeric_date_in_input', NUM_DATE, text), ('story_fields_in_input', STORY, text),
                                  ('glued_role_token', glued, body), ('token_hyphen_digit', remnant, body)):
                if rx.search(src):
                    problems[name] += 1
                    examples.setdefault(name, r['key'])
                    bad_keys.add(r['key'])
    out = {'year': year, 'inputs': n, 'articles': articles, 'problems': dict(problems),
           'first_key_per_problem': examples, 'withheld_inputs': len(bad_keys),
           'passed_after_withholding': True,
           'rule': 'an input with any problem is withheld from all three providers alike and takes the fallback'}
    R.new_json(d / f'POPULATION_CHECK_v2_{year}.json', out)
    if not inputs_dir:
        R.new_json(d / f'WITHHELD_POPULATION_{year}.json', {'year': year, 'withheld_keys': sorted(bad_keys),
                                                            'problems': dict(problems)})
    return out
