"""D8 repair pass: rebuild the firm-weeks the main build withheld only because an article kept an unbracketed Reuters
story field (for example 'STORY_TIME: hh:mm:ss PM GMT'), with ONE pre-mask rule.

Why: TD3's masker strips the bracketed form (td3_mask.STORY_FIELDS, 'STORY_TIME:[...]') and its residual check rejects
any remaining 'STORY_NUMBER/DATE/TIME'. The 200-firm-week pilot contained no unbracketed variant, but a few such
articles (108 in the 2024 article store, 137 in 2025) are shown for widely connected partners, so they withhold many
firm-weeks (2024: 522 of 45,942; 2025: far more in some weeks). Withholding is safe but costs coverage.
The rule: before masking, every unbracketed story field 'STORY_(NUMBER|DATE|TIME) : <value without "(", "[" or a line
break, at most 40 characters>' in a headline or body is replaced by the same number of spaces, so TD3's 200/1000
character truncation windows are unchanged; everything else (masker, residual checks, render) is TD3 v6 unchanged.
Bracketed fields are left to TD3's own rule. An input that still has any residual stays withheld. Repaired records
carry td4_repair = 'story_field_spaces_v1' in their build receipt; they are verified like the main inputs and called
after them. Flagged for review (TD4_HANDOFF.md, Deviations).
"""
import json
import re

import pandas as pd

import td4_spec as T
import td4_env as E

RULE = 'story_field_spaces_v1'
UNBRACKETED = re.compile(r'STORY_(?:NUMBER|DATE|TIME)\s*:\s*(?!\[)[^\n(\[]{0,40}')


def repair_text(text):
    if not text:
        return text, 0
    n = 0

    def spaces(m):
        nonlocal n
        n += 1
        return ' ' * len(m.group(0))
    return UNBRACKETED.sub(spaces, text), n


def story_only_keys(year):
    import td4_inputs as I
    E.paths()
    import td3_runtime as R
    keys = set()
    for r in R.read_jsonl(I.INPUTS / f'period_{year}' / 'WITHHELD.jsonl'):
        if r['reason'] == 'date_residual' and re.search(r"residual for sample rank \d+: \{'story_fields': \d+\}$",
                                                         r['message']):
            keys.add((r['firm_id'], pd.Timestamp(r['cutoff'])))
    return keys


def run(year):
    import td4_inputs as I
    if year not in T.PERIODS:
        raise PermissionError('periods only')
    if not (I.INPUTS / f'period_{year}' / 'BUILD.json').exists():
        raise PermissionError('main build of the period not finished')
    keys = story_only_keys(year)

    def only(elig):
        return elig[[(a, pd.Timestamp(c)) in keys for a, c in zip(elig.firm_id, elig.cutoff)]]
    rec = I.run_build((year,), I.repair_dir(year), frame_filter=only, audit_n=len(keys), tag=f'repair_{year}',
                      text_hook=repair_text)
    return {'year': year, 'rule': RULE, 'story_only_withheld_keys': len(keys), 'counts': rec['counts'],
            'elapsed_seconds': rec['elapsed_seconds'], 'peak_working_set_bytes': rec['peak_working_set_bytes']}
