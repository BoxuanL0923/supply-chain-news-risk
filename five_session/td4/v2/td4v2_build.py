"""TD4 v2 input builds (amendment 2): the 2021-2022 selection subsample and the 2023 test year.

Both use the v1 build path unchanged (td4_inputs.run_build: TD3 v6 masker and prompt, chunked and resumable), the v1
D8 repair rule (td4_repair.repair_text) for firm-weeks withheld only by an unbracketed story field, the population
checks and the audit samples. The 2024 and 2025 inputs are the v1 builds, reused exactly as built.

Selection subsample: the TD3-eligible 2021-2022 firm-weeks (TD2 call unit + a verified partner article) are ordered by
SHA256("TD4-select-2122|" + key), key = TD3's record key SHA256("TD3-key|<firm_id>|<cutoff ISO>")[:16]; the first
CANDIDATES in that order are built, and the subsample is the first SUBSAMPLE_N of them that are TD3-eligible (built,
or withheld by a masking residual). TD3 pilot firm-weeks are included when drawn; their calls are fresh.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

import td4_spec as T
import td4_env as E

SUBSAMPLE_N = 20000
CANDIDATES = 20400             # margin for candidates without a verified article (TD3: 0 of 200 examined)
# select_2122/ holds an abandoned first attempt (hash-order chunks spanned every 2021-2022 week, so per-chunk edge maps
# for about 100 origins exceeded the memory cap in chunk 1); the build runs in select_2122_b/ in (cutoff, firm_id)
# order. Its article store (same candidate keys) is reused under the tag select_2122.
SELECT_DIR = T.OUT / 'inputs' / 'select_2122_b'
SELECT_REPAIR_DIR = T.OUT / 'inputs' / 'select_2122_b_repair'
YEAR_DIR = {2023: T.OUT / 'inputs' / 'period_2023', 2024: T.OUT / 'inputs' / 'period_2024',
            2025: T.OUT / 'inputs' / 'period_2025'}
REPAIR_DIR = {2023: T.OUT / 'inputs' / 'repair_2023', 2024: T.OUT / 'inputs' / 'repair_2024',
              2025: T.OUT / 'inputs' / 'repair_2025'}


def record_key(firm_id, cutoff):
    """TD3 td3_build.record_key (same string)."""
    return hashlib.sha256(f'TD3-key|{firm_id}|{pd.Timestamp(cutoff).isoformat()}'.encode()).hexdigest()[:16]


def select_rank(key):
    return hashlib.sha256(('TD4-select-2122|' + key).encode()).hexdigest()


def _order(elig):
    r = [select_rank(record_key(a, b)) for a, b in zip(elig.firm_id, elig.cutoff)]
    return elig.assign(_r=r).sort_values('_r', kind='stable').drop(columns='_r').reset_index(drop=True)


def build_select():
    import td4_inputs as I
    E.paths()
    # membership = first CANDIDATES by hash rank; build order = (cutoff, firm_id) as in the year builds (bounded
    # memory: a chunk covers a few weeks); the subsample is taken in hash order afterwards (subsample_keys)
    rec = I.run_build((2021, 2022), SELECT_DIR, frame_filter=lambda e: _order(e).iloc[:CANDIDATES],
                      audit_n=300, tag='select_2122')
    return rec['counts']


def build_year(year):
    import td4_inputs as I
    if year != 2023:
        raise PermissionError('v2 builds 2023 only (2024 and 2025 are the v1 builds)')
    rec = I.run_build((2023,), YEAR_DIR[2023], tag='2023')       # same call as td4_inputs.build for 2024/2025
    return rec['counts']


def story_only(dir_):
    import td3_runtime as R
    import re
    out = set()
    for r in R.read_jsonl(Path(dir_) / 'WITHHELD.jsonl'):
        if r['reason'] == 'date_residual' and re.search(r"residual for sample rank \d+: \{'story_fields': \d+\}$",
                                                         r['message']):
            out.add((r['firm_id'], pd.Timestamp(r['cutoff'])))
    return out


def repair(which):
    """D8 repair (v1 rule td4_repair.repair_text) of a v2 build: 'select' or 2023."""
    import td4_inputs as I
    import td4_repair as P
    if which == 'select':
        src, out, years, tag = SELECT_DIR, SELECT_REPAIR_DIR, (2021, 2022), 'select_2122_repair'
    elif which == 2023:
        src, out, years, tag = YEAR_DIR[2023], REPAIR_DIR[2023], (2023,), 'repair_2023'
    else:
        raise ValueError(which)
    if not (src / 'BUILD.json').exists():
        raise PermissionError('main build not finished: ' + str(src))
    keys = story_only(src)
    if not keys:
        return {'which': str(which), 'story_only_withheld_keys': 0, 'built': False}

    def only(e):
        return e[[(a, pd.Timestamp(c)) in keys for a, c in zip(e.firm_id, e.cutoff)]]
    rec = I.run_build(years, out, frame_filter=only, audit_n=len(keys), tag=tag, text_hook=P.repair_text)
    return {'which': str(which), 'rule': P.RULE, 'story_only_withheld_keys': len(keys), 'counts': rec['counts']}


def subsample_keys():
    """The selection subsample: first SUBSAMPLE_N TD3-eligible candidates in hash order -> {record key: status}."""
    import pyarrow.parquet as pq
    e = pq.read_table(SELECT_DIR / 'ELIGIBLE.parquet').to_pandas()
    e['rk'] = [record_key(a, b) for a, b in zip(e.firm_id, e.cutoff)]
    e['rank'] = [select_rank(k) for k in e.rk]
    e = e.sort_values('rank', kind='stable')
    td3_eligible = e[e.status.isin(['input_built']) | e.status.str.startswith('withheld_')]
    sub = td3_eligible.iloc[:SUBSAMPLE_N]
    if len(sub) < SUBSAMPLE_N:
        raise ValueError(f'Only {len(sub)} TD3-eligible candidates built; raise CANDIDATES')
    return sub
