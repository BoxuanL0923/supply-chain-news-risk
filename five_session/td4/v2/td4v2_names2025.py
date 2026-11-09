"""2025 alias source for masker v2: the CRSP CIZ security-information rows that start in 2025 (coordinator pull of
2026-11-06 19:30Z, read-only, reviewed query; file and SHA256 below).

For the 2025 build only, td3_build.name_sources is wrapped so that every PERMNO's CRSP name rows also carry its CIZ
2025 rows (secinfostartdt, issuernm, ticker): TD3 aliases_for and td4v2_mask.add_v2 then derive name variants,
tickers, initialisms, code and unit prefixes from them like from any CRSP name row (any start date in 2025; masking a
name before its start date only removes text). This replaces the hand rename table (its 'renames' section is not used
in the 2025 build) and the post-WRDS withholding rule. A population check (td4v2_rename.check_inputs, run by
td4v2_verify2 for 2025) withholds, for every arm alike, any 2025 input in which a 2025-only CIZ name or ticker of its
focal or a shown firm is still readable. impact() measures how many 2025 inputs the source changes. The 2021-2022,
2023 and 2024 builds do not use this file. No label, return or provider output is read.
"""
import csv
import hashlib
import json
from pathlib import Path

import td4_spec as T
import td4_env as E

CIZ = Path('D:/OptResearch_data/wrds_20261031/td4_names_2025/CRSP_CIZ_SECINFO_2025_v1.csv')
CIZ_SHA256 = '70cf97251b2494460f8f0bab148455ca6ce4d40599b138540c0f76615315f445'
_STATE = {'orig': None, 'rows': None}


def load_ciz(path=CIZ):
    """{permno: [(start day, issuer name, ticker)]} of the CIZ rows dated in 2025 (refuses a changed file)."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if path == CIZ and h != CIZ_SHA256:
        raise PermissionError('CIZ 2025 name file changed')
    out = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as fh:
        for r in csv.DictReader(fh):
            d = (r.get('secinfostartdt') or '')[:10]
            if not ('2025-01-01' <= d <= T.HARD_SEAL):
                raise PermissionError('CIZ row outside 2025: ' + d)
            p = int(float(r['permno']))
            name = ' '.join((r.get('issuernm') or '').upper().split())
            tick = (r.get('ticker') or '').strip().upper()
            out.setdefault(p, []).append((d, name, tick))
    return out, h


def install():
    """Wrap td3_build.name_sources (2025 build only)."""
    E.paths()
    import td3_build as B
    rows, h = load_ciz()
    _STATE['rows'] = rows
    if _STATE['orig'] is None:
        _STATE['orig'] = B.name_sources

    def name_sources_2025(sources, permnos, ricmap):
        ns = _STATE['orig'](sources, permnos, ricmap)
        for p in list(ns['names']):
            for d, name, tick in rows.get(int(p), ()):
                if d <= '2025-12-31':
                    ns['names'][p].append((d, name, tick))
        return ns
    B.name_sources = name_sources_2025
    return h


def build(out_dir=None, frame_filter=None, order=None, audit_n=300, limit=None, chunk=None, tag=None, use_ciz=True):
    """The 2025 masker-v2 build with the CIZ 2025 alias source (td4v2_build2.build otherwise unchanged)."""
    import td4_inputs as I
    import td3_runtime as R
    import td4v2_mask as MV
    import td4v2_build2 as B2
    info = B2.install((2025,))
    ciz_sha = install() if use_ciz else None
    tables = dict(MV._STATE['tables'])
    tables['renames'] = {}                      # replaced by the CIZ source (coordinator 2026-11-06)
    MV._STATE['tables'] = tables
    out_dir = Path(out_dir or B2.PERIOD_DIR[2025])
    kw = {} if chunk is None else {'chunk': chunk}
    rec = I.run_build((2025,), out_dir, frame_filter=frame_filter, order=order, audit_n=audit_n, limit=limit,
                      tag=tag or B2.STORE_TAG[2025], text_hook=MV.story_fields_v2, **kw)
    receipt = {'period': '2025', **info, 'build_receipt_sha256': R.sha(out_dir / 'BUILD.json'),
               'd8_rule': 'td4v2_mask.story_fields_v2', 'mistag_rule': 'td4v2_mask.MISTAG',
               'v1_masker_inputs_used': False, 'ciz_2025_names_file': str(CIZ) if use_ciz else None,
               'ciz_2025_names_sha256': ciz_sha, 'hand_rename_table_used': False,
               'code_sha256_names2025': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    R.new_json(out_dir / 'MASK_V2_RECEIPT.json', receipt)
    return {'counts': rec['counts'], 'elapsed_seconds': rec['elapsed_seconds'],
            'peak_working_set_bytes': rec['peak_working_set_bytes'], 'out_dir': str(out_dir)}


def impact(n=1000):
    """How many 2025 inputs the CIZ source changes: n built inputs (hash-ranked) rebuilt without it and compared byte
    for byte with the final build; written to OUT/CIZ_2025_IMPACT.json."""
    import td3_runtime as R
    import pandas as pd
    import pyarrow.parquet as pq
    import td4v2_build2 as B2
    d = Path(B2.PERIOD_DIR[2025])
    e = pq.read_table(d / 'ELIGIBLE.parquet').to_pandas()
    e = e[e.status == 'input_built']
    e['r'] = [hashlib.sha256(f'TD4-ciz-impact|{k}'.encode()).hexdigest() for k in e.key]
    pick = e.sort_values('r').iloc[:n]
    pairs = {(a, pd.Timestamp(c)) for a, c in zip(pick.firm_id, pick.cutoff)}
    out_dir = T.OUT / 'inputs_m2_dev' / f'ciz_impact_without_{R.utc().replace(":", "")}'
    build(out_dir=out_dir, frame_filter=B2.keys_filter(pairs), audit_n=0, chunk=500, use_ciz=False)
    final = {}
    with (d / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        want = set(pick.key)
        for line in f:
            r = json.loads(line)
            if r['key'] in want:
                final[r['key']] = r['input_sha256']
    other = {r['key']: r['input_sha256'] for r in R.read_jsonl(out_dir / 'INPUTS.jsonl')}
    common = sorted(set(final) & set(other))
    changed = [k for k in common if final[k] != other[k]]
    res = {'sample_inputs': len(common), 'changed_by_ciz': len(changed),
           'share_changed': len(changed) / len(common) if common else None,
           'withheld_only_without_ciz': len(set(final) - set(other)),
           'sample_rule': "first n built 2025 keys by SHA256('TD4-ciz-impact|' + key)", 'without_dir': str(out_dir),
           'ciz_sha256': CIZ_SHA256}
    R.new_json(T.OUT / 'CIZ_2025_IMPACT.json', res)
    return res
