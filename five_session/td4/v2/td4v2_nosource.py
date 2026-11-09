"""MASKER_NO_SOURCE (rule v2.1, coordinator technical correction of 2026-11-07, fixed before any provider call and before
any outcome is read): for 2023, 2024 and 2025 it replaces the M4 firm-level classes NONAME and NONAME_AFTER_2022 as a
withholding class.

Reason: s13 NONAME tests for no CRSP name row dated on or before 2024-12-31 (it does not read the CIZ 2025 source) and
NONAME_AFTER_2022 for a first CRSP name after 2022-12-31; both stand in for "the masker had no name to mask for this
firm". In 2023-2025 that premise fails (recently listed firms are ordinary and masker v2 masked them from its tables);
their text residues are covered by the span detectors that ran and were redacted.

Rule: an input is withheld (every arm alike) when its focal firm or a shown firm (menu partner) has no name and no
ticker entry in the exact name tables masker v2 used for that period's build:
  - CRSP stocknames / names rows (the firm's own rows and its issuer siblings' rows, i.e. PERMNOs sharing a CUSIP6) dated
    on or before the build's last day (MASK_V2_RECEIPT 'last_day'), as td3_build.load_names / issuer_siblings read them;
  - the FSB1 FactSet names and US tickers linked by issuer CUSIP6 (start_ on or before the last day;
    td3_build.factset_aliases);
  - for 2025, the bound CRSP CIZ 2025 file (MASK_V2_RECEIPT 'ciz_2025_names_sha256'; td4v2_names2025.load_ciz).
RIC roots from the Reuters identity table are not a name table and do not count. Computed from these tables (not from
s13). 2021-2022 is unchanged (NONAME / NONAME_AFTER_2022 stay withholding classes there; they flag no input); the check is
still computed for the record. Writes <masker-v2 build>/MASKER_NO_SOURCE_<UTC>.json. Reads no label, return or provider
output.
"""
import hashlib
import json
from pathlib import Path

import td4_spec as T
import td4_env as E

YEARS = {'2122': (2021, 2022), '2023': (2023,), '2024': (2024,), '2025': (2025,)}


def _sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sourced(permno, names, siblings, fsnames, fstickers, ciz, last_day):
    """True when the firm has at least one name or ticker entry in the masker's tables: an own or issuer-sibling CRSP
    row dated <= last_day with a company name or a ticker, an issuer-linked FactSet name or US ticker, or (2025) a CIZ
    row with an issuer name or ticker."""
    p = int(permno)
    for q in [p] + [int(x) for x in siblings.get(p, [])]:
        for d, comnam, tick in names.get(q, []):
            if d <= last_day and ((comnam or '').strip() or (tick or '').strip()):
                return True
    if fsnames.get(p) or fstickers.get(p):
        return True
    for d, name, tick in ciz.get(p, ()):
        if d <= last_day and ((name or '').strip() or (tick or '').strip()):
            return True
    return False


def shown_permnos(meta):
    return [int(meta['focal_permno'])] + [int(s['permno']) for s in meta['shown']]


def flag_inputs(metas, unsourced):
    """{key: sorted unsourced PERMNOs} of the inputs whose focal or a shown firm is unsourced."""
    out = {}
    for m in metas:
        bad = sorted({p for p in shown_permnos(m) if p in unsourced})
        if bad:
            out[m['key']] = bad
    return out


def tables(period, permnos):
    """The masker's name tables for the period's build, restricted to the given PERMNOs and their issuer siblings."""
    E.paths()
    import td3_build as B
    import td3_spec as S3
    import td4_inputs as I
    import td4v2_build2 as B2
    key = '2122' if period == '2122' else int(period)
    rec = json.loads((Path(B2.PERIOD_DIR[key]) / 'MASK_V2_RECEIPT.json').read_text(encoding='utf-8'))
    I.set_period(YEARS[period])
    last_day = S3.LAST_DAY
    if last_day != rec['last_day']:
        raise PermissionError('period parameters differ from the build receipt')
    crsp, fsb1 = tuple(T.CRSP_NAME_FILES), T.FSB1_COMPANY
    siblings, cusip6 = B.issuer_siblings(crsp, permnos)
    allp = set(int(p) for p in permnos) | {q for v in siblings.values() for q in v}
    names = B.load_names(crsp, allp)
    fsn, fst = B.factset_aliases(fsb1, cusip6)
    ciz, ciz_sha = {}, None
    if period == '2025':
        import td4v2_names2025 as N25
        ciz, ciz_sha = N25.load_ciz()
        if ciz_sha != rec.get('ciz_2025_names_sha256'):
            raise PermissionError('CIZ file differs from the 2025 build receipt')
    files = {str(p): _sha(p) for p in crsp + (fsb1,)}
    if period == '2025':
        import td4v2_names2025 as N25
        files[str(N25.CIZ)] = ciz_sha
    return {'names': names, 'siblings': siblings, 'fsnames': fsn, 'fstickers': fst, 'ciz': ciz, 'last_day': last_day,
            'files_sha256': files, 'receipt': rec}


def run(period):
    """MASKER_NO_SOURCE for one period's callable inputs; written to the build folder."""
    E.paths()
    import time
    import td3_runtime as R
    import td4v2_inputs as IV
    period = str(period)
    t0 = time.time()
    d = IV.period_dir(period)
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    metas = []
    with (Path(d) / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            if m['key'] not in pop:
                metas.append({'key': m['key'], 'focal_permno': m['focal_permno'],
                              'shown': [{'permno': s['permno']} for s in m['shown']]})
    permnos = sorted({p for m in metas for p in shown_permnos(m)})
    tb = tables(period, permnos)
    unsourced = {p for p in permnos if not sourced(p, tb['names'], tb['siblings'], tb['fsnames'], tb['fstickers'],
                                                   tb['ciz'], tb['last_day'])}
    flagged = flag_inputs(metas, unsourced)
    focal_only = sum(1 for m in metas if int(m['focal_permno']) in unsourced)
    ciz_only = 0
    if period == '2025':
        empty = {}
        ciz_only = sum(1 for p in permnos if p not in unsourced and
                       not sourced(p, tb['names'], tb['siblings'], tb['fsnames'], tb['fstickers'], empty, tb['last_day']))
    res = {'period': period, 'rule': 'MASKER_NO_SOURCE (v2.1): focal or shown firm without any name or ticker entry in '
                                     'the masker-v2 name tables of the build (CRSP own and issuer-sibling rows <= last '
                                     'day, FSB1 issuer-linked names and US tickers, CIZ 2025 for 2025)',
           'applied': period != '2122', 'build_dir': str(d), 'last_day': tb['last_day'],
           'receipt_sha256': R.sha(Path(d) / 'MASK_V2_RECEIPT.json'), 'tables_files_sha256': tb['files_sha256'],
           'callable': len(metas), 'shown_or_focal_permnos': len(permnos), 'unsourced_permnos': len(unsourced),
           'sourced_only_by_ciz_2025': ciz_only, 'flagged_inputs': len(flagged),
           'flagged_share': round(len(flagged) / len(metas), 5) if metas else None,
           'flagged_by_unsourced_focal': focal_only, 'flagged_keys': sorted(flagged),
           'unsourced_permnos_list': sorted(unsourced),
           'code_sha256': R.sha(Path(__file__)), 'elapsed_s': round(time.time() - t0), **R.peak()}
    R.new_json(Path(d) / f'MASKER_NO_SOURCE_{R.utc().replace(":", "")}.json', res)
    return {k: v for k, v in res.items() if k not in ('flagged_keys', 'unsourced_permnos_list')}


def latest(d):
    runs = sorted(Path(d).glob('MASKER_NO_SOURCE_*.json'))
    if not runs:
        raise PermissionError('MASKER_NO_SOURCE not computed for ' + str(d))
    return runs[-1]


def stand_in(d):
    """Rehearsal only: an empty MASKER_NO_SOURCE file for a sandbox build under OUT/rehearsal_v2_<id>/."""
    import td3_runtime as R
    if 'rehearsal_v2_' not in str(d):
        raise PermissionError('stand-in only in a rehearsal sandbox')
    return R.new_json(Path(d) / f'MASKER_NO_SOURCE_{R.utc().replace(":", "")}.json',
                      {'flagged_keys': [], 'flagged_inputs': 0, 'applied': True, 'unsourced_permnos': 0,
                       'rehearsal_stand_in': True})
