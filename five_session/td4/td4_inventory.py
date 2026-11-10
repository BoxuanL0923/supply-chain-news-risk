"""2025 coverage inventory of every local source TD1/TD2/TD3 use (counts only; no outcome is computed).

Parquet: row-group statistics first; the date column alone is read only when the statistics show no row after
2025-12-31 (otherwise row groups after the seal are skipped and a straddling group is reported, not read). CSV: the
csv module tokenises each row to locate its date field; no other field of a row dated after 2025-12-31 is accessed.
Nothing here reads returns as labels: CRSP is counted by session date and permno only.
"""
import csv
csv.field_size_limit(8 * 1024 * 1024)   # FSB5 interval-list fields (as TD2 decoder_v3)
import datetime as dt
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import td4_spec as T
import td4_env as E

SEAL = T.HARD_SEAL


def _year(day):
    return day[:4] if day and len(day) >= 4 and day[:4].isdigit() else 'blank'


def parquet_stats(path, datecol):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    ix = pf.schema_arrow.get_field_index(datecol)
    groups = []
    for g in range(pf.num_row_groups):
        st = pf.metadata.row_group(g).column(ix).statistics
        groups.append({'group': g, 'rows': pf.metadata.row_group(g).num_rows,
                       'min': str(st.min)[:10] if st is not None and st.has_min_max else None,
                       'max': str(st.max)[:10] if st is not None and st.has_min_max else None})
    return pf, groups


def parquet_years(path, datecol, extra=None):
    """Counts by year of datecol (and optional non-null counts of extra columns by year) for row groups whose max is
    on or before the seal; groups after the seal are skipped; a straddling group is reported and not read."""
    import pyarrow.compute as pc
    E.guard()
    pf, groups = parquet_stats(path, datecol)
    years, nonnull = Counter(), defaultdict(Counter)
    skipped, straddling = [], []
    cols = [datecol] + list(extra or [])
    for g in groups:
        if g['min'] is not None and g['min'] > SEAL:
            skipped.append(g['group'])
            continue
        if g['max'] is None or g['max'] > SEAL:
            straddling.append(g['group'])
            continue
        for b in pf.iter_batches(batch_size=262144, row_groups=[g['group']], columns=cols, use_threads=False):
            d = b.column(0)
            ys = pc.utf8_slice_codeunits(pc.cast(d, 'string'), 0, 4).to_pylist()
            years.update(ys)
            for c in extra or []:
                valid = pc.is_valid(b.column(c)).to_pylist()
                for y, v in zip(ys, valid):
                    if v:
                        nonnull[c][y] += 1
        E.guard()
    return {'path': str(path), 'date_column': datecol, 'rows_total_metadata': pf.metadata.num_rows,
            'row_groups': len(groups), 'max_date_metadata': max((g['max'] for g in groups if g['max']), default=None),
            'groups_after_seal_skipped': skipped, 'straddling_groups_not_read': straddling,
            'rows_by_year': {k: years[k] for k in sorted(years) if k >= '2022'},
            'nonnull_by_year': {c: {k: v[k] for k in sorted(v) if k >= '2022'} for c, v in nonnull.items()}}


def csv_years(path, datecol, second=None, delimiter=',', quotechar='"', filt=None):
    """Counts by year of a CSV date column; optional second date column counted only for rows on or before the seal;
    filt(row_dict_of_needed_fields) restricts rows (fields requested by name are accessed only after the date check)."""
    E.guard()
    years, years2, n = Counter(), Counter(), 0
    with Path(path).open(encoding='utf-8-sig', errors='replace', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        header = next(reader)
        di = header.index(datecol)
        si = header.index(second) if second else None
        fi = {k: header.index(k) for k in (filt[0] if filt else [])}
        for row in reader:
            n += 1
            if len(row) <= di:
                years['short_row'] += 1
                continue
            day = row[di].strip()[:10]
            y = _year(day)
            if day and day > SEAL:
                years['after_seal'] += 1          # nothing else of this row is accessed
                continue
            if filt and not filt[1]({k: row[i] for k, i in fi.items()}):
                continue
            years[y] += 1
            if si is not None:
                d2 = row[si].strip()[:10]
                years2['after_seal' if d2 and d2 > SEAL else _year(d2)] += 1
            if n % 500000 == 0:
                E.guard()
    out = {'path': str(path), 'date_column': datecol, 'rows_read': n,
           'rows_by_year': {k: years[k] for k in sorted(years) if k >= '2022' or not k[0].isdigit()}}
    if second:
        out['second_column'] = second
        out['second_by_year'] = {k: years2[k] for k in sorted(years2) if k >= '2022' or not k[0].isdigit()}
    return out


def crsp_2025():
    """CRSP daily compact: sessions and firms per year (date and permno only) and share_type 0/2 coverage."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    pf, groups = parquet_stats(T.CRSP_DAILY, 'date')
    sessions, firms, rows = defaultdict(set), defaultdict(set), Counter()
    for g in groups:
        if g['max'] is None or g['max'] < '2023-01-01' or g['min'] > SEAL:
            continue
        for b in pf.iter_batches(batch_size=262144, row_groups=[g['group']], columns=['date', 'permno', 'share_type'],
                                 use_threads=False):
            keep = pc.is_in(b.column('share_type'), value_set=__import__('pyarrow').array([0, 2], type=b.column('share_type').type))
            b = b.filter(keep)
            for d, p in zip(b.column('date').to_pylist(), b.column('permno').to_pylist()):
                y = d.year
                if y >= 2023:
                    sessions[y].add(d)
                    firms[y].add(p)
                    rows[y] += 1
        E.guard()
    return {'path': str(T.CRSP_DAILY), 'max_date_metadata': max(g['max'] for g in groups),
            'share_type_0_2': {str(y): {'rows': rows[y], 'sessions': len(sessions[y]), 'permnos': len(firms[y]),
                                        'last_session': str(max(sessions[y])) if sessions[y] else None}
                               for y in sorted(rows)}}


def factset_relationships():
    """Direct CUSTOMER/SUPPLIER rows of the October 2025 snapshot by START_DATE year and END_DATE status."""
    E.guard()
    starts, ends, n = Counter(), Counter(), 0
    with T.FACTSET_RELATIONSHIPS.open(encoding='latin-1', newline='') as f:
        reader = csv.reader(f, delimiter='|', quotechar='"')
        header = next(reader)
        it, si, ei = header.index('REL_TYPE'), header.index('START_DATE'), header.index('END_DATE')
        for row in reader:
            n += 1
            if len(row) <= ei or row[it] not in ('CUSTOMER', 'SUPPLIER'):
                continue
            s = row[si].strip()[:10]
            if s > SEAL:
                starts['after_seal'] += 1
                continue
            starts[_year(s)] += 1
            e = row[ei].strip()[:10]
            ends['blank' if not e else ('after_seal' if e > SEAL else _year(e))] += 1
    return {'path': str(T.FACTSET_RELATIONSHIPS), 'rows_read': n,
            'start_by_year': {k: starts[k] for k in sorted(starts) if k >= '2020' or not k[0].isdigit()},
            'end_by_year': {k: ends[k] for k in sorted(ends) if k >= '2022' or not k[0].isdigit()},
            'snapshot_note': 'File dated 2025-10-02 (October 2025 DataFeed snapshot); no record published after '
                             'the snapshot exists, and END_DATE values after the snapshot are unknown.'}


def run():
    start = time.perf_counter()
    E.guard()
    import td3_runtime as R
    out = {'version': T.VERSION, 'seal': SEAL, 'purpose': '2025 coverage of TD1/TD2/TD3 sources (amendment item 4)'}
    out['crsp_daily'] = crsp_2025()
    out['jkp_chars'] = parquet_years(T.JKP, 'eom', extra=['ocf_at'])
    out['reuters_articles'] = parquet_years(T.ARTICLES, 'date', extra=['rics'])
    out['compustat_quarterly'] = csv_years(T.QUARTERLY, 'datadate')
    out['earnings_dates'] = csv_years(T.EARNINGS, 'datadate', second='rdq')
    out['factset_to_permno'] = csv_years(T.FS_MAP, 'link_bdate', second='link_edate')
    out['crsp_stocknames'] = csv_years(T.CRSP_NAME_FILES[0], 'namedt', second='nameenddt')
    out['crsp_names'] = csv_years(T.CRSP_NAME_FILES[1], 'namedt', second='nameenddt')
    out['fsb5_candidates'] = csv_years(T.FSB5, 'bridge_start')
    out['factset_relationships'] = factset_relationships()
    out['factset_supply_chain_derived'] = csv_years(T.FACTSET_SUPPLY_CHAIN, 'START_DATE', second='END_DATE',
                                                    delimiter='|')
    out['fsb1_company_names'] = {k: v for k, v in parquet_years(T.FSB1_COMPANY, 'start_').items()}
    out['panel_keys'] = parquet_years(T.PANEL, 'date')
    out['td2_reuters_identity'] = {'path': str(T.TD2 / 'sources/reuters_identity_development_v1.json'),
                                   'note': 'frozen development RIC->PERMNO mapping (keys <= 2022), used for every '
                                           'year by TD2 amendment A4; reused unchanged for 2024-2025'}
    out['elapsed_seconds'] = time.perf_counter() - start
    out.update(R.peak())
    path = R.new_json(T.OUT / 'inventory' / 'SOURCE_INVENTORY_2025_v1.json', out)
    return path, out
