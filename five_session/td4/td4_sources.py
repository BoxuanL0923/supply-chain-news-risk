"""2025 source extracts with the TD1/TD2 rules, a seal parameter and parity hooks (features only; no label).

Each extractor is a copy of the TD1/TD2 logic it replaces, with the hard-coded seal turned into a parameter:
- csv_dated            <- TD1 v2 sources.csv_dated (Compustat quarterly, earnings dates)
- parquet_dated        <- TD1 v2 sources.dated (CRSP daily, JKP cash flow) via a pyarrow scan predicate
- graph_extract        <- the FSB3 'o' rows behind TD2 sources/graph_october_sealed.parquet, re-read from the raw
                          October 2025 relationships file (direct CUSTOMER/SUPPLIER rows, START_DATE <= seal, END_DATE
                          after the seal censored to null with status censored_after_cutoff)
- links_extract        <- TD2 decoder_v3.mapping (factset_to_permno.csv, start <= seal, end censored at the seal)
- news_scores          <- TD2 decoder_v3.build_news (frozen development RIC identities, earliest day per alt_id)
- universe_keys        <- C:/OptResearch/code/hist_panel_2015_2024/build_panel.py universe and monthly roots, plus the
                          GARCH(1,1) fit-success filter of its assemble stage; the label filter (y finite and > -20)
                          is NOT applied here (it reads future returns) and is applied at evaluation to every arm.
With seal 2024-12-31 (or 2022-12-31 for the panel) each extractor must reproduce the frozen TD1/TD2 file; see
td4_parity. Nothing dated after 2025-12-31 is used: the CRSP, JKP and Reuters files hold no row after 2025-12-31
(row-group statistics, SOURCE_INVENTORY_2025_v1.json); CSV rows dated after the seal are skipped after the date field
is located, before any other field is used.
"""
import csv
csv.field_size_limit(8 * 1024 * 1024)
import datetime as dt
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E

SNAP = T.OUT / 'td1_inputs'          # TD1-style snapshots for 2025 (and re-sealed accounting tables)
TD2SRC = T.OUT / 'td2_sources'       # TD2-style graph, links and news scores sealed at 2025-12-31
KEYS = T.OUT / 'panel_keys'


def _check_seal(hi):
    if hi > T.HARD_SEAL:
        raise PermissionError('TD4 seal: nothing after 2025-12-31 is read')


# ----------------------------------------------------------------------------------------------- TD1 snapshot readers
def parquet_dated(path, cols, date_col='date', lo='2014-01-01', hi=T.HARD_SEAL):
    """TD1 sources.dated with a seal parameter: dates-only pass first, then the projection, both under the scan
    predicate; refuses if any returned date is after hi."""
    _check_seal(hi)
    E.guard()
    schema = pq.read_schema(path)
    typ = schema.field(date_col).type
    convert = str if pa.types.is_string(typ) or pa.types.is_large_string(typ) else (
        dt.date.fromisoformat if pa.types.is_date(typ) else pd.Timestamp)
    filt = (ds.field(date_col) >= pa.scalar(convert(lo), type=typ)) & (
        ds.field(date_col) <= pa.scalar(convert(hi), type=typ))
    dataset = ds.dataset(path, format='parquet')
    dates = dataset.to_table(columns=[date_col], filter=filt, use_threads=False)[date_col].to_pandas()
    if dates.isna().any() or (pd.to_datetime(dates) > pd.Timestamp(hi)).any():
        raise ValueError('Date seal')
    del dates
    result = dataset.to_table(columns=list(dict.fromkeys([date_col] + list(cols))), filter=filt,
                              use_threads=False)
    if pc.max(pc.cast(result[date_col], pa.string())).as_py()[:10] > hi:
        raise ValueError('Date seal after projection')
    return result


def csv_dated(path, cols, date_col='datadate', lo='2012-01-01', hi=T.HARD_SEAL, second_date=None):
    """TD1 sources.csv_dated with a seal parameter (rows outside [lo, hi] and rows whose second date is after the seal
    are skipped before any other field is used)."""
    _check_seal(hi)
    E.guard()
    rows = []
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        di = header.index(date_col)
        ix = [header.index(c) for c in cols]
        si = header.index(second_date) if second_date else None
        for n, values in enumerate(reader):
            day = values[di][:10]
            if not (lo <= day <= hi):
                continue
            if si is not None and values[si] and values[si][:10] > hi:
                continue
            rows.append([values[i] for i in ix])
            if n % 50000 == 0:
                E.guard()
    return pd.DataFrame(rows, columns=cols)


def td1_snapshots(seal=T.HARD_SEAL, years=(T.TEST_YEAR,)):
    """CRSP daily and JKP cash-flow snapshots for the new years, and accounting tables re-sealed at `seal` (TD1's
    quarters.parquet / earnings.parquet were cut at 2024-12-31, so 2025 releases are absent from them)."""
    out = {}
    for y in years:
        p = SNAP / f'crsp_{y}.parquet'
        if not p.exists():
            E.write_parquet(p, parquet_dated(T.CRSP_DAILY, ['date', 'permno', 'ret', 'me', 'ff49', 'share_type'],
                                             lo=f'{y}-01-01', hi=f'{y}-12-31'))
        out[str(p)] = None
        p = SNAP / f'cashflow_{y}.parquet'
        if not p.exists():
            E.write_parquet(p, parquet_dated(T.JKP, ['eom', 'permno', 'ocf_at', 'common', 'primary_sec'],
                                             date_col='eom', lo=f'{y}-01-01', hi=f'{y}-12-31'))
        out[str(p)] = None
    tag = seal[:4]
    p = SNAP / f'quarters_seal{tag}.parquet'
    if not p.exists():
        E.write_parquet(p, csv_dated(T.QUARTERLY, ['gvkey', 'datadate', 'atq', 'cheq', 'invtq', 'dlcq', 'dlttq'],
                                     hi=seal))
    out[str(p)] = None
    p = SNAP / f'earnings_seal{tag}.parquet'
    if not p.exists():
        E.write_parquet(p, csv_dated(T.EARNINGS, ['gvkey', 'datadate', 'rdq'], second_date='rdq', hi=seal))
    out[str(p)] = None
    return out


# ------------------------------------------------------------------------------------------------ TD2 graph and links
GRAPH_COLUMNS = ['id', 'date', 'end', 'status', 'typ', 'src', 'tgt', 'rownum']


def _valid_day(text):
    try:
        dt.date.fromisoformat(text)
        return True
    except ValueError:
        return False


def graph_extract(seal=T.HARD_SEAL, path=None):
    """Direct CUSTOMER/SUPPLIER rows of the October 2025 relationships file with START_DATE <= seal, as the FSB3 'o'
    rows TD2 read: rownum = 1-based data row number; date = START_DATE day; END_DATE <= seal kept (valid), blank
    (missing) or after the seal (nulled, censored_after_cutoff); unparseable dates and the 91,869 FSB3 quarantine IDs
    are excluded; sorted by id, start, rownum."""
    import pyarrow.csv as pcsv
    _check_seal(seal)
    E.guard()
    path = Path(path or T.FACTSET_RELATIONSHIPS)
    quarantine = json.loads((T.TD2 / 'sources/quarantine_ids_v1.json').read_text(encoding='utf-8'))[
        'relationship_ids'] if path == T.FACTSET_RELATIONSHIPS else []
    names = ['ID', 'REL_TYPE', 'SOURCE_FACTSET_ENTITY_ID', 'TARGET_FACTSET_ENTITY_ID', 'START_DATE', 'END_DATE']
    reader = pcsv.open_csv(path, read_options=pcsv.ReadOptions(encoding='latin-1', use_threads=False,
                                                               block_size=16 << 20),
                           parse_options=pcsv.ParseOptions(delimiter='|', quote_char='"', newlines_in_values=False),
                           convert_options=pcsv.ConvertOptions(include_columns=names,
                                                               column_types={n: pa.string() for n in names}))
    kept, n_rows, other = [], 0, 0
    value_set = pa.array(['CUSTOMER', 'SUPPLIER'])
    for batch in reader:                       # streamed: only CUSTOMER/SUPPLIER rows are retained
        rn = pa.array(np.arange(n_rows + 1, n_rows + batch.num_rows + 1, dtype=np.int64))   # 1-based data row
        n_rows += batch.num_rows
        b = pa.Table.from_batches([batch]).append_column('rownum', rn)
        typ_ok = pc.is_in(b['REL_TYPE'], value_set=value_set)
        other += int(b.num_rows - pc.sum(typ_ok).as_py())
        kept.append(b.filter(typ_ok))
        E.guard()
    t = pa.concat_tables(kept)
    del kept
    excluded = {'rows_in_file': n_rows, 'other_type': other}
    start = pc.utf8_slice_codeunits(pc.utf8_trim_whitespace(t['START_DATE']), 0, 10)
    end = pc.utf8_slice_codeunits(pc.utf8_trim_whitespace(t['END_DATE']), 0, 10)
    day_rx = r'^\d{4}-\d{2}-\d{2}$'
    start_ok = pc.match_substring_regex(start, day_rx)
    end_blank = pc.equal(pc.utf8_length(end), 0)
    end_ok = pc.or_(end_blank, pc.match_substring_regex(end, day_rx))
    parse_ok = pc.and_(start_ok, end_ok)
    excluded['unparseable'] = int(t.num_rows - pc.sum(parse_ok).as_py())
    t, start, end, end_blank = (x.filter(parse_ok) for x in (t, start, end, end_blank))
    in_seal = pc.less_equal(start, seal)
    excluded['start_after_seal'] = int(t.num_rows - pc.sum(in_seal).as_py())
    t, start, end, end_blank = (x.filter(in_seal) for x in (t, start, end, end_blank))
    if quarantine:
        keep = pc.invert(pc.is_in(t['ID'], value_set=pa.array(quarantine, type=pa.string())))
        excluded['quarantine'] = int(t.num_rows - pc.sum(keep).as_py())
        t, start, end, end_blank = (x.filter(keep) for x in (t, start, end, end_blank))
    after = pc.and_(pc.invert(end_blank), pc.greater(end, seal))
    status = pc.if_else(end_blank, 'missing', pc.if_else(after, 'censored_after_cutoff', 'valid'))
    end_v = pc.if_else(pc.or_(end_blank, after), pa.scalar(None, pa.string()), end)
    out = pa.table({'id': t['ID'], 'date': start, 'end': end_v, 'status': status, 'typ': t['REL_TYPE'],
                    'src': t['SOURCE_FACTSET_ENTITY_ID'], 'tgt': t['TARGET_FACTSET_ENTITY_ID'], 'rownum': t['rownum']})
    del t
    out = out.take(pc.sort_indices(out, sort_keys=[('id', 'ascending'), ('date', 'ascending'),
                                                    ('rownum', 'ascending')]))
    out = pa.table({k: (out[k].cast(pa.large_string()) if k != 'rownum' else out[k]) for k in GRAPH_COLUMNS})
    return out, excluded


def links_extract(seal=T.HARD_SEAL):
    """TD2 decoder_v3.mapping with a seal parameter (start <= seal; end censored at the seal; duplicates dropped)."""
    _check_seal(seal)
    rows = []
    with T.FS_MAP.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            start = r['link_bdate'][:10]
            if not start or start > seal:
                continue
            try:
                p = int(float(r['PERMNO']))
            except (ValueError, TypeError):
                continue
            end = r['link_edate'][:10]
            rows.append((r['FACTSET_ENTITY_ID'], p, start, min(end or seal, seal)))
    f = pd.DataFrame(rows, columns=['entity', 'permno', 'start', 'end']).drop_duplicates()
    return f.rename(columns={'start': 'date'})


# -------------------------------------------------------------------------------------------------------- news scores
def _td2_score():
    E.paths()
    import engine
    return engine.score


def article_batches(lo, hi, columns, batch_size=2048):
    """TD2 decoder_v3.parquet over the Reuters archive with a seal parameter (row-group skip by statistics, then a
    date mask per 2048-row batch before conversion)."""
    _check_seal(hi)
    pf = pq.ParquetFile(T.ARTICLES, pre_buffer=False, buffer_size=65536)
    typ = pf.schema_arrow.field('date').type
    ix = pf.schema_arrow.get_field_index('date')
    low, high = pa.scalar(lo, type=typ), pa.scalar(hi, type=typ)
    cols = list(dict.fromkeys(['date'] + list(columns)))
    for group in range(pf.num_row_groups):
        st = pf.metadata.row_group(group).column(ix).statistics
        if st and (st.max < lo or st.min > hi):
            continue
        if st and st.max[:10] > T.HARD_SEAL:
            raise PermissionError('Archive row group extends past the 2025 seal')
        for raw in pf.iter_batches(batch_size=batch_size, row_groups=[group], columns=cols, use_threads=False):
            d = raw.column(raw.schema.get_field_index('date'))
            safe = raw.filter(pc.and_(pc.greater_equal(d, low), pc.less_equal(d, high)))
            if safe.num_rows:
                yield safe.to_pandas()
        E.guard()


def news_scores(year, seen, seal=T.HARD_SEAL):
    """TD2 decoder_v3.build_news for one year: alt_ids already seen in earlier partitions are skipped; bindings via
    the frozen development RIC identities; earliest day per alt_id; TD2 lexicon scores of headline[:200]+body[:1000]."""
    _check_seal(f'{year}-12-31' if f'{year}-12-31' <= seal else seal)
    E.paths()
    from engine import day_clock
    score = _td2_score()
    ricmap = json.loads((T.TD2 / 'sources/reuters_identity_development_v1.json').read_text(encoding='utf-8'))['mapping']
    rows = {}
    for batch in article_batches(f'{year}-01-01', f'{year}-12-31', ['alt_id', 'rics', 'headline', 'body_short']):
        for r in batch.itertuples(index=False):
            aid = r.alt_id
            if not aid or aid in seen or not r.rics:
                continue
            day = r.date[:10]
            if day > f'{year}-12-31':
                raise ValueError('Article date seal')
            bindings = sorted({tuple(ricmap[t.strip()]) for t in r.rics.split(',') if t.strip() in ricmap})
            if not bindings:
                continue
            text = (r.headline or '')[:200] + (r.body_short or '')[:1000]
            record = {'date': pd.Timestamp(day), 'alt_id': aid, 'firms': [p for p, m in bindings],
                      'first_months': [m for p, m in bindings], 'available_at': day_clock(day), **score(text)}
            if aid not in rows or day < str(rows[aid]['date'].date()):
                rows[aid] = record
    seen.update(rows)
    f = pd.DataFrame(list(rows.values()), columns=['date', 'alt_id', 'firms', 'first_months', 'available_at'] + T.NAMES)
    gc.collect()
    return f


def seen_ids(years, roots):
    """alt_ids of earlier news partitions (TD2 build_eval_news order: 2014 upward)."""
    seen = set()
    for y in years:
        p = roots[y]
        seen.update(pq.read_table(p, columns=['alt_id']).column('alt_id').to_pylist())
    return seen


def news_path(year):
    return (T.TD2 / f'news_scores/year_{year}.parquet') if year <= 2024 else (TD2SRC / f'news_scores_{year}.parquet')


# ------------------------------------------------------------------------------------------- panel universe and keys
def load_crsp(lo, hi):
    """CRSP daily (share_type 0/2, first row per permno-date) between lo and hi as a long frame."""
    t = parquet_dated(T.CRSP_DAILY, ['permno', 'ret', 'me', 'share_type'], lo=lo, hi=hi)
    d = t.to_pandas()
    d = d[d.share_type.isin([0, 2])].drop_duplicates(['permno', 'date'])
    d['date'] = pd.to_datetime(d.date)
    return d


def load_supply_chain_edges():
    """build_panel.load_edges: derived supplier/customer rows, START/END normalised to days, self-links dropped.
    START_DATE values after the seal are dropped on read (none exist in the October 2025 snapshot)."""
    import pyarrow.csv as pcsv
    t = pcsv.read_csv(T.FACTSET_SUPPLY_CHAIN, read_options=pcsv.ReadOptions(encoding='latin-1'),
                      parse_options=pcsv.ParseOptions(delimiter='|'),
                      convert_options=pcsv.ConvertOptions(
                          include_columns=['SUPPLIER_FACTSET_ENTITY_ID', 'CUSTOMER_FACTSET_ENTITY_ID', 'START_DATE',
                                           'END_DATE'],
                          column_types={'START_DATE': pa.string(), 'END_DATE': pa.string()}))
    start = pc.utf8_slice_codeunits(t['START_DATE'], 0, 10)
    keep = pc.less_equal(start, T.HARD_SEAL)
    t = t.filter(keep)
    end = pc.utf8_slice_codeunits(t['END_DATE'], 0, 10).to_pandas()
    e = pd.DataFrame({'s': t['SUPPLIER_FACTSET_ENTITY_ID'].to_pandas(),
                      'c': t['CUSTOMER_FACTSET_ENTITY_ID'].to_pandas(),
                      'start': pd.to_datetime(pc.utf8_slice_codeunits(t['START_DATE'], 0, 10).to_pandas()),
                      'end': pd.to_datetime(end.mask(end == ''))})
    del t
    e = e[e.s != e.c]
    e['s'] = e.s.astype('category')
    e['c'] = e.c.astype('category')
    return e


def load_panel_mapping():
    """build_panel.load_mapping (factset_to_permno.csv link intervals; no link end lies after 2025-12-31)."""
    m = pd.read_csv(T.FS_MAP, usecols=['FACTSET_ENTITY_ID', 'PERMNO', 'link_bdate', 'link_edate'])
    m.columns = ['entity', 'permno', 'b', 'e']
    m['b'], m['e'] = pd.to_datetime(m.b), pd.to_datetime(m.e)
    if (m.e > pd.Timestamp(T.HARD_SEAL)).any() or (m.b > pd.Timestamp(T.HARD_SEAL)).any():
        m = m[(m.b <= pd.Timestamp(T.HARD_SEAL))]
        m.loc[m.e > pd.Timestamp(T.HARD_SEAL), 'e'] = pd.Timestamp(T.HARD_SEAL)
    return m


def active_edges(e, asof):
    return e[(e.start < asof + pd.Timedelta(days=1)) & (e.end.isna() | (e.end > asof))]


def mapping_at(m, asof, me_at):
    a = m[(m.b <= asof) & (m.e >= asof)].copy()
    a['me'] = a.permno.map(me_at).fillna(-1)
    a = a.sort_values('me', ascending=False).drop_duplicates('entity').drop_duplicates('permno')
    return dict(zip(a.entity, a.permno.astype(int)))


def garch_ok(R, cal, perms, fy):
    """build_panel.stage_garch inclusion: GARCH(1,1) (zero mean, normal) on the 504 sessions ending at the prior
    year end, >= 400 finite returns, fit without exception, omega > 0, alpha >= 0, beta >= 0, alpha + beta < 0.9999.
    Uses only returns dated on or before fy (the last session before the year)."""
    from arch import arch_model
    end_i = cal.get_loc(fy)
    start = end_i - 503
    ok = {}
    for p in perms:
        if p not in R.columns:
            ok[p] = False
            continue
        r = R[p].values.astype(float) * 100
        win = r[start:end_i + 1]
        if np.isfinite(win).sum() < 400:
            ok[p] = False
            continue
        try:
            res = arch_model(win[np.isfinite(win)], mean='Zero', vol='GARCH', p=1, q=1, dist='normal',
                             rescale=False).fit(disp='off', show_warning=False)
            om, al, be = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
        except Exception:
            ok[p] = False
            continue
        ok[p] = bool(om > 0 and al >= 0 and be >= 0 and al + be < 0.9999)
    return ok


def year_origins(cal, year):
    """Last session of each W-SUN week of `year` whose five following sessions exist on or before the year's last
    session (no label window crosses the year end; for 2025 none reaches 2026)."""
    cal = pd.DatetimeIndex(cal)
    s = pd.Series(cal, index=cal)
    origins = s.groupby(cal.to_period('W-SUN')).max()
    pos = pd.Series(np.arange(len(cal)), index=cal)
    last_pos = pos[cal[cal.year == year][-1]]
    return pd.DatetimeIndex([o for o in origins.values if pd.Timestamp(o).year == year
                             and pos[pd.Timestamp(o)] + 5 <= last_pos])


def universe_keys(year, crsp_lo=None, last_origin=None):
    """Panel keys for one year by the build_panel rules: universe = 1,000 largest CRSP share_type 0/2 stocks by ME at
    the last session of year-1 mapping to a FactSet entity with an active supply-chain edge that day; monthly roots
    = universe firms whose entity is in the active graph at the last session before the month; origins = last
    session of each W-SUN week of the year whose five following sessions exist on or before Dec 31 of the year (the
    2015-2024 panel stopped at 2024-12-20, the same position); GARCH fit success on the prior 504 sessions. The
    label filter (y finite and > -20) is not applied (future returns)."""
    if year > T.TEST_YEAR:
        raise PermissionError('Panel keys after 2025 are not built')
    lo = crsp_lo or f'{year - 3}-11-01'
    d = load_crsp(lo, f'{year}-12-31')
    cal = pd.DatetimeIndex(sorted(d.date.unique()))
    R = d.pivot(index='date', columns='permno', values='ret').reindex(cal).astype('float32')
    me_by_date = {k: g.set_index('permno').me for k, g in d.groupby('date')}
    del d
    gc.collect()
    origins = year_origins(cal, year)
    if last_origin is not None:
        origins = origins[origins <= pd.Timestamp(last_origin)]
    e, m = load_supply_chain_edges(), load_panel_mapping()
    fy = cal[cal < pd.Timestamp(year, 1, 1)][-1]
    me_f = me_by_date.get(fy, pd.Series(dtype=float))
    e2p_f = mapping_at(m, fy, me_f)
    act_f = active_edges(e, fy)
    linked = set(act_f.s) | set(act_f.c)
    cand = pd.DataFrame([(p, me_f.get(p, np.nan)) for ent, p in e2p_f.items() if ent in linked], columns=['permno', 'me'])
    universe = cand.dropna().sort_values('me', ascending=False).head(T.PANEL_UNIVERSE_N).permno.astype(int).tolist()
    ok = garch_ok(R, cal, universe, fy)
    rows, months = [], {}
    for mm in sorted({o.month for o in origins}):
        asof = cal[cal < pd.Timestamp(year, mm, 1)][-1]
        ent2perm = mapping_at(m, asof, me_by_date.get(asof, pd.Series(dtype=float)))
        act = active_edges(e, asof)
        ents = set(pd.unique(pd.concat([act.s.astype(str), act.c.astype(str)])))
        perm2ent = {p: ent for ent, p in ent2perm.items()}
        roots = [p for p in universe if perm2ent.get(p) in ents]
        months[mm] = {'asof': str(asof.date()), 'roots': len(roots), 'roots_garch_ok': sum(ok[p] for p in roots)}
        for o in origins[origins.month == mm]:
            rows.extend((o, p) for p in roots if ok[p])
    keys = pd.DataFrame(rows, columns=['date', 'permno'])
    keys['permno'] = keys.permno.astype('int64')
    info = {'year': year, 'universe_size': len(universe), 'universe_asof': str(fy.date()),
            'garch_ok': int(sum(ok.values())), 'origins': len(origins), 'first_origin': str(origins[0].date()),
            'last_origin': str(origins[-1].date()), 'keys': len(keys), 'months': months}
    return keys, info, R, cal


# ------------------------------------------------------------------------------------------------------------- driver
def build_all():
    """Every 2025 extract (exclusive writes; an existing output is kept and re-hashed, never rebuilt)."""
    import td3_runtime as R_
    start = time.perf_counter()
    E.guard()
    receipt = {'version': T.VERSION, 'seal': T.HARD_SEAL, 'labels_read': False}
    receipt['td1_snapshots'] = list(td1_snapshots())
    p = TD2SRC / 'graph_october_sealed_2025.parquet'
    if not p.exists():
        table, excluded = graph_extract()
        E.write_parquet(p, table)
        receipt['graph_excluded'] = excluded
        del table
        gc.collect()
    p = TD2SRC / 'factset_links_2025.parquet'
    if not p.exists():
        E.write_parquet(p, links_extract())
    p = news_path(T.TEST_YEAR)
    if not p.exists():
        seen = seen_ids(range(2014, 2025), {y: news_path(y) for y in range(2014, 2025)})
        f = news_scores(T.TEST_YEAR, seen)
        E.write_parquet(p, f)
        receipt['news_rows_2025'] = len(f)
        del f, seen
        gc.collect()
    p = KEYS / f'panel_keys_{T.TEST_YEAR}.parquet'
    if not p.exists():
        keys, info, _, _ = universe_keys(T.TEST_YEAR)
        E.write_parquet(p, keys)
        receipt['panel_keys'] = info
    outputs = sorted([*SNAP.glob('*.parquet'), *TD2SRC.glob('*.parquet'), *KEYS.glob('*.parquet')])
    receipt['outputs_sha256'] = {str(q): R_.sha(q) for q in outputs}
    receipt['elapsed_seconds'] = time.perf_counter() - start
    receipt.update(R_.peak())
    tag = R_.sha_text(json.dumps(receipt['outputs_sha256'], sort_keys=True))[:8]
    R_.new_json(T.OUT / f'SOURCES_2025_{tag}.json', receipt)
    return {k: v for k, v in receipt.items() if k != 'outputs_sha256'}
