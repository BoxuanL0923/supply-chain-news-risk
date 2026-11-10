"""TD1 firm-week manifest rows for 2025 (features only; no label is computed for any year after 2022).

accounting(), common_all() and build_year() are copies of TD1 v2 build.py with three changes, each marked `TD4:`:
(1) the hard-coded 2024-12-31 seal is a parameter (2025-12-31 for 2025 rows); (2) inputs are read from explicit
paths (TD1 inputs for <= 2024, TD4 snapshots for 2025) through td4_sources.parquet_dated; (3) the year < 2022 label
branch is unchanged and never reached (only 2024 parity and 2025 are built), and no label column is written for
years >= 2023 (as TD1). TD1's pure functions (clock, own_inputs, require_past) and FEATURES are TD1's own modules,
loaded by td4_env.td1. parity2024 rebuilds 2024 with seal 2024-12-31 and compares every column with the frozen TD1
partition.
"""
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4_sources as S

MANIFEST = T.OUT / 'firm_week_manifest'


def _td1():
    return E.td1('core'), E.td1('spec')


def read_snapshot(path, date_col='date', lo='2012-01-01', hi=T.HARD_SEAL):
    return S.parquet_dated(path, pq.read_schema(path).names, date_col=date_col, lo=lo, hi=hi).to_pandas()


def crsp_path(year):
    return (T.TD1 / f'inputs/crsp_{year}.parquet') if year <= 2024 else (S.SNAP / f'crsp_{year}.parquet')


def cashflow_path(year):
    return (T.TD1 / f'inputs/cashflow_{year}.parquet') if year <= 2024 else (S.SNAP / f'cashflow_{year}.parquet')


def accounting(seal):
    """TD1 build.accounting with the seal as a parameter (TD4: seal; TD4: explicit snapshot paths)."""
    core, _ = _td1()
    clock = core.clock
    tag = seal[:4]
    if seal == T.TD2_SEAL:
        qp, ep = T.TD1 / 'inputs/quarters.parquet', T.TD1 / 'inputs/earnings.parquet'
    else:
        qp, ep = S.SNAP / f'quarters_seal{tag}.parquet', S.SNAP / f'earnings_seal{tag}.parquet'
    q = read_snapshot(qp, date_col='datadate', hi=seal)
    e = read_snapshot(ep, date_col='datadate', hi=seal)
    for f in [q, e]:
        f['gvkey'] = pd.to_numeric(f.gvkey, errors='coerce')
        f['datadate'] = pd.to_datetime(f.datadate, errors='coerce')
    e['rdq'] = pd.to_datetime(e.rdq, errors='coerce')
    e = e[(e.rdq.isna()) | (e.rdq <= pd.Timestamp(seal))]                       # TD4: seal
    releases = e[e.rdq >= e.datadate].groupby(['gvkey', 'datadate'], as_index=False).rdq.min()
    q = q.drop_duplicates()
    q = q[~q.duplicated(['gvkey', 'datadate'], keep=False)]
    q = q.merge(releases, on=['gvkey', 'datadate'], how='left')
    q['availability_rule'] = np.where(q.rdq.notna(), 'rdq_end_of_day', 'fiscal_end_plus_120d')
    q['available_day'] = q.rdq.fillna(q.datadate + pd.Timedelta(days=120))
    q = q[q.available_day <= pd.Timestamp(seal)]                                 # TD4: seal
    for c in ['atq', 'cheq', 'invtq', 'dlcq', 'dlttq']:
        q[c] = pd.to_numeric(q[c], errors='coerce')
    valid = q.atq > 0
    q['cash'] = (q.cheq / q.atq).where(valid)
    q['inventory'] = (q.invtq / q.atq).where(valid)
    q['leverage'] = ((q.dlcq + q.dlttq) / q.atq).where(valid)
    link = pq.read_table(T.TD1 / 'inputs/identity.parquet', columns=['gvkey', 'permno']).to_pandas()
    q = q.merge(link, on='gvkey', how='inner')
    e = e[e.rdq.notna()].merge(link, on='gvkey', how='inner')
    q['availability'] = q.available_day.map({v: clock(v) for v in q.available_day.unique()})
    e['availability'] = e.rdq.map({v: clock(v) for v in e.rdq.unique()})
    return q.sort_values(['availability', 'datadate']), e.sort_values('availability')


def cashflow(seal, years):
    """TD1 build.manifests cash-flow preparation (TD4: seal and explicit years)."""
    core, _ = _td1()
    cf = pd.concat([read_snapshot(cashflow_path(y), date_col='eom', lo=f'{y}-01-01', hi=min(f'{y}-12-31', seal))
                    for y in years], ignore_index=True)
    cf = cf[(cf.common == 1) & (cf.primary_sec == 1) & cf.permno.notna()].copy()
    cf['permno'] = cf.permno.astype(int)
    cf['eom'] = pd.to_datetime(cf.eom)
    if cf.duplicated(['eom', 'permno']).any():
        raise ValueError('Ambiguous JKP primary security')
    cf['available_day'] = cf.eom + pd.DateOffset(months=6)
    cf = cf[cf.available_day <= pd.Timestamp(seal)]                             # TD4: seal
    mapping = {v: core.clock(v) for v in cf.available_day.unique()}
    cf['availability'] = cf.available_day.map(mapping)
    return cf


def common_all(r, me, ff, t):
    """TD1 build.common_all (unchanged)."""
    n = r.shape[1]
    if t < 22:
        return {c: np.full(n, np.nan) for c in ['market_log_rv5', 'market_log_rv22', 'industry_log_rv5']}
    rr = r[t - 21:t + 1]
    ww = me[t - 22:t]
    good = np.isfinite(rr) & np.isfinite(ww) & (ww > 0)
    rw = np.where(good, rr * ww, 0.)
    w = np.where(good, ww, 0.)
    den = w.sum(1)
    mkt = np.divide(rw.sum(1), den, out=np.full(22, np.nan), where=(den > 0) & (good.sum(1) >= 50))
    result = {f'market_log_rv{k}': np.full(n, np.log(max(np.mean(mkt[-k:] ** 2), 1e-8))
                                           if np.isfinite(mkt[-k:]).all() else np.nan) for k in [5, 22]}
    industry = np.full(n, np.nan)
    for g in range(1, 50):
        same = ff[t] == g
        if not same.any():
            continue
        nums = rw[-5:, same].sum(1)[:, None] - rw[-5:, same]
        dens = w[-5:, same].sum(1)[:, None] - w[-5:, same]
        ir = np.divide(nums, dens, out=np.full_like(nums, np.nan), where=dens > 0)
        valid = np.isfinite(ir).all(0)
        values = np.log(np.maximum(np.mean(ir ** 2, axis=0), 1e-8))
        values[~valid] = np.nan
        industry[same] = values
    result['industry_log_rv5'] = industry
    return result


def build_year(year, keys, q, e, cf, seal):
    """TD1 build.build_year (TD4: explicit CRSP paths and seal; labels only for year <= 2022 as TD1, never used)."""
    core, spec = _td1()
    clock, own_inputs, require_past, FEATURES = core.clock, core.own_inputs, core.require_past, spec.FEATURES
    E.guard()
    start = time.perf_counter()
    chunks = [read_snapshot(crsp_path(y), lo=f'{y}-01-01', hi=f'{y}-12-31') for y in [year - 1, year]]
    if year < 2022:
        raise PermissionError('TD4 builds manifests for 2024 (parity) and 2025 only')
    d = pd.concat(chunks, ignore_index=True)
    del chunks
    d = d[d.share_type.isin([0, 2])].copy()
    d['date'] = pd.to_datetime(d.date)
    if d.duplicated(['date', 'permno']).any():
        raise ValueError('Duplicate CRSP firm-session')
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    firms = pd.Index(sorted(d.permno.unique()))

    def wide(c):
        return d.pivot(index='date', columns='permno', values=c).reindex(index=dates, columns=firms).to_numpy(dtype=float)
    r, me, ff = wide('ret'), wide('me'), wide('ff49')
    del d
    gc.collect()
    k = keys[keys.date.dt.year == year].copy().sort_values(['date', 'permno'])
    matched = set(q.permno.astype(int))
    eg = {int(p): v.availability.to_numpy(dtype='datetime64[ns]') for p, v in e.groupby('permno')}
    rows = []
    for origin, group in k.groupby('date', sort=True):
        E.guard()
        t = dates.get_indexer([origin])[0]
        cutoff = clock(origin)
        avail = clock(origin, '20:00:00')
        if t < 0:
            raise ValueError('Origin missing from exchange calendar')
        common = common_all(r, me, ff, t)
        qq = q[(q.availability < cutoff) & (q.datadate >= origin - pd.Timedelta(days=550))]
        qr = {int(p): v for p, v in qq.sort_values(['datadate', 'availability']).drop_duplicates('permno', keep='last')
              .set_index('permno').iterrows()}
        cc = cf[(cf.availability < cutoff) & (cf.eom >= origin - pd.Timedelta(days=550))]
        cr = {int(p): v for p, v in cc.sort_values('eom').drop_duplicates('permno', keep='last')
              .set_index('permno').iterrows()}
        for p in group.permno:
            j = firms.get_indexer([p])[0]
            row = {'firm_id': hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24],
                   'origin_session': origin, 'cutoff': cutoff, 'universe_included': True,
                   'fundamental_identity_matched': int(p) in matched}
            if j >= 0:
                row.update(own_inputs(dates, r[:, j], me[:, j], t))
                row.update({c: v[j] for c, v in common.items()})
            else:
                row.update({c: np.nan for c in FEATURES})
            for c in ['size', 'rv1', 'rv5', 'rv22', 'ret1', 'ret5', 'ret22', 'own_downside5',
                      'market_log_rv5', 'market_log_rv22', 'industry_log_rv5']:
                row[c + '__available_at'] = avail if np.isfinite(row.get(c, np.nan)) else pd.NaT
            chosen = qr.get(int(p))
            row['fundamental_rule'] = str(chosen.availability_rule) if chosen is not None else 'missing'
            row['fundamental_fiscal_end'] = chosen.datadate if chosen is not None else pd.NaT
            for c in ['cash', 'inventory', 'leverage']:
                v = float(chosen[c]) if chosen is not None else np.nan
                row[c] = v
                row[c + '__available_at'] = chosen.availability if np.isfinite(v) else pd.NaT
            picked = cr.get(int(p))
            row['cash_flow'] = float(picked.ocf_at) if picked is not None and pd.notna(picked.ocf_at) else np.nan
            row['cash_flow__available_at'] = picked.availability if picked is not None and np.isfinite(
                row['cash_flow']) else pd.NaT
            row['cash_flow_source_eom'] = picked.eom if picked is not None else pd.NaT
            announcements = eg.get(int(p), np.array([], dtype='datetime64[ns]'))
            past = announcements[announcements < cutoff.to_datetime64()]
            covered = len(past) > 0 and past[-1] >= clock(dates[max(0, t - 299)])
            row['earnings_covered'] = float(covered)
            row['post_earnings'] = float(len(past) > 0 and past[-1] >= clock(dates[max(0, t - 4)])) if len(past) \
                else np.nan
            row['earnings_scheduled'] = np.nan
            for c in ['earnings_covered', 'post_earnings']:
                row[c + '__available_at'] = cutoff - pd.Timedelta(seconds=1) if np.isfinite(row[c]) else pd.NaT
            row['earnings_scheduled__available_at'] = pd.NaT
            row['eligible_features'] = bool(j >= 0 and np.isfinite(row['rv22']) and np.isfinite(row['size']))
            for c in FEATURES:
                row.setdefault(c, np.nan)
                row[c + '__missing'] = not np.isfinite(row[c])
                row[c + '__included'] = True
                if not row[c + '__missing']:
                    require_past([row[c + '__available_at']], cutoff, c)
            rows.append(row)
    f = pd.DataFrame(rows)
    for c in [x for x in f if x.endswith('__available_at')] + ['cutoff', 'origin_session', 'fundamental_fiscal_end',
                                                               'cash_flow_source_eom']:
        f[c] = pd.to_datetime(f[c])
    if any(c in f for c in ['y', 'label_end', 'label_observed']):
        raise ValueError('Label column in a TD4 manifest')
    return f, {'year': year, 'rows': len(f), 'eligible_features': int(f.eligible_features.sum()),
               'fundamentals_coverage': {c: float(f[c].notna().mean()) for c in ['cash', 'cash_flow', 'leverage',
                                                                                 'inventory']},
               'elapsed_seconds': time.perf_counter() - start}


def manifest_path(year):
    return (T.TD1 / f'firm_week_manifest/year_{year}.parquet') if year <= 2024 else (MANIFEST / f'year_{year}.parquet')


def build(year):
    import td3_runtime as R
    if year != T.TEST_YEAR:
        raise PermissionError('TD4 builds the 2025 manifest only (2015-2024 are TD1 partitions)')
    path = manifest_path(year)
    if path.exists():
        raise FileExistsError(path)
    keys = pq.read_table(S.KEYS / f'panel_keys_{year}.parquet').to_pandas()
    keys['date'] = pd.to_datetime(keys.date)
    if keys.duplicated(['date', 'permno']).any():
        raise ValueError('Duplicate panel key')
    q, e = accounting(T.HARD_SEAL)
    cf = cashflow(T.HARD_SEAL, range(2014, year + 1))
    f, info = build_year(year, keys, q, e, cf, T.HARD_SEAL)
    E.write_parquet(path, f)
    info.update(sha256=R.sha(path), keys_sha256=R.sha(S.KEYS / f'panel_keys_{year}.parquet'), labels=False, **R.peak())
    R.new_json(T.OUT / f'MANIFEST_{year}_{info["sha256"][:8]}.json', info)
    return info


def parity2024(**_):
    """Rebuild 2024 with seal 2024-12-31 from TD1's own inputs and compare every column with TD1's partition."""
    from td4_parity import _frames_equal
    keys = pq.read_table(T.TD1 / 'inputs/panel_keys.parquet').to_pandas()
    keys['date'] = pd.to_datetime(keys.date)
    q, e = accounting(T.TD2_SEAL)
    cf = cashflow(T.TD2_SEAL, range(2014, 2025))
    f, info = build_year(2024, keys, q, e, cf, T.TD2_SEAL)
    del q, e, cf
    gc.collect()
    frozen = pq.read_table(T.TD1 / 'firm_week_manifest/year_2024.parquet').to_pandas()
    out = _frames_equal(f[frozen.columns], frozen) if set(f.columns) == set(frozen.columns) else \
        {'equal': False, 'column_sets_differ': sorted(set(f.columns) ^ set(frozen.columns))}
    out['info'] = info
    return out
