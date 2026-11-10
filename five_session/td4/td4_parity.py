"""Parity of the TD4 copies with the frozen TD1/TD2/TD3 outputs (offline; only development-era labels, never
2023-2025 outcomes). Each check writes PARITY_<name>_<tag>.json under the private root.

graph2024     graph_extract(seal 2024-12-31)            == TD2 sources/graph_october_sealed.parquet
links2024     links_extract(seal 2024-12-31)            == TD2 sources/factset_links.parquet
td1snap2024   csv_dated/parquet_dated (seal 2024-12-31) == TD1 inputs quarters/earnings/crsp_2024/cashflow_2024
news2024      news_scores(2024)                          == TD2 news_scores/year_2024.parquet
panel2022     universe_keys(2022) + panel label filter   == hist panel 2022 keys (panel y recomputed from CRSP)
panel2024     universe_keys(2024) (no label filter)      contains the hist panel 2024 keys; differences counted
manifest2024  td4_manifest rows (seal 2024-12-31)        == TD1 firm_week_manifest/year_2024.parquet
features2024  td4_features rows for a sample of origins  == TD2 features_v2/evidence_v2 year_2024 rows
inputs_pilot  td4_inputs on the 200 TD3 pilot firm-weeks == TD3 build_v7 INPUTS.jsonl (byte-identical texts)
refit2022     td4_models refit through 2022              == TD1 model_frozen_v2.pkl predictions, TD2 MODEL_v2 betas
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E


def _frames_equal(a, b, sort=None):
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if sort:
        a = a.sort_values(sort).reset_index(drop=True)
        b = b.sort_values(sort).reset_index(drop=True)
    out = {'rows_td4': len(a), 'rows_frozen': len(b), 'columns_equal': list(a.columns) == list(b.columns)}
    if len(a) != len(b) or not out['columns_equal']:
        out['equal'] = False
        return out
    diffs = {}
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            bad = ~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))
        else:
            bad = ~((x.astype(object).values == y.astype(object).values) | (pd.isna(x).values & pd.isna(y).values))
        if bad.any():
            diffs[c] = int(bad.sum())
    out['column_differences'] = diffs
    out['equal'] = not diffs
    return out


def graph2024(**_):
    import td4_sources as S
    mine, excluded = S.graph_extract(seal=T.TD2_SEAL)
    frozen = pq.read_table(T.TD2 / 'sources/graph_october_sealed.parquet')
    out = {'rows_td4': mine.num_rows, 'rows_frozen': frozen.num_rows,
           'columns_equal': mine.column_names == frozen.column_names}
    diffs = {}
    if out['columns_equal'] and mine.num_rows == frozen.num_rows:
        for c in mine.column_names:
            if not mine[c].cast(frozen[c].type).equals(frozen[c]):
                diffs[c] = 'differs'
    out['column_differences'] = diffs
    out['equal'] = out['columns_equal'] and mine.num_rows == frozen.num_rows and not diffs
    out['excluded'] = excluded
    return out


def links2024(**_):
    import td4_sources as S
    mine = S.links_extract(seal=T.TD2_SEAL)
    frozen = pq.read_table(T.TD2 / 'sources/factset_links.parquet').to_pandas()
    return _frames_equal(mine, frozen)


def td1snap2024(**_):
    import td4_sources as S
    out = {}
    q = S.csv_dated(T.QUARTERLY, ['gvkey', 'datadate', 'atq', 'cheq', 'invtq', 'dlcq', 'dlttq'], hi=T.TD2_SEAL)
    out['quarters'] = _frames_equal(q, pq.read_table(T.TD1 / 'inputs/quarters.parquet').to_pandas())
    e = S.csv_dated(T.EARNINGS, ['gvkey', 'datadate', 'rdq'], second_date='rdq', hi=T.TD2_SEAL)
    out['earnings'] = _frames_equal(e, pq.read_table(T.TD1 / 'inputs/earnings.parquet').to_pandas())
    c = S.parquet_dated(T.CRSP_DAILY, ['date', 'permno', 'ret', 'me', 'ff49', 'share_type'], lo='2024-01-01',
                        hi='2024-12-31').to_pandas()
    out['crsp_2024'] = _frames_equal(c, pq.read_table(T.TD1 / 'inputs/crsp_2024.parquet').to_pandas())
    j = S.parquet_dated(T.JKP, ['eom', 'permno', 'ocf_at', 'common', 'primary_sec'], date_col='eom',
                        lo='2024-01-01', hi='2024-12-31').to_pandas()
    fj = pq.read_table(T.TD1 / 'inputs/cashflow_2024.parquet').to_pandas()
    j['eom'] = pd.to_datetime(j.eom)
    out['cashflow_2024'] = _frames_equal(j, fj)
    out['equal'] = all(v['equal'] for v in out.values())
    return out


def news2024(**_):
    import td4_sources as S
    seen = S.seen_ids(range(2014, 2024), {y: S.news_path(y) for y in range(2014, 2024)})
    mine = S.news_scores(2024, seen, seal=T.TD2_SEAL)
    frozen = pq.read_table(T.TD2 / 'news_scores/year_2024.parquet').to_pandas()
    for f in (mine, frozen):
        f['firms'] = f.firms.map(lambda v: tuple(int(x) for x in v))
        f['first_months'] = f.first_months.map(lambda v: tuple(int(x) for x in v))
    return _frames_equal(mine, frozen)


def panel_y(R, cal, keys):
    """Hist panel y = log(max(mean of r^2 over t+1..t+5 (float32, all five present), 1e-12)); used only for the
    development-year panel parity (2022)."""
    sq = R.pow(2)
    fwd = sq[::-1].rolling(5, min_periods=5).mean()[::-1].shift(-1)
    y = np.log(fwd.clip(lower=1e-12))
    st = y.stack(future_stack=True).rename('y').reset_index()
    st.columns = ['date', 'permno', 'y']
    return keys.merge(st, on=['date', 'permno'], how='left')


def panel2022(**_):
    import td4_sources as S
    keys, info, R, cal = S.universe_keys(2022, crsp_lo='2019-09-01')
    # the panel's y for late-December origins needs January 2023 sessions (development data, label end <= 2023-01)
    extra = S.load_crsp('2023-01-01', '2023-01-15')
    R2 = pd.concat([R, extra.pivot(index='date', columns='permno', values='ret').astype('float32')]).sort_index()
    del extra
    k = panel_y(R2, R2.index, keys)
    k = k[np.isfinite(k.y) & (k.y > -20)]
    frozen = pq.read_table(T.PANEL, columns=['date', 'permno']).to_pandas()
    frozen = frozen[frozen.date.dt.year == 2022]
    a = set(zip(k.date, k.permno.astype(int)))
    b = set(zip(frozen.date, frozen.permno.astype(int)))
    last = max(k.date)
    b_common = {x for x in b if x[0] <= last}
    return {'info': info, 'reconstructed_after_label_filter': len(a), 'panel_2022': len(b),
            'only_reconstructed': len(a - b), 'only_panel': len(b - a),
            'panel_origins_after_last_reconstructed_origin': sorted({str(d.date()) for d, _ in b if d > last}),
            'panel_rows_after_last_reconstructed_origin': len(b - b_common),
            'equal_on_reconstructed_origins': a == b_common,
            'equal': a == b_common,
            'note': 'the 2015-2024 panel kept every week of each year (labels from data to 2025-02); TD4 drops the '
                    'final origins whose five following sessions cross the year end, as the amendment requires for '
                    '2025; on the remaining origins the reconstruction must equal the panel exactly',
            'examples_only_reconstructed': [(str(d.date()), p) for d, p in sorted(a - b)[:5]],
            'examples_only_panel': [(str(d.date()), p) for d, p in sorted(b - a)[:5]]}


def panel2024(**_):
    """2024 keys up to the GARCH filter (the label filter would read 2024 returns, which the build never does)."""
    import td4_sources as S
    keys, info, _, _ = S.universe_keys(2024)
    frozen = pq.read_table(T.PANEL, columns=['date', 'permno']).to_pandas()
    frozen = frozen[frozen.date.dt.year == 2024]
    a = set(zip(keys.date, keys.permno.astype(int)))
    b = set(zip(frozen.date, frozen.permno.astype(int)))
    return {'info': info, 'reconstructed_before_label_filter': len(a), 'panel_2024': len(b),
            'panel_not_reconstructed': len(b - a), 'reconstructed_not_in_panel': len(a - b),
            'equal_up_to_label_filter': len(b - a) == 0,
            'note': 'reconstructed_not_in_panel rows are those the panel dropped by its label filter (y finite and '
                    '> -20); they are not checked here because that needs 2024 future returns'}


def rebuild(year, limit=None):
    """The first `limit` eligible firm-weeks of a period rebuilt by the current td4_inputs code; texts must equal the
    full build's records with the same keys (the full build ran from the same build path)."""
    import td4_inputs as I
    import td3_runtime as R
    n = limit or 150
    tag = R.utc().replace(':', '').replace('-', '')
    out = I.INPUTS / f'rebuild_{year}_{tag}'
    rec = I.run_build((year,), out, limit=n, audit_n=0, tag=f'rebuild_{year}_{tag}')
    full = {}
    for r in R.read_jsonl(out / 'INPUTS.jsonl'):
        full[r['key']] = r['input_sha256']
    built = I.load_inputs(year)
    same = sum(1 for k, h in full.items() if k in built and built[k]['input_sha256'] == h)
    return {'year': year, 'rebuilt': len(full), 'identical_to_full_build': same, 'equal': same == len(full) > 0,
            'counts': rec['counts']}


def rebuild2024(limit=None, **_):
    return rebuild(2024, limit)


def rebuild2025(limit=None, **_):
    return rebuild(2025, limit)


CHECKS = {'graph2024': graph2024, 'links2024': links2024, 'td1snap2024': td1snap2024, 'news2024': news2024,
          'panel2022': panel2022, 'panel2024': panel2024, 'rebuild2024': rebuild2024, 'rebuild2025': rebuild2025}


def _late_checks():
    out = {}
    try:
        import td4_manifest
        out['manifest2024'] = td4_manifest.parity2024
    except ImportError:
        pass
    try:
        import td4_features
        out['features2024'] = td4_features.parity2024
        out['features2024_td4src'] = td4_features.parity2024_td4src
    except ImportError:
        pass
    try:
        import td4_inputs
        out['inputs_pilot'] = td4_inputs.parity_pilot
    except ImportError:
        pass
    try:
        import td4_models
        out['refit2022'] = td4_models.parity2022
    except ImportError:
        pass
    return out


def run(name, limit=None):
    import td3_runtime as R
    checks = {**CHECKS, **_late_checks()}
    if name not in checks:
        raise ValueError(f'Unknown parity check {name}; known: {sorted(checks)}')
    start = time.perf_counter()
    result = checks[name](limit=limit)
    result = {'check': name, 'version': T.VERSION, **result, 'elapsed_seconds': time.perf_counter() - start,
              **R.peak()}
    tag = R.sha_text(json.dumps(result, sort_keys=True, default=str))[:8]
    R.new_json(T.OUT / 'parity' / f'PARITY_{name}_{tag}.json', result)
    return {k: v for k, v in result.items() if k not in ('examples_only_reconstructed', 'examples_only_panel')}
