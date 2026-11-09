"""Scoring frames shared by the 2024 selection and the 2025 evaluation (identical rows and values for every arm).

frame(): TD2-style feature rows of the year (primary variant) joined one-to-one with the MLP baseline b of the
refit through year-1 and the labels; rows without a finite label, with y <= -20 (panel rule) or without b are dropped
for every arm alike. provider_outputs(): the final parsed output of each provider arm per row (valid, invalid,
no_output, not_called), aligned with the rows; nothing is backfilled.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4_gates as Q

ZCOLS = {arm: [('Z_' if arm == 'main' else f'Z{str(arm)[0]}_') + n for n in T.NAMES] for arm in T.HOP_ARMS}


def frame(year, labels, b):
    """labels: firm_id, cutoff, y, label_end, label_observed; b: firm_id, cutoff, b."""
    import td4_models as MD
    f = MD.feature_frame(year)
    n0 = len(f)
    f = f.merge(b[['firm_id', 'cutoff', 'b']], on=['firm_id', 'cutoff'], how='left', validate='one_to_one')
    f = f.merge(labels[['firm_id', 'cutoff', 'y', 'label_end', 'label_observed']], on=['firm_id', 'cutoff'],
                how='left', validate='one_to_one')
    if len(f) != n0:
        raise ValueError('Row alignment changed')
    keep = f.label_observed.fillna(False).astype(bool) & np.isfinite(f.y.astype(float)) & \
        (f.y.astype(float) > -20.0) & np.isfinite(f.b.astype(float))
    if (pd.to_datetime(f.loc[keep, 'label_end']) > pd.Timestamp(f'{year}-12-31')).any():
        raise ValueError('Label window crosses the year end')
    info = {'feature_rows': int(n0), 'scored_rows': int(keep.sum()),
            'dropped_no_label_or_floor': int((~(f.label_observed.fillna(False).astype(bool) &
                                                np.isfinite(f.y.astype(float)) & (f.y.astype(float) > -20.0))).sum()),
            'dropped_no_baseline': int((~np.isfinite(f.b.astype(float))).sum())}
    return f[keep].reset_index(drop=True), info


def available(f):
    return ~f[['missing_' + n for n in T.NAMES]].to_numpy(bool)


def provider_outputs(year, f, collected, eligible_map):
    """collected: td4_runner.collect rows (pass 'main'); eligible_map: firm_id, cutoff -> key (inputs built) from the
    period ELIGIBLE.parquet. Returns {arm: {'ordering','choice','yes_scores','valid','status'}}."""
    by = {}
    for r in collected:
        if r['pass'] == 'main':
            by[(r['key'], r['selector'])] = r
    keys = [eligible_map.get((a, pd.Timestamp(c))) for a, c in zip(f.firm_id, f.cutoff)]
    out = {}
    for arm in T.PROVIDER_ARMS:
        orders, choices, scores, valid, status = [], [], [], [], []
        for k in keys:
            r = by.get((k, arm)) if k is not None else None
            if k is None:
                st = 'not_called'
            elif r is None:
                st = 'no_output'
            elif not r['http_ok']:
                st = 'no_output'
            else:
                st = r['status']
            ok = st == 'valid'
            orders.append(r['ordering'] if ok and arm == 'deepseek_order' else None)
            choices.append(r['choice'] if ok and arm == 'jev_choice' else None)
            scores.append(r['yes_scores'] if ok and arm == 'jev_noul' else None)
            valid.append(ok)
            status.append(st)
        out[arm] = {'ordering': orders, 'choice': choices, 'yes_scores': scores, 'valid': np.array(valid, bool),
                    'status': status}
    return out


def eligible_map(year):
    import td4_inputs as I
    e = pq.read_table(I.INPUTS / f'period_{year}' / 'ELIGIBLE.parquet').to_pandas()
    rp = I.repair_dir(year) / 'ELIGIBLE.parquet'
    if rp.exists():
        e = pd.concat([e, pq.read_table(rp).to_pandas()], ignore_index=True)
    e = e[e.status == 'input_built']
    if e.key.duplicated().any():
        raise ValueError('Duplicate input key across main and repair builds')
    bad = I.withheld_population(year)
    return {(a, pd.Timestamp(c)): k for a, c, k in zip(e.firm_id, e.cutoff, e.key) if k not in bad}


def coverage(f, outs):
    news = f.has_news.to_numpy(bool)
    res = {'rows': int(len(f)), 'rows_with_news': int(news.sum())}
    for arm, o in outs.items():
        st = pd.Series(o['status'])
        res[arm] = {'status_counts_rows_with_news': st[news].value_counts().to_dict(),
                    'valid_share_of_rows_with_news': float(np.mean(o['valid'][news])) if news.any() else None}
    return res


def mse(y, pred):
    return float(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2))


def local_ranks(f, kind, seed, order):
    return Q._gv().rank_matrix(f, T.NAMES, kind, seed, order if kind == 'training_beta' else None)
