"""Scoring frames for the v2 selection (2021-2022 subsample) and the v2 test (2023, 2024, 2025).

Selection rows: TD2 features_v2 primary rows of the 20,000 subsample firm-weeks; b = TD1 selection_mlp prediction
(fit 2015-2020; TD2 decoder_v3.development_baseline formula); heads = TD2 2015-2020 training heads (training_beta);
training-|beta| order = TD2 training_beta_order. Test rows: every TD2-style feature row (primary) of the year; b = TD1
predictions_v2 (2023, 2024; the frozen refitted network) or the same frozen network on the 2025 manifest; heads = TD2
MODEL_v2 beta (refit through 2022). Rows without a finite label, with y <= -20, or without b are dropped for every arm.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4v2_spec as S


def td2_model():
    import td4_models as MD
    return MD.td2_model()


def selection_heads():
    m = td2_model()
    return {'beta': m['heads']['main']['training_beta'], 'training_beta_order': m['training_beta_order'],
            'heads': {a: {'beta': m['heads'][a]['training_beta']} for a in ('1r', '2r', '3r')},
            'source': 'TD2 MODEL_v2.json heads[*].training_beta (2015-2020 fit, penalty chosen on 2021-2022)'}


def test_heads():
    m = td2_model()
    return {'beta': m['beta'], 'training_beta_order': m['training_beta_order'],
            'heads': {a: {'beta': m['heads'][a]['beta']} for a in ('1r', '2r', '3r')},
            'source': 'TD2 MODEL_v2.json beta and heads[*].beta (refit through 2022)'}


def selection_b(years):
    """TD2 decoder_v3.development_baseline: TD1 selection_mlp on the manifest rows (no label read)."""
    import td4_models as MD
    bundle = MD.td1_bundle()
    core = E.td1('core')
    out = []
    for y in years:
        f = MD.manifest_frame(y, bundle['features'])
        x = core.transform(f[bundle['features']].to_numpy(float), bundle['preprocess'])
        f['b'] = bundle['selection_mlp'].predict(x) * bundle['yscale'] + bundle['ymean']
        out.append(f[['firm_id', 'cutoff', 'b']])
    return pd.concat(out, ignore_index=True)


def test_b(year):
    import td4_models as MD
    p = T.TD1 / f'predictions_v2/year_{year}.parquet'
    if p.exists():
        return pq.read_table(p, columns=['firm_id', 'cutoff', 'b']).to_pandas()
    bundle = MD.td1_bundle()
    return MD.predict_year(bundle['mlp'], year)[['firm_id', 'cutoff', 'b']]


def _join(f, b, labels, bound):
    n0 = len(f)
    f = f.merge(b, on=['firm_id', 'cutoff'], how='left', validate='one_to_one')
    f = f.merge(labels[['firm_id', 'cutoff', 'y', 'label_end', 'label_observed']], on=['firm_id', 'cutoff'],
                how='left', validate='one_to_one')
    if len(f) != n0:
        raise ValueError('Row alignment changed')
    ok_label = f.label_observed.fillna(False).astype(bool) & np.isfinite(f.y.astype(float)) & \
        (f.y.astype(float) > -20.0)
    keep = ok_label & np.isfinite(f.b.astype(float))
    if (pd.to_datetime(f.loc[keep, 'label_end']) > pd.Timestamp(bound)).any():
        raise ValueError('Label window beyond the bound')
    info = {'feature_rows': int(n0), 'scored_rows': int(keep.sum()), 'dropped_no_label_or_floor': int((~ok_label).sum()),
            'dropped_no_baseline': int((ok_label & ~np.isfinite(f.b.astype(float))).sum())}
    return f[keep].reset_index(drop=True), info


def selection_frame(sub_pairs, labels):
    """sub_pairs: set of (firm_id, cutoff Timestamp) of the subsample."""
    import td4_models as MD
    f = pd.concat([MD.feature_frame(y) for y in (2021, 2022)], ignore_index=True)
    f = f[[(a, pd.Timestamp(c)) in sub_pairs for a, c in zip(f.firm_id, f.cutoff)]].reset_index(drop=True)
    return _join(f, selection_b((2021, 2022)), labels, S.SELECT_LABEL_END)


def test_frame(year, labels):
    import td4_models as MD
    f = MD.feature_frame(year)
    return _join(f, test_b(year), labels, S.LABEL_END_BOUND)


def concat_outputs(parts):
    """Concatenate per-year provider-output dicts (td4_score.provider_outputs) in row order."""
    out = {}
    for arm in T.PROVIDER_ARMS:
        out[arm] = {k: [] for k in ('ordering', 'choice', 'yes_scores', 'status')}
        out[arm]['valid'] = []
        for p in parts:
            for k in ('ordering', 'choice', 'yes_scores', 'status'):
                out[arm][k] += list(p[arm][k])
            out[arm]['valid'] += list(p[arm]['valid'])
        out[arm]['valid'] = np.array(out[arm]['valid'], bool)
    return out
