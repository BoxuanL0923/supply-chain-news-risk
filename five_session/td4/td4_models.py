"""Refits with frozen hyperparameters: the TD1 MLP and the TD2 ridge heads through 2023 (selection) and 2024 (test).

TD1 procedure (fit.py, unchanged in substance): preprocessing (median imputation, scaling, missing indicators) and the
target standardisation are TD1's frozen 2015-2020 statistics from model_frozen_v2.pkl and are not refitted; the
network is MLPRegressor((64, 32), relu, adam, alpha 0.01, batch 4096, lr 0.001, seed 41037) trained by exactly the TD1
selected 14 epochs of partial_fit on every eligible row with origin year <= Y and label end <= Y-12-31, rows in TD1
manifest order (years ascending, file order). model() below is a copy of TD1 fit.model.
TD2 procedure (head_v2.fit_arm refit branch): ridge without intercept on Z scaled by TD2's frozen 2015-2020 training
RMS, residual y - b_in_sample(refit MLP of the same Y), penalty frozen per arm from MODEL_v2.json (main, 1r, 2r, 3r).
parity2022 reproduces TD1's frozen model predictions and TD2's frozen betas with Y = 2022 (development labels only).
refit_all needs clearance A (it reads labels of 2023 and 2024).
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4_gate as G
import td4_labels as L
import td4_manifest as M

MODELS = T.OUT / 'models'
ARMS = ['main', '1r', '2r', '3r']


def model_paths(year_end):
    """(MLP pickle, out-of-sample b for year_end + 1, heads JSON) of the refit through year_end."""
    return (MODELS / f'MLP_THROUGH_{year_end}.pkl', MODELS / f'B_THROUGH_{year_end}_PREDICT_{year_end + 1}.parquet',
            MODELS / f'HEADS_THROUGH_{year_end}.json')


def td1_bundle():
    import td3_runtime as R
    if R.sha(T.TD1 / 'model_frozen_v2.pkl') != T.TD1_MODEL_SHA256:
        raise ValueError('TD1 frozen model changed')
    with (T.TD1 / 'model_frozen_v2.pkl').open('rb') as f:
        return pickle.load(f)


def td2_model():
    import td3_runtime as R
    if R.sha(T.TD2 / 'MODEL_v2.json') != T.TD2_MODEL_SHA256:
        raise ValueError('TD2 frozen model changed')
    return json.loads((T.TD2 / 'MODEL_v2.json').read_text(encoding='utf-8'))


def model(lr, alpha, seed):
    """TD1 fit.model (copy)."""
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=alpha, batch_size=4096,
                        learning_rate_init=lr, random_state=seed, max_iter=1, shuffle=True, early_stopping=False,
                        tol=0.)


def manifest_frame(year, features):
    cols = ['firm_id', 'cutoff', 'origin_session', 'eligible_features'] + features
    return pq.read_table(M.manifest_path(year), columns=cols).to_pandas()


def labels_for(year, keys, year_end, token):
    """Labels for one origin year under a refit through year_end: TD1 development labels where they exist (label end
    <= 2022-12-31), otherwise TD4 labels (token) with allowed end year_end-12-31."""
    allowed_end = f'{year_end}-12-31'
    if year <= 2022:
        dev = L.development(year)[['firm_id', 'cutoff', 'y', 'label_end', 'label_observed']]
        out = keys.merge(dev, on=['firm_id', 'cutoff'], how='left', validate='one_to_one')
        clipped = (~out.label_observed.fillna(False).astype(bool)) & \
            (pd.to_datetime(out.label_end) > pd.Timestamp(T.DEV_SEAL))
        if year_end > 2022 and clipped.any():
            fresh = L.compute(year, keys[clipped.values], allowed_end, token)
            out = out.set_index(['firm_id', 'cutoff'])
            out.update(fresh.set_index(['firm_id', 'cutoff']))
            out = out.reset_index()
        return out
    return L.compute(year, keys, allowed_end, token).merge(keys[['firm_id', 'cutoff']], on=['firm_id', 'cutoff'])


def training_frame(year_end, token=None, td1_flags=False):
    """Rows 2015..year_end in TD1 order with features and labels; td1_flags=True uses TD1's refit_included (2022)."""
    bundle = td1_bundle()
    features = bundle['features']
    chunks = []
    for y in range(T.FIRST_YEAR, year_end + 1):
        E.guard()
        f = manifest_frame(y, features)
        if td1_flags:
            lab = pq.read_table(T.TD1 / f'firm_week_manifest/year_{y}.parquet',
                                columns=['firm_id', 'cutoff', 'y', 'label_end', 'refit_included']).to_pandas()
            f = f.merge(lab, on=['firm_id', 'cutoff'], how='left', validate='one_to_one')
            f['include'] = f.refit_included.fillna(False).astype(bool)
        else:
            lab = labels_for(y, f[['firm_id', 'cutoff', 'origin_session']], year_end, token)
            f = f.merge(lab[['firm_id', 'cutoff', 'y', 'label_end', 'label_observed']], on=['firm_id', 'cutoff'],
                        how='left', validate='one_to_one')
            f['include'] = f.eligible_features.astype(bool) & f.label_observed.fillna(False).astype(bool) & \
                (pd.to_datetime(f.label_end) <= pd.Timestamp(f'{year_end}-12-31'))
        f['year'] = y
        chunks.append(f)
    f = pd.concat(chunks, ignore_index=True)
    if (pd.to_datetime(f.loc[f.include, 'label_end']) > pd.Timestamp(f'{year_end}-12-31')).any():
        raise ValueError('Training label crosses the refit year')
    return f, bundle


def fit_mlp(year_end, token=None, td1_flags=False):
    core = E.td1('core')
    f, bundle = training_frame(year_end, token, td1_flags)
    features, prep, sel = bundle['features'], bundle['preprocess'], bundle['selected']
    seed = E.td1('spec').SEED
    ymean, yscale = bundle['ymean'], bundle['yscale']
    raw = f[features].to_numpy(float)
    x = core.transform(raw, prep)
    ref = f.include.to_numpy(bool)
    target = (f.y.to_numpy(float) - ymean) / yscale
    net = model(sel['lr'], sel['alpha'], seed)
    xr, yr = x[ref], target[ref]
    for _ in range(sel['best_epoch']):
        E.guard()
        net.partial_fit(xr, yr)
    f['b'] = net.predict(x) * yscale + ymean
    f.loc[~f.eligible_features.astype(bool), 'b'] = np.nan
    return net, f, {'year_end': year_end, 'rows': int(ref.sum()), 'epochs': sel['best_epoch'], 'lr': sel['lr'],
                    'alpha': sel['alpha'], 'seed': seed, 'preprocessing': 'TD1 frozen 2015-2020 statistics'}


def predict_year(net, year):
    core = E.td1('core')
    bundle = td1_bundle()
    f = manifest_frame(year, bundle['features'])
    f = f[f.eligible_features].copy()
    f['b'] = net.predict(core.transform(f[bundle['features']].to_numpy(float), bundle['preprocess'])) * \
        bundle['yscale'] + bundle['ymean']
    return f[['firm_id', 'cutoff', 'origin_session', 'b']]


def feature_frame(year, names=T.NAMES, variant='primary'):
    import td4_inputs as I
    cols = ['firm_id', 'cutoff', 'origin_session', 'variant', 'has_news', 'mapped_focal'] + \
        ['missing_' + n for n in names] + ['Z_' + n for n in names] + \
        [f'Z{h}_{n}' for h in T.HOPS for n in names] + \
        [f'H{h}_{c}' for h in T.HOPS for c in ('graph_partners', 'news_partners', 'selected_partners')]
    f = pq.read_table(I.feature_path(year), columns=cols).to_pandas()
    return f[f.variant == variant].drop(columns='variant').reset_index(drop=True)


def prefix(arm):
    return 'Z_' if arm == 'main' else f'Z{str(arm)[0]}_'


def fit_heads(year_end, mlp_frame, td2=None):
    """TD2 refit branch per arm with the frozen penalty and training RMS scale; b = in-sample refit MLP."""
    E.paths()
    from engine import ridge
    td2 = td2 or td2_model()
    chunks = []
    for y in range(T.FIRST_YEAR, year_end + 1):
        z = feature_frame(y)
        m = mlp_frame[mlp_frame.year == y][['firm_id', 'cutoff', 'y', 'b', 'include']]
        chunks.append(z.merge(m, on=['firm_id', 'cutoff'], how='left', validate='one_to_one'))
    f = pd.concat(chunks, ignore_index=True)
    ref = f.include.fillna(False).astype(bool).to_numpy()
    if not np.isfinite(f.loc[ref, 'b']).all():
        raise ValueError('Baseline missing on a refit row')
    heads = {}
    for arm in ARMS:
        h = td2['heads'][arm]
        scale = np.asarray(h['scale_training_rms'], float)
        x = f[[prefix(arm) + n for n in td2['names']]].to_numpy(float) / scale
        res = f.y.to_numpy(float) - f.b.to_numpy(float)
        coef = ridge(x[ref], res[ref], h['penalty'])
        heads[arm] = {'penalty': h['penalty'], 'scale_training_rms': scale.tolist(), 'final_coef_std': coef.tolist(),
                      'beta': (coef / scale).tolist(), 'rows': int(ref.sum())}
    order = sorted(td2['names'], key=lambda n: (-abs(dict(zip(td2['names'], heads['main']['final_coef_std']))[n]), n))
    return {'year_end': year_end, 'names': td2['names'], 'heads': heads, 'beta': heads['main']['beta'],
            'training_beta_order': order,
            'training_beta_order_rule': 'order by |standardised coefficient| (final_coef_std) of this main head'}


def parity2022(**_):
    """Y = 2022 with TD1's refit flags must reproduce TD1's frozen MLP predictions (2015-2024) and TD2's betas."""
    net, f, info = fit_mlp(2022, td1_flags=True)
    out = {'mlp_info': info}
    diffs = {}
    for y in (2022, 2023, 2024):
        mine = predict_year(net, y)
        frozen = pq.read_table(T.TD1 / f'predictions_v2/year_{y}.parquet', columns=['firm_id', 'cutoff', 'b']).to_pandas()
        j = frozen.merge(mine, on=['firm_id', 'cutoff'], how='left', suffixes=('_td1', '_td4'), validate='one_to_one')
        d = np.abs(j.b_td1.to_numpy(float) - j.b_td4.to_numpy(float))
        diffs[str(y)] = {'rows': len(j), 'max_abs_diff': float(np.nanmax(d)), 'exact': bool((d == 0).all())}
    out['mlp_predictions'] = diffs
    # TD2 used TD1's refit b; the head refit is checked with both TD1's b and the TD4 refit b
    td2 = td2_model()
    for label, frame in (('td4_refit_b', f),):
        heads = fit_heads(2022, frame, td2)
        out['heads_' + label] = {arm: {'max_abs_beta_diff': float(np.max(np.abs(np.asarray(heads['heads'][arm]['beta'])
                                                                                - np.asarray(td2['heads'][arm]['beta'])))),
                                       'exact': heads['heads'][arm]['beta'] == td2['heads'][arm]['beta']}
                                 for arm in ARMS}
    out['equal'] = all(v['exact'] for v in diffs.values()) and all(
        v['exact'] for v in out['heads_td4_refit_b'].values())
    return out


def refit_all():
    """Clearance A step: refits through 2023 and 2024, predictions, heads; exclusive files under models/."""
    import td3_runtime as R
    token, freeze = G.admit('refit')
    start = time.perf_counter()
    receipt = {'version': T.VERSION, 'clearance_sha256': token.clearance_sha, 'refits': {}}
    for year_end in (T.TRAIN_END[T.SELECTION_YEAR], T.TRAIN_END[T.TEST_YEAR]):
        target_year = year_end + 1
        out_m, out_b, out_h = model_paths(year_end)
        if out_m.exists():
            raise FileExistsError(out_m)
        net, f, info = fit_mlp(year_end, token=token)
        with E.private(out_m).open('xb') as fh:
            pickle.dump({'mlp': net, 'info': info}, fh, protocol=5)
        pred = predict_year(net, target_year)
        E.write_parquet(out_b, pred)
        heads = fit_heads(year_end, f)
        R.new_json(out_h, heads)
        receipt['refits'][str(year_end)] = {'mlp': info, 'mlp_sha256': R.sha(out_m),
                                            'predictions_sha256': R.sha(out_b), 'heads_sha256': R.sha(out_h),
                                            'head_rows': heads['heads']['main']['rows'],
                                            'training_beta_order': heads['training_beta_order']}
        del net, f
    receipt['elapsed_seconds'] = time.perf_counter() - start
    receipt.update(R.peak())
    R.new_json(MODELS / 'REFIT_RECEIPT_v1.json', receipt)
    return receipt
