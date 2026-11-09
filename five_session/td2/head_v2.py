"""Shared five-session ridge head, local selector harness and hop ablation heads (development only).

Fit 2015-2020, select penalty/K/fallback on 2021-2022, refit through 2022, freeze. No row
with a label end after 2022-12-31 is read; no 2023-2024 label exists in any input here.
"""
import time
import numpy as np
import pandas as pd
import io_v2 as io
import gates_v2 as gv
from engine import ridge
from selectors import validate_slot
from runtime_v3 import OUT, guard, new_json, peak, sha
from spec_v2 import (NAMES, HOPS, K_GRID, SELECTORS, FALLBACKS, RIDGE_GRID, SLOT_SCHEMAS, VERSION,
                     AMENDMENTS, RULES)

def checked_join(frame, other, keys=('firm_id', 'cutoff')):
    """Left join that refuses duplicate keys and any change in the row count."""
    n = len(frame)
    out = frame.merge(other, on=list(keys), how='left', validate='one_to_one', suffixes=('', '_dup'))
    if len(out) != n:
        raise ValueError('Row alignment changed')
    return out

def prefix(arm):
    if arm == 'main':
        return 'Z_'
    return f'Z{str(arm)[0]}_'

def dev_columns(names):
    cols = ['firm_id', 'cutoff', 'variant', 'has_news'] + ['missing_' + n for n in names]
    cols += ['Z_' + n for n in names] + [f'Z{h}_{n}' for h in HOPS for n in names]
    return cols

def load_development(names, years=range(2015, 2023), feature_root=None):
    years = list(years)
    if any(not 2015 <= y <= 2022 for y in years):
        raise PermissionError('Development years only')  # checked before any payload is opened
    model = io.dec.td1_model(); chunks = []
    for year in years:
        guard()
        f = io.read_features(year, dev_columns(names), feature_root)
        f = f[f.variant == 'primary'].drop(columns='variant')
        labels = io.manifest_labels(year)[['firm_id', 'cutoff'] + io.LABEL_COLUMNS]
        f = checked_join(f, labels)
        f = checked_join(f, io.dec.development_baseline(year, model))
        f = checked_join(f, io.dec.baseline(year)[['firm_id', 'cutoff', 'b']])
        chunks.append(f)
    return pd.concat(chunks, ignore_index=True)

def masks(f):
    tr = f.fit_included.to_numpy(bool); va = f.selection_included.to_numpy(bool)
    ref = f.refit_included.to_numpy(bool)
    if not tr.any() or not va.any() or not ref.any():
        raise ValueError('Empty development partition')
    for m in (tr, va, ref):
        if not np.isfinite(f.loc[m, 'y'].to_numpy(float)).all():
            raise ValueError('Nonfinite development label')
    ends = pd.to_datetime(f.label_end)
    if (ends[tr] > pd.Timestamp('2020-12-31')).any(): raise ValueError('Train boundary')
    if (ends[va] > pd.Timestamp('2022-12-31')).any(): raise ValueError('Selection boundary')
    if (ends[ref] > pd.Timestamp('2022-12-31')).any(): raise ValueError('Refit boundary')
    if (pd.to_datetime(f.origin_session[va]).dt.year < 2021).any(): raise ValueError('Selection years')
    return tr, va, ref

def fit_arm(z, y, b_dev, b_ref, tr, va, ref, grid=RIDGE_GRID):
    """Ridge on RMS-scaled Z (scale from training rows); penalty by selection MSE; refit on ref rows."""
    z = np.asarray(z, float); y = np.asarray(y, float)
    scale = np.sqrt(np.mean(z[tr] ** 2, axis=0)); scale[scale == 0] = 1.0
    x = z / scale; res = y - np.asarray(b_dev, float)
    table = []; best = None
    for lam in grid:
        coef = ridge(x[tr], res[tr], lam)
        mse = float(np.mean((res[va] - x[va] @ coef) ** 2))
        table.append({'penalty': lam, 'selection_mse': mse})
        if best is None or mse < best[0]:
            best = (mse, lam, coef.copy())
    final_std = ridge(x[ref], y[ref] - np.asarray(b_ref, float)[ref], best[1])
    return {'scale_training_rms': scale.tolist(), 'penalty': best[1], 'selection_mse': best[0],
            'training_coef_std': best[2].tolist(), 'training_beta': (best[2] / scale).tolist(),
            'final_coef_std': final_std.tolist(), 'beta': (final_std / scale).tolist(), 'ridge_grid': table,
            'baseline_selection_mse': float(np.mean(res[va] ** 2))}

def ranks_for(frame, names, kind, seed, order):
    return gv.rank_matrix(frame, names, kind, seed, order if kind == 'training_beta' else None)

def selection_sweep(v, names, heads, order):
    """K/fallback grid on 2021-2022 rows with the training heads and TD1 selection MLP."""
    available = ~v[['missing_' + n for n in names]].to_numpy(bool)
    y = v.y.to_numpy(float); b = v.b_development.to_numpy(float); news = v.has_news.to_numpy(bool)
    grid = []; chosen = {}; ablation = []
    for kind, seed in SELECTORS:
        ranks = ranks_for(v, names, kind, seed, order); key = f'{kind}:{seed}'
        for k in K_GRID:
            for fb in FALLBACKS:
                q = gv.gates(ranks, available, news, k, fb)
                pred = gv.predict(b, v[[prefix('main') + n for n in names]].to_numpy(float), q, heads['main']['training_beta'])
                rec = {'selector': kind, 'seed': seed, 'K': k, 'fallback': fb,
                       'selection_mse': float(np.mean((y - pred) ** 2))}
                grid.append(rec)
                if key not in chosen or rec['selection_mse'] < chosen[key]['selection_mse']:
                    chosen[key] = rec
        # Development-only ablation table at the chosen budget (information; nothing is selected on it).
        c = chosen[key]; q = gv.gates(ranks, available, news, c['K'], c['fallback'])
        for arm in ['main', 1, 2, 3, '1r', '2r', '3r']:
            beta = heads['main']['training_beta'] if arm in (1, 2, 3, 'main') else heads[arm]['training_beta']
            pred = gv.predict(b, v[[prefix(arm) + n for n in names]].to_numpy(float), q, beta)
            ablation.append({'selector': kind, 'seed': seed, 'K': c['K'], 'arm': str(arm),
                             'selection_mse': float(np.mean((y - pred) ** 2))})
    return grid, chosen, ablation

def fit(names=None):
    start = time.perf_counter(); guard(); names = list(NAMES if names is None else names)
    for p in (OUT / 'MODEL_v2.json', OUT / 'SELECTION_v2.json'):
        if p.exists():
            raise FileExistsError(p)
    for year in range(2015, 2025):
        if not (io.COVERAGE / f'year_{year}.json').exists():
            raise PermissionError(f'Feature year {year} incomplete')
    f = load_development(names)
    tr, va, ref = masks(f)
    y = f.y.to_numpy(float); bdev = f.b_development.to_numpy(float); bref = f.b.to_numpy(float)
    heads = {}
    for arm in ['main', '1r', '2r', '3r']:
        heads[arm] = fit_arm(f[[prefix(arm) + n for n in names]].to_numpy(float), y, bdev, bref, tr, va, ref)
    order = gv.beta_order(names, heads['main']['training_coef_std'])
    v = f.loc[va].reset_index(drop=True)
    grid, chosen, ablation = selection_sweep(v, names, heads, order)
    slots = {}
    for kind in SLOT_SCHEMAS:
        slot = io.json_read(OUT / f'selector_slots/{kind}_v1.json'); validate_slot(slot, kind, names)
        slots[kind] = {'path': str(OUT / f'selector_slots/{kind}_v1.json'),
                       'sha256': sha(OUT / f'selector_slots/{kind}_v1.json'),
                       'status': 'EMPTY_PENDING_SELECTION', 'K': None, 'threshold': None, 'fallback': None,
                       'records': len(slot['records'])}
    counts = {'fit_rows': int(tr.sum()), 'selection_rows': int(va.sum()), 'refit_rows': int(ref.sum()),
              'development_rows': int(len(f))}
    new_json(OUT / 'SELECTION_v2.json', {'version': VERSION, 'selection_period': '2021-2022 only',
        'ridge_grids': {a: h['ridge_grid'] for a, h in heads.items()}, 'selector_grid': grid,
        'chosen_local_gates': chosen, 'development_ablation_at_chosen_K': ablation,
        'threshold_grid_for_future_noul': [.25, .5, .75],
        'provider_selection_status': 'No provider records; no provider K, threshold or fallback selected',
        **counts, 'elapsed_seconds': time.perf_counter() - start, **peak()})
    bundle = {'schema': 'TD2_model_v2', 'version': VERSION, 'names': names, 'exploratory': True,
        'q_during_fitting': 1, 'intercept': 0, 'heads': heads, 'beta': heads['main']['beta'],
        'training_beta_order': order, 'selected_local_gates': chosen, 'provider_gates': slots,
        'hop_arms': {'main': 'Z_ with main beta', '1': 'Z1_ with main beta', '2': 'Z2_ with main beta',
                     '3': 'Z3_ with main beta', '1r': 'Z1_ with H1 refit head', '2r': 'Z2_ with H2 refit head',
                     '3r': 'Z3_ with H3 refit head'},
        'baseline_development': 'TD1 selection_mlp (fit 2015-2020) for training and selection',
        'baseline_refit_and_evaluation': 'TD1 refitted b from predictions_v2',
        'amendments': AMENDMENTS, 'rules': RULES, **counts, 'evaluation_run': False}
    new_json(OUT / 'MODEL_v2.json', bundle)
    print('MODEL_V2', str(OUT / 'MODEL_v2.json'), 'penalty', heads['main']['penalty'],
          'beta', heads['main']['beta'], 'order', order, flush=True)
