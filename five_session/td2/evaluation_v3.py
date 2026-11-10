"""Claude-only, once-only, hash-bound 2023-2024 TD2 evaluation (exploratory historical reproduction).

Importing this module reads nothing. run() first requires
D:/OptResearch_data/five_session/clearance/TD2_CLEARANCE_v1.json; until it exists and every bound
hash matches, no return, label, feature or model payload is opened.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from runtime_v3 import OUT, TD1, CODE, guard, sha, new_json, peak
from spec_v2 import VERSION, EVAL_YEARS, MAX_LABEL_END

BINDINGS = ['version', 'code_sha256', 'artifact_sha256', 'td1_sha256', 'model_sha256', 'qualification_sha256']

@dataclass(frozen=True)
class Paths:
    out: Path
    td1: Path
    code: Path
    clearance: Path

REAL = Paths(OUT, TD1, CODE, TD1 / 'clearance/TD2_CLEARANCE_v1.json')

def validate_clearance(clearance, freeze, freeze_hash):
    if clearance.get('schema') != 'TD2_CLEARANCE_v1' or clearance.get('status') != 'CLEARED' or 'request_only' in clearance:
        raise PermissionError('Clearance schema/status (a request is never a clearance)')
    if freeze.get('schema') != 'TD2_FREEZE_v2':
        raise PermissionError('Freeze schema')
    if clearance.get('freeze_sha256') != freeze_hash:
        raise PermissionError('Freeze hash mismatch')
    for key in BINDINGS:
        if key not in freeze or clearance.get(key) != freeze[key]:
            raise PermissionError('Clearance binding ' + key)
    for key, value in [('evaluation_years', list(EVAL_YEARS)), ('maximum_label_end', MAX_LABEL_END)]:
        if clearance.get(key) != value or freeze.get(key) != value:
            raise PermissionError('Scope mismatch ' + key)
    if freeze.get('version') != VERSION:
        raise PermissionError('Version mismatch')

def admit(paths=REAL):
    guard()
    if not Path(paths.clearance).is_file():
        raise PermissionError('STOP: TD2_CLEARANCE_v1.json absent; no evaluation label read or computed')
    started = Path(paths.out) / 'EVALUATION_STARTED_v2.json'; result = Path(paths.out) / 'EVALUATION_RESULT_v2.json'
    if started.exists() or result.exists():
        raise PermissionError('Evaluation once-only latch exists; no replay')
    clearance = json.loads(Path(paths.clearance).read_text(encoding='utf-8'))
    fpath = Path(paths.out) / 'FREEZE_v2.json'
    freeze = json.loads(fpath.read_text(encoding='utf-8'))
    validate_clearance(clearance, freeze, sha(fpath))
    if {str(p) for p in Path(paths.code).glob('*.py')} != set(freeze['code_sha256']):
        raise PermissionError('Code inventory changed')
    for field, root in [('code_sha256', paths.code), ('artifact_sha256', paths.out), ('td1_sha256', paths.td1),
                        ('qualification_sha256', paths.out)]:
        for p, h in freeze[field].items():
            resolved = Path(p).resolve()
            if not resolved.is_relative_to(Path(root).resolve()) or sha(resolved) != h:
                raise PermissionError('Hash mismatch ' + p)
    if sha(Path(paths.out) / 'MODEL_v2.json') != freeze['model_sha256']:
        raise PermissionError('Head changed')
    new_json(started, {'freeze_sha256': sha(fpath), 'clearance_sha256': sha(paths.clearance), 'version': VERSION,
                       'exploratory': True})
    import io_v2
    io_v2._evaluation_token = object()
    return io_v2._evaluation_token

def label(dates, returns, pos, token):
    """TD1 rule: log(mean of the next five squared daily total returns); end <=2024-12-31."""
    import io_v2
    if token is None or token is not io_v2._evaluation_token:
        raise PermissionError('Label admission required')
    import numpy as np
    import pandas as pd
    if pos < 0 or pos + 5 >= len(dates):
        return None, None
    end = pd.Timestamp(dates[pos + 5])
    if end > pd.Timestamp(MAX_LABEL_END):
        return None, None  # the return fields of a clipped window are never touched
    w = np.asarray(returns[pos + 1:pos + 6], float)
    if len(w) != 5 or not np.isfinite(w).all() or np.mean(w * w) <= 0:
        return None, None
    return float(np.log(np.mean(w * w))), end

def run(paths=REAL):
    token = admit(paths)  # First action: nothing below runs without a matching clearance.
    import numpy as np
    import pandas as pd
    import io_v2 as io
    import gates_v2 as gv
    from head_v2 import checked_join, prefix
    from statistics import paired, clustered_mean
    from spec_v2 import K_GRID, SELECTORS, HOPS, HOP_ARMS
    start = time.perf_counter(); out = Path(paths.out)
    model = io.json_read(out / 'MODEL_v2.json'); names = model['names']
    cols = ['firm_id', 'cutoff', 'variant', 'has_news', 'mapped_focal'] + ['missing_' + n for n in names]
    cols += ['Z_' + n for n in names] + [f'Z{h}_{n}' for h in HOPS for n in names]
    cols += [f'H{h}_{c}' for h in HOPS for c in ('news_partners', 'selected_partners')]
    frames = []; coverage = []
    for year in EVAL_YEARS:
        guard()
        r = io.admitted_returns(paths.td1, year, token)
        r = r[r.share_type.isin([0, 2])].copy(); r['date'] = pd.to_datetime(r.date)
        if r.duplicated(['permno', 'date']).any():
            raise ValueError('Duplicate return session')
        dates = pd.DatetimeIndex(sorted(r.date.unique()))
        wide = r.pivot(index='date', columns='permno', values='ret').reindex(dates)
        mapper = {hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]: p for p in wide.columns}
        f = io.read_features(year, cols, out / 'features_v2')
        base = io.baseline_frame(paths.td1, year)[['firm_id', 'cutoff', 'b']]
        keys = f.loc[f.variant == 'primary', ['firm_id', 'cutoff', 'origin_session']]
        rows = []
        for firm, g in keys.groupby('firm_id', sort=True):
            p = mapper.get(firm)
            if p is None:
                continue
            ret = wide[p].to_numpy(float)
            for c, o in zip(g.cutoff, g.origin_session):
                y, end = label(dates, ret, dates.get_indexer([pd.Timestamp(o)])[0], token)
                if y is not None:
                    rows.append({'firm_id': firm, 'cutoff': c, 'y': y, 'label_end': end})
        labels = pd.DataFrame(rows, columns=['firm_id', 'cutoff', 'y', 'label_end'])
        for variant in ('primary', 'fsb5_sensitivity'):
            g = checked_join(f[f.variant == variant], base)
            g = checked_join(g, labels); g = g[g.y.notna()].copy()
            if g.b.isna().any():
                raise ValueError('Frozen MLP prediction missing on a paired row')
            frames.append(g)
            coverage.append({'year': year, 'variant': variant, 'feature_rows': int((f.variant == variant).sum()),
                'paired_rows': int(len(g)), 'rows_without_label': int((f.variant == variant).sum() - len(g)),
                'paired_rows_with_news': int(g.has_news.sum()), 'mapped_focal': int(g.mapped_focal.sum()),
                'missing_characteristics': {n: int(g['missing_' + n].sum()) for n in names},
                'hop_news_rows': {str(h): int((g[f'H{h}_news_partners'] > 0).sum()) for h in HOPS},
                'hop_selected_rows': {str(h): int((g[f'H{h}_selected_partners'] > 0).sum()) for h in HOPS}})
    f = pd.concat(frames, ignore_index=True)
    if (pd.to_datetime(f.label_end) > pd.Timestamp(MAX_LABEL_END)).any():
        raise ValueError('Label date seal')
    arms = []; increments = []
    for variant, g in f.groupby('variant', sort=True):
        g = g.reset_index(drop=True)
        available = ~g[['missing_' + n for n in names]].to_numpy(bool); news = g.has_news.to_numpy(bool)
        y = g.y.to_numpy(float); b = g.b.to_numpy(float)
        z = {arm: g[[prefix(arm) + n for n in names]].to_numpy(float) for arm in HOP_ARMS}
        beta = {arm: (model['beta'] if arm in ('main', 1, 2, 3) else model['heads'][arm]['beta']) for arm in HOP_ARMS}
        for kind, seed in SELECTORS:
            key = f'{kind}:{seed}'; chosen = model['selected_local_gates'][key]
            ranks = gv.rank_matrix(g, names, kind, seed, model['training_beta_order'] if kind == 'training_beta' else None)
            se = {'firm_id': g.firm_id, 'cutoff': g.cutoff, 'origin_session': g.origin_session,
                  'baseline_squared_error': (y - b) ** 2}
            for k in K_GRID:
                q = gv.gates(ranks, available, news, k, chosen['fallback']); preds = {}
                for arm in HOP_ARMS:
                    pred = gv.predict(b, z[arm], q, beta[arm]); preds[arm] = pred
                    arms.append(dict(variant=variant, selector=kind, seed=seed, K=k, arm=str(arm),
                                     fallback=chosen['fallback'], selected_K_on_2021_2022=(k == chosen['K']),
                                     **paired(y, b, pred, g.firm_id, g.origin_session)))
                    se[f'se_K{k}_{arm}'] = (y - pred) ** 2
                for family, (a1, a2, a3) in [('frozen_main_beta', (1, 2, 3)), ('per_hop_refit', ('1r', '2r', '3r'))]:
                    for hi, lo, name in [(a2, a1, 'H2-H1'), (a3, a2, 'H3-H2')]:
                        gain = (y - preds[lo]) ** 2 - (y - preds[hi]) ** 2
                        increments.append(dict(variant=variant, selector=kind, seed=seed, K=k, family=family,
                            contrast=name, positive='lower squared error for the higher exact hop',
                            **clustered_mean(gain, g.firm_id, g.origin_session)))
            io.write_frame(out / f'evaluation_v2/squared_errors_{variant}_{kind}_{seed}.parquet', pd.DataFrame(se))
    new_json(out / 'EVALUATION_RESULT_v2.json', {'version': VERSION, 'exploratory': True,
        'label': 'historical reproduction on 2023-2024, already examined in the existing risk study; not a new test',
        'baseline': 'frozen TD1 MLP b (predictions_v2)', 'arms': arms, 'hop_incremental_gains': increments,
        'coverage': coverage, 'provider_selectors': 'empty slots; not evaluated',
        'elapsed_seconds': time.perf_counter() - start, **peak(), 'free_bytes': guard()})
    print('EVALUATION_RESULT_V2', str(out / 'EVALUATION_RESULT_v2.json'), flush=True)
