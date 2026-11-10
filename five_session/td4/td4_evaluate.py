"""Once-only 2025 evaluation (clearance B). CONFIRMATORY for this pipeline, with the pre-registered caveats.

admit_b() runs first (clearance A still valid, clearance B present and matching the selection freeze, the
pre-registration, the 2025 ledger digest and the through-2024 models) and writes the once-only latch before any 2025
label is computed. Then: scored rows (identical for every arm), the frozen 2024 selections applied to the 2025
provider outputs, the heads refitted through 2024 (one beta for every arm), the MLP refitted through 2024 as the
benchmark; P1 and P2 as pre-registered; descriptive arms, references, coverage, cost, H1/H2/H3 ablations and
increments with the same gates.
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4_gate as G
import td4_gates as Q
import td4_score as SC

EV = T.OUT / 'evaluation'


def confirmed(stat):
    iv = stat.get('normal_95_interval')
    return bool(stat.get('uncertainty_status') == 'SUPPORTED' and iv is not None and iv[0] > 0)


def gates_from_config(cfg, outs, avail, news, order, f):
    if cfg['arm'] == 'all_on':
        return Q.all_on_gate(avail)
    return Q.gate_for(cfg['arm'], cfg['rule'], cfg['param'], cfg['fallback'], outs[cfg['arm']], avail)[0]


def evaluate_core(f, outs, heads, sel, costs=None):
    """Pure evaluation on scored rows with the frozen selection. Returns the result dict (P1, P2, descriptive)."""
    st = E.td2_statistics()
    y, b = f.y.to_numpy(float), f.b.to_numpy(float)
    firms, dates = f.firm_id.to_numpy(), f.origin_session.to_numpy()
    avail, news = SC.available(f), f.has_news.to_numpy(bool)
    z = {arm: f[SC.ZCOLS[arm]].to_numpy(float) for arm in T.HOP_ARMS}
    beta = {arm: (heads['beta'] if arm in ('main', 1, 2, 3) else heads['heads'][arm]['beta']) for arm in T.HOP_ARMS}
    order = sel['training_beta_order_frozen']
    arms = {}
    configs = dict(sel['chosen_provider_configs'])
    configs['all_on'] = sel['all_on']
    q = {name: gates_from_config(cfg, outs, avail, news, order, f) for name, cfg in configs.items()}
    pred = {name: Q.predict(b, z['main'], qq, beta['main']) for name, qq in q.items()}
    for name in configs:
        arms[name] = {'config': configs[name], **st.paired(y, b, pred[name], firms, dates)}
    best, jev, ds = sel['best_pipeline_arm'], sel['better_jev_arm'], 'deepseek_order'
    p1 = st.paired(y, b, pred[best], firms, dates)
    gain2 = (y - pred[ds]) ** 2 - (y - pred[jev]) ** 2
    p2 = st.clustered_mean(gain2, firms, dates)
    p2.update(deepseek_mse=float(np.mean((y - pred[ds]) ** 2)), jev_mse=float(np.mean((y - pred[jev]) ** 2)),
              sign='positive gain means lower squared error for the Jev arm than for matched-input DeepSeek')
    jev_vs_ds = {}
    for arm in ('jev_choice', 'jev_noul'):
        g = (y - pred[ds]) ** 2 - (y - pred[arm]) ** 2
        jev_vs_ds[arm] = {'descriptive': arm != jev, **st.clustered_mean(g, firms, dates)}
    pre = T.PREREGISTRATION['stop_rule_readings']
    readings = {'P1': pre['P1_confirmed'] if confirmed(p1) else pre['P1_not_confirmed'],
                'P2': 'Jev improves on matched-input DeepSeek in this test (interval excludes zero).' if confirmed(p2)
                else pre['P2_not_confirmed']}
    refs = {}
    for kind, seed in T.LOCAL_SELECTORS:
        c = sel['reference_configs'][f'{kind}:{seed}']
        ranks = SC.local_ranks(f, kind, seed, order)
        qq = Q._gv().gates(ranks, avail, news, c['K'], c['fallback'])
        refs[f'{kind}:{seed}'] = {'config': c, **st.paired(y, b, Q.predict(b, z['main'], qq, beta['main']), firms, dates)}
    ablations, increments = [], []
    for name in list(T.PROVIDER_ARMS) + ['all_on']:
        hp = {}
        for arm in T.HOP_ARMS:
            hp[arm] = Q.predict(b, z[arm], q[name], beta[arm])
            ablations.append({'selector_arm': name, 'hop_arm': str(arm), **st.paired(y, b, hp[arm], firms, dates)})
        for family, (a1, a2, a3) in (('frozen_main_beta', (1, 2, 3)), ('per_hop_refit', ('1r', '2r', '3r'))):
            for hi, lo, label in ((a2, a1, 'H2-H1'), (a3, a2, 'H3-H2')):
                g = (y - hp[lo]) ** 2 - (y - hp[hi]) ** 2
                increments.append({'selector_arm': name, 'family': family, 'contrast': label,
                                   'positive': 'lower squared error for the higher exact hop',
                                   **st.clustered_mean(g, firms, dates)})
    descriptive_grid = []
    for arm in T.PROVIDER_ARMS:
        for rule, param, fb in Q.configs(arm):
            qq, _ = Q.gate_for(arm, rule, param, fb, outs[arm], avail)
            descriptive_grid.append({'arm': arm, 'rule': rule, 'param': param, 'fallback': fb,
                                     'mse_test': SC.mse(y, Q.predict(b, z['main'], qq, beta['main'])),
                                     'selected_on_selection_year': (rule, param, fb) == (configs[arm]['rule'],
                                                                              configs[arm]['param'],
                                                                              configs[arm]['fallback'])})
    missing = {n: int(f['missing_' + n].sum()) for n in T.NAMES}
    hop_cov = {str(h): {'rows_with_graph_partner': int((f[f'H{h}_graph_partners'] > 0).sum()),
                        'rows_with_news_partner': int((f[f'H{h}_news_partners'] > 0).sum()),
                        'rows_with_selected_partner': int((f[f'H{h}_selected_partners'] > 0).sum())} for h in T.HOPS}
    return {'P1': {'comparison': 'best pipeline arm (chosen on 2024) vs firm-only MLP', 'arm': best,
                   'confirmed': confirmed(p1), **p1},
            'P2': {'comparison': 'better Jev arm (chosen on 2024) vs matched-input DeepSeek, same head', 'jev_arm': jev,
                   'confirmed': confirmed(p2), **p2},
            'stop_rule_readings': readings, 'jev_vs_matched_deepseek': jev_vs_ds,
            'claims': ('No causal three-hop channel is inferred from a path or from a predictive gain; graph paths are '
                       'reported business links, not verified physical product flows (physical_channel_verified is '
                       'false and intermediate product evidence is unavailable in the sources).'),
            'arms': arms, 'references': refs, 'hop_ablations': ablations,
            'hop_increments': increments, 'descriptive_grid_test': descriptive_grid,
            'missingness': {'rows': int(len(f)), 'missing_characteristics': missing, 'hop_coverage': hop_cov},
            'coverage': SC.coverage(f, outs), 'cost': costs}


def actual_costs(year):
    import td4_runner as RU
    import td4_inputs as I
    comp = RU.completion(year, I.callable_inputs(year))
    return {'usd_by_provider_including_retries_and_reasks': comp['usd_by_provider'],
            'jev_requests': comp['jev_requests']}


def product_continuity(year):
    """Counts of witness rows by intermediate_product_evidence (TD2: unavailable_in_sources) for the year."""
    import td4_inputs as I
    pf = pq.ParquetFile(I.evidence_path(year))
    counts = {}
    for b in pf.iter_batches(batch_size=200000, columns=['variant', 'hop', 'intermediate_product_evidence',
                                                         'physical_channel_verified'], use_threads=False):
        d = b.to_pandas()
        d = d[d.variant == 'primary']
        for (h, e, v), n in d.groupby(['hop', 'intermediate_product_evidence', 'physical_channel_verified']).size().items():
            k = f'H{h}|{e}|physical_channel_verified={v}'
            counts[k] = counts.get(k, 0) + int(n)
    return counts


def run():
    import td3_runtime as R
    import td4_labels as L
    import td4_runner as RU
    import td4_models as MD
    token, freeze, clearance_b = G.admit_b()
    R.new_json(G.EVAL_LATCH, {'version': T.VERSION, 'clearance_b_sha256': token.clearance_sha,
                              'selection_freeze_sha256': R.sha(G.SELECTION_FREEZE)})
    start = time.perf_counter()
    sel = json.loads(G.SELECTION_FREEZE.read_text(encoding='utf-8'))
    for p, h in clearance_b.get('models_files_sha256', {}).items():
        if R.sha(p) != h:
            raise PermissionError('Through-2024 model changed: ' + p)
    mlp_p, b_p, heads_p = MD.model_paths(T.TRAIN_END[T.TEST_YEAR])
    heads = json.loads(heads_p.read_text(encoding='utf-8'))
    b = pq.read_table(b_p).to_pandas()
    keys = MD.feature_frame(T.TEST_YEAR)[['firm_id', 'cutoff', 'origin_session']]
    labels = L.compute(T.TEST_YEAR, keys, f'{T.TEST_YEAR}-12-31', token)
    f, info = SC.frame(T.TEST_YEAR, labels, b)
    outs = SC.provider_outputs(T.TEST_YEAR, f, RU.collect(T.TEST_YEAR), SC.eligible_map(T.TEST_YEAR))
    res = evaluate_core(f, outs, heads, sel, costs=actual_costs(T.TEST_YEAR))
    res['product_continuity_evidence'] = product_continuity(T.TEST_YEAR)
    out = {'schema': 'TD4_EVALUATION_RESULT_v1', 'version': T.VERSION, 'evaluation_year': T.TEST_YEAR,
           'label': 'CONFIRMATORY for this pipeline (pre-registered P1 and P2 only); caveats: '
                    + ' '.join(T.PREREGISTRATION['labels_and_caveats']),
           'rows_info': info, **res, 'selection_freeze_sha256': R.sha(G.SELECTION_FREEZE),
           'preregistration_sha256': R.sha(G.PREREG), 'clearance_b_sha256': token.clearance_sha,
           'elapsed_seconds': time.perf_counter() - start, **R.peak()}
    R.new_json(G.EVAL_RESULT, out)
    return {'P1': {k: out['P1'][k] for k in ('arm', 'mean', 'se', 'normal_95_interval', 'confirmed')},
            'P2': {k: out['P2'][k] for k in ('jev_arm', 'mean', 'se', 'normal_95_interval', 'confirmed')},
            'stop_rule_readings': out['stop_rule_readings'], 'result': str(G.EVAL_RESULT)}
