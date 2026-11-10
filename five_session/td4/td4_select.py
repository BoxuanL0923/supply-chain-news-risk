"""2024 selection (clearance A, step 'select'): K / threshold / fallback per arm, best pipeline arm and better Jev arm,
written to the hashed SELECTION_FREEZE_v1.json; then the clearance-B request once the 2025 calls are complete.

Selection protocol (TD2 head_v2.selection_sweep, transposed to the amendment): the 2024 rows are scored with the MLP
and heads refitted through 2023 (out of sample for 2024), never with a head refitted on 2024. Per arm, the lowest 2024
MSE wins; ties keep the first grid entry (K order 0,1,2,3,5,all, then thresholds 0.25, 0.5, 0.75; fallback none
before all). Every 2024 number is selection-only.
"""
import hashlib
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

SEL = T.OUT / 'selection'


def select_core(f, outs, heads):
    """Pure selection on scored rows. Returns the selection record (configs, grids, MSEs, best arms)."""
    y, b = f.y.to_numpy(float), f.b.to_numpy(float)
    z = f[SC.ZCOLS['main']].to_numpy(float)
    beta = heads['beta']
    avail = SC.available(f)
    news = f.has_news.to_numpy(bool)
    grid, chosen = [], {}
    for arm in T.PROVIDER_ARMS:
        for rule, param, fb in Q.configs(arm):
            q, _ = Q.gate_for(arm, rule, param, fb, outs[arm], avail)
            rec = {'arm': arm, 'rule': rule, 'param': param, 'fallback': fb,
                   'mse_selection': SC.mse(y, Q.predict(b, z, q, beta))}
            grid.append(rec)
            if arm not in chosen or rec['mse_selection'] < chosen[arm]['mse_selection']:
                chosen[arm] = rec
    all_on = {'arm': 'all_on', 'rule': 'all', 'param': 'all', 'fallback': None,
              'mse_selection': SC.mse(y, Q.predict(b, z, Q.all_on_gate(avail), beta))}
    refs, ref_grid = {}, []
    for kind, seed in T.LOCAL_SELECTORS:
        ranks = SC.local_ranks(f, kind, seed, heads['training_beta_order'])
        key = f'{kind}:{seed}'
        for k in T.K_GRID:
            for fb in T.FALLBACKS:
                q = Q._gv().gates(ranks, avail, news, k, fb)
                rec = {'selector': kind, 'seed': seed, 'K': k, 'fallback': fb,
                       'mse_selection': SC.mse(y, Q.predict(b, z, q, beta))}
                ref_grid.append(rec)
                if key not in refs or rec['mse_selection'] < refs[key]['mse_selection']:
                    refs[key] = rec
    candidates = [chosen['deepseek_order'], chosen['jev_choice'], chosen['jev_noul'], all_on]
    best = candidates[0]
    for c in candidates[1:]:
        if c['mse_selection'] < best['mse_selection']:
            best = c
    jev = chosen['jev_choice'] if chosen['jev_choice']['mse_selection'] <= chosen['jev_noul']['mse_selection'] \
        else chosen['jev_noul']
    return {'chosen_provider_configs': chosen, 'all_on': all_on, 'provider_grid_selection': grid,
            'reference_configs': refs, 'reference_grid_selection': ref_grid,
            'baseline_mse_selection': SC.mse(y, b), 'best_pipeline_arm': best['arm'], 'best_pipeline_config': best,
            'better_jev_arm': jev['arm'], 'better_jev_config': jev,
            'matched_deepseek_config': chosen['deepseek_order'], 'rows': int(len(f))}


def ledger_digest(year, mode='real'):
    """SHA256 over the sorted (req_key, attempt, http_status, response_sha256) settle rows of a period."""
    import td4_runner as RU
    import td3_runtime as R
    rows = []
    for p in sorted(RU.period_dir(year, mode).glob('session_*/ledger.jsonl')):
        for r in RU.iter_jsonl(p):
            if r['kind'] == 'settle':
                rows.append(f"{r['req_key']}|{r['attempt']}|{r['http_status']}|{r['response_sha256']}")
    return hashlib.sha256('\n'.join(sorted(rows)).encode()).hexdigest(), len(rows)


def run():
    import td3_runtime as R
    import td4_labels as L
    import td4_runner as RU
    import td4_inputs as I
    import td4_models as MD
    token, freeze = G.admit('select')
    if G.SELECTION_FREEZE.exists():
        raise FileExistsError(G.SELECTION_FREEZE)
    start = time.perf_counter()
    records = I.callable_inputs(T.SELECTION_YEAR)
    comp = RU.completion(T.SELECTION_YEAR, records)
    if not comp['main_complete']:
        raise PermissionError('STOP: selection-year main pass incomplete: ' + json.dumps(comp))
    mlp_p, b_p, heads_p = MD.model_paths(T.TRAIN_END[T.SELECTION_YEAR])
    heads = json.loads(heads_p.read_text(encoding='utf-8'))
    b = pq.read_table(b_p).to_pandas()
    keys = MD.feature_frame(T.SELECTION_YEAR)[['firm_id', 'cutoff', 'origin_session']]
    labels = L.compute(T.SELECTION_YEAR, keys, f'{T.SELECTION_YEAR}-12-31', token)
    f, info = SC.frame(T.SELECTION_YEAR, labels, b)
    outs = SC.provider_outputs(T.SELECTION_YEAR, f, RU.collect(T.SELECTION_YEAR), SC.eligible_map(T.SELECTION_YEAR))
    res = select_core(f, outs, heads)
    digest, n_settle = ledger_digest(T.SELECTION_YEAR)
    out = {'schema': 'TD4_SELECTION_FREEZE_v1', 'version': T.VERSION,
           'label': f'SELECTION ONLY ({T.SELECTION_YEAR})',
           'selection_year': T.SELECTION_YEAR, **res, 'rows_info': info,
           'coverage_selection': SC.coverage(f, outs), 'training_beta_order_frozen': heads['training_beta_order'],
           'models_sha256': {str(p): R.sha(p) for p in (mlp_p, b_p, heads_p)},
           'preregistration_sha256': R.sha(G.PREREG), 'aggregation_freeze_sha256': R.sha(G.AGGREGATION_FREEZE),
           'ledger_digest_selection': digest, 'ledger_settle_rows_selection': n_settle,
           'clearance_a_sha256': token.clearance_sha, 'elapsed_seconds': time.perf_counter() - start}
    R.new_json(G.SELECTION_FREEZE, out)
    se = f[['firm_id', 'cutoff', 'origin_session']].copy()
    E.write_parquet(SEL / f'selection_rows_{T.SELECTION_YEAR}.parquet', se)
    req = request_b(quiet=True)
    return {'selection_freeze': str(G.SELECTION_FREEZE), 'sha256': R.sha(G.SELECTION_FREEZE),
            'best_pipeline_arm': res['best_pipeline_arm'], 'better_jev_arm': res['better_jev_arm'],
            'request_b': req}


def request_b(quiet=False):
    """Write CLEARANCE_REQUEST_TD4_B_v1.json (request only) once the selection freeze exists and the 2025 main pass is
    complete. Reads ledgers and hashes only (no outcome)."""
    import td3_runtime as R
    import td4_runner as RU
    import td4_inputs as I
    import td4_models as MD
    if not G.SELECTION_FREEZE.exists():
        raise PermissionError('Selection freeze absent')
    comp = RU.completion(T.TEST_YEAR, I.callable_inputs(T.TEST_YEAR))
    if not comp['main_complete']:
        if quiet:
            return {'written': False, 'reason': 'test-year main pass incomplete', 'completion': comp}
        raise PermissionError('test-year main pass incomplete: ' + json.dumps(comp))
    if G.REQUEST_B.exists():
        raise FileExistsError(G.REQUEST_B)
    digest, n = ledger_digest(T.TEST_YEAR)
    models = {str(p): R.sha(p) for p in MD.model_paths(T.TRAIN_END[T.TEST_YEAR])}
    req = {'schema': T.CLEARANCE_B_SCHEMA, 'status': 'REQUESTED', 'request_only': True,
           'clearance_a_sha256': R.sha(T.CLEARANCE_A) if T.CLEARANCE_A.exists() else None,
           'selection_freeze_sha256': R.sha(G.SELECTION_FREEZE), 'preregistration_sha256': R.sha(G.PREREG),
           'ledger_digest_2025': digest, 'ledger_settle_rows_2025': n,
           'models_sha256': hashlib.sha256(json.dumps(models, sort_keys=True).encode()).hexdigest(),
           'models_files_sha256': models, 'evaluation_year': T.TEST_YEAR, 'maximum_label_end': T.HARD_SEAL,
           'note': 'The reviewer copies this file to ' + str(T.CLEARANCE_B) + ' with status CLEARED and without '
                   'request_only after checking the selection freeze; the evaluator then runs once.'}
    R.new_json(G.REQUEST_B, req)
    return {'written': True, 'path': str(G.REQUEST_B)}
