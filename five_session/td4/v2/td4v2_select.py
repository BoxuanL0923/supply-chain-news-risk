"""v2 selection on the 2021-2022 subsample (clearance A v2, step 'select') -> SELECTION_FREEZE_v2.json, and the
clearance-B v2 request once the 2023, 2024 and 2025 main passes are complete.

The selection itself is v1 td4_select.select_core, unchanged (grids, fallback, references, ties, best pipeline arm,
better Jev arm), applied with the TD2 selection models (TD1 selection_mlp b, TD2 training heads) as TD2's protocol
prescribes. Every number written here is selection-only.
"""
import hashlib
import json
import time

import pandas as pd

import td4_spec as T
import td4_env as E
import td4v2_spec as S
import td4v2_gate as G
import td4v2_inputs as IV
import td4v2_runner as RV
import td4v2_score as SC2


def ledger_digest(period, mode='real'):
    import td4_runner as RU
    rows = []
    for p in sorted(RV.period_root(period, mode).glob('session_*/ledger.jsonl')):
        for r in RU.iter_jsonl(p):
            if r['kind'] == 'settle':
                rows.append(f"{r['req_key']}|{r['attempt']}|{r['http_status']}|{r['response_sha256']}")
    return hashlib.sha256('\n'.join(sorted(rows)).encode()).hexdigest(), len(rows)


def run():
    import td3_runtime as R
    import td4_select as SL1
    import td4_score as SC1
    import td4v2_outcomes as O
    token, freeze = G.admit('select')
    if S.SELECTION_FREEZE.exists():
        raise FileExistsError(S.SELECTION_FREEZE)
    t0 = time.perf_counter()
    comp = RV.completion(S.SELECT)
    if not comp['main_complete']:
        raise PermissionError('STOP: selection-period main pass incomplete: ' + json.dumps(comp))
    sub = IV.subsample_frame()                    # masker-v2 build of the selection period
    pairs = {(a, pd.Timestamp(c)) for a, c in zip(sub.firm_id, sub.cutoff)}
    labels = O.selection_labels((2021, 2022), token)
    f, info = SC2.selection_frame(pairs, labels)
    heads = SC2.selection_heads()
    outs = SC1.provider_outputs(None, f, RV.collect(S.SELECT), IV.eligible_map(S.SELECT))
    res = SL1.select_core(f, outs, heads)
    digest, n = ledger_digest(S.SELECT)
    out = {'schema': 'TD4_SELECTION_FREEZE_v2', 'version': S.VERSION,
           'label': 'SELECTION ONLY (2021-2022 subsample); scored by TD1 selection_mlp and the TD2 2015-2020 heads',
           **res, 'rows_info': info, 'subsample_firm_weeks': len(pairs),
           'coverage_selection': SC1.coverage(f, outs), 'training_beta_order_frozen': heads['training_beta_order'],
           'heads_source': heads['source'], 'preregistration_sha256': R.sha(S.PREREG),
           'aggregation_freeze_sha256': R.sha(S.AGGREGATION_FREEZE), 'ledger_digest_selection': digest,
           'ledger_settle_rows_selection': n, 'clearance_a_sha256': token.clearance_sha,
           'elapsed_seconds': time.perf_counter() - t0}
    R.new_json(S.SELECTION_FREEZE, out)
    return {'selection_freeze': str(S.SELECTION_FREEZE), 'sha256': R.sha(S.SELECTION_FREEZE),
            'best_pipeline_arm': res['best_pipeline_arm'], 'better_jev_arm': res['better_jev_arm'],
            'request_b': request_b(quiet=True)}


def combined_test_digest(mode='real'):
    """Digest of the settle rows of the 2023, 2024 and 2025 ledgers (recomputed by the clearance-B gate, R1)."""
    digests = {str(y): ledger_digest(str(y), mode) for y in S.TEST_YEARS}
    combined = hashlib.sha256(json.dumps({k: v[0] for k, v in digests.items()}, sort_keys=True).encode()).hexdigest()
    return combined, digests


def models_files():
    import td4_inputs as I
    files = [T.TD1 / 'model_frozen_v2.pkl', T.TD2 / 'MODEL_v2.json']
    files += [T.TD1 / f'predictions_v2/year_{y}.parquet' for y in S.TEST_YEARS
              if (T.TD1 / f'predictions_v2/year_{y}.parquet').exists()]
    return files


def request_b(quiet=False):
    import td3_runtime as R
    if not S.SELECTION_FREEZE.exists():
        raise PermissionError('Selection freeze v2 absent')
    for y in S.TEST_YEARS:
        c = RV.completion(str(y))
        if not c['main_complete']:
            if quiet:
                return {'written': False, 'reason': f'{y} main pass incomplete', 'completion': c}
            raise PermissionError(f'{y} main pass incomplete: ' + json.dumps(c))
    if S.REQUEST_B.exists():
        raise FileExistsError(S.REQUEST_B)
    combined, digests = combined_test_digest()
    models = {str(p): R.sha(p) for p in models_files()}
    req = {'schema': S.CLEARANCE_B_SCHEMA, 'status': 'REQUESTED', 'request_only': True,
           'clearance_a_sha256': R.sha(S.CLEARANCE_A) if S.CLEARANCE_A.exists() else None,
           'selection_freeze_sha256': R.sha(S.SELECTION_FREEZE), 'preregistration_sha256': R.sha(S.PREREG),
           'ledger_digest_test': combined, 'ledger_digests_by_year': {k: v[0] for k, v in digests.items()},
           'ledger_settle_rows_by_year': {k: v[1] for k, v in digests.items()},
           'models_sha256': hashlib.sha256(json.dumps(models, sort_keys=True).encode()).hexdigest(),
           'models_files_sha256': models, 'evaluation_years': S.TEST_YEARS,
           'maximum_label_end': S.LABEL_END_BOUND,
           'note': 'Request only. The reviewer writes ' + str(S.CLEARANCE_B) + ' with status CLEARED and without '
                   'request_only after checking the selection freeze; the evaluator then runs once.'}
    R.new_json(S.REQUEST_B, req)
    return {'written': True, 'path': str(S.REQUEST_B)}
