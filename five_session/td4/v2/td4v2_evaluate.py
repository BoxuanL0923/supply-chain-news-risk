"""Once-only v2 evaluation on pooled 2023-2025 (clearance B v2.3). P1 and P2 as pre-registered; per-year results,
every arm, references, coverage, cost and H1/H2/H3 are descriptive.

The computation is v1 td4_evaluate.evaluate_core, unchanged, applied to the pooled rows (and, descriptively, to
each year); the frozen TD2 MODEL_v2 beta and the TD1 refitted MLP b are used for every arm; the 2021-2022 selections
are read from the selection freeze. The latch is written before any test label is computed.

v2.3 (TD4C clearance-B review item B6-M, fix brief F1-F5; no number, rule, grid or confirmation logic changes):
F1 actual_costs() (ledgers and inputs only, never a label) runs right after admit_b, before any provider output is
   collected and before the latch; only its small result is kept and the completion structures are freed.
F2 provider outputs are collected compactly per test year (td4v2_slim.collect_slim: main pass only, only the fields
   td4_score.provider_outputs reads); the unchanged td4_score.provider_outputs builds the arm values from them.
F3 per-year intermediates (keys, labels, compact outputs, eligible maps, the pooled frame) are released once used.
F4 as soon as the pooled evaluate_core result and its stop-rule readings exist they are written to
   EVALUATION_PRIMARY_v2.json, before any descriptive part runs. Every descriptive part (per year, 2025 halves, S1,
   product continuity) runs in its own try/except that records the error and continues; the final result file
   repeats the primary result as written. There is no replay or rerun path: admit_b refuses when the latch, the
   primary file or the result exists.
F5 wording only: the P1/P2 'comparison' text says "chosen on 2021-2022" (the v1 text said 2024), and P1/P2 state
   that "confirmed" means the interval lies above zero in an exploratory historical reproduction of 2023-2025.
"""
import gc
import json
import math
import time

import pandas as pd

import td4_spec as T
import td4_env as E
import td4v2_spec as S
import td4v2_gate as G
import td4v2_inputs as IV
import td4v2_runner as RV
import td4v2_score as SC2

P1_COMPARISON = 'best pipeline arm (chosen on 2021-2022) vs firm-only MLP'
P2_COMPARISON = 'better Jev arm (chosen on 2021-2022) vs matched-input DeepSeek, same head'
CONFIRMATION_MEANING = ('"confirmed" means only that the two-sided normal 95% interval of the mean paired gain '
                        '(two-way clustered by firm and calendar week) lies above zero (interval above zero), in an '
                        'exploratory historical reproduction of 2023-2025: the TODO notes that the earlier risk study '
                        'already examined this period.')
EXPLORATORY_NOTE = ('exploratory = True is the marker that TD2 statistics.paired sets on every paired gain; it is kept '
                    'and agrees with the TODO caveat: the pooled 2023-2025 test is an exploratory historical '
                    'reproduction, and its pre-registered rule reads "confirmed" as "interval above zero" within it.')
RESULT_LABEL = ('Pre-registered P1 and P2 on pooled 2023-2025, an exploratory historical reproduction (TODO caveat); '
                '"confirmed" means the two-sided 95% interval lies above zero (interval above zero). ')


def readings(res):
    pre = S.PREREGISTRATION['stop_rule_readings']
    return {'P1': pre['P1_confirmed'] if res['P1']['confirmed'] else pre['P1_not_confirmed'],
            'P2': ('Jev improves on matched-input DeepSeek in this test (interval above zero).' if res['P2']['confirmed']
                   else pre['P2_not_confirmed'])}


def v2_wording(res):
    """F5: wording of the pooled P1/P2 entries only; every number and the confirmed flags stay as computed."""
    res['P1']['comparison'] = P1_COMPARISON
    res['P2']['comparison'] = P2_COMPARISON
    for k in ('P1', 'P2'):
        res[k]['confirmation_meaning'] = CONFIRMATION_MEANING
        res[k]['exploratory'] = True
        res[k]['exploratory_note'] = EXPLORATORY_NOTE
    return res


def actual_costs():
    out = {}
    for p in S.RUN_ORDER:
        c = RV.completion(p)
        out[p] = {'usd_by_provider_including_retries_and_reasks': c['usd_by_provider'],
                  'jev_requests': c['jev_requests']}
        del c
        gc.collect()
    return out


def _subset_outputs(o, idx):
    return {arm: {k: ([v[i] for i in idx] if k != 'valid' else v[idx]) for k, v in o[arm].items()} for arm in o}


def _cfg(c):
    return {k: c.get(k) for k in ('arm', 'rule', 'param', 'fallback', 'mse_selection')}


def secondary_s1(frames, outs, heads, sel):
    """Pre-registered DESCRIPTIVE secondary S1 (coordinator, 2026-11-06): the same frozen through-2022 models, grids,
    fallback and tie rules; selection on 2023 instead of 2021-2022 (td4_select.select_core with the test heads);
    S1-P1 and S1-P2 on pooled 2024-2025 with the same two-way clustered intervals. Never confirmatory, no stop-rule
    reading. Also (a) whether the 2021-2022 and 2023 selections pick the same configuration per arm and (b) per test
    year the in-year best configuration per arm (an oracle table, not a valid forecast). Computed in the same once-only
    run from the outcomes already read for the primary (each outcome is read once)."""
    import td4_evaluate as EV1
    import td4_select as SL1
    if not {2023, 2024, 2025} <= set(S.TEST_YEARS):
        return {'skipped': 'test years differ (rehearsal)'}
    i23 = S.TEST_YEARS.index(2023)
    f23 = frames[i23].reset_index(drop=True)
    sel_s1 = {**SL1.select_core(f23, outs[i23], heads), 'training_beta_order_frozen': heads['training_beta_order']}
    del f23
    later = [S.TEST_YEARS.index(y) for y in (2024, 2025)]
    f2425 = pd.concat([frames[i] for i in later], ignore_index=True)
    o2425 = SC2.concat_outputs([outs[i] for i in later])
    r = EV1.evaluate_core(f2425, o2425, heads, sel_s1)
    del f2425, o2425
    gc.collect()
    p1 = {k: r['P1'][k] for k in ('arm', 'n', 'mean', 'se', 'normal_95_interval', 'relative_gain', 'baseline_mse',
                                  'arm_mse') if k in r['P1']}
    p2 = {k: r['P2'][k] for k in ('jev_arm', 'n', 'mean', 'se', 'normal_95_interval') if k in r['P2']}
    del r
    same = {}
    for arm in T.PROVIDER_ARMS:
        a, b = sel['chosen_provider_configs'][arm], sel_s1['chosen_provider_configs'][arm]
        same[arm] = {'same_config': all(a.get(k) == b.get(k) for k in ('rule', 'param', 'fallback')),
                     'selection_2122': _cfg(a), 'selection_2023': _cfg(b)}
    same['best_pipeline_arm'] = {'selection_2122': sel['best_pipeline_arm'], 'selection_2023': sel_s1['best_pipeline_arm'],
                                 'same': sel['best_pipeline_arm'] == sel_s1['best_pipeline_arm']}
    same['better_jev_arm'] = {'selection_2122': sel['better_jev_arm'], 'selection_2023': sel_s1['better_jev_arm'],
                              'same': sel['better_jev_arm'] == sel_s1['better_jev_arm']}
    oracle = {}
    for y, f, o in zip(S.TEST_YEARS, frames, outs):
        s = SL1.select_core(f.reset_index(drop=True), o, heads)
        oracle[str(y)] = {'label': 'ORACLE: in-year best configuration per arm; not a valid forecast',
                          'configs': {arm: _cfg(c) for arm, c in s['chosen_provider_configs'].items()},
                          'all_on_mse': s['all_on']['mse_selection'], 'baseline_mse': s['baseline_mse_selection'],
                          'best_pipeline_arm': s['best_pipeline_arm'], 'better_jev_arm': s['better_jev_arm']}
        del s
        gc.collect()
    return {'label': 'SECONDARY S1, DESCRIPTIVE ONLY: selection on 2023, tested on pooled 2024-2025; never '
                     'confirmatory; no stop-rule reading',
            'selection_2023': {'chosen_provider_configs': {a: _cfg(c) for a, c in
                                                           sel_s1['chosen_provider_configs'].items()},
                               'best_pipeline_arm': sel_s1['best_pipeline_arm'],
                               'better_jev_arm': sel_s1['better_jev_arm'], 'rows': sel_s1['rows']},
            'S1_P1_arm_vs_mlp_2024_2025': p1, 'S1_P2_jev_vs_deepseek_2024_2025': p2,
            'a_same_configuration_2122_vs_2023': same, 'b_in_year_oracle': oracle}


def year_2025_halves(frames, outs, heads, sel):
    """Descriptive 2025 split at the FactSet relationship horizon (origins on or before 2025-06-06 versus after)."""
    import td4_evaluate as EV1
    halves = {}
    if 2025 not in S.TEST_YEARS:                       # (the rehearsal tests on 2022 only)
        return halves
    f25, o25 = frames[S.TEST_YEARS.index(2025)], outs[S.TEST_YEARS.index(2025)]
    h = pd.Timestamp(S.FACTSET_HORIZON)
    for name, mask in (('origins_to_' + S.FACTSET_HORIZON, f25.origin_session <= h),
                       ('origins_after_' + S.FACTSET_HORIZON, f25.origin_session > h)):
        mask = mask.to_numpy()
        part = f25[mask].reset_index(drop=True)
        if not len(part):
            continue
        idx = [i for i, v in enumerate(mask) if v]
        po = {arm: {k: ([v[i] for i in idx] if k != 'valid' else v[idx]) for k, v in o25[arm].items()}
              for arm in o25}
        try:
            r = EV1.evaluate_core(part, po, heads, sel)
            halves[name] = {'descriptive': True, 'n': int(len(part)),
                            'P1_arm_vs_mlp': {k: r['P1'][k] for k in ('arm', 'mean', 'se', 'normal_95_interval')},
                            'P2_jev_vs_deepseek': {k: r['P2'][k] for k in ('jev_arm', 'mean', 'se',
                                                                           'normal_95_interval')}}
        except Exception as e:
            halves[name] = {'descriptive': True, 'n': int(len(part)), 'error': str(e)[:200]}
        del part, po
        gc.collect()
    return halves


def per_year_part(f, o, heads, sel):
    """Descriptive result of one test year (the v2.2 per-year entry, unchanged)."""
    import td4_evaluate as EV1
    r = EV1.evaluate_core(f.reset_index(drop=True), o, heads, sel)
    out = {'descriptive': True, 'P1_arm_vs_mlp': {k: r['P1'][k] for k in
                                                  ('arm', 'n', 'mean', 'se', 'normal_95_interval',
                                                   'relative_gain', 'baseline_mse', 'arm_mse')},
           'P2_jev_vs_deepseek': {k: r['P2'][k] for k in ('jev_arm', 'n', 'mean', 'se', 'normal_95_interval')},
           'arms': r['arms'], 'references': r['references'], 'coverage': r['coverage']}
    out['P1_arm_vs_mlp'].pop('confirmed', None)
    return out


def _error(e):
    return {'error': f'{type(e).__name__}: {str(e)[:300]}', 'descriptive': True}


def guarded(fn, *args):
    """F4: one descriptive part; any exception (MemoryError included) or a value that the strict JSON writer
    refuses is recorded as an error and the run continues."""
    import td3_runtime as R
    try:
        value = fn(*args)
        R.dumps(value)                                 # allow_nan=False: a non-finite float is recorded, not written
        return value
    except Exception as e:
        return _error(e)
    finally:
        gc.collect()


def _finite(x):
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def write_primary(payload):
    """F4: the primary result, written exclusively before any descriptive part. A non-finite float (refused by the
    strict writer before the file is created) is written as its repr string, and the file says so."""
    import td3_runtime as R
    try:
        return R.new_json(S.EVAL_PRIMARY, payload)
    except ValueError as e:
        if S.EVAL_PRIMARY.exists():
            raise
        return R.new_json(S.EVAL_PRIMARY, {**_finite(payload), 'nonfinite_floats_written_as_strings': str(e)[:200]})


def run():
    import td3_runtime as R
    import td4_evaluate as EV1
    import td4_score as SC1
    import td4v2_outcomes as O
    import td4_models as MD
    import td4v2_slim as SL3
    phases = []
    token, freeze, clearance_b = G.admit_b()          # R1: every check (no replay, ledgers, model files) first
    sel = json.loads(S.SELECTION_FREEZE.read_text(encoding='utf-8'))
    heads = SC2.test_heads()
    costs = actual_costs()                            # F1: ledgers and inputs only; before any output and the latch
    gc.collect()
    phases.append(SL3.memory('costs'))
    slim = {}
    for y in S.TEST_YEARS:                            # F2: compact main-pass outputs (N4, N5); no label yet
        slim[y] = SL3.collect_slim(str(y))
        gc.collect()
        phases.append(SL3.memory(f'collect_{y}'))
    E.guard()                                         # the disk and memory floors that test_labels checks, pre-latch
    R.new_json(S.EVAL_LATCH, {'version': S.VERSION, 'code_version': S.CODE_VERSION,
                              'clearance_b_sha256': token.clearance_sha,
                              'selection_freeze_sha256': R.sha(S.SELECTION_FREEZE)})
    t0 = time.perf_counter()
    frames, outs, infos = [], [], {}
    for y in S.TEST_YEARS:
        keys = MD.feature_frame(y)[['firm_id', 'cutoff', 'origin_session']]
        labels = O.test_labels(y, keys, token)
        del keys
        f, info = SC2.test_frame(y, labels)
        del labels
        f['test_year'] = y
        infos[str(y)] = info
        so = slim.pop(y)                              # F3: the compact outputs of the year are released here
        outs.append(SC1.provider_outputs(None, f, so.rows(), so.eligible_map))
        del so
        frames.append(f)
        del f
        gc.collect()
        phases.append(SL3.memory(f'labels_frame_{y}'))
    pooled = pd.concat(frames, ignore_index=True)
    pooled_outs = SC2.concat_outputs(outs)
    res = EV1.evaluate_core(pooled, pooled_outs, heads, sel, costs=costs)
    res['stop_rule_readings'] = readings(res)
    v2_wording(res)
    del pooled, pooled_outs
    gc.collect()
    phases.append(SL3.memory('pooled_primary'))
    primary = {'schema': 'TD4_EVALUATION_PRIMARY_v2', 'version': S.VERSION, 'code_version': S.CODE_VERSION,
               'evaluation_years': S.TEST_YEARS,
               'label': RESULT_LABEL + ' '.join(S.PREREGISTRATION['labels_and_caveats']),
               'rows_info': infos, **res,
               'selection_freeze_sha256': R.sha(S.SELECTION_FREEZE), 'preregistration_sha256': R.sha(S.PREREG),
               'clearance_b_sha256': token.clearance_sha,
               'note': 'Primary result (pooled P1/P2 and the stop-rule readings) written before any descriptive part; '
                       'EVALUATION_RESULT_v2.json repeats it unchanged and adds the descriptive parts.'}
    del res
    primary_path = write_primary(primary)             # F4: before any descriptive part
    del primary
    primary_sha = R.sha(primary_path)
    written = json.loads(primary_path.read_text(encoding='utf-8'))
    phases.append(SL3.memory('primary_written'))
    per_year = {}
    for y, f, o in zip(S.TEST_YEARS, frames, outs):   # F4: each descriptive part on its own
        per_year[str(y)] = guarded(per_year_part, f, o, heads, sel)
    phases.append(SL3.memory('per_year'))
    halves = guarded(year_2025_halves, frames, outs, heads, sel)
    phases.append(SL3.memory('halves_2025'))
    s1 = guarded(secondary_s1, frames, outs, heads, sel)
    phases.append(SL3.memory('secondary_s1'))
    del frames, outs
    gc.collect()
    continuity = {str(y): guarded(EV1.product_continuity, y) for y in S.TEST_YEARS}
    phases.append(SL3.memory('product_continuity'))
    out = {**written, 'schema': 'TD4_EVALUATION_RESULT_v2', 'primary_written_utc': written.get('written_utc'),
           'primary_result_path': str(primary_path), 'primary_result_sha256': primary_sha,
           'per_year_descriptive': per_year, 'product_continuity_evidence': continuity,
           'year_2025_halves_descriptive': halves, 'factset_horizon': S.FACTSET_HORIZON,
           'secondary_S1_descriptive': s1,
           'names_2025_source': 'CRSP CIZ 2025 security names (alias source of the 2025 build; td4v2_names2025)',
           'elapsed_seconds': time.perf_counter() - t0, 'memory_phases': phases, **R.peak()}
    out.pop('note', None)
    out.pop('written_utc', None)
    result_error = None
    try:
        R.new_json(S.EVAL_RESULT, out)
    except Exception as e:                            # the primary file stays as written
        result_error = f'{type(e).__name__}: {str(e)[:300]}'
        try:
            R.new_json(S.EVAL_RESULT, {'schema': 'TD4_EVALUATION_RESULT_v2', 'code_version': S.CODE_VERSION,
                                       'primary_result_path': str(primary_path), 'primary_result_sha256': primary_sha,
                                       'descriptive_result_write_error': result_error})
        except Exception:
            pass
    return {'P1': {k: written['P1'][k] for k in ('arm', 'mean', 'se', 'normal_95_interval', 'confirmed')},
            'P2': {k: written['P2'][k] for k in ('jev_arm', 'mean', 'se', 'normal_95_interval', 'confirmed')},
            'stop_rule_readings': written['stop_rule_readings'], 'primary': str(primary_path),
            'primary_sha256': primary_sha, 'result': str(S.EVAL_RESULT), 'result_write_error': result_error}
