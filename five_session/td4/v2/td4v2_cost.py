"""v2 cost and wall-time projection per period (2122, 2023, 2024, 2025) from the inputs actually sent and the TD3 pilot's
measured tokens (v1 td4_cost.pilot_fit / project_period / caps_from, unchanged); balance USD 82.57 after the top-up.
"""
import json
import time

import td4_spec as T
import td4_env as E
import td4v2_spec as S
import td4v2_inputs as IV


def _redaction_file(p):
    try:
        return str(IV.redaction_file(IV.period_dir(p)))
    except PermissionError as e:
        return 'missing: ' + str(e)[:80]


def project(write=True):
    import td4_cost as C
    import td3_runtime as R
    t0 = time.perf_counter()
    fit = C.pilot_fit()
    periods = {}
    for p in S.RUN_ORDER:
        try:
            b = [r['input_bytes'] for r in IV.callable_inputs(p).values()]
        except PermissionError as e:
            periods[p] = {'basis': 'inputs not ready: ' + str(e)[:80], 'firm_weeks': None}
            continue
        periods[p] = {'basis': 'frozen inputs sent', **C.project_period(b, fit)}
    built = [v for v in periods.values() if v.get('firm_weeks')]
    keys = ('deepseek_usd_off_peak', 'deepseek_usd_peak', 'deepseek_usd_off_peak_no_cache', 'jev_usd',
            'total_usd_off_peak', 'deepseek_calls', 'jev_requests', 'deepseek_busy_hours', 'jev_busy_hours')
    total = {k: float(sum(v[k] for v in built)) for k in keys}
    usable = S.BALANCE_USD - T.DEEPSEEK_BALANCE_FLOOR_USD
    cum, stop = 0.0, None
    for p in S.RUN_ORDER:
        v = periods[p]
        if v.get('firm_weeks'):
            cum += v['deepseek_usd_off_peak']
            v['deepseek_cumulative_usd_off_peak'] = cum
            if stop is None and cum > usable:
                stop = p
    out = {'version': S.VERSION, 'rule_version': S.RULE_VERSION,
           'redaction_files': {p: _redaction_file(p) for p in S.RUN_ORDER}, 'pilot_fit': fit, 'periods': periods,
           'all_periods': total,
           'deepseek_balance': {'balance_usd': S.BALANCE_USD, 'note': S.BALANCE_NOTE,
                                'floor_usd': T.DEEPSEEK_BALANCE_FLOOR_USD, 'usable_usd': usable,
                                'projected_spend_off_peak': total['deepseek_usd_off_peak'],
                                'shortfall_usd': max(0.0, total['deepseek_usd_off_peak'] - usable),
                                'first_period_reaching_floor': stop},
           'caps': {p: C.caps_from(v) for p, v in periods.items() if v.get('firm_weeks')},
           'elapsed_seconds': time.perf_counter() - t0}
    if write:
        tag = R.sha_text(json.dumps(out['periods'], sort_keys=True, default=str))[:8]
        R.new_json(S.OUT / f'COST_PROJECTION_{S.RULE_VERSION.replace(".", "_")}_{tag}.json', out)
    return {k: out[k] for k in ('periods', 'all_periods', 'deepseek_balance', 'caps')}
