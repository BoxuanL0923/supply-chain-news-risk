"""Cost and wall-time projection from the eligible firm-week counts and the TD3 pilot's measured tokens (TODO 'First
handoff': the old intraday dollar estimate does not apply).

Token model, per selector, fitted on the 200 pass-1 pilot calls (TD3 real run real_0b596ec45502): input tokens =
a + c x input_bytes (OLS on the frozen pilot inputs), applied to every frozen TD4 input of the period. DeepSeek
output tokens and cache-hit tokens are the pilot means. Prices are TD3 v6 constants (off-peak only for DeepSeek,
since the runner never starts a DeepSeek call at peak; the peak figure is shown for reference). Re-asks: REASK_N
firm-weeks x 3 selectors per period. Caps bound in the freeze: DeepSeek USD and Jev USD at 1.3 x projection, Jev
requests at 1.1 x planned requests + 500.
"""
import json
import time

import numpy as np

import td4_spec as T
import td4_env as E

BALANCE_USD = 62.57             # DeepSeek balance read 2026-11-06 05:47Z (brief); re-read by the runner at start
BALANCE_READ_UTC = '2026-11-06T05:47:00Z'


def _mods():
    E.paths()
    import td3_spec as S3
    import td3_runtime as R
    return S3, R


def pilot_fit():
    S3, R = _mods()
    inputs = {r['key']: r['input_bytes'] for r in R.read_jsonl(T.TD3 / 'build_v7/INPUTS.jsonl')}
    rows = {s: [] for s in S3.SELECTORS}
    lat = {s: [] for s in S3.SELECTORS}
    hit = []
    for r in R.read_jsonl(T.TD3_PILOT_RUN / 'responses.jsonl'):
        if r['http_status'] != 200 or r['unit']['pass'] != 1:
            continue
        s = r['unit']['selector']
        u = r['parsed'].get('usage', {})
        tok = u.get('prompt_tokens') if s == 'deepseek_order' else u.get('input_tokens')
        out = u.get('completion_tokens') if s == 'deepseek_order' else u.get('output_tokens')
        rows[s].append((inputs[r['unit']['key']], tok, out))
        lat[s].append(r['latency_s'])
        if s == 'deepseek_order':
            hit.append(u.get('prompt_cache_hit_tokens', 0))
    fit = {}
    for s, v in rows.items():
        x = np.array([a for a, _, _ in v], float)
        y = np.array([b for _, b, _ in v], float)
        c, a = np.polyfit(x, y, 1)
        fit[s] = {'n': len(v), 'intercept_tokens': float(a), 'tokens_per_input_byte': float(c),
                  'mean_input_tokens': float(y.mean()), 'mean_input_bytes': float(x.mean()),
                  'mean_output_tokens': float(np.mean([o for _, _, o in v])),
                  'mean_latency_s': float(np.mean(lat[s])), 'p90_latency_s': float(np.percentile(lat[s], 90))}
    fit['deepseek_order']['mean_cache_hit_tokens'] = float(np.mean(hit))
    return fit


def project_period(n_bytes_list, fit, reask=T.REASK_N):
    S3, _ = _mods()
    n = len(n_bytes_list)
    bytes_ = np.asarray(n_bytes_list, float)
    mean_b = float(bytes_.mean()) if n else 0.0
    out = {'firm_weeks': n, 'mean_input_bytes': mean_b}
    ds = fit['deepseek_order']
    calls = n + reask
    tin = calls * (ds['intercept_tokens'] + ds['tokens_per_input_byte'] * mean_b)
    hit = calls * ds['mean_cache_hit_tokens']
    tout = calls * ds['mean_output_tokens']
    for w in ('off_peak', 'peak'):
        p = S3.PRICES[w]
        out[f'deepseek_usd_{w}'] = float(((tin - hit) * p['input_miss'] + hit * p['input_hit'] + tout * p['output']) / 1e6)
    out['deepseek_usd_off_peak_no_cache'] = float((tin * S3.PRICES['off_peak']['input_miss'] +
                                                   tout * S3.PRICES['off_peak']['output']) / 1e6)
    out['deepseek_calls'] = calls
    jev_tok = 0.0
    for s in ('jev_choice', 'jev_noul'):
        jev_tok += calls * (fit[s]['intercept_tokens'] + fit[s]['tokens_per_input_byte'] * mean_b)
    out['jev_requests'] = 2 * calls
    out['jev_input_tokens'] = float(jev_tok)
    out['jev_usd'] = float(jev_tok * S3.JEV_PRICE_INPUT_PER_M / 1e6)
    out['total_usd_off_peak'] = out['deepseek_usd_off_peak'] + out['jev_usd']
    # wall time: concurrent workers, pilot latencies; DeepSeek only off-peak (17 of 24 h on weekdays)
    ds_h = calls * ds['mean_latency_s'] / T.DEEPSEEK_CONCURRENCY / 3600
    jev_h = 2 * calls * np.mean([fit[s]['mean_latency_s'] for s in ('jev_choice', 'jev_noul')]) / T.JEV_CONCURRENCY / 3600
    out['deepseek_busy_hours'] = float(ds_h)
    out['jev_busy_hours'] = float(jev_h)
    out['wall_hours_weekend_or_offpeak'] = float(max(ds_h, jev_h))
    out['wall_hours_weekday_worst'] = float(max(ds_h * 24 / 17, jev_h) + 7 * (ds_h > 0))
    return out


def caps_from(proj):
    return {'deepseek_max_usd': round(1.3 * proj['deepseek_usd_off_peak'], 2),
            'jev_max_usd': round(1.3 * proj['jev_usd'], 2),
            'jev_max_requests': int(1.1 * proj['jev_requests']) + 500}


def caps_for(year, n):
    """Caps for a dry run over n inputs (pilot mean bytes)."""
    fit = pilot_fit()
    proj = project_period([fit['deepseek_order']['mean_input_bytes']] * n, fit, reask=0)
    return caps_from(proj)


def project(write=True):
    S3, R = _mods()
    import td4_inputs as I
    start = time.perf_counter()
    fit = pilot_fit()
    periods = {}
    for y in T.PERIODS:
        p = I.INPUTS / f'period_{y}' / 'INPUTS.jsonl'
        if p.exists():
            b = [r['input_bytes'] for r in I.callable_inputs(y).values()]
            periods[str(y)] = {'basis': 'frozen TD4 inputs', **project_period(b, fit)}
        else:
            periods[str(y)] = {'basis': 'inputs not built', 'firm_weeks': None}
    built = [v for v in periods.values() if v.get('firm_weeks')]
    total = {k: float(sum(v[k] for v in built)) for k in ('deepseek_usd_off_peak', 'deepseek_usd_peak',
                                                         'deepseek_usd_off_peak_no_cache', 'jev_usd',
                                                         'total_usd_off_peak', 'deepseek_calls', 'jev_requests')}
    usable = BALANCE_USD - T.DEEPSEEK_BALANCE_FLOOR_USD
    first = periods.get(str(T.SELECTION_YEAR), {})
    out = {'version': T.VERSION, 'pilot_fit': fit, 'periods': periods, 'both_periods': total,
           'deepseek_balance': {'balance_usd': BALANCE_USD, 'read_utc': BALANCE_READ_UTC,
                                'floor_usd': T.DEEPSEEK_BALANCE_FLOOR_USD, 'usable_usd': usable,
                                'projected_spend_both_off_peak': total['deepseek_usd_off_peak'],
                                'top_up_needed_for_both': total['deepseek_usd_off_peak'] > usable,
                                'shortfall_usd': max(0.0, total['deepseek_usd_off_peak'] - usable),
                                'selection_year_fits': (first.get('deepseek_usd_off_peak') or 0) <= usable},
           'caps': {y: caps_from(v) for y, v in periods.items() if v.get('firm_weeks')},
           'prices': S3.PRICE_SOURCE, 'excluded': 'calls withheld for masking residuals; Jev and DeepSeek retries '
                                                  'beyond the pilot rate (0 retries in the pilot)',
           'elapsed_seconds': time.perf_counter() - start}
    if write:
        tag = R.sha_text(json.dumps(out['periods'], sort_keys=True))[:8]
        R.new_json(T.OUT / f'COST_PROJECTION_{tag}.json', out)
    return {k: out[k] for k in ('periods', 'both_periods', 'deepseek_balance', 'caps')}
