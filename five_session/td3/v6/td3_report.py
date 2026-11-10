"""Pilot report: selections, agreement, stability, invalid outputs, tokens/cost and a projected full-run cost.
v2 (TD3B, C4): Jev USD (input tokens x USD 0.042 per million, output free) is reported and projected. v3, v4: unchanged
except this docstring.

Reads only the run's own response and ledger files, the frozen inputs and the hash-bound TD2 token-cost counts.
No five-session label, return or loss is read or computed.
"""
import csv
import io
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sps
import td3_spec as S
from td3_runtime import read_jsonl, sha

NAMES = S.NAMES


def _ordered_files(run_dir, name):
    return sorted(Path(run_dir).glob(f'session_*/{name}'), key=lambda p: int(p.parent.name.split('_')[1]))


def load_run(run_dir):
    responses, ledger = [], []
    for p in _ordered_files(run_dir, 'responses.jsonl'):
        responses += [{**r, '_session': int(p.parent.name.split('_')[1])} for r in read_jsonl(p)]
    for p in _ordered_files(run_dir, 'ledger.jsonl'):
        ledger += [{**r, '_session': int(p.parent.name.split('_')[1])} for r in read_jsonl(p)]
    return responses, ledger


def final_units(responses):
    """Per (key, selector, pass): the last attempt with HTTP 200 if any, else the last attempt."""
    out = {}
    for r in sorted(responses, key=lambda r: (r['_session'], r['call_seq'])):
        u = (r['unit']['key'], r['unit']['selector'], r['unit']['pass'])
        if u in out and out[u]['http_status'] == 200 and r['http_status'] != 200:
            continue
        out[u] = r
    return out


def selections(final, keys):
    rows = {}
    for key in keys:
        for p in (1, 2):
            row = {}
            for sel in S.SELECTORS:
                r = final.get((key, sel, p))
                parsed = r['parsed'] if r else {}
                row[sel] = {'status': parsed.get('status', 'not_attempted') if r else 'not_attempted',
                            'issues': parsed.get('issues', []),
                            'value': (parsed.get('ordering') if sel == 'deepseek_order' else
                                      parsed.get('probabilities') if sel == 'jev_choice' else parsed.get('yes_scores')),
                            'choice': parsed.get('choice') if sel == 'jev_choice' else None}
            rows[(key, p)] = row
    return rows


def _vec(sel, row):
    v = row[sel]
    if v['status'] != 'valid':
        return None
    if sel == 'deepseek_order':
        return np.array([len(NAMES) - v['value'].index(n) for n in NAMES], float)
    return np.array([v['value'][n] for n in NAMES], float)


def _top1(sel, row):
    v = row[sel]
    if v['status'] != 'valid':
        return None, False
    if sel == 'deepseek_order':
        return v['value'][0], False
    if sel == 'jev_choice':
        return v['choice'], False
    best = max(v['value'].values())
    winners = [n for n in NAMES if v['value'][n] == best]
    return winners[0], len(winners) > 1


def _summary(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return {'n_defined': int(len(v)), 'n_undefined': int(len([x for x in values if x is None or not np.isfinite(x)])),
            'mean': float(v.mean()) if len(v) else None, 'median': float(np.median(v)) if len(v) else None,
            'share_positive': float((v > 0).mean()) if len(v) else None}


def _corr(a, b, kind):
    if a is None or b is None or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(sps.spearmanr(a, b)[0] if kind == 'spearman' else sps.kendalltau(a, b)[0])


def agreement(rows, keys):
    first = [rows[(k, 1)] for k in keys]
    out = {}
    for a, b in (('deepseek_order', 'jev_choice'), ('deepseek_order', 'jev_noul'), ('jev_choice', 'jev_noul')):
        pairs, none_rows, ties = [], 0, 0
        for r in first:
            ta, tie_a = _top1(a, r)
            tb, tie_b = _top1(b, r)
            if ta is None or tb is None:
                continue
            ties += tie_a + tie_b
            if S.NONE_OPTION in (ta, tb):
                none_rows += 1
                continue
            pairs.append((ta, tb))
        ma, mb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
        n = len(pairs)
        chance = sum(ma[m] * mb[m] for m in NAMES) / n ** 2 if n else None
        corr = {kind: _summary([_corr(_vec(a, r), _vec(b, r), kind) for r in first
                                if _vec(a, r) is not None and _vec(b, r) is not None])
                for kind in ('spearman', 'kendall_tau_b')}
        out[f'{a}__vs__{b}'] = {'n_both_valid_non_none': n, 'rows_with_none_choice': none_rows,
                                'noul_top1_ties': ties,
                                'top1_agreement': sum(x == y for x, y in pairs) / n if n else None,
                                'top1_agreement_chance_from_marginals': chance, 'rank_correlation': corr}
    marg = {}
    for sel in S.SELECTORS:
        tops = [_top1(sel, r)[0] for r in first]
        marg[sel] = dict(Counter(t for t in tops if t is not None))
    out['top1_marginals_pass1'] = marg
    return out


def stability(rows, reask_keys):
    out = {}
    for sel in S.SELECTORS:
        both = [(rows[(k, 1)], rows[(k, 2)]) for k in reask_keys]
        valid = [(a, b) for a, b in both if a[sel]['status'] == 'valid' and b[sel]['status'] == 'valid']
        status_change = sum(a[sel]['status'] != b[sel]['status'] for a, b in both)
        res = {'reask_units': len(both), 'both_valid': len(valid), 'status_changed': status_change}
        if sel == 'deepseek_order':
            res['identical_ordering_rate'] = (sum(a[sel]['value'] == b[sel]['value'] for a, b in valid) / len(valid)
                                              if valid else None)
            res['top1_same_rate'] = (sum(a[sel]['value'][0] == b[sel]['value'][0] for a, b in valid) / len(valid)
                                     if valid else None)
            res['kendall_tau_b'] = _summary([_corr(_vec(sel, a), _vec(sel, b), 'kendall') for a, b in valid])
        elif sel == 'jev_choice':
            opts = NAMES + [S.NONE_OPTION]
            tv = [0.5 * sum(abs(a[sel]['value'][o] - b[sel]['value'][o]) for o in opts) for a, b in valid]
            mx = [max(abs(a[sel]['value'][o] - b[sel]['value'][o]) for o in opts) for a, b in valid]
            res['same_choice_rate'] = (sum(a[sel]['choice'] == b[sel]['choice'] for a, b in valid) / len(valid)
                                       if valid else None)
            res['total_variation_mean'] = float(np.mean(tv)) if tv else None
            res['max_abs_probability_change'] = float(max(mx)) if mx else None
        else:
            d = [abs(a[sel]['value'][n] - b[sel]['value'][n]) for a, b in valid for n in NAMES]
            res['mean_abs_yes_change'] = float(np.mean(d)) if d else None
            res['max_abs_yes_change'] = float(max(d)) if d else None
            res['top1_same_rate'] = (sum(_top1(sel, a)[0] == _top1(sel, b)[0] for a, b in valid) / len(valid)
                                     if valid else None)
            res['spearman'] = _summary([_corr(_vec(sel, a), _vec(sel, b), 'spearman') for a, b in valid])
        out[sel] = res
    return out


def invalid_counts(final, keys, reask_keys):
    out = {}
    for sel in S.SELECTORS:
        for p, ks in ((1, keys), (2, reask_keys)):
            st, iss = Counter(), Counter()
            for k in ks:
                r = final.get((k, sel, p))
                if r is None:
                    st['not_attempted'] += 1
                    continue
                st[r['parsed']['status']] += 1
                iss.update(r['parsed'].get('issues', []))
            out[f'{sel}_pass{p}'] = {'status': dict(st), 'issues': dict(iss)}
    return out


def _dist(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return {'n': 0}
    return {'n': int(len(v)), 'sum': float(v.sum()), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90)), 'max': float(v.max())}


def tokens_cost(ledger):
    settles = [r for r in ledger if r['kind'] == 'settle']
    out = {}
    for sel in S.SELECTORS:
        rows = [r for r in settles if r['selector'] == sel]
        ok = [r for r in rows if r['http_status'] == 200]
        key_in, key_out = (('prompt_tokens', 'completion_tokens') if sel == 'deepseek_order'
                           else ('input_tokens', 'output_tokens'))
        ok1 = [r for r in ok if r['unit']['pass'] == 1]
        out[sel] = {'attempts': len(rows), 'http_200': len(ok),
                    'input_tokens': _dist([r['usage'].get(key_in) for r in ok]),
                    'output_tokens': _dist([r['usage'].get(key_out) for r in ok]),
                    'pass1_input_tokens': _dist([r['usage'].get(key_in) for r in ok1]),
                    'pass1_output_tokens': _dist([r['usage'].get(key_out) for r in ok1]),
                    'pass1_cache_hit_tokens': _dist([r['usage'].get('prompt_cache_hit_tokens', 0) for r in ok1]),
                    'latency_s': _dist([r['latency_s'] for r in rows]),
                    'models_returned': dict(Counter(str(r['model_returned']) for r in rows)),
                    'price_windows': dict(Counter(r['price_window'] for r in rows))}
        if sel == 'deepseek_order':
            out[sel]['cache_hit_tokens'] = _dist([r['usage'].get('prompt_cache_hit_tokens') for r in ok])
            out[sel]['reasoning_tokens'] = _dist([r['usage'].get('reasoning_tokens') for r in ok])
        out[sel]['usd_estimate_total'] = float(sum(r['usd_estimate'] or 0 for r in rows))
        out[sel]['usd_basis'] = dict(Counter(r['usd_basis'] for r in rows))
    reserves = [r for r in ledger if r['kind'] == 'reserve']
    jev_usd = out['jev_choice']['usd_estimate_total'] + out['jev_noul']['usd_estimate_total']
    out['totals'] = {'deepseek_usd_estimate': out['deepseek_order']['usd_estimate_total'],
                     'jev_usd_estimate': jev_usd,
                     'total_usd_estimate': out['deepseek_order']['usd_estimate_total'] + jev_usd,
                     'jev_requests': sum(r['provider'] == 'jev' for r in reserves),
                     'deepseek_attempts': sum(r['provider'] == 'deepseek' for r in reserves),
                     'jev_price_note': f'USD {S.JEV_PRICE_INPUT_PER_M} per million input tokens, output free '
                                       '(official Jev models page, read 2026-11-05)'}
    return out


def td2_call_counts(path=S.TD2_COST, verify=True):
    if verify:
        bound = json.loads(S.TD2_FREEZE.read_text(encoding='utf-8'))['artifact_sha256']
        if sha(S.TD2_FREEZE) != S.TD2_FREEZE_SHA256 or bound.get(str(Path(path))) != sha(path):
            raise ValueError('TD2 token-cost estimate differs from FREEZE_v2.json')
    counts = json.loads(Path(path).read_text(encoding='utf-8'))['counts_by_year']
    return {y: int(v['with_news_and_characteristic']) for y, v in counts.items()}


def projection(cost, counts, pass_rate):
    ds = cost['deepseek_order']
    tin, tout = ds['pass1_input_tokens'].get('mean'), ds['pass1_output_tokens'].get('mean')
    hit = ds['pass1_cache_hit_tokens'].get('mean') or 0.0
    ch, nl = cost['jev_choice'], cost['jev_noul']
    sets = {'selection_2021_2022': ('2021', '2022'), 'evaluation_2023_2024': ('2023', '2024'),
            'selection_plus_evaluation': ('2021', '2022', '2023', '2024')}
    out = {'unit': 'one call per selector per firm-week; TD2 count = primary firm-weeks with news and at least one '
                   'characteristic (TOKEN_COST_ESTIMATE_v2.json), scaled by the pilot re-verification pass rate',
           'pilot_pass_rate': pass_rate, 'deepseek_mean_input_tokens': tin, 'deepseek_mean_output_tokens': tout,
           'deepseek_mean_cache_hit_tokens': hit, 'token_basis': 'pass-1 HTTP 200 calls only',
           'excludes': 're-asks, retries, malformed-answer repeats', 'sets': {}}
    for name, ys in sets.items():
        n = sum(counts[y] for y in ys) * pass_rate
        row = {'td2_firm_weeks': sum(counts[y] for y in ys), 'projected_calls_per_selector': n}
        if tin is not None and tout is not None:
            for w, p in S.PRICES.items():
                row[f'deepseek_usd_{w}_no_cache'] = n * (tin * p['input_miss'] + tout * p['output']) / 1e6
                row[f'deepseek_usd_{w}_measured_cache'] = n * ((tin - hit) * p['input_miss'] + hit * p['input_hit']
                                                               + tout * p['output']) / 1e6
        row['jev_requests'] = 2 * n
        if ch['pass1_input_tokens'].get('mean') is not None and nl['pass1_input_tokens'].get('mean') is not None:
            row['jev_input_tokens'] = n * (ch['pass1_input_tokens']['mean'] + nl['pass1_input_tokens']['mean'])
            row['jev_output_tokens'] = n * ((ch['pass1_output_tokens'].get('mean') or 0)
                                            + (nl['pass1_output_tokens'].get('mean') or 0))
            row['jev_usd'] = (row['jev_input_tokens'] * S.JEV_PRICE_INPUT_PER_M
                              + row['jev_output_tokens'] * S.JEV_PRICE_OUTPUT_PER_M) / 1e6
            for w in S.PRICES:
                if f'deepseek_usd_{w}_no_cache' in row:
                    row[f'total_usd_{w}_no_cache'] = row[f'deepseek_usd_{w}_no_cache'] + row['jev_usd']
        else:
            row['jev_usd'] = None
        out['sets'][name] = row
    return out


def selections_csv(rows, keys, reask_keys):
    buf = io.StringIO()
    cols = ['key', 'pass', 'deepseek_status', 'deepseek_ordering', 'choice_status', 'choice'] + \
        [f'p_{o}' for o in NAMES + [S.NONE_OPTION]] + ['noul_status'] + [f'yes_{n}' for n in NAMES]
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator='\n')
    w.writeheader()
    for p, ks in ((1, keys), (2, reask_keys)):
        for k in ks:
            r = rows[(k, p)]
            row = {'key': k, 'pass': p, 'deepseek_status': r['deepseek_order']['status'],
                   'deepseek_ordering': '>'.join(r['deepseek_order']['value']) if r['deepseek_order']['value'] else '',
                   'choice_status': r['jev_choice']['status'], 'choice': r['jev_choice']['choice'] or '',
                   'noul_status': r['jev_noul']['status']}
            for o in NAMES + [S.NONE_OPTION]:
                row[f'p_{o}'] = r['jev_choice']['value'][o] if r['jev_choice']['value'] else ''
            for n in NAMES:
                row[f'yes_{n}'] = r['jev_noul']['value'][n] if r['jev_noul']['value'] else ''
            w.writerow(row)
    return buf.getvalue()


def ledger_csv(ledger):
    buf = io.StringIO()
    cols = ['session', 'call_seq', 'selector', 'key', 'pass', 'utc_start', 'utc_end', 'latency_s', 'http_status',
            'status', 'model_returned', 'input_tokens', 'output_tokens', 'cache_hit_tokens', 'price_window',
            'usd_estimate', 'usd_basis', 'cumulative_usd_after', 'jev_requests_after']
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator='\n')
    w.writeheader()
    for r in ledger:
        if r['kind'] != 'settle':
            continue
        u = r['usage'] or {}
        w.writerow({'session': r['_session'], 'call_seq': r['call_seq'], 'selector': r['selector'],
                    'key': r['unit']['key'], 'pass': r['unit']['pass'], 'utc_start': r['utc_start'],
                    'utc_end': r['utc_end'], 'latency_s': round(r['latency_s'], 3), 'http_status': r['http_status'],
                    'status': r['status'], 'model_returned': r['model_returned'],
                    'input_tokens': u.get('prompt_tokens', u.get('input_tokens', '')),
                    'output_tokens': u.get('completion_tokens', u.get('output_tokens', '')),
                    'cache_hit_tokens': u.get('prompt_cache_hit_tokens', ''), 'price_window': r['price_window'],
                    'usd_estimate': r['usd_estimate'], 'usd_basis': r['usd_basis'],
                    'cumulative_usd_after': r.get('cumulative_usd_after'), 'jev_requests_after': r.get('jev_requests_after')})
    return buf.getvalue()


def build_report(run_dir, records, counts, pass_rate, run_mode):
    responses, ledger = load_run(run_dir)
    final = final_units(responses)
    keys = [r['key'] for r in sorted(records, key=lambda r: r['sample_rank'])]
    reask_keys = [r['key'] for r in sorted((r for r in records if r['reask']), key=lambda r: r['reask_rank'])]
    rows = selections(final, keys)
    cost = tokens_cost(ledger)
    report = {'version': S.VERSION, 'run_mode': run_mode, 'exploratory': True,
              'mock_providers': run_mode != 'real',
              'scope': 'Selector pilot on 200 selection-period (2021-2022) firm-weeks; no five-session label, return '
                       'or loss is read or computed.',
              'firm_weeks': len(keys), 'reask_firm_weeks': len(reask_keys),
              'invalid_counts': invalid_counts(final, keys, reask_keys), 'agreement_pass1': agreement(rows, keys),
              'stability_reask': stability(rows, reask_keys), 'tokens_and_cost': cost,
              'projection': projection(cost, counts, pass_rate)}
    return report, selections_csv(rows, keys, reask_keys), ledger_csv(ledger)
