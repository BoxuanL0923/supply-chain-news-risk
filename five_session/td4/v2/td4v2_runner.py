"""v2 provider calls per period, in the order 2122 (selection subsample), 2023, 2024, 2025.

The v1 runner core is reused unchanged (td4_runner.run_session / Session / Budget / make_plan / read_balance /
Keyring / SimClock / RealClock): keyed idempotent ledger, response row before settle row, DeepSeek balance floor
USD 8, off-peak DeepSeek only, Jev 429 backoff, caps, TD3 request bodies, parsers and returned-model checks. New here
(v2 paths, the v2 clearance, the period order rule, per-period caps from FREEZE_TD4_A_v2.json) and the TD4C review
fixes:
N1 terminal client errors: a unit with a deterministic HTTP 4xx (td4v2_spec.TERMINAL_STATUS) in >= 2 separate
   sessions and never an HTTP 200 is settled without output (fallback) within a cap of max(3, 0.2% of the units) per
   provider and period; completion() counts it as done only within the cap. It is skipped by later sessions (the v1
   prior_state is wrapped for the session call only; the v1 file is unchanged).
N2 a session halted for a returned model outside the allowlist blocks every later session of the period until a
   reviewer writes overrides/MODEL_OVERRIDE_<period>_<session>.json (status REVIEWED).
N4 collect() reads sessions in numeric order (session_2 before session_10).
N5 collect() keeps an output only when its stored input SHA256 equals the sent input's SHA256 (and its key matches).
"""
import json
import uuid
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_spec as S
import td4v2_gate as G
import td4v2_inputs as IV


def period_root(period, mode='real'):
    if period not in S.RUN_ORDER:
        raise ValueError('Unknown period ' + str(period))
    return S.RUNS / f'{mode}_{period}'


def _q():
    E.paths()
    import td3_spec as S3
    menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
    q = S3.question_bundle(menu)[1]
    model_for = {'deepseek_order': S3.DEEPSEEK_MODEL, 'jev_choice': S3.JEV_MODEL, 'jev_noul': S3.JEV_MODEL}
    return S3, q, model_for


def sessions(pdir, name):
    """Files `name` of the period's sessions in numeric session order (N4)."""
    def num(p):
        try:
            return int(p.parent.name.split('_')[1])
        except (IndexError, ValueError):
            return 10 ** 9
    return sorted(Path(pdir).glob('session_*/' + name), key=lambda p: (num(p), str(p)))


def terminal_units(pdir):
    """N1: req_key -> provider for units with a deterministic 4xx in >= TERMINAL_MIN_SESSIONS sessions, no HTTP 200."""
    import td4_runner as RU
    by_key, ok = {}, set()
    for p in sessions(pdir, 'ledger.jsonl'):
        for r in RU.iter_jsonl(p):
            if r['kind'] != 'settle':
                continue
            if r['http_status'] == 200:
                ok.add(r['req_key'])
            elif r['http_status'] in S.TERMINAL_STATUS:
                by_key.setdefault(r['req_key'], [r['provider'], set()])[1].add(p.parent.name)
    return {k: v[0] for k, v in by_key.items() if k not in ok and len(v[1]) >= S.TERMINAL_MIN_SESSIONS}


def terminal_cap(units):
    return max(S.TERMINAL_CAP_MIN, int(S.TERMINAL_CAP_FRACTION * units))


def completion(period, records=None, mode='real'):
    import td4_runner as RU
    S3, q, model_for = _q()
    records = IV.callable_inputs(period) if records is None else records
    y = S.PERIOD_REQ_YEAR[period]
    pdir = period_root(period, mode)
    done, usd, jev = RU.prior_state(pdir)
    need = {}
    for k, r in records.items():
        for s in S3.SELECTORS:
            need[RU.req_key(y, 'main', s, k, r['input_sha256'], q, model_for[s])] = s
    have = sum(1 for k in need if k in done)
    term = {k: p for k, p in terminal_units(pdir).items() if k in need and k not in done}
    per_provider = Counter(term.values())
    units_by_provider = Counter('deepseek' if s == 'deepseek_order' else 'jev' for s in need.values())
    within = all(per_provider[p] <= terminal_cap(units_by_provider[p]) for p in per_provider)
    complete = (have + len(term) == len(need)) and within
    return {'period': period, 'units': len(need), 'completed': have, 'terminal_no_output': dict(per_provider),
            'terminal_cap': {p: terminal_cap(n) for p, n in units_by_provider.items()},
            'terminal_within_cap': within, 'main_complete': complete, 'usd_by_provider': usd, 'jev_requests': jev}


def check_order(period, completion_fn):
    """A later period refuses until every earlier period of RUN_ORDER has a complete main pass."""
    for prev in S.RUN_ORDER[:S.RUN_ORDER.index(period)]:
        st = completion_fn(prev)
        if not st['main_complete']:
            raise PermissionError(f'STOP: period {prev} must be complete before {period}: ' + json.dumps(st))
    return True


def model_halts(period, mode='real'):
    """N2: sessions of the period that halted on a returned-model violation, and whether each has a reviewed
    override."""
    out = []
    for p in sessions(period_root(period, mode), 'SESSION_SUMMARY.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        if str(d.get('halt') or '').startswith('model_not_allowed'):
            o = S.OVERRIDES / f'MODEL_OVERRIDE_{period}_{p.parent.name}.json'
            ok = o.exists() and json.loads(o.read_text(encoding='utf-8')).get('status') == 'REVIEWED'
            out.append({'session': p.parent.name, 'halt': d['halt'], 'override': str(o), 'reviewed': ok})
    return out


def _session_prior_state(pdir, terminal):
    """v1 prior_state with the N1 terminal units added to `done` (they are not re-sent)."""
    import td4_runner as RU
    orig = RU.prior_state

    def wrapped(d):
        done, usd, jev = orig(d)
        if Path(d) == Path(pdir):
            for k in terminal:
                done.setdefault(k, {'terminal_no_output': True})
        return done, usd, jev
    return orig, wrapped


def run_real(period, reask=False):
    import td4_runner as RU
    import td3_runtime as R
    import td3_providers as P
    token, freeze = G.admit('reask' if reask else 'call')
    if R.mode() != 'online':
        raise PermissionError('online runtime required')
    records = IV.callable_inputs(period)
    if not reask:
        check_order(period, completion)
    blocked = [h for h in model_halts(period) if not h['reviewed']]
    if blocked:
        raise PermissionError('STOP: returned-model halt without a reviewed override: ' + json.dumps(blocked))
    pdir = period_root(period)
    term = terminal_units(pdir)
    comp = completion(period, records)
    if not comp['terminal_within_cap']:
        raise PermissionError('STOP: terminal client errors above the cap; review: ' + json.dumps(comp))
    lock = R.RunLock(S.RUNS / 'TD4_RUN_v2.lock').acquire()
    orig, wrapped = _session_prior_state(pdir, term)
    try:
        RU.prior_state = wrapped
        red = R.Redactor()
        kr = RU.Keyring.from_environment(red)
        caps = freeze['caps'][period]
        d, s = RU.run_session(S.PERIOD_REQ_YEAR[period], records, P.UrllibTransport(), kr,
                              lambda: RU.read_balance(kr.get('deepseek')), RU.RealClock(), caps, 'real',
                              {'clearance_sha256': token.clearance_sha, 'freeze_sha256': G.sha(S.FREEZE_A),
                               'period': period, 'code_sha256_runner': G.sha(Path(__file__))},
                              reask=reask, root=pdir)
    finally:
        RU.prior_state = orig
        lock.release()
    return {'session_dir': str(d), **{k: s[k] for k in ('halt', 'counts', 'usd_period_after',
                                                        'balance_projected_end_usd', 'skipped_done_before')},
            'terminal_units_skipped': len(term), 'completion': completion(period)}


def dry_run(period, limit=None):
    """Mock providers over the period's real inputs (simulated clock starting a minute before a weekday peak)."""
    import td4_runner as RU
    import td4_cost as C
    import td3_runtime as R
    import td3_providers as P
    records = IV.callable_inputs(period)
    if limit:
        keep = sorted(records, key=lambda k: records[k]['sample_rank'])[:limit]
        records = {k: records[k] for k in keep}
    kr = RU.Keyring(R.Redactor(), {'deepseek': 'sk-mock-deepseek-0000', 'jev': 'mock-jev-key-0000'})
    root = S.RUNS / f'dryrun_{S.RULE_VERSION.replace(".", "_")}_{period}_{uuid.uuid4().hex[:8]}'
    tr = P.MockTransport()
    caps = C.caps_for(period, len(records))
    clock = RU.SimClock('2026-11-09T00:59:00')
    d, s = RU.run_session(S.PERIOD_REQ_YEAR[period], records, tr, kr, lambda: S.BALANCE_USD, clock, caps, 'dryrun',
                          {'dry_run': True, 'period': period}, root=root, concurrency={'deepseek': 4, 'jev': 2})
    done, usd, jev = RU.prior_state(root)
    n = len(tr.calls)
    d2, s2 = RU.run_session(S.PERIOD_REQ_YEAR[period], records, tr, kr, lambda: S.BALANCE_USD, clock, caps, 'dryrun',
                            {'dry_run': True, 'resume_check': True}, root=root, concurrency={'deepseek': 4, 'jev': 2})
    out = {'period': period, 'records': len(records), 'mock_calls': n, 'halt': s['halt'], 'usd_mock_tokens': usd,
           'jev_requests': jev, 'counts': s['counts'], 'simulated_wait_s': clock.slept,
           'resume_check': {'skipped_done_before': s2['skipped_done_before'], 'new_mock_calls': len(tr.calls) - n},
           'run_dir': str(root), 'code_sha256_runner': G.sha(Path(__file__)), 'rule_version': S.RULE_VERSION,
           'redaction_file': str(IV.redaction_file(IV.period_dir(period)))}
    R.new_json(root / 'DRYRUN_SUMMARY.json', out)
    return out


def collect_root(root, records=None):
    """td4_runner.collect with sessions in numeric order (N4) and the input-hash check (N5): an output whose stored
    input SHA256 differs from the sent input's (records[key]['input_sha256']) is treated as no output."""
    import td4_runner as RU
    best = {}
    for p in sessions(root, 'responses.jsonl'):
        for r in RU.iter_jsonl(p):
            r.pop('response_raw', None)
            k = (r['key'], r['selector'], r['pass'])
            ok = r['http_status'] == 200 and r['parsed'].get('status') != 'model_not_allowed'
            if records is not None:
                rec = records.get(r['key'])
                if rec is None or rec['input_sha256'] != r.get('input_sha256'):
                    ok = False
            prev = best.get(k)
            if prev is None or (ok and not prev['_ok']) or (ok == prev['_ok'] and not prev['_ok']):
                pr = r['parsed']
                best[k] = {'_ok': ok, 'usd_estimate': r.get('usd_estimate'), 'input_sha256': r.get('input_sha256'),
                           'parsed': {f: pr.get(f) for f in ('status', 'ordering', 'ordering_raw', 'choice',
                                                             'probabilities', 'yes_scores', 'issues',
                                                             'model_returned', 'usage')}}
    out = []
    for (key, sel, pas), r in best.items():
        pr = r['parsed']
        out.append({'key': key, 'selector': sel, 'pass': pas, 'http_ok': r['_ok'], 'status': pr.get('status'),
                    'ordering': pr.get('ordering'), 'ordering_raw': pr.get('ordering_raw'),
                    'choice': pr.get('choice'), 'probabilities': pr.get('probabilities'),
                    'yes_scores': pr.get('yes_scores'), 'issues': pr.get('issues'),
                    'model_returned': pr.get('model_returned'), 'usage': pr.get('usage'),
                    'usd_estimate': r.get('usd_estimate'), 'input_sha256': r.get('input_sha256')})
    return out


def collect(period, mode='real'):
    return collect_root(period_root(period, mode), IV.callable_inputs(period))
