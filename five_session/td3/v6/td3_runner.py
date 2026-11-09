"""Selector runner: dry run (mock providers, offline) and real run (gated by a hash-bound clearance file). v6 (TD3F).

v6: the v6 clearance file (clearance/TD3_CLEARANCE_v6.json, schema TD3_CLEARANCE_v6) with the same bindings as v5;
otherwise unchanged from v5.

v5: the v5 clearance file (clearance/TD3_CLEARANCE_v5.json, schema TD3_CLEARANCE_v5) with the same bindings as v4
(the freeze, code, inputs, questions, mask table, caps, models, Jev price and probability-sum tolerance); the D:
free-space guard of td3_runtime is 25 GiB (TD3C4 N-e); otherwise unchanged from v4.

v4: the v4 clearance file (clearance/TD3_CLEARANCE_v4.json) with the v4 bindings (Jev Choice probability-sum
tolerance 0.03, TD3C3 N5) and the frozen mask table's sha256 equal in the clearance and the freeze; otherwise
unchanged from v3.

v3: separate consecutive-failure counters per provider (D4: a DeepSeek outage halts even while Jev succeeds, and vice
versa); a unit refused by the returned-model rules is never re-asked under the same clearance (m4: a resume after a
model_not_allowed response refuses unless the clearance file's sha256 differs from that of the session that recorded
the refusal); the v3 clearance file and bindings (deepseek-flash, thinking disabled).

Call order: for each sampled firm-week by sample_rank, DeepSeek order, Jev Choice, Jev Noul (pass 1); then the 20
re-ask firm-weeks by reask_rank with the identical request bodies (pass 2). Each attempt is reserved in the ledger
(USD upper bound for both providers, C4) before sending. A transport error or HTTP 429/5xx is retried at most twice
(5 s, 20 s back-off), each retry reserved and counted. Halts: HTTP 401/402/403; any HTTP 3xx (redirects are never
followed, R2); a third other 4xx response from one provider; a cap stop; 3 consecutive HTTP-200 responses of one
selector that are all structurally malformed or all invalid with a shared structural issue (C3); an HTTP-200
response whose returned model breaks the frozen rules (C5); 3 consecutive units whose final attempt ended in a
transport error or HTTP 429/5xx (R1), counted per provider (D4). A real run holds the run lock for its whole session
(R3). Invalid content is never re-asked. Nothing is backfilled.
"""
import json
import time
import uuid
from collections import Counter
from pathlib import Path
import td3_spec as S
from td3_runtime import (JsonlWriter, Redactor, RunLock, load_key, new_json, new_text, read_jsonl, sha, sha_bytes,
                         sha_text, utc, say, mode, peak)
from td3_ledger import Ledger, CapStop, prior_spend, deepseek_upper_usd, jev_upper_usd
from td3_providers import selectors, TransportError, UrllibTransport, MockTransport
import td3_report


class KeyRing:
    """Provider keys in memory only; every value is registered with the redactor."""

    def __init__(self, redactor, keys):
        self._keys = dict(keys)
        for v in self._keys.values():
            redactor.add(v)

    @classmethod
    def from_environment(cls, redactor):
        return cls(redactor, {p: load_key(var) for p, var in S.KEY_VARS.items()})

    def get(self, provider):
        return self._keys[provider]


def code_hashes(folder=S.CODE):
    return {str(p): sha(p) for p in sorted(Path(folder).glob('*.py'))}


def load_inputs(build_dir):
    records = read_jsonl(Path(build_dir) / 'INPUTS.jsonl')
    for r in records:
        if sha_text(r['input_text']) != r['input_sha256']:
            raise ValueError('Frozen input changed: ' + r['key'])
    if len({r['key'] for r in records}) != len(records):
        raise ValueError('Duplicate input key')
    return records


def plan(records):
    ordered = sorted(records, key=lambda r: r['sample_rank'])
    reask = sorted((r for r in records if r['reask']), key=lambda r: r['reask_rank'])
    return ([(r['key'], s, 1) for r in ordered for s in S.SELECTORS] +
            [(r['key'], s, 2) for r in reask for s in S.SELECTORS])


def structural_halt(recent):
    """C3: halt reason when the last STRUCTURAL_PROBE_N HTTP-200 parses of a selector share a structural defect."""
    if len(recent) < S.STRUCTURAL_PROBE_N:
        return None
    last = recent[-S.STRUCTURAL_PROBE_N:]
    if all(p['status'] == 'malformed_body' for p in last):
        return 'schema_mismatch'
    if all(p['status'] == 'invalid' for p in last):
        shared = set(S.STRUCTURAL_ISSUES)
        for p in last:
            shared &= set(p.get('issues', []))
        if shared:
            return 'structural_invalid_' + sorted(shared)[0]
    return None


def execute(units, by_key, sels, transport, keyring, ledger, writer, redactor, run_id, run_mode, session_name,
            done=frozenset(), sleep=time.sleep, clock=utc, retries=S.TRANSPORT_RETRIES, backoff=S.BACKOFF_S,
            progress_every=30):
    summary = {'units_planned': len(units), 'units_skipped_done': 0, 'units_attempted': 0, 'attempts': 0,
               'halt': None, 'client_errors': Counter()}
    recent = {s: [] for s in S.SELECTORS}   # HTTP-200 parses of each selector, in order
    failure_run = {p: 0 for p in sorted(set(S.PROVIDER.values()))}   # D4: one counter per provider
    for i, (key, sel_name, pas) in enumerate(units, 1):
        if (key, sel_name, pas) in done:
            summary['units_skipped_done'] += 1
            continue
        sel, rec = sels[sel_name], by_key[key]
        body = sel.body(rec['input_text'])
        unit = {'key': key, 'selector': sel_name, 'pass': pas}
        summary['units_attempted'] += 1
        for attempt in range(1, retries + 2):
            try:
                seq, upper = ledger.reserve(sel.provider, sel_name, unit, body)
            except CapStop as e:
                summary['halt'] = str(e)
                return summary
            t0, p0 = clock(), time.perf_counter()
            status = raw = terr = None
            try:
                status, raw = transport.post(sel.url, sel.headers(keyring.get(sel.provider)), body)
            except TransportError as e:
                terr = str(e)
            except Exception as e:  # sanitized: type name only
                terr = 'unexpected_' + type(e).__name__
            latency, t1 = time.perf_counter() - p0, clock()
            truncated = raw is not None and len(raw) > S.MAX_RESPONSE_BYTES
            if terr is not None:
                parsed = {'selector': sel_name, 'status': 'transport_error', 'issues': ['transport_error'],
                          'usage': {}, 'model_returned': None}
            elif truncated:
                parsed = {'selector': sel_name, 'status': 'response_too_large', 'issues': ['response_too_large'],
                          'usage': {}, 'model_returned': None}
            else:
                parsed = sel.apply_model_check(status, sel.parse(status, raw))
            row = ledger.settle(seq, sel.provider, sel_name, unit, t0, t1, latency, status, parsed.get('usage', {}),
                                parsed.get('model_returned'), parsed['status'], upper)
            writer.write({'run_id': run_id, 'run_mode': run_mode, 'session': session_name, 'call_seq': seq,
                          'unit': unit, 'attempt': attempt, 'utc_start': t0, 'utc_end': t1, 'latency_s': latency,
                          'http_status': status, 'transport_error': terr,
                          'request': {'url': sel.url, 'model': sel.model, 'question_version': S.QUESTION_VERSION,
                                      'input_sha256': rec['input_sha256'], 'body_sha256': sha_bytes(body),
                                      'body_bytes': len(body)},
                          'response_raw': raw[:S.MAX_RESPONSE_BYTES].decode('utf-8', 'replace') if raw is not None
                          else None, 'response_truncated': truncated, 'parsed': parsed,
                          'usd_estimate': row['usd_estimate'], 'usd_basis': row['usd_basis'],
                          'price_window': row['price_window'], 'cumulative_usd_after': row['cumulative_usd_after'],
                          'jev_requests_after': row['jev_requests_after']})
            summary['attempts'] += 1
            if status == 200:
                if parsed['status'] == 'model_not_allowed':
                    summary['halt'] = f"model_not_allowed_{sel_name}:{parsed.get('model_violation')}"
                    return summary
                recent[sel_name].append(parsed)
                reason = structural_halt(recent[sel_name])
                if reason:
                    summary['halt'] = f'{reason}_{sel_name}'
                    return summary
            if (terr is not None or status in S.RETRY_STATUS) and attempt <= retries:
                sleep(backoff[min(attempt - 1, len(backoff) - 1)])
                continue
            if status in S.HALT_STATUS:
                summary['halt'] = f'http_{status}_{sel.provider}'
                return summary
            if status is not None and 300 <= status < 400:
                summary['halt'] = f'http_redirect_{status}_{sel.provider}'
                return summary
            if status is not None and 400 <= status < 500 and status not in S.RETRY_STATUS:
                summary['client_errors'][sel.provider] += 1
                if summary['client_errors'][sel.provider] >= S.CLIENT_ERROR_LIMIT:
                    summary['halt'] = f'client_error_limit_{sel.provider}'
                    return summary
            break
        failed = terr is not None or status in S.RETRY_STATUS     # R1: final 429/5xx counts like a transport error
        failure_run[sel.provider] = failure_run[sel.provider] + 1 if failed else 0     # D4: per provider
        if failure_run[sel.provider] >= S.FAILURE_UNIT_LIMIT:
            summary['halt'] = f'failure_unit_limit_{sel.provider}'
            return summary
        if progress_every and summary['units_attempted'] % progress_every == 0:
            say(redactor, 'PROGRESS', i, '/', len(units), 'usd', round(ledger.usd(), 6), 'jev', ledger.jev())
    return summary


def model_refusals(run_dir):
    """m4: sessions of a run that recorded a model_not_allowed response, with the clearance sha256 they ran under."""
    out = []
    for p in sorted(Path(run_dir).glob('session_*/responses.jsonl')):
        n = sum(1 for r in read_jsonl(p) if r['http_status'] == 200 and
                r['parsed'].get('status') == 'model_not_allowed')
        if n:
            started = p.parent / 'SESSION_STARTED.json'
            sha_c = json.loads(started.read_text(encoding='utf-8')).get('binding', {}).get('clearance_sha256')                 if started.exists() else None
            out.append({'session': p.parent.name, 'refusals': n, 'clearance_sha256': sha_c})
    return out


def done_units(run_dir):
    """Units with an HTTP-200 response, except responses refused by the returned-model rules. Such units are re-asked
    on a resume only, and a resume after a refusal requires a new clearance (run_real, m4)."""
    done = set()
    for p in Path(run_dir).glob('session_*/responses.jsonl'):
        for r in read_jsonl(p):
            if r['http_status'] == 200 and r['parsed'].get('status') != 'model_not_allowed':
                done.add((r['unit']['key'], r['unit']['selector'], r['unit']['pass']))
    return done


def next_session(run_dir):
    n = 1 + max([int(p.name.split('_')[1]) for p in Path(run_dir).glob('session_*')], default=0)
    return Path(run_dir) / f'session_{n}'


def session(run_dir, run_id, run_mode, build_dir, transport, keyring, redactor, caps, prior, menu, counts,
            pass_rate, done=frozenset(), sleep=time.sleep, clock=utc, binding=None):
    sess = next_session(run_dir)
    records = load_inputs(build_dir)
    by_key = {r['key']: r for r in records}
    sels = selectors(menu)
    units = plan(records)
    new_json(sess / 'SESSION_STARTED.json', {
        'version': S.VERSION, 'run_id': run_id, 'run_mode': run_mode, 'session': sess.name,
        'build_dir': str(build_dir), 'inputs_sha256': sha(Path(build_dir) / 'INPUTS.jsonl'),
        'question_sha256': S.question_bundle(menu)[1], 'units_planned': len(units), 'units_done_before': len(done),
        'max_usd': caps[0], 'max_jev_requests': caps[1], 'prior_usd': prior[0], 'prior_jev_requests': prior[1],
        'binding': binding or {}, 'deepseek': {'url': S.DEEPSEEK_URL, 'model': S.DEEPSEEK_MODEL,
                                                  'thinking': S.DEEPSEEK_THINKING},
        'jev': {'url': S.JEV_URL, 'model': S.JEV_MODEL}, 'model_checks': S.model_checks(),
        'jev_price_input_per_m': S.JEV_PRICE_INPUT_PER_M}, redactor)
    ledger = Ledger(sess / 'ledger.jsonl', redactor, caps[0], caps[1], prior[0], prior[1])
    writer = JsonlWriter(sess / 'responses.jsonl', redactor)
    try:
        summary = execute(units, by_key, sels, transport, keyring, ledger, writer, redactor, run_id, run_mode,
                          sess.name, done=done, sleep=sleep, clock=clock)
    finally:
        ledger.close()
        writer.close()
    summary = {**summary, 'client_errors': dict(summary['client_errors']), 'cumulative_usd': ledger.usd(),
               'cumulative_jev_requests': ledger.jev(), 'redactions': redactor.count}
    new_json(sess / 'SESSION_SUMMARY.json', {'version': S.VERSION, 'run_id': run_id, 'session': sess.name,
                                             **summary, **peak()}, redactor)
    report, sel_csv, led_csv = td3_report.build_report(run_dir, records, counts, pass_rate, run_mode)
    report['session_summary'] = summary
    new_json(sess / 'PILOT_REPORT.json', report, redactor)
    new_text(sess / 'selections.csv', sel_csv, redactor)
    new_text(sess / 'ledger.csv', led_csv, redactor)
    say(redactor, 'SESSION_DONE', run_id, sess.name, json.dumps({k: summary[k] for k in
                                                                ('units_attempted', 'attempts', 'halt',
                                                                 'cumulative_usd', 'cumulative_jev_requests')}))
    return sess, summary, report


def clearance_bindings():
    """Values the clearance and the freeze must both carry (besides hashes)."""
    return {'max_usd': S.MAX_USD, 'max_jev_requests': S.MAX_JEV_REQUESTS, 'deepseek_model': S.DEEPSEEK_MODEL,
            'deepseek_thinking': S.DEEPSEEK_THINKING,
            'jev_model': S.JEV_MODEL, 'jev_price_input_per_m': S.JEV_PRICE_INPUT_PER_M,
            'deepseek_returned_model_allowlist': sorted(S.DEEPSEEK_RETURNED_MODEL_PRICES),
            'probability_sum_tolerance': S.PROBABILITY_SUM_TOLERANCE}


def clearance_check(clearance_path=S.CLEARANCE, out=S.OUT, code_dir=S.CODE, menu=None):
    """(ok, reasons, freeze). Every binding must match; adding or editing any td3/v6 *.py file blocks the run."""
    p = Path(clearance_path)
    if not p.exists():
        return False, ['clearance file absent: ' + str(p)], None
    try:
        c = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False, ['clearance file unreadable'], None
    reasons = []
    if c.get('schema') != S.CLEARANCE_SCHEMA:
        reasons.append('schema')
    if c.get('status') != 'CLEARED':
        reasons.append('status is not CLEARED')
    if 'request_only' in c:
        reasons.append('request_only key present (a request is never a clearance)')
    fpath = Path(c.get('freeze_path') or (Path(out) / S.FREEZE_NAME))
    if not fpath.resolve().is_relative_to(Path(out).resolve()) or not fpath.exists():
        return False, reasons + ['freeze file missing or outside the TD3 private root'], None
    if sha(fpath) != c.get('freeze_sha256'):
        reasons.append('freeze_sha256')
    f = json.loads(fpath.read_text(encoding='utf-8'))
    if f.get('schema') != S.FREEZE_SCHEMA:
        reasons.append('freeze schema')
    current = code_hashes(code_dir)
    if c.get('code_sha256') != current:
        reasons.append('clearance code_sha256 differs from the current td3/v6 code inventory')
    if f.get('code_sha256') != current:
        reasons.append('freeze code_sha256 differs from the current td3/v6 code inventory')
    if not f.get('mask_tables_sha256') or c.get('mask_tables_sha256') != f.get('mask_tables_sha256'):
        reasons.append('mask_tables_sha256')
    if fpath.name in S.SUPERSEDED_FREEZES:
        reasons.append('a superseded freeze is never cleared')
    inputs = Path(f.get('inputs_path', ''))
    if not inputs.exists() or sha(inputs) != f.get('inputs_sha256') or c.get('inputs_sha256') != f.get('inputs_sha256'):
        reasons.append('inputs_sha256')
    for k, v in clearance_bindings().items():
        if c.get(k) != v or f.get(k) != v:
            reasons.append(k)
    q = S.question_bundle(menu if menu is not None else S.menu_rows())[1]
    if c.get('question_sha256') != q or f.get('question_sha256') != q:
        reasons.append('question_sha256')
    build_dir = Path(f.get('build_dir', ''))
    if not build_dir.resolve().is_relative_to(Path(out).resolve()) or inputs.parent.resolve() != build_dir.resolve():
        reasons.append('build_dir')
    return not reasons, reasons, f


def _pass_rate(build_dir):
    b = json.loads((Path(build_dir) / 'BUILD.json').read_text(encoding='utf-8'))['counts']
    return b['accepted'] / b['examined']


def run_dry(build_dir, runs_root=S.RUNS, menu=None, counts=None, transport=None, clearance_path=S.CLEARANCE):
    """Mock providers over the real frozen inputs; never reads a key and never opens a socket."""
    menu = menu if menu is not None else S.menu_rows()
    counts = counts if counts is not None else td3_report.td2_call_counts()
    redactor = Redactor()
    keyring = KeyRing(redactor, {'deepseek': 'DRYRUN-NO-KEY-deepseek', 'jev': 'DRYRUN-NO-KEY-jev'})
    run_id = 'dryrun_' + uuid.uuid4().hex[:12]
    run_dir = Path(runs_root) / run_id
    records = load_inputs(build_dir)
    sels = selectors(menu)
    units = plan(records)
    reask = [r for r in records if r['reask']]
    bodies = {s: [sels[s].body(r['input_text']) for r in records] for s in S.SELECTORS}
    reask_bodies = {s: [sels[s].body(r['input_text']) for r in reask] for s in S.SELECTORS}
    upper_ds = sum(deepseek_upper_usd(b) for b in bodies['deepseek_order'])
    upper_jev = sum(jev_upper_usd(b) for s in ('jev_choice', 'jev_noul') for b in bodies[s])
    est = {}
    for w, p in S.PRICES.items():
        est[w] = sum((len(b) * S.ESTIMATE_TOKENS_PER_BYTE * p['input_miss'] + S.ESTIMATE_OUTPUT_TOKENS * p['output'])
                     / 1e6 for b in bodies['deepseek_order'] + reask_bodies['deepseek_order'])
    jev_est = sum(len(b) * S.ESTIMATE_TOKENS_PER_BYTE * S.JEV_PRICE_INPUT_PER_M / 1e6
                  for s in ('jev_choice', 'jev_noul') for b in bodies[s] + reask_bodies[s])
    ok, reasons, _ = clearance_check(clearance_path, menu=menu)
    sess, summary, report = session(run_dir, run_id, 'dryrun', build_dir, transport or MockTransport(), keyring,
                                    redactor, (S.MAX_USD, S.MAX_JEV_REQUESTS), (0.0, 0), menu, counts,
                                    _pass_rate(build_dir), sleep=lambda s: None)
    plan_info = {'version': S.VERSION, 'run_id': run_id, 'mock_providers': True,
                 'planned_calls': {s: sum(u[1] == s for u in units) for s in S.SELECTORS},
                 'planned_jev_requests': sum(S.PROVIDER[u[1]] == 'jev' for u in units),
                 'deepseek_precall_upper_bound_usd_pass1': upper_ds,
                 'jev_precall_upper_bound_usd_pass1': upper_jev,
                 'total_precall_upper_bound_usd_pass1': upper_ds + upper_jev,
                 'deepseek_estimate_usd_all_calls': est, 'jev_estimate_usd_all_calls': jev_est,
                 'estimate_basis': f'request bytes x {S.ESTIMATE_TOKENS_PER_BYTE} tokens/byte, '
                                   f'{S.ESTIMATE_OUTPUT_TOKENS} DeepSeek output tokens, no cache; Jev input at '
                                   f'USD {S.JEV_PRICE_INPUT_PER_M}/M, output free',
                 'request_body_bytes': {s: td3_report._dist([len(b) for b in bodies[s]]) for s in S.SELECTORS},
                 'real_run_currently_allowed': ok, 'clearance_reasons': reasons,
                 'session': sess.name, 'summary': summary}
    new_json(run_dir / 'DRYRUN_PLAN.json', plan_info, redactor)
    return run_dir, plan_info


def run_real(resume=None, runs_root=S.RUNS, clearance_path=S.CLEARANCE, out=S.OUT, code_dir=S.CODE, menu=None,
             counts=None, transport=None, keyring=None, redactor=None, sleep=time.sleep, clock=utc, lock_path=None):
    """Real provider calls. Requires the online runtime (or an injected test transport) and a valid clearance."""
    if transport is None and mode() != 'online':
        raise PermissionError('Real calls need the online runtime')
    menu = menu if menu is not None else S.menu_rows()
    ok, reasons, freeze = clearance_check(clearance_path, out, code_dir, menu)
    if not ok:
        raise PermissionError('STOP: TD3 clearance not valid: ' + '; '.join(reasons))
    runs_root = Path(runs_root)
    with RunLock(lock_path or (runs_root / S.RUN_LOCK.name)):
        existing = sorted(p.name for p in runs_root.glob('real_*')) if runs_root.exists() else []
        if resume is None:
            if existing:
                raise PermissionError('A real run already exists (' + ', '.join(existing) + '); use --resume RUN_ID')
            run_id = 'real_' + uuid.uuid4().hex[:12]
        else:
            if resume not in existing:
                raise PermissionError('Unknown real run: ' + str(resume))
            run_id = resume
            current = sha(clearance_path)
            stale = [r for r in model_refusals(runs_root / run_id) if r['clearance_sha256'] in (None, current)]
            if stale:
                raise PermissionError('STOP: a returned-model refusal was recorded under this clearance (' +
                                      ', '.join(r['session'] for r in stale) + '); refused units are never re-asked '
                                      'without a new clearance')
        run_dir = runs_root / run_id
        redactor = redactor or Redactor()
        keyring = keyring or KeyRing.from_environment(redactor)
        transport = transport or UrllibTransport()
        prior = prior_spend(runs_root)
        counts = counts if counts is not None else td3_report.td2_call_counts()
        clearance = json.loads(Path(clearance_path).read_text(encoding='utf-8'))
        binding = {'clearance_path': str(clearance_path), 'clearance_sha256': sha(clearance_path),
                   'freeze_sha256': clearance.get('freeze_sha256'), 'code_sha256': code_hashes(code_dir),
                   'prior_ledgers': prior[2]}
        return session(run_dir, run_id, 'real', freeze['build_dir'], transport, keyring, redactor,
                       (S.MAX_USD, S.MAX_JEV_REQUESTS), prior[:2], menu, counts, _pass_rate(freeze['build_dir']),
                       done=frozenset(done_units(run_dir)), sleep=sleep, clock=clock, binding=binding)
