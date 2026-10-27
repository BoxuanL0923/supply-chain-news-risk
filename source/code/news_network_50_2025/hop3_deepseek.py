"""deepseek-v4-pro: resume-safe, budgeted development-only hop ablation.

No Qwen transport is instantiated. Prepared, label-free host evidence is reused.
The only network destination is the explicitly authorized DeepSeek endpoint.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

from . import hop3_guard as gates

MODEL = 'deepseek-v4-pro'
gates.MODEL = MODEL  # Label resource receipts and all writes in this run correctly.
URL = 'https://api.deepseek.com/chat/completions'
CAP = 15.0
MAX_TOKENS = 16000
PIN, PHIT, POUT = .435, .0435, .87
ARMS = ('OWN', 'H1', 'H2', 'H3', 'H3_GRAPH')
MODES = ('all_paths', 'continuity_only')
KNOWN_REJECTIONS = {'REJECTED_429_NO_INFERENCE', 'REJECTED_402_NO_INFERENCE'}
ROOT = gates.ROOT
OUT = ROOT / 'hop_ablation_v2'
LEDGER = ROOT / 'deepseek_costs_hop3.csv'
LOCK = threading.RLock()


def write(path, value):
    gates.write(path, {'model': MODEL, **value})
    with Path(path).open('r+b') as f:
        os.fsync(f.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


class DeepSeekGuard(gates.Guard):
    def install(self):
        self.network_requests = 0
        self.authorized_ips = set()
        def hook(event, args):
            if event == 'open':
                mode, flags = args[1], args[2] or 0
                writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
                self.check_path(args[0], writing)
            elif event == 'urllib.Request':
                if args[0] != URL:
                    raise PermissionError('Only the authorized DeepSeek URL is permitted')
                self.network_requests += 1
            elif event == 'socket.getaddrinfo':
                if args[0] != 'api.deepseek.com' or args[1] != 443:
                    raise PermissionError('Only DeepSeek DNS is permitted')
            elif event == 'socket.connect':
                address = args[1]
                if not isinstance(address, tuple) or address[1] != 443 or address[0] not in self.authorized_ips:
                    raise PermissionError('Only resolved DeepSeek HTTPS addresses are permitted')
        sys.addaudithook(hook)

    def resolve(self):
        self.authorized_ips.update(row[4][0] for row in socket.getaddrinfo('api.deepseek.com', 443, type=socket.SOCK_STREAM))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise PermissionError('HTTP redirects forbidden')


def request_body(prompt):
    if len(prompt.encode('utf-8')) + 1024 >= 24000:
        raise ValueError('Prompt exceeds conservative 24k token bound')
    return {'model': MODEL, 'temperature': 0, 'seed': 20261026,
            'max_tokens': MAX_TOKENS, 'thinking': {'type': 'enabled'},
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': prompt}]}


def token_cost(usage):
    total = int(usage['prompt_tokens'])
    hit = int(usage.get('prompt_cache_hit_tokens', 0))
    miss = int(usage.get('prompt_cache_miss_tokens', total - hit))
    if min(total, hit, miss, int(usage['completion_tokens'])) < 0 or hit + miss != total:
        raise ValueError('Inconsistent provider usage')
    return (miss * PIN + hit * PHIT + int(usage['completion_tokens']) * POUT) / 1e6


def rows():
    if not LEDGER.exists():
        return []
    with LEDGER.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def spent():
    return sum(float(r['est_cost_usd']) for r in rows())


def log_response(saved):
    key = saved['request_sha256']
    if key in {r['request_sha256'] for r in rows()}:
        return
    u = saved['envelope']['usage']
    row = {'time': saved['received_at'], 'model': saved['server_model'], 'request_sha256': key,
           'prompt_tokens': u['prompt_tokens'], 'prompt_cache_hit_tokens': u.get('prompt_cache_hit_tokens', 0),
           'prompt_cache_miss_tokens': u.get('prompt_cache_miss_tokens', u['prompt_tokens'] - u.get('prompt_cache_hit_tokens', 0)),
           'completion_tokens': u['completion_tokens'],
           'reasoning_tokens': (u.get('completion_tokens_details') or {}).get('reasoning_tokens', ''),
           'seconds': saved['seconds'], 'est_cost_usd': format(token_cost(u), '.12f')}
    new = not LEDGER.exists()
    with LEDGER.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def reconcile():
    with LOCK:
        for path in (OUT / 'responses').glob('*.json'):
            saved = gates.read(path)
            if saved['request']['model'] != MODEL:
                raise ValueError('Mixed-model response cache forbidden')
            log_response(saved)
        for path in (OUT / 'requests').glob('*.json'):
            if not (OUT / 'responses' / path.name).exists():
                state = gates.read(path)
                if state.get('status') in KNOWN_REJECTIONS and state['reserved_usd']:
                    state['reserved_usd'] = 0
                    write(path, state)
        pending = [p for p in (OUT / 'requests').glob('*.json') if not (OUT / 'responses' / p.name).exists()
                   and gates.read(p).get('status') not in KNOWN_REJECTIONS]
        if pending:
            raise RuntimeError(f'{len(pending)} unresolved prior requests: refusing automatic duplicate calls; their maximum reservations remain charged')


def parse_answer(saved, evidence_ids):
    raw = saved['envelope']
    if saved['server_model'] != MODEL or raw.get('model') != MODEL:
        raise ValueError('Unexpected provider model ID')
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Incomplete model answer; response preserved, no automatic retry')
    content = choice['message'].get('content') or ''
    # Reasoning is preserved in the raw envelope, never parsed as the score.
    answer = json.loads(content)
    if type(answer.get('score')) not in (int, float) or not math.isfinite(answer['score']):
        raise ValueError('Invalid finite score')
    ids = answer.get('evidence_ids')
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError('Invalid or ineligible evidence IDs')
    resolved, aliases, unresolved, issues = [], [], [], []
    if len(ids) > 8:issues.append('more_than_eight_returned_ids')
    if len(set(ids)) != len(ids):issues.append('duplicate_returned_ids')
    for source in ids:
        if source in evidence_ids:
            resolved.append(source)
            continue
        # Provider sometimes omits only the hash from RTRS:month:line:hash.
        # Accept a byte-exact unique month/line prefix, never fuzzy/title matches.
        parts = source.split(':')
        candidates = [e for e in evidence_ids if e.startswith(source + ':')] if len(parts) == 3 else []
        kind = 'omitted_hash_suffix'
        if source.startswith('RTRS:') and len(parts) == 4:
            candidates = [e for e in evidence_ids if e.split(':')[0] == 'RTRS' and e.split(':')[2:] == parts[2:]]
            kind = 'month_typo_exact_line_and_hash'
        if not source.startswith('RTRS:') or len(candidates) != 1:
            unresolved.append(source)
            continue
        resolved.append(candidates[0])
        aliases.append({'returned_id': source, 'canonical_id': candidates[0], 'resolution_kind': kind})
    if len(set(resolved)) != len(resolved):issues.append('duplicate_resolved_ids')
    if unresolved:issues.append('unresolved_or_ineligible_returned_ids')
    answer['returned_evidence_ids'] = ids
    answer['evidence_ids'] = list(dict.fromkeys(resolved))
    answer['citation_validation_issues'] = issues
    answer['unresolved_evidence_ids'] = unresolved
    answer['citation_validation_status'] = 'FLAGGED' if issues else 'PASS_WITH_EXACT_ALIAS_RESOLUTION' if aliases else 'PASS'
    if aliases:
        answer['evidence_id_alias_resolution'] = aliases
    if not isinstance(answer.get('reason'), str):
        raise ValueError('Missing final explanation')
    if raw['usage']['prompt_tokens'] >= 24000:
        raise ValueError('Provider prompt count exceeds 24k')
    token_cost(raw['usage'])
    return answer


def call(payload, guard):
    body = request_body(payload['prompt'])
    key = gates.digest(body)
    target = OUT / 'responses' / (key + '.json')
    with LOCK:
        if target.exists():
            saved = gates.read(target)
            if gates.digest(saved['request']) != key or saved['request_sha256'] != key:
                raise ValueError('Response cache request mismatch')
            log_response(saved)
            return saved
        balance_file = OUT / 'ACCOUNT_BALANCE_BLOCK.json'
        if balance_file.exists() and gates.read(balance_file)['status'] == 'BLOCKED':
            raise RuntimeError('DeepSeek account balance blocked; replenish and use --resume-after-balance before new requests')
        request_path = OUT / 'requests' / (key + '.json')
        prior = gates.read(request_path) if request_path.exists() else {}
        if prior and prior.get('status') not in KNOWN_REJECTIONS:
            raise RuntimeError('Request outcome unresolved; refusing duplicate call')
        # Reserve byte-bound cache-miss input and the entire reasoning+answer cap.
        worst = ((len(payload['prompt'].encode()) + 1024) * PIN + MAX_TOKENS * POUT) / 1e6
        reserved = sum(gates.read(p)['reserved_usd'] for p in (OUT / 'requests').glob('*.json') if p != request_path and not (OUT / 'responses' / p.name).exists())
        if spent() + reserved + worst > CAP:
            raise RuntimeError('Hard $15 budget cap: next request cannot be reserved')
        key_value = os.environ.get('DEEPSEEK_API_KEY')
        if not key_value:
            raise RuntimeError('DEEPSEEK_API_KEY is not set in the process environment')
        attempts = list(prior.get('attempts', []))
        write(request_path, {'request_sha256': key, 'request': body, 'reserved_usd': worst, 'started_at': now(), 'status': 'SENT_OUTCOME_PENDING', 'attempts': attempts})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + key_value, 'Content-Type': 'application/json'})
    del key_value
    started = time.monotonic()
    # Retry only explicit pre-inference rate-limit rejections. Unknown outcomes,
    # transport failures and other HTTP errors retain their reservation and stop.
    for retry in range(9):
        try:
            with opener.open(req, timeout=900) as response:
                raw = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            raw_bytes = exc.read()
            error_text = raw_bytes.decode('utf-8', errors='replace')
            attempt_file = key + f'_attempt_{len(attempts)+1}.json'
            with LOCK:
                write(OUT / 'http_errors' / attempt_file, {'request_sha256': key, 'status': exc.code,
                      'response_body': error_text, 'received_at': now()})
                attempts.append({'http_status': exc.code, 'response_file': attempt_file, 'at': now()})
                try:
                    error_type = json.loads(error_text)['error']['type']
                    is_rate_limit = exc.code == 429 and error_type == 'rate_limit_error'
                    is_balance = exc.code == 402 and error_type == 'insufficient_balance_error'
                except (ValueError, KeyError, TypeError): is_rate_limit = is_balance = False
                if is_balance:
                    write(balance_file, {'status': 'BLOCKED', 'detected_at': now(), 'http_status': 402,
                          'reason': 'DeepSeek insufficient_balance_error; no new calls until external account replenishment and explicit resume flag.', 'actual_cost_usd': spent(), 'last_rejected_request': key})
                status = 'REJECTED_402_NO_INFERENCE' if is_balance else 'REJECTED_429_NO_INFERENCE' if is_rate_limit else 'SENT_OUTCOME_PENDING'
                reserve = 0 if is_balance or (is_rate_limit and retry == 8) else worst
                write(request_path, {'request_sha256': key, 'request': body, 'reserved_usd': reserve,
                      'started_at': now(), 'status': status, 'attempts': attempts})
            if not is_rate_limit or retry == 8:
                raise RuntimeError(f'DeepSeek HTTP {exc.code}; response preserved') from None
            print(MODEL, 'RATE_LIMIT_REJECTION', key[:8], 'retry_in_seconds', 30, flush=True)
            time.sleep(30)
            with LOCK:
                state = gates.read(request_path)
                state['status'] = 'SENT_OUTCOME_PENDING'
                write(request_path, state)
    saved = {'model': MODEL, 'request_sha256': key, 'request': body, 'envelope': raw,
             'server_model': raw.get('model'), 'answer': ((raw.get('choices') or [{}])[0].get('message') or {}).get('content'),
             'seconds': time.monotonic() - started, 'received_at': now(),
             'prompt_byte_upper_bound': len(payload['prompt'].encode()) + 1024}
    with LOCK:
        write(target, saved)  # Persist before validating final JSON or accounting.
        log_response(saved)
        state = gates.read(request_path)
        state.update(status='RESPONSE_RECEIVED', reserved_usd=0, received_at=saved['received_at'])
        write(request_path, state)
    return saved


def prepare():
    """Preserve the completed Qwen preparation, change only model identity/transport."""
    if (OUT / 'MODEL_SWITCH.json').exists():
        validate_inputs()
        return
    if any((OUT / 'responses').glob('*.json')) or (OUT / 'prediction_freeze.json').exists():
        raise ValueError('Cannot switch a scored experiment in place')
    ready = gates.read(OUT / 'OFFLINE_READY.json')
    if ready['protocol_sha256'] != gates.sha(OUT / 'PROTOCOL.json') or ready['inputs_sha256'] != gates.sha(OUT / 'input_index.json'):
        raise ValueError('Previous offline preparation incomplete or changed')
    old = gates.read(OUT / 'PROTOCOL.json')
    index = gates.read(OUT / 'input_index.json')
    cases = gates.read(ROOT / 'cases.json')['cases']
    for c in cases:
        gates.check_case(c)
    if len(cases) != 300 or gates.digest(cases) != old['cases_hash']:
        raise ValueError('Unexpected prepared case set')
    manifest = gates.read(ROOT / 'data_manifest.json')
    if gates.sha(ROOT / 'data_manifest.json') != old['data_manifest_sha256']:
        raise ValueError('Prepared pool manifest changed')
    for name, expected in manifest['outputs'].items():
        if gates.sha(ROOT / name) != expected:
            raise ValueError('Prepared artifact hash mismatch: ' + name)
    audit = gates.read(OUT / 'full_graph_radius_audit.json')
    if audit['mismatches'] or audit['data_manifest_sha256'] != gates.sha(ROOT / 'data_manifest.json'):
        raise ValueError('Independent exact-radius audit not satisfied')
    archive = OUT / 'handoff_archive_qwen3.8-27b'
    archive.mkdir(exist_ok=True)
    for name in ('PROTOCOL.json', 'input_index.json', 'OFFLINE_READY.json', 'PRE_MODEL_HOLD.json', 'REPORT.md', 'TESTS_PASSED.json', 'FILES_CHANGED.json'):
        path = OUT / name
        if path.exists():
            (archive / name).write_bytes(path.read_bytes())
    converted, adjustments = {}, []
    for case in cases:
        items = [i for i in index['records'] if i['case_id'] == case['case_id']]
        payloads = {}
        for item in items:
            path = OUT / 'inputs' / item['input']
            if gates.sha(path) != item['input_sha256']:
                raise ValueError('Prepared prompt changed')
            payloads.setdefault(item['input'], gates.read(path))
        original_cap = next(iter(payloads.values()))['excerpt_chars']
        cap = original_cap
        def convert(payload):
            policy, p = payload['prompt'].split('\nINPUT\n', 1)
            p = json.loads(p)
            p.update(model=MODEL, excerpt_char_limit=cap)
            for evidence in p['evidence']:
                evidence['truncated'] = evidence['truncated'] or len(evidence['text']) > cap
                evidence['text'] = evidence['text'][:cap]
            return policy.replace('qwen3.8-27b', MODEL) + '\nINPUT\n' + json.dumps(p, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
        while max(len(convert(p).encode()) + 1024 for p in payloads.values()) >= 24000:
            cap -= 10
            if cap < 50:
                raise ValueError('Metadata alone exceeds prompt bound')
        if cap != original_cap:
            adjustments.append({'case_id': case['case_id'], 'old_cap': original_cap, 'new_cap': cap})
        for old_file, payload in payloads.items():
            prompt = convert(payload)
            request_body(prompt)
            payload.update(model=MODEL, prompt=prompt, prompt_sha256=gates.digest(prompt), excerpt_chars=cap)
            file = gates.digest(prompt) + '.json'
            write(OUT / 'inputs' / file, payload)
            converted[old_file] = (file, gates.sha(OUT / 'inputs' / file))
    for item in index['records']:
        item['input'], item['input_sha256'] = converted[item['input']]
    switch = {'previous_model': 'qwen3.8-27b', 'new_model': MODEL, 'decision_date': '2026-10-26',
              'reason': 'Explicit user decision: Reuters text may be sent to DeepSeek; engineering run, not look-ahead clean. Local Qwen is shut down; do not call it or wait for PID 23012.',
              'preserved': 'Completed exact-shortest pools, excluded own-entity aliases, controls, case set, seed, deterministic excerpts, source inspection, conservative continuity validator and audits.',
              'overwrite_reason': 'Replace pending Qwen protocol, prompt index, readiness and report with the authorized DeepSeek run; originals are preserved in handoff_archive_qwen3.8-27b.',
              'prepared_at': now(), 'previous_offline_ready': ready}
    switch['excerpt_cap_adjustments'] = adjustments
    switch['excerpt_cap_reason'] = 'Longer DeepSeek model identifier can push a boundary prompt over the conservative byte bound; reduce the same per-case cap by 10 characters across every arm/panel only when necessary.'
    protocol = {**old, 'model': MODEL, 'model_endpoint': URL, 'thinking': True,
                'max_completion_tokens': MAX_TOKENS, 'seed_note': 'Identical requested seed 20261026 and temperature=0; provider determinism is not guaranteed.',
                'fixed_policy': old['fixed_policy'].replace('qwen3.8-27b', MODEL),
                'retrieval': 'Preserved deterministic host-side Toolkit retrieval and source inspection; DeepSeek scores fixed excerpts. No native model tool-calling claim.',
                'model_switch': switch, 'budget_usd': CAP, 'cost_log': str(LEDGER),
                'prices_per_million_usd': {'cache_miss_input': PIN, 'cache_hit_input': PHIT, 'completion_including_reasoning': POUT},
                'pilot': '20 distinct requests, four per arm, deterministic prompt-byte quartiles without outcome selection; project cache-miss input plus arm-average completion usage before full run.',
                'deduplication': 'Identical full request bodies reuse one frozen response across arms/panels; no extra stochastic calls for zero evidence increments.',
                'network_policy': 'Only api.deepseek.com:443; proxies, redirects, local Qwen and all other APIs forbidden.',
                'execution_source_sha256': gates.sha(Path(__file__))}
    write(OUT / 'PROTOCOL.json', protocol)
    write(OUT / 'input_index.json', {**index, 'model': MODEL})
    write(OUT / 'OFFLINE_READY.json', {'protocol_sha256': gates.sha(OUT / 'PROTOCOL.json'), 'inputs_sha256': gates.sha(OUT / 'input_index.json'), 'unique_prompts': len(converted)})
    write(OUT / 'MODEL_SWITCH.json', switch)
    write(OUT / 'PRE_MODEL_HOLD.json', {'status': 'DeepSeek offline validation required before pilot', 'model_calls': 0, 'supersedes': 'Archived stale full_graph_radius_audit hold; that audit completed successfully.'})
    validate_inputs()


def validate_inputs():
    ready = gates.read(OUT / 'OFFLINE_READY.json')
    protocol = gates.read(OUT / 'PROTOCOL.json')
    if protocol['model'] != MODEL or ready['protocol_sha256'] != gates.sha(OUT / 'PROTOCOL.json') or ready['inputs_sha256'] != gates.sha(OUT / 'input_index.json'):
        raise ValueError('DeepSeek readiness mismatch')
    index = gates.read(OUT / 'input_index.json')['records']
    cases = gates.read(ROOT / 'cases.json')['cases']
    for c in cases:
        gates.check_case(c)
    if gates.digest(cases) != protocol['cases_hash']:
        raise ValueError('Case hash changed')
    expected = {(c['case_id'], mode, arm) for c in cases for mode in MODES for arm in ARMS}
    if {(i['case_id'], i['mode'], i['arm']) for i in index} != expected or len(index) != 3000:
        raise ValueError('Full matched 3000-record design required')
    unique = {}
    for item in index:
        path = OUT / 'inputs' / item['input']
        if gates.sha(path) != item['input_sha256']:
            raise ValueError('Input hash changed')
        if item['input'] not in unique:
            payload = gates.read(path)
            if payload['model'] != MODEL or gates.digest(payload['prompt']) != payload['prompt_sha256'] or 'qwen3.8-27b' in payload['prompt']:
                raise ValueError('Prompt identity mismatch')
            request_body(payload['prompt'])
            unique[item['input']] = payload
    # Strong cumulative-packet check, including unchanged own excerpts/controls.
    lookup = {(i['case_id'], i['mode'], i['arm']): i['input'] for i in index}
    order_changes = 0
    for c in cases:
        for mode in MODES:
            prior = None
            for arm in ARMS:
                p = unique[lookup[c['case_id'], mode, arm]]['prompt'].split('\nINPUT\n', 1)[1]
                p = json.loads(p)
                common = {k: v for k, v in p.items() if k not in ('evidence', 'graph_target', 'graph_pair_fields')}
                ev = [{k: v for k, v in e.items() if k != 'graph_pair'} for e in p['evidence']]
                if prior:
                    before = {e['source_id']: e for e in prior[1]}
                    after = {e['source_id']: e for e in ev}
                    if common != prior[0] or any(after.get(s) != e for s, e in before.items()):
                        raise ValueError('Cumulative packet or common controls changed')
                    order_changes += int(ev[:len(prior[1])] != prior[1])
                prior = (common, ev)
    write(OUT / 'DEEPSEEK_INPUT_CHECKS.json', {'records': len(index), 'unique_requests': len(unique), 'maximum_prompt_byte_bound': max(len(p['prompt'].encode()) + 1024 for p in unique.values()), 'cases_hash': gates.digest(cases), 'evaluation_accesses': 0, 'nonprefix_order_expansions': order_changes, 'order_note': 'Existing source contents remain identical; broader host inspection can interleave newly accessible sources. Original deterministic ordering preserved.', 'resources': gates.resources('deepseek_input_checks')})
    return index, unique


def pilot_selection(index, unique):
    selected, seen = [], set()
    for arm in ARMS:
        candidates = {}
        for i in index:
            if i['arm'] == arm and i['input'] not in seen:
                candidates.setdefault(i['input'], i)
        ordered = sorted(candidates.values(), key=lambda i: (len(unique[i['input']]['prompt'].encode()), i['case_id'], i['mode']))
        for q in (.125, .375, .625, .875):
            position = min(len(ordered) - 1, int(q * len(ordered)))
            item = ordered[position]
            selected.append(item)
            seen.add(item['input'])
    if len(seen) != 20:
        raise ValueError('Pilot must contain 20 distinct requests')
    return selected


def pilot(guard):
    index, unique = validate_inputs()
    tests = gates.read(OUT / 'TESTS_PASSED.json')
    if tests['model'] != MODEL or tests['exit_code'] != 0:
        raise ValueError('DeepSeek offline tests required before any model call')
    reconcile()
    selection = pilot_selection(index, unique)
    write(OUT / 'PILOT_PLAN.json', {'records': selection, 'selection': 'Four distinct prompt-byte quartiles per arm, deduplicated before calls; no labels used.'})
    results = []
    guard.resolve()
    def one(item):
        payload = unique[item['input']]
        saved = call(payload, guard)
        parse_answer(saved, payload['evidence_ids'])
        return item, saved
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(one, item) for item in selection]
        for future in futures:
            results.append(future.result())
            print(MODEL, 'PILOT', len(results), 'cost_usd', round(spent(), 6), flush=True)
    # Allocate every unique request to its first arm; use observed input-token / byte
    # ratios and arm-average output tokens, ignoring cache discounts in projection.
    by_arm = {a: [s for i, s in results if i['arm'] == a] for a in ARMS}
    assigned = {}
    for i in index:
        assigned.setdefault(i['input'], i['arm'])
    projection = 0.0
    details = {}
    for arm in ARMS:
        samples = by_arm[arm]
        ratio = sum(s['envelope']['usage']['prompt_tokens'] for s in samples) / sum(len(s['request']['messages'][0]['content'].encode()) for s in samples)
        output = sum(s['envelope']['usage']['completion_tokens'] for s in samples) / len(samples)
        inputs = sum(len(unique[k]['prompt'].encode()) * ratio for k, a in assigned.items() if a == arm)
        n = sum(a == arm for a in assigned.values())
        cost = (inputs * PIN + n * output * POUT) / 1e6
        projection += cost
        details[arm] = {'unique_requests': n, 'pilot_n': len(samples), 'input_tokens_per_byte': ratio, 'mean_completion_tokens': output, 'projected_usd': cost}
    value = {'pilot_calls': 20, 'pilot_actual_usd': sum(token_cost(s['envelope']['usage']) for _, s in results),
             'projected_full_usd': projection, 'projected_with_25pct_contingency_usd': projection * 1.25,
             'budget_usd': CAP, 'full_run_allowed': projection <= CAP, 'by_arm': details,
             'method': 'Full unique prompt universe, per-arm pilot token/byte ratios and mean reasoning+answer tokens; all input at cache-miss price. 25% contingency shown separately. Per-request worst-case reservations enforce hard cap regardless of projection.'}
    write(OUT / 'PILOT_PROJECTION.json', value)
    print(MODEL, 'PILOT_PROJECTION', json.dumps(value), flush=True)
    if projection > CAP:
        raise RuntimeError('Pilot projects more than $15; full run forbidden')
    return index, unique


def predict(guard, workers=8):
    if (OUT / 'prediction_freeze.json').exists():
        gates.verify_freeze(OUT, gates.read(ROOT / 'cases.json')['cases'])
        return
    index, unique = pilot(guard)
    start = time.monotonic()
    completed = 0
    def task(pair):
        file, payload = pair
        saved = call(payload, guard)
        parse_answer(saved, payload['evidence_ids'])
        return file, saved['request_sha256']
    request_keys = {}
    # Bound the live queue, refill on completion, and drain without new calls
    # after any error. A slow reasoning response does not stall every worker.
    items = list(unique.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        queue = iter(items)
        pending = {executor.submit(task, item) for item in [next(queue, None) for _ in range(workers)] if item is not None}
        errors = []
        last_progress = 0
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    file, key = future.result()
                    request_keys[file] = key
                    completed += 1
                except Exception as exc:
                    errors.append(exc)
            if completed - last_progress >= 10 or not pending:
                write(OUT / 'progress.json', {'stage': 'deepseek_predictions', 'completed_unique_requests': completed, 'total_unique_requests': len(unique), 'actual_cost_usd': spent(), 'seconds': time.monotonic() - start, 'errors_draining': len(errors)})
                print(MODEL, 'PREDICTIONS', completed, '/', len(unique), 'cost_usd', round(spent(), 6), flush=True)
                gates.resources('deepseek_predictions')
                last_progress = completed
            if not errors:
                for _ in done:
                    item = next(queue, None)
                    if item is not None:
                        pending.add(executor.submit(task, item))
        if errors:
            raise errors[0]
    records = []
    for item in index:
        key = request_keys[item['input']]
        saved = gates.read(OUT / 'responses' / (key + '.json'))
        answer = parse_answer(saved, unique[item['input']]['evidence_ids'])
        file = gates.digest({k: item[k] for k in ('case_id', 'mode', 'arm')}) + '.json'
        write(OUT / 'predictions' / file, {**item, 'score': float(answer['score']), 'request_sha256': key,
              'server_model': MODEL, 'prediction': answer, 'usage': saved['envelope']['usage'],
              'response_file_sha256': gates.sha(OUT / 'responses' / (key + '.json'))})
        records.append({k: item[k] for k in ('case_id', 'mode', 'arm')} | {'file': file, 'sha256': gates.sha(OUT / 'predictions' / file)})
    cases = gates.read(ROOT / 'cases.json')['cases']
    write(OUT / 'prediction_freeze.json', {'cases_hash': gates.digest(cases), 'protocol_sha256': gates.sha(OUT / 'PROTOCOL.json'),
          'execution_code_sha256': gates.sha(Path(__file__)), 'frozen_at': now(), 'actual_cost_usd': spent(), 'predictions': records})
    write(OUT / 'prediction_freeze.sha256.json', {'sha256': gates.sha(OUT / 'prediction_freeze.json')})
    gates.verify_freeze(OUT, cases)
    print(MODEL, 'ALL_3000_PREDICTIONS_FROZEN', 'cost_usd', spent(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'pilot', 'predict', 'report'), required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--resume-after-balance', action='store_true', help='Use only after the API account has been replenished; never changes the $15 job cap')
    args = ap.parse_args()
    guard = DeepSeekGuard()
    guard.install()
    from filelock import FileLock
    with FileLock(str(OUT / 'deepseek_run.lock'), timeout=0):
        if args.resume_after_balance:
            write(OUT / 'ACCOUNT_BALANCE_BLOCK.json', {'status': 'RESUME_REQUESTED', 'at': now(), 'reason': 'Explicit --resume-after-balance invocation after account replenishment; original job cap retained.'})
        if args.stage == 'prepare':
            prepare()
        elif args.stage == 'pilot':
            pilot(guard)
        elif args.stage == 'predict':
            predict(guard, args.workers)
        else:
            from .hop3_deepseek_report import report
            report(guard)


if __name__ == '__main__':
    main()
