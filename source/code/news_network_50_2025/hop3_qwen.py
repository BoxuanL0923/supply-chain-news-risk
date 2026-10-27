"""Sequential Qwen-only engineering rerun. All labels stay gated by full freeze."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request

from . import hop3_guard as gates
from .config import CONTROLS

MODEL = gates.MODEL
ROOT = gates.ROOT
PREP = ROOT / 'hop_ablation_v2'
OUT = PREP / 'qwen'
TARGET = gates.REPORT
DONE = ROOT / 'QWEN_RUN_DONE_v2.json'
SERVER_LOG = Path(os.environ.get('HOP3_QWEN_SERVER_LOG', gates.PROJECT/'data/qwen/server.log')).resolve()
ARMS = ('OWN', 'H1', 'H2', 'H3', 'H3_GRAPH')
MODES = ('all_paths', 'continuity_only')
MAX_TOKENS = 7000
URL = 'http://127.0.0.1:8081/v1/chat/completions'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    """Atomic, flushed artifacts; the completion signal is written last."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump({'model': MODEL, **value}, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise PermissionError('Redirects forbidden; loopback endpoint only')


class QwenGuard(gates.Guard):
    def __init__(self):
        super().__init__()
        self.network_requests = 0

    def install(self):
        super().install()
        def hook(event, args):
            if event == 'socket.connect':
                self.network_requests += 1
        sys.addaudithook(hook)


def request_body(prompt):
    bound = len(prompt.encode('utf-8')) + 1024
    if bound >= 24000 or bound + MAX_TOKENS > 32768:
        raise ValueError('Prompt/completion exceeds frozen context bound')
    return {'model': MODEL, 'temperature': 0, 'seed': 20261026,
            'max_tokens': MAX_TOKENS,
            'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': prompt}]}


def parse_answer(saved, evidence_ids):
    raw = saved['envelope']
    if saved['server_model'] != MODEL or raw.get('model') != MODEL:
        raise ValueError('Mixed or unexpected server model')
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Incomplete final answer; cached, no automatic retry')
    answer = json.loads(choice['message']['content'])
    if type(answer.get('score')) not in (int, float) or not math.isfinite(answer['score']):
        raise ValueError('Invalid finite JSON score')
    ids = answer.get('evidence_ids')
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError('Invalid evidence_ids structure')
    if not isinstance(answer.get('reason'), str):
        raise ValueError('Missing JSON explanation')
    # Numeric prediction validity and citation quality are separate, declared
    # before inference. Preserve every finite score; never select cases by IDs.
    issues, aliases, resolved, unresolved = [], [], [], []
    if len(ids) > 8:
        issues.append('more_than_eight_returned_ids')
    if len(set(ids)) != len(ids):
        issues.append('duplicate_returned_ids')
    for sid in ids:
        if sid in evidence_ids:
            resolved.append(sid)
            continue
        parts = sid.split(':')
        candidates = [e for e in evidence_ids if e.startswith(sid + ':')] if len(parts) == 3 else []
        kind = 'omitted_hash_suffix'
        if sid.startswith('RTRS:') and len(parts) == 4:
            candidates = [e for e in evidence_ids if e.split(':')[0] == 'RTRS' and e.split(':')[2:] == parts[2:]]
            kind = 'month_typo_exact_line_and_hash'
        if sid.startswith('RTRS:') and len(candidates) == 1:
            resolved.append(candidates[0])
            aliases.append({'returned_id': sid, 'canonical_id': candidates[0], 'kind': kind})
        else:
            unresolved.append(sid)
    if len(set(resolved)) != len(resolved):
        issues.append('duplicate_resolved_ids')
    if unresolved:
        issues.append('unresolved_or_ineligible_returned_ids')
    answer.update(returned_evidence_ids=ids, evidence_ids=list(dict.fromkeys(resolved)),
                  unresolved_evidence_ids=unresolved, evidence_id_alias_resolution=aliases,
                  citation_validation_issues=issues)
    usage = raw['usage']
    p, c = usage['prompt_tokens'], usage['completion_tokens']
    if type(p) is not int or type(c) is not int or not 0 < p < 24000 or not 0 < c <= MAX_TOKENS or p + MAX_TOKENS > 32768:
        raise ValueError('Invalid or excessive actual token usage')
    return answer


def server_log_tail():
    with SERVER_LOG.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 32768))
        tail = f.read().decode('utf-8', errors='replace')
    return size, tail


def check_server_log():
    _, tail = server_log_tail()
    if re.search(r'CUDA error|CUDA out of memory|CUDA_ERROR|cudaMalloc.*failed', tail, re.I):
        raise RuntimeError('CUDA failure in server.log; stop without server intervention')


class QwenClient:
    """Same call interface as the local runtime, with durable raw receipts."""
    def __init__(self, out=OUT):
        self.out = Path(out)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(self, prompt, purpose='score', nonce='', retry=False):
        body = request_body(prompt)
        key = gates.digest(body)
        target = self.out / 'responses' / (key + '.json')
        if target.exists():
            saved = gates.read(target)
            if saved['request_sha256'] != key or gates.digest(saved['request']) != key or saved['request'] != body:
                raise ValueError('Cached response request mismatch')
            raw_path = self.out / 'raw_responses' / (key + '.json')
            if gates.sha(raw_path) != saved['raw_response_sha256']:
                raise ValueError('Cached raw response changed')
            return saved
        pending = self.out / 'requests' / (key + '.json')
        if pending.exists():
            raise RuntimeError('Prior request has no validated receipt; do not repeat unknown outcome')
        check_server_log()
        gates.resources('qwen_request')
        write(pending, {'request_sha256': key, 'request': body, 'status': 'SENT_OUTCOME_PENDING', 'started_at': now()})
        started = time.monotonic()
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(req, timeout=300) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            write(self.out / 'http_errors' / (key + '.json'),
                  {'status': exc.code, 'body': exc.read().decode('utf-8', errors='replace'), 'at': now()})
            raise
        # Persist bytes BEFORE JSON or score validation, including malformed data.
        raw_path = self.out / 'raw_responses' / (key + '.json')
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open('wb') as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())
        raw = json.loads(raw_bytes)
        saved = {'request_sha256': key, 'request': body, 'envelope': raw,
                 'server_model': raw.get('model'), 'received_at': now(),
                 'seconds': time.monotonic() - started, 'raw_response_sha256': gates.sha(raw_path),
                 'prompt_byte_upper_bound': len(prompt.encode()) + 1024}
        write(target, saved)
        check_server_log()
        return gates.read(target)


def deepseek_snapshot():
    """Hash only, never import scores into the Qwen panel."""
    files = [p for folder in ('requests', 'responses', 'partial_predictions_deepseek-v4-pro') for p in (PREP / folder).glob('*') if p.is_file()]
    files += [p for p in PREP.glob('*') if p.is_file()]
    if (ROOT / 'deepseek_costs_hop3.csv').exists():
        files += [ROOT / 'deepseek_costs_hop3.csv']
    return {str(p.relative_to(ROOT)): gates.sha(p) for p in sorted(files)}


def prepare():
    from .hop3_v2 import POLICY, GRAPH_POLICY
    archive = PREP / 'handoff_archive_qwen3.8-27b'
    # A fresh Qwen-only run uses the original label-free preparation directly;
    # the historical rerun uses its archive after the DeepSeek conversion.
    if not archive.exists():
        archive = PREP
    if (OUT / 'PROTOCOL.json').exists():
        return validate_inputs()
    if any((OUT / 'responses').glob('*.json')):
        raise ValueError('Do not rebuild a Qwen protocol after responses')
    gates.resources('qwen_prepare')
    cases = gates.read(ROOT / 'cases.json')['cases']
    for c in cases:
        gates.check_case(c)
    manifest = gates.read(ROOT / 'data_manifest.json')
    if not manifest['prefix_support_complete']:
        raise ValueError('Incomplete pool support')
    for name, expected in manifest['outputs'].items():
        if gates.sha(ROOT / name) != expected:
            raise ValueError('Prepared source changed: ' + name)
    radius = gates.read(PREP / 'full_graph_radius_audit.json')
    if radius['mismatches'] or radius['data_manifest_sha256'] != gates.sha(ROOT / 'data_manifest.json'):
        raise ValueError('Exact radius audit failed')
    original_ready = gates.read(archive / 'OFFLINE_READY.json')
    if original_ready['protocol_sha256'] != gates.sha(archive / 'PROTOCOL.json') or original_ready['inputs_sha256'] != gates.sha(archive / 'input_index.json'):
        raise ValueError('Archived Qwen preparation hash mismatch')
    protocol = gates.read(archive / 'PROTOCOL.json')
    if protocol['fixed_policy'] != POLICY or protocol['graph_policy'] != GRAPH_POLICY:
        raise ValueError('Original Qwen policy changed')
    index = gates.read(archive / 'input_index.json')
    for name in {i['input'] for i in index['records']}:
        source = PREP / 'inputs' / name
        if not source.exists():
            raise ValueError('Original Qwen prompt missing: ' + name)
        target = OUT / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write(OUT / 'input_index.json', index)
    snapshot = deepseek_snapshot()
    write(OUT / 'DEEPSEEK_PRESERVED.json', {'files': snapshot,
          'distinct_responses': len(list((PREP / 'responses').glob('*.json'))),
          'last_progress_requests': gates.read(PREP / 'progress.json')['completed_unique_requests'] if (PREP / 'progress.json').exists() else 0,
          'old_partial_freeze_cells': len(gates.read(PREP / 'PARTIAL_PREDICTION_FREEZE_deepseek-v4-pro.json')['predictions']) if (PREP / 'PARTIAL_PREDICTION_FREEZE_deepseek-v4-pro.json').exists() else 0,
          'expected_requests': 1778, 'complete': False,
          'decision': 'Preserve any prior DeepSeek artifacts by hash; no combined-model analysis.'})
    protocol.update(version='hop3-ablation-v2-qwen-only-rerun', model=MODEL, thinking=False,
        max_completion_tokens=MAX_TOKENS, context_tokens=32768, concurrency=1,
        network_policy='Only 127.0.0.1:8081; no proxy, redirects or paid APIs',
        seed_note='Same seed and temperature for every arm; bitwise backend determinism not guaranteed',
        deduplication='Identical request bodies share one response across arms/panels; zero increments stay exactly zero',
        citation_quality_policy='Every finite final-JSON score retained. Exact unique month/line or line/hash ID aliases allowed; unresolved, duplicate or >8 IDs are quality flags, never verified evidence or case exclusions.',
        failure_policy='Cache response bytes first; stop on connection failure, timeout, CUDA failure, malformed score or unknown prior request; never restart/reconfigure/kill server',
        model_switch={'from': 'deepseek-v4-pro' if archive != PREP else None, 'to': MODEL,
          'reason': 'Separate Qwen-only panel; every arm and case scored from Qwen prompts. Prior model responses are never reused.',
          'prompts': 'Original Qwen-preparation prompts restored byte-for-byte, including original common per-case excerpt caps'},
        inference='HC3; CR1 effective-rank correction; two-sided 90% t intervals; g3 bound is sensitivity, not prespecified equivalence',
        execution_source_sha256=gates.sha(Path(__file__)),
        analysis_source_sha256=gates.sha(Path(__file__).with_name('hop3_analysis.py')),
        guard_source_sha256=gates.sha(Path(gates.__file__)),
        preparation_audits_sha256={n: gates.sha(PREP / n) for n in ('scope_audit.json', 'continuity_summary.json', 'continuity_source_audit.json', 'full_graph_radius_audit.json')})
    write(OUT / 'PROTOCOL.json', protocol)
    write(OUT / 'MODEL_SWITCH.json', protocol['model_switch'])
    write(OUT / 'OFFLINE_READY.json', {'protocol_sha256': gates.sha(OUT / 'PROTOCOL.json'), 'inputs_sha256': gates.sha(OUT / 'input_index.json')})
    return validate_inputs()


def validate_inputs():
    ready, protocol = (gates.read(OUT / n) for n in ('OFFLINE_READY.json', 'PROTOCOL.json'))
    if protocol['model'] != MODEL or ready['protocol_sha256'] != gates.sha(OUT / 'PROTOCOL.json') or ready['inputs_sha256'] != gates.sha(OUT / 'input_index.json'):
        raise ValueError('Qwen readiness/hash mismatch')
    if protocol['execution_source_sha256'] != gates.sha(Path(__file__)) or protocol['analysis_source_sha256'] != gates.sha(Path(__file__).with_name('hop3_analysis.py')) or protocol['guard_source_sha256'] != gates.sha(Path(gates.__file__)):
        raise ValueError('Execution/analysis/guard code changed since Qwen protocol')
    for name, expected in protocol['preparation_audits_sha256'].items():
        if gates.sha(PREP / name) != expected:
            raise ValueError('Preparation audit changed')
    cases = gates.read(ROOT / 'cases.json')['cases']
    for c in cases:
        gates.check_case(c)
    if len(cases) != 300 or gates.digest(cases) != protocol['cases_hash']:
        raise ValueError('Case set changed')
    index = gates.read(OUT / 'input_index.json')['records']
    expected = {(c['case_id'], m, a) for c in cases for m in MODES for a in ARMS}
    if len(index) != 3000 or {(i['case_id'], i['mode'], i['arm']) for i in index} != expected:
        raise ValueError('Incomplete or duplicate design')
    unique = {}
    for item in index:
        path = OUT / 'inputs' / item['input']
        if gates.sha(path) != item['input_sha256']:
            raise ValueError('Input bytes changed')
        if item['input'] not in unique:
            p = gates.read(path)
            if p['model'] != MODEL or gates.digest(p['prompt']) != p['prompt_sha256'] or 'deepseek-v4-pro' in p['prompt']:
                raise ValueError('Prompt identity mismatch')
            request_body(p['prompt'])
            packet = json.loads(p['prompt'].split('\nINPUT\n', 1)[1])
            if packet['model'] != MODEL or packet['case_id'] != item['case_id'] or [e['source_id'] for e in packet['evidence']] != p['evidence_ids']:
                raise ValueError('Prompt packet identity mismatch')
            unique[item['input']] = p
    lookup = {(i['case_id'], i['mode'], i['arm']): i['input'] for i in index}
    reorder = 0
    for c in cases:
        for mode in MODES:
            prior = None
            for arm in ARMS:
                p = json.loads(unique[lookup[c['case_id'], mode, arm]]['prompt'].split('\nINPUT\n', 1)[1])
                common = {k: v for k, v in p.items() if k not in ('evidence', 'graph_target', 'graph_pair_fields')}
                ev = [{k: v for k, v in e.items() if k != 'graph_pair'} for e in p['evidence']]
                if prior:
                    after = {e['source_id']: e for e in ev}
                    if common != prior[0] or any(after.get(e['source_id']) != e for e in prior[1]):
                        raise ValueError('Common controls or cumulative source content changed')
                    reorder += int(ev[:len(prior[1])] != prior[1])
                prior = (common, ev)
    write(OUT / 'INPUT_CHECKS.json', {'records': len(index), 'unique_requests': len(unique),
          'maximum_prompt_byte_bound': max(len(p['prompt'].encode()) + 1024 for p in unique.values()),
          'nonprefix_order_expansions': reorder, 'evaluation_accesses': 0})
    return index, unique


def verify_complete():
    cases = gates.read(ROOT / 'cases.json')['cases']
    frozen = gates.verify_freeze(OUT, cases)
    index, unique = validate_inputs()
    client = QwenClient()
    quality = {}
    expected = {(i['case_id'], i['mode'], i['arm']): i for i in index}
    for item in frozen['predictions']:
        p = gates.read(OUT / 'predictions' / item['file'])
        original = expected[item['case_id'], item['mode'], item['arm']]
        if any(p[k] != v for k, v in original.items()) or p['model'] != MODEL or p['server_model'] != MODEL:
            raise ValueError('Prediction differs from its matched input cell')
        payload = unique[p['input']]
        request_key = gates.digest(request_body(payload['prompt']))
        if not (OUT / 'responses' / (request_key + '.json')).exists():
            raise ValueError('Complete freeze has a missing response; no new calls allowed')
        saved = client.call(payload['prompt'])  # Must already exist; no calls allowed here.
        if saved['request_sha256'] != p['request_sha256'] or gates.sha(OUT / 'responses' / (p['request_sha256'] + '.json')) != p['response_file_sha256']:
            raise ValueError('Frozen response changed')
        answer = parse_answer(saved, payload['evidence_ids'])
        if p['score'] != float(answer['score']) or p['prediction'] != answer:
            raise ValueError('Frozen score differs from response')
        quality[p['request_sha256']] = {'issues': answer['citation_validation_issues'],
            'unresolved': answer['unresolved_evidence_ids'], 'aliases': answer['evidence_id_alias_resolution']}
    if len(quality) != len(unique) or len(list((OUT / 'responses').glob('*.json'))) != len(unique):
        raise ValueError('Missing or extra Qwen responses')
    return frozen, quality


def predict():
    if not (OUT / 'TESTS_PASSED.json').exists() or gates.read(OUT / 'TESTS_PASSED.json')['exit_code'] != 0:
        raise ValueError('Passing offline tests required before calls')
    index, unique = validate_inputs()
    if (OUT / 'prediction_freeze.json').exists():
        return verify_complete()
    client, records, started = QwenClient(), [], time.monotonic()
    for n, item in enumerate(index, 1):
        payload = unique[item['input']]
        saved = client.call(payload['prompt'])
        answer = parse_answer(saved, payload['evidence_ids'])
        key = saved['request_sha256']
        file = gates.digest({k: item[k] for k in ('case_id', 'mode', 'arm')}) + '.json'
        write(OUT / 'predictions' / file, {**item, 'score': float(answer['score']), 'server_model': saved['server_model'],
              'request_sha256': key, 'prediction': answer, 'response_file_sha256': gates.sha(OUT / 'responses' / (key + '.json'))})
        records.append({k: item[k] for k in ('case_id', 'mode', 'arm')} | {'file': file, 'sha256': gates.sha(OUT / 'predictions' / file)})
        if n % 10 == 0:
            count = len(list((OUT / 'responses').glob('*.json')))
            write(OUT / 'progress.json', {'stage': 'predictions', 'completed_records': n, 'total_records': len(index),
                  'unique_responses': count, 'total_unique_requests': len(unique), 'seconds_this_session': time.monotonic() - started,
                  'last_case': item['case_id'], 'updated_at': now()})
            print(MODEL, 'PREDICTIONS', n, '/', len(index), 'unique', count, '/', len(unique), flush=True)
    write(OUT / 'prediction_freeze.json', {'cases_hash': gates.digest(gates.read(ROOT / 'cases.json')['cases']),
          'protocol_sha256': gates.sha(OUT / 'PROTOCOL.json'), 'frozen_at': now(), 'predictions': records})
    write(OUT / 'prediction_freeze.sha256.json', {'sha256': gates.sha(OUT / 'prediction_freeze.json')})
    return verify_complete()


def load_labels(guard, split):
    """Guard native Arrow reads explicitly; source files never copied or mixed."""
    if split not in ('train', 'valid'):
        raise ValueError('Evaluation labels forbidden')
    cases = gates.read(ROOT / 'cases.json')['cases']
    gates.verify_freeze(OUT, cases)
    prelabel = gates.read(OUT / 'PRE_LABEL_VERIFICATION.json')
    if not prelabel['response_hashes_and_scores_verified'] or prelabel['prediction_freeze_sha256'] != gates.sha(OUT / 'prediction_freeze.json'):
        raise ValueError('Complete response verification required')
    guard.labels_allowed = True
    path = gates.LIVE / ('labels_' + split + '.parquet')
    guard.check_path(path)
    manifest = gates.read(ROOT / 'label_source_manifest.json')
    if gates.sha(path) != manifest['expected_sha256'][path.name]:
        raise ValueError('Development label source changed')
    import pandas as pd
    frame = pd.read_parquet(path)
    allowed = {c['case_id'] for c in cases if c['split'] == split}
    if set(frame.case_id) != allowed or frame.case_id.duplicated().any() or set(frame['split']) != {split} or any(str(d)[:10] not in gates.SPLITS for d in frame.date):
        raise ValueError('Unexpected development label cases')
    return frame[['case_id', 'y']]


def report(guard):
    import numpy as np
    import pandas as pd
    from .hop3_analysis import analyze
    gates.resources('qwen_analysis')
    frozen, quality = verify_complete()
    prelabel = now()
    write(OUT / 'PRE_LABEL_VERIFICATION.json', {'verified_at': prelabel, 'predictions': 3000,
          'unique_responses': len(quality), 'labels_opened': False, 'response_hashes_and_scores_verified': True,
          'prediction_freeze_sha256': gates.sha(OUT / 'prediction_freeze.json')})
    labels = pd.concat([load_labels(guard, s) for s in ('train', 'valid')], ignore_index=True)
    base = []
    for day in gates.DATES:
        batch = gates.read(ROOT / 'pools' / (day + '.json'))
        for pool in batch['pools'].values():
            c = gates.check_case(pool['case'])
            base.append({'model': MODEL, 'case_id': c['case_id'], 'split': c['split'], 'date': day, 'ticker': c['ticker'],
                         'anchor': c['b'], **{k: pool['controls'][k] for k in CONTROLS},
                         **{f'candidate_exact_h{h}': v for h, v in pool['coverage']['candidate_exact_hops'].items()}})
    frames, analyses = {}, {}
    for mode in MODES:
        frame = pd.DataFrame(base).merge(labels, on='case_id', validate='one_to_one')
        for arm in ARMS:
            predictions = {r['case_id']: gates.read(OUT / 'predictions' / r['file']) for r in frozen['predictions'] if r['mode'] == mode and r['arm'] == arm}
            frame[arm] = frame.case_id.map({c: p['score'] for c, p in predictions.items()})
            frame[arm + '_citation_flagged'] = frame.case_id.map({c: bool(p['prediction']['citation_validation_issues']) for c, p in predictions.items()})
        frame['mode'] = mode
        frames[mode], analyses[mode] = frame, analyze(frame)
    continuity = frames['continuity_only']
    if not np.array_equal(continuity.H1, continuity.H2) or not np.array_equal(continuity.H2, continuity.H3):
        raise ValueError('Continuity-only increments should be exactly zero')
    matched = frames['all_paths'].copy()
    scope = gates.read(PREP / 'scope_audit.json')['records']
    for r in scope:
        for h in (1, 2, 3):
            matched.loc[matched.case_id == r['case_id'], f"H{r['radius']}_accepted_exact_h{h}"] = r[f'accepted_exact_h{h}']
    for arm in ARMS:
        matched['continuity_' + arm] = continuity[arm].to_numpy()
    matched[matched['split'] == 'valid'].to_csv(OUT / 'matched_validation.csv', index=False)
    matched.to_csv(OUT / 'matched_development_qwen3.8-27b.csv', index=False)
    responses = [gates.read(p) for p in (OUT / 'responses').glob('*.json')]
    preserved = gates.read(OUT / 'DEEPSEEK_PRESERVED.json')
    if preserved['files'] != deepseek_snapshot():
        raise ValueError('DeepSeek partial artifacts changed during Qwen run')
    summary = {'status': 'COMPLETE', 'engineering_only': True, 'look_ahead_clean': False, 'train_n': 200, 'valid_n': 100,
        'evaluation_cases_opened': False, 'evaluation_labels_opened': False, 'prediction_records': len(frozen['predictions']),
        'unique_model_calls': len(responses), 'server_models': sorted({r['server_model'] for r in responses}),
        'maximum_prompt_tokens': max(r['envelope']['usage']['prompt_tokens'] for r in responses),
        'maximum_prompt_byte_bound': max(r['prompt_byte_upper_bound'] for r in responses),
        'maximum_prompt_plus_max_tokens': max(r['envelope']['usage']['prompt_tokens'] + MAX_TOKENS for r in responses),
        'total_model_seconds': sum(r['seconds'] for r in responses),
        'usage_totals': {k: sum(r['envelope']['usage'][k] for r in responses) for k in ('prompt_tokens', 'completion_tokens')},
        'analysis': analyses, 'coverage': gates.read(ROOT / 'coverage_audit.json')['by_split'],
        'continuity': gates.read(PREP / 'continuity_summary.json'),
        'inspected_h3_coverage': {s: {'sources': sum(r['accepted_exact_h3'] for r in scope if r['split'] == s and r['radius'] == 3),
          'cases': sum(r['accepted_exact_h3'] > 0 for r in scope if r['split'] == s and r['radius'] == 3)} for s in ('train', 'valid')},
        'citation_quality': {'unique_responses': len(quality), 'flagged': sum(bool(q['issues']) for q in quality.values()),
          'with_exact_aliases': sum(bool(q['aliases']) for q in quality.values()), 'with_unresolved': sum(bool(q['unresolved']) for q in quality.values())},
        'deepseek_partial': {k: v for k, v in preserved.items() if k not in ('files', 'model')} | {'model': 'deepseek-v4-pro', 'unchanged_hashes': True},
        'prediction_freeze_sha256': gates.sha(OUT / 'prediction_freeze.json'), 'frozen_at': frozen['frozen_at'],
        'pre_label_verification_at': prelabel, 'checks': gates.read(OUT / 'TESTS_PASSED.json'),
        'resources': gates.resources('qwen_report_complete')}
    write(OUT / 'citation_quality_audit.json', {'records': quality, **summary['citation_quality']})
    write(OUT / 'summary.json', summary)
    write(OUT / 'data_access_audit.json', {'live_files_opened': sorted(guard.opened), 'evaluation_accesses': 0,
          'labels_opened_after': prelabel, 'prediction_freeze_sha256': summary['prediction_freeze_sha256'],
          'network_requests_in_runner': guard.network_requests})
    text = render_report(summary)
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding='utf-8')
    write(OUT / 'FINAL_VERIFICATION.json', {'report_sha256': gates.sha(OUT / 'REPORT.md'),
          'copy_sha256': gates.sha(TARGET), 'deepseek_unchanged': True, 'at': now()})
    print(MODEL, json.dumps({'valid_raw_mse': analyses['all_paths']['raw_residual_mse']['valid'],
          'g3': analyses['all_paths']['pooled_train_valid']['coefficients']['H3_minus_H2'], 'report': str(TARGET)}), flush=True)


def fmt(v):
    if v is None:
        return 'unidentified'
    if isinstance(v, (tuple, list)):
        return '[' + ', '.join(fmt(x) for x in v) + ']'
    return f'{v:.6g}'


def render_report(s):
    a = s['analysis']
    lines = ['# Cumulative hop-radius ablation - qwen3.8-27b', '',
        '**COMPLETE Qwen-only engineering run. Qwen3.8-27B is not look-ahead clean.** 50 firms; 200 train cases on October 1/8/15/22, 2025; 100 validation cases on October 30 and November 6. All five arms and both path panels use the same model. Evaluation cases and labels remain sealed.', '',
        'This is the fresh second Qwen attempt: every cell was rescored with a uniform 7,000-token completion allowance. Attempt 1 stopped on one truncated answer at its 700-token cap; its 116 response receipts and 191 predictions remain archived in `../qwen_attempt_01_m700_20261026/`. No attempt-1 response was reused. The attempt-1 failure signal is `QWEN_RUN_DONE_attempt01.json`; this runner writes `QWEN_RUN_DONE_v2.json` only after its reports and audit artifacts.', '',
        '| Arm | Train raw residual MSE | Valid raw residual MSE | Continuity train MSE | Continuity valid MSE |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        values = [a[m]['raw_residual_mse'][split][arm] for m in MODES for split in ('train', 'valid')]
        lines.append('| ' + arm + ' | ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines += ['', 'Raw residual MSE = mean((y - raw GARCH anchor - model score)^2). Zero-score baselines: ' +
        ', '.join(f"{split} {fmt(a['all_paths']['zero_score_mse'][split])}" for split in ('train', 'valid')) + '.', '',
        'The anchor is frozen outcome-free raw GARCH, rather than PR2\'s development-fitted OLS anchor. All 11 conventional controls enter the post-freeze regression. This preserves the requirement to make every forecast before opening labels.', '',
        'Nested equation: y = intercept + X b + g_own OWN + g1 H1 + g2(H2-H1) + g3(H3-H2) + e.', '',
        '| Panel | Controls + OWN valid MSE | + H1 | + (H2-H1) | + (H3-H2) |', '|---|---:|---:|---:|---:|']
    for m in MODES:
        lines.append('| ' + m + ' | ' + ' | '.join(fmt(n['valid_mse']) for n in a[m]['train_fitted_valid']['nested']) + ' |')
    lines += ['', 'These coefficients are fitted only on the four train dates and applied unchanged to the two validation dates. Pooled estimates below use all six train+valid dates for exploratory inference.', '',
        '| Panel | Pooled term | Estimate | HC3 SE | HC3 90% CI | Date-cluster SE | Date-cluster 90% CI |',
        '|---|---|---:|---:|---|---:|---|']
    for m in MODES:
        for term, co in a[m]['pooled_train_valid']['coefficients'].items():
            lines.append(f"| {m} | {term} | {fmt(co['estimate'])} | {fmt(co['HC3']['se'])} | {fmt(co['HC3']['ci90'])} | {fmt(co['date_cluster']['se'])} | {fmt(co['date_cluster']['ci90'])} |")
    lines += ['', 'HC3 uses residual effective-rank degrees of freedom. CR1 uses G/(G-1)*(n-1)/(n-rank), with t intervals on five date degrees of freedom. Redundant controls or score contrasts are marked unidentified. Six date clusters make the clustered intervals fragile. Full train fits, nested ranks, validation predictions and pooled outputs are in `summary.json` and the matched CSVs.', '',
        '| Panel | Smallest symmetric g3 bound, HC3 | Date-cluster bound | Partial SD of H3-H2 |', '|---|---:|---:|---:|']
    for m in MODES:
        p = a[m]['pooled_train_valid']
        b = p['gamma3_smallest_symmetric_equivalence_bound']
        lines.append(f"| {m} | {fmt(b['HC3'])} | {fmt(b['date_cluster'])} | {fmt(p['power']['partial_increment_sd'])} |")
    lines += ['', 'The bound is max(abs(lower 90% CI), abs(upper 90% CI)), in log variance per unit increment score. A strictly larger symmetric margin would satisfy the interval criterion. This is observed design sensitivity; no practical equivalence margin was prespecified.', '',
        '| Panel | Assumed g3 | RMS log-risk effect | Cases for 80% power, HC3 | Cases for 80% power, new date clusters |', '|---|---:|---:|---:|---:|']
    for m in MODES:
        for p in a[m]['pooled_train_valid']['power']['grid']:
            lines.append(f"| {m} | {p['gamma3_effect']} | {fmt(p['rms_log_risk_effect'])} | {fmt(p['HC3_cases_for_80pct_power'])} | {fmt(p['date_cluster_cases_for_80pct_power'])} |")
    lines += ['', 'Power is the local normal approximation n_new=n*((z_.95+z_.80)*SE/effect)^2, alpha=.10 two-sided. It assumes stable signal/design/noise. The cluster projection rounds to 50 cases per new independent date; extra firms on the same dates do not create independent time variation. These are exploratory extrapolations, not finite-sample guarantees.', '',
        'Continuity-only H1=H2=H3 exactly: the preparation supplies no complete verified multi-hop product chain. Thus g2/g3, their bounds and their power sample sizes are unidentified. Direct paths are retained without claiming verified product exposure. Missing continuity evidence does not establish absence of economic transmission. H3_GRAPH can differ through graph instructions and direct-pair numerical receipts.', '',
        '| Split | Exact H1 candidates | Exact H2 | Exact H3 | Any H3 candidate | Accessible H3 sources | Cases with accessible H3 |', '|---|---:|---:|---:|---:|---:|---:|']
    for split, cov in s['coverage'].items():
        i = s['inspected_h3_coverage'][split]
        lines.append(f"| {split} | {cov['candidate_counts']['1']} | {cov['candidate_counts']['2']} | {cov['candidate_counts']['3']} | {cov['share_any_h3']:.1%} | {i['sources']} | {i['cases']} |")
    lines += ['', 'Prepared 8/6/6 reserves fill to 20 network candidates using outcome-free frozen MiniLM and recency. Exact shortest distances use full-graph BFS; own-entity aliases and co-mentions are excluded. One deterministic shortest witness per mapped entity and its prefixes are retained. The original capped-enumeration preflight had 930 false H3 rows and two own aliases; this was corrected before scoring. The independent corrected-pool radius audit has zero mismatches.', '',
        'Product continuity requires dated inspected verbatim product evidence on every supply edge, an explicit input/output link through each intermediate, and an affected-item endpoint receipt. Topology, revenue percentages, name/sector similarity and keyword overlap cannot supply these facts. The vendor keyword audit and undated-profile exclusions are preserved under the parent preparation directory. Reuters pages were inspected for retrieval but were not exhaustively converted to structured flow facts; this conservative procedure may miss real continuity.', '',
        'Qwen uses the authorized host-side Toolkit retrieval fallback. All accessible pages pass exact-prefix inspection gates; the model scores deterministic leading excerpts with the same per-case cap in every arm. This is not autonomous Qwen tool calling. Wider arms may change source order and attention and are not matched-compute or causal transmission comparisons. Graph skills use the repository degree/triangle implementations at thresholds 90/75/0.5; their prepared numerical receipts remain available. The June effective-date graph and mapped membership are fixed and are not certified archived monthly acquisition vintages.', '',
        f"DeepSeek is an **incomplete second model**. Its last progress checkpoint records {s['deepseek_partial']['last_progress_requests']}/1,778 requests; the preserved cache contains {s['deepseek_partial']['distinct_responses']} response files, consistent with additional in-flight completions before the stop. The older partial freeze contains {s['deepseek_partial']['old_partial_freeze_cells']} arm/panel cells and was not refreshed. The user stopped it for budget reasons; the actual price was approximately four times the logged estimate. Its partial freeze, caches and cost ledger were hash-checked unchanged. Its scores are not combined with Qwen or used in the results above. All Qwen predictions were produced from scratch using the original Qwen prompts; no DeepSeek response was reused.", '',
        'Possible errors and checks performed', '',
        f"- {s['checks']['tests']} offline unittest checks passed. Tests cover continuity receipts and missing/future/forged facts, shortest radii and prefix gates, evaluation/freeze guards, regression recovery/rank deficiency and CR1 correction, Qwen context/JSON/model checks, cache reuse and transport failure. Test logs and input checks are in this Qwen directory.",
        f"- All {s['prediction_records']:,} predictions and {s['unique_model_calls']:,} unique responses were frozen and reverified before train/valid labels opened. Freeze: {s['frozen_at']}; pre-label verification: {s['pre_label_verification_at']}. Freeze SHA256: `{s['prediction_freeze_sha256']}`. Evaluation files/cases remain blocked, including native Arrow reads behind explicit checks.",
        f"- Every response reports {MODEL}. Maximum prompt: {s['maximum_prompt_tokens']:,} actual tokens; conservative byte-plus-framing bound: {s['maximum_prompt_byte_bound']:,}; maximum prompt plus the {MAX_TOKENS:,}-token output allowance: {s['maximum_prompt_plus_max_tokens']:,} of 32,768. One sequential request, temperature zero, seed 20261026, enable_thinking=false. Backend bitwise determinism is not assumed.",
        f"- Citation audit: {s['citation_quality']['flagged']} flagged responses; {s['citation_quality']['with_exact_aliases']} with exact unique ID alias resolution; {s['citation_quality']['with_unresolved']} with unresolved IDs. Raw IDs/content and all finite scores remain preserved. Unresolved citations are not verified evidence; no case was dropped or rescored for citation quality.",
        '- Requests are persisted before sending, raw responses before JSON validation, and identical request bodies reuse a cached receipt. Unknown outcomes stop rather than trigger automatic duplicate inference. Only 127.0.0.1:8081 is permitted; proxies/redirects are disabled. No paid API call, server reconfiguration/restart or process kill occurred.',
        '- Windows native available-commit and process peak-working-set checks replace permission-denied CIM. Stages wait below 5 GB available commit and fail at 3 GB process RAM. BLAS uses one thread; prepared data are read one date at a time.',
        '- Exact case joins, finite scores/labels, common controls and source content, temporal train/validation ordering, pool/input/code hashes, raw response/request identity, and the unchanged DeepSeek snapshot were checked. Historical model familiarity, short excerpts, incomplete continuity and six dates limit predictive interpretation.', '',
        'Reproduce: see docs/HOP3_ABLATION.md. Use the configured interpreter, PYTHONPATH=source/code and single-threaded BLAS; run `-m news_network_50_2025.hop3_qwen --stage run` after preparation and passing tests. Use a fresh run root after code changes; frozen protocols reject changed code hashes.', '']
    return '\n'.join(lines)


def failure_report(error, guard):
    """An incomplete run never unlocks labels or invents performance numbers."""
    import csv
    if guard.labels_allowed:
        write(OUT / 'summary.json', {'status': 'ANALYSIS_FAILURE_AFTER_FREEZE', 'error': error,
              'complete_prediction_freeze': True, 'train_valid_labels_opened': True,
              'evaluation_labels_opened': False, 'at': now()})
        text = ('# qwen3.8-27b - analysis failure\n\nEngineering run; not look-ahead clean. '
                'Predictions were frozen before train/valid labels opened, but final analysis failed. '
                'No final performance result is published. Evaluation labels remain sealed.\n\n'
                'Error: ' + error['message'] + '\n\nPossible errors and checks performed\n\n'
                'Inspect RUN_FAILURE.json and the complete prediction freeze; rerun reporting offline.\n')
        (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
        TARGET.write_text(text, encoding='utf-8')
        return
    cases = gates.read(ROOT / 'cases.json')['cases']
    for c in cases:
        gates.check_case(c)
    values = {}
    for path in (OUT / 'predictions').glob('*.json'):
        p = gates.read(path)
        values[p['case_id'], p['mode'], p['arm']] = p['score']
    rows = []
    for c in cases:
        if c['split'] == 'valid':
            rows.append({'model': MODEL, 'status': 'INCOMPLETE_NO_LABELS', 'case_id': c['case_id'],
                         'split': c['split'], 'date': c['date'],
                         **{arm: values.get((c['case_id'], 'all_paths', arm)) for arm in ARMS},
                         **{'continuity_' + arm: values.get((c['case_id'], 'continuity_only', arm)) for arm in ARMS}})
    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'matched_validation.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    valid, incomplete = 0, 0
    for path in (OUT / 'responses').glob('*.json'):
        response = gates.read(path)
        try:
            packet = json.loads(response['request']['messages'][0]['content'].split('\nINPUT\n', 1)[1])
            parse_answer(response, [e['source_id'] for e in packet['evidence']])
            valid += 1
        except (ValueError, KeyError, TypeError):
            incomplete += 1
    preserved_path = OUT / 'DEEPSEEK_PRESERVED.json'
    preserved = gates.read(preserved_path) if preserved_path.exists() else {}
    label_files = [p for p in guard.opened if Path(p).name.startswith('labels_')]
    if label_files or guard.labels_allowed:
        raise ValueError('Failure reporting expects a pre-label failure')
    summary = {'status': 'INCOMPLETE_FAILURE', 'engineering_only': True, 'look_ahead_clean': False,
          'responses_cached': valid + incomplete, 'valid_responses': valid, 'invalid_or_incomplete_responses': incomplete,
          'prediction_records': len(values), 'expected_responses': 1778, 'expected_predictions': 3000,
          'evaluation_cases_opened': False, 'evaluation_labels_opened': False, 'train_valid_labels_opened': False,
          'complete_prediction_freeze': False, 'raw_residual_mse': None, 'nested_regression': None,
          'g3_CI_equivalence_power': None, 'error': error, 'at': now(),
          'deepseek_last_progress': preserved.get('last_progress_requests'),
          'deepseek_cached_files': preserved.get('distinct_responses'),
          'deepseek_unchanged': preserved.get('files') == deepseek_snapshot() if preserved else None}
    write(OUT / 'summary.json', summary)
    write(OUT / 'data_access_audit.json', {'live_files_opened': sorted(guard.opened), 'evaluation_accesses': 0,
          'train_valid_labels_opened': False, 'complete_prediction_freeze': False})
    lines = ['# Hop-radius ablation - qwen3.8-27b - INCOMPLETE', '',
        '**Engineering run; Qwen3.8-27B is not look-ahead clean. No performance result is available.**', '',
        f"Cached {valid + incomplete}/1,778 Qwen responses: {valid} valid JSON scores and {incomplete} incomplete/invalid response(s). Saved {len(values)}/3,000 arm/panel predictions. Every cached request uses qwen3.8-27b; no DeepSeek score was reused.", '',
        'Failure: ' + error['message'] + '. The initial attempt used a 700-token completion cap and truncated one reason string, with finish_reason=length. That runner stopped and wrote its failure signal; a subsequent diagnostic found the local endpoint refusing connections. No process was killed and the server was not restarted or reconfigured by this task.', '',
        '**All train, validation and evaluation labels remain sealed.** There is no complete prediction freeze. Raw residual MSE, train-fitted validation regressions, pooled HC3/date-cluster estimates, 90% CIs, g3 equivalence bounds and outcome-based power are not estimated. `matched_validation.csv` is a label-free partial-score roster, not validation performance.', '',
        f"DeepSeek remains an incomplete second model: last progress {preserved.get('last_progress_requests')}/1,778, with {preserved.get('distinct_responses')} cached response files (including later in-flight completions). The old partial freeze and all caches/ledger are preserved. The user stopped it for budget reasons; actual price was approximately 4x the logged estimate. No Qwen/DeepSeek predictions are combined.", '',
        'A corrected fresh scoring version uses a uniform 7,000-token completion allowance. With prompts conservatively below 24,000 including framing, the total stays below 31,000, within the server\'s 32,768 context. All cases and arms must be re-scored with that uniform setting. The initial 700-token attempt is retained for audit and cannot be reused in the corrected version. No model calls are possible while the endpoint is unavailable.', '',
        'Possible errors and checks performed', '',
        '- 75 offline unittest checks passed before launch; the first 23 distinct responses parsed, reported qwen3.8-27b and had no citation flags. Their 4.66-second mean projected about 2.3 hours; the attempt stopped before completion.',
        '- The full input check passed 3,000 matched records, 1,778 unique original Qwen prompts, unchanged common controls and cumulative source contents. The conservative maximum prompt bound was 23,998. The prepared independent full-graph radius audit has zero mismatches.',
        '- The output-cap failure was cached byte-for-byte and was not repaired by extracting an early score from incomplete JSON. Existing finite scores were preserved; none were used to open labels or select cases.',
        '- The conservative continuity validator found no complete verified multi-hop product chains. It requires dated inspected edge-item facts, every intermediate input/output link and endpoint affected-item evidence. Thus continuity-only deeper-hop contrasts would be unidentified, not evidence of zero economic transmission.',
        '- Prepared graph receipts use /sc-graph-degree and /sc-graph-triangles at thresholds 90/75/0.5. Retrieval uses the documented host-side Toolkit fallback and common per-case leading excerpts. Historical model familiarity, short excerpts, fixed graph membership and only six dates limit interpretation.',
        '- Network was restricted to 127.0.0.1:8081, with proxies/redirects disabled. One request at a time; enable_thinking=false. Available-commit and process-memory checks enforced the resource limits. Scoring peak working set was under 0.1 GB.',
        f"- DeepSeek snapshot unchanged: {summary['deepseek_unchanged']}. `data_access_audit.json` confirms no label files opened. Original Qwen request/response bytes, failure traceback and server diagnostic are preserved.", '',
        'Repository changes remain uncommitted; see `FILES_CHANGED.json`. The server must be made available externally before a new detached scoring attempt; this task will not restart it.', '']
    text = '\n'.join(lines)
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding='utf-8')


def run(guard):
    started = now()
    error = None
    try:
        predict()
        report(guard)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        write(OUT / 'RUN_FAILURE.json', {**error, 'at': now(), 'server_log_tail': server_log_tail()[1] if SERVER_LOG.exists() else None})
        failure_report(error, guard)
        raise
    finally:
        # The external watcher may shut down the server as soon as this appears.
        # No more calls or artifact mutations follow this completion signal.
        write(DONE, {'status': 'failure' if error else 'success', 'started_at': started, 'finished_at': now(),
            'counts': {'responses': len(list((OUT / 'responses').glob('*.json'))),
                       'predictions': len(list((OUT / 'predictions').glob('*.json'))), 'expected_responses': 1778, 'expected_predictions': 3000},
            'report_path': str(OUT / 'REPORT.md') if (OUT / 'REPORT.md').exists() else None,
            'published_report_path': str(TARGET) if TARGET.exists() else None, 'error': error})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=('prepare', 'predict', 'report', 'run'))
    a = p.parse_args()
    guard = QwenGuard()
    guard.install()
    from filelock import FileLock
    with FileLock(str(ROOT / 'qwen_only_run.lock'), timeout=0):
        if a.stage == 'prepare':
            prepare()
        elif a.stage == 'predict':
            predict()
        elif a.stage == 'report':
            report(guard)
        else:
            if DONE.exists():
                raise RuntimeError('Completion signal already exists; inspect it before any resume')
            run(guard)


if __name__ == '__main__':
    main()
