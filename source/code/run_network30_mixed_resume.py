"""Finish the December panel without repeating completed paired Opus forecasts.

The resulting panel deliberately mixes two tool contracts. It is exploratory,
not a single frozen-protocol test; every chosen pair retains its source version.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading

from filelock import FileLock

import run_network30_frozen_2025 as base
from audit_network30_opus_smoke import audit
from news_agent_batched_runtime_2025 import Budget, digest, read, write
from news_network_50_2025.config import OUT


OLD = OUT / 'network30_dec2025_opus46_v5'
NEW = OUT / 'network30_dec2025_opus46_v6'
MIXED = OUT / 'network30_dec2025_opus46_mixed_v5_v6'
MODEL = 'claude-opus-4-6'
LIMITS = {**base.LIMITS, 'hard_total_provider_attempts': 320,
          'hard_total_execution_minutes': 720,
          'input_token_stop': 250_000_000, 'output_token_stop': 7_000_000,
          'provider_list_price_stop_usd': 500.}


def pair(folder, case):
    stem = case['case_id'].replace(':', '_')
    paths = [folder / 'predictions' / f'{stem}_{arm}.json' for arm in ('OWN', 'NET')]
    if not all(path.exists() for path in paths):
        return None
    own, net = [read(path) for path in paths]
    version = 'grounded-v11' if folder == NEW else 'grounded-v10'
    for record, arm in ((own, 'OWN'), (net, 'NET')):
        forecast = record['accepted']['forecast']
        if (record['case_id'] != case['case_id'] or record['arm'] != arm or
                record['model'] != MODEL or forecast['version'] != version):
            raise ValueError(f'Invalid cached record: {case["case_id"]} {arm} {folder}')
        trace = read(folder / 'traces' / f'{record["key"]}.json')
        if (not trace['submissions'] or
                digest(trace['submissions'][-1]) != digest(record['accepted']) or
                trace.get('labels_accessible') is not False):
            raise ValueError(f'Invalid receipt or label access: {case["case_id"]} {arm}')
    if digest(net['accepted']['forecast']['frozen_own']) != digest(own['accepted']['forecast']):
        raise ValueError(f'NET changed OWN: {case["case_id"]}')
    return own, net


def chosen(case):
    revised = pair(NEW, case)
    return (NEW, revised) if revised else (OLD, pair(OLD, case))


def run():
    base.MODEL, base.DEST, base.LIMITS = MODEL, NEW, LIMITS
    NEW.mkdir(parents=True, exist_ok=True)
    MIXED.mkdir(parents=True, exist_ok=True)
    with FileLock(str(NEW / '.run.lock'), timeout=0):
        cases, controls, anchors, protocol = base.prepare()
        if not (NEW / 'SMOKE_ACCEPTED.json').exists():
            raise ValueError('Revised-tool smoke requires label-free acceptance first')
        previous = read(OLD / 'PROTOCOL.json')
        if (previous['anchor_sha256'] != protocol['anchor_sha256'] or
                previous['firms'] != protocol['firms'] or previous['dates'] != protocol['dates']):
            raise ValueError('The old and revised arms do not share the same anchor or panel')
        pending = [case for case in cases if chosen(case)[1] is None]
        budget = Budget(NEW, LIMITS)
        budget.start()
        errors, done, lock = [], [], threading.Lock()
        halted = threading.Event()

        def task(case):
            if halted.is_set():
                return
            try:
                own = base.score(case, False, None, controls, anchors, protocol, budget)
                net = base.score(case, True, own['accepted']['forecast'], controls,
                                 anchors, protocol, budget)
                with lock:
                    done.append(case['case_id'])
                    write(MIXED / 'progress.json', {
                        'complete_new_pairs': len(done), 'requested_new_pairs': len(pending),
                        'last_case': case['case_id'], 'errors': errors,
                        'at': datetime.now(timezone.utc).isoformat()})
                    print(case['case_id'], 'OWN', own['score'], 'NET', net['score'], flush=True)
            except Exception as exc:
                halted.set()
                with lock:
                    errors.append({'case_id': case['case_id'], 'error': str(exc)})
                    write(MIXED / 'progress.json', {'complete_new_pairs': len(done),
                          'requested_new_pairs': len(pending), 'errors': errors,
                          'at': datetime.now(timezone.utc).isoformat()})

        print('Reusing completed pairs; new pairs needed:', len(pending), flush=True)
        with ThreadPoolExecutor(max_workers=base.WORKERS) as executor:
            futures = [executor.submit(task, case) for case in pending]
            for future in as_completed(futures):
                future.result()
        if errors:
            raise ValueError(f'Incomplete revised run: {errors}')

        (MIXED / 'predictions').mkdir(parents=True, exist_ok=True)
        (MIXED / 'traces').mkdir(parents=True, exist_ok=True)
        provenance = {}
        for case in cases:
            folder, records = chosen(case)
            if records is None:
                raise ValueError(f'Missing final pair: {case["case_id"]}')
            provenance[case['case_id']] = 'v6_revised' if folder == NEW else 'v5_original'
            for record in records:
                stem = case['case_id'].replace(':', '_')
                shutil.copy2(folder / 'predictions' / f'{stem}_{record["arm"]}.json',
                             MIXED / 'predictions' / f'{stem}_{record["arm"]}.json')
                shutil.copy2(folder / 'traces' / f'{record["key"]}.json',
                             MIXED / 'traces' / f'{record["key"]}.json')
        for filename in ('PROTOCOL.json', 'selected_cases.json'):
            shutil.copy2(NEW / filename, MIXED / filename)
        access_path = NEW / 'TWO_NETWORK_ADDITIONS_DIAGNOSTIC.json'
        inspected = read(access_path)['outcome_labels_opened'] if access_path.exists() else []
        manifest = {'design': 'mixed_tool_version_exploratory_not_frozen',
                    'model': MODEL, 'cases': provenance,
                    'counts': {v: list(provenance.values()).count(v)
                               for v in ('v5_original', 'v6_revised')},
                    'old_implementation': previous['implementation'],
                    'new_implementation': protocol['implementation'],
                    'anchor_sha256': protocol['anchor_sha256'],
                    'test_labels_read_before_complete_predictions': bool(inspected),
                    'previously_inspected_case_ids': inspected}
        write(MIXED / 'PROVENANCE.json', manifest)
        result = audit(MIXED, expected_cases=120)
        write(MIXED / 'FULL_AUDIT.json', result)
        if not result['passed_mechanical']:
            raise ValueError(f'Full label-free audit failed: {result["issues"][:12]}')
        base.DEST = MIXED
        report = base.evaluate(cases, controls, anchors)
        print('Complete mixed-panel NET minus OWN direct R2 pp:',
              report['frozen_direct']['network_minus_own_r2_pp'], flush=True)


if __name__ == '__main__':
    run()
