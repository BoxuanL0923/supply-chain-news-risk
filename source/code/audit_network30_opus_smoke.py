"""Label-free audit of paired Opus smoke traces before the full panel."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from news_agent_batched_runtime_2025 import digest, read, write
from network50_grounded_toolkit import (CERTAIN_WINDOW, NUMERIC_RECAP,
                                         UNPRICED_REACTION, UNSOURCED_PATTERN)


def audit(folder: Path, expected_cases: int) -> dict:
    protocol = read(folder / 'PROTOCOL.json')
    selected = read(folder / 'selected_cases.json')
    smoke = {(c['date'], c['ticker']) for c in selected if
             (folder / 'predictions' / f"{c['case_id'].replace(':', '_')}_OWN.json").exists()}
    issues, call_errors, pairs = [], Counter(), []
    if protocol['model'] != 'claude-opus-4-6':
        issues.append('Unexpected model in protocol')
    for day, ticker in sorted(smoke):
        case = next(c for c in selected if c['date'] == day and c['ticker'] == ticker)
        outputs = {}
        for arm in ('OWN', 'NET'):
            path = folder / 'predictions' / f"{case['case_id'].replace(':', '_')}_{arm}.json"
            if not path.exists():
                issues.append(f"Missing {case['case_id']} {arm}")
                continue
            pred = read(path)
            trace = read(folder / 'traces' / f"{pred['key']}.json")
            forecast = pred['accepted']['forecast']
            outputs[arm] = forecast
            if pred['model'] != protocol['model'] or forecast['case_id'] != case['case_id']:
                issues.append(f"Wrong model or case: {case['case_id']} {arm}")
            if not trace['submissions'] or digest(trace['submissions'][-1]) != digest(pred['accepted']):
                issues.append(f"Receipt mismatch: {case['case_id']} {arm}")
            if trace.get('labels_accessible') is not False:
                issues.append(f"Labels accessible: {case['case_id']} {arm}")
            used = {call['tool'] for call in trace['calls'] if call['ok']}
            mandatory = {'market_state', 'inventory', 'search_archive'}
            if arm == 'NET':
                mandatory |= {'network_leads'}
            if not mandatory <= used:
                issues.append(f"Missing tools: {case['case_id']} {arm} {mandatory-used}")
            if arm == 'NET' and not any(call['tool'] == 'network_leads' and call['ok'] and
                                        not call['args'].get('query') for call in trace['calls']):
                issues.append(f"No broad network discovery: {case['case_id']}")
            for call in trace['calls']:
                if not call['ok']:
                    error = call['result'].get('error', {})
                    code = error.get('code', str(error)[:80]) if isinstance(error, dict) else str(error)[:80]
                    call_errors[(call['tool'], code)] += 1
            cutoff = datetime.fromisoformat(case['asof'])
            for fact in forecast['facts'].values():
                if datetime.fromisoformat(fact['available_at']) >= cutoff:
                    issues.append(f"Future cited fact: {case['case_id']} {arm} {fact['source_id']}")
                if fact.get('minutes_before_decision', 0) <= 0:
                    issues.append(f"Invalid source-time receipt: {case['case_id']} {arm}")
            for driver in forecast['drivers']:
                if any(NUMERIC_RECAP.search(driver[field]) for field in ('interpretation', 'limitations')):
                    issues.append(f"Numeric recap accepted: {case['case_id']} {arm}")
                if UNPRICED_REACTION.search(driver['interpretation']):
                    issues.append(f"Unpriced reaction accepted: {case['case_id']} {arm}")
                if UNSOURCED_PATTERN.search(driver['interpretation']):
                    issues.append(f"Unverified attribution or pattern accepted: {case['case_id']} {arm}")
                if CERTAIN_WINDOW.search(driver['interpretation']):
                    receipt = forecast['dates'].get(driver['timing_id']) if driver['timing_id'] else None
                    if receipt is None and arm == 'NET' and driver['timing_id']:
                        receipt = outputs.get('OWN', {}).get('dates', {}).get(driver['timing_id'])
                    if not receipt or receipt['in_forecast_window'] is not True:
                        issues.append(f"Unverified scheduled window: {case['case_id']} {arm}")
                if not set(driver['fact_ids']) <= set(forecast['facts']) | set(
                        outputs.get('OWN', {}).get('facts', {})):
                    issues.append(f"Unbound driver fact: {case['case_id']} {arm}")
                if arm == 'NET' and not set(driver['relation_ids']) <= set(trace['relations']):
                    issues.append(f"Unbound graph path: {case['case_id']} {arm}")
        if 'OWN' in outputs and 'NET' in outputs:
            if digest(outputs['NET']['frozen_own']) != digest(outputs['OWN']):
                issues.append(f"OWN changed in NET: {case['case_id']}")
            if not outputs['NET']['drivers'] and outputs['NET']['score'] != outputs['OWN']['score']:
                issues.append(f"Score changed without NET event: {case['case_id']}")
            pairs.append({'case_id': case['case_id'], 'own_score': outputs['OWN']['score'],
                          'net_score': outputs['NET']['score'],
                          'new_network_events': len(outputs['NET']['drivers'])})
    if len(pairs) != expected_cases:
        issues.append(f"Expected {expected_cases} complete pairs, found {len(pairs)}")
    return {'passed_mechanical': not issues, 'issues': issues, 'pairs': pairs,
            'invalid_tool_calls': [{'tool': tool, 'code': code, 'count': count}
                                   for (tool, code), count in sorted(call_errors.items())],
            'test_labels_read': False,
            'manual_semantic_review_required': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--expected-cases', type=int, default=8)
    args = parser.parse_args()
    result = audit(args.folder, args.expected_cases)
    write(args.folder / 'smoke_audit.json', result)
    print(f"Mechanical pass={result['passed_mechanical']}; pairs={len(result['pairs'])}; issues={result['issues']}")
    if not result['passed_mechanical']:
        raise SystemExit(1)
