"""Dress rehearsal of every real step on the development years, in a sandbox (offline; no provider; no 2023+ label).

The real-step code (freeze, gate admission, refit_all, the runner session, re-ask, select.run, request_b,
evaluate.run) is executed unchanged after the period constants are pointed at the development years:
selection year 2021 (models refitted through 2020), test year 2022 (models through 2021), 150 firm-weeks per year
built with td4_inputs, mock providers (td3_providers.MockTransport) on a simulated clock, a mock balance, and
sandbox clearance files written from the sandbox requests. Every path (outputs, clearances, latches) is under
D:/OptResearch_data/five_session/td4/rehearsal_<id>/; the real clearance folder is never touched. All labels read are
development labels (label end <= 2022-12-31). The numbers it produces are engineering checks, not results.
Must run in a fresh process (run_td4.py rehearsal) because it re-points td4_spec constants before importing the
other TD4 modules.
"""
import json
import sys
import time
import uuid
from pathlib import Path

import td4_spec as T
import td4_env as E

REAL_OUT = T.OUT


def run(n=150):
    for m in ('td4_gate', 'td4_inputs', 'td4_models', 'td4_runner', 'td4_select', 'td4_evaluate', 'td4_freeze',
              'td4_cost', 'td4_score', 'td4_labels'):
        if m in sys.modules:
            raise RuntimeError('rehearsal must run in a fresh process (' + m + ' already imported)')
    E.paths()
    rid = uuid.uuid4().hex[:8]
    box = REAL_OUT / f'rehearsal_{rid}'
    T.OUT = box
    T.SELECTION_YEAR, T.TEST_YEAR = 2021, 2022
    T.PERIODS = (2021, 2022)
    T.TRAIN_END = {2021: 2020, 2022: 2021}
    T.CLEARANCE_A = box / 'clearance' / 'TD4_CLEARANCE_A_v1.json'
    T.CLEARANCE_B = box / 'clearance' / 'TD4_CLEARANCE_B_v1.json'
    import td3_runtime as R
    import td4_gate as G
    import td4_inputs as I
    import td4_runner as RU
    import td4_freeze as FZ
    import td4_cost as C
    import td4_models as MD
    import td4_select as SL
    import td4_evaluate as EV
    log = {'sandbox': str(box), 'years': {'selection': 2021, 'test': 2022}, 'steps': {}}
    t0 = time.perf_counter()
    import td4_verify as V
    for y in T.PERIODS:
        rec = I.run_build((y,), I.INPUTS / f'period_{y}', limit=n, tag=str(y), audit_n=10)
        log['steps'][f'inputs_{y}'] = rec['counts']
        log['steps'][f'verify_{y}'] = V.population_checks(y)
    log['steps']['cost'] = C.project()
    q = sorted(REAL_OUT.glob('QUALIFICATION_TD4_*.json'), key=lambda p: p.stat().st_mtime)
    qual = json.loads(q[-1].read_text(encoding='utf-8')) if q else {}
    if qual.get('code_sha256') != G.code_inventory():
        qual = {'tests': 0, 'failures': 0, 'errors': 0, 'code_sha256': G.code_inventory(),
                'note': 'rehearsal stand-in: no current qualification receipt'}
    R.new_json(box / f'QUALIFICATION_TD4_rehearsal.json', {**qual, 'rehearsal_copy_of': str(q[-1]) if q else None})
    FZ.write_prereg()
    log['steps']['freeze'] = FZ.freeze()
    req = json.loads(G.REQUEST_A.read_text(encoding='utf-8'))
    req.pop('request_only')
    req.pop('written_utc', None)
    req['status'] = 'CLEARED'
    R.new_json(T.CLEARANCE_A, req)
    log['steps']['gate_a'] = G.status()
    log['steps']['refit'] = MD.refit_all()
    token, freeze = G.admit('call')
    for y in T.PERIODS:
        records = I.callable_inputs(y)
        kr = RU.Keyring(R.Redactor(), {'deepseek': 'sk-rehearsal-0000', 'jev': 'rehearsal-jev-0000'})
        caps = freeze['caps'][str(y)]
        d, s = RU.run_session(y, records, __import__('td3_providers').MockTransport(), kr, lambda: 62.57,
                              RU.SimClock('2026-11-09T00:59:00'), caps, 'real', {'rehearsal': True})
        d2, s2 = RU.run_session(y, records, __import__('td3_providers').MockTransport(), kr, lambda: 62.57,
                                RU.SimClock('2026-11-09T12:00:00'), caps, 'real', {'rehearsal': True}, reask=True)
        log['steps'][f'calls_{y}'] = {'halt': s['halt'], 'counts': s['counts'], 'reask_halt': s2['halt'],
                                      'completion': RU.completion(y, records)}
    log['steps']['select'] = SL.run()
    if not G.REQUEST_B.exists():
        log['steps']['request_b'] = SL.request_b()
    reqb = json.loads(G.REQUEST_B.read_text(encoding='utf-8'))
    reqb.pop('request_only')
    reqb.pop('written_utc', None)
    reqb['status'] = 'CLEARED'
    reqb['clearance_a_sha256'] = G.sha(T.CLEARANCE_A)
    R.new_json(T.CLEARANCE_B, reqb)
    log['steps']['evaluate'] = EV.run()
    try:
        EV.run()
        log['steps']['evaluate_replay'] = 'NOT REFUSED (defect)'
    except PermissionError as e:
        log['steps']['evaluate_replay'] = 'refused: ' + str(e)[:120]
    log['elapsed_seconds'] = time.perf_counter() - t0
    log.update(R.peak())
    T.OUT = REAL_OUT
    R.new_json(REAL_OUT / f'REHEARSAL_{rid}.json', log)
    return {'rehearsal': str(REAL_OUT / f'REHEARSAL_{rid}.json'), 'sandbox': str(box),
            'evaluate_replay': log['steps']['evaluate_replay'], 'P1_engineering_only': log['steps']['evaluate']['P1']}
