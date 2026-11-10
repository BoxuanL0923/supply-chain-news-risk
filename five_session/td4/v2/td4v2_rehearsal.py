"""Dress rehearsal of every v2 real step in a sandbox (offline; mock providers; development labels only).

The v2 real-step code runs unchanged after its period constants are pointed at development data: the selection
subsample is 150 hash-ranked 2021-2022 firm-weeks (built with td4v2_build.build_select), the test year is 2022 (150
firm-weeks, labels bounded at 2022-12-31 so no 2023 return is read), every output, clearance and latch path is under
D:/OptResearch_data/five_session/td4/rehearsal_v2_<id>/. Sandbox clearances are written from the sandbox requests;
the real clearance folder is never touched. The numbers it produces are engineering checks, not results.
Run in a fresh process (run_td4v2.py rehearsal).
"""
import json
import sys
import time
import uuid

import td4_spec as T
import td4_env as E
import td4v2_spec as S


def run(n=150):
    if S.PROVIDER_CALLS_CLOSED:
        raise PermissionError('STOP: superseded in v2.3. This dev-year rehearsal starts its real-step path with '
                              'td4v2_gate.admit("call"), which v2.3 closes, and its select/evaluate steps read '
                              '2021-2022 development labels. The v2.3 evaluator is rehearsed at full size by '
                              'td4v2_memrehearsal (run_td4v2.py mem-rehearsal), without any label read.')
    for m in ('td4v2_gate', 'td4v2_runner', 'td4v2_select', 'td4v2_evaluate', 'td4v2_freeze', 'td4v2_cost',
              'td4v2_inputs', 'td4v2_build'):
        if m in sys.modules:
            raise RuntimeError('rehearsal must run in a fresh process (' + m + ' already imported)')
    E.paths()
    import td3_runtime as R
    rid = uuid.uuid4().hex[:8]
    box = T.OUT / f'rehearsal_v2_{rid}'
    S.OUT = box
    S.RUNS, S.SEL, S.EV, S.LABELS = box / 'runs_v2', box / 'selection_v2', box / 'evaluation_v2', box / 'labeling_v2'
    S.LABELS_R = box / 'labeling_v2r'
    S.CLEARANCE_A, S.CLEARANCE_B = box / 'clearance' / 'TD4_CLEARANCE_A_v2.json', box / 'clearance' / 'TD4_CLEARANCE_B_v2.json'
    S.FREEZE_A, S.REQUEST_A, S.PREREG = box / 'FREEZE_TD4_A_v2.json', box / 'CLEARANCE_REQUEST_TD4_A_v2.json', \
        box / 'PREREGISTRATION_v2.json'
    S.SUPERSEDED_V2 = {}
    S.SELECTION_FREEZE, S.REQUEST_B = S.SEL / 'SELECTION_FREEZE_v2.json', S.SEL / 'CLEARANCE_REQUEST_TD4_B_v2.json'
    S.EVAL_LATCH, S.EVAL_RESULT = S.EV / 'EVALUATION_STARTED_v2.json', S.EV / 'EVALUATION_RESULT_v2.json'
    S.EVAL_PRIMARY, S.V23 = S.EV / 'EVALUATION_PRIMARY_v2.json', box / 'v2_3'
    S.TEST_YEARS, S.RUN_ORDER = [2022], [S.SELECT, '2022']
    S.PERIOD_REQ_YEAR = {S.SELECT: 2122, '2022': 2022}
    S.LABEL_END_BOUND = '2022-12-31'
    S.SUBSAMPLE_N = n
    import td4v2_build as BV
    import td4v2_build2 as B2
    BV.SUBSAMPLE_N, BV.CANDIDATES = n, n + 10
    B2.PERIOD_DIR = {'2122': box / 'inputs_m2' / 'select_2122', 2022: box / 'inputs_m2' / 'period_2022'}
    B2.YEARS = {'2122': (2021, 2022), 2022: (2022,)}
    B2.STORE_TAG = {'2122': f'rh2_{rid}_select', 2022: f'rh2_{rid}_2022'}
    import td4_inputs as I
    import td4_runner as RU
    import td3_providers as P
    import td4v2_verify2 as V
    import td4v2_cost as C2
    import td4v2_freeze as FZ
    import td4v2_gate as G
    import td4v2_runner as RV
    import td4v2_select as SL
    import td4v2_evaluate as EV
    import td4v2_inputs as IV
    log = {'sandbox': str(box), 'steps': {}}
    t0 = time.perf_counter()
    sel = B2.build('2122', audit_n=5)                 # masker v2, own sandbox store
    log['steps']['build_select'] = sel['counts']
    rec = B2.build(2022, limit=n, audit_n=5)
    log['steps']['build_2022'] = rec['counts']
    import td4v2_flags as FL
    import td4v2_redact_final as RF
    import td4v2_nosource as NS
    for p in ('2122', 2022):
        log['steps'][f'verify_{p}'] = V.population(p)
        FL.stand_in(B2.PERIOD_DIR[p])                  # sandbox: empty detector-flag file (engineering check)
        RF.stand_in(B2.PERIOD_DIR[p])                  # sandbox: identity redaction (engineering check)
        NS.stand_in(B2.PERIOD_DIR[p])                  # sandbox: empty MASKER_NO_SOURCE (rule v2.1)
    log['steps']['cost'] = C2.project()
    q = sorted(T.OUT.glob('QUALIFICATION_TD4v2_*.json'), key=lambda p: p.stat().st_mtime)
    qual = json.loads(q[-1].read_text(encoding='utf-8')) if q else {}
    if qual.get('code_sha256') != G.code_inventory():
        qual = {'tests': 0, 'failures': 0, 'errors': 0, 'code_sha256': G.code_inventory(),
                'note': 'rehearsal stand-in: no current v2 qualification receipt'}
    R.new_json(box / 'QUALIFICATION_TD4v2_rehearsal.json', {**qual, 'rehearsal_copy_of': str(q[-1]) if q else None})
    log['steps']['prereg'] = FZ.write_prereg()
    log['steps']['freeze'] = FZ.freeze()
    req = json.loads(S.REQUEST_A.read_text(encoding='utf-8'))
    req.pop('request_only')
    req.pop('written_utc', None)
    req['status'] = 'CLEARED'
    R.new_json(S.CLEARANCE_A, req)
    token, freeze = G.admit('call')
    for p in S.RUN_ORDER:
        RV.check_order(p, RV.completion)
        recs = IV.callable_inputs(p)
        kr = RU.Keyring(R.Redactor(), {'deepseek': 'sk-rehearsal-v2-0000', 'jev': 'rehearsal-v2-jev-0000'})
        d, s = RU.run_session(S.PERIOD_REQ_YEAR[p], recs, P.MockTransport(), kr, lambda: S.BALANCE_USD,
                              RU.SimClock('2026-11-09T00:59:00'), freeze['caps'][p], 'real', {'rehearsal': True},
                              root=RV.period_root(p), concurrency={'deepseek': 3, 'jev': 2})
        d2, s2 = RU.run_session(S.PERIOD_REQ_YEAR[p], recs, P.MockTransport(), kr, lambda: S.BALANCE_USD,
                                RU.SimClock('2026-11-09T12:00:00'), freeze['caps'][p], 'real', {'rehearsal': True},
                                reask=True, root=RV.period_root(p), concurrency={'deepseek': 3, 'jev': 2})
        log['steps'][f'calls_{p}'] = {'halt': s['halt'], 'reask_halt': s2['halt'], 'completion': RV.completion(p)}
    log['steps']['select'] = SL.run()
    if not S.REQUEST_B.exists():
        log['steps']['request_b'] = SL.request_b()
    reqb = json.loads(S.REQUEST_B.read_text(encoding='utf-8'))
    reqb.pop('request_only')
    reqb.pop('written_utc', None)
    reqb['status'] = 'CLEARED'
    reqb['clearance_a_sha256'] = G.sha(S.CLEARANCE_A)
    R.new_json(S.CLEARANCE_B, reqb)
    log['steps']['evaluate'] = EV.run()
    try:
        EV.run()
        log['steps']['evaluate_replay'] = 'NOT REFUSED (defect)'
    except PermissionError as e:
        log['steps']['evaluate_replay'] = 'refused: ' + str(e)[:120]
    log['elapsed_seconds'] = time.perf_counter() - t0
    log.update(R.peak())
    R.new_json(T.OUT / f'REHEARSAL_v2_{rid}.json', log)
    return {'rehearsal': str(T.OUT / f'REHEARSAL_v2_{rid}.json'), 'evaluate_replay': log['steps']['evaluate_replay'],
            'note': 'engineering check on development years with mock providers; not a result'}
