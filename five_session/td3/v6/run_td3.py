"""TD3 v6 (TD3F) entry point. The runtime contract (2.0e9-byte job memory cap, audit hook) is installed before any
scientific import. Every command is offline except `run`, which may resolve only api.deepseek.com and
api.typesafe.ai and refuses unless clearance/TD3_CLEARANCE_v6.json matches the frozen v6 code and inputs.

    python -B run_td3.py qualify               offline unit tests with mocked providers
    python -B run_td3.py build   --tag v7      offline: 200-firm-week sample, frozen inputs, scan hits, adjacency
                                               report and adjacency sweep (needs the frozen private mask table)
    python -B run_td3.py verify  --tag v7      offline: independent re-check (needs SCAN_ADJUDICATION.json,
                                               ADJACENCY_REVIEW.json and ADJACENCY_SWEEP_REVIEW.json)
    python -B run_td3.py dry-run --tag v7      offline: mock providers over the frozen inputs
    python -B run_td3.py freeze  --tag v7      offline: FREEZE_TD3_v6.json + CLEARANCE_REQUEST_TD3_v6.json
    python -B run_td3.py run [--resume RUN_ID] REAL provider calls (clearance v6 required)
    python -B run_td3.py report  --run RUN_ID  offline: recompute a run's report into a new file
    python -B run_td3.py status
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import uuid
from pathlib import Path
import td3_spec as S
import td3_runtime as R


def main(argv=None):
    p = argparse.ArgumentParser(description='TD3 provider-selector pilot (v6)')
    p.add_argument('command', choices=['qualify', 'build', 'verify', 'dry-run', 'freeze', 'run', 'report', 'status'])
    p.add_argument('--tag', default='v7')
    p.add_argument('--resume')
    p.add_argument('--run')
    a = p.parse_args(argv)
    R.install('online' if a.command == 'run' else 'offline')
    build_dir = S.OUT / f'build_{a.tag}'
    if a.command == 'qualify':
        import td3_tests
        return td3_tests.qualify()
    if a.command == 'status':
        import td3_runner
        ok, reasons, _ = td3_runner.clearance_check()
        print('CODE', S.CODE, 'CLEARANCE_PATH', S.CLEARANCE)
        print('BUILD', build_dir, (build_dir / 'BUILD.json').exists(), 'VERIFY', (build_dir / 'VERIFY.json').exists())
        print('FREEZE', (S.OUT / S.FREEZE_NAME).exists(), 'REQUEST', (S.OUT / S.CLEARANCE_REQUEST_NAME).exists())
        print('CLEARANCE_VALID', ok, reasons)
        print('REAL_RUNS', sorted(x.name for x in S.RUNS.glob('real_*')) if S.RUNS.exists() else [])
        return 0
    if a.command == 'run':
        import td3_runner
        td3_runner.run_real(resume=a.resume)
        return 0
    from td3_freeze import latest_qualification
    print('QUALIFIED_BY', latest_qualification(), flush=True)
    if a.command == 'build':
        import td3_build
        receipt = td3_build.build(build_dir)
        print('BUILD_DONE', build_dir, json.dumps(receipt['counts']), json.dumps(receipt['independent_scan']),
              flush=True)
    elif a.command == 'verify':
        import td3_build
        result = td3_build.verify(build_dir)
        print('VERIFY', json.dumps({k: result[k] for k in ('records', 'reask', 'problems', 'passed',
                                                           'scan_adjudication', 'adjacency_review',
                                                           'adjacency_sweep', 'review_findings',
                                                           'ric_checks')}), flush=True)
        return 0 if result['passed'] else 1
    elif a.command == 'dry-run':
        import td3_runner
        run_dir, info = td3_runner.run_dry(build_dir)
        print('DRYRUN_DONE', run_dir, json.dumps({k: info[k] for k in ('planned_calls', 'planned_jev_requests',
                                                                       'total_precall_upper_bound_usd_pass1',
                                                                       'deepseek_estimate_usd_all_calls',
                                                                       'jev_estimate_usd_all_calls',
                                                                       'real_run_currently_allowed')}), flush=True)
    elif a.command == 'freeze':
        import td3_freeze
        print('FROZEN', *td3_freeze.freeze(build_dir), flush=True)
    elif a.command == 'report':
        import td3_report
        import td3_runner
        if not a.run:
            p.error('--run RUN_ID required')
        run_dir = S.RUNS / a.run
        started = json.loads(sorted(run_dir.glob('session_*/SESSION_STARTED.json'))[0].read_text(encoding='utf-8'))
        records = td3_runner.load_inputs(started['build_dir'])
        report, sel_csv, led_csv = td3_report.build_report(run_dir, records, td3_report.td2_call_counts(),
                                                           td3_runner._pass_rate(started['build_dir']),
                                                           started['run_mode'])
        tag = uuid.uuid4().hex[:8]
        R.new_json(run_dir / f'PILOT_REPORT_recomputed_{tag}.json', report)
        R.new_text(run_dir / f'selections_recomputed_{tag}.csv', sel_csv)
        R.new_text(run_dir / f'ledger_recomputed_{tag}.csv', led_csv)
        print('REPORT', run_dir, tag, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
