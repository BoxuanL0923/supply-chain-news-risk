"""TD4 entry point. The TD3 v6 runtime contract (job memory cap, audit hook, exclusive writes under the TD4 private
root) is installed before any scientific import. Only `call` runs online, and only after clearance A.

Offline build (no clearance; no label, no provider call):
    python -B run_td4.py qualify                         unit tests, synthetic fixtures, mocked providers
    python -B run_td4.py inventory                       2025 coverage of every local source (counts only)
    python -B run_td4.py sources                         2025 extracts (TD1 snapshots, graph, links, news, keys)
    python -B run_td4.py manifest  --year 2025           TD1 feature rows (no labels)
    python -B run_td4.py features  --year 2025           TD2 E/Z/evidence rows
    python -B run_td4.py parity    --check NAME          parity of the TD4 copies with TD1/TD2/TD3 frozen outputs
    python -B run_td4.py inputs    --year 2024|2025      selector inputs (TD3 masker and prompt, unchanged)
    python -B run_td4.py audit-sample --year Y           re-mask from raw text, strict residual, scanner (audit sample)
    python -B run_td4.py verify --year Y                 text-level checks over every input of the period
    python -B run_td4.py repair --year Y                 D8: rebuild story-field-withheld firm-weeks (one pre-mask rule)
    python -B run_td4.py verify --year Y --check repair  text-level checks over the repaired inputs
    python -B run_td4.py packet                          blinded human-labeling packet (2024 contexts)
    python -B run_td4.py score-labels --labels FILE      diagnostics once humans have filled the template
    python -B run_td4.py cost                            cost and wall-time projection
    python -B run_td4.py dry-run   --year Y              mock providers over the real inputs (ledger exercised)
    python -B run_td4.py prereg                          PREREGISTRATION_v1.json (hashed) + AGGREGATION_FREEZE_v1
    python -B run_td4.py rehearsal                       sandboxed dress rehearsal of every real step on 2021/2022
    python -B run_td4.py freeze                          FREEZE_TD4_A + CLEARANCE_REQUEST_TD4_A (request only)
    python -B run_td4.py summary                         TD4_QUALIFICATION_<utc>.json (tests, parity, hashes)

Real steps (each refuses unless clearance A, or B for evaluate, matches the frozen package):
    python -B run_td4.py refit                           MLP + heads through 2023 and 2024 (reads labels <= 2024)
    python -B run_td4.py call      --year 2024|2025      paid provider calls (resumable, idempotent)
    python -B run_td4.py reask     --year 2024|2025      fixed re-ask subset (stability diagnostic)
    python -B run_td4.py select                          2024 selection -> SELECTION_FREEZE (+ request B if 2025 done)
    python -B run_td4.py request-b                       offline: CLEARANCE_REQUEST_TD4_B after the 2025 calls
    python -B run_td4.py evaluate                        once-only 2025 evaluation (clearance B)
    python -B run_td4.py status
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import td4_spec as T
import td4_env as E

OFFLINE = ['qualify', 'inventory', 'sources', 'manifest', 'features', 'parity', 'inputs', 'audit-sample', 'packet',
           'cost', 'dry-run', 'prereg', 'freeze', 'refit', 'select', 'request-b', 'evaluate', 'status',
           'score-labels', 'rehearsal', 'summary', 'verify', 'repair']
ONLINE = ['call', 'reask']


def main(argv=None):
    p = argparse.ArgumentParser(description='TD4 provider selectors (2024 selection, 2025 test)')
    p.add_argument('command', choices=OFFLINE + ONLINE)
    p.add_argument('--year', type=int)
    p.add_argument('--check')
    p.add_argument('--limit', type=int)
    p.add_argument('--labels')
    a = p.parse_args(argv)
    if a.command == 'qualify':
        E.paths()
        import td4_tests
        return td4_tests.qualify()
    R = E.install('online' if a.command in ONLINE else 'offline')
    if a.command == 'inventory':
        import td4_inventory
        path, out = td4_inventory.run()
        print('INVENTORY', path, flush=True)
    elif a.command == 'sources':
        import td4_sources
        print('SOURCES', json.dumps(td4_sources.build_all(), default=str), flush=True)
    elif a.command == 'manifest':
        import td4_manifest
        print('MANIFEST', json.dumps(td4_manifest.build(a.year), default=str), flush=True)
    elif a.command == 'features':
        import td4_features
        print('FEATURES', json.dumps(td4_features.build(a.year), default=str), flush=True)
    elif a.command == 'parity':
        import td4_parity
        print('PARITY', json.dumps(td4_parity.run(a.check, limit=a.limit), default=str), flush=True)
    elif a.command == 'inputs':
        import td4_inputs
        print('INPUTS', json.dumps(td4_inputs.build(a.year, limit=a.limit), default=str), flush=True)
    elif a.command == 'verify':
        import td4_verify
        print('VERIFY', json.dumps(td4_verify.population_checks(a.year, repair=bool(a.check == 'repair')),
                                   default=str), flush=True)
    elif a.command == 'repair':
        import td4_repair
        print('REPAIR', json.dumps(td4_repair.run(a.year), default=str), flush=True)
    elif a.command == 'audit-sample':
        import td4_inputs
        if a.check == 'repair':
            out = td4_inputs.audit_sample(a.year, period_dir=td4_inputs.repair_dir(a.year), store_tag=f'repair_{a.year}')
        else:
            out = td4_inputs.audit_sample(a.year)
        print('AUDIT', json.dumps(out, default=str), flush=True)
    elif a.command == 'packet':
        import td4_diagnostics
        print('PACKET', json.dumps(td4_diagnostics.build_packet(), default=str), flush=True)
    elif a.command == 'score-labels':
        import td4_diagnostics
        print('LABEL_SCORES', json.dumps(td4_diagnostics.score_labels(a.labels), default=str), flush=True)
    elif a.command == 'cost':
        import td4_cost
        print('COST', json.dumps(td4_cost.project(), default=str), flush=True)
    elif a.command == 'dry-run':
        import td4_runner
        print('DRYRUN', json.dumps(td4_runner.dry_run(a.year, limit=a.limit), default=str), flush=True)
    elif a.command == 'prereg':
        import td4_freeze
        print('PREREG', json.dumps(td4_freeze.write_prereg(), default=str), flush=True)
    elif a.command == 'freeze':
        import td4_freeze
        print('FREEZE', json.dumps(td4_freeze.freeze(), default=str), flush=True)
    elif a.command == 'refit':
        import td4_models
        print('REFIT', json.dumps(td4_models.refit_all(), default=str), flush=True)
    elif a.command == 'call':
        import td4_runner
        print('CALL', json.dumps(td4_runner.run_real(a.year), default=str), flush=True)
    elif a.command == 'reask':
        import td4_runner
        print('REASK', json.dumps(td4_runner.run_real(a.year, reask=True), default=str), flush=True)
    elif a.command == 'select':
        import td4_select
        print('SELECT', json.dumps(td4_select.run(), default=str), flush=True)
    elif a.command == 'request-b':
        import td4_select
        print('REQUEST_B', json.dumps(td4_select.request_b(), default=str), flush=True)
    elif a.command == 'evaluate':
        import td4_evaluate
        print('EVALUATE', json.dumps(td4_evaluate.run(), default=str), flush=True)
    elif a.command == 'summary':
        import td4_freeze
        print('SUMMARY', json.dumps(td4_freeze.summary(), default=str), flush=True)
    elif a.command == 'rehearsal':
        import td4_rehearsal
        print('REHEARSAL', json.dumps(td4_rehearsal.run(), default=str), flush=True)
    elif a.command == 'status':
        import td4_gate
        print('STATUS', json.dumps(td4_gate.status(), default=str, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
