"""TD4 v2 entry point (amendment 2: select on a 2021-2022 subsample, test on pooled 2023-2025, frozen TD1/TD2 models).

The v1 package (td4/*.py, FREEZE_TD4_A_v1.json) is left byte-identical; v2 code lives in td4/v2/ and imports the v1
modules it reuses unchanged. The TD3 v6 runtime contract is installed before any scientific import.

Offline (no clearance; no label, no provider call):
    python -B run_td4v2.py qualify                 v2 tests (synthetic fixtures, mocked providers)
    python -B run_td4v2.py build-m2 --period 2122|2023|2024|2025   inputs with masker v2 (td4v2_build2; all periods)
    python -B run_td4v2.py verify-m2 --period P    population checks + v2 scanner (withholding exact classes)
    python -B run_td4v2.py audit-m2 --period P     fresh hash-ranked audit sample (re-mask, residuals, both scanners)
    python -B run_td4v2.py flags-prepare --period P   detector-flag rule: copied reviewer detectors + RUN_DETECTORS.sh
    bash <run folder>/RUN_DETECTORS.sh                 the detectors, one process at a time (outside the runtime)
    python -B run_td4v2.py flags --period P --which RUN   withheld keys (td4v2_flags; DETECTOR_FLAGS_M2.json)
    python -B run_td4v2.py redact --period P --which R    span-level redaction round R = 1, 2, 3 (td4v2_redact)
    python -B run_td4v2.py redact-verify --period P --which R   ledger/diff invariants and idempotency of round R
    python -B run_td4v2.py redact-prepare --period P --which R  detector re-run on the round-R inputs
    bash <run folder>/RUN_DETECTORS.sh                 the re-run (outside the runtime)
    python -B run_td4v2.py redact-residual --period P --which RUN   still-flagged inputs after the re-run
    python -B run_td4v2.py redact-final --period P     final inputs and withheld set (after round 3)
    python -B run_td4v2.py nosource --period P         rule v2.1: MASKER_NO_SOURCE (firms without a masker name source)
    python -B run_td4v2.py redact-final-v21 --period P   rule v2.1 final withheld set (REDACTION_FINAL_v2_1_<UTC>)
    python -B run_td4v2.py redact22 --period P --which R          v2.2 round R (td4v2_redact22; rounds 1-2 ledger-only)
    python -B run_td4v2.py redact22-prepare --period P --which R  detector re-run on the v2.2 round-R text
    bash <run folder>/RUN_DETECTORS.sh                 the re-run (outside the runtime)
    python -B run_td4v2.py redact22-residual --period P --which RUN   still flagged after the v2.2 re-run
    python -B run_td4v2.py redact22-verify --period P --which R   rendering, invariants, P6, idempotency of round R
    python -B run_td4v2.py redact22-final --period P   REDACTION_FINAL_v2_2_<UTC> (after round 3)
    python -B run_td4v2.py check22 --period P          review classes, P6 and over-redaction, v2.1 vs v2.2
    python -B run_td4v2.py parity-construction         request bodies from the redacted text (200 inputs x 3)
    python -B run_td4v2.py packet-r                    labeling_v2r: redacted versions of the changed packet contexts
    python -B run_td4v2.py packet-r2                   labeling_v2r2: contexts whose v2.2 text differs from the labelled text
    python -B run_td4v2.py convert-labels-r --labels DIR   relabels of labeling_v2r -> label template
    python -B run_td4v2.py convert-labels-r2 --labels DIR  relabels of labeling_v2r2 (v2.2 text) -> label template
    python -B run_td4v2.py parity2025 --which graph2025_frozen_code|links2025_frozen_code|features2025_frozen_code
    python -B run_td4v2.py ciz-impact [--limit N]  2025 inputs changed by the CIZ 2025 alias source (sample rebuild)
    python -B run_td4v2.py build-select            [superseded: v1 masker] 2021-2022 selection subsample inputs
    python -B run_td4v2.py build-2023              2023 test inputs (same code as the v1 2024/2025 builds)
    python -B run_td4v2.py repair --which select|2023
    python -B run_td4v2.py verify --which select|select-repair|2023|2023-repair
    python -B run_td4v2.py audit --which select|select-repair|2023|2023-repair
    python -B run_td4v2.py packet                  LABEL_PACKET_v2 (150 contexts, batches of 10, Codex schema)
    python -B run_td4v2.py convert-labels --labels DIR   Codex JSON outputs -> validated label template
    python -B run_td4v2.py cost
    python -B run_td4v2.py parity                  v2 selection path reproduces TD2's 2021-2022 selection MSEs
    python -B run_td4v2.py dry-run --period P --limit N
    python -B run_td4v2.py prereg
    python -B run_td4v2.py rehearsal
    python -B run_td4v2.py freeze | summary | status
    python -B run_td4v2.py slim-parity --period P   v2.3 F2: compact outputs equal the v2.2 path (no label read)
    python -B run_td4v2.py mem-rehearsal            v2.3 F6: full-size evaluate in a sandbox, synthetic labels
Real steps (clearance A v2.3; evaluate needs clearance B v2.3):
    python -B run_td4v2.py call | reask             CLOSED in v2.3 (every provider call ran under clearance A v2.2)
    python -B run_td4v2.py select                   writes selection_v2_3/ (selection_v2/ stays untouched)
    python -B run_td4v2.py request-b
    python -B run_td4v2.py evaluate
    python -B run_td4v2.py score-labels --labels FILE[,V2R[,V2R2]]   (Codex labels converted to the template)
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import td4_spec as T
import td4_env as E

OFFLINE = ['qualify', 'redact', 'redact-prepare', 'redact-residual', 'redact-verify', 'redact-final',
           'parity-construction', 'packet-r', 'convert-labels-r', 'nosource', 'redact-final-v21',
           'redact22', 'redact22-prepare', 'redact22-residual', 'redact22-verify', 'redact22-final', 'check22',
           'packet-r2', 'convert-labels-r2',
           'flags-prepare', 'flags', 'build-m2', 'verify-m2', 'audit-m2', 'parity2025', 'ciz-impact', 'build-select', 'build-2023', 'repair', 'verify', 'audit', 'packet', 'convert-labels', 'cost',
           'dry-run', 'prereg', 'rehearsal', 'freeze', 'summary', 'status', 'select', 'request-b', 'evaluate',
           'score-labels', 'parity', 'slim-parity', 'mem-rehearsal']
ONLINE = ['call', 'reask']                             # closed in v2.3: refused before the runtime is installed


def main(argv=None):
    p = argparse.ArgumentParser(description='TD4 v2')
    p.add_argument('command', choices=OFFLINE + ONLINE)
    p.add_argument('--which')
    p.add_argument('--period')
    p.add_argument('--limit', type=int)
    p.add_argument('--labels')
    a = p.parse_args(argv)
    E.paths()
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if a.command == 'qualify':
        import td4v2_tests
        return td4v2_tests.qualify()
    if a.command in ONLINE:                            # v2.3: no further provider call (the gate refuses as well)
        import td4v2_spec as SP
        raise PermissionError(f'STOP: {a.command} is closed in v2.3: every provider call ran under clearance A '
                              f'v2.2 ({SP.CLEARANCE_A_V2_2_SHA256}); no further provider call is possible')
    if a.command == 'mem-rehearsal':                   # installs its file-open trace before the runtime and any reader
        import td4v2_memrehearsal as MR
        out = MR.run()
        print(a.command.upper(), json.dumps(out, default=str), flush=True)
        return 0 if out.get('pass') else 1
    E.install('offline')
    out = None
    if a.command == 'flags-prepare':                   # detector-flag rule step 1 (after verify-m2)
        import td4v2_flags as FL
        out = FL.prepare_period(a.period)
    elif a.command == 'flags':                         # step 3, after the run folder's RUN_DETECTORS.sh (step 2)
        import td4v2_flags as FL
        out = FL.run(a.period, a.which)
    elif a.command == 'redact':                        # span-level redaction round (td4v2_redact)
        import td4v2_redact as RD
        out = RD.redact_round(a.period, int(a.which), dev_limit=a.limit)
    elif a.command == 'redact-prepare':                # detector re-run on the round's inputs
        import td4v2_redact as RD
        out = RD.prepare_rerun(a.period, int(a.which))
    elif a.command == 'redact-residual':               # after the run folder's RUN_DETECTORS.sh
        import td4v2_redact as RD
        rnd = int(a.which.split('_rd', 1)[1].split('_', 1)[0])
        out = RD.residual(a.period, rnd, a.which)
    elif a.command == 'redact-verify':
        import td4v2_redact as RD
        out = RD.verify_round(a.period, int(a.which), idem_n=a.limit or 2000)
    elif a.command == 'redact-final':
        import td4v2_redact_final as RF
        out = RF.finalize(a.period)
    elif a.command == 'nosource':                      # rule v2.1: MASKER_NO_SOURCE per period
        import td4v2_nosource as NS
        out = NS.run(a.period)
    elif a.command == 'redact-final-v21':              # rule v2.1 final withheld set
        import td4v2_redact_final as RF
        out = RF.finalize_v21(a.period)
    elif a.command == 'redact22':                      # v2.2 round (td4v2_redact22)
        import td4v2_redact22 as R22
        out = R22.redact_round(a.period, int(a.which))
    elif a.command == 'redact22-prepare':
        import td4v2_redact22 as R22
        out = R22.prepare_rerun(a.period, int(a.which))
    elif a.command == 'redact22-residual':
        import td4v2_redact22 as R22
        rnd = int(a.which.split('_v22rd', 1)[1].split('_', 1)[0])
        out = R22.residual(a.period, rnd, a.which)
    elif a.command == 'redact22-verify':
        import td4v2_redact22 as R22
        out = R22.verify_round22(a.period, int(a.which), idem_n=a.limit or 2000)
    elif a.command == 'redact22-final':
        import td4v2_redact22 as R22
        out = R22.finalize22(a.period)
    elif a.command == 'check22':
        import td4v2_check22 as C22
        out = C22.run(a.period)
    elif a.command == 'build-m2':
        if str(a.period) == '2025':                    # 2025 adds the CRSP CIZ 2025 alias source
            import td4v2_names2025 as N25
            out = N25.build()
        else:
            import td4v2_build2 as B2
            out = B2.build(a.period)
    elif a.command == 'verify-m2':
        import td4v2_verify2 as V2
        out = V2.population(a.period)
    elif a.command == 'audit-m2':
        import td4v2_verify2 as V2
        out = V2.audit(a.period)
    elif a.command == 'parity2025':
        import td4v2_parity2025 as P25
        out = P25.run(a.which, limit=a.limit)
    elif a.command == 'ciz-impact':
        import td4v2_names2025 as N25
        out = N25.impact(a.limit or 1000)
    elif a.command == 'build-select':
        import td4v2_build as B
        out = B.build_select()
    elif a.command == 'build-2023':
        import td4v2_build as B
        out = B.build_year(2023)
    elif a.command == 'repair':
        import td4v2_build as B
        out = B.repair('select' if a.which == 'select' else int(a.which))
    elif a.command == 'verify':
        import td4v2_verify as V
        out = V.run(a.which)
    elif a.command == 'audit':
        import td4v2_verify as V
        out = V.audit(a.which)
    elif a.command == 'packet':
        import td4v2_labels as LB
        out = LB.build_packet()
        out['marker'] = LB.packet_marker()
    elif a.command == 'convert-labels':
        import td4v2_labels as LB
        out = LB.convert(a.labels)
    elif a.command == 'score-labels':
        import td4v2_labels as LB
        lab, _, rest = a.labels.partition(',')           # v2 template[,v2r relabels[,v2r2 relabels]]
        lab_r, _, lab_r2 = rest.partition(',')
        out = LB.score(lab, lab_r or None, lab_r2 or None)
    elif a.command == 'cost':
        import td4v2_cost as C
        out = C.project()
    elif a.command == 'dry-run':
        import td4v2_runner as RU
        out = RU.dry_run(a.period, limit=a.limit)
    elif a.command == 'prereg':
        import td4v2_freeze as FZ
        out = FZ.write_prereg()
    elif a.command == 'rehearsal':
        import td4v2_rehearsal as RH
        out = RH.run()
    elif a.command == 'freeze':
        import td4v2_freeze as FZ
        out = FZ.freeze()
    elif a.command == 'summary':
        import td4v2_freeze as FZ
        out = FZ.summary()
    elif a.command == 'status':
        import td4v2_gate as G
        out = G.status()
    elif a.command in ('call', 'reask'):
        import td4v2_runner as RU
        out = RU.run_real(a.period, reask=a.command == 'reask')
    elif a.command == 'parity':
        import td4v2_parity as PV
        out = PV.run()
    elif a.command == 'parity-construction':
        import td4v2_parity as PV
        out = PV.construction_redacted(a.limit or 200)
    elif a.command == 'packet-r':
        import td4v2_labels as LB
        out = LB.redacted_packet()
    elif a.command == 'convert-labels-r':
        import td4v2_labels as LB
        out = LB.convert(a.labels, revised=True)
    elif a.command == 'convert-labels-r2':
        import td4v2_labels as LB
        out = LB.convert(a.labels, revised='v2r2')
    elif a.command == 'packet-r2':
        import td4v2_labels as LB
        out = LB.redacted_packet_v22()
    elif a.command == 'select':
        import td4v2_select as SL
        out = SL.run()
    elif a.command == 'request-b':
        import td4v2_select as SL
        out = SL.request_b()
    elif a.command == 'evaluate':
        import td4v2_evaluate as EV
        out = EV.run()
    elif a.command == 'slim-parity':
        import td4v2_slim as SL3
        out = SL3.parity(a.period)
    print(a.command.upper(), json.dumps(out, default=str), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
