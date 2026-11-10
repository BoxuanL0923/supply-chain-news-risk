"""v2 pre-registration, freeze A v2 and the clearance-A v2 request (request only), and the v2 qualification summary.

FREEZE_TD4_A_v1.json and every file it binds stay byte-identical; v2 binds the v1 files it reuses unchanged (by
their current hashes, equal to the v1 freeze) plus the new v2 files. The inputs are the masker-v2 builds of every
period, and the span-level redaction of them (td4v2_redact, researcher decision 2026-11-07): the final (round-3)
redacted INPUTS, every round's INPUTS, ledger, round, verify and residual file, the detector re-run folders (run copies,
run script, run log) and the REDACTION_FINAL file of each build are bound. The round-0 detector-flag files
(td4v2_flags) stay bound as the record of round 0; they no longer withhold. The masker-v3 modules are kept for the
record; no v2 path imports them and no masker-v3 output is bound (researcher decision 2026-11-06).
Rule v2.1 (2026-11-07): the freeze is written as FREEZE_TD4_A_v2_1.json with the request CLEARANCE_REQUEST_TD4_A_v2_1.json
and PREREGISTRATION_v2_1.json; it also binds every build's MASKER_NO_SOURCE and REDACTION_FINAL_v2_1 files, the masker's
name tables (CRSP name files, FSB1 company file, CIZ 2025) and, as the superseded record, the v2 freeze, request and
pre-registration (never cleared).
v2.2 (2026-11-07 fix round after the TD4M3 redaction review): FREEZE_TD4_A_v2_2.json, CLEARANCE_REQUEST_TD4_A_v2_2.json
and PREREGISTRATION_v2_2.json; it binds the v2.2 rounds (td4v2_redact22: ledgers, hashes, round, verify and residual
files, round-3 INPUTS, the v2.2 detector re-run folders), REDACTION_FINAL_v2_2, the CHECK22 files, the labeling_v2r2
packet and, as the superseded record, the v2.1 freeze, request and pre-registration (never cleared); the v2.1 rounds
stay bound as the record.
v2.3 (2026-11-09, evaluator memory and crash-safety fix, brief F7): FREEZE_TD4_A_v2_3.json and
CLEARANCE_REQUEST_TD4_A_v2_3.json bind every td4/*.py and td4/v2/*.py, every artifact and external file that freeze
A v2.2 bound (re-hashed equal, or the freeze refuses), the same pre-registration (PREREGISTRATION_v2_2.json) and
models, the v2.2 freeze and request as the record, the v2.3 evidence (slim-output parity receipts, the full-size
memory rehearsal, the qualification) and `provider_outputs_record`: every runs_v2/real_<P> session ran under
clearance A v2.2 (SHA256 checked against td4v2_spec.CLEARANCE_A_V2_2_SHA256), with the per-period ledger digest and
the SHA256 of every run file taken at the freeze. Under v2.3, call and reask refuse; select writes selection_v2_3/.
"""
import glob
import json
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_spec as S
import td4v2_gate as G
import td4v2_build2 as B2
import td4v2_names2025 as N25

DIR_FILES = ('INPUTS.jsonl', 'INPUT_META_COMPACT.jsonl', 'ELIGIBLE.parquet', 'BUILD.json', 'WITHHELD.jsonl',
             'AUDIT_META.jsonl', 'AUDIT_SWEEP.jsonl', 'MASK_V2_RECEIPT.json', 'POPULATION_CHECK_M2.json',
             'WITHHELD_POPULATION_M2.json', 'SCAN_HITS_M2.jsonl', 'DETECTOR_FLAGS_M2.json')


def _R():
    E.paths()
    import td3_runtime as R
    return R


def write_prereg():
    R = _R()
    if not S.PREREG.exists():
        R.new_text(S.PREREG, S.preregistration_text())
    return {'path': str(S.PREREG), 'sha256': R.sha(S.PREREG), 'matches_spec': R.sha(S.PREREG) == S.preregistration_sha256(),
            'aggregation_freeze_reused': str(S.AGGREGATION_FREEZE), 'aggregation_sha256': R.sha(S.AGGREGATION_FREEZE)}


def latest(pattern):
    files = sorted(glob.glob(str(S.OUT / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return files[-1] if files else None


def period_dirs():
    """The masker-v2 build of every period (no v1-masker input is bound)."""
    return [Path(B2.PERIOD_DIR[S.SELECT])] + [Path(B2.PERIOD_DIR[y]) for y in S.TEST_YEARS]


def flag_run_dirs():
    """The detector run folders named in the bound DETECTOR_FLAGS_M2.json files."""
    import td4v2_flags as FL
    out = []
    for d in period_dirs():
        for p in sorted(d.glob('DETECTOR_FLAGS_M2*.json')):
            r = json.loads(p.read_text(encoding='utf-8')).get('run_dir')
            if r and Path(r) not in out:
                out.append(Path(r))
    return out


def redaction_files():
    """Every file of the span-level redaction: per period the rounds' INPUTS, LEDGER, HASHES, ROUND, VERIFY and
    RESIDUAL files of v2/v2.1 (td4v2_redact) and v2.2 (td4v2_redact22), the detector re-run folders named in the
    residual files, the build's REDACTION_FINAL files and the v2.2 CHECK22 files."""
    import td4v2_redact as RD
    import td4v2_redact22 as R22
    paths, runs = [], []
    for key in [S.SELECT] + list(S.TEST_YEARS):
        for r in range(1, RD.ROUNDS + 1):
            for rd in (RD.round_dir(str(key), r), R22.round_dir(str(key), r)):
                paths += [rd / n for n in ('INPUTS.jsonl', 'LEDGER.jsonl', 'HASHES.jsonl', 'ROUND.json')]
                for x in sorted(rd.glob('VERIFY_*.json')) + sorted(rd.glob('VERIFY22_*.json')) + \
                        sorted(rd.glob('RESIDUAL_*.json')):
                    paths.append(x)
                    if x.name.startswith('RESIDUAL_'):
                        run = Path(json.loads(x.read_text(encoding='utf-8'))['run_dir'])
                        if run not in runs:
                            runs.append(run)
        paths += sorted(Path(B2.PERIOD_DIR[key]).glob('REDACTION_FINAL_*.json'))
        paths += sorted((R22.ROOT / str(key)).glob('CHECK22_*.json'))
    for run in runs:
        paths += sorted(run.rglob('*.py')) + sorted(run.glob('*.sh')) + [run / 'run.log']
    return paths, runs


def artifact_list():
    paths = []
    for d in period_dirs():
        paths += [d / n for n in DIR_FILES]
        paths += sorted(d.glob('AUDIT_SAMPLE_M2_*.json')) + sorted(d.glob('AUDIT_SCAN_HITS_M2_*.jsonl'))
        paths += sorted(d.glob('POPULATION_CHECK_M2_*.json')) + sorted(d.glob('WITHHELD_POPULATION_M2_*.json'))
        paths += sorted(d.glob('SCAN_HITS_M2_*.jsonl')) + sorted(d.glob('NAMES2025_CHECK_M2*.json'))
        paths += sorted(d.glob('DETECTOR_FLAGS_M2_*.json')) + sorted(d.glob('MASKER_NO_SOURCE_*.json'))
    for r in flag_run_dirs():                         # the run copies of the detectors and their run log
        paths += sorted(r.rglob('*.py')) + sorted(r.glob('*.sh')) + [r / 'run.log']
    paths += redaction_files()[0]
    for tag in B2.STORE_TAG.values():
        paths.append(T.OUT / 'inputs' / f'article_store_{tag}.parquet')
    paths += sorted((T.OUT / 'parity').glob('PARITY_*_frozen_code_*.json'))
    paths += sorted((T.OUT / 'parity').glob('PARITY_v2_selection_td2_*.json'))
    paths += [p for p in (T.OUT / 'CIZ_2025_IMPACT.json', T.OUT / 'proof_m2' / 'PROOF_M2.json') if p.exists()]
    paths += [B2.TABLES_V2, T.OUT / 'tables_v2' / 'COMMON_WORDS_v2.json']
    paths += sorted((T.OUT / 'tables_v2').glob('CRSP_WORD_DF_v2_*.json'))
    for sub in ('td1_inputs', 'td2_sources', 'panel_keys'):
        paths += sorted((T.OUT / sub).glob('*.parquet'))
    paths += [T.OUT / 'firm_week_manifest/year_2025.parquet', T.OUT / 'features/year_2025.parquet',
              T.OUT / 'evidence/year_2025.parquet', S.PREREG, S.AGGREGATION_FREEZE]
    paths += sorted(S.LABELS.glob('batch_*.json')) + [S.LABELS / n for n in ('INSTRUCTIONS_v2.txt', 'PACKET_KEYS_v2.json',
                                                                             'PACKET_INDEX_v2.json')]
    paths += sorted(S.LABELS_R.glob('batch_*.json')) + [S.LABELS_R / n for n in ('INSTRUCTIONS_v2r.txt',
                                                                                 'PACKET_KEYS_v2r.json',
                                                                                 'PACKET_INDEX_v2r.json')]
    paths += sorted((T.OUT / 'parity').glob('PARITY_construction_redacted_*.json'))
    paths += sorted(S.LABELS_R2.glob('batch_*.json')) + [S.LABELS_R2 / n for n in ('INSTRUCTIONS_v2r2.txt',
                                                                                   'PACKET_KEYS_v2r2.json',
                                                                                   'PACKET_INDEX_v2r2.json')]
    paths += [x for x in S.SUPERSEDED_V2.values()]                       # rule-v2 freeze, request, prereg (record)
    paths += sorted((T.OUT / 'inputs_redacted' / '2025').glob('SPOTCHECK_v2_1_*.json'))
    for pat in ('COST_PROJECTION_v2_*.json', 'QUALIFICATION_TD4v2_*.json'):
        p = latest(pat)
        if p:
            paths.append(Path(p))
    paths += list(v23_receipts().values())                              # v2.3 evidence
    v22 = S.SUPERSEDED_V2.get('freeze_v2_2')
    if v22 is not None and Path(v22).exists():                          # v2.3: everything freeze A v2.2 bound
        paths += [Path(p) for p in json.loads(Path(v22).read_text(encoding='utf-8'))['artifact_sha256']]
    out, seen = [], set()
    for p in paths:
        if Path(p).exists() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def latest_rehearsal():
    """The latest v2.3 full-size memory rehearsal receipt (td4v2_memrehearsal)."""
    files = sorted(S.OUT.glob('rehearsal_mem_v2_3_*/MEMREHEARSAL_v2_3.json'), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def v23_receipts():
    """v2.3 evidence: the latest slim-output parity receipt per period and the latest memory rehearsal."""
    import td4v2_slim as SL3
    out = {}
    for p in S.RUN_ORDER:
        x = SL3.latest_parity(p)
        if x is not None:
            out[f'slim_parity_{p}'] = x
    r = latest_rehearsal()
    if r is not None:
        out['memory_rehearsal'] = r
    return out


def provider_outputs_record():
    """v2.3 F7: the record that every provider output in runs_v2/real_<P> was produced under clearance A v2.2, with
    the per-period ledger digest and the SHA256 of every run file taken now (td4v2_gate.runs_digests)."""
    R = _R()
    import td4v2_runner as RV
    c = S.CLEARANCE_A_V2_2
    c_sha = R.sha(c) if c.exists() else None
    if c_sha != S.CLEARANCE_A_V2_2_SHA256:
        raise PermissionError('clearance A v2.2 missing or different from ' + S.CLEARANCE_A_V2_2_SHA256)
    f22 = S.SUPERSEDED_V2['freeze_v2_2']
    f22_sha = R.sha(f22)
    if json.loads(c.read_text(encoding='utf-8')).get('freeze_sha256') != f22_sha:
        raise PermissionError('clearance A v2.2 does not bind FREEZE_TD4_A_v2_2.json')
    runner_sha = R.sha(S.CODE_V2 / 'td4v2_runner.py')
    sessions = {}
    for p in S.RUN_ORDER:
        starts = sorted(RV.period_root(p).glob('session_*/SESSION_STARTED.json'))
        n, bound = len(starts), Counter()
        for s in starts:
            b = json.loads(s.read_text(encoding='utf-8')).get('binding') or {}
            bound['clearance_a_v2_2'] += int(b.get('clearance_sha256') == S.CLEARANCE_A_V2_2_SHA256)
            bound['freeze_a_v2_2'] += int(b.get('freeze_sha256') == f22_sha)
            bound['runner_code_unchanged'] += int(b.get('code_sha256_runner') == runner_sha)
            bound['period'] += int(b.get('period') == p)
        sessions[p] = {'sessions': n, **{k: bound[k] for k in ('clearance_a_v2_2', 'freeze_a_v2_2',
                                                               'runner_code_unchanged', 'period')}}
        if not n or any(sessions[p][k] != n for k in ('clearance_a_v2_2', 'freeze_a_v2_2', 'period')):
            raise PermissionError(f'a session of {p} is not bound to clearance A v2.2: ' + json.dumps(sessions[p]))
    import td4v2_gate as G
    return {'statement': 'Every provider call of 2122, 2023, 2024 and 2025 ran under clearance A v2.2; the outputs in '
                         'runs_v2/real_<P> are those of these sessions. Under v2.3 call and reask refuse, so no '
                         'further provider call is possible.',
            'clearance_a_v2_2': {'path': str(c), 'sha256': c_sha}, 'freeze_a_v2_2_sha256': f22_sha,
            'request_a_v2_2_sha256': R.sha(S.SUPERSEDED_V2['request_v2_2']),
            'runner_code_sha256': runner_sha, 'sessions': sessions,
            'digest_rule': 'per period: SHA256 of the sorted settle rows "req_key|attempt|http_status|response_sha256" '
                           'over every session ledger (td4v2_select.ledger_digest); files_sha256 = every file under '
                           'runs_v2/real_<P>/; td4v2_gate.check_runs_record recomputes both',
            'periods': G.runs_digests(), 'taken_utc': R.utc(),
            'selection_v2_record_sha256': {k: R.sha(v) for k, v in S.SELECTION_V2_RECORD.items() if Path(v).exists()},
            'provider_calls_after_v2_3': 'refused: call and reask (td4v2_gate.CLOSED_STEPS)'}


def v23_evidence(inv):
    """The v2.3 evidence the freeze requires: the 2023 slim-output parity (and every other period's receipt present)
    equal on the current code, and a passing full-size memory rehearsal on the current code."""
    ev = v23_receipts()
    out, bad = {}, []
    for name, p in sorted(ev.items()):
        d = json.loads(Path(p).read_text(encoding='utf-8'))
        if d.get('code_sha256') != inv:
            bad.append(name + ': receipt not on the current code')
        if name.startswith('slim_parity_'):
            out[name] = {'path': str(p), 'equal': d.get('equal'), 'rows': d.get('rows'),
                         'differences': d.get('differences')}
            if d.get('equal') is not True:
                bad.append(name + ': not equal')
        else:
            out[name] = {'path': str(p), 'pass': d.get('pass'), 'peak_working_set_bytes': d.get('peak_working_set_bytes'),
                         'peak_private_bytes': d.get('peak_private_bytes'), 'job_peak': d.get('job_peak'),
                         'target_bytes': d.get('target_bytes'), 'job_memory_cap': d.get('job_memory_cap'),
                         'trace_clean': (d.get('trace') or {}).get('clean')}
            if d.get('pass') is not True:
                bad.append(name + ': rehearsal did not pass')
    for need in ('slim_parity_2023', 'memory_rehearsal'):
        if need not in ev:
            bad.append(need + ' missing')
    if bad:
        raise PermissionError('v2.3 evidence missing or failing: ' + '; '.join(bad))
    return out


def compare_v2_2(f):
    """Every artifact and external file of freeze A v2.2 is bound again with the same SHA256, and the models,
    pre-registration, caps and question are the same."""
    f22 = json.loads(Path(S.SUPERSEDED_V2['freeze_v2_2']).read_text(encoding='utf-8'))
    out = {}
    for field in ('artifact_sha256', 'external_sha256'):
        a, b = f22[field], f[field]
        out[field] = {'v2_2': len(a), 'v2_3': len(b), 'same_sha256': sum(b.get(p) == h for p, h in a.items()),
                      'changed_or_missing': sorted(p for p, h in a.items() if b.get(p) != h),
                      'added': sorted(p for p in b if p not in a)}
    for k in ('models', 'preregistration_sha256', 'aggregation_freeze_sha256', 'caps', 'question_sha256', 'periods',
              'run_order', 'hard_seal', 'test_years'):
        out['same_' + k] = f22.get(k) == f.get(k)
    bad = [k for k, v in out.items() if k.startswith('same_') and not v]
    bad += [fld for fld in ('artifact_sha256', 'external_sha256') if out[fld]['changed_or_missing']]
    if bad:
        raise PermissionError('freeze A v2.3 differs from v2.2 beyond the v2.3 changes: ' + ', '.join(bad))
    return out


def external_list():
    import td4_freeze as F1
    import td4v2_flags as FL
    ext = list(F1.external_list())
    ext += [T.TD1 / f'predictions_v2/year_{y}.parquet' for y in (2021, 2023, 2024)]
    ext += [T.TD2 / f'evidence_v2/year_{y}.parquet' for y in (2021, 2022)]
    ext += [T.TD1 / f'inputs/crsp_{y}.parquet' for y in (2021,)]
    ext += [N25.CIZ, N25.CIZ.with_name('CRSP_CIZ_SECINFO_2025_v1.manifest.json')]
    ext += [Path(p) for p in FL.manifest()['script_sha256']] + list(FL.ADJUDICATED)   # reviewer detector originals
    ext += list(T.CRSP_NAME_FILES) + [T.FSB1_COMPANY]                    # masker-v2 name tables (MASKER_NO_SOURCE)
    out, seen = [], set()
    for p in ext:
        if Path(p).exists() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


V1_REUSED_DATA = ('the v1 article stores of 2024 and 2025 and the 2021-2022 selection store (archive rows only, no '
                  'masked text; reused by tag), the 2025 extracts (td1_inputs, td2_sources, panel_keys, '
                  'firm_week_manifest/year_2025, features/year_2025, evidence/year_2025), AGGREGATION_FREEZE_v1.json. '
                  'No input built with the v1 masker is bound: every period is rebuilt with masker v2 (inputs_m2/).')
V1_SUPERSEDED = ('the v1-masker inputs (inputs/period_2024, period_2025, repair_2024, repair_2025, select_2122, '
                 'select_2122_b; masking reviews td4m M1-M4 FIX REQUIRED), the v1 D8 repair rule (td4_repair.RULE; '
                 'TD4C B1), PREREGISTRATION_v1.json, FREEZE_TD4_A_v1.json, CLEARANCE_REQUEST_TD4_A_v1.json, the v1 cost '
                 'projection, the v1 labeling packet (labeling/), and the v1 entry points refit/call/select/evaluate '
                 '(td4_models.refit_all, td4_runner.run_real, td4_select.run, td4_evaluate.run and td4_gate paths)')


BUILD_CODE = ('td4v2_mask.py', 'td4v2_build2.py', 'td4v2_words.py', 'td4v2_build.py')


def build_receipts():
    """Per period: the masker-v2 receipt, its SHA256, and the build code files changed since that build (N9)."""
    out = {}
    cur = B2.code_files()
    for key in [S.SELECT] + list(S.TEST_YEARS):
        d = Path(B2.PERIOD_DIR[key])
        p = d / 'MASK_V2_RECEIPT.json'
        if not p.exists():
            out[str(key)] = {'receipt': None, 'changed_build_code': ['receipt missing']}
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        files = BUILD_CODE + (('td4v2_names2025.py',) if str(key) == '2025' else ())
        changed = [f for f in files if r['code_sha256'].get(f) != cur.get(f)]
        if str(key) == '2025' and r.get('ciz_2025_names_sha256') != N25.CIZ_SHA256:
            changed.append('CIZ 2025 name source not used or changed')
        import td4v2_flags as FL
        import td4v2_redact_final as RF
        flags = FL.flags_path(d)
        fl = json.loads(flags.read_text(encoding='utf-8')) if flags.exists() else {}
        if not fl:
            changed.append('round-0 detector flags missing')
        try:
            rfp = RF.final_path(d)
            rf = json.loads(rfp.read_text(encoding='utf-8'))
            if G.sha(rf['final_inputs']) != rf['final_inputs_sha256']:
                changed.append('redacted inputs changed after REDACTION_FINAL')
        except PermissionError:
            rfp, rf = None, {}
            changed.append('span-level redaction not finalised')
        if rfp is not None and rf.get('rule_version') != S.RULE_VERSION:
            changed.append(f'rule-{S.RULE_VERSION} redaction result missing')
        import td4v2_nosource as NS
        try:
            nsp = NS.latest(d)
            nsr = json.loads(nsp.read_text(encoding='utf-8'))
        except PermissionError:
            nsp, nsr = None, {}
            changed.append('MASKER_NO_SOURCE missing')
        out[str(key)] = {'receipt': str(p), 'receipt_sha256': G.sha(p), 'dir': str(d),
                         'build_code_sha256': {f: r['code_sha256'].get(f) for f in files},
                         'ciz_2025_names_sha256': r.get('ciz_2025_names_sha256'),
                         'tables_v2_sha256': r.get('tables_v2_sha256'),
                         'common_words_sha256': r.get('common_words_sha256'),
                         'crsp_word_df_sha256': r.get('crsp_word_df_sha256'),
                         'detector_flags_round0': {'file': str(flags), 'sha256': G.sha(flags) if fl else None,
                                                   'run_dir': fl.get('run_dir'),
                                                   'whole_input_rule_share_not_applied': fl.get('withheld_share'),
                                                   'callable_before_flags': fl.get('callable_before_flags')},
                         'redaction': {'file': str(rfp) if rfp else None, 'sha256': G.sha(rfp) if rfp else None,
                                       'rule_version': rf.get('rule_version'),
                                       'final_inputs': rf.get('final_inputs'),
                                       'final_inputs_sha256': rf.get('final_inputs_sha256'),
                                       'withheld_inputs': rf.get('withheld_inputs'),
                                       'withheld_share': rf.get('withheld_share'), 'callable': rf.get('callable'),
                                       'withheld_reasons': rf.get('withheld_reasons'),
                                       'waived_literal_placeholder_hits': rf.get('waived_literal_placeholder_hits'),
                                       'subsample': rf.get('subsample')},
                         'masker_no_source': {'file': str(nsp) if nsp else None, 'sha256': G.sha(nsp) if nsp else None,
                                              'applied': nsr.get('applied'), 'flagged_inputs': nsr.get('flagged_inputs'),
                                              'unsourced_permnos': nsr.get('unsourced_permnos')},
                         'changed_build_code': changed}
    return out


def terminal_caps():
    """N1: the reviewed cap of terminal client-error units per provider and period (td4v2_runner.terminal_cap)."""
    import td4v2_runner as RV
    import td4v2_inputs as IV
    out = {}
    for p in S.RUN_ORDER:
        n = len(IV.callable_inputs(p))
        out[p] = {'deepseek': RV.terminal_cap(n), 'jev': RV.terminal_cap(2 * n)}
    return out


def freeze():
    R = _R()
    import td3_spec as S3
    if S.FREEZE_A.exists() or S.REQUEST_A.exists():
        raise FileExistsError('freeze or request v2 exists')
    q = latest('QUALIFICATION_TD4v2_*.json')
    if not q:
        raise PermissionError('Run the v2 qualify first')
    qual = json.loads(Path(q).read_text(encoding='utf-8'))
    inv = G.code_inventory()
    if qual.get('code_sha256') != inv or qual.get('failures') != 0 or qual.get('errors') != 0:
        raise PermissionError('v2 qualification missing, failing or stale')
    receipts = build_receipts()
    stale = {p: r['changed_build_code'] for p, r in receipts.items() if r['changed_build_code']}
    if stale:
        raise PermissionError('build code changed after a masker-v2 build, or redaction missing: ' +
                              json.dumps(stale))
    pre = write_prereg()
    if pre['sha256'] != S.preregistration_sha256():
        raise PermissionError('PREREGISTRATION_v2_2.json differs from the spec text')
    evidence = v23_evidence(inv)
    record = provider_outputs_record()
    cost = json.loads(Path(latest('COST_PROJECTION_v2_*.json')).read_text(encoding='utf-8'))
    menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
    import td4_freeze as F1
    f = {'schema': 'TD4_FREEZE_A_v2_3', 'code_version': S.CODE_VERSION, 'rule_version': S.RULE_VERSION,
         'version': S.VERSION, 'periods': S.RUN_ORDER, 'run_order': S.RUN_ORDER,
         'test_years': S.TEST_YEARS, 'hard_seal': T.HARD_SEAL, 'code_sha256': inv,
         'artifact_sha256': {str(p): R.sha(p) for p in artifact_list()},
         'external_sha256': {str(p): R.sha(p) for p in external_list()},
         'provenance_sha256': {str(p): R.sha(p) for p in F1.provenance_list() if Path(p).exists()},
         'preregistration_sha256': pre['sha256'], 'aggregation_freeze_sha256': pre['aggregation_sha256'],
         'caps': cost['caps'], 'cost_projection': cost['all_periods'], 'deepseek_balance': cost['deepseek_balance'],
         'models': {'deepseek_model': S3.DEEPSEEK_MODEL, 'deepseek_thinking': S3.DEEPSEEK_THINKING,
                    'deepseek_returned_model_allowlist': sorted(S3.DEEPSEEK_RETURNED_MODEL_PRICES),
                    'jev_model': S3.JEV_MODEL, 'jev_price_input_per_m': S3.JEV_PRICE_INPUT_PER_M,
                    'probability_sum_tolerance': S3.PROBABILITY_SUM_TOLERANCE,
                    'td1_model_sha256': R.sha(T.TD1 / 'model_frozen_v2.pkl'),
                    'td2_model_sha256': R.sha(T.TD2 / 'MODEL_v2.json')},
         'question_sha256': S3.question_bundle(menu)[1],
         'runner': {'deepseek_balance_floor_usd': T.DEEPSEEK_BALANCE_FLOOR_USD, 'peak_windows_utc': S3.PEAK_WINDOWS_UTC,
                    'peak_guard_s': T.PEAK_GUARD_S, 'deepseek_concurrency': T.DEEPSEEK_CONCURRENCY,
                    'jev_concurrency': T.JEV_CONCURRENCY, 'reask_n': T.REASK_N},
         'mask_v2_builds': receipts,
         'masker': {'version': 'TD4-mask-v2', 'v1_masker_inputs_bound': False, 'masker_v3_used': False,
                    'withholding_rule': 'span-level redaction v2.2, three rounds (td4v2_redact22 with the reviewers\' '
                                        'detectors, F1 word boundaries, F2 builder locators, F3 no GB waiver; '
                                        'pre-registered in PREREGISTRATION_v2_2 inputs.redaction.v2_2_fix); rule '
                                        'v2.1 kept: MASKER_NO_SOURCE replaces NONAME/NONAME_AFTER_2022 for 2023-2025 '
                                        '(td4v2_nosource; inputs.redaction.v2_1_correction)',
                    'detector_flag_rule': 'superseded as a withholding rule (round 0 of the redaction)'},
         'redaction_runs': [str(r) for r in redaction_files()[1]],
         'superseded_v2_sha256': {k: R.sha(v) for k, v in S.SUPERSEDED_V2.items() if v.exists()},
         'terminal_client_error_caps': terminal_caps(),
         'names_2025_source': {'path': str(N25.CIZ), 'sha256': R.sha(N25.CIZ)},
         'v1_freeze_sha256': R.sha(T.OUT / 'FREEZE_TD4_A_v1.json'), 'v1_reused_data': V1_REUSED_DATA,
         'v1_superseded': V1_SUPERSEDED, 'qualification': q,
         'clearance_a_path': str(S.CLEARANCE_A), 'clearance_b_path': str(S.CLEARANCE_B),
         'provider_outputs_record': record, 'provider_calls': 'closed in v2.3: call and reask refuse',
         'selection_paths': {'selection_freeze': str(S.SELECTION_FREEZE), 'request_b': str(S.REQUEST_B),
                             'selection_v2_untouched': {k: str(v) for k, v in S.SELECTION_V2_RECORD.items()}},
         'evaluation_paths': {'latch': str(S.EVAL_LATCH), 'primary_result': str(S.EVAL_PRIMARY),
                              'result': str(S.EVAL_RESULT)},
         'evaluator_memory': {'job_memory_cap_bytes': T.JOB_MEMORY_CAP, 'target_peak_bytes': S.EVAL_MEMORY_TARGET_BYTES,
                              'evidence': evidence},
         'superseded_v2_2': {'freeze': str(S.SUPERSEDED_V2['freeze_v2_2']),
                             'request': str(S.SUPERSEDED_V2['request_v2_2']),
                             'clearance_a_v2_2_sha256': S.CLEARANCE_A_V2_2_SHA256}}
    f['compared_with_v2_2'] = compare_v2_2(f)
    R.new_json(S.FREEZE_A, f)
    req = {'schema': S.CLEARANCE_A_SCHEMA, 'status': 'REQUESTED', 'request_only': True,
           'freeze_sha256': R.sha(S.FREEZE_A), **{k: f[k] for k in G.A_BINDINGS},
           'note': 'Request only. The reviewer writes ' + str(S.CLEARANCE_A) + ' with status CLEARED and without '
                   'request_only. It authorises select (writes ' + str(S.SELECTION_FREEZE) + ') and request-b '
                   'only: call and reask are closed in v2.3, because every provider output was produced under '
                   'clearance A v2.2 (' + S.CLEARANCE_A_V2_2_SHA256 + ', provider_outputs_record). The pooled '
                   '2023-2025 evaluation needs clearance B v2.3 (' + str(S.CLEARANCE_B) + '), which binds this '
                   'clearance A v2.3 by its SHA256.'}
    R.new_json(S.REQUEST_A, req)
    return {'freeze': str(S.FREEZE_A), 'freeze_sha256': R.sha(S.FREEZE_A), 'request': str(S.REQUEST_A),
            'request_sha256': R.sha(S.REQUEST_A), 'artifacts': len(f['artifact_sha256']),
            'external': len(f['external_sha256']), 'code_files': len(inv),
            'compared_with_v2_2': {k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, list)
                                                                                else len(vv)) for kk, vv in v.items()})
                                   for k, v in f['compared_with_v2_2'].items()}}


def summary():
    R = _R()
    out = {'version': S.VERSION, 'code_sha256': G.code_inventory()}
    q = latest('QUALIFICATION_TD4v2_*.json')
    if q:
        d = json.loads(Path(q).read_text(encoding='utf-8'))
        out['qualification'] = {'path': q, 'sha256': R.sha(q), **{k: d[k] for k in ('tests', 'failures', 'errors',
                                                                                    'skipped')}}
    par = {}
    for p in sorted((T.OUT / 'parity').glob('PARITY_*.json'), key=lambda x: x.stat().st_mtime):
        d = json.loads(p.read_text(encoding='utf-8'))
        par[d['check']] = {'path': str(p), 'sha256': R.sha(p), 'equal': d.get('equal')}
    out['parity_latest_per_check'] = par
    checks = {}
    for d in period_dirs():
        for name in ['BUILD.json'] + [x.name for x in sorted(d.glob('POPULATION_CHECK_*.json'))] + \
                [x.name for x in sorted(d.glob('AUDIT_SAMPLE_*.json'))]:
            p = d / name
            if p.exists():
                x = json.loads(p.read_text(encoding='utf-8'))
                checks[f'{d.name}/{name}'] = {'sha256': R.sha(p), **{k: x.get(k) for k in
                                                                     ('counts', 'inputs', 'problems', 'withheld_inputs',
                                                                      'audit_inputs', 'scan_hits', 'input_bytes',
                                                                      'peak_working_set_bytes')}}
    out['builds_and_checks'] = checks
    for pat, key in (('REHEARSAL_v2_*.json', 'rehearsal'), ('COST_PROJECTION_v2_*.json', 'cost_projection')):
        p = latest(pat)
        if p:
            out[key] = {'path': p, 'sha256': R.sha(p)}
    out['dry_runs'] = {}
    for p in sorted(S.RUNS.glob('dryrun_*/DRYRUN_SUMMARY.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        out['dry_runs'][p.parent.name] = {k: d.get(k) for k in ('period', 'records', 'mock_calls', 'halt',
                                                                'usd_mock_tokens', 'resume_check')}
    for k, p in (('freeze_a_v2_3', S.FREEZE_A), ('request_a_v2_3', S.REQUEST_A), ('preregistration_v2_2', S.PREREG),
                 ('freeze_a_v1_unchanged', T.OUT / 'FREEZE_TD4_A_v1.json')):
        out[k] = {'path': str(p), 'sha256': R.sha(p)} if p.exists() else None
    out['label_note'] = S.LLM_LABEL_NOTE
    path = R.new_json(S.OUT / f'TD4_QUALIFICATION_v2_{R.utc().replace(":", "")}.json', out)
    return {'path': str(path), 'sha256': R.sha(path)}
