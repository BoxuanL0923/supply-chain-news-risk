"""Clearance gates and admission tokens for the v2 real steps (select needs A v2.3; evaluate needs B v2.3).

Same rules as v1 td4_gate with v2 paths and schemas. The code inventory is every td4/*.py (the v1 modules v2 reuses,
unchanged) plus every td4/v2/*.py; adding or editing any of them invalidates the clearance.

v2.3 (2026-11-09, fix brief F7): every provider call of 2122, 2023, 2024 and 2025 ran under clearance A v2.2; call and
reask now refuse (no further provider call is possible). Freeze A v2.3 carries a record of the provider outputs
(`provider_outputs_record`: per-period ledger digest and the SHA256 of every runs_v2/real_<P> file, taken at the
freeze); check_a recomputes it, so select and evaluate refuse if any ledger or output changed. admit_b checks the
once-only latch, the primary-result file and the result file first (no replay), then everything else, before the
evaluator writes the latch.
"""
import hashlib
import json
from pathlib import Path

import td4_spec as T
import td4v2_spec as S

STEPS_A = ('call', 'reask', 'select')
CLOSED_STEPS = ('call', 'reask')                  # v2.3: refused whatever the clearance
_tokens = {}


class Token:
    def __init__(self, step, clearance_sha):
        self.step, self.clearance_sha = step, clearance_sha


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def code_inventory(v1_dir=None, v2_dir=None):
    v1_dir, v2_dir = Path(v1_dir or T.CODE), Path(v2_dir or S.CODE_V2)
    return {str(p): sha(p) for p in sorted(list(v1_dir.glob('*.py')) + list(v2_dir.glob('*.py')))}


A_BINDINGS = ('version', 'code_version', 'code_sha256', 'artifact_sha256', 'external_sha256', 'preregistration_sha256',
              'aggregation_freeze_sha256', 'caps', 'models', 'question_sha256', 'periods', 'hard_seal',
              'run_order', 'provider_outputs_record')


def run_files(period, mode='real', runs=None):
    """Every file of runs_v2/<mode>_<period>/ (sessions, ledgers, responses, summaries), sorted."""
    root = Path(runs or S.RUNS) / f'{mode}_{period}'
    return sorted(p for p in root.rglob('*') if p.is_file())


def runs_digests(mode='real', runs=None):
    """Per period: the ledger digest (td4v2_select.ledger_digest: SHA256 of the sorted settle rows
    'req_key|attempt|http_status|response_sha256'), its settle-row count and the SHA256 of every run file."""
    import td4v2_select as SL
    saved = S.RUNS
    try:
        if runs is not None:
            S.RUNS = Path(runs)
        out = {}
        for p in S.RUN_ORDER:
            digest, n = SL.ledger_digest(p, mode)
            out[p] = {'ledger_digest': digest, 'settle_rows': n,
                      'files_sha256': {str(x): sha(x) for x in run_files(p, mode)}}
        return out
    finally:
        S.RUNS = saved


def check_runs_record(record, mode='real', runs=None):
    """Reasons why the provider outputs differ from the freeze's record (empty list: unchanged)."""
    if not isinstance(record, dict) or not isinstance(record.get('periods'), dict):
        return ['provider_outputs_record missing from the freeze']
    now = runs_digests(mode, runs)
    reasons = []
    for p in S.RUN_ORDER:
        a, b = record['periods'].get(p) or {}, now[p]
        if a.get('ledger_digest') != b['ledger_digest'] or a.get('settle_rows') != b['settle_rows']:
            reasons.append(f'provider outputs changed since the freeze: ledger digest of {p}')
        if a.get('files_sha256') != b['files_sha256']:
            reasons.append(f'provider outputs changed since the freeze: run files of {p}')
    return reasons


def check_a(clearance_path=None, freeze_path=None, inventory=None):
    clearance_path = Path(clearance_path or S.CLEARANCE_A)
    freeze_path = Path(freeze_path or S.FREEZE_A)
    if not clearance_path.is_file():
        return False, ['clearance file absent: ' + str(clearance_path)], None
    if not freeze_path.is_file():
        return False, ['freeze absent: ' + str(freeze_path)], None
    c = json.loads(clearance_path.read_text(encoding='utf-8'))
    f = json.loads(freeze_path.read_text(encoding='utf-8'))
    reasons = []
    if c.get('schema') != S.CLEARANCE_A_SCHEMA or c.get('status') != 'CLEARED' or 'request_only' in c:
        reasons.append('schema/status (a request is never a clearance)')
    if c.get('freeze_sha256') != sha(freeze_path):
        reasons.append('freeze_sha256')
    for k in A_BINDINGS:
        if c.get(k) != f.get(k):
            reasons.append('binding ' + k)
    if (inventory if inventory is not None else code_inventory()) != f.get('code_sha256'):
        reasons.append('code inventory differs from the freeze (added, removed or edited td4/*.py or td4/v2/*.py)')
    for field in ('artifact_sha256', 'external_sha256'):
        for p, h in (f.get(field) or {}).items():
            if not Path(p).exists() or sha(p) != h:
                reasons.append(f'{field} changed: {p}')
    if f.get('preregistration_sha256') != S.preregistration_sha256():
        reasons.append('pre-registration text in td4v2_spec differs from the freeze')
    if not S.PREREG.exists() or sha(S.PREREG) != f.get('preregistration_sha256'):
        reasons.append('PREREGISTRATION_v2_2.json missing or changed')
    if not reasons:                                   # v2.3: the provider outputs are those of the record
        reasons += check_runs_record(f.get('provider_outputs_record'))
    return not reasons, reasons, f


def admit(step, clearance_path=None, freeze_path=None, inventory=None):
    if step not in STEPS_A:
        raise ValueError('step')
    if step in CLOSED_STEPS:
        raise PermissionError(f'STOP: {step} is closed in v2.3: every provider call ran under clearance A v2.2 '
                              f'({S.CLEARANCE_A_V2_2_SHA256}); no further provider call is possible')
    ok, reasons, f = check_a(clearance_path, freeze_path, inventory)
    if not ok:
        raise PermissionError('STOP: clearance A v2.3 not valid for ' + step + ': ' + '; '.join(reasons[:6]))
    tok = Token(step, sha(clearance_path or S.CLEARANCE_A))
    _tokens[id(tok)] = tok
    return tok, f


B_BINDINGS = ('selection_freeze_sha256', 'preregistration_sha256', 'ledger_digest_test', 'models_sha256',
              'models_files_sha256', 'evaluation_years', 'maximum_label_end')


def once_only_files():
    """The once-only evaluation files: latch, primary result (v2.3 F4) and result. Any of them refuses admit_b."""
    return [p for p in (S.EVAL_LATCH, S.EVAL_PRIMARY, S.EVAL_RESULT) if Path(p).exists()]


def admit_b():
    """R1 (TD4C): every check runs here, before the evaluator writes the once-only latch: no latch, primary or
    result file (checked first: no replay); clearance A v2.3 still valid (including the provider-output record);
    clearance B v2.3 equals request B v2.3 on every binding and binds clearance A v2.3; the ledger digest is
    recomputed from the actual ledgers; the model-file map is mandatory and every file is re-hashed."""
    import td4v2_select as SL
    present = once_only_files()
    if present:
        raise PermissionError('STOP: once-only evaluation latch exists; no replay: ' + '; '.join(map(str, present)))
    ok, reasons, f = check_a()
    if not ok:
        raise PermissionError('STOP: clearance A v2.3 no longer valid: ' + '; '.join(reasons[:6]))
    if not S.CLEARANCE_B.is_file():
        raise PermissionError('STOP: clearance B v2.3 absent: ' + str(S.CLEARANCE_B) + '; no test label read or computed')
    b = json.loads(S.CLEARANCE_B.read_text(encoding='utf-8'))
    req = json.loads(S.REQUEST_B.read_text(encoding='utf-8')) if S.REQUEST_B.exists() else {}
    bad = []
    if b.get('schema') != S.CLEARANCE_B_SCHEMA or b.get('status') != 'CLEARED' or 'request_only' in b:
        bad.append('schema/status')
    if b.get('clearance_a_sha256') != sha(S.CLEARANCE_A):
        bad.append('clearance_a_sha256 (clearance B v2.3 must bind clearance A v2.3)')
    if not S.SELECTION_FREEZE.exists() or b.get('selection_freeze_sha256') != sha(S.SELECTION_FREEZE):
        bad.append('selection_freeze_sha256')
    if b.get('preregistration_sha256') != sha(S.PREREG):
        bad.append('preregistration_sha256')
    for k in B_BINDINGS:
        if b.get(k) != req.get(k):
            bad.append('binding ' + k)
    if b.get('evaluation_years') != S.TEST_YEARS or b.get('maximum_label_end') != S.LABEL_END_BOUND:
        bad.append('scope')
    digest, _ = SL.combined_test_digest()
    if b.get('ledger_digest_test') != digest:
        bad.append('ledger digest recomputed from the ledgers differs')
    mf = b.get('models_files_sha256')
    if not isinstance(mf, dict) or not mf:
        bad.append('models_files_sha256 missing')
    else:
        for p, h in mf.items():
            if not Path(p).exists() or sha(p) != h:
                bad.append('model file changed: ' + p)
        if b.get('models_sha256') != hashlib.sha256(json.dumps(mf, sort_keys=True).encode()).hexdigest():
            bad.append('models_sha256 differs from the file map')
        if sorted(mf) != sorted(str(p) for p in SL.models_files()):
            bad.append('model file list differs')
    if bad:
        raise PermissionError('STOP: clearance B v2.3 mismatch: ' + '; '.join(bad))
    tok = Token('evaluate', sha(S.CLEARANCE_B))
    _tokens[id(tok)] = tok
    return tok, f, b


def valid(token, steps):
    return isinstance(token, Token) and _tokens.get(id(token)) is token and token.step in steps


def status():
    ok, reasons, _ = check_a()
    return {'version': S.VERSION, 'code_version': S.CODE_VERSION,
            'provider_calls': 'closed in v2.3 (call and reask refuse; every period ran under clearance A v2.2)',
            'clearance_a_path': str(S.CLEARANCE_A), 'clearance_a_valid': ok, 'clearance_a_reasons': reasons[:10],
            'clearance_b_path': str(S.CLEARANCE_B), 'clearance_b_present': S.CLEARANCE_B.exists(),
            'freeze_a_v2_3': S.FREEZE_A.exists(), 'request_a_v2_3': S.REQUEST_A.exists(),
            'preregistration_v2_2': S.PREREG.exists(),
            'selection_freeze_v2_3': S.SELECTION_FREEZE.exists(), 'request_b_v2_3': S.REQUEST_B.exists(),
            'selection_v2_record_present': {k: Path(v).exists() for k, v in S.SELECTION_V2_RECORD.items()},
            'evaluation_latch_v2': S.EVAL_LATCH.exists(), 'evaluation_primary_v2': S.EVAL_PRIMARY.exists(),
            'evaluation_result_v2': S.EVAL_RESULT.exists()}
