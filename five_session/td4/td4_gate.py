"""Clearance gates and admission tokens for the TD4 real steps (refit, call, reask, select, evaluate).

A step runs only when the reviewer's clearance file exists at its named path and matches the frozen package:
- clearance A (D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_A_v1.json): schema TD4_CLEARANCE_A_v1, status
  CLEARED, no request_only key, freeze_sha256 = SHA256 of FREEZE_TD4_A_v1.json, and every binding of that freeze
  (code inventory of td4/*.py, private input and source hashes, pre-registration and aggregation hashes, caps, models)
  equal; the code folder inventory and every bound file are re-hashed before admission;
- clearance B (TD4_CLEARANCE_B_v1.json): schema TD4_CLEARANCE_B_v1, binds clearance A's sha256, the selection freeze
  SELECTION_FREEZE_v1.json sha256, the pre-registration sha256 and the 2025 call-ledger digest; once-only latch.
Labels (td4_labels) and paid calls (td4_runner) refuse without the token returned by admit(). Nothing runs on import.
"""
import json
from pathlib import Path

import td4_spec as T

FREEZE_A = T.OUT / 'FREEZE_TD4_A_v1.json'
REQUEST_A = T.OUT / 'CLEARANCE_REQUEST_TD4_A_v1.json'
SELECTION_FREEZE = T.OUT / 'selection' / 'SELECTION_FREEZE_v1.json'
REQUEST_B = T.OUT / 'selection' / 'CLEARANCE_REQUEST_TD4_B_v1.json'
PREREG = T.OUT / 'PREREGISTRATION_v1.json'
AGGREGATION_FREEZE = T.OUT / 'AGGREGATION_FREEZE_v1.json'
EVAL_LATCH = T.OUT / 'evaluation' / 'EVALUATION_STARTED_v1.json'
EVAL_RESULT = T.OUT / 'evaluation' / 'EVALUATION_RESULT_v1.json'

STEPS_A = ('refit', 'call', 'reask', 'select')
_tokens = {}


class Token:
    def __init__(self, step, clearance_sha):
        self.step, self.clearance_sha = step, clearance_sha


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def code_inventory(folder=T.CODE):
    return {str(p): sha(p) for p in sorted(Path(folder).glob('*.py'))}


A_BINDINGS = ('version', 'code_sha256', 'artifact_sha256', 'external_sha256', 'preregistration_sha256',
              'aggregation_freeze_sha256', 'caps', 'models', 'question_sha256', 'periods', 'hard_seal')


def check_a(clearance_path=T.CLEARANCE_A, freeze_path=FREEZE_A, code_dir=T.CODE):
    """(ok, reasons, freeze). Re-hashes the code inventory and every bound artifact."""
    reasons = []
    if not Path(clearance_path).is_file():
        return False, ['clearance file absent: ' + str(clearance_path)], None
    if not Path(freeze_path).is_file():
        return False, ['freeze absent: ' + str(freeze_path)], None
    c = json.loads(Path(clearance_path).read_text(encoding='utf-8'))
    f = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if c.get('schema') != T.CLEARANCE_A_SCHEMA or c.get('status') != 'CLEARED' or 'request_only' in c:
        reasons.append('schema/status (a request is never a clearance)')
    if c.get('freeze_sha256') != sha(freeze_path):
        reasons.append('freeze_sha256')
    for k in A_BINDINGS:
        if c.get(k) != f.get(k):
            reasons.append('binding ' + k)
    inv = code_inventory(code_dir)
    if inv != f.get('code_sha256'):
        reasons.append('code inventory differs from the freeze (added, removed or edited td4/*.py)')
    for field in ('artifact_sha256', 'external_sha256'):
        for p, h in (f.get(field) or {}).items():
            if not Path(p).exists() or sha(p) != h:
                reasons.append(f'{field} changed: {p}')
    if f.get('preregistration_sha256') != T.preregistration_sha256():
        reasons.append('pre-registration text in td4_spec differs from the freeze')
    if not PREREG.exists() or sha(PREREG) != f.get('preregistration_sha256'):
        reasons.append('PREREGISTRATION_v1.json missing or changed')
    return not reasons, reasons, f


def admit(step, clearance_path=T.CLEARANCE_A, freeze_path=FREEZE_A, code_dir=T.CODE):
    if step not in STEPS_A:
        raise ValueError('step')
    ok, reasons, f = check_a(clearance_path, freeze_path, code_dir)
    if not ok:
        raise PermissionError('STOP: clearance A not valid for ' + step + ': ' + '; '.join(reasons[:6]))
    tok = Token(step, sha(clearance_path))
    _tokens[id(tok)] = tok
    return tok, f


def admit_b(clearance_b=T.CLEARANCE_B, clearance_a=T.CLEARANCE_A, freeze_path=FREEZE_A, code_dir=T.CODE):
    ok, reasons, f = check_a(clearance_a, freeze_path, code_dir)
    if not ok:
        raise PermissionError('STOP: clearance A no longer valid: ' + '; '.join(reasons[:6]))
    if not Path(clearance_b).is_file():
        raise PermissionError('STOP: clearance B absent: ' + str(clearance_b) + '; no 2025 label read or computed')
    if EVAL_LATCH.exists() or EVAL_RESULT.exists():
        raise PermissionError('STOP: once-only evaluation latch exists; no replay')
    b = json.loads(Path(clearance_b).read_text(encoding='utf-8'))
    req = json.loads(REQUEST_B.read_text(encoding='utf-8')) if REQUEST_B.exists() else {}
    bad = []
    if b.get('schema') != T.CLEARANCE_B_SCHEMA or b.get('status') != 'CLEARED' or 'request_only' in b:
        bad.append('schema/status')
    if b.get('clearance_a_sha256') != sha(clearance_a):
        bad.append('clearance_a_sha256')
    if not SELECTION_FREEZE.exists() or b.get('selection_freeze_sha256') != sha(SELECTION_FREEZE):
        bad.append('selection_freeze_sha256')
    if b.get('preregistration_sha256') != sha(PREREG):
        bad.append('preregistration_sha256')
    for k in ('selection_freeze_sha256', 'preregistration_sha256', 'ledger_digest_2025', 'models_sha256',
              'evaluation_year', 'maximum_label_end'):
        if b.get(k) != req.get(k):
            bad.append('binding ' + k)
    if b.get('evaluation_year') != T.TEST_YEAR or b.get('maximum_label_end') != T.HARD_SEAL:
        bad.append('scope')
    if bad:
        raise PermissionError('STOP: clearance B mismatch: ' + '; '.join(bad))
    tok = Token('evaluate', sha(clearance_b))
    _tokens[id(tok)] = tok
    return tok, f, b


def valid(token, steps):
    return isinstance(token, Token) and _tokens.get(id(token)) is token and token.step in steps


def status():
    ok, reasons, _ = check_a()
    return {'code': str(T.CODE), 'private_root': str(T.OUT), 'clearance_a_path': str(T.CLEARANCE_A),
            'clearance_a_valid': ok, 'clearance_a_reasons': reasons[:10], 'clearance_b_path': str(T.CLEARANCE_B),
            'clearance_b_present': T.CLEARANCE_B.exists(), 'freeze_a': FREEZE_A.exists(),
            'request_a': REQUEST_A.exists(), 'preregistration': PREREG.exists(),
            'selection_freeze': SELECTION_FREEZE.exists(), 'evaluation_latch': EVAL_LATCH.exists()}
