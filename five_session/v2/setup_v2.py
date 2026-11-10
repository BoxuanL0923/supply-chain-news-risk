"""Create an isolated, append-only correction delivery from reviewed sources."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
BASE = CODE.parent
sys.path.insert(0, str(BASE))
import runtime
runtime.limit()
import difflib
import json

OUT = runtime.OUT
RECEIPTS = OUT/'repair_v2'
RECEIPTS.mkdir(exist_ok=True)

def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one source fragment: ' + old)
    return text.replace(old, new)

def create(path, text):
    runtime.guard()
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)

def main():
    frozen = json.loads((OUT/'MODEL_FREEZE_v2.json').read_text())
    baseline = {}
    for field in ('code_sha256', 'input_sha256', 'manifest_sha256', 'prediction_sha256', 'selection_sha256'):
        for name, expected in frozen[field].items():
            actual = runtime.sha(name)
            if actual != expected:
                raise ValueError('Reviewed artifact changed: ' + name)
            baseline[name] = actual
    for name in ('model_frozen.pkl', 'SOURCE_SNAPSHOTS.json', 'MANIFEST_BUILD.json',
                 'MODEL_FREEZE.json', 'MODEL_FREEZE_v2.json', 'CLEARANCE_REQUEST.json', 'CLEARANCE_REQUEST_v2.json'):
        baseline[str(OUT/name)] = runtime.sha(OUT/name)
    runtime.new_json(RECEIPTS/'baseline.json', {'sha256': baseline})
    for path in sorted(BASE.glob('*.py')):
        text = path.read_text(encoding='utf-8')
        if path.name == 'core.py':
            text = replace_one(text,
                'def old_fit(x,y):\n    p=prepare_fit(x); z=np.column_stack([np.ones(len(x)),transform(x,p,False)])',
                'def old_fit(x,y,preprocess=None):\n    p=prepare_fit(x) if preprocess is None else {k:v.copy() for k,v in preprocess.items()}\n    z=np.column_stack([np.ones(len(x)),transform(x,p,False)])')
        if path.name == 'fit.py':
            text = replace_one(text, 'old_final=old_fit(f.loc[ref,OLD].to_numpy(float),y[ref])',
                "old_final=old_fit(f.loc[ref,OLD].to_numpy(float),y[ref],preprocess=old['preprocess'])")
            text = text.replace("MODEL_FREEZE.json", "MODEL_FREEZE_v3.json").replace("CLEARANCE_REQUEST.json", "CLEARANCE_REQUEST_v3.json")
        if path.name in ('fit.py', 'sources.py', 'verify.py', 'evaluate.py'):
            text = text.replace('model_frozen.pkl', 'model_frozen_v2.pkl').replace('predictions/year_', 'predictions_v2/year_')
        if path.name in ('fit.py', 'evaluate.py', 'verify.py', 'tests.py'):
            text = text.replace('TD1_CLEARANCE_v1', 'TD1_CLEARANCE_v2')
        if path.name == 'runtime.py':
            text = replace_one(text,
                "    amended=OUT/'MODEL_FREEZE_v2.json'\n    return amended if amended.exists() else OUT/'MODEL_FREEZE.json'",
                "    return OUT/'MODEL_FREEZE_v3.json'  # Correction delivery; no fallback to reviewed artifacts.")
        if path.name == 'run_td1.py':
            text = replace_one(text, 'limit()\nimport argparse', 'limit()\nimport runtime_safety\nruntime_safety.seal_labels()\nimport argparse')
            text = replace_one(text, "'evaluate','tests'", "'evaluate','tests','repair','correction-tests','traps'")
            text = replace_one(text, "    if a.step in ['snapshots','manifest']:",
                "    if a.step=='repair':\n        from repair import run\n        return run()\n    if a.step=='correction-tests':\n        from correction_tests import run\n        return run()\n    if a.step=='traps':\n        import lookahead_traps\n        return\n    if a.step in ['snapshots','manifest']:")
        if path.name == 'amend_freeze.py':
            # The old code-only amendment must never target the new numerical delivery.
            text = '"""The correction requires repair.py, not the legacy code-only amendment."""\nraise RuntimeError("Use run_td1.py repair; all reviewed freezes are immutable")\n'
        if path.name == 'verify.py':
            text = replace_one(text, 'from spec import FEATURES', 'from spec import FEATURES, OLD\nfrom validate_correction import assert_preprocessing')
            text = replace_one(text, 'checks+=2; training=[]; training_y=[]; selection=[]',
                'checks+=2; training=[]; training_y=[]; selection=[]; training_old=[]; refit_old=[]')
            text = replace_one(text, '            if year<=2020:\n',
                "            refit_old.append(f.loc[f.refit_included,OLD+['y']])\n            if year<=2020:\n                training_old.append(f.loc[f.fit_included,OLD])\n")
            text = replace_one(text, '    ty=np.concatenate(training_y)',
                "    # Independent statistics retain the fitter's column-major OLD layout.\n    checks+=assert_preprocessing(pd.concat(training_old,ignore_index=True)[OLD].to_numpy(float),bundle)\n    rf=pd.concat(refit_old,ignore_index=True)\n    from validate_correction import independent_beta\n    beta=independent_beta(rf[OLD].to_numpy(float),rf.y.to_numpy(float),bundle['selection_old']['preprocess'])\n    if not np.array_equal(beta,bundle['old']['beta']): raise ValueError('Comparator coefficients were not refit through 2022')\n    checks+=1\n    ty=np.concatenate(training_y)")
            text = replace_one(text, "target=OUT/'VERIFICATION.json'", "target=OUT/'VERIFICATION_v2.json'")
            text = text.replace("f'VERIFICATION_{uuid.uuid4().hex}.json'", "f'VERIFICATION_v2_{uuid.uuid4().hex}.json'")
        create(CODE/path.name, text)
    # Keep the independent 16 traps, adapting only their delivery paths and schema.
    review = Path('D:/OptResearch_data/five_session_review/td1c')
    traps = (review/'lookahead_traps.py').read_text(encoding='utf-8')
    traps = traps.replace('import offline_base as audit', 'import audit_adapter as audit')
    traps = traps.replace('model_frozen.pkl', 'model_frozen_v2.pkl').replace('TD1_CLEARANCE_v1', 'TD1_CLEARANCE_v2')
    create(CODE/'lookahead_traps.py', traps)
    diff = []
    for path in sorted(CODE.glob('*.py')):
        old = BASE/path.name
        if path.name == 'lookahead_traps.py': old = review/path.name
        before = old.read_text(encoding='utf-8') if old.exists() else ''
        diff.extend(difflib.unified_diff(before.splitlines(True), path.read_text(encoding='utf-8').splitlines(True),
            fromfile=str(old) if old.exists() else '/dev/null', tofile=str(path)))
    create(RECEIPTS/'SOURCE_DIFF_v2.patch', ''.join(diff))
    print('Versioned source:', CODE)
    print('Preserved artifact hashes:', len(baseline))

if __name__ == '__main__': main()
