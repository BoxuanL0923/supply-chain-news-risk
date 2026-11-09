"""Hash-bound TD2 v2 freeze and clearance REQUEST (never a clearance). No label is read."""
import json
from pathlib import Path
import io_v2 as io
from selectors import validate_slot
from runtime_v3 import OUT, TD1, CODE, guard, sha, new_json
from spec_v2 import VERSION, NAMES, SLOT_SCHEMAS, EVAL_YEARS, MAX_LABEL_END

CLEARANCE = TD1 / 'clearance/TD2_CLEARANCE_v1.json'
BINDINGS = ['version', 'code_sha256', 'artifact_sha256', 'td1_sha256', 'model_sha256', 'qualification_sha256']

def artifact_paths(out=OUT):
    out = Path(out)
    paths = [out / p for p in ('risk_menu.csv', 'EXPOSURE_DICTIONARY_v1.json', 'PREPARED_v1.json', 'NEWS_EVAL_v2.json',
                               'MODEL_v2.json', 'SELECTION_v2.json', 'TOKEN_COST_ESTIMATE_v2.json')]
    paths += sorted(p for p in (out / 'sources').glob('*') if p.is_file())
    paths += sorted((out / 'news_scores').glob('year_*.parquet'))
    paths += sorted((out / 'selector_slots').glob('*_v1.json'))
    for sub, pattern in [('features_v2', 'year_*.parquet'), ('evidence_v2', 'year_*.parquet'), ('coverage_v2', 'year_*.json')]:
        paths += sorted((out / sub).glob(pattern))
    return paths

def td1_paths(td1=TD1):
    td1 = Path(td1)
    paths = [td1 / 'MODEL_FREEZE_v3.json', td1 / 'model_frozen_v2.pkl']
    paths += [td1 / f'firm_week_manifest/year_{y}.parquet' for y in range(2015, 2025)]
    paths += [td1 / f'predictions_v2/year_{y}.parquet' for y in range(2015, 2025)]
    paths += [td1 / f'inputs/crsp_{y}.parquet' for y in EVAL_YEARS]
    return paths

def check_td1(hashes, td1=TD1):
    freeze = json.loads((Path(td1) / 'MODEL_FREEZE_v3.json').read_text(encoding='utf-8'))
    if hashes[str(Path(td1) / 'model_frozen_v2.pkl')] != freeze['model_sha256']:
        raise ValueError('TD1 model differs from its freeze')
    for field in ('manifest_sha256', 'prediction_sha256', 'input_sha256'):
        for p, h in hashes.items():
            if p in freeze[field] and freeze[field][p] != h:
                raise ValueError('TD1 artifact differs from its freeze: ' + p)
    return True

def build(qualification):
    guard()
    for p in (OUT / 'FREEZE_v2.json', OUT / 'CLEARANCE_REQUEST_v2.json'):
        if p.exists():
            raise FileExistsError(p)
    for year in range(2015, 2025):
        cov = io.json_read(io.COVERAGE / f'year_{year}.json')
        if sha(io.FEATURES / f'year_{year}.parquet') != cov['feature_sha256'] or \
           sha(io.EVIDENCE / f'year_{year}.parquet') != cov['evidence_sha256']:
            raise ValueError(f'Feature/evidence year {year} changed after build')
    model = io.json_read(OUT / 'MODEL_v2.json')
    for kind in SLOT_SCHEMAS:
        validate_slot(io.json_read(OUT / f'selector_slots/{kind}_v1.json'), kind, model['names'])
    code = {str(p): sha(p) for p in sorted(CODE.glob('*.py'))}
    artifacts = {str(p): sha(p) for p in artifact_paths()}
    td1 = {str(p): sha(p) for p in td1_paths()}
    check_td1(td1)
    binding = {'schema': 'TD2_FREEZE_v2', 'version': VERSION, 'exploratory': True, 'code_sha256': code,
               'artifact_sha256': artifacts, 'td1_sha256': td1, 'model_sha256': sha(OUT / 'MODEL_v2.json'),
               'qualification_sha256': {str(Path(qualification)): sha(qualification)},
               'evaluation_years': list(EVAL_YEARS), 'maximum_label_end': MAX_LABEL_END,
               'evaluation_run': False, 'clearance_expected': str(CLEARANCE),
               'scope': '2023-2024 exploratory historical reproduction; paired squared-error gains versus the frozen '
                        'TD1 MLP; label ends <=2024-12-31; 2023-2025 already examined in the existing risk study'}
    new_json(OUT / 'FREEZE_v2.json', binding)
    request = {'schema': 'TD2_CLEARANCE_v1', 'status': 'REQUESTED', 'request_only': True,
               'freeze_sha256': sha(OUT / 'FREEZE_v2.json'), **{k: binding[k] for k in BINDINGS},
               'evaluation_years': list(EVAL_YEARS), 'maximum_label_end': MAX_LABEL_END,
               'issuer_instructions': 'After independent review, the issuer writes ' + str(CLEARANCE) +
                                      ' with these fields, status CLEARED and no request_only key.'}
    new_json(OUT / 'CLEARANCE_REQUEST_v2.json', request)
    print('FREEZE_V2', str(OUT / 'FREEZE_v2.json'), 'REQUEST_ONLY', str(OUT / 'CLEARANCE_REQUEST_v2.json'), flush=True)
