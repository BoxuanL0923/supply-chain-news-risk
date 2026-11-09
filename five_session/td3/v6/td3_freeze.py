"""Freeze the TD3 pilot (v6, TD3F) and write the clearance REQUEST (never a clearance).

v6: the manual-read record must also list the audit-listed inputs ('must_read'), each read in its final content.

The freeze binds the td3/v6 code inventory, the frozen private mask table, the frozen inputs, the build and
verification receipts, the independent scan hits and their adjudication, the adjacency report and the builder's
adjacency review, the adjacency sweep and its row-by-row review, the manual-read record, the passing qualification
receipt, the questions, the models, thinking mode and returned-model rules, the caps and the prices (DeepSeek and
Jev). The inputs must have been built by the same input-producing code and mask table that are frozen. The manual read
(v5, td3_spec.MANUAL_READ_RULE) must cover, by readers that did not write the masking code, every input that no
reviewer read end to end and every input whose content changed from the previous build, with every reported item
adjudicated, zero confirmed names and zero masks that give their word back.
"""
import json
from pathlib import Path
import td3_spec as S
from td3_runtime import new_json, sha
from td3_runner import clearance_bindings


INPUT_MODULES = ('td3_spec.py', 'td3_mask.py', 'td3_build.py', 'td3_runtime.py', 'td3_scan.py')


def code_hashes(folder=S.CODE):
    return {str(p): sha(p) for p in sorted(Path(folder).glob('*.py'))}


def input_code(hashes, folder=S.CODE):
    return {str(Path(folder) / n): hashes.get(str(Path(folder) / n)) for n in INPUT_MODULES}


def latest_qualification(out=S.OUT, code_dir=S.CODE):
    current = code_hashes(code_dir)
    for p in sorted(Path(out).glob(S.QUALIFICATION_PREFIX + '*.json'), key=lambda q: q.stat().st_mtime, reverse=True):
        r = json.loads(p.read_text(encoding='utf-8'))
        if (r.get('tests', 0) > 0 and r.get('passed') == r.get('tests') and not r.get('failures')
                and not r.get('errors') and r.get('code_sha256') == current):
            return p
    raise PermissionError('A passing ' + S.QUALIFICATION_PREFIX + ' receipt for the current td3/v6 code is required')


def manual_read_check(m, inputs_sha):
    """Problems of a manual-read record (empty list when acceptable). v5 (MANUAL_READ_RULE): the earlier reviewer
    end-to-end read sets ('reviewer_read_sets') and the inputs changed from the previous build ('changed_inputs') are
    listed; every input outside the reviewer sets and every changed input is read end to end ('reads': {reader: ranks})
    by readers that are not the builder; every input whose content changed after its read was read again
    ('reread_after_change' empty or covered); every reported item is adjudicated; zero confirmed names and zero masks
    that give their word back."""
    out = []
    if m.get('inputs_sha256') != inputs_sha:
        out.append('manual read is for other inputs')
    reads = m.get('reads') or {}
    if not reads or any('builder' in str(k).lower() for k in reads):
        out.append('reads must be listed by reader, and no reader may be the builder')
    read = set()
    for v in reads.values():
        read |= set(v)
    if len(read) < S.MANUAL_READ_MIN:
        out.append(f'manual read covers fewer than {S.MANUAL_READ_MIN} inputs')
    if m.get('confirmed_focal_or_shown_names', 1) != 0 or m.get('inferable_masks', 1) != 0:
        out.append('manual read records confirmed names or masks that give their word back')
    reviewer = set()
    for v in (m.get('reviewer_read_sets') or {}).values():
        reviewer |= set(v)
    if not reviewer:
        out.append('earlier reviewer read sets not listed')
    changed = set(m.get('changed_inputs') or [])
    if 'changed_inputs' not in m:
        out.append('changed inputs not listed')
    must = set(m.get('must_read') or [])                     # v6: audit-listed inputs
    if not must:
        out.append('audit-listed inputs (must_read) not listed')
    required = (set(range(1, S.SAMPLE_N + 1)) - reviewer) | changed | must
    if not required <= read:
        out.append('a required input (never read by a reviewer, changed or audit-listed) is missing from the reads')
    final_read = set(m.get('final_content_read') or [])
    if not (changed | must) <= final_read:
        out.append('a changed or audit-listed input was not read in its final content')
    items = m.get('reported_items')
    if items is None or any(not it.get('adjudication') for it in items):
        out.append('reported items missing or not adjudicated')
    return out


def freeze(build_dir, out=S.OUT, code_dir=S.CODE, manual_read=None):
    build_dir = Path(build_dir)
    target, request = Path(out) / S.FREEZE_NAME, Path(out) / S.CLEARANCE_REQUEST_NAME
    for p in (target, request):
        if p.exists():
            raise FileExistsError(p)
    q = latest_qualification(out, code_dir)
    current = code_hashes(code_dir)
    build = json.loads((build_dir / 'BUILD.json').read_text(encoding='utf-8'))
    verify = json.loads((build_dir / 'VERIFY.json').read_text(encoding='utf-8'))
    inputs = build_dir / 'INPUTS.jsonl'
    if input_code(build['td3_code_sha256'], code_dir) != input_code(current, code_dir):
        raise PermissionError('Input-producing td3/v6 modules changed since the build; rebuild with a new --tag')
    tables = Path(S.MASK_TABLES)
    if not tables.exists() or sha(tables) != build.get('mask_tables_sha256') or \
            verify.get('mask_tables_sha256') != build.get('mask_tables_sha256'):
        raise PermissionError('The frozen mask table differs from the one used by the build and verify')
    if not verify.get('passed') or verify.get('inputs_sha256') != sha(inputs):
        raise PermissionError('VERIFY.json must pass for the current INPUTS.jsonl')
    adjudication = build_dir / 'SCAN_ADJUDICATION.json'
    if verify.get('scan_adjudication', {}).get('sha256') != sha(adjudication):
        raise PermissionError('SCAN_ADJUDICATION.json changed since VERIFY.json')
    adjacency, adj_review = build_dir / 'ADJACENCY.jsonl', build_dir / 'ADJACENCY_REVIEW.json'
    if verify.get('adjacency_review', {}).get('sha256') != sha(adj_review):
        raise PermissionError('ADJACENCY_REVIEW.json changed since VERIFY.json')
    sweep, sweep_review = build_dir / 'ADJACENCY_SWEEP.jsonl', build_dir / 'ADJACENCY_SWEEP_REVIEW.json'
    if verify.get('adjacency_sweep', {}).get('sha256') != sha(sweep_review):
        raise PermissionError('ADJACENCY_SWEEP_REVIEW.json changed since VERIFY.json')
    manual = Path(manual_read) if manual_read else build_dir / 'MANUAL_READ.json'
    if not manual.exists():
        raise PermissionError('The builder manual-read record is required: ' + str(manual))
    m = json.loads(manual.read_text(encoding='utf-8'))
    problems = manual_read_check(m, sha(inputs))
    if problems:
        raise PermissionError('Manual read not acceptable: ' + '; '.join(problems))
    for p, h in build['outputs_sha256'].items():
        if sha(p) != h:
            raise ValueError('Build output changed: ' + p)
    menu = build['menu']
    if menu != S.menu_rows():
        raise ValueError('Menu differs from TD2 risk_menu.csv')
    bundle, qsha = S.question_bundle(menu)
    if build['question_sha256'] != qsha:
        raise ValueError('Question bundle differs from the build')
    value = {'schema': S.FREEZE_SCHEMA, 'version': S.VERSION, 'exploratory': True,
             'code_dir': str(code_dir), 'code_sha256': current, 'td2_code_sha256': build['td2_code_sha256'],
             'build_code_sha256': build['td3_code_sha256'], 'input_modules': list(INPUT_MODULES),
             'input_modules_unchanged_since_build': True,
             'build_dir': str(build_dir), 'inputs_path': str(inputs), 'inputs_sha256': sha(inputs),
             'build_sha256': sha(build_dir / 'BUILD.json'), 'verify_sha256': sha(build_dir / 'VERIFY.json'),
             'sample_sha256': sha(build_dir / 'PILOT_SAMPLE.csv'), 'meta_sha256': sha(build_dir / 'INPUT_META.jsonl'),
             'mask_audit_sha256': sha(build_dir / 'MASK_AUDIT.json'),
             'scan_hits_sha256': sha(build_dir / 'SCAN_HITS.jsonl'), 'scan_adjudication_sha256': sha(adjudication),
             'adjacency_sha256': sha(adjacency), 'adjacency_review_sha256': sha(adj_review),
             'adjacency_sweep_sha256': sha(sweep), 'adjacency_sweep_review_sha256': sha(sweep_review),
             'mask_tables_path': str(tables), 'mask_tables_sha256': sha(tables),
             'manual_read_path': str(manual), 'manual_read_sha256': sha(manual),
             'external_source_sha256': build['external_source_sha256'],
             'qualification_path': str(q), 'qualification_sha256': sha(q),
             'question_sha256': qsha, 'questions': bundle, 'rules': S.rules(),
             **clearance_bindings(), 'model_checks': S.model_checks(),
             'deepseek_url': S.DEEPSEEK_URL, 'jev_url': S.JEV_URL, 'prices_usd_per_million': S.PRICES,
             'deepseek_returned_model_prices': S.DEEPSEEK_RETURNED_MODEL_PRICES,
             'jev_price_output_per_m': S.JEV_PRICE_OUTPUT_PER_M, 'price_source': S.PRICE_SOURCE,
             'interpretation_dated_state': S.INTERPRETATION_DATED_STATE,
             'limitation_context_dates': S.LIMITATION_CONTEXT_DATES,
             'limitation_context_identification': S.LIMITATION_CONTEXT_IDENTIFICATION,
             'td2_freeze_sha256': S.TD2_FREEZE_SHA256, 'clearance_expected': str(S.CLEARANCE),
             'supersedes': {'freezes': [str(Path(out) / n) for n in S.SUPERSEDED_FREEZES],
                            'reason': 'TD3C (v1), TD3BC (v2), TD3C3 (v3), TD3C4 (v4) and TD3C5 (v5) reviews NOT '
                                      'CLEARED; v1 to v5 are never to be cleared'},
             'review_findings_c4_path': str(S.REVIEW_FINDINGS_C4),
             'review_findings_c4_sha256': sha(S.REVIEW_FINDINGS_C4) if Path(S.REVIEW_FINDINGS_C4).exists() else None,
             'review_findings_c5_path': str(S.REVIEW_FINDINGS_C5),
             'review_findings_c5_sha256': sha(S.REVIEW_FINDINGS_C5) if Path(S.REVIEW_FINDINGS_C5).exists() else None,
             'scope': 'Selector pilot on 200 selection-period firm-weeks; no label, return or loss computed'}
    new_json(target, value)
    req = {'schema': S.CLEARANCE_SCHEMA, 'status': 'REQUESTED', 'request_only': True,
           'freeze_path': str(target), 'freeze_sha256': sha(target), 'code_sha256': current,
           'inputs_sha256': value['inputs_sha256'], 'question_sha256': qsha,
           'mask_tables_sha256': value['mask_tables_sha256'], **clearance_bindings(),
           'note': ('A reviewer who clears TD3 v6 writes ' + str(S.CLEARANCE) + ' with these fields, status CLEARED '
                    'and no request_only key. Editing or adding any td3/v6 *.py file, changing the inputs or the mask '
                    'table invalidates it.')}
    new_json(request, req)
    return target, request
