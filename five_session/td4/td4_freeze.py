"""Pre-registration, aggregation freeze, qualification lookup, freeze A and the clearance-A request (request only).

The freeze binds (re-hashed at every real-step admission): the td4/*.py inventory; TD4 private artifacts (inputs and
eligibility per period, article stores, 2025 extracts, manifest, features and evidence, pre-registration, aggregation
freeze, cost projection, qualification receipt); TD1/TD2/TD3 frozen artifacts the real steps read (manifests,
predictions, model, CRSP snapshots for labels, TD2 model, features, evidence, news scores, sources, TD3 mask table);
the TD1/TD2/TD3 code files TD4 imports; caps; models; question hash. Raw upstream sources (Reuters archive, JKP, CRSP
daily, FactSet files, CRSP names) are recorded under provenance_sha256 (hashed once, not re-hashed at admission:
everything the real steps need from them is in the bound extracts and stores).
"""
import glob
import json
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4_gate as G


def _R():
    E.paths()
    import td3_runtime as R
    return R


def write_prereg():
    R = _R()
    out = {}
    if not G.PREREG.exists():
        R.new_text(G.PREREG, T.preregistration_text())
    out['preregistration'] = {'path': str(G.PREREG), 'sha256': R.sha(G.PREREG),
                              'matches_spec': R.sha(G.PREREG) == T.preregistration_sha256()}
    if not G.AGGREGATION_FREEZE.exists():
        E.paths()
        import td3_spec as S3
        import spec as td2spec
        agg = {'schema': 'TD4_AGGREGATION_FREEZE_v1', 'version': T.VERSION,
               'td3_aggregation_rule': S3.AGGREGATION_RULE, 'td3_article_rule': S3.ARTICLE_RULE,
               'td3_input_rule': S3.INPUT_RULE, 'td2_aggregation_rule': td2spec.RULES['aggregation'],
               'td2_article_rule': td2spec.RULES['article'], 'td2_partner_cap': td2spec.PARTNER_CAP,
               'td2_lookback_days': td2spec.LOOKBACK_DAYS,
               'frozen_before': 'any 2025 outcome is read (the evaluator requires this file through the freeze)'}
        R.new_json(G.AGGREGATION_FREEZE, agg)
    out['aggregation_freeze'] = {'path': str(G.AGGREGATION_FREEZE), 'sha256': R.sha(G.AGGREGATION_FREEZE)}
    return out


def latest(pattern):
    files = sorted(glob.glob(str(T.OUT / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return files[-1] if files else None


def artifact_list():
    paths = []
    for y in T.PERIODS:
        d = T.OUT / 'inputs' / f'period_{y}'
        paths += [d / n for n in ('INPUTS.jsonl', 'INPUT_META_COMPACT.jsonl', 'ELIGIBLE.parquet', 'BUILD.json',
                                  'WITHHELD.jsonl', 'AUDIT_META.jsonl', 'AUDIT_SWEEP.jsonl',
                                  f'POPULATION_CHECK_v2_{y}.json', f'WITHHELD_POPULATION_{y}.json',
                                  f'AUDIT_SAMPLE_{y}.json')]
        paths.append(T.OUT / 'inputs' / f'article_store_{y}.parquet')
        r = T.OUT / 'inputs' / f'repair_{y}'
        paths += [r / n for n in ('INPUTS.jsonl', 'INPUT_META_COMPACT.jsonl', 'ELIGIBLE.parquet', 'BUILD.json',
                                  'WITHHELD.jsonl', 'AUDIT_META.jsonl', 'AUDIT_SWEEP.jsonl',
                                  f'POPULATION_CHECK_v2_{y}.json', f'WITHHELD_POPULATION_{y}.json',
                                  f'AUDIT_SAMPLE_{y}.json')]
        paths.append(T.OUT / 'inputs' / f'article_store_repair_{y}.parquet')
    for sub in ('td1_inputs', 'td2_sources', 'panel_keys'):
        paths += sorted((T.OUT / sub).glob('*.parquet'))
    paths += [T.OUT / 'labeling' / n for n in ('LABEL_PACKET_v1.jsonl', 'PACKET_KEYS_v1.json', 'LABEL_TEMPLATE_v1.csv',
                                               'LABELING_INSTRUCTIONS_v1.txt')]
    paths += [T.OUT / 'firm_week_manifest/year_2025.parquet', T.OUT / 'features/year_2025.parquet',
              T.OUT / 'evidence/year_2025.parquet', G.PREREG, G.AGGREGATION_FREEZE]
    for pat in ('COST_PROJECTION_*.json', 'QUALIFICATION_TD4_*.json'):
        p = latest(pat)
        if p:
            paths.append(Path(p))
    return [p for p in paths if Path(p).exists()]


def external_list():
    ext = [T.TD1 / 'MODEL_FREEZE_v3.json', T.TD1 / 'model_frozen_v2.pkl', T.TD1 / 'inputs/identity.parquet']
    ext += [T.TD1 / f'firm_week_manifest/year_{y}.parquet' for y in range(2015, 2025)]
    ext += [T.TD1 / f'predictions_v2/year_{y}.parquet' for y in range(2022, 2025)]
    ext += [T.TD1 / f'inputs/crsp_{y}.parquet' for y in (2022, 2023, 2024)]
    ext += [T.TD2 / n for n in ('FREEZE_v2.json', 'MODEL_v2.json', 'risk_menu.csv', 'EXPOSURE_DICTIONARY_v1.json')]
    ext += [T.TD2 / f'features_v2/year_{y}.parquet' for y in range(2015, 2025)]
    ext += [T.TD2 / f'evidence_v2/year_{y}.parquet' for y in (2023, 2024)]
    ext += [T.TD2 / f'news_scores/year_{y}.parquet' for y in range(2014, 2025)]
    ext += [T.TD2 / 'sources' / n for n in ('graph_october_sealed.parquet', 'factset_links.parquet',
                                           'identity_sensitivity_v1.json', 'reuters_identity_development_v1.json',
                                           'quarantine_ids_v1.json')]
    ext += [T.TD3 / 'tables_v6/MASK_TABLES_v4.json', T.TD3 / 'build_v7/INPUTS.jsonl',
            T.TD3 / 'build_v7/PILOT_SAMPLE.csv', T.TD3_PILOT_RUN / 'responses.jsonl', T.TD3 / 'FREEZE_TD3_v6.json']
    ext += sorted(T.TD3_CODE.glob('td3_*.py'))
    ext += [T.TD2_CODE / n for n in ('engine.py', 'fast_builder.py', 'build_v2.py', 'decoder_v3.py', 'spec.py',
                                     'spec_v2.py', 'io_v2.py', 'runtime_v3.py', 'gates_v2.py', 'statistics.py',
                                     'selectors.py')]
    ext += [T.TD1_CODE / 'spec.py', T.TD1_CODE / 'core.py']
    return [p for p in ext if Path(p).exists()]


def provenance_list():
    return [T.CRSP_DAILY, T.JKP, T.QUARTERLY, T.EARNINGS, T.GVKEY_LINK, T.FS_MAP, T.ARTICLES, *T.CRSP_NAME_FILES,
            T.PANEL, T.FACTSET_RELATIONSHIPS, T.FACTSET_SUPPLY_CHAIN, T.FSB5, T.FSB1_COMPANY,
            Path('C:/Users/Boxuan/AppData/Roaming/nltk_data/corpora/words/en')]


def freeze():
    R = _R()
    import td4_cost as C
    E.paths()
    import td3_spec as S3
    if G.FREEZE_A.exists() or G.REQUEST_A.exists():
        raise FileExistsError('freeze A or request A already exists')
    q = latest('QUALIFICATION_TD4_*.json')
    if not q:
        raise PermissionError('Run qualify first')
    qual = json.loads(Path(q).read_text(encoding='utf-8'))
    inv = G.code_inventory()
    if qual.get('code_sha256') != inv:
        raise PermissionError('Qualification receipt does not match the current code inventory; re-run qualify')
    if not (qual.get('failures') == 0 and qual.get('errors') == 0):
        raise PermissionError('Qualification has failures')
    prereg = write_prereg()
    cost = json.loads(Path(latest('COST_PROJECTION_*.json')).read_text(encoding='utf-8'))
    menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
    f = {'schema': 'TD4_FREEZE_A_v1', 'version': T.VERSION, 'periods': list(T.PERIODS), 'hard_seal': T.HARD_SEAL,
         'code_sha256': inv,
         'artifact_sha256': {str(p): R.sha(p) for p in artifact_list()},
         'external_sha256': {str(p): R.sha(p) for p in external_list()},
         'provenance_sha256': {str(p): R.sha(p) for p in provenance_list() if Path(p).exists()},
         'preregistration_sha256': prereg['preregistration']['sha256'],
         'aggregation_freeze_sha256': prereg['aggregation_freeze']['sha256'],
         'caps': cost['caps'], 'cost_projection': cost['both_periods'],
         'models': {'deepseek_model': S3.DEEPSEEK_MODEL, 'deepseek_thinking': S3.DEEPSEEK_THINKING,
                    'deepseek_returned_model_allowlist': sorted(S3.DEEPSEEK_RETURNED_MODEL_PRICES),
                    'jev_model': S3.JEV_MODEL, 'jev_price_input_per_m': S3.JEV_PRICE_INPUT_PER_M,
                    'probability_sum_tolerance': S3.PROBABILITY_SUM_TOLERANCE},
         'question_sha256': S3.question_bundle(menu)[1],
         'runner': {'deepseek_balance_floor_usd': T.DEEPSEEK_BALANCE_FLOOR_USD, 'peak_windows_utc': S3.PEAK_WINDOWS_UTC,
                    'peak_guard_s': T.PEAK_GUARD_S, 'deepseek_concurrency': T.DEEPSEEK_CONCURRENCY,
                    'jev_concurrency': T.JEV_CONCURRENCY, 'jev_429_backoff_s': list(T.JEV_429_BACKOFF_S),
                    'reask_n': T.REASK_N},
         'qualification': q, 'clearance_a_path': str(T.CLEARANCE_A), 'clearance_b_path': str(T.CLEARANCE_B)}
    R.new_json(G.FREEZE_A, f)
    req = {'schema': T.CLEARANCE_A_SCHEMA, 'status': 'REQUESTED', 'request_only': True,
           'freeze_sha256': R.sha(G.FREEZE_A), **{k: f[k] for k in G.A_BINDINGS},
           'note': 'Request only. The reviewer writes ' + str(T.CLEARANCE_A) + ' with status CLEARED, without '
                   'request_only, after the audit. It authorises refit, call (2024 then 2025), reask and select; the '
                   '2025 evaluation needs clearance B.'}
    R.new_json(G.REQUEST_A, req)
    return {'freeze': str(G.FREEZE_A), 'freeze_sha256': R.sha(G.FREEZE_A), 'request': str(G.REQUEST_A),
            'request_sha256': R.sha(G.REQUEST_A), 'artifacts': len(f['artifact_sha256']),
            'external': len(f['external_sha256']), 'code_files': len(inv)}


def summary():
    """TD4_QUALIFICATION_v1.json: tests, parity receipts, rehearsal, audit samples, dry runs, cost, code and freeze
    hashes (one JSON for the reviewer)."""
    R = _R()
    out = {'version': T.VERSION, 'code_sha256': G.code_inventory()}
    q = latest('QUALIFICATION_TD4_*.json')
    out['qualification'] = {'path': q, 'sha256': R.sha(q), **{k: json.loads(Path(q).read_text(encoding='utf-8'))[k]
                                                             for k in ('tests', 'failures', 'errors', 'skipped')}} \
        if q else None
    par = {}
    for p in sorted((T.OUT / 'parity').glob('PARITY_*.json'), key=lambda x: x.stat().st_mtime):
        d = json.loads(p.read_text(encoding='utf-8'))
        par[d['check']] = {'path': str(p), 'sha256': R.sha(p), 'equal': d.get('equal'),
                           'written_utc': d.get('written_utc')}
    out['parity_latest_per_check'] = par
    reh = latest('REHEARSAL_*.json')
    if reh:
        d = json.loads(Path(reh).read_text(encoding='utf-8'))
        out['rehearsal'] = {'path': reh, 'sha256': R.sha(reh), 'evaluate_replay': d['steps'].get('evaluate_replay'),
                            'select_best_arm': d['steps'].get('select', {}).get('best_pipeline_arm')}
    out['audit_samples'] = {}
    for y in T.PERIODS:
        p = T.OUT / 'inputs' / f'period_{y}' / f'AUDIT_SAMPLE_{y}.json'
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            out['audit_samples'][str(y)] = {'path': str(p), 'sha256': R.sha(p), 'audit_inputs': d['audit_inputs'],
                                            'scan_hits': d['scan_hits'], 'problems': d['problems']}
        for kind, sub in (('main', f'period_{y}'), ('repair', f'repair_{y}')):
            b = T.OUT / 'inputs' / sub / 'BUILD.json'
            if b.exists():
                d = json.loads(b.read_text(encoding='utf-8'))
                out.setdefault('builds', {})[f'{y}_{kind}'] = {'path': str(b), 'sha256': R.sha(b),
                                                                'counts': d['counts'], 'input_bytes': d['input_bytes'],
                                                                'elapsed_seconds': d['elapsed_seconds'],
                                                                'peak_working_set_bytes': d['peak_working_set_bytes']}
            for name in (f'POPULATION_CHECK_v2_{y}.json', f'AUDIT_SAMPLE_{y}.json'):
                q = T.OUT / 'inputs' / sub / name
                if q.exists():
                    d = json.loads(q.read_text(encoding='utf-8'))
                    out.setdefault('checks', {})[f'{y}_{kind}_{name.split("_")[0].lower()}'] = {
                        'path': str(q), 'sha256': R.sha(q),
                        **{k: d.get(k) for k in ('inputs', 'audit_inputs', 'problems', 'withheld_inputs', 'scan_hits')}}
    out['dry_runs'] = {}
    for p in sorted((T.OUT / 'runs').glob('dryrun_*/DRYRUN_SUMMARY.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        out['dry_runs'][p.parent.name] = {k: d.get(k) for k in ('year', 'records', 'mock_calls', 'halt',
                                                                'usd_by_provider_mock_tokens', 'resume_check',
                                                                'simulated_peak_wait_s')}
    c = latest('COST_PROJECTION_*.json')
    if c:
        d = json.loads(Path(c).read_text(encoding='utf-8'))
        out['cost_projection'] = {'path': c, 'sha256': R.sha(c), 'both_periods': d['both_periods'],
                                  'deepseek_balance': d['deepseek_balance'], 'caps': d['caps']}
    for k, p in (('freeze_a', G.FREEZE_A), ('request_a', G.REQUEST_A), ('preregistration', G.PREREG),
                 ('aggregation_freeze', G.AGGREGATION_FREEZE)):
        out[k] = {'path': str(p), 'sha256': R.sha(p)} if p.exists() else None
    path = R.new_json(T.OUT / f'TD4_QUALIFICATION_{R.utc().replace(":", "")}.json', out)
    return {'path': str(path), 'sha256': R.sha(path)}
