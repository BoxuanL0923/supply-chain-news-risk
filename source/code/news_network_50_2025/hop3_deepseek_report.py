"""deepseek-v4-pro: post-freeze development readout, never evaluation data."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CONTROLS
from .hop3_analysis import analyze
from . import hop3_deepseek as ds


def fmt(value):
    if value is None:
        return 'unidentified'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(fmt(v) for v in value) + ']'
    return f'{value:.6g}'


def report(guard):
    out, root, gates = ds.OUT, ds.ROOT, ds.gates
    if not (out/'prediction_freeze.json').exists():
        if (out/'ACCOUNT_BALANCE_BLOCK.json').exists() and gates.read(out/'ACCOUNT_BALANCE_BLOCK.json')['status'] == 'BLOCKED':
            return account_balance_report(guard)
        projection = gates.read(out/'PILOT_PROJECTION.json')
        if projection['projected_full_usd'] > ds.CAP:
            return budget_stop_report(guard)
        raise ValueError('Full prediction freeze required before statistical reporting')
    resources = gates.resources('deepseek_post_freeze_analysis')
    cases = gates.read(root/'cases.json')['cases']
    freeze = gates.verify_freeze(out, cases)
    index, unique = ds.validate_inputs()
    ds.reconcile()
    if freeze['actual_cost_usd'] != ds.spent() or ds.spent() > ds.CAP:
        raise ValueError('Cost ledger/freeze mismatch')
    # Validate response bytes, request identity and every final answer before labels.
    quality_by_request = {}
    for item in freeze['predictions']:
        prediction = gates.read(out/'predictions'/item['file'])
        key = prediction['request_sha256']
        response_path = out/'responses'/(key+'.json')
        if prediction['response_file_sha256'] != gates.sha(response_path):
            raise ValueError('Frozen response bytes changed')
        saved = gates.read(response_path)
        if gates.digest(saved['request']) != key:
            raise ValueError('Response request hash mismatch')
        if saved['request'] != ds.request_body(unique[prediction['input']]['prompt']):
            raise ValueError('Response differs from frozen model input')
        answer = ds.parse_answer(saved, unique[prediction['input']]['evidence_ids'])
        if prediction['score'] != float(answer['score']):
            raise ValueError('Frozen score differs from cached final content')
        quality_by_request[key] = {'case_id': item['case_id'], 'request_sha256': key,
             'status': answer['citation_validation_status'], 'issues': answer['citation_validation_issues'],
             'returned_evidence_ids': answer['returned_evidence_ids'], 'canonical_eligible_ids': answer['evidence_ids'],
             'unresolved_evidence_ids': answer['unresolved_evidence_ids'],
             'exact_alias_resolution': answer.get('evidence_id_alias_resolution', [])}
    quality = {'unique_responses': len(quality_by_request),
               'with_exact_alias_resolution': sum(bool(q['exact_alias_resolution']) for q in quality_by_request.values()),
               'with_citation_quality_flags': sum(bool(q['issues']) for q in quality_by_request.values()),
               'with_unresolved_citations': sum(bool(q['unresolved_evidence_ids']) for q in quality_by_request.values()),
               'policy': 'All finite scores retained unchanged; unresolved IDs are never treated as verified sources. Raw responses and returned IDs preserved. No outcome-based case selection.'}
    ds.write(out/'citation_quality_audit_deepseek-v4-pro.json', {**quality, 'records': list(quality_by_request.values())})
    prelabel = datetime.now(timezone.utc).isoformat()
    ds.write(out/'PRE_LABEL_VERIFICATION.json', {'prediction_freeze_sha256': gates.sha(out/'prediction_freeze.json'),
             'verified_at': prelabel, 'predictions': 3000, 'unique_responses': len(unique), 'labels_opened': False,
             'response_hashes_and_scores_verified': True})
    # First and only label source: exact train/valid parquet files, behind full freeze.
    labels = pd.concat([gates.load_labels(guard, out, cases, split) for split in ('train', 'valid')], ignore_index=True)
    base = []
    for day in gates.DATES:
        batch = gates.read(root/'pools'/(day+'.json'))
        for pool in batch['pools'].values():
            c = gates.check_case(pool['case'])
            base.append({'model': ds.MODEL, 'case_id': c['case_id'], 'split': c['split'], 'date': day,
                         'ticker': c['ticker'], 'anchor': c['b'], **{k: pool['controls'][k] for k in CONTROLS},
                         **{f'candidate_exact_h{h}': v for h, v in pool['coverage']['candidate_exact_hops'].items()}})
    frames, analyses = {}, {}
    for mode in ds.MODES:
        frame = pd.DataFrame(base).merge(labels, on='case_id', validate='one_to_one')
        for arm in ds.ARMS:
            predictions = {r['case_id']: gates.read(out/'predictions'/r['file']) for r in freeze['predictions'] if r['mode'] == mode and r['arm'] == arm}
            scores = {cid: p['score'] for cid, p in predictions.items()}
            frame[arm] = frame.case_id.map(scores)
            frame[arm+'_citation_flagged'] = frame.case_id.map({cid: bool(p['prediction']['citation_validation_issues']) for cid, p in predictions.items()})
        frame['mode'] = mode
        frames[mode], analyses[mode] = frame, analyze(frame)
    matched = frames['all_paths'].copy()
    scope = gates.read(out/'scope_audit.json')['records']
    for r in scope:
        for h in (1, 2, 3):
            matched.loc[matched.case_id == r['case_id'], f"H{r['radius']}_accepted_exact_h{h}"] = r[f'accepted_exact_h{h}']
    for arm in ds.ARMS:
        matched['continuity_'+arm] = frames['continuity_only'][arm].to_numpy()
        matched['continuity_'+arm+'_citation_flagged'] = frames['continuity_only'][arm+'_citation_flagged'].to_numpy()
    if len(matched) != 300 or not np.isfinite(matched[['y', 'anchor', *ds.ARMS]].to_numpy(float)).all():
        raise ValueError('Incomplete development join')
    matched[matched['split'] == 'valid'].to_csv(out/'matched_validation.csv', index=False)
    matched.to_csv(out/'matched_development_deepseek-v4-pro.csv', index=False)
    coverage = gates.read(root/'coverage_audit.json')['by_split']
    continuity = gates.read(out/'continuity_summary.json')
    inspected = {s: {'exact_h3_sources': sum(r['accepted_exact_h3'] for r in scope if r['split'] == s and r['radius'] == 3),
                    'cases_with_h3': sum(r['accepted_exact_h3'] > 0 for r in scope if r['split'] == s and r['radius'] == 3),
                    'total_cases': sum(r['split'] == s and r['radius'] == 3 for r in scope)} for s in ('train', 'valid')}
    caps = [p['excerpt_chars'] for p in unique.values()]
    responses = [gates.read(p) for p in sorted((out/'responses').glob('*.json'))]
    http_errors = [gates.read(p) for p in (out/'http_errors').glob('*.json')]
    rate_rejections = sum(e.get('status') == 429 for e in http_errors)
    pilot = gates.read(out/'PILOT_PROJECTION.json')
    usage_totals = {k: sum(int(s['envelope']['usage'].get(k, 0)) for s in responses)
                    for k in ('prompt_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'completion_tokens')}
    continuity_frame = frames['continuity_only']
    if not np.array_equal(continuity_frame.H1.to_numpy(), continuity_frame.H2.to_numpy()) or not np.array_equal(continuity_frame.H2.to_numpy(), continuity_frame.H3.to_numpy()):
        raise ValueError('Unexpected nonzero continuity-only increment without verified chains')
    summary = {'status': 'COMPLETE', 'engineering_only': True, 'look_ahead_clean': False,
               'train_n': 200, 'valid_n': 100, 'evaluation_cases_opened': False, 'evaluation_labels_opened': False,
               'prediction_freeze_sha256': gates.sha(out/'prediction_freeze.json'),
               'protocol_sha256': gates.sha(out/'PROTOCOL.json'), 'frozen_at': freeze['frozen_at'],
               'pre_label_verification_at': prelabel, 'label_source_manifest_sha256': gates.sha(root/'label_source_manifest.json'),
               'analysis': analyses, 'coverage': coverage, 'inspected_h3_coverage': inspected,
               'continuity_preparation': continuity, 'continuity_source_audit': gates.read(out/'continuity_source_audit.json'),
               'continuity_increment_identification': 'H2=H1 and H3=H2 exactly; no verified multihop sources. g2/g3 are unidentified, no finite supported equivalence bound or power sample size.',
               'excerpt_char_cap': {'min': min(caps), 'median_unique_prompts': float(np.median(caps)), 'max': max(caps)},
               'input_checks': gates.read(out/'DEEPSEEK_INPUT_CHECKS.json'), 'model_switch': gates.read(out/'MODEL_SWITCH.json'),
               'unique_model_calls': len(responses), 'prediction_records': 3000,
               'http_rate_limit_rejections': rate_rejections,
               'server_models': sorted({s['server_model'] for s in responses}),
               'maximum_prompt_tokens': max(s['envelope']['usage']['prompt_tokens'] for s in responses),
               'maximum_prompt_byte_bound': max(s['prompt_byte_upper_bound'] for s in responses),
               'usage_totals': usage_totals, 'pilot_projection': pilot, 'actual_total_cost_usd': ds.spent(),
               'citation_quality': quality,
               'cost_ledger': str(ds.LEDGER), 'total_request_seconds': sum(s['seconds'] for s in responses),
               'resources': gates.resources('deepseek_report_complete'), 'initial_resources': resources, 'checks': gates.read(out/'TESTS_PASSED.json'),
               'report_code_sha256': gates.sha(Path(__file__)),
               'analysis_code_sha256': gates.sha(Path(__file__).with_name('hop3_analysis.py'))}
    if summary['server_models'] != [ds.MODEL] or len(responses) != len(unique):
        raise ValueError('Mixed models or missing responses')
    ds.write(out/'summary.json', summary)
    lines = ['# Cumulative hop-radius ablation — deepseek-v4-pro', '',
             '**Engineering run; deepseek-v4-pro is not look-ahead clean.** 50 firms; 200 train cases on October 1/8/15/22, 2025; 100 validation cases on October 30 and November 6. Evaluation cases and labels were never opened. These are predictive scope contrasts, with no causal transmission claim.', '',
             f"Pilot: 20 distinct calls, actual ${pilot['pilot_actual_usd']:.6f}; projected full run **${pilot['projected_full_usd']:.4f}**, or ${pilot['projected_with_25pct_contingency_usd']:.4f} with 25% contingency. Actual total **${ds.spent():.6f}**, under the $15 hard cap. {len(responses):,} distinct DeepSeek requests supply all 3,000 arm/panel records; identical request bodies reuse responses. Ledger: `{ds.LEDGER}`.", '',
             'The user authorized sending Reuters text to DeepSeek on 2026-10-26. Completed label-free Qwen preparation was retained; the pending protocol/index/report were superseded, and originals are in `handoff_archive_qwen3.8-27b/`. No Qwen endpoint or PID gate was used by this run. The prior session had one refused local connection attempt and no successful model response.', '',
             '| Arm | Train raw residual MSE | Valid raw residual MSE | Continuity train MSE | Continuity valid MSE |',
             '|---|---:|---:|---:|---:|']
    for arm in ds.ARMS:
        values = [analyses[m]['raw_residual_mse'][s][arm] for m, s in (('all_paths', 'train'), ('all_paths', 'valid'), ('continuity_only', 'train'), ('continuity_only', 'valid'))]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', f"Zero-score raw GARCH MSE: train {analyses['all_paths']['zero_score_mse']['train']:.6f}, valid {analyses['all_paths']['zero_score_mse']['valid']:.6f}.", '',
              'Raw residual MSE = mean((y − raw GARCH b − model score)²). The outcome-free raw GARCH anchor differs from PR #2’s development-fitted OLS scoring anchor; fitting that anchor before predictions would violate the requested label seal. All eleven conventional controls enter the post-freeze regressions.', '',
              'Regression: y = intercept + Xβ + g_own OWN + g1 H1 + g2(H2 − H1) + g3(H3 − H2) + e. X contains b, rv1, rv5, rv22, earnings_confirmed, calendar_unknown, post_earnings, pending_cash_deal, log_news_count, log_neighbor_count and log_degree. The graph-skills arm is reported as a separate raw forecast contrast.', '',
              '| Panel | Train-fit valid MSE: X + OWN | + H1 | + (H2 − H1) | + (H3 − H2) | H3 valid MSE gain over H2 |',
              '|---|---:|---:|---:|---:|---:|']
    for mode, analysis in analyses.items():
        nested = analysis['train_fitted_valid']['nested']
        lines.append('| '+mode+' | '+' | '.join(f"{n['valid_mse']:.6f}" for n in nested)+f" | {nested[-1]['valid_mse_gain_vs_previous']:.6f} |")
    lines += ['', 'Every regression in this table is fitted on the four train dates and evaluated without refitting on the two validation dates. Full train coefficient vectors, design ranks and nested results are in `summary.json`.', '',
              'Pooled train+valid score coefficients (deepseek-v4-pro; HC3 heteroskedastic and CR1 date-cluster inference):', '',
              '| Panel | Term | Estimate | HC3 SE | HC3 90% CI | Date-cluster SE | Date-cluster 90% CI |',
              '|---|---|---:|---:|---|---:|---|']
    for mode, analysis in analyses.items():
        for term in ('OWN', 'H1', 'H2_minus_H1', 'H3_minus_H2'):
            c = analysis['pooled_train_valid']['coefficients'][term]
            lines.append(f"| {mode} | {term} | {fmt(c['estimate'])} | {fmt(c['HC3']['se'])} | {fmt(c['HC3']['ci90'])} | {fmt(c['date_cluster']['se'])} | {fmt(c['date_cluster']['ci90'])} |")
    lines += ['', 'All pooled control coefficients and both SE types are also in `summary.json`. Intervals use t critical values, n minus design rank for HC3 and dates minus one for clustered inference. There are only six pooled date clusters (four train, two valid), so clustered intervals are fragile and exploratory. Rank-deficient terms are explicitly marked unidentified; pseudoinverse fitted predictions do not make those coefficients identifiable.', '',
              '| Panel | g3 smallest symmetric bound, HC3 | g3 smallest symmetric bound, date cluster | Partial H3-increment SD |',
              '|---|---:|---:|---:|']
    for mode, analysis in analyses.items():
        pooled = analysis['pooled_train_valid']
        bound = pooled['gamma3_smallest_symmetric_equivalence_bound']
        lines.append(f"| {mode} | {fmt(bound['HC3'])} | {fmt(bound['date_cluster'])} | {fmt(pooled['power']['partial_increment_sd'])} |")
    lines += ['', 'The sensitivity bound is max(abs(lower 90% CI), abs(upper 90% CI)), in units of log variance per one-unit H3−H2 score change. A strictly larger prespecified symmetric margin would pass a TOST-style interval criterion; no scientific equivalence margin was prespecified here. This outcome-dependent bound is a description of what the design can rule out, not proof of practical equivalence.', '',
              '| Panel | Assumed g3 effect | RMS log-risk effect | Cases for 80% power, HC3 | Cases for 80% power, new date clusters |',
              '|---|---:|---:|---:|---:|']
    for mode, analysis in analyses.items():
        for p in analysis['pooled_train_valid']['power']['grid']:
            lines.append(f"| {mode} | {p['gamma3_effect']} | {fmt(p['rms_log_risk_effect'])} | {fmt(p['HC3_cases_for_80pct_power'])} | {fmt(p['date_cluster_cases_for_80pct_power'])} |")
    lines += ['', 'Power uses the local normal approximation n_new = n[(z_.95 + z_.80)SE/effect]², two-sided α=.10 and 80% power. It treats the observed design and noise as stable. Cluster extrapolation rounds to 50 cases/date and requires new independent dates, not more firms on the existing dates. It is not a finite-sample guarantee, especially if the implied sample is too small to fit the design. Unidentified increments have no finite supported bound or sample-size estimate.', '',
              '**Continuity-only limitation:** no complete product-flow receipts were available. Direct typed paths remain, while every unsupported H2/H3 article is removed. Request deduplication makes H1=H2=H3 exactly in this panel; g2 and g3 are unidentified. This is missing evidence of continuity, not evidence of no economic transmission. H3_GRAPH may still differ because it adds direct-pair graph receipts and graph instructions.', '',
              '| Split | Exact H1 candidates | Exact H2 candidates | Exact H3 candidates | Share with any H3 candidate | Accessible H3 articles | Cases with accessible H3 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for split, cov in coverage.items():
        i = inspected[split]
        lines.append(f"| {split} | {cov['candidate_counts']['1']} | {cov['candidate_counts']['2']} | {cov['candidate_counts']['3']} | {cov['share_any_h3']:.1%} | {i['exact_h3_sources']} | {i['cases_with_h3']}/{i['total_cases']} |")
    lines += ['', 'Candidate counts refer to case-source rows. The new pool uses 8/6/6 reserves, fills to 20 network candidates and ranks outcome-free with frozen local MiniLM plus recency. Shortest distances are full-graph BFS distances, with one deterministic shortest witness per mapped reachable entity and all prefixes. Own-entity stock aliases and co-mentions are excluded. Additional dated intermediate articles unlock host inspection but do not enter the primary scoring slots.', '',
              'The earlier capped-enumeration preflight mislabeled 930 candidate rows as H3 when omitted two-hop routes existed, and admitted two own-entity aliases. Those inputs were archived without model forecasts or label reads. The corrected pool has zero independent full-graph radius mismatches. Alternative shortest witnesses are not exhaustively enumerated; source mapping and prefix coverage can still exclude useful evidence.', '',
              'The continuity validator requires dated, inspected verbatim product receipts for each directed supply edge, an explicit input-to-output link through every intermediate and an affected-item receipt at the endpoint. Edge type, graph vintage, revenue percentage, names, sector overlap and search-keyword overlap cannot supply those facts. Dated vendor keywords were audited for every multihop case-path; undated company profiles were excluded. Reuters text was host-inspected for retrieval, but this preparation did not manually extract new structured product-flow facts from all articles. That conservative choice can miss real continuity. See `continuity_audit/`, `continuity_source_audit.json` and the retained source receipts.', '',
              'Graph skills use the repository implementations of `/sc-graph-degree` and `/sc-graph-triangles`, with thresholds 90/75/0.5. Host numerical receipts are preserved in `graph_skill_receipts/`. The June effective-date reconstruction and mapped membership are held fixed, rather than certified archived acquisition vintages or changing monthly membership.', '',
              'Retrieval is a documented host-side fallback, inherited from preparation: all accessible source pages are inspected through Toolkit prefix gates; DeepSeek receives deterministic leading excerpts and selects up to eight supplied source IDs. This is not an autonomous DeepSeek retrieval agent or an exact replication of Claude’s native retrieval. Wider arms can contain more text and alter attention/order, so radius effects are not matched-compute effects or isolated causal graph transmission.', '',
              'Possible errors and checks performed', '',
              '- Only 200 train and 100 valid cases passed explicit split/date/window allowlists. Mixed live cases/controls, test/evaluation and sealed label files are blocked. Arrow label reads run only after verifying every one of the 3,000 frozen predictions and hashes. Response hashes, request bodies, final scores and model IDs were rechecked before the first label read.',
              f"- Freeze time: {freeze['frozen_at']}; pre-label verification: {prelabel}. Prediction freeze SHA256: `{summary['prediction_freeze_sha256']}`. `data_access_audit.json` records opened live paths; no evaluation cases or labels were opened.",
              f"- {summary['checks']['tests']} offline unittest checks passed, including complete/missing intermediate receipts, forged/future quotes, expired edges, graph directions, exact-radius regressions, host prefix support, inference recovery/rank failure, final-content parsing, token accounting, cost refusal, cache replay and unresolved-request refusal. Logs: `tests_deepseek-v4-pro.log`.",
              f"- Common controls, cases and existing source excerpts match across cumulative arms. Maximum actual prompt: {summary['maximum_prompt_tokens']:,} tokens; conservative maximum UTF-8-byte-plus-framing bound: {summary['maximum_prompt_byte_bound']:,}, both below 24,000. Excerpt caps: {summary['excerpt_char_cap']}. Two case caps were reduced by ten characters equally across all arms because the longer model identifier crossed the byte bound; this is recorded in MODEL_SWITCH.json. Leading excerpts can omit qualifications.",
              '- All requests use deepseek-v4-pro, temperature zero, the same requested seed and 16,000 maximum completion tokens for reasoning plus JSON. Provider determinism is not guaranteed. Only final message content supplies the score; reasoning is retained solely in the cached provider envelope. Identical requests reuse exactly the same answer.',
              f"- Citation audit across {quality['unique_responses']} unique scored responses: {quality['with_exact_alias_resolution']} use a unique byte-exact source-ID resolution; {quality['with_citation_quality_flags']} have citation quality flags, including {quality['with_unresolved_citations']} with unresolved IDs. Exact resolution accepts an omitted hash with unique month/line, or a mistyped month with exact line/hash. Other IDs remain explicitly unresolved. Returned IDs, raw content and score are preserved; invalid citations are not host-verified evidence. Every finite final-JSON score remains in the full matched panel, with per-arm citation flags in the CSV. See `citation_quality_audit_deepseek-v4-pro.json` and the pre-label protocol revision log.",
              '- Every request is persisted before sending; every received response is cached before score validation. The separate DeepSeek ledger counts cache-miss input, discounted cache-hit input and all completion tokens. Concurrent reservations use the full prompt byte bound and full completion cap, preventing the job from exceeding $15. An interrupted request with no cached response is never resent automatically and retains its full reservation.',
              f"- DeepSeek issued {rate_rejections} explicit HTTP 429 rate-limit rejections before inference. Their response bodies and request attempts were preserved; only these known unaccepted attempts may retry after 30 seconds. The run reduced concurrency from 64 to 48 after the provider reported a limit of 61. Unknown transport outcomes are not automatically retried. Successful scoring responses were not repeated.",
              '- Network is restricted to api.deepseek.com:443. Proxies and redirects are disabled. No other paid API, Qwen request, process kill or server restart was used. O4’s cost log was not touched.',
              '- CIM FreeVirtualMemory is permission-denied; Windows GlobalMemoryStatusEx and GetProcessMemoryInfo provide available commit and process peak working set. Stages wait below 5 GB available commit and fail at a 3 GB process peak. Model concurrency uses threads with small cached payloads; BLAS is single-threaded.',
              '- Exact joins, finite scores/labels, unique keys, input/protocol/pool checksums, strictly ordered train/valid dates and rank checks passed. The post-observation model, short excerpts, incomplete continuity receipts, fixed graph membership and six-date design materially limit inference.', '',
              'Reproduce/resume with the supplied interpreter, PYTHONPATH=source/code and single-threaded BLAS:', '',
              '```powershell',
              'python -m news_network_50_2025.hop3_deepseek --stage predict --workers 8',
              'python -m news_network_50_2025.hop3_deepseek --stage report',
              '```', '',
              'Repository changes remain uncommitted on pr2-local-extensions; see `FILES_CHANGED.json`. Large prompts, source text, request/response caches and scored development rows remain on D:.']
    text = '\n'.join(lines)+'\n'
    (out/'REPORT.md').write_text(text, encoding='utf-8')
    target = gates.REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    ds.write(out/'data_access_audit.json', {'live_files_opened': sorted(guard.opened), 'evaluation_accesses': 0,
             'prediction_freeze_sha256': summary['prediction_freeze_sha256'], 'labels_opened_after': prelabel,
             'network_requests_in_reporting_process': guard.network_requests})
    print(ds.MODEL, json.dumps({'pilot_projected_full_usd': pilot['projected_full_usd'],
          'actual_total_cost_usd': ds.spent(), 'valid_raw_mse': analyses['all_paths']['raw_residual_mse']['valid'],
          'g3': analyses['all_paths']['pooled_train_valid']['coefficients']['H3_minus_H2'], 'report': str(target)}), flush=True)


def budget_stop_report(guard):
    """A completed pilot may forbid the full run. Keep all labels sealed."""
    out, gates = ds.OUT, ds.gates
    projection = gates.read(out/'PILOT_PROJECTION.json')
    if projection['projected_full_usd'] <= ds.CAP:
        raise ValueError('Cost-stop report requires a projection over the user cap')
    index, unique = ds.validate_inputs()
    ds.reconcile()
    plan = gates.read(out/'PILOT_PLAN.json')['records']
    by_cell, manifest = {}, []
    for item in plan:
        payload = unique[item['input']]
        key = gates.digest(ds.request_body(payload['prompt']))
        response = out/'responses'/(key+'.json')
        saved = gates.read(response)
        answer = ds.parse_answer(saved, payload['evidence_ids'])
        by_cell[item['case_id'], item['mode'], item['arm']] = answer['score']
        manifest.append({**item, 'response_file': str(response), 'response_sha256': gates.sha(response),
                         'request_sha256': key, 'score': answer['score']})
    ds.write(out/'pilot_prediction_freeze_deepseek-v4-pro.json', {'predictions': manifest, 'frozen_at': ds.now(),
             'complete_3000_record_freeze': False, 'labels_must_remain_sealed': True})
    # Explicitly incomplete validation roster. No label field is opened or exported.
    cases = gates.read(ds.ROOT/'cases.json')['cases']
    roster = []
    for case in cases:
        gates.check_case(case)
        if case['split'] != 'valid':
            continue
        row = {'model': ds.MODEL, 'case_id': case['case_id'], 'date': case['date'], 'ticker': case['ticker'],
               'split': 'valid', 'status': 'INCOMPLETE_COST_PROJECTION_STOP', 'labels_opened': False}
        for arm in ds.ARMS:
            row[arm] = by_cell.get((case['case_id'], 'all_paths', arm))
            row['continuity_'+arm] = by_cell.get((case['case_id'], 'continuity_only', arm))
        roster.append(row)
    pd.DataFrame(roster).to_csv(out/'matched_validation.csv', index=False)
    summary = {'status': 'STOPPED_PILOT_PROJECTION_EXCEEDS_CAP', 'engineering_only': True,
               'look_ahead_clean': False, 'pilot_projection': projection, 'actual_total_cost_usd': ds.spent(),
               'unique_model_calls': len(list((out/'responses').glob('*.json'))),
               'requested_prediction_records': 3000, 'full_run_started': False,
               'train_n': 200, 'valid_n': 100, 'evaluation_cases_opened': False, 'evaluation_labels_opened': False,
               'train_labels_opened': False, 'valid_labels_opened': False,
               'analysis': None, 'analysis_unavailable_reason': 'User explicitly requires stopping before the full run if pilot projection exceeds $15. Complete predictions were not frozen, so no train/valid label may be read.',
               'matched_validation_note': 'Incomplete 100-case roster with pilot scores only; no labels or MSEs. Empty scores are unscored, never zero.',
               'pilot_prediction_freeze_sha256': gates.sha(out/'pilot_prediction_freeze_deepseek-v4-pro.json'),
               'coverage': gates.read(ds.ROOT/'coverage_audit.json')['by_split'],
               'continuity_preparation': gates.read(out/'continuity_summary.json'),
               'input_checks': gates.read(out/'DEEPSEEK_INPUT_CHECKS.json'),
               'checks': gates.read(out/'TESTS_PASSED.json'), 'model_switch': gates.read(out/'MODEL_SWITCH.json'),
               'cost_ledger': str(ds.LEDGER), 'resources': gates.resources('deepseek_cost_stop_report'),
               'report_code_sha256': gates.sha(Path(__file__))}
    ds.write(out/'summary.json', summary)
    lines = ['# Cumulative hop-radius ablation — deepseek-v4-pro', '',
             '**Stopped after the pilot as required by the $15 projection gate. Engineering run; not look-ahead clean.**', '',
             f"The 20-call pilot across OWN, H1, H2, H3 and H3_GRAPH cost **${projection['pilot_actual_usd']:.6f}**. Actual total job cost is **${ds.spent():.6f}**. Actual token usage projects **${projection['projected_full_usd']:.4f}** for all {len(unique):,} unique requests ({3_000:,} arm/panel records), exceeding the user's **$15 hard cap**. The 25% contingency projection is ${projection['projected_with_25pct_contingency_usd']:.4f}. The full run was not started.", '',
             'No train, validation or evaluation labels were read. Raw MSE, nested train-fit/valid regression, pooled HC3/date-cluster coefficients, 90% intervals, equivalence bounds and outcome-based power calculations are unavailable because all 3,000 predictions must be frozen before any label access. No partial-pilot performance is substituted for the requested experiment.', '',
             '| Pilot arm | Pilot calls | Mean completion tokens, including reasoning | Projected unique requests | Projected full cost, USD |',
             '|---|---:|---:|---:|---:|']
    for arm, value in projection['by_arm'].items():
        lines.append(f"| {arm} | {value['pilot_n']} | {value['mean_completion_tokens']:.1f} | {value['unique_requests']} | {value['projected_usd']:.6f} |")
    lines += ['', projection['method'], '',
              'The pilot selects four distinct prompt-byte quartiles per arm without outcomes. All unique prompt lengths are known; per-arm actual input-token/byte ratios and average completion tokens project the full run. Cache-hit discounts are ignored in the projection, but included in actual costs. These are estimated token costs at the supplied prices, rather than an independently reconciled provider invoice.', '',
              'The prepared design remains 50 firms, 200 train cases on October 1/8/15/22, 2025 and 100 validation cases on October 30/November 6. The model switch was explicitly authorized on 2026-10-26, including sending Reuters text to DeepSeek. Identical cases, seed, controls, evidence and model apply across arms; temperature is zero and reasoning is enabled with 16,000 completion tokens. Only final content is parsed as JSON scores.', '',
              'Completed preparation retained the corrected full-graph BFS shortest distances, own-entity alias exclusion, fresh 8/6/6 candidate reserves, host prefix inspection and deterministic excerpts. The independent radius audit has zero mismatches. The earlier capped preflight’s 930 false H3 rows and two own-entity aliases were archived before model forecasting. Prior Qwen preparation protocol/index/readiness/report are preserved under `handoff_archive_qwen3.8-27b/`; all new scoring outputs identify deepseek-v4-pro.', '',
              'No complete multihop product-continuity receipts exist in the prepared evidence. Every edge requires dated inspected product facts, every intermediate an explicit input/output link and the endpoint an affected-item receipt. Dated relationship keywords and sector overlap are insufficient; undated profiles were excluded. The continuity-only panel retains direct paths and has no identifiable H2/H3 increment. This is missing verification, not proof of no economic transmission. Reuters articles were not manually converted into new structured flow facts, so conservative classification can miss real continuity.', '',
              '| Split | Exact H1 candidates | Exact H2 candidates | Exact H3 candidates | Cases with any H3 |',
              '|---|---:|---:|---:|---:|']
    for split, cov in summary['coverage'].items():
        lines.append(f"| {split} | {cov['candidate_counts']['1']} | {cov['candidate_counts']['2']} | {cov['candidate_counts']['3']} | {cov['share_any_h3']:.1%} |")
    lines += ['', 'Artifacts: `PROTOCOL.json` records the model switch and reason; `PILOT_PLAN.json`, `PILOT_PROJECTION.json` and `pilot_prediction_freeze_deepseek-v4-pro.json` preserve the pilot. `matched_validation.csv` is an explicitly incomplete 100-case roster with available pilot scores only, no labels and blank unscored cells. `summary.json` records the cost stop and unavailable statistics. Requests and raw responses remain cached on D:; successful calls need not be repeated.', '',
              'Possible errors and checks performed', '',
              f"- {summary['checks']['tests']} offline unittest checks passed: continuity toy chains/missing intermediates, product and date receipts, forged quotes, graph direction, radius correction, prefix support, label/freeze guards, inference recovery/rank failure, cache replay, final-content parsing, price accounting, hard-cap refusal and unresolved-request refusal.",
              '- All 3,000 input records and 1,778 unique prompts passed hashes, split/date/window checks, equal-controls and cumulative-source-content checks. Two cases needed a ten-character excerpt-cap reduction equally across all arms to keep the longer model name within the conservative UTF-8-byte-plus-framing limit. Maximum bound is 23,998, below 24,000 tokens; actual pilot counts are retained in responses.',
              '- The guard blocks evaluation/sealed labels and mixed live cases/controls. No complete prediction freeze exists, so labels remain locked. The pilot freeze explicitly cannot unlock labels. No MSE, regression, confidence interval or power result was fabricated.',
              '- Requests are saved before sending, responses before JSON validation, and costs in the separate `deepseek_costs_hop3.csv`. Cache-miss input is $0.435/million, hits $0.0435/million and completion including reasoning $0.87/million. Full output reservations enforce the $15 cap; O4’s log is untouched. Network is limited to api.deepseek.com:443, with proxies and redirects blocked. This run made no Qwen call, waited for no PID, killed no process and restarted no server. The prior session had a refused local connection, without a successful model response.',
              '- CIM memory queries are permission-denied; Windows native commit and process memory checks were used, with a 3 GB per-process limit. Pilot execution is small in memory. Model familiarity with post-observation history, short excerpts, a fixed June graph/membership and only six dates limit any later empirical interpretation.',
              '- Pilot completion-token variability and small strata make the projection uncertain. The declared decision uses the point projection; no cheaper reasoning configuration or other model was substituted after observing costs. Any future configuration change requires a separately labeled protocol and pilot.', '',
              f"Cost ledger: `{ds.LEDGER}`. Repository edits remain uncommitted on pr2-local-extensions; see `FILES_CHANGED.json`."]
    text = '\n'.join(lines)+'\n'
    (out/'REPORT.md').write_text(text, encoding='utf-8')
    target = gates.REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    ds.write(out/'data_access_audit.json', {'live_files_opened': sorted(guard.opened), 'evaluation_accesses': 0,
             'train_labels_opened': False, 'valid_labels_opened': False, 'network_requests_in_reporting_process': guard.network_requests})
    print(ds.MODEL, json.dumps({'pilot_projected_full_usd': projection['projected_full_usd'], 'actual_total_cost_usd': ds.spent(),
          'main_statistics': 'NOT_ESTIMATED_LABELS_SEALED_COST_STOP', 'report': str(target)}), flush=True)


def account_balance_report(guard):
    """Reviewable partial result; insufficient provider funds cannot unlock labels."""
    out, gates = ds.OUT, ds.gates
    if gates.read(out/'ACCOUNT_BALANCE_BLOCK.json')['status'] != 'BLOCKED':
        raise ValueError('Account-balance report requires a confirmed block')
    index, unique = ds.validate_inputs()
    ds.reconcile()
    pilot = gates.read(out/'PILOT_PROJECTION.json')
    cells, records, by_request = {}, [], {}
    for item in index:
        payload = unique[item['input']]
        key = gates.digest(ds.request_body(payload['prompt']))
        response = out/'responses'/(key+'.json')
        if not response.exists():
            continue
        saved = gates.read(response)
        if saved['request_sha256'] != key or gates.digest(saved['request']) != key:
            raise ValueError('Partial response identity mismatch')
        answer = ds.parse_answer(saved, payload['evidence_ids'])
        value = {**item, 'score': float(answer['score']), 'prediction': answer, 'request_sha256': key,
                 'server_model': ds.MODEL, 'response_file_sha256': gates.sha(response),
                 'status': 'PARTIAL_ACCOUNT_BALANCE_BLOCK'}
        file = gates.digest({k: item[k] for k in ('case_id', 'mode', 'arm')})+'.json'
        ds.write(out/'partial_predictions_deepseek-v4-pro'/file, value)
        records.append({k: item[k] for k in ('case_id', 'mode', 'arm')} | {'file': file, 'sha256': gates.sha(out/'partial_predictions_deepseek-v4-pro'/file)})
        cells[item['case_id'], item['mode'], item['arm']] = value
        by_request[key] = answer
    cases = gates.read(ds.ROOT/'cases.json')['cases']
    ds.write(out/'PARTIAL_PREDICTION_FREEZE_deepseek-v4-pro.json', {'complete_3000_record_freeze': False,
             'labels_must_remain_sealed': True, 'frozen_at': ds.now(), 'cases_hash': gates.digest(cases),
             'protocol_sha256': gates.sha(out/'PROTOCOL.json'), 'predictions': records,
             'prediction_directory': 'partial_predictions_deepseek-v4-pro', 'unique_responses': len(by_request)})
    roster = []
    for case in cases:
        gates.check_case(case)
        row = {'model': ds.MODEL, 'case_id': case['case_id'], 'date': case['date'], 'ticker': case['ticker'],
               'split': case['split'], 'anchor': case['b'], 'status': 'PARTIAL_ACCOUNT_BALANCE_BLOCK', 'labels_opened': False}
        for mode, prefix in (('all_paths', ''), ('continuity_only', 'continuity_')):
            for arm in ds.ARMS:
                value = cells.get((case['case_id'], mode, arm))
                row[prefix+arm] = value['score'] if value else None
                row[prefix+arm+'_citation_flagged'] = bool(value['prediction']['citation_validation_issues']) if value else None
        roster.append(row)
    frame = pd.DataFrame(roster)
    frame[frame['split'] == 'valid'].to_csv(out/'matched_validation.csv', index=False)
    frame.to_csv(out/'matched_development_partial_deepseek-v4-pro.csv', index=False)
    error_records = [gates.read(p) for p in (out/'http_errors').glob('*.json')]
    rejected_402 = sum(e['status'] == 402 for e in error_records)
    rejected_429 = sum(e['status'] == 429 for e in error_records)
    quality = {'unique_scored_responses': len(by_request),
               'with_exact_alias_resolution': sum(bool(a.get('evidence_id_alias_resolution')) for a in by_request.values()),
               'with_citation_quality_flags': sum(bool(a['citation_validation_issues']) for a in by_request.values()),
               'with_unresolved_citations': sum(bool(a['unresolved_evidence_ids']) for a in by_request.values())}
    ds.write(out/'citation_quality_partial_deepseek-v4-pro.json', {**quality,
             'records': [{'request_sha256': key, 'status': a['citation_validation_status'],
                          'issues': a['citation_validation_issues'], 'returned_ids': a['returned_evidence_ids'],
                          'unresolved_ids': a['unresolved_evidence_ids'], 'exact_alias_resolution': a.get('evidence_id_alias_resolution', [])} for key, a in by_request.items()]})
    coverage = gates.read(ds.ROOT/'coverage_audit.json')['by_split']
    scope = gates.read(out/'scope_audit.json')['records']
    inspected = {s: {'h3_sources': sum(r['accepted_exact_h3'] for r in scope if r['split'] == s and r['radius'] == 3),
                     'cases': sum(r['accepted_exact_h3'] > 0 for r in scope if r['split'] == s and r['radius'] == 3)} for s in ('train', 'valid')}
    summary = {'status': 'BLOCKED_DEEPSEEK_ACCOUNT_INSUFFICIENT_BALANCE', 'engineering_only': True,
               'look_ahead_clean': False, 'train_n': 200, 'valid_n': 100,
               'pilot_projection': pilot, 'actual_total_cost_usd': ds.spent(), 'budget_usd': ds.CAP,
               'budget_remaining_usd': ds.CAP-ds.spent(), 'full_run_started': True,
               'unique_model_calls': len(by_request), 'required_unique_requests': len(unique),
               'remaining_unique_requests': len(unique)-len(by_request), 'partial_prediction_records': len(records),
               'required_prediction_records': 3000, 'complete_prediction_freeze': False,
               'partial_prediction_freeze_sha256': gates.sha(out/'PARTIAL_PREDICTION_FREEZE_deepseek-v4-pro.json'),
               'train_labels_opened': False, 'valid_labels_opened': False,
               'evaluation_cases_opened': False, 'evaluation_labels_opened': False,
               'analysis': None, 'analysis_unavailable_reason': 'HTTP 402 insufficient_balance_error stopped new inference. The full 3000 prediction freeze is incomplete, so all labels remain sealed; no partial-pilot MSE or regression is reported.',
               'http_status_counts': {'402': rejected_402, '429': rejected_429},
               'provider_block': gates.read(out/'ACCOUNT_BALANCE_BLOCK.json'), 'citation_quality_partial': quality,
               'matched_validation_note': '100-case incomplete roster; available numeric scores only, blank cells unscored, no label column.',
               'coverage': coverage, 'inspected_h3_coverage': inspected,
               'continuity': gates.read(out/'continuity_summary.json'),
               'checks': gates.read(out/'TESTS_PASSED.json'), 'input_checks': gates.read(out/'DEEPSEEK_INPUT_CHECKS.json'),
               'cost_ledger': str(ds.LEDGER), 'resources': gates.resources('deepseek_account_balance_report'),
               'report_code_sha256': gates.sha(Path(__file__))}
    ds.write(out/'summary.json', summary)
    lines = ['# Cumulative hop-radius ablation — deepseek-v4-pro', '',
             '**BLOCKED: DeepSeek API account returned HTTP 402, “Insufficient Balance.” Engineering run; not look-ahead clean.**', '',
             f"Pilot projection: **${pilot['projected_full_usd']:.6f}** for the full run, below the **$15 job cap**. Pilot actual: **${pilot['pilot_actual_usd']:.6f}** for 20 calls. Actual job total: **${ds.spent():.9f}**. Remaining job allowance: **${ds.CAP-ds.spent():.9f}**. The API account balance, rather than the job cap, prevented completion.", '',
             f"Saved: **{len(by_request):,}/{len(unique):,} distinct scored responses**, covering **{len(records):,}/3,000 arm/panel prediction cells**. Still needed: **{len(unique)-len(by_request):,} distinct requests**. All successful responses, original request bodies, usage, final content and reasoning are cached on D:. No successful scoring call was repeated.", '',
             '**No train, validation or evaluation labels were opened.** The full prediction freeze is incomplete. Raw residual MSE, the nested train-fitted validation regression, pooled HC3/date-cluster estimates, 90% CIs, g3 equivalence bound and outcome-based power calculations therefore remain **not estimated**. Partial scores are not used to infer performance or select cases.', '',
             'The prepared experiment is unchanged: 50 firms, 200 train cases (October 1/8/15/22, 2025), 100 validation cases (October 30/November 6); OWN, H1, H2, H3 and H3_GRAPH; all-path and continuity-only panels; common cases, seed, controls and deepseek-v4-pro. The user authorized sending Reuters text to DeepSeek on 2026-10-26. Protocol metadata records the switch from Qwen and its reason. The prior completed preparation and its original protocol/index/report are retained under `handoff_archive_qwen3.8-27b/`.', '',
             '| Split | Exact H1 candidates | Exact H2 candidates | Exact H3 candidates | Share with H3 candidate | Accessible H3 articles | Cases with accessible H3 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for split, cov in coverage.items():
        i = inspected[split]
        lines.append(f"| {split} | {cov['candidate_counts']['1']} | {cov['candidate_counts']['2']} | {cov['candidate_counts']['3']} | {cov['share_any_h3']:.1%} | {i['h3_sources']} | {i['cases']}/{cov['cases']} |")
    lines += ['', 'Coverage counts are case-source rows. Fresh 8/6/6 reserves use outcome-free MiniLM/recency selection and fill to 20 candidates. Full-graph BFS distances and own-entity alias exclusions replaced the initial capped enumeration. The independent corrected-pool radius audit has zero mismatches. The archived preflight had 930 false H3 rows and two own-entity aliases, detected before successful model inference or label reads. One deterministic shortest witness per mapped entity and all prefixes are retained; alternative paths are not exhaustive.', '',
              'No complete multi-hop product-continuity receipts exist in the prepared evidence. The validator requires a dated inspected product receipt for every edge, an explicit input/output link at every intermediate and an affected-item endpoint receipt. Relationship keywords, sector overlap, revenue percentages and topology do not establish this. Undated profiles were excluded; Reuters text was not manually converted into new structured flow facts. This conservative choice can miss real continuity. The continuity-only panel retains direct paths; H2−H1 and H3−H2 are structurally unidentified there. No finite equivalence bound or supported power sample size can be estimated for those absent increments.', '',
              'Scoring uses the preserved host-side Toolkit retrieval fallback and deterministic leading excerpts, not a native DeepSeek retrieval loop. Wider arms can contain more text and change source order/attention. H3_GRAPH adds host-computed degree/triangle receipts with thresholds 90/75/0.5. The June effective-date graph and mapped membership are held fixed and are not a certified archived monthly vintage.', '',
              'Possible errors and checks performed', '',
              f"- {summary['checks']['tests']} offline unittest checks pass. Coverage includes continuity toy graphs, missing/forged/future facts, exact radii, prefix inspection, evaluation/freeze guards, inference recovery and rank failure, effective-rank CR1 correction, token prices, final-content parsing, citation flags, cache reuse, hard-cap refusal, known 429 retries, and blocking after HTTP 402.",
              '- All 3,000 input records and 1,778 unique prompts pass hashes, cumulative-source and common-control checks. Maximum conservative prompt bound is 23,998 UTF-8 bytes plus framing, below 24,000; reported actual tokens are cached. Two case excerpt caps were reduced by ten characters across all their arms for the longer model identifier. Excerpts may omit qualifications.',
              f"- Among {quality['unique_scored_responses']} cached unique scores, {quality['with_exact_alias_resolution']} needed exact source-ID resolution and {quality['with_citation_quality_flags']} have citation-quality flags, including {quality['with_unresolved_citations']} with unresolved IDs. Exact resolution accepts omitted hashes with unique month/line or a mistyped month with exact line/hash. Other IDs stay unresolved, never host-verified. Numeric scores and raw answers remain unchanged; no cases are dropped. See `citation_quality_partial_deepseek-v4-pro.json`.",
              f"- API receipts contain {rejected_429} explicit pre-inference HTTP 429 rejection(s) and {rejected_402} HTTP 402 insufficient-balance rejections. Rejected calls produced no inference usage. All error bodies and attempt records are preserved. Confirmed 402/429 rejections release their reservation and may resume; unknown transport outcomes remain blocked to prevent duplicate inference. Concurrency was reduced from 64 to 48 when the provider reported a limit of 61.",
              '- The separate `deepseek_costs_hop3.csv` charges actual reported cache-miss input at $0.435/million, cache hits at $0.0435/million and all completion/reasoning tokens at $0.87/million. This is a usage-based estimate at the user-supplied prices, not a reconciled provider invoice. O4’s ledger was not accessed or changed. Full prompt/output reservations enforce the $15 cap.',
              '- Only api.deepseek.com:443 is authorized; proxies and redirects are disabled. This run made no Qwen request, waited for no PID, killed no process and restarted no server. The prior session logged a refused local connection and no successful model response.',
              '- Labels are blocked until a complete 3,000-record durable freeze and all hashes pass. The partial freeze explicitly cannot unlock labels. Mixed live case/control files and evaluation/sealed labels remain forbidden. `matched_validation.csv` is an incomplete 100-case roster with available scores, blank unscored cells and no labels. It is not a validation performance result.',
              '- Windows native commit/process memory checks replace permission-denied CIM queries. Scoring peak working set stayed below 0.3 GB; all stages enforce the 3 GB process cap. Model history familiarity, short excerpts, incomplete continuity, fixed graph membership and six pooled dates limit future inference.', '',
              'After replenishing the DeepSeek API account balance, resume with the original key in the environment; the $15 job cap and existing caches remain in force:', '',
              '```powershell', "$env:PYTHONPATH='source/code'", "$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'",
              'python -m news_network_50_2025.hop3_deepseek --stage predict --workers 24 --resume-after-balance',
              'python -m news_network_50_2025.hop3_deepseek --stage report', '```', '',
              'The first command reuses all cached responses and sends only missing or explicitly rejected requests. Unknown-outcome requests are never resent automatically. The second command verifies the complete prediction/response freeze before opening only train/valid labels, then produces the specified statistics and replaces this partial report. Repository changes remain uncommitted on pr2-local-extensions; `FILES_CHANGED.json` lists them.']
    text = '\n'.join(lines)+'\n'
    (out/'REPORT.md').write_text(text, encoding='utf-8')
    target = gates.REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    ds.write(out/'data_access_audit.json', {'live_files_opened': sorted(guard.opened), 'evaluation_accesses': 0,
             'train_labels_opened': False, 'valid_labels_opened': False, 'complete_prediction_freeze': False,
             'network_requests_in_reporting_process': guard.network_requests})
    print(ds.MODEL, json.dumps({'pilot_projected_full_usd': pilot['projected_full_usd'], 'actual_total_cost_usd': ds.spent(),
          'cached_responses': len(by_request), 'remaining_requests': len(unique)-len(by_request),
          'main_statistics': 'NOT_ESTIMATED_LABELS_SEALED_HTTP_402', 'report': str(target)}), flush=True)
