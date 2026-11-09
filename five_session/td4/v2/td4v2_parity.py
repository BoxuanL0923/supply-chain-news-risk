"""v2 parity: the v2 selection scoring path reproduces TD2's own 2021-2022 selection numbers.

On TD2's selection rows (TD1 selection_included, development labels only), the v2 path (td4v2_score.selection_b =
TD1 selection_mlp b, td4v2_score.selection_heads = TD2 2015-2020 training heads, TD2 gates) must give the TD2
SELECTION_v2.json MSE of every chosen local gate (training_beta K=1 none; random seeds K=5 none).
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E


def selection_td2(**_):
    import td4_models as MD
    import td4_score as SC1
    import td4_gates as Q
    import td4v2_score as SC2
    f = pd.concat([MD.feature_frame(y) for y in (2021, 2022)], ignore_index=True)
    lab = pd.concat([pq.read_table(T.TD1 / f'firm_week_manifest/year_{y}.parquet',
                                   columns=['firm_id', 'cutoff', 'y', 'selection_included']).to_pandas()
                     for y in (2021, 2022)], ignore_index=True)
    f = f.merge(lab, on=['firm_id', 'cutoff'], how='left', validate='one_to_one')
    f = f.merge(SC2.selection_b((2021, 2022)), on=['firm_id', 'cutoff'], how='left', validate='one_to_one')
    f = f[f.selection_included.fillna(False).astype(bool)].reset_index(drop=True)
    heads = SC2.selection_heads()
    td2 = json.loads((T.TD2 / 'SELECTION_v2.json').read_text(encoding='utf-8'))['chosen_local_gates']
    y, b = f.y.to_numpy(float), f.b.to_numpy(float)
    z = f[SC1.ZCOLS['main']].to_numpy(float)
    avail, news = SC1.available(f), f.has_news.to_numpy(bool)
    out = {'rows': int(len(f)), 'checks': {}}
    for key, c in td2.items():
        kind, seed = c['selector'], c['seed']
        ranks = SC1.local_ranks(f, kind, seed, heads['training_beta_order'])
        q = Q._gv().gates(ranks, avail, news, c['K'], c['fallback'])
        mse = SC1.mse(y, Q.predict(b, z, q, heads['beta']))
        out['checks'][key] = {'td2_mse': c['selection_mse'], 'v2_mse': mse,
                              'abs_diff': abs(mse - c['selection_mse'])}
    out['max_abs_diff'] = max(v['abs_diff'] for v in out['checks'].values())
    out['equal'] = out['max_abs_diff'] <= 1e-12
    return out


def run(name='v2_selection_td2'):
    import td3_runtime as R
    t0 = time.perf_counter()
    res = {'check': name, 'version': 'TD4-provider-selectors-v2', **selection_td2(),
           'elapsed_seconds': time.perf_counter() - t0, **R.peak()}
    tag = R.sha_text(json.dumps(res, sort_keys=True, default=str))[:8]
    R.new_json(T.OUT / 'parity' / f'PARITY_{name}_{tag}.json', res)
    return res


def construction_redacted(n=200):
    """Request construction from the redacted text: for the first n sent 2021-2022 subsample inputs that the redaction
    changed (sample-rank order) and each of the 3 selectors, the body built by the unchanged TD3 selector classes (as
    td4_runner.attempt calls them, sel.body(rec['input_text'])) must equal the documented payload built from the
    redacted text, hold the redacted text verbatim, and differ from the body of the masker-v2 text."""
    E.paths()
    import hashlib
    import td3_runtime as R
    import td3_providers as P
    import td3_spec as S3
    import td4_inputs as I
    import td4v2_inputs as IV
    import td4v2_spec as S
    t0 = time.perf_counter()
    menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
    sels = P.selectors(menu)
    recs = IV.callable_inputs(S.SELECT)
    d = IV.period_dir(S.SELECT)
    v2 = I.load_inputs(None, d / 'INPUTS.jsonl')
    keys = [k for k in sorted(recs, key=lambda k: recs[k]['sample_rank'])
            if recs[k]['input_sha256'] != v2[k]['input_sha256']][:n]
    ok, bad = 0, []
    for k in keys:
        txt, old = recs[k]['input_text'], v2[k]['input_text']
        for name, sel in sels.items():
            body = sel.body(txt)
            if name == 'deepseek_order':
                want = {'model': S3.DEEPSEEK_MODEL,
                        'messages': [{'role': 'system', 'content': S3.SYSTEM_DEEPSEEK},
                                     {'role': 'user', 'content': S3.deepseek_question(menu) + '\n\n' + txt}],
                        'thinking': dict(S3.DEEPSEEK_THINKING), 'temperature': S3.DEEPSEEK_TEMPERATURE,
                        'max_tokens': S3.DEEPSEEK_MAX_TOKENS, 'response_format': {'type': 'json_object'},
                        'stream': False}
                got = json.loads(body)['messages'][1]['content'][-len(txt):]
            else:
                qs = S3.jev_choice_question(menu) if name == 'jev_choice' else S3.jev_noul_questions(menu)
                want = {'state': txt, 'model': S3.JEV_MODEL, 'questions': qs}
                got = json.loads(body)['state']
            exp = json.dumps(want, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            if body == exp and got == txt and body != sel.body(old):
                ok += 1
            elif len(bad) < 10:
                bad.append([hashlib.sha256(k.encode()).hexdigest()[:12], name])
    res = {'check': 'construction_redacted', 'rule_version': S.RULE_VERSION, 'version': S.VERSION,
           'period': S.SELECT, 'inputs': len(keys),
           'bodies': len(keys) * len(sels), 'equal_bodies': ok, 'equal': ok == len(keys) * len(sels) and len(keys) == n,
           'failures_hashed': bad, 'redaction_file': str(IV.redaction_file(d)),
           'redaction_file_sha256': R.sha(IV.redaction_file(d)), 'selectors': list(sels),
           'elapsed_seconds': time.perf_counter() - t0, **R.peak()}
    tag = R.sha_text(json.dumps(res, sort_keys=True, default=str))[:8]
    R.new_json(T.OUT / 'parity' / f'PARITY_construction_redacted_{S.RULE_VERSION.replace(".", "_")}_{tag}.json', res)
    return res
