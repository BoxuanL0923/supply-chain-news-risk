"""Model-independent nested predictive regressions and sensitivity/power."""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from .config import CONTROLS

TERMS=['intercept',*CONTROLS,'OWN','H1','H2_minus_H1','H3_minus_H2']

def design(frame, depth=3):
    x=np.c_[np.ones(len(frame)),frame[list(CONTROLS)].to_numpy(float),frame.OWN.to_numpy(float)]
    columns=[frame.H1.to_numpy(float),(frame.H2-frame.H1).to_numpy(float),(frame.H3-frame.H2).to_numpy(float)]
    return np.c_[x,*columns[:depth]] if depth else x

def rank(x):
    scales=np.linalg.norm(x,axis=0); scales[scales==0]=1
    return np.linalg.matrix_rank(x/scales)

def regress(frame, train=None):
    y=frame.y.to_numpy(float); x=design(frame)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite regression data')
    model=sm.OLS(y,x).fit(); r=int(rank(x)); g=int(frame.date.nunique())
    hc=model.get_robustcov_results(cov_type='HC3',use_t=True)
    cluster=model.get_robustcov_results(cov_type='cluster',groups=frame.date.to_numpy(),use_correction=True,df_correction=True,use_t=True)
    # statsmodels CR1 uses the number of columns, including redundant controls.
    # Use effective design rank in its residual degrees-of-freedom correction.
    cluster_rank_scale=math.sqrt((len(frame)-x.shape[1])/(len(frame)-r))
    coefficients={}
    for j,name in enumerate(TERMS):
        identified=rank(x)>rank(np.delete(x,j,axis=1))
        row={'identified':bool(identified),'estimate':float(model.params[j]) if identified else None}
        for label,result,df in [('HC3',hc,len(frame)-r),('date_cluster',cluster,g-1)]:
            se=float(result.bse[j]) if identified else None
            if label=='date_cluster' and se is not None:se*=cluster_rank_scale
            if se is not None and (not math.isfinite(se) or se<0):se=None
            half=float(stats.t.ppf(.95,df)*se) if se is not None else None
            row[label]={'se':se,'df':df,'ci90':[row['estimate']-half,row['estimate']+half] if half is not None else None}
        coefficients[name]=row
    nested=[]
    for depth in range(4):
        xx=design(frame,depth); fit=sm.OLS(y,xx).fit()
        entry={'depth':depth,'rank':int(rank(xx)),'r2':float(fit.rsquared),'n':len(frame)}
        if train is not None:
            tx=design(train,depth); tfit=sm.OLS(train.y.to_numpy(float),tx).fit()
            prediction=xx@tfit.params
            entry.update(train_rank=int(rank(tx)),valid_mse=float(np.mean((y-prediction)**2)),
                         train_coefficients=dict(zip(TERMS[:tx.shape[1]],tfit.params.tolist())))
            if depth:
                previous=design(frame,depth-1)@sm.OLS(train.y.to_numpy(float),design(train,depth-1)).fit().params
                entry['valid_mse_gain_vs_previous']=float(np.mean((y-previous)**2)-entry['valid_mse'])
        nested.append(entry)
    g3=coefficients['H3_minus_H2']
    bound={kind:(max(abs(v) for v in g3[kind]['ci90']) if g3[kind]['ci90'] is not None else None) for kind in ('HC3','date_cluster')}
    # This is sensitivity, not an outcome-selected equivalence test. Strict TOST
    # equivalence would require a scientifically chosen margin larger than bound.
    z=float(stats.norm.ppf(.95)+stats.norm.ppf(.8))
    power=[]
    residual=x[:,-1]-x[:,:-1]@np.linalg.lstsq(x[:,:-1],x[:,-1],rcond=None)[0]
    sd_inc=float(np.sqrt(np.mean(residual**2)))
    sd_y=float(np.std(y,ddof=1))
    for effect in (.1,.25,.5,1.0):
        row={'gamma3_effect':effect,'rms_log_risk_effect':effect*sd_inc}
        for kind in ('HC3','date_cluster'):
            se=g3[kind]['se']
            needed=math.ceil(len(frame)*(z*se/effect)**2) if g3['identified'] and se is not None else None
            if kind=='date_cluster' and needed is not None: needed=int(50*math.ceil(needed/50))
            row[kind+'_cases_for_80pct_power']=needed
        power.append(row)
    return {'n':len(frame),'dates':g,'rank':r,'columns':len(TERMS),'r2':float(model.rsquared),
            'cluster_CR1_correction':'G/(G-1) * (n-1)/(n-effective_design_rank)',
            'coefficients':coefficients,'nested':nested,'gamma3_smallest_symmetric_equivalence_bound':bound,
            'power':{'alpha':.10,'two_sided':True,'power':.80,'partial_increment_sd':sd_inc,'y_sd':sd_y,'grid':power,
                     'method':'Normal approximation n_new=n*((z_.95+z_.80)*SE/effect)^2. Cluster version requires new independent dates with 50 cases/date, not more firms on the same dates.',
                     'unidentified':not g3['identified']},
            'ci_warning':'Only six pooled date clusters (four train, two valid); clustered t intervals and extrapolated power are unstable and exploratory.'}

def analyze(frame):
    train=frame[frame['split']=='train'];valid=frame[frame['split']=='valid']
    if len(train)!=200 or len(valid)!=100: raise ValueError('Matched 200/100 required')
    if max(train.date)>=min(valid.date): raise ValueError('Time split overlap')
    arms=('OWN','H1','H2','H3','H3_GRAPH')
    validation=regress(valid,train)
    return {'raw_residual_mse':{s:{a:float(np.mean((f.y-f.anchor-f[a])**2)) for a in arms}
                              for s,f in [('train',train),('valid',valid)]},
            'zero_score_mse':{s:float(np.mean((f.y-f.anchor)**2)) for s,f in [('train',train),('valid',valid)]},
            'train_fitted_valid':{'train_n':len(train),'valid_n':len(valid),'nested':validation['nested'],
                'estimation':'Nested train_coefficients fitted on train dates only; valid_mse evaluated on validation dates without refitting. Pooled coefficients are reported separately.'},
            'pooled_train_valid':regress(frame),
            'train':regress(train)}


def cached_hop_report(root):
    """Post-freeze reporting from Qwen cache only; no runner/provider imports.

    Read the development CSV already produced by the completed run. Never open
    live labels or rewrite its protocol/freeze/reports. Hash every artifact used.
    """
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd
    root = Path(root)
    provenance = {}

    def blob(name, expected=None):
        path = root / name
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError('Cache path escapes run directory')
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError('Cache hash mismatch: ' + name)
        provenance[name] = actual
        return raw

    def read(name, expected=None):
        return json.loads(blob(name, expected))

    frozen = read('prediction_freeze.json')
    seal = read('prediction_freeze.sha256.json')
    if provenance['prediction_freeze.json'] not in seal.values():
        raise ValueError('Prediction freeze seal mismatch')
    index = read('input_index.json')['records']
    import io
    frame = pd.read_csv(io.BytesIO(blob('matched_development_qwen3.8-27b.csv')), float_precision='round_trip')
    if (frame.case_id.duplicated().any() or set(frame['split']) != {'train', 'valid'} or
            any(str(d) >= '2025-11-14' for d in frame.date)):
        raise ValueError('Only unique sealed development cases may be reported')
    train = frame[frame['split'] == 'train']
    if max(train.date) >= min(frame.loc[frame['split'] == 'valid', 'date']):
        raise ValueError('Development dates overlap')
    inputs = {(r['case_id'], r['mode'], r['arm']): r for r in index}
    predictions = {(r['case_id'], r['mode'], r['arm']): r for r in frozen['predictions']}
    arms = ('OWN', 'H1', 'H2', 'H3', 'H3_GRAPH')
    modes = ('all_paths', 'continuity_only')
    expected = {(c, m, a) for c in frame.case_id for m in modes for a in arms}
    if len(inputs) != len(index) or len(predictions) != len(frozen['predictions']) or set(inputs) != expected or set(predictions) != expected:
        raise ValueError('Incomplete or duplicate cached design')
    cells, responses = {}, {}
    for key in sorted(expected):
        entry, freeze = inputs[key], predictions[key]
        inp = read('inputs/' + entry['input'], entry['input_sha256'])
        pred = read('predictions/' + freeze['file'], freeze['sha256'])
        packet = json.loads(inp['prompt'].split('\nINPUT\n', 1)[1])
        if ((pred['case_id'], pred['mode'], pred['arm']) != key or pred['input'] != entry['input'] or
                pred['input_sha256'] != entry['input_sha256'] or packet['case_id'] != key[0] or
                [e['source_id'] for e in packet['evidence']] != inp['evidence_ids']):
            raise ValueError('Cached cell binding mismatch')
        if len(set(inp['evidence_ids'])) != len(inp['evidence_ids']):
            raise ValueError('Duplicate scoring article')
        from .graph_skills import utc
        for event in packet['evidence']:
            if utc(event['available_at']) >= utc(packet['asof']):
                raise ValueError('Post-cutoff scoring article')
        request = pred['request_sha256']
        if request not in responses:
            response = read('responses/' + request + '.json', pred['response_file_sha256'])
            raw_request = json.dumps(response['request'], sort_keys=True, allow_nan=False).encode()
            if hashlib.sha256(raw_request).hexdigest() != request or response['request_sha256'] != request:
                raise ValueError('Response request hash mismatch')
            responses[request] = response
        response = responses[request]
        if (response['request']['messages'][0]['content'] != inp['prompt'] or
                hashlib.sha256(blob('responses/' + request + '.json')).hexdigest() != pred['response_file_sha256']):
            raise ValueError('Response prompt binding mismatch')
        row = frame[frame.case_id == key[0]].iloc[0]
        column = key[2] if key[1] == 'all_paths' else 'continuity_' + key[2]
        if float(row[column]) != float(pred['score']):
            raise ValueError('CSV score differs from frozen prediction')
        cells[key] = dict(request=request, evidence=packet['evidence'])

    def costs(keys):
        requests = {cells[k]['request'] for k in keys}
        selected = [responses[r] for r in requests]
        usage = [r['envelope']['usage'] for r in selected]
        return dict(model_calls=len(requests), prompt_tokens=sum(u['prompt_tokens'] for u in usage),
                    completion_tokens=sum(u['completion_tokens'] for u in usage),
                    tokens=sum(u['prompt_tokens'] + u['completion_tokens'] for u in usage),
                    seconds=sum(r['seconds'] for r in selected),
                    articles_retrieved=sum(len(cells[k]['evidence']) for k in keys)), requests

    rows = []
    for mode in modes:
        data = frame.copy()
        if mode == 'continuity_only':
            for arm in arms:
                data[arm] = data['continuity_' + arm]
        training = data[data['split'] == 'train']
        coefficients = {d: np.linalg.lstsq(design(training, d), training.y.to_numpy(float), rcond=None)[0] for d in range(4)}
        for split in ('train', 'valid'):
            f = data[data['split'] == split]
            for depth, arm, previous in ((1, 'H1', 'OWN'), (2, 'H2', 'H1'), (3, 'H3', 'H2')):
                keys = [(c, mode, arm) for c in f.case_id]
                prior = [(c, mode, previous) for c in f.case_id]
                cost, requests = costs(keys)
                old_cost, old_requests = costs(prior)
                new_requests = requests - old_requests
                exact = [sum(e['hop'] == depth for e in cells[k]['evidence']) for k in keys]
                raw = float(np.mean((f.y - f.anchor - f[arm])**2))
                old_raw = float(np.mean((f.y - f.anchor - f[previous])**2))
                nested = float(np.mean((f.y - design(f, depth) @ coefficients[depth])**2))
                old_nested = float(np.mean((f.y - design(f, depth-1) @ coefficients[depth-1])**2))
                rows.append(dict(mode=mode, split=split, contrast='H1' if depth == 1 else arm + '-' + previous,
                    n=len(f), exact_hop=depth, cases_with_evidence=sum(n > 0 for n in exact),
                    share_any_evidence=sum(n > 0 for n in exact)/len(f),
                    missing_share=sum(n == 0 for n in exact)/len(f), exact_hop_articles=sum(exact),
                    candidate_share_any=float(np.mean(f['candidate_exact_h' + str(depth)] > 0)),
                    raw_residual_mse=raw, raw_mse_gain_vs_previous=old_raw-raw,
                    nested_train_fitted_mse=nested, nested_mse_gain_vs_previous=old_nested-nested,
                    cost=cost, cost_change_vs_previous={k: cost[k]-old_cost[k] for k in cost},
                    additional_unique_requests=len(new_requests),
                    additional_request_tokens=sum(responses[r]['envelope']['usage']['prompt_tokens'] + responses[r]['envelope']['usage']['completion_tokens'] for r in new_requests),
                    additional_request_seconds=sum(responses[r]['seconds'] for r in new_requests),
                    retrieval_seconds=None, end_to_end_seconds=None))
    unique_cost, _ = costs(list(expected))
    return dict(schema_version=1, model='qwen3.8-27b', new_model_calls=0, evaluation_labels_accessed=False,
                rows=rows, unique_run_cost=unique_cost, artifact_sha256=provenance,
                definitions=dict(missingness='No retained scoring article at this exact hop; not a zero forecast increment or absence of economic exposure.',
                    articles='Retained source IDs per case in cumulative scoring packets, including OWN; repeated across cases count separately. Host-only supporting reads are excluded.',
                    costs='Unique cached scoring requests per arm/split/mode. Additional requests are set difference from previous arm; signed cost changes are separate. Shared requests are counted once in unique_run_cost.',
                    seconds='Recorded model response service seconds. Retrieval and end-to-end timing were not cached and remain null; no timing is imputed.',
                    loss='Raw residual MSE and nested OLS MSE with coefficients fitted on training cases only. Positive gain means lower loss. H1 is compared with OWN.'))


def cached_report_markdown(report):
    lines = ['# Cached Qwen exact-hop missingness and cost', '',
             'Derived offline from frozen development artifacts; zero new model calls. Evaluation labels remain sealed.', '',
             '| Mode | Split | Contrast | n | Any exact-hop evidence | Missing | Raw MSE | Raw gain | Nested MSE | Nested gain | Calls (additional unique) | Cumulative tokens | Cumulative articles | Model seconds |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        c = row['cost']
        lines.append(f"| {row['mode']} | {row['split']} | {row['contrast']} | {row['n']} | {row['share_any_evidence']:.1%} | {row['missing_share']:.1%} | {row['raw_residual_mse']:.6f} | {row['raw_mse_gain_vs_previous']:.6f} | {row['nested_train_fitted_mse']:.6f} | {row['nested_mse_gain_vs_previous']:.6f} | {c['model_calls']} ({row['additional_unique_requests']}) | {c['tokens']:,} | {c['articles_retrieved']} | {c['seconds']:.2f} |")
    lines += ['', '| Mode | Split | Contrast | Additional-request tokens | Additional-request seconds | Article change | Token change | Model-second change |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        delta = row['cost_change_vs_previous']
        lines.append(f"| {row['mode']} | {row['split']} | {row['contrast']} | {row['additional_request_tokens']:,} | {row['additional_request_seconds']:.2f} | {delta['articles_retrieved']:+d} | {delta['tokens']:+,d} | {delta['seconds']:+.2f} |")
    lines += ['', *report['definitions'].values(), '',
              'Cumulative costs describe standalone arms. They are not matched compute. The JSON also reports signed cost changes and additional-request cost.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Offline post-freeze Qwen coverage/cost report; no provider code imported')
    parser.add_argument('--cached-qwen', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    result = cached_hop_report(args.cached_qwen)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with args.markdown.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(cached_report_markdown(result))
    print(json.dumps({k: v for k, v in result.items() if k not in {'rows', 'artifact_sha256', 'definitions'}}, indent=2))
