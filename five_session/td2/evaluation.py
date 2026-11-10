"""Claude-only once-only evaluation. No evaluation payload is read on import."""
import hashlib
import json
from pathlib import Path
from runtime_v3 import CODE,OUT,TD1,guard,sha,new_json,peak
from spec import VERSION,K_GRID,SEEDS

CLEARANCE=TD1/'clearance/TD2_CLEARANCE_v1.json'
BINDINGS=['code_sha256','artifact_sha256','td1_freeze_sha256','td1_evaluation_inputs_sha256','model_sha256']

def validate_clearance(clearance,freeze,freeze_hash):
    if clearance.get('schema')!='TD2_CLEARANCE_v1' or clearance.get('status')!='CLEARED':
        raise PermissionError('Clearance schema/status')
    if clearance.get('freeze_sha256')!=freeze_hash: raise PermissionError('Freeze hash mismatch')
    for key in BINDINGS:
        if key not in freeze or clearance.get(key)!=freeze[key]: raise PermissionError('Clearance binding '+key)
    for key,value in [('version',VERSION),('evaluation_years',[2023,2024]),('maximum_label_end','2024-12-31')]:
        if clearance.get(key)!=value or freeze.get(key)!=value: raise PermissionError('Scope mismatch '+key)

def admit():
    guard()
    if not CLEARANCE.is_file(): raise PermissionError('STOP: TD2_CLEARANCE_v1.json absent; no evaluation label read or computed')
    if (OUT/'EVALUATION_STARTED_v1.json').exists() or (OUT/'EVALUATION_RESULT_v1.json').exists():
        raise PermissionError('Evaluation once-only latch exists; no replay')
    # Only clearance and freeze metadata are opened until they agree.
    clearance=json.loads(CLEARANCE.read_text(encoding='utf-8'))
    freeze=json.loads((OUT/'FREEZE_v1.json').read_text(encoding='utf-8'))
    validate_clearance(clearance,freeze,sha(OUT/'FREEZE_v1.json'))
    if {str(p) for p in CODE.glob('*.py')}!=set(freeze['code_sha256']): raise PermissionError('Code inventory changed')
    for field,root in [('code_sha256',CODE),('artifact_sha256',OUT),('td1_evaluation_inputs_sha256',TD1)]:
        for p,h in freeze[field].items():
            path=Path(p).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path)!=h: raise PermissionError('Hash mismatch '+p)
    if sha(TD1/'MODEL_FREEZE_v3.json')!=freeze['td1_freeze_sha256']: raise PermissionError('TD1 freeze changed')
    if sha(OUT/'MODEL_v1.json')!=freeze['model_sha256']: raise PermissionError('Head changed')
    new_json(OUT/'EVALUATION_STARTED_v1.json',{'freeze_sha256':sha(OUT/'FREEZE_v1.json'),
        'clearance_sha256':sha(CLEARANCE),'version':VERSION,'exploratory':True})
    import decoder
    decoder._evaluation_token=object()
    return decoder._evaluation_token

def label(dates,returns,pos,token):
    import decoder
    if token is None or token is not decoder._evaluation_token: raise PermissionError('Label admission required')
    import numpy as np
    import pandas as pd
    if pos<0 or pos+5>=len(dates): return None,None
    end=pd.Timestamp(dates[pos+5])
    if end>pd.Timestamp('2024-12-31'): return None,None
    r=np.asarray(returns[pos+1:pos+6],float)
    if len(r)!=5 or not np.isfinite(r).all() or np.mean(r*r)<=0: return None,None
    return float(np.log(np.mean(r*r))),end

def run():
    token=admit()  # First action: no return/feature/model decoder before this succeeds.
    import time
    import numpy as np
    import pandas as pd
    import decoder as io
    from pipeline import features,checked_join,predictions
    from statistics import paired,clustered_mean
    start=time.perf_counter(); model=io.json_read(OUT/'MODEL_v1.json'); cases=[]; coverage=[]
    for year in (2023,2024):
        returns=io.admitted_returns(year,token); returns=returns[returns.share_type.isin([0,2])]
        returns['date']=pd.to_datetime(returns.date)
        if returns.duplicated(['permno','date']).any(): raise ValueError('Duplicate return session')
        dates=pd.DatetimeIndex(sorted(returns.date.unique()))
        wide=returns.pivot(index='date',columns='permno',values='ret').reindex(dates)
        mapper={hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]:p for p in wide.columns}
        f=features(year); baseline=io.baseline(year)[['firm_id','cutoff','b']]
        primary=f[f.variant=='primary']; rows=[]
        for r in primary.itertuples():
            p=mapper.get(r.firm_id); pos=dates.get_indexer([r.origin_session])[0]
            if p is None: continue
            y,end=label(dates,wide[p].to_numpy(float),pos,token)
            if y is not None: rows.append({'firm_id':r.firm_id,'cutoff':r.cutoff,'y':y,'label_end':end})
        labels=pd.DataFrame(rows)
        # Exactly the same observed-label keys for both identity variants and every hop/selector.
        for variant in ['primary','fsb5_sensitivity']:
            g=checked_join(f[f.variant==variant],baseline)
            g=checked_join(g,labels); g=g[g.y.notna()].copy(); cases.append(g)
            coverage.append({'year':year,'variant':variant,'feature_rows':len(primary),'paired_rows':len(g),
                'news_rows':int(g.has_news.sum()),'mapped_focal':int(g.mapped_focal.sum()),
                'missing_characteristics':{n:int(g['missing_'+n].sum()) for n in model['names']},
                'hop_news_rows':{str(h):int((g[f'H{h}_news_partners']>0).sum()) for h in (1,2,3)}})
    f=pd.concat(cases,ignore_index=True); result=[]; paired_rows=[]; increments=[]
    for variant,g in f.groupby('variant',sort=True):
        for kind,seed in [('training_beta',0)]+[('random',s) for s in SEEDS]:
            key=f'{kind}:{seed}'; chosen=model['selected_local_gates'][key]
            for k in K_GRID:
                byhop={}
                for hop in ['main',1,2,3]:
                    pred=predictions(g,model,kind,k,seed,hop,chosen['fallback']); byhop[hop]=pred
                    metrics=paired(g.y,g.b,pred,g.firm_id,g.origin_session)
                    result.append(dict(variant=variant,selector=kind,seed=seed,K=k,hop=hop,
                        selected_on_2021_2022=k==chosen['K'],**metrics))
                    paired_rows.append(pd.DataFrame({'firm_id':g.firm_id,'origin_session':g.origin_session,
                        'variant':variant,'selector':kind,'seed':seed,'K':str(k),'hop':str(hop),
                        'baseline_squared_error':(g.y-g.b)**2,'arm_squared_error':(g.y-pred)**2}))
                for a,b in [(2,1),(3,2)]:
                    gain=(g.y.to_numpy()-byhop[b])**2-(g.y.to_numpy()-byhop[a])**2
                    increments.append(dict(variant=variant,selector=kind,seed=seed,K=k,
                        contrast=f'H{a}-H{b}',positive='lower loss for higher exact hop',
                        **clustered_mean(gain,g.firm_id,g.origin_session)))
    io.write_frame(OUT/'evaluation/paired_squared_errors.parquet',pd.concat(paired_rows,ignore_index=True))
    new_json(OUT/'EVALUATION_RESULT_v1.json',{'version':VERSION,'exploratory':True,'arms':result,
        'hop_incremental_gains':increments,'coverage':coverage,'provider_selectors':'empty, not evaluated',
        'elapsed_seconds':time.perf_counter()-start,**peak(),'free_bytes':guard()})
    print('EVALUATION_RESULT',str(OUT/'EVALUATION_RESULT_v1.json'),flush=True)
