"""Offline development builder, common head, local selector sweep, and freeze."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import decoder as io
from runtime_v3 import CODE,OUT,TD1,guard,new_json,sha,peak
from spec import VERSION,NAMES,MENU,RULES,LEXICONS,SLOT_SCHEMAS,K_GRID,RIDGE_GRID,SEEDS
from engine import graph_at,news_pool,exposure,ridge
from selectors import ordering,gate,validate_slot

def read_menu():
    plan=io.json_read(OUT/'EXPOSURE_DICTIONARY_v1.json')
    return plan['eligible_names']

def require_synthetic():
    for p in sorted(OUT.glob('SYNTHETIC_TESTS_*.json')):
        r=io.json_read(p); b=r.get('boundary_tests',{})
        if r.get('failures')==0 and r.get('errors')==0 and b.get('failures')==0 and b.get('errors')==0:
            return str(p)
    raise PermissionError('Synthetic fixture and boundary suites must pass before development reads')

def prepare():
    start=time.perf_counter(); test=require_synthetic(); guard()
    if (OUT/'EXPOSURE_DICTIONARY_v1.json').exists(): raise FileExistsError('Plan already frozen')
    counts={n:0 for n in NAMES}; train_values={n:[] for n in NAMES}
    for year in range(2015,2023):
        f=io.manifest(year); f=f[f.eligible_features]
        for n in NAMES:
            finite=np.isfinite(f[n].to_numpy(float)); counts[n]+=int(finite.sum())
            if year<=2020: train_values[n].extend(f.loc[finite,n].to_numpy(float).tolist())
    menu=[]
    for row in MENU:
        n=row['name']; eligible=counts[n]>0 and len(train_values[n])>1 and np.std(train_values[n])>0
        reason=('Observed and variable on training; menu and coverage frozen using 2015-2022 only. '
                'Exploratory availability-lag reconstruction; no revision-free PIT certification.') if eligible else 'No variable pre-2021 values or no development coverage'
        menu.append(dict(row,eligible=eligible,reason=reason))
    names=[m['name'] for m in menu if m['eligible']]
    if not names: raise ValueError('No eligible menu items')
    with (OUT/'risk_menu.csv').open('x',encoding='utf-8',newline='') as f:
        import csv
        writer=csv.DictWriter(f,fieldnames=['name','firm_column','economic_meaning','risk_channel','asof_source','eligible','reason'])
        writer.writeheader(); writer.writerows(menu)
    for kind,slot in SLOT_SCHEMAS.items():
        validate_slot(slot,kind,names)
        new_json(OUT/f'selector_slots/{kind}_v1.json',slot,timed=False)
    plan={'schema':'TD2_exposure_dictionary_v1','version':VERSION,'eligible_names':names,
        'menu_frozen_period':'2015-2022','development_nonmissing_counts':counts,
        'rules':RULES,'lexicons':LEXICONS,'synthetic_receipt':test,
        'source_paths':{'October_date_sealed_projection':str(io.GRAPH),'FactSet_links':str(io.FSMAP),
            'FSB5_candidates':str(io.FS5),'Reuters_archive':str(io.ARTICLE),
            'Reuters_tag_identity_keys':str(io.PROCESSED/'regression_panel.parquet'),
            'original_news_adapter':'C:/OptResearch/code/sc_risk/cc1_articles.py'},
        'source_limitations':['FactSet stored start date is a retrospective validity proxy, not verified historical publication availability',
            'Reuters archive provides a day field; no intraday timestamp or original-version guarantee',
            'RIC mapping is one-to-one over development keys, using first observed month conservatively',
            'Intermediate product continuity is unobserved; paths are graph witnesses only'],
        'elapsed_seconds':time.perf_counter()-start,**peak()}
    new_json(OUT/'EXPOSURE_DICTIONARY_v1.json',plan)
    graph,q=io.graph_rows()
    io.write_frame(OUT/'sources/graph_october_sealed.parquet',graph.rename(columns={'start':'date'}))
    io.write_frame(OUT/'sources/factset_links.parquet',io.mapping().rename(columns={'start':'date'}))
    new_json(OUT/'sources/quarantine_ids_v1.json',{'relationship_ids':sorted(q)},timed=False)
    new_json(OUT/'sources/identity_sensitivity_v1.json',{'links':[list(x) for x in sorted(io.sensitivity_links())]},timed=False)
    # Development identities are frozen before any evaluation-era tag keys are used.
    mapping=io.ric_mapping(2022)
    new_json(OUT/'sources/reuters_identity_development_v1.json',{'mapping':mapping},timed=False)
    seen=set(); news_counts={}
    for year in range(2014,2023):
        guard(); news_counts[str(year)]=io.build_news(year,mapping,seen)
        print('NEWS',year,news_counts[str(year)],flush=True)
    sources={str(p):sha(p) for p in sorted((OUT/'sources').glob('*')) if p.is_file()}
    sources.update({str(p):sha(p) for p in sorted((OUT/'news_scores').glob('*'))})
    new_json(OUT/'PREPARED_v1.json',{'version':VERSION,'news_counts':news_counts,'source_sha256':sources,
        'source_graph_audit_sha256':sha(io.GRAPH),'FSB5_sha256':sha(io.FS5),
        'td1_freeze_sha256':sha(TD1/'MODEL_FREEZE_v3.json'),'elapsed_seconds':time.perf_counter()-start,**peak()})
    print('PREPARED',str(OUT/'risk_menu.csv'),flush=True)

def build_year(year):
    start=time.perf_counter(); require_synthetic(); names=read_menu()
    if not 2015<=year<=2024: raise ValueError('Build year 2015-2024 only')
    path=OUT/f'features/year_{year}.parquet'
    if path.exists(): raise FileExistsError(path)
    if not (OUT/'PREPARED_v1.json').exists(): raise PermissionError('Prepare incomplete')
    prepared=io.json_read(OUT/'PREPARED_v1.json')
    for p,h in prepared['source_sha256'].items():
        if sha(p)!=h: raise ValueError('Prepared source changed')
    graph=io.read_frame(OUT/'sources/graph_october_sealed.parquet',
        ['id','end','status','typ','src','tgt','rownum'],'date','1900-01-01','2024-12-31').rename(columns={'date':'start'})
    links=io.read_frame(OUT/'sources/factset_links.parquet',['entity','permno','end'],'date','1900-01-01','2024-12-31').rename(columns={'date':'start'})
    suspects={tuple(x) for x in io.json_read(OUT/'sources/identity_sensitivity_v1.json')['links']}
    # Evaluation-era score partitions are features only. No returns or labels are read here.
    if year>=2023:
        devmap=io.json_read(OUT/'sources/reuters_identity_development_v1.json')['mapping']
        seen=set()
        for y in range(2014,year+1):
            p=OUT/f'news_scores/year_{y}.parquet'
            if not p.exists(): io.build_news(y,devmap,seen)
            f=io.read_frame(p,['alt_id'],'date',f'{y}-01-01',f'{y}-12-31'); seen.update(f.alt_id)
    f=io.manifest(year); f=f[f.eligible_features].copy()
    if f.duplicated(['firm_id','cutoff']).any(): raise ValueError('Duplicate manifest firm-week')
    news=io.news_for_year(year); records=[]; cov=[]
    evidencepath=OUT/f'evidence/year_{year}.jsonl'; evidencepath.parent.mkdir(parents=True,exist_ok=True)
    with evidencepath.open('x',encoding='utf-8',newline='\n') as evidencefile:
        for origin,g in f.groupby('origin_session',sort=True):
            guard(); cutoffs=g.cutoff.unique()
            if len(cutoffs)!=1: raise ValueError('Cutoff inconsistent within origin')
            cutoff=pd.Timestamp(cutoffs[0]); stories=io.week_stories(news,origin)
            pool=news_pool(stories,cutoff,origin)
            # Opaque TD1 ID derived from mapped PERMNO; no identity from returns needed.
            idmap={hashlib.sha256(f'TD1-firm:{p}'.encode()).hexdigest()[:24]:int(p) for p in links.permno.unique()}
            for variant,badlinks in [('primary',None),('fsb5_sensitivity',suspects)]:
                adj=graph_at(io.mapped_edges(graph,links,origin,badlinks),cutoff,origin)
                for r in g.itertuples():
                    root=idmap.get(r.firm_id,-1)
                    chars={n:float(getattr(r,n)) for n in names}
                    available={n:getattr(r,n+'__available_at') for n in names}
                    values,witness=exposure(root,cutoff,origin,chars,available,adj,pool,names)
                    row={'firm_id':r.firm_id,'cutoff':cutoff,'origin_session':pd.Timestamp(origin),'variant':variant,
                         'mapped_focal':root!=-1}
                    for n in names:
                        row['C_'+n]=chars[n]
                        row['available_'+n]=available[n]
                    for v in values:
                        h,n=v['hop'],v['name']; row[f'Z{h}_{n}']=v['Z']; row[f'E{h}_{n}']=v['E']
                        row[f'missing_{n}']=v['characteristic_missing']
                        for c in ['graph_partners','news_partners','selected_partners','unique_articles']:
                            row[f'H{h}_{c}']=v[c]
                    for n in names: row['Z_'+n]=sum(row[f'Z{h}_{n}'] for h in (1,2,3))/3
                    row['has_news']=any(row[f'H{h}_selected_partners']>0 for h in (1,2,3))
                    records.append(row)
                    if witness:
                        evidencefile.write(json.dumps({'firm_id':r.firm_id,'cutoff':str(cutoff),
                            'variant':variant,'witnesses':witness},allow_nan=False)+'\n')
            print('BUILD',year,str(pd.Timestamp(origin).date()),len(records),flush=True)
    result=pd.DataFrame(records)
    io.write_frame(path,result)
    for variant,g in result.groupby('variant'):
        cov.append({'variant':variant,'rows':len(g),'mapped_focal':int(g.mapped_focal.sum()),
            'firm_weeks_with_news':int(g.has_news.sum()),
            'missing_characteristics':{n:int(g['missing_'+n].sum()) for n in names},
            'hop_coverage':{str(h):{'graph':int((g[f'H{h}_graph_partners']>0).sum()),
                'news':int((g[f'H{h}_news_partners']>0).sum()),
                'selected_partner_packets':int(g[f'H{h}_selected_partners'].sum()),
                'intermediate_product_evidence_observed':0} for h in (1,2,3)}})
    new_json(OUT/f'coverage/year_{year}.json',{'coverage':cov,'feature_sha256':sha(path),
        'evidence_sha256':sha(evidencepath),'elapsed_seconds':time.perf_counter()-start,**peak()})

def features(year):
    names=read_menu()
    cols=['firm_id','cutoff','variant','mapped_focal','has_news']
    cols += ['C_'+n for n in names]+['available_'+n for n in names]+['missing_'+n for n in names]
    cols += ['Z_'+n for n in names]+[f'Z{h}_{n}' for h in (1,2,3) for n in names]
    cols += [f'H{h}_{c}' for h in (1,2,3) for c in ['graph_partners','news_partners','selected_partners','unique_articles']]
    return io.read_frame(OUT/f'features/year_{year}.parquet',cols,'origin_session',f'{year}-01-01',f'{year}-12-31')

def checked_join(features_frame,labels):
    keys=['firm_id','cutoff']; n=len(features_frame)
    result=features_frame.merge(labels,on=keys,how='left',validate='one_to_one',suffixes=('','_manifest'))
    if len(result)!=n: raise ValueError('Row alignment changed')
    return result

def predictions(frame,model,kind,k,seed,hop='main',fallback='none'):
    names=model['names']; q=[]
    for r in frame.itertuples():
        available=[n for n in names if not getattr(r,'missing_'+n)]
        order=ordering(names,kind,f'{r.firm_id}|{r.cutoff}',seed,model['training_beta'])
        gatevalues=gate(names,order,k,available)
        if not r.has_news: gatevalues=gate(names,order,'all' if fallback=='all' else 0,available)
        q.append(gatevalues)
    prefix='Z_' if hop=='main' else f'Z{hop}_'
    z=frame[[prefix+n for n in names]].to_numpy(float)
    return frame.b.to_numpy(float)+np.sum(z*np.asarray(q)*np.asarray(model['beta']),axis=1)

def fit():
    start=time.perf_counter(); require_synthetic(); names=read_menu()
    if (OUT/'MODEL_v1.json').exists(): raise FileExistsError('Head already fit')
    model=io.td1_model(); chunks=[]
    for year in range(2015,2023):
        f=features(year); f=f[f.variant=='primary']
        labels=io.manifest(year,labels=True)[['firm_id','cutoff','y','label_end','fit_included','selection_included','refit_included']]
        f=checked_join(f,labels)
        f=checked_join(f,io.development_baseline(year,model))
        f=checked_join(f,io.baseline(year)[['firm_id','cutoff','b']])
        chunks.append(f)
    f=pd.concat(chunks,ignore_index=True)
    tr=f.fit_included.to_numpy(bool); va=f.selection_included.to_numpy(bool); ref=f.refit_included.to_numpy(bool)
    if not tr.any() or not va.any() or not np.isfinite(f.loc[ref,'y']).all(): raise ValueError('Invalid development rows')
    if (pd.to_datetime(f.loc[tr,'label_end'])>pd.Timestamp('2020-12-31')).any(): raise ValueError('Train boundary')
    if (pd.to_datetime(f.loc[ref,'label_end'])>pd.Timestamp('2022-12-31')).any(): raise ValueError('Refit boundary')
    z=f[['Z_'+n for n in names]].to_numpy(float)
    scale=np.sqrt(np.mean(z[tr]**2,axis=0)); scale[scale==0]=1.
    x=z/scale; residual=f.y.to_numpy(float)-f.b_development.to_numpy(float)
    candidates=[]; best=None
    for penalty in RIDGE_GRID:
        coef=ridge(x[tr],residual[tr],penalty)
        pred=f.loc[va,'b_development'].to_numpy(float)+x[va]@coef
        loss=float(np.mean((f.loc[va,'y'].to_numpy(float)-pred)**2))
        candidates.append({'penalty':penalty,'selection_mse':loss})
        if best is None or loss<best[0]: best=(loss,penalty,coef.copy())
    training_beta=dict(zip(names,(best[2]/scale).tolist()))
    # Select local gates against the training-fit head and training-fit TD1 baseline.
    temp={'names':names,'beta':(best[2]/scale).tolist(),'training_beta':training_beta}
    validation=f.loc[va].copy(); validation['b']=validation.b_development
    grid=[]; selected={}
    for kind,seed in [('training_beta',0)]+[('random',s) for s in SEEDS]:
        for k in K_GRID:
            for fallback in ['none','all']:
                pred=predictions(validation,temp,kind,k,seed,fallback=fallback)
                mse=float(np.mean((validation.y.to_numpy(float)-pred)**2))
                record={'selector':kind,'seed':seed,'K':k,'fallback':fallback,'selection_mse':mse}
                grid.append(record); key=f'{kind}:{seed}'
                if key not in selected or mse<selected[key]['selection_mse']: selected[key]=record
    final=ridge(x[ref],f.loc[ref,'y'].to_numpy(float)-f.loc[ref,'b'].to_numpy(float),best[1])/scale
    bundle={'schema':'TD2_model_v1','version':VERSION,'names':names,'beta':final.tolist(),
        'training_beta':training_beta,'scale_training_rms':scale.tolist(),'penalty':best[1],
        'q_during_fitting':1,'intercept':0,'fit_rows':int(tr.sum()),'selection_rows':int(va.sum()),
        'refit_rows':int(ref.sum()),'selected_local_gates':selected,
        'provider_gates':{k:{'status':'EMPTY_PENDING_SELECTION','K':None,'threshold':None,'fallback':None} for k in SLOT_SCHEMAS},
        'evaluation_run':False,'exploratory':True}
    new_json(OUT/'SELECTION_v1.json',{'ridge_grid':candidates,'selector_grid':grid,'chosen':selected,
        'selection_period':'2021-2022 only','threshold_grid_for_future_noul':[.25,.5,.75],
        'provider_selection_status':'No provider records; no provider threshold or K selected',
        'elapsed_seconds':time.perf_counter()-start,**peak()})
    new_json(OUT/'MODEL_v1.json',bundle)
    print('MODEL',str(OUT/'MODEL_v1.json'),'BETA',final.tolist(),flush=True)

def cost(input_rate=None,output_rate=None):
    """Measured masked prompt bytes; tokenizer and dollar amounts remain explicit estimates."""
    names=read_menu(); counts={}; contexts=[]
    for year in range(2015,2025):
        if not (OUT/f'features/year_{year}.parquet').exists(): continue
        f=features(year); f=f[(f.variant=='primary')&f.has_news]
        counts[str(year)]=len(f)
        for r in f.head(8).itertuples():
            # Values and identities replaced; no article text is included.
            contexts.append({'firm':'MASKED','cutoff':'YYYY-MM-DD','characteristics':dict.fromkeys(names,'MASKED_NUMERIC'),
                'exact_hop_coverage':[getattr(r,f'H{h}_selected_partners') for h in (1,2,3)],
                'articles':['[MASKED ARTICLE: '+('x'*1200)+']']*sum(getattr(r,f'H{h}_selected_partners') for h in (1,2,3))})
    if not contexts:
        contexts=[{'firm':'SYNTHETIC','characteristics':dict.fromkeys(names,'MASKED'),
                   'articles':['[MASKED '+('x'*1200)+']']*n} for n in (1,5,15)]
    prompts=['Return a complete ordering of this firm-characteristic menu for five-session risk. '
        'No causal claims. JSON ordering only. Menu: '+json.dumps(names)+'. Context: '+json.dumps(c,sort_keys=True) for c in contexts]
    sizes=[len(p.encode('utf-8')) for p in prompts]; outbytes=len(json.dumps({'ordering':names}).encode('utf-8'))
    calls=sum(counts.values()); mean=float(np.mean(sizes))
    # Byte bounds are estimates, not a claim to use DeepSeek's actual tokenizer.
    low=calls*mean/4; high=calls*mean; output=calls*outbytes/3
    dollars=None
    if input_rate is not None and output_rate is not None:
        if min(input_rate,output_rate)<0: raise ValueError('Nonnegative user-supplied prices')
        dollars={'low':(low*input_rate+output*output_rate)/1e6,'high':(high*input_rate+output*output_rate)/1e6}
    new_json(OUT/'TOKEN_COST_ESTIMATE_v1.json',{'eligible_firm_weeks_with_news':counts,'one_ordering_call_per_firm_week':calls,
        'measured_prompt_bytes':sizes,'prompt_sample_type':'masked or synthetic; no provider calls',
        'estimated_input_tokens_range':[low,high],'estimated_output_tokens':output,
        'tokenizer':'UTF-8 bytes /4 to bytes /1; proxy bounds, not measured provider tokens',
        'user_supplied_input_usd_per_million':input_rate,'user_supplied_output_usd_per_million':output_rate,
        'estimated_usd':dollars,'cost_formula':'calls * (mean input tokens * input price + output tokens * output price) / 1e6',
        'excluded':'Retries, system/provider framing, failed calls, cache discounts; no price verified offline',
        'all_years_available':set(counts)=={str(y) for y in range(2015,2025)},'actual_provider_calls':0})

def freeze():
    require_synthetic(); model=io.json_read(OUT/'MODEL_v1.json'); td1=io.verify_td1()
    for year in range(2015,2025):
        if not (OUT/f'features/year_{year}.parquet').exists(): raise PermissionError('Missing feature year '+str(year))
    for kind in SLOT_SCHEMAS:
        validate_slot(io.json_read(OUT/f'selector_slots/{kind}_v1.json'),kind,model['names'])
    code={str(p):sha(p) for p in sorted(CODE.glob('*.py'))}
    private={str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('FREEZE_v1.json','CLEARANCE_REQUEST_v1.json')}
    expected_inputs={str(TD1/f'inputs/crsp_{y}.parquet'):td1['input_sha256'][str(TD1/f'inputs/crsp_{y}.parquet')] for y in (2023,2024)}
    expected_inputs.update({str(TD1/f'predictions_v2/year_{y}.parquet'):td1['prediction_sha256'][str(TD1/f'predictions_v2/year_{y}.parquet')] for y in (2023,2024)})
    binding={'schema':'TD2_FREEZE_v1','version':VERSION,'code_sha256':code,'artifact_sha256':private,
        'td1_freeze_sha256':sha(TD1/'MODEL_FREEZE_v3.json'),'td1_evaluation_inputs_sha256':expected_inputs,
        'model_sha256':sha(OUT/'MODEL_v1.json'),'exploratory':True,'evaluation_years':[2023,2024],
        'maximum_label_end':'2024-12-31','evaluation_run':False}
    new_json(OUT/'FREEZE_v1.json',binding)
    request={'schema':'TD2_CLEARANCE_v1','status':'CLEARED','version':VERSION,
        'freeze_sha256':sha(OUT/'FREEZE_v1.json'),
        **{k:binding[k] for k in ('code_sha256','artifact_sha256','td1_freeze_sha256','td1_evaluation_inputs_sha256','model_sha256')},
        'evaluation_years':[2023,2024],'maximum_label_end':'2024-12-31'}
    new_json(OUT/'CLEARANCE_REQUEST_v1.json',request,timed=False)
    print('CLEARANCE_REQUEST_ONLY',str(OUT/'CLEARANCE_REQUEST_v1.json'),flush=True)
