"""qwen3.8-27b: new development-only PR2 8/6/6 pool from real Reuters indexes."""
from __future__ import annotations
import gc
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .hop3_guard import MODEL, LIVE, ROOT, PROJECT, EXPOSURE, REUTERS_INDEX, DATES, SPLITS, Guard, check_case, digest, read, write, sha, resources

def metadata(out):
    resources('metadata')
    cases=[]; inputs={}
    for day in DATES:
        source=LIVE/'pools'/(day+'.json')
        original=read(source)
        pools={}; docs={}
        for cid,p in original['pools'].items():
            c=check_case(p['case'])
            if c['date']!=day: raise ValueError('Cross-date live pool')
            cases.append(c)
            pools[cid]={k:p[k] for k in ('case','calendar','controls','own_ids','common_ids','history_ids','own_packet_hash')}
            docs.update({sid:original['documents'][sid] for sid in p['own_ids']})
        if len(pools)!=50: raise ValueError('Expected exactly 50 firms per day')
        write(out/'base'/(day+'.json'),{'pools':pools,'documents':docs})
        inputs[str(source)]=sha(source)
        del original,pools,docs; gc.collect()
    if len(cases)!=300 or len({c['case_id'] for c in cases})!=300: raise ValueError('Expected 300 development cases')
    write(out/'cases.json',{'cases':sorted(cases,key=lambda c:(c['date'],c['ticker']))})
    label_metadata=read(LIVE/'numerical_manifest.json')
    write(out/'label_source_manifest.json',{'expected_sha256':{f'labels_{s}.parquet':label_metadata['outputs'][f'labels_{s}.parquet'] for s in ('train','valid')},
        'source_manifest':str(LIVE/'numerical_manifest.json'),'source_manifest_sha256':sha(LIVE/'numerical_manifest.json'),
        'note':'Existing metadata hashes only; no label file opened'})
    from news_exposure_2025.graph import Graph
    graph=Graph(read(LIVE/'graph.json'))
    inputs[str(LIVE/'graph.json')]=sha(LIVE/'graph.json')
    paths={}
    from .hop3_shortest import shortest_pool_paths
    for ticker in sorted({c['ticker'] for c in cases}):
        relations=shortest_pool_paths(graph,ticker,3)
        if any(r.get('type')=='comention' for r in relations): raise ValueError('Unexpected co-mention')
        root_permnos=sorted({p for n in graph.data['targets'][ticker] for p in graph.data['nodes'][n]['permnos']})
        write(out/'relations'/(ticker+'.json'),{'relations':relations,'root_permnos':root_permnos,
            'witness_policy':'Exact full-graph BFS distances. One witness per mapped entity, prefer mapped intermediates, fewer competitor steps, stable IDs; retain every witness prefix.'})
        paths[ticker]={'counts':{h:sum(p['hops']==h for p in relations) for h in (1,2,3)},
                       'higher_hop_limit_per_depth':None,'possibly_truncated':False}
    universe=sorted(n for n,v in graph.data['nodes'].items() if v.get('permnos'))
    snapshot={'nodes':list(graph.data['nodes']), 'edges':[{k:e[k] for k in ('id','source','target','type','valid_from','vintage')} for e in graph.data['edges']],
              'targets':graph.data['targets'],'vintage':graph.data['audit']['graph_date'],
              'monthly_universe':{m:universe for m in ('2025-10','2025-11')}}
    write(out/'graph_snapshot.json',snapshot)
    write(out/'preparation_protocol.json',{'max_hops':3,'version':'hop3-pool-v3-exact-shortest','splits':SPLITS,'inputs':inputs,
        'graph_assumption':graph.data['audit'],'enumeration':paths,'outcomes_read':False,
        'monthly_universe':'June mapped entities held fixed; not certified monthly membership',
        'own_common':'Byte-identical source packets from six original development pools; no historical network candidates reused',
        'network_selection':'Fresh shortest typed paths, 8/6/6 slots, diversified MiniLM headline + recency rank, fill to 20',
        'continuity_fields':{'edge_fields':sorted({k for e in graph.data['edges'] for k in e}),
                             'node_fields':sorted({k for n in graph.data['nodes'].values() for k in n}),
                             'product_receipts_available':False}})
    resources('metadata_complete')

def prepare_day(out,day):
    if day not in DATES: raise ValueError('Evaluation pool preparation forbidden')
    resources('prepare_'+day)
    from .data import three_hop_candidates, substantive_neighbor, graph_control_scope
    from news_exposure_2025.data import resolve_history, relevant_common
    import news_event_risk_data_2025 as legacy
    legacy.ROOT=PROJECT # read-only helper module paths; no numerical helpers invoked for labels
    base=read(out/'base'/(day+'.json'))
    relation_records={p['case']['ticker']:read(out/'relations'/(p['case']['ticker']+'.json')) for p in base['pools'].values()}
    relations={ticker:r['relations'] for ticker,r in relation_records.items()}
    eligible={n for rs in relations.values() for r in rs for n in r['permnos']}
    idx=[]; inputs={}
    # Only September onward is needed for the 31-day network window. All known
    # versions in these months are resolved before the window/eligibility filter.
    for month in ('2025-09','2025-10','2025-11'):
        if month>day[:7]: continue
        meta=read(REUTERS_INDEX/(month+'.json'))
        ip=EXPOSURE/'reuters_index'/(month+'.jsonl.gz')
        saved=read(ip.with_suffix('.manifest.json'))
        actual=sha(ip)
        if actual!=saved['sha256']: raise ValueError('Index checksum mismatch')
        inputs[str(ip)]=actual
        idx.append({**meta,'index':str(ip)})
    cutoff=pd.Timestamp(next(iter(base['pools'].values()))['case']['asof'])
    allrows=resolve_history(idx,cutoff.isoformat(),eligible)
    rows=[r for r in allrows if r['usable'] and not legacy.NON_SUBSTANTIVE.search(r['headline'])
          and 0<(cutoff-pd.Timestamp(r['available_at'])).total_seconds()<=31*86400]
    del allrows; gc.collect()
    resources('before_minilm_'+day)
    # Dependencies must be installed in the selected interpreter in advance.
    # No per-user site-packages injection, installation or download.
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',local_files_only=True,device='cpu')
    q=model.encode(['supplier customer competitor disruption demand bottleneck capacity substitution earnings'],normalize_embeddings=True)[0]
    import sqlite3
    with sqlite3.connect(out/'headline_embeddings_qwen3.8-27b.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS vectors (headline_hash TEXT PRIMARY KEY, value BLOB NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        fingerprint=digest({'encoder':'sentence-transformers/all-MiniLM-L6-v2',
                            'commit':getattr(model[0].auto_model.config,'_commit_hash',None),'normalize':True})
        previous=db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if previous and previous[0]!=fingerprint:raise ValueError('Local embedding encoder changed')
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('fingerprint',fingerprint))
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('run_model',MODEL))
        cached={k:np.frombuffer(v,dtype=np.float32) for k,v in db.execute('SELECT headline_hash,value FROM vectors')}
        keys=[digest(r['headline']) for r in rows]
        missing={k:r['headline'] for k,r in zip(keys,rows) if k not in cached}
        items=list(missing.items())
        print(MODEL,'HEADLINES',day,'total',len(rows),'uncached',len(items),flush=True)
        for start in range(0,len(items),1024):
            part=items[start:start+1024]
            encoded=model.encode([t for _,t in part],batch_size=64,normalize_embeddings=True,show_progress_bar=False)
            db.executemany('INSERT OR IGNORE INTO vectors VALUES (?,?)',[(k,v.astype(np.float32).tobytes()) for (k,_),v in zip(part,encoded)])
            db.commit()
            cached.update({k:v for (k,_),v in zip(part,encoded)})
            if start%5120==0:print(MODEL,'EMBEDDED',day,start+len(part),'/',len(items),flush=True)
        vectors=np.stack([cached[k] for k in keys])
        del cached,missing,items
    ages=np.array([(cutoff-pd.Timestamp(r['available_at'])).total_seconds()/86400 for r in rows])
    scores=vectors@q+.12*np.exp(-ages/7)
    del model,vectors; gc.collect()
    # Exact line parser from the read-only news_percentile/build.py. Loading
    # legacy.news_helpers would unnecessarily import incompatible GARCH code.
    from types import SimpleNamespace
    def parse_line(raw):
        position=raw.find(b'{')
        if position<0:return None
        value=raw[position:].rstrip().rstrip(b',')
        for cut in range(4):
            try:return json.loads(value[:-cut] if cut else value)
            except (ValueError,UnicodeDecodeError):continue
        return None
    parser=SimpleNamespace(parse_line=parse_line)
    docs=base['documents']; pools={}; audit=[]
    for cid,p in base['pools'].items():
        c=check_case(p['case']); rel=relations[c['ticker']]
        neighbors={n for r in rel for n in r['permnos']}
        own_equivalent=set(relation_records[c['ticker']]['root_permnos'])|{c['permno']}
        net_idx=[i for i,r in enumerate(rows) if not own_equivalent.intersection(r['ids']) and not relevant_common(r)
                 and set(r['ids'])&neighbors and substantive_neighbor(r['headline'])]
        selected=three_hop_candidates(rows,scores,net_idx,rel)
        selected=[r for r in selected if r['source_id'] not in p['own_ids']]
        for r in selected:
            sid=r['source_id']
            if sid not in docs:
                body=legacy.read_article(r,parser)
                docs[sid]={k:r[k] for k in ('source_id','headline','available_at','ids','story_id','raw_sha256')}
                docs[sid]['body']=body or r['headline']
        direct,di=graph_control_scope(rel,rows,net_idx,3)
        controls={**p['controls'],'log_neighbor_count':float(np.log1p(sum(ages[i]<7 for i in di))),
                  'log_degree':float(np.log1p(len(direct)))}
        from .hop_ablation import exact_hop_counts
        counts=exact_hop_counts(selected,rel,document_mode=True)
        pools[cid]={**p,'controls':controls,'network_ids':[r['source_id'] for r in selected],
                    'coverage':{'max_hops':3,'candidate_exact_hops':counts,'network_candidates':len(selected)}}
        audit.append({'model':MODEL,'case_id':cid,'split':c['split'],'date':day,**{f'exact_h{h}':v for h,v in counts.items()},
                      'any_h3':bool(counts[3])})
    write(out/'pools'/(day+'.json'),{'pools':pools,'documents':docs,'audit':audit,'index_inputs':inputs})
    resources('pool_complete_'+day)

def finish(out):
    cases=read(out/'cases.json')['cases']; rows=[]; hashes={}
    support_complete=True
    for day in DATES:
        p=out/'pools'/(day+'.json'); v=read(p); rows+=v['audit']; hashes[str(p.relative_to(out))]=sha(p)
        support_complete=support_complete and v.get('prefix_support_complete',False)
        for pool in v['pools'].values(): check_case(pool['case'])
    coverage={s:{'cases':sum(r['split']==s for r in rows),
                 'candidate_counts':{h:sum(r[f'exact_h{h}'] for r in rows if r['split']==s) for h in (1,2,3)},
                 'share_any_h3':float(np.mean([r['any_h3'] for r in rows if r['split']==s]))} for s in ('train','valid')}
    write(out/'coverage_audit.json',{'by_split':coverage,'cases':rows})
    pd.DataFrame(rows).to_csv(out/'coverage_audit_qwen3.8-27b.csv',index=False)
    for p in [out/'cases.json',out/'graph_snapshot.json',out/'preparation_protocol.json',out/'label_source_manifest.json',*sorted((out/'relations').glob('*.json'))]:
        hashes[str(p.relative_to(out))]=sha(p)
    write(out/'data_manifest.json',{'max_hops':3,'version':'hop3-pool-v3-exact-shortest','n':len(cases),'coverage':coverage,'outputs':hashes,
                                  'outcomes_read':False,'evaluation_cases_opened':False,'prefix_support_complete':support_complete})
    print(MODEL,'NEW_POOL_COMPLETE',coverage,flush=True)

def prefix_support(out,day):
    """Supplemental host-only prefix inspections; never occupy scoring slots."""
    if day not in DATES:raise ValueError('Evaluation support forbidden')
    resources('prefix_support_'+day)
    from news_exposure_2025.data import resolve_history
    import news_event_risk_data_2025 as legacy
    legacy.ROOT=PROJECT
    from types import SimpleNamespace
    batch=read(out/'pools'/(day+'.json'))
    if batch.get('prefix_support_complete'):return
    relations={p['case']['ticker']:read(out/'relations'/(p['case']['ticker']+'.json'))['relations'] for p in batch['pools'].values()}
    eligible={n for rs in relations.values() for r in rs for n in r['permnos']}
    idx=[]
    for month in ('2025-09','2025-10','2025-11'):
        if month>day[:7]:continue
        meta=read(REUTERS_INDEX/(month+'.json'))
        ip=EXPOSURE/'reuters_index'/(month+'.jsonl.gz')
        if sha(ip)!=batch['index_inputs'][str(ip)]:raise ValueError('Support index changed')
        idx.append({**meta,'index':str(ip)})
    cutoff=pd.Timestamp(next(iter(batch['pools'].values()))['case']['asof'])
    rows=resolve_history(idx,cutoff.isoformat(),eligible)
    rows=[r for r in rows if r['usable'] and not legacy.NON_SUBSTANTIVE.search(r['headline'])
          and 0<(cutoff-pd.Timestamp(r['available_at'])).total_seconds()<=31*86400]
    by_permno={}
    for r in sorted(rows,key=lambda r:(r['available_at'],r['source_id']),reverse=True):
        for p in r['ids']:
            by_permno.setdefault(p,r)
    def parse_line(raw):
        pos=raw.find(b'{')
        if pos<0:return None
        value=raw[pos:].rstrip().rstrip(b',')
        for cut in range(4):
            try:return json.loads(value[:-cut] if cut else value)
            except (ValueError,UnicodeDecodeError):continue
        return None
    parser=SimpleNamespace(parse_line=parse_line);support_count=0;covered=0
    for pool in batch['pools'].values():
        rel=relations[pool['case']['ticker']];paths={tuple(r['path']):r for r in rel}
        support_for={};support_paths={}
        for sid in pool['network_ids']:
            doc=batch['documents'][sid]
            matching=[p for p in rel if set(p['permnos'])&set(doc['ids'])]
            shortest=min(p['hops'] for p in matching)
            for path in sorted([p for p in matching if p['hops']==shortest],key=lambda p:p['path']):
                witnesses=[]
                for depth in range(1,shortest):
                    prefix=paths.get(tuple(path['path'][:depth+1]))
                    choices=[by_permno[p] for p in prefix['permnos'] if p in by_permno] if prefix else []
                    if not choices:break
                    witnesses.append(max(choices,key=lambda r:(r['available_at'],r['source_id'])))
                if len(witnesses)!=shortest-1:continue
                support_for[sid]=[r['source_id'] for r in witnesses]
                support_paths[sid]=path['path']
                for r in witnesses:
                    ss=r['source_id']
                    if ss not in batch['documents']:
                        batch['documents'][ss]={k:r[k] for k in ('source_id','headline','available_at','ids','story_id','raw_sha256')}
                        batch['documents'][ss]['body']=legacy.read_article(r,parser) or r['headline']
                if shortest==3:covered+=1
                break
        pool['support_for']=support_for;pool['support_paths']=support_paths
        pool['support_ids']=sorted({s for ids in support_for.values() for s in ids})
        support_count+=len(pool['support_ids'])
    batch['prefix_support_complete']=True
    batch['prefix_support_policy']='Latest dated pre-cutoff usable 31-day article for each intermediate on one complete shortest path. Full host inspection only; never included in Qwen scoring unless already a primary/OWN source. No product continuity inference.'
    write(out/'pools'/(day+'.json'),batch)
    print(MODEL,'PREFIX_SUPPORT',day,'support_case_sources',support_count,'h3_with_complete_prefixes',covered,flush=True)
    resources('prefix_support_complete_'+day)

def prepare(out=ROOT,max_hops=3,stage='all',day=None):
    if max_hops!=3: raise ValueError('This development-only version requires --max-hops 3')
    Guard().install(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if stage=='metadata': metadata(out)
    elif stage=='day': prepare_day(out,day)
    elif stage=='finish': finish(out)
    elif stage=='support':prefix_support(out,day)
    else:
        import subprocess,sys,os
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        jobs=[['metadata']]+[['day',d] for d in DATES]+[['support',d] for d in DATES]+[['finish']]
        for job in jobs:
            artifact=out/('preparation_protocol.json' if job[0]=='metadata' else 'data_manifest.json' if job[0]=='finish' else 'pools/'+job[1]+'.json')
            if job[0]=='support':
                if artifact.exists() and read(artifact).get('prefix_support_complete'):continue
            elif artifact.exists() and job[0]!='finish': continue
            cmd=[sys.executable,'-m','news_network_50_2025.hop3_prepare','--out',str(out),'--stage',job[0]]
            if len(job)>1: cmd+=['--day',job[1]]
            with (out/('prepare_'+ '_'.join(job)+'_qwen3.8-27b.log')).open('a',encoding='utf-8') as log:
                subprocess.run(cmd,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT);p.add_argument('--stage',default='all');p.add_argument('--day')
    a=p.parse_args();prepare(a.out,stage=a.stage,day=a.day)
