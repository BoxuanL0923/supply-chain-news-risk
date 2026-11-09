"""Pure as-of graph/news interactions. No readers, returns, labels, or providers."""
from collections import deque
import hashlib
import re
import numpy as np
import pandas as pd
from spec import NAMES,LEXICONS,PARTNER_CAP,LOOKBACK_DAYS

def day_clock(day):
    return (pd.Timestamp(day).normalize()+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC').tz_localize(None)

def past(available,cutoff):
    if pd.isna(available) or pd.Timestamp(available)>=pd.Timestamp(cutoff):
        raise ValueError('Future or missing availability')

def score(text):
    words=set(re.findall('[a-z]+',text.lower()))
    return {n:len(words.intersection(LEXICONS[n]))/len(LEXICONS[n]) for n in NAMES}

def graph_at(edges,cutoff,origin):
    """Edges are already date-sealed and mapped as of origin by the decoder."""
    adj={}
    for edge in edges:
        past(edge['available_at'],cutoff)
        if pd.Timestamp(edge['start'])>=pd.Timestamp(origin):
            raise ValueError('Future edge start')
        if edge.get('end') and pd.Timestamp(edge['end'])<=pd.Timestamp(origin):
            continue
        a,b=edge['a'],edge['b']
        if a==b: continue
        for u,v in [(a,b),(b,a)]:
            # Preserve a deterministic source witness when parallel links exist.
            old=adj.setdefault(u,{}).get(v)
            if old is None or edge['id']<old['id']:
                adj[u][v]=edge
    return adj

def exact_paths(adj,root):
    found={root:([],[])}; queue=deque([root])
    while queue:
        u=queue.popleft(); nodes,edges=found[u]
        if len(edges)==3: continue
        for v in sorted(adj.get(u,{})):
            if v not in found:
                found[v]=(nodes+[v],edges+[adj[u][v]])
                queue.append(v)
    return {h:{v:p for v,p in found.items() if len(p[1])==h} for h in (1,2,3)}

def news_pool(stories,cutoff,origin):
    lo=pd.Timestamp(origin)-pd.Timedelta(days=LOOKBACK_DAYS)
    pool={}
    for row in stories:
        past(row['available_at'],cutoff)
        d=pd.Timestamp(row['date'])
        if d>=pd.Timestamp(origin): raise ValueError('Origin-day or future news')
        if d<lo: continue
        for p in row['firms']:
            pool.setdefault(p,{})[row['alt_id']]=row
    return pool

def exposure(root,cutoff,origin,characteristics,availability,adj,pool,names):
    paths=exact_paths(adj,root); records=[]; evidence=[]
    for h in (1,2,3):
        candidates=[p for p in paths[h] if p in pool]
        selected=sorted(candidates,key=lambda p:hashlib.sha256(f'{cutoff}|{p}'.encode()).hexdigest())[:PARTNER_CAP]
        ev={m:0. for m in names}; unique=set()
        for p in selected:
            stories=list(pool[p].values()); unique.update(r['alt_id'] for r in stories)
            for m in names:
                ev[m]+=max((r['scores'][m] for r in stories),default=0.)/PARTNER_CAP
            nodes,edges=paths[h][p]
            evidence.append({'hop':h,'partner':p,'nodes':[root]+nodes,
                'intermediate_nodes':nodes[:-1],'relationship_ids':[e['id'] for e in edges],
                'relationship_source_rows':[e['rownum'] for e in edges],
                'reporting_roles':[e['typ'] for e in edges],
                'max_edge_available_at':max(str(e['available_at']) for e in edges),
                'article_ids':sorted(r['alt_id'] for r in stories),
                'max_news_available_at':max(str(r['available_at']) for r in stories),
                'intermediate_product_evidence':'unavailable_in_sources',
                'physical_channel_verified':False})
        for m in names:
            c=characteristics.get(m,np.nan); missing=not np.isfinite(c)
            if not missing: past(availability.get(m),cutoff)
            records.append({'hop':h,'name':m,'C':None if missing else float(c),
                'E':ev[m],'Z':0. if missing else float(c)*ev[m],
                'characteristic_missing':missing,'graph_partners':len(paths[h]),
                'news_partners':len(candidates),'selected_partners':len(selected),
                'unique_articles':len(unique),'physical_channel_verified':False})
    return records,evidence

def feature_packet(record,names):
    """Strict schema makes future labels/returns impossible selector inputs."""
    allowed={'firm_id','cutoff','origin_session','C','available_at','E','news_ids','graph_evidence'}
    if set(record)-allowed: raise ValueError('Forbidden feature input fields')
    if set(record['C'])!=set(names) or set(record['E'])!=set(names):
        raise ValueError('Menu-to-feature mapping mismatch')
    for m,v in record['C'].items():
        if v is not None and np.isfinite(v): past(record['available_at'][m],record['cutoff'])
    return record

def ridge(x,residual,penalty):
    if penalty<=0: raise ValueError('Positive ridge required')
    return np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]),x.T@residual)
