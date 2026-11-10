"""Same frozen exposures using bounded sparse shortest distances and selected witnesses."""
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import engine
from spec import PARTNER_CAP

class Graph:
    def __init__(self,adj):
        self.adj=adj; self.nodes=sorted(adj); self.index={p:i for i,p in enumerate(self.nodes)}
        ii=[]; jj=[]
        for a,neighbors in adj.items():
            ii.extend([self.index[a]]*len(neighbors)); jj.extend(self.index[b] for b in neighbors)
        self.matrix=csr_matrix((np.ones(len(ii),np.int8),(ii,jj)),shape=(len(self.nodes),len(self.nodes)))
        self.matrix.sort_indices()

def graph_at(edges,cutoff,origin):
    return Graph(engine.graph_at(edges,cutoff,origin))

def witness(graph,root,partner,h,dist):
    adj=graph.adj
    if h==1: nodes=[root,partner]
    elif h==2:
        u=min(set(adj[root]).intersection(adj[partner])); nodes=[root,u,partner]
    else:
        nodes=None; target_neighbors=set(adj[partner])
        for u in sorted(adj[root]):
            middle=[v for v in adj[u] if v in target_neighbors and dist[graph.index[v]]==2]
            if middle:
                nodes=[root,u,min(middle),partner]; break
        if nodes is None: raise ValueError('No exact-hop witness')
    edges=[adj[a][b] for a,b in zip(nodes[:-1],nodes[1:])]
    return nodes,edges

def exposure(root,cutoff,origin,characteristics,availability,adj,pool,names):
    graph=adj if isinstance(adj,Graph) else Graph(adj)
    if root in graph.index:
        dist=dijkstra(graph.matrix,directed=False,indices=graph.index[root],unweighted=True,limit=3)
    else: dist=np.full(len(graph.nodes),np.inf)
    result=[]; evidence=[]
    for h in (1,2,3):
        partners=[graph.nodes[i] for i in np.flatnonzero(dist==h)]
        candidates=[p for p in partners if p in pool]
        selected=sorted(candidates,key=lambda p:hashlib.sha256(f'{cutoff}|{p}'.encode()).hexdigest())[:PARTNER_CAP]
        ev=dict.fromkeys(names,0.); unique=set()
        for p in selected:
            stories=list(pool[p].values()); unique.update(r['alt_id'] for r in stories)
            for m in names: ev[m]+=max((r['scores'][m] for r in stories),default=0.)/PARTNER_CAP
            nodes,edges=witness(graph,root,p,h,dist)
            evidence.append({'hop':h,'partner':p,'nodes':nodes,'intermediate_nodes':nodes[1:-1],
                'relationship_ids':[e['id'] for e in edges],'relationship_source_rows':[e['rownum'] for e in edges],
                'reporting_roles':[e['typ'] for e in edges],
                'max_edge_available_at':max(str(e['available_at']) for e in edges),
                'article_ids':sorted(r['alt_id'] for r in stories),
                'max_news_available_at':max(str(r['available_at']) for r in stories),
                'intermediate_product_evidence':'unavailable_in_sources','physical_channel_verified':False})
        for m in names:
            c=characteristics.get(m,np.nan); missing=not np.isfinite(c)
            if not missing: engine.past(availability.get(m),cutoff)
            result.append({'hop':h,'name':m,'C':None if missing else float(c),'E':ev[m],
                'Z':0. if missing else float(c)*ev[m],'characteristic_missing':missing,
                'graph_partners':len(partners),'news_partners':len(candidates),
                'selected_partners':len(selected),'unique_articles':len(unique),'physical_channel_verified':False})
    return result,evidence

def mapped_edges(graph,links,origin,suspects=None):
    day=str(pd.Timestamp(origin).date())
    if day>'2024-12-31': raise ValueError('Graph date seal')
    eligible=graph[graph.start<day].drop_duplicates(['id','start'],keep='first').drop_duplicates('id',keep='last')
    eligible=eligible[eligible.end.isna()|(eligible.end>day)]
    lm=links[(links.start<day)&(links.end>=day)].drop_duplicates(['entity','permno'])
    if suspects: lm=lm[[(r.entity,int(r.permno)) not in suspects for r in lm.itertuples()]]
    lm=lm[~lm.duplicated('entity',keep=False)]; lookup=dict(zip(lm.entity,lm.permno))
    eligible=eligible.assign(a=eligible.src.map(lookup),b=eligible.tgt.map(lookup))
    eligible=eligible[eligible.a.notna()&eligible.b.notna()&(eligible.a!=eligible.b)]
    clocks={s:engine.day_clock(s) for s in eligible.start.unique()}
    return [{'id':r.id,'a':int(r.a),'b':int(r.b),'start':r.start,'end':r.end,
        'available_at':clocks[r.start],'rownum':int(r.rownum),'typ':r.typ} for r in eligible.itertuples()]

def install(pipeline,decoder):
    pipeline.graph_at=graph_at; pipeline.exposure=exposure; decoder.mapped_edges=mapped_edges
