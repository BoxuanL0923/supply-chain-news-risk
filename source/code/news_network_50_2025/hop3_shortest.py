"""qwen3.8-27b: exact typed-hop distances, without capped-enumeration bias."""
from collections import deque

def shortest_pool_paths(graph,ticker,max_hops=3):
    """One shortest witness for every mapped reachable entity, plus prefixes.

    BFS visits the complete typed graph. Witnesses prefer mapped intermediates,
    then fewer competitor steps, then stable entity/edge IDs. These outcome-free
    tie-breaks improve inspectability without changing distance. All needed
    prefixes survive, including unmapped intermediates. Alternative shortest
    paths are not exhaustively enumerated and continuity is never inferred.
    """
    if max_hops not in (1,2,3):raise ValueError('Hop radius must be 1..3')
    roots=graph.data['targets'][ticker]
    distance=dict.fromkeys(roots,0);queue=deque(roots)
    while queue:
        node=queue.popleft()
        if distance[node]>=max_hops:continue
        for neighbor,_,edge in graph.adj[node]:
            if neighbor not in distance:
                distance[neighbor]=distance[node]+1;queue.append(neighbor)
    best={r:{'path':[r],'steps':[],'mapped':0,'competitors':0} for r in roots}
    for depth in range(1,max_hops+1):
        for node in sorted(n for n,d in distance.items() if d==depth):
            if depth==max_hops and not graph.data['nodes'][node]['permnos']:continue
            choice=None;chosen_key=None
            for parent,_,edge in graph.adj[node]:
                if distance.get(parent)!=depth-1:continue
                previous=best[parent]
                role=('competitor' if edge['type']=='competitor' else 'customer' if edge['source']==parent else 'supplier')
                mapped=previous['mapped']+int(parent not in roots and bool(graph.data['nodes'][parent]['permnos']))
                competitors=previous['competitors']+int(role=='competitor')
                chain=previous['path']+[node]
                key=(-mapped,competitors,tuple(chain),str(edge['id']))
                if chosen_key is None or key<chosen_key:
                    chosen_key=key
                    choice={'path':chain,'mapped':mapped,'competitors':competitors,
                            'steps':previous['steps']+[{'from':parent,'to':node,'neighbor_role':role,
                                      'edge_id':edge['id'],'revenue_pct':edge.get('revenue_pct')}]}
            if choice is None:raise ValueError('Missing BFS predecessor')
            best[node]=choice
    keep={n for n,d in distance.items() if 0<d<=max_hops and graph.data['nodes'][n]['permnos']}
    keep.update(n for terminal in list(keep) for n in best[terminal]['path'][1:-1])
    return [{'entity':n,**graph.data['nodes'][n],'hops':distance[n],
             'path':best[n]['path'],'steps':best[n]['steps'],'degree':len(graph.adj[n]),
             'vintage':graph.data['audit']['graph_date'],'mechanism_verified':False,
             'full_graph_shortest_distance':True} for n in sorted(keep,key=lambda n:(distance[n],n))]

def audit_pool(root):
    """Independent set-adjacency BFS against every selected primary source."""
    from collections import defaultdict,Counter
    from .hop3_guard import LIVE,DATES,MODEL,read,write,sha,resources,check_case
    resources('independent_full_graph_radius_audit')
    data=read(LIVE/'graph.json');adj=defaultdict(set);mapping=defaultdict(set)
    for edge in data['edges']:
        adj[edge['source']].add(edge['target']);adj[edge['target']].add(edge['source'])
    for node,value in data['nodes'].items():
        for permno in value['permnos']:mapping[permno].add(node)
    distances={}
    for ticker,roots in data['targets'].items():
        d=dict.fromkeys(roots,0);q=deque(roots)
        while q:
            node=q.popleft()
            if d[node]>=3:continue
            for neighbor in adj[node]:
                if neighbor not in d:d[neighbor]=d[node]+1;q.append(neighbor)
        distances[ticker]=d
    mismatches=[];cross=Counter()
    for day in DATES:
        batch=read(root/'pools'/(day+'.json'))
        for cid,pool in batch['pools'].items():
            check_case(pool['case'])
            rel=read(root/'relations'/(pool['case']['ticker']+'.json'))['relations']
            d=distances[pool['case']['ticker']]
            for sid in pool['network_ids']:
                ids=batch['documents'][sid]['ids']
                reported=min(p['hops'] for p in rel if set(p['permnos'])&set(ids))
                actual=min(d.get(n,99) for permno in ids for n in mapping.get(permno,()))
                cross[(pool['case']['split'],reported,actual)]+=1
                if reported!=actual:mismatches.append({'case_id':cid,'source_id':sid,'reported':reported,'actual':actual})
    value={'mismatches':mismatches,'counts':[{'split':s,'reported':r,'actual':a,'n':n} for (s,r,a),n in sorted(cross.items())],
           'data_manifest_sha256':sha(root/'data_manifest.json'),'graph_source_sha256':sha(LIVE/'graph.json'),
           'resources':resources('full_graph_radius_audit_complete')}
    write(root/'hop_ablation_v2'/'full_graph_radius_audit.json',value)
    print(MODEL,'FULL_GRAPH_RADIUS_MISMATCHES',len(mismatches),flush=True)
    if mismatches:raise ValueError('Prepared radius is not the full-graph shortest radius')
    return value

if __name__=='__main__':
    from .hop3_guard import ROOT,Guard
    Guard().install();audit_pool(ROOT)
