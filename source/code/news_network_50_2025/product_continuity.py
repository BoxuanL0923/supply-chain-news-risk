"""qwen3.8-27b: conservative, source-bound multihop product continuity.

Topology, revenue percentages, names, and industry similarity never establish
continuity. A verified chain requires (1) supply direction consistent at every
step; (2) a dated item receipt on EACH edge; (3) a dated intermediate receipt
explicitly linking the two incident items; (4) a dated affected-item receipt
from the endpoint event. Every receipt must quote an actually inspected source
verbatim, bind its exact edge/node IDs, and precede cutoff. Generic segment
fields qualify only through this same evidence schema, never by word overlap.
The current FactSet cache contains none of these product receipts: path_only.
"""
from __future__ import annotations
from .graph_skills import utc

def validate_continuity(path, cutoff, *, edges, facts=(), inspected_sources=None, affected_item=None):
    inspected_sources=inspected_sources or {}
    hops=path.get('hops'); chain=path.get('path',[]); steps=path.get('steps',[])
    if hops not in (2,3) or len(chain)!=hops+1 or len(steps)!=hops or len(set(chain))!=len(chain):
        raise ValueError('Expected a simple two- or three-hop path')
    at=utc(cutoff)
    def receipt_ok(f):
        source=inspected_sources.get(f.get('source_id'))
        if not source or not f.get('available_at') or not f.get('quote'):
            return False
        try:
            return (utc(f['available_at']) < at and utc(source['available_at']) < at
                    and utc(f['available_at']) >= utc(source['available_at'])
                    and len(f['quote'].strip())>=12 and f['quote'] in source['body'])
        except (ValueError,KeyError,TypeError): return False
    valid=[f for f in facts if receipt_ok(f)]
    reasons=[]; evidence=[]; items=[]; roles=[]
    if not path.get('vintage') or utc(path['vintage'])>=at:
        reasons.append('missing_or_post_cutoff_path_vintage')
    for a,b,s in zip(chain,chain[1:],steps):
        e=edges.get(str(s.get('edge_id')))
        if isinstance(e,list):
            candidates=[row for row in e if {row.get('source'),row.get('target')}=={a,b}]
            e=candidates[0] if len(candidates)==1 else None
        if not e or {e.get('source'),e.get('target')}!={a,b}:
            reasons.append('missing_or_mismatched_edge'); items.append(None); continue
        if (not e.get('vintage') or not e.get('valid_from') or utc(e['vintage'])>=at or
            utc(e['valid_from'])>=at or (e.get('valid_to') and utc(e['valid_to'])<=at)):
            reasons.append('undated_future_or_expired_edge')
        role='customer' if e['source']==a else 'supplier'
        roles.append(role)
        if e.get('type')!='supplies' or s.get('neighbor_role')!=role:
            reasons.append('not_consistent_supply_step')
        matched=[f for f in valid if f.get('kind')=='edge_product' and str(f.get('edge_id'))==str(s['edge_id'])
                 and f.get('supplier')==e['source'] and f.get('customer')==e['target'] and f.get('item')]
        unique={f['item'] for f in matched}
        if len(unique)!=1:
            reasons.append('missing_or_ambiguous_edge_product'); items.append(None)
        else: items.append(next(iter(unique))); evidence.extend(matched)
    if len(set(roles))!=1: reasons.append('direction_change')
    if not affected_item or not any(f.get('kind')=='affected_item' and f.get('entity')==chain[-1]
            and f.get('item')==affected_item for f in valid):
        reasons.append('missing_endpoint_affected_item_receipt')
    elif items[-1]!=affected_item: reasons.append('affected_item_not_on_endpoint_edge')
    else: evidence.extend(f for f in valid if f.get('kind')=='affected_item' and f.get('entity')==chain[-1] and f.get('item')==affected_item)
    intermediate=[]
    for i,node in enumerate(chain[1:-1],1):
        incoming,outgoing=(items[i],items[i-1]) if roles and roles[0]=='supplier' else (items[i-1],items[i])
        matches=[f for f in valid if f.get('kind')=='intermediate_flow' and f.get('entity')==node
                 and incoming and outgoing and f.get('input_item')==incoming and f.get('output_item')==outgoing
                 and set(map(str,f.get('edge_ids',[])))=={str(steps[i-1].get('edge_id')),str(steps[i].get('edge_id'))}]
        intermediate.append({'entity':node,'verified':bool(matches),'input_item':incoming,'output_item':outgoing})
        if not matches: reasons.append('missing_intermediate_product_flow:'+node)
        evidence.extend(matches)
    return {'model':'qwen3.8-27b','status':'path_only' if reasons else 'continuity_verified',
            'hops':hops,'path':chain,'cutoff':cutoff,'intermediates':intermediate,
            'reasons':sorted(set(reasons)), 'evidence':evidence,
            'meaning':'evidence supports product continuity, not economic materiality or causal propagation'}
