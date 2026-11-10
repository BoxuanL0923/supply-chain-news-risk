"""Pure future-slot adapter: no transport, SDK, credentials, or provider calls."""
import hashlib
import json
import numpy as np
import pandas as pd
from engine import feature_packet
from selectors import validate_slot,gate,choice_gate,noul_gate
from spec import K_GRID

def context_hash(packet,names):
    safe=feature_packet(packet,names)
    return hashlib.sha256(json.dumps(safe,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()

def slot_lookup(slot,kind,names,expected_hashes):
    validate_slot(slot,kind,names)
    if slot['records'] and slot['model_version']=='REQUIRED': raise ValueError('Pin actual provider version before accepting records')
    result={}
    for r in slot['records']:
        key=(r['firm_id'],str(pd.Timestamp(r['cutoff'])))
        if key not in expected_hashes or expected_hashes[key]!=r['input_sha256']:
            raise ValueError('Provider input differs from frozen eligible context')
        result[key]=r
    return result

def gates(frame,names,kind,slot,k,threshold,fallback,expected_hashes):
    if fallback not in ('none','all'): raise ValueError('Fallback must be selected on development')
    lookup=slot_lookup(slot,kind,names,expected_hashes); q=[]; missing=[]
    for r in frame.itertuples():
        available=[n for n in names if not getattr(r,'missing_'+n)]
        key=(r.firm_id,str(pd.Timestamp(r.cutoff))); row=lookup.get(key)
        if row is None:
            missing.append({'firm_id':r.firm_id,'cutoff':str(r.cutoff),'reason':'no_valid_slot_record'})
            q.append(gate(names,names,0 if fallback=='none' else 'all',available)); continue
        if kind=='deepseek_order': q.append(gate(names,row['ordering'],k,available))
        elif kind=='jev_choice': q.append(choice_gate(names,row['choice'],available))
        else: q.append(noul_gate(names,row['yes_scores'],available,k,threshold))
    return np.asarray(q),missing

def tune_selection(frame,names,beta,kind,slot,expected_hashes):
    """Caller supplies baseline and Z; no label is ever placed in selector context."""
    origins=pd.to_datetime(frame.origin_session)
    if not origins.dt.year.isin([2021,2022]).all(): raise PermissionError('Only 2021-2022 can select provider K/threshold/fallback')
    if (pd.to_datetime(frame.label_end)>pd.Timestamp('2022-12-31')).any(): raise PermissionError('Selection window crosses evaluation')
    validate_slot(slot,kind,names)
    if not slot['records']:
        return {'status':'EMPTY_PENDING_SELECTION','selected':None,'grid':[]}
    rows=[]; best=None
    candidates=[(1,None)] if kind=='jev_choice' else [(k,None) for k in K_GRID]
    if kind=='jev_noul': candidates += [('all',t) for t in (.25,.5,.75)]
    z=frame[['Z_'+n for n in names]].to_numpy(float)
    for k,threshold in candidates:
        for fallback in ('none','all'):
            q,missing=gates(frame,names,kind,slot,k,threshold,fallback,expected_hashes)
            pred=frame.b.to_numpy(float)+np.sum(z*q*np.asarray(beta),axis=1)
            row={'K':k,'threshold':threshold,'fallback':fallback,
                'selection_mse':float(np.mean((frame.y.to_numpy(float)-pred)**2)),
                'missing_provider_records':len(missing)}
            rows.append(row)
            if best is None or row['selection_mse']<best['selection_mse']: best=row
    return {'status':'SELECTED_ON_2021_2022','selected':best,'grid':rows}
