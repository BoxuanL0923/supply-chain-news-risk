"""Offline selectors and strict empty slots; never performs a provider call."""
import hashlib
import numpy as np
from spec import SLOT_SCHEMAS

def ordering(names,kind,key,seed=0,beta=None):
    if kind=='random':
        return sorted(names,key=lambda n:hashlib.sha256(f'{seed}|{key}|{n}'.encode()).hexdigest())
    if kind=='training_beta':
        return sorted(names,key=lambda n:(-abs(beta[n]),n))
    raise ValueError('Unknown local selector')

def gate(names,order,k,available=None):
    if len(order)!=len(set(order)) or set(order)!=set(names):
        raise ValueError('Ordering must contain every menu item once; no backfill')
    available=set(names) if available is None else set(available)
    if not available.issubset(names): raise ValueError('Unknown eligible characteristic')
    filtered=[n for n in order if n in available]
    count=len(filtered) if k=='all' else int(k)
    if count<0: raise ValueError('Negative K')
    active=set(filtered[:count])
    return np.array([float(n in active) for n in names])

def choice_gate(names,choice,available):
    if choice!='none' and choice not in names: raise ValueError('Unknown Choice item')
    return np.array([float(n==choice and n in available) for n in names])

def validate_slot(slot,kind,names,allow_empty=True):
    if kind not in SLOT_SCHEMAS or slot.get('schema')!=SLOT_SCHEMAS[kind]['schema']:
        raise ValueError('Slot schema')
    if slot.get('question_version')!=SLOT_SCHEMAS[kind]['question_version']:
        raise ValueError('Question version')
    if not isinstance(slot.get('model_version'),str) or not slot['model_version']:
        raise ValueError('Model version required')
    rows=slot.get('records')
    if not isinstance(rows,list) or (not rows and not allow_empty): raise ValueError('Slot records')
    seen=set()
    for row in rows:
        base={'firm_id','cutoff','input_sha256','receipt_id'}
        extra={'ordering'} if kind=='deepseek_order' else ({'choice','probabilities'} if kind=='jev_choice' else {'yes_scores'})
        if set(row)!=base|extra: raise ValueError('Unexpected or missing slot fields')
        key=(row['firm_id'],row['cutoff'])
        if key in seen: raise ValueError('Duplicate firm-week slot')
        seen.add(key)
        if len(row['input_sha256'])!=64 or not row['receipt_id']: raise ValueError('Input hash/receipt')
        if kind=='deepseek_order': gate(names,row['ordering'],'all')
        elif kind=='jev_choice':
            probs=row['probabilities']
            if set(probs)!=set(names)|{'none'} or row['choice'] not in probs: raise ValueError('Choice distribution')
            if not all(isinstance(v,(int,float)) and 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>1e-8:
                raise ValueError('Choice probabilities')
        else:
            scores=row['yes_scores']
            if set(scores)!=set(names) or not all(isinstance(v,(int,float)) and 0<=v<=1 for v in scores.values()):
                raise ValueError('Noul scores')
    return True

def noul_gate(names,scores,available,k='all',threshold=None):
    order=sorted(names,key=lambda n:(-scores[n],n))
    if threshold is not None:
        available=[n for n in available if scores[n]>=threshold]
    return gate(names,order,k,available)
