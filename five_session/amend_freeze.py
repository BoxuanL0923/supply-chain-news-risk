"""Append a code-only freeze amendment; never refit or overwrite an artifact."""
from runtime import limit, OUT, CODE, sha, new_json, utc
limit()
import json

def main():
    parent=OUT/'MODEL_FREEZE.json'
    freeze=json.loads(parent.read_text())
    for field in ['input_sha256','manifest_sha256','prediction_sha256','selection_sha256']:
        for p,h in freeze[field].items():
            if sha(p)!=h: raise ValueError('Immutable experiment artifact changed: '+p)
    if sha(OUT/'model_frozen.pkl')!=freeze['model_sha256']:
        raise ValueError('Model changed')
    freeze['parent_freeze_sha256']=sha(parent)
    freeze['amendment']='Verifier reconstructs fitting array in C order, preserving exact preprocessing equality. No data, model, prediction, selection or numerical fitting change.'
    freeze['code_sha256']={str(p):sha(p) for p in sorted(CODE.glob('*.py'))}
    freeze['utc']=utc(); path=OUT/'MODEL_FREEZE_v2.json'; new_json(path,freeze)
    request={'schema':'TD1_CLEARANCE_v1','status':'CLEARED','freeze_sha256':sha(path),
        'scope':'2023-2024 exploratory paired losses only; all label ends <=2024-12-31'}
    for key in ['version','model_sha256','code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256']:
        request[key]=freeze[key]
    new_json(OUT/'CLEARANCE_REQUEST_v2.json',request)
    print('Amended freeze:',path,flush=True)

if __name__=='__main__': main()
