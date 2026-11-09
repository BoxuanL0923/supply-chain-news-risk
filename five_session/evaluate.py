"""Claude-only, hash-bound, once-only exploratory 2023-2024 evaluation.

Importing this module computes no target. The clearance is checked before any
return, feature, prediction, or model payload is read. No clearance is issued.
"""
import hashlib
import json
import time
from pathlib import Path
from runtime import OUT, CODE, guard, sha, new_json, utc, peak_bytes, freeze_path
from spec import VERSION

CLEARANCE=OUT/'clearance/TD1_CLEARANCE_v1.json'
BINDINGS=['version','model_sha256','code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256']

def validate_clearance(clearance,freeze,freeze_hash):
    if clearance.get('schema')!='TD1_CLEARANCE_v1' or clearance.get('status')!='CLEARED':
        raise PermissionError('Clearance schema/status not CLEARED')
    if clearance.get('freeze_sha256')!=freeze_hash:
        raise PermissionError('Freeze hash does not match clearance')
    for key in BINDINGS:
        if key not in clearance or clearance[key]!=freeze[key]:
            raise PermissionError('Clearance binding mismatch: '+key)

def admit():
    guard()
    if not CLEARANCE.is_file():
        raise PermissionError('STOP: TD1_CLEARANCE_v1.json absent; no evaluation target computed')
    if (OUT/'EVALUATION_STARTED.json').exists() or (OUT/'EVALUATION_RESULT.json').exists():
        raise PermissionError('Evaluation already started; once-only runner refuses replay')
    clearance=json.loads(CLEARANCE.read_text())
    path=freeze_path(); freeze=json.loads(path.read_text())
    validate_clearance(clearance,freeze,sha(path))
    if {str(p) for p in CODE.glob('*.py')} != set(freeze['code_sha256']):
        raise PermissionError('Delivered Python inventory changed')
    for field in ['code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256']:
        for p,h in freeze[field].items():
            # Real path is bound to the intended source-code or private root.
            resolved=Path(p).resolve()
            root=CODE.resolve() if field=='code_sha256' else OUT.resolve()
            if not resolved.is_relative_to(root) or sha(resolved)!=h:
                raise PermissionError('Frozen artifact changed: '+p)
    if sha(OUT/'model_frozen.pkl')!=freeze['model_sha256']:
        raise PermissionError('Frozen model changed')
    receipt={'freeze_sha256':sha(path),'clearance_sha256':sha(CLEARANCE),
             'version':VERSION,'exploratory':True}
    receipt['utc']=utc(); new_json(OUT/'EVALUATION_STARTED.json',receipt)
    return freeze

def run():
    start=time.perf_counter()
    admit()  # Nothing below this line executes without matching clearance.
    import numpy as np
    import pandas as pd
    import sources as io
    from core import label, paired
    pairs=[]; coverage=[]
    for year in [2023,2024]:
        guard()
        d=io.dated(OUT/f'inputs/crsp_{year}.parquet',
                   ['date','permno','ret','share_type'],lo=f'{year}-01-01',hi=f'{year}-12-31')
        if year==2023:
            d=pd.concat([d,io.dated(OUT/'inputs/crsp_2024.parquet',
                ['date','permno','ret','share_type'],lo='2024-01-01',hi='2024-01-15')],ignore_index=True)
        d=d[d.share_type.isin([0,2])].copy(); d['date']=pd.to_datetime(d.date)
        dates=pd.DatetimeIndex(sorted(d.date.unique()))
        returns=d.pivot(index='date',columns='permno',values='ret').reindex(dates)
        idmap={hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]:p for p in returns.columns}
        pred=io.dated(OUT/f'predictions/year_{year}.parquet',
            ['firm_id','origin_session','cutoff','b','old_b'],date_col='origin_session',
            lo=f'{year}-01-01',hi=f'{year}-12-31')
        rows=[]
        for firm,g in pred.groupby('firm_id'):
            p=idmap.get(firm)
            if p is None: continue
            r=returns[p].to_numpy(float)
            for record in g.to_dict('records'):
                t=dates.get_indexer([record['origin_session']])[0]
                if t<0: continue
                y,end,valid=label(dates,r,t,'2024-12-31')
                if not valid: continue
                record.update(y=y,label_end=end,mlp_squared_loss=(y-record['b'])**2,
                              old_squared_loss=(y-record['old_b'])**2)
                rows.append(record)
        f=pd.DataFrame(rows); pairs.append(f)
        coverage.append({'year':year,'feature_predictions':len(pred),'paired_label_rows':len(f)})
    f=pd.concat(pairs,ignore_index=True)
    if (f.label_end>pd.Timestamp('2024-12-31')).any(): raise ValueError('Label date seal')
    result=paired(f.y.to_numpy(),f.b.to_numpy(),f.old_b.to_numpy(),f.firm_id.to_numpy(),f.origin_session.to_numpy())
    io.write_parquet(OUT/'evaluation/paired_losses_2023_2024.parquet',f)
    report={'version':VERSION,'exploratory':True,'paired':result,'coverage':coverage,
            'elapsed_seconds':time.perf_counter()-start,**peak_bytes(),'free_bytes':guard()}
    report['utc']=utc(); new_json(OUT/'EVALUATION_RESULT.json',report)
    print(json.dumps(report),flush=True)
