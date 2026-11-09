"""Refit only HAR coefficients; preserve MLP state and every reviewed artifact."""
import copy
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import sources as io
from core import old_fit, old_predict
from runtime import OUT, CODE, guard, sha, new_json, utc, peak_bytes
from spec import OLD
from validate_correction import assert_preprocessing, independent_beta, extreme_selection_missingness_trap

RECEIPTS=OUT/'repair_v2'

def object_hash(value):
    return hashlib.sha256(pickle.dumps(value,protocol=5)).hexdigest()

def weight_hash(model):
    h=hashlib.sha256()
    for name in ('coefs_', 'intercepts_'):
        for i,a in enumerate(getattr(model,name)):
            h.update(f'{name}/{i}/{a.dtype.str}/{a.shape}'.encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def check_originals():
    baseline=json.loads((RECEIPTS/'baseline.json').read_text())['sha256']
    for p,expected in baseline.items():
        if sha(p)!=expected: raise ValueError('Original artifact changed: '+p)
    return baseline

def run():
    guard()
    for p in (OUT/'model_frozen_v2.pkl', OUT/'predictions_v2', OUT/'MODEL_FREEZE_v3.json', OUT/'CLEARANCE_REQUEST_v3.json'):
        if p.exists(): raise FileExistsError('Preserve existing output: '+str(p))
    original_hashes=check_originals()
    parent=OUT/'MODEL_FREEZE_v2.json'; frozen=json.loads(parent.read_text())
    if sha(OUT/'model_frozen.pkl')!=frozen['model_sha256']: raise ValueError('Original model hash')
    with (OUT/'model_frozen.pkl').open('rb') as stream: original=pickle.load(stream)
    cols=OLD+['y','fit_included','refit_included','label_end','origin_session']
    f=pd.concat([io.dated(OUT/f'firm_week_manifest/year_{year}.parquet',cols,
        date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31') for year in range(2015,2023)],ignore_index=True)
    tr=f.fit_included.to_numpy(bool); ref=f.refit_included.to_numpy(bool)
    if not (f.loc[tr,'origin_session'].dt.year.between(2015,2020).all() and
            f.loc[tr,'label_end'].le(pd.Timestamp('2020-12-31')).all() and
            f.loc[ref,'label_end'].le(pd.Timestamp('2022-12-31')).all()):
        raise ValueError('Development split/date boundary')
    y=f.y.to_numpy(float)
    final=old_fit(f.loc[ref,OLD].to_numpy(float), y[ref], preprocess=original['selection_old']['preprocess'])
    corrected=dict(original,old=final)
    checks=assert_preprocessing(f.loc[tr,OLD].to_numpy(float),corrected)
    np.testing.assert_array_equal(final['beta'],independent_beta(f.loc[ref,OLD].to_numpy(float),y[ref],original['selection_old']['preprocess']))
    assert extreme_selection_missingness_trap()
    for key in original:
        if key!='old': assert object_hash(original[key])==object_hash(corrected[key]), key
    with (OUT/'model_frozen_v2.pkl').open('xb') as stream: pickle.dump(corrected,stream,protocol=5)
    with (OUT/'model_frozen_v2.pkl').open('rb') as stream: reloaded=pickle.load(stream)
    mlp_hashes={}
    for key in ('mlp','selection_mlp'):
        before=weight_hash(original[key]); after=weight_hash(reloaded[key])
        assert before==after
        assert object_hash(original[key])==object_hash(reloaded[key])
        mlp_hashes[key]={'original_weights_sha256':before,'corrected_weights_sha256':after,
            'original_complete_state_sha256':object_hash(original[key]),'corrected_complete_state_sha256':object_hash(reloaded[key])}
    predictions={}; partition_facts=[]
    for year in range(2015,2025):
        guard()
        path=OUT/f'firm_week_manifest/year_{year}.parquet'
        if year>=2023 and set(io.schema_names(path)) & {'y','label_end','label_observed','fit_included','selection_included','refit_included'}:
            raise ValueError('Evaluation label fields in manifest')
        v=io.dated(path,['firm_id','cutoff','origin_session','eligible_features']+OLD,
            date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
        v=v[v.eligible_features]
        before=io.dated(OUT/f'predictions/year_{year}.parquet',io.schema_names(OUT/f'predictions/year_{year}.parquet'),
            date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
        if set(before.columns)&{'y','label_end','target','loss'}: raise ValueError('Outcome in original predictions')
        pd.testing.assert_frame_equal(before[['firm_id','cutoff','origin_session']],v[['firm_id','cutoff','origin_session']].reset_index(drop=True))
        pred=before.copy(); pred['old_b']=old_predict(v[OLD].to_numpy(float),final)
        pd.testing.assert_frame_equal(before.drop(columns='old_b'),pred.drop(columns='old_b'),check_exact=True)
        dest=OUT/f'predictions_v2/year_{year}.parquet'; io.write_parquet(dest,pred)
        saved=io.dated(dest,list(pred.columns),date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
        pd.testing.assert_frame_equal(pred,saved,check_exact=True)
        b_before=hashlib.sha256(before.b.to_numpy().tobytes()).hexdigest()
        b_after=hashlib.sha256(saved.b.to_numpy().tobytes()).hexdigest(); assert b_before==b_after
        partition_facts.append({'year':year,'rows':len(saved),'old_b_changed_over_1e_8':int((np.abs(saved.old_b-before.old_b)>1e-8).sum()),
            'original_b_sha256':b_before,'corrected_b_sha256':b_after,'other_columns_exactly_preserved':True})
        predictions[str(dest)]=sha(dest)
        print('Recreated old_b partition',year,len(saved),flush=True)
    check_originals()
    preserved={'original_artifacts_sha256':original_hashes,'original_artifact_count':len(original_hashes),
        'mlp_hashes':mlp_hashes,'partitions':partition_facts,
        'fit_rows':int(tr.sum()),'refit_rows':int(ref.sum()),'independent_preprocessing_assertions':checks,
        'extreme_selection_missingness_trap_passed':True,'all_non_old_bundle_fields_preserved_by_hash':True,
        'source_manifests_sha256':{str(OUT/n):sha(OUT/n) for n in ('SOURCE_SNAPSHOTS.json','MANIFEST_BUILD.json')},
        'evaluation_labels_read_or_computed':False,**peak_bytes(),'free_bytes':guard()}
    new_json(RECEIPTS/'PRESERVATION_v2.json',preserved)
    freeze=copy.deepcopy(frozen)
    freeze.update(parent_freeze_path=str(parent),parent_freeze_sha256=sha(parent),
        artifact_revision='TD1B-comparator-preprocessing-v2',model_path=str(OUT/'model_frozen_v2.pkl'),
        model_sha256=sha(OUT/'model_frozen_v2.pkl'),code_sha256={str(p):sha(p) for p in sorted(CODE.glob('*.py'))},
        prediction_sha256=predictions,clearance_expected=str(OUT/'clearance/TD1_CLEARANCE_v2.json'),
        reviewed_code_sha256=frozen['code_sha256'],preservation_sha256=sha(RECEIPTS/'PRESERVATION_v2.json'),
        source_diff_sha256=sha(RECEIPTS/'SOURCE_DIFF_v2.patch'),evaluation_run=False,
        amendment='HAR refit coefficients through 2022 with copied selection HAR preprocessing fitted only on 2015-2020. MLPs and source manifests unchanged.')
    # Do not carry old fitting-stage resource measurements into this repair.
    for key in ('utc','elapsed_seconds','peak_working_set_bytes','peak_private_bytes','free_bytes'):
        freeze.pop(key,None)
    freeze['utc']=utc(); new_json(OUT/'MODEL_FREEZE_v3.json',freeze)
    request={'schema':'TD1_CLEARANCE_v2','status':'CLEARED','request_only':True,
        'freeze_sha256':sha(OUT/'MODEL_FREEZE_v3.json'),
        'scope':'2023-2024 exploratory paired losses only; all label ends <=2024-12-31'}
    for key in ('version','model_sha256','code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256'):
        request[key]=freeze[key]
    new_json(OUT/'CLEARANCE_REQUEST_v3.json',request)
    print('MODEL:',OUT/'model_frozen_v2.pkl',flush=True)
    print('FREEZE:',OUT/'MODEL_FREEZE_v3.json',flush=True)
    print('REQUEST:',OUT/'CLEARANCE_REQUEST_v3.json',flush=True)
