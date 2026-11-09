"""Post-fit contract verification; evaluation outcomes are never projected."""
import json
import uuid
import numpy as np
import pandas as pd
import sources as io
from runtime import OUT, CODE, guard, sha, new_json, utc, peak_bytes, freeze_path
from spec import FEATURES, OLD
from validate_correction import assert_preprocessing
from core import transform, old_predict, prepare_fit

def run():
    checks=0; rows=0; predictions=0
    freeze=json.loads(freeze_path().read_text())
    for field in ['code_sha256','input_sha256','manifest_sha256','prediction_sha256','selection_sha256']:
        for p,h in freeze[field].items():
            if sha(p)!=h: raise ValueError('Hash mismatch: '+p)
            checks+=1
    if sha(OUT/'model_frozen_v2.pkl')!=freeze['model_sha256']: raise ValueError('Model hash')
    checks+=1
    bundle=io.read_frozen_model()
    if bundle['features']!=FEATURES or bundle['version']!=freeze['version']:
        raise ValueError('Model version or feature order')
    if [v.shape for v in bundle['mlp'].coefs_]!=[(36,64),(64,32),(32,1)]:
        raise ValueError('Model architecture')
    checks+=2; training=[]; training_y=[]; selection=[]; training_old=[]; refit_old=[]
    for year in range(2015,2025):
        guard(); path=OUT/f'firm_week_manifest/year_{year}.parquet'
        names=io.schema_names(path)
        if year>=2023 and any(c in names for c in ['y','label_end','label_observed','fit_included','selection_included','refit_included']):
            raise ValueError('Evaluation outcomes in feature partition')
        f=io.dated(path,names,date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
        if f.duplicated(['firm_id','cutoff']).any(): raise ValueError('Duplicate firm-week')
        checks+=2; rows+=len(f)
        for c in FEATURES:
            finite=np.isfinite(f[c].to_numpy(float))
            if not np.array_equal(~finite,f[c+'__missing'].to_numpy(bool)):
                raise ValueError('Missing flag mismatch '+c)
            a=f[c+'__available_at']
            if (a[finite]>=f.loc[finite,'cutoff']).any() or a[finite].isna().any():
                raise ValueError('Future/unknown feature '+c)
            if not a[~finite].isna().all(): raise ValueError('Missing feature has availability '+c)
            checks+=3
        if year<=2022:
            valid=f.y.notna()
            if (f.loc[valid,'label_end']>pd.Timestamp('2022-12-31')).any(): raise ValueError('Development label boundary')
            if (f.loc[f.fit_included,'label_end']>pd.Timestamp('2020-12-31')).any(): raise ValueError('Fitting boundary')
            checks+=2
            refit_old.append(f.loc[f.refit_included,OLD+['y']])
            if year<=2020:
                training_old.append(f.loc[f.fit_included,OLD])
                training.append(f.loc[f.fit_included,FEATURES].to_numpy(float))
                training_y.append(f.loc[f.fit_included,'y'].to_numpy(float))
            else: selection.append(f.loc[f.selection_included,['firm_id','cutoff','origin_session','y']+FEATURES])
        p=OUT/f'predictions_v2/year_{year}.parquet'; pn=io.schema_names(p)
        if any(c in pn for c in ['y','label_end','target','loss']): raise ValueError('Outcomes in prediction file')
        pr=io.dated(p,pn,date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
        expected=f.loc[f.eligible_features,['firm_id','cutoff']]
        if len(pr)!=len(expected) or len(pr.merge(expected,on=['firm_id','cutoff'],validate='one_to_one'))!=len(expected):
            raise ValueError('Prediction/manifest row mismatch')
        if not np.isfinite(pr[['b','old_b']].to_numpy(float)).all(): raise ValueError('Nonfinite prediction')
        if not pr.model_version.eq(bundle['version']).all(): raise ValueError('Prediction model version')
        role='refit_in_sample' if year<=2022 else 'features_only_exploratory'
        if not pr.prediction_role.eq(role).all(): raise ValueError('Prediction role')
        checks+=2
        eligible=f[f.eligible_features]
        if not eligible[['firm_id','cutoff']].reset_index(drop=True).equals(pr[['firm_id','cutoff']].reset_index(drop=True)):
            raise ValueError('Prediction order differs from feature order')
        b=bundle['mlp'].predict(transform(eligible[FEATURES].to_numpy(float),bundle['preprocess']))*bundle['yscale']+bundle['ymean']
        ob=old_predict(eligible[bundle['old_features']].to_numpy(float),bundle['old'])
        if not np.allclose(pr.b.to_numpy(),b,rtol=1e-11,atol=1e-11): raise ValueError('Saved MLP prediction mismatch')
        if not np.allclose(pr.old_b.to_numpy(),ob,rtol=1e-11,atol=1e-11): raise ValueError('Saved old prediction mismatch')
        checks+=3
        predictions+=len(pr); checks+=3
    # The fitter's boolean row selection produces C-contiguous arrays. Match
    # that layout to keep reduction order and the exact equality check intact.
    actual=prepare_fit(np.ascontiguousarray(np.concatenate(training)))
    for k,v in actual.items():
        if not np.array_equal(v,bundle['preprocess'][k]): raise ValueError('Preprocessing was not fit on fitting rows only')
        checks+=1
    # Independent statistics retain the fitter's column-major OLD layout.
    checks+=assert_preprocessing(pd.concat(training_old,ignore_index=True)[OLD].to_numpy(float),bundle)
    rf=pd.concat(refit_old,ignore_index=True)
    from validate_correction import independent_beta
    beta=independent_beta(rf[OLD].to_numpy(float),rf.y.to_numpy(float),bundle['selection_old']['preprocess'])
    if not np.array_equal(beta,bundle['old']['beta']): raise ValueError('Comparator coefficients were not refit through 2022')
    checks+=1
    ty=np.concatenate(training_y)
    if float(ty.mean())!=bundle['ymean'] or float(ty.std())!=bundle['yscale']:
        raise ValueError('Target transform was not fit on fitting rows only')
    checks+=2
    sf=pd.concat(selection,ignore_index=True)
    path=OUT/'selection/paired_predictions_2021_2022.parquet'
    se=io.dated(path,io.schema_names(path),date_col='origin_session',lo='2021-01-01',hi='2022-12-31')
    if not sf[['firm_id','cutoff','y']].equals(se[['firm_id','cutoff','y']]): raise ValueError('Selection row/target mismatch')
    sp=bundle['selection_mlp'].predict(transform(sf[FEATURES].to_numpy(float),bundle['preprocess']))*bundle['yscale']+bundle['ymean']
    so=old_predict(sf[bundle['old_features']].to_numpy(float),bundle['selection_old'])
    if not np.allclose(se.b_selection,sp,rtol=1e-11,atol=1e-11): raise ValueError('Selection MLP mismatch')
    if not np.allclose(se.old_selection,so,rtol=1e-11,atol=1e-11): raise ValueError('Selection old baseline mismatch')
    if not np.allclose(se.mlp_squared_loss,(se.y-sp)**2,rtol=1e-11,atol=1e-11): raise ValueError('Selection loss mismatch')
    if not np.allclose(se.old_squared_loss,(se.y-so)**2,rtol=1e-11,atol=1e-11): raise ValueError('Selection old loss mismatch')
    checks+=5
    report={'contract_checks_passed':checks,'manifest_rows':rows,'prediction_rows':predictions,
        'evaluation_targets_computed':False,'evaluation_gate_file_exists':(OUT/'clearance/TD1_CLEARANCE_v2.json').exists(),
        **peak_bytes(),'free_bytes':guard()}
    target=OUT/'VERIFICATION_v2.json'
    if target.exists(): target=OUT/f'VERIFICATION_v2_{uuid.uuid4().hex}.json'
    report['utc']=utc(); new_json(target,report)
    print(json.dumps(report),flush=True)
