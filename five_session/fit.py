"""2015-2020 fit, 2021-2022 selection, through-2022 refit, append-only freeze."""
import copy
import json
import pickle
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
from runtime import CODE, OUT, guard, new_json, peak_bytes, sha, utc
import sources as io
from spec import VERSION, FEATURES, OLD, GRID, MAX_EPOCHS, PATIENCE, MIN_DELTA, SEED, CLOCK_RULES
from core import prepare_fit, transform, old_fit, old_predict, paired

def model(lr,alpha):
    return MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',solver='adam',
        alpha=alpha,batch_size=4096,learning_rate_init=lr,random_state=SEED,
        max_iter=1,shuffle=True,early_stopping=False,tol=0.)

def run():
    start=time.perf_counter(); guard()
    if (OUT/'MODEL_FREEZE.json').exists(): raise FileExistsError('Freeze already exists')
    cols=['firm_id','cutoff','origin_session']+FEATURES+['y','label_end','fit_included','selection_included','refit_included']
    f=pd.concat([io.dated(OUT/f'firm_week_manifest/year_{y}.parquet',cols,
        date_col='origin_session',lo=f'{y}-01-01',hi=f'{y}-12-31') for y in range(2015,2023)],ignore_index=True)
    tr=f.fit_included.to_numpy(bool); va=f.selection_included.to_numpy(bool); ref=f.refit_included.to_numpy(bool)
    if not tr.any() or not va.any(): raise ValueError('Empty fitting or selection split')
    if (f.loc[tr,'label_end']>pd.Timestamp('2020-12-31')).any(): raise ValueError('Training label crosses selection')
    if (f.loc[ref,'label_end']>pd.Timestamp('2022-12-31')).any(): raise ValueError('Refit label crosses evaluation')
    raw=f[FEATURES].to_numpy(float); y=f.y.to_numpy(float)
    prep=prepare_fit(raw[tr]); x=transform(raw,prep)
    ymean=float(y[tr].mean()); yscale=float(y[tr].std()) or 1.
    target=(y-ymean)/yscale
    xtr,xva=x[tr],x[va]; ytr=target[tr]; yva=y[va]
    selected=None; histories=[]
    for lr,alpha in GRID:
        guard(); m=model(lr,alpha); best=np.inf; best_epoch=None; best_model=None; misses=0; history=[]
        for epoch in range(1,MAX_EPOCHS+1):
            guard(); m.partial_fit(xtr,ytr)
            pv=m.predict(xva)*yscale+ymean
            loss=float(np.mean((yva-pv)**2)); history.append({'epoch':epoch,'selection_mse':loss})
            # Actual minimum retained; patience reset requires a material improvement.
            oldbest=best
            if loss<best:
                best=loss; best_epoch=epoch; best_model=copy.deepcopy(m)
            misses=0 if loss<oldbest-MIN_DELTA else misses+1
            if epoch%10==0:
                print('select',lr,alpha,'epoch',epoch,'mse',loss,flush=True)
            if misses>=PATIENCE: break
        record={'lr':lr,'alpha':alpha,'best_epoch':best_epoch,'selection_mse':best,'history':history}
        histories.append(record)
        if selected is None or best<selected['selection_mse']:
            selected={k:v for k,v in record.items() if k!='history'}; chosen=best_model
        print('candidate',lr,alpha,best,best_epoch,flush=True)
    old=old_fit(f.loc[tr,OLD].to_numpy(float),y[tr])
    selection_pred=chosen.predict(xva)*yscale+ymean
    selection_old=old_predict(f.loc[va,OLD].to_numpy(float),old)
    result=paired(yva,selection_pred,selection_old,f.loc[va,'firm_id'].to_numpy(),f.loc[va,'origin_session'].to_numpy())
    select=f.loc[va,['firm_id','cutoff','origin_session','y']].copy()
    select['b_selection']=selection_pred; select['old_selection']=selection_old
    select['mlp_squared_loss']=(yva-selection_pred)**2; select['old_squared_loss']=(yva-selection_old)**2
    io.write_parquet(OUT/'selection/paired_predictions_2021_2022.parquet',select)
    summary={'version':VERSION,'selected':selected,'grid':histories,'paired_selection':result,
        'selection_is_used_for_tuning':True,'fit_rows':int(tr.sum()),'selection_rows':int(va.sum()),
        'refit_rows':int(ref.sum()),'preprocessing_fit_period':'2015-2020, label end by 2020-12-31'}
    summary['utc']=utc(); new_json(OUT/'SELECTION.json',summary)
    # Refit network only, exact selected number of epochs; frozen training-only
    # imputer, feature scaler and target transform are reused without refitting.
    final=model(selected['lr'],selected['alpha'])
    xr=x[ref]; yr=target[ref]
    for epoch in range(selected['best_epoch']):
        guard(); final.partial_fit(xr,yr)
    old_final=old_fit(f.loc[ref,OLD].to_numpy(float),y[ref])
    bundle={'version':VERSION,'features':FEATURES,'old_features':OLD,'preprocess':prep,
        'ymean':ymean,'yscale':yscale,'mlp':final,'old':old_final,
        'selection_mlp':chosen,'selection_old':old,'selected':selected}
    with (OUT/'model_frozen.pkl').open('xb') as stream: pickle.dump(bundle,stream,protocol=5)
    predictions={}
    for year in range(2015,2025):
        guard()
        p=OUT/f'firm_week_manifest/year_{year}.parquet'
        v=io.dated(p,['firm_id','cutoff','origin_session','eligible_features']+FEATURES,
                   date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
        v=v[v.eligible_features].copy()
        pred=v[['firm_id','cutoff','origin_session']].copy()
        pred['b']=final.predict(transform(v[FEATURES].to_numpy(float),prep))*yscale+ymean
        pred['old_b']=old_predict(v[OLD].to_numpy(float),old_final)
        pred['model_version']=VERSION
        pred['prediction_role']='refit_in_sample' if year<=2022 else 'features_only_exploratory'
        path=OUT/f'predictions/year_{year}.parquet'; io.write_parquet(path,pred)
        predictions[str(path)]=sha(path)
    hashes={str(p):sha(p) for p in sorted(CODE.glob('*.py'))}
    manifests={str(p):sha(p) for p in sorted((OUT/'firm_week_manifest').glob('*.parquet'))}
    inputs=json.loads((OUT/'SOURCE_SNAPSHOTS.json').read_text())['snapshot_sha256']
    selection_hashes={str(p):sha(p) for p in [OUT/'SELECTION.json',OUT/'selection/paired_predictions_2021_2022.parquet']}
    freeze={'version':VERSION,'exploratory':True,'architecture':[len(FEATURES)*2,64,32,1],
        'activation':'ReLU','loss':'squared_error','seed':SEED,'selected':selected,
        'rules':CLOCK_RULES,'training_selection_summary_sha256':sha(OUT/'SELECTION.json'),
        'model_sha256':sha(OUT/'model_frozen.pkl'),'code_sha256':hashes,
        'selection_sha256':selection_hashes,
        'manifest_sha256':manifests,'input_sha256':inputs,'prediction_sha256':predictions,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'fit_rows':int(tr.sum()),'selection_rows':int(va.sum()),'refit_rows':int(ref.sum()),
        'elapsed_seconds':time.perf_counter()-start,**peak_bytes(),'free_bytes':guard(),
        'evaluation_run':False,'clearance_expected':'D:/OptResearch_data/five_session/clearance/TD1_CLEARANCE_v1.json'}
    freeze['utc']=utc(); new_json(OUT/'MODEL_FREEZE.json',freeze)
    expected={'schema':'TD1_CLEARANCE_v1','status':'CLEARED','freeze_sha256':sha(OUT/'MODEL_FREEZE.json'),
              'version':VERSION,'scope':'2023-2024 exploratory paired losses only; all label ends <=2024-12-31',
              'code_sha256':hashes,'model_sha256':freeze['model_sha256'],
              'selection_sha256':selection_hashes,
              'manifest_sha256':manifests,'input_sha256':inputs,'prediction_sha256':predictions}
    # This is a REQUEST in the root, never a self-issued clearance in clearance/.
    new_json(OUT/'CLEARANCE_REQUEST.json',expected)
    print('frozen',str(OUT/'MODEL_FREEZE.json'),'selection',json.dumps(result),flush=True)
