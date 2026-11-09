# TD1B correction handoff v2

The final HAR comparator copies the selection comparator's 2015-2020 medians, means and scales without fitting them again. Only HAR coefficients are refit on eligible development rows through 2022, retaining the intercept and numerical ridge of 1e-6. The MLPs, their preprocessing and target transform, source snapshots, manifests, selection artifacts and original delivery are preserved.

Active source: `D:/OptResearch_data/external/supply-chain-news-risk/five_session/v2/`. All numerical artifacts and receipts: `D:/OptResearch_data/five_session/`. Reviewed source files in the parent directory are immutable. Use the versioned runner below. The original model specification/version identifier is retained; artifact filenames and `artifact_revision` distinguish this repair.

The verifier independently reconstructs every HAR median, mean and scale from eligible 2015-2020 rows without calling `core.prepare_fit`, requires exact equality with `selection_old.preprocess`, and requires exact equality of final `old.preprocess` with that selection preprocessing. It independently reconstructs through-2022 HAR coefficients and requires exact equality. `v2/validate_correction.py` retains the extreme-selection/missingness regression, an unsafe-refit control and copied-array checks. Delivered tests and independent access traps retain their checks, with updated versioned paths/schema; correction tests exercise each corrupted statistic and the versioned gate.

## Exact commands

These commands use local installed dependencies. The constructor and repair commands were completed once and refuse existing targets. Do not rerun construction over this delivery. No `fit`, snapshot build, WRDS, provider, network or git command was run for the correction.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$td1Python = 'C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe'
$td1V2 = 'D:/OptResearch_data/external/supply-chain-news-risk/five_session/v2'

# Completed once; append-only source construction and comparator-only repair.
& $td1Python -B "$td1V2/setup_v2_1.py"
& $td1Python -B "$td1V2/run_td1.py" repair

# Completed; reruns append new private fixtures and receipts.
& $td1Python -B "$td1V2/run_td1.py" tests
& $td1Python -B "$td1V2/run_td1.py" correction-tests
& $td1Python -B "$td1V2/run_td1.py" traps
& $td1Python -B "$td1V2/run_td1.py" verify
```

`setup_v2_1.py` is the active source constructor. `runtime_safety_checked.py` handles Windows' serialized subprocess audit arguments and permits only the exact `date.exe -u` timestamp command during repair/tests/verification. Earlier constructor/guard source drafts are retained under the no-deletion/no-overwrite rule. The active runner applies task-only network/write/label seals during repair, tests and verification; later evaluation retains its separate hash-bound clearance gate.

## Artifacts and review boundary

| Item | Path |
| --- | --- |
| Corrected model bundle | `D:/OptResearch_data/five_session/model_frozen_v2.pkl` |
| Corrected predictions, all ten partitions | `D:/OptResearch_data/five_session/predictions_v2/year_2015.parquet` through `year_2024.parquet` |
| Active numerical freeze | `D:/OptResearch_data/five_session/MODEL_FREEZE_v3.json` |
| Active clearance request | `D:/OptResearch_data/five_session/CLEARANCE_REQUEST_v3.json` |
| Required independent clearance | `D:/OptResearch_data/five_session/clearance/TD1_CLEARANCE_v2.json` |
| Contract-verifier receipt | `D:/OptResearch_data/five_session/VERIFICATION_v2.json` |
| Original and MLP preservation hashes | `D:/OptResearch_data/five_session/repair_v2/PRESERVATION_v2.json` |
| Original baseline hashes | `D:/OptResearch_data/five_session/repair_v2/baseline.json` |
| Complete source diff | `D:/OptResearch_data/external/supply-chain-news-risk/five_session/TD1_SOURCE_DIFF_v2.patch` |
| Bound private source-diff copy | `D:/OptResearch_data/five_session/repair_v2/SOURCE_DIFF_v2.patch` |
| Test logs and independent-trap receipts | `D:/OptResearch_data/five_session/repair_v2/` |
| Private handoff and final integrity receipt | `D:/OptResearch_data/five_session/TD1_HANDOFF_v2.md`; `repair_v2/FINAL_INTEGRITY_v2.json` |

Only `old_b` changes in prediction frames; `b` and all other columns are preserved exactly. The preservation receipt records SHA256 for both MLP weights and complete serialized states before/after, per-partition `b` hashes, all preserved originals and both source-manifest files. The new freeze binds the full active Python inventory, model, all predictions, inputs, manifests, selection artifacts, source diff, preservation receipt and parent reviewed freeze. Original models, prediction partitions, freezes and requests remain available.

Clearance schema and filename are now `TD1_CLEARANCE_v2`. `CLEARANCE_REQUEST_v3.json` is a request template with `request_only: true`, never an issued clearance. No clearance or evaluation-start/result file was created. The evaluator now reads `MODEL_FREEZE_v3.json`, `model_frozen_v2.pkl` and `predictions_v2/`; it checks the full source inventory and all bindings before decoder/label access and retains the global once-only receipt. Independent re-review is required before later evaluation. No evaluation command was run.

Every Python process installs the 2,500,000,000-byte Windows process/job cap before scientific imports. D: free space is repeatedly checked against 40 GiB. Runtime artifacts stay under the private D: root; bytecode writes are disabled. No task file was intentionally created on C:. Nothing was deleted or overwritten. Protected intraday/IDS2/WRDS-lock paths and original mixed-period sources were not opened. Real data access used only bounded private snapshots and manifests. Invented future-date poison fixtures only test rejection. No real 2025 data or any 2023-2024 five-session label was read or computed. Timestamps come from running `date.exe -u` immediately before writing them.

The audit's separate repository-wide packaging finding remains outside this numerical correction: legacy data/reports elsewhere in the existing repository were preserved. The versioned subtree contains source and permitted documentation/patches, with numerical artifacts private. No clean-export or repository-wide packaging clearance is claimed.

## Numerical and verifier source diff

The full patch linked above also includes artifact paths, gate, tests and resource guard. The exact main correction is:

```diff
--- D:\OptResearch_data\external\supply-chain-news-risk\five_session\core.py
+++ D:\OptResearch_data\external\supply-chain-news-risk\five_session\v2\core.py
@@ -61,8 +61,9 @@
     # No winsorization or clipping estimated using selection/evaluation rows.
     return np.column_stack([z,missing.astype(float)]) if indicators else z
 
-def old_fit(x,y):
-    p=prepare_fit(x); z=np.column_stack([np.ones(len(x)),transform(x,p,False)])
+def old_fit(x,y,preprocess=None):
+    p=prepare_fit(x) if preprocess is None else {k:v.copy() for k,v in preprocess.items()}
+    z=np.column_stack([np.ones(len(x)),transform(x,p,False)])
     beta=np.linalg.solve(z.T@z+1e-6*np.diag(np.r_[0.,np.ones(x.shape[1])]),z.T@y)
     return {'preprocess':p,'beta':beta}
 
--- D:\OptResearch_data\external\supply-chain-news-risk\five_session\fit.py
+++ D:\OptResearch_data\external\supply-chain-news-risk\five_session\v2\fit.py
@@ -20,7 +20,7 @@
 
 def run():
     start=time.perf_counter(); guard()
-    if (OUT/'MODEL_FREEZE.json').exists(): raise FileExistsError('Freeze already exists')
+    if (OUT/'MODEL_FREEZE_v3.json').exists(): raise FileExistsError('Freeze already exists')
     cols=['firm_id','cutoff','origin_session']+FEATURES+['y','label_end','fit_included','selection_included','refit_included']
     f=pd.concat([io.dated(OUT/f'firm_week_manifest/year_{y}.parquet',cols,
         date_col='origin_session',lo=f'{y}-01-01',hi=f'{y}-12-31') for y in range(2015,2023)],ignore_index=True)
@@ -71,11 +71,11 @@
     xr=x[ref]; yr=target[ref]
     for epoch in range(selected['best_epoch']):
         guard(); final.partial_fit(xr,yr)
-    old_final=old_fit(f.loc[ref,OLD].to_numpy(float),y[ref])
+    old_final=old_fit(f.loc[ref,OLD].to_numpy(float),y[ref],preprocess=old['preprocess'])
     bundle={'version':VERSION,'features':FEATURES,'old_features':OLD,'preprocess':prep,
         'ymean':ymean,'yscale':yscale,'mlp':final,'old':old_final,
         'selection_mlp':chosen,'selection_old':old,'selected':selected}
-    with (OUT/'model_frozen.pkl').open('xb') as stream: pickle.dump(bundle,stream,protocol=5)
+    with (OUT/'model_frozen_v2.pkl').open('xb') as stream: pickle.dump(bundle,stream,protocol=5)
     predictions={}
     for year in range(2015,2025):
         guard()
@@ -88,7 +88,7 @@
         pred['old_b']=old_predict(v[OLD].to_numpy(float),old_final)
         pred['model_version']=VERSION
         pred['prediction_role']='refit_in_sample' if year<=2022 else 'features_only_exploratory'
-        path=OUT/f'predictions/year_{year}.parquet'; io.write_parquet(path,pred)
+        path=OUT/f'predictions_v2/year_{year}.parquet'; io.write_parquet(path,pred)
         predictions[str(path)]=sha(path)
     hashes={str(p):sha(p) for p in sorted(CODE.glob('*.py'))}
     manifests={str(p):sha(p) for p in sorted((OUT/'firm_week_manifest').glob('*.parquet'))}
@@ -97,19 +97,19 @@
     freeze={'version':VERSION,'exploratory':True,'architecture':[len(FEATURES)*2,64,32,1],
         'activation':'ReLU','loss':'squared_error','seed':SEED,'selected':selected,
         'rules':CLOCK_RULES,'training_selection_summary_sha256':sha(OUT/'SELECTION.json'),
-        'model_sha256':sha(OUT/'model_frozen.pkl'),'code_sha256':hashes,
+        'model_sha256':sha(OUT/'model_frozen_v2.pkl'),'code_sha256':hashes,
         'selection_sha256':selection_hashes,
         'manifest_sha256':manifests,'input_sha256':inputs,'prediction_sha256':predictions,
         'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
         'fit_rows':int(tr.sum()),'selection_rows':int(va.sum()),'refit_rows':int(ref.sum()),
         'elapsed_seconds':time.perf_counter()-start,**peak_bytes(),'free_bytes':guard(),
-        'evaluation_run':False,'clearance_expected':'D:/OptResearch_data/five_session/clearance/TD1_CLEARANCE_v1.json'}
-    freeze['utc']=utc(); new_json(OUT/'MODEL_FREEZE.json',freeze)
-    expected={'schema':'TD1_CLEARANCE_v1','status':'CLEARED','freeze_sha256':sha(OUT/'MODEL_FREEZE.json'),
+        'evaluation_run':False,'clearance_expected':'D:/OptResearch_data/five_session/clearance/TD1_CLEARANCE_v2.json'}
+    freeze['utc']=utc(); new_json(OUT/'MODEL_FREEZE_v3.json',freeze)
+    expected={'schema':'TD1_CLEARANCE_v2','status':'CLEARED','freeze_sha256':sha(OUT/'MODEL_FREEZE_v3.json'),
               'version':VERSION,'scope':'2023-2024 exploratory paired losses only; all label ends <=2024-12-31',
               'code_sha256':hashes,'model_sha256':freeze['model_sha256'],
               'selection_sha256':selection_hashes,
               'manifest_sha256':manifests,'input_sha256':inputs,'prediction_sha256':predictions}
     # This is a REQUEST in the root, never a self-issued clearance in clearance/.
-    new_json(OUT/'CLEARANCE_REQUEST.json',expected)
-    print('frozen',str(OUT/'MODEL_FREEZE.json'),'selection',json.dumps(result),flush=True)
+    new_json(OUT/'CLEARANCE_REQUEST_v3.json',expected)
+    print('frozen',str(OUT/'MODEL_FREEZE_v3.json'),'selection',json.dumps(result),flush=True)
--- D:\OptResearch_data\external\supply-chain-news-risk\five_session\verify.py
+++ D:\OptResearch_data\external\supply-chain-news-risk\five_session\v2\verify.py
@@ -5,7 +5,8 @@
 import pandas as pd
 import sources as io
 from runtime import OUT, CODE, guard, sha, new_json, utc, peak_bytes, freeze_path
-from spec import FEATURES
+from spec import FEATURES, OLD
+from validate_correction import assert_preprocessing
 from core import transform, old_predict, prepare_fit
 
 def run():
@@ -15,14 +16,14 @@
         for p,h in freeze[field].items():
             if sha(p)!=h: raise ValueError('Hash mismatch: '+p)
             checks+=1
-    if sha(OUT/'model_frozen.pkl')!=freeze['model_sha256']: raise ValueError('Model hash')
+    if sha(OUT/'model_frozen_v2.pkl')!=freeze['model_sha256']: raise ValueError('Model hash')
     checks+=1
     bundle=io.read_frozen_model()
     if bundle['features']!=FEATURES or bundle['version']!=freeze['version']:
         raise ValueError('Model version or feature order')
     if [v.shape for v in bundle['mlp'].coefs_]!=[(36,64),(64,32),(32,1)]:
         raise ValueError('Model architecture')
-    checks+=2; training=[]; training_y=[]; selection=[]
+    checks+=2; training=[]; training_y=[]; selection=[]; training_old=[]; refit_old=[]
     for year in range(2015,2025):
         guard(); path=OUT/f'firm_week_manifest/year_{year}.parquet'
         names=io.schema_names(path)
@@ -45,11 +46,13 @@
             if (f.loc[valid,'label_end']>pd.Timestamp('2022-12-31')).any(): raise ValueError('Development label boundary')
             if (f.loc[f.fit_included,'label_end']>pd.Timestamp('2020-12-31')).any(): raise ValueError('Fitting boundary')
             checks+=2
+            refit_old.append(f.loc[f.refit_included,OLD+['y']])
             if year<=2020:
+                training_old.append(f.loc[f.fit_included,OLD])
                 training.append(f.loc[f.fit_included,FEATURES].to_numpy(float))
                 training_y.append(f.loc[f.fit_included,'y'].to_numpy(float))
             else: selection.append(f.loc[f.selection_included,['firm_id','cutoff','origin_session','y']+FEATURES])
-        p=OUT/f'predictions/year_{year}.parquet'; pn=io.schema_names(p)
+        p=OUT/f'predictions_v2/year_{year}.parquet'; pn=io.schema_names(p)
         if any(c in pn for c in ['y','label_end','target','loss']): raise ValueError('Outcomes in prediction file')
         pr=io.dated(p,pn,date_col='origin_session',lo=f'{year}-01-01',hi=f'{year}-12-31')
         expected=f.loc[f.eligible_features,['firm_id','cutoff']]
@@ -75,6 +78,13 @@
     for k,v in actual.items():
         if not np.array_equal(v,bundle['preprocess'][k]): raise ValueError('Preprocessing was not fit on fitting rows only')
         checks+=1
+    # Independent statistics retain the fitter's column-major OLD layout.
+    checks+=assert_preprocessing(pd.concat(training_old,ignore_index=True)[OLD].to_numpy(float),bundle)
+    rf=pd.concat(refit_old,ignore_index=True)
+    from validate_correction import independent_beta
+    beta=independent_beta(rf[OLD].to_numpy(float),rf.y.to_numpy(float),bundle['selection_old']['preprocess'])
+    if not np.array_equal(beta,bundle['old']['beta']): raise ValueError('Comparator coefficients were not refit through 2022')
+    checks+=1
     ty=np.concatenate(training_y)
     if float(ty.mean())!=bundle['ymean'] or float(ty.std())!=bundle['yscale']:
         raise ValueError('Target transform was not fit on fitting rows only')
@@ -91,9 +101,9 @@
     if not np.allclose(se.old_squared_loss,(se.y-so)**2,rtol=1e-11,atol=1e-11): raise ValueError('Selection old loss mismatch')
     checks+=5
     report={'contract_checks_passed':checks,'manifest_rows':rows,'prediction_rows':predictions,
-        'evaluation_targets_computed':False,'evaluation_gate_file_exists':(OUT/'clearance/TD1_CLEARANCE_v1.json').exists(),
+        'evaluation_targets_computed':False,'evaluation_gate_file_exists':(OUT/'clearance/TD1_CLEARANCE_v2.json').exists(),
         **peak_bytes(),'free_bytes':guard()}
-    target=OUT/'VERIFICATION.json'
-    if target.exists(): target=OUT/f'VERIFICATION_{uuid.uuid4().hex}.json'
+    target=OUT/'VERIFICATION_v2.json'
+    if target.exists(): target=OUT/f'VERIFICATION_v2_{uuid.uuid4().hex}.json'
     report['utc']=utc(); new_json(target,report)
     print(json.dumps(report),flush=True)

```
