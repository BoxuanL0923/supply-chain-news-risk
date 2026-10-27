"""Model-independent nested predictive regressions and sensitivity/power."""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from .config import CONTROLS

TERMS=['intercept',*CONTROLS,'OWN','H1','H2_minus_H1','H3_minus_H2']

def design(frame, depth=3):
    x=np.c_[np.ones(len(frame)),frame[list(CONTROLS)].to_numpy(float),frame.OWN.to_numpy(float)]
    columns=[frame.H1.to_numpy(float),(frame.H2-frame.H1).to_numpy(float),(frame.H3-frame.H2).to_numpy(float)]
    return np.c_[x,*columns[:depth]] if depth else x

def rank(x):
    scales=np.linalg.norm(x,axis=0); scales[scales==0]=1
    return np.linalg.matrix_rank(x/scales)

def regress(frame, train=None):
    y=frame.y.to_numpy(float); x=design(frame)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite regression data')
    model=sm.OLS(y,x).fit(); r=int(rank(x)); g=int(frame.date.nunique())
    hc=model.get_robustcov_results(cov_type='HC3',use_t=True)
    cluster=model.get_robustcov_results(cov_type='cluster',groups=frame.date.to_numpy(),use_correction=True,df_correction=True,use_t=True)
    # statsmodels CR1 uses the number of columns, including redundant controls.
    # Use effective design rank in its residual degrees-of-freedom correction.
    cluster_rank_scale=math.sqrt((len(frame)-x.shape[1])/(len(frame)-r))
    coefficients={}
    for j,name in enumerate(TERMS):
        identified=rank(x)>rank(np.delete(x,j,axis=1))
        row={'identified':bool(identified),'estimate':float(model.params[j]) if identified else None}
        for label,result,df in [('HC3',hc,len(frame)-r),('date_cluster',cluster,g-1)]:
            se=float(result.bse[j]) if identified else None
            if label=='date_cluster' and se is not None:se*=cluster_rank_scale
            if se is not None and (not math.isfinite(se) or se<0):se=None
            half=float(stats.t.ppf(.95,df)*se) if se is not None else None
            row[label]={'se':se,'df':df,'ci90':[row['estimate']-half,row['estimate']+half] if half is not None else None}
        coefficients[name]=row
    nested=[]
    for depth in range(4):
        xx=design(frame,depth); fit=sm.OLS(y,xx).fit()
        entry={'depth':depth,'rank':int(rank(xx)),'r2':float(fit.rsquared),'n':len(frame)}
        if train is not None:
            tx=design(train,depth); tfit=sm.OLS(train.y.to_numpy(float),tx).fit()
            prediction=xx@tfit.params
            entry.update(train_rank=int(rank(tx)),valid_mse=float(np.mean((y-prediction)**2)),
                         train_coefficients=dict(zip(TERMS[:tx.shape[1]],tfit.params.tolist())))
            if depth:
                previous=design(frame,depth-1)@sm.OLS(train.y.to_numpy(float),design(train,depth-1)).fit().params
                entry['valid_mse_gain_vs_previous']=float(np.mean((y-previous)**2)-entry['valid_mse'])
        nested.append(entry)
    g3=coefficients['H3_minus_H2']
    bound={kind:(max(abs(v) for v in g3[kind]['ci90']) if g3[kind]['ci90'] is not None else None) for kind in ('HC3','date_cluster')}
    # This is sensitivity, not an outcome-selected equivalence test. Strict TOST
    # equivalence would require a scientifically chosen margin larger than bound.
    z=float(stats.norm.ppf(.95)+stats.norm.ppf(.8))
    power=[]
    residual=x[:,-1]-x[:,:-1]@np.linalg.lstsq(x[:,:-1],x[:,-1],rcond=None)[0]
    sd_inc=float(np.sqrt(np.mean(residual**2)))
    sd_y=float(np.std(y,ddof=1))
    for effect in (.1,.25,.5,1.0):
        row={'gamma3_effect':effect,'rms_log_risk_effect':effect*sd_inc}
        for kind in ('HC3','date_cluster'):
            se=g3[kind]['se']
            needed=math.ceil(len(frame)*(z*se/effect)**2) if g3['identified'] and se is not None else None
            if kind=='date_cluster' and needed is not None: needed=int(50*math.ceil(needed/50))
            row[kind+'_cases_for_80pct_power']=needed
        power.append(row)
    return {'n':len(frame),'dates':g,'rank':r,'columns':len(TERMS),'r2':float(model.rsquared),
            'cluster_CR1_correction':'G/(G-1) * (n-1)/(n-effective_design_rank)',
            'coefficients':coefficients,'nested':nested,'gamma3_smallest_symmetric_equivalence_bound':bound,
            'power':{'alpha':.10,'two_sided':True,'power':.80,'partial_increment_sd':sd_inc,'y_sd':sd_y,'grid':power,
                     'method':'Normal approximation n_new=n*((z_.95+z_.80)*SE/effect)^2. Cluster version requires new independent dates with 50 cases/date, not more firms on the same dates.',
                     'unidentified':not g3['identified']},
            'ci_warning':'Only six pooled date clusters (four train, two valid); clustered t intervals and extrapolated power are unstable and exploratory.'}

def analyze(frame):
    train=frame[frame['split']=='train'];valid=frame[frame['split']=='valid']
    if len(train)!=200 or len(valid)!=100: raise ValueError('Matched 200/100 required')
    if max(train.date)>=min(valid.date): raise ValueError('Time split overlap')
    arms=('OWN','H1','H2','H3','H3_GRAPH')
    validation=regress(valid,train)
    return {'raw_residual_mse':{s:{a:float(np.mean((f.y-f.anchor-f[a])**2)) for a in arms}
                              for s,f in [('train',train),('valid',valid)]},
            'zero_score_mse':{s:float(np.mean((f.y-f.anchor)**2)) for s,f in [('train',train),('valid',valid)]},
            'train_fitted_valid':{'train_n':len(train),'valid_n':len(valid),'nested':validation['nested'],
                'estimation':'Nested train_coefficients fitted on train dates only; valid_mse evaluated on validation dates without refitting. Pooled coefficients are reported separately.'},
            'pooled_train_valid':regress(frame),
            'train':regress(train)}
