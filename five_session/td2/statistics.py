"""Paired squared-error gains with two-way firm/calendar-week clustering."""
import numpy as np
import pandas as pd

def clustered_mean(values,firms,dates):
    x=np.asarray(values,float); n=len(x)
    if n<2 or not np.isfinite(x).all(): raise ValueError('Invalid paired observations')
    weeks=pd.DatetimeIndex(pd.to_datetime(dates)).to_period('W-SUN').astype(str)
    f=pd.DataFrame({'u':x-x.mean(),'firm':firms,'week':weeks})
    def term(cols):
        sums=f.groupby(cols).u.sum().to_numpy(); g=len(sums)
        return (float(sums@sums)*g/(g-1) if g>1 else None),g
    a,gf=term(['firm']); b,gw=term(['week']); c,gx=term(['firm','week'])
    var=None if None in (a,b,c) else (a+b-c)/n**2
    # A negative inclusion-exclusion estimate is reported unsupported, never clipped to zero.
    se=float(np.sqrt(var)) if var is not None and var>=0 else None
    return {'n':n,'mean':float(x.mean()),'variance':var,'se':se,
        'clusters_firm':gf,'clusters_calendar_week':gw,'clusters_intersection':gx,
        'uncertainty_status':'SUPPORTED' if se is not None else 'UNSUPPORTED',
        'normal_95_interval':None if se is None else [float(x.mean()-1.96*se),float(x.mean()+1.96*se)]}

def paired(y,baseline,pred,firms,dates):
    y=np.asarray(y,float); baseline=np.asarray(baseline,float); pred=np.asarray(pred,float)
    if not (np.isfinite(y).all() and np.isfinite(baseline).all() and np.isfinite(pred).all()):
        raise ValueError('Nonfinite paired rows')
    base=(y-baseline)**2; arm=(y-pred)**2
    result=clustered_mean(base-arm,firms,dates)
    result.update(baseline_mse=float(base.mean()),arm_mse=float(arm.mean()),
        relative_gain=None if base.mean()==0 else float((base.mean()-arm.mean())/base.mean()),
        sign='positive gain means lower squared error than frozen firm-only MLP',exploratory=True)
    return result
