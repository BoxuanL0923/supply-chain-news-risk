"""Pure availability, feature, target, and preprocessing functions."""
import numpy as np
import pandas as pd
from spec import FEATURES, OLD

def clock(day, at='23:59:59'):
    return (pd.Timestamp(day).normalize()+pd.Timedelta(at)).tz_localize(
        'America/New_York').tz_convert('UTC').tz_localize(None)

def require_past(available, cutoff, kind='feature'):
    a = pd.to_datetime(available)
    if pd.isna(a).any() or (a >= pd.Timestamp(cutoff)).any():
        raise ValueError(f'Future or unknown availability: {kind}')

def gate_values(values, availability, cutoff, choice):
    if choice == 'none':
        return {k: 0.0 for k in values}
    if choice not in values:
        raise ValueError('Invalid characteristic')
    if not np.isfinite(values[choice]):
        return {k:0.0 for k in values}
    require_past([availability[choice]], cutoff)
    return {k: float(v) if k == choice and np.isfinite(v) else 0.0 for k,v in values.items()}

def own_inputs(dates, returns, me, pos):
    """Feature function never touches positions after the origin."""
    if pos < 21:
        return {k: np.nan for k in ['rv1','rv5','rv22','ret1','ret5','ret22','own_downside5','size']}
    past = np.asarray(returns[max(0,pos-21):pos+1],dtype=float)
    result = {}
    for n in [1,5,22]:
        w = past[-n:]
        valid = len(w) == n and np.isfinite(w).all()
        result[f'rv{n}'] = float(np.log(max(np.mean(w*w),1e-8))) if valid else np.nan
        result[f'ret{n}'] = float(np.prod(1+w)-1) if valid else np.nan
    result['own_downside5'] = float(np.maximum(-past[-5:],0).mean()) if np.isfinite(past[-5:]).all() else np.nan
    result['size'] = float(np.log(me[pos]*1e6)) if np.isfinite(me[pos]) and me[pos]>0 else np.nan
    return result

def label(dates, returns, pos, allowed_end):
    """Call only for development, or behind the evaluation clearance gate."""
    if pos+5 >= len(dates):
        return np.nan, pd.NaT, False
    end = pd.Timestamp(dates[pos+5])
    if end > pd.Timestamp(min(allowed_end,'2024-12-31')):
        return np.nan, end, False  # do not inspect the future-return fields
    w = np.asarray(returns[pos+1:pos+6],float)
    valid = len(w)==5 and np.isfinite(w).all() and np.mean(w*w)>0
    return (float(np.log(np.mean(w*w))) if valid else np.nan), end, bool(valid)

def prepare_fit(x):
    x = np.asarray(x,float)
    med = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0 for v in x.T])
    z = np.where(np.isfinite(x),x,med)
    mu = z.mean(0); sd = z.std(0); sd[sd==0]=1
    return {'median':med,'mean':mu,'scale':sd}

def transform(x, p, indicators=True):
    x=np.asarray(x,float); missing=~np.isfinite(x)
    z=(np.where(missing,p['median'],x)-p['mean'])/p['scale']
    # No winsorization or clipping estimated using selection/evaluation rows.
    return np.column_stack([z,missing.astype(float)]) if indicators else z

def old_fit(x,y):
    p=prepare_fit(x); z=np.column_stack([np.ones(len(x)),transform(x,p,False)])
    beta=np.linalg.solve(z.T@z+1e-6*np.diag(np.r_[0.,np.ones(x.shape[1])]),z.T@y)
    return {'preprocess':p,'beta':beta}

def old_predict(x,m):
    return np.column_stack([np.ones(len(x)),transform(x,m['preprocess'],False)])@m['beta']

def paired(y,p,q,firms,weeks):
    if not all(np.isfinite(v).all() for v in [y,p,q]):
        raise ValueError('Nonfinite paired rows')
    a=(y-p)**2; b=(y-q)**2; d=a-b; n=len(d)
    frame=pd.DataFrame({'d':d-d.mean(),'firm':firms,'week':weeks})
    def meat(cols):
        u=frame.groupby(cols).d.sum().to_numpy(); g=len(u)
        return (g/(g-1))*float(u@u) if g>1 else 0.0
    variance=(meat(['firm'])+meat(['week'])-meat(['firm','week']))/(n*n)
    return {'n':n,'mlp_mse':float(a.mean()),'old_mse':float(b.mean()),
            'delta_mse_mlp_minus_old':float(d.mean()),'relative_gain':float(1-a.mean()/b.mean()),
            'two_way_firm_week_se':float(np.sqrt(max(0,variance))),
            'two_way_raw_variance':float(variance),'exploratory':True}
