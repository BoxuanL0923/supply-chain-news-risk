"""Create date-sealed source snapshots and one manifest partition per origin year."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import OUT, guard, new_json, sha, utc, peak_bytes
import sources as io
from core import clock, own_inputs, label, require_past
from spec import FEATURES, CLOCK_RULES, VERSION

INPUTS=OUT/'inputs'
MANIFEST=OUT/'firm_week_manifest'

def cached(name, build_fn):
    path=INPUTS/name
    if path.exists():
        raise FileExistsError(path)
    f=build_fn(); io.write_parquet(path,f); del f; gc.collect()

def snapshots():
    start=time.perf_counter(); guard()
    cached('panel_keys.parquet',lambda: io.dated(io.PANEL,['date','permno'],lo='2015-01-01'))
    cached('identity.parquet',io.links)
    cached('quarters.parquet',lambda: io.csv_dated(io.QUARTER,
        ['gvkey','datadate','atq','cheq','invtq','dlcq','dlttq']))
    cached('earnings.parquet',lambda: io.csv_dated(io.EARNINGS,
        ['gvkey','datadate','rdq'],second_date='rdq'))
    for year in range(2014,2025):
        guard(); print('snapshot',year,flush=True)
        cached(f'crsp_{year}.parquet',lambda y=year: io.dated(io.CRSP,
            ['date','permno','ret','me','ff49','share_type'],lo=f'{y}-01-01',hi=f'{y}-12-31'))
        cached(f'cashflow_{year}.parquet',lambda y=year: io.dated(io.JKP,
            ['eom','permno','ocf_at','common','primary_sec'],date_col='eom',
            lo=f'{y}-01-01',hi=f'{y}-12-31'))
    hashes={str(p):sha(p) for p in sorted(INPUTS.glob('*.parquet'))}
    report={'version':VERSION,'source_paths':[str(p) for p in io.SOURCE_PATHS],
            'snapshot_sha256':hashes,'elapsed_seconds':time.perf_counter()-start,
            **peak_bytes(),'free_bytes':guard()}
    report['utc']=utc(); new_json(OUT/'SOURCE_SNAPSHOTS.json',report)

def read_input(name, cols=None, date_col='date', lo='2012-01-01', hi='2024-12-31'):
    path=INPUTS/name
    return io.dated(path,cols or io.schema_names(path),date_col,lo,hi)

def accounting():
    q=read_input('quarters.parquet',date_col='datadate')
    e=read_input('earnings.parquet',date_col='datadate')
    for f in [q,e]:
        f['gvkey']=pd.to_numeric(f.gvkey,errors='coerce')
        f['datadate']=pd.to_datetime(f.datadate,errors='coerce')
    e['rdq']=pd.to_datetime(e.rdq,errors='coerce')
    # Earliest valid release wins duplicate report-date rows; conflicting quarter
    # values are excluded rather than retrospectively selecting a revision.
    e=e[(e.rdq.isna())|(e.rdq<=pd.Timestamp('2024-12-31'))]
    releases=e[e.rdq>=e.datadate].groupby(['gvkey','datadate'],as_index=False).rdq.min()
    q=q.drop_duplicates()
    q=q[~q.duplicated(['gvkey','datadate'],keep=False)]
    q=q.merge(releases,on=['gvkey','datadate'],how='left')
    q['availability_rule']=np.where(q.rdq.notna(),'rdq_end_of_day','fiscal_end_plus_120d')
    q['available_day']=q.rdq.fillna(q.datadate+pd.Timedelta(days=120))
    q=q[q.available_day<=pd.Timestamp('2024-12-31')]
    for c in ['atq','cheq','invtq','dlcq','dlttq']:
        q[c]=pd.to_numeric(q[c],errors='coerce')
    valid=q.atq>0
    q['cash']=(q.cheq/q.atq).where(valid)
    q['inventory']=(q.invtq/q.atq).where(valid)
    # Both debt components must be observed. Missing debt is not zero debt.
    q['leverage']=((q.dlcq+q.dlttq)/q.atq).where(valid)
    # Key-only undated private snapshot; central decoder handles this read.
    link=io.read_keys(INPUTS/'identity.parquet')
    q=q.merge(link,on='gvkey',how='inner')
    e=e[e.rdq.notna()].merge(link,on='gvkey',how='inner')
    q['availability']=q.available_day.map({v:clock(v) for v in q.available_day.unique()})
    e['availability']=e.rdq.map({v:clock(v) for v in e.rdq.unique()})
    return q.sort_values(['availability','datadate']), e.sort_values('availability')

def common_inputs(r, me, ff, origin_position, firm_position):
    """Origin and prior sessions only, lagged market-equity weights, LOO industry."""
    t=origin_position; j=firm_position
    if t<22:
        return {c:np.nan for c in ['market_log_rv5','market_log_rv22','industry_log_rv5']}
    rr=np.asarray(r[t-21:t+1],float)
    ww=np.asarray(me[t-22:t],float)
    good=np.isfinite(rr)&np.isfinite(ww)&(ww>0)
    rw=np.where(good,rr*ww,0); w=np.where(good,ww,0)
    den=w.sum(1)
    mkt=np.divide(rw.sum(1),den,out=np.full(len(rr),np.nan),where=(den>0)&(good.sum(1)>=50))
    out={f'market_log_rv{n}':float(np.log(max(np.mean(mkt[-n:]**2),1e-8)))
         if np.isfinite(mkt[-n:]).all() else np.nan for n in [5,22]}
    group=ff[t,j]
    same=(ff[t]==group)&np.isfinite(group)&(group>=1)&(group<=49)
    # Classification at the origin is known before cutoff; no future classification.
    nums=rw[-5:,same].sum(1)-rw[-5:,j]
    dens=w[-5:,same].sum(1)-w[-5:,j]
    ir=np.divide(nums,dens,out=np.full(5,np.nan),where=dens>0)
    out['industry_log_rv5']=float(np.log(max(np.mean(ir**2),1e-8))) if same[j] and np.isfinite(ir).all() else np.nan
    return out

def common_all(r,me,ff,t):
    n=r.shape[1]
    if t<22:
        return {c:np.full(n,np.nan) for c in ['market_log_rv5','market_log_rv22','industry_log_rv5']}
    rr=r[t-21:t+1]; ww=me[t-22:t]
    good=np.isfinite(rr)&np.isfinite(ww)&(ww>0)
    rw=np.where(good,rr*ww,0.); w=np.where(good,ww,0.)
    den=w.sum(1)
    mkt=np.divide(rw.sum(1),den,out=np.full(22,np.nan),where=(den>0)&(good.sum(1)>=50))
    result={f'market_log_rv{k}':np.full(n,np.log(max(np.mean(mkt[-k:]**2),1e-8))
            if np.isfinite(mkt[-k:]).all() else np.nan) for k in [5,22]}
    industry=np.full(n,np.nan)
    for g in range(1,50):
        same=ff[t]==g
        if not same.any(): continue
        nums=rw[-5:,same].sum(1)[:,None]-rw[-5:,same]
        dens=w[-5:,same].sum(1)[:,None]-w[-5:,same]
        ir=np.divide(nums,dens,out=np.full_like(nums,np.nan),where=dens>0)
        valid=np.isfinite(ir).all(0)
        values=np.log(np.maximum(np.mean(ir**2,axis=0),1e-8)); values[~valid]=np.nan
        industry[same]=values
    result['industry_log_rv5']=industry
    return result

def build_year(year,keys,q,e,cf):
    guard(); start=time.perf_counter()
    # One preceding year for rolling history. Future sessions are present only
    # to construct development labels; feature functions slice at the origin.
    chunks=[read_input(f'crsp_{y}.parquet',lo=f'{y}-01-01',hi=f'{y}-12-31')
            for y in [year-1,year]]
    if year<2022:
        chunks.append(read_input(f'crsp_{year+1}.parquet',lo=f'{year+1}-01-01',hi=f'{year+1}-01-15'))
    d=pd.concat(chunks,ignore_index=True); del chunks
    d=d[d.share_type.isin([0,2])].copy(); d['date']=pd.to_datetime(d.date)
    if d.duplicated(['date','permno']).any():
        raise ValueError('Duplicate CRSP firm-session')
    dates=pd.DatetimeIndex(sorted(d.date.unique())); firms=pd.Index(sorted(d.permno.unique()))
    def wide(c):
        return d.pivot(index='date',columns='permno',values=c).reindex(index=dates,columns=firms).to_numpy(dtype=float)
    r,me,ff=wide('ret'),wide('me'),wide('ff49')
    del d; gc.collect()
    k=keys[keys.date.dt.year==year].copy().sort_values(['date','permno'])
    matched=set(q.permno.astype(int))
    eg={int(p):v.availability.to_numpy(dtype='datetime64[ns]') for p,v in e.groupby('permno')}
    rows=[]
    for origin,group in k.groupby('date',sort=True):
        guard(); t=dates.get_indexer([origin])[0]; cutoff=clock(origin); avail=clock(origin,'20:00:00')
        if t<0:
            raise ValueError('Origin missing from exchange calendar')
        common=common_all(r,me,ff,t)
        qq=q[(q.availability<cutoff)&(q.datadate>=origin-pd.Timedelta(days=550))]
        qr={int(p):v for p,v in qq.sort_values(['datadate','availability']).drop_duplicates('permno',keep='last').set_index('permno').iterrows()}
        cc=cf[(cf.availability<cutoff)&(cf.eom>=origin-pd.Timedelta(days=550))]
        cr={int(p):v for p,v in cc.sort_values('eom').drop_duplicates('permno',keep='last').set_index('permno').iterrows()}
        for p in group.permno:
            j=firms.get_indexer([p])[0]
            row={'firm_id':hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24],
                 'origin_session':origin,'cutoff':cutoff,'universe_included':True,
                 'fundamental_identity_matched':int(p) in matched}
            if j>=0:
                row.update(own_inputs(dates,r[:,j],me[:,j],t))
                # The market part does not depend on j. LOO industry does.
                row.update({c:v[j] for c,v in common.items()})
            else:
                row.update({c:np.nan for c in FEATURES})
            for c in ['size','rv1','rv5','rv22','ret1','ret5','ret22','own_downside5',
                      'market_log_rv5','market_log_rv22','industry_log_rv5']:
                row[c+'__available_at']=avail if np.isfinite(row.get(c,np.nan)) else pd.NaT
            chosen=qr.get(int(p))
            row['fundamental_rule']=str(chosen.availability_rule) if chosen is not None else 'missing'
            row['fundamental_fiscal_end']=chosen.datadate if chosen is not None else pd.NaT
            for c in ['cash','inventory','leverage']:
                v=float(chosen[c]) if chosen is not None else np.nan
                row[c]=v; row[c+'__available_at']=chosen.availability if np.isfinite(v) else pd.NaT
            picked=cr.get(int(p))
            row['cash_flow']=float(picked.ocf_at) if picked is not None and pd.notna(picked.ocf_at) else np.nan
            row['cash_flow__available_at']=picked.availability if picked is not None and np.isfinite(row['cash_flow']) else pd.NaT
            row['cash_flow_source_eom']=picked.eom if picked is not None else pd.NaT
            announcements=eg.get(int(p),np.array([],dtype='datetime64[ns]'))
            past=announcements[announcements<cutoff.to_datetime64()]
            # No claim of no announcement when the source has never covered a firm.
            covered=len(past)>0 and past[-1]>=clock(dates[max(0,t-299)])
            row['earnings_covered']=float(covered)
            row['post_earnings']=float(len(past)>0 and past[-1]>=clock(dates[max(0,t-4)])) if len(past) else np.nan
            row['earnings_scheduled']=np.nan
            # The observable past-only calendar query is available at cutoff-1s.
            for c in ['earnings_covered','post_earnings']:
                row[c+'__available_at']=cutoff-pd.Timedelta(seconds=1) if np.isfinite(row[c]) else pd.NaT
            row['earnings_scheduled__available_at']=pd.NaT
            row['eligible_features']=bool(j>=0 and np.isfinite(row['rv22']) and np.isfinite(row['size']))
            for c in FEATURES:
                row.setdefault(c,np.nan)
                row[c+'__missing']=not np.isfinite(row[c])
                row[c+'__included']=True
                if not row[c+'__missing']:
                    require_past([row[c+'__available_at']],cutoff,c)
            if year<=2022:
                y,end,valid=label(dates,r[:,j],t,'2022-12-31') if j>=0 else (np.nan,pd.NaT,False)
                row.update(y=y,label_end=end,label_observed=valid,
                    fit_included=bool(row['eligible_features'] and valid and year<=2020 and end<=pd.Timestamp('2020-12-31')),
                    selection_included=bool(row['eligible_features'] and valid and year>=2021),
                    refit_included=bool(row['eligible_features'] and valid))
            rows.append(row)
    f=pd.DataFrame(rows)
    # Columns that are entirely missing still have stable datetime schemas.
    for c in [x for x in f if x.endswith('__available_at')]+['cutoff','origin_session','fundamental_fiscal_end','cash_flow_source_eom']:
        f[c]=pd.to_datetime(f[c])
    if year>=2023 and any(c in f for c in ['y','label_end','label_observed']):
        raise ValueError('Evaluation label leaked into feature manifest')
    path=MANIFEST/f'year_{year}.parquet'; io.write_parquet(path,f)
    result={'year':year,'rows':len(f),'eligible_features':int(f.eligible_features.sum()),
            'fit_rows':int(f.fit_included.sum()) if year<=2022 else None,
            'selection_rows':int(f.selection_included.sum()) if year<=2022 else None,
            'fundamentals_coverage':{c:float(f[c].notna().mean()) for c in ['cash','cash_flow','leverage','inventory']},
            'sha256':sha(path),'elapsed_seconds':time.perf_counter()-start,**peak_bytes()}
    print(json.dumps(result),flush=True)
    return result

def manifests():
    start=time.perf_counter()
    snapshots_info=json.loads((OUT/'SOURCE_SNAPSHOTS.json').read_text())
    for p,h in snapshots_info['snapshot_sha256'].items():
        if sha(p)!=h: raise ValueError('Snapshot changed')
    keys=read_input('panel_keys.parquet',lo='2015-01-01'); keys['date']=pd.to_datetime(keys.date)
    if keys.duplicated(['date','permno']).any(): raise ValueError('Duplicate panel key')
    q,e=accounting()
    cf=pd.concat([read_input(f'cashflow_{y}.parquet',date_col='eom',lo=f'{y}-01-01') for y in range(2014,2025)],ignore_index=True)
    cf=cf[(cf.common==1)&(cf.primary_sec==1)&cf.permno.notna()].copy()
    cf['permno']=cf.permno.astype(int); cf['eom']=pd.to_datetime(cf.eom)
    if cf.duplicated(['eom','permno']).any(): raise ValueError('Ambiguous JKP primary security')
    cf['available_day']=cf.eom+pd.DateOffset(months=6)
    cf=cf[cf.available_day<=pd.Timestamp('2024-12-31')]
    # Map distinct dates, not every firm-month, to keep timezone conversion small.
    mapping={v:clock(v) for v in cf.available_day.unique()}
    cf['availability']=cf.available_day.map(mapping)
    results=[build_year(y,keys,q,e,cf) for y in range(2015,2025)]
    result={'version':VERSION,'rules':CLOCK_RULES,'partitions':results,
            'source_snapshot_manifest_sha256':sha(OUT/'SOURCE_SNAPSHOTS.json'),
            'elapsed_seconds':time.perf_counter()-start,**peak_bytes(),'free_bytes':guard()}
    result['utc']=utc(); new_json(OUT/'MANIFEST_BUILD.json',result)
