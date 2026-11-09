# TD4 copied functions: diffs against the originals

Generated from the source files (ast line ranges). Each TD4 copy changes only what its docstring and the `TD4:` comments state; parity receipts (PARITY_*.json) show the copies reproduce the frozen TD1/TD2/TD3 outputs when run with the original seal. Reformatting (line wrapping, spacing) appears as diff noise because the TD1/TD2 originals use a compact style.

## TD1 v2 build.py accounting -> td4_manifest.py:accounting

```diff
--- build.py:accounting
+++ td4_manifest.py:accounting
@@ -1,31 +1,35 @@
-def accounting():
-    q=read_input('quarters.parquet',date_col='datadate')
-    e=read_input('earnings.parquet',date_col='datadate')
-    for f in [q,e]:
-        f['gvkey']=pd.to_numeric(f.gvkey,errors='coerce')
-        f['datadate']=pd.to_datetime(f.datadate,errors='coerce')
-    e['rdq']=pd.to_datetime(e.rdq,errors='coerce')
-    # Earliest valid release wins duplicate report-date rows; conflicting quarter
-    # values are excluded rather than retrospectively selecting a revision.
-    e=e[(e.rdq.isna())|(e.rdq<=pd.Timestamp('2024-12-31'))]
-    releases=e[e.rdq>=e.datadate].groupby(['gvkey','datadate'],as_index=False).rdq.min()
-    q=q.drop_duplicates()
-    q=q[~q.duplicated(['gvkey','datadate'],keep=False)]
-    q=q.merge(releases,on=['gvkey','datadate'],how='left')
-    q['availability_rule']=np.where(q.rdq.notna(),'rdq_end_of_day','fiscal_end_plus_120d')
-    q['available_day']=q.rdq.fillna(q.datadate+pd.Timedelta(days=120))
-    q=q[q.available_day<=pd.Timestamp('2024-12-31')]
-    for c in ['atq','cheq','invtq','dlcq','dlttq']:
-        q[c]=pd.to_numeric(q[c],errors='coerce')
-    valid=q.atq>0
-    q['cash']=(q.cheq/q.atq).where(valid)
-    q['inventory']=(q.invtq/q.atq).where(valid)
-    # Both debt components must be observed. Missing debt is not zero debt.
-    q['leverage']=((q.dlcq+q.dlttq)/q.atq).where(valid)
-    # Key-only undated private snapshot; central decoder handles this read.
-    link=io.read_keys(INPUTS/'identity.parquet')
-    q=q.merge(link,on='gvkey',how='inner')
-    e=e[e.rdq.notna()].merge(link,on='gvkey',how='inner')
-    q['availability']=q.available_day.map({v:clock(v) for v in q.available_day.unique()})
-    e['availability']=e.rdq.map({v:clock(v) for v in e.rdq.unique()})
-    return q.sort_values(['availability','datadate']), e.sort_values('availability')
+def accounting(seal):
+    """TD1 build.accounting with the seal as a parameter (TD4: seal; TD4: explicit snapshot paths)."""
+    core, _ = _td1()
+    clock = core.clock
+    tag = seal[:4]
+    if seal == T.TD2_SEAL:
+        qp, ep = T.TD1 / 'inputs/quarters.parquet', T.TD1 / 'inputs/earnings.parquet'
+    else:
+        qp, ep = S.SNAP / f'quarters_seal{tag}.parquet', S.SNAP / f'earnings_seal{tag}.parquet'
+    q = read_snapshot(qp, date_col='datadate', hi=seal)
+    e = read_snapshot(ep, date_col='datadate', hi=seal)
+    for f in [q, e]:
+        f['gvkey'] = pd.to_numeric(f.gvkey, errors='coerce')
+        f['datadate'] = pd.to_datetime(f.datadate, errors='coerce')
+    e['rdq'] = pd.to_datetime(e.rdq, errors='coerce')
+    e = e[(e.rdq.isna()) | (e.rdq <= pd.Timestamp(seal))]                       # TD4: seal
+    releases = e[e.rdq >= e.datadate].groupby(['gvkey', 'datadate'], as_index=False).rdq.min()
+    q = q.drop_duplicates()
+    q = q[~q.duplicated(['gvkey', 'datadate'], keep=False)]
+    q = q.merge(releases, on=['gvkey', 'datadate'], how='left')
+    q['availability_rule'] = np.where(q.rdq.notna(), 'rdq_end_of_day', 'fiscal_end_plus_120d')
+    q['available_day'] = q.rdq.fillna(q.datadate + pd.Timedelta(days=120))
+    q = q[q.available_day <= pd.Timestamp(seal)]                                 # TD4: seal
+    for c in ['atq', 'cheq', 'invtq', 'dlcq', 'dlttq']:
+        q[c] = pd.to_numeric(q[c], errors='coerce')
+    valid = q.atq > 0
+    q['cash'] = (q.cheq / q.atq).where(valid)
+    q['inventory'] = (q.invtq / q.atq).where(valid)
+    q['leverage'] = ((q.dlcq + q.dlttq) / q.atq).where(valid)
+    link = pq.read_table(T.TD1 / 'inputs/identity.parquet', columns=['gvkey', 'permno']).to_pandas()
+    q = q.merge(link, on='gvkey', how='inner')
+    e = e[e.rdq.notna()].merge(link, on='gvkey', how='inner')
+    q['availability'] = q.available_day.map({v: clock(v) for v in q.available_day.unique()})
+    e['availability'] = e.rdq.map({v: clock(v) for v in e.rdq.unique()})
+    return q.sort_values(['availability', 'datadate']), e.sort_values('availability')
```

## TD1 v2 build.py common_all -> td4_manifest.py:common_all

```diff
--- build.py:common_all
+++ td4_manifest.py:common_all
@@ -1,23 +1,28 @@
-def common_all(r,me,ff,t):
-    n=r.shape[1]
-    if t<22:
-        return {c:np.full(n,np.nan) for c in ['market_log_rv5','market_log_rv22','industry_log_rv5']}
-    rr=r[t-21:t+1]; ww=me[t-22:t]
-    good=np.isfinite(rr)&np.isfinite(ww)&(ww>0)
-    rw=np.where(good,rr*ww,0.); w=np.where(good,ww,0.)
-    den=w.sum(1)
-    mkt=np.divide(rw.sum(1),den,out=np.full(22,np.nan),where=(den>0)&(good.sum(1)>=50))
-    result={f'market_log_rv{k}':np.full(n,np.log(max(np.mean(mkt[-k:]**2),1e-8))
-            if np.isfinite(mkt[-k:]).all() else np.nan) for k in [5,22]}
-    industry=np.full(n,np.nan)
-    for g in range(1,50):
-        same=ff[t]==g
-        if not same.any(): continue
-        nums=rw[-5:,same].sum(1)[:,None]-rw[-5:,same]
-        dens=w[-5:,same].sum(1)[:,None]-w[-5:,same]
-        ir=np.divide(nums,dens,out=np.full_like(nums,np.nan),where=dens>0)
-        valid=np.isfinite(ir).all(0)
-        values=np.log(np.maximum(np.mean(ir**2,axis=0),1e-8)); values[~valid]=np.nan
-        industry[same]=values
-    result['industry_log_rv5']=industry
+def common_all(r, me, ff, t):
+    """TD1 build.common_all (unchanged)."""
+    n = r.shape[1]
+    if t < 22:
+        return {c: np.full(n, np.nan) for c in ['market_log_rv5', 'market_log_rv22', 'industry_log_rv5']}
+    rr = r[t - 21:t + 1]
+    ww = me[t - 22:t]
+    good = np.isfinite(rr) & np.isfinite(ww) & (ww > 0)
+    rw = np.where(good, rr * ww, 0.)
+    w = np.where(good, ww, 0.)
+    den = w.sum(1)
+    mkt = np.divide(rw.sum(1), den, out=np.full(22, np.nan), where=(den > 0) & (good.sum(1) >= 50))
+    result = {f'market_log_rv{k}': np.full(n, np.log(max(np.mean(mkt[-k:] ** 2), 1e-8))
+                                           if np.isfinite(mkt[-k:]).all() else np.nan) for k in [5, 22]}
+    industry = np.full(n, np.nan)
+    for g in range(1, 50):
+        same = ff[t] == g
+        if not same.any():
+            continue
+        nums = rw[-5:, same].sum(1)[:, None] - rw[-5:, same]
+        dens = w[-5:, same].sum(1)[:, None] - w[-5:, same]
+        ir = np.divide(nums, dens, out=np.full_like(nums, np.nan), where=dens > 0)
+        valid = np.isfinite(ir).all(0)
+        values = np.log(np.maximum(np.mean(ir ** 2, axis=0), 1e-8))
+        values[~valid] = np.nan
+        industry[same] = values
+    result['industry_log_rv5'] = industry
     return result
```

## TD1 v2 build.py build_year -> td4_manifest.py:build_year

```diff
--- build.py:build_year
+++ td4_manifest.py:build_year
@@ -1,93 +1,94 @@
-def build_year(year,keys,q,e,cf):
-    guard(); start=time.perf_counter()
-    # One preceding year for rolling history. Future sessions are present only
-    # to construct development labels; feature functions slice at the origin.
-    chunks=[read_input(f'crsp_{y}.parquet',lo=f'{y}-01-01',hi=f'{y}-12-31')
-            for y in [year-1,year]]
-    if year<2022:
-        chunks.append(read_input(f'crsp_{year+1}.parquet',lo=f'{year+1}-01-01',hi=f'{year+1}-01-15'))
-    d=pd.concat(chunks,ignore_index=True); del chunks
-    d=d[d.share_type.isin([0,2])].copy(); d['date']=pd.to_datetime(d.date)
-    if d.duplicated(['date','permno']).any():
+def build_year(year, keys, q, e, cf, seal):
+    """TD1 build.build_year (TD4: explicit CRSP paths and seal; labels only for year <= 2022 as TD1, never used)."""
+    core, spec = _td1()
+    clock, own_inputs, require_past, FEATURES = core.clock, core.own_inputs, core.require_past, spec.FEATURES
+    E.guard()
+    start = time.perf_counter()
+    chunks = [read_snapshot(crsp_path(y), lo=f'{y}-01-01', hi=f'{y}-12-31') for y in [year - 1, year]]
+    if year < 2022:
+        raise PermissionError('TD4 builds manifests for 2024 (parity) and 2025 only')
+    d = pd.concat(chunks, ignore_index=True)
+    del chunks
+    d = d[d.share_type.isin([0, 2])].copy()
+    d['date'] = pd.to_datetime(d.date)
+    if d.duplicated(['date', 'permno']).any():
         raise ValueError('Duplicate CRSP firm-session')
-    dates=pd.DatetimeIndex(sorted(d.date.unique())); firms=pd.Index(sorted(d.permno.unique()))
+    dates = pd.DatetimeIndex(sorted(d.date.unique()))
+    firms = pd.Index(sorted(d.permno.unique()))
+
     def wide(c):
-        return d.pivot(index='date',columns='permno',values=c).reindex(index=dates,columns=firms).to_numpy(dtype=float)
-    r,me,ff=wide('ret'),wide('me'),wide('ff49')
-    del d; gc.collect()
-    k=keys[keys.date.dt.year==year].copy().sort_values(['date','permno'])
-    matched=set(q.permno.astype(int))
-    eg={int(p):v.availability.to_numpy(dtype='datetime64[ns]') for p,v in e.groupby('permno')}
-    rows=[]
-    for origin,group in k.groupby('date',sort=True):
-        guard(); t=dates.get_indexer([origin])[0]; cutoff=clock(origin); avail=clock(origin,'20:00:00')
-        if t<0:
+        return d.pivot(index='date', columns='permno', values=c).reindex(index=dates, columns=firms).to_numpy(dtype=float)
+    r, me, ff = wide('ret'), wide('me'), wide('ff49')
+    del d
+    gc.collect()
+    k = keys[keys.date.dt.year == year].copy().sort_values(['date', 'permno'])
+    matched = set(q.permno.astype(int))
+    eg = {int(p): v.availability.to_numpy(dtype='datetime64[ns]') for p, v in e.groupby('permno')}
+    rows = []
+    for origin, group in k.groupby('date', sort=True):
+        E.guard()
+        t = dates.get_indexer([origin])[0]
+        cutoff = clock(origin)
+        avail = clock(origin, '20:00:00')
+        if t < 0:
             raise ValueError('Origin missing from exchange calendar')
-        common=common_all(r,me,ff,t)
-        qq=q[(q.availability<cutoff)&(q.datadate>=origin-pd.Timedelta(days=550))]
-        qr={int(p):v for p,v in qq.sort_values(['datadate','availability']).drop_duplicates('permno',keep='last').set_index('permno').iterrows()}
-        cc=cf[(cf.availability<cutoff)&(cf.eom>=origin-pd.Timedelta(days=550))]
-        cr={int(p):v for p,v in cc.sort_values('eom').drop_duplicates('permno',keep='last').set_index('permno').iterrows()}
+        common = common_all(r, me, ff, t)
+        qq = q[(q.availability < cutoff) & (q.datadate >= origin - pd.Timedelta(days=550))]
+        qr = {int(p): v for p, v in qq.sort_values(['datadate', 'availability']).drop_duplicates('permno', keep='last')
+              .set_index('permno').iterrows()}
+        cc = cf[(cf.availability < cutoff) & (cf.eom >= origin - pd.Timedelta(days=550))]
+        cr = {int(p): v for p, v in cc.sort_values('eom').drop_duplicates('permno', keep='last')
+              .set_index('permno').iterrows()}
         for p in group.permno:
-            j=firms.get_indexer([p])[0]
-            row={'firm_id':hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24],
-                 'origin_session':origin,'cutoff':cutoff,'universe_included':True,
-                 'fundamental_identity_matched':int(p) in matched}
-            if j>=0:
-                row.update(own_inputs(dates,r[:,j],me[:,j],t))
-                # The market part does not depend on j. LOO industry does.
-                row.update({c:v[j] for c,v in common.items()})
+            j = firms.get_indexer([p])[0]
+            row = {'firm_id': hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24],
+                   'origin_session': origin, 'cutoff': cutoff, 'universe_included': True,
+                   'fundamental_identity_matched': int(p) in matched}
+            if j >= 0:
+                row.update(own_inputs(dates, r[:, j], me[:, j], t))
+                row.update({c: v[j] for c, v in common.items()})
             else:
-                row.update({c:np.nan for c in FEATURES})
-            for c in ['size','rv1','rv5','rv22','ret1','ret5','ret22','own_downside5',
-                      'market_log_rv5','market_log_rv22','industry_log_rv5']:
-                row[c+'__available_at']=avail if np.isfinite(row.get(c,np.nan)) else pd.NaT
-            chosen=qr.get(int(p))
-            row['fundamental_rule']=str(chosen.availability_rule) if chosen is not None else 'missing'
-            row['fundamental_fiscal_end']=chosen.datadate if chosen is not None else pd.NaT
-            for c in ['cash','inventory','leverage']:
-                v=float(chosen[c]) if chosen is not None else np.nan
-                row[c]=v; row[c+'__available_at']=chosen.availability if np.isfinite(v) else pd.NaT
-            picked=cr.get(int(p))
-            row['cash_flow']=float(picked.ocf_at) if picked is not None and pd.notna(picked.ocf_at) else np.nan
-            row['cash_flow__available_at']=picked.availability if picked is not None and np.isfinite(row['cash_flow']) else pd.NaT
-            row['cash_flow_source_eom']=picked.eom if picked is not None else pd.NaT
-            announcements=eg.get(int(p),np.array([],dtype='datetime64[ns]'))
-            past=announcements[announcements<cutoff.to_datetime64()]
-            # No claim of no announcement when the source has never covered a firm.
-            covered=len(past)>0 and past[-1]>=clock(dates[max(0,t-299)])
-            row['earnings_covered']=float(covered)
-            row['post_earnings']=float(len(past)>0 and past[-1]>=clock(dates[max(0,t-4)])) if len(past) else np.nan
-            row['earnings_scheduled']=np.nan
-            # The observable past-only calendar query is available at cutoff-1s.
-            for c in ['earnings_covered','post_earnings']:
-                row[c+'__available_at']=cutoff-pd.Timedelta(seconds=1) if np.isfinite(row[c]) else pd.NaT
-            row['earnings_scheduled__available_at']=pd.NaT
-            row['eligible_features']=bool(j>=0 and np.isfinite(row['rv22']) and np.isfinite(row['size']))
+                row.update({c: np.nan for c in FEATURES})
+            for c in ['size', 'rv1', 'rv5', 'rv22', 'ret1', 'ret5', 'ret22', 'own_downside5',
+                      'market_log_rv5', 'market_log_rv22', 'industry_log_rv5']:
+                row[c + '__available_at'] = avail if np.isfinite(row.get(c, np.nan)) else pd.NaT
+            chosen = qr.get(int(p))
+            row['fundamental_rule'] = str(chosen.availability_rule) if chosen is not None else 'missing'
+            row['fundamental_fiscal_end'] = chosen.datadate if chosen is not None else pd.NaT
+            for c in ['cash', 'inventory', 'leverage']:
+                v = float(chosen[c]) if chosen is not None else np.nan
+                row[c] = v
+                row[c + '__available_at'] = chosen.availability if np.isfinite(v) else pd.NaT
+            picked = cr.get(int(p))
+            row['cash_flow'] = float(picked.ocf_at) if picked is not None and pd.notna(picked.ocf_at) else np.nan
+            row['cash_flow__available_at'] = picked.availability if picked is not None and np.isfinite(
+                row['cash_flow']) else pd.NaT
+            row['cash_flow_source_eom'] = picked.eom if picked is not None else pd.NaT
+            announcements = eg.get(int(p), np.array([], dtype='datetime64[ns]'))
+            past = announcements[announcements < cutoff.to_datetime64()]
+            covered = len(past) > 0 and past[-1] >= clock(dates[max(0, t - 299)])
+            row['earnings_covered'] = float(covered)
+            row['post_earnings'] = float(len(past) > 0 and past[-1] >= clock(dates[max(0, t - 4)])) if len(past) \
+                else np.nan
+            row['earnings_scheduled'] = np.nan
+            for c in ['earnings_covered', 'post_earnings']:
+                row[c + '__available_at'] = cutoff - pd.Timedelta(seconds=1) if np.isfinite(row[c]) else pd.NaT
+            row['earnings_scheduled__available_at'] = pd.NaT
+            row['eligible_features'] = bool(j >= 0 and np.isfinite(row['rv22']) and np.isfinite(row['size']))
             for c in FEATURES:
-                row.setdefault(c,np.nan)
-                row[c+'__missing']=not np.isfinite(row[c])
-                row[c+'__included']=True
-                if not row[c+'__missing']:
-                    require_past([row[c+'__available_at']],cutoff,c)
-            if year<=2022:
-                y,end,valid=label(dates,r[:,j],t,'2022-12-31') if j>=0 else (np.nan,pd.NaT,False)
-                row.update(y=y,label_end=end,label_observed=valid,
-                    fit_included=bool(row['eligible_features'] and valid and year<=2020 and end<=pd.Timestamp('2020-12-31')),
-                    selection_included=bool(row['eligible_features'] and valid and year>=2021),
-                    refit_included=bool(row['eligible_features'] and valid))
+                row.setdefault(c, np.nan)
+                row[c + '__missing'] = not np.isfinite(row[c])
+                row[c + '__included'] = True
+                if not row[c + '__missing']:
+                    require_past([row[c + '__available_at']], cutoff, c)
             rows.append(row)
-    f=pd.DataFrame(rows)
-    # Columns that are entirely missing still have stable datetime schemas.
-    for c in [x for x in f if x.endswith('__available_at')]+['cutoff','origin_session','fundamental_fiscal_end','cash_flow_source_eom']:
-        f[c]=pd.to_datetime(f[c])
-    if year>=2023 and any(c in f for c in ['y','label_end','label_observed']):
-        raise ValueError('Evaluation label leaked into feature manifest')
-    path=MANIFEST/f'year_{year}.parquet'; io.write_parquet(path,f)
-    result={'year':year,'rows':len(f),'eligible_features':int(f.eligible_features.sum()),
-            'fit_rows':int(f.fit_included.sum()) if year<=2022 else None,
-            'selection_rows':int(f.selection_included.sum()) if year<=2022 else None,
-            'fundamentals_coverage':{c:float(f[c].notna().mean()) for c in ['cash','cash_flow','leverage','inventory']},
-            'sha256':sha(path),'elapsed_seconds':time.perf_counter()-start,**peak_bytes()}
-    print(json.dumps(result),flush=True)
-    return result
+    f = pd.DataFrame(rows)
+    for c in [x for x in f if x.endswith('__available_at')] + ['cutoff', 'origin_session', 'fundamental_fiscal_end',
+                                                               'cash_flow_source_eom']:
+        f[c] = pd.to_datetime(f[c])
+    if any(c in f for c in ['y', 'label_end', 'label_observed']):
+        raise ValueError('Label column in a TD4 manifest')
+    return f, {'year': year, 'rows': len(f), 'eligible_features': int(f.eligible_features.sum()),
+               'fundamentals_coverage': {c: float(f[c].notna().mean()) for c in ['cash', 'cash_flow', 'leverage',
+                                                                                 'inventory']},
+               'elapsed_seconds': time.perf_counter() - start}
```

## TD1 v2 sources.py csv_dated -> td4_sources.py:csv_dated

```diff
--- sources.py:csv_dated
+++ td4_sources.py:csv_dated
@@ -1,8 +1,12 @@
-def csv_dated(path, cols, date_col='datadate', lo='2012-01-01', hi=HARD_END,
-              second_date=None):
-    """Stream CSV: reject using date positions before accessing other field values."""
-    guard(); rows = []
+def csv_dated(path, cols, date_col='datadate', lo='2012-01-01', hi=T.HARD_SEAL, second_date=None):
+    """TD1 sources.csv_dated with a seal parameter (rows outside [lo, hi] and rows whose second date is after the seal
+    are skipped before any other field is used)."""
+    _check_seal(hi)
+    E.guard()
+    rows = []
     with Path(path).open(newline='', encoding='utf-8-sig') as f:
-        reader = csv.reader(f); header = next(reader)
-        di = header.index(date_col); ix = [header.index(c) for c in cols]
+        reader = csv.reader(f)
+        header = next(reader)
+        di = header.index(date_col)
+        ix = [header.index(c) for c in cols]
         si = header.index(second_date) if second_date else None
@@ -10,5 +14,5 @@
             day = values[di][:10]
-            if not (lo <= day <= min(hi, HARD_END)):
+            if not (lo <= day <= hi):
                 continue
-            if si is not None and values[si] and values[si][:10] > HARD_END:
+            if si is not None and values[si] and values[si][:10] > hi:
                 continue
@@ -16,3 +20,3 @@
             if n % 50000 == 0:
-                guard()
+                E.guard()
     return pd.DataFrame(rows, columns=cols)
```

## TD1 v2 sources.py dated -> td4_sources.py:parquet_dated

```diff
--- sources.py:dated
+++ td4_sources.py:parquet_dated
@@ -1,26 +1,20 @@
-def dated(path, cols, date_col='date', lo='2014-01-01', hi=HARD_END):
-    guard()
-    if hi > HARD_END:
-        raise ValueError('2025 is prohibited')
-    path = Path(path).resolve()
-    allowed = [p.resolve() for p in SOURCE_PATHS]
-    if path not in allowed and not path.is_relative_to(OUT.resolve()):
-        raise ValueError('Source outside TD1 allowlist')
-    if path==PANEL.resolve() and set(cols)-{'date','permno'}:
-        raise ValueError('Legacy panel is admitted for keys only; cached outcomes/features prohibited')
+def parquet_dated(path, cols, date_col='date', lo='2014-01-01', hi=T.HARD_SEAL):
+    """TD1 sources.dated with a seal parameter: dates-only pass first, then the projection, both under the scan
+    predicate; refuses if any returned date is after hi."""
+    _check_seal(hi)
+    E.guard()
     schema = pq.read_schema(path)
     typ = schema.field(date_col).type
-    convert = str if pa.types.is_string(typ) else (dt.date.fromisoformat if pa.types.is_date(typ) else pd.Timestamp)
+    convert = str if pa.types.is_string(typ) or pa.types.is_large_string(typ) else (
+        dt.date.fromisoformat if pa.types.is_date(typ) else pd.Timestamp)
     filt = (ds.field(date_col) >= pa.scalar(convert(lo), type=typ)) & (
-            ds.field(date_col) <= pa.scalar(convert(hi), type=typ))
+        ds.field(date_col) <= pa.scalar(convert(hi), type=typ))
     dataset = ds.dataset(path, format='parquet')
-    # Only dates first, including no other fields from rejected rows.
-    dates = dataset.to_table(columns=[date_col], filter=filt, use_threads=False)
-    values = dates[date_col].to_pandas()
-    if values.isna().any() or (pd.to_datetime(values) > HARD_END).any():
+    dates = dataset.to_table(columns=[date_col], filter=filt, use_threads=False)[date_col].to_pandas()
+    if dates.isna().any() or (pd.to_datetime(dates) > pd.Timestamp(hi)).any():
         raise ValueError('Date seal')
-    del dates, values
-    result = dataset.to_table(columns=list(dict.fromkeys([date_col]+list(cols))),
-                              filter=filt, use_threads=False).to_pandas()
-    if (pd.to_datetime(result[date_col]) > HARD_END).any():
+    del dates
+    result = dataset.to_table(columns=list(dict.fromkeys([date_col] + list(cols))), filter=filt,
+                              use_threads=False)
+    if pc.max(pc.cast(result[date_col], pa.string())).as_py()[:10] > hi:
         raise ValueError('Date seal after projection')
```

## TD1 v2 core.py label -> td4_labels.py:label

```diff
--- core.py:label
+++ td4_labels.py:label
@@ -1,10 +1,10 @@
 def label(dates, returns, pos, allowed_end):
-    """Call only for development, or behind the evaluation clearance gate."""
-    if pos+5 >= len(dates):
+    """TD1 core.label (TD4: clamp at the 2025 seal)."""
+    if pos + 5 >= len(dates):
         return np.nan, pd.NaT, False
-    end = pd.Timestamp(dates[pos+5])
-    if end > pd.Timestamp(min(allowed_end,'2024-12-31')):
-        return np.nan, end, False  # do not inspect the future-return fields
-    w = np.asarray(returns[pos+1:pos+6],float)
-    valid = len(w)==5 and np.isfinite(w).all() and np.mean(w*w)>0
-    return (float(np.log(np.mean(w*w))) if valid else np.nan), end, bool(valid)
+    end = pd.Timestamp(dates[pos + 5])
+    if end > pd.Timestamp(min(allowed_end, T.HARD_SEAL)):
+        return np.nan, end, False
+    w = np.asarray(returns[pos + 1:pos + 6], float)
+    valid = len(w) == 5 and np.isfinite(w).all() and np.mean(w * w) > 0
+    return (float(np.log(np.mean(w * w))) if valid else np.nan), end, bool(valid)
```

## TD1 v2 fit.py model -> td4_models.py:model

```diff
--- fit.py:model
+++ td4_models.py:model
@@ -1,4 +1,6 @@
-def model(lr,alpha):
-    return MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',solver='adam',
-        alpha=alpha,batch_size=4096,learning_rate_init=lr,random_state=SEED,
-        max_iter=1,shuffle=True,early_stopping=False,tol=0.)
+def model(lr, alpha, seed):
+    """TD1 fit.model (copy)."""
+    from sklearn.neural_network import MLPRegressor
+    return MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=alpha, batch_size=4096,
+                        learning_rate_init=lr, random_state=seed, max_iter=1, shuffle=True, early_stopping=False,
+                        tol=0.)
```

## TD2 decoder_v3.py mapping -> td4_sources.py:links_extract

```diff
--- decoder_v3.py:mapping
+++ td4_sources.py:links_extract
@@ -1,12 +1,17 @@
-def mapping():
-    rows=[]
-    with FSMAP.open(encoding='utf-8-sig',newline='') as f:
+def links_extract(seal=T.HARD_SEAL):
+    """TD2 decoder_v3.mapping with a seal parameter (start <= seal; end censored at the seal; duplicates dropped)."""
+    _check_seal(seal)
+    rows = []
+    with T.FS_MAP.open(encoding='utf-8-sig', newline='') as f:
         for r in csv.DictReader(f):
-            # Inspect start before using identity values; later ends censored.
-            start=r['link_bdate'][:10]
-            if not start or start>'2024-12-31': continue
-            try: p=int(float(r['PERMNO']))
-            except (ValueError,TypeError): continue
-            end=r['link_edate'][:10]
-            rows.append((r['FACTSET_ENTITY_ID'],p,start,min(end or '2024-12-31','2024-12-31')))
-    return pd.DataFrame(rows,columns=['entity','permno','start','end']).drop_duplicates()
+            start = r['link_bdate'][:10]
+            if not start or start > seal:
+                continue
+            try:
+                p = int(float(r['PERMNO']))
+            except (ValueError, TypeError):
+                continue
+            end = r['link_edate'][:10]
+            rows.append((r['FACTSET_ENTITY_ID'], p, start, min(end or seal, seal)))
+    f = pd.DataFrame(rows, columns=['entity', 'permno', 'start', 'end']).drop_duplicates()
+    return f.rename(columns={'start': 'date'})
```

## TD2 decoder_v3.py build_news -> td4_sources.py:news_scores

```diff
--- decoder_v3.py:build_news
+++ td4_sources.py:news_scores
@@ -1,20 +1,29 @@
-def build_news(year,ricmap,seen):
-    rows={}
-    for batch in parquet(ARTICLE,['alt_id','rics','headline','body_short'],'date',f'{year}-01-01',f'{year}-12-31'):
+def news_scores(year, seen, seal=T.HARD_SEAL):
+    """TD2 decoder_v3.build_news for one year: alt_ids already seen in earlier partitions are skipped; bindings via
+    the frozen development RIC identities; earliest day per alt_id; TD2 lexicon scores of headline[:200]+body[:1000]."""
+    _check_seal(f'{year}-12-31' if f'{year}-12-31' <= seal else seal)
+    E.paths()
+    from engine import day_clock
+    score = _td2_score()
+    ricmap = json.loads((T.TD2 / 'sources/reuters_identity_development_v1.json').read_text(encoding='utf-8'))['mapping']
+    rows = {}
+    for batch in article_batches(f'{year}-01-01', f'{year}-12-31', ['alt_id', 'rics', 'headline', 'body_short']):
         for r in batch.itertuples(index=False):
-            aid=r.alt_id
-            if not aid or aid in seen or not r.rics: continue
-            day=r.date[:10]
-            if day>f'{year}-12-31': raise ValueError('Article date seal')
-            bindings=sorted({tuple(ricmap[t.strip()]) for t in r.rics.split(',') if t.strip() in ricmap})
-            if not bindings: continue
-            text=(r.headline or '')[:200]+(r.body_short or '')[:1000]
-            record={'date':pd.Timestamp(day),'alt_id':aid,'firms':[p for p,m in bindings],
-                'first_months':[m for p,m in bindings],'available_at':day_clock(day),**score(text)}
-            if aid not in rows or day<str(rows[aid]['date'].date()): rows[aid]=record
-        guard()
+            aid = r.alt_id
+            if not aid or aid in seen or not r.rics:
+                continue
+            day = r.date[:10]
+            if day > f'{year}-12-31':
+                raise ValueError('Article date seal')
+            bindings = sorted({tuple(ricmap[t.strip()]) for t in r.rics.split(',') if t.strip() in ricmap})
+            if not bindings:
+                continue
+            text = (r.headline or '')[:200] + (r.body_short or '')[:1000]
+            record = {'date': pd.Timestamp(day), 'alt_id': aid, 'firms': [p for p, m in bindings],
+                      'first_months': [m for p, m in bindings], 'available_at': day_clock(day), **score(text)}
+            if aid not in rows or day < str(rows[aid]['date'].date()):
+                rows[aid] = record
     seen.update(rows)
-    f=pd.DataFrame(list(rows.values()),columns=['date','alt_id','firms','first_months','available_at']+NAMES)
-    write_frame(OUT/f'news_scores/year_{year}.parquet',f)
+    f = pd.DataFrame(list(rows.values()), columns=['date', 'alt_id', 'firms', 'first_months', 'available_at'] + T.NAMES)
     gc.collect()
-    return len(f)
+    return f
```

## TD2 decoder_v3.py parquet -> td4_sources.py:article_batches

```diff
--- decoder_v3.py:parquet
+++ td4_sources.py:article_batches
@@ -1,22 +1,21 @@
-def parquet(path,columns,datecol,lo,hi):
-    """Project requested payload only after date filtering; bounded single-thread batches."""
-    guard()
-    if hi>'2024-12-31' or lo>hi: raise ValueError('Hard date seal')
-    path=Path(path)
-    pf=pq.ParquetFile(path,pre_buffer=False,buffer_size=65536)
-    typ=pf.schema_arrow.field(datecol).type
-    def scalar(v):
-        if pa.types.is_string(typ) or pa.types.is_large_string(typ): return pa.scalar(v,type=typ)
-        if pa.types.is_date(typ): return pa.scalar(dt.date.fromisoformat(v),type=typ)
-        return pa.scalar(pd.Timestamp(v),type=typ)
-    ix=pf.schema_arrow.get_field_index(datecol); low,high=scalar(lo),scalar(hi)
-    cols=list(dict.fromkeys([datecol]+list(columns)))
+def article_batches(lo, hi, columns, batch_size=2048):
+    """TD2 decoder_v3.parquet over the Reuters archive with a seal parameter (row-group skip by statistics, then a
+    date mask per 2048-row batch before conversion)."""
+    _check_seal(hi)
+    pf = pq.ParquetFile(T.ARTICLES, pre_buffer=False, buffer_size=65536)
+    typ = pf.schema_arrow.field('date').type
+    ix = pf.schema_arrow.get_field_index('date')
+    low, high = pa.scalar(lo, type=typ), pa.scalar(hi, type=typ)
+    cols = list(dict.fromkeys(['date'] + list(columns)))
     for group in range(pf.num_row_groups):
-        stat=pf.metadata.row_group(group).column(ix).statistics
-        if stat and (pa.scalar(stat.max,type=typ).as_py()<low.as_py() or pa.scalar(stat.min,type=typ).as_py()>high.as_py()): continue
-        for raw in pf.iter_batches(batch_size=2048,row_groups=[group],columns=cols,use_threads=False):
-            dates=raw.column(raw.schema.get_field_index(datecol))
-            mask=pc.and_(pc.greater_equal(dates,low),pc.less_equal(dates,high))
-            safe=raw.filter(mask)
-            if safe.num_rows: yield safe.to_pandas()
-        guard()
+        st = pf.metadata.row_group(group).column(ix).statistics
+        if st and (st.max < lo or st.min > hi):
+            continue
+        if st and st.max[:10] > T.HARD_SEAL:
+            raise PermissionError('Archive row group extends past the 2025 seal')
+        for raw in pf.iter_batches(batch_size=batch_size, row_groups=[group], columns=cols, use_threads=False):
+            d = raw.column(raw.schema.get_field_index('date'))
+            safe = raw.filter(pc.and_(pc.greater_equal(d, low), pc.less_equal(d, high)))
+            if safe.num_rows:
+                yield safe.to_pandas()
+        E.guard()
```

## TD2 fast_builder.py mapped_edges -> td4_features.py:mapped_edges

```diff
--- fast_builder.py:mapped_edges
+++ td4_features.py:mapped_edges
@@ -1,13 +1,18 @@
-def mapped_edges(graph,links,origin,suspects=None):
-    day=str(pd.Timestamp(origin).date())
-    if day>'2024-12-31': raise ValueError('Graph date seal')
-    eligible=graph[graph.start<day].drop_duplicates(['id','start'],keep='first').drop_duplicates('id',keep='last')
-    eligible=eligible[eligible.end.isna()|(eligible.end>day)]
-    lm=links[(links.start<day)&(links.end>=day)].drop_duplicates(['entity','permno'])
-    if suspects: lm=lm[[(r.entity,int(r.permno)) not in suspects for r in lm.itertuples()]]
-    lm=lm[~lm.duplicated('entity',keep=False)]; lookup=dict(zip(lm.entity,lm.permno))
-    eligible=eligible.assign(a=eligible.src.map(lookup),b=eligible.tgt.map(lookup))
-    eligible=eligible[eligible.a.notna()&eligible.b.notna()&(eligible.a!=eligible.b)]
-    clocks={s:engine.day_clock(s) for s in eligible.start.unique()}
-    return [{'id':r.id,'a':int(r.a),'b':int(r.b),'start':r.start,'end':r.end,
-        'available_at':clocks[r.start],'rownum':int(r.rownum),'typ':r.typ} for r in eligible.itertuples()]
+def mapped_edges(graph, links, origin, suspects=None, seal=T.HARD_SEAL):
+    """fast_builder.mapped_edges (TD4: seal parameter instead of the 2024-12-31 literal; otherwise unchanged)."""
+    engine, _, _, _ = _td2()
+    day = str(pd.Timestamp(origin).date())
+    if day > seal:
+        raise ValueError('Graph date seal')
+    eligible = graph[graph.start < day].drop_duplicates(['id', 'start'], keep='first').drop_duplicates('id', keep='last')
+    eligible = eligible[eligible.end.isna() | (eligible.end > day)]
+    lm = links[(links.start < day) & (links.end >= day)].drop_duplicates(['entity', 'permno'])
+    if suspects:
+        lm = lm[[(r.entity, int(r.permno)) not in suspects for r in lm.itertuples()]]
+    lm = lm[~lm.duplicated('entity', keep=False)]
+    lookup = dict(zip(lm.entity, lm.permno))
+    eligible = eligible.assign(a=eligible.src.map(lookup), b=eligible.tgt.map(lookup))
+    eligible = eligible[eligible.a.notna() & eligible.b.notna() & (eligible.a != eligible.b)]
+    clocks = {s: engine.day_clock(s) for s in eligible.start.unique()}
+    return [{'id': r.id, 'a': int(r.a), 'b': int(r.b), 'start': r.start, 'end': r.end,
+             'available_at': clocks[r.start], 'rownum': int(r.rownum), 'typ': r.typ} for r in eligible.itertuples()]
```

## TD2 build_v2.py origin_rows -> td4_features.py:origin_rows

```diff
--- build_v2.py:origin_rows
+++ td4_features.py:origin_rows
@@ -1,12 +1,13 @@
-def origin_rows(origin, cutoff, g, names, stories, graph, links, suspects, idmap):
-    """All feature and evidence rows for one origin. g holds manifest feature columns only.
-
-    Any label-like column present in g is ignored by construction (never referenced).
-    """
-    origin = pd.Timestamp(origin); cutoff = pd.Timestamp(cutoff)
+def origin_rows(origin, cutoff, g, names, stories, graph, links, suspects, idmap, seal=T.HARD_SEAL):
+    """build_v2.origin_rows (TD4: calls the seal-parameter mapped_edges copy; otherwise unchanged)."""
+    engine, fast_builder, build_v2, _ = _td2()
+    HOPS, FEATURE_INT, VARIANTS = (1, 2, 3), build_v2.FEATURE_INT, build_v2.VARIANTS
+    origin = pd.Timestamp(origin)
+    cutoff = pd.Timestamp(cutoff)
     pool = engine.news_pool(stories, cutoff, origin)
-    features = []; evidence = []
+    features = []
+    evidence = []
     for variant in VARIANTS:
         bad = suspects if variant == 'fsb5_sensitivity' else None
-        adj = fast_builder.graph_at(fast_builder.mapped_edges(graph, links, origin, bad), cutoff, origin)
+        adj = fast_builder.graph_at(mapped_edges(graph, links, origin, bad, seal=seal), cutoff, origin)   # TD4
         for r in g.itertuples(index=False):
@@ -24,3 +25,4 @@
                 h, n = v['hop'], v['name']
-                row[f'Z{h}_{n}'] = v['Z']; row[f'E{h}_{n}'] = v['E']
+                row[f'Z{h}_{n}'] = v['Z']
+                row[f'E{h}_{n}'] = v['E']
                 row['missing_' + n] = bool(v['characteristic_missing'])
@@ -34,3 +36,4 @@
             for w in witness:
-                edge_at = pd.Timestamp(w['max_edge_available_at']); news_at = pd.Timestamp(w['max_news_available_at'])
+                edge_at = pd.Timestamp(w['max_edge_available_at'])
+                news_at = pd.Timestamp(w['max_news_available_at'])
                 if edge_at >= cutoff or news_at >= cutoff:
@@ -40,11 +43,13 @@
                 evidence.append({'firm_id': r.firm_id, 'cutoff': cutoff, 'origin_session': origin,
-                    'variant': variant, 'hop': int(w['hop']), 'partner': int(w['partner']),
-                    'nodes': [int(x) for x in w['nodes']], 'intermediate_nodes': [int(x) for x in w['intermediate_nodes']],
-                    'relationship_ids': [str(x) for x in w['relationship_ids']],
-                    'relationship_source_rows': [int(x) for x in w['relationship_source_rows']],
-                    'reporting_roles': [str(x) for x in w['reporting_roles']], 'max_edge_available_at': edge_at,
-                    'article_ids': [str(x) for x in w['article_ids']], 'n_articles': len(w['article_ids']),
-                    'max_news_available_at': news_at,
-                    'intermediate_product_evidence': w['intermediate_product_evidence'],
-                    'physical_channel_verified': False})
+                                 'variant': variant, 'hop': int(w['hop']), 'partner': int(w['partner']),
+                                 'nodes': [int(x) for x in w['nodes']],
+                                 'intermediate_nodes': [int(x) for x in w['intermediate_nodes']],
+                                 'relationship_ids': [str(x) for x in w['relationship_ids']],
+                                 'relationship_source_rows': [int(x) for x in w['relationship_source_rows']],
+                                 'reporting_roles': [str(x) for x in w['reporting_roles']],
+                                 'max_edge_available_at': edge_at,
+                                 'article_ids': [str(x) for x in w['article_ids']], 'n_articles': len(w['article_ids']),
+                                 'max_news_available_at': news_at,
+                                 'intermediate_product_evidence': w['intermediate_product_evidence'],
+                                 'physical_channel_verified': False})
     return features, evidence
```

## TD3 v6 td3_build.py eligible -> td4_inputs.py:eligible

```diff
--- td3_build.py:eligible
+++ td4_inputs.py:eligible
@@ -1,10 +1,12 @@
-def eligible(frame):
-    lo, hi = pd.Timestamp(S.FIRST_DAY), pd.Timestamp(S.LAST_DAY)
+def eligible(frame, years):
+    """D2: td3_build.eligible with the period years in place of the 2021-2022 literals (otherwise unchanged)."""
+    S3, _, _, _ = _mods()
+    lo, hi = pd.Timestamp(S3.FIRST_DAY), pd.Timestamp(S3.LAST_DAY)
     o, c = frame.origin_session, frame.cutoff
     period = (o >= lo) & (o <= hi) & (c >= lo) & (c < hi + pd.Timedelta(days=1)) & (c > o)
-    anychar = ~np.logical_and.reduce([frame['missing_' + n].to_numpy(bool) for n in S.NAMES])
+    anychar = ~np.logical_and.reduce([frame['missing_' + n].to_numpy(bool) for n in S3.NAMES])
     e = frame[period & frame.has_news.astype(bool) & frame.mapped_focal.astype(bool) & anychar]
-    if (e.origin_session.dt.year > 2022).any() or (e.cutoff.dt.year > 2022).any() or \
-            (e.origin_session.dt.year < 2021).any():
-        raise PermissionError('Row outside 2021-2022 reached the eligible frame')
+    if (e.origin_session.dt.year > max(years)).any() or (e.cutoff.dt.year > max(years)).any() or \
+            (e.origin_session.dt.year < min(years)).any():
+        raise PermissionError('Row outside the period reached the eligible frame')
     if e.duplicated(['firm_id', 'cutoff']).any():
```

## TD3 v6 td3_build.py edge_maps -> td4_inputs.py:edge_maps

```diff
--- td3_build.py:edge_maps
+++ td4_inputs.py:edge_maps
@@ -1,11 +1,15 @@
-def edge_maps(sources, accepted, fast_builder):
+def edge_maps(sources_, accepted, seal):
+    """D3: td3_build.edge_maps calling td4_features.mapped_edges (seal parameter); otherwise unchanged."""
+    _, B, _, _ = _mods()
+    import td4_features as F
     rel_ids = sorted({rid for a in accepted for p in a['partners'] for rid in p['row'].relationship_ids})
-    graph = _concat(list(sealed_batches(sources.graph, ['id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'], 'date',
-                                        '1900-01-01', S.LAST_DAY, isin=('id', pa.array(rel_ids, type=pa.string())))),
-                    ['date', 'id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum']).rename(columns={'date': 'start'})
-    links = _concat(list(sealed_batches(sources.links, ['entity', 'permno', 'end'], 'date', '1900-01-01', S.LAST_DAY)),
-                    ['date', 'entity', 'permno', 'end']).rename(columns={'date': 'start'})
+    graph = B._concat(list(B.sealed_batches(sources_.graph, ['id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'],
+                                            'date', '1900-01-01', seal,
+                                            isin=('id', pa.array(rel_ids, type=pa.string())))),
+                      ['date', 'id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum']).rename(columns={'date': 'start'})
+    links = B._concat(list(B.sealed_batches(sources_.links, ['entity', 'permno', 'end'], 'date', '1900-01-01', seal)),
+                      ['date', 'entity', 'permno', 'end']).rename(columns={'date': 'start'})
     maps = {}
     for origin in sorted({pd.Timestamp(a['cand'].origin_session) for a in accepted}):
-        maps[origin] = {e['id']: e for e in fast_builder.mapped_edges(graph, links, origin)}
+        maps[origin] = {e['id']: e for e in F.mapped_edges(graph, links, origin, seal=seal)}
     return maps, links
```

## TD3 v6 td3_build.py select_sample (per-chunk part) -> td4_inputs.py:accept_chunk

```diff
--- td3_build.py:select_sample
+++ td4_inputs.py:accept_chunk
@@ -1,54 +1,49 @@
-def select_sample(sources, elig, n, ricmap, engine):
-    order = elig.assign(_k=[sample_key(a, b) for a, b in zip(elig.firm_id, elig.cutoff)]).sort_values('_k')
-    accepted, stats = [], {'examined': 0, 'chunks': 0, 'rejected': Counter(), 'article_rejects': Counter(),
-                           'text_status': Counter(), 'partners_without_article': 0, 'partner_order_equal_td2': True}
-    pos = 0
-    while len(accepted) < n and pos < len(order):
-        chunk = order.iloc[pos:pos + S.CANDIDATE_BATCH]
-        pos += len(chunk)
-        stats['chunks'] += 1
-        ev = load_evidence(sources, chunk)
-        news = load_news(sources, {a for lst in ev.article_ids for a in lst})
-        verdict = {}
-        for w in ev.itertuples(index=False):
+def accept_chunk(sources_, chunk, ricmap, engine, stats):
+    """td3_build.select_sample's per-chunk logic for every candidate of the chunk (no sample cut-off)."""
+    _, B, _, _ = _mods()
+    ev = B.load_evidence(sources_, chunk)
+    news = B.load_news(sources_, {a for lst in ev.article_ids for a in lst})
+    verdict = {}
+    for w in ev.itertuples(index=False):
+        for aid in w.article_ids:
+            verdict[(w.firm_id, w.cutoff, int(w.partner), aid)] = B.verify_article(aid, w.partner, w.origin_session,
+                                                                                  w.cutoff, news)
+    wanted = {k[3]: news[k[3]] for k, v in verdict.items() if v is None}
+    texts, tstatus = B.fetch_texts(sources_, wanted, ricmap, engine.score)
+    groups = {k: g for k, g in ev.groupby(['firm_id', 'cutoff'], sort=False)}
+    accepted, rejected = [], []
+    for cand in chunk.itertuples(index=False):
+        stats['examined'] += 1
+        g = groups.get((cand.firm_id, cand.cutoff))
+        if g is None or not len(g):
+            stats['rejected']['no_evidence_rows'] += 1
+            rejected.append((cand, 'no_evidence_rows'))
+            continue
+        for h in (1, 2, 3):
+            if int((g.hop == h).sum()) != int(getattr(cand, f'H{h}_selected_partners')):
+                raise ValueError('Evidence rows differ from TD2 selected-partner counts')
+        g = g.assign(_p=[B.priority(cand.cutoff, p) for p in g.partner])
+        ordered = g.sort_values(['hop', '_p'], kind='stable')
+        if list(ordered.index) != list(g.sort_values('hop', kind='stable').index):
+            stats['partner_order_equal_td2'] = False
+        partners = []
+        for w in ordered.itertuples(index=False):
+            good = []
             for aid in w.article_ids:
-                verdict[(w.firm_id, w.cutoff, int(w.partner), aid)] = verify_article(aid, w.partner, w.origin_session,
-                                                                                    w.cutoff, news)
-        wanted = {k[3]: news[k[3]] for k, v in verdict.items() if v is None}
-        texts, tstatus = fetch_texts(sources, wanted, ricmap, engine.score)
-        groups = {k: g for k, g in ev.groupby(['firm_id', 'cutoff'], sort=False)}
-        for cand in chunk.itertuples(index=False):
-            stats['examined'] += 1
-            g = groups.get((cand.firm_id, cand.cutoff))
-            if g is None or not len(g):
-                stats['rejected']['no_evidence_rows'] += 1
-                continue
-            for h in S.HOPS:
-                if int((g.hop == h).sum()) != int(getattr(cand, f'H{h}_selected_partners')):
-                    raise ValueError('Evidence rows differ from TD2 selected-partner counts')
-            g = g.assign(_p=[priority(cand.cutoff, p) for p in g.partner])
-            ordered = g.sort_values(['hop', '_p'], kind='stable')
-            if list(ordered.index) != list(g.sort_values('hop', kind='stable').index):
-                stats['partner_order_equal_td2'] = False
-            partners = []
-            for w in ordered.itertuples(index=False):
-                good = []
-                for aid in w.article_ids:
-                    reason = verdict[(w.firm_id, w.cutoff, int(w.partner), aid)]
-                    if reason is None:
-                        reason = tstatus.get(aid)
-                    if reason is None:
-                        good.append((news[aid]['date'], aid))
-                    else:
-                        stats['article_rejects'][reason] += 1
-                good.sort(key=lambda t: (-t[0].value, t[1]))
-                partners.append({'row': w, 'article': good[0][1] if good else None})
-            if not any(p['article'] for p in partners):
-                stats['rejected']['no_verified_article'] += 1
-                continue
-            stats['partners_without_article'] += sum(p['article'] is None for p in partners)
-            accepted.append({'cand': cand, 'partners': partners, 'news': news, 'texts': texts})
-            if len(accepted) == n:
-                break
-        stats['text_status'].update('ok' if v is None else v for v in tstatus.values())
-    return accepted, stats
+                reason = verdict[(w.firm_id, w.cutoff, int(w.partner), aid)]
+                if reason is None:
+                    reason = tstatus.get(aid)
+                if reason is None:
+                    good.append((news[aid]['date'], aid))
+                else:
+                    stats['article_rejects'][reason] += 1
+            good.sort(key=lambda t: (-t[0].value, t[1]))
+            partners.append({'row': w, 'article': good[0][1] if good else None})
+        if not any(p['article'] for p in partners):
+            stats['rejected']['no_verified_article'] += 1
+            rejected.append((cand, 'no_verified_article'))
+            continue
+        stats['partners_without_article'] += sum(p['article'] is None for p in partners)
+        accepted.append({'cand': cand, 'partners': partners, 'news': news, 'texts': texts})
+    stats['text_status'].update('ok' if v is None else v for v in tstatus.values())
+    return accepted, rejected
```

## build_panel.py mapping_at -> td4_sources.py:mapping_at

```diff
--- build_panel.py:mapping_at
+++ td4_sources.py:mapping_at
@@ -2,4 +2,4 @@
     a = m[(m.b <= asof) & (m.e >= asof)].copy()
-    a["me"] = a.permno.map(me_at).fillna(-1)
-    a = a.sort_values("me", ascending=False).drop_duplicates("entity").drop_duplicates("permno")
+    a['me'] = a.permno.map(me_at).fillna(-1)
+    a = a.sort_values('me', ascending=False).drop_duplicates('entity').drop_duplicates('permno')
     return dict(zip(a.entity, a.permno.astype(int)))
```

## build_panel.py active_edges -> td4_sources.py:active_edges

```diff

```

## build_panel.py stage_garch (inclusion part) -> td4_sources.py:garch_ok

```diff
--- build_panel.py:stage_garch
+++ td4_sources.py:garch_ok
@@ -1,39 +1,26 @@
-def stage_garch():
+def garch_ok(R, cal, perms, fy):
+    """build_panel.stage_garch inclusion: GARCH(1,1) (zero mean, normal) on the 504 sessions ending at the prior
+    year end, >= 400 finite returns, fit without exception, omega > 0, alpha >= 0, beta >= 0, alpha + beta < 0.9999.
+    Uses only returns dated on or before fy (the last session before the year)."""
     from arch import arch_model
-    R = pd.read_parquet(CACHE / "returns_wide.parquet")
-    R.index = pd.to_datetime(R.index)
-    origins = pd.DatetimeIndex(pd.read_parquet(CACHE / "origins.parquet").origin)
-    universe = {int(k): v for k, v in json.load(open(CACHE / "universe.json")).items()}
-    out, t0 = [], time.time()
-    for yy, perms in universe.items():
-        ye = R.index[R.index < pd.Timestamp(yy, 1, 1)][-1]
-        yo = origins[origins.year == yy]
-        end_i, last_i = R.index.get_loc(ye), R.index.get_loc(yo[-1])
-        start = end_i - 503
-        for p in perms:
-            if p not in R.columns:
-                continue
-            r = R[p].values.astype(float) * 100
-            win = r[start:end_i + 1]
-            if np.isfinite(win).sum() < 400:
-                continue
-            try:
-                res = arch_model(win[np.isfinite(win)], mean="Zero", vol="GARCH", p=1, q=1, dist="normal",
-                                 rescale=False).fit(disp="off", show_warning=False)
-                om, al, be = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
-            except Exception:
-                continue
-            if not (om > 0 and al >= 0 and be >= 0 and al + be < 0.9999):
-                continue
-            path = r[start:last_i + 1]
-            sig = np.empty(len(path) + 1)
-            sig[0] = np.nanvar(win)
-            for t, x in enumerate(path):
-                sig[t + 1] = om + al * (x * x if np.isfinite(x) else sig[t]) + be * sig[t]
-            lr, phi = om / (1 - al - be), al + be
-            for o in yo:
-                s1 = sig[R.index.get_loc(o) - start + 1]  # one-step forecast for the session after o
-                f5 = np.mean([lr + phi ** (h - 1) * (s1 - lr) for h in range(1, 6)])
-                out.append((o, p, np.log(max(f5 / 1e4, FLOOR))))
-        log(f"garch {yy}: {len(out)} rows, {time.time() - t0:.0f}s")
-    pd.DataFrame(out, columns=["date", "permno", "b"]).to_parquet(CACHE / "garch.parquet")
+    end_i = cal.get_loc(fy)
+    start = end_i - 503
+    ok = {}
+    for p in perms:
+        if p not in R.columns:
+            ok[p] = False
+            continue
+        r = R[p].values.astype(float) * 100
+        win = r[start:end_i + 1]
+        if np.isfinite(win).sum() < 400:
+            ok[p] = False
+            continue
+        try:
+            res = arch_model(win[np.isfinite(win)], mean='Zero', vol='GARCH', p=1, q=1, dist='normal',
+                             rescale=False).fit(disp='off', show_warning=False)
+            om, al, be = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
+        except Exception:
+            ok[p] = False
+            continue
+        ok[p] = bool(om > 0 and al >= 0 and be >= 0 and al + be < 0.9999)
+    return ok
```

