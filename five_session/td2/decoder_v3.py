"""All private payload readers live here. Evaluation returns require admission token."""
import csv
csv.field_size_limit(8 * 1024 * 1024)  # bounded FSB5 interval-list fields
import datetime as dt
import gc
import json
import pickle
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from runtime_v3 import OUT,TD1,guard,sha
from spec import NAMES
from engine import day_clock,score

PROCESSED=Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed')
FSB=Path('C:/OptResearch/results/sc_risk/t7v3_20261028/model01/mech_round3/fsb')
GRAPH=FSB/'fsb3/FSB3_AUDIT_CACHE_v1.sqlite'
FS5=FSB/'fsb5/FSB5_FSMAP_CANDIDATES_v1.csv'
FSMAP=PROCESSED/'factset_to_permno.csv'
ARTICLE=PROCESSED/'articles_classified.parquet'
_evaluation_token=None

def parquet(path,columns,datecol,lo,hi):
    """Project requested payload only after date filtering; bounded single-thread batches."""
    guard()
    if hi>'2024-12-31' or lo>hi: raise ValueError('Hard date seal')
    path=Path(path)
    pf=pq.ParquetFile(path,pre_buffer=False,buffer_size=65536)
    typ=pf.schema_arrow.field(datecol).type
    def scalar(v):
        if pa.types.is_string(typ) or pa.types.is_large_string(typ): return pa.scalar(v,type=typ)
        if pa.types.is_date(typ): return pa.scalar(dt.date.fromisoformat(v),type=typ)
        return pa.scalar(pd.Timestamp(v),type=typ)
    ix=pf.schema_arrow.get_field_index(datecol); low,high=scalar(lo),scalar(hi)
    cols=list(dict.fromkeys([datecol]+list(columns)))
    for group in range(pf.num_row_groups):
        stat=pf.metadata.row_group(group).column(ix).statistics
        if stat and (pa.scalar(stat.max,type=typ).as_py()<low.as_py() or pa.scalar(stat.min,type=typ).as_py()>high.as_py()): continue
        for raw in pf.iter_batches(batch_size=2048,row_groups=[group],columns=cols,use_threads=False):
            dates=raw.column(raw.schema.get_field_index(datecol))
            mask=pc.and_(pc.greater_equal(dates,low),pc.less_equal(dates,high))
            safe=raw.filter(mask)
            if safe.num_rows: yield safe.to_pandas()
        guard()

def read_frame(path,columns,datecol,lo,hi):
    chunks=list(parquet(path,columns,datecol,lo,hi))
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=list(dict.fromkeys([datecol]+columns)))

def write_frame(path,frame):
    path=Path(path).resolve(); guard()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Private output root')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: frame.to_parquet(f,index=False,compression='zstd')

def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_td1():
    freeze=json_read(TD1/'MODEL_FREEZE_v3.json')
    if sha(TD1/'model_frozen_v2.pkl')!=freeze['model_sha256']: raise ValueError('TD1 model hash')
    return freeze

def td1_model():
    verify_td1()
    with (TD1/'model_frozen_v2.pkl').open('rb') as f: return pickle.load(f)

def manifest(year,labels=False):
    if not 2015<=year<=2024 or (labels and year>2022): raise PermissionError('Development labels only')
    freeze=verify_td1(); path=TD1/f'firm_week_manifest/year_{year}.parquet'
    if sha(path)!=freeze['manifest_sha256'][str(path)]: raise ValueError('TD1 manifest hash')
    columns=['firm_id','cutoff','origin_session','eligible_features']+NAMES+[m+'__available_at' for m in NAMES]
    if labels: columns+=['y','label_end','fit_included','selection_included','refit_included']
    f=read_frame(path,columns,'origin_session',f'{year}-01-01',f'{year}-12-31')
    if labels and (pd.to_datetime(f.label_end.dropna())>pd.Timestamp('2022-12-31')).any():
        raise ValueError('Development label end')
    return f

def baseline(year):
    freeze=verify_td1(); path=TD1/f'predictions_v2/year_{year}.parquet'
    if sha(path)!=freeze['prediction_sha256'][str(path)]: raise ValueError('TD1 baseline hash')
    return read_frame(path,['firm_id','cutoff','b'],'origin_session',f'{year}-01-01',f'{year}-12-31')

def development_baseline(year,model):
    if year>2022: raise PermissionError('Selection model only for development')
    cols=['firm_id','cutoff','origin_session']+model['features']
    # Source hash already checked by manifest(). Only past inputs are projected here.
    f=read_frame(TD1/f'firm_week_manifest/year_{year}.parquet',cols,'origin_session',f'{year}-01-01',f'{year}-12-31')
    x=f[model['features']].to_numpy(float); p=model['preprocess']; missing=~np.isfinite(x)
    x=(np.where(missing,p['median'],x)-p['mean'])/p['scale']
    x=np.column_stack([x,missing.astype(float)])
    f['b_development']=model['selection_mlp'].predict(x)*model['yscale']+model['ymean']
    return f[['firm_id','cutoff','b_development']]

def graph_rows():
    db=sqlite3.connect(GRAPH.as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-32768')
    db.set_authorizer(lambda action,a,b,c,d: sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK)
    quarantine={r[0] for r in db.execute("SELECT id FROM classes WHERE category='added'")}
    if len(quarantine)!=91869: raise ValueError('FSB3 quarantine must contain 91869 IDs')
    rows=[]
    sql="SELECT id,start,end,status,typ,src,tgt,rownum FROM o WHERE start<='2024-12-31' AND typ IN ('CUSTOMER','SUPPLIER') ORDER BY id,start,rownum"
    for r in db.execute(sql):
        if r[0] in quarantine or r[3]=='unparseable': continue
        if r[2] and r[2]>'2024-12-31': raise ValueError('Unsealed graph end')
        rows.append(r)
    db.close()
    return pd.DataFrame(rows,columns=['id','start','end','status','typ','src','tgt','rownum']),quarantine

def mapping():
    rows=[]
    with FSMAP.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            # Inspect start before using identity values; later ends censored.
            start=r['link_bdate'][:10]
            if not start or start>'2024-12-31': continue
            try: p=int(float(r['PERMNO']))
            except (ValueError,TypeError): continue
            end=r['link_edate'][:10]
            rows.append((r['FACTSET_ENTITY_ID'],p,start,min(end or '2024-12-31','2024-12-31')))
    return pd.DataFrame(rows,columns=['entity','permno','start','end']).drop_duplicates()

def sensitivity_links():
    result=set()
    with FS5.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['bridge_start'][:10]>'2024-12-31': continue
            result.add((r['entity_id'],int(float(r['A_PERMNO']))))
    return result

def mapped_edges(graph,links,origin,suspects=None):
    day=str(pd.Timestamp(origin).date())
    if day>'2024-12-31': raise ValueError('Graph date seal')
    eligible=graph[graph.start<day].drop_duplicates(['id','start'],keep='first').drop_duplicates('id',keep='last')
    eligible=eligible[eligible.end.isna()|(eligible.end>day)]
    lm=links[(links.start<day)&(links.end>=day)].drop_duplicates(['entity','permno'])
    if suspects:
        lm=lm[[ (r.entity,int(r.permno)) not in suspects for r in lm.itertuples() ]]
    lm=lm[~lm.duplicated('entity',keep=False)]
    lookup=dict(zip(lm.entity,lm.permno))
    result=[]
    for r in eligible.itertuples():
        a,b=lookup.get(r.src),lookup.get(r.tgt)
        if a is None or b is None or a==b: continue
        result.append({'id':r.id,'a':int(a),'b':int(b),'start':r.start,
            'end':r.end,'available_at':day_clock(r.start),'rownum':int(r.rownum),'typ':r.typ})
    return result

def ric_mapping(endyear):
    if endyear not in (2022,2024): raise ValueError('Identity freeze period')
    pf=pq.ParquetFile(PROCESSED/'regression_panel.parquet',pre_buffer=False)
    # Project keys and month only. No cached outcomes or other features.
    choices={}; first={}
    for raw in pf.iter_batches(batch_size=4096,columns=['ric','ym_int','permno'],use_threads=False):
        months=raw.column(1); safe=raw.filter(pc.and_(pc.greater_equal(months,201001),pc.less_equal(months,endyear*100+12)))
        for ric,month,p in zip(*[safe.column(i).to_pylist() for i in range(3)]):
            if ric is None or p is None: continue
            choices.setdefault(ric,set()).add(int(p)); first[ric]=min(first.get(ric,999999),month)
    return {ric:(next(iter(v)),first[ric]) for ric,v in choices.items() if len(v)==1}

def build_news(year,ricmap,seen):
    rows={}
    for batch in parquet(ARTICLE,['alt_id','rics','headline','body_short'],'date',f'{year}-01-01',f'{year}-12-31'):
        for r in batch.itertuples(index=False):
            aid=r.alt_id
            if not aid or aid in seen or not r.rics: continue
            day=r.date[:10]
            if day>f'{year}-12-31': raise ValueError('Article date seal')
            bindings=sorted({tuple(ricmap[t.strip()]) for t in r.rics.split(',') if t.strip() in ricmap})
            if not bindings: continue
            text=(r.headline or '')[:200]+(r.body_short or '')[:1000]
            record={'date':pd.Timestamp(day),'alt_id':aid,'firms':[p for p,m in bindings],
                'first_months':[m for p,m in bindings],'available_at':day_clock(day),**score(text)}
            if aid not in rows or day<str(rows[aid]['date'].date()): rows[aid]=record
        guard()
    seen.update(rows)
    f=pd.DataFrame(list(rows.values()),columns=['date','alt_id','firms','first_months','available_at']+NAMES)
    write_frame(OUT/f'news_scores/year_{year}.parquet',f)
    gc.collect()
    return len(f)

def news_for_year(year):
    chunks=[]
    for y in (year-1,year):
        lo=f'{year-1}-12-20' if y<year else f'{year}-01-01'
        p=OUT/f'news_scores/year_{y}.parquet'
        chunks.append(read_frame(p,['alt_id','firms','first_months','available_at']+NAMES,'date',lo,f'{y}-12-31'))
    return pd.concat(chunks,ignore_index=True)

def week_stories(news,origin):
    day=pd.Timestamp(origin); lower=day-pd.Timedelta(days=7)
    month=day.year*100+day.month
    f=news[(news.date>=lower)&(news.date<day)]; result=[]
    for r in f.itertuples():
        firms=[int(p) for p,m in zip(r.firms,r.first_months) if m<month]
        if not firms: continue
        result.append({'date':r.date,'alt_id':r.alt_id,'firms':firms,
            'available_at':r.available_at,'scores':{n:float(getattr(r,n)) for n in NAMES}})
    return result

def admitted_returns(year,token):
    if token is None or token is not _evaluation_token: raise PermissionError('No evaluation admission')
    if year not in (2023,2024): raise PermissionError('Evaluation years only')
    cols=['date','permno','ret','share_type']
    f=read_frame(TD1/f'inputs/crsp_{year}.parquet',cols,'date',f'{year}-01-01',f'{year}-12-31')
    if year==2023:
        f=pd.concat([f,read_frame(TD1/'inputs/crsp_2024.parquet',cols,'date','2024-01-01','2024-01-15')],ignore_index=True)
    return f
