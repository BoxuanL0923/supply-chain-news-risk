"""The only TD1 production source decoder. Date filtering precedes field access.

Arrow's scan predicate drops out-of-range rows before returning projected fields
to Python. A date-only pass is also made first. Compressed mixed-date pages may
be decoded by Arrow internally; no 2025 payload is returned or inspected.
"""
from pathlib import Path
import csv
import datetime as dt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from runtime import guard, OUT

ROOT = Path('C:/OptResearch')
SRC = Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed')
PANEL = ROOT/'results/hist_panel_2015_2024/panel.parquet'
CRSP = ROOT/'data/macro/exp_crsp_daily_compact.parquet'
QUARTER = SRC/'compustat_quarterly_full.csv'
EARNINGS = SRC/'earnings_dates.csv'
LINK = SRC/'gvkey_permno_link.csv'
JKP = SRC/'jkp_full_chars.parquet'
HARD_END = '2024-12-31'
SOURCE_PATHS = [PANEL, CRSP, QUARTER, EARNINGS, LINK, JKP]

def dated(path, cols, date_col='date', lo='2014-01-01', hi=HARD_END):
    guard()
    if hi > HARD_END:
        raise ValueError('2025 is prohibited')
    path = Path(path).resolve()
    allowed = [p.resolve() for p in SOURCE_PATHS]
    if path not in allowed and not path.is_relative_to(OUT.resolve()):
        raise ValueError('Source outside TD1 allowlist')
    if path==PANEL.resolve() and set(cols)-{'date','permno'}:
        raise ValueError('Legacy panel is admitted for keys only; cached outcomes/features prohibited')
    schema = pq.read_schema(path)
    typ = schema.field(date_col).type
    convert = str if pa.types.is_string(typ) else (dt.date.fromisoformat if pa.types.is_date(typ) else pd.Timestamp)
    filt = (ds.field(date_col) >= pa.scalar(convert(lo), type=typ)) & (
            ds.field(date_col) <= pa.scalar(convert(hi), type=typ))
    dataset = ds.dataset(path, format='parquet')
    # Only dates first, including no other fields from rejected rows.
    dates = dataset.to_table(columns=[date_col], filter=filt, use_threads=False)
    values = dates[date_col].to_pandas()
    if values.isna().any() or (pd.to_datetime(values) > HARD_END).any():
        raise ValueError('Date seal')
    del dates, values
    result = dataset.to_table(columns=list(dict.fromkeys([date_col]+list(cols))),
                              filter=filt, use_threads=False).to_pandas()
    if (pd.to_datetime(result[date_col]) > HARD_END).any():
        raise ValueError('Date seal after projection')
    return result

def csv_dated(path, cols, date_col='datadate', lo='2012-01-01', hi=HARD_END,
              second_date=None):
    """Stream CSV: reject using date positions before accessing other field values."""
    guard(); rows = []
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f); header = next(reader)
        di = header.index(date_col); ix = [header.index(c) for c in cols]
        si = header.index(second_date) if second_date else None
        for n, values in enumerate(reader):
            day = values[di][:10]
            if not (lo <= day <= min(hi, HARD_END)):
                continue
            if si is not None and values[si] and values[si][:10] > HARD_END:
                continue
            rows.append([values[i] for i in ix])
            if n % 50000 == 0:
                guard()
    return pd.DataFrame(rows, columns=cols)

def links():
    # This undated key-only table has no data date or outcome fields.
    l = pd.read_csv(LINK, usecols=['permno','gvkey']).dropna().astype('int64').drop_duplicates()
    p = l.groupby('gvkey').permno.transform('nunique')
    g = l.groupby('permno').gvkey.transform('nunique')
    return l[(p == 1) & (g == 1)]

def write_parquet(path, frame):
    guard(); path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Private outputs outside TD1 root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        frame.to_parquet(f, index=False, compression='zstd')

def schema_names(path):
    return pq.read_schema(path).names

def read_keys(path):
    path=Path(path).resolve()
    if path != (OUT/'inputs/identity.parquet').resolve():
        raise ValueError('Only the undated identity snapshot is admitted')
    return pq.read_table(path,columns=['gvkey','permno'],use_threads=False).to_pandas()

def read_frozen_model():
    """Trusted, locally produced pickle; caller verifies frozen SHA256 first."""
    import pickle
    with (OUT/'model_frozen_v2.pkl').open('rb') as stream:
        return pickle.load(stream)
