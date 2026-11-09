"""Metadata and date-first local inspection; never load existing outcome columns."""
from runtime import limit, guard
limit()
import importlib.util
import json
from pathlib import Path
import pyarrow.parquet as pq

for module in ['numpy','pandas','pyarrow','sklearn','torch','psutil','pypdf','pdfplumber','duckdb','pytest']:
    print(module, bool(importlib.util.find_spec(module)))
for path in [Path('C:/OptResearch/results/hist_panel_2015_2024/panel.parquet'),
             Path('C:/OptResearch/data/macro/exp_crsp_daily_compact.parquet'),
             Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/ctl_compustat_quarterly_labor.parquet')]:
    guard(); p = pq.ParquetFile(path)
    print(str(path), p.metadata.num_rows, p.num_row_groups, p.schema_arrow)
    for j in [0, p.num_row_groups-1]:
        print('rowgroup',j, [(p.metadata.row_group(j).column(i).path_in_schema,
                str(p.metadata.row_group(j).column(i).statistics))
               for i in range(p.metadata.num_columns) if p.metadata.row_group(j).column(i).path_in_schema in ['date','datadate']])
try:
    from pypdf import PdfReader
    p = PdfReader('C:/Users/Boxuan/.claude/uploads/c5308263-d41b-41ec-bcae-26e718b5a584/1219673d-TODO.pdf')
    print('TODO math', '\n'.join(x.extract_text() for x in p.pages))
except ImportError:
    print('No PDF parser installed')
