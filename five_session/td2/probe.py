"""Read schemas and pre-2025 audit metadata only; no label or return payload."""
from runtime import limit, OUT, new_json, peak
limit()
import sqlite3
from pathlib import Path
import pyarrow.parquet as pq

processed=Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed')
dbpath=Path('C:/OptResearch/results/sc_risk/t7v3_20261028/model01/mech_round3/fsb/fsb3/FSB3_AUDIT_CACHE_v1.sqlite')
db=sqlite3.connect(dbpath.as_uri()+'?mode=ro&immutable=1',uri=True)
result={'schemas':{},'sqlite':{}}
for table in ['o','classes']:
    result['sqlite'][table]=list(db.execute('PRAGMA table_info('+table+')'))
result['quarantine_count']=db.execute("SELECT count(*) FROM classes WHERE category='added'").fetchone()[0]
for path in [processed/'articles_classified.parquet',processed/'regression_panel.parquet',
             Path('D:/OptResearch_data/sc_risk/weekly_cocust_gates/article_gate/stories_2015.parquet')]:
    result['schemas'][str(path)]=str(pq.read_schema(path))
try:
    from pypdf import PdfReader
    pdf=Path('C:/Users/Boxuan/.claude/uploads/c5308263-d41b-41ec-bcae-26e718b5a584/1219673d-TODO.pdf')
    result['math_pdf_text']='\n'.join(p.extract_text() for p in PdfReader(pdf).pages)
except ImportError:
    result['math_pdf_status']='pypdf unavailable; Markdown equations inspected'
result.update(peak())
new_json(OUT/'SOURCE_PROBE_v1.json',result)
print(result)
