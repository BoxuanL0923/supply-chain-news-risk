"""Append-only repair of the bounded FSB5 decoder and resumable preparation."""
import sys
import time
import decoder_v2
sys.modules['decoder']=decoder_v2
import evaluation_v2
sys.modules['evaluation']=evaluation_v2
import pipeline
from runtime_v3 import OUT,TD1,new_json,sha,peak,guard
from spec import VERSION,RULES,LEXICONS

def resume_prepare():
    if not (OUT/'EXPOSURE_DICTIONARY_v1.json').exists():
        return _original_prepare()
    if (OUT/'PREPARED_v1.json').exists(): raise FileExistsError('Preparation already complete')
    pipeline.require_synthetic(); start=time.perf_counter(); io=decoder_v2
    plan=io.json_read(OUT/'EXPOSURE_DICTIONARY_v1.json')
    if plan['rules']!=RULES or plan['lexicons']!=LEXICONS: raise ValueError('Partial plan differs from current specification')
    required=['risk_menu.csv','sources/graph_october_sealed.parquet',
              'sources/factset_links.parquet','sources/quarantine_ids_v1.json']
    if not all((OUT/p).exists() for p in required): raise PermissionError('Partial preparation is not the expected safe resume point')
    suspects=OUT/'sources/identity_sensitivity_v1.json'
    if not suspects.exists():
        new_json(suspects,{'links':[list(x) for x in sorted(io.sensitivity_links())]},timed=False)
    mappingpath=OUT/'sources/reuters_identity_development_v1.json'
    if not mappingpath.exists():
        new_json(mappingpath,{'mapping':io.ric_mapping(2022)},timed=False)
    mapping=io.json_read(mappingpath)['mapping']; seen=set(); news_counts={}
    for year in range(2014,2023):
        guard(); p=OUT/f'news_scores/year_{year}.parquet'
        if not p.exists(): io.build_news(year,mapping,seen)
        f=io.read_frame(p,['alt_id'],'date',f'{year}-01-01',f'{year}-12-31')
        seen.update(f.alt_id); news_counts[str(year)]=len(f)
        print('NEWS',year,len(f),flush=True)
    source_hashes={str(p):sha(p) for p in sorted((OUT/'sources').glob('*')) if p.is_file()}
    source_hashes.update({str(p):sha(p) for p in sorted((OUT/'news_scores').glob('*'))})
    new_json(OUT/'PREPARED_v1.json',{'version':VERSION,'news_counts':news_counts,'source_sha256':source_hashes,
        'source_graph_audit_sha256':sha(io.GRAPH),'FSB5_sha256':sha(io.FS5),
        'td1_freeze_sha256':sha(TD1/'MODEL_FREEZE_v3.json'),'resume_after':'bounded CSV field limit',
        'elapsed_seconds':time.perf_counter()-start,**peak()})
    print('PREPARED',str(OUT/'risk_menu.csv'),flush=True)

_original_prepare=pipeline.prepare
pipeline.prepare=resume_prepare
