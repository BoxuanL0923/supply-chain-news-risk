"""qwen3.8-27b: audit dated vendor keyword support without promoting it to flow.

The local FactSet v1.4.3 guide pp.14-15 defines relationship keywords as search
leads, and summary overlaps as sector counts. p.17 profiles are product-line
descriptions without dates. None asserts intermediate input/output continuity.
This audit annotates every existing multihop receipt with available dated
keyword overlap; statuses and all frozen scoring prompts remain unchanged.
"""
from collections import Counter,defaultdict
import csv
from pathlib import Path
from .hop3_guard import ROOT,FACTSET,DATES,MODEL,Guard,read,write,sha,resources

SOURCE=FACTSET

def audit(root=ROOT):
    Guard().install();resources('continuity_vendor_fields')
    out=root/'hop_ablation_v2';cases=read(root/'cases.json')['cases']
    relations={c['ticker']:read(root/'relations'/(c['ticker']+'.json'))['relations'] for c in cases}
    edge_ids={str(s['edge_id']) for rs in relations.values() for r in rs for s in r['steps']}
    keyword_file=SOURCE/'scr_relationships_keyword.txt';records=[];scanned=0
    # This vendor export contains Latin-1 bytes (e.g. e9), not valid UTF-8.
    # Byte-preserving decoding avoids silently replacing evidence characters.
    with keyword_file.open(encoding='latin-1',newline='') as stream:
        for row in csv.DictReader(stream,delimiter='|'):
            scanned+=1
            if row['ID'] not in edge_ids or not row['START_DATE'] or row['START_DATE'][:10]>=DATES[-1]:continue
            end=row['END_DATE']
            if end and end.upper()!='NULL' and end[:10]<=DATES[0]:continue
            terms=sorted({row[f'RELATIONSHIP_KEYWORD{i}'].strip().lower() for i in range(1,11)
                          if row[f'RELATIONSHIP_KEYWORD{i}'].strip() and row[f'RELATIONSHIP_KEYWORD{i}'].upper()!='NULL'})
            records.append({'keyword_id':row['KEYWORD_ID'],'edge_id':row['ID'],
                'start_date':row['START_DATE'],'end_date':row['END_DATE'],'terms':terms})
    write(out/'dated_relationship_keywords_qwen3.8-27b.json',{'records':records,'source':str(keyword_file),
        'source_sha256':sha(keyword_file),'meaning':'Dated relationship retrieval keywords, not verified input/output flow facts'})
    counts=Counter()
    for day in DATES:
        terms=defaultdict(set)
        for record in records:
            if record['start_date'][:10]<day and (not record['end_date'] or record['end_date'].upper()=='NULL' or record['end_date'][:10]>day):
                terms[record['edge_id']].update(record['terms'])
        for case in [c for c in cases if c['date']==day]:
            rs=[r for r in relations[case['ticker']] if r['hops']>1]
            path=out/'continuity_audit'/(case['case_id'].replace(':','_')+'.json')
            value=read(path)
            if len(rs)!=len(value['paths']):raise ValueError('Continuity path audit length changed')
            for relation,result in zip(rs,value['paths']):
                if relation['path']!=result['path'] or result['status']!='path_only':
                    raise ValueError('Do not overwrite changed continuity decisions')
                edge_terms=[terms[str(step['edge_id'])] for step in relation['steps']]
                shared=[a&b for a,b in zip(edge_terms,edge_terms[1:])]
                result['partial_vendor_support']={'edge_ids':[str(s['edge_id']) for s in relation['steps']],
                    'edges_with_dated_keywords':sum(bool(v) for v in edge_terms),
                    'intermediate_shared_keywords':[sorted(v)[:10] for v in shared],
                    'qualifies_as_continuity':False,
                    'reason':'Search keywords do not state that an affected input continues into the outgoing product; profiles are undated and sector overlap is not product flow.'}
                key=(case['split'],relation['hops'])
                counts[key+('paths',)]+=1
                counts[key+('all_edges_have_keywords',)]+=int(all(edge_terms))
                counts[key+('every_intermediate_has_keyword_overlap',)]+=int(all(shared))
            value['vendor_field_audit']='Dated keywords checked; no product-flow verification inferred. See continuity_source_audit.json.'
            write(path,value)
        print(MODEL,'VENDOR_CONTINUITY_AUDIT',day,flush=True)
    summary={'keyword_rows_scanned':scanned,'matched_development_keyword_records':len(records),
        'path_support':[{'split':s,'hops':h,'metric':m,'n':v} for (s,h,m),v in sorted(counts.items())],
        'eligibility':'Publication START_DATE strictly before origin calendar date; END_DATE after origin or null; conservative with unspecified vendor timezone.',
        'text_encoding':'Latin-1 byte-preserving decode; no lossy replacement',
        'sources':{'keywords':{'path':str(keyword_file),'sha256':sha(keyword_file)},
                   'profiles':{'path':str(SOURCE/'ent_scr_profiles.txt'),'used':False,'reason':'No availability or effective date fields'},
                   'sector_summary':{'path':str(SOURCE/'scr_relationships_summary.txt'),'used':False,'reason':'Sector-overlap counts, no product identity or intermediate flow'}},
        'documentation':'Local FactSet Supply Chain v1.4.3 UserGuide, pp.14-15 and 17; extracted relevant pages saved locally.',
        'source_text_limit':'Full Reuters pages were host-inspected for retrieval; no free-text quotation was automatically promoted into a structured intermediate-flow assertion. Positive continuity needs separately inspected, explicit source-bound facts.',
        'decisions_changed':0,'predictions_changed':0,'resources':resources('continuity_vendor_audit_complete')}
    write(out/'continuity_source_audit.json',summary)
    print(MODEL,'VENDOR_AUDIT_COMPLETE',summary['path_support'],flush=True)

if __name__=='__main__':audit()
