"""One development origin, five firms, two arms. No prompt fitting or panel rerun."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from datetime import datetime, timezone

from filelock import FileLock
from news_agent_batched_runtime_2025 import read, write, locked_write, digest, Budget, SYSTEM
from news_agent_r2_2025.config import CLAUDE
from news_exposure_2025.workflow import json_answer
from news_network_50_2025.config import OUT, ROOT, MODEL, DSPY_PYTHON, CONTRACT, SEED
from news_network_50_2025.run import resource_limits
from news_network_50_2025.workflow import Runner
from network50_cached_followups import risk_arithmetic


DAY = '2025-10-22'
FIRMS = ['AMD','ALGM','CFLT','CAT','WMT']
DEST = OUT/'oneweek_diagnostic'
RULES = (
    'Use inventory, market_state, and source-reading tools. Review own AND common macro/financial/sector news. '
    'Search the full eligible archive to resolve a specific shortlist gap, including a focused common-scope '
    'query. Inspect sources before citing. Retain denials, qualifications, timing and unknowns. '
    'Distinguish new or still-pending uncertainty from old background. A target price cut is not necessarily '
    'a rating downgrade and is not an operating disclosure. A prior price movement or lower trailing variance '
    'is not proof that uncertainty has been resolved. Calendar and cash-deal indicators are incomplete '
    'text proxies, not authoritative event databases; a zero or unknown does not establish event absence. '
    'Keep context-only or rejected material in background, not in events. Do not fill event slots. '
    'A future economic activity can affect present uncertainty, but describe the transmission and distinguish '
    'it from a known event occurring within these five sessions. Unknown impact magnitude is allowed; '
    'unsupported transmission is not an admitted network risk driver. Do not infer geography from headquarters. '
    'Before finalizing call risk_check with your proposed score, read the variance ratios, and ensure that '
    'risk_direction_vs_anchor, exact forecast_end and reason agree with the computed quantities. '
    'There is no fixed tool-turn limit; stop when further searches do not address a concrete information gap. '
)
SCHEMA = (
    'Return JSON only: {score:number,risk_direction_vs_anchor:"lower"|"equal"|"higher",forecast_end:string,'
    'events:[{source_id:string,page:integer,fact:string,qualifications:string,mechanism:string,'
    'status:"pending"|"resolved"|"ongoing"|"unknown",transmission_supported:boolean}],'
    'background:[{source_id:string,reason:string}],missing:[string],reason:string}. '
    'At most eight OWN events or five NETWORK events. Each event must cite an inspected page. '
    'No second numerical risk forecast. Score is in log-variance residual units, not sentiment or confidence.'
)


def validate(answer, trace, cap, anchor, own=None):
    score = answer.get('score')
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
        raise ValueError('Finite numerical residual score required')
    computed = risk_arithmetic(cap['case'],anchor,score)
    if answer.get('risk_direction_vs_anchor') != computed['direction_vs_anchor']:
        raise ValueError('Score direction contradicts computed anchor comparison')
    if answer.get('forecast_end') != computed['forecast_end']:
        raise ValueError('Incorrect five-session forecast end')
    calls=trace.get('calls',[])
    used={c['tool'] for c in calls if c['ok']}
    if not {'market_state','inventory','search_archive','risk_check'} <= used:
        raise ValueError('Required native tool use missing')
    if not any(c['tool']=='risk_check' and c['ok'] and c['args'].get('score')==score for c in calls):
        raise ValueError('Submitted score was not checked through the arithmetic tool')
    if own is None and not any(c['tool']=='search_archive' and c['ok'] and c['args'].get('scope')=='common' for c in calls):
        raise ValueError('OWN arm omitted common-news archive search')
    if own is not None and 'exposure_paths' not in used:
        raise ValueError('Network arm omitted graph lookup')
    events=answer.get('events')
    if not isinstance(events,list) or len(events)>(5 if own is not None else 8):
        raise ValueError('Incorrect event list')
    inspections={}
    for call in calls:
        if call['ok'] and call['tool']=='inspect_archive':
            inspections[call['result']['source_id']]=call['result']
    admitted=[]
    for event in events:
        sid,page=event.get('source_id'),event.get('page')
        if isinstance(page,bool) or not isinstance(page,int) or str(page) not in trace.get('reads',{}).get(sid,{}):
            raise ValueError('Citation was not actually inspected')
        for field in ('fact','qualifications','mechanism'):
            if not isinstance(event.get(field),str) or not event[field].strip():
                raise ValueError('Missing evidence field '+field)
        if event.get('status') not in {'pending','resolved','ongoing','unknown'}:
            raise ValueError('Invalid event state')
        if own is not None:
            if event.get('transmission_supported') is not True:
                raise ValueError('Unsupported network story admitted as a risk driver')
            if sid in {e['source_id'] for e in own['answer']['events']}:
                raise ValueError('Network supplement duplicated OWN evidence')
            if sid not in cap['documents'] and inspections.get(sid,{}).get('scope')!='network':
                raise ValueError('Non-network archive source admitted to network supplement')
        admitted.append({**event,'quote':trace['reads'][sid][str(page)]})
    if own is not None and not admitted and score!=own['answer']['score']:
        raise ValueError('Empty network supplement must preserve the frozen OWN score')
    return {'computed':computed,'events':admitted,'tool_calls':len(calls),'checked':True}


def native(prompt, capfile, key, network, budget):
    target=DEST/'responses'/f'{key}.json';trace=DEST/'traces'/f'{key}.json';config=DEST/'mcp'/f'{key}.json'
    if target.exists():return read(target),read(trace)
    write(config,{'mcpServers':{'news':{'command':str(DSPY_PYTHON),
        'args':['-X','utf8',str(ROOT/'code/network50_enhanced_toolkit.py'),str(capfile),str(trace)],
        'env':{'PYTHONPATH':str(ROOT/'code')}}}})
    names=['inventory','market_state','search_events','inspect_event','inspect_events','event_logbook',
        'search_archive','inspect_archive','risk_check']+(['exposure_paths'] if network else [])
    allowed=','.join('mcp__news__'+n for n in names)
    cmd=[str(CLAUDE),'--restricted','--strict-mcp-config','--mcp-config',str(config),'--tools',allowed,
        '--allowedTools',allowed,'--permission-mode','dontAsk','--disable-slash-commands','--setting-sources','',
        '--settings','{"disableAllHooks":true}','--no-session-persistence','--model',MODEL,'--output-format','json',
        '--system-prompt',SYSTEM,'-p']
    env={**os.environ,'PYTHONPATH':str(ROOT/'code'),'MAX_THINKING_TOKENS':'0','CLAUDE_CODE_MAX_OUTPUT_TOKENS':'6000',
        'CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1','CLAUDE_CODE_DISABLE_ATTACHMENTS':'1','CLAUDE_CODE_DISABLE_ADVISOR_TOOL':'1',
        'MAX_STRUCTURED_OUTPUT_RETRIES':'0','ANTHROPIC_MAX_RETRIES':'0','ENABLE_TOOL_SEARCH':'false'}
    ticket=budget.reserve(key,'oneweek_diagnostic',150000,12000)
    envelope=None;started=time.monotonic()
    try:
        proc=subprocess.run(cmd,input=prompt,capture_output=True,text=True,encoding='utf-8',errors='replace',
            cwd=OUT/'isolated_cli',env=env,timeout=max(1,min(900,budget.remaining_seconds())),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        envelope=json.loads(proc.stdout)
        if proc.returncode or envelope.get('is_error'):raise ValueError(str(envelope)[-1800:])
        if set(envelope.get('modelUsage',{}))!={MODEL}:raise ValueError('Model mismatch')
        result={'envelope':envelope,'prompt':prompt,'seconds':time.monotonic()-started,'key':key}
        write(target,result);budget.finish(ticket,envelope)
        return result,read(trace)
    except Exception as exc:
        budget.finish(ticket,envelope,exc)
        write(DEST/'failures'/f'{key}.json',{'error':str(exc),'envelope':envelope})
        raise


def run():
    with FileLock(str(DEST)+'.lock',timeout=0):
        files=['network50_oneweek_diagnostic.py','network50_enhanced_toolkit.py','network50_archive_tools.py',
            'network50_availability_toolkit.py','network50_cached_followups.py']
        implementation=digest({f:(ROOT/'code'/f).read_text(encoding='utf-8') for f in files})
        protocol={'origin':DAY,'firms':FIRMS,'arms':['OWN','NET'],'model':MODEL,'implementation':implementation,
            'target':'Five-session log mean squared daily total returns; raw residual versus existing training OLS anchor.',
            'scope':'One previously used development origin. Correctness diagnostic, not an unseen forecast test. No optimization.',
            'selection':'Two semiconductor firms, cloud software, industrial equipment, retail; fixed before new diagnostic forecasts.',
            'controls':'Original conventional anchor kept unchanged; cash-deal and calendar proxy defects remain disclosed.',
            'workflow_change':'Combined native evidence and scoring, balanced own/common retrieval, explicit network admission, host arithmetic checks. Not an isolated tool-only ablation.',
            'no_full_panel_launch':True,'no_test_labels':True,'label_read_rule':'Only selected training labels, after all diagnostic answers are frozen.'}
        locked_write(DEST/'PROTOCOL.json',protocol)
        cases=read(OUT/'cases.json');anchors=read(OUT/'anchor.json')['anchors']
        def no_provider(*args,**kwargs):raise ValueError('Cached original capability only')
        runner=Runner(OUT,SimpleNamespace(native=no_provider,call=no_provider),cases,anchors,digest(read(OUT/'RUN_MANIFEST.json')))
        budget=Budget(OUT,resource_limits(OUT));budget.start()
        records=[]
        try:
            for ticker in FIRMS:
                cid=f'{ticker}:{DAY}';case=next(c for c in cases if c['case_id']==cid)
                assert case['split']=='train'
                own=None
                for arm,network in [('OWN',False),('NET',True)]:
                    write(DEST/'progress.json',{'stage':'predict','case_id':cid,'arm':arm,'completed':len(records),
                                              'at':datetime.now(timezone.utc).isoformat()})
                    if network and (own is None or not own['valid']):
                        records.append({'case_id':cid,'arm':arm,'valid':False,'error':'OWN diagnostic invalid; NET skipped'})
                        continue
                    ev=runner.evidence(case,network);cap=read(OUT/'capabilities'/f"{ev['key']}.json")
                    capfile=DEST/'capabilities'/f'{ticker}_{arm}.json';write(capfile,cap)
                    task=('OWN forecast. Include own-company and common news; no network available.\n' if not network else
                        'NETWORK supplement to the frozen OWN forecast supplied below. Do not rewrite its evidence. '
                        'Look up graph paths and related news, then admit only sources with a concrete, source-supported '
                        'transmission mechanism. Put unsupported leads in background. Score is TOTAL residual versus the same '
                        'OLS anchor, not the incremental difference. If no network risk drivers are admitted, return exactly '
                        'the OWN score without revising it. A nonempty supplement may raise, lower or leave risk unchanged.\n')
                    inputs={'case_id':cid,'company':case['company'],'asof':case['asof'],'next_sessions':case['next_sessions'],
                            'anchor':anchors[cid],'raw_garch':case['b']}
                    if own is not None:inputs['frozen_own']={'answer':own['answer'],'verified_events':own['audit']['events']}
                    prompt=CONTRACT+'\n'+SEED+'\n'+RULES+'\n'+task+SCHEMA+'\n'+json.dumps(inputs,ensure_ascii=True)
                    key=digest({'prompt':prompt,'implementation':implementation,'capability':cap,'model':MODEL})
                    response,trace=native(prompt,capfile,key,network,budget)
                    record={'case_id':cid,'arm':arm,'request':key,'list_price_usd':response['envelope'].get('total_cost_usd')}
                    try:
                        answer=json_answer(response['envelope']['result'])
                        record.update(answer=answer,audit=validate(answer,trace,cap,anchors[cid],own),valid=True)
                    except (ValueError,KeyError,TypeError) as exc:
                        record.update(valid=False,error=str(exc),answer=response['envelope'].get('result'))
                    locked_write(DEST/'predictions'/f'{ticker}_{arm}.json',record)
                    records.append(record)
                    if not network:own=record
                    print(cid,arm,'valid=',record['valid'],record.get('error',''),flush=True)
            locked_write(DEST/'frozen_predictions.json',records)
            # Only the five selected development labels are joined, after forecast generation.
            import pandas as pd
            labels=pd.read_parquet(OUT/'labels_train.parquet').set_index('case_id')
            scored=[]
            for record in records:
                if not record['valid']:continue
                cid=record['case_id'];a=anchors[cid];y=float(labels.loc[cid,'y']);s=record['answer']['score']
                scored.append({'case_id':cid,'arm':record['arm'],'y':y,'anchor':a,'score':s,
                    'target_residual':y-a,'squared_error':(y-a-s)**2,'anchor_squared_error':(y-a)**2})
            write(DEST/'losses.json',scored)
            write(DEST/'progress.json',{'stage':'complete','valid':sum(r['valid'] for r in records),'total':len(records),
                                      'at':datetime.now(timezone.utc).isoformat(),'full_pipeline_launched':False})
            lines=['# One-Week Two-Arm Diagnostic','',protocol['scope'],'',
                'Origin: '+DAY+'. Firms: '+', '.join(FIRMS)+'. Only this five-session forecast window is evaluated.',
                'No statistical significance or pooled regression R-squared claim is made from five firms.',
                'Historical sources can include the previous 31 days; the outcome horizon is five sessions.','',
                '| Firm | Arm | Valid | Score | Squared residual error |','|---|---|---|---:|---:|']
            lookup={(r['case_id'],r['arm']):r for r in scored}
            for r in records:
                loss=lookup.get((r['case_id'],r['arm']))
                lines.append(f"| {r['case_id'].split(':')[0]} | {r['arm']} | {r['valid']} | "+
                    (f"{loss['score']:.4f} | {loss['squared_error']:.4f} |" if loss else '- | - |'))
            lines+=['','Validation failures:']+[f"- {r['case_id']} {r['arm']}: {r['error']}" for r in records if not r['valid']]
            lines+=['','Inspect predictions and tool traces for semantic correctness; structural validity does not prove material transmission.',
                'The original anchor was fitted on the development period, so these losses are diagnostic, not independent out-of-sample results.',
                'Original experiment artifacts are unchanged. No full-panel follow-up is authorized by passing this diagnostic.']
            (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        except Exception as exc:
            write(DEST/'progress.json',{'stage':'stopped','error':str(exc),'completed':len(records),
                                      'at':datetime.now(timezone.utc).isoformat()})
            raise


if __name__=='__main__':
    run()
