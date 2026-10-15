"""Sonnet-only, label-free debugging. Does not launch forecasting or prompt optimization."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from filelock import FileLock
from news_agent_batched_runtime_2025 import read, write, locked_write, digest, Budget, SYSTEM
from news_agent_r2_2025.config import CLAUDE
from news_network_50_2025.config import ROOT, OUT, DSPY_PYTHON, CONTRACT, SEED
from news_network_50_2025.run import resource_limits
from network50_grounded_toolkit import VERSION, COMMON_TOOLS


MODEL = 'claude-sonnet-4-5-20250929'
DEST = OUT/'grounded_debug_v3'
PRIOR = OUT/'failure_case_model_comparison/sonnet'
CASES = ['CFLT:2025-12-08','DE:2025-10-15']
RULES = '''
This is a label-free correctness/debugging run, not optimization on outcomes.
Use inventory, market_state, own/common news, the logbook and the eligible archive. Search focused
entity/event terms; search_archive matches headline words conjunctively, so shorten empty queries.
Read sources, including material qualifications and denials. No news found is not proof of absence.
For transactions prefer the full UPDATE article over briefs before declaring terms unavailable.
The source IDs from search_archive can be read with inspect_event, including sources omitted
from the short list. Check page_count and read later pages when transaction details are missing.
News counts and a calendar-coverage flag do not encode the semantic content of a particular article.

Call record_facts with exact source_id/page/quote records. A fact is the quoted text, NOT your
paraphrase or inference. Category is reported_fact, financing, consideration, timing, exposure, or
broker_commentary. You may bind several facts at once. Each transaction field needs its own cited
fact ID; funding method is distinct from payment to shareholders. Never infer cash consideration
from cash-on-hand funding. Missing terms stay unknown. Do not invent counterparty/product facts.

For event timing, bind its source wording and call resolve_event_date(fact_id, phrase) with a
verbatim timing phrase. Use the returned timing_id. Do not invent exact dates from approximate
wording. Article time and event time are different. Unknown earnings coverage is not no earnings.

Keep interpretation and limitations qualitative and explicitly conditional where exposure is not
documented. Put numeric financial details in the source quotes, not invented factual recaps.
The interpretation field accepts no digits: reference bound facts instead of restating numbers.
The host renders risk multiples, RMS units and horizon dates. Do not write a separate numerical
summary. A price movement alone does not establish that uncertainty has resolved. Broker-target
BUZZ articles are background even if they quote numerical demand/backlog or tariff forecasts;
do not rewrite a broker's view as something the company reported. Retain relevant common news
even without a firm tag.

Driver schema: {event_key:string,fact_ids:[bound IDs],interpretation:string,limitations:string,
timing_id:returned ID or null,relation_ids:[],exposure_fact_ids:[]}.
Use concise stable event keys; multiple articles about one event are corroboration, not new shocks.
No hard tool-turn cap. Stop when additional searches do not address a concrete evidence gap.
After successful submission, return ONLY its receipt. Correct structured errors within the tool loop.
'''
OWN = '''
OWN arm. Use submit_own(forecast={score:number,drivers:[Driver],background_fact_ids:[IDs],
transaction_facts:{announcement:ID,financing:ID,consideration:ID,expected_close:ID},unknowns:[strings]}).
Only include supported transaction fields; omit the rest. No generated transaction terms, dates,
fact prose, or numerical summaries are accepted as top-level fields. At least inspect relevant
own and common coverage. Score is the log-risk residual versus the supplied OLS anchor.
'''
NET = '''
NET arm is an ADDITION to frozen OWN, not a new OWN forecast. Read fact_ledger for inherited facts.
Use exposure_paths to obtain stable relation_ids, inspect related sources, and bind new facts.
Do not rewrite OWN transaction terms or copy OWN drivers into new_events. An unchanged event
covered by another source belongs in corroborating_fact_ids, not new_events. A graph path is a
lead, not proof of materiality. Co-mention alone needs separate exposure facts. Preserve denials.
Use submit_network(supplement={total_score:number,new_events:[Driver],corroborating_fact_ids:[IDs],
rejected_fact_ids:[IDs],unknowns:[strings]}). Each new event requires relation_ids matching its
related-company news and a concrete conditional mechanism, with unknown materiality stated.
If new_events is empty, total_score MUST equal the frozen OWN score. The host, not you, composes
the final answer and preserves inherited facts. Do not resubmit OWN/common sources as new events.
'''


def capability(cid, network, own=None):
    original = read(PRIOR/'predictions'/f"{cid.replace(':','_')}_{'NET' if network else 'OWN'}.json")
    cap = read(PRIOR/'capabilities'/f"{original['key']}.json")
    cap.pop('diagnostic_submission', None)
    cap.pop('diagnostic_own', None)
    if own is not None:
        cap['grounded_own'] = own
    return cap


async def smoke():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from network50_archive_tools import load
    from network50_grounded_toolkit import GroundedToolkit
    folder = DEST/'native_smoke'
    cap = capability(CASES[0],False)
    cp, tp = folder/'capability.json', folder/'trace.json'
    write(cp,cap)
    params = StdioServerParameters(command=str(DSPY_PYTHON),
        args=['-X','utf8',str(ROOT/'code/network50_grounded_toolkit.py'),str(cp),str(tp)],
        env={**os.environ,'PYTHONPATH':str(ROOT/'code')})
    async with stdio_client(params) as (rx, tx):
        async with ClientSession(rx,tx) as session:
            await session.initialize()
            definitions = {t.name:t for t in (await session.list_tools()).tools}
            names = set(definitions)
            for tool, field in [('inspect_archive','page'),('search_archive','limit'),('inventory','limit')]:
                assert definitions[tool].input_schema['properties'][field]['type']=='integer'
            assert definitions['search_archive'].input_schema['properties']['scope']['enum']==['own','common','network']
            assert 'submit_own' in names and 'submit_network' not in names and 'submit_forecast' not in names
            for name, args in [('market_state',{}),('inventory',{}),
                ('inspect_event',{'source_id':'RTRS:2025-12:86309:6e468aecc471','page':0})]:
                result=await session.call_tool(name,args)
                assert not result.is_error, str(result)
            result=await session.call_tool('record_facts',{'facts':[{
                'source_id':'RTRS:2025-12:86309:6e468aecc471','page':0,
                'quote':'IBM will fund the Confluent deal with cash on hand','category':'financing'}]})
            assert not result.is_error, str(result)
    trace=read(tp)
    assert len(trace['facts'])==1
    cflt_read = trace['reads']['RTRS:2025-12:86309:6e468aecc471']['0']
    assert 'cash on hand' in cflt_read and '2026' in cflt_read
    own = read(OUT/'grounded_debug_v2/predictions/DE_2025-10-15_OWN.json')['accepted']['forecast']
    de_cap = capability('DE:2025-10-15',True,own)
    de_archive,_ = load('DE:2025-10-15',True)
    try:
        kit = GroundedToolkit(de_cap,de_archive,read(OUT/'anchor.json')['anchors']['DE:2025-10-15'])
        sid='RTRS:2025-10:65980:4ba498a70bae'
        actual=kit.call('inspect_event',source_id=sid,page=0)
        assert 'Truist, Jefferies' in actual['headline'] and 'text' in actual
        quote='Notes co is preparing for higher tariff costs and taking steps such as cost cuts, supply chain shifts and price changes to manage the impact, though some backlog orders may still be at risk'
        bound=kit.call('record_facts',facts=[{'source_id':sid,'page':0,'quote':quote}])['facts'][0]
        assert bound['category']=='broker_commentary', bound
    finally:
        de_archive.close()
    write(folder/'acceptance.json',{'passed':True,'labels_read':False,'llm_calls':0,
        'checks':{'cflt_archive_event_reader':True,'de_broker_source_classification':True},
        'tools':sorted(names)})
    print('Native MCP smoke passed; no LLM calls.',flush=True)


def run():
    DEST.mkdir(parents=True,exist_ok=True)
    with FileLock(str(DEST/'.run.lock'),timeout=0):
        implementation=digest({f:(ROOT/'code'/f).read_text(encoding='utf-8') for f in
            ['network50_grounded_debug.py','network50_grounded_toolkit.py','network50_enhanced_toolkit.py',
             'network50_archive_tools.py','network50_availability_toolkit.py','network50_cached_followups.py']})
        protocol={'version':VERSION,'model':MODEL,'cases':CASES,'arms':['OWN','NET'],
            'implementation':implementation,'labels_read':False,'optimization':False,
            'no_opus_calls':True,'no_forecasting_evaluation':True,
            'scope':'Known failure-case debugging only; no automatic one-week/full-panel/Opus follow-up.',
            'controls':'Original anchor unchanged. New quoted transaction state is not a refitted control.',
            'usage':'Shared cumulative ledger retained; at most four uncached agent sessions in this version.'}
        locked_write(DEST/'PROTOCOL.json',protocol)
        if not read(DEST/'native_smoke/acceptance.json')['passed']:
            raise ValueError('Native MCP smoke must pass before paid calls')
        budget=Budget(OUT,resource_limits(OUT));budget.start()
        budget.begin_authorized_continuation('user_sonnet_grounded_debug_2026_10_13',90)
        anchors=read(OUT/'anchor.json')['anchors']
        records=[]
        try:
            for cid in CASES:
                own=None
                for arm,network in [('OWN',False),('NET',True)]:
                    write(DEST/'progress.json',{'stage':'running','case_id':cid,'arm':arm,'completed':len(records),
                        'at':datetime.now(timezone.utc).isoformat()})
                    cap=capability(cid,network,own)
                    case=cap['case']
                    prompt=CONTRACT+'\n'+SEED+'\n'+RULES+'\n'+(NET if network else OWN)+'\n'+json.dumps({
                        'case_id':cid,'company':case['company'],'asof':case['asof'],
                        'next_sessions':case['next_sessions'],'anchor':anchors[cid],'raw_garch':case['b'],
                        **({'frozen_own':own} if network else {})},ensure_ascii=True)
                    key=digest({'prompt':prompt,'capability':cap,'implementation':implementation,'model':MODEL})
                    cp,tp,mp,rp=[DEST/d/f'{key}.json' for d in ('capabilities','traces','mcp','responses')]
                    write(cp,cap)
                    names=COMMON_TOOLS+(['exposure_paths','submit_network'] if network else ['submit_own'])
                    write(mp,{'mcpServers':{'news':{'command':str(DSPY_PYTHON),
                        'args':['-X','utf8',str(ROOT/'code/network50_grounded_toolkit.py'),str(cp),str(tp)],
                        'env':{'PYTHONPATH':str(ROOT/'code')}}}})
                    if rp.exists():
                        response=read(rp)
                    else:
                        allowed=','.join('mcp__news__'+n for n in names)
                        cmd=[str(CLAUDE),'--restricted','--strict-mcp-config','--mcp-config',str(mp),
                            '--tools',allowed,'--allowedTools',allowed,'--permission-mode','dontAsk',
                            '--disable-slash-commands','--setting-sources','','--settings','{"disableAllHooks":true}',
                            '--no-session-persistence','--model',MODEL,'--output-format','json','--system-prompt',SYSTEM,'-p']
                        env={**os.environ,'PYTHONPATH':str(ROOT/'code'),'MAX_THINKING_TOKENS':'0',
                            'CLAUDE_CODE_MAX_OUTPUT_TOKENS':'6000','CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1',
                            'CLAUDE_CODE_DISABLE_ATTACHMENTS':'1','CLAUDE_CODE_DISABLE_ADVISOR_TOOL':'1',
                            'MAX_STRUCTURED_OUTPUT_RETRIES':'0','ANTHROPIC_MAX_RETRIES':'0','ENABLE_TOOL_SEARCH':'false'}
                        ticket=budget.reserve(key,'grounded_sonnet_debug',200000,16000)
                        envelope=None; started=time.monotonic()
                        try:
                            proc=subprocess.run(cmd,input=prompt,capture_output=True,text=True,encoding='utf-8',
                                errors='replace',cwd=OUT/'isolated_cli',env=env,
                                timeout=max(1,min(900,budget.remaining_seconds())),
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            envelope=json.loads(proc.stdout)
                            if proc.returncode or envelope.get('is_error'):
                                raise ValueError(str(envelope)[-1800:])
                            if set(envelope.get('modelUsage',{}))!={MODEL}:
                                raise ValueError('Model mismatch; only pinned Sonnet authorized for debugging')
                            response={'envelope':envelope,'prompt':prompt,'key':key,'seconds':time.monotonic()-started}
                            write(rp,response);budget.finish(ticket,envelope)
                        except Exception as exc:
                            budget.finish(ticket,envelope,exc)
                            write(DEST/'failures'/f'{key}.json',{'error':str(exc),'envelope':envelope})
                            raise
                    trace=read(tp)
                    if not trace['submissions']:
                        raise ValueError(f'{cid} {arm}: no accepted submission; free text is not a forecast')
                    accepted=trace['submissions'][-1]
                    record={'case_id':cid,'arm':arm,'key':key,'model':MODEL,'accepted':accepted,
                            'tool_calls':len(trace['calls']),'cost_usd':response['envelope']['total_cost_usd']}
                    locked_write(DEST/'predictions'/f"{cid.replace(':','_')}_{arm}.json",record)
                    records.append(record)
                    if not network:
                        own=accepted['forecast']
                    write(DEST/'summary.json',{'records':records,'labels_read':False,'opus_calls':0})
                    print(cid,arm,'accepted',len(trace['calls']),'tools',record['cost_usd'],flush=True)
            write(DEST/'progress.json',{'stage':'complete','sessions':len(records),'labels_read':False,
                'opus_calls':0,'at':datetime.now(timezone.utc).isoformat()})
        except Exception as exc:
            write(DEST/'progress.json',{'stage':'stopped','completed':len(records),'error':str(exc),
                'at':datetime.now(timezone.utc).isoformat()})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        asyncio.run(smoke())
    else:
        run()
