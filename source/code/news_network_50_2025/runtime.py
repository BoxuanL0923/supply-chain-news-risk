"""Pinned, metered native-tool transport with no built-in filesystem or web tools."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
from filelock import FileLock
from news_agent_batched_runtime_2025 import Client, ProviderStop, SYSTEM, digest, read, write
from .config import ROOT, DSPY_PYTHON, MODEL, TOOL_NAMES


class QwenClient:
    """Local OpenAI-compatible score client and explicit host retrieval fallback.

    call/native return NativeClient's answer/request_sha256 envelope. native()
    is deterministic host Toolkit inspection, NOT model-directed tool use.
    The Claude subprocess/MCP transport cannot be used with the Qwen endpoint.
    """
    def __init__(self, root):
        from .hop3_guard import MODEL as QWEN
        import urllib.request
        from .prompt_gd import NoRedirect
        self.root=Path(root); self.model=QWEN
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())

    def call(self,prompt,purpose='score',nonce='',retry=False):
        import urllib.request, math
        from .hop3_guard import read,write,digest,resources
        # UTF-8 byte count is a conservative upper bound for this byte-level
        # tokenizer, with 1024 additional tokens reserved for chat framing.
        if len(prompt.encode('utf-8'))+1024 >= 24000:
            raise ValueError('Prompt exceeds conservative 24k-token bound')
        body={'model':self.model,'temperature':0,'seed':20261026,'max_tokens':700,
              'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object'},
              'messages':[{'role':'user','content':prompt}]}
        key=digest(body); target=self.root/'responses'/(key+'.json')
        if target.exists():
            saved=read(target)
            if saved['request_sha256']!=key or digest(saved['request'])!=key:
                raise ValueError('Qwen cache request changed')
            return saved
        if not (self.root/'SERVER_RELEASED.json').exists():
            raise RuntimeError('Server-owner release gate has not been satisfied')
        resources('qwen_request')
        started=time.monotonic()
        req=urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with self.opener.open(req,timeout=300) as r:
            raw=json.load(r)
        # Cache every received response BEFORE parsing/validating model JSON.
        value={'request_sha256':key,'request':body,'envelope':raw,
               'server_model':raw.get('model'),'answer':raw['choices'][0]['message']['content'],
               'seconds':time.monotonic()-started,'prompt_byte_upper_bound':len(prompt.encode())+1024}
        write(target,value)
        return read(target)

    def native(self,prompt,capability_path,nonce=''):
        from .hop3_guard import read,write,digest
        from .toolkit import Toolkit
        cap=read(capability_path) if isinstance(capability_path,(str,Path)) else capability_path
        kit=Toolkit(cap); events=[]; skipped=[]
        if cap['network']:
            kit.call('exposure_paths',hops=1)
            kit.call('search_events',query='earnings demand production disruption supply',hops=1)
        pending=list(cap['documents'])
        while pending:
            progress=False
            for sid in pending[:]:
                doc=cap['documents'][sid]
                paths=[p for p in kit.paths_for(doc) if kit._accessible_path(p,3)] if cap['network'] else []
                if cap['network'] and not paths: continue
                pages=[]
                for page in range(len(kit.paragraphs(doc))):
                    result=kit.call('inspect_event',source_id=sid,page=page)
                    if 'error' in result: raise ValueError(result['error'])
                    pages.append(result['text'])
                path=min(paths,key=lambda p:(p['hops'],p['path'])) if paths else None
                events.append({'source_id':sid,'available_at':doc['available_at'],'headline':doc['headline'],
                    'body':'\n'.join(pages),'hop':path['hops'] if path else 0,
                    'path':path['path'] if path else [],'verified_paths':paths,'ids':doc['ids']})
                pending.remove(sid); progress=True
            if not progress: break
        trace={'calls':[{'tool':c['tool'],'args':c['args'],'ok':c['ok']} for c in kit.trace],
               'reads':{sid:{p:hashlib.sha256(t.encode()).hexdigest() for p,t in pages.items()} for sid,pages in kit.reads.items()},
               'inspected_paths':[list(p) for p in sorted(kit.inspected_paths)]}
        value={'transport':'host-side Toolkit inspection; no Qwen tool loop','events':events,
               'missing_prefix_sources':pending,'trace':trace}
        return {'request_sha256':digest(value),'answer':json.dumps(value),'trace':trace,'host_evidence':value}


class NativeClient(Client):
    def native(self, prompt, capability_path, nonce=""):
        capability = read(capability_path)
        key = digest({"prompt": prompt, "capability": capability, "model": MODEL,
                      "native": 2, "tool_turn_cap": None, "output_tokens": 6000, "nonce": nonce,
                      "toolkit_sha256": hashlib.sha256(Path(__file__).with_name("toolkit.py").read_bytes()).hexdigest()})
        target = self.root / "native_responses" / (key+".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        trace = self.root / "native_traces" / (key+".json")
        config = self.root / "mcp_configs" / (key+".json")
        write(config, {"mcpServers": {"news": {"command": str(DSPY_PYTHON),
            "args": ["-m", "news_network_50_2025.toolkit", str(capability_path), str(trace)],
            "env": {"PYTHONPATH": str(ROOT / "code")}}}})
        with FileLock(str(target)+".lock", timeout=1200):
            if target.exists():
                return read(target)
            cmd = [str(self.executable), "--restricted", "--strict-mcp-config", "--mcp-config", str(config),
                   "--tools", ",".join("mcp__news__"+x for x in TOOL_NAMES),
                   "--allowedTools", ",".join("mcp__news__"+x for x in TOOL_NAMES),
                   "--permission-mode", "dontAsk", "--disable-slash-commands", "--setting-sources", "",
                   "--settings", '{"disableAllHooks":true}', "--no-session-persistence", "--model", MODEL,
                   "--output-format", "json", "--debug-file", str(self.root / "native_debug" / (key+".log")),
                   "--system-prompt", SYSTEM, "-p"]
            env = {**os.environ, "PYTHONPATH": str(ROOT / "code"), "MAX_THINKING_TOKENS": "0",
                   "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "6000", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                   "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1", "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
                   "MAX_STRUCTURED_OUTPUT_RETRIES": "0", "ANTHROPIC_MAX_RETRIES": "0"}
            env["ENABLE_TOOL_SEARCH"] = "false"
            with self.semaphore:
                ticket = self.budget.reserve(key, "native_evidence", 80000, 8000)
                start, envelope = time.monotonic(), None
                try:
                    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=self.work, env=env,
                        timeout=max(1, min(900, self.budget.remaining_seconds())),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    envelope = json.loads(proc.stdout)
                    exhausted = (envelope.get("terminal_reason") == "max_turns" or
                                 envelope.get("subtype") == "error_max_turns")
                    if (proc.returncode or envelope.get("is_error")) and not exhausted:
                        raise ProviderStop(str(envelope)[-1800:])
                    if set(envelope.get("modelUsage", {})) != {MODEL}:
                        raise ProviderStop("Model mismatch: " + str(envelope.get("modelUsage")))
                    if not trace.exists() or not read(trace).get("calls"):
                        raise ValueError("Native tool use missing: do not silently fall back to prompt-described tools")
                    result = {"request_sha256": key, "answer": envelope.get("result"), "envelope": envelope,
                              "retrieval_turn_limit": exhausted,
                              "trace": read(trace), "prompt": prompt, "seconds": time.monotonic()-start}
                    write(target, result)
                    self.budget.finish(ticket, envelope)
                    return result
                except Exception as exc:
                    self.budget.finish(ticket, envelope, exc)
                    write(self.root / "failures" / (key+".json"), {"error": str(exc), "envelope": envelope})
                    raise
