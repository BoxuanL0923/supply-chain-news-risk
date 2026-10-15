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
