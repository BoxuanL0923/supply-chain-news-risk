"""One shared policy; explicit JSON tool dispatch, no external native CLI tools."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import re
import threading

from filelock import FileLock
from news_agent_batched_runtime_2025 import digest, read, write, Client, MODEL
from .config import CONTRACT, NETWORK_RULES, DEMONSTRATIONS, MAX_ACTION_ROUNDS, MAX_TOOL_CALLS, WORKERS, VERSION
from .tools import ToolSession


def json_answer(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder(parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    try:
        value = decoder.decode(text)
    except json.JSONDecodeError:
        # A harmless prose preamble must not consume the agent's search rounds.
        # Parse complete objects with the JSON decoder, never brace-counting regex.
        values, start = [], 0
        while (start := text.find("{", start)) >= 0:
            try:
                value, end = decoder.raw_decode(text, start)
                values.append(value)
                start = end
            except json.JSONDecodeError:
                start += 1
        if len(values) != 1:
            raise ValueError("Expected one unambiguous JSON object")
        value = values[0]
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object, not another root type")
    return value


def validate(answer, session):
    if not session.read_ids:
        raise ValueError("Inspect at least one eligible source passage before forecasting")
    if session.network and not any(t["tool"] == "search_related_events" for t in session.trace):
        raise ValueError("Search related events before the network-arm forecast; weak results may be rejected")
    for key in ("score", "expected_log_risk"):
        if isinstance(answer.get(key), bool) or not isinstance(answer.get(key), (float, int)) or not math.isfinite(answer[key]):
            raise ValueError("Missing or nonfinite risk value")
    if not math.isclose(answer["score"]+session.case["anchor"], answer["expected_log_risk"], rel_tol=0, abs_tol=.0051):
        raise ValueError("Score/expected-log-risk arithmetic mismatch")
    evidence = answer.get("evidence")
    if not isinstance(evidence, list) or len(evidence) > 12:
        raise ValueError("Evidence must be a concise source list")
    for item in evidence:
        if not isinstance(item, dict) or item.get("source_id") not in session.read_ids:
            raise ValueError("Citation was not inspected in this arm")
        if not all(isinstance(item.get(k), str) for k in ("fact", "interpretation")):
            raise ValueError("Facts and interpretations must be separated")
    if not isinstance(answer.get("analysis"), dict) or set(answer["analysis"]) != {"own", "common", "network", "state"}:
        raise ValueError("Four structured analysis fields required")
    if not all(isinstance(v, str) for v in answer["analysis"].values()) or not isinstance(answer.get("missing"), list) or not isinstance(answer.get("reason"), str):
        raise ValueError("Invalid structured explanation")
    if answer.get("calendar_status") != session.calendar["status"]:
        raise ValueError("Echo the supplied calendar_status exactly; unknown is not an empty calendar")
    explanation = " ".join(answer["analysis"].values())+" "+answer["reason"]
    if session.calendar["status"] == "unknown" and re.search(r"\bno earnings (?:are )?scheduled\b|\bcalendar is clean\b", explanation, re.I):
        raise ValueError("Unknown calendar coverage cannot support an assertion of no scheduled earnings")
    return answer


class ForecastRunner:
    def __init__(self, client, out, cases, calendars, graph, manifest_hash):
        self.client, self.out, self.cases, self.calendars, self.graph = client, out, cases, calendars, graph
        self.manifest_hash = manifest_hash

    def forecast(self, case, policy, network=False, nonce=""):
        key = digest({"case": case["case_id"], "policy": policy, "network": network, "nonce": nonce,
                      "manifest": self.manifest_hash, "version": VERSION, "model": MODEL})
        target = self.out / "forecasts" / (key + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target)+".lock", timeout=1200):
            if target.exists():
                value = read(target)
                if value["key"] != key or not value["valid"]:
                    raise ValueError("Invalid forecast cache")
                return value
            session = ToolSession(case, self.calendars[case["case_id"]], self.graph, self.out, network)
            intro = session.intro()
            prefix = CONTRACT + "\n" + DEMONSTRATIONS + "\nPOLICY\n" + json.dumps(policy) + ("\n"+NETWORK_RULES if network else "")
            prefix += "\nINPUTS\n" + json.dumps(intro, ensure_ascii=True)
            # Tools must actually be requested by the model. Host provides discovery,
            # numerical state and calendar only; it does not preselect source bodies.
            history, requests = [], []
            answer = None
            for step in range(MAX_ACTION_ROUNDS+1):
                final = step == MAX_ACTION_ROUNDS
                prompt = prefix + "\nTOOL HISTORY\n" + json.dumps(history, ensure_ascii=True)
                prompt += ("\nTool round budget exhausted. Return your forecast, disclosing evidence gaps." if final else
                           f"\nAt most {MAX_ACTION_ROUNDS-step} action rounds remain. Request up to {MAX_TOOL_CALLS} local calls together, or forecast after inspection.")
                response = self.client.call(prompt, nonce=nonce)
                requests.append(response["request_sha256"])
                try:
                    decision = json_answer(response["answer"])
                    if decision.get("action") == "forecast":
                        answer = validate(decision, session)
                        break
                    calls = decision.get("calls")
                    if final or decision.get("action") != "tools" or not isinstance(calls, list) or not 1 <= len(calls) <= MAX_TOOL_CALLS:
                        raise ValueError("Expected bounded tool requests or a forecast")
                    results = []
                    for call in calls:
                        try:
                            result = session.call(call["tool"], call.get("args", {}))
                        except (ValueError, KeyError, TypeError) as exc:
                            result = {"tool_error": str(exc)}
                        results.append({"call": call, "result": result})
                    history.append({"round": step+1, "results": results})
                except (ValueError, KeyError, TypeError) as exc:
                    if final:
                        repair = self.client.call(prompt + "\nFORMAT ERROR: " + str(exc) +
                                  "\nPrevious answer:\n" + response["answer"] +
                                  "\nReturn a corrected forecast only; do not change the target or invent citations.", nonce=nonce+":repair", retry=True)
                        requests.append(repair["request_sha256"])
                        answer = validate(json_answer(repair["answer"]), session)
                        break
                    history.append({"format_error": str(exc), "previous_answer": response["answer"]})
            if answer is None:
                raise ValueError("Agent failed to complete a forecast")
            record = {"key": key, "case_id": case["case_id"], "policy": policy, "network": network,
                      "nonce": nonce, "manifest": self.manifest_hash, "valid": True, "prediction": answer,
                      "arithmetic": {"expected_log_risk_from_score": case["anchor"]+answer["score"],
                                     "reported_rounding_difference": answer["expected_log_risk"]-case["anchor"]-answer["score"]},
                      "requests": requests, "tool_trace": session.trace, "inspected_sources": sorted(session.read_ids),
                      "cited_event_keys": sorted({session.rows[e["source_id"]]["event_key"] for e in answer["evidence"]}),
                      "cited_families": sorted({session.rows[e["source_id"]].get("family", "unknown") for e in answer["evidence"]})}
            write(target, record)
            return record

    def panel(self, cases, policy, network=False, nonce=""):
        # Forecast tasks share a policy, never conversational memory or outcomes.
        pool = ThreadPoolExecutor(max_workers=WORKERS)
        futures = []
        try:
            futures = [pool.submit(self.forecast, c, policy, network, nonce) for c in cases]
            records = []
            for c, future in zip(cases, futures):
                records.append(future.result())
                print("SCORED", "NET" if network else "OWN", digest(policy)[:8], c["case_id"], flush=True)
            return records
        finally:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
