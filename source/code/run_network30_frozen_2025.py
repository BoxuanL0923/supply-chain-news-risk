"""Frozen OWN/NET December 2025 panel with the source-grounded toolkit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from filelock import FileLock
import numpy as np
import pandas as pd

from news_agent_batched_runtime_2025 import Budget, SYSTEM, digest, locked_write, read, write
from news_agent_r2_2025.config import CLAUDE
from news_network_50_2025.config import ROOT, OUT, DSPY_PYTHON, CONTROLS
from news_network_50_2025.workflow import Runner
from network50_grounded_debug import MODEL, RULES, OWN, NET
from network50_grounded_toolkit import COMMON_TOOLS, VERSION


DEST = OUT / "network30_dec2025_frozen_v6"
DATES = ("2025-12-01", "2025-12-08", "2025-12-15", "2025-12-22")
CONTROLS_V4 = tuple("recent_ma_news_31d" if c == "pending_cash_deal" else c
                    for c in CONTROLS if c != "calendar_unknown")
MA_NEWS = re.compile(r"\b(?:acquisition|merger|takeover|buyout|to buy|to acquire|agrees? to buy)\b", re.I)
SMOKE = (("2025-12-08", "AMD"), ("2025-12-08", "CFLT"),
         ("2025-12-08", "SMCI"), ("2025-12-08", "CAT"),
         ("2025-12-08", "WMT"), ("2025-12-01", "AMD"),
         ("2025-12-15", "WMT"), ("2025-12-22", "SMCI"),
         ("2025-12-01", "HPE"), ("2025-12-01", "MDB"))
STRICT_RULES = '''
The OLS anchor contains only the controls shown by market_state. In this pilot it has no VIX,
rates, or other macroeconomic factor; do not claim an article's semantic content is controlled.
Use historical news as context, not as a new driver after two weeks without a fresh update.
Exceptions require a computed upcoming event or a documented live acquisition. A Reuters firm
tag is only a retrieval lead; confirm the target is actually named in its article before using
it as own-company evidence. Direct graph edges do not turn generic partner earnings or small-
partner tariff anecdotes into target-specific shocks. Operational interruptions can be reasoned
through a direct supplier/customer path, with materiality and alternatives left unknown.
If a proposed network event fails an exposure check, search once for concrete target evidence;
otherwise move it to rejected/background facts and do not repeat the unsupported proposal.
In NET, first call network_leads with an empty query for balanced direct-partner discovery,
then use focused queries for OWN drivers with plausible transmission channels (operational
shocks, export policy, tariffs, or demand). A zero-hit narrow query does not mean no network
news. Inspect a source before admitting its lead. Topic overlap is not an economic channel.
Two-hop paths remain
leads unless cited facts explicitly connect the target through the intermediate firm.
OWN can search only own and common scopes; NET can also search network. Inventory limit is at
most twenty; inspect_events accepts at most six source/page requests. Do not assert initial
market reaction must occur after the decision close for pre-close articles: source facts show
minutes before the decision, but whether news was absorbed is unknown. Interpretations and
limitations must be qualitative: spell out no numbers or ordinal dates there. Keep exact
figures in inspected Reuters quotes and computed dates in timing receipts.
Interpretation is a conditional economic mechanism, not a recap of who said what. Reporter
narration is not a company statement. Do not claim documented price co-movement or prior
market reactions without a cited empirical source; if not supported, state the channel as
a hypothesis and the transmission strength as unknown. A computed date range is not a
single decision day. Approximate timing does not certify a future-window event. For a
scheduled event said to fall inside the forecast window, bind and resolve the shortest
exact date or weekday phrase in a source. If timing is unresolved, describe inclusion
only as possible, not certain.
market_state also supplies host-verified target earnings-calendar fact_id and timing_id
receipts when an earnings release is confirmed in the forecast window. Cite both IDs
for that target earnings driver; do not borrow its receipt for Fed or peer events.
An earnings article dated on the decision day without a release time does not prove
whether the event was before or after the close. You may cite its resolved same-day
date as a conditional driver only if limitations explicitly say release time is unknown;
it is not a confirmed forecast-window event. Seek an eligible source with the actual
release time before assigning a large adjustment. A score whose magnitude exceeds
log(2) implies more than a twofold variance change and requires a cited target-specific
driver; macro stories, background facts, and unresolved same-day timing alone cannot
justify that size. Unknowns must not silently supply the reason for the numeric score.
'''
WORKERS = 4
LIMITS = {"hard_total_provider_attempts": 280, "hard_total_execution_minutes": 480,
          "input_token_stop": 180_000_000, "output_token_stop": 5_000_000,
          "provider_list_price_stop_usd": 250., "shared_retry_reserve": 0}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_firms():
    universe = pd.read_csv(OUT / "universe.csv")
    chosen = []
    for theme, group in universe.groupby("theme"):
        for size_group in (group[group["size"] < 30e9], group[group["size"] >= 30e9]):
            ranked = size_group.sort_values(["median_dollars", "ticker"], ascending=[False, True]).head(3)
            if len(ranked) != 3:
                raise ValueError(f"Insufficient July-eligible firms in {theme}")
            chosen.extend(ranked.ticker.tolist())
    if len(chosen) != 30 or len(set(chosen)) != 30:
        raise ValueError("Expected 30 distinct tickers")
    return tuple(sorted(chosen))


def recent_ma(case, pool_file):
    pool = pool_file["pools"][case["case_id"]]
    cutoff = datetime.fromisoformat(case["asof"])
    return int(any(
        case["permno"] in pool_file["documents"][sid]["ids"]
        and cutoff - timedelta(days=31) <= datetime.fromisoformat(pool_file["documents"][sid]["available_at"]) < cutoff
        and MA_NEWS.search(pool_file["documents"][sid]["headline"])
        for sid in pool["history_ids"]
    ))


def prepare():
    firms = selected_firms()
    all_cases = read(OUT / "cases.json")
    cases = [c for c in all_cases if c["ticker"] in firms and c["date"] in DATES]
    if len(cases) != 120 or len({c["case_id"] for c in cases}) != 120:
        raise ValueError("Expected 30 firms x 4 nonoverlapping origins")
    controls = pd.read_parquet(OUT / "controls.parquet").copy()
    controls["recent_ma_news_31d"] = 0
    for day in sorted(controls.date.unique()):
        source = read(OUT / "pools" / (day + ".json"))
        for case in (c for c in all_cases if c["date"] == day):
            mask = controls.case_id.eq(case["case_id"])
            controls.loc[mask, "recent_ma_news_31d"] = recent_ma(case, source)
    if len(controls) != 600 or not controls.case_id.is_unique:
        raise ValueError("Conventional-control coverage changed")
    if int(controls.loc[controls.case_id.eq("CFLT:2025-12-08"), "recent_ma_news_31d"].iloc[0]) != 1:
        raise ValueError("CFLT acquisition control still missing")
    if not (controls.earnings_confirmed + controls.calendar_unknown).eq(1).all():
        raise ValueError("Calendar redundancy assumption changed")
    labels = pd.read_parquet(OUT / "labels_train.parquet", columns=["case_id", "y"])
    train = labels.merge(controls, on="case_id", validate="one_to_one")
    if len(train) != 200:
        raise ValueError("Historical training rows changed")
    x = np.c_[np.ones(len(train)), train[list(CONTROLS_V4)].to_numpy(float)]
    beta, _, rank, _ = np.linalg.lstsq(x, train.y.to_numpy(float), rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Conventional anchor remains rank deficient")
    all_x = np.c_[np.ones(len(controls)), controls[list(CONTROLS_V4)].to_numpy(float)]
    anchors = dict(zip(controls.case_id, (all_x @ beta).tolist()))
    if any(not np.isfinite(anchors[c["case_id"]]) for c in cases):
        raise ValueError("Nonfinite anchor")
    output = {"coefficients": beta.tolist(), "rank": int(rank), "controls": list(CONTROLS_V4),
              "train_ids": sorted(train.case_id.tolist()), "anchors": anchors,
              "definition": "Pooled OLS fitted only to October 1-22 2025 labels, replacing the degenerate cash-deal flag with point-in-time recent M&A news and omitting calendar_unknown, an exact complement of earnings_confirmed."}
    locked_write(DEST / "anchor.json", output)
    controls.drop(columns=["pending_cash_deal", "calendar_unknown"]).to_parquet(DEST / "controls.parquet", index=False)
    implementation = digest({name: file_hash(ROOT / "code" / name) for name in (
        "run_network30_frozen_2025.py", "network50_grounded_toolkit.py", "network50_grounded_debug.py",
        "network50_enhanced_toolkit.py", "network50_archive_tools.py", "network50_availability_toolkit.py",
        "news_network_50_2025/toolkit.py")})
    protocol = {"version": VERSION, "model": MODEL, "dates": list(DATES), "firms": list(firms),
                "selection": "Within each original July-eligible theme: three highest July median-dollar-volume firms below $30bn and three at/above $30bn; no outcome-based selection.",
                "arms": ["OWN", "NET"], "prompt_sha256": digest((RULES, STRICT_RULES, OWN, NET)),
                "implementation": implementation, "workers": WORKERS,
                "anchor_sha256": digest(output), "baseline": output["definition"],
                "source_data_sha256": {name: file_hash(OUT / name) for name in
                    ("data_manifest.json", "universe.csv", "cases.json", "labels_train.parquet")},
                "forecast_target": "log mean of next five sessions' squared close-to-close total returns",
                "evaluation": "Frozen direct anchor+score primary; pooled refit R2 descriptive only; no significance claim with four dates.",
                "history": "December cases and failure examples previously reviewed; exploratory, not pristine holdout.",
                "network_caveat": "NET adds tools, related-company text and compute; no placebo, so not an isolated graph-topology effect.",
                "labels_available_to_agent": False}
    locked_write(DEST / "PROTOCOL.json", protocol)
    locked_write(DEST / "selected_cases.json", [{k: c[k] for k in ("case_id", "date", "ticker", "asof", "next_sessions")} for c in cases])
    return cases, controls.set_index("case_id"), anchors, protocol


def capability(case, network, own, controls):
    cap = Runner(OUT, None, None, None, None).capability(case, network)
    row = controls.loc[case["case_id"]]
    cap["controls"] = {c: float(row[c]) for c in CONTROLS_V4}
    if network:
        cap["grounded_own"] = own
    if cap["case"]["asof"] != case["asof"] or any(
        datetime.fromisoformat(d["available_at"]) >= datetime.fromisoformat(case["asof"])
        for d in cap["documents"].values()
    ):
        raise ValueError("Future or mismatched source in capability")
    return cap


def prompt_for(case, cap, anchor, network, own):
    from news_network_50_2025.config import CONTRACT, SEED
    return CONTRACT + "\n" + SEED + "\n" + RULES + "\n" + STRICT_RULES + "\n" + (NET if network else OWN) + "\n" + json.dumps({
        "case_id": case["case_id"], "company": case["company"], "asof": case["asof"],
        "next_sessions": case["next_sessions"], "anchor": anchor, "raw_garch": case["b"],
        **({"frozen_own": own} if network else {})}, ensure_ascii=True)


def score(case, network, own, controls, anchors, protocol, budget):
    cid, arm = case["case_id"], "NET" if network else "OWN"
    cap = capability(case, network, own, controls)
    prompt = prompt_for(case, cap, anchors[cid], network, own)
    key = digest({"prompt": prompt, "capability": cap,
                  "implementation": protocol["implementation"], "model": MODEL})
    pred_path = DEST / "predictions" / f"{cid.replace(':', '_')}_{arm}.json"
    if pred_path.exists():
        record = read(pred_path)
        if record["key"] != key:
            raise ValueError(f"Cached {cid} {arm} has a different prompt or source capability")
        return record
    cp, tp, mp, rp = [DEST / folder / (key + ".json") for folder in
                       ("capabilities", "traces", "mcp", "responses")]
    write(cp, cap)
    write(mp, {"mcpServers": {"news": {"command": str(DSPY_PYTHON),
        "args": ["-X", "utf8", str(ROOT / "code/network50_grounded_toolkit.py"), str(cp), str(tp)],
        "env": {"PYTHONPATH": str(ROOT / "code")}}}})
    names = COMMON_TOOLS + (["exposure_paths", "network_leads", "submit_network"] if network else ["submit_own"])
    allowed = ",".join("mcp__news__" + name for name in names)
    cmd = [str(CLAUDE), "--restricted", "--strict-mcp-config", "--mcp-config", str(mp),
           "--tools", allowed, "--allowedTools", allowed, "--permission-mode", "dontAsk",
           "--disable-slash-commands", "--setting-sources", "", "--settings", '{"disableAllHooks":true}',
           "--no-session-persistence", "--model", MODEL, "--output-format", "json",
           "--system-prompt", SYSTEM, "-p"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "code"), "MAX_THINKING_TOKENS": "0",
           "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "6000", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1", "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
           "MAX_STRUCTURED_OUTPUT_RETRIES": "0", "ANTHROPIC_MAX_RETRIES": "0", "ENABLE_TOOL_SEARCH": "false"}
    if rp.exists():
        response = read(rp)
    else:
        ticket = budget.reserve(key, "network30_frozen_" + arm.lower(), 200000, 16000)
        envelope = None
        start = time.monotonic()
        try:
            timeout = max(1, min(900, budget.remaining_seconds()))
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", cwd=OUT / "isolated_cli", env=env, timeout=timeout,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            envelope = json.loads(proc.stdout)
            if proc.returncode or envelope.get("is_error"):
                raise ValueError(f"CLI error {proc.returncode}: {str(envelope)[-1200:]} {proc.stderr[-600:]}")
            if set(envelope.get("modelUsage", {})) != {MODEL}:
                raise ValueError("Model identity mismatch")
            response = {"envelope": envelope, "prompt": prompt, "key": key,
                        "seconds": time.monotonic() - start}
            write(rp, response)
            budget.finish(ticket, envelope)
        except Exception as exc:
            budget.finish(ticket, envelope, exc)
            write(DEST / "failures" / (key + ".json"), {"case_id": cid, "arm": arm,
                  "error": str(exc), "envelope": envelope})
            raise
    trace = read(tp)
    if not trace["submissions"]:
        raise ValueError(f"{cid} {arm}: no accepted structured submission")
    forecast = trace["submissions"][-1]["forecast"]
    if forecast["arm"] != arm or forecast["case_id"] != cid:
        raise ValueError("Wrong accepted case or arm")
    if network and digest(forecast["frozen_own"]) != digest(own):
        raise ValueError("NET modified OWN")
    record = {"case_id": cid, "date": case["date"], "ticker": case["ticker"], "arm": arm,
              "key": key, "score": forecast["score"], "accepted": trace["submissions"][-1],
              "tool_calls": len(trace["calls"]), "cost_usd": response["envelope"]["total_cost_usd"],
              "model": MODEL, "capability_hash": digest(cap)}
    write(pred_path, record)
    return record


def evaluate(cases, controls, anchors):
    from news_network_50_2025.metrics import fit as pooled_fit
    records = {}
    for case in cases:
        for arm in ("OWN", "NET"):
            record = read(DEST / "predictions" / f"{case['case_id'].replace(':', '_')}_{arm}.json")
            records[(case["case_id"], arm)] = record
    labels = pd.read_parquet(OUT / "labels_test.parquet", columns=["case_id", "y"])
    selected = {c["case_id"] for c in cases}
    frame = labels[labels.case_id.isin(selected)].merge(controls.reset_index(), on="case_id", validate="one_to_one")
    if len(frame) != 120 or set(frame.case_id) != selected:
        raise ValueError("December evaluation panel incomplete")
    frame["p0"] = frame.case_id.map(anchors)
    for arm in ("OWN", "NET"):
        frame[arm] = frame.case_id.map({cid: r["score"] for (cid, a), r in records.items() if a == arm})
        frame["p_" + arm] = frame.p0 + frame[arm]
    y = frame.y.to_numpy(float)
    sst = float(np.square(y-y.mean()).sum())
    if not np.isfinite(frame[["y", "p0", "p_OWN", "p_NET"]].to_numpy(float)).all() or sst <= 0:
        raise ValueError("Invalid evaluation values")
    sse = {name: float(np.square(y-frame[name].to_numpy(float)).sum()) for name in ("p0", "p_OWN", "p_NET")}
    base = pooled_fit(frame, CONTROLS_V4)
    own = pooled_fit(frame, CONTROLS_V4, "OWN")
    net = pooled_fit(frame, CONTROLS_V4, "NET")
    by_date = {}
    for day, group in frame.groupby("date"):
        by_date[day] = {"n": len(group), "own_mse": float(np.mean(np.square(group.y-group.p_OWN))),
                        "net_mse": float(np.mean(np.square(group.y-group.p_NET))),
                        "net_minus_own_mse": float(np.mean(np.square(group.y-group.p_NET)-np.square(group.y-group.p_OWN)))}
    report = {"n": len(frame), "firms": frame.ticker.nunique(), "dates": DATES,
              "frozen_direct": {"centered_r2": {k: 1-v/sst for k, v in sse.items()},
                                "increment_over_anchor": {k: 1-sse[k]/sse["p0"] for k in ("p_OWN", "p_NET")},
                                "network_minus_own_r2_pp": 100*(sse["p_OWN"]-sse["p_NET"])/sst,
                                "mse": {k: v/len(frame) for k, v in sse.items()}},
              "pooled_refit_descriptive": {"controls_r2": base["r2"], "own_r2": own["r2"],
                  "net_r2": net["r2"], "network_minus_own_r2_pp": 100*(net["r2"]-own["r2"])},
              "by_date": by_date, "network_abstentions": sum(not records[(c["case_id"], "NET")]["accepted"]["forecast"]["drivers"] for c in cases),
              "returned_list_price_usd": sum(r["cost_usd"] for r in records.values()),
              "interpretation": "Exploratory four-date panel. Direct anchor+score uses no fitted December coefficient; refit R2 is descriptive, not OOS. No statistical-significance claim or isolated graph effect."}
    write(DEST / "evaluation.json", report)
    frame[["case_id", "date", "ticker", "y", "p0", "OWN", "NET", "p_OWN", "p_NET"]].to_parquet(DEST / "matched_evaluation.parquet", index=False)
    return report


def run(smoke=False, full=False):
    DEST.mkdir(parents=True, exist_ok=True)
    with FileLock(str(DEST / ".run.lock"), timeout=0):
        cases, controls, anchors, protocol = prepare()
        selected = [c for c in cases if (c["date"], c["ticker"]) in SMOKE] if smoke else cases
        if smoke and len(selected) != len(SMOKE):
            raise ValueError("Missing predeclared smoke firm")
        if not smoke and not full:
            print("Prepared fixed panel and October-only anchor; no LLM calls.", flush=True)
            return
        if full and not (DEST / "SMOKE_ACCEPTED.json").exists():
            raise ValueError("Review and accept the label-free smoke before full scoring")
        budget = Budget(DEST, LIMITS)
        budget.start()
        halted = threading.Event()
        results, errors = [], []
        lock = threading.Lock()

        def task(case):
            if halted.is_set():
                return
            try:
                own = score(case, False, None, controls, anchors, protocol, budget)
                net = score(case, True, own["accepted"]["forecast"], controls, anchors, protocol, budget)
                with lock:
                    results.append((own, net))
                    write(DEST / "progress.json", {"stage": "smoke" if smoke else "full", "complete_case_pairs": len(results),
                          "requested_case_pairs": len(selected), "last_case": case["case_id"],
                          "at": datetime.now(timezone.utc).isoformat(), "errors": errors})
                    print(case["case_id"], "OWN", own["score"], "NET", net["score"], flush=True)
            except Exception as exc:
                halted.set()
                with lock:
                    errors.append({"case_id": case["case_id"], "error": str(exc)})
                    write(DEST / "progress.json", {"stage": "stopped", "complete_case_pairs": len(results),
                          "last_case": case["case_id"], "errors": errors,
                          "at": datetime.now(timezone.utc).isoformat()})

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(task, case) for case in selected]
            for future in as_completed(futures):
                future.result()
        if errors:
            raise ValueError(f"Incomplete run: {errors}")
        if smoke:
            write(DEST / "smoke_summary.json", {"cases": [c["case_id"] for c in selected],
                  "accepted": len(results) == len(SMOKE), "test_labels_read": False,
                  "manual_review_required": True})
        else:
            report = evaluate(cases, controls, anchors)
            print("Complete. Direct NET-OWN R2 increment (pp):", report["frozen_direct"]["network_minus_own_r2_pp"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opus", action="store_true", help="Use the pinned Opus 4.6 ID and a separate output directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.opus:
        MODEL = "claude-opus-4-6"
        DEST = OUT / "network30_dec2025_opus46_v6"
        LIMITS = {**LIMITS, "hard_total_provider_attempts": 320,
                  "hard_total_execution_minutes": 720,
                  "input_token_stop": 250_000_000, "output_token_stop": 7_000_000,
                  "provider_list_price_stop_usd": 500.}
    run(args.smoke, args.full)
