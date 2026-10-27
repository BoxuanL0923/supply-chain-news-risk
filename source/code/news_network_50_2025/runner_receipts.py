"""Host Skill -> Python receipts used by the live retrieval/training path."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from .graph_router import RouterRules, digest, route_case, verify_receipt
from .graph_skills import Snapshot, utc
from .graph_projection import bound_paths, indexed_snapshot
from .label_firewall import boundary, check_payload

REPO = Path(__file__).resolve().parents[3]


def skill_read(name, function):
    path = REPO / "skills" / name / "SKILL.md"
    raw = path.read_bytes()
    return dict(event="skill_read", skill=name, path=path.relative_to(REPO).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(), python=function)


def acceptance_manifest(bundles):
    """Freeze gate bundle and aggregate-input bytes before any scoring."""
    result = {}
    for policy_hash, name in sorted(bundles.items()):
        path = Path(name)
        raw = path.read_bytes()
        value = json.loads(raw)
        check_payload(value, development=True)
        if value['policy_sha256'] != policy_hash:
            raise ValueError('Acceptance manifest policy binding mismatch')
        config = value['config']
        names = [config[k] for k in ('validation', 'sign_stability', 'rewire_placebo')]
        names += [config['industry_peer'][k] for k in ('rows', 'contract')]
        inputs = {}
        for name in names:
            target = path.parent/name
            if Path(name).name != name or target.is_symlink():
                raise ValueError('Unsafe gate input path')
            inputs[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        result[policy_hash] = dict(bundle_sha256=hashlib.sha256(raw).hexdigest(), input_sha256=inputs)
    return result


def route_capability(capability, reads=None, *, inspected_paths=(), rules=None):
    """Derive exact prefix receipts from host reads, never model assertions."""
    check_payload(capability)
    snapshot = capability.get("graph_snapshot")
    if snapshot is None:
        raise ValueError("Enabled graph router requires a dated graph snapshot")
    case = capability["case"]
    target, cutoff = capability["graph_firm_id"], case["asof"]
    graph = Snapshot(snapshot, cutoff)
    projection = indexed_snapshot(snapshot)
    inspected_paths = {tuple(p) for p in inspected_paths}
    by_prefix, candidates = {}, set()
    for sid, doc in capability["documents"].items():
        paths = bound_paths(capability["relations"], doc, cutoff, projection)
        for path in paths:
            chain = path.get("path", [])
            if not chain:
                continue
            if chain[0] != target:
                raise ValueError("Graph path belongs to another focal firm")
            candidates.add(chain[1])
            if reads and reads.get(sid) and tuple(chain) in inspected_paths and 1 <= path["hops"] <= 2:
                by_prefix.setdefault(tuple(chain), []).append(sid)
    # A second-hop read cannot attest to an uninspected first-hop prefix.
    inspections = []
    for chain, ids in sorted(by_prefix.items(), key=lambda p: (len(p[0]), p[0])):
        if len(chain) == 3 and chain[:2] not in by_prefix:
            raise ValueError("Host read lacks its exact earlier prefix")
        inspections.append(dict(path=list(chain), case_id=case["case_id"], cutoff=cutoff,
            snapshot_sha256=digest(snapshot), inspected=True, source_ids=sorted(set(ids)),
            available_at=max((capability["documents"][sid]["available_at"] for sid in ids), key=utc)))
    route = route_case(snapshot, dict(case_id=case["case_id"], firm_id=target, cutoff=cutoff,
                       candidate_neighbours=sorted(candidates), inspections=inspections), rules)
    trace = [skill_read("sc-graph-router", "graph_router.route_case")]
    seen = set()
    for call in route["calls"]:
        tool = call["tool"]
        if tool.startswith("/sc-") and tool not in seen:
            trace.append(skill_read(tool[1:], "graph_skills.Snapshot"))
            seen.add(tool)
        trace.append(dict(event="python_call", function=tool, receipt=call))
    trace.append(dict(event="router_decision", receipt=route))
    return route, trace


def verify_case_trace(trace):
    """Verify bindings, router hashes/replay and the final host selection call."""
    body = {k: v for k, v in trace.items() if k != "sha256"}
    if digest(body) != trace.get("sha256"):
        raise ValueError("Case trace hash mismatch")
    check_payload(body)
    routes = [e["receipt"] for e in trace["events"] if e["event"] == "router_decision"]
    for receipt in routes:
        verify_receipt(receipt)
        if receipt["case_id"] != trace["case_id"] or receipt["cutoff"] != trace["cutoff"]:
            raise ValueError("Cross-case router receipt")
    calls = [e for e in trace["events"] if e["event"] == "select_context"]
    skills = [e for e in trace["events"] if e["event"] == "skill_read" and e["skill"] == "supply-chain-context-selection"]
    if not skills or not calls or not calls[-1]["receipt"]["tool_called"]:
        raise ValueError("Skill read and Python select_context receipt required")
    if trace["events"].index(skills[0]) > trace["events"].index(calls[-1]):
        raise ValueError("Skill must precede select_context")
    for event in trace["events"]:
        if event["event"] == "skill_read":
            actual = skill_read(event["skill"], event["python"])
            if actual != event:
                raise ValueError("Skill bytes changed")
        if event["event"] == "acceptance_gates" and event["binding"]["policy_sha256"] != trace["policy_sha256"]:
            raise ValueError("Acceptance receipt belongs to another policy")
    if any(not a["allowed"] for a in trace["label_accesses"]):
        raise ValueError("Forbidden label access recorded")
    return True


def training_gate(out, policy, cases, bundle_path=None):
    """Run existing acceptance scripts before a trained candidate can replace seed.

    These historical gate adapters do not refit 2025 Q1 data. Missing/incompatible
    inputs fail closed; receipt input hashes preserve the host fitting attestation.
    """
    check_payload(cases, development=True)
    events = [skill_read("sc-acceptance-ledger", "acceptance_ledger.run")]
    if bundle_path is None:
        return dict(decision="NOT_RUN", accepted=False, screening_only=True,
                    reason="No source-bound historical gate inputs supplied", events=events)
    with boundary(training=True) as access:
        bundle_path = Path(bundle_path).resolve()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if set(bundle) != {"policy_sha256", "case_ids", "config"} or bundle["policy_sha256"] != digest(policy):
            raise ValueError("Gate bundle must bind the exact proposed policy")
        if sorted(bundle["case_ids"]) != sorted(c["case_id"] for c in cases):
            raise ValueError("Gate bundle case set differs from development selection")
        config = bundle["config"]
        names = [config[k] for k in ("validation", "sign_stability", "rewire_placebo")]
        names += [config["industry_peer"][k] for k in ("rows", "contract")]
        hashes = {}
        for name in names:
            p = bundle_path.parent / name
            if Path(name).name != name or p.is_symlink() or not p.is_file():
                raise ValueError("Gate input must be a local regular filename")
            raw = p.read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            if p.suffix == ".json":
                check_payload(json.loads(raw), development=True)
            elif p.suffix == ".csv":
                import csv
                check_payload(list(csv.DictReader(raw.decode("utf-8-sig").splitlines())), development=True)
            else:
                raise ValueError("Runner gate inputs use inspectable CSV/JSON only")
        for name in ("validation", "sign-stability", "industry-peer", "rewire-placebo"):
            events.append(skill_read("sc-gate-" + name, "gate_" + name.replace("-", "_")))
        path = REPO / "skills/sc-acceptance-ledger/scripts/acceptance_ledger.py"
        spec = importlib.util.spec_from_file_location("runner_acceptance_ledger", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ledger = Path(out) / "acceptance" / (digest(dict(bundle=bundle, input_sha256=hashes)) + ".jsonl")
        existing = module.read_ledger(ledger)
        row = existing[0] if existing else module.run(config, bundle_path.parent, ledger)
        if len(module.read_ledger(ledger)) != 1 or row["payload"]["config_sha256"] != module.digest(config):
            raise ValueError("Gate ledger binding mismatch")
        binding = dict(policy_sha256=digest(policy), case_ids=bundle["case_ids"],
                       input_sha256=hashes, ledger_row_hash=row["row_hash"])
        events.append(dict(event="acceptance_gates", binding=binding, receipt=row))
        return dict(decision=row["payload"]["decision"], accepted=row["payload"]["decision"] == "ACCEPT",
                    screening_only=True, binding=binding, events=events, label_accesses=list(access["accesses"]))


if __name__ == '__main__':
    import argparse
    from .label_firewall import static_verify
    parser = argparse.ArgumentParser(description='Verify local gate-4 traces without labels or model calls')
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    verified = []
    with boundary():
        static = static_verify()
        for path in sorted((args.run / 'case_traces').glob('*.json')):
            verify_case_trace(json.loads(path.read_text(encoding='utf-8')))
            verified.append(path.name)
    if not verified:
        raise SystemExit('No case traces; refusing a vacuous verification')
    print(json.dumps(dict(static=static, runtime_traces_verified=len(verified),
                         test_labels_accessed=False, evaluation_labels_accessed=False), indent=2))
