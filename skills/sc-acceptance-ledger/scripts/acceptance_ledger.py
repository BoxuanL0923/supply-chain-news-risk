"""Compose repository gate scripts and append verified hash-chained decisions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ZERO = "0" * 64
ORDER = ("validation", "sign", "stability", "industry_peer", "rewire_placebo")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_gate(name):
    filename = "gate_" + name + ".py"
    path = Path(__file__).resolve().parents[2] / ("sc-gate-" + name.replace("_", "-")) / "scripts" / filename
    spec = importlib.util.spec_from_file_location("acceptance_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_ledger(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Partial ledger row; preserve file and stop")
    rows, previous = [], ZERO
    for index, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        body = {k: v for k, v in row.items() if k != "row_hash"}
        if row["index"] != index or row["previous_hash"] != previous or digest(body) != row["row_hash"]:
            raise ValueError("Broken ledger hash chain")
        rows.append(row); previous = row["row_hash"]
    return rows


@contextmanager
def ledger_lock(path):
    lock = Path(str(path) + ".lock")
    # Fail immediately on contention. A stale foreign lock is never removed.
    stream = lock.open("x", encoding="utf-8")
    try:
        stream.write(str(os.getpid())); stream.flush()
        yield
    finally:
        stream.close()
        lock.unlink()


def append(path, payload, expected_head=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_lock(path):
        rows = read_ledger(path)
        previous = rows[-1]["row_hash"] if rows else ZERO
        if expected_head is not None and previous != expected_head:
            raise ValueError("Ledger differs from trusted expected head")
        if any(r["payload"]["attempt_id"] == payload["attempt_id"] for r in rows):
            raise ValueError("attempt_id already recorded; append a new attempt ID")
        body = dict(index=len(rows) + 1, previous_hash=previous, payload=payload)
        row = dict(body, row_hash=digest(body))
        with path.open("ab") as stream:
            stream.write((canonical(row) + "\n").encode())
            stream.flush(); os.fsync(stream.fileno())
        return row


def run(config, directory, ledger, expected_head=None):
    directory = Path(directory)
    allowed = {"attempt_id", "profile", "validation", "sign_stability", "industry_peer", "rewire_placebo",
               "se_multiplier", "required_years", "t_cutoff", "draws", "placebo_threshold", "two_sided", "action"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(set(config) - allowed)))
    if config.get("action", "add") not in ("add", "replace"):
        raise ValueError("Only addition/replacement screens; removal uses a separate nonworsening exception")
    profile = config.get("profile", "agent-v4")
    if profile not in ("agent-v4", "loop", "o2"):
        raise ValueError("Unknown profile")
    if not isinstance(config["attempt_id"], str) or not config["attempt_id"].strip():
        raise ValueError("Nonempty attempt_id required")
    result = dict(attempt_id=config["attempt_id"], profile=profile, order=list(ORDER),
                  screening_only=True,
                  config_sha256=digest(config), evaluated_at=datetime.now(timezone.utc).isoformat(),
                  checks={k: None for k in ORDER}, receipts={}, decision="REJECT", first_failed_gate=None)
    modules = {name: load_gate(name) for name in ("validation", "sign_stability", "industry_peer", "rewire_placebo")}
    result["implementation_sha256"] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                        for name, module in modules.items()}
    result["implementation_sha256"]["ledger"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    read_json = lambda name: json.loads((directory / config[name]).read_text(encoding="utf-8"))
    current = "validation"
    try:
        receipt = modules["validation"].score(modules["validation"].read_rows(directory / config["validation"]),
                                               profile, config.get("se_multiplier", 1.))
        result["receipts"][current] = receipt
        result["checks"][current] = receipt["passed"]
        if receipt["passed"]:
            current = "sign"
            declaration = read_json("sign_stability")
            if declaration.get("two_sided", False) != config.get("two_sided", False):
                raise ValueError("Sign and placebo sidedness declarations disagree")
            receipt = modules["sign_stability"].evaluate(declaration, profile,
                config.get("required_years"), config.get("t_cutoff", 1.96))
            result["receipts"]["sign_stability"] = receipt
            result["checks"].update(receipt["checks"])
            current = receipt["first_failed_gate"] or "industry_peer"
            if receipt["passed"]:
                peer = config["industry_peer"]
                receipt = modules["industry_peer"].evaluate(modules["validation"].read_rows(directory / peer["rows"]),
                    json.loads((directory / peer["contract"]).read_text(encoding="utf-8")), profile,
                    config.get("se_multiplier", 1.))
                result["receipts"][current] = receipt
                result["checks"][current] = receipt["passed"]
                if receipt["passed"]:
                    current = "rewire_placebo"
                    bundle = read_json("rewire_placebo")
                    if config.get("two_sided", False) and bundle.get("ambiguity_rationale") != declaration.get("ambiguity_rationale"):
                        raise ValueError("Two-sided rationale differs between sign and placebo inputs")
                    receipt = modules["rewire_placebo"].evaluate(bundle, profile,
                        config.get("placebo_threshold", .05), config.get("draws", 100), config.get("two_sided", False))
                    result["receipts"][current] = receipt
                    result["checks"][current] = receipt["passed"]
        if all(v is True for v in result["checks"].values()):
            result["decision"] = "ACCEPT"
        else:
            result["first_failed_gate"] = next(k for k, v in result["checks"].items() if v is False)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result.update(decision="ERROR", first_failed_gate=current, error=f"{type(error).__name__}: {error}")
        result["checks"][current] = False
    return append(ledger, result, expected_head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    try:
        if args.verify_only:
            rows = read_ledger(args.ledger)
            head = rows[-1]["row_hash"] if rows else ZERO
            if args.expected_head is not None and head != args.expected_head:
                raise ValueError("Ledger differs from trusted expected head")
            result = dict(verified=True, rows=len(rows), head=head)
            code = 0
        else:
            if not args.config:
                raise ValueError("--config is required unless --verify-only")
            path = Path(args.config)
            result = run(json.loads(path.read_text(encoding="utf-8")), path.parent, args.ledger, args.expected_head)
            decision = result["payload"]["decision"]
            code = 0 if decision == "ACCEPT" else 1 if decision == "REJECT" else 2
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status="ERROR", error=str(error))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
