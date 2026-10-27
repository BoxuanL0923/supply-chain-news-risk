"""Standalone same-operator industry-peer loss benchmark."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Finite numeric inputs required")
    return float(value)


def evaluate(rows, contract, profile="agent-v4", se_multiplier=1.0):
    if profile not in ("agent-v4", "loop", "o2"):
        raise ValueError("Unknown profile")
    if not isinstance(contract.get("operator"), str) or not contract["operator"].strip():
        raise ValueError("Name the identical graph/peer feature operator")
    for key in ("same_operator", "exclude_self_and_supply_links", "shared_missingness_controls", "point_in_time"):
        if contract.get(key) is not True:
            raise ValueError(f"Required benchmark attestation: {key}")
    if finite(contract["size_factor"]) != 2:
        raise ValueError("Documented lagged-size matching factor is 2")
    peers = ("ff49", "naics4") if profile == "o2" else ("indpeer",)
    if set(contract["industries"]) != set(peers):
        raise ValueError("Peer industry contract disagrees with profile")
    multiplier = finite(se_multiplier)
    if multiplier <= 0:
        raise ValueError("SE multiplier must be positive")
    totals, clusters, ids, parities = {}, {}, set(), set()
    for row in rows:
        identity = str(row["row_id"])
        if not identity or identity in ids:
            raise ValueError("Unique row_id required")
        ids.add(identity)
        day = date.fromisoformat(str(row["date"]))
        iso = day.isocalendar()
        fit = finite(row["fit_parity"])
        if day.year not in range(2015, 2022) or fit not in (0, 1) or fit == iso.week % 2:
            raise ValueError("Pre-2022 alternate-week training folds required")
        if profile != "loop" and iso.week % 2 != 1:
            raise ValueError("agent-v4/o2 score odd ISO weeks")
        parities.add(iso.week % 2)
        n = finite(row["n"])
        graph = finite(row["sse_graph"])
        if n <= 0 or n != int(n) or graph < 0:
            raise ValueError("Positive integer n, nonnegative SSE required")
        for peer in peers:
            loss = finite(row[f"sse_{peer}"])
            if loss < 0:
                raise ValueError("Negative peer SSE")
            total = totals.setdefault(peer, [0., 0., 0.])
            total[0] += loss; total[1] += graph; total[2] += n
            block = clusters.setdefault(peer, {}).setdefault((iso.year, iso.week), [0., 0.])
            block[0] += loss - graph; block[1] += n
    if not totals or (profile == "loop" and parities != {0, 1}):
        raise ValueError("Empty data or missing bidirectional fold")
    comparisons = {}
    for peer, (old, new, n) in totals.items():
        if len(clusters[peer]) < 2:
            raise ValueError("At least two week clusters required")
        gain = (old - new) / n
        se = math.sqrt(sum((d - gain * count)**2 for d, count in clusters[peer].values())) / n
        comparisons[peer] = dict(graph_advantage=gain, week_se=se, N=int(n),
                                 passed=gain > (multiplier * se if profile == "loop" else 0) and
                                 (profile != "loop" or se > 0))
    payload = dict(rows=rows, contract=contract)
    return dict(gate="industry_peer", profile=profile, passed=all(v["passed"] for v in comparisons.values()),
                comparisons=comparisons, se_multiplier=multiplier if profile == "loop" else None,
                input_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                contract=contract, verification="Aligned row/count/fold loss comparison; peer construction and common-base fitting are caller-attested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--profile", choices=("agent-v4", "loop", "o2"), default="agent-v4")
    parser.add_argument("--se-multiplier", type=float, default=1.)
    args = parser.parse_args()
    try:
        path = Path(args.input)
        if path.suffix == ".parquet":
            import pandas as pd
            rows = pd.read_parquet(path).to_dict("records")
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            raise ValueError("Expected CSV or Parquet")
        result = evaluate(rows, json.loads(Path(args.contract).read_text(encoding="utf-8")), args.profile, args.se_multiplier)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status="ERROR", error=str(error))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
