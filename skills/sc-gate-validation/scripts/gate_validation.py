"""Standalone alternate-ISO-week validation gate; CSV or optional Parquet."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path

TRAIN = tuple(range(2015, 2022))
PROFILES = ("agent-v4", "loop", "o2")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numerical observation")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite numerical observation")
    return result


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == ".parquet":
        import pandas as pd  # optional; CSV needs only the standard library
        return pd.read_parquet(path).to_dict("records")
    raise ValueError("Expected .csv or .parquet")


def normalize(rows, profile="agent-v4"):
    if profile not in PROFILES:
        raise ValueError("Unknown policy profile")
    answer, identities, parities = [], set(), set()
    for row in rows:
        identity = str(row["row_id"])
        if not identity or identity in identities:
            raise ValueError("row_id must be nonempty and unique")
        identities.add(identity)
        day = date.fromisoformat(str(row["date"]))
        if day.year not in TRAIN:
            raise ValueError("Only calendar training years 2015--2021 are admissible")
        iso = day.isocalendar()
        parity = iso.week % 2
        fit = finite(row["fit_parity"])
        if fit not in (0, 1) or fit == parity or (profile != "loop" and parity != 1):
            raise ValueError("Held-out ISO week must oppose fit_parity; agent-v4/o2 score odd weeks only")
        parities.add(parity)
        n, old, new = (finite(row[k]) for k in ("n", "sse_incumbent", "sse_candidate"))
        if n <= 0 or n != int(n) or old < 0 or new < 0:
            raise ValueError("Positive integer n and nonnegative SSE required")
        answer.append(dict(row_id=identity, date=day.isoformat(), week=f"{iso.year}-W{iso.week:02}",
                           n=int(n), sse_incumbent=old, sse_candidate=new, fit_parity=int(fit)))
    if not answer or (profile == "loop" and parities != {0, 1}):
        raise ValueError("Empty data or missing bidirectional loop fold")
    return sorted(answer, key=lambda r: r["row_id"])


def score(rows, profile="agent-v4", se_multiplier=1.0):
    rows = normalize(rows, profile)
    multiplier = finite(se_multiplier)
    if multiplier <= 0:
        raise ValueError("SE multiplier must be positive")
    weeks = {}
    for row in rows:
        block = weeks.setdefault(row["week"], [0.0, 0])
        block[0] += row["sse_incumbent"] - row["sse_candidate"]
        block[1] += row["n"]
    if len(weeks) < 2:
        raise ValueError("At least two ISO-year/week clusters required")
    total_n = sum(r["n"] for r in rows)
    old, new = (sum(r[k] for r in rows) for k in ("sse_incumbent", "sse_candidate"))
    if old <= 0:
        raise ValueError("Incumbent SSE must be positive")
    gain = (old - new) / total_n
    se = math.sqrt(sum((d - gain * n)**2 for d, n in weeks.values())) / total_n
    # Legacy loop/O2 use a finite t_week > 1; upstream dm returns NaN
    # for zero score variance. v4 explicitly handles that edge case by SE.
    passed = gain > 0 and gain > multiplier * se and (profile == "agent-v4" or se > 0)
    return dict(gate="validation", profile=profile, passed=passed,
                mse_reduction=gain, week_se=se, se_multiplier=multiplier,
                gain_pct=100 * (1 - new / old), N=total_n, weeks=len(weeks),
                SSE_incumbent=old, SSE_candidate=new, input_sha256=digest(rows),
                folds="both even/odd directions" if profile == "loop" else "fit even, score odd",
                verification="ISO-year/week cluster scores; no finite-sample correction; input fit provenance is caller-attested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="agent-v4")
    parser.add_argument("--se-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    try:
        result = score(read_rows(args.input), args.profile, args.se_multiplier)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status="ERROR", error=str(error))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
