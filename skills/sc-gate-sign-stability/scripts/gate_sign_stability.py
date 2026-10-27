"""Standalone predeclared-sign and complete-training-year stability gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def number(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Finite numerical estimates required")
    return float(value)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone-aware declaration and evaluation timestamps required")
    return result.astimezone(timezone.utc)


def evaluate(data, profile="agent-v4", required_years=None, t_cutoff=1.96):
    if profile not in ("agent-v4", "loop", "o2"):
        raise ValueError("Unknown profile")
    if timestamp(data["declared_at"]) >= timestamp(data["evaluated_at"]):
        raise ValueError("Sign must be declared before evaluation")
    if data["predicted_sign"] not in ("+", "-"):
        raise ValueError("predicted_sign must be + or -")
    two = data.get("two_sided", False)
    if type(two) is not bool:
        raise ValueError("two_sided must be boolean")
    if two and (profile != "agent-v4" or not data.get("ambiguity_rationale", "").strip() or data.get("post_hoc_of")):
        raise ValueError("Two-sided requires agent-v4, ex ante ambiguity rationale and no post-hoc reversal")
    expected_split = "odd" if profile == "agent-v4" else "pooled"
    if data["estimate_split"] != expected_split:
        raise ValueError(f"{profile} requires {expected_split} training coefficient estimates")
    pooled = number(data["pooled"]["estimate"])
    sign = (1 if pooled > 0 else -1 if pooled < 0 else 0) if two else (1 if data["predicted_sign"] == "+" else -1)
    required = (5 if profile == "o2" else 4) if required_years is None else required_years
    if type(required) is not int or not 1 <= required <= 7 or number(t_cutoff) <= 0:
        raise ValueError("required_years must be 1..7; t_cutoff must be positive")
    years = data["years"]
    if set(years) - {str(y) for y in range(2015, 2022)}:
        raise ValueError("Only calendar training years 2015--2021 allowed")
    sign_pass = pooled * sign > 0
    passing = []
    if sign_pass:
        for year, stat in sorted(years.items()):
            right = number(stat["estimate"]) * sign > 0
            if profile == "loop":
                right = right and all(number(stat[k]) * sign > t_cutoff for k in ("t_day", "t_week"))
            if right:
                passing.append(year)
    complete = set(years) == {str(y) for y in range(2015, 2022)}
    stability = complete and len(passing) >= required if sign_pass else None
    body = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(gate="sign_stability", profile=profile, passed=bool(sign_pass and stability),
                checks={"sign": sign_pass, "stability": stability},
                first_failed_gate="sign" if not sign_pass else "stability" if not stability else None,
                effective_sign=sign, exploratory=two, passing_years=passing, complete_years=complete,
                required_years=required, t_cutoff=t_cutoff if profile == "loop" else None,
                estimate_split=expected_split, input_sha256=hashlib.sha256(body.encode()).hexdigest(),
                verification="Declaration chronology checked; historical preregistration and coefficient fitting are caller-attested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", choices=("agent-v4", "loop", "o2"), default="agent-v4")
    parser.add_argument("--required-years", type=int)
    parser.add_argument("--t-cutoff", type=float, default=1.96)
    args = parser.parse_args()
    try:
        result = evaluate(json.loads(Path(args.input).read_text(encoding="utf-8")), args.profile, args.required_years, args.t_cutoff)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status="ERROR", error=str(error))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
