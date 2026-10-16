#!/usr/bin/env python3
"""Validate shareable dictionaries without opening raw data or model outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FORECAST_COLUMNS = {
    "name",
    "role",
    "units",
    "raw_fields",
    "formula",
    "timing",
    "caveat",
    "source_reference",
}
FORECAST_VARIABLES = {
    "y",
    "b",
    "rv1",
    "rv5",
    "rv22",
    "earnings_confirmed",
    "calendar_unknown",
    "post_earnings",
    "pending_cash_deal",
    "log_news_count",
    "log_neighbor_count",
    "log_degree",
    "anchor",
    "score",
}
CAUSAL_COLUMNS = {
    "name",
    "economic_meaning",
    "causal_role",
    "entity_level",
    "unit",
    "construction",
    "observation_time",
    "availability_time",
    "source_id",
    "raw_source_fields",
    "transformation",
    "missingness_meaning",
    "point_in_time_required",
    "predetermined_relative_to_shock",
    "leakage_risk",
    "dictionary_version",
}
CAUSAL_VARIABLES = {
    "supplier_disruption_shock",
    "supplier_to_customer_exposure",
    "inventory_buffer",
    "supplier_redundancy",
    "customer_operating_cash_flow",
    "customer_sales_or_production_shortfall",
    "customer_future_return",
    "customer_future_volatility",
}
CORE_HISTORICAL_FILES = {
    "source/code/news_network_50_2025/run.py",
    "source/code/news_network_50_2025/test_design.py",
    "source/code/news_network_50_2025/workflow.py",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check_manifest(repo: Path) -> list[str]:
    path = repo / "metadata/file_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("files", manifest if isinstance(manifest, list) else [])
    failures: list[str] = []
    for entry in entries:
        relative = entry.get("path")
        expected = entry.get("sha256") or entry.get("source_sha256")
        target = repo / str(relative)
        if not relative or not target.is_file():
            failures.append(str(relative or "<missing-path>"))
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected and actual != expected:
            failures.append(str(relative))
    return failures


def make_result(
    mode: str,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    status = "BLOCKED" if errors else ("QUALIFIED" if warnings else "PASS")
    return {
        "status": status,
        "mode": mode,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def audit(repo: Path, mode: str) -> tuple[dict[str, Any], int]:
    errors: list[str] = []
    warnings: list[str] = []
    dictionary_path = repo / "dictionaries/variables.csv"
    raw_fields_path = repo / "dictionaries/raw_fields.csv"
    verification_path = repo / "metadata/verification.json"
    source_manifest_path = repo / "metadata/source_manifest.csv"
    package_manifest_path = repo / "metadata/file_manifest.json"

    required_paths = (
        dictionary_path,
        raw_fields_path,
        verification_path,
        source_manifest_path,
        package_manifest_path,
    )
    for path in required_paths:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(repo)}")
    if errors:
        return make_result(mode, errors, warnings, {}), 2

    columns, rows = read_csv(dictionary_path)
    names = {(row.get("name") or "").strip() for row in rows}
    required_columns = FORECAST_COLUMNS if mode == "forecasting" else CAUSAL_COLUMNS
    required_variables = FORECAST_VARIABLES if mode == "forecasting" else CAUSAL_VARIABLES
    missing_columns = sorted(required_columns - set(columns))
    missing_variables = sorted(required_variables - names)
    if missing_columns:
        errors.append("missing dictionary columns: " + ", ".join(missing_columns))
    if missing_variables:
        errors.append("missing required variables: " + ", ".join(missing_variables))

    if mode == "forecasting":
        for row in rows:
            if row.get("name") in FORECAST_VARIABLES:
                empty = sorted(
                    column
                    for column in FORECAST_COLUMNS
                    if not (row.get(column) or "").strip()
                )
                if empty:
                    errors.append(f"{row.get('name')}: empty fields: {', '.join(empty)}")

    raw_columns, raw_rows = read_csv(raw_fields_path)
    expected_raw = {"dataset", "field", "type", "nullable", "meaning_and_use"}
    missing_raw_columns = sorted(expected_raw - set(raw_columns))
    if missing_raw_columns:
        errors.append("missing raw-field columns: " + ", ".join(missing_raw_columns))
    if any(
        not (row.get(column) or "").strip()
        for row in raw_rows
        for column in expected_raw
    ):
        errors.append("raw_fields.csv contains empty required cells")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("passed") is not True:
        errors.append("metadata verification did not pass")
    if verification.get("counts_and_garch_refitting"):
        warnings.append("counts and GARCH were not recomputed by package verification")
    if (
        verification.get("training_design_columns_including_intercept")
        != verification.get("training_design_rank")
    ):
        warnings.append("training design is rank deficient")

    _, source_rows = read_csv(source_manifest_path)
    mismatches = sorted(
        row.get("path", "")
        for row in source_rows
        if row.get("historical_status") == "differs_from_recorded_run"
    )
    core_mismatches = sorted(set(mismatches) & CORE_HISTORICAL_FILES)
    if core_mismatches:
        warnings.append("core source files differ from the recorded historical run")

    manifest_failures = check_manifest(repo)
    if manifest_failures:
        errors.append("package manifest mismatch: " + ", ".join(manifest_failures))

    details = {
        "dictionary_rows": len(rows),
        "raw_field_rows": len(raw_rows),
        "missing_columns": missing_columns,
        "missing_variables": missing_variables,
        "historical_source_mismatches": mismatches,
        "core_historical_source_mismatches": core_mismatches,
        "package_manifest_mismatches": manifest_failures,
    }
    report = make_result(mode, errors, warnings, details)
    return report, 2 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=("forecasting", "causal"), required=True)
    args = parser.parse_args()
    report, code = audit(args.repo.resolve(), args.mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
