#!/usr/bin/env python3
"""Tests for the fail-closed dictionary validator."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_dictionary.py")
SPEC = importlib.util.spec_from_file_location("validate_dictionary", MODULE_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ValidatorTests(unittest.TestCase):
    def fixture(self) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        repo = Path(temp.name)
        rows = [
            {
                column: (name if column == "name" else "defined")
                for column in VALIDATOR.FORECAST_COLUMNS
            }
            for name in sorted(VALIDATOR.FORECAST_VARIABLES)
        ]
        write_csv(
            repo / "dictionaries/variables.csv",
            sorted(VALIDATOR.FORECAST_COLUMNS),
            rows,
        )
        write_csv(
            repo / "dictionaries/raw_fields.csv",
            ["dataset", "field", "type", "nullable", "meaning_and_use"],
            [
                {
                    "dataset": "source",
                    "field": "value",
                    "type": "float",
                    "nullable": "false",
                    "meaning_and_use": "test",
                }
            ],
        )
        (repo / "metadata").mkdir(parents=True, exist_ok=True)
        (repo / "metadata/verification.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "training_design_columns_including_intercept": 1,
                    "training_design_rank": 1,
                }
            ),
            encoding="utf-8",
        )
        write_csv(
            repo / "metadata/source_manifest.csv",
            ["path", "historical_status"],
            [
                {
                    "path": "source/code/example.py",
                    "historical_status": "matches_recorded_run",
                }
            ],
        )
        files = []
        for path in sorted(repo.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(repo)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        (repo / "metadata/file_manifest.json").write_text(
            json.dumps({"files": files}), encoding="utf-8"
        )
        return repo

    def test_forecasting_fixture_passes(self) -> None:
        report, code = VALIDATOR.audit(self.fixture(), "forecasting")
        self.assertEqual(0, code)
        self.assertEqual("PASS", report["status"])

    def test_causal_mode_fails_closed(self) -> None:
        report, code = VALIDATOR.audit(self.fixture(), "causal")
        self.assertEqual(2, code)
        self.assertEqual("BLOCKED", report["status"])
        self.assertTrue(report["details"]["missing_columns"])
        self.assertTrue(report["details"]["missing_variables"])

    def test_manifest_mismatch_blocks(self) -> None:
        repo = self.fixture()
        with (repo / "dictionaries/variables.csv").open("a", encoding="utf-8") as handle:
            handle.write("tampered\n")
        report, code = VALIDATOR.audit(repo, "forecasting")
        self.assertEqual(2, code)
        self.assertIn("package manifest mismatch", " ".join(report["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
