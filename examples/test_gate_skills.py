"""Offline acceptance arithmetic, structural rewiring and ledger invariants."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples import gate_skills_synthetic as toy

validation = toy.module("gate_validation")
sign = toy.module("gate_sign_stability")
peer = toy.module("gate_industry_peer")
rewire = toy.module("gate_rewire_placebo")

_path = toy.REPO / "skills/sc-acceptance-ledger/scripts/acceptance_ledger.py"
_spec = importlib.util.spec_from_file_location("acceptance_ledger", _path)
ledger = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ledger)


class ValidationTests(unittest.TestCase):
    def test_known_pass_fail_and_zero_gain(self):
        value = validation.score(toy.loss_rows())
        self.assertTrue(value["passed"])
        self.assertAlmostEqual(value["mse_reduction"], .5)
        self.assertEqual(value["week_se"], 0)
        self.assertFalse(validation.score(toy.loss_rows(11))["passed"])
        self.assertFalse(validation.score(toy.loss_rows(10))["passed"])
        self.assertFalse(validation.score(toy.loss_rows(), "o2")["passed"])

    def test_cluster_score_formula_and_strict_boundary(self):
        rows = toy.loss_rows()[:2]
        rows[1]["sse_candidate"] = 9.
        result = validation.score(rows)
        self.assertAlmostEqual(result["mse_reduction"], .3)
        self.assertAlmostEqual(result["week_se"], (8**.5)/20)
        multiplier = result["mse_reduction"] / result["week_se"]
        self.assertFalse(validation.score(rows, se_multiplier=multiplier)["passed"])

    def test_folds_training_dates_and_cluster_year_key(self):
        rows = toy.loss_rows()[:2]
        for key, value in (("fit_parity", 1), ("date", "2025-01-16"), ("n", 0), ("sse_candidate", float("nan"))):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.assertRaises(ValueError): validation.score(bad)
        with self.assertRaises(ValueError): validation.score(rows, "loop")
        rows = [dict(r, date="2016-01-21" if i else "2015-01-15") for i, r in enumerate(rows)]
        self.assertEqual(validation.score(rows)["weeks"], 2)
        rows[1].update(date="2015-01-22", fit_parity=1, sse_candidate=6.)
        self.assertTrue(validation.score(rows, "loop")["passed"])

    def test_prediction_and_sse_equivalence_and_parquet(self):
        # Per-row squared errors sum to the documented cross-products.
        rows = toy.loss_rows()[:2]
        for row in rows:
            y, old, new = [0., 0.], [1., 3.], [1., 2.]
            row.update(n=2, sse_incumbent=sum((a-b)**2 for a,b in zip(y,old)),
                       sse_candidate=sum((a-b)**2 for a,b in zip(y,new)))
        self.assertAlmostEqual(validation.score(rows)["mse_reduction"], 2.5)
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"toy.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            self.assertEqual(validation.score(validation.read_rows(path)), validation.score(rows))


class SignPeerTests(unittest.TestCase):
    def test_agent_four_of_seven_and_missing_year(self):
        data = toy.declaration()
        self.assertTrue(sign.evaluate(data)["passed"])
        data["years"].pop("2021")
        self.assertFalse(sign.evaluate(data)["passed"])
        data = toy.declaration(); data["pooled"]["estimate"] = -.2
        result = sign.evaluate(data)
        self.assertEqual(result["first_failed_gate"], "sign")
        self.assertIsNone(result["checks"]["stability"])

    def test_source_policy_differences_and_strict_t(self):
        data = toy.declaration(); data["estimate_split"] = "pooled"
        self.assertFalse(sign.evaluate(data, "o2")["passed"])
        self.assertTrue(sign.evaluate(data, "loop")["passed"])
        data["years"]["2015"]["t_week"] = 1.96
        self.assertFalse(sign.evaluate(data, "loop")["passed"])
        data["years"]["2019"]["estimate"] = .1
        self.assertTrue(sign.evaluate(data, "o2")["passed"])

    def test_declaration_and_two_sided(self):
        data = toy.declaration(); data.update(two_sided=True, ambiguity_rationale="Two predeclared mechanisms")
        self.assertTrue(sign.evaluate(data)["exploratory"])
        data.pop("ambiguity_rationale")
        with self.assertRaises(ValueError): sign.evaluate(data)
        data = toy.declaration(); data["declared_at"] = data["evaluated_at"]
        with self.assertRaises(ValueError): sign.evaluate(data)

    def test_peer_both_benchmarks_and_strict_tie(self):
        rows, contract = toy.peer_rows(), toy.peer_contract()
        self.assertTrue(peer.evaluate(rows, contract)["passed"])
        self.assertFalse(peer.evaluate([dict(r, sse_indpeer=5) for r in rows], contract)["passed"])
        contract["industries"] = ["ff49", "naics4"]
        rows = [dict(r, sse_ff49=7., sse_naics4=4.) for r in rows]
        self.assertFalse(peer.evaluate(rows, contract, "o2")["passed"])
        contract["same_operator"] = False
        with self.assertRaises(ValueError): peer.evaluate(rows, contract, "o2")


class PlaceboTests(unittest.TestCase):
    def test_seeded_directed_undirected_and_industry_degrees(self):
        for directed in (True, False):
            graph = toy.graph(directed)
            graph["nodes"].update(A="one", C="one", E="one", G="one", B="two", D="two", F="two", H="two")
            changed, audit = rewire.rewire(graph, 42)
            self.assertEqual(rewire.rewire(graph, 42)[0], changed)
            self.assertTrue(audit["industry_degrees_preserved"])
            self.assertGreater(audit["swaps"], 0)
            bad = copy.deepcopy(changed); bad["edges"][0]["old"] = False
            with self.assertRaises(ValueError): rewire.verify_rewire(graph, bad)

    def test_frozen_strata_report_no_movement(self):
        graph = toy.graph(); graph["edges"] = graph["edges"][:1]
        changed, receipt = rewire.rewire(graph)
        self.assertEqual(graph, changed)
        self.assertEqual(receipt["changed_fraction"], 0)

    def test_p_formulas_threshold_ties_and_two_sided(self):
        bundle = toy.placebo()
        value = rewire.evaluate(bundle)
        self.assertAlmostEqual(value["p"], 1/101)
        self.assertTrue(value["passed"])
        self.assertFalse(rewire.evaluate(bundle, threshold=1/101)["passed"])
        self.assertEqual(rewire.evaluate(bundle, "o2")["p"], 0)
        bundle["ambiguity_rationale"] = "Predeclared direction ambiguity"
        self.assertTrue(rewire.evaluate(bundle, two_sided=True)["exploratory"])
        for item in bundle["draws"]: item["rows"] = toy.loss_rows()
        self.assertEqual(rewire.evaluate(bundle)["p"], 1)

    def test_fail_closed_draw_count_sample_and_graph(self):
        bundle = toy.placebo(2)
        with self.assertRaises(ValueError): rewire.evaluate(bundle)
        bundle["draws"][0]["rows"][0]["n"] = 20
        with self.assertRaises(ValueError): rewire.evaluate(bundle, draws=2)
        bundle = toy.placebo(2); bundle["draws"][0]["graph"]["edges"][0]["target"] = "A"
        with self.assertRaises(ValueError): rewire.evaluate(bundle, draws=2)


class LedgerTests(unittest.TestCase):
    def test_pass_append_prefix_preserved_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            toy.write(directory)
            config = json.loads((Path(directory)/"acceptance.json").read_text())
            path = Path(directory)/"decisions.jsonl"
            first = ledger.run(config, directory, path)
            self.assertEqual(first["payload"]["decision"], "ACCEPT")
            before = path.read_bytes()
            config["attempt_id"] = "toy-second"
            second = ledger.run(config, directory, path, first["row_hash"])
            self.assertTrue(path.read_bytes().startswith(before))
            self.assertEqual(second["previous_hash"], first["row_hash"])
            self.assertEqual(len(ledger.read_ledger(path)), 2)
            with self.assertRaises(ValueError): ledger.run(config, directory, path)
            path.write_bytes(path.read_bytes().replace(b'"ACCEPT"', b'"REJECT"', 1))
            with self.assertRaises(ValueError): ledger.read_ledger(path)

    def test_first_failure_never_opens_later_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.csv"
            rows = toy.loss_rows(11)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            config = dict(attempt_id="fail-first", validation="bad.csv")
            value = ledger.run(config, directory, Path(directory)/"decisions.jsonl")["payload"]
            self.assertEqual(value["first_failed_gate"], "validation")
            self.assertIsNone(value["checks"]["rewire_placebo"])
            self.assertEqual(list(value["receipts"]), ["validation"])

    def test_partial_chain_expected_head_and_foreign_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"decisions.jsonl"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError): ledger.read_ledger(path)
            path.write_bytes(b"")
            with self.assertRaises(ValueError): ledger.append(path, dict(attempt_id="x"), "bad")
            lock = Path(str(path)+".lock"); lock.write_text("foreign")
            with self.assertRaises(FileExistsError): ledger.append(path, dict(attempt_id="x"))
            self.assertEqual(lock.read_text(), "foreign")

    def test_sign_stability_and_peer_failures_stop_before_placebo(self):
        for failed in ("sign", "stability", "industry_peer"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                toy.write(directory)
                root = Path(directory)
                config = json.loads((root/"acceptance.json").read_text())
                config["rewire_placebo"] = "must-not-be-opened.json"
                declaration = toy.declaration()
                if failed == "sign":
                    declaration["pooled"]["estimate"] = -.1
                elif failed == "stability":
                    declaration["years"].pop("2021")
                else:
                    rows = [dict(r, sse_graph=8.) for r in toy.peer_rows()]
                    with (root/"peers.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                (root/"sign.json").write_text(json.dumps(declaration))
                result = ledger.run(config, directory, root/"decisions.jsonl")["payload"]
                self.assertEqual(result["decision"], "REJECT")
                self.assertEqual(result["first_failed_gate"], failed)
                self.assertIsNone(result["checks"]["rewire_placebo"])

    def test_all_standalone_clis(self):
        with tempfile.TemporaryDirectory() as directory:
            toy.write(directory)
            commands = [(validation.__file__, "--input", "validation.csv"),
                        (sign.__file__, "--input", "sign.json"),
                        (peer.__file__, "--input", "peers.csv", "--contract", "peer_contract.json"),
                        (rewire.__file__, "evaluate", "--input", "placebo.json"),
                        (str(_path), "--config", "acceptance.json", "--ledger", "cli.jsonl")]
            for command in commands:
                proc = subprocess.run([sys.executable, *command], cwd=directory, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
                self.assertIsInstance(json.loads(proc.stdout), dict)

    def test_review_manifest_excludes_dirty_ablation_and_covers_new_files(self):
        from examples.prepare_review_manifest import prepare
        original = (toy.REPO/"metadata/file_manifest.json").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(toy.REPO, toy.REPO/"docs/PR2_EXTENSION_COMMIT_FILES.txt", root/"manifest.json",
                    root/"refresh.patch", root/"snapshot")
            entries = json.loads((root/"manifest.json").read_text())["files"]
            self.assertNotIn("metadata/file_manifest.json", {e["path"] for e in entries})
            # PR4 has since tracked this module. The helper must cover tracked
            # files as well as the explicit candidate additions.
            self.assertIn("source/code/news_network_50_2025/hop3_qwen.py", {e["path"] for e in entries})
            self.assertEqual((root/"snapshot/source/code/news_network_50_2025/graph_router.py").read_bytes(),
                             (toy.REPO/"source/code/news_network_50_2025/graph_router.py").read_bytes())
            for name in ("data.py", "runtime.py"):
                relative = "source/code/news_network_50_2025/" + name
                expected = subprocess.run(["git", "cat-file", "--filters", "HEAD:"+relative],
                    cwd=toy.REPO, capture_output=True, check=True).stdout
                self.assertEqual((root/"snapshot"/relative).read_bytes(), expected)
            for entry in entries:
                blob = (root/"snapshot"/entry["path"]).read_bytes()
                self.assertEqual(validation.hashlib.sha256(blob).hexdigest(), entry["sha256"])
        self.assertEqual((toy.REPO/"metadata/file_manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
