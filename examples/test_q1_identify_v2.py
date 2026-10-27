"""Offline V2 checks. Register the design before executing optimizer tests."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("Q1_SEAL_DATE", "2025-11-03")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/code"))
from news_network_50_2025 import prompt_gd_identify_v2 as v


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration, cls.cases, cls.bank = v.register()

    def test_quantile_endpoints_monotonicity_and_derivative(self):
        for mapping in self.bank.maps:
            values = [mapping.at(q)[0] for q in np.linspace(0, 1, 101)]
            self.assertEqual(values[0], mapping.knots[0])
            self.assertEqual(values[-1], mapping.knots[-1])
            self.assertTrue(np.all(np.diff(values) >= -1e-12))
            for q in (.13, .47, .82):
                derivative = (mapping.at(q+1e-6)[0]-mapping.at(q-1e-6)[0])/2e-6
                self.assertAlmostEqual(mapping.at(q)[1], derivative, delta=2e-7)

    def test_reduced_full_and_scalar_score(self):
        bank = self.bank
        full_bank = v.Evidence(bank.original, bank.maps)
        for x in (np.zeros(4), np.ones(4), np.array([.2, .7, .3, .6])):
            s, j = bank.score(x)
            full = bank.full(x)
            fs, fj = full_bank.score(full)
            np.testing.assert_allclose(s, fs, atol=1e-15)
            np.testing.assert_allclose(j, fj[:, bank.active], atol=1e-15)
            b = bank.original
            expected = []
            for i in range(len(s)):
                total = 0.
                for k in range(v.pg.BUDGET):
                    value = b.base[i, k]
                    hop = b.hop[i, k]
                    value *= full[0] if hop == 2 else full[1] if hop == 3 else 1.
                    if hop:
                        h, r, c, m = [float(v.sigmoid((raw-map_.at(q)[0])/map_.temperature)) for raw, map_, q in
                                     zip((b.degree[i,k], b.ra_pct[i,k], b.clust[i,k], b.mat[i,k]), bank.maps, full[2:6])]
                        value *= (1-.5*h)*(1+.25*r*b.ra_positive[i,k])*(1-.5*c)*m
                    total += value
                expected.append(total*full[6])
            np.testing.assert_allclose(s, expected, atol=1e-15)

    def test_analytic_jacobian(self):
        self.assertLess(v.numerical_checks(self.bank)["jacobian_max_absolute_error"], 1e-8)

    def test_duplicate_moments_gradient_and_gn(self):
        rng = np.random.default_rng(44)
        idx = rng.choice(120, 300)
        y = rng.normal(0, .1, 300)
        moment = v.Moments(self.bank, idx, y)
        x = np.array([.3, .4, .6, .7])
        s, j = self.bank.score(x)
        loss, grad = moment.mse(x)
        self.assertAlmostEqual(loss, np.mean((s[idx]-y)**2), places=14)
        np.testing.assert_allclose(grad, 2*j[idx].T@(s[idx]-y)/len(y), atol=1e-14)
        np.testing.assert_allclose(moment.gn(x), 2*j[idx].T@j[idx]/len(y), atol=1e-14)
        numeric = [(moment.mse(x+d)[0]-moment.mse(x-d)[0])/2e-6 for d in np.eye(4)*1e-6]
        np.testing.assert_allclose(grad, numeric, atol=1e-10)

    def test_invalid_box_and_zero_scale(self):
        for x in ([0]*3, [-.1]*4, [np.nan]*4, [1.1]*4):
            with self.assertRaises(ValueError):
                self.bank.score(x)
        score, jac = self.bank.score([.5, .5, .5, 0])
        np.testing.assert_array_equal(score, 0)
        np.testing.assert_array_equal(jac[:, :3], 0)

    def test_quantile_roundtrip_and_prompt_physical_thresholds(self):
        for mapping in self.bank.maps:
            restored = v.Quantile(settings=mapping.settings())
            self.assertEqual(restored.at(.35), mapping.at(.35))
        prompt = self.bank.prompt([.5]*4)
        for mapping in self.bank.maps:
            self.assertIn(f"{mapping.at(.5)[0]:.8g}", prompt)

    def test_evidence_only_screen(self):
        d = self.registration["settings"]["diagnosis"]
        self.assertEqual(d["active"], [0, 1, 4, 6])
        self.assertGreater(d["remapped_gate_activity"]["THRESH_CLUST"]["all_neighbour_share"], .8)
        self.assertEqual(d["old"]["gates_at_DEFAULT"]["THRESH_CLUST"]["transition_share_0.1_to_0.9"], 0)

    def test_box_optimizers_and_kkt(self):
        class Quadratic:
            scale = 1.
            bank = type("Bank", (), {"active": [0, 6]})()
            def mse(self, x):
                residual = np.asarray(x)-np.array([.3, -.2])
                return float(residual@residual), 2*residual
            scaled = mse
        for method in ("GD", "L-BFGS-B"):
            result = v.fit(Quadratic(), [.8, .7], method)
            np.testing.assert_allclose(result["theta"], [.3, 0], atol=1e-7)
            self.assertLessEqual(result["projected_gradient_norm"], v.GTOL)

    def test_criterion_requires_all_cells_and_parameters(self):
        cells = [{"method": m, "r": r, "n": n, "agreement_share": .8, "MAE": [.1]*4}
                 for m in ("GD", "L-BFGS-B") for r in (.2, .4) for n in (500, 2000)]
        self.assertEqual(v.eligibility(cells), {"GD": True, "L-BFGS-B": True})
        cells[0]["MAE"][2] = .100001
        self.assertFalse(v.eligibility(cells)["GD"])
        self.assertFalse(v.eligibility(cells[:-1])["L-BFGS-B"])

    def test_immutable_and_source_guards(self):
        v.check_registration(v.OUT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"immutable.json"
            v.immutable_write(path, {"x": 1})
            v.immutable_write(path, {"x": 1})
            with self.assertRaises(ValueError):
                v.immutable_write(path, {"x": 2})
        with patch.object(v, "source_hashes", return_value={}):
            with self.assertRaises(ValueError):
                v.check_registration(v.OUT)

    def test_labels_forbidden_on_failed_recovery(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(v, "check_registration"), patch.object(v.pg, "load_labels") as loader:
            v.pg.write(Path(directory)/"PROTOCOL_IDENTIFY_V2.json", {})
            v.pg.write(Path(directory)/"synthetic_results.json", {
                "protocol_sha256": v.old.sha(Path(directory)/"PROTOCOL_IDENTIFY_V2.json"), "cells": []})
            for split in ("train", "valid", "eval"):
                with self.assertRaises(ValueError):
                    v.guarded_labels(self.cases, split, "L-BFGS-B", directory)
            loader.assert_not_called()

    def test_validation_freeze_required_after_pass(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(v, "check_registration"), patch.object(v.pg, "load_labels") as loader:
            v.pg.write(Path(directory)/"PROTOCOL_IDENTIFY_V2.json", {})
            cells = [{"method": "L-BFGS-B", "r": r, "n": n, "agreement_share": 1., "MAE": [0]*4}
                     for r in (.2, .4) for n in (500, 2000)]
            v.pg.write(Path(directory)/"synthetic_results.json", {
                "protocol_sha256": v.old.sha(Path(directory)/"PROTOCOL_IDENTIFY_V2.json"), "cells": cells})
            with self.assertRaisesRegex(ValueError, "Freeze"):
                v.guarded_labels(self.cases, "valid", "L-BFGS-B", directory)
            loader.assert_not_called()

    def test_singular_condition_and_projected_gradient(self):
        self.assertIsNone(v.condition([0, 1]))
        self.assertEqual(v.condition([.5, 1]), 2.)
        self.assertEqual(v.projected_norm(np.array([0., 1.]), np.array([2., -2.])), 0.)


if __name__ == "__main__":
    unittest.main()
