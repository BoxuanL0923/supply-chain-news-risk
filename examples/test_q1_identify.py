"""Offline unittest checks; no real labels or model clients are used."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("Q1_SEAL_DATE", "2025-11-03")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/code"))
from news_network_50_2025 import prompt_gd as pg
from news_network_50_2025 import prompt_gd_identify as identify


class IdentifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.features = identify.load_evidence()
        cls.rows = [cls.features[c["case_id"]] for c in cls.cases]
        cls.bank = identify.Evidence(cls.rows)

    def test_all_120_cases_twenty_random_theta_score_and_jacobian(self):
        self.assertEqual(len(self.rows), 120)
        for x in np.random.default_rng(20261027).uniform(0, 1, (20, 7)):
            score, jac = self.bank.score(x)
            reference = [pg.smooth_score(x, row) for row in self.rows]
            np.testing.assert_allclose(score, [r[0] for r in reference], atol=1e-12, rtol=0)
            np.testing.assert_allclose(jac, [r[1] for r in reference], atol=1e-12, rtol=0)

    def test_boundary_equivalence_and_zero_scale_shape_unidentified(self):
        for x in (np.zeros(7), np.ones(7), np.array([0, 1, 0, 1, 0, 1, 0.])):
            s, j = self.bank.score(x)
            reference = [pg.smooth_score(x, row) for row in self.rows]
            np.testing.assert_allclose(s, [r[0] for r in reference], atol=1e-12, rtol=0)
            np.testing.assert_allclose(j, [r[1] for r in reference], atol=1e-12, rtol=0)
            if x[6] == 0:
                np.testing.assert_array_equal(j[:, :6], 0)
        with self.assertRaises(ValueError):
            self.bank.score(np.full(7, -1.))

    def test_truncation_padding_and_ra_zero(self):
        rows = [[{"relevance": .8, "direction": (-1)**h, "materiality": .7, "hop": h % 4,
                  "graph": {"degree_percentile": 90., "ra_percentile": 80., "ra": 0., "clustering": .3}}
                 for h in range(count)] for count in (1, 4, 11)]
        bank = identify.Evidence(rows)
        s, j = bank.score(pg.DEFAULT)
        ref = [pg.smooth_score(pg.DEFAULT, row) for row in rows]
        np.testing.assert_allclose(s, [r[0] for r in ref], atol=1e-12, rtol=0)
        np.testing.assert_allclose(j, [r[1] for r in ref], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(j[:, 3], 0)

    def make_moment(self):
        rng = np.random.default_rng(13)
        indices = rng.choice(120, 500, replace=True)
        y = rng.normal(.02, .3, 500)
        return identify.Moments(self.bank, indices, y), indices, y

    def test_duplicate_moments_equal_expanded_original_objective(self):
        moment, indices, y = self.make_moment()
        x = pg.DEFAULT.copy()
        s, j = self.bank.score(x)
        loss, grad = moment.mse(x)
        error = s[indices]-y
        self.assertAlmostEqual(loss, np.mean(error**2), places=14)
        np.testing.assert_allclose(grad, 2*j[indices].T@error/len(y), atol=1e-14, rtol=0)
        np.testing.assert_allclose(moment.gn(x), 2*j[indices].T@j[indices]/len(y), atol=1e-14, rtol=0)
        # Also verify the original residualization convention, including anchors.
        cases = self.cases[:10]
        labels = {c["case_id"]: c["anchor"]+float(y[i]) for i, c in enumerate(cases)}
        reference = pg.objective(x, cases, self.features, labels)
        compact = identify.Moments(self.bank, np.arange(10), y[:10]).mse(x)
        self.assertAlmostEqual(reference[0], compact[0], places=14)
        np.testing.assert_allclose(reference[1], compact[1], atol=1e-14, rtol=0)

    def test_mse_ridge_and_profile_gradients(self):
        moment, indices, y = self.make_moment()
        x = np.array([.3, .2, .6, .4, .5, .7, .8])
        for rho in (0., .013):
            loss, grad = moment.mse(x, rho)
            numeric = [(moment.mse(x+d, rho)[0]-moment.mse(x-d, rho)[0])/2e-6
                       for d in np.eye(7)*1e-6]
            np.testing.assert_allclose(grad, numeric, atol=3e-10, rtol=1e-6)
        loss, grad, w = moment.profile(x[:6])
        s = self.bank.score(np.r_[x[:6], 1.])[0][indices]
        self.assertAlmostEqual(w, np.dot(s, y)/np.dot(s, s), places=12)
        self.assertAlmostEqual(loss, np.mean((w*s-y)**2), places=14)
        numeric = [(moment.profile(x[:6]+d)[0]-moment.profile(x[:6]-d)[0])/2e-6
                   for d in np.eye(6)*1e-6]
        np.testing.assert_allclose(grad, numeric, atol=3e-10, rtol=1e-6)

    def test_signed_profile_negative_scale_and_zero_denominator(self):
        s = self.bank.score(np.r_[pg.DEFAULT[:6], 1.])[0]
        m = identify.Moments(self.bank, np.arange(120), -2*s)
        loss, grad, w = m.profile(pg.DEFAULT[:6])
        self.assertAlmostEqual(w, -2.)
        self.assertAlmostEqual(loss, 0., places=14)
        np.testing.assert_allclose(grad, 0, atol=1e-14)
        empty = identify.Evidence([[]])
        loss, grad, w = identify.Moments(empty, [0], [3.]).profile(pg.DEFAULT[:6])
        self.assertEqual((loss, w), (9., 0.))
        np.testing.assert_array_equal(grad, 0)

    def test_original_descent_stop_and_trajectory_equivalence(self):
        cases = self.cases[:10]
        y = np.linspace(-.2, .4, 10)
        labels = {c["case_id"]: c["anchor"]+float(v) for c, v in zip(cases, y)}
        moment = identify.Moments(self.bank, np.arange(10), y)
        for schedule in pg.PROTOCOL["schedules"]:
            direct = pg.descend(lambda x: pg.objective(x, cases, self.features, labels),
                                identify.STARTS[0], schedule, iterations=12)
            fast = pg.descend(moment.mse, identify.STARTS[0], schedule, iterations=12)
            self.assertEqual(direct["stop"], fast["stop"])
            np.testing.assert_allclose(direct["theta"], fast["theta"], atol=1e-13, rtol=0)
            np.testing.assert_allclose([v["loss"] for v in direct["trajectory"]],
                                       [v["loss"] for v in fast["trajectory"]], atol=1e-13, rtol=0)

    def test_hessian_and_population_correlation_calibration(self):
        moment, _, _ = self.make_moment()
        h = identify.hessian(moment, pg.DEFAULT[:6])
        np.testing.assert_allclose(h, h.T, atol=1e-15)
        np.testing.assert_allclose(h, identify.hessian(moment, pg.DEFAULT[:6], 5e-5), atol=1e-8)
        variance = np.var(self.bank.score(pg.DEFAULT)[0])
        for r in (.1, .2, .4):
            sigma2 = variance*(r**-2-1)
            self.assertAlmostEqual(np.sqrt(variance/(variance+sigma2)), r, places=14)

    def test_registration_and_validation_guards_before_label_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(pg, "load_labels", side_effect=AssertionError("Label access forbidden")) as loader:
                registration = identify.register(out)
                self.assertEqual(registration["settings"]["module_sha256"], identify.sha(identify.__file__))
                self.assertEqual(registration, identify.register(out))
                with self.assertRaises(ValueError):
                    identify.guarded_labels(identify.DATA, self.cases, "evaluation", out)
                with self.assertRaises(ValueError):
                    identify.guarded_labels(identify.DATA, self.cases, "valid", out)
                loader.assert_not_called()
            calls = []
            def fake_loader(data, cases, split):
                self.assertTrue((out/"PROTOCOL_IDENTIFY.json").exists())
                self.assertTrue((out/"label_access_audit.json").exists())
                calls.append(split)
                return {}
            with patch.object(pg, "load_labels", side_effect=fake_loader):
                identify.guarded_labels(identify.DATA, self.cases, "train", out)
                pg.write(out/"validation_prediction_freeze.json", {"test_only": True})
                identify.guarded_labels(identify.DATA, self.cases, "valid", out)
            self.assertEqual(calls, ["train", "valid"])
            registration["settings"]["module_sha256"] = "changed"
            pg.write(out/"PROTOCOL_IDENTIFY.json", registration)
            with self.assertRaises(ValueError):
                identify.register(out)
            with patch.object(pg, "load_labels") as loader:
                with self.assertRaises(ValueError):
                    identify.guarded_labels(identify.DATA, self.cases, "train", out)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
