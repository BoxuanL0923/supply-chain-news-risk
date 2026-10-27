"""Offline, preregistered separation of Q1 optimizer recovery and evidence signal.

No client is constructed and no evaluation-label path is accepted. The protected
prompt_gd module supplies constants, provenance guards and the unchanged descent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import prompt_gd as pg

DATA = Path("D:/OptResearch_data/agent_q1/data_v2")
CACHE = Path("D:/OptResearch_data/agent_q1/run_v2/features.json")
OUT = Path("D:/OptResearch_data/agent_q1/run_v2_identify")
REPO = Path(__file__).resolve().parents[3]
REPORT = REPO / "docs/Q1_IDENTIFY_REPORT.md"
REPORT_COPY = Path("C:/OptResearch/results/sc_risk/agent_q1/Q1_IDENTIFY_REPORT.md")
SEED = 20261026
R_VALUES = (0., .1, .2, .4)
N_VALUES = (80, 200, 500, 2000)
REPLICATIONS = 10
STARTS = np.random.default_rng(SEED).uniform(.1, .9, (5, 7))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_settings(data=DATA, cache=CACHE):
    return {
        "module_sha256": sha(__file__),
        "protected_source_sha256": {name: sha(Path(__file__).with_name(name))
                                    for name in ("prompt_gd.py", "hop_ablation.py", "q1_run_v2.py")},
        "data": str(data), "cache": str(cache), "model": pg.MODEL,
        "cases_sha256": sha(Path(data)/"cases.json"), "cache_sha256": sha(cache),
        "seal": pg.SEAL, "engineering_only": True, "network_calls": False,
        "label_access": "Only prompt_gd.load_labels(data, cases, train/valid); never evaluation labels",
        "known_results_before_registration": {
            "source": "User supplied; gd.json was not opened for this analysis",
            "all_ten_original_runs_w": 0, "train_mse": .9925, "valid_baseline_mse": 1.5259,
            "train_corr_midpoint": -.067, "valid_corr_midpoint": .065},
        "names": list(pg.NAMES), "default": pg.DEFAULT.tolist(), "seed": SEED,
        "starts": STARTS.tolist(), "schedules": list(pg.PROTOCOL["schedules"]),
        "optimizer": {"implementation": "prompt_gd.descend", "iterations": 400, "step": .5,
                      "epsilon": pg.PROTOCOL["epsilon"], "delta": pg.PROTOCOL["delta"],
                      "patience": pg.PROTOCOL["patience"], "rm_exponent": .602,
                      "rm_denominator": "(1 + iteration / 10)**.602"},
        "D1": {"shapes": "DEFAULT and all-six-at-0.5", "scale": "signed s'y / s's; no intercept",
               "admissibility": "Negative w is diagnostic only, not an admissible prompt setting"},
        "VP": {"starts": "first six coordinates of the same five starts", "schedules": 2,
               "loss": "mean(y^2) - (s'y)^2/(n*s's)", "scale": "signed; no intercept",
               "zero_denominator": "w=0 and gradient=0 if s's <= 1e-24",
               "selection": "minimum final training loss; first in traversal breaks ties",
               "hessian": "central difference of analytic profile gradient, symmetric smooth extension at boundary",
               "hessian_steps": [1e-4, 5e-5], "hessian_primary_step": 1e-4,
               "R2": "uncentered versus zero residual; also report centered R2",
               "validation": "train-fitted signed w and max(0,w); also clip(w,0,1) for full admissibility"},
        "MAP": {"rho_multipliers": [1e-3, 1e-2, 1e-1], "rho_scale": "mean(y_train^2)",
                "penalty": "rho * sum((x-DEFAULT)^2), all seven normalized coordinates",
                "starts_and_schedules": "same five starts x two schedules; all three rho reported"},
        "synthetic": {"r": list(R_VALUES), "n": list(N_VALUES), "replications": REPLICATIONS,
                      "theta_star": pg.DEFAULT.tolist(), "starts": "same five starts for each replication",
                      "schedule": "constant", "fit": "original bounded MSE, no ridge, no profiling",
                      "seed_rule": "SeedSequence([20261026, 410, r_index, n_index, replication])",
                      "rng_order": "train indices, train Gaussian noise, validation indices, validation Gaussian noise",
                      "evidence": "all 120 cached development cases; n=80 without replacement; n>120 with replacement",
                      "validation": "fresh independent uniform evidence draw with replacement, same n and sigma",
                      "sigma_positive_r": "std(w_star*s(DEFAULT), ddof=0) over 120 cases * sqrt(r**-2 - 1)",
                      "correlation": "target population Pearson correlation under uniform empirical evidence distribution; realized correlations reported, not forced",
                      "r_zero": "pure-noise null y=eps, sigma=std(w_star*s(DEFAULT)); finite Gaussian noise cannot yield exactly zero population correlation with fixed nonzero signal",
                      "r_zero_parameter_error": "distance from DEFAULT reported descriptively; no recoverable true shape under null",
                      "selection": "minimum training MSE among five starts; no validation selection",
                      "MAE": "mean across replications of absolute error of training-best theta",
                      "GN": "eigenvalues of 2*J.T@J/n at training-best theta; full seven-dimensional normalized coordinates",
                      "gain": "fresh validation zero-score MSE minus fitted MSE; training-best main and per-run retained",
                      "runtime": "duplicate sampled cases compressed by exact count/sum(y)/sum(y^2) sufficient statistics"},
        "criteria": {"agreement_max_normalized_range": .05, "flat_eigenvalue_cutoff": 1e-6,
                     "corner_tolerance": 1e-12, "recovery_share": .8, "per_parameter_MAE": .10,
                     "optimizer_convergent_with_signal": "every r>=0.2,n>=500 cell has agreement share>=0.8 AND every parameter MAE<=0.10",
                     "real_data_identifiable": "VP primary Hessian has no eigenvalue below 1e-6 AND all ten starts/schedules agree",
                     "minimum_n": "smallest tested n passing cell agreement and every-parameter MAE thresholds; r=0 not meaningful",
                     "flat_parameter_diagnostic": "named parameters with DEFAULT GN diagonal<1e-6 or failed strong-cell MAE; inspect gate activity and Jacobian correlations"},
        "checks": {"equivalence": "all 120 cases, 20 uniform random theta, seed 20261027; score AND Jacobian atol=1e-12, rtol=0",
                   "finite_differences": "MSE, profile, ridge gradients; Hessian step sensitivity",
                   "memory": "single process, <=120 x 8 evidence arrays; peak working set reported; budget 2 GiB"},
        "validation_rule": "freeze all VP/MAP predictions on disk before loading valid labels; never select on validation",
        "outputs": ["PROTOCOL_IDENTIFY.json", "identify.json", "validation_prediction_freeze.json",
                    "label_access_audit.json", "recovery_error.png", "corner_rate.png",
                    str(REPORT), str(REPORT_COPY)],
    }


def register(out=OUT, data=DATA, cache=CACHE):
    if pg.SEAL != "2025-11-03":
        raise ValueError("Run with Q1_SEAL_DATE=2025-11-03")
    path = Path(out)/"PROTOCOL_IDENTIFY.json"
    settings = protocol_settings(data, cache)
    if path.exists():
        existing = pg.read(path)
        if existing["settings"] != settings:
            raise ValueError("Immutable preregistration differs; do not overwrite or silently amend")
        return existing
    value = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
             "labels_opened_before_registration": False, "settings": settings}
    pg.write(path, value)
    return value


def load_evidence(data=DATA, cache=CACHE):
    cases, saved = pg.load_cases(Path(data)/"cases.json"), pg.read(cache)
    if saved["model"] != pg.MODEL or saved["cases_hash"] != pg.digest(cases):
        raise ValueError("Cache provenance mismatch")
    features = {c["case_id"]: c["features"] for c in saved["cases"]}
    if len(features) != len(saved["cases"]) or set(features) != {c["case_id"] for c in cases}:
        raise ValueError("Cache case IDs mismatch")
    for case in cases:
        rows = features[case["case_id"]]
        if len(rows) != len(case["candidates"]):
            raise ValueError("Candidate count mismatch")
        for row, original in zip(rows, case["candidates"]):
            if any(row[k] != original[k] for k in ("source_id", "hop", "graph")):
                raise ValueError("Host-owned fields changed")
            for key, lo, hi in (("relevance", 0, 1), ("materiality", 0, 1), ("direction", -1, 1)):
                if type(row[key]) not in (int, float) or not np.isfinite(row[key]) or not lo <= row[key] <= hi:
                    raise ValueError("Invalid feature")
    if len(cases) != 120 or sum(c["split"] == "train" for c in cases) != 80:
        raise ValueError("Expected the registered 80/40 development bank")
    return cases, features


class Evidence:
    """Vectorized exact fixed-eight evidence score, without parameter division."""

    def __init__(self, rows):
        n = len(rows)
        self.hop = np.zeros((n, pg.BUDGET), dtype=int)
        self.base = np.zeros((n, pg.BUDGET))
        self.mat = np.zeros_like(self.base)
        self.degree = np.zeros_like(self.base)
        self.ra_pct = np.zeros_like(self.base)
        self.ra_positive = np.zeros_like(self.base, dtype=bool)
        self.clust = np.zeros_like(self.base)
        self.present = np.zeros_like(self.base, dtype=bool)
        for i, features in enumerate(rows):
            for j, e in enumerate(features[:pg.BUDGET]):
                self.present[i, j] = True
                self.hop[i, j] = e["hop"]
                self.base[i, j] = e["relevance"]*e["direction"]*e["materiality"]/pg.BUDGET
                self.mat[i, j] = e["materiality"]
                if e["hop"]:
                    g = e["graph"]
                    self.degree[i, j], self.ra_pct[i, j] = g["degree_percentile"], g["ra_percentile"]
                    self.ra_positive[i, j] = g["ra"] > 0
                    self.clust[i, j] = g["clustering"]
        self.neighbour = self.hop > 0
        self.h2, self.h3 = self.hop == 2, self.hop == 3

    def score(self, x, *, extension=False):
        x = np.asarray(x, float)
        if x.shape != (7,) or not np.isfinite(x).all() or (not extension and np.any((x < 0) | (x > 1))):
            raise ValueError("Seven finite unit-box parameters required")
        gate = lambda z: 1/(1+np.exp(-np.clip(z, -60, 60)))
        m = gate((self.mat-x[5])/.1)
        h = gate((self.degree-(50+50*x[2]))/10)
        r = gate((self.ra_pct-(50+50*x[3]))/10)*self.ra_positive
        c = gate((self.clust-x[4])/.1)
        M = np.where(self.neighbour, m, 1.)
        H = np.where(self.neighbour, 1-.5*h, 1.)
        R = np.where(self.neighbour, 1+.25*r, 1.)
        C = np.where(self.neighbour, 1-.5*c, 1.)
        L = np.where(self.h2, x[0], np.where(self.h3, x[1], 1.))
        base = self.base
        raw = base*L*M*H*R*C
        s = raw.sum(axis=1)
        j = np.empty((len(s), 7))
        j[:, 0] = (base*self.h2*M*H*R*C).sum(axis=1)*x[6]
        j[:, 1] = (base*self.h3*M*H*R*C).sum(axis=1)*x[6]
        common = base*L*self.neighbour
        j[:, 2] = (common*M*(2.5*h*(1-h))*R*C).sum(axis=1)*x[6]
        j[:, 3] = (common*M*H*(-1.25*r*(1-r))*C).sum(axis=1)*x[6]
        j[:, 4] = (common*M*H*R*(5*c*(1-c))).sum(axis=1)*x[6]
        j[:, 5] = (common*(-10*m*(1-m))*H*R*C).sum(axis=1)*x[6]
        j[:, 6] = s
        return s*x[6], j


class Moments:
    """Exact sampled MSE sufficient statistics, including duplicate evidence."""

    def __init__(self, bank, indices, y):
        self.bank = bank
        indices, y = np.asarray(indices, int), np.asarray(y, float)
        self.n = len(y)
        if not self.n or len(indices) != self.n or not np.isfinite(y).all():
            raise ValueError("Finite aligned nonempty observations required")
        self.count = np.bincount(indices, minlength=len(bank.base))/self.n
        self.sum_y = np.bincount(indices, weights=y, minlength=len(bank.base))/self.n
        self.y2 = float(np.mean(y*y))
        self.variance = float(np.var(y))

    def mse(self, x, rho=0.):
        s, j = self.bank.score(x)
        loss = self.y2 + np.dot(self.count, s*s) - 2*np.dot(self.sum_y, s)
        grad = 2*j.T@(self.count*s-self.sum_y)
        d = np.asarray(x)-pg.DEFAULT
        return float(loss+rho*np.dot(d, d)), grad+2*rho*d

    def profile(self, shape, *, extension=False):
        s, j = self.bank.score(np.r_[shape, 1.], extension=extension)
        a, b = np.dot(s, self.sum_y), np.dot(self.count, s*s)
        if b <= 1e-24:
            return self.y2, np.zeros(6), 0.
        w = a/b
        grad = 2*w*j[:, :6].T@(self.count*w*s-self.sum_y)
        return float(self.y2-a*a/b), grad, float(w)

    def gn(self, x):
        j = self.bank.score(x)[1]
        return 2*j.T@(self.count[:, None]*j)


def optimize(moment, *, rho=0., profile=False, schedules=None, compact=False):
    runs = []
    for schedule in schedules or pg.PROTOCOL["schedules"]:
        for i, initial in enumerate(STARTS):
            if profile:
                def fun(x):
                    loss, grad, _ = moment.profile(x[:6])
                    return loss, np.r_[grad, 0.]
            else:
                fun = lambda x: moment.mse(x, rho)
            run = pg.descend(fun, initial, schedule)
            if profile:
                run["theta"] = run["theta"][:6]
                run["parameters"].pop("w")
                run["w_signed"] = moment.profile(run["theta"])[2]
            run.update(start=i, schedule=schedule, initial_theta=initial[:6 if profile else 7].tolist())
            run["final"] = run["trajectory"][-1]
            if compact:
                run.pop("trajectory")
            runs.append(run)
    return runs


def agreement(runs):
    ranges = np.ptp([r["theta"] for r in runs], axis=0)
    return {"theta_range": ranges.tolist(), "all_starts_agree": bool(ranges.max() <= .05)}


def hessian(moment, shape, step=1e-4):
    eye = np.eye(6)*step
    h = np.column_stack([(moment.profile(np.asarray(shape)+d, extension=True)[1]
                          - moment.profile(np.asarray(shape)-d, extension=True)[1])/(2*step) for d in eye])
    return (h+h.T)/2


def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def check_equivalence(bank, rows):
    score_error, jac_error = 0., 0.
    for x in np.random.default_rng(SEED+1).uniform(0, 1, (20, 7)):
        s, j = bank.score(x)
        reference = [pg.smooth_score(x, row) for row in rows]
        sr, jr = np.array([v[0] for v in reference]), np.array([v[1] for v in reference])
        score_error = max(score_error, float(np.max(np.abs(s-sr))))
        jac_error = max(jac_error, float(np.max(np.abs(j-jr))))
    if max(score_error, jac_error) > 1e-12:
        raise AssertionError("Vectorized equivalence failed")
    return {"cases": len(rows), "random_theta": 20, "max_score_error": score_error,
            "max_jacobian_error": jac_error, "passed": True}


def evidence_diagnostics(bank):
    mask = bank.neighbour & bank.present
    s, j = bank.score(pg.DEFAULT)
    gn = 2*j.T@j/len(s)
    eigen, vectors = np.linalg.eigh(gn)
    def summary(a):
        return {"min": float(np.min(a)), "median": float(np.median(a)), "max": float(np.max(a)),
                "std": float(np.std(a))}
    result = {"used_evidence_rows": int(bank.present.sum()),
              "hop_counts": {str(h): int(((bank.hop == h)&bank.present).sum()) for h in range(4)},
              "nonzero_contribution_hop_counts": {str(h): int(((bank.hop == h)&(bank.base != 0)).sum()) for h in range(4)},
              "degree_percentile": summary(bank.degree[mask]), "ra_percentile": summary(bank.ra_pct[mask]),
              "clustering": summary(bank.clust[mask]), "materiality": summary(bank.mat[mask]),
              "positive_ra_share": float(bank.ra_positive[mask].mean()),
              "default_score": summary(s), "default_GN_diagonal": np.diag(gn).tolist(),
              "default_GN_eigenvalues": eigen.tolist(), "default_weakest_eigenvector": vectors[:, 0].tolist(),
              "default_GN_flat_count": int((eigen < 1e-6).sum()),
              "default_J_column_norms": np.linalg.norm(j, axis=0).tolist(),
              "default_J_cosine": (j.T@j / np.maximum(np.outer(np.linalg.norm(j, axis=0), np.linalg.norm(j, axis=0)), 1e-30)).tolist()}
    gate = lambda z: 1/(1+np.exp(-np.clip(z, -60, 60)))
    gates = {"THRESH_HUB": gate((bank.degree[mask]-90)/10),
             "THRESH_RA": gate((bank.ra_pct[mask]-75)/10)*bank.ra_positive[mask],
             "THRESH_CLUST": gate((bank.clust[mask]-.5)/.1),
             "materiality": gate((bank.mat[mask]-.5)/.1)}
    result["gates_at_DEFAULT"] = {k: {"value": summary(v), "transition_share_0.1_to_0.9": float(((v >= .1)&(v <= .9)).mean())} for k, v in gates.items()}
    return result


def synthetic_experiment(bank, out):
    planted = bank.score(pg.DEFAULT)[0]
    sd = float(np.std(planted))
    if sd == 0:
        raise ValueError("Cannot calibrate correlation with a constant planted score")
    cells = []
    for ri, r in enumerate(R_VALUES):
        sigma = sd*np.sqrt(r**-2-1) if r else sd
        for ni, n in enumerate(N_VALUES):
            reps = []
            for rep in range(REPLICATIONS):
                seed = [SEED, 410, ri, ni, rep]
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                indices = rng.choice(len(planted), n, replace=n > len(planted))
                y = (planted[indices] if r else 0.) + rng.normal(0, sigma, n)
                vi = rng.choice(len(planted), n, replace=True)
                vy = (planted[vi] if r else 0.) + rng.normal(0, sigma, n)
                moment = Moments(bank, indices, y)
                runs = optimize(moment, schedules=["constant"], compact=True)
                best_index = min(range(len(runs)), key=lambda k: runs[k]["final"]["loss"])
                best = runs[best_index]
                theta = np.array(best["theta"])
                eig = np.linalg.eigvalsh(moment.gn(theta))
                baseline = float(np.mean(vy*vy))
                for run in runs:
                    pred = bank.score(run["theta"])[0][vi]
                    run["validation_mse"] = float(np.mean((vy-pred)**2))
                    run["validation_gain"] = baseline-run["validation_mse"]
                    run["corner"] = bool(run["theta"][6] <= 1e-12)
                reps.append({"replication": rep, "seed_sequence": seed,
                             "train_unique_evidence": int(len(np.unique(indices))),
                             "train_sample_correlation": corr(planted[indices], y),
                             "validation_sample_correlation": corr(planted[vi], vy),
                             "best_run_index": best_index, "runs": runs, **agreement(runs),
                             "absolute_error": np.abs(theta-pg.DEFAULT).tolist(),
                             "GN_eigenvalues": eig.tolist(), "GN_flat_count": int((eig < 1e-6).sum()),
                             "validation_baseline_mse": baseline,
                             "validation_gain": best["validation_gain"]})
            mae = np.mean([q["absolute_error"] for q in reps], axis=0)
            share = float(np.mean([q["all_starts_agree"] for q in reps]))
            all_runs = [a for q in reps for a in q["runs"]]
            cell = {"r": r, "n": n, "sigma": float(sigma), "replications": reps,
                    "agreement_share": share, "MAE": mae.tolist(),
                    "GN_flat_count_mean": float(np.mean([q["GN_flat_count"] for q in reps])),
                    "GN_flat_counts": [q["GN_flat_count"] for q in reps],
                    "corner_share": float(np.mean([a["corner"] for a in all_runs])),
                    "validation_gain_mean": float(np.mean([q["validation_gain"] for q in reps])),
                    "validation_gain_std": float(np.std([q["validation_gain"] for q in reps], ddof=1)),
                    "validation_gain_positive_share": float(np.mean([q["validation_gain"] > 0 for q in reps])),
                    "train_sample_correlation_mean": float(np.mean([q["train_sample_correlation"] for q in reps])),
                    "stop_counts": {stop: sum(a["stop"] == stop for a in all_runs)
                                    for stop in ("max_iterations", "projected_gradient", "relative_loss")},
                    "passes_cell_recovery": bool(share >= .8 and np.all(mae <= .1))}
            cells.append(cell)
            pg.write(Path(out)/"synthetic.partial.json", {"cells": cells})
            print(json.dumps({"r": r, "n": n, "agreement": share, "MAE": mae.tolist(),
                              "corner": cell["corner_share"], "GN_flat": cell["GN_flat_count_mean"]}), flush=True)
    required = [c for c in cells if c["r"] >= .2 and c["n"] >= 500]
    minimum = {str(r): next((c["n"] for c in cells if c["r"] == r and c["passes_cell_recovery"]), None)
               if r else None for r in R_VALUES}
    return {"cells": cells, "optimizer_convergent_with_signal": all(c["passes_cell_recovery"] for c in required),
            "minimum_tested_n_passing": minimum,
            "strong_cell_failed_MAE_parameters": [name for i, name in enumerate(pg.NAMES) if any(c["MAE"][i] > .1 for c in required)]}


def guarded_labels(data, cases, split, out):
    registration = pg.read(Path(out)/"PROTOCOL_IDENTIFY.json")
    if registration["settings"]["module_sha256"] != sha(__file__):
        raise ValueError("Source changed after registration")
    if split not in ("train", "valid"):
        raise ValueError("Only development labels allowed")
    if split == "valid" and not (Path(out)/"validation_prediction_freeze.json").exists():
        raise ValueError("Validation predictions must be frozen first")
    audit_path = Path(out)/"label_access_audit.json"
    audit = pg.read(audit_path) if audit_path.exists() else {"accesses": []}
    audit["accesses"].append({"split": split, "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                              "module_sha256": sha(__file__), "registered_at_utc": registration["registered_at_utc"]})
    pg.write(audit_path, audit)
    return pg.load_labels(data, cases, split)


def real_experiment(bank, cases, data, out):
    train = np.array([i for i, c in enumerate(cases) if c["split"] == "train"])
    valid = np.array([i for i, c in enumerate(cases) if c["split"] == "valid"])
    labels = guarded_labels(data, cases, "train", out)
    y = np.array([labels[cases[i]["case_id"]]-cases[i]["anchor"] for i in train])
    moment = Moments(bank, train, y)
    d1 = {}
    for name, shape in (("DEFAULT", pg.DEFAULT[:6]), ("midpoint", np.full(6, .5))):
        loss, _, w = moment.profile(shape)
        s = bank.score(np.r_[shape, 1.])[0]
        d1[name] = {"shape": shape.tolist(), "w_signed": w, "sign": "negative" if w < 0 else "nonnegative",
                    "train_mse_signed": loss, "train_correlation": corr(s[train], y),
                    "admissible": bool(0 <= w <= 1)}
    vruns = optimize(moment, profile=True)
    best_index = min(range(len(vruns)), key=lambda k: vruns[k]["final"]["loss"])
    best = vruns[best_index]
    shape, w = np.array(best["theta"]), best["w_signed"]
    h = hessian(moment, shape)
    h2 = hessian(moment, shape, 5e-5)
    eig = np.linalg.eigvalsh(h)
    agree = agreement(vruns)
    loss = best["final"]["loss"]
    vp = {"runs": vruns, "best_run_index": best_index, "theta_hat": shape.tolist(),
          "w_signed": w, "w_sign": "negative" if w < 0 else "nonnegative",
          "train_mse": loss, "train_R2_zero_baseline": 1-loss/moment.y2,
          "train_R2_centered": 1-loss/moment.variance, **agree,
          "hessian": h.tolist(), "hessian_eigenvalues": eig.tolist(),
          "hessian_eigenvalues_half_step": np.linalg.eigvalsh(h2).tolist(),
          "hessian_step_max_absolute_difference": float(np.max(np.abs(h-h2))),
          "hessian_flat_count": int((eig < 1e-6).sum()),
          "real_data_identifiable": bool(np.all(eig >= 1e-6) and agree["all_starts_agree"])}
    maps = []
    frozen = {"VP": {}, "MAP": [], "D1": {}, "valid_ids": [cases[i]["case_id"] for i in valid]}
    for label, scale in (("signed", w), ("nonnegative", max(0., w)), ("unit_box", np.clip(w, 0, 1))):
        frozen["VP"][label] = (bank.score(np.r_[shape, 1.])[0][valid]*scale).tolist()
    for name, row in d1.items():
        s = bank.score(np.r_[row["shape"], 1.])[0][valid]
        frozen["D1"][name] = {"raw": s.tolist(), "signed": (s*row["w_signed"]).tolist()}
    for multiplier in (1e-3, 1e-2, 1e-1):
        rho = multiplier*moment.y2
        runs = optimize(moment, rho=rho)
        bi = min(range(len(runs)), key=lambda k: runs[k]["final"]["loss"])
        theta = np.array(runs[bi]["theta"])
        maps.append({"rho_multiplier": multiplier, "rho": rho, "runs": runs, "best_run_index": bi,
                     "theta_hat": theta.tolist(), **agreement(runs),
                     "signed_distance_from_DEFAULT": (theta-pg.DEFAULT).tolist(),
                     "absolute_distance_from_DEFAULT": np.abs(theta-pg.DEFAULT).tolist(),
                     "train_mse": moment.mse(theta)[0], "penalized_train_loss": runs[bi]["final"]["loss"]})
        frozen["MAP"].append({"rho_multiplier": multiplier,
                               "all_run_predictions": [bank.score(run["theta"])[0][valid].tolist() for run in runs],
                               "best_run_index": bi})
    # Real fitting and training-only selection are complete before this write/load.
    pg.write(Path(out)/"validation_prediction_freeze.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
             "module_sha256": sha(__file__), "VP": vp, "MAP": maps, "predictions": frozen})
    validation = guarded_labels(data, cases, "valid", out)
    vy = np.array([validation[cases[i]["case_id"]]-cases[i]["anchor"] for i in valid])
    baseline = float(np.mean(vy*vy))
    vp["validation_mse"] = {name: float(np.mean((vy-np.array(pred))**2)) for name, pred in frozen["VP"].items()}
    for row, preds in zip(maps, frozen["MAP"]):
        for run, pred in zip(row["runs"], preds["all_run_predictions"]):
            run["validation_mse"] = float(np.mean((vy-np.array(pred))**2))
        row["validation_mse"] = row["runs"][row["best_run_index"]]["validation_mse"]
    for name, row in d1.items():
        row["validation_correlation"] = corr(frozen["D1"][name]["raw"], vy)
        row["validation_mse_signed"] = float(np.mean((vy-np.array(frozen["D1"][name]["signed"]))**2))
    return {"train_n": len(train), "valid_n": len(valid), "train_baseline_mse": moment.y2,
            "validation_baseline_mse": baseline, "D1": d1, "VP": vp, "MAP": maps,
            "real_data_map_location": {"n": len(train), "absolute_train_correlation_midpoint": abs(d1["midpoint"]["train_correlation"]),
                                       "interpretation": "near r=0 to 0.1, n=80; real signed correlation is negative"}}


def peak_memory_bytes():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except ImportError:
        return None


def figures(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = result["synthetic"]["cells"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for i, name in enumerate(pg.NAMES):
        ax = axes.flat[i]
        for r in R_VALUES:
            rows = [c for c in cells if c["r"] == r]
            ax.plot([c["n"] for c in rows], [c["MAE"][i] for c in rows], "o-", label=f"r={r:g}")
        ax.axhline(.1, color="black", ls="--", lw=.8)
        ax.set_title(name); ax.set_xscale("log"); ax.set_xlabel("Sample size n"); ax.set_ylabel("Mean absolute error")
    axes.flat[7].axis("off")
    axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Planted recovery: 10 replications, training-best of 5 starts\nr=0 is a pure-noise control; its DEFAULT error is descriptive")
    fig.tight_layout(); fig.savefig(Path(out)/"recovery_error.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in R_VALUES:
        rows = [c for c in cells if c["r"] == r]
        axes[0].plot([c["n"] for c in rows], [c["corner_share"] for c in rows], "o-", label=f"r={r:g}")
    axes[0].set_xscale("log"); axes[0].set_ylim(-.02, 1.02)
    axes[0].set_xlabel("Sample size n"); axes[0].set_ylabel("Share of all 50 runs at w=0"); axes[0].legend()
    axes[0].set_title("Corner rate, original bounded objective")
    for c in cells:
        axes[1].scatter(c["n"], c["r"], color="tab:green" if c["passes_cell_recovery"] else "tab:gray", s=65)
    loc = result["real"]["real_data_map_location"]
    axes[1].scatter(loc["n"], loc["absolute_train_correlation_midpoint"], marker="*", color="tab:red", s=180, label="Real train |corr| (negative sign)")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Sample size n"); axes[1].set_ylabel("Target r / real |correlation|")
    axes[1].set_title("Recovery map: green passes, gray fails"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out)/"corner_rate.png", dpi=150); plt.close(fig)


def render_report(result, out, report=REPORT, report_copy=REPORT_COPY):
    real, synth, evidence = result["real"], result["synthetic"], result["evidence"]
    vp = real["VP"]
    fmt = lambda v: f"{v:.6g}"
    vec = lambda v: ", ".join(fmt(a) for a in v)
    lines = ["# Q1 Scheme A: optimizer recovery and evidence identifiability", "",
             "This is an engineering-only diagnostic of the cached smooth score, not a causal or financial validation.", "",
             "## Plain verdict", "",
             f"**(i) Optimizer convergent with signal: {'PASS' if synth['optimizer_convergent_with_signal'] else 'FAIL'} under the pre-declared joint convergence-and-recovery criterion.** "
             "A failed recovery criterion does not by itself prove faulty optimization: weak curvature, confounding, noise, and the fixed stopping budget can all prevent parameter recovery.", "",
             f"**(ii) Real data identifiable: {'YES' if vp['real_data_identifiable'] else 'NO'}.** "
             f"The signed-profile fit has {vp['hessian_flat_count']} Hessian eigenvalues below 1e-6; all starts agree: {vp['all_starts_agree']}. "
             f"At the midpoint, training correlation is {real['D1']['midpoint']['train_correlation']:.4f}, validation correlation is {real['D1']['midpoint']['validation_correlation']:.4f}. "
             f"Signed VP validation MSE is {vp['validation_mse']['signed']:.6f} against zero-score {real['validation_baseline_mse']:.6f}. "
             "These small development samples do not establish a reproducible, admissible predictive signal; they also cannot prove population signal is absent.", "",
             "The real-data corner is allowed. Negative signed scales are diagnostics, not admissible prompt settings. Ridge regularization supplies prior curvature and cannot establish data identification.", "",
             "## What was run and registered", "",
             f"Protocol: `{Path(out)/'PROTOCOL_IDENTIFY.json'}`. Registered UTC: {result['registration']['registered_at_utc']}. "
             f"Module SHA-256: `{result['registration']['settings']['module_sha256']}`. User-supplied original-run numbers were known in advance and recorded; no label file was read before registration.", "",
             "The cached qwen3.8-27b evidence comprises 80 Aug–Sep training and 40 October validation cases, sealed before 2025-11-03. "
             "All runs use normalized unit-box parameters, fixed evidence budget eight, seed 20261026, the original five uniform [.1,.9] starts, "
             "step 0.5, 400 iterations, projected-gradient tolerance 1e-5, relative-loss tolerance 1e-7 and patience five. "
             "VP/MAP use constant and Robbins–Monro schedules. Synthetic fits use constant only. No validation-based selection, network requests or LLM calls were made.", "",
             "The score/Jacobian are vectorized across cases. Repeated synthetic evidence is compressed into exact counts, sums of labels, and sums of squared labels; "
             "this retains the original MSE and gradient. The original `prompt_gd.descend` performs every update and stopping decision. "
             "For VP it receives six shape gradients and a zero dummy seventh gradient; the dummy coordinate is excluded from reported ranges and Hessians.", "",
             "## D1: signed scale diagnostic", "",
             "Here y means outcome minus anchor. Scale is s'y/s's without an intercept; a negative value is inadmissible. "
             "Pearson correlation is centered, whereas this scale uses uncentered inner products, so their signs need not agree in general.", "",
             "| Fixed shape | Signed w | Train correlation | Valid correlation | Train signed MSE | Valid signed MSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in real["D1"].items():
        lines.append(f"| {name} | {fmt(row['w_signed'])} | {fmt(row['train_correlation'])} | {fmt(row['validation_correlation'])} | {fmt(row['train_mse_signed'])} | {fmt(row['validation_mse_signed'])} |")
    lines += ["", "## Variable projection", "",
              f"Training-best shape: [{vec(vp['theta_hat'])}]. Signed scale: {fmt(vp['w_signed'])} ({vp['w_sign']}). "
              f"Train R² versus zero: {fmt(vp['train_R2_zero_baseline'])}; conventional centered R²: {fmt(vp['train_R2_centered'])}.", "",
              f"Final shape ranges over 5 starts × 2 schedules: [{vec(vp['theta_range'])}]. All agree at range ≤0.05: {vp['all_starts_agree']}.", "",
              f"Finite-difference Hessian eigenvalues: [{vec(vp['hessian_eigenvalues'])}]. "
              f"Half-step check: [{vec(vp['hessian_eigenvalues_half_step'])}]; largest Hessian entry difference {fmt(vp['hessian_step_max_absolute_difference'])}. "
              "These are the symmetric Hessian of the smooth extension at boundary solutions, not a constrained uniqueness certificate; negative eigenvalues count as below cutoff.", "",
              "| Prediction | Validation MSE | Gain vs zero |", "|---|---:|---:|",
              f"| Zero score | {fmt(real['validation_baseline_mse'])} | 0 |"]
    for name, mse in vp["validation_mse"].items():
        lines.append(f"| VP {name} | {fmt(mse)} | {fmt(real['validation_baseline_mse']-mse)} |")
    lines += ["", "`nonnegative` uses exactly max(0,w), as requested; `unit_box` additionally enforces the original upper bound one. All scales were fitted on train only.", "",
              "## MAP: all registered ridge strengths", "",
              "Agreement is empirical agreement across starts and schedules, not a proof of a globally unique minimum. All seven coordinates are penalized toward DEFAULT.", "",
              "| rho / mean(y²) | rho | All agree | Max theta range | Train MSE | Valid MSE | Gain vs zero |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in real["MAP"]:
        lines.append(f"| {row['rho_multiplier']:g} | {fmt(row['rho'])} | {row['all_starts_agree']} | {fmt(max(row['theta_range']))} | {fmt(row['train_mse'])} | {fmt(row['validation_mse'])} | {fmt(real['validation_baseline_mse']-row['validation_mse'])} |")
    lines += ["", "| Parameter | DEFAULT | " + " | ".join(f"rho×{row['rho_multiplier']:g}: estimate / signed movement" for row in real["MAP"]) + " |",
              "|---|---:|" + "---:|"*3]
    for i, name in enumerate(pg.NAMES):
        lines.append(f"| {name} | {fmt(pg.DEFAULT[i])} | " + " | ".join(f"{fmt(row['theta_hat'][i])} / {fmt(row['signed_distance_from_DEFAULT'][i])}" for row in real["MAP"]) + " |")
    lines += ["", "All per-parameter ranges and absolute prior distances, all starts, stopping reasons, gradients and trajectories are in `identify.json`.", "",
              "## Planted-signal recovery", "",
              "Theta* is DEFAULT = [" + vec(pg.DEFAULT) + "]. For positive r, y=w*s(theta*)+epsilon, with independent Gaussian noise and "
              "sigma=std(w*s(theta*)) sqrt(r^-2−1), calibrated over the uniform empirical bank of all 120 cases. "
              "Thus r is population correlation in that empirical design, not an enforced finite-sample correlation. "
              "At n=80, evidence is sampled without replacement; larger training samples and all fresh validation samples use replacement. "
              "Each cell has ten seeded replications and five starts. SeedSequence([20261026,410,r_index,n_index,replication]) fixes all draws. "
              "The r=0 row is pure noise with sigma equal to planted-score standard deviation: finite independent Gaussian noise cannot make a nonzero fixed signal exactly uncorrelated. "
              "Its errors against DEFAULT are descriptive only; true shape recovery is undefined in the null.", "",
              "MAE and GN eigenvalue counts use the minimum-training-loss start within each replication. Corner share uses all 50 runs. "
              "Validation gain is zero-score MSE minus fitted MSE on an independent synthetic draw of the same n; no model is chosen using it.", "",
              "| r | n | Agreement share | Max parameter MAE | Mean flat GN count | Corner share | Mean valid gain | SD valid gain | Realized train corr | Pass |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in synth["cells"]:
        lines.append(f"| {c['r']:g} | {c['n']} | {c['agreement_share']:.2f} | {max(c['MAE']):.4f} | {c['GN_flat_count_mean']:.2f} | {c['corner_share']:.2f} | {fmt(c['validation_gain_mean'])} | {fmt(c['validation_gain_std'])} | {c['train_sample_correlation_mean']:.3f} | {c['passes_cell_recovery']} |")
    lines += ["", "| r | n | " + " | ".join(pg.NAMES) + " |", "|---:|---:|" + "---:|"*7]
    for c in synth["cells"]:
        lines.append(f"| {c['r']:g} | {c['n']} | " + " | ".join(f"{v:.4f}" for v in c["MAE"]) + " |")
    lines += ["", "Minimum tested n meeting both ≥80% start agreement and ≤0.10 MAE for every parameter:", ""]
    for r, n in synth["minimum_tested_n_passing"].items():
        lines.append(f"- r={r}: " + ("not applicable (pure-noise null)." if float(r) == 0 else (str(n) if n else "not reached through n=2000.")))
    lines += ["", "Parameters exceeding the MAE threshold in at least one required strong-signal cell: " + ", ".join(synth["strong_cell_failed_MAE_parameters"]) + ".", "",
              f"Real data sit at n=80, |training correlation|={real['real_data_map_location']['absolute_train_correlation_midpoint']:.4f}, between the r=0 and 0.1 grid rows; "
              "the actual sign is negative. This location is descriptive and is not a formal power estimate for the real-data distribution.", "",
              "## Evidence geometry and weak parameters", "",
              f"Only the first eight evidence rows per case enter the score: {evidence['used_evidence_rows']} rows in total. Hop counts: {evidence['hop_counts']}; "
              f"nonzero signed-contribution counts: {evidence['nonzero_contribution_hop_counts']}.", "",
              "| Parameter | DEFAULT GN diagonal | Jacobian column norm |", "|---|---:|---:|"]
    for i, name in enumerate(pg.NAMES):
        lines.append(f"| {name} | {fmt(evidence['default_GN_diagonal'][i])} | {fmt(evidence['default_J_column_norms'][i])} |")
    lines += ["", f"DEFAULT full-bank GN eigenvalues: [{vec(evidence['default_GN_eigenvalues'])}]. "
              f"Weakest eigenvector in parameter order: [{vec(evidence['default_weakest_eigenvector'])}]. "
              "Diagonal curvature alone does not establish joint identification; correlated Jacobian columns can produce much weaker eigen-directions.", "",
              "| Neighbor feature | Min | Median | Max | Std |", "|---|---:|---:|---:|---:|"]
    for name in ("degree_percentile", "ra_percentile", "clustering", "materiality"):
        row = evidence[name]
        lines.append(f"| {name} | " + " | ".join(fmt(row[k]) for k in ("min", "median", "max", "std")) + " |")
    lines += ["", "| Gate | Median sigmoid at DEFAULT | Fraction in [0.1,0.9] |", "|---|---:|---:|"]
    for name, row in evidence["gates_at_DEFAULT"].items():
        lines.append(f"| {name} | {fmt(row['value']['median'])} | {fmt(row['transition_share_0.1_to_0.9'])} |")
    flat = [name for i, name in enumerate(pg.NAMES) if evidence["default_GN_diagonal"][i] < 1e-6]
    lines += ["", "Individually nearly flat DEFAULT directions (GN diagonal <1e-6): " + (", ".join(flat) or "none") + ". "
              "In particular, the clustering gate is evaluated far above most observed clustering values when its threshold is 0.5, so its sigmoid derivative is small. "
              "The tables quantify the activation and limited graph variation; the full Jacobian cosine matrix is retained in JSON to expose confounding. "
              "Weak gate activation limits recovery even when labels are generated by this model; a large n formed by resampling cannot add new graph support.", "",
              "## Limitations", "",
              "This tests a smooth engineering surrogate using previously cached LLM features, not fresh prompting or a structural economic model. "
              "The empirical bank has only 120 distinct cases; n=2000 reduces independent label noise conditional on resampled evidence, but is not 2000 independent real-world news environments. "
              "Training/validation synthetic draws can share evidence identities while their noise is independent. "
              "Gaussian synthetic noise and known correct model specification are favorable assumptions. Only ten replications estimate each cell's shares. "
              "The fixed 400-step, unscaled MSE gradient descent may stop before weak coordinates move appreciably. Failure of this exact protocol is not proof that no other optimizer or larger evidence design can recover parameters. "
              "Signed profiling permits negative or >1 scales solely for diagnosis; ridge can stabilize a prior-driven answer without creating evidence signal. "
              "No tested sample size beyond 2000 is inferred, and no interior real-data solution was forced.", "",
              "## Possible errors and checks performed", "",
              f"- Vectorization: all 120 cached cases at 20 seeded random theta, max score discrepancy {fmt(result['checks']['equivalence']['max_score_error'])}, "
              f"max Jacobian discrepancy {fmt(result['checks']['equivalence']['max_jacobian_error'])}, both below 1e-12.",
              "- New unittest suite checks boundary scores/Jacobians, finite-difference objective/profile/ridge gradients, duplicate-case sufficient statistics, scalar optimizer equivalence, and preregistration/validation guards.",
              f"- Hessian differencing: steps 1e-4 and 5e-5 agree to max entry difference {fmt(vp['hessian_step_max_absolute_difference'])}; boundary smooth-extension interpretation is explicit.",
              "- Cached model, case digest, feature IDs, host graph fields and feature bounds are checked; development origins and target sessions pass the original seal guards.",
              "- Real labels are residualized once by subtracting anchor. Only original load_labels(train/valid) is used. The prediction freeze precedes validation access, and label access times are recorded.",
              "- Source SHA-256 is registered before label access. The three protected source hashes are checked again after the experiment; the existing runner was not interrupted or modified.",
              "- Synthetic seeds, realized correlations, all final fits, stopping reasons, eigenvalues, prior distances and fresh validation gains are retained for audit. Population-r calibration and the r=0 exception are registered explicitly.",
              f"- Measured peak working set before plotting: {result['checks']['peak_memory_bytes_before_figures']} bytes. Compute elapsed: {result['elapsed_seconds_before_figures']:.2f} seconds. "
              "Arrays scale with 120 cached cases, and no parallel fitting workers are launched.", "",
              "## Artifacts", "",
              f"- Full results: `{Path(out)/'identify.json'}`",
              f"- Recovery figure: `{Path(out)/'recovery_error.png'}`",
              f"- Corner/map figure: `{Path(out)/'corner_rate.png'}`",
              f"- This report: `{report}`; copied to `{report_copy}`."]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines)+"\n", encoding="utf-8")
    Path(report_copy).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report, report_copy)


def run(data=DATA, cache=CACHE, out=OUT):
    started = time.perf_counter()
    registration = register(out, data, cache)  # Must precede every possible label access.
    cases, features = load_evidence(data, cache)
    rows = [features[c["case_id"]] for c in cases]
    bank = Evidence(rows)
    equivalence = check_equivalence(bank, rows)
    evidence = evidence_diagnostics(bank)
    synth = synthetic_experiment(bank, out)  # Main recovery experiment needs no real labels.
    real = real_experiment(bank, cases, data, out)
    protected = registration["settings"]["protected_source_sha256"]
    unchanged = all(sha(Path(__file__).with_name(name)) == value for name, value in protected.items())
    if not unchanged:
        raise ValueError("Protected dependency changed during this experiment")
    memory = peak_memory_bytes()
    if memory is not None and memory >= 2*1024**3:
        raise RuntimeError("Memory budget exceeded")
    result = {"registration": registration, "real": real, "synthetic": synth, "evidence": evidence,
              "checks": {"equivalence": equivalence, "protected_sources_unchanged": unchanged,
                         "evaluation_labels_opened": False, "network_or_LLM_calls": 0,
                         "peak_memory_bytes_before_figures": memory},
              "elapsed_seconds_before_figures": time.perf_counter()-started}
    figures(result, out)
    result["checks"]["peak_memory_bytes_after_figures"] = peak_memory_bytes()
    if result["checks"]["peak_memory_bytes_after_figures"] is not None and result["checks"]["peak_memory_bytes_after_figures"] >= 2*1024**3:
        raise RuntimeError("Memory budget exceeded during plotting")
    pg.write(Path(out)/"identify.json", result)
    render_report(result, out)
    print(json.dumps({"optimizer_convergent_with_signal": synth["optimizer_convergent_with_signal"],
                      "real_data_identifiable": real["VP"]["real_data_identifiable"],
                      "D1_midpoint_w": real["D1"]["midpoint"]["w_signed"],
                      "VP_w": real["VP"]["w_signed"], "VP_R2": real["VP"]["train_R2_zero_baseline"],
                      "VP_valid": real["VP"]["validation_mse"], "baseline": real["validation_baseline_mse"],
                      "minimum_n": synth["minimum_tested_n_passing"], "report": str(REPORT_COPY)}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(args.out, args.data, args.cache), indent=2))
    else:
        run(args.data, args.cache, args.out)


if __name__ == "__main__":
    main()
