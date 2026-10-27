"""Offline Q1 V2: evidence-only design, immutable registration, recovery gate.

No network/client entry points; evaluation labels are never accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from . import prompt_gd_identify as old

pg = old.pg
OUT = Path("D:/OptResearch_data/agent_q1/run_v2_identify_v2")
REPORT = old.REPO / "docs/Q1_IDENTIFY_V2_REPORT.md"
REPORT_COPY = Path("C:/OptResearch/results/sc_risk/agent_q1/Q1_IDENTIFY_V2_REPORT.md")
SEED = 20261026
GATE_NAMES = tuple(pg.NAMES[2:6])
MAXITER = 4000
GTOL = 1e-7
FTOL = 1e-12


def utc():
    return datetime.now(timezone.utc).isoformat()


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


class Quantile:
    """Smooth monotone Bernstein approximation to the empirical quantile.

Twenty equal quantile intervals; endpoints are exact observed min/max.
Unlike raw inverse ECDF, ties do not create derivative discontinuities.
"""
    def __init__(self, values=None, settings=None):
        self.knots = np.quantile(values, np.linspace(0, 1, 21)) if settings is None else np.array(settings["knots"])
        self.temperature = max(float(np.std(values)) * .5, 1e-12) if settings is None else settings["temperature"]
        self.coef = np.array([math.comb(20, k) for k in range(21)])
        self.dcoef = np.array([math.comb(19, k) for k in range(20)])
        self.k = np.arange(21)

    def at(self, q):
        weights = self.coef * q**self.k * (1-q)**(20-self.k)
        dw = self.dcoef * q**self.k[:-1] * (1-q)**(19-self.k[:-1])
        return float(weights @ self.knots), float(20 * dw @ np.diff(self.knots))

    def settings(self):
        return {"knots": self.knots.tolist(), "temperature": self.temperature}


class Evidence:
    def __init__(self, original, maps, active=None):
        self.original = original
        self.base = original.base
        self.maps = maps
        self.active = list(range(7)) if active is None else list(active)
        self.names = [pg.NAMES[i] for i in self.active]
        self.fixed_gates = None
        if self.active == [0, 1, 4, 6]:
            (h, r, _, m), _ = self.gates(np.full(7, .5))
            self.fixed_gates = self.base * np.where(original.neighbour, m*(1-.5*h)*(1+.25*r), 1.)

    def full(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.active),) or not np.isfinite(theta).all() or np.any((theta < 0) | (theta > 1)):
            raise ValueError("Finite unit-box coordinates required")
        full = np.full(7, .5)
        full[self.active] = theta
        return full

    def gates(self, full):
        b = self.original
        inputs = [b.degree, b.ra_pct, b.clust, b.mat]
        gates, derivatives = [], []
        for q, values, mapping in zip(full[2:6], inputs, self.maps):
            threshold, slope = mapping.at(q)
            g = sigmoid((values-threshold)/mapping.temperature)
            dg = -g*(1-g)*slope/mapping.temperature
            gates.append(g)
            derivatives.append(dg)
        gates[1] = gates[1]*b.ra_positive
        derivatives[1] = derivatives[1]*b.ra_positive
        return gates, derivatives

    def score(self, theta):
        x = self.full(theta)
        b = self.original
        if self.fixed_gates is not None:
            threshold, slope = self.maps[2].at(x[4])
            c = sigmoid((b.clust-threshold)/self.maps[2].temperature)
            C = np.where(b.neighbour, 1-.5*c, 1.)
            L = np.where(b.h2, x[0], np.where(b.h3, x[1], 1.))
            raw = self.fixed_gates*C
            s = (raw*L).sum(axis=1)
            j = np.column_stack([(raw*b.h2).sum(axis=1)*x[6],
                                 (raw*b.h3).sum(axis=1)*x[6],
                                 (self.fixed_gates*L*b.neighbour*.5*c*(1-c)*slope/self.maps[2].temperature).sum(axis=1)*x[6], s])
            return s*x[6], j
        (h, r, c, m), (dh, dr, dc, dm) = self.gates(x)
        H = np.where(b.neighbour, 1-.5*h, 1.)
        R = np.where(b.neighbour, 1+.25*r, 1.)
        C = np.where(b.neighbour, 1-.5*c, 1.)
        M = np.where(b.neighbour, m, 1.)
        L = np.where(b.h2, x[0], np.where(b.h3, x[1], 1.))
        raw = b.base*L*M*H*R*C
        s = raw.sum(axis=1)
        common = b.base*L*b.neighbour
        j = np.column_stack([
            (b.base*b.h2*M*H*R*C).sum(axis=1)*x[6],
            (b.base*b.h3*M*H*R*C).sum(axis=1)*x[6],
            (common*M*(-.5*dh)*R*C).sum(axis=1)*x[6],
            (common*M*H*(.25*dr)*C).sum(axis=1)*x[6],
            (common*M*H*R*(-.5*dc)).sum(axis=1)*x[6],
            (common*dm*H*R*C).sum(axis=1)*x[6], s])
        return s*x[6], j[:, self.active]

    def prompt(self, theta):
        x = self.full(theta)
        thresholds = [m.at(q)[0] for m, q in zip(self.maps, x[2:6])]
        return (f"Use the first eight evidence rows. Multiply signed relevance times materiality by "
                f"hop-2 weight {x[0]:.8g}, hop-3 weight {x[1]:.8g} (hop 0/1 weight 1). "
                f"For neighbours use materiality threshold {thresholds[3]:.8g}; damp high-degree hubs "
                f"above degree percentile {thresholds[0]:.8g} by up to 50%; boost positive RA above "
                f"RA percentile {thresholds[1]:.8g} by up to 25%; damp clustering above "
                f"{thresholds[2]:.8g} by up to 50%. Use smooth logistic gates with respective "
                f"hub/RA/clustering/materiality temperatures {[m.temperature for m in self.maps]}. "
                f"Sum, divide by eight, then multiply by nonnegative scale {x[6]:.8g}. "
                "Add this residual score to the anchor; direct evidence bypasses neighbour gates.")


def geometry(j):
    norms = np.linalg.norm(j, axis=0)
    gram = j.T @ j
    return {"rms_sensitivity": np.sqrt(np.mean(j*j, axis=0)).tolist(),
            "mean_signed_sensitivity": np.mean(j, axis=0).tolist(),
            "max_abs_sensitivity": np.max(np.abs(j), axis=0).tolist(),
            "cosine": (gram / np.maximum(np.outer(norms, norms), 1e-300)).tolist(),
            "GN_eigenvalues": np.linalg.eigvalsh(2*gram/len(j)).tolist()}


def diagnose(original):
    mask = original.present & original.neighbour
    inputs = [original.degree, original.ra_pct, original.clust, original.mat]
    maps = [Quantile(values[mask]) for values in inputs]
    candidate = Evidence(original, maps)
    original_geom = geometry(original.score(pg.DEFAULT)[1])
    remapped = geometry(candidate.score(np.full(7, .5))[1])
    sensitivities = np.array(remapped["rms_sensitivity"])
    cosine = np.array(remapped["cosine"])
    # Scale is economically essential. Gates with <1% scale sensitivity are
    # weak/inactive locally; pairwise cosine >= .8 is a redundancy screen.
    active = [6]
    decisions = []
    for i in sorted(range(6), key=lambda k: (-sensitivities[k], k)):
        redundant = next((k for k in active if abs(cosine[i, k]) >= .8), None)
        weak = sensitivities[i] < .01*sensitivities[6]
        keep = not weak and redundant is None
        decisions.append({"name": pg.NAMES[i], "index": i, "keep": keep,
                          "rms": float(sensitivities[i]), "relative_to_w": float(sensitivities[i]/sensitivities[6]),
                          "reason": "weak (<1% of w RMS)" if weak else
                          (f"collinear with {pg.NAMES[redundant]} (absolute cosine >=0.8)" if redundant is not None else "active, independent under screen"),
                          "partner_cosine": float(cosine[i, redundant]) if redundant is not None else None})
        if keep:
            active.append(i)
    active.sort()
    gates, _ = candidate.gates(np.full(7, .5))
    activity = {}
    for name, g, mapping in zip(GATE_NAMES, gates, maps):
        relevant = mask & (original.ra_positive if name == "THRESH_RA" else True)
        contributing = relevant & (original.base != 0)
        trans = (g >= .1) & (g <= .9)
        activity[name] = {"all_neighbour_share": float(trans[mask].mean()),
                          "contributing_share": float(trans[contributing].mean()),
                          "threshold_at_half": mapping.at(.5)[0], "slope_at_half": mapping.at(.5)[1],
                          "input_min": float(mapping.knots[0]), "input_max": float(mapping.knots[-1]),
                          "temperature": mapping.temperature}
    reduced = Evidence(original, maps, active)
    sweep = {str(q): geometry(candidate.score(np.full(7, q))[1]) for q in (.2, .5, .8)}
    return {"old": old.evidence_diagnostics(original), "original_geometry": original_geom,
            "remapped_geometry": remapped, "remapped_gate_activity": activity,
            "sensitivity_sweep": sweep, "decisions": decisions, "active": active,
            "names": reduced.names, "maps": [m.settings() for m in maps],
            "reduced_geometry": geometry(reduced.score(np.full(len(active), .5))[1]),
            "prompt_at_midpoint": reduced.prompt(np.full(len(active), .5))}, reduced


def evidence_only():
    cases, features = old.load_evidence()
    original = old.Evidence([features[c["case_id"]] for c in cases])
    diagnosis, bank = diagnose(original)
    return cases, diagnosis, bank


class Moments(old.Moments):
    def __init__(self, bank, indices, y):
        super().__init__(bank, indices, y)
        midpoint = bank.score(np.full(len(bank.active), .5))[0]
        self.scale = float(np.mean(midpoint*midpoint))
        if self.scale <= 0:
            raise ValueError("Degenerate evidence score")

    def mse(self, x):
        s, j = self.bank.score(x)
        return (float(self.y2 + self.count @ (s*s) - 2*self.sum_y @ s),
                2*j.T @ (self.count*s-self.sum_y))

    def scaled(self, x):
        loss, grad = self.mse(x)
        return loss/self.scale, grad/self.scale


def projected_norm(x, gradient):
    return float(np.max(np.abs(x-np.clip(x-gradient, 0, 1))))


def fit(moment, initial, method):
    """Same box, objective and stopping diagnostics; GD keeps original step .5."""
    x = np.array(initial, float)
    started = time.perf_counter()
    if method == "GD":
        loss, grad = moment.scaled(x)
        stable = 0
        stop = "iteration_cap"
        iteration = 0
        for iteration in range(MAXITER+1):
            if projected_norm(x, grad) <= GTOL:
                stop = "projected_gradient"
                break
            if iteration == MAXITER:
                break
            new = np.clip(x-.5*moment.scale*grad, 0, 1)
            next_loss, next_grad = moment.scaled(new)
            relative = abs(next_loss-loss)/max(1., abs(loss), abs(next_loss))
            stable = stable+1 if relative <= FTOL else 0
            x, loss, grad = new, next_loss, next_grad
            if stable >= 5:
                iteration += 1
                stop = "relative_loss"
                break
        message = "constant raw-MSE step 0.5"
    elif method == "L-BFGS-B":
        from scipy.optimize import minimize
        result = minimize(moment.scaled, x, jac=True, method="L-BFGS-B",
                          bounds=[(0., 1.)]*len(x),
                          options={"maxiter": MAXITER, "maxfun": 100000, "gtol": GTOL,
                                   "ftol": FTOL, "maxls": 50, "maxcor": 10})
        x = result.x
        iteration = int(result.nit)
        loss, grad = moment.scaled(x)
        message = str(result.message)
        stop = ("projected_gradient" if projected_norm(x, grad) <= GTOL else
                "relative_loss" if result.success else "iteration_cap" if iteration >= MAXITER else "solver_failure")
    else:
        raise ValueError("Unregistered optimizer")
    raw_loss, raw_grad = moment.mse(x)
    return {"theta": x.tolist(), "initial": list(map(float, initial)), "iterations": iteration,
            "loss": raw_loss, "scaled_loss": loss, "stop": stop, "message": message,
            "projected_gradient_norm": projected_norm(x, grad),
            "raw_projected_gradient_norm": projected_norm(x, raw_grad),
            "seconds": time.perf_counter()-started,
            "corner_any": bool(np.any((x <= 1e-8) | (x >= 1-1e-8))),
            "corner_w_zero": bool(x[moment.bank.active.index(6)] <= 1e-8)}


def source_hashes():
    paths = [Path(__file__), Path(old.__file__),
             old.REPO / "examples/test_q1_identify_v2.py"]
    paths += [Path(__file__).with_name(n) for n in ("prompt_gd.py", "hop_ablation.py", "q1_run_v2.py")]
    return {str(p): old.sha(p) for p in paths}


def settings(diagnosis, bank):
    import scipy
    return {"source_hashes": source_hashes(), "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "cases_sha256": old.sha(old.DATA/"cases.json"), "cache_sha256": old.sha(old.CACHE),
            "seal": pg.SEAL, "diagnosis": diagnosis, "active": bank.active, "names": bank.names,
            "theta_star": [.5]*len(bank.active), "starts": old.STARTS[:, bank.active].tolist(),
            "design": {"mapping": "degree-20 Bernstein approximation of empirical quantiles at 0,.05,...,1; exact endpoints",
                       "inputs": "all used neighbour rows, including zero contributions; bank of 80 train + 40 valid covariates only",
                       "temperature": "0.5 * population standard deviation of each gate input; floor 1e-12",
                       "screen": "at all-0.5 candidate: retain w; remaining descending RMS; drop RMS<.01*w_RMS or |cosine|>=.8 with a retained column",
                       "fixed_coordinates": "removed gates fixed at empirical-smoothed median (q=.5); economic factors retained",
                       "selection_disclosure": "screen chosen after inspecting prior report, before synthetic fits; evidence-only local heuristic, not global identifiability proof"},
            "optimizers": ["GD", "L-BFGS-B"], "optimizer_settings": {
                "iterations": MAXITER, "GD_raw_MSE_step": .5,
                "scaled_objective": "raw MSE / mean(full-bank midpoint prediction squared), no penalty or intercept",
                "projected_gradient": "infinity norm x-clip(x-gradient_of_scaled_objective,0,1)",
                "gtol": GTOL, "relative_loss_tolerance": FTOL, "GD_relative_loss_patience": 5,
                "LBFGSB_relative_loss": "SciPy native consecutive-iterate rule; report stop and projected norm separately",
                "LBFGSB_maxfun": 100000, "LBFGSB_maxls": 50, "LBFGSB_maxcor": 10},
            "synthetic": {"r": list(old.R_VALUES), "n": list(old.N_VALUES), "replications": 10,
                "seed": "SeedSequence([20261026, 420, r_index, n_index, replication])",
                "draw_order": "train indices; train noise; OOS indices; OOS noise",
                "sampling": "uniform 120-case bank; n=80 without replacement, others with replacement; OOS same n always with replacement",
                "positive_r": "y=f(theta*)+Gaussian(0,sigma); sigma=std(f(theta*)) * sqrt(r^-2-1); population empirical-bank correlation",
                "r_zero": "y=pure Gaussian noise with sigma=std(f(theta*)); theta MAE is descriptive, no identified shape",
                "paired": "both optimizers use identical samples, noise, and five starts",
                "best": "minimum raw training MSE; first start breaks ties; no OOS selection",
                "GN": "raw-MSE 2*J.T*counts*J at training-best; condition infinity if lambda_min<=max(1e-18,lambda_max*1e-14)",
                "corner": "any coordinate within 1e-8 of either box edge; separately report w=0",
                "iterations": "all 50 runs per cell; mean, median, maximum, cap fraction and stop counts",
                "gain": "independent synthetic OOS zero-score MSE minus fitted MSE"},
            "criteria": {"agreement": .05, "agreement_share": .8, "every_parameter_MAE": .1,
                         "required": "every r>=.2 and n>=500 cell must pass; per optimizer",
                         "real_selection": "L-BFGS-B if passing, otherwise GD if passing, otherwise no real labels or fit"},
            "real": {"train": 80, "valid": 40, "labels": "only guarded prompt_gd.load_labels(train/valid), residual=outcome-anchor",
                     "freeze": "write immutable all-start validation predictions and train-best index before any valid label access",
                     "prior_knowledge": "Prior Q1 report already exposed aggregate real train/validation results; this is a follow-up, not a new untouched holdout"},
            "no_network_or_LLM": True, "evaluation_labels": "forbidden", "memory_limit_bytes": 2*1024**3}


def immutable_write(path, value):
    path = Path(path)
    if path.exists():
        if pg.read(path) != value:
            raise ValueError(f"Immutable artifact differs: {path}")
    else:
        pg.write(path, value)


def register(out=OUT):
    if pg.SEAL != "2025-11-03":
        raise ValueError("Q1_SEAL_DATE must be 2025-11-03")
    cases, diagnosis, bank = evidence_only()
    config = settings(diagnosis, bank)
    path = Path(out)/"PROTOCOL_IDENTIFY_V2.json"
    if path.exists():
        registration = pg.read(path)
        if registration["settings"] != config:
            raise ValueError("Immutable preregistration changed")
    else:
        registration = {"registered_at_utc": utc(), "new_label_files_opened": False,
                        "synthetic_fits_before_registration": 0, "settings": config}
        pg.write(path, registration)
    immutable_write(Path(out)/"evidence_diagnosis.json", diagnosis)
    return registration, cases, bank


def check_registration(out):
    reg = pg.read(Path(out)/"PROTOCOL_IDENTIFY_V2.json")
    if reg["settings"]["source_hashes"] != source_hashes():
        raise ValueError("Source changed after preregistration")
    if pg.SEAL != reg["settings"]["seal"]:
        raise ValueError("Seal mismatch")
    for path, key in ((old.DATA/"cases.json", "cases_sha256"), (old.CACHE, "cache_sha256")):
        if old.sha(path) != reg["settings"][key]:
            raise ValueError("Evidence changed after preregistration")
    return reg


def condition(eigen):
    return float(eigen[-1]/eigen[0]) if eigen[0] > max(1e-18, eigen[-1]*1e-14) else None


def passes_cell(cell):
    return bool(cell["agreement_share"] >= .8 and np.all(np.asarray(cell["MAE"]) <= .1))


def eligibility(cells):
    result = {}
    for method in ("GD", "L-BFGS-B"):
        required = [c for c in cells if c["method"] == method and c["r"] >= .2 and c["n"] >= 500]
        pairs = {(c["r"], c["n"]) for c in required}
        result[method] = len(required) == 4 and pairs == {(.2, 500), (.2, 2000), (.4, 500), (.4, 2000)} and all(passes_cell(c) for c in required)
    return result


def synthetic_experiment(bank, registration, out=OUT):
    check_registration(out)
    config = registration["settings"]
    theta = np.array(config["theta_star"])
    planted = bank.score(theta)[0]
    sd = float(np.std(planted))
    cells = []
    for ri, r in enumerate(old.R_VALUES):
        sigma = float(sd*np.sqrt(r**-2-1)) if r else sd
        for ni, n in enumerate(old.N_VALUES):
            by_method = {m: [] for m in config["optimizers"]}
            for rep in range(10):
                seed = [SEED, 420, ri, ni, rep]
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                idx = rng.choice(len(planted), n, replace=n > len(planted))
                y = (planted[idx] if r else 0.) + rng.normal(0, sigma, n)
                vi = rng.choice(len(planted), n, replace=True)
                vy = (planted[vi] if r else 0.) + rng.normal(0, sigma, n)
                moment = Moments(bank, idx, y)
                for method in config["optimizers"]:
                    runs = [fit(moment, start, method) for start in config["starts"]]
                    best = min(range(5), key=lambda k: runs[k]["loss"])
                    estimate = np.array(runs[best]["theta"])
                    eig = np.linalg.eigvalsh(moment.gn(estimate))
                    ranges = np.ptp([run["theta"] for run in runs], axis=0)
                    baseline = float(np.mean(vy*vy))
                    for run in runs:
                        pred = bank.score(run["theta"])[0][vi]
                        run["oos_mse"] = float(np.mean((vy-pred)**2))
                        run["oos_gain"] = baseline-run["oos_mse"]
                    by_method[method].append({"replication": rep, "seed": seed, "runs": runs,
                        "best_start": best, "theta_range": ranges.tolist(), "agree": bool(np.max(ranges) <= .05),
                        "absolute_error": np.abs(estimate-theta).tolist(), "GN_eigenvalues": eig.tolist(),
                        "GN_condition": condition(eig), "oos_gain": runs[best]["oos_gain"],
                        "oos_baseline_mse": baseline, "train_correlation": old.corr(planted[idx], y),
                        "oos_correlation": old.corr(planted[vi], vy), "train_unique_cases": int(len(np.unique(idx)))})
            for method, reps in by_method.items():
                runs = [run for rep in reps for run in rep["runs"]]
                conds = [rep["GN_condition"] for rep in reps]
                finite = [c for c in conds if c is not None]
                iters = [run["iterations"] for run in runs]
                cell = {"method": method, "r": r, "n": n, "sigma": sigma, "replications": reps,
                    "agreement_share": float(np.mean([rep["agree"] for rep in reps])),
                    "MAE": np.mean([rep["absolute_error"] for rep in reps], axis=0).tolist(),
                    "GN_condition_median": float(np.median(conds)) if len(finite) == 10 else None,
                    "GN_condition_finite_median": float(np.median(finite)) if finite else None,
                    "GN_singular_share": 1-len(finite)/10,
                    "iterations_mean": float(np.mean(iters)), "iterations_median": float(np.median(iters)),
                    "iterations_max": int(max(iters)),
                    "cap_share": float(np.mean([run["stop"] == "iteration_cap" for run in runs])),
                    "stop_counts": {s: sum(run["stop"] == s for run in runs) for s in sorted({run["stop"] for run in runs})},
                    "corner_share": float(np.mean([run["corner_any"] for run in runs])),
                    "w_zero_share": float(np.mean([run["corner_w_zero"] for run in runs])),
                    "oos_gain_mean": float(np.mean([rep["oos_gain"] for rep in reps])),
                    "oos_gain_sd": float(np.std([rep["oos_gain"] for rep in reps], ddof=1)),
                    "projected_gradient_median": float(np.median([run["projected_gradient_norm"] for run in runs]))}
                cell["pass"] = passes_cell(cell) if r else False
                cells.append(cell)
                print(json.dumps({k: cell[k] for k in ("method", "r", "n", "agreement_share", "MAE", "iterations_mean", "cap_share")}), flush=True)
            pg.write(Path(out)/"synthetic.partial.json", {"cells": cells})
    eligible = eligibility(cells)
    result = {"completed_at_utc": utc(), "protocol_sha256": old.sha(Path(out)/"PROTOCOL_IDENTIFY_V2.json"),
              "cells": cells, "eligible": eligible,
              "minimum_n": {method: {str(r): next((c["n"] for c in cells if c["method"] == method and c["r"] == r and c["pass"]), None)
                                       for r in (.1, .2, .4)} for method in config["optimizers"]},
              "planted_score_sd": sd}
    pg.write(Path(out)/"synthetic_results.json", result)
    return result


def guarded_labels(cases, split, method, out=OUT):
    if split not in ("train", "valid"):
        raise ValueError("Evaluation labels forbidden")
    check_registration(out)
    synthetic = pg.read(Path(out)/"synthetic_results.json")
    if synthetic["protocol_sha256"] != old.sha(Path(out)/"PROTOCOL_IDENTIFY_V2.json") or not eligibility(synthetic["cells"]).get(method, False):
        raise ValueError("Recovery criterion failed; real labels forbidden")
    freeze_path = Path(out)/"validation_prediction_freeze.json"
    if split == "valid" and not freeze_path.exists():
        raise ValueError("Freeze validation predictions first")
    audit_path = Path(out)/"label_access_audit.json"
    audit = pg.read(audit_path) if audit_path.exists() else {"accesses": []}
    audit["accesses"].append({"split": split, "at_utc": utc(), "method": method,
                              "freeze_sha256": old.sha(freeze_path) if split == "valid" else None})
    pg.write(audit_path, audit)
    return pg.load_labels(old.DATA, cases, split)


def real_experiment(bank, cases, method, registration, out=OUT):
    train = np.array([i for i, c in enumerate(cases) if c["split"] == "train"])
    valid = np.array([i for i, c in enumerate(cases) if c["split"] == "valid"])
    labels = guarded_labels(cases, "train", method, out)
    y = np.array([labels[cases[i]["case_id"]]-cases[i]["anchor"] for i in train])
    moment = Moments(bank, train, y)
    runs = [fit(moment, start, method) for start in registration["settings"]["starts"]]
    best = min(range(5), key=lambda i: runs[i]["loss"])
    predictions = [bank.score(run["theta"])[0][valid].tolist() for run in runs]
    frozen = {"at_utc": utc(), "method": method, "runs": runs, "best_train_start": best,
              "case_ids": [cases[i]["case_id"] for i in valid], "residual_predictions": predictions,
              "outcome_predictions": (np.array(predictions)+np.array([cases[i]["anchor"] for i in valid])).tolist(),
              "prompt": bank.prompt(runs[best]["theta"]),
              "protocol_sha256": old.sha(Path(out)/"PROTOCOL_IDENTIFY_V2.json")}
    immutable_write(Path(out)/"validation_prediction_freeze.json", frozen)
    labels = guarded_labels(cases, "valid", method, out)
    vy = np.array([labels[cases[i]["case_id"]]-cases[i]["anchor"] for i in valid])
    baseline = float(np.mean(vy*vy))
    mse = float(np.mean((vy-np.array(predictions[best]))**2))
    return {"run": True, "method": method, "theta": runs[best]["theta"], "runs": runs,
            "train_mse": runs[best]["loss"], "train_baseline": moment.y2,
            "train_gain": moment.y2-runs[best]["loss"], "valid_mse": mse,
            "valid_baseline": baseline, "valid_gain": baseline-mse,
            "prompt": frozen["prompt"], **old.agreement(runs)}


def peak_memory():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def numerical_checks(bank):
    max_error = 0.
    rng = np.random.default_rng(SEED+2)
    for x in rng.uniform(.1, .9, (12, len(bank.active))):
        jac = bank.score(x)[1]
        numeric = np.column_stack([(bank.score(x+d)[0]-bank.score(x-d)[0])/2e-6
                                   for d in np.eye(len(x))*1e-6])
        max_error = max(max_error, float(np.max(np.abs(jac-numeric))))
    if max_error > 1e-8:
        raise AssertionError("Analytic Jacobian check failed")
    return {"jacobian_max_absolute_error": max_error}


def figures(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = result["synthetic"]["cells"]
    names = result["registration"]["settings"]["names"]
    fig, axes = plt.subplots(2, len(names), figsize=(4*len(names), 7), squeeze=False)
    for row, method in enumerate(("GD", "L-BFGS-B")):
        for col, name in enumerate(names):
            ax = axes[row, col]
            for r in old.R_VALUES:
                selected = [c for c in cells if c["method"] == method and c["r"] == r]
                ax.plot([c["n"] for c in selected], [c["MAE"][col] for c in selected], "o-", label=f"r={r:g}")
            ax.axhline(.1, color="black", ls="--", lw=.8)
            ax.set(xscale="log", xlabel="n", ylabel="Parameter MAE", title=f"{method}: {name}", ylim=(0, .6))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pre-registered V2 recovery; r=0 error is descriptive only")
    fig.tight_layout(); fig.savefig(Path(out)/"recovery_mae.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, method in enumerate(("GD", "L-BFGS-B")):
        for col, (key, ylabel) in enumerate((("agreement_share", "All-start agreement share"),
                                            ("iterations_mean", "Mean iterations"), ("corner_share", "Any-boundary rate"))):
            ax = axes[row, col]
            for r in old.R_VALUES:
                selected = [c for c in cells if c["method"] == method and c["r"] == r]
                ax.plot([c["n"] for c in selected], [c[key] for c in selected], "o-", label=f"r={r:g}")
            ax.set(xscale="log", xlabel="n", ylabel=ylabel, title=method)
            if col != 1:
                ax.set_ylim(-.03, 1.03)
            if col == 0:
                ax.axhline(.8, color="black", ls="--", lw=.8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out)/"optimizer_diagnostics.png", dpi=150); plt.close(fig)
    diag = result["registration"]["settings"]["diagnosis"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in zip(axes, ("original_geometry", "remapped_geometry"), ("Original DEFAULT", "Quantile candidate midpoint")):
        im = ax.imshow(diag[key]["cosine"], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(7), pg.NAMES, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(7), pg.NAMES, fontsize=8)
        ax.set_title(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="Jacobian-column cosine")
    fig.subplots_adjust(bottom=.28, wspace=.45, right=.82)
    fig.savefig(Path(out)/"jacobian_cosines.png", dpi=150); plt.close(fig)


def report(result, out):
    config = result["registration"]["settings"]
    diagnosis, synthetic = config["diagnosis"], result["synthetic"]
    old_diag = diagnosis["old"]
    fmt = lambda x: "infinite/singular" if x is None else f"{x:.6g}"
    names = config["names"]
    old_eig = diagnosis["original_geometry"]["GN_eigenvalues"]
    new_eig = diagnosis["reduced_geometry"]["GN_eigenvalues"]
    lines = ["# Q1 identification V2: registered evidence correction and recovery gate", "",
        "## Verdict", "",
        f"**Convergent with signal under the requested joint criterion: {synthetic['eligible']}.**",
        "The criterion is at least 80% five-start agreement (every coordinate range <=0.05) and every parameter MAE <=0.10 in EACH of r={0.2,0.4}, n={500,2000}. "
        "Optimization agreement and statistical parameter recovery are reported separately. A positive predictive gain does not establish parameter recovery.", "",
        "## Registration and evidence-only design", "",
        f"Registered at {result['registration']['registered_at_utc']}; protocol `{Path(out)/'PROTOCOL_IDENTIFY_V2.json'}`. "
        f"Protocol SHA-256 `{synthetic['protocol_sha256']}`. Source and dependency hashes, exact quantile coefficients, starts, seeds, screens, stopping rules and branch decision are in that immutable artifact. "
        "Registration preceded all new synthetic fits and label access. The prior report was read first and its aggregate real-data outcomes were already known. "
        "No new label file was used for parameterization or method selection. This is not a fresh, untouched validation study.", "",
        f"The cached bank has 120 development cases (80 train / 40 valid), {old_diag['used_evidence_rows']} used rows, eight rows maximum per case, and seal 2025-11-03. "
        f"Hop row counts: {old_diag['hop_counts']}; nonzero signed contribution counts: {old_diag['nonzero_contribution_hop_counts']}. "
        "Quantile maps use all used neighbour rows, including zero-contribution rows. Validation covariates enter this label-free design, so the experiment is transductive with respect to the 120-case bank.", "",
        "## Per-parameter diagnosis", "",
        "Here s denotes the FINAL residual prediction including w; ds/dtheta uses normalized coordinates. RMS sensitivities summarize all 120 case derivatives, "
        "not just positive contributions. Transition means sigmoid in [0.1,0.9], among all used neighbour rows. Hop weights and w have no gate (N/A). "
        "Mean signed and maximum absolute derivatives and the q=0.2/0.5/0.8 sensitivity sweep are in evidence_diagnosis.json.", "",
        "| Parameter | Original transition share | Original RMS ds/dtheta | Remapped transition share | Remapped RMS ds/dtheta | Decision |",
        "|---|---:|---:|---:|---:|---|"]
    decisions = {d["name"]: d for d in diagnosis["decisions"]}
    for i, name in enumerate(pg.NAMES):
        before = fmt(old_diag["gates_at_DEFAULT"][name]["transition_share_0.1_to_0.9"]) if name in GATE_NAMES else "N/A"
        after = fmt(diagnosis["remapped_gate_activity"][name]["all_neighbour_share"]) if name in GATE_NAMES else "N/A"
        action = "retain essential nonnegative scale" if i == 6 else ("retain: " if decisions[name]["keep"] else "fix q=0.5: ")+decisions[name]["reason"]
        lines.append(f"| {name} | {before} | {fmt(diagnosis['original_geometry']['rms_sensitivity'][i])} | {after} | {fmt(diagnosis['remapped_geometry']['rms_sensitivity'][i])} | {action} |")
    lines += ["", "Thresholds use the degree-20 Bernstein polynomial with coefficients equal to empirical quantiles at 0,0.05,...,1. "
        "It is a smooth, monotone APPROXIMATION to the empirical inverse CDF, with exact observed minimum and maximum at q=0 and q=1. "
        "Smoothing avoids plateaus/discontinuous derivatives from tied graph inputs; q is a smoothed quantile coordinate, not an exact percentile between knots. "
        "Each temperature is half the input population standard deviation (floor 1e-12). Physical thresholds remain explicit in the prompt. "
        "No basis rotation, regularization, intercept, negative scale, or truth-dependent bounds are introduced.", "",
        "| Gate | Input min | Input max | Threshold at q=.5 | Temperature | Transition among contributing rows |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, row in diagnosis["remapped_gate_activity"].items():
        lines.append(f"| {name} | "+" | ".join(fmt(row[k]) for k in ("input_min", "input_max", "threshold_at_half", "temperature", "contributing_share"))+" |")
    lines += ["", "The evidence-only screen retains w, then visits remaining columns in descending midpoint RMS sensitivity. "
        "It fixes a parameter if RMS is below 1% of w RMS, or absolute Jacobian cosine is at least 0.8 with an already retained parameter. "
        "It retains hop-2, hop-3, clustering and scale. Hub/clustering cosine is 0.973424, RA/clustering is -0.898303. "
        "Materiality RMS is 0.0000686006 (0.096784% of w); many rows near its transition carry little or zero signed materiality. "
        "Fixing these gates preserves hub damping, RA enhancement and materiality screening at their label-free midpoint values, but relinquishes independent learning of those thresholds. "
        "The local weak-materiality decision does not claim the derivative vanishes throughout the box. The retained clustering gate repairs its original saturation; "
        "its original RA cosine was -0.843637. These screens were chosen before any V2 synthetic results; no post-result pruning was done.", "",
        f"Original DEFAULT GN eigenvalues: {old_eig}. Reduced midpoint eigenvalues: {new_eig}. "
        f"Condition number improves from {fmt(old_eig[-1]/old_eig[0])} to {fmt(new_eig[-1]/new_eig[0])}; "
        "eigenvalues below 1e-6 fall from two to zero. This improves local geometry but is not a guarantee of statistical power or global uniqueness.", ""]
    for key, title in (("original_geometry", "Original DEFAULT pairwise Jacobian cosines"), ("remapped_geometry", "Remapped seven-parameter candidate midpoint cosines")):
        lines += [f"### {title}", "", "| Parameter | "+" | ".join(pg.NAMES)+" |", "|---|"+"---:|"*7]
        for name, row in zip(pg.NAMES, diagnosis[key]["cosine"]):
            lines.append(f"| {name} | "+" | ".join(f"{v:.4f}" for v in row)+" |")
        lines.append("")
    lines += ["### Economically interpretable prompt at the planted midpoint", "", diagnosis["prompt_at_midpoint"], "",
        "## Optimizers and recovery protocol", "",
        "Both methods use the same five original seeded starts restricted to retained coordinates, exact MSE sufficient statistics, and box [0,1]. "
        f"Starts: `{config['starts']}`. Theta* is interior `{config['theta_star']}`. "
        f"Projected GD keeps raw-MSE fixed step 0.5 and now permits {MAXITER} iterations. L-BFGS-B uses SciPy {config['scipy_version']}, "
        f"box bounds, {MAXITER} iterations, maxfun=100000, maxls=50, maxcor=10. "
        "For numerical stopping and L-BFGS-B, MSE is divided by the label-free midpoint mean squared prediction; the GD update compensates for that factor and exactly retains raw step 0.5. "
        f"Projected gradient infinity norm is ||x-clip(x-grad,0,1)|| <= {GTOL:g} on that scaled objective. Relative loss tolerance is {FTOL:g}, "
        "using max(1,|old loss|,|new loss|) in the denominator: five consecutive small changes for GD, the native consecutive-iterate rule for L-BFGS-B. "
        "Relative-loss stops are not represented as projected-gradient certificates; stop counts and final norms are reported.", "",
        "For positive r, y=s(theta*)+epsilon with independent Gaussian noise of SD std(s(theta*)) sqrt(r^-2-1), calibrated over the uniform 120-case empirical bank. "
        f"Planted score SD is {fmt(synthetic['planted_score_sd'])}. r is population correlation, not forced finite-sample correlation. "
        "At r=0, y is pure noise with SD equal to the planted score SD; MAE against theta* is descriptive only. "
        "Ten replications per cell use SeedSequence([20261026,420,r_index,n_index,replication]). "
        "n=80 samples without replacement; larger n samples with replacement. Fresh OOS evidence is sampled with replacement, same n, with independent noise. "
        "Both optimizers receive identical evidence/noise. The minimum-training-loss start determines MAE, GN condition and OOS gain. "
        "Agreement and corner/iteration summaries use all five starts. No OOS result is used for selection. All 1,600 final fits and seeds are retained.", "",
        "## Recovery results", "",
        "GN condition is the median over training-best fits; 'infinite/singular' flags any singular replication conservatively. "
        "Finite-only medians and the singular fraction are retained in JSON. A singular eigenvalue means min eigenvalue <=max(1e-18,1e-14*max eigenvalue). "
        "Corner means any parameter within 1e-8 of either boundary; w=0 is separately reported. OOS gain is zero-score MSE minus fitted MSE.", "",
        "| Optimizer | r | n | Agree share | Max MAE | Median GN cond | Mean iterations | Cap share | Any corner | w=0 | OOS gain mean (SD) | Cell pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in synthetic["cells"]:
        lines.append(f"| {c['method']} | {c['r']:g} | {c['n']} | {c['agreement_share']:.2f} | {max(c['MAE']):.4f} | {fmt(c['GN_condition_median'])} | {c['iterations_mean']:.1f} | {c['cap_share']:.2f} | {c['corner_share']:.2f} | {c['w_zero_share']:.2f} | {fmt(c['oos_gain_mean'])} ({fmt(c['oos_gain_sd'])}) | {c['pass']} |")
    lines += ["", "### MAE for every retained parameter", "", "| Optimizer | r | n | "+" | ".join(names)+" |", "|---|---:|---:|"+"---:|"*len(names)]
    for c in synthetic["cells"]:
        lines.append(f"| {c['method']} | {c['r']:g} | {c['n']} | "+" | ".join(f"{a:.5f}" for a in c["MAE"])+" |")
    lines += ["", "### Stopping evidence", "", "| Optimizer | r | n | Iterations median / max | Median projected norm | Stop counts |", "|---|---:|---:|---:|---:|---|"]
    for c in synthetic["cells"]:
        lines.append(f"| {c['method']} | {c['r']:g} | {c['n']} | {c['iterations_median']:g} / {c['iterations_max']} | {fmt(c['projected_gradient_median'])} | {c['stop_counts']} |")
    lines += ["", "### Minimum tested recovery cells", "", "The entries below require both agreement and every-parameter MAE thresholds. None means not attained through n=2000; no extrapolated minimum is claimed.", ""]
    for method, minimum in synthetic["minimum_n"].items():
        lines.append(f"- {method}: "+"; ".join(f"r={r}: {n if n is not None else 'none'}" for r, n in minimum.items())+".")
    lines += ["", "### Figures", "", "![Parameter recovery](Q1_IDENTIFY_V2_recovery_mae.png)", "", "![Optimizer agreement, iterations and corners](Q1_IDENTIFY_V2_optimizer_diagnostics.png)", "", "![Jacobian cosines](Q1_IDENTIFY_V2_jacobian_cosines.png)", "", "## Real development data", ""]
    real = result["real"]
    if not real["run"]:
        lines += ["**Not run. Neither optimizer passed the pre-declared joint recovery criterion in every required signal/sample-size cell.** "
                  "No new training or validation label file was opened; no real-data optimization or validation prediction file was produced. "
                  "The prior report's real-data results were not reused as V2 results."]
    else:
        lines += [f"Selected {real['method']} by the preregistered priority. Best train theta: {real['theta']}; five-start agreement: {real['all_starts_agree']}. "
                  "All validation predictions were frozen and hashed before opening validation labels.", "",
                  "| Split | Zero-score MSE | Fitted MSE | Gain |", "|---|---:|---:|---:|",
                  f"| Train (80) | {fmt(real['train_baseline'])} | {fmt(real['train_mse'])} | {fmt(real['train_gain'])} |",
                  f"| Validation (40) | {fmt(real['valid_baseline'])} | {fmt(real['valid_mse'])} | {fmt(real['valid_gain'])} |", "", real["prompt"]]
    lines += ["", "## Interpretation and limits", "",
        "The reduction restores useful local curvature and allows a meaningful test of the optimizer. Recovery can still fail because the independent shape effects are small relative to total-score noise, "
        "or because fixed-step GD remains too slow. Five-start agreement tests agreement of these five runs, not a global optimum theorem. "
        "Large simulated n repeats only 120 distinct evidence cases; it reduces independent Gaussian label noise without creating new graph support. "
        "Synthetic OOS cases may share evidence identities with training while their noise is independent. Correct model specification, Gaussian noise and an interior planted truth are favorable assumptions. "
        "Ten replications give coarse 10-percentage-point agreement shares. Fixed thresholds are engineering restrictions, not discoveries that their economic mechanisms are irrelevant. "
        "This study concerns a smooth cached-feature surrogate, not actual LLM prompt execution or causal financial identification.", "",
        "## Possible errors and checks performed", "",
        f"- Analytic Jacobian: 12 random interior points, central differences of size 1e-6; maximum absolute error {fmt(result['checks']['jacobian_max_absolute_error'])}.",
        "- Unittests cover quantile monotonicity/endpoints/derivatives, scalar score equivalence, reduced/full Jacobian agreement, finite-difference MSE gradients, exact duplicate sufficient statistics, constrained optimizer/KKT behavior, registration immutability, and label/recovery/freeze guards. Test logs are in the artifact directory.",
        "- Cached case/model hashes, host-owned graph fields, feature ranges, timestamps, split counts, and original seal guards are checked before use. Only the first eight rows enter the score. RA=0 cannot receive an RA bonus.",
        "- Quantile smoothing handles ties without a nondifferentiable raw inverse CDF. Activity among contributing rows is reported separately to avoid mistaking many zero-score rows for information. Temperatures and mappings never use labels.",
        "- Exact count/sum(y)/sum(y²) compression preserves sampled MSE, gradient and GN; fresh OOS noise is independent. No fabricated population correlation is imposed on individual samples.",
        "- The real branch recomputes eligibility from all four required cells and checks the protocol hash before allowing train labels. Validation additionally requires a frozen prediction file. Evaluation split requests are rejected before any read.",
        f"- Source/dependency SHA-256 checks passed after the run: {result['checks']['sources_unchanged']}. Protected prompt_gd.py, hop_ablation.py and q1_run_v2.py were not edited. Existing unrelated working-tree modifications were preserved.",
        f"- No network or LLM calls, process termination, package installation or evaluation-label reads. New real-label reads: {result['checks']['real_label_reads']}. Single-process serial fitting; native Windows peak working set {result['checks']['peak_memory_bytes']} bytes ({result['checks']['peak_memory_bytes']/1024**2:.2f} MiB), below 2 GiB. Runtime including figures: {result['elapsed_seconds']:.2f} seconds.",
        "- Relative-loss termination and high projected norms are distinguished from stationary convergence. Singular GN matrices are not silently assigned finite conditions. Null-cell shape error has no recovery interpretation.", "",
        "## Artifacts and changed files", "",
        f"Full results and all fits: `{Path(out)/'identify_v2.json'}`; synthetic-only gate artifact: `{Path(out)/'synthetic_results.json'}`; evidence-only diagnosis: `{Path(out)/'evidence_diagnosis.json'}`.", "",
        "New repository files: `source/code/news_network_50_2025/prompt_gd_identify_v2.py`, `examples/test_q1_identify_v2.py`, `docs/Q1_IDENTIFY_V2_REPORT.md`, "
        "`docs/Q1_IDENTIFY_V2_recovery_mae.png`, `docs/Q1_IDENTIFY_V2_optimizer_diagnostics.png`, `docs/Q1_IDENTIFY_V2_jacobian_cosines.png`. Changes are uncommitted; .git was not modified.", "",
        f"Report copy: `{REPORT_COPY}`. Figure copies are beside both reports."]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    REPORT_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORT, REPORT_COPY)
    for name in ("recovery_mae", "optimizer_diagnostics", "jacobian_cosines"):
        for parent in (REPORT.parent, REPORT_COPY.parent):
            shutil.copyfile(Path(out)/f"{name}.png", parent/f"Q1_IDENTIFY_V2_{name}.png")


def run(out=OUT):
    started = time.perf_counter()
    registration, cases, bank = register(out)
    checks = numerical_checks(bank)
    synthetic = synthetic_experiment(bank, registration, out)
    selected = next((m for m in ("L-BFGS-B", "GD") if synthetic["eligible"][m]), None)
    real = real_experiment(bank, cases, selected, registration, out) if selected else {
        "run": False, "reason": "No optimizer passes every required recovery cell"}
    result = {"registration": registration, "synthetic": synthetic, "real": real, "checks": checks}
    figures(result, out)
    check_registration(out)
    memory = peak_memory()
    if memory >= 2*1024**3:
        raise RuntimeError("Memory limit exceeded")
    audit_path = Path(out)/"label_access_audit.json"
    checks.update(sources_unchanged=True, peak_memory_bytes=memory,
                  real_label_reads=len(pg.read(audit_path)["accesses"]) if audit_path.exists() else 0,
                  evaluation_label_reads=0, network_or_LLM_calls=0)
    result["elapsed_seconds"] = time.perf_counter()-started
    pg.write(Path(out)/"identify_v2.json", result)
    report(result, out)
    print(json.dumps({"convergent_with_signal": synthetic["eligible"], "minimum_n": synthetic["minimum_n"],
                      "required_cells": [{k: c[k] for k in ("method", "r", "n", "agreement_share", "MAE", "iterations_mean")}
                                         for c in synthetic["cells"] if c["r"] >= .2 and c["n"] >= 500],
                      "real": {k: v for k, v in real.items() if k != "runs"}, "peak_MiB": memory/1024**2,
                      "report": str(REPORT), "report_copy": str(REPORT_COPY)}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "register", "run"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "diagnose":
        print(json.dumps(evidence_only()[1], indent=2))
    elif args.command == "register":
        value, _, _ = register(args.out)
        print(json.dumps({"registered_at_utc": value["registered_at_utc"], "names": value["settings"]["names"],
                          "protocol": str(args.out/"PROTOCOL_IDENTIFY_V2.json")}, indent=2))
    else:
        run(args.out)


if __name__ == "__main__":
    main()
