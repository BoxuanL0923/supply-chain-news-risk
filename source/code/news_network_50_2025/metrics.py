"""Matched pooled explanatory and genuinely frozen-regression forecast comparisons."""
import numpy as np
import pandas as pd
from news_agent_batched_runtime_2025 import read, write, locked_write
from news_exposure_2025.metrics import regression
from .config import CONTROLS, ARMS


def matrix(frame, controls=CONTROLS, score=None):
    x = np.c_[np.ones(len(frame)), frame[list(controls)].to_numpy(float)]
    return np.c_[x, frame[score].to_numpy(float)] if score else x


def fit(frame, controls=CONTROLS, score=None):
    return regression(frame.y, matrix(frame, controls, score))


def anchor(out, cases):
    controls = pd.read_parquet(out / "controls.parquet")
    labels = pd.read_parquet(out / "labels_train.parquet")
    train = labels[["case_id", "y"]].merge(controls, on="case_id", validate="one_to_one")
    assert len(train) == 200
    fitted = fit(train)
    coeff = np.asarray(fitted["coefficient"])
    result = dict(zip(controls.case_id, matrix(controls) @ coeff))
    locked_write(out / "anchor.json", {"coefficients": coeff.tolist(), "controls": list(CONTROLS),
          "training_case_ids": train.case_id.tolist(), "n": len(train),
          "definition": "Development-only pooled OLS on conventional controls. Training residuals in-sample as in KL; later forecasts frozen.",
          "rank": fitted["rank"], "anchors": result})
    return result


def matched(out, cases, records, split):
    expected = {c["case_id"] for c in cases if c["split"] == split}
    controls = pd.read_parquet(out / "controls.parquet")
    frame = pd.read_parquet(out / f"labels_{split}.parquet")[["case_id", "y"]].merge(
        controls, on="case_id", validate="one_to_one")
    if set(frame.case_id) != expected:
        raise ValueError("Missing outcomes/controls; no silent row dropping")
    for arm, rows in records.items():
        selected = [r for r in rows if r["case_id"] in expected]
        if len(selected) != len(expected) or len({r["case_id"] for r in selected}) != len(expected):
            raise ValueError("Incomplete or duplicated arm " + arm)
        scores = {r["case_id"]: r["score"] for r in selected}
        frame[arm] = frame.case_id.map(scores)
    if not np.isfinite(frame[["y", *CONTROLS, *records]].to_numpy(float)).all():
        raise ValueError("Nonfinite comparison")
    return frame


def summarize(train, test):
    report = {"n": len(test), "dates": test.date.nunique(), "firms": test.ticker.nunique(), "arms": {}}
    frozen_anchor = matrix(test) @ np.asarray(fit(train)["coefficient"])
    residual = test.y.to_numpy()-frozen_anchor
    for arm in ARMS:
        result = {}
        for name, control in (("garch", ("b",)), ("expanded", CONTROLS)):
            base, full = fit(test, control), fit(test, control, arm)
            training_base, training_full = fit(train, control), fit(train, control, arm)
            e0 = test.y.to_numpy()-matrix(test, control)@np.asarray(training_base["coefficient"])
            e1 = test.y.to_numpy()-matrix(test, control, arm)@np.asarray(training_full["coefficient"])
            result[name] = {"pooled_r2": full["r2"], "baseline_r2": base["r2"],
                "increment_pp": 100*(full["r2"]-base["r2"]),
                "adjusted_increment_pp": 100*(full["adjusted_r2"]-base["adjusted_r2"]),
                "score_coefficient": full["coefficient"][-1], "rank": full["rank"],
                "frozen_incremental_r2": float(1-np.sum(e1**2)/np.sum(e0**2)),
                "frozen_log_mse": float(np.mean(e1**2)), "frozen_baseline_log_mse": float(np.mean(e0**2))}
        result["leave_one_date_out_pp"] = {d: 100*(fit(test[test.date.ne(d)], score=arm)["r2"]-
            fit(test[test.date.ne(d)])["r2"]) for d in sorted(test.date.unique())}
        result["leave_one_firm_out_pp"] = {t: 100*(fit(test[test.ticker.ne(t)], score=arm)["r2"]-
            fit(test[test.ticker.ne(t)])["r2"]) for t in sorted(test.ticker.unique())}
        result["non_earnings_pp"] = 100*(fit(test[test.earnings_confirmed.eq(0)], score=arm)["r2"]-
            fit(test[test.earnings_confirmed.eq(0)])["r2"])
        raw_error = residual-test[arm].to_numpy()
        result["raw_residual_mse"] = float(np.mean(raw_error**2))
        result["raw_score_forecast_r2_vs_anchor"] = float(1-np.sum(raw_error**2)/np.sum(residual**2))
        result["risk_score_direction_supported"] = result["expanded"]["score_coefficient"] > 0
        report["arms"][arm] = result
    contrasts = [("network_fixed", "NET_FIXED", "OWN_FIXED"), ("own_optimization", "OWN_OPT", "OWN_FIXED"),
                 ("network_optimization", "NET_OPT", "NET_FIXED"), ("network_optimized", "NET_OPT", "OWN_OPT")]
    rng = np.random.default_rng(20251001)
    dates = sorted(test.date.unique())
    groups = [test[test.date.eq(d)] for d in dates]
    boot = {name: [] for name, _, _ in contrasts}
    for _ in range(999):
        starts = rng.integers(0, len(dates), size=3)
        sample = pd.concat([groups[i] for s in starts for i in (s, (s+1) % len(dates))])
        values = {arm: fit(sample, score=arm)["r2"] for arm in ARMS}
        for name, a, b in contrasts:
            boot[name].append(100*(values[a]-values[b]))
    report["contrasts"] = {name: {"difference_pp": 100*(fit(test, score=a)["r2"]-fit(test, score=b)["r2"]),
        "date_block_bootstrap_interval_pp": np.quantile(boot[name], [.025, .975]).tolist()}
        for name, a, b in contrasts}
    report["inference"] = "Exploratory six-date-cluster uncertainty, conditional on generated scores; not a confirmatory significance claim. Dates were used in prior design work."
    report["placebo"] = "Omitted at user request. Network comparison includes additional evidence and compute, not isolated graph topology."
    return report
