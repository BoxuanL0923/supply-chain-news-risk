"""Matched explanatory regressions, not frozen-coefficient forecast R-squared."""
import numpy as np
import pandas as pd

from .config import CONTROLS, ARMS, SEED_NUMBER


def regression(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    if not np.isfinite(np.c_[y, x]).all() or y.size != len(x):
        raise ValueError("Nonfinite/incomplete regression")
    # Scaling affects numerical conditioning only, not the regressor span.
    scale = np.std(x, axis=0)
    scale[scale < 1e-10] = 1
    beta, _, rank, _ = np.linalg.lstsq(x / scale, y, rcond=None)
    beta /= scale
    if len(y) <= rank+1:
        raise ValueError("Too few independent residual degrees of freedom")
    e, tss = y-x@beta, float(np.sum((y-y.mean())**2))
    if tss <= 0:
        raise ValueError("Constant target")
    sse = float(e@e)
    r2 = 1-sse/tss
    return {"n": len(y), "rank": int(rank), "r2": r2, "adjusted_r2": 1-(1-r2)*(len(y)-1)/(len(y)-rank),
            "mse": sse/len(y), "sse": sse, "tss": tss, "coefficient": beta.tolist(), "errors": e.tolist()}


def fit(frame, score=None, controls=CONTROLS):
    x = np.c_[np.ones(len(frame)), frame[list(controls)].to_numpy(float)]
    if score is not None:
        x = np.c_[x, frame[score].to_numpy(float)]
    return regression(frame.y, x)


def metric(frame, score="score", controls=CONTROLS):
    base, full = fit(frame, controls=controls), fit(frame, score, controls)
    delta = 100*(full["r2"]-base["r2"])
    identity = 100*len(frame)*(base["mse"]-full["mse"])/base["tss"]
    if not np.isclose(delta, identity, atol=1e-8):
        raise AssertionError("Profiled MSE/R2 identity failed")
    return {"baseline": base, "full": full, "increment_pp": delta,
            "adjusted_increment_pp": 100*(full["adjusted_r2"]-base["adjusted_r2"]), "profiled_mse": full["mse"]}


def matched_frame(records, cases, controls, labels):
    expected = [c["case_id"] for c in cases]
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("No silent missing-row dropping or duplicate predictions")
    if not all(r.get("valid") for r in records):
        raise ValueError("Panel has failed forecasts")
    c = controls.set_index("case_id").loc[expected]
    lab = labels.set_index("case_id").loc[expected, ["y", "anchor", "available_at"]]
    scores = pd.DataFrame({"case_id": ids, "score": [r["prediction"]["score"] for r in records]}).set_index("case_id")
    return c.join(lab).join(scores).reset_index()


def sensitivity(frame, score="score"):
    return {field: {str(value): metric(frame[frame[field] != value], score)["increment_pp"]
                    for value in sorted(frame[field].unique())}
            for field in ("date", "ticker")}


def paired_uncertainty(frame, a, b, repetitions=1999):
    """Date-block resampling; scores and selected policies are held fixed."""
    dates = sorted(frame.date.unique())
    if len(dates) < 6:
        return {"status": "insufficient_time_clusters", "dates": len(dates)}
    rng = np.random.default_rng(SEED_NUMBER)
    groups = [frame[frame.date.eq(d)] for d in dates]
    values = []
    observed = 100*(fit(frame, a)["r2"]-fit(frame, b)["r2"])
    for _ in range(repetitions):
        starts = rng.integers(0, len(dates), size=(len(dates)+1)//2)
        indices = [i for start in starts for i in (start, (start+1) % len(dates))][:len(dates)]
        sampled = pd.concat([groups[i] for i in indices], ignore_index=True)
        values.append(100*(fit(sampled, a)["r2"]-fit(sampled, b)["r2"]))
    values = np.asarray(values)
    p = float((1+np.sum(np.abs(values-observed) >= abs(observed))) / (repetitions+1))
    return {"observed_pp": observed, "ci95_pp": np.quantile(values, [.025, .975]).tolist(),
            "approximate_two_sided_p": p, "replicates": repetitions, "block_dates": 2,
            "caution": "Exploratory small-date-cluster uncertainty conditional on generated scores, not a rerun of prompt selection."}


def summarize(frame, complete=False):
    value = {"n": len(frame), "origins": int(frame.date.nunique()), "complete": complete, "arms": {}}
    for arm in ARMS:
        value["arms"][arm] = {"garch": metric(frame, arm, ("b",)), "expanded": metric(frame, arm),
                              "sensitivity": sensitivity(frame, arm)}
    value["contrasts"] = {"optimization": paired_uncertainty(frame, "OWN_OPT", "OWN_FIXED") if complete else {"status": "endpoint_only"},
                           "network": paired_uncertainty(frame, "NET_FIXED", "OWN_FIXED") if complete else {"status": "endpoint_only"}}
    if complete and "event_keys" in frame:
        counts = {}
        for group in frame.event_keys:
            for key in group:
                counts[key] = counts.get(key, 0)+1
        repeated = sorted((k for k, n in counts.items() if n >= 2), key=lambda k: (-counts[k], k))[:12]
        value["repeated_source_event_sensitivity"] = {}
        for key in repeated:
            rest = frame[~frame.event_keys.map(lambda items: key in items)]
            if len(rest) > 20:
                value["repeated_source_event_sensitivity"][key] = {"removed_rows": counts[key],
                    **{a: metric(rest, a)["increment_pp"] for a in ARMS}}
        value["event_sensitivity_note"] = "Repeated cited Reuters story/event keys, not a gold causal-episode partition. No observations removed from primary estimates."
    if complete and all("approximate_two_sided_p" in r for r in value["contrasts"].values()):
        order = sorted(value["contrasts"], key=lambda k: value["contrasts"][k]["approximate_two_sided_p"])
        previous = 0.
        for i, key in enumerate(order):
            p = min(1., max(previous, (len(order)-i)*value["contrasts"][key]["approximate_two_sided_p"]))
            value["contrasts"][key]["holm_two_comparison_p"] = p
            previous = p
    return value
