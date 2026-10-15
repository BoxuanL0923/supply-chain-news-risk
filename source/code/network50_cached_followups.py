"""Post-run CPU diagnostics, not a new holdout or permission to change forecasts."""
import math
import numpy as np
import pandas as pd
from news_agent_batched_runtime_2025 import read, write
from news_network_50_2025.config import OUT


def risk_arithmetic(case, anchor, score):
    """Compute units and comparison directions; no model or realized outcome access."""
    if not all(math.isfinite(v) for v in (anchor, score, case["b"])):
        raise ValueError("Finite log-risk values required")
    sessions = case["next_sessions"]
    if len(sessions) != 5 or sessions != sorted(set(sessions)):
        raise ValueError("Five ordered unique exchange sessions required")
    def direction(delta):
        return "higher" if delta > 0 else "lower" if delta < 0 else "equal"
    return {"next_sessions":sessions, "forecast_end":sessions[-1],
            "anchor_daily_geometric_rms_percent":100*math.exp(anchor/2),
            "forecast_daily_geometric_rms_percent":100*math.exp((anchor+score)/2),
            "variance_multiple_vs_anchor":math.exp(score), "direction_vs_anchor":direction(score),
            "variance_multiple_vs_garch":math.exp(anchor+score-case["b"]),
            "direction_vs_garch":direction(anchor+score-case["b"]),
            "lagged_variance_vs_garch":{str(n):{
                "ratio":math.exp(case["state"][f"log_ret_rv{n}"]-case["b"]),
                "direction":direction(case["state"][f"log_ret_rv{n}"]-case["b"])} for n in (1,5,22)},
            "units_note":"Geometric-risk translation of log mean squared returns; not an arithmetic conditional variance forecast or a learned calibration."}


def main():
    if read(OUT / "progress.json").get("status") != "complete":
        raise ValueError("Wait for frozen primary results")
    errors = pd.read_csv(OUT / "diagnostics/case_errors.csv")
    results = {}
    for arm in ("OWN_FIXED", "NET_FIXED"):
        frame = errors[errors.arm.eq(arm)]
        train = frame[frame.split.eq("train")]
        s, y = train.score.to_numpy(), train.target_residual.to_numpy()
        beta = max(0.,float(np.sum((s-s.mean())*(y-y.mean()))/np.sum((s-s.mean())**2)))
        alternatives = {"unaltered":(0.,1.), "train_centered_unit_slope":(float(y.mean()-s.mean()),1.),
                        "train_positive_slope":(float(y.mean()-beta*s.mean()),beta)}
        results[arm] = {}
        for name,(a,b) in alternatives.items():
            split_results = {}
            for split in ("train","valid","test"):
                part = frame[frame.split.eq(split)]
                err = part.target_residual-a-b*part.score
                mse, base_mse = float(np.mean(err**2)), float(np.mean(part.target_residual**2))
                split_results[split] = {"n":len(part),"mse":mse,"anchor_mse":base_mse,
                    "forecast_r2_vs_anchor":1-mse/base_mse}
            results[arm][name] = {"intercept":a,"slope":b,"splits":split_results}
    cases = {c["case_id"]:c for c in read(OUT / "cases.json")}
    anchors = read(OUT / "anchor.json")["anchors"]
    arithmetic = {}
    for cid,arm in (("AMD:2025-10-01","OWN_FIXED"),("ALGM:2025-10-22","OWN_FIXED"),
                    ("CFLT:2025-12-08","NET_FIXED"),("CFLT:2025-12-22","OWN_FIXED")):
        score = float(errors.loc[errors.case_id.eq(cid)&errors.arm.eq(arm),"score"].iloc[0])
        arithmetic[cid+":"+arm] = risk_arithmetic(cases[cid],anchors[cid],score)
    controls = pd.read_parquet(OUT / "controls.parquet")
    result = {"scope":"Post-run diagnostic; no LLM calls, no primary forecasts changed. Not a preselected strategy or independent holdout.",
              "calibration":results,"arithmetic":arithmetic,
              "cash_deal_flag_counts":controls.pending_cash_deal.value_counts().to_dict(),
              "interpretation":"A constant shift or positive rescaling leaves refitted OLS R-squared with an intercept unchanged. Forecast-MSE improvements from calibration do not establish improved news interpretation. Do not select a calibration using these test comparisons."}
    write(OUT / "diagnostics/cached_followups.json",result)
    for arm, alternatives in results.items():
        print(arm,{name:{split:round(v["forecast_r2_vs_anchor"],4) for split,v in values["splits"].items()}
                   for name,values in alternatives.items()})
    print("Arithmetic checks",arithmetic)


if __name__ == "__main__":
    main()
