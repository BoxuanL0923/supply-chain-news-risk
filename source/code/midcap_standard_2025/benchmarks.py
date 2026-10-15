"""Causal, return-based standard five-session variance forecasts."""
from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model

from config import RISK_FLOOR

MODELS = ("hist22", "ewma094", "garch11", "gjr_garch11")
HORIZON = 5
TRAILING_SESSIONS = 504
MIN_OBS = 400


def historical_22(returns: np.ndarray) -> float:
    if len(returns) < 22 or not np.isfinite(returns[-22:]).all():
        return float("nan")
    return max(float(np.mean(returns[-22:] ** 2)), RISK_FLOOR)


def riskmetrics_ewma(returns: np.ndarray, decay: float = 0.94) -> float:
    recent = np.asarray(returns[-TRAILING_SESSIONS:], dtype=float)
    if len(recent) < MIN_OBS or not np.isfinite(recent[-1]):
        return float("nan")
    recent = recent[np.isfinite(recent)]
    if len(recent) < MIN_OBS:
        return float("nan")
    variance = float(np.mean(recent[:22] ** 2))
    for ret in recent[22:]:
        variance = decay * variance + (1.0 - decay) * float(ret ** 2)
    return max(variance, RISK_FLOOR)


def garch_forecast(returns: np.ndarray, asymmetric: bool) -> tuple[float, str]:
    recent = np.asarray(returns[-TRAILING_SESSIONS:], dtype=float)
    if len(recent) < MIN_OBS or not np.isfinite(recent[-1]):
        return float("nan"), "insufficient_or_stale"
    recent = recent[np.isfinite(recent)]
    if len(recent) < MIN_OBS:
        return float("nan"), "insufficient"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(recent * 100.0, mean="Zero", vol="GARCH",
                               p=1, o=int(asymmetric), q=1, dist="normal",
                               rescale=False)
            fitted = model.fit(disp="off", show_warning=False)
            if fitted.convergence_flag != 0:
                return float("nan"), f"nonconverged:{fitted.convergence_flag}"
            forecasts = fitted.forecast(horizon=HORIZON, method="analytic")
            variance = float(np.mean(forecasts.variance.iloc[-1].to_numpy())) / 10000.0
    except (ValueError, FloatingPointError, ArithmeticError) as exc:
        return float("nan"), f"fit_error:{type(exc).__name__}"
    if not np.isfinite(variance) or variance <= 0:
        return float("nan"), "invalid_variance"
    return max(variance, RISK_FLOOR), "ok"


def forecasts(returns: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
    x = np.asarray(returns, dtype=float)
    hist = historical_22(x)
    ewma = riskmetrics_ewma(x)
    garch, garch_state = garch_forecast(x, False)
    gjr, gjr_state = garch_forecast(x, True)
    values = dict(hist22=hist, ewma094=ewma, garch11=garch, gjr_garch11=gjr)
    status = dict(hist22="ok" if np.isfinite(hist) else "missing",
                  ewma094="ok" if np.isfinite(ewma) else "missing",
                  garch11=garch_state, gjr_garch11=gjr_state)
    return values, status
