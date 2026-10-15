"""CPU-only, causal preparation of the frozen exploratory midcap pilot.

Run with ``python -B code/midcap_tools/prepare.py``. No LLM or network calls.
All generated artifacts, including intermediate inputs, stay under config.OUT.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CFG = load_module("midcap_frozen_config", Path(__file__).with_name("config.py"))
ROOT, OUT = CFG.ROOT, CFG.OUT
SOURCE = Path(r"C:\Users\Boxuan\Documents\GitHub\ReutersNewstTraining")
PROCESSED = SOURCE / "processed"
RAW_NEWS = SOURCE / "Reuters"
COMPACT = ROOT / "data/macro/exp_crsp_daily_compact.parquet"
NAMES = ROOT / "data/gn/gn_crsp_names.parquet"
RIC = PROCESSED / "ric_to_ticker_map.csv"
LINKS = ROOT / "data/nbr/links.parquet"
CM_EDGES = ROOT / "data/plc/S2_comention_edges.parquet"
CM_MONTHLY = ROOT / "data/plc/S2_comention_monthly.parquet"
FORMATION = pd.Timestamp(CFG.FORMATION)
FIRST = pd.Timestamp(CFG.FIRST)
DATES = pd.date_range(FIRST, periods=CFG.N_WEEKS, freq="W-FRI")
READ_START = pd.Timestamp("2020-11-01")
READ_END = pd.Timestamp("2025-11-21")
FACTSET_ASOF = pd.Timestamp("2025-06-30")
ADV_MIN = 10_000_000.0
MEDIAN_DOLLAR_MIN = 5_000_000.0
VOLUME_MIN = 100_000.0
LIQUIDITY_DAYS = 60
LIQUIDITY_MIN_OBS = 55
BETA_DAYS = 252
BETA_MIN_OBS = 126
RIDGE_ALPHA = 10.0
MIN_TRAIN = 52
FLOOR = 1e-12
FEATURES = ([f"log_{kind}_rv{w}" for kind in ("total", "idio", "mkt") for w in (1, 5, 22)]
            + ["log_ind_rv5", "log_ind_rv22", "negret1", "posret1", "beta_mkt", "beta_ind", "log_me"])
TARGETS = ("idio", "total")
LABEL_FIELDS = ["case_id", "date", "permno", "log_y_idio", "log_y_total", "y_idio", "y_total"]
T0 = time.monotonic()


def log(*values):
    print(f"[{time.monotonic() - T0:7.1f}s]", *values, flush=True)


def json_default(value):
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_frame(frame, name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def read_frame(path, columns=None, lo=None, hi=None, ids=None):
    source = ds.dataset(str(path), format="parquet")
    filt = None
    if lo is not None:
        string_date = str(source.schema.field("date").type) == "string"
        a, b = pd.Timestamp(lo), pd.Timestamp(hi)
        a, b = (a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) if string_date else (a.date(), b.date())
        filt = (ds.field("date") >= a) & (ds.field("date") <= b)
    if ids is not None:
        id_filter = ds.field("permno").isin([int(p) for p in ids])
        filt = id_filter if filt is None else filt & id_filter
    result = source.to_table(columns=columns, filter=filt).to_pandas()
    if "date" in result:
        result["date"] = pd.to_datetime(result.date)
    return result


def read_raw(lo, hi, ids=None):
    parts = []
    for filename, lower, upper in [
        ("crsp_daily.parquet", pd.Timestamp(lo), min(pd.Timestamp(hi), pd.Timestamp("2024-12-31"))),
        ("crsp_daily_2025.parquet", max(pd.Timestamp(lo), pd.Timestamp("2025-01-01")), pd.Timestamp(hi)),
    ]:
        if lower <= upper:
            frame = read_frame(PROCESSED / filename, ["permno", "date", "ret", "prc", "shrout", "vol"], lower, upper, ids)
            frame["source_file"] = filename
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate raw CRSP firm-session rows")
    return result


def cutoff(date):
    return (pd.Timestamp(date).normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")


def news_window(calendar, date):
    pos = calendar.get_indexer([pd.Timestamp(date)])[0]
    if pos < 5:
        raise ValueError("Decision requires five preceding exchange sessions")
    return cutoff(calendar[pos - 5]), cutoff(date)


def future_positions(calendar, date):
    pos = calendar.get_indexer([pd.Timestamp(date)])[0]
    if pos < 0 or pos + 5 >= len(calendar):
        raise ValueError("Decision does not have five calendar outcome sessions")
    return np.arange(pos + 1, pos + 6)


def split_role(index):
    if index < 4:
        return "train", "calibration"
    if index < 8:
        return "train", "reflection"
    if index < 12:
        return "validation", "validation"
    return "test", "test"


def selected_llm(firm_index, date_index):
    return date_index >= 12 or firm_index // 4 == date_index % 3


def active_names(names, date):
    current = names[(names.namedt <= date) & (names.nameenddt >= date)].copy()
    return current.sort_values(["permno", "namedt"]).drop_duplicates("permno", keep="last")


class NewsMapper:
    def __init__(self, names, ric):
        self.names = names
        self.ric = ric[ric.is_us.fillna(False).astype(bool) & ric.ric.notna() & ric.ticker.notna()].copy()
        self.ric["ticker"] = self.ric.ticker.str.upper()
        self.cache = {}

    def for_date(self, date):
        date = pd.Timestamp(date).normalize()
        if date not in self.cache:
            names = active_names(self.names, date)
            names = names[names.ticker.notna()].copy()
            names["ticker"] = names.ticker.str.upper()
            n = names.groupby("ticker").permno.nunique()
            names = names[names.ticker.isin(n[n == 1].index)]
            lookup = dict(zip(names.ticker, names.permno.astype(int)))
            mapped = self.ric.assign(permno=self.ric.ticker.map(lookup)).dropna(subset=["permno"])
            counts = mapped.groupby("ric").permno.nunique()
            mapped = mapped[mapped.ric.isin(counts[counts == 1].index)]
            self.cache[date] = dict(zip(mapped.ric, mapped.permno.astype(int)))
        return self.cache[date]


def news_helpers():
    # Import the old parsers by absolute file, restoring its legacy dependency alias.
    prior = sys.modules.get("common")
    sys.modules["common"] = load_module("midcap_legacy_common", ROOT / "code/news_percentile/common.py")
    try:
        return load_module("midcap_legacy_news_helpers", ROOT / "code/news_percentile/build.py")
    finally:
        if prior is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = prior


def news_records(month, mapper, lo, hi, archive):
    helper = news_helpers()
    source = RAW_NEWS / f"STORY.RTRS.{month}.REC.JSON.txt"
    counts = Counter()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream, gzip.open(archive, "wt", encoding="utf-8") as saved:
        for line in stream:
            counts["lines"] += 1
            obj = helper.parse_line(line)
            if not obj or not isinstance(obj.get("data"), dict):
                continue
            data = obj["data"]
            if data.get("language") not in (None, "en"):
                continue
            try:
                ts = pd.Timestamp(data.get("versionCreated") or data.get("firstCreated"))
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            except (ValueError, TypeError):
                counts["invalid_timestamp"] += 1
                continue
            if pd.isna(ts) or not lo <= ts < hi:
                continue
            lookup = mapper.for_date(ts.tz_convert("America/New_York").tz_localize(None).normalize())
            ids = sorted({lookup[t] for t in helper.parse_tags(data.get("subjects")) if t in lookup})
            headline = str(data.get("headline") or "")
            body = str(data.get("body") or "")
            kind, family, _ = helper.common_category(headline, body, "")
            kind = kind or "other"
            family = family or helper.substantive_category(headline, body)
            alt = str(data.get("altId") or data.get("id") or obj.get("guid") or "")
            if not alt:
                alt = hashlib.sha256((headline + str(data.get("firstCreated"))).encode()).hexdigest()[:24]
            event_id = hashlib.sha256((alt + "|" + ts.isoformat() + "|" + headline + "|" + body).encode()).hexdigest()[:24]
            row = {"event_id": event_id, "ts": ts.isoformat(), "alt": alt, "headline": headline,
                   "body": body, "ids": ids, "kind": kind, "family": family,
                   "urgency": int(data.get("urgency") or obj.get("urgency") or 3)}
            row["_usable"] = data.get("pubStatus", "stat:usable") not in {"stat:canceled", "stat:cancelled", "stat:withdrawn"}
            saved.write(json.dumps({**row, "source_month": month, "source_line": counts["lines"],
                                    "versionCreated": data.get("versionCreated"),
                                    "firstCreated": data.get("firstCreated"), "subjects": data.get("subjects"),
                                    "pubStatus": data.get("pubStatus")}, ensure_ascii=True) + "\n")
            counts["eligible_versions"] += 1
            yield row
    log("Reuters scanned", month, dict(counts))


def keep_latest(latest, row):
    old = latest.get(row["alt"])
    if old is None or (row["ts"], row["event_id"]) > (old["ts"], old["event_id"]):
        latest[row["alt"]] = row


def liquidity_metrics(raw, sessions):
    frame = raw[raw.date.isin(sessions)].copy()
    valid = frame.vol.ge(0) & frame.prc.abs().gt(0) & np.isfinite(frame.vol) & np.isfinite(frame.prc)
    frame["dollar_volume"] = (frame.vol * frame.prc.abs()).where(valid)
    frame["observed_positive"] = valid & frame.vol.gt(0)
    result = frame.groupby("permno").agg(
        adv60_usd=("dollar_volume", "mean"), median_dollar_volume60_usd=("dollar_volume", "median"),
        average_shares60=("vol", "mean"), liquidity_observations=("dollar_volume", "count"),
        positive_volume_sessions=("observed_positive", "sum"), liquidity_first=("date", "min"), liquidity_last=("date", "max"))
    result["liquid"] = (result.adv60_usd.ge(ADV_MIN) & result.median_dollar_volume60_usd.ge(MEDIAN_DOLLAR_MIN)
                        & result.average_shares60.ge(VOLUME_MIN) & result.liquidity_observations.ge(LIQUIDITY_MIN_OBS)
                        & result.positive_volume_sessions.ge(LIQUIDITY_MIN_OBS))
    return result.reset_index(), frame


def select_universe(names, mapper):
    snap = read_frame(COMPACT, lo=FORMATION, hi=FORMATION)
    save_frame(snap, "inputs/formation_snapshot.parquet")
    candidates = snap[snap.me.between(2000, 20000) & snap.prc.abs().ge(5)
                      & snap.share_type.isin([0, 2]) & snap.exchcd.isin([1, 2, 3])].copy()
    current = active_names(names, FORMATION)
    candidates = candidates.merge(current[["permno", "ticker", "comnam", "carried_fwd"]], on="permno", validate="one_to_one")
    candidates = candidates[candidates.ticker.notna() & candidates.ff49.notna()].copy()
    history = read_frame(COMPACT, ["permno", "date", "ret"], "2023-01-01", FORMATION, candidates.permno)
    save_frame(history, "inputs/eligibility_history.parquet")
    counts = history[np.isfinite(history.ret)].groupby("permno").size()
    candidates["n_history"] = candidates.permno.map(counts).fillna(0).astype(int)
    calendar = pd.DatetimeIndex(sorted(history.date.unique()))
    sessions = calendar[calendar <= FORMATION][-LIQUIDITY_DAYS:]
    raw = read_raw(sessions[0], FORMATION, candidates.permno)
    liquidity, daily_liquidity = liquidity_metrics(raw, sessions)
    save_frame(daily_liquidity, "inputs/liquidity_daily.parquet")
    candidates = candidates.merge(liquidity, on="permno", how="left", validate="one_to_one")
    last = raw[raw.date.eq(FORMATION)].copy()
    last["raw_me_millions"] = last.prc.abs() * last.shrout / 1000
    candidates = candidates.merge(last[["permno", "raw_me_millions"]], on="permno", how="left")
    candidates["market_cap_usd"] = candidates.me * 1_000_000.0
    candidates["size_units_verified"] = np.isclose(candidates.me, candidates.raw_me_millions, rtol=1e-5, atol=.01)
    latest = {}
    for row in news_records("2025-07", mapper, pd.Timestamp("2025-07-01", tz="America/New_York").tz_convert("UTC"),
                            cutoff(FORMATION), OUT / "inputs/news_versions/formation_july.jsonl.gz"):
        keep_latest(latest, row)
    coverage = Counter(p for row in latest.values() if row["_usable"] for p in row["ids"])
    candidates["july_news_events"] = candidates.permno.map(coverage).fillna(0).astype(int)
    candidates["eligible"] = (candidates.liquid.fillna(False) & candidates.n_history.ge(400)
                              & candidates.july_news_events.ge(5) & candidates.size_units_verified)
    eligible = candidates[candidates.eligible].sort_values(
        ["july_news_events", "adv60_usd", "permno"], ascending=[False, False, True]).copy()
    eligible["company_key"] = eligible.comnam.str.upper().str.replace(r"\W+", "", regex=True)
    universe = eligible.drop_duplicates("company_key").drop_duplicates("ff49").head(CFG.N_FIRMS).copy()
    if len(universe) != CFG.N_FIRMS:
        raise RuntimeError(f"Only {len(universe)} distinct industries passed the frozen screen")
    industry_names = {}
    for line in (ROOT / "data/macro/exp_ff49_siccodes.txt").read_text().splitlines():
        match = re.match(r"^\s*(\d{1,2})\s+\S+\s+(.+)$", line)
        if match:
            industry_names[int(match[1])] = match[2].strip()
    universe["industry"] = universe.ff49.astype(int).map(industry_names)
    universe["company"] = universe.comnam
    universe["formation"] = FORMATION.strftime("%Y-%m-%d")
    universe["selection_rank"] = np.arange(1, len(universe) + 1)
    candidates["selected"] = candidates.permno.isin(universe.permno)
    save_frame(candidates, "inputs/universe_screen.parquet")
    save_frame(universe, "universe.parquet")
    universe.to_csv(OUT / "universe.csv", index=False)
    log("Selected July-only universe", universe.ticker.tolist(), "eligible candidates", int(candidates.eligible.sum()))
    return universe.reset_index(drop=True)


def weighted_factors(compact, calendar):
    x = compact.sort_values(["permno", "date"]).copy()
    if x.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate compact CRSP rows")
    x["session"] = calendar.get_indexer(x.date)
    grouped = x.groupby("permno", sort=False)
    for source, dest in [("me", "lag_me"), ("ff49", "lag_ff49"), ("share_type", "lag_share_type"), ("session", "lag_session")]:
        x[dest] = grouped[source].shift(1)
    valid = (x.lag_share_type.isin([0, 2]) & x.lag_me.gt(0) & np.isfinite(x.ret)
             & x.session.sub(x.lag_session).eq(1))
    x["weight"] = x.lag_me.where(valid, 0.0)
    x["weighted_return"] = (x.weight * x.ret).where(valid, 0.0)
    market = x.groupby("date").agg(mkt_num=("weighted_return", "sum"), mkt_den=("weight", "sum"))
    market["mkt_ret"] = market.mkt_num / market.mkt_den.where(market.mkt_den.gt(0))
    industry = x[x.weight.gt(0) & x.lag_ff49.notna()].groupby(["date", "lag_ff49"]).agg(
        ind_num=("weighted_return", "sum"), ind_den=("weight", "sum"), ind_members=("permno", "size")).reset_index()
    return x, market, industry


def leave_one_out(num, den, own_weight, own_return):
    own_weight = np.nan_to_num(np.asarray(own_weight, dtype=float), nan=0.0)
    own_contribution = np.where(own_weight > 0, own_weight * np.asarray(own_return), 0.0)
    denominator = np.asarray(den) - own_weight
    return np.divide(np.asarray(num) - own_contribution, denominator,
                     out=np.full(len(denominator), np.nan), where=denominator > 0)


def fit_beta(ret, factors, origin, include_origin=True):
    end = origin + int(include_origin)
    begin = max(0, end - BETA_DAYS)
    y, f = ret[begin:end], factors[begin:end]
    valid = np.isfinite(y) & np.isfinite(f).all(axis=1)
    if valid.sum() < BETA_MIN_OBS:
        return np.full(3, np.nan), int(valid.sum())
    design = np.column_stack([np.ones(valid.sum()), f[valid]])
    beta = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return beta, int(valid.sum())


def target_variances(ret, factors, beta, positions):
    future = ret[positions]
    total = max(float(np.mean(future ** 2)), FLOOR) if np.isfinite(future).all() else np.nan
    f = factors[positions]
    if np.isfinite(future).all() and np.isfinite(f).all() and np.isfinite(beta).all():
        residual = future - (beta[0] + f @ beta[1:])
        idio = max(float(np.mean(residual ** 2)), FLOOR)
    else:
        idio = np.nan
    return idio, total


def build_weekly_rows(universe):
    log("Reading factor inputs and raw focal histories")
    compact = read_frame(COMPACT, lo=READ_START, hi=READ_END)
    save_frame(compact, "inputs/factor_constituents_sealed.parquet")
    calendar = pd.DatetimeIndex(sorted(compact.date.unique()))
    if not DATES.isin(calendar).all():
        raise RuntimeError("A frozen Friday is missing from the exchange calendar")
    save_frame(pd.DataFrame({"date": calendar}), "inputs/session_calendar.parquet")
    horizons = {d.strftime("%Y-%m-%d"): [str(s.date()) for s in calendar[future_positions(calendar, d)]] for d in DATES}
    overlaps = [{"first_origin": a, "second_origin": b, "shared_sessions": sorted(set(horizons[a]) & set(horizons[b]))}
                for a, b in zip(list(horizons)[:-1], list(horizons)[1:]) if set(horizons[a]) & set(horizons[b])]
    dump_json(OUT / "outcome_horizons.json", {"sessions": horizons, "adjacent_overlaps": overlaps,
              "rule": "Fixed Friday origins and exactly next5 exchange sessions are authoritative. Holiday-induced overlaps are preserved and disclosed; training still requires full maturity."})
    weighted, market, industries = weighted_factors(compact, calendar)
    del compact
    save_frame(market.reset_index(), "inputs/market_factors_sealed.parquet")
    save_frame(industries, "inputs/industry_factor_sums_sealed.parquet")
    raw = read_raw(READ_START, READ_END, universe.permno)
    save_frame(raw, "inputs/raw_focal_returns_sealed.parquet")
    factor_focal = weighted[weighted.permno.isin(universe.permno)].copy()
    del weighted
    save_frame(factor_focal, "inputs/focal_factor_weights_sealed.parquet")
    origins = calendar[(calendar.dayofweek == 4) & (calendar >= "2022-01-01") & (calendar <= DATES[-1])]
    rows, daily_panels = [], []
    for firm in universe.itertuples():
        panel = raw[raw.permno.eq(firm.permno)].set_index("date").reindex(calendar)
        panel["permno"] = int(firm.permno)
        panel.index.name = "date"
        panel["me"] = panel.prc.abs() * panel.shrout / 1000
        panel = panel.join(market[["mkt_ret"]])
        focal = factor_focal[factor_focal.permno.eq(firm.permno)].set_index("date").reindex(calendar)
        # Classification is carried forward from the latest strictly earlier known session.
        panel["lag_ff49"] = focal.ff49.ffill().shift(1)
        panel = panel.reset_index().merge(industries, on=["date", "lag_ff49"], how="left", validate="one_to_one").set_index("date")
        own_weight = focal.weight.where(focal.lag_ff49.eq(panel.lag_ff49), 0.0).fillna(0)
        panel["ind_ret"] = leave_one_out(panel.ind_num, panel.ind_den, own_weight, focal.ret)
        panel["industry_peers"] = panel.ind_members - own_weight.gt(0).astype(int)
        panel.loc[panel.industry_peers.lt(2), "ind_ret"] = np.nan
        ret = panel.ret.to_numpy(float)
        factors = panel[["mkt_ret", "ind_ret"]].to_numpy(float)
        residual = np.full(len(panel), np.nan)
        for i in range(BETA_MIN_OBS, len(panel)):
            beta, _ = fit_beta(ret, factors, i, include_origin=False)
            if np.isfinite(beta).all() and np.isfinite(ret[i]) and np.isfinite(factors[i]).all():
                residual[i] = ret[i] - beta[0] - factors[i] @ beta[1:]
        panel["causal_daily_residual"] = residual
        for kind, values in [("total", panel.ret), ("idio", panel.causal_daily_residual), ("mkt", panel.mkt_ret)]:
            for w in (1, 5, 22):
                panel[f"log_{kind}_rv{w}"] = np.log(values.pow(2).rolling(w, min_periods=w).mean().clip(lower=FLOOR))
        for w in (5, 22):
            panel[f"log_ind_rv{w}"] = np.log(panel.ind_ret.pow(2).rolling(w, min_periods=w).mean().clip(lower=FLOOR))
        panel["negret1"] = (-panel.ret).clip(lower=0)
        panel["posret1"] = panel.ret.clip(lower=0)
        panel["log_me"] = np.log(panel.me.where(panel.me.gt(0)))
        for date in origins:
            pos = calendar.get_loc(date)
            beta, n_beta = fit_beta(ret, factors, pos)
            positions = future_positions(calendar, date)
            idio, total = target_variances(ret, factors, beta, positions)
            row = {"case_id": f"{firm.permno}:{date:%Y-%m-%d}", "date": date, "permno": int(firm.permno),
                   "outcome_start": calendar[positions[0]], "outcome_end": calendar[positions[-1]],
                   "beta_start": calendar[max(0, pos + 1 - BETA_DAYS)], "beta_end": date, "beta_n": n_beta,
                   "beta_alpha": beta[0], "beta_mkt": beta[1], "beta_ind": beta[2],
                   "y_idio": idio, "y_total": total, "log_y_idio": np.log(idio), "log_y_total": np.log(total),
                   "observed_outcome_returns": int(np.isfinite(ret[positions]).sum()),
                   "observed_outcome_factors": int(np.isfinite(factors[positions]).all(axis=1).sum())}
            row.update(panel.loc[date, [c for c in FEATURES if c not in {"beta_mkt", "beta_ind"}]].to_dict())
            rows.append(row)
        daily_panels.append(panel.reset_index())
        log("Computed causal risk state", firm.ticker)
    save_frame(pd.concat(daily_panels, ignore_index=True), "inputs/daily_focal_state_sealed.parquet")
    weekly = pd.DataFrame(rows).sort_values(["date", "permno"]).reset_index(drop=True)
    feature_columns = ["case_id", "date", "permno", "outcome_start", "outcome_end", "beta_start", "beta_end", "beta_n", "beta_alpha"] + FEATURES
    save_frame(weekly[feature_columns], "inputs/weekly_features.parquet")
    save_frame(weekly[weekly.date < FIRST][LABEL_FIELDS + ["outcome_start", "outcome_end"]], "inputs/weekly_labels_pre_pilot.parquet")
    save_frame(weekly[["case_id", "date", "permno", "outcome_start", "outcome_end", "observed_outcome_returns", "observed_outcome_factors"]],
               "inputs/outcome_coverage.parquet")
    return weekly, calendar


def ridge_forecast(train, current, target):
    x = train[FEATURES].to_numpy(float)
    z = current[FEATURES].to_numpy(float)
    missing_x, missing_z = ~np.isfinite(x), ~np.isfinite(z)
    med = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in x.T])
    x, z = np.where(missing_x, med, x), np.where(missing_z, med, z)
    x, z = np.column_stack([x, missing_x]), np.column_stack([z, missing_z])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = Ridge(alpha=RIDGE_ALPHA).fit((x - mean) / scale, train[target].to_numpy(float))
    predicted = float(model.predict((z - mean) / scale)[0])
    params = {"intercept": float(model.intercept_), "coefficients": model.coef_.tolist(),
              "medians": med.tolist(), "means": mean.tolist(), "scales": scale.tolist()}
    return predicted, params


def causal_training_rows(rows, origin, target):
    return rows[(rows.date >= origin - pd.DateOffset(years=2)) & (rows.date < origin)
                & (rows.outcome_end <= origin) & np.isfinite(rows[target])].copy()


def empirical_percentile(reference, value):
    ref = np.sort(np.asarray(reference, dtype=float))
    return float((np.searchsorted(ref, value, side="right") + .5) / (len(ref) + 1.0))


def fit_baselines(weekly, universe):
    vintages, params, predictions, reference_rows = [], [], [], []
    for firm in universe.itertuples():
        rows = weekly[weekly.permno.eq(firm.permno)].sort_values("date")
        for current in rows[rows.date.ge("2024-01-01")].itertuples():
            date = current.date
            if date < FIRST and current.outcome_end >= FIRST:
                continue
            if date >= FIRST and date not in DATES:
                continue
            forecast = {"case_id": current.case_id, "date": date, "permno": int(firm.permno)}
            point = rows[rows.date.eq(date)]
            for kind in TARGETS:
                target = f"log_y_{kind}"
                train = causal_training_rows(rows, date, target)
                if len(train) < MIN_TRAIN:
                    if date >= FIRST:
                        raise RuntimeError(f"Insufficient mature history for {firm.ticker} at {date}: {len(train)}")
                    continue
                prediction, parameters = ridge_forecast(train, point, target)
                forecast[f"baseline_{kind}"] = prediction
                forecast[f"historical_mean_{kind}"] = float(train[target].mean())
                fit_id = f"{current.case_id}:{kind}"
                vintages.append({"fit_id": fit_id, "permno": int(firm.permno), "date": date, "target": kind,
                                 "train_first_origin": train.date.min(), "train_last_origin": train.date.max(),
                                 "train_max_maturity": train.outcome_end.max(), "train_n": len(train),
                                 "train_window_start": date - pd.DateOffset(years=2), "ridge_alpha": RIDGE_ALPHA,
                                 "train_case_ids": train.case_id.tolist()})
                params.append({"fit_id": fit_id, **parameters})
                if date < FIRST and math.isfinite(getattr(current, target)):
                    reference_rows.append({"case_id": current.case_id, "permno": int(firm.permno), "date": date,
                                           "outcome_end": current.outcome_end, "target": kind,
                                           "prediction": prediction, "residual": getattr(current, target) - prediction})
            if date >= FIRST:
                predictions.append(forecast)
        log("Refit weekly ridge models", firm.ticker)
    refs = pd.DataFrame(reference_rows)
    references = {}
    for firm in universe.itertuples():
        references[str(firm.permno)] = {}
        for kind in TARGETS:
            values = sorted(refs.loc[refs.permno.eq(firm.permno) & refs.target.eq(kind), "residual"].astype(float).tolist())
            if len(values) < 26:
                raise RuntimeError(f"Too few pre-pilot OOS residuals for {firm.ticker}: {len(values)}")
            references[str(firm.permno)][kind] = values
    save_frame(refs, "inputs/reference_oos_forecasts.parquet")
    save_frame(pd.DataFrame(vintages), "model_vintages.parquet")
    dump_json(OUT / "model_parameters.json", params)
    dump_json(OUT / "references.json", references)
    return pd.DataFrame(predictions), references


def write_cases_labels(weekly, predictions, universe, references):
    cases = []
    lookup = predictions.set_index("case_id")
    for i, date in enumerate(DATES):
        split, role = split_role(i)
        for j, firm in enumerate(universe.itertuples()):
            case_id = f"{firm.permno}:{date:%Y-%m-%d}"
            row = {"case_id": case_id, "permno": int(firm.permno), "ticker": firm.ticker,
                   "company": firm.company, "industry": firm.industry, "date": date.strftime("%Y-%m-%d"),
                   "asof": cutoff(date).isoformat(), "split": split, "role": role,
                   "selected_llm": bool(selected_llm(j, i))}
            for kind in TARGETS:
                ref = references[str(firm.permno)][kind]
                row[f"baseline_{kind}"] = float(lookup.loc[case_id, f"baseline_{kind}"])
                row[f"historical_mean_{kind}"] = float(lookup.loc[case_id, f"historical_mean_{kind}"])
                row[f"q_{kind}"] = {str(p / 100): float(np.quantile(ref, p / 100)) for p in (5, 10, 25, 50, 75, 90, 95)}
                row[f"p0_{kind}"] = empirical_percentile(ref, 0.0)
            cases.append(row)
    tmp = OUT / "cases.jsonl.tmp"
    tmp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in cases), encoding="utf-8")
    tmp.replace(OUT / "cases.jsonl")
    labels = weekly[weekly.date.isin(DATES)][LABEL_FIELDS].copy()
    save_frame(labels[labels.date < DATES[12]], "labels_development.parquet")
    save_frame(labels[labels.date >= DATES[12]], "labels_test_sealed.parquet")
    return cases


def invert_role(role):
    return {"customer": "supplier", "supplier": "customer", "competitor": "competitor"}[role]


def directed_factset_rows(links, focals):
    records = {}
    for row in links.itertuples():
        if row.role not in {"customer", "supplier"}:
            continue
        for focal, neighbor, role in [(row.focal_permno, row.nbr_permno, row.role),
                                      (row.nbr_permno, row.focal_permno, invert_role(row.role))]:
            if focal not in focals or focal == neighbor:
                continue
            rev = None if pd.isna(row.rev_pct) else float(row.rev_pct)
            key = (int(focal), int(neighbor), role)
            old = records.get(key)
            if old is None or (rev is not None and (old["rev_pct"] is None or rev > old["rev_pct"])):
                records[key] = {"focal_permno": int(focal), "permno": int(neighbor), "role": role,
                                "source": "FactSet", "rev_pct": rev, "asof": "2025-06-30"}
    return list(records.values())


def prior_months(ym):
    period = pd.Period(str(ym), freq="M")
    return [int((period - k).strftime("%Y%m")) for k in range(1, 13)]


def build_relations(universe, names):
    links = pd.read_parquet(LINKS)
    links = links[(links.s <= FACTSET_ASOF) & (links.e.fillna(pd.Timestamp("2030-01-01")) >= FACTSET_ASOF)]
    focals = set(universe.permno.astype(int))
    links = links[links.focal_permno.isin(focals) | links.nbr_permno.isin(focals)]
    save_frame(links, "inputs/factset_frozen_rows.parquet")
    factset = directed_factset_rows(links, focals)
    # This EXACT full-calendar coder is part of the old graph's file format, not a universe screen.
    codes = read_frame(COMPACT, ["permno"], "2013-06-01", "2025-12-31")
    permnos = np.sort(codes.permno.unique())
    del codes
    save_frame(pd.DataFrame({"code": np.arange(len(permnos)), "permno": permnos}), "inputs/comention_decoder.parquet")
    cm = pd.read_parquet(CM_EDGES)
    months = sorted(set(int(d.strftime("%Y%m")) for d in DATES))
    cm = cm[cm.ym_event.isin(months)].copy()
    if cm[["pc_i", "pc_j"]].min().min() < 0 or cm[["pc_i", "pc_j"]].max().max() >= len(permnos):
        raise RuntimeError("Co-mention code is outside the generation calendar")
    cm["a"], cm["b"] = permnos[cm.pc_i.to_numpy(int)], permnos[cm.pc_j.to_numpy(int)]
    cm = cm[cm.a.isin(focals) | cm.b.isin(focals)].copy()
    monthly = pd.read_parquet(CM_MONTHLY)
    focal_codes = set(np.flatnonzero(np.isin(permnos, list(focals))))
    monthly = monthly[monthly.pc_i.isin(focal_codes) | monthly.pc_j.isin(focal_codes)].copy()
    needed_months = {m for ym in months for m in prior_months(ym)}
    monthly = monthly[monthly.ym.isin(needed_months)]
    checks = []
    for ym in months:
        agg = monthly[monthly.ym.isin(prior_months(ym))].groupby(["pc_i", "pc_j"]).agg(cnt12=("cnt", "sum"), w12=("w", "sum"))
        comparison = cm[cm.ym_event.eq(ym)].merge(agg, on=["pc_i", "pc_j"], how="outer", suffixes=("_stored", "_rebuilt"))
        ok = (comparison.cnt12_stored.eq(comparison.cnt12_rebuilt)
              & np.isclose(comparison.w12_stored, comparison.w12_rebuilt, rtol=1e-9, atol=1e-9))
        if not ok.all():
            raise RuntimeError(f"Co-mention preceding-12-month reconstruction differs at {ym}")
        checks.append({"ym_event": ym, "rows_verified": len(comparison), "source_months": prior_months(ym)})
    save_frame(cm, "inputs/comention_edges_decoded.parquet")
    save_frame(monthly, "inputs/comention_prior_monthly.parquet")
    dump_json(OUT / "comention_generation_check.json", checks)
    result = {}
    for date in DATES:
        records = list(factset)
        ym = int(date.strftime("%Y%m"))
        for row in cm[cm.ym_event.eq(ym) & cm.significant.eq(1)].itertuples():
            for focal, neighbor in ((row.a, row.b), (row.b, row.a)):
                if focal in focals:
                    records.append({"focal_permno": int(focal), "permno": int(neighbor), "role": "co-mention",
                                    "source": "Reuters PRIOR12MONTH", "rev_pct": None,
                                    "asof": (date.to_period("M").start_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d")})
        name_map = active_names(names, date).set_index("permno")
        output = {str(p): [] for p in sorted(focals)}
        for record in records:
            item = {k: v for k, v in record.items() if k != "focal_permno"}
            neighbor = item["permno"]
            if neighbor in name_map.index:
                n = name_map.loc[neighbor]
                item.update(ticker=str(n.ticker) if pd.notna(n.ticker) else "", company=str(n.comnam) if pd.notna(n.comnam) else "")
            else:
                item.update(ticker="", company="")
            output[str(record["focal_permno"])].append(item)
        for items in output.values():
            items.sort(key=lambda v: (v["source"], v["role"], v["permno"]))
        key = date.strftime("%Y-%m-%d")
        dump_json(OUT / "relations" / f"{key}.json", output)
        result[key] = output
    log("Verified and wrote relations", len(result), "dates")
    return result


def build_events(universe, calendar, mapper, relations):
    windows = {date.strftime("%Y-%m-%d"): news_window(calendar, date) for date in DATES}
    latest = {date: {} for date in windows}
    lo = min(v[0] for v in windows.values())
    hi = max(v[1] for v in windows.values())
    for month in pd.period_range(lo.tz_localize(None), hi.tz_localize(None), freq="M"):
        for row in news_records(str(month), mapper, lo, hi, OUT / "inputs/news_versions" / f"{month}.jsonl.gz"):
            ts = pd.Timestamp(row["ts"])
            for key, (lower, upper) in windows.items():
                if lower <= ts < upper:
                    keep_latest(latest[key], row)
    counts = []
    for date, candidates in latest.items():
        neighbors = {item["permno"] for items in relations[date].values() for item in items}
        allowed = neighbors | set(universe.permno.astype(int))
        events = []
        for row in candidates.values():
            if row["_usable"] and (row["kind"] in {"macro", "finance"} or allowed.intersection(row["ids"])):
                events.append({k: v for k, v in row.items() if not k.startswith("_")})
        events.sort(key=lambda row: (row["ts"], row["event_id"]))
        path = OUT / "events" / f"{date}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(events, stream, ensure_ascii=True, allow_nan=False)
        counts.append({"date": date, "window_start": windows[date][0].isoformat(), "cutoff": windows[date][1].isoformat(),
                       "all_latest_stories": len(candidates), "events": len(events),
                       "macro": sum(r["kind"] == "macro" for r in events), "finance": sum(r["kind"] == "finance" for r in events),
                       "body_characters": sum(len(r["body"]) for r in events)})
        log("Wrote full-body events", date, len(events))
    save_frame(pd.DataFrame(counts), "event_counts.parquet")
    return counts


def write_structural_audit(universe, cases, calendar):
    """Inspect availability, never outcome magnitudes or predictive performance."""
    frame = pd.DataFrame(cases)
    numerical = [f"{field}_{kind}" for field in ("baseline", "historical_mean") for kind in TARGETS]
    finite = np.isfinite(frame[numerical]).all(axis=1)
    if not finite.all():
        raise RuntimeError("Nonfinite conventional case forecasts")
    labels = pd.concat([pd.read_parquet(OUT / filename) for filename in
                        ("labels_development.parquet", "labels_test_sealed.parquet")], ignore_index=True)
    availability = labels[["case_id"]].copy()
    for kind in TARGETS:
        availability[f"finite_{kind}_label"] = np.isfinite(labels[f"log_y_{kind}"]) & np.isfinite(labels[f"y_{kind}"])
    availability = frame[["case_id", "permno", "ticker", "date", "split", "role", "selected_llm"]].merge(availability, on="case_id", validate="one_to_one")
    save_frame(availability, "label_availability.parquet")
    raw = pd.read_parquet(OUT / "inputs/raw_focal_returns_sealed.parquet", columns=["permno", "date", "ret"])
    records = []
    for firm in universe.itertuples():
        observed = raw[raw.permno.eq(firm.permno) & np.isfinite(raw.ret)]
        rows = availability[availability.permno.eq(firm.permno)]
        records.append({"permno": int(firm.permno), "ticker": firm.ticker, "finite_raw_return_count": len(observed),
                        "first_finite_raw_return": str(observed.date.min().date()), "last_finite_raw_return": str(observed.date.max().date()),
                        "pilot_missing_idio_labels": int((~rows.finite_idio_label).sum()),
                        "pilot_missing_total_labels": int((~rows.finite_total_label).sum())})
    summary = []
    for (split, sampled), rows in availability.groupby(["split", "selected_llm"]):
        summary.append({"split": split, "selected_llm": bool(sampled), "cases": len(rows),
                        "finite_idio_labels": int(rows.finite_idio_label.sum()), "finite_total_labels": int(rows.finite_total_label.sum())})
    missing_permnos = set(availability.loc[~availability.finite_idio_label | ~availability.finite_total_label, "permno"])
    evidence = {}
    with gzip.open(OUT / "inputs/news_versions/formation_july.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if missing_permnos.intersection(row["ids"]) and re.search(r"merger|merg|acquisition|delist|skydance", row["headline"], re.I):
                keep_latest(evidence, row)
    dump_json(OUT / "formation_corporate_action_evidence.json", {
        "purpose": "Post-selection structural attrition diagnosis using only formation-available news; no re-selection or successor return splicing.",
        "formation_cutoff": cutoff(FORMATION).isoformat(),
        "records": sorted(evidence.values(), key=lambda row: (row["ts"], row["event_id"]))})
    horizons = {d: set(calendar[future_positions(calendar, d)]) for d in DATES}
    boundaries = []
    for left, right, a, b in [("calibration", "reflection", DATES[:4], DATES[4:8]),
                               ("reflection", "validation", DATES[4:8], DATES[8:12]),
                               ("train", "validation", DATES[:8], DATES[8:12]),
                               ("validation", "test", DATES[8:12], DATES[12:])]:
        first = set().union(*(horizons[d] for d in a))
        second = set().union(*(horizons[d] for d in b))
        boundary = {"left": left, "right": right, "last_left_origin": str(a[-1].date()),
                    "last_left_label_maturity": str(max(first).date()), "first_right_origin": str(b[0].date()),
                    "shared_target_sessions": [str(d.date()) for d in sorted(first & second)],
                    "all_left_labels_mature_at_first_right_close": bool(max(first) <= b[0])}
        if boundary["shared_target_sessions"] or not boundary["all_left_labels_mature_at_first_right_close"]:
            raise RuntimeError(f"Development boundary is not causal: {boundary}")
        boundaries.append(boundary)
    features = pd.read_parquet(OUT / "inputs/weekly_features.parquet", columns=["case_id", "date", "permno", "outcome_end"])
    purges = []
    for date in DATES:
        prior = features[(features.date < date) & (features.date >= date - pd.DateOffset(years=2))]
        excluded = prior[prior.outcome_end > date]
        purges.append({"date": date, "excluded_unmatured_case_ids": excluded.case_id.tolist(),
                       "excluded_unmatured_rows": len(excluded),
                       "rule": "Exclude every training outcome with fifth session after this origin, even if its starting Friday is earlier."})
    save_frame(pd.DataFrame(purges), "causal_purge_audit.parquet")
    dump_json(OUT / "boundary_timing_audit.json", {"boundaries": boundaries,
              "close_convention": "Conventional state uses the decision close, so labels maturing at that close are causal. News timestamps are strictly earlier than16:00 NY.",
              "graph_refresh": [{"decision_date": str(d.date()), "ym_event": int(d.strftime("%Y%m")),
                                  "first_source_month": min(prior_months(int(d.strftime("%Y%m")))),
                                  "last_source_month": max(prior_months(int(d.strftime("%Y%m")))),
                                  "available_after_month_end": str((d.to_period("M").start_time - pd.Timedelta(days=1)).date())} for d in DATES]})
    audit = {"case_rows": len(cases), "all_baselines_and_historical_means_finite": bool(finite.all()),
             "counts": summary, "firms": records,
             "attrition_policy": "Retain every selected firm and every case. Missing outcomes are NaN, not zeros. No ex-post replacement and no unverified successor return splice. Exclude unavailable labels explicitly when scoring; record changed denominators.",
             "corporate_action_evidence": "formation_corporate_action_evidence.json",
             "delisting_status": "No dlret or dated delisting-event file available. Missing terminal coverage is observed; a dated delisting adjustment is not certified."}
    dump_json(OUT / "structural_coverage_audit.json", audit)
    log("Structural audit", {"cases": len(cases), "finite_baselines": int(finite.sum()), "firms_with_missing_labels": sorted(missing_permnos)})
    return audit


def metadata(universe, cases, event_counts):
    inputs = []
    for path in [COMPACT, NAMES, RIC, LINKS, CM_EDGES, CM_MONTHLY, PROCESSED / "crsp_daily.parquet",
                 PROCESSED / "crsp_daily_2025.parquet", Path(__file__).with_name("config.py"),
                 Path(__file__).with_name("PROTOCOL.md")]:
        stat = path.stat()
        inputs.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    definition = {
        "status": "complete", "formation": str(FORMATION.date()), "dates": [d.strftime("%Y-%m-%d") for d in DATES],
        "n_firms": len(universe), "n_cases": len(cases), "n_llm_development": sum(r["selected_llm"] and r["split"] != "test" for r in cases),
        "n_llm_test": sum(r["selected_llm"] and r["split"] == "test" for r in cases),
        "labels_development_rows": sum(r["split"] != "test" for r in cases), "labels_test_sealed_rows": sum(r["split"] == "test" for r in cases),
        "features": FEATURES, "missingness_indicator_features": FEATURES, "ridge_alpha": RIDGE_ALPHA,
        "ridge_training": "Per-firm weekly HAR-type ridge, trailing two calendar years, >=52 complete target rows; refit every origin.",
        "causal_maturity": "Training origin < prediction origin; fifth outcome session <= prediction origin close.",
        "risk_features": "1/5/22-session total and strictly causal daily factor-residual squared-return means, through current close.",
        "feature_scope": "Market/industry risk, log USD-million size, downside/upside returns as leverage-effect proxies, factor loadings. No balance-sheet or macro data used.",
        "factor_definition": "Lagged actual previous-session market caps and share-type membership; value-weight market and lag-FF49 industry. Focal contribution removed from industry; >=2 peers.",
        "beta_definition": "OLS intercept + market + leave-focal-out industry, trailing252 sessions, >=126 complete observations. Outcome beta estimated through origin and held fixed for next5 sessions.",
        "daily_residual_definition": "One-step daily residual uses beta fitted strictly before that daily return.",
        "label_definition": "log(max(mean of next5 squared daily returns or frozen-beta residuals,1e-12)); any missing target session makes its label missing.",
        "reference_definition": "Sorted residuals of weekly rolling causal forecasts from January2024, with outcome_end strictly before August1; frozen for entire pilot.",
        "reference_cdf": "p0=(searchsorted(sorted_residuals,0,side=right)+0.5)/(n+1), stored at full precision; q keys are probability strings0.05,0.1,0.25,0.5,0.75,0.9,0.95.",
        "universe_selection": "July31 snapshot; share_type in(0,2), exchcd in(1,2,3), cap USD2-20bn inclusive, price>=5, >=400 returns since2023Jan1; >=5 July stories strictly before formation close. Rank by July story count, ADV, permno; one company per FF49 industry.",
        "liquidity": {"sessions": LIQUIDITY_DAYS, "minimum_observed_positive_sessions": LIQUIDITY_MIN_OBS,
                      "mean_dollar_volume_min_usd": ADV_MIN, "median_dollar_volume_min_usd": MEDIAN_DOLLAR_MIN,
                      "average_shares_min": VOLUME_MIN, "formula": "abs(raw prc)*raw vol; no volume imputation", "end": "2025-07-31"},
        "units": {"compact_me": "USD millions", "raw_shrout": "thousands of shares", "raw_vol": "shares", "dollar_volume": "USD"},
        "news": "English Reuters versions, lower bound inclusive at close five exchange sessions earlier, upper bound strictly before16:00 NY. Latest altId version as of each cutoff, no headline/body truncation or priority/body-length cap; relevant focal/neighbor or macro/finance stories retained.",
        "relations": "FactSetJune30 endpoint frozen, reversed roles inverted, prior12month co-mentions decoded with exact legacy calendar and counts rebuilt. rev_pct is supplier revenue exposure in either direction, not customer procurement share.",
        "graph_availability": {
            "FactSet": {"endpoint": "2025-06-30", "selection": "s<=2025-06-30<=e in the supplied interval table; held constant at all16origins", "historical_release_vintage_verified": False},
            "comention": {"August": "2024-08-01 through2025-07-31", "September": "2024-09-01 through2025-08-31", "October": "2024-10-01 through2025-09-30", "November": "2024-11-01 through2025-10-31", "cumulative_rule": "Each monthly graph is refreshed using preceding12 calendar months, excluding all articles dated in or after the event month; earlier pilot news can enter later graphs when its month has completed.", "threshold": "cnt12>=2 and binomial z>3 as generated in S2_07_networks.py", "decoding": "Sorted unique compact permnos from2013-06-01 through2025-12-31, exactly the generation-time codebook; future entries used solely to decode IDs, never to screen firms or estimate edge counts."},
        },
        "limitations": [
            "Raw CRSP has no dlret or delisting event field; terminal returns cannot be certified delisting-adjusted. Missing sessions remain missing, never zero-filled or silently dropped.",
            "gn_crsp_names carries some December2024 names forward through2025; carried_fwd saved in universe and all name input rows.",
            "RIC crosswalk is a static source mapped through date-valid CRSP ticker intervals; fully point-in-time symbology vintages are unavailable.",
            "Compact share_type may use same-month JKP membership and historical availability; supplied compact metadata is not independently vintage-certified.",
            "FactSet historical link intervals in the existing source collapse overlapping and disjoint records; original disclosure vintages are unavailable. June30 freeze is the available endpoint, not a guarantee of original public release timing.",
            "Co-mention graph uses historical article dates and static extraction, with monthly counts checked; article-level historical version vintages are not independently certified.",
            "All files with sealed in their names contain returns/outcomes and must not enter LLM retrieval. Test labels are not used for firm selection, hyperparameters, or frozen reference CDFs.",
        ],
        "event_counts": event_counts, "inputs": inputs,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    dump_json(OUT / "preparation_metadata.json", definition)
    return definition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "numeric", "news", "audit"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = pd.read_parquet(NAMES)
    for col in ("namedt", "nameenddt"):
        names[col] = pd.to_datetime(names[col])
    ric = pd.read_csv(RIC)
    save_frame(names, "inputs/crsp_names.parquet")
    save_frame(ric, "inputs/ric_crosswalk.parquet")
    mapper = NewsMapper(names, ric)
    if args.stage in {"all", "numeric"}:
        universe = select_universe(names, mapper)
        weekly, calendar = build_weekly_rows(universe)
        predictions, references = fit_baselines(weekly, universe)
        cases = write_cases_labels(weekly, predictions, universe, references)
        log("Wrote numerical contract", len(cases), "cases; test outcomes saved without inspecting performance")
    else:
        universe = pd.read_parquet(OUT / "universe.parquet")
        calendar = pd.DatetimeIndex(pd.read_parquet(OUT / "inputs/session_calendar.parquet").date)
        cases = [json.loads(line) for line in (OUT / "cases.jsonl").read_text().splitlines()]
    write_structural_audit(universe, cases, calendar)
    if args.stage in {"all", "news"}:
        relations = build_relations(universe, names)
        counts = build_events(universe, calendar, mapper, relations)
        result = metadata(universe, cases, counts)
        log("COMPLETE", {k: result[k] for k in ("n_firms", "n_cases", "n_llm_development", "n_llm_test")})
    elif args.stage == "numeric":
        log("Numeric stage complete; run --stage news to finish event and relation inputs")
    else:
        log("Audit stage complete; no numerical forecasts, labels, universe or news files changed")


if __name__ == "__main__":
    main()
