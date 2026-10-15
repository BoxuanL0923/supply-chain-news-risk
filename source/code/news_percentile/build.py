from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    CRSP,
    CRSP_NAMES,
    RAW_REUTERS,
    RESULTS,
    RIC_MAP,
    RIC_NAMES,
    VOL_RESULTS,
    assign_decision_date,
    cutoff_utc,
    empirical_percentile,
    ensure_dirs,
    experiment_dates,
    load_config,
    stable_hash,
)


TAG_RE = re.compile(r"(?:^|\|)R:([^|]+)")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.I,
)
PREFIX_RE = re.compile(
    r"^(?:(?:UPDATE|WRAPUP|CORRECTED|REFILE|RPT|ANALYSIS|FACTBOX|INSTANT VIEW|BRIEF)\s*\d*[-: ]+)+",
    re.I,
)
WORD_RE = re.compile(r"[a-z0-9]+")
REF_RE = re.compile(r"\b[nN][A-Za-z0-9]{7,15}\b")

US_TOKEN = r"(?<![A-Za-z])(?:(?-i:U\.S\.|US)|United States)(?![A-Za-z])"
MACRO_PATTERNS = {
    "fomc": re.compile(
        r"(?:\bFOMC\b|\bPowell\b|"
        r"\bFED(?:'S)?\s+(?:WILLIAMS|BRAINARD|WALLER|BOSTIC|DALY|MESTER|BULLARD|KASHKARI|BARKIN|EVANS|GEORGE)\b|"
        r"\b(?:Federal Reserve|Fed)\b.{0,80}\b(?:rate|inflation|monetary policy|econom|meeting|decision|hike|cut|financial stability)\w*\b|"
        r"\b(?:rate|inflation|monetary policy|meeting|decision|hike|cut)\w*\b.{0,80}\b(?:Federal Reserve|Fed)\b)",
        re.I,
    ),
    "cpi": re.compile(
        rf"(?:{US_TOKEN}.{{0,35}}\bCPI\b|\bCPI\b.{{0,35}}{US_TOKEN}|"
        rf"{US_TOKEN}.{{0,35}}\bconsumer prices?\b.{{0,25}}\b(?:rise|rose|fall|fell|increase|data|index)\w*\b|"
        rf"\bconsumer prices?\b.{{0,25}}\b(?:rise|rose|fall|fell|increase|data|index)\w*\b.{{0,35}}{US_TOKEN})",
        re.I,
    ),
    "payrolls": re.compile(r"\b(?:U\.S\. nonfarm payrolls?|US nonfarm payrolls?|U\.S\. jobs report|US jobs report)\b", re.I),
    "ppi": re.compile(r"\b(?:U\.S\.|US) (?:producer prices?|PPI)\b", re.I),
    "gdp": re.compile(r"\b(?:U\.S\.|US) (?:gross domestic product|GDP)\b", re.I),
    "retail": re.compile(r"\b(?:U\.S\.|US) retail sales\b", re.I),
    "ism": re.compile(r"\bISM (?:manufacturing|services|non-manufacturing|index|data|survey|PMI)\b", re.I),
    "claims": re.compile(r"\b(?:U\.S\.|US) (?:weekly )?(?:jobless|unemployment) claims\b", re.I),
    "policy": re.compile(
        r"(?:\bU\.S\. (?:midterm )?elections?\b|\bUS (?:midterm )?elections?\b|"
        r"\b(?:White House|U\.S\. Congress|US Congress|U\.S\. Senate|U\.S\. House|Biden)\b"
        r".{0,100}\b(?:U\.S\. economy|US economy|tax (?:cuts?|rates?|bill|policy|increase)|spending|debt ceiling|tariff|sanction|regulat|antitrust|trade polic)\w*\b)",
        re.I,
    ),
}
US_EQUITY_RE = re.compile(
    r"\b(?:U\.S\. stocks?|US stocks?|Wall St(?:reet)? (?:stocks?|rises?|falls?|gains?|drops?|slips?|jumps?|closes?)|S&P\s*500|Dow Jones|"
    r"Nasdaq(?: Composite| 100)? (?:index |futures? )?(?:rises?|falls?|up|down|gains?|drops?|slips?|jumps?|closes?|futures?))\b",
    re.I,
)
US_RATES_RE = re.compile(
    rf"(?:{US_TOKEN}.{{0,35}}\bTreasur(?:y|ies|ys)\b.{{0,80}}\b(?:yield|note|bond|bill|auction|curve|futures?|bps|basis points?)\b|"
    rf"\bTreasur(?:y|ies|ys)\b.{{0,80}}\b(?:yield|note|bond|bill|auction|curve|futures?|bps|basis points?)\b.{{0,35}}{US_TOKEN})",
    re.I,
)
US_CREDIT_RE = re.compile(
    r"\b(?:U\.S\.|US) (?:corporate bonds?|credit spreads?|high[- ]yield|investment[- ]grade|funding market|banking system)\b",
    re.I,
)
US_VOL_RE = re.compile(
    r"\bCBOE volatility index\b|\bVIX\b.{0,30}\b(?:index|futures?|rises?|falls?|up|down|hits?|points?|level)\b|"
    r"\b(?:index|futures?|rises?|falls?|up|down|hits?|points?|level)\b.{0,30}\bVIX\b|"
    r"\b(?:U\.S\.|US) options market\b",
    re.I,
)
US_FX_RE = re.compile(r"\b(?:U\.S\. dollar index|US dollar index|dollar index)\b", re.I)
MARKET_WRAP_RE = re.compile(
    r"^(?:US STOCKS|GLOBAL MARKETS|LIVE MARKETS|WALL ST|WALL STREET|TREASURIES|FOREX|MARKET PULSE)",
    re.I,
)

CATEGORY_RULES = [
    ("earnings", re.compile(r"\b(?:earnings|quarterly results?|EPS|revenue guidance|profit forecast)\b", re.I)),
    ("legal_policy", re.compile(r"\b(?:lawsuit|court|regulator|antitrust|probe|investigation|FDA|FTC|DOJ|sanction|tariff)\b", re.I)),
    ("ma_ownership", re.compile(r"\b(?:acquisition|merger|takeover|buyout|tender offer|to buy|stake|spin[- ]?off)\b", re.I)),
    ("capital_financing", re.compile(r"\b(?:dividend|buyback|share repurchase|bond offering|debt offering|credit facility|refinanc|bankrupt|Chapter 11|credit rating)\b", re.I)),
    ("operations", re.compile(r"\b(?:supply chain|shortage|disruption|plant|factory|shutdown|outage|strike|logistics|freight|capacity|recall|cyberattack|ransomware)\b", re.I)),
    ("product_contract", re.compile(r"\b(?:launch|new product|contract award|wins? a contract|partnership|clinical trial|approval|clearance)\b", re.I)),
]

CATEGORICAL = ["permno", "dow"]
PRICE_FEATURES = [
    "log_rv1", "log_rv5", "log_rv22", "negret1", "posret1",
    "log_mkt_rv1", "log_mkt_rv5", "log_mkt_rv22",
    "log_ind_rv5", "log_ind_rv22", "beta60", "log_me",
]
OPTION_FEATURES = ["log_iv_variance", "iv_change_l1", "skew_25d_l1", "iv_term_l1", "term_10_60_l1"]
NEWS_FEATURES = [
    "own_active", "log_own_n_articles", "log_own_news_words",
    "own_lm_neg_share", "own_lm_pos_share", "own_lm_unc_share", "own_post_close",
    "network_active", "log_network_n_articles", "log_network_n_sources", "log_network_news_words",
    "network_lm_neg_share", "network_lm_pos_share", "network_lm_unc_share", "network_post_close",
    "network_rev_pct_max", "network_comention_z_max",
    "log_graph_degree", "log_graph_supplier_degree", "log_graph_customer_degree",
    "log_graph_revenue_links", "log_graph_comention_links",
]
STRONG_FEATURES = PRICE_FEATURES + OPTION_FEATURES + NEWS_FEATURES


def log(start: float, *parts: object) -> None:
    print(f"[{time.time() - start:7.1f}s]", *parts, flush=True)


def parse_line(raw: bytes) -> dict | None:
    position = raw.find(b"{")
    if position < 0:
        return None
    value = raw[position:].rstrip().rstrip(b",")
    for cut in range(4):
        try:
            return json.loads(value[:-cut] if cut else value)
        except Exception:
            continue
    return None


def clean_text(value: object, limit: int = 4000) -> str:
    text = HTML_RE.sub(" ", str(value or ""))
    return SPACE_RE.sub(" ", text).strip()[:limit]


def parse_tags(subjects: object) -> list[str]:
    if isinstance(subjects, list):
        text = "|".join(str(x) for x in subjects)
    else:
        text = str(subjects or "")
    return list(dict.fromkeys(TAG_RE.findall("|" + text)))


def common_category(headline: str, body: str, subjects: str) -> tuple[str | None, str | None, bool]:
    # Common news is deliberately high precision: a firm story that merely mentions
    # inflation, Nasdaq, dollars, or a foreign treasury is not a market-wide event.
    text = headline.strip()
    if re.match(r"^(?:BUZZ|DIARY)[-:]", text, re.I) or re.search(r"STOCKS ON THE MOVE", text, re.I):
        return None, None, False
    for name, pattern in MACRO_PATTERNS.items():
        if pattern.search(text):
            return "macro", name, True
    for family, pattern in [
        ("equities", US_EQUITY_RE), ("rates_credit", US_RATES_RE),
        ("rates_credit", US_CREDIT_RE), ("volatility", US_VOL_RE), ("fx", US_FX_RE),
    ]:
        if pattern.search(text):
            if family == "rates_credit" and re.search(r"\b(?:ETF|NET ASSET VALUE)\b", text, re.I):
                continue
            return "finance", family, True
    return None, None, False


def substantive_category(headline: str, body: str) -> str:
    text = headline + " " + body[:1200]
    for name, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return name
    return "other"


def canonical_headline(value: str) -> str:
    text = PREFIX_RE.sub("", value.strip())
    tokens = [x for x in WORD_RE.findall(text.lower()) if len(x) > 1]
    return " ".join(tokens[:20])


def prepare_market_maps(cfg: dict) -> dict:
    universe = pd.read_parquet(VOL_RESULTS / "universe.parquet").head(int(cfg["universe_size"])).copy()
    formation = pd.Timestamp(cfg["formation_date"])
    snapshot = ds.dataset(str(CRSP), format="parquet").to_table(
        columns=["permno", "date", "ff49", "sic2", "me"],
        filter=ds.field("date") == formation.date(),
    ).to_pandas()
    snapshot["me_rank"] = snapshot.me.rank(ascending=False, method="first")
    snapshot = snapshot.drop_duplicates("permno")
    ff49 = snapshot.set_index("permno").ff49.to_dict()
    me = snapshot.set_index("permno").me.to_dict()
    me_rank = snapshot.set_index("permno").me_rank.to_dict()

    names = pd.read_parquet(CRSP_NAMES)
    names = names[(names.namedt <= formation) & (formation <= names.nameenddt)].copy()
    counts = names.groupby(names.ticker.astype(str).str.upper()).permno.nunique()
    unique_tickers = set(counts[counts.eq(1)].index)
    names["ticker_key"] = names.ticker.astype(str).str.upper()
    names = names[names.ticker_key.isin(unique_tickers)].drop_duplicates("ticker_key")
    ticker_to_permno = dict(zip(names.ticker_key, names.permno.astype(int)))

    ric = pd.read_csv(RIC_MAP)
    ric = ric[ric.is_us.fillna(False)].dropna(subset=["ric", "ticker"]).copy()
    ric["ticker_key"] = ric.ticker.astype(str).str.upper()
    ric["permno"] = ric.ticker_key.map(ticker_to_permno)
    ric = ric[ric.permno.notna()].copy()
    ric["permno"] = ric.permno.astype(int)
    ric_to_permno = dict(zip(ric.ric.astype(str), ric.permno))

    target_permnos = set(universe.permno.astype(int))
    target_sectors = set(universe.ff49.dropna().astype(int))
    graph = pd.read_parquet(VOL_RESULTS / "graph_monthly.parquet")
    graph = graph[
        graph.focal_permno.isin(target_permnos)
        & graph.ym.isin([202210, 202211])
    ].copy()
    neighbor_permnos = set(graph.nbr_permno.astype(int))
    sector_permnos = set(snapshot.loc[snapshot.ff49.isin(target_sectors), "permno"].astype(int))
    relevant_permnos = target_permnos | neighbor_permnos | sector_permnos
    relevant_rics = {code for code, permno in ric_to_permno.items() if permno in relevant_permnos}

    aliases = defaultdict(list)
    corpus_names = pd.read_parquet(RIC_NAMES, columns=["ric", "name", "support", "share"])
    corpus_names = corpus_names[
        corpus_names.ric.astype(str).isin(relevant_rics)
        & corpus_names.share.ge(0.5)
        & corpus_names.name.notna()
    ].sort_values(["ric", "support"], ascending=[True, False])
    for row in corpus_names.itertuples():
        if len(str(row.name).strip()) >= 3 and len(aliases[str(row.ric)]) < 3:
            aliases[str(row.ric)].append(str(row.name).strip())
    for row in ric[ric.ric.astype(str).isin(relevant_rics)].itertuples():
        aliases[str(row.ric)].append(str(row.ticker).upper())

    return {
        "universe": universe,
        "snapshot": snapshot,
        "ff49": ff49,
        "me": me,
        "me_rank": me_rank,
        "ric_to_permno": ric_to_permno,
        "relevant_rics": relevant_rics,
        "aliases": dict(aliases),
        "graph": graph,
        "target_permnos": target_permnos,
        "target_sectors": target_sectors,
    }


def scan_reuters(cfg: dict, maps: dict, calendar: pd.DatetimeIndex, rebuild: bool, start: float) -> pd.DataFrame:
    path = RESULTS / "focused_versions.parquet"
    if path.exists() and not rebuild:
        frame = pd.read_parquet(path)
        log(start, "loaded focused Reuters cache", len(frame))
        return frame

    scan_start = pd.Timestamp(cfg["reflection_start"]) - pd.Timedelta(days=4)
    scan_end = pd.Timestamp(cfg["test_end"]) + pd.Timedelta(days=1)
    months = pd.period_range(scan_start, scan_end, freq="M")
    rows: list[dict] = []
    seen = 0
    for month in months:
        source = RAW_REUTERS / f"News.RTRS.{month.strftime('%Y%m')}.0214.txt"
        if not source.exists():
            raise FileNotFoundError(source)
        kept_before = len(rows)
        with source.open("rb") as handle:
            for raw_line in handle:
                if len(raw_line) < 60:
                    continue
                obj = parse_line(raw_line)
                if not obj:
                    continue
                seen += 1
                data = obj.get("data") or {}
                language = str(data.get("language") or "").lower()
                if language and language != "en":
                    continue
                first_created = data.get("firstCreated")
                version_created = data.get("versionCreated") or first_created
                stamp = str(version_created or "")[:10]
                if not stamp or stamp < scan_start.strftime("%Y-%m-%d") or stamp > scan_end.strftime("%Y-%m-%d"):
                    continue
                headline = clean_text(data.get("headline"), 400)
                body = clean_text(data.get("body"), 4000)
                subjects_list = data.get("subjects") or []
                subjects = "|".join(str(x) for x in subjects_list)
                rics = parse_tags(subjects_list)
                relevant = [code for code in rics if code in maps["relevant_rics"]]
                common_kind, common_family, us_relevant = common_category(headline, body, subjects)
                if not relevant and common_kind is None:
                    continue
                rows.append(
                    {
                        "alt_id": str(data.get("altId") or ""),
                        "guid": str(obj.get("guid") or ""),
                        "first_created": first_created,
                        "version_created": version_created,
                        "take_seq": data.get("takeSequence"),
                        "urgency": obj.get("urgency") if obj.get("urgency") is not None else data.get("urgency"),
                        "msg_type": data.get("messageType"),
                        "headline": headline,
                        "body": body,
                        "subjects": subjects,
                        "rics": relevant,
                        "common_kind": common_kind,
                        "common_family": common_family,
                        "us_relevant": us_relevant,
                    }
                )
        log(start, source.name, "scanned", seen, "kept", len(rows) - kept_before)

    frame = pd.DataFrame(rows)
    frame["first_ts"] = pd.to_datetime(frame.first_created, utc=True, errors="coerce")
    frame["version_ts"] = pd.to_datetime(frame.version_created, utc=True, errors="coerce").fillna(frame.first_ts)
    frame = frame[frame.version_ts.notna()].copy()
    frame["decision_date"] = assign_decision_date(frame.version_ts, calendar, int(cfg["cutoff_hour_et"]))
    wanted = set(experiment_dates(cfg, calendar))
    frame = frame[frame.decision_date.isin(wanted)].copy()
    frame["story_key"] = (
        frame.alt_id.astype(str)
        + "|"
        + frame.first_ts.dt.strftime("%Y-%m-%d").fillna(frame.version_ts.dt.strftime("%Y-%m-%d"))
    )
    frame["take_seq"] = pd.to_numeric(frame.take_seq, errors="coerce").fillna(-1)
    frame = frame.sort_values(["story_key", "decision_date", "version_ts", "take_seq"]).drop_duplicates(
        ["story_key", "decision_date"], keep="last"
    )
    cutoffs = pd.Series(
        cutoff_utc(pd.DatetimeIndex(frame.decision_date), int(cfg["cutoff_hour_et"])),
        index=frame.index,
    )
    if (frame.version_ts > cutoffs).any():
        raise AssertionError("Reuters version later than decision cutoff")
    frame.to_parquet(path, index=False)
    log(start, "focused final versions", len(frame), "stories", frame.story_key.nunique())
    return frame


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def mask_event_text(text: str, rics: list[str], aliases: dict[str, list[str]]) -> str:
    output = text
    terms = {term for ric in rics for term in aliases.get(ric, []) if len(term) >= 3}
    for term in sorted(terms, key=len, reverse=True):
        output = re.sub(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", "[FIRM]", output, flags=re.I)
    output = DATE_RE.sub("[DATE]", output)
    output = REF_RE.sub("[STORY]", output)
    return SPACE_RE.sub(" ", output).strip()


def build_events(cfg: dict, versions: pd.DataFrame, maps: dict, start: float) -> pd.DataFrame:
    frame = versions.copy().reset_index(drop=True)
    frame["canonical"] = frame.headline.map(canonical_headline)
    frame["firm_permnos"] = frame.rics.map(
        lambda values: list(dict.fromkeys(maps["ric_to_permno"][x] for x in values if x in maps["ric_to_permno"]))
    )
    frame["category"] = [
        common if common in {"macro", "finance"} else substantive_category(headline, body)
        for common, headline, body in zip(frame.common_kind, frame.headline, frame.body)
    ]
    frame["scheduled_key"] = None
    for index, row in frame.iterrows():
        if row.common_kind == "macro" and row.common_family:
            frame.at[index, "scheduled_key"] = f"macro:{row.common_family}:{row.decision_date.date()}"
        elif row.common_kind == "finance" and row.common_family:
            frame.at[index, "scheduled_key"] = f"finance:{row.common_family}:{row.decision_date.date()}"

    uf = UnionFind(len(frame))
    keyed: dict[tuple, int] = {}
    alt_to_index: dict[str, int] = {}
    for index, row in frame.iterrows():
        if pd.notna(row.scheduled_key) and row.scheduled_key:
            key = (row.decision_date, row.scheduled_key)
        else:
            entity_key = tuple(sorted(row.rics)[:3])
            key = (row.decision_date, row.category, row.canonical, entity_key)
        if key in keyed:
            uf.union(index, keyed[key])
        else:
            keyed[key] = index
        alt_to_index[str(row.alt_id)] = index
    for index, row in frame.iterrows():
        for reference in REF_RE.findall(row.body):
            if reference in alt_to_index:
                other = alt_to_index[reference]
                current_scheduled = pd.notna(row.scheduled_key) and bool(row.scheduled_key)
                other_scheduled = pd.notna(frame.at[other, "scheduled_key"]) and bool(frame.at[other, "scheduled_key"])
                if not current_scheduled and not other_scheduled and frame.at[other, "decision_date"] == row.decision_date:
                    uf.union(index, other)
    frame["cluster"] = [uf.find(i) for i in range(len(frame))]

    rows = []
    category_weight = {
        "macro": 3.0, "finance": 2.5, "operations": 2.5, "legal_policy": 2.0,
        "capital_financing": 2.0, "ma_ownership": 2.0, "earnings": 1.5,
        "product_contract": 1.0, "other": 0.0,
    }
    for (_, cluster), group in frame.groupby(["decision_date", "cluster"], sort=True):
        if group.common_kind.notna().any():
            ordered = group.assign(content_length=group.body.str.len()).sort_values(
                ["content_length", "version_ts"], ascending=[False, False]
            )
        else:
            ordered = group.sort_values(["urgency", "version_ts"], ascending=[True, False])
        representative = ordered.iloc[0]
        rics = list(dict.fromkeys(code for values in group.rics for code in values))
        permnos = list(dict.fromkeys(int(value) for values in group.firm_permnos for value in values))
        ff49s = sorted({int(maps["ff49"][value]) for value in permnos if pd.notna(maps["ff49"].get(value))})
        related_headlines = list(dict.fromkeys(group.headline.astype(str)))[:3]
        category = Counter(group.category).most_common(1)[0][0]
        urgency_values = pd.to_numeric(group.urgency, errors="coerce")
        urgency = int(urgency_values.min()) if urgency_values.notna().any() else 3
        urgency_points = 4.0 if urgency <= 1 else 2.0 if urgency == 2 else 0.0
        ranks = [maps["me_rank"].get(value, np.nan) for value in permnos]
        ranks = [float(x) for x in ranks if pd.notna(x)]
        prominence = (
            urgency_points
            + category_weight.get(category, 0.0)
            + math.log1p(group.story_key.nunique())
            + 0.7 * math.log1p(len(permnos))
            + (1.0 if ranks and min(ranks) <= 200 else 0.0)
        )
        headline = mask_event_text(str(representative.headline), rics, maps["aliases"])
        body = mask_event_text(str(representative.body), rics, maps["aliases"])
        event_id = "E" + stable_hash([str(representative.decision_date.date()), sorted(group.story_key)])[:11].upper()
        rows.append(
            {
                "event_id": event_id,
                "decision_date": representative.decision_date,
                "category": category,
                "common_family": representative.common_family,
                "us_relevant": bool(group.us_relevant.any()),
                "first_ts": group.version_ts.min(),
                "last_ts": group.version_ts.max(),
                "urgency": urgency,
                "n_stories": int(group.story_key.nunique()),
                "n_firms": len(permnos),
                "rics": rics,
                "firm_permnos": permnos,
                "ff49s": ff49s,
                "min_me_rank": min(ranks) if ranks else np.nan,
                "prominence": float(prominence),
                "headline_masked": headline,
                "related_headlines_masked": [
                    mask_event_text(value, rics, maps["aliases"]) for value in related_headlines
                ],
                "text_masked": body[: int(cfg["event_excerpt_chars"])],
                "story_keys": list(group.story_key),
            }
        )
    events = pd.DataFrame(rows).sort_values(["decision_date", "prominence", "last_ts"], ascending=[True, False, False])
    events.to_parquet(RESULTS / "events.parquet", index=False)
    log(start, "economic event cards", len(events), "days", events.decision_date.nunique())
    return events


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy().sort_values(["permno", "date"])
    for raw, logged in [
        ("rv1", "log_rv1"), ("rv5", "log_rv5"), ("rv22", "log_rv22"),
        ("mkt_rv1", "log_mkt_rv1"), ("mkt_rv5", "log_mkt_rv5"), ("mkt_rv22", "log_mkt_rv22"),
        ("ind_rv5", "log_ind_rv5"), ("ind_rv22", "log_ind_rv22"),
    ]:
        frame[logged] = np.log(pd.to_numeric(frame[raw], errors="coerce").clip(lower=1e-12))
    iv = frame.atm_iv_l1.where(frame.atm_iv_l1.notna(), frame.atm_iv_30_l1)
    frame["log_iv_variance"] = np.log((pd.to_numeric(iv, errors="coerce").pow(2) / 252).clip(lower=1e-12))
    frame["iv_change_l1"] = frame.groupby("permno").atm_iv_l1.diff()
    frame["iv_term_l1"] = frame.atm_iv_91_l1 - frame.atm_iv_30_l1
    frame["own_active"] = frame.own_n_articles.gt(0).astype(float)
    frame["network_active"] = frame.network_n_articles.gt(0).astype(float)
    for raw in ["own_n_articles", "own_news_words", "network_n_articles", "network_n_sources", "network_news_words"]:
        frame[f"log_{raw}"] = np.log1p(pd.to_numeric(frame[raw], errors="coerce").fillna(0))
    for raw in ["graph_degree", "graph_supplier_degree", "graph_customer_degree", "graph_revenue_links", "graph_comention_links"]:
        frame[f"log_{raw}"] = np.log1p(pd.to_numeric(frame[raw], errors="coerce").fillna(0))
    return frame


def estimator() -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]),
                STRONG_FEATURES,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", prep), ("ridge", Ridge(alpha=10.0))])


def purged_end(calendar: pd.DatetimeIndex, end: pd.Timestamp, horizon: int) -> pd.Timestamp:
    end_index = int(np.searchsorted(calendar.values, end.to_datetime64(), side="right") - 1)
    index = end_index - horizon + 1
    if index < 0:
        raise RuntimeError("Insufficient history for purge")
    return pd.Timestamp(calendar[index])


def build_targets(cfg: dict, start: float) -> tuple[pd.DataFrame, dict]:
    panel = prepare_panel(pd.read_parquet(VOL_RESULTS / "panel.parquet"))
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    score_days = experiment_dates(cfg, calendar)
    columns = STRONG_FEATURES + CATEGORICAL
    outputs, references, scales = [], [], {}
    for horizon in [int(x) for x in cfg["horizons"]]:
        target = f"log_q{horizon}"
        fold_predictions = []
        for month in pd.period_range(cfg["cdf_start"], cfg["cdf_end"], freq="M"):
            lo = max(pd.Timestamp(cfg["cdf_start"]), month.start_time)
            hi = min(pd.Timestamp(cfg["cdf_end"]), month.end_time)
            previous = calendar[calendar < lo][-1]
            train_end = purged_end(calendar, pd.Timestamp(previous), horizon)
            train = panel[panel.date.between(cfg["baseline_start"], train_end) & panel[target].notna()]
            test = panel[panel.date.between(lo, hi) & panel[target].notna()]
            model = estimator().fit(train[columns], train[target])
            fold = test[["permno", "date", target]].copy()
            fold["pred_log"] = model.predict(test[columns])
            fold_predictions.append(fold)
        calibration = pd.concat(fold_predictions, ignore_index=True)
        calibration["residual"] = calibration[target] - calibration.pred_log
        reference = calibration.residual.to_numpy(float)
        references.append(pd.DataFrame({"horizon": horizon, "residual": reference}))

        final_end = purged_end(calendar, pd.Timestamp(cfg["baseline_end"]), horizon)
        train = panel[panel.date.between(cfg["baseline_start"], final_end) & panel[target].notna()]
        score = panel[panel.date.isin(score_days) & panel[target].notna()].copy()
        model = estimator().fit(train[columns], train[target])
        prediction = model.predict(score[columns])
        residual = score[target].to_numpy(float) - prediction
        out = score[["permno", "firm_id", "date", f"q{horizon}", target]].copy()
        out = out.rename(columns={f"q{horizon}": "actual_q", target: "actual_log_q"})
        out["horizon"] = horizon
        out["pred_log"] = prediction
        out["pred_q"] = np.exp(prediction)
        out["residual"] = residual
        out["actual_percentile"] = empirical_percentile(residual, reference)
        outputs.append(out)
        quantiles = np.quantile(reference, [0.1, 0.25, 0.5, 0.75, 0.9])
        scales[str(horizon)] = {
            "percentiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            "log_variance_residuals": [float(x) for x in quantiles],
            "variance_multipliers": [float(math.exp(x)) for x in quantiles],
            "cdf_rows": len(reference),
            "final_model_train_end": str(final_end.date()),
        }
    targets = pd.concat(outputs, ignore_index=True)
    targets.to_parquet(RESULTS / "baseline_targets.parquet", index=False)
    pd.concat(references, ignore_index=True).to_parquet(RESULTS / "cdf_residuals.parquet", index=False)
    (RESULTS / "target_scale.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    log(start, "baseline targets", len(targets), "score dates", targets.date.nunique())
    return targets, scales


def exposure_bucket(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unreported"
    if value >= 20:
        return "20pct-plus"
    if value >= 5:
        return "5-to-20pct"
    return "below-5pct"


def build_packets(cfg: dict, events: pd.DataFrame, targets: pd.DataFrame, scales: dict, maps: dict, start: float) -> pd.DataFrame:
    universe = maps["universe"].set_index("permno")
    panel = pd.read_parquet(VOL_RESULTS / "panel.parquet").set_index(["permno", "date"])
    primary = targets[targets.horizon.eq(int(cfg["primary_horizon"]))].set_index(["permno", "date"])
    events_by_day = {day: group.copy() for day, group in events.groupby("decision_date")}
    graph = maps["graph"].copy()
    graph_lookup = {
        (int(ym), int(target)): group.copy()
        for (ym, target), group in graph.groupby(["ym", "focal_permno"])
    }
    rows = []
    for day in experiment_dates(cfg, pd.DatetimeIndex(sorted(pd.read_parquet(VOL_RESULTS / "panel.parquet", columns=["date"]).date.unique()))):
        day_events = events_by_day.get(day, events.iloc[0:0])
        common = day_events[
            day_events.category.isin(["macro", "finance"]) & day_events.us_relevant
        ].sort_values(["prominence", "last_ts"], ascending=[False, False]).head(int(cfg["max_common_events"]))
        common_ids = common.event_id.tolist()
        ym = int(day.strftime("%Y%m"))
        for permno in maps["universe"].sort_values("universe_rank").permno.astype(int):
            u = universe.loc[permno]
            state = panel.loc[(permno, day)]
            base = primary.loc[(permno, day)]
            graph_rows = graph_lookup.get((ym, permno), graph.iloc[0:0])
            neighbors = set(graph_rows.nbr_permno.astype(int))
            graph_by_neighbor = graph_rows.sort_values("rev_pct", ascending=False).drop_duplicates("nbr_permno").set_index("nbr_permno")

            firm_sets = day_events.firm_permnos.map(set)
            own_mask = firm_sets.map(lambda values: permno in values)
            own = day_events[
                own_mask & ~day_events.category.isin(["macro", "finance"])
            ].sort_values(["prominence", "last_ts"], ascending=[False, False]).head(int(cfg["max_own_events"]))
            network_mask = firm_sets.map(lambda values: bool(values & neighbors) and permno not in values)
            network_candidates = day_events[network_mask & ~day_events.category.isin(["macro", "finance"])].copy()
            network_items = []
            source_counts: Counter[int] = Counter()
            ranked_items = []
            for event in network_candidates.itertuples():
                sources = sorted(set(int(x) for x in event.firm_permnos) & neighbors)
                if not sources:
                    continue
                best = max(
                    sources,
                    key=lambda source: float(graph_by_neighbor.loc[source].rev_pct)
                    if source in graph_by_neighbor.index and pd.notna(graph_by_neighbor.loc[source].rev_pct)
                    else -1.0,
                )
                relation = str(graph_by_neighbor.loc[best].role) if best in graph_by_neighbor.index else "linked"
                rev = float(graph_by_neighbor.loc[best].rev_pct) if best in graph_by_neighbor.index and pd.notna(graph_by_neighbor.loc[best].rev_pct) else np.nan
                score = float(event.prominence) + (2.0 if pd.notna(rev) and rev >= 20 else 1.0 if pd.notna(rev) and rev >= 5 else 0.0)
                ranked_items.append((score, event.last_ts, event.event_id, best, relation, rev))
            ranked_items.sort(reverse=True)
            for _, _, event_id, source, relation, rev in ranked_items:
                if source_counts[source] >= int(cfg["max_network_events_per_source"]):
                    continue
                source_counts[source] += 1
                network_items.append(
                    {
                        "event_id": event_id,
                        "source_id": f"N{len(source_counts):02d}",
                        "relation_to_target": relation,
                        "reported_revenue_exposure": exposure_bucket(rev),
                    }
                )
                if len(network_items) >= int(cfg["max_network_events"]):
                    break
            network_ids = [item["event_id"] for item in network_items]

            target_ff49 = int(u.ff49)
            sector_mask = day_events.ff49s.map(lambda values: target_ff49 in set(values))
            sector = day_events[
                sector_mask
                & ~own_mask
                & ~network_mask
                & ~day_events.category.isin(["macro", "finance"])
            ].copy()
            sector = sector[
                sector.n_firms.ge(2)
                | sector.urgency.le(2)
                | sector.min_me_rank.le(200)
                | sector.category.isin(["operations", "legal_policy", "capital_financing", "ma_ownership"])
            ].sort_values(["prominence", "last_ts"], ascending=[False, False]).head(
                int(cfg["max_sector_events"])
            )
            sector_ids = sector.event_id.tolist()
            own_ids = own.event_id.tolist()
            all_ids = list(dict.fromkeys(common_ids + sector_ids + own_ids + network_ids))
            hist_vol = math.sqrt(max(float(state.rv22), 0.0) * 252) if pd.notna(state.rv22) else np.nan
            iv = state.atm_iv_l1 if pd.notna(state.atm_iv_l1) else state.atm_iv_30_l1
            case_id = "C" + stable_hash([cfg["run_id"], str(day.date()), permno])[:12].upper()
            payload = {
                "case_id": case_id,
                "target": {
                    "industry_code": f"FF49-{target_ff49:02d}",
                    "size_bucket": "mega" if int(u.universe_rank) <= 10 else "large",
                    "historical_annualized_volatility": round(hist_vol, 3) if np.isfinite(hist_vol) else None,
                    "lagged_option_iv": round(float(iv), 3) if pd.notna(iv) else None,
                    "conventional_forecast_annualized_volatility": round(math.sqrt(float(base.pred_q) * 252), 3),
                    "prior_session_return": round(float(state.ret1), 4) if pd.notna(state.ret1) else None,
                },
                "common_event_ids": common_ids,
                "sector_event_ids": sector_ids,
                "own_event_ids": own_ids,
                "network_events": network_items,
            }
            rows.append(
                {
                    "case_id": case_id,
                    "decision_date": day,
                    "permno": permno,
                    "firm_id": u.firm_id,
                    "base_active": bool(common_ids or sector_ids or own_ids),
                    "network_active": bool(network_ids),
                    "requires_llm": bool(all_ids),
                    "common_event_ids": json.dumps(common_ids),
                    "sector_event_ids": json.dumps(sector_ids),
                    "own_event_ids": json.dumps(own_ids),
                    "network_event_ids": json.dumps(network_ids),
                    "all_event_ids": json.dumps(all_ids),
                    "case_json": json.dumps(payload, separators=(",", ":")),
                }
            )
    cases = pd.DataFrame(rows).sort_values(["decision_date", "firm_id"])
    cases.to_parquet(RESULTS / "case_packets.parquet", index=False)
    log(start, "case packets", len(cases), "dates", cases.decision_date.nunique())
    return cases


def write_audit(cfg: dict, versions: pd.DataFrame, events: pd.DataFrame, cases: pd.DataFrame, targets: pd.DataFrame) -> None:
    def lengths(column: str) -> pd.Series:
        return cases[column].map(lambda value: len(json.loads(value)))

    audit = {
        "run_id": cfg["run_id"],
        "focused_final_versions": len(versions),
        "event_cards": len(events),
        "cases": len(cases),
        "dates": int(cases.decision_date.nunique()),
        "cases_by_period": {
            name: int(cases.decision_date.between(cfg[f"{name}_start"], cfg[f"{name}_end"]).sum())
            for name in ["reflection", "selection", "test"]
        },
        "event_count_distributions": {
            name: lengths(column).describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict()
            for name, column in [
                ("common", "common_event_ids"), ("sector", "sector_event_ids"),
                ("own", "own_event_ids"), ("network", "network_event_ids"),
            ]
        },
        "target_rows": len(targets),
        "config_hash": stable_hash(cfg),
    }
    if audit["cases_by_period"] != {"reflection": 150, "selection": 150, "test": 300}:
        raise AssertionError(f"Unexpected period sizes: {audit['cases_by_period']}")
    if cases.decision_date.between(cfg["embargo_start"], cfg["embargo_end"]).any():
        raise AssertionError("Embargo dates entered cases")
    (RESULTS / "build_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the event-collapsed news percentile pilot.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild-scan", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    started = time.time()
    panel_dates = pd.read_parquet(VOL_RESULTS / "panel.parquet", columns=["date"])
    calendar = pd.DatetimeIndex(sorted(panel_dates.date.unique()))
    maps = prepare_market_maps(cfg)
    log(started, "relevant RICs", len(maps["relevant_rics"]), "graph rows", len(maps["graph"]))
    versions = scan_reuters(cfg, maps, calendar, args.rebuild_scan, started)
    events = build_events(cfg, versions, maps, started)
    targets, scales = build_targets(cfg, started)
    cases = build_packets(cfg, events, targets, scales, maps, started)
    write_audit(cfg, versions, events, cases, targets)


if __name__ == "__main__":
    main()
