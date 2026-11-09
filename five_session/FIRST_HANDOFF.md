# TD1 first handoff: exploratory five-session benchmark

This is a local source inventory and an availability reconstruction for the single outcome `log(mean(r[t+1:t+5]^2))`. The researcher decision is the last entry in `C:/OptResearch/results/sc_risk/t7v3_20261028/model01/MODEL01_FREEZE.md`: fit 2015-2020, select 2021-2022, refit through 2022, freeze, and evaluate 2023-2024 once later. Historical evaluation is exploratory. TD1 makes no provider calls.

## Source inventory and exact paths

| Purpose | Exact local source or implementation | TD1 use |
| --- | --- | --- |
| Collaborator instructions | `C:/Users/Boxuan/.claude/uploads/c5308263-d41b-41ec-bcae-26e718b5a584/81e51e41-TODO.md` | Read instructions. |
| Mathematical companion | `C:/Users/Boxuan/.claude/uploads/c5308263-d41b-41ec-bcae-26e718b5a584/1219673d-TODO.pdf` | Read both pages using pypdf; same five-session log target. |
| Existing source configuration | `C:/OptResearch/code/sc_risk/config.py` | Inventory only; never imported its legacy loaders. |
| Historical firm-week panel | `C:/OptResearch/results/hist_panel_2015_2024/panel.parquet` | Read only `date,permno`; never project its cached `y`, earnings or graph fields. Metadata: 460,913 rows. |
| Panel construction and documentation | `C:/OptResearch/code/hist_panel_2015_2024/build_panel.py`; `C:/OptResearch/results/hist_panel_2015_2024/README.md` | Original weekly origins are the last exchange session in each week, ending 2024-12-20. Universe is inherited: prior-year largest graph-linked common stocks. |
| Original panel caches | `C:/OptResearch/results/hist_panel_2015_2024/cache/calendar.parquet`; `origins.parquet`, `returns_wide.parquet`, `states_at_origins.parquet`, `market.parquet`, `industry.parquet`, `garch.parquet`, `graph_wide.parquet`, `universe.json` in that same directory | Located only; not needed for TD1 decoding. `states_at_origins` contains old labels. |
| Daily total returns and size | `C:/OptResearch/data/macro/exp_crsp_daily_compact.parquet` | Fields `date,permno,ret,me,ff49,share_type`; retain ordinary common shares `share_type in {0,2}`. Date predicate <=2024-12-31 is applied before projected fields reach Python. |
| Quarterly balance-sheet characteristics | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/compustat_quarterly_full.csv` | `gvkey,datadate,atq,cheq,invtq,dlcq,dlttq`. This file lacks operating cash flow. |
| Smaller quarterly extract | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/compustat_quarterly.csv` | Header checked; also lacks operating cash flow. Not used in model. |
| Earnings/report dates | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/earnings_dates.csv` | `gvkey,datadate,rdq`. Realized report dates, not a publication-stamped forward schedule. |
| Firm identity link | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/gvkey_permno_link.csv` | Key-only undated mapping. Retain only globally one-to-one pairs; otherwise mark fundamentals missing. No dated CCM guarantee. |
| Cash-flow characteristic | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/jkp_full_chars.parquet` | `eom,permno,ocf_at,common,primary_sec`; `ocf_at` is the old catalog's operating cash flow / assets. Use six-month conservative lag after eom. |
| Existing accounting logic | `C:/OptResearch/code/sc_risk/firm_month_fin.py`; `C:/OptResearch/code/sc_risk/accounting.py`; `C:/OptResearch/code/sc_risk/t7v3/nc24_20261028.py`; `C:/OptResearch/code/sc_risk/t7v3/enrich_20261028.py` | Definitions and source leads only. These modules were read as text, not imported. |

The panel README explicitly identifies look-ahead in `earnings_confirmed`: realized future announcements stand in for schedules. TD1 does not reuse it. No 2025 row, return, article, or other field is returned to TD1 from a source decoder. Arrow can internally decode a compressed page spanning the date boundary; filtering excludes the disallowed rows before Python receives projected fields. CSV rows are rejected by dates before the other values are accessed.

## Old characteristic catalog

The exact catalog is `C:/OptResearch/results/sc_risk/intraday/news_conditioned_build_20261028/MENU_FROZEN.json`; its implementation is `C:/OptResearch/code/sc_risk/t7v3/nc24_20261028.py` (the `NAMES`, `PRIMITIVES`, `normalized_firms`, and `scores` definitions). This is an intraday mechanism catalog, not a fitted five-session model. Its names are:

`ann_size`, `resp_size`, `ann_cash`, `resp_cash`, `ann_cashflow`, `resp_cashflow`, `ann_leverage`, `resp_leverage`, `resp_inventory`, `shared_supplier_inventory`, `shared_supplier_cash`, `resp_volatility`, `pair_return_rank_gap`, `shared_supplier_count`, `shared_customer_count`, `pair_ra`, `pair_jaccard`, `shared_supplier_customer_count`, `resp_investment`, `resp_profitability`, `resp_supplier_count`, `resp_customer_count`, `pair_reciprocity`, `pair_local_clustering`.

The nine primitive columns are `me,cash_at,ocf_at,debt_at,inv_act,ivol_ff3_21d,ret_12_1,capx_at,gp_sale`. The old code ranks primitives within industry and computes pair/neighborhood summaries. TD1 does not import those rankings, graph support masks, 14 routing columns, or 98 intraday interaction slopes. The related seven-flag candidate-routing artifact is `C:/OptResearch/results/sc_risk/t7v3_20261028/combread/PRIOR_CANDIDATES.json`; implementation: `C:/OptResearch/code/sc_risk/t7v3/combread_prior.py` and `combread_core.py` in the same directory.

TD1's scalar fundamentals are `size=log(CRSP me*1e6)`, `cash=cheq/atq`, `cash_flow=JKP ocf_at`, `leverage=(dlcq+dlttq)/atq`, and `inventory=invtq/atq`. Inventory/assets differs from the old catalog's inventory/current-assets primitive; it is recorded explicitly rather than assigned the old name.

## Candidate graph/news definitions and clocks

These are future incremental-arm definitions, not TD1 MLP inputs or fitted effects. For characteristic `m`, a candidate is `Z_m=C_m*E_m`, with all timestamps strictly before the focal cutoff and a single aggregation rule frozen on development data. A concrete candidate `E_m` is the fraction of valid exact-shortest-hop partners with at least one relevant story during the previous seven calendar days; count distinct canonical story chains, use the latest eligible version before cutoff, and report the eligible-partner denominator and missing coverage separately. Count each partner once per hop. Do not substitute missing coverage with absence of exposure. Candidate relevance channels: size/attention; cash/liquidity disruption; cash flow/demand and cost interruption; leverage/financing pressure; inventory/physical-input interruption. These are proposed channels, not verified physical paths. H1, H2 and H3 must be reported separately; product-continuity evidence for intermediate nodes remains necessary.

| Source | Exact path / definition | Availability rule required for incremental work |
| --- | --- | --- |
| FactSet raw supply links | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/FactSetRevere/FactSetRevere/ent_scr_supply_chain.txt` | Original panel treats `START_DATE` as first publication, excludes competitor links and uses prior-month snapshots. Require publication/identity time < cutoff; do not infer physical continuity from reachability. |
| Dated mapping source | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/factset_to_permno.csv` | Use link interval active as of graph date; historical ambiguities require separate audit. |
| Graph implementation | `C:/OptResearch/code/hist_panel_2015_2024/build_panel.py` (`shells`); `C:/OptResearch/code/sc_risk/graph_ops.py`; `graph_clocks_audit.py` in that directory | Exact shortest shells, not overlapping walks; only dated edges. |
| Pre-existing graph snapshots | `D:/OptResearch_data/sc_risk/o5_exposure_tail/v2_rw/graphs/` | Source lead from t7v3 `common.py`; not read for TD1. |
| Reuters article source | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/articles.parquet` | Located only; no article text opened or copied. The public clone's `D:/OptResearch_data/external/supply-chain-news-risk/DATA_DICTIONARY.md` documents ordinary-news availability as `max(firstCreated,versionCreated,recorded timestamps)`, with at least one publication timestamp. Resolve latest versions before usability/tag/relevance filtering; a later deletion must not resurrect an older version. These generic clock rules are source leads; TD1 reads no article payload. |
| Body-free daily metadata | `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed/firm_daily_counts.parquet` | Counts cannot certify an intraday version clock; conservative prior-day use until clock is verified. Not used in TD1. |
| Story-clock adapter and index | `C:/OptResearch/code/sc_risk/t7v3/news_members.py` (`resolve_index`, `NewsIndex`); `C:/OptResearch/code/sc_risk/reuters_timestamps.py`; `D:/OptResearch_data/sc_risk/t7v3/news/firm_story_index/{YYYY-MM}.parquet`, declared by `D:/OptResearch_data/sc_risk/t7v3/news/NEWS_READY.json` | `NewsIndex` accepts `ts/available_ts/availability_ts/timestamp/time`; distinct story chains with timestamps < cutoff. `firstCreated` alone does not authorize later updates. Only the readiness metadata was read; no index payload was read here. |
| Old five-session candidate primitives | `C:/OptResearch/code/sc_risk/optimizer.py`; `library.py`; `daily.py`; `node_extra.py` in that directory | Own/neighbor log RV5, downside, volatility surprise, prior returns, news bursts; pre-cutoff only. Legacy graph/news earnings shares need the same schedule repair. |

## TD1 availability clocks and missingness

- Cutoff: the original weekly origin session at 23:59:59 New York, saved as a timezone-naive UTC timestamp. Daily fields are assumed available at 20:00 New York on that session. This convention includes origin-session data and predicts the following five sessions.
- Quarterly fields: valid `rdq>=datadate`, otherwise `datadate+120` calendar days, at 23:59:59 New York; a report on the cutoff day is excluded because equality is not strictly before cutoff. Fiscal data older than 550 calendar days is missing. Both debt components must be observed; missing debt is not zero.
- Cash flow: primary common JKP security only, `eom+6 calendar months`, at 23:59:59 New York, with a 550-day source-age cap. This is an explicitly conservative proxy clock, not a verified publication or revision history.
- Earnings: past reports only, with end-of-day conservative availability. `post_earnings` uses the prior five exchange sessions; `earnings_covered` uses 300. `earnings_scheduled` is missing because a timestamped advance schedule is unavailable.
- Own history: log realized variance over 1/5/22 sessions (feature floor 1e-8), compounded own returns over 1/5/22 sessions, mean downside over five sessions, log size. Market: value-weighted daily return using prior-session ME, then RV5/RV22. Industry: leave-self-out FF49 value-weighted daily return with lagged ME, then RV5. No graph or news features.
- Every feature has `__available_at`, `__missing`, and `__included`; null availability accompanies missing values. Eligibility requires full own 22-session history and positive size. Missing fundamentals remain as rows and receive training-fitted imputation plus missing indicators.
- Local Compustat/JKP extracts are not vintage archives. The manifest is an availability-lag reconstruction; revisions and undated identity links remain limitations. The original graph-conditioned universe is retained without using graph values as predictors.

## Existing outcomes, predictions, and baseline-loss code

| Artifact or function | Exact path | Relationship to TD1 |
| --- | --- | --- |
| Original next-five-session log outcome | `C:/OptResearch/results/hist_panel_2015_2024/panel.parquet`, column `y` | Located by schema only; never projected. TD1 recomputes development labels from CRSP, with all five returns and no target floor. |
| Matched log-target predictions | `D:/OptResearch_data/sc_risk/weekly_cocust_gates/predictions_5.parquet` | Existing `target,own_har` and incremental predictions; file existence checked, payload never read. |
| Later neighbor-study predictions | `C:/OptResearch/results/sc_risk/neighbor_factors_20261027/artifacts/weekly_predictions_5.parquet` | Existing five-session reproduction predictions; existence checked, payload never read. |
| Original log-target benchmark and loss | `C:/OptResearch/code/sc_risk/cc1_gates.py`: `forward_log_target`, `train_predict`, `paired_loss`; caller `C:/OptResearch/code/sc_risk/cc1_run.py` | TD1's old comparison follows CC1 primary own HAR: RV1/RV5/RV22, own downside, size, market RV5/RV22, industry RV5; OLS with intercept, median fill, scaling and numerical ridge 1e-6. |
| Reproduction loss checks | `C:/OptResearch/code/sc_risk/neighbor_verify_report_20261027.py`: `verify`; `C:/OptResearch/code/sc_risk/neighbor_inference_20261027.py` | Pairing and old own HAR provenance. Not executed. |
| Older volatility-level ladder | `C:/OptResearch/code/sc_risk/r2_ladder.py`: `fit_predict`, `dm` | Uses annualized volatility level `sqrt(exp(y)*252)`, GARCH and realized forward earnings flags. It has a different target and contains unsealed mixed-period readers; not executed or used as the primary log-target comparison. |

TD1 fits the old log-target own HAR on exactly the same development rows as the MLP, and reports paired 2021-2022 squared losses before refitting. This is a same-row reconstruction of the old primary specification, not a promise of bit-for-bit reproduction of cached features. Any 2023-2024 target/loss computation is confined to `evaluate.py` after independent hash-matching clearance. The active request is `D:/OptResearch_data/five_session/CLEARANCE_REQUEST_v2.json`, binding `D:/OptResearch_data/five_session/MODEL_FREEZE_v2.json`; neither is a clearance. Original freeze/request files are preserved; the amendment changes verification array layout and code hashes only, with identical data, model and predictions. See `TD1_HANDOFF.md` for commands.

All new executable code and documents are in `D:/OptResearch_data/external/supply-chain-news-risk/five_session/`. All private snapshots, manifests, fitted models, predictions and test receipts are in `D:/OptResearch_data/five_session/`. No article text, identifiers from observations, credentials or model outputs belong in the public code repository.
