# Data dictionary and variable lineage

## Scope and notation

The authoritative completed specification is `source/code/news_network_50_2025/`. Its saved input table is `results/news_network_50_2025/controls.parquet`. Let `t` be the origin exchange session, `r[i,t]` the stored decimal daily total return, `a[t]` the timestamp of the actual XNYS close, and `S[t]` the next five XNYS session dates. All logarithms are natural. A value of `0.01` for a return means one percent. Variances are in decimal-return-squared units and are not annualized.

The generated CSV/JSON dictionaries provide exact source functions and line references. The definitions below describe the implementation used, including imperfect proxies. They do not retrospectively repair the completed experiment.

## Source files and raw identifiers

| Local source, relative to its data root | Fields used | Role |
|---|---|---|
| `SOURCE/FactSetRevere/FactSetRevere/ent_scr_supply_chain.txt` | `ID`, `SUPPLIER_FACTSET_ENTITY_ID`, `CUSTOMER_FACTSET_ENTITY_ID`, `START_DATE`, `END_DATE`, `REVENUE_PCT`; `SOURCE_FACTSET_ENTITY_ID` exists but is not stored in the constructed supply edge | Pipe-delimited historical supplier-to-customer relationships |
| `SOURCE/FactSetRevere/FactSetRevere/ent_scr_relationships.txt` | `ID`, `REL_TYPE`, `SOURCE_FACTSET_ENTITY_ID`, `TARGET_FACTSET_ENTITY_ID`, `START_DATE`, `END_DATE` | Only `REL_TYPE == 'COMPETITOR'` is added from this table; its `REVENUE_PCT` is not used |
| `SOURCE/FactSetRevere/FactSetRevere/mapping.csv` | `FACTSET_ENTITY_ID`, `ENTITY_PROPER_NAME` | Entity names for identity aids, not dated operational profiles |
| `SOURCE/processed/factset_to_permno.csv` | `FACTSET_ENTITY_ID`, `PERMNO`, `link_bdate`, `link_edate`; other observed columns are inventoried | Dated entity-to-security link |
| `SOURCE/processed/ric_to_ticker_map.csv` | `ric`, `ticker`, `is_us` | US Reuters RIC-to-ticker bridge, then dated CRSP names map ticker to `permno` |
| `SOURCE/processed/crsp_daily.parquet`, `crsp_daily_2025.parquet` | `permno`, `date`, `ret`, `prc`, `shrout`, `vol` | Daily total returns, prices, share counts, and trading volume |
| `ROOT/data/macro/exp_crsp_daily_compact.parquet` | `permno`, `date`, `ret`, `prc` | Fills security-session rows absent from the raw-precision source |
| `ROOT/data/gn/gn_crsp_names.parquet` | `permno`, `ticker`, `namedt`, `nameenddt`, `comnam` | Dated security identity and company names |
| `SOURCE/Reuters/STORY.RTRS.2025-MM.REC.JSON.txt` | Nested Reuters fields listed below | Raw story versions and diary calendars |
| `ROOT/data/macro/cal_trading_days.parquet`, plus `exchange_calendars` XNYS schedule | `date`; session opens/closes from XNYS | Return-calendar validation, origin and future-session clocks |

`SOURCE` denotes the local ReutersNewstTraining data directory and `ROOT` the OptResearch directory. Exact local locations and metadata are recorded in `dictionaries/observed_schemas.json`. No raw data rows are included.

### FactSet relationship direction and weights

The supply edge goes from `SUPPLIER_FACTSET_ENTITY_ID` to `CUSTOMER_FACTSET_ENTITY_ID`. For a focal supplier, the neighbor role is customer; for a focal customer, it is supplier. In the directly reported relationships table, source is the reporting entity and target is the named related entity. The implementation exposes competitor links in both traversal directions; that is a modeling convention, not a claim that both companies independently reported the relationship.

The locally available *FactSet Standard DataFeed Supply Chain Relationships V1.4.3 UserGuide* explains that customer revenue percentage is the share of the supplier's revenue obtained from that customer. It is not the customer's procurement share or input dependency. The raw percentage scale is retained in the graph and used to rank paths; it is not converted to a causal exposure weight. Existing agent capabilities strip these numerical weights, and percentages are never multiplied along paths. The table-specific revenue field in directly reported relationships applies to reported customer revenue when disclosed. The exact guide pages and file hash are recorded by the builder when the guide can be parsed; the guide itself is excluded.

The active-record rule is `START_DATE <= 2025-06-06 23:59:59` and `(END_DATE is null or END_DATE > that timestamp)`. The same rule is applied to mapping intervals `link_bdate/link_edate`. The graph is held fixed through December 31 by assumption. The guide describes publication-history dates and also notes exceptional publication discrepancies; effective-date reconstruction is not a certified archived delivery vintage.

`Graph.paths` keeps direct edges and balances higher-hop path types/intermediaries, with a default higher-hop limit of 1,500 per depth. Path identities, direction, and intermediary steps are retained. Generic same-industry or co-mention adjacency is not proof of a supplier/customer/competitor transmission mechanism.

### Reuters raw fields and derived records

| Raw field | Implemented use |
|---|---|
| `data.altId`, fallback `data.id`, fallback top-level `guid` | `story_id`; required identity for superseding versions |
| `data.firstCreated`, `data.versionCreated` | Publication/version timestamps; UTC-aware parsing |
| Top-level `timestamps[]` entries with `name == 'recorded'` | Their `timestamp` values also enter availability |
| `data.takeSequence` | Integer sequence, default zero |
| `data.pubStatus` | Default `stat:usable`; unusable latest versions are not replaced by an older usable story |
| `data.provider`, `data.language` | Default provider `NS:RTRS`; usable news requires Reuters and language `en` or missing |
| `data.subjects` | Native RIC/entity tags and `N2:` topic tags; topic codes are metadata, not inferred firm mentions |
| `data.headline`, `data.body` | Ranking/classification and full article inspection |

For ordinary news, `available_at = max(firstCreated, versionCreated, recorded timestamps)`, over supplied valid timestamps. At least one publication timestamp is required. Only records with `available_at < a[t]` are eligible. Latest versions are resolved across months before usability, firm-tag, and relevance filtering. The ranking key is `(version_created, take_sequence, available_at, source_id)`. A later deletion or loss of a firm tag cannot resurrect an older version.

The diary parser additionally recognizes `data.ts` and `data.version` as timestamp/version fallbacks. It preserves the source object for diary receipts locally. Those objects are not in this sharing package.

`source_id` is `RTRS:{YYYY-MM}:{raw line number}:{first 12 hex characters of raw-line SHA256}`. Full raw hashes, source month/line, byte offset, and byte length support direct source verification. These are provenance locators, not permanent vendor story identifiers.

`NewsMapper.for_date` filters the RIC mapping to US entries and uses CRSP name intervals `namedt <= date <= nameenddt`. Ambiguous ticker-to-PERMNO or RIC-to-PERMNO mappings are omitted. Date is the New York date of story availability. A tag is a retrieval lead; it does not establish that the article's economic event concerns the target.

### Daily return handling

`ret` is used as the stored CRSP decimal total return. The project does not rebuild it from prices or subtract a risk-free return. There is no separate delisting-return adjustment in these functions. Upstream extraction/revision/delisting conventions must be checked before claiming stronger coverage guarantees.

Raw-precision rows are preferred; compact rows are added only for absent `(permno,date)` keys. A missing raw return is not filled using compact data. Finite overlapping returns must agree within `rtol=1e-6, atol=1e-8`. Duplicates fail. Missing/invalid future returns fail rather than being silently zero-filled. In GARCH estimation, finite history observations are retained after minimum-length/staleness checks, as implemented in `garch_forecast`.

`abs(prc)` gives the displayed price magnitude; a negative CRSP price encoding is not a negative dollar price. Formation market equity is `abs(prc)*shrout*1000`; the multiplier reflects CRSP shares-outstanding units. Trading dollar volume is `abs(prc)*vol`. These formation quantities are not regressors in the completed expanded specification.

## Response, benchmark, and exact covariates

| Variable | Exact formula or rule | Timing and caveat |
|---|---|---|
| `y` | `log(mean(r[i,s]**2 for s in S[t]))` | Exactly five finite future returns; all-zero windows fail; no response floor |
| `b` | `log(max(mean(hat_h[1:5]),1e-12))` | Zero-mean Gaussian GARCH(1,1), latest 504 sessions through origin, at least 400 finite observations; percent-input variance divided by 10,000 |
| `rv1` | `log(max(r[i,t]**2,1e-12))` | Includes the origin return |
| `rv5` | `log(max(mean(last 5 r**2 through t),1e-12))` | Five finite observations required |
| `rv22` | `log(max(mean(last 22 r**2 through t),1e-12))` | Twenty-two finite observations required |
| `earnings_confirmed` | `int(calendar.status == 'confirmed_in_window')` | Source-backed earnings release whose possible reaction sessions all fall in the target window |
| `calendar_unknown` | `int(calendar.status == 'unknown')` | Missing/ambiguous schedule evidence, not a statement that no earnings occur |
| `post_earnings` | Indicator that a recent own headline matches `(?:reports?\|posts?) .*(?:quarter\|results\|earnings)\|quarter.*(?:profit\|revenue)` (regex alternatives are `|` in code) | Case-insensitive headline proxy, not a verified post-release trading-day variable |
| `pending_cash_deal` | Indicator that one of the retained historical own headlines matches `(?:acquir\|buy\|takeover\|acquisition\|merger).*\b(?:cash\|agreement\|agrees)\b\|all-cash` (alternatives are `|` in code) | Historical own-event list is at most 12 latest usable matching stories since July; not proof of cash consideration or still-pending state |
| `log_news_count` | `log(1 + len(recent))` | `recent`: eligible own-tagged, non-excluded latest stories with age strictly below seven calendar days |
| `log_neighbor_count` | `log(1 + sum(age < 7 for net_idx))` | Target excluded, common stories excluded, linked PERMNO tag and substantive-headline filter required; counts before the 20-story shortlist cap |
| `log_degree` | `log(1 + number of unique mapped PERMNOs in the stored relation candidates)` | Combines available graph paths up to two hops and frozen co-mention leads, not direct supplier degree alone |

The JSON dictionary stores the regexes without Markdown table escaping. All story counts refer to latest surviving story records before headline-family shortlist deduplication; they are not counts of all Reuters versions, inspected sources, or gold-standard events. Their seven-day window is a calendar-time window, not seven exchange sessions. Age is measured from `available_at`, and ordinary candidate news must also be within the 31-calendar-day eligible pool.

`b` is computed with `arch.arch_model(recent*100, mean='Zero', vol='GARCH', p=1, o=0, q=1, dist='normal', rescale=False)`. The five-step analytic forecast is averaged and divided by 10,000. A deterministic numerical retry standardizes the same returns, refits the same model with `maxiter=3000, ftol=1e-9`, and rescales by the return standard deviation squared. It does not switch to another economic model. The exact rolling recursion is `h[t] = omega + alpha*r[t-1]**2 + beta*h[t-1]`.

The conventional anchor is pooled OLS on an intercept and the eleven named controls, estimated using the 200 training rows only. The historical design includes rank redundancy: `calendar_unknown` complements `earnings_confirmed` in the saved sample and `pending_cash_deal` is always zero. Least squares uses the implementation's pseudoinverse/least-squares routine; no ridge penalty is used here. The generated verification file records these facts.

`score` is the LLM estimate of conditional future log risk minus the supplied anchor. Direct log-risk prediction is `anchor + score`; `exp(score)` is a geometric variance ratio and `exp(score/2)` a geometric RMS ratio. There is no extra fitted score slope in this direct forecast. Separate explanatory regressions fit `y = intercept + X*theta + gamma*score + error` on the displayed evaluation rows; their R-squared is not a frozen-coefficient forecasting result.

## Earnings calendar construction

Calendar receipts come from eligible Reuters DIARY news, including earlier months. The parser recognizes explicit dated table rows with a supported target RIC suffix (`'', O, OQ, N`) and an earnings/results release description. It extracts `ticker`, `ric`, `event_date`, `period`, `time_text`, `timezone`, `action`, exact source span, and receipt identity. It does not infer a date from the firm's historical reporting season.

Receipt versions, withdrawals, conflicts, and undated replacements are resolved conservatively. After-market-close (`AMC`) releases map to the next close-to-close return; before-market-open (`BMO`) releases map to the same session. Exact clock times require a recognized zone. Unknown/DBH/unzoned times span the possible day and remain unknown if the interval crosses an origin or target-window boundary. An already-passed planned time does not prove that the release actually occurred. A wholly future interval whose possible first-reaction sessions all lie in `S[t]` gives `confirmed_in_window`. Observed schedules outside the window can give `known_outside_window`, but do not establish that there is no other in-window event.

For the completed 50-firm run, diary observations are filtered to strict pre-cutoff availability before `feature_per_case`. Undated evidence can mark coverage uncertain rather than certify a future event. Calendar output includes `status`, `earnings_in_window`, `confirmed_release_count`, `events`, and `source_versions`; absence is never represented as a verified negative earnings event.

## Graph and co-mention construction

Co-mention leads use only July 2025 latest usable stories resolved before August 1 UTC. Retain stories with between two and twenty distinct mapped PERMNOs. Let `N[i,j]` be pair story count and `N[i]` the number of retained multi-tag stories containing `i`. Weight is `N[i,j]/sqrt(N[i]*N[j])`; each firm's top twenty neighbors are retained by weight, count, then identifier. This is attention overlap, not verified economic exposure.

Relation candidates for the 50-firm run are `Graph.paths(ticker, 2)` plus those co-mention neighbors. An unmapped foreign/private entity can remain in the graph but cannot retrieve PERMNO-tagged news through this bridge. No separate first/first-plus-second/first-plus-second-plus-third topology forecasting comparison is represented by this package.

## Later corrected December specification

`run_network30_frozen_2025.py` removes `calendar_unknown` and replaces `pending_cash_deal` with `recent_ma_news_31d`. The latter tests for a target-tagged, pre-cutoff, retained historical story in the previous 31 days matching `\b(?:acquisition|merger|takeover|buyout|to buy|to acquire|agrees? to buy)\b`, case-insensitively. It is an M&A-news proxy, not proof of a pending fixed-cash transaction. The anchor is reestimated on the same 200 training rows using the revised controls.

The original paused grounded MCP server could read the global 50-firm anchor while the December prompt used the revised anchor. The separate `EvidenceToolkit` requires an explicit matching `forecast_anchor` to prevent that mismatch. It adds financial-history receipts, conservative deal-term calculations, and typed network-channel prerequisites. These repairs are not substituted into the completed 50-firm definitions or results. No paid rerun is launched by this package.

## Availability and remaining upstream gaps

The builder exports the exact twelve-origin schedule from the saved numerical manifest, including actual exchange closes and next-session opens used for label maturity. Raw return vintages and historical vendor delivery vintages are not certified. RIC and FactSet-to-PERMNO bridges are project-generated mappings; their upstream production scripts are not part of the tool server's search access. The dataset dictionaries describe the actual local inputs, not every possible field in commercial Reuters, LSEG, FactSet, or CRSP products. Official vendor manuals other than the identified FactSet guide were not supplied with this package; fields without verified vendor definitions are explicitly marked as such.
