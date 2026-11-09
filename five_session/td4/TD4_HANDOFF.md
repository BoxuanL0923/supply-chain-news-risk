# TD4 handoff: provider selectors on the frozen five-session head (2024 selection, 2025 test)

Written 2026-11-06T17:02:49Z (from `date -u`). Builder: Claude subagent. Status: **built, tested offline, frozen as a
request; not cleared.** No DeepSeek or Jev call was made, no clearance file was written, and no 2023, 2024 or 2025
outcome was read. Figures below are counts, hashes, estimates or engineering checks; none is a forecasting result.

Code: this folder (`five_session/td4/`, public clone; code and Markdown only, no data, keys or article text). Private
inputs, outputs and receipts: `D:/OptResearch_data/five_session/td4/`. Diffs of every copied function against its
original: `TD4_COPIED_FUNCTIONS.md` (this folder).

## 1. Scope (researcher's amendment of 2026-11-06)

- Selector calls for 2024 (selection) and 2025 (confirmatory test) only; 2024 runs first, then 2025.
- 2025 data are used for this experiment only. Every label window ends by 2025-12-31; nothing dated 2026 or later is
  decoded or used (the CRSP daily, JKP and Reuters files hold no row after 2025-12-31 per their row-group statistics;
  CSV rows dated after 2025-12-31 are skipped once the date field is located; link ends are censored at the seal).
- Hyperparameters frozen from TD1 (MLP) and TD2 (menu, exposure dictionary, ridge penalties, aggregation rule).
  Selection models are refitted through 2023, test models through 2024, with those hyperparameters; nothing is re-tuned.
- Pre-registered before any 2025 outcome: `PREREGISTRATION_v1.json` (sha256 `56415d4e...94f86`), bound in the freeze
  and required by the evaluator. **P1**: best pipeline arm chosen on 2024 (selected DeepSeek, Jev Choice, Jev Noul or
  all-on) versus the firm-only MLP. **P2**: better Jev arm on 2024 versus matched-input DeepSeek with the same head.
  Paired squared-error gain, two-way clustered by firm and calendar week (TD2 `statistics.clustered_mean`), two-sided
  normal 95% interval; confirmed only if the interval lies above zero. Everything else is descriptive. The evaluator
  prints the TODO stop-rule readings verbatim from the pre-registration.
- 2024 results are selection-only. The 2025 result is CONFIRMATORY for this pipeline, with two pre-registered caveats:
  (a) the TODO says the earlier risk study had examined 2023-2025; (b) both providers' models were trained after
  2025, so masked articles may still be recognised.

## 2. What was built

| File | Role |
| --- | --- |
| `run_td4.py` | CLI; installs the TD3 v6 runtime contract before scientific imports (job memory cap 1.4e9 bytes; audit hook: offline = no socket, online = DNS for api.deepseek.com and api.typesafe.ai only; exclusive writes under the TD4 private root; no delete or rename; only `date -u` subprocesses) |
| `td4_spec.py` | Constants, rules, pre-registration text (hashed) |
| `td4_env.py` | Import paths (TD4, TD3 v6, TD2 appended last), runtime install, D: >= 25 GiB and C: >= 15 GiB guards, pause below 1,500 MB available memory, loaders for TD1 `spec`/`core` and TD2 `statistics` under private names (module-name clashes) |
| `td4_inventory.py` | 2025 coverage of every local source (counts only) |
| `td4_sources.py` | 2025 extracts with the TD1/TD2 rules and a seal parameter (CRSP/JKP snapshots, Compustat/earnings re-sealed, October-2025 graph and links, 2025 news scores, 2025 panel keys) |
| `td4_manifest.py` | TD1 firm-week feature rows for 2025 (copies of TD1 `accounting`, `common_all`, `build_year`; no label) |
| `td4_features.py` | TD2 E, Z and witness evidence for 2025 (TD2 primitives unchanged; copies of `mapped_edges`, `origin_rows`) |
| `td4_inputs.py` | Selector inputs for every eligible 2024/2025 firm-week with the cleared TD3 v6 functions; chunked, resumable; audit sample; `callable_inputs` |
| `td4_repair.py` | D8 repair pass for story-field withholdings (section 5) |
| `td4_verify.py` | Text-level population checks over every input; writes the population withheld list |
| `td4_parity.py` | Parity of every copy with the frozen TD1/TD2/TD3 outputs |
| `td4_gate.py` | Clearance A/B validation, admission tokens, once-only latch |
| `td4_labels.py` | Labels behind tokens (TD1 rule; clamp at the 2025 seal) |
| `td4_models.py` | Refits through 2023/2024 with frozen TD1/TD2 hyperparameters; 2022 parity |
| `td4_runner.py` | Paid calls: keyed idempotent ledger, balance floor, off-peak DeepSeek, Jev 429 backoff, caps, concurrency, re-ask |
| `td4_gates.py`, `td4_score.py` | Provider gates and shared-head predictions; scoring frames (identical rows for every arm) |
| `td4_select.py` | 2024 selection -> `selection/SELECTION_FREEZE_v1.json`; clearance-B request |
| `td4_evaluate.py` | Once-only 2025 evaluation (P1, P2, all arms, references, H1/H2/H3, coverage, cost) |
| `td4_diagnostics.py` | Re-ask stability; blinded human-labeling packet; scorer (NDCG, top-one, Jev calibration) |
| `td4_cost.py` | Cost and wall-time projection from counts and the pilot's measured tokens; caps |
| `td4_freeze.py` | Pre-registration, aggregation freeze, freeze A, clearance-A request, qualification summary |
| `td4_rehearsal.py` | Sandboxed dress rehearsal of every real step on 2021/2022 (development labels, mock providers) |
| `td4_tests.py` | Offline tests (synthetic fixtures, mocked providers) |

## 3. Sources and the 2025 rows (amendment item 4)

Inventory: `inventory/SOURCE_INVENTORY_2025_v1.json` (counts by year from metadata and date fields only).

| Source | 2025 coverage | Use | Point-in-time status |
| --- | --- | --- | --- |
| CRSP daily compact | 250 sessions to 2025-12-31; 5,162 PERMNOs (share_type 0/2) | features, universe, labels | session data; labels only behind tokens |
| JKP characteristics | ocf_at for 55,723 firm-months | cash_flow (month end + 6 months) | latest extract with availability lag (as TD1) |
| Compustat quarterly | 51,118 rows with 2025 fiscal dates | cash, leverage, inventory | latest extract, not revision-free (as TD1) |
| Earnings dates | 28,323 rdq in 2025 | release clock, earnings flags | realized dates (as TD1) |
| Reuters archive | 636,702 articles in 2025 | news scores, article text | day stamps, available next New York midnight (TD2) |
| FactSet relationships, October 2025 snapshot | 174,832 direct CUSTOMER/SUPPLIER versions start in 2025 | graph | **not point-in-time**: vendor record dates; nothing published after 2025-10-02, so late-2025 cutoffs see a stale graph; backdated 2025 additions cannot be measured (FSB3 found about 11% for the 2023-2024 cohorts in a later copy) |
| factset_to_permno links | to 2025-12-31 (active links end-coded 2025-12) | entity to PERMNO | extract built after 2025; ends censored at the seal |
| CRSP name files | **end 2024-12-31** | masking names only | 2025 renames not covered (WRDS query, section 10) |
| FactSet FSB1 names | end 2024-12-31 | masking names only | as TD3 |
| Historical panel keys | end 2024-12-20 | 2015-2024 universe | 2025 keys rebuilt (below) |
| Reuters RIC identity | frozen development mapping, keys <= 2022 | article tags | TD2 amendment A4, unchanged |

2025 rows built with the TD1/TD2 rules (all hashes are in the freeze):
- Panel keys (`panel_keys/panel_keys_2025.parquet`): `build_panel.py` rules: 1,000 largest share_type 0/2 stocks by
  market equity at 2024-12-31 with an active FactSet edge that day; monthly roots; GARCH(1,1) fit success on the 504
  sessions to 2024-12-31 (925 of 1,000); origins = last session of each W-SUN week of 2025 whose five following sessions
  end by 2025-12-31 (2025-01-03 to 2025-12-19, 51 origins); 46,514 keys. The panel's label filter (y finite and
  > -20) reads future returns and is applied at evaluation to every arm alike.
- TD1 manifest 2025: 46,514 rows, 46,469 with eligible features; fundamentals coverage cash 85.2%, cash_flow 99.5%,
  leverage 78.5%, inventory 81.9% (2024: 85.3%, 99.5%, 79.1%, 82.0%).
- TD2 features 2025: 46,469 primary rows, 43,147 with news (2024: 48,993 and 45,942); rows with a selected partner
  H1 34,678, H2 42,865, H3 43,140.

Selector inputs (TD3 eligibility: news, mapped focal, a non-missing characteristic, a verified partner article):

| | 2024 | 2025 |
| --- | --- | --- |
| Eligible firm-weeks (TD2 call unit) | 45,942 | 43,147 |
| Built in the main build | 45,417 | 40,109 |
| Withheld at build (TD3 residual checks) | 525 (522 story field, 3 strict) | 3,038 (all date/boilerplate; 3,036 story field only) |
| Rebuilt by the D8 repair pass | 522 of 522 | 3,036 of 3,036 |
| Withheld by the population checks | 44 (token-hyphen-digit) | 459 (196 token-hyphen-digit, 263 numeric date) |
| **Inputs sent to the providers** | **45,895** | **42,686** |
| Not called (row takes the arm's fallback) | 47 | 461 |
| Mean input size | 14,796 bytes | 14,027 bytes |
| Articles in the inputs (main + repair) | 570,166 | 531,930 |

All three selectors receive byte-identical texts per firm-week. Key hashes: `inputs/period_2024/INPUTS.jsonl`
`a71f515b...`, `inputs/repair_2024/INPUTS.jsonl` `5501dad2...`, `inputs/period_2025/INPUTS.jsonl` `c283d32a...`,
`inputs/repair_2025/INPUTS.jsonl` `8d5fcc0a...` (full list in the freeze).

## 4. Parity: the copies reproduce the frozen TD1/TD2/TD3 outputs

Receipts: `parity/PARITY_<check>_<tag>.json`. Every check is exact (0 differing values) unless stated.

| Check | Compared | Result |
| --- | --- | --- |
| graph2024 | graph extract from the raw October file, seal 2024-12-31, vs TD2 `graph_october_sealed.parquet` | 1,769,950 rows, 8 columns identical |
| links2024 | links extract vs TD2 `factset_links.parquet` | 60,448 rows identical |
| td1snap2024 | Compustat, earnings, CRSP 2024, JKP 2024 extracts vs TD1 `inputs/` | identical |
| news2024 | 2024 news scores vs TD2 `news_scores/year_2024.parquet` | 83,567 rows identical |
| panel2022 | universe + roots + GARCH + panel label filter (development year) vs hist panel 2022 | identical on the reconstructed origins (44,215 keys); the panel's two final 2022 origins are dropped by the year-end window rule required for 2025 |
| panel2024 | 2024 keys before the label filter vs the panel | contains every panel key; 86 extra = rows the panel's label filter dropped (unchecked: needs 2024 returns) |
| manifest2024 | TD4 copy of TD1 `build_year`, seal 2024-12-31, vs TD1 `year_2024.parquet` | 48,993 rows, every column identical |
| features2024, features2024_td4src | TD4 copies on six 2024 origins with TD2's sources and with the 2025-sealed TD4 sources vs TD2 `features_v2`/`evidence_v2` | 11,518 feature and 135,236 evidence rows identical, both |
| inputs_pilot | the 200 TD3 pilot firm-weeks rebuilt by the final `td4_inputs` (2021-2022 parameters, four chunks) vs TD3 `build_v7/INPUTS.jsonl` | 200 of 200 texts byte-identical |
| rebuild2024, rebuild2025 | first 150 eligible firm-weeks rebuilt by the final code vs the full builds | 150/150 and 52/52 identical (98 of the first 150 2025 firm-weeks are story-field withholdings) |
| refit2022 | TD4 refit through 2022 with TD1's refit rows vs TD1 `model_frozen_v2.pkl` and TD2 `MODEL_v2.json` | MLP predictions 2022-2024 identical (max difference 0.0); main, 1r, 2r, 3r betas identical |

## 5. Deviations from TD2/TD3 and why (each isolated and flagged for review)

1. **Period parameters (TD3).** TD3 hard-codes 2021-2022. `td4_inputs.set_period` sets td3_spec YEARS, FIRST_DAY,
   LAST_DAY, LAST_MONTH, NEWS_YEARS, ARTICLE_LO per period and points the Sources object at the period's files.
   Masking name sources are rows dated on or before the period end, as in TD3.
2. **Copied functions with a seal parameter.** TD3 `eligible`, `edge_maps`, `select_sample` (per-chunk part); TD2
   `fast_builder.mapped_edges`, `build_v2.origin_rows`, `decoder_v3.mapping`/`build_news`/`parquet`; TD1
   `sources.dated`/`csv_dated`, `build.accounting`/`common_all`/`build_year`, `core.label` (2024 clamp), `fit.model`;
   hist-panel `mapping_at`, `active_edges`, GARCH inclusion. Diffs in `TD4_COPIED_FUNCTIONS.md`; parity in section 4.
3. **Population instead of a sample (TD3).** Every eligible firm-week in (cutoff, firm_id) order, chunks of 1,500,
   resumable by chunk. Archive texts come from a period article store (archive rows of the period's evidence article
   IDs, physical order kept), so `fetch_texts`' first-physical-row rule is unchanged.
4. **Performance and memory shims (outputs unchanged; pilot and rebuild parity).** `aliases_for` memoised per PERMNO
   (bounded LRU 4,000; comnams_after_origin recomputed per origin); `td3_mask.variant_regex` memoised; larger Python
   regex cache; name sources per chunk (an alias reads only the firm's own rows, issuer siblings and near-name entries
   keyed by its own anchors); pure caches emptied between chunks (added after the 2024 build started; used by the 2025
   build and the repair builds). About 0.37 s per input.
5. **Residuals withhold a firm-week instead of aborting the build (TD3).** `build_inputs` runs per firm-week; an input
   with a strict or date residual is withheld from all three providers alike (`WITHHELD.jsonl`) and its row takes the
   arm's fallback.
6. **Population checks (TD3 verify, text level).** Every input is checked for years, URLs, numeric dates, story fields,
   glued role tokens and token-hyphen-digit remnants (TD3C4 class D2), the frozen header and the article count. 503
   inputs failed (mostly "[token]-140"-style codes and numeric dates) and are withheld from all providers alike
   (`WITHHELD_POPULATION_<year>.json`); the runner and the scorer never call or use them.
7. **Audit sample instead of manual reads (TD3).** TD3 read all 200 inputs and adjudicated every scanner hit; that
   does not scale to 88,581 inputs. TD4 keeps full TD3 metadata for a hash-ranked audit sample (300 per main build;
   all 522 / 3,036 repaired inputs, of which 522 / 300 audited), re-masks it from raw text (identical in every case),
   re-runs the strict residual (0) and the independent TD3 scanner, and writes the scanner hits for the reviewer
   (2024 main 296 inputs, 385 hits; 2024 repair 522, 542; 2025 main 282, 313; 2025 repair 300, 249). No LLM labels them.
8. **D8 repair of unbracketed story fields (new pre-mask rule).** TD3 strips `STORY_TIME:[...]` and rejects any other
   story field. A few articles carry the unbracketed form (`STORY_TIME: hh:mm:ss PM GMT`; 108 in the 2024 store, 137
   in 2025) and are shown for widely connected partners, so they withheld 522 (2024) and 3,036 (2025, 7.0%) firm-weeks.
   `td4_repair.py` rebuilds only those firm-weeks with one rule: the unbracketed field (value without "(", "[" or a
   line break, at most 40 characters) is replaced by the same number of spaces before masking, so TD3's truncation
   windows are unchanged; masker, residual checks and render are TD3 v6 unchanged. An input that differs from the
   main build only through this rule would otherwise have been withheld. Repaired inputs are verified and audited like
   the others and called after the main ones (sample_rank + 1,000,000). The reviewer may instead reject the repair, in
   which case `inputs/repair_*` should be unbound and those rows take the fallback.
9. **Input header and questions.** The frozen TD3 header `FROZEN INPUT (TD3-input-v7)` and the question bundle
   (sha256 `8e1284dd...`, equal to TD3's) are kept byte for byte, so request bodies are those of TD3.
10. **2025 universe.** Rebuilt by the hist-panel rules (exact on 2022); the label filter is applied at evaluation, so
    calls are made for the few rows it later drops (2024: 86 of 49,079 keys).
11. **Refit rules.** MLP: TD1's refit procedure (frozen 2015-2020 preprocessing and target scaling, 14 epochs, seed
    41037) on every eligible row with origin year <= Y and label end <= Y-12-31 (late-December rows of Y-1 whose
    windows end in January of Y enter the through-Y refit, as 2021 rows entered TD1's through-2022 refit). Heads: TD2's
    refit branch with TD2's frozen penalties (main, 1r, 2r 0.01; 3r 1e5) and 2015-2020 training RMS scale; residual =
    y minus the in-sample prediction of the MLP refitted through the same Y.
12. **Selection protocol.** As TD2: the selection year is scored by models fitted before it (through 2023 for 2024);
    the training-|beta| reference order is |standardised coefficient| of the through-2023 main head, frozen into the
    selection file for 2025. Grids: DeepSeek and Noul K in {0,1,2,3,5,all}; Noul thresholds {0.25, 0.5, 0.75} (TD2's
    frozen grid); Choice top-one-or-none; fallback none/all for rows with news and no valid output (invalid, failed,
    withheld or not called). Ties keep the first grid entry (TD2 A10).
13. **Gate details.** A DeepSeek ordering with an invalid, duplicate or missing name is invalid as a whole (never
    partially used). Choice: an unavailable chosen item activates nothing; `none` activates nothing (an answer, not a
    fallback). Noul top-K ties by menu order.
14. **Runner (new).** Keyed idempotent ledger, balance floor, off-peak pause, Jev 429 backoff, per-period caps and
    concurrency (section 7). TD3's request bodies, parsers, returned-model checks (deepseek-flash, jev-1.14.0 in every
    response), halts and structural probe are reused; the probe runs per selector over completion order.
15. **Two clearances.** A authorises refit, calls, re-ask and selection; B (after the reviewer checks the selection
    freeze) authorises the once-only evaluation and binds the selection freeze, the pre-registration, the 2025 ledger
    digest and the through-2024 models.
16. **Identity-sensitivity variant.** Built for 2025 (no FSB5 bridge starts after 2024) but only the primary variant is
    called and evaluated.

## 6. Tests and checks

- `run_td4.py qualify`: **46 tests, 46 passed, 0 failures, 0 errors** (`QUALIFICATION_TD4_21d4169f...json`, sha256
  `de3c996b...`, bound in the freeze). Coverage: pre-registration and seals; gates (top-K with missing items, invalid
  outputs never backfilled, Choice none and unavailable choices, Noul ties and thresholds, gate-to-feature mapping,
  K=0 = baseline); labels (token required; a window past its allowed end is never opened; 2025 clamp; value); no label
  column in the 2025 feature files; scoring refuses a window crossing the year end; copied functions on synthetic
  fixtures (seal refusal, future and ended edges, latest version, graph memory filter exact, period guard, missing
  characteristics, future-dated news via TD3 `verify_article`, year-end origin rule, graph extract parsing, story-field
  repair rule); runner with mocked providers (complete then resume sends nothing; unsettled reserve counted and re-sent
  once; response row written before the settle row; balance floor stops before the floor; stale balance read handled
  conservatively; balance-read failure halts; no DeepSeek call starts in a peak window; Jev 429 backoff; HTTP 401 halt;
  disallowed returned model halt; invalid output logged, not backfilled; keys never written even when echoed; Jev
  request cap; re-ask only after the main pass, with new keys); gate (absent clearance, request_only, code inventory
  change); selection and evaluation cores on synthetic data (P1/P2, stop-rule readings, hop increments); diagnostics;
  public folder (code and Markdown only; no audit-sample firm name or eight-word text window in any file here).
- Parity receipts: section 4. Population checks and audit samples: section 5 (items 6-7).
- **Dress rehearsal** (`run_td4.py rehearsal`, `REHEARSAL_d8a2e345.json`, run with the final code): in a sandbox it
  built 150 inputs per year for 2021/2022, wrote a freeze and request, a sandbox clearance A, refitted through 2020
  (274,249 rows, equal to TD1's fitting rows) and 2021 (318,649), ran mock calls (450 of 450 units per year, no halt)
  and re-asks, the selection (freeze, then request B), a sandbox clearance B and the evaluation; a second evaluation was
  refused by the once-only latch. Its P1/P2 numbers come from mock outputs on development years and are not results.
- **Dry runs** over 2,000 real inputs per period with mock providers: 6,000 calls each, no halt, DeepSeek waited out a
  simulated peak window while Jev continued, a resume re-sent 0 calls (`runs/dryrun_2024_d39bff96`,
  `runs/dryrun_2025_cc255752`).
- **Freeze self-check**: a simulated clearance written under `fixtures/` (never in the clearance folder) validates
  against `FREEZE_TD4_A_v1.json` with no reason; `run_td4.py status` reports only "clearance file absent".

## 7. Cost, time and memory (estimates)

Token model fitted on the 200 pass-1 pilot calls (input tokens = a + c x input bytes per selector; DeepSeek output
23 tokens and 255 cached tokens per call), applied to every input sent; prices are TD3 v6's (deepseek-flash off-peak
USD 0.15 miss / 0.003 hit / 0.60 output per million; Jev USD 0.042 per million input). Includes the 100-firm-week
re-ask per period; excludes retries (0 in the pilot). `COST_PROJECTION_810c4941.json`.

| Period | Firm-weeks | DeepSeek calls | DeepSeek off-peak (peak) | Jev requests | Jev | Total off-peak |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | 45,895 | 45,995 | USD 31.17 (62.34) | 91,990 | USD 20.10 | USD 51.27 |
| 2025 | 42,686 | 42,786 | USD 27.64 (55.29) | 85,572 | USD 17.89 | USD 45.54 |
| Both | 88,581 | 88,781 | USD 58.82 (117.63) | 177,562 | USD 37.99 | USD 96.81 |

- **DeepSeek balance.** USD 62.57 (2026-11-06 05:47Z) minus the USD 8 floor leaves USD 54.57: 2024 fits; both periods
  fall short by about USD 4.25. The runner will stop cleanly near the end of the 2025 calls; **top up at least USD 10
  before the 2025 calls** (or resume after a top-up; completed requests are never re-billed).
- **Caps bound in the freeze** (1.3 x projection; Jev requests 1.1 x planned + 500): 2024 DeepSeek USD 40.52, Jev USD
  26.13 and 101,689 requests; 2025 DeepSeek USD 35.94, Jev USD 23.26 and 94,629 requests.
- **Wall time.** At the pilot's latencies (DeepSeek 1.13 s, Jev 0.15 s) and 6 DeepSeek / 3 Jev workers: DeepSeek busy
  2.4 h (2024) and 2.2 h (2025), Jev 1.3 h and 1.2 h. Each weekday has 7 peak hours with no DeepSeek call, so a period
  takes about 2.5 h on a weekend and up to about 10 h if started just before a weekday peak. Refits take minutes;
  selection and evaluation a few minutes each.
- **Memory (measured peaks).** Main input builds 1.01 GB (2024) and 0.73 GB (2025); repair builds 0.48 and 0.76 GB;
  2025 sources 1.07 GB; manifest 0.89 GB; features 0.92 GB; parity runs up to 1.20 GB (comparison frames); rehearsal
  0.93 GB (refits and selection included); dry runs low (inputs read lazily). Four steps exceeded the 1 GB target
  slightly; all stayed under the 1.4 GB job cap; one heavy process ran at a time.
- **Disk.** TD4 private root 4.1 GB (inputs, stores, extracts, fixtures, rehearsal sandbox); D: 41.4 GB free, C: 20.7
  GB free at the end.

## 8. Runbook

PowerShell from this folder; interpreter `C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe` (TD1-TD3
used the same; no venv). Set `$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'`, then
`& $PY -B run_td4.py <command>`.

Already run by the builder (offline, no clearance): `inventory`, `sources`, `manifest --year 2025`,
`features --year 2025`, `parity --check <each>`, `inputs --year 2024`, `inputs --year 2025`, `verify --year Y`,
`repair --year Y`, `verify --year Y --check repair`, `audit-sample --year Y [--check repair]`, `packet`, `cost`,
`dry-run --year Y --limit 2000`, `prereg`, `rehearsal`, `qualify`, `freeze`, `summary`.

Real steps (each refuses unless its clearance file matches the freeze; run in this order):

| Step | Command | Gate | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | `status` | none | hashes | nothing |
| 2 | `refit` | A | labels <= 2024-12-31 | `models/MLP_THROUGH_{2023,2024}.pkl`, `B_THROUGH_*`, `HEADS_THROUGH_*`, `REFIT_RECEIPT_v1.json` |
| 3 | `call --year 2024` (repeat to resume) | A, online | 2024 inputs; keys from DEEPSEEK_API_KEY, TYPESAFE_API_KEY | `runs/real_2024/session_n/` |
| 4 | `reask --year 2024` | A, online | 100 fixed firm-weeks | same folder, pass reask1 |
| 5 | `call --year 2025` (refuses until 2024 is complete) | A, online | 2025 inputs | `runs/real_2025/` |
| 6 | `reask --year 2025` | A, online | | |
| 7 | `select` | A | 2024 labels and outputs | `selection/SELECTION_FREEZE_v1.json`; `CLEARANCE_REQUEST_TD4_B_v1.json` once 2025 is complete (else `request-b`) |
| 8 | reviewer checks the selection freeze and writes clearance B | | | |
| 9 | `evaluate` (once) | B | 2025 labels and outputs | `evaluation/EVALUATION_STARTED_v1.json` latch, `EVALUATION_RESULT_v1.json` |
| 10 | `score-labels --labels <filled template>` (after human labeling) | none | labels, 2024 outputs | `labeling/LABEL_SCORES_*.json` |

A session stops cleanly on the DeepSeek balance floor, a cap, HTTP 401/402/403, a redirect, a third other 4xx per
provider, three consecutive failed units per provider, the TD3 structural probe, a disallowed returned model, or a
balance-read failure. Re-running the same `call` command resumes; completed requests are skipped and never re-billed.
After a floor stop, top up and resume. A `model_not_allowed` halt needs a review before resuming.

Labeling (diagnostics, TODO section 4): `labeling/LABEL_PACKET_v1.jsonl` (150 masked 2024 contexts, no outcome, no
selector output, no record key; sha256 `bd1491fc...`), `LABEL_TEMPLATE_v1.csv`, `LABELING_INSTRUCTIONS_v1.txt`. Humans
fill graded relevance (0-3) per item and a top choice or none; `score-labels` then reports NDCG@1,2,3,5 for the
DeepSeek and Noul rankings, top-one accuracy, Jev Choice multiclass Brier and ECE, Noul Brier and ECE against
relevance >= 2, and the re-ask stability of both periods.

## 9. Clearance files (the reviewer creates them; the builder wrote requests only)

- **Clearance A**: `D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_A_v1.json`, schema `TD4_CLEARANCE_A_v1`,
  `status` CLEARED, no `request_only` key, `freeze_sha256` = SHA256 of `FREEZE_TD4_A_v1.json`, and the fields of
  `CLEARANCE_REQUEST_TD4_A_v1.json` (version, code_sha256 of the 24 `td4/*.py`, artifact_sha256 (63 private files),
  external_sha256 (80 TD1/TD2/TD3 files and code), preregistration_sha256, aggregation_freeze_sha256, caps, models,
  question_sha256, periods, hard_seal). Adding or editing any `td4/*.py` or changing any bound file invalidates it.
- **Clearance B**: `D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_B_v1.json`, schema `TD4_CLEARANCE_B_v1`,
  copied from `selection/CLEARANCE_REQUEST_TD4_B_v1.json` (clearance_a_sha256, selection_freeze_sha256,
  preregistration_sha256, ledger_digest_2025, models_sha256, evaluation_year 2025, maximum_label_end 2025-12-31).

| File (private root) | SHA256 |
| --- | --- |
| `FREEZE_TD4_A_v1.json` (manifest) | `cb4bc0a0b7992ffdbfaaa08f50a2f7b184e6c69688279097abfb61be64f7a2d2` |
| `CLEARANCE_REQUEST_TD4_A_v1.json` | `e6e3a35bc3ec7e7e2b3f864c02a206abe7651063fdc91fb681c69f125b9a4c72` |
| `PREREGISTRATION_v1.json` | `56415d4e48cd305f9315785dc7dc33524e91f7e832d4a1a8c711139125194f86` |
| `AGGREGATION_FREEZE_v1.json` | `ca80062682b48c1740c1783ac46f2ab1e7dc6f361ef78da6d0a97317d34152d3` |
| `TD4_QUALIFICATION_2026-11-06T170200Z.json` (summary: tests, parity, checks, rehearsal, dry runs, cost; not bound) | `a2188a5405d340ee211f2feee21bf390babf99d51c5f5e2f7d755df4370498e5` |

## 10. WRDS query for 2025 masking names (not run; for review)

The local CRSP name files end at 2024-12-31. Names are a masking source only. If approved, Claude runs it through the
shared lock; the inputs would then be rebuilt and re-frozen (new clearance).

```sql
-- availability probes
SELECT 1 FROM crsp.stksecurityinfohist LIMIT 1;
SELECT 1 FROM crsp.stocknames LIMIT 1;
-- counts first
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT permno) AS n_permnos FROM crsp.stksecurityinfohist
WHERE secinfostartdt >= '2025-01-01' AND secinfostartdt <= '2025-12-31';
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT permno) AS n_permnos FROM crsp.stocknames
WHERE namedt >= '2025-01-01' AND namedt <= '2025-12-31';
-- pull (CIZ format; legacy fallback below)
SELECT permno, secinfostartdt, LEAST(secinfoenddt, DATE '2025-12-31') AS secinfoenddt_censored,
       ticker, issuernm, securitynm, cusip
FROM crsp.stksecurityinfohist WHERE secinfostartdt >= '2025-01-01' AND secinfostartdt <= '2025-12-31';
SELECT permno, namedt, LEAST(nameenddt, DATE '2025-12-31') AS nameenddt_censored, ncusip, ticker, comnam, shrcd, exchcd
FROM crsp.stocknames WHERE namedt >= '2025-01-01' AND namedt <= '2025-12-31';
```

Nothing else is missing for this design. The FactSet graph cannot be refreshed past October 2025 from WRDS (the
account cannot read `factset_revere`); the Baidu June-2026 copy is tests-only and holds 2026 rows, so it was not used.

## 11. Open risks

1. **Masking at scale is not human-verified.** TD3 needed six review rounds and end-to-end reads of all 200 pilot
   inputs; its firm-specific mask table covers only the pilot's firms. The TD4 inputs rely on the same automated rules;
   firm-owned product or unit names of other firms can give a masked name back. The 1,489 audit-sample scanner hits
   are unadjudicated candidates. Recommend adjudicating them and reading a sample end to end before clearing the calls.
2. **Recognition through model memory** (pre-registered caveat b).
3. **DeepSeek balance**: a top-up of about USD 4.25 (recommend USD 10) is needed to finish 2025.
4. **FactSet graph is not point-in-time** (October 2025 snapshot; pre-registered caveat).
5. **2025 masking names end 2024-12-31** (CRSP and FactSet name files).
6. **D8 repair** changes masker input for 3,558 inputs (section 5, item 8); flagged for review.
7. **Withheld rows**: 508 eligible firm-weeks (0.6%) are not called and take the fallback in every provider arm.
8. **News coverage** depends on the frozen 2022 RIC identity (TD2 A4); coverage, not leakage.
9. **Structural-probe halts** over about 89,000 calls per selector are possible (three consecutive invalid outputs
   with the same structural issue); resumable after inspection. The pilot had 0 invalid outputs in 660 calls.
10. **Jev rate limits are unknown**; persistent 429s end in a failed-unit halt (three consecutive failed Jev units).
11. **Peak boundary**: no DeepSeek call starts within two minutes of a peak window; billing by completion time could
    price a straddling call at peak (at most a handful).
12. **Fundamentals are latest extracts** with availability lags (TD1 caveat).

## 12. Disclosures

- No provider call, no network use (every command ran under the offline audit hook except a few offline read-only
  probes run as plain Python before the package existed: parquet metadata, masker timing on the TD3 pilot metadata, a
  feature-parity debug, a story-field profile of the article stores, and freeze/hash summaries), no git, no WRDS, no
  clearance file, no 2023-2025 outcome or label. Development labels (label end <= 2022-12-31) were read by the
  refit2022 parity and the rehearsal.
- Nothing pre-existing was edited or deleted. New TD4 files were edited during the build before the freeze. Extra
  private folders from development remain under the TD4 root (trial `period_2024_limit300`, `scratch_prof80`, three
  `parity_pilot_*` and two `rebuild_*` builds, `fixtures/`, `rehearsal_d8a2e345/`, early POPULATION_CHECK files without
  key lists); none is bound except as listed in the freeze. The TD3 runtime forbids deletes, so none was removed.
- `td4_inputs.py` was edited after the 2024 build started (new functions only: cache trimming between chunks, repair
  hook, callable inputs, bounded audit metadata); the build path that produced the inputs is otherwise unchanged, and
  the final code reproduces the pilot (200/200) and the first 2024/2025 firm-weeks of the full builds.
- Four steps peaked slightly above the 1 GB per-process target (section 7).
- The pre-registration text was written to its hashed file before any 2025 outcome exists and must not be changed.
