# TD2 handoff: five-session risk head, exposures and selector harness

Written 2026-11-05T02:55:53Z (from `date -u`). **Exploratory.** 2023-2025 was already examined in the existing risk study,
so any 2023-2024 run is a historical reproduction, not a new test. Facts only. Model outputs (coverage counts, coefficients,
selection MSEs) are kept out of this public folder, in
`D:/OptResearch_data/five_session/td2/TD2_HANDOFF_PRIVATE_v1.md` and the private receipts.

What was and was not done: no 2023-2024 five-session label was read or computed, and no evaluation was run. No LLM or provider
call, WRDS connection, network access or git command happened. No 2025 value enters any output. Nothing was deleted or
overwritten. The frozen TD1 MLP and manifest are byte-identical before and after (hashes below).

## Item status

"Found" is the state left by the previous builder (Codex, stopped by a usage limit). "Now" is the state after this
completion. Existing files were never edited. Changes went into new versioned files.

| Item | Found | Now | Evidence (private root `D:/OptResearch_data/five_session/td2/`) |
| --- | --- | --- | --- |
| (1) risk_menu.csv | Complete: 7 required columns, 5 firm characteristics, all eligible, frozen on 2015-2022 non-missing counts | Complete, unchanged | `risk_menu.csv`; counts in `EXPOSURE_DICTIONARY_v1.json` |
| (2) Exposure dictionary E | Partial: rules, lexicons and aggregation rule frozen (`EXPOSURE_DICTIONARY_v1.json`, 2026-11-05T00:20:02Z); sources and 2014-2022 news scores prepared; no E computed; the 2015/2016 builds were interrupted and left partial `evidence/*.jsonl` and no features | Complete: E for exact-hop H1, H2 and H3 on the graph valid before each cutoff, primary and FSB5-sensitivity identity variants, every eligible firm-week 2015-2024; witness evidence with intermediate nodes; coverage per year; 2023-2024 news scores built with the frozen development identities | `features_v2/`, `evidence_v2/`, `coverage_v2/`, `NEWS_EVAL_v2.json` |
| (3) Z = C x E | Missing | Complete: `Z1_m`, `Z2_m`, `Z3_m` and main `Z_m` = mean over hops for the 5 items on 921,710 rows (460,855 eligible firm-weeks x 2 identity variants) | `features_v2/year_2015..2024.parquet` |
| (4) Shared ridge head | Partial: code (`pipeline.fit`) never run; v1 penalty grid could not shrink (A1) | Complete: no intercept, all Z active (q=1), penalty grid 1e-2..1e8 chosen on 2021-2022, refit through 2022, frozen; intraday slopes not used | `MODEL_v2.json`, `SELECTION_v2.json` |
| (5) Selector harness | Partial: code and empty slots, never run; `training_beta` ordered by raw-unit beta (A2) | Complete: K in 0,1,2,3,5,all and fallback none/all chosen on 2021-2022 for `training_beta` and random seeds 72101/72103/72109. DeepSeek order, Jev Choice and Jev Noul slots are empty, schema-validated and hash-recorded as `EMPTY_PENDING_SELECTION` (no K, threshold or fallback chosen) | `SELECTION_v2.json`, `MODEL_v2.json` `provider_gates`, `selector_slots/*_v1.json` |
| (6) H1/H2/H3 ablations | Partial: only inside the v1 evaluator, never run | Complete as construction: arms `main`, `1`, `2`, `3` (frozen main beta on exact-hop Z) plus refit heads `1r`, `2r`, `3r`, all on identical rows with the same selectors and K. A development-only table exists. 2023-2024 numbers appear only after clearance | `MODEL_v2.json` `heads`, `SELECTION_v2.json` |
| (7) Gated 2023-2024 runner | Partial: `evaluation_v2.py` bound to `FREEZE_v1.json`/`MODEL_v1.json`, which never existed; never run end to end | Complete: `evaluation_v3.py` passed an end-to-end synthetic run (gate, labels, paired gains, two-way clustering, once-only latch). On the real path it refused because no clearance exists | `logs_v2/evaluate_refusal.log` |
| (8) DeepSeek token/cost estimate | Missing (code only) | Complete: estimate below | `TOKEN_COST_ESTIMATE_v2.json` |

Synthetic fixture tests (spec list) and the tests in `tests_v2.py` that cover them:
- Future-dated news and edges rejected: `test_story_filter_excludes_origin_day_future_window_and_month`,
  `test_exact_hops_and_future_edges_excluded`, `test_injected_future_news_rejected`, plus the v1 fixture tests.
- Missing characteristics and `none`: `test_missing_characteristic_recorded_and_zero`, `test_k_zero_is_baseline_and_none`,
  `test_missing_items_do_not_consume_budget`.
- Gate-to-feature mapping: `test_gate_matrix_parity_with_v1`, `test_random_rank_parity_with_v1_ordering`,
  `test_fallback_only_rows_without_news`.
- No future returns in any input: `test_label_columns_in_manifest_are_ignored`, `test_feature_projection_requests_no_label_columns`,
  `test_2023_labels_refused_before_any_read`, `test_schema_roundtrip_and_no_outcome_columns`, `test_returns_reader_refuses_without_token`,
  `test_label_requires_admission_and_skips_clipped_window`.
- Quarantine (91,869 FSB3 IDs) and date seal: the `Quarantine` class.

## Amendments made before any head fit (recorded in `spec_v2.py` and `MODEL_v2.json`)

`spec_v2.py` was bound by qualification receipt `QUALIFICATION_v2_5b188782a8094aebba15a79c96b0f069.json`
(command started 02:25:32Z, receipt `written_utc` 02:25:35Z), before the fit command started at 02:51:59Z.
- A1: penalty grid 1e-2..1e8. The v1 maximum of 1e3 is below 0.4% of the x'x diagonal (~2.7e5 rows, unit-RMS columns).
- A2: `training_beta` orders by |standardized 2015-2020 coefficient|, because raw-unit order depends on units.
- A3: primary ablation unchanged from v1 (frozen main beta on exact-hop Z, no refit), plus secondary refit heads 1r/2r/3r.
- A4: the frozen development RIC/PERMNO mapping (panel keys up to 2022) is used for every year including 2023-2024, as the v1
  code did. This supersedes the v1 rule text "<=2024 for evaluation".
- A5/A6: parquet witness evidence; main `E_m` stored. A7: vectorized gates with tested parity. A8: the same penalty at refit.
  A9: the v1 partial JSONL files are retained and unused. A10: the tie-break rule. For local selectors the fallback cannot
  change predictions, because rows without news have Z=0.

## Exact commands that were run

Git Bash, with the TD1 interpreter. Each command refuses to overwrite an existing output, so reruns of build/fit/cost/freeze
stop with `FileExistsError`.

```bash
cd "D:/OptResearch_data/external/supply-chain-news-risk/five_session/td2/"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py qualify     # 02:25:32Z, 107/107
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py news-eval   # 02:25:42Z, 21 s
# three concurrent chains, 02:26:13Z-02:51:44Z; logs D:/OptResearch_data/five_session/td2/logs_v2/build_<year>.log
for y in 2015 2018 2021 2024; do "C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py build --year $y >> D:/OptResearch_data/five_session/td2/logs_v2/build_${y}.log 2>&1 || exit 1; done
for y in 2016 2019 2022;      do "C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py build --year $y >> D:/OptResearch_data/five_session/td2/logs_v2/build_${y}.log 2>&1 || exit 1; done
for y in 2017 2020 2023;      do "C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py build --year $y >> D:/OptResearch_data/five_session/td2/logs_v2/build_${y}.log 2>&1 || exit 1; done
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py qualify     # 02:51:51Z, 107/107 (after the tests_v2.py fixture change)
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py fit    > D:/OptResearch_data/five_session/td2/logs_v2/fit.log 2>&1
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py cost   > D:/OptResearch_data/five_session/td2/logs_v2/cost.log 2>&1
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py freeze > D:/OptResearch_data/five_session/td2/logs_v2/freeze.log 2>&1
"C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe" -B run_td2_v3.py evaluate > D:/OptResearch_data/five_session/td2/logs_v2/evaluate_refusal.log 2>&1
# -> exit 1: "PermissionError: STOP: TD2_CLEARANCE_v1.json absent; no evaluation label read or computed"; no latch written
```

Test counts: both qualification runs reported 107 tests, 107 passed, 0 failures, 0 errors. The suites were v1 fixture 30,
v1 boundary 14, v1 extended 12, v1 fast parity 2, v1 provider 8 and v2 synthetic 41. Receipts:
`QUALIFICATION_v2_5b188782a8094aebba15a79c96b0f069.json` (bound the builds) and
`QUALIFICATION_v2_322ceedd76a0472b9c504cf7c3799e3c.json` (bound the fit, cost and freeze, and is bound in `FREEZE_v2.json`). Between the two receipts
only `tests_v2.py` changed: the end-to-end fixture now uses the real CRSP input schema (date32, int32, float32, int8) and
a `share_type`=1 row that must be filtered out. The build-code hashes are identical in both receipts. The previous
builder's report (`QUALIFICATION_d15e5b2da6d24718a09e87a739c948c2.json`, 66 tests) covers the same v1 suites; the v1 provider
suite is counted there.

## Gated 2023-2024 evaluation (Claude runs it once)

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'; $env:PYTHONIOENCODING = 'utf-8'
Set-Location 'D:/OptResearch_data/external/supply-chain-news-risk/five_session/td2'
& 'C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe' -B run_td2_v3.py evaluate
```

Before it opens any return, label, feature or model payload, `evaluation_v3.admit` requires
`D:/OptResearch_data/five_session/clearance/TD2_CLEARANCE_v1.json` with:
- `schema` = `TD2_CLEARANCE_v1`, `status` = `CLEARED`, and no `request_only` key;
- `freeze_sha256` = SHA256 of `FREEZE_v2.json`;
- `version`, `code_sha256`, `artifact_sha256`, `td1_sha256`, `model_sha256` and `qualification_sha256` equal to `FREEZE_v2.json`;
- `evaluation_years` = [2023, 2024] and `maximum_label_end` = `2024-12-31`.

It then re-hashes every bound file, requires the code-folder `*.py` inventory to equal the freeze (so adding or editing any
`.py` here blocks it), and writes the once-only latch `EVALUATION_STARTED_v2.json`. `CLEARANCE_REQUEST_v2.json` contains these
fields pre-filled with `status` = `REQUESTED` and `request_only` = true. It is a request, never a clearance. Outputs:
`EVALUATION_RESULT_v2.json` and `evaluation_v2/squared_errors_<variant>_<selector>_<seed>.parquet` (8 files). The runner reports
paired squared-error gains against the frozen TD1 MLP `b` for 2 identity variants x 4 selector keys x 6 K x 7 hop arms, plus
H2-H1 and H3-H2 increments for both ablation families. Each comes with firm and calendar-week two-way clustered uncertainty
and coverage. Labels follow the TD1 rule (log mean of the next five squared CRSP daily total returns, `share_type` in {0, 2},
window end <=2024-12-31). The inactive v1 evaluator would stop with `FileNotFoundError` on the missing `FREEZE_v1.json` before
reading any label.

## Frozen artifact hashes (SHA256)

| Artifact | SHA256 |
| --- | --- |
| TD1 `D:/OptResearch_data/five_session/model_frozen_v2.pkl` (before = after = `MODEL_FREEZE_v3.json` value) | `7ce4015df2fed29134363893afe1f54d713cfaf57ad99ce99acf2f8f511df8ce` |
| TD1 `D:/OptResearch_data/five_session/MODEL_FREEZE_v3.json` (before = after) | `05d18f9b6e2677faa4ea3a9303de199ede1bc98fde76b68f562411692c0e4a0d` |
| TD1 `inputs/crsp_2023.parquet` / `crsp_2024.parquet` (bound) | `c492035f7aa31575c4be11f8906a2bc19c686008edca4286d1264701be8a7e4d` / `14351b4572b9e3ea8eaba9875655ee3e2baf259009272343fa60a67f40880279` |
| TD2 `FREEZE_v2.json` | `2d383ffb4894f4054728b30f15c9ce2197a5d03537aac95033e011b66aff03c5` |
| TD2 `MODEL_v2.json` | `d5f0d41546450b19fa84be52657bb460ec203de6b15c303a0902d2a0be3e4426` |
| TD2 `CLEARANCE_REQUEST_v2.json` | `db02f10392fefa0fd55e6a6ca4739e3c300d081ca5b13e983e0c8be8207e2c33` |
| TD2 `SELECTION_v2.json` | `ca985bab63bbcc190674d512bc08798a765b9098b0a4108b8ba6b22fa9c5956b` |
| TD2 `TOKEN_COST_ESTIMATE_v2.json` | `9183a4887961e3f6b75668c078eedb059ab9913d097c0ced14d737a2a2765be7` |
| TD2 `QUALIFICATION_v2_322ceedd76a0472b9c504cf7c3799e3c.json` | `45e32567ff9268d3929d194e7ca71cbe102be5b85ab93eb889bb005e3793b258` |

The 10 TD1 manifest partitions and 10 `predictions_v2` partitions equal `MODEL_FREEZE_v3.json` before and after. The comparisons
are in `receipts_v2/FROZEN_HASHES_BEFORE_e8b8bcad.json` (02:09:08Z) and `receipts_v2/FROZEN_HASHES_AFTER_7f8b3aee.json`
(02:53:34Z; `td1_all_equal` true). All 28 pre-existing TD2 private files (`FROZEN_HASHES_BEFORE_TD2PRE_077147f4.json`) and all
32 pre-existing code files (hashes in the Codex qualification report) are unchanged. `FREEZE_v2.json` binds 43 code files,
56 TD2 artifacts, 24 TD1 files and the qualification receipt.

## Active and inactive files

Active code (this folder): `run_td2_v3.py` (CLI), `qualify_v2.py`, `spec_v2.py`, `io_v2.py`, `gates_v2.py`, `build_v2.py`,
`head_v2.py`, `cost_v2.py`, `freeze_v2.py`, `evaluation_v3.py` and `tests_v2.py`. It also reuses unchanged v1 modules:
`runtime_v3.py`, `decoder_v3.py`, `engine.py`, `fast_builder.py`, `selectors.py`, `statistics.py`, `spec.py` and
`provider_harness.py` (future provider K/threshold/fallback tuning on 2021-2022). The qualification also runs `fixture_tests.py`,
`boundary_tests.py`, `extended_tests.py`, `fast_tests.py` and `provider_tests.py`, which import `delivery.py`, `pipeline.py`,
`evaluation_v2.py` and `decoder_v2.py`. Inactive and kept: `run_td2.py`, `run_td2_v2.py`, `run_td2_final.py`, `td2.py`,
`batch_remaining.py`, `qualify.py`, `run_checks.py`, `run_fast_checks.py`, `decoder.py`, `evaluation.py`, `runtime.py`,
`runtime_v2.py` and `probe*.py`.

Active private artifacts: `risk_menu.csv`, `EXPOSURE_DICTIONARY_v1.json`, `PREPARED_v1.json`, `sources/`, `news_scores/`
(2014-2024), `NEWS_EVAL_v2.json`, `selector_slots/`, `features_v2/`, `evidence_v2/`, `coverage_v2/`, `MODEL_v2.json`,
`SELECTION_v2.json`, `TOKEN_COST_ESTIMATE_v2.json`, `FREEZE_v2.json` and `CLEARANCE_REQUEST_v2.json`. Inactive and kept:
`evidence/year_2015.jsonl` and `evidence/year_2016.jsonl` (interrupted v1 builds), and the earlier `SYNTHETIC_TESTS_*`,
`FAST_PARITY_*` and `QUALIFICATION_d15e...` receipts. Test fixtures live in `fixtures_v2/` (28 synthetic directories).

## Memory, disk and run time (estimates from measurements on this machine)

- qualify: about 3 s, peak working set about 130 MB. news-eval: 21 s. fit: 9 s, peak 402 MB. cost: 15 s, peak 263 MB.
  freeze: 1.5 s.
- build: 295-431 s per year with three concurrent processes, peak working set 729-879 MB per process, 26 min wall time for
  all 10 years. Each process runs under the 2.4e9-byte Windows job cap from `runtime_v3.limit()`.
- evaluate (not run; estimate): reads CRSP 2023/2024 (1,195,055 and 1,149,278 rows) and about 95k feature rows per year.
  Expected a few minutes and well under 1.5 GB. Writes 8 squared-error parquet files (estimate below 200 MB) plus one JSON.
- Disk: `features_v2` 97.3 MB, `evidence_v2` 240.3 MB, 2023-2024 news scores 1.8 MB, `fixtures_v2` 19.4 MB, logs and
  receipts under 1 MB. The private td2 root totals 886.3 MB, of which the retained v1 partial JSONL is 520 MB. Code folder
  227 KB. C: received only scratch scripts (under 100 KB). D: had 89.4 GB free at the end (40 GB stop rule).

## DeepSeek token/cost estimate (estimate only; 0 provider calls)

- Calls: one DeepSeek ordering call per primary firm-week with news and at least one eligible characteristic. That gives
  82,445 calls for selection 2021-2022, 88,839 for evaluation 2023-2024, and 171,284 for both. Running the same calls for the
  FSB5-sensitivity contexts that differ would add 27,156.
- Measured masked prompts: 400 deterministic 2021-2022 firm-weeks were measured with a sizing draft of the question
  (`cost_v2.masked_prompt`; not the frozen provider question). Article text is replaced by `x` repeated to its measured length
  (headline up to 200 plus body up to 1,000 characters). With all unique articles of the selected partners, prompts average
  40,424 bytes (median 35,148; 60.3 articles). With one article per selected partner, they average 6,494 bytes (median 6,159;
  13.2 articles).
- Tokens: 0.2386 tokens per byte, measured from file sizes of the 2 Markdown prompts in
  `D:/OptResearch_data/sc_risk/o4_llm_loop/deepseek_costs.csv`, up to 0.30 per character (commonly cited DeepSeek guidance,
  not verified offline). Output is either 20 tokens (JSON ordering only, non-thinking) or 1,752 tokens (median completion of
  113 earlier flash calls on a different task, as a thinking-mode scenario).
- Prices: deepseek-v4-flash off-peak USD 0.15 per million input tokens (cache miss), 0.003 (cache hit) and 0.60 for output;
  peak is twice that. Source: the project memory note dated 2026-10-28; not re-verified here (no network).

USD for 171,284 calls (selection + evaluation), no cache, low-high tokens-per-byte range:

| Prompt policy | Output | Off-peak | Peak |
| --- | --- | --- | --- |
| One article per selected partner | Non-thinking | 41.90-52.15 | 83.81-104.29 |
| One article per selected partner | Thinking | 219.86-230.10 | 439.72-460.21 |
| All articles of selected partners | Non-thinking | 249.90-313.67 | 499.80-627.35 |
| All articles of selected partners | Thinking | 427.86-491.63 | 855.71-983.26 |

Caching the fixed 941-byte header lowers the off-peak non-thinking figures to 36.25-45.04 (one article) and 244.25-306.57 (all
articles). Not included: retries, repeats after malformed answers, Jev calls (no price available) and provider framing tokens.
Per-period tables are in `TOKEN_COST_ESTIMATE_v2.json`.

## Source limitations (facts)

- Graph: October 2025 FactSet snapshot. Start dates are retrospective validity proxies, not point-in-time publication dates.
  The 91,869 FSB3 `added` relationship IDs are excluded throughout. FSB5 candidate wrong-issuer links are excluded only in the
  `fsb5_sensitivity` variant. Paths are graph witnesses: `physical_channel_verified` = false and
  `intermediate_product_evidence` = `unavailable_in_sources` on every witness row.
- News: Reuters timestamps are day-level, so a story counts as available at the next New York midnight. The origin day is
  excluded, and so is a story whose firm mapping first appears in the origin month. One archive row group straddles
  2024-12-31. Its 2025 rows inside shared Arrow batches are decoded and then dropped by the date mask before conversion, so no
  2025 value is used or stored.
- Characteristics: TD1 latest-extract fundamentals with availability lags, not certified revision-free point-in-time data.
- K=5 and K=all are identical because all 5 menu items are eligible.
