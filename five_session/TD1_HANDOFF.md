# TD1 handoff: exploratory historical reconstruction

Code and documents: `D:/OptResearch_data/external/supply-chain-news-risk/five_session/`.
Private data and outputs: `D:/OptResearch_data/five_session/`.
Source inventory, old catalog, candidate graph/news definitions and clocks: `D:/OptResearch_data/external/supply-chain-news-risk/five_session/FIRST_HANDOFF.md`.

## Specification

One outcome: `log(mean(next five squared decimal daily total returns))`. All five subsequent exchange-session returns must be finite and their mean square positive. No response floor or annualization. Label windows cannot end after 2024-12-31.

Model version: `TD1-firm-only-64x32-v1`. Eighteen features plus eighteen missingness indicators feed ReLU layers of 64 and 32 units and a scalar identity output. `spec.py` records the feature order, clocks, fixed seed and hyperparameter grid. Own features contain RV1/RV5/RV22, returns over 1/5/22 sessions, five-session downside, size, and past earnings/calendar coverage. Common states contain market RV5/RV22 and leave-self-out industry RV5. No graph, partner news, article text or vendor labels enter the MLP.

Quarterly cash/assets, inventory/assets and debt/assets use valid rdq or fiscal-end +120 days, at end of day New York; equality with cutoff is excluded. Cash flow/assets comes from primary-common JKP `ocf_at`, available at source month-end +six calendar months, end of day New York. Source age is capped at 550 days. Missing fundamentals remain with missing indicators. No dated advance earnings schedule was available for this build, so the scheduled-earnings feature is missing. These are conservative availability reconstructions from latest extracts, not certified revision-free historical vintages. The gvkey link is undated and restricted to one-to-one pairs. The inherited firm universe is conditional on prior-year graph-linked selection.

Weekly cutoff means original panel session at 23:59:59 New York, stored as naive UTC; daily data is assumed available at 20:00 New York. Partitions and year splits refer to the origin session's year. Each feature has `__available_at`, `__missing` and `__included`. Missing feature values have null availability. A row must have full 22-session own history and positive size for feature eligibility.

Fit cutoffs 2015-2020, with label end <=2020-12-31. Select cutoffs 2021-2022, with label end <=2022-12-31. Fit feature medians, means/scales and target transformation on the fitting rows only. Select learning rate in `{0.001,0.0003}`, alpha in `{0.0001,0.01}` and retained epoch using selection MSE; maximum 80 epochs, patience 10, material-improvement threshold 1e-5. Refit the network through 2022 for the selected number of epochs, keeping original fitting-only preprocessing fixed. Freeze before evaluation.

The old comparison is CC1's primary log-target own HAR: RV1/RV5/RV22, downside, size, market RV5/RV22 and industry RV5, median imputation, standardization, intercept, numerical ridge 1e-6. It is fitted on the same fitting rows and compared with the selected MLP on the same 2021-2022 rows. This comparison uses no GARCH-level target or realized forward earnings flag. It reconstructs the old specification on repaired inputs; it is not a byte-for-byte reproduction of cached predictions. Selection losses are tuning diagnostics, not an untouched test.

## Exact commands

Use the installed local interpreter; no dependency installation, network, WRDS, git or provider call is required.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$td1Python = 'C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe'
$td1Runner = 'D:/OptResearch_data/external/supply-chain-news-risk/five_session/run_td1.py'

# Invented fixtures only; every invocation writes a new private test receipt.
& $td1Python -B $td1Runner tests

# Initial construction commands. These exclusively create files and refuse
# existing output paths; do not rerun them over the delivered artifacts.
& $td1Python -B $td1Runner snapshots
& $td1Python -B $td1Runner manifest
& $td1Python -B $td1Runner fit

# Append the delivered verifier correction to the original immutable freeze.
# This verifies that model/data/predictions/selection are unchanged.
& $td1Python -B 'D:/OptResearch_data/external/supply-chain-news-risk/five_session/amend_freeze.py'

# Verify hashes, feature clocks, row coverage, fitting-only preprocessing,
# architecture, and every prediction reloaded from the frozen model.
& $td1Python -B $td1Runner verify
```

The code uses installed numpy, pandas, pyarrow and scikit-learn. Exact installed versions, chosen parameters, seed, data/model/code hashes and active freeze clock are in `D:/OptResearch_data/five_session/MODEL_FREEZE_v2.json`. The original `D:/OptResearch_data/five_session/MODEL_FREEZE.json` is preserved. Wall-clock stamps are obtained by executing `C:/Program Files/Git/usr/bin/date.exe -u` immediately before writing them. All artifact writes are new files; no existing research file is replaced or deleted.

The sole code-freeze amendment corrects verification array layout. Reloaded training blocks were Fortran-contiguous, while the fitter's row selection was C-contiguous; different reduction order produced means differing by about 5e-12. The verifier now explicitly restores C order and retains bit-for-bit preprocessing equality. `amend_freeze.py` checks that input, manifest, selection, model and prediction hashes match the original freeze before recording new code hashes. Model weights, feature values, selected parameters and predictions were not changed. The active evaluator and verifier choose `MODEL_FREEZE_v2.json` when present.

## Private artifacts

| Artifact | Exact local path |
| --- | --- |
| Date-filtered source copies | `D:/OptResearch_data/five_session/inputs/` |
| Source-copy hashes and resources | `D:/OptResearch_data/five_session/SOURCE_SNAPSHOTS.json` |
| Firm-week manifest | `D:/OptResearch_data/five_session/firm_week_manifest/year_2015.parquet` through `year_2024.parquet` in that directory |
| Manifest construction receipt | `D:/OptResearch_data/five_session/MANIFEST_BUILD.json` |
| Development selection losses | `D:/OptResearch_data/five_session/selection/paired_predictions_2021_2022.parquet` |
| Parameter/epoch search and paired development MSE | `D:/OptResearch_data/five_session/SELECTION.json` |
| Fitted model, preprocessing and old baseline | `D:/OptResearch_data/five_session/model_frozen.pkl` |
| Active freeze record | `D:/OptResearch_data/five_session/MODEL_FREEZE_v2.json`; original `D:/OptResearch_data/five_session/MODEL_FREEZE.json` preserved |
| Versioned predictions | `D:/OptResearch_data/five_session/predictions/year_2015.parquet` through `year_2024.parquet` in that directory |
| Synthetic fixtures and receipts | `D:/OptResearch_data/five_session/fixtures/`; `D:/OptResearch_data/five_session/SYNTHETIC_TESTS_*.json` |
| Contract verification receipt | `D:/OptResearch_data/five_session/VERIFICATION.json`; repeat verification appends a UUID-named receipt |
| Active independent-clearance request | `D:/OptResearch_data/five_session/CLEARANCE_REQUEST_v2.json`; original `D:/OptResearch_data/five_session/CLEARANCE_REQUEST.json` preserved and superseded |

Manifest partitions for 2023-2024 contain features and feature eligibility only: no `y`, label-end date, label-observed flag or fit/selection/refit flag. Development labels are written only when the window ends by 2022-12-31. Predictions for every feature-eligible row contain `b`, `old_b`, version and prediction role, with no labels in any prediction partition. The 2015-2022 frozen-model predictions are explicitly refit in-sample; use the separate selection prediction file for the development comparison.

## Claude-only later evaluation

Required independent clearance file: `D:/OptResearch_data/five_session/clearance/TD1_CLEARANCE_v1.json`. `CLEARANCE_REQUEST_v2.json` lists the active expected schema/status and exact hashes, including `MODEL_FREEZE_v2.json`; it is a request, not an issued clearance. The older request's hashes will be rejected. TD1 does not create the required clearance file.

The independent reviewer must check the frozen delivery and supply a JSON object matching the request's bindings: schema `TD1_CLEARANCE_v1`, status `CLEARED`, version, freeze SHA256, model SHA256, code SHA256 map, manifest SHA256 map, input SHA256 map, prediction SHA256 map and selection-artifact SHA256 map. Bindings refer to immutable date-filtered source copies, avoiding any need to hash/read 2025 payload in mixed original files.

Only after that file exists and all bindings match, Claude runs:

```powershell
& 'C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe' -B 'D:/OptResearch_data/external/supply-chain-news-risk/five_session/run_td1.py' evaluate
```

`evaluate.py` refuses missing/altered clearance or code/data/model/predictions, and refuses any second start after exclusively creating `D:/OptResearch_data/five_session/EVALUATION_STARTED.json`. A failed hash admission does not reserve a run or read return payload. The admitted run computes only 2023-2024 labels with window ends <=2024-12-31, pairs MLP and old-baseline losses on identical finite rows, reports label coverage, and writes `D:/OptResearch_data/five_session/evaluation/paired_losses_2023_2024.parquet` plus `D:/OptResearch_data/five_session/EVALUATION_RESULT.json`. Reported uncertainty is a two-way firm/week clustered standard error of the mean squared-loss difference. This is an exploratory historical comparison; no confirmatory or LLM advantage is claimed.

## Resources and verification

Resource estimates for this implementation: peak Python memory about 1.5 GB; private D: artifacts about 0.3-0.8 GB; total construction and fitting runtime about 15-40 minutes on one CPU thread. These are estimates, not measured run facts. Exact measured stage durations, peak working set/private memory and free space are in the private receipts. The enforced Windows process/job private-memory ceiling is 2.5 billion bytes, below the task's 2.6 GB limit. Every production entry point installs that cap before scientific imports. D: free space is checked repeatedly; below 40 GiB, execution stops with a note and deletes nothing. Code/documents are on D:, and no task data/model artifact is intentionally written on C:.

Tests cover future/equal/unknown feature clocks; future news and edges; changing future returns and size without changing model inputs; exact five-subsequent-session labels; label boundaries before reading future return fields; missing characteristics and `none`; training-only imputation/scaling; old baseline recovery; paired rows and finite losses; string/date Arrow and CSV date filtering; quarterly rdq/fallback clocks; a full synthetic development/features-only manifest build; model shape; source allowlists; append-only writes; and clearance hash mismatch/missing-file admission. Production verification independently checks all feature clocks, missingness flags, target boundaries, row coverage, code/data/model hashes, fitting-only preprocessing and saved-model prediction parity. Test counts and pass/fail facts are recorded in the private receipts rather than placing experiment outputs in this public repository.

No repair job directory, IDS2 directory or shared WRDS lock is used by this code. There is no provider/LLM call or network client. The shared research files remain unchanged.
