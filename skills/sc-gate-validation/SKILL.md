---
name: sc-gate-validation
description: Compute one incremental alternate-ISO-week validation gate from aligned held-out loss summaries against an incumbent.
---

Invoke explicitly:
`/sc-gate-validation test incremental gain; inputs: CSV/Parquet loss summaries, profile; script: skills/sc-gate-validation/scripts/gate_validation.py; constraints: training calendar years 2015-2021, opposite ISO-week fit/score parity, SE_MULTIPLIER=1; deliverable: JSON gain, week SE, pass/fail and input hash; verify cluster formula and folds; stop after one receipt or input error.`

Before execution state the required output, verification, and stop condition above.
Run the self-contained standard-library script (Parquet optionally uses pandas/pyarrow):
`python skills/sc-gate-validation/scripts/gate_validation.py --input validation.csv --profile agent-v4 --se-multiplier 1`

Inputs have unique `row_id`, ISO `date`, positive integer `n`, nonnegative
`sse_incumbent`, `sse_candidate`, and `fit_parity` (0=even, 1=odd). SSEs are
summed squared held-out errors on the identical rows. Predictions may be
converted to SSE by summing `(outcome-prediction)**2`; upstream fits remain fixed.
Dates must be calendar 2015-2021. `agent-v4` (default) and `o2` fit even and score
odd ISO weeks. `loop` requires both opposite-parity folds.

For ISO-year/week clusters, compute `gain=sum(old-new)/N` and
`SE=sqrt(sum_week((delta_week-gain*n_week)**2))/N`, without a finite-sample
correction. Require gain > 0 and gain > 1 SE (strict). `--se-multiplier` is
exposed; changing it changes the declared screen. Legacy loop/O2 additionally
require nonzero SE, matching their finite `t_week > 1` rule; v4 accepts a
positive gain with zero SE. Require at least two clusters. Never merge equal
week numbers from different ISO years. Fit provenance is caller-attested.

Required output: JSON `passed`, `mse_reduction`, `week_se`, `gain_pct`, `N`,
`weeks`, folds, profile, thresholds and input SHA256. Verify counts and parity;
stop on receipt (exit 0 pass, 1 fail) or error (exit 2). Do not tune after viewing
validation outcomes. [Schemas and synthetic example](../../docs/SKILLS_AND_ROUTER.md).
