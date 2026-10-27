---
name: sc-gate-sign-stability
description: Check a predeclared mechanism direction and the complete training-year sign stability gate from coefficient summaries.
---

Invoke explicitly:
`/sc-gate-sign-stability check declared sign then yearly stability; inputs: timestamped coefficient-summary JSON, profile; script: skills/sc-gate-sign-stability/scripts/gate_sign_stability.py; constraints: pre-stated direction, all 7 years 2015-2021, agent-v4 >=4 right-sign years; deliverable: JSON c/d checks, count and hash; verify chronology, split and coverage; stop at sign failure or after stability receipt.`

Before execution state the required output, verification, and stop above.
Run the self-contained standard-library script:
`python skills/sc-gate-sign-stability/scripts/gate_sign_stability.py --input sign.json --profile agent-v4`

JSON fields: timezone-aware `declared_at < evaluated_at`, `predicted_sign`
(`+` or `-`), `estimate_split`, `pooled: {estimate}`, and `years` mapping all
calendar years `2015` through `2021` to `{estimate}`. A zero estimate fails sign.
Missing years fail stability. `agent-v4` defaults to odd-week coefficients and
at least 4/7 strictly right-sign years, without yearly significance filtering.
`o2` uses pooled coefficients and >=5/7 right-sign years. `loop` uses pooled
coefficients, >=4/7 years, and both signed `t_day > 1.96` and `t_week > 1.96`
in each passing year. Set `estimate_split` to `odd` or `pooled` accordingly.
Expose `--required-years` and `--t-cutoff 1.96` if changing the screen; never
choose them from observed estimates. The guideline's 4/6 rule is a different
training calendar; do not silently relabel it 4/7.

Only v4 permits `two_sided: true`, with an ex ante `ambiguity_rationale`, and
no `post_hoc_of`. It uses the nonzero pooled odd-week estimate's direction for
yearly stability and labels the receipt exploratory. Keep the same declaration
for the placebo skill; its default threshold is 0.05 (guideline 0.025).

Required output: JSON c/d checks (`stability: null` if sign fails), first failure,
passing years, declared/effective direction, profile and input hash. Verify
chronology, complete seven-year coverage and correct split. Historical
preregistration and coefficient fitting are caller-attested. Stop on first
failure or receipt; exit 0/1/2 = pass/fail/error. [Full schema](../../docs/SKILLS_AND_ROUTER.md).
