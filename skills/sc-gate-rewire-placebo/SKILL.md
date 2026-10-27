---
name: sc-gate-rewire-placebo
description: Generate stratified degree-preserving graph draws or test aligned real/null validation gains with reproducible topology receipts.
---

Invoke explicitly:
`/sc-gate-rewire-placebo test graph specificity; inputs: graph plus seeded draw graphs and aligned held-out SSEs; script: skills/sc-gate-rewire-placebo/scripts/gate_rewire_placebo.py; constraints: 100 draws, 20 sweeps, calendar month x endpoint industry strata, p < 0.05, declared sidedness; deliverable: JSON p-value, gains and preservation receipts; verify every seeded draw and common sample; stop on receipt or first invalid input.`

Before execution state the required output, verification, and stop above.
The self-contained standard-library script has two bounded operations:

```
python skills/sc-gate-rewire-placebo/scripts/gate_rewire_placebo.py generate --input graph.json --draws 100 --seed 20261026 --sweeps 20
python skills/sc-gate-rewire-placebo/scripts/gate_rewire_placebo.py evaluate --input placebo.json --profile agent-v4 --draws 100 --threshold 0.05
```

Graph JSON: `directed` boolean, `nodes` mapping string IDs to industry strata,
`edges` containing `source`, `target`, `period` (`YYYY-MM`, 2015-2021).
Extra edge-slot attributes stay in place. Switch `(a,b),(c,d)` to `(a,d),(c,b)`
within month and matching endpoint industries; reject loops, duplicates and
collisions. Directed switches preserve in/out and industry-specific degrees;
undirected switches preserve degree and industry mixing. Frozen strata remain
unchanged and report movement=0. Seeded Python RNG is deterministic here; its
draws are not byte-identical to upstream Numba draws.

Bundle JSON: `graph`, `operator`, `same_operator_refit: true`, `observed_rows`
(validation skill schema), `draws: [{seed, sweeps:20, graph, rows}]`.
The host recomputes the SAME feature operator and refits on each graph,
holding outcomes/news/incumbent/sample fixed. The script reproduces each
seeded draw, verifies degrees/attributes and identical row IDs, dates, counts,
baseline SSE and folds; host feature construction/refits remain attested.
For a 10% intact-event subsample, use its real score as well as its null scores.
Do not compare sampled nulls with the full-sample real gain.

Default v4/loop p is `(1+count(null_gain >= real_gain-1e-12))/(B+1)`;
O2 uses `count/B` and relative SSE percent rather than MSE units. Default B=100,
threshold=0.05, strict inequality. Expose `--draws` and `--threshold` (0 < p
threshold <= .05); non-100 draws are exploratory. `--two-sided` requires v4
and an ex ante `ambiguity_rationale`: upstream v4 uses the SAME upper-tail
direction-free predictive gain, not `2*p` or abs(gain). Label exploratory.
The user's default 0.05 overrides guideline 8.3's two-sided 0.025; pass
`--threshold 0.025` explicitly for that stricter screen.

Required output: p/formula/sidedness/threshold, observed/null gains, draw count,
sample hash and per-draw preservation/movement receipts. Stop on the receipt
or first invalid input; exit 0/1/2 = pass/fail/error.
[Detailed input contract](../../docs/SKILLS_AND_ROUTER.md).
