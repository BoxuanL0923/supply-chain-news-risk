---
name: sc-gate-industry-peer
description: Compare graph validation losses with an aligned same-operator industry-peer counterpart benchmark.
---

Invoke explicitly:
`/sc-gate-industry-peer compare graph and peer counterparts; inputs: aligned CSV/Parquet SSEs and benchmark-contract JSON, profile; script: skills/sc-gate-industry-peer/scripts/gate_industry_peer.py; constraints: same operator and controls, pre-cutoff nonlinked peers, size factor 2; deliverable: JSON graph advantage and pass/fail per counterpart; verify alignment and attestations; stop after first receipt or error.`

Before execution state the required output, verification, and stop above.
Run the self-contained script (standard library; optional pandas for Parquet):
`python skills/sc-gate-industry-peer/scripts/gate_industry_peer.py --input peers.csv --contract peer_contract.json --profile agent-v4 --se-multiplier 1`

Rows: unique `row_id`, ISO `date`, positive integer `n`, `fit_parity`,
`sse_graph`, and `sse_indpeer`. `o2` instead requires `sse_ff49` and
`sse_naics4`. All losses share outcomes, rows, incumbent regressors and both
missingness controls. Use 2015-2021 calendar years and opposite ISO-week folds;
v4/O2 score odd weeks, loop scores both directions.

Contract JSON names `operator`, lists `industries` (`["indpeer"]` or
`["ff49","naics4"]`), sets `size_factor: 2`, and attests `same_operator`,
`exclude_self_and_supply_links`, `shared_missingness_controls`, `point_in_time`
as true. Construct peers from same-industry firms within a factor of 2 in
lagged size, nearest absolute log-size distance, ties by node ID, up to target
degree, excluding self and direct supply links. For undirected operators use
the symmetric union. Keep shortages; never fill from another industry.

v4 requires graph gain strictly larger than indpeer gain, equivalently lower
SSE on this common base/sample. O2 must beat BOTH FF49 and NAICS4. Loop
requires graph-vs-peer gain > 1 week SE and finite t; expose
`--se-multiplier 1`. Cluster SE uses the validation skill's score formula.
These adapters verify arithmetic/alignment and contract fields; construction
and fitting of supplied peers remain host-attested.

Required output: per-peer graph advantage, SE/counts, thresholds, profile,
contract and input hash. Verify same operator/rows/controls and complete folds.
Stop on receipt or error; exit 0/1/2 = pass/fail/error.
[Schemas and synthetic example](../../docs/SKILLS_AND_ROUTER.md).
