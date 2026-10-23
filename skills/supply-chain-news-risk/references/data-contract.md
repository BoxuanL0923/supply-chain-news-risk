# Claim gates and data contract

Use the narrowest gate that matches the requested claim. Passing an earlier
gate does not imply that a later gate passes.

## Gate 1: completed-experiment documentation

The machine-readable dictionary must define the completed response, benchmark,
controls, anchor, and score with nonempty units, raw fields, formula, timing,
caveat, and source reference. Formula tests and package-manifest hashes must
pass. Disclosed proxy limitations remain limitations.

Status:

- `PASS`: required definitions and deterministic checks are complete.
- `QUALIFIED`: definitions are complete but disclosed vintage, rank, proxy, or
  historical-source limitations remain.
- `BLOCKED`: a required definition or deterministic check is missing or fails.

## Gate 2: exact historical replication

Every source file used by the historical run must have a recorded hash and the
packaged hash must match it. Required licensed inputs, caches, environment, and
provider configuration must be available. Recompute the results without
silently substituting later toolkit repairs.

Any core file marked `differs_from_recorded_run` blocks an exact-replication
claim even when the package manifest is internally consistent.

## Gate 3: frozen forecasting value

Use untouched chronological origins and frozen coefficient-one predictions.
The primary endpoint is out-of-sample loss relative to the pre-fitted expanded
anchor. Date is the dependence unit. Fitted explanatory R-squared, reversed
score coefficients, retrieval rates, and prompt acceptance are diagnostics,
not substitutes for deployable forecast improvement.

To identify topology, hold model, prompt, candidate count, article count, tool
calls, tokens, and retrieval steps fixed across OWN, matched extra news,
industry neighbors, true supply-chain neighbors, and degree-preserving shuffled
neighbors.

For a cumulative-radius study, freeze the same firm-period panel, target,
anchor, numerical controls, model, and selection policy. Compare `H1` (direct),
`H2` (through two hops), and `H3` (through three hops); report the paired
increments `H2-H1` and `H3-H2`. Mark an increment unidentified when its
out-of-sample contrast has no support or a regression term has zero rank.
Report exact-hop candidate and accepted counts, selected source overlap with
OWN, co-mention exclusions, source availability, graph vintage, token/tool
cost, and unsupported product links by radius. A cumulative-radius contrast
can show predictive value of wider retrieval, but a matched-cost graph versus
generic-news contrast is needed to attribute value to topology itself.

## Gate 4: causal supply-chain mechanism

Each variable must have these structured fields:

`name`, `economic_meaning`, `causal_role`, `entity_level`, `unit`,
`construction`, `observation_time`, `availability_time`, `source_id`,
`raw_source_fields`, `transformation`, `missingness_meaning`,
`point_in_time_required`, `predetermined_relative_to_shock`, `leakage_risk`,
and `dictionary_version`.

The contract must include, at minimum:

- `supplier_disruption_shock`
- `supplier_to_customer_exposure`
- `inventory_buffer`
- `supplier_redundancy`
- `customer_operating_cash_flow`
- `customer_sales_or_production_shortfall`
- `customer_future_return`
- `customer_future_volatility`

Required sequence:

`predetermined relationship -> supplier shock -> exposure -> buffer or
substitutability -> operational mediator -> cash-flow mediator -> market
outcome`.

Undefined meaning, missing availability time, missing-versus-zero ambiguity,
uncertified post-cutoff use, or topology-only evidence makes the causal gate
`BLOCKED`.

For h2/h3, bind the event fact to the endpoint and independent product or
order-continuity facts at each intermediate and at the focal target. Record
source IDs and availability times for each link. The current
`check_network_channel` implementation rejects paths above one hop; a
model-written receipt, ordered role sequence, or multiple graph paths cannot
replace that missing validation. A conditional direct receipt itself leaves
materiality and causal effect unknown.
