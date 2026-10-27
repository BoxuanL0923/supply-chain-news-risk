---
name: sc-acceptance-ledger
description: Run deterministic mechanism screens in host execution order, stop at the first failure, and append a verified hash-chained decision.
---

Invoke explicitly:
`/sc-acceptance-ledger evaluate one candidate; inputs: config JSON, gate input paths and ledger path; script: skills/sc-acceptance-ledger/scripts/acceptance_ledger.py; constraints: validation >1 SE, sign then >=4/7 years, peer advantage >0, 100 rewires p<0.05; deliverable: append-only JSONL row and head hash; verify existing chain and individual gate receipts; stop at first failure or after one append.`

Before execution state the required output, verification, and stop above.
Run the standard-library orchestrator with the four sibling gate skills installed:
`python skills/sc-acceptance-ledger/scripts/acceptance_ledger.py --config acceptance.json --ledger D:/review/decisions.jsonl`
Verify without running gates:
`python skills/sc-acceptance-ledger/scripts/acceptance_ledger.py --ledger D:/review/decisions.jsonl --verify-only --expected-head TRUSTED_SHA256`

Config names a unique `attempt_id`, `profile` (default `agent-v4`),
`validation` CSV/Parquet path, `sign_stability` JSON path, `industry_peer`
`{rows,contract}` paths, and `rewire_placebo` JSON path, relative to the config.
Exposed controls: `se_multiplier:1`, `required_years` (v4/loop 4, O2 5),
`t_cutoff:1.96` (loop), `draws:100`, `placebo_threshold:0.05`, `two_sided:false`.
Keep declaration/rationale identical across sign and placebo inputs.

Sequential order is **a validation -> c sign -> d stability -> e peer -> b
rewire**, the actual v4/legacy host scheduling. Guideline 8.3 lists a,b,c,d,e;
the expensive placebo is last in the host. O2 uses the same wrapper schedule
with its explicitly selected policy thresholds. Do not open later gate input
files after failure. Record later checks as null. Removal's nonworsening-only
exception is outside this candidate-addition/replacement skill.

Before append verify every existing index, previous hash and canonical row
hash. Use an exclusive fail-fast writer lock, append/flush/fsync only, and
refuse duplicate attempt IDs. Partial/corrupt chains, foreign locks and a
mismatched expected head stop without an append. Never repair or truncate.
A chain alone cannot detect loss of a valid suffix; save the returned head
outside the ledger and supply `--expected-head` on subsequent runs.

Required output: ACCEPT/REJECT/ERROR, all five checks, first failure, executed
receipts, input/config/implementation hashes, previous hash and new row hash.
Stop after one append; exit 0/1/2 = accepted/rejected/error. These are screening
receipts on supplied sufficient statistics, not a production freeze or
verification of private full-calendar/refit provenance.
[Schemas, source policy differences and example](../../docs/SKILLS_AND_ROUTER.md).
