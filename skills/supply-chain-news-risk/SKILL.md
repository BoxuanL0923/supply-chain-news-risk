---
name: supply-chain-news-risk
description: Audit this repository's forecasting data dictionary, point-in-time evidence contract, historical reproducibility, or proposed causal supply-chain mechanism. Use only when explicitly invoked; do not use it to run production forecasts or treat graph paths as causal evidence.
---

# Supply Chain News Risk

Invoke explicitly as `/supply-chain-news-risk` in Claude Code or
`$supply-chain-news-risk` in Codex.

## Workflow

1. Identify the requested claim: completed-experiment documentation, exact
   historical replication, frozen forecasting value, or causal transmission.
2. Run the deterministic validator before interpreting the package:

   ```bash
   python3 skills/supply-chain-news-risk/scripts/validate_dictionary.py \
     --repo . --mode forecasting
   ```

   Use `--mode causal` for any shock-propagation or mechanism claim.
3. Read only the files needed for the requested claim. Start with
   `DATA_DICTIONARY.md`, `dictionaries/variables.csv`,
   `metadata/verification.json`, and `metadata/source_manifest.csv`. Read
   `TOOL_SEARCH_UNIVERSE.md` only for retrieval or tool questions.
4. Apply the claim gates in [references/data-contract.md](references/data-contract.md).
   For a network result, record whether the candidate came from an exact
   typed hop (1, 2, or 3), a co-mention, or no graph path. Check the event's
   publication time against the forecast origin and the graph vintage against
   the same origin. A path motif and a finance hypothesis are selector inputs,
   not evidence that the affected product reaches the focal firm.
5. Report the repository commit, validator mode, qualified findings, blocking
   gaps, and the smallest next action. Distinguish verified checks from author
   assertions and untested code.

## Non-negotiable boundaries

- Treat `metadata/verification.json: passed` as a package check, not a result
  replication, forecasting improvement, or causal result.
- Do not claim exact historical replication when a core source hash differs
  from the recorded run.
- Missing is not zero, lack of retrieved news is not no event, and a graph path
  is not a transmission mechanism.
- A direct `conditional_channel` receipt does not certify a second- or
  third-hop path. For h2/h3, demand source-bound affected-item continuity at
  every intermediate before discussing a product channel. Do not convert a
  supplier revenue percentage into focal-firm input dependence or call
  multiple paths measured supplier redundancy.
- To claim an incremental h2 or h3 forecast contribution, use the same test
  firm-periods and frozen target/anchor, report `H1`, `H2-H1`, and `H3-H2`
  with missing and zero increments distinct, and disclose different retrieval
  cost. A fitted regression or synthetic test alone is not held-out value.
- A causal claim requires a measured shock, predetermined exposure, buffer or
  substitution measure, operational mediator, cash-flow mediator, and market
  outcome, each with observation and availability times.
- Do not run production runners, contact a model provider, open licensed raw
  data, or read outcome/score files unless the user explicitly authorizes that
  action. The validator itself reads only dictionaries and metadata.
- Return `BLOCKED` rather than infer a field's meaning, timing, or provenance
  from its name.

## Output contract

Return four short sections: `Verdict`, `Verified`, `Blocking gaps`, and
`Next action`. Use `PASS`, `QUALIFIED`, or `BLOCKED`; never collapse those
statuses into a generic success statement.
