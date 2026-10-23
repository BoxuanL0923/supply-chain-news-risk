---
name: supply-chain-context-selection
description: Select point-in-time, source-backed supply-chain evidence for a bounded forecast context. Invoke explicitly for context selection, never for a causal claim.
---

# Supply Chain Context Selection

Use the supply-chain-context-selection skill when ranking inspected evidence.

Execution boundary: the DSPy `ContextProgram` agent must call
`use_supply_chain_context_skill` to read this reviewed local file before it
calls the Python `select_context` tool. The host rejects a missing or
out-of-order call. This happens after the evidence host has called graph and
article tools. Steps 1
and 2 describe evidence that the host must obtain and verify; the selector
cannot itself call those tools. The native retrieval agent currently has its
own instructions in `Runner.evidence`. This file is not a hosted Skill mount.

1. Start from the focal company and its dated supply-chain graph. Use
   `exposure_paths` for typed supplier, customer, or competitor direction;
   a co-mention is only a search lead. For a third-hop event, inspect a direct
   neighbor, then the second-hop intermediate, before querying the third hop
   with that inspected second-hop entity as `via`. Search and inspect the
   related-company event with source ID, publication time, and an exact passage.
2. To claim a product or input channel, bind both the neighbor event fact and
   separate target-exposure fact, then use `check_network_channel`. A
   `conditional_channel` receipt is a prerequisite for a mechanism-specific
   description; `path_only` remains an unverified retrieval lead. Two- and
   three-hop paths need product continuity at every intermediate, not just
   a sequence of graph edges. The current `check_network_channel` receipt
   validates a direct path only; do not promote a multi-hop lead with it.
3. Rank source IDs for the five-session **log mean squared return** forecast.
   Read the Python-projected `role_motif`, `finance_hypothesis`, `evidence_gap`,
   `alternative_route`, `path_pattern`, `path_count_bin`, `mechanism_hint`,
   and `event_state` together. Apply these rules:

   | Graph/event combination | Permitted ranking interpretation |
   | --- | --- |
   | All-supplier path plus supplier disruption | `upstream_input_risk` is a hypothesis for focal input availability, including h2/h3, with product continuity still unverified. |
   | All-customer path plus customer demand | `downstream_demand_risk` is a hypothesis for sales or order exposure; no sign or magnitude follows. |
   | Direct competitor plus substitution event | `competitive_substitution_risk` is a hypothesis; competitor-containing multi-hop paths are ambiguous. |
   | Mixed direction, conflicting shortest motifs, or unaligned event | `unaligned_or_ambiguous`; do not invent a finance mechanism. |
   | Co-mention or no typed path | Retrieval lead or unlinked event only; never a supply edge. |

   `alternative_route` records observed path topology, **not** supplier
   redundancy or substitution capacity. `path_count_bin=multiple` does not
   measure independent routes. A FactSet supplier revenue percentage is not
   the focal firm's procurement dependence. `event_state=resolved` needs
   source-backed remaining exposure before priority rises. Preserve
   uncertainty, mitigations, and contradictions; do not infer the sign of a
   future stock return.
4. Pass the ranked IDs to the Python `select_context` tool. The tool enforces the
   as-of cutoff and event budget; do not fabricate selected context or cite unread sources.
5. Treat missing evidence as unknown. `path_only` always remains an unverified
   candidate, including when a model-provided row claims a channel receipt.
   The selector cannot validate the product link. A graph path alone does not
   prove transmission. Do not describe this volatility target as directional
   stock returns.

Only training feedback may revise the ranking instruction. Validation chooses a
candidate; held-out labels must stay sealed until final evaluation.
