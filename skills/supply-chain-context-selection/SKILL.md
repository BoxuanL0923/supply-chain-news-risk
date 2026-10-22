---
name: supply-chain-context-selection
description: Select point-in-time, source-backed supply-chain evidence for a bounded forecast context. Invoke explicitly for context selection, never for a causal claim.
---

# Supply Chain Context Selection

Use the supply-chain-context-selection skill when ranking inspected evidence.

1. Rank source IDs by an operating mechanism that could change five-session risk.
2. Distinguish supplier disruption, customer demand, shared bottleneck, and routine news.
3. Keep uncertainty, mitigations, and contradictory evidence visible.
4. Pass the ranked IDs to the Python `select_context` tool. The tool enforces the
   as-of cutoff and event budget; do not fabricate selected context or cite unread sources.
5. Treat missing evidence as unknown. A graph path alone does not prove transmission.

Only training feedback may revise the ranking instruction. Validation chooses a
candidate; held-out labels must stay sealed until final evaluation.
