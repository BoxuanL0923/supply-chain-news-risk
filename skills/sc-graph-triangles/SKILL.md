---
name: sc-graph-triangles
description: Compute dated target triangles statistics with Python, explicitly invoked for supply-chain context ranking, with exploration bounded to three hops.
---

Invoke explicitly:
`/sc-graph-triangles compute target triangles; inputs: firm_id, cutoff, dated directed snapshot with monthly_universe, max_hops, inspected_prefixes; script: skills/sc-graph-triangles/scripts/graph_triangles.py; constraints: strictly pre-cutoff edges, max_hops <= 3, inspected exact prefixes; deliverable: JSON statistics and verification receipt.`

Execute the self-contained repository script (standard library; shared numerical implementation in `source/code/news_network_50_2025/graph_skills.py`):
`python skills/sc-graph-triangles/scripts/graph_triangles.py --snapshot SNAPSHOT.json --firm-id ID --cutoff ISO_DATE --max-hops 1`
For deeper paths pass `--inspected-prefixes prefixes.json`. Triangles accepts `--neighbour ID`.

Expose numerical thresholds in the task: `--thresh-hub 90`, `--thresh-ra 75`, `--thresh-clust 0.5`. RA >= THRESH_RA (default 75th percentile of target-to-month-universe pairs), with RA > 0, flags specialized shared exposure. Clustering >= THRESH_CLUST (default 0.5) flags common exposure: do not count neighbour news as independent evidence. Triangles/clustering are not robust predictive effects. A nonadjacent pair has shared partners but zero closed pair triangles.

Success before execution: require JSON containing model, cutoff/vintage, target statistics, path hop counts and truncation flag; degree also requires in/out/total degree, monthly percentile, hub flag and degrees of returned-path neighbours; triangles requires triangle count, clustering, pair shared count, RA and thresholds. Verify graph vintage and every edge's vintage/valid_from precede cutoff; expired edges are excluded, duplicates and self loops do not inflate statistics. Check every deeper path's exact prefixes have inspected source receipts. Stop after one valid receipt, or report the error/coverage gap. Never expand beyond three hops; a truncated result is incomplete, not no exposure. The host supplies the monthly universe; never substitute the retrieved sample. Degrees include supplier and competitor links in the supplied snapshot, with reciprocal directed partners counted twice in total degree. RA uses unique undirected degree. Graph topology alone does not verify a product channel.

Model label: qwen3.8-27b; numerical computation itself is deterministic Python. Engineering only; this model is not look-ahead-clean.
