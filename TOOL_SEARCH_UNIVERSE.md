# Tool search universe and Python implementation

## What the agent can access

The analyst does not search a collection of Python scripts. The host exposes a finite set of read-only news/graph/state tools through MCP. Python scripts implement ingestion, selection, search, validation, and forecasting. Source files are provided here so the collaborator can inspect that implementation. The forecaster does not receive arbitrary Python execution, shell, filesystem, live web, URLs, or outcome-label access.

There are two layers of news retrieval: local deterministic preprocessing chooses candidates using native Reuters firm/topic tags, frozen graph candidates, recency, and local embeddings; the LLM chooses focused searches and article-page reads through the permitted tools. CPU search executes those queries. A title/search result alone is not a verified source passage.

## Completed 50-firm pipeline

| Python path under `source/code/` | Responsibility |
|---|---|
| `news_network_50_2025/config.py` | Firms, origins/splits, exact controls, pinned model, prompt/score contract |
| `news_network_50_2025/data.py` | Universe, numerical preparation, diary features, MiniLM ranking, scopes, counts, candidate pools |
| `news_event_risk_data_2025.py` | Source normalization/version metadata, byte-addressed article reads, daily-return union, risk inputs/response |
| `midcap_tools/prepare.py` | Parquet input reads and dated RIC-to-PERMNO mapping; legacy parser loader |
| `news_percentile/build.py`, `news_percentile/common.py` | Bounded Reuters parsing, native-tag extraction, topic/headline classifiers |
| `midcap_standard_2025/benchmarks.py`, `config.py` | Standard zero-mean GARCH fitting and variance floor |
| `news_event_calendar_2025.py` | Diary row parsing, receipt histories, conservative event timing |
| `news_exposure_2025/graph.py` | June 6 graph reconstruction, typed path traversal, July co-mention normalization |
| `news_exposure_2025/data.py` | Latest-version resolution and common-news eligibility |
| `news_network_50_2025/toolkit.py` | Original six native MCP tools and source-page read receipts |
| `news_network_50_2025/runtime.py` | Pinned model transport, MCP allowlist, disabled external tools, metering/cache |
| `news_network_50_2025/workflow.py` | Case/arm capabilities, OWN evidence and separate NET supplement, citation checks |
| `news_network_50_2025/optimize.py` | GEPA reflection/training/validation, residual-MSE objective, policy acceptance |
| `news_network_50_2025/metrics.py` | Training-only OLS anchor, explanatory and frozen forecasting comparisons |
| `news_network_50_2025/run.py` | Experiment orchestration; not executed by packaging |

The full source manifest includes additional local import dependencies and explicit dynamic-loader files. Byte-for-byte snapshots are distinguished from historical run versions by hashes. A hash mismatch is retained as a disclosed version difference, not silently described as the exact historical source.

### Initial scope construction

Resolve latest known Reuters versions across July-December indexes. Retain usable, non-excluded news with age at most 31 calendar days. Initial semantic ranking uses locally available `sentence-transformers/all-MiniLM-L6-v2` on headlines and three fixed queries; normalized cosine scores receive a `0.12*exp(-age/7)` recency term. This is headline semantic ranking, not a full-body vector database.

OWN candidates have native target tags. The own shortlist is at most ten, deduplicated by normalized headline family and availability day. Common macro/financial candidates require the implemented US/common filters and are balanced over five topic classes, at most two candidates per class. Ranking within a class adds `1.5*exp(-age/2)` and prefers stories at most eight calendar days old when available. Retained historical own events add up to twelve latest usable matching stories since July. The OWN capability contains the union of own/common/history IDs.

NET candidates exclude target-tagged and common stories and require at least one mapped neighbor tag. Initial capacity is twenty stories: twelve direct and eight higher-hop candidates, with remaining capacity filled if necessary. This is a candidate cap, not a global tool-turn cap. It does not mean every neighbor story is passed to the model or every article is read. Source/page eligibility is arm- and cutoff-specific. The original NET capability has supplement documents and frozen OWN factual context; it is not an unrestricted copy of all OWN documents.

The initial paragraph search is lexical within that stored shortlist. Article chunks preserve whole paragraphs, grouping at roughly 5,000 characters and splitting unusually long paragraphs at sentence boundaries. Page reads return source IDs, timestamps, page numbers, and paths. This design caused the original CFLT retrieval gap: a missing shortlist item could not be recovered by more shortlist searches.

## Native tools and later repairs

| Tool | Input / return purpose | Availability |
|---|---|---|
| `market_state` | Exact pre-cutoff numerical state, controls, calendar and anchor description | OWN and NET |
| `search_events` | Focused query, limit, hops/via; ranked scoped source/page leads | OWN and NET, different document capabilities |
| `inspect_event`, `inspect_events` | Source ID/page reads; batch at most six requests | OWN and NET |
| `event_logbook` | Source-backed historical search plus calendar; previous forecasts are not facts | OWN and NET |
| `exposure_paths` | Typed partner paths with query/pagination; higher hops need inspected intermediate | NET only |
| `inventory` | Available scopes, source categories, blocked paths and coverage | Later toolkit |
| `search_archive`, `inspect_archive` | Recover eligible sources outside the initial shortlist | Later toolkit |
| `network_leads` | Balanced direct-economic event discovery and focused related-firm search | Grounded NET toolkit |
| `record_facts`, `fact_ledger` | Bind exact inspected quotes to typed facts and return receipts | Grounded toolkit |
| `resolve_event_date` | Host-computed source-bound event intervals and forecast-window status | Grounded toolkit |
| `risk_check` | Host arithmetic for log-risk, variance/RMS ratios and forecast units | Later toolkit |
| `submit_own`, `submit_network` | Separate validated own decision and network-addition schemas | Grounded toolkit; one submission tool per arm |
| `financial_history` | Dated pre-cutoff observed statistics; no inferred historical earnings pattern | Separate evidence toolkit |
| `deal_terms` | Source-bound funding/consideration distinction and quoted-offer arithmetic | Separate evidence toolkit |
| `check_network_channel` | Typed graph direction, event/product/exposure prerequisite receipt | Separate evidence toolkit, NET only |

`dictionaries/tool_registry.json` gives method signatures and exact implementation locations, including inherited methods. It explicitly distinguishes the original six-tool registration from the grounded and evidence variants. Registration is not proof that every model used a tool correctly.

### Archive search versus semantic retrieval

`network50_archive_tools.py` builds a local SQLite FTS5 headline index of latest eligible own/common/network records within 31 days. It can read omitted full articles directly at verified byte offsets. Archive search is lexical headline retrieval, not semantic search over all bodies or all global Reuters news. Scope priority is own first, then common, then permitted network. Broad direct-network discovery uses a 14-day window, event-type balancing, target-topic overlap, and exclusions for routine market/broker headlines. Higher-hop news needs evidence through an inspected intermediate. A default twenty-distinct-network-article read budget is separate from the absence of a hard agent-turn limit.

`network50_grounded_toolkit.py` adds source, timing, relation and submission receipts; `network50_evidence_toolkit.py` adds stricter historical-reaction, deal-term and product-channel checks. Receipt checks constrain evidence prerequisites. They do not certify causal materiality, future risk magnitude, or predictive benefit.

## Optimization scope

The completed larger experiment uses 200 individual training firm-origins and 100 separate validation firm-origins. GEPA maximizes negative raw squared residual error: `-(y-anchor-score)**2`. Reflection batches have sixteen cases; two proposals per arm are allowed. Candidate text cannot change retrieval, controls, numerical response, or evidence contract. Separate validation compares paired losses on the two validation origins. Both OWN and NET retained their starting policies; the reported OPT columns are not independent improved forecasts.

The later grounded/evidence tools are engineering improvements, not secretly substituted optimized variants of the completed experiment. Source snapshots include their implementation for review, while keeping that attribution distinction explicit.

## Replication boundaries

The source snapshot is copied without executing it. Static imports and known dynamic-loader files are inventoried, but historical caches, upstream mapping builders, proprietary data and provider credentials are not included. Full replication needs those separately. Safely running the synthetic tests needs only Python's standard library. Executing a production runner requires configuration review and can incur model usage; packaging does not do so.
