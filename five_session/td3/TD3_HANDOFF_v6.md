# TD3 handoff v6 (TD3F): provider-selector pilot, sixth round (narrow fix)

Written 2026-11-05T14:52:33Z (from `date -u`). **Exploratory engineering pilot.** Status: the one blocking finding of
the TD3C5 review (D1: a shown partner's unit named with the partner's two-letter initialism) is fixed by one private
table entry and one general rule (a firm ticker or initialism followed by a corporate-unit word), with residual, verify
and scanner checks, tests with invented firms and one line in the limitation text. Everything else is unchanged from
v5. The inputs were rebuilt as `build_v7`: **only rank 46 changed in content** (199 inputs differ only in the version
header); `PILOT_SAMPLE.csv` is byte-identical to build_v5/build_v6. Tests 125/125, verify passed with problems `{}`, a
reader subagent re-read rank 46, mock dry run of 660 units, `FREEZE_TD3_v6.json` and a clearance **request**
`CLEARANCE_REQUEST_TD3_v6.json`. **No DeepSeek or Jev call of any kind was made** and the real-run mode was not run.
The v6 runner accepts only `D:/OptResearch_data/five_session/clearance/TD3_CLEARANCE_v6.json` and refuses while it is
absent (`status`: `CLEARANCE_VALID False ['clearance file absent: ...']`). v1 to v5 must never be cleared.

Examples in this file are invented. Figures marked "estimate" are estimates.

## Diff from v5

**Code** (`five_session/td3/v6/`, copied from `v5/`; lines added/removed against v5):

| File | Change |
| --- | --- |
| `td3_mask.py` (+63 -1) | New constant `UNIT_WORDS` (frozen corporate-unit words with a source note: the TD3C5 list plus singular forms); new functions `unit_prefixes(codes)` (code prefixes whose rule is `ticker` or `initialism`, >= 2 letters, not on `CODE_PREFIX_STOP` or `ACRONYM_STOP`), `unit_regex(prefixes)`, `unit_ok(m)`; `Masker.__init__` builds `unit_rx` from the alias key `unit_prefixes`; new `Masker._unit_pass` (after the name rules and the embedded/misspelling pass, before the product loop); `Masker.mask` calls it; `Masker.residual` reports `initialism_unit`; docstring |
| `td3_build.py` (+26 -8) | `aliases_for` adds `unit_prefixes`; `review_findings_check` checks the TD3C5 finding (`REVIEW_FINDINGS_C5`); `verify` reads `unit_prefixes`, counts the residual kind `initialism_unit` as `d1c5_initialism_unit`, requires the TD3C5 check, lists the new checks; docstrings |
| `td3_scan.py` (+38 -0) | New `initialism_unit` check (`_LEGAL_FORMS`, `_SCAN_UNITS`, `FirmTerms.unit_terms`, a loop in `scan_article`): a ticker or name initialism (descriptive words kept, legal forms dropped) followed by a corporate-unit word or legal suffix |
| `td3_spec.py` (+45 -22) | `VERSION` v6, `INPUT_VERSION` `TD3-input-v7`, clearance/freeze/request/qualification names v6, `FREEZE_TD3_v5.json` added to `SUPERSEDED_FREEZES`, `MASK_TABLES` -> `tables_v6/MASK_TABLES_v4.json`, new `REVIEW_FINDINGS_C5`; `MASKING_RULE` (unit-name sentence, acceptance), `MANUAL_READ_RULE` (v6 sentence), `LIMITATION_CONTEXT_IDENTIFICATION` (one line), `LIMITATION_EXAMPLES` (+2: "ZB Ventures sold its stake" -> "[X] sold its stake"; "ZRBX Ventures rose" -> "[X] Ventures rose", the ticker being masked first by the ticker rule) |
| `td3_freeze.py` (+19 -12) | `manual_read_check` requires `must_read` (audit-listed inputs) read in final content; supersedes reason v1-v5; binds the TD3C5 findings file; v6 messages |
| `td3_runner.py` (+7 -4), `run_td3.py` (+9 -9) | v6 docstrings and messages; default `--tag v7` |
| `td3_tests.py` (+61 -14) | `aliases()` helper adds `unit_prefixes`; new `TestV6UnitNames` (2 tests) and `TestIndependentScanner.test_scanner_v6_initialism_unit`; `test_v6_paths_and_versions`; v6 clearance names; `must_read` assertions; the two new limitation examples. 125 tests (+3) |
| `td3_providers.py`, `td3_ledger.py`, `td3_report.py`, `td3_runtime.py` | Unchanged |

**Tables** (private): `tables_v6/MASK_TABLES_v4.json` (sha256 `ba2fdc7b814faf9e5199b4054861eddf99282f4d956d9b9ad1a213243725fb8b`) = `tables_v5/MASK_TABLES_v3.json` plus one `common_names` entry (the TD3C5 unit name under its firm's comnam prefix, with the audit's source note) and updated title/review/notes fields; no other entry changed (41 common-name, 48 firm-products, 13 fixed-phrase, 7 code-prefix entries). New `tables_v6/TD3C5_FINDINGS.json` (1 item, sha256 `e966cda49fd62b757ad1a638612ee00f284a0785e9677234266404e19e9f2a45`).

**Inputs**: `build_v7/INPUTS.jsonl` sha256 `b4de0b6ddfee40b0dd096c2e8bb583282e8f6ec2d104183b437f68282df2ba81`. Changed in content: rank 46 only (one body line: the unit name became the firm's token). 199 inputs differ only in the header `TD3-input-v7`. `SCAN_HITS.jsonl` is identical to build_v6; `ADJACENCY.jsonl` (5,755 rows, +1) and `ADJACENCY_SWEEP.jsonl` (1,966 rows, +1) differ only by the rank 46 token.

## Verification

| Check | Result |
| --- | --- |
| Tests (`run_td3.py qualify`) | 125/125 (`QUALIFICATION_TD3v6_55426cf53c0945e3aab945321a67d047.json`) |
| Strict residual (incl. the new `initialism_unit` check) | 0 hits |
| `verify --tag v7` | passed, problems `{}`; 2,606 articles re-masked from raw text and identical; 1,966 sweep rows recomputed with hidden words and identical; findings TD3C 79, TD3BC 19, TD3C3 32+3, TD3C4 15, TD3C5 1: 0 present |
| Independent scanner (`td3_scan` v6) | 143 hits (focal 4, shown 118, link 21), identical to build_v6, all false positives (adjudications carried after re-read); 0 `initialism_unit` hits. Positive control: the v6 scanner run over build_v6 reports the TD3C5 unit name (`dev_v7/control_unit_rule.py`) |
| General rule alone | With the table entry removed, the v6 rule masks the rank 46 unit name (`initialism_unit`), same control |
| Adjacency sweep | 1,966 rows: 1,965 identical to build_v6 groups (reasons carried after re-read), 1 new (the rank 46 token, whole unit name masked); 0 glued, 1 number neighbour (unchanged ordinal); 0 give back |
| Reads | Rank 46 re-read end to end by a reader subagent that did not write the masker (on `dev_v7/r1`, byte-identical to build_v7): 0 A, 0 B, 1 C (the intended whole-name mask, checked against the raw text). No other input changed, so no other read was required; the earlier end-to-end reads (reviewers TD3C-TD3C5 and the v5 readers) cover all 200 inputs |
| Public folder (v6 code) | 0 eight-word and 0 six-word windows of any raw or masked sampled article; no sampled firm name or ticker introduced (new words are general unit vocabulary) |
| Dry run | 660 mock units, no halt, mock-ledger total USD 0.153 |

## Exact commands (as run, PowerShell, from `five_session/td3/v6`)

```
$PY = 'C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe'
& $PY -B run_td3.py qualify              # 125/125 -> QUALIFICATION_TD3v6_55426cf5...json
& $PY -B run_td3.py build   --tag v7     # build_v7 (needs tables_v6/MASK_TABLES_v4.json)
$D = 'D:/OptResearch_data/five_session/td3'
& $PY -B "$D/dev_v7/make_scan_adj7.py"     "$D/build_v7" build_v7
& $PY -B "$D/dev_v7/make_sweep_review7.py" "$D/build_v7" build_v7
& $PY -B "$D/dev_v7/make_manual_read7.py"  "$D/build_v7" "$D/dev_v7/reads" "$D/dev_v7/r1"
& $PY -B run_td3.py verify  --tag v7     # passed, problems {}
& $PY -B run_td3.py dry-run --tag v7     # 660 mock units, no halt
& $PY -B run_td3.py freeze  --tag v7     # FREEZE_TD3_v6.json + CLEARANCE_REQUEST_TD3_v6.json
& $PY -B run_td3.py status               # CLEARANCE_VALID False (file absent)
```

## Freeze, request and the clearance the runner requires

- `FREEZE_TD3_v6.json` sha256 `3e1f12028d7acdc63939c718d287758f3c21b5108dd5ca81b25c3de3de7c1c7a`.
- `CLEARANCE_REQUEST_TD3_v6.json` (request only) sha256 `1626c9e3411094c674039c9d0b95ca9d47a6ad9329338bec0a584f40ac5c8b3e`.
- Required: `D:/OptResearch_data/five_session/clearance/TD3_CLEARANCE_v6.json`, schema `TD3_CLEARANCE_v6`, `status`
  CLEARED, no `request_only` key, with the request's fields (`freeze_sha256` above, `code_sha256` of the 12
  `td3/v6/*.py` files, `inputs_sha256` above, `question_sha256`
  `8e1284dd158ec26d1ba94880434319f0e4b0327c8226f5e6d2cb0a831132c03a`, `mask_tables_sha256` above, `max_usd` 5.0,
  `max_jev_requests` 600, `deepseek_model` `deepseek-flash`, `deepseek_thinking` `{"type": "disabled"}`,
  `deepseek_returned_model_allowlist` `["deepseek-flash"]`, `jev_model` `jev-1.14.0`, `jev_price_input_per_m` 0.042,
  `probability_sum_tolerance` 0.03). Editing or adding any `td3/v6/*.py` file, or changing the inputs or the table,
  invalidates it.

## Cost, memory and time

- Dry-run cost estimate for the real pilot (660 requests): DeepSeek USD 0.097 off-peak / 0.195 peak, Jev USD 0.056,
  total about USD 0.15-0.25 (estimate); pre-call upper bound of all first attempts USD 0.51; joint cap USD 5.00 and
  600 Jev requests (unchanged).
- Measured on this machine: build 188 s, peak 747 MB private (782 MB working set); tests about 3 s; verify about 4
  minutes; dry run about 1 minute. Real run: roughly 15-35 minutes plus retries (estimate).

## Private outputs

`D:/OptResearch_data/five_session/td3/`: `tables_v6/`, `build_v7/` (13 files), `FREEZE_TD3_v6.json`,
`CLEARANCE_REQUEST_TD3_v6.json`, `QUALIFICATION_TD3v6_55426cf5...json`, `runs/dryrun_1a34aa57fba5/`, `logs/`
(`qualify_v6.log`, `build_v7.log`, `verify_v7.log`, `dryrun_v7.log`, `freeze_v6.log`, `status_v6.log`, dev logs) and
`dev_v7/` (dev build `r1`, byte-identical to build_v7 in inputs, metadata, sweep, adjacency and scan hits; the rank 46
read batch, notes and adjudication; positive controls; the builder's tools).

## Do not push

Push only `td3/v6/` and this file, after the reviewer's check. Do not push: the v1 files and `TD3_HANDOFF.md` at
`td3/`, `TD3_HANDOFF_v2.md` to `TD3_HANDOFF_v5.md`, the `td3/v2/`, `td3/v3/`, `td3/v4/` and `td3/v5/` folders
(superseded; v2-v4 hold one sampled drug code in a comment) and the `td3/__pycache__/` and `td3/v2/__pycache__/`
folders. None of them was edited or deleted in this round.

## Disclosures

- Nothing was deleted or edited that existed before this round, and no file created in this round was deleted. No
  network use, no git command, no WRDS connection, no 2025 data, no label, return or loss; no key value was read;
  nothing was written under `D:/OptResearch_data/five_session/clearance/`.
- The unit rule runs after the name rules, so it acts only where the ticker or initialism is still readable; a ticker
  already masked by the ticker rules keeps the unit word after its token (stated in the masking rule and the
  limitation line). This placement is why only rank 46 changed.
