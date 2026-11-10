# TD4 v2 handoff: provider selectors, selection on 2021-2022, test on pooled 2023-2025, masker v2 with span-level redaction

Written 2026-11-07T09:57:03Z (from `date -u`). Builder: Claude subagent. Nothing real has run: no provider call, no
2021-2025 outcome read, no clearance file written, no git. Code: `td4/v2/` (new files; every v1 file and every file
bound by `FREEZE_TD4_A_v1.json` is byte-identical). Private outputs: `D:/OptResearch_data/five_session/td4/`.

## 1. Scope

- Amendment 2 (researcher, 2026-11-06 about 12:30 EST): frozen TD1 `model_frozen_v2.pkl` and TD2 `MODEL_v2.json`, no
  refit; selection on a 20,000-firm-week hash-ranked subsample of the TD3-eligible 2021-2022 population with TD2's
  protocol (TD1 `selection_mlp` b, TD2 2015-2020 training heads); test on every eligible firm-week of 2023, 2024 and
  2025, evaluated once on the pooled years (clearance B); P1 and P2 as pre-registered; descriptive secondary S1;
  Codex labels for the diagnostics (LLM labels from OpenAI, not human ground truth).
- Masking reviews td4m M1-M4 (round 1, FIX REQUIRED): masker v2 implements every M1-M4 class (section 3); every period
  is rebuilt with it. No input built with the v1 masker is bound.
- Masking reviews td4m2 MA1 and MA2 (round 2, 2021-2022 masker-v2 inputs, FIX REQUIRED). Researcher decision 2026-11-06
  about 18:40 EST: masker v3 is skipped (record only, `masker_v3_record/`; no v2 path imports it).
- The whole-input detector-flag rule (2026-11-06) would have withheld 33.2% / 58.2% / 61.8% / 63.4% of the callable
  inputs (2021-22 / 2023 / 2024 / 2025); it went back to the researcher (`WITHHOLDING_DECISION_NOTE_v1.md`) and was
  **not applied**. **Researcher decision 2026-11-07: span-level redaction** (the note's option 2), section 6b: every
  detector-matched string is replaced in article text by a placeholder, for every arm alike, in at most three rounds of
  redact-and-re-run; the remainder is withheld.
- Code review TD4C (interim): B1, R1, N1, N2, N4, N5, N7, N8, N9 fixed in v2; status of every item in section 7.
- Coordinator 2026-11-06 19:30Z: the CRSP CIZ 2025 security-name pull is an extra alias source for the 2025 inputs
  (population check: no 2025-only name or ticker of a focal or shown firm readable).

## 2. What was built

Public code, `td4/v2/` (all new; nothing runs on import):

| File | Role |
| --- | --- |
| `run_td4v2.py` | entry point (offline commands; `call`/`reask` online, clearance A v2) |
| `td4v2_spec.py` | constants, periods, pre-registration text (written to `PREREGISTRATION_v2.json`) |
| `td4v2_mask.py` | masker v2: `MaskerV2` subclasses the TD3 v6 `td3_mask.Masker` |
| `td4v2_words.py` | frozen word tables |
| `td4v2_build2.py`, `td4v2_names2025.py` | masker-v2 builds of every period; 2025 with the CRSP CIZ 2025 name source |
| `td4v2_scan.py`, `td4v2_verify2.py`, `td4v2_rename.py` | independent v2 scanner, population checks, audit samples, 2025-only-name check |
| `td4v2_flags.py` | the reviewers' detectors: copies, run script, extraction of flagged classes (round 0 of the redaction) |
| **`td4v2_redact.py`** | **span-level redaction**: locators per detector, rounds, ledger, detector re-runs, residual flags, invariant checks |
| **`td4v2_redact_final.py`** | **final redacted inputs and withheld set per period** (`REDACTION_FINAL_<UTC>.json`) |
| `td4v2_inputs.py` | inputs sent per period: the final redacted INPUTS minus population-check and redaction withholding; subsample |
| `td4v2_runner.py` | v2 calls (period order, N1 terminal 4xx, N2 model override, N4 numeric sessions, N5 input hash) |
| `td4v2_gate.py`, `td4v2_select.py`, `td4v2_evaluate.py` | clearances A/B v2 (R1), selection on 2021-2022, once-only pooled evaluation (with S1) |
| `td4v2_score.py`, `td4v2_outcomes.py` | selection/test frames and the label gates |
| `td4v2_labels.py` | Codex label packet, redacted packet (`labeling_v2r`), strict converter, scorer |
| `td4v2_cost.py`, `td4v2_freeze.py`, `td4v2_rehearsal.py`, `td4v2_parity.py`, `td4v2_parity2025.py` | cost, freeze/request, rehearsal, parity (incl. construction from redacted text) |
| `td4v2_build.py` | record key, hash ranks, candidate rule |
| `td4v2_tests.py`, `td4v2_tests_m2.py`, **`td4v2_tests_redact.py`** | tests (invented fixtures) |

Private (`D:/OptResearch_data/five_session/td4/`): `inputs_m2/<build>/` (masker-v2 builds, population checks, audit
samples, round-0 `DETECTOR_FLAGS_M2_<UTC>.json`, `REDACTION_FINAL_<UTC>.json`); `inputs_redacted/<period>/round_<r>/`
(INPUTS, LEDGER, ROUND, VERIFY, RESIDUAL per round; the round-3 INPUTS are sent); `detector_flags/<period>_<UTC>/`
(round 0) and `detector_flags/<period>_rd<r>_<UTC>/` (re-runs; copies, `RUN_DETECTORS.sh`, `run.log`, outputs);
`labeling_v2/` (Codex packet, unchanged) and `labeling_v2r/` (redacted versions of the changed contexts); `parity/`;
`proof_m2/`; cost, prereg, freeze, request. Records, not read by v2: `inputs_m2r/` and
`detector_flags/2122_rd1_2026-11-07T063525Z/` (a first round-1 attempt, stopped after a locator fix, section 6b);
`inputs_m2r_dev/` (development run of 3,000 records); `inputs_m2/select_2122`, `period_2023`, `period_2024` (stopped
first build attempts); `inputs_m2_dev/`, `inputs_m3_dev/`, `masker_v3_record/`.

## 3. Masker v2 (td4v2_mask; TD3 v6 files unchanged)

Standard (coordinator): blocking = a readable name, name part, initialism, ticker or firm code of a token firm; a
product name built from the firm name; a give-back phrase; a unit acronym or model number next to a token; any
calendar date or year residue. Owned brands without a firm word, executive names and unique descriptions stay the
disclosed TD3 limitation (executive names next to a token are counted by the v2 scanner, class S15).

Every rule has (a) a residual check: `MaskerV2.residual` re-applies every TD3 and v2 rule to the final text and adds
the v2 after-token checks (hyphenated own word, RIC suffix, model number, hyphen acronym, v2 unit forms);
`date_residuals_v2` adds the v2 date forms, story-field stubs and clock times with seconds; a residual withholds the
firm-week from all providers alike; and (b) an independent check in `td4v2_scan` (terms from the CRSP name files only).

| Review class | v2 rule |
| --- | --- |
| M1 B1, M2, M4 plural-English tickers | English-ticker guard needs `_is_english` (stem >= 3) and the common-word table; a ticker that is the firm's initialism is never kept |
| M2, M3 P4, M4 A1 month/English tickers heading a bullet | a stop-list, month or English ticker in subject position (bullet, BRIEF-, '[T] -') before a company verb is masked; in the firm's own article its tickers, initialisms, first core words and collapsed acronyms (stop lists included) are masked (upper case anywhere when not a common word; otherwise as a subject or before a company verb); a month ticker before a day number stays a date |
| M2, M3 P2, M4 Title-case tickers | Title-case form of a non-common ticker of >= 3 letters in the firm's own article |
| M2, M4 A1 two-letter initialisms | initials of the first two core words wherever the firm is a token (stop lists, ratings and units excluded); initialisms of the first k >= 3 name words; initialism + next name word |
| M1 B3, M3 P1, P9, M2 digit and core variants | words with digits in any case; single letter + next word; the first three words when the second is an initial; numbers before the first word; the core word after a generic first word; distinctive words also in lower case |
| M1 B4, M2 spaced names | camel-case (FactSet) and English-word splits of one-word names, with a space or hyphen between the parts |
| M1 B5 | 'U.S. <W>' from UNITED STATES names |
| M1 B6 | distinctive non-first name words (not English, not common, at most 5 CRSP PERMNOs carry the word); English ones by the private owner table |
| M1 B7 | ampersand initialisms from ' & ' names and ampersand first words |
| M1 B8 | v2 unit words and '<prefix> <Word> <legal suffix>' for non-English prefixes; a token followed by unit nouns and a legal suffix absorbs them |
| M1 B9, M2 own-article guards | the private index and exchange guard is bypassed for index words in the firm's own article unless an exact index name follows; stop lists do not protect a firm's own aliases in its own article |
| M1 B10, M3 P7 | leading parts of one-word names at a natural boundary (English remainder or camel boundary), distinctive |
| M1 B11 | the subject of an earnings-summary or preview template in an article chosen for exactly one firm |
| M1 B12, M4 A2, M2 brands | first core word or FactSet camel part at a camel boundary; in upper-case tokens when not a common word; distinctive name words followed by any letters; a ticker glued after an English word; in the firm's own article its first core word followed by any letters (token not English) |
| M1 B13, M3 P8 | an own name word joined to a token by a hyphen is absorbed |
| M3 P3, M4 A1 glued short names | first word or ticker of 2-4 letters glued to a prefix of the next name word or a frozen suffix list |
| M1 B2, M3 P6 codes | code prefixes = subsequences of a ticker or of the first name word that keep the first letter: >= 3 letters wherever the firm is a token, 2 letters in its own article |
| M2, M3 P5, M4 A3 give-backs | weekday and month words are never one-word aliases; a one-word alias absorbs an adjacent capitalised word in running text; dictionary-word aliases are kept in frozen collocations, two-word place phrases, Reuters slugs, index-movement lists and after adverbs in upper-case text; first core words that TD3 does not use alone are masked only in company context (possessive, company verb, 'according to', 'shares of') |
| M4 A4 | the private table's 'always' acronyms are active wherever the firm is a token and the firm's product acronyms in its own article (also after a hyphen, 'BRIEF-<ACR>'); a 3-digit aircraft model number after a token is absorbed |
| M2 RIC suffix | an exchange suffix after a token ('[T].XX') is absorbed |
| M2, M4 A5 dates | a dateline day glued to the next word; 'MONTH, D'; ordinals in any case; 'CY26', apostrophe years; '(Reuters) -' becomes a space |
| TD4C B1, M1, M3, M4 D8 | each unbracketed STORY_NUMBER/DATE/TIME field blanked by its own value pattern, equal length, `[ \t]*` only (no match where '[' follows); applied to every archive text before truncation in every build; residuals for label stubs and clock times with seconds |
| M1 advisory | mis-tag withdrawal notices are not eligible shown articles (the partner's next eligible article is shown) |
| M2, M4 rename gap | 2025 build: CRSP CIZ 2025 security names and tickers are added to every PERMNO's name rows; population check for 2025-only names (section 6) |

Firm-specific entries are in the private table `tables_v2/MASK_TABLES_TD4_v2.json` (source note per entry): common
names (unit and short names of nine firms), owner aliases (five firms, own articles only), one 'always' acronym, and a
three-row hand rename table that the 2025 build does not use (replaced by the CIZ source). The common-word table is
built from the 2021-2022 selection store (dated <= 2022-12-31): a word written in lower case at least 50 times is a
common English word.

## 4. Proof on the cited inputs and on the TD3 pilot

Private package `proof_m2/` (`PROOF_M2.json`, SHA256 af0b7d9582fc76811dca1efe77f1e3a973497fe5cede191d766d724e23d82572): every input the four reviews cite (M1 blocking
contexts and the 24 scan-found inputs, M3 non-clean inputs: 111 keys of 2024; M4 sample and M2 blocking or borderline
items: 169 keys of 2025) rebuilt with the final masker v2 (the 2025 keys with the CIZ source), plus the 200 TD3 pilot
firm-weeks with the 2021-2022 parameters.

| Class (cited surface forms) | v1 text | v2 text |
| --- | --- | --- |
| 2024 M1_B1_plural_english_ticker | 9 | 0 |
| 2024 M1_B2_M3_P6_code_subsequence | 14 | 0 |
| 2024 M1_B3_M3_P9_digit_names | 17 | 0 |
| 2024 M1_B4_M2_split_names | 2 | 0 |
| 2024 M1_B5_US_word | 25 | 1 |
| 2024 M1_B6_nonfirst_words | 29 | 0 |
| 2024 M1_B7_ampersand | 3 | 0 |
| 2024 M1_B8_unit | 1 | 0 |
| 2024 M1_B9_own_stoplist | 25 | 17 |
| 2024 M1_B10_M3_P7_prefix | 4 | 0 |
| 2024 M1_B11_template | 1 | 0 |
| 2024 M1_B12_glued_word | 1 | 0 |
| 2024 M1_B13_M3_P8_hyphen_remnant | 6 | 0 |
| 2024 M4_A1_initialism_ticker | 5 | 3 |
| 2024 M4_A2_compound | 0 | 0 |
| 2024 M4_A3_M2_giveback | 4 | 2 |
| 2024 M4_A4_unit_acronym_model | 3 | 0 |
| 2024 M4_A5_M2_dates | 0 | 0 |
| 2024 M2_renames | 0 | 0 |
| 2024 M2_title_ticker_core | 0 | 0 |
| 2024 D8_story_remnant | 58 | 0 |
| 2024 A1_mistag_notice | 38 | 0 |
| 2025 M1_B1_plural_english_ticker | 3 | 1 |
| 2025 M1_B2_M3_P6_code_subsequence | 0 | 0 |
| 2025 M1_B3_M3_P9_digit_names | 1 | 0 |
| 2025 M1_B4_M2_split_names | 3 | 0 |
| 2025 M1_B5_US_word | 3 | 3 |
| 2025 M1_B6_nonfirst_words | 9 | 0 |
| 2025 M1_B7_ampersand | 0 | 0 |
| 2025 M1_B8_unit | 0 | 0 |
| 2025 M1_B9_own_stoplist | 30 | 25 |
| 2025 M1_B10_M3_P7_prefix | 1 | 1 |
| 2025 M1_B11_template | 1 | 0 |
| 2025 M1_B12_glued_word | 0 | 0 |
| 2025 M1_B13_M3_P8_hyphen_remnant | 0 | 0 |
| 2025 M4_A1_initialism_ticker | 24 | 2 |
| 2025 M4_A2_compound | 11 | 0 |
| 2025 M4_A3_M2_giveback | 48 | 0 |
| 2025 M4_A4_unit_acronym_model | 14 | 2 |
| 2025 M4_A5_M2_dates | 23 | 0 |
| 2025 M2_renames | 7 | 0 |
| 2025 M2_title_ticker_core | 3 | 1 |
| 2025 D8_story_remnant | 77 | 0 |
| 2025 A1_mistag_notice | 45 | 0 |

Every form still found in a v2 text was checked against the input's token map: each belongs to a third party (the
firm is not a token of that input) or is a scanner false positive (a place or an ordinary word); none is a token
firm's name, ticker, code, unit or date.

TD3 pilot versus `build_v7` (182 of 200 firm-weeks identical; changes by reason {"td3:ric|v2:v2_ric_suffix": 1, "other": 8, "D_v2_dates": 1, "renumbering_or_whitespace": 13, "A_mistag_notice_dropped": 4, "td3:product_absorb|v2_absorb:name": 2, "v2_absorb:name": 2, "v2:init2": 2, "v2:core_ctx": 1, "v2:v2_camel_caps": 1}): every change is listed in `proof_m2/PILOT_DIFF_V7.txt` with its
reason (mis-tag notice replaced by the partner's next article; v2 date rule; RIC suffix absorbed; two-letter
initialism; capitalised neighbour word absorbed; product or unit word absorbed; own-article word; camel-case
compound; renumbered articles).

## 5. Inputs per period (masker v2, after redaction)

| Period | Eligible | Built | Withheld at build | Population check | Redaction rule | Withheld (union) | Sent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-22 | 20,400 | 20,394 | 6 | 15 | 1,184 | 1,199 | 18,814 |
| 2023 | 42,897 | 42,896 | 1 | 560 | 719 | 1,279 | 41,617 |
| 2024 | 45,942 | 45,934 | 8 | 184 | 4,010 | 4,194 | 41,740 |
| 2025 | 43,147 | 43,033 | 114 | 603 | 6,741 | 7,344 | 35,689 |

Selection subsample: 20,000 firm-weeks ({"input_built": 19994, "withheld_date_residual": 6}); of them 15 population-withheld and 1,165 withheld by the redaction rule (fallback rows); 18,820 sent. Mis-tag notices rejected as shown articles: 2021-22 341, 2023 2724, 2024 1678, 2025 5414.

The 2025 build ran with the CRSP CIZ 2025 alias source from its first chunk (receipt binds SHA256 70cf9725...); CIZ
impact: 41 of 1000 sampled inputs change (4.1%), 0 would be withheld only without it (file SHA256 8126d915e75d8c49e400c85b8ab3184cb6fc456018e3bed9488de5a7b5d93889).

## 6. Population checks and audit samples

`td4v2_verify2.population` runs over every input of every build: the TD3/v1 text checks (header, firm id, article
count, year, year glued to a time, URL, numeric date, story fields, glued role token, token-hyphen-digit), the v2
story-field stubs and clock times with seconds, and the independent v2 scanner. An input is withheld for every arm
alike when a text check fails or the scanner hits an exact or narrow class (S7 own-article code subsequence, S8
hyphenated name remnant, S9 give-back collocation/index list/slug/name remainder, S10 date or year, S11 story-field
stub, S12 mis-tag notice, S13 model number after a token, S14 RIC suffix after a token; for 2025 also S16, a 2025-only
CIZ name or ticker of a focal or shown firm). The other scanner classes (S1 tickers, S2 initialisms, S3 name words, S4
compounds, S5 digit names, S6 splits, S9 weekday slot) are candidates for the reviewers (`SCAN_HITS_M2*.jsonl`,
private); S15 counts executive names next to a token (disclosed limitation).

| Period | Inputs | Withheld | Problems (inputs) | Executive names next to a token (inputs, counted) |
| --- | --- | --- | --- | --- |
| 2021-22 | 20,394 | 15 | scan_S10_date 2, scan_S7_code_subsequence 6, scan_S9_giveback_name_remainder 3, token_hyphen_digit 6 | 796 |
| 2023 | 42,896 | 560 | scan_S10_date 110, scan_S7_code_subsequence 1, scan_S9_giveback_index_list 208, scan_S9_giveback_name_remainder 35, token_hyphen_digit 206 | 3,700 |
| 2024 | 45,934 | 184 | scan_S10_date 73, scan_S13_model_after_token 37, scan_S7_code_subsequence 1, scan_S9_giveback_name_remainder 27, scan_S9_giveback_slug 6, token_hyphen_digit 46 | 5,913 |
| 2025 | 43,033 | 603 | numeric_date_in_input 263, scan_S10_date 32, scan_S13_model_after_token 5, scan_S16_2025_only_name_readable 0, scan_S7_code_subsequence 1, scan_S9_giveback_collocation 27, scan_S9_giveback_index_list 18, scan_S9_giveback_name_remainder 63, scan_S9_giveback_slug 1, token_hyphen_digit 196 | 5,739 |

2025 CIZ check: {"permnos_with_2025_only_names": 2423, "inputs_withheld": 0}.

Two forms that masker v2 leaves are caught only by this check and withheld: a 'CYnn' year followed by a comma or period
(the v2 rule's lookahead excludes them) and a drug code whose prefix is on TD3's acronym stop list; both are counted in
the table (S10, S7).

Fresh audit samples: the 300 build-time audit records of each period ranked by SHA256('TD4-audit-m2|<period>|firm_id|
cutoff'); every article is re-masked from its raw text with `MaskerV2` (must be identical), the strict and date
residuals are re-run, and the TD3 scanner and the v2 scanner hits are written for the reviewers
(`AUDIT_SAMPLE_M2_<period>_<UTC>.json`, `AUDIT_SCAN_HITS_M2_<UTC>.jsonl`, private).

| Period | Audit inputs | Articles | Re-mask identical | Strict / date residual | TD3 scanner hits |
| --- | --- | --- | --- | --- | --- |
| 2021-22 | 300 | 4009 | yes | 0 / 0 | 184 |
| 2023 | 300 | 3777 | yes | 0 / 0 | 295 |
| 2024 | 300 | 3663 | yes | 0 / 0 | 263 |
| 2025 | 299 | 3591 | yes | 0 / 0 | 255 |

## 6b. Span-level redaction (researcher decision 2026-11-07; pre-registered)

**Transform.** `td4v2_redact` is a separate, deterministic post-processing step on the masker-v2 INPUTS (the masker and
its builds are unchanged), applied identically for every arm and reference selector before any provider call and any
outcome read. Every string matched by an included detector class (the classes, exclusions and article-part scope of
`td4v2_flags`) is replaced in article text only (the headline and text of each PARTNER NEWS article):

- `[REDACTED]`: name residues (names, name parts, tickers, RIC parts, initialisms, codes, glued compounds, story IDs,
  URL residues, mis-tag wording; M1, M2, M3, M4 and MA1 name classes, MA2 B1);
- `[DATE]`: date, year and story-time residues (M1 D_*, M2 year2/month_day, M4 DATE_MD/YEAR4/STORY_REM, every MA1
  date class except story IDs/URLs/mis-tag wording, MA2 D and D12);
- `[TERM]`: give-back slots, i.e. the one role-token occurrence that a give-back detector names (MA1 GB and IDX, MA2
  G1-G3, s41 token patterns, s41b, M2 surface, M3 P5, M4 GB_*). The slot is replaced as a whole token: this is the only
  place where a pseudonym-token occurrence in article text changes (each one is a ledger entry); every other pseudonym
  token, the header (instructions, risk menu, characteristics, graph state), the article labels and the field labels
  are byte-identical. A `[REDACTED]`/`[DATE]` span is trimmed to article text and never covers a bracket token.

**Location.** From each detector's own output: the matched string and its context window are re-found by rebuilding
the detector's context string at every candidate position (exact match; M1, M4, MA1 EMB/OWN/CO/PFX/IDX/REM, MA2 B1 and
D); the forms of `s11` (M2) by string search in s11's own article string; the key-only classes (s41, s41b, ma2_s12) and
the date classes of s11, s14, s15, s16 by re-running the detector's own regular expressions (read from the run copy;
every match in article text is replaced, not only the 8 the detector prints); GB groups by re-scanning every occurrence
with s17b's neighbour rule; MA2 G1-G3 by re-running MA2's raw-versus-masked word alignment. A diagnostic over every
round-0 hit located 100% of the text hits in every period (except header-text matches, which the flag rule does not
count). A flagged class with no located span in article text withholds the input; this covers the two M4 firm-level
classes (below). The reviewers' adjudicated 2021-2022 key sets are withheld.

**Rounds and "still flagged".** Round r = 1, 2, 3 redacts the matches of the detector run on the round r-1 inputs
(round 0 = the masker-v2 inputs and their detector-flag run) and then re-runs every detector on the round-r inputs
(copies that differ from the round-0 copies only in the INPUTS path and the output folder; population, metadata,
article store and withheld set of the masker-v2 build). An input is still flagged after round r when
`td4v2_flags.extract` (same classes and exclusions) lists it in the round-r re-run, or when a frozen GB group still has
a slot in its round-r text. Population-level detectors are frozen across rounds: an s18 GB group (permno, word, side,
neighbour) kept in any round so far stays a group and is re-scanned at every occurrence, so a group that falls below
s18's ratio after redaction cannot release its remaining slots. Per-class hit caps (s10 and s12 12, s14-s16 8, s17b 30
per class and input) only bound what one round locates from the outputs: an input with a further match is listed again
by the re-run and is still flagged. Inputs still flagged after the round-3 re-run are withheld for all arms alike.
A detector match that rests on a redaction placeholder is the redaction itself and counts neither as unlocatable nor
as still flagged: the M4 give-back patterns GB_* accept any bracket token in their slot, so after the second word of a
two-word place name (also a word of a token firm's name) was redacted, the pattern word followed by `[REDACTED]` matched
the firm-specific GB_* pattern again; such matches name no firm (`td4v2_redact_final.placeholder_dependent`; set when the round-1 residuals
showed them, before any round-3 result and before any outcome). Counting them would withhold the number in the
column "if placeholder-dependent flags counted".

**Per period.**

| Period | Callable | Inputs with >= 1 redaction | Redactions per input, all callable: mean / p50 / p95 (redacted inputs only) | Article characters redacted | Flagged before round 1 | Still flagged after round 1 / 2 / 3 | Final withheld (share) | Over 2% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-22 | 20,379 | 6,745 (33.1%) | 0.70 / 0 / 3 (2.11 / 2 / 5) | 211,798 of 100,231,331 (0.21%) | 6,741 | 66 / 0 / 0 | 1,184 (5.81%) | YES |
| 2023 | 42,336 | 24,397 (57.6%) | 1.61 / 1 / 6 (2.80 / 2 / 7) | 1,053,578 of 361,675,365 (0.29%) | 24,660 | 1,080 / 730 / 730 | 719 (1.70%) | no |
| 2024 | 45,750 | 26,562 (58.1%) | 1.45 / 1 / 5 (2.50 / 2 / 6) | 952,256 of 406,529,280 (0.23%) | 28,276 | 4,401 / 4,013 / 4,013 | 4,010 (8.77%) | YES |
| 2025 | 42,430 | 24,109 (56.8%) | 1.38 / 1 / 5 (2.43 / 2 / 6) | 793,166 of 344,495,176 (0.23%) | 26,895 | 8,134 / 7,326 / 7,256 | 6,741 (15.89%) | YES |

Why inputs are withheld (an input can fall under several):

| Period | Adjudicated 2021-22 key sets | Unlocatable (any round) | Still flagged after round 3 | Firm-level M4 flags (NONAME, NONAME_AFTER_2022) | Withheld without the firm-level classes (share) | Withheld if placeholder-dependent flags counted (share) |
| --- | --- | --- | --- | --- | --- | --- |
| 2021-22 | 1,184 | 0 | 0 | 0 | 1,184 (5.81%) | 1,184 (5.81%) |
| 2023 | 0 | 719 | 719 | 719 | 0 (0.00%) | 730 (1.72%) |
| 2024 | 0 | 4,010 | 4,010 | 4,010 | 0 (0.00%) | 4,013 (8.77%) |
| 2025 | 0 | 6,741 | 6,741 | 6,741 | 0 (0.00%) | 7,256 (17.10%) |

**Final withheld share above 2%: 2021-22, 2024, 2025.**
The two M4 firm-level classes NONAME (a token firm with no CRSP name row dated on or before 2024-12-31) and
NONAME_AFTER_2022 (a token firm whose first CRSP name row is after 2022-12-31) are round-1 detectors written for the
2021-2022 period: they flag the presence of a firm in the input, not a string, so redaction cannot clear them and every
re-run lists them again; under the rule ("a flagged class with no located span withholds") they withhold. In 2023-2025
they flag the firms that listed after 2022, i.e. the legitimate current firms of those years (the 2025 inputs were
masked with the CIZ 2025 names and the population check found no readable 2025-only name). The share without them is
in the column "Withheld without the firm-level classes"; the rule was not changed. For 2021-2022 the adjudicated key
sets (coordinator rule) are the whole withheld set: without them 0 inputs would be withheld
(0 in the subsample).

Replacements by placeholder (all rounds):

| Period | [REDACTED] | [DATE] | [TERM] | Ledger entries |
| --- | --- | --- | --- | --- |
| 2021-22 | 6,482 | 1,060 | 6,691 | 14,233 |
| 2023 | 35,174 | 3,936 | 29,134 | 68,244 |
| 2024 | 36,192 | 6,488 | 23,772 | 66,452 |
| 2025 | 33,230 | 5,501 | 19,848 | 58,579 |

Rounds (inputs, spans, unlocated inputs by class; invariant check of every record against its masker-v2 input and the
cumulative ledger, and the idempotency re-check):

| Period | Round | Inputs with a new redaction | New spans | Unlocated inputs (classes) | Verify (records, problems, idempotency) |
| --- | --- | --- | --- | --- | --- |
| 2021-22 | 1 | 6,741 | 14,167 | 0 (-) | 20,394, none, 0 new spans on 2,000 re-checked |
| 2021-22 | 2 | 66 | 66 | 0 (-) | 20,394, none, 0 new spans on 66 re-checked |
| 2021-22 | 3 | 0 | 0 | 0 (-) | 20,394, none, 0 new spans on 0 re-checked |
| 2023 | 1 | 24,391 | 67,875 | 719 (M4:NONAME_AFTER_2022 719) | 42,896, none, 0 new spans on 2,000 re-checked |
| 2023 | 2 | 356 | 369 | 730 (M4:GB_DEVELOP 4, M4:GB_ROUNDUP 7, M4:NONAME_AFTER_2022 719) | 42,896, none, 0 new spans on 356 re-checked |
| 2023 | 3 | 0 | 0 | 730 (M4:GB_DEVELOP 4, M4:GB_ROUNDUP 7, M4:NONAME_AFTER_2022 719) | 42,896, none, 0 new spans on 0 re-checked |
| 2024 | 1 | 26,554 | 65,997 | 4,010 (M4:NONAME_AFTER_2022 4010) | 45,934, none, 0 new spans on 2,000 re-checked |
| 2024 | 2 | 421 | 452 | 4,013 (M4:GB_DEVELOP 1, M4:GB_ROUNDUP 3, M4:NONAME_AFTER_2022 4010) | 45,934, none, 0 new spans on 421 re-checked |
| 2024 | 3 | 1 | 3 | 4,013 (M4:GB_DEVELOP 1, M4:GB_ROUNDUP 3, M4:NONAME_AFTER_2022 4010) | 45,934, none, 0 new spans on 1 re-checked |
| 2025 | 1 | 24,003 | 57,273 | 6,741 (M4:NONAME 1072, M4:NONAME_AFTER_2022 5832) | 43,033, none, 0 new spans on 2,000 re-checked |
| 2025 | 2 | 1,045 | 1,222 | 7,256 (M4:GB_LOGONE 622, M4:GB_ROUNDUP 1, M4:GB_THERMAL 2, M4:NONAME 1072, M4:NONAME_AFTER_2022 5832) | 43,033, none, 0 new spans on 1,045 re-checked |
| 2025 | 3 | 84 | 84 | 7,256 (M4:GB_LOGONE 622, M4:GB_ROUNDUP 1, M4:GB_THERMAL 2, M4:NONAME 1072, M4:NONAME_AFTER_2022 5832) | 43,033, none, 0 new spans on 84 re-checked |

Classes still flagged after round 3:

| Class still flagged after round 3 | 2021-22 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| M4:GB_DEVELOP | 0 | 4 | 1 | 0 |
| M4:GB_LOGONE | 0 | 0 | 0 | 622 |
| M4:GB_ROUNDUP | 0 | 7 | 3 | 1 |
| M4:GB_THERMAL | 0 | 0 | 0 | 2 |
| M4:NONAME | 0 | 0 | 0 | 1,072 |
| M4:NONAME_AFTER_2022 | 0 | 719 | 4,010 | 5,832 |

**Ledger and invariants.** Every replacement is a private ledger entry (key, article, field, offset and length in the
masker-v2 text, classes, placeholder, SHA256 of the original string, round; `inputs_redacted/<period>/round_<r>/
LEDGER.jsonl`); no plaintext is in a public file. `td4v2_redact.verify_round` checks every record of every round
against its masker-v2 input with an implementation independent of the renderer: outside the ledger spans the texts are
byte-identical; each span reads as its placeholder; the original string's SHA256 matches; a `[TERM]` span is exactly
one role token; a `[REDACTED]`/`[DATE]` span holds no bracket token and lies in one article content field; the header,
the article and field labels and every bracket token outside `[TERM]` spans are byte-identical and in order; record
hashes match. Idempotency: the round's locator re-run with the same detector outputs on the round's output finds no
further span. The request body is built by the unchanged TD3/TD4 construction code from the redacted text (section 9).

**Locator fix before the real rounds.** A first round-1 attempt (`inputs_m2r/`, stopped) left 26 2023 inputs
unlocated because the M2 `english_short_name` forms end with a verb that s11's pattern does not bound on the right
(a two-word form followed directly by a plural 's'); the locator drops the right-boundary test for that class and for forms cut at
s11's 40 characters, and the real rounds (`inputs_redacted/`) were run with the fixed module only (SHA256 in every
ROUND, VERIFY and RESIDUAL file and in freeze A v2).

**Label packet.** Of the 150 Codex-labelled contexts (`labeling_v2/`, unchanged), **38 change** under the redaction: (IDs in the private copy). Their redacted texts (the texts now sent) are in `labeling_v2r/` in the labeling_v2 batch format (same context ids; batches R01, R02, R03, R04; `PACKET_INDEX_v2r.json`, SHA256 ab11829799345d6c585de56dea88537e36f0967eca501fdfe44e453089eeb1c6) for the researcher's relabelling; Codex was not called. 7 contexts are withheld under the redaction rule and are excluded from the diagnostics: (IDs in the private copy). `score-labels --labels <v2 template>,<v2r template>` uses the relabels for the changed contexts and excludes a changed context without a relabel.

## 7. Code-review findings (TD4C interim) and status

| Item | Status |
| --- | --- |
| B1 D8 rule | fixed in v2: per-field value patterns, equal length, `[ \t]*` only, before truncation in every build; label-stub and clock-with-seconds residuals in the build and the population check (`td4v2_mask.story_fields_v2`; `td4v2_verify2`) |
| B2 (v1-only) | not applicable to v2: no v2 path imports `td4_models.refit_all`; frozen TD1/TD2 models only |
| R1 clearance B | fixed: `td4v2_gate.admit_b` recomputes the test ledger digest, requires and re-hashes every model file, every check before the once-only latch (`td4v2_evaluate.run`; tested) |
| N1 terminal 4xx | fixed: deterministic 400/404/413/422 in two sessions and no 200 -> settled without output within max(3, 0.2% of units) per provider and period (caps in freeze A v2) |
| N2 model violation | fixed: a returned-model halt blocks the period until a REVIEWED override file exists |
| N3 balance projection (non-blocking) | unchanged by design: USD 8 floor against the USD 5 hard floor; every session re-reads the balance |
| N4 session order | fixed: sessions read in numeric order |
| N5 output matching | fixed: an output counts only when its stored input SHA256 equals the sent input's (the redacted text's) SHA256 |
| N6 Jev 429 / transport upper bound (non-blocking) | unchanged by design: fails closed (caps, structural probe) |
| N7 | fixed: header, firm-id and article-count failures withhold (`td4v2_verify2.population`) |
| N8 FactSet horizon | fixed: pre-registration states that the sealed 2025 graph has no relationship start or end after 2025-06-06; 2025 halves reported descriptively |
| N9 code hashes | fixed: every build receipt, population check, audit, redaction round, parity receipt and dry run records the code SHA256; 2025 graph, links and features re-run with the frozen code |
| N10 in-process token registry (non-blocking) | unchanged by design: trusted-operator boundary; no runbook command registers a token |
| N11 (v1-only) | v1 `qualify` path; v2 `qualify` writes only new `QUALIFICATION_TD4v2_<uuid>.json` files |

## 8. Pre-registration v2

`PREREGISTRATION_v2.json` (SHA256 0185572b3bb5fedf272b656ed46936905f47bbff16237324a484c2ddd876929a) was written before any provider call and before any 2021-2025
outcome is read; it is bound in freeze A v2 and clearance B v2. It holds: the amendment-2 design (selection on the
2021-2022 subsample with TD2's protocol; P1 and P2 on pooled 2023-2025, two-way clustered by firm and calendar week,
two-sided 95%, confirmed only when the interval lies above zero; per-year results descriptive); the descriptive
secondary **S1** (selection on 2023, S1-P1 and S1-P2 on pooled 2024-2025, never confirmatory, plus the selection
agreement and the in-year oracle table); the masker-v2 and withholding rules; the **redaction rule** (transform,
placeholders, never-touched parts, location, three rounds, "still flagged" for population-level detectors and hit
caps, residual withholding, ledger, checks, reporting; `inputs.redaction`) with the detector manifest (included and
excluded classes, SHA256 of every reviewer script) and the redaction manifest (SHA256 of `td4v2_redact.py` and
`td4v2_redact_final.py`, placeholders, rounds, firm-level classes); the CIZ 2025 name source; the **N8** FactSet
statement (no relationship start or end after 2025-06-06; 2025 halves descriptive); the runner rules (N1, N2, N4/N5);
and the caveats: **historical reproduction** (2023-2025 is a period the earlier risk study examined), model memory,
non-point-in-time FactSet, the masking caveat (residual identifiability from executives, owned brands and unique
descriptions remains in most inputs; 2023-2025 not human-read beyond the round-1 samples; no redacted input was
human-read; detector recall unmeasured; false positives redacted for every arm), the relabelling of changed packet
contexts, and the Codex label note.

## 9. Tests, parity, rehearsal and dry runs

- Qualification: `QUALIFICATION_TD4v2_8638bad35a6046b78fcfe53067136b84.json` (SHA256 c8c1332fa6e2d7010590f8c3d88291cdd024938c19dbf47edd893be320955fba): 96 tests (46 v1, 50 v2, of which 12 new redaction and construction tests), 0 failures, 0 errors, 0 skipped; it records the SHA256 of every `td4/*.py` and `td4/v2/*.py`.
- Construction from the redacted text: the unit test `test_construction_parity_600_on_fixed_text` builds 200 fixed
  texts x 3 selectors with the unchanged TD3 selector classes (as `td4_runner.attempt` calls them) and checks 600/600
  bodies against the documented payload with the text verbatim; receipt on real inputs: `PARITY_construction_redacted_7f65846b.json` (SHA256 aaa4ea79f47e7bb2b49d2e185f3f8e9dba8897840f5b002c0c8ace5d96573f67): the first 200 sent 2021-22 subsample inputs that the redaction changed x 3 selectors = 600 bodies, 600 equal to the documented payload, with the redacted text verbatim and different from the masker-v2 body (equal: True).
- Parity (unchanged since the 84/84 pre-run; `parity/`): v2 selection path vs TD2 `SELECTION_v2.json` (max absolute
  difference 0); TD4C N9 frozen-code re-runs of the 2025 graph, links, features and the v1 checks (equal).
- Rehearsal (sandbox, mock providers, development years only; identity stand-in redaction): not re-run after the redaction change, because the rehearsal's select and evaluate steps read 2021-2022 development labels and the coordinator's 2026-11-07 instruction is no outcome read. The last rehearsal, `REHEARSAL_v2_5e8e8422.json` (SHA256 01621f4757488b5d0c7b9699018622b5518b4a56cc347ebb8f4c429a62c6d330, written 2026-11-07T02:40:30Z, pre-redaction code), ran build, verify, cost, prereg, freeze, clearance A, mock calls and re-asks, select, clearance B and evaluate, and refused a second evaluate (refused: STOP: once-only evaluation latch exists; no replay). The redaction changes reach the real steps only through `td4v2_inputs` (the sent text and the withheld set); that path is covered by the tests, the construction receipt and the 2025 dry run; the rehearsal code has an identity stand-in redaction for its sandbox.
- Dry runs (mock providers over the sent redacted inputs, simulated clock starting a minute before a weekday peak):
  `dryrun_2122_ea5f5c0e`: period 2122, 2000 records, 6000 mock calls, halt None, resume check {'skipped_done_before': 6000, 'new_mock_calls': 0}; `dryrun_2023_634753a8`: period 2023, 2000 records, 6000 mock calls, halt None, resume check {'skipped_done_before': 6000, 'new_mock_calls': 0}; `dryrun_2025_aa81bbb7`: period 2025, 2000 records, 6000 mock calls, halt None, resume check {'skipped_done_before': 6000, 'new_mock_calls': 0}; `dryrun_2024_41a21aff`: period 2024, 2000 records, 6000 mock calls, halt None, resume check {'skipped_done_before': 6000, 'new_mock_calls': 0}.

## 10. Cost and wall time (projection from the redacted inputs actually sent and the TD3 pilot's measured tokens)

| Period | Firm-weeks sent | DeepSeek calls | DeepSeek USD (off-peak) | Jev requests | Jev USD | Total USD | DeepSeek cumulative | DeepSeek busy h | Jev busy h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2122 | 18,814 | 18,914 | 9.78 | 37,828 | 6.46 | 16.24 | 9.78 | 1.0 | 0.5 |
| 2023 | 41,617 | 41,717 | 27.66 | 83,434 | 17.87 | 45.53 | 37.44 | 2.2 | 1.2 |
| 2024 | 41,740 | 41,840 | 28.35 | 83,680 | 18.28 | 46.62 | 65.79 | 2.2 | 1.2 |
| 2025 | 35,689 | 35,789 | 23.12 | 71,578 | 14.97 | 38.09 | 88.91 | 1.9 | 1.0 |
| all | | 138,260 | 88.91 | 276,520 | 57.57 | 146.48 | | 7.2 | 3.9 |

Projection file `COST_PROJECTION_v2_a002b3f6.json` (SHA256 7553efdd73c1603874da5b75abe3adf6c492b180624b487751c89b33474f0ea6).

DeepSeek balance USD 82.57 (coordinator, after the top-up), floor USD 8 (usable USD 74.57); off-peak DeepSeek only.
Projected DeepSeek spend USD 88.91 (off-peak) against usable USD 74.57; shortfall USD 14.34; first period reaching the floor: 2025. Every session re-reads the balance and stops before it would fall below USD 8; a later period refuses
until the earlier one is complete. Caps per period (bound in freeze A v2) are 1.3 times the projected USD per provider
and 1.1 times the projected Jev requests plus 500 (TD4 v1 rule `td4_cost.caps_from`).

Wall time of the builder's work: redaction stage 2026-11-07 06:07-10:00 UTC (about 3.9 h: locator build and checks about 0.6 h, three redact-and-re-run rounds 06:43-09:55 UTC with at most three detector runs in parallel, final steps about 10 min).

## 11. Runbook

PowerShell from `td4/v2`; `$PY = 'C:/Users/Boxuan/AppData/Local/Programs/Python/Python313/python.exe'`;
`$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'`; then `& $PY -B run_td4v2.py <command>`.

Run by the builder (offline, no clearance): `build-m2 --period 2122|2023|2024|2025`, `verify-m2 --period P`,
`audit-m2 --period P`, `flags-prepare --period P`, then `bash <run folder>/RUN_DETECTORS.sh`, then `flags --period P
--which <run folder name>` (detector-flag rule), `redact --which 1|2|3`, `redact-verify`, `redact-prepare`, then `bash <run folder>/RUN_DETECTORS.sh`, then `redact-residual --which <run folder name>` (rounds 1-3), `redact-final`, `ciz-impact`, `parity`, `parity-construction`, `packet-r`, `parity2025 --which <check>`, `packet`,
`cost`, `dry-run --period P --limit 2000`, `rehearsal` (sandbox; stand-in empty flag files, the reviewer detectors are
not run on development data), `qualify`, `prereg`, `freeze`, `summary`.

Real steps (each refuses unless clearance A v2 matches `FREEZE_TD4_A_v2.json`; in this order):

| Step | Command | Gate | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | `status` | none | hashes | nothing |
| 2 | `call --period 2122` (repeat to resume) | A, online | 2021-2022 subsample inputs; keys from DEEPSEEK_API_KEY, TYPESAFE_API_KEY | `runs_v2/real_2122/session_n/` |
| 3 | `call --period 2023`, then `2024`, then `2025` | A, online | each refuses until every earlier period's main pass is complete | `runs_v2/real_<P>/` |
| 4 | `reask --period P` (any order after its main pass) | A, online | 100 fixed firm-weeks | same folder, pass reask1 |
| 5 | `select` | A | 2021-2022 development labels (end <= 2022-12-31) and outputs | `selection_v2/SELECTION_FREEZE_v2.json`; `CLEARANCE_REQUEST_TD4_B_v2.json` when 2023-2025 are complete (else `request-b`) |
| 6 | reviewer checks the selection freeze and writes clearance B v2 | | | |
| 7 | `evaluate` (once) | B | 2023-2025 labels (window end <= 2025-12-31) and outputs | `evaluation_v2/EVALUATION_STARTED_v2.json` latch, `EVALUATION_RESULT_v2.json` |
| 8 | `convert-labels --labels <folder>` then `score-labels --labels <csv>` | none | Codex outputs, 2021-2022 outputs | label template and scores (LLM labels from OpenAI, not human ground truth) |

A session stops on the DeepSeek balance floor (USD 8), a cap, HTTP 401/402/403, a redirect, a third other 4xx per
provider, three consecutive failed units, the TD3 structural probe, a disallowed returned model, or a balance-read
failure; re-running the same `call` resumes and never re-bills a completed request. A deterministic 4xx unit that
failed in two sessions is settled without output (N1) within its cap. A returned-model halt needs the reviewed override
file before any further session of that period (N2). DeepSeek runs off-peak only (no new call in 01:00-04:00 and
06:00-10:00 UTC on weekdays, 2-minute guard); Jev 429 backoff 5/15/45/120/300 s.

## 12. Clearance files (the reviewer creates them; the builder wrote requests only)

- Clearance A v2: `D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_A_v2.json`, schema `TD4_CLEARANCE_A_v2`,
  status CLEARED, no `request_only`, `freeze_sha256` = SHA256 of `FREEZE_TD4_A_v2.json`, and the fields of
  `CLEARANCE_REQUEST_TD4_A_v2.json` (version, code_sha256 of every `td4/*.py` and `td4/v2/*.py`, artifact_sha256,
  external_sha256, preregistration_sha256, aggregation_freeze_sha256, caps, models, question_sha256, periods,
  hard_seal, run_order). Adding or editing any of those `.py` files or changing a bound file invalidates it.
- Clearance B v2: `D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_B_v2.json`, schema `TD4_CLEARANCE_B_v2`,
  copied from `selection_v2/CLEARANCE_REQUEST_TD4_B_v2.json` (clearance_a_sha256, selection_freeze_sha256,
  preregistration_sha256, ledger_digest_test, models_sha256, models_files_sha256, evaluation_years [2023, 2024, 2025],
  maximum_label_end 2025-12-31); the evaluator recomputes the ledger digest and every model hash before its latch.

| File (private root) | SHA256 |
| --- | --- |
| `FREEZE_TD4_A_v2.json` | 28b302b9eb545815f6d89de4c838c686c2d2fbd32010372beee97fa322561c7a |
| `CLEARANCE_REQUEST_TD4_A_v2.json` | 8f1f4ea2119c208d59f478da43ccef2ca80ad3e0973f539bc97a188ad46e7fa4 |
| `PREREGISTRATION_v2.json` | 0185572b3bb5fedf272b656ed46936905f47bbff16237324a484c2ddd876929a |
| `v2_PACKET_READY.json` | 3776318c696a26a8f5de4b39a761f6db8646d818a50957e4f5fa0f232747bd7a |
| `inputs_m2/select_2122_r3/REDACTION_FINAL_2026-11-07T085440Z.json` | ab4a63c43e6240ae9b4156fba891a7feed0f87ee389cbd2b2486411cd7a2528a |
| `inputs_m2/select_2122_r3/DETECTOR_FLAGS_M2_2026-11-07T043322Z.json` | a7f2886aa55b8c7cabbb75738cd675129200bf3d0028c3a2ab3dd076c9c9b1ce |
| `inputs_m2/select_2122_r3/MASK_V2_RECEIPT.json` | e791a1da8dd36a940bf40fc52766262f0b79fd4798a581568e14c7123456614c |
| `inputs_m2/period_2023_r3/REDACTION_FINAL_2026-11-07T093645Z.json` | 0b771200bde8bc97dd99189e0baac87f14e17b085d3d7e3be2d048518b2744e1 |
| `inputs_m2/period_2023_r3/DETECTOR_FLAGS_M2_2026-11-07T043352Z.json` | 5e67e62830f266bfc2b4b8d0a6fd59b81d718fd76d5eb95f807c828559174a77 |
| `inputs_m2/period_2023_r3/MASK_V2_RECEIPT.json` | 4dd867dba2a0f98673091a196f0caf24c6a4a4651f9577c1ffec32be288621a5 |
| `inputs_m2/period_2024_r3/REDACTION_FINAL_2026-11-07T095449Z.json` | 6cf6a4a7f0e3da56a6ce883ed5f7cdb2a4d64fd95b55675251b4b82d3dabff9c |
| `inputs_m2/period_2024_r3/DETECTOR_FLAGS_M2_2026-11-07T041730Z.json` | c29b346c71ae6eb2bba29db13a2288c5649a204991ea57a0e5182bbf28aadce9 |
| `inputs_m2/period_2024_r3/MASK_V2_RECEIPT.json` | 0c8f38396bb4536d5e46947a0337b45f000a04d1d723921576d07f7d4c006dfd |
| `inputs_m2/period_2025_r3/REDACTION_FINAL_2026-11-07T093844Z.json` | 87ab892a4aa529788699d4bc7be3140e033aebc1ee62183e85bb1456203faec1 |
| `inputs_m2/period_2025_r3/DETECTOR_FLAGS_M2_2026-11-07T054640Z.json` | 9d2f4d9c4c9a299945124cd4b301e5a8cdcb1bac4641cd129370134b6898fc52 |
| `inputs_m2/period_2025_r3/MASK_V2_RECEIPT.json` | 1c04857406995f6868d801efa1c06173ac355e8176061bf16453e3956808cc1f |
| `inputs_redacted/2122/round_3/INPUTS.jsonl` (sent text) | 0fe1e73f55caa0918e119cf008b21a6d9c526e6bd6298472b363eba07bf308fd |
| `inputs_redacted/2023/round_3/INPUTS.jsonl` (sent text) | cb40a2efb5a7ffd41ee1f09abca2f182c201e3f6a92c7a0132b44c8ffdccc899 |
| `inputs_redacted/2024/round_3/INPUTS.jsonl` (sent text) | 57e19595e49762a4cfcc4e32ce8f514ae9fcaf35c23fec83a696cb4e2e31bdd7 |
| `inputs_redacted/2025/round_3/INPUTS.jsonl` (sent text) | 1ceb3ced0efe0c3f69b62744940a798234fee24edaf9fbae76b7d5c1331ec489 |
| `labeling_v2r/PACKET_INDEX_v2r.json` | ab11829799345d6c585de56dea88537e36f0967eca501fdfe44e453089eeb1c6 |
| `parity/PARITY_construction_redacted_7f65846b.json` | aaa4ea79f47e7bb2b49d2e185f3f8e9dba8897840f5b002c0c8ace5d96573f67 |

## 13. v1 files reused unchanged, and files superseded

Every file bound by `FREEZE_TD4_A_v1.json` is byte-identical (167 of 167 files re-hashed equal).

Reused unchanged (bound again in freeze A v2 by their current hashes): the 24 v1 modules in `td4/`; the 2025 extracts
(`td1_inputs/`, `td2_sources/`, `panel_keys/`, `firm_week_manifest/year_2025`, `features/year_2025`,
`evidence/year_2025`); the article stores of 2023, 2024 and 2025 and the 2021-2022 selection store; the masker-v2
builds `inputs_m2/*_r3` (their INPUTS are the round-0 text of the redaction); `AGGREGATION_FREEZE_v1.json`; the Codex
packet `labeling_v2/` (unchanged; its changed contexts are relabelled from `labeling_v2r/`).

Superseded (not used by v2), and why:

- all v1-masker inputs (`inputs/period_2024`, `period_2025`, `repair_*`, `select_2122*`; masking reviews M1-M4);
- the v1 D8 repair rule and builds (TD4C B1); `PREREGISTRATION_v1.json`, `FREEZE_TD4_A_v1.json`,
  `CLEARANCE_REQUEST_TD4_A_v1.json`, the v1 cost projection and labeling packet (amendment 2);
- the v1 entry points `refit`, `call`, `select`, `evaluate` (TD4C B2 is v1-only);
- `td4v2_build.build_select`, `build_year`, `repair`, `td4v2_verify` (v1-masker build path of the first v2 draft);
- the hand rename table of `MASK_TABLES_TD4_v2.json` for 2025 (replaced by the CIZ source);
- masker v3 (researcher decision 2026-11-06; record only);
- the **whole-input detector-flag rule** (`td4v2_flags` withholding; 2026-11-06): its files stay bound as round 0 of
  the redaction and no longer withhold (researcher decision 2026-11-07);
- the first round-1 redaction attempt `inputs_m2r/` and its stopped re-run folder (locator fix, section 6b);
- the earlier dry runs of 2021-22, 2023 and 2024 and the qualification `QUALIFICATION_TD4v2_f8af255c...` (84/84) and
  rehearsal `REHEARSAL_v2_5e8e8422` (both on the pre-redaction code).

## 14. Code changes since the 84/84 pre-run (`QUALIFICATION_TD4v2_f8af255c...`, 2026-11-07 00:33Z)

File-level comparison of the current `td4/*.py` and `td4/v2/*.py` SHA256 with the inventory recorded in
`QUALIFICATION_TD4v2_f8af255cb55041d3b4882a0c6428d517.json` (84 tests, 0 failures; written 2026-11-07T00:33:17Z). No v1
file (`td4/*.py`) changed. New or changed v2 files:

| File | Change |
| --- | --- |
| `td4v2_redact.py` (new) | span-level redaction: `Doc` (article content segments, protected bracket tokens, role tokens), placeholders and class kinds, one locator per detector family (M1, M2 forms/header/surface/RX, M3, M4, MA1 dates, s17b EMB/OWN/CO/PFX/IDX/REM, GB frozen groups, s41, s41b, MA2 B1/D/D12 and the G1-G3 word alignment), `redact_round`, `prepare_rerun` (detector copies with the INPUTS path changed), `residual`, `check_record`/`render`/`verify_round`/`idempotency`. The stopped first attempt (`inputs_m2r/`) ran an earlier version without the M2 `english_short_name` right-boundary fix (section 6b); every round in `inputs_redacted/` ran with the final file (SHA256 in each ROUND/VERIFY/RESIDUAL file) |
| `td4v2_redact_final.py` (new) | `finalize` (final inputs, withheld set with the placeholder-dependence rule, per-period statistics, `REDACTION_FINAL_<UTC>.json` in the build folder), `final_path`, rehearsal `stand_in` |
| `td4v2_tests_redact.py` (new) | 12 tests: placeholder kinds; spans cut to article text and around tokens (menu, labels and label tokens untouched); invariants hold and tampering is caught (outside change, token change, header change, token under `[REDACTED]`); idempotency; exact-context location; round mapping to v2 offsets; MA2 alignment positions; GB frozen-group rescan; placeholder-dependent flags; reviewer constants read without running the script; inputs require the redaction result and its hash; construction parity 600/600 on fixed text |
| `td4v2_inputs.py` | the sent text is the final redacted INPUTS named by the latest `REDACTION_FINAL` (file SHA256 checked); withheld = population check + redaction rule; the detector-flag file no longer withholds; `counts()` reports the redaction |
| `td4v2_labels.py` | `redacted_packet()` writes `labeling_v2r/` (changed contexts, same ids, same batch format); `convert(..., revised=True)`; `score(path, path_r)` uses relabels for changed contexts and excludes a changed context without one; withheld contexts excluded as before |
| `td4v2_spec.py` | `LABELS_R`; pre-registration `inputs.detectors` and `inputs.redaction` (replacing `inputs.detector_flags`), masking and label caveats, `redaction_manifest` in `preregistration_text()` |
| `td4v2_freeze.py` | binds every redaction file (rounds' INPUTS, LEDGER, ROUND, VERIFY, RESIDUAL; re-run folders' copies, scripts, logs; `REDACTION_FINAL`), `labeling_v2r/`, the construction-parity receipt; build receipts require the redaction result (redacted-file hash re-checked); freeze field `redaction_runs`; masker field names the withholding rule |
| `td4v2_parity.py` | `construction_redacted()` (receipt `PARITY_construction_redacted_<tag>.json`) |
| `td4v2_rehearsal.py` | identity stand-in redaction in the sandbox; sandbox `LABELS_R` |
| `run_td4v2.py` | commands `redact`, `redact-verify`, `redact-prepare`, `redact-residual`, `redact-final`, `parity-construction`, `packet-r`, `convert-labels-r`; `score-labels` takes an optional relabel template |
| `td4v2_tests.py` | `qualify()` adds the redaction tests |
| `td4v2_tests_m2.py` | the inputs-gate test requires the redaction result (a detector-flag file alone no longer withholds) |
| `td4v2_flags.py` | `run_plan` exports `PYTHONIOENCODING=utf-8` (the first 2024 run's `s41_targeted` crashed printing examples on the Windows console); no class or rule change: the 2021-22 and 2023 flag files were re-extracted with it with identical counts |
| `td4v2_verify2.py` | memory: the 2025 population check frees the metadata before the CIZ check; the scan-hit file name with a timestamp is recognised |
| `td4v2_rename.py` | memory: the 2025-only-name check streams INPUT_META_COMPACT into a per-key firm map (the first 2025 run hit the job memory cap) |


## 15. Open risks and disclosures

- Masking (pre-registered caveat): masker v2 is FIX REQUIRED by the round-2 reviews and is not revised; the
  span-level redaction replaces what the reviewers' detectors flag, but detector recall is not measured, so some
  give-backs, own-article tickers and name residues remain in the sent inputs. Residual identifiability from executive
  names, owned brands and unique descriptions remains in most inputs (disclosed). The 2023-2025 masker-v2 inputs were not
  human-read beyond the round-1 samples, and no redacted input was human-read. A `[TERM]` placeholder removes the
  link between a give-back slot and its firm but leaves the surrounding words (for example a remaining product number
  next to it).
- Information removed: detector false positives are redacted too (ordinary words such as 'PRICE' or 'BUSINESS' after
  give-back or own-article detectors, prices written with periods by `d_dmy_periods`, product model numbers by
  `x_two_digit_fiscal`, third-party brief headers), for every arm alike; the share of article characters replaced is in
  section 6b. A placeholder shows that something was removed.
- Withheld shares: above 2% in 2021-22, 2024, 2025; the withheld rows take each arm's fallback, so the
  selectors are compared on fewer firm-weeks, and the withheld inputs are not a random subset (firm-level M4 flags select
  firm-weeks with recently listed firms; the 2021-2022 adjudicated sets were selected by the reviewers' reading).
- The inputs changed after the reviewers read them: the round-2 reviews read the masker-v2 2021-2022 text, not the
  redacted text.
- Disclosed context limitation and model memory: as before (executive names, owned brands, unique descriptions; both
  providers' models were trained after 2025).
- The 2025 FactSet graph is frozen at 2025-06-06 for the 2025 origins after that date (N8).
- DeepSeek balance: Projected DeepSeek spend USD 88.91 (off-peak) against usable USD 74.57; shortfall USD 14.34; first period reaching the floor: 2025.
- Memory and disk: every redaction process stayed below 196 MB working set (redaction, verify, residual and finalize processes); detector re-runs at most about 241 MB in their MA2 population scan; at most three detector runs in parallel; D: and C:
  stayed above 25 GB and 15 GB free.
- No provider call, no network, no git, no WRDS; no 2021-2025 label, return or loss was read; nothing dated after
  2025-12-31 was decoded. No clearance file was written. Existing data files were not edited and nothing was deleted
  (stopped attempts and development runs remain, labelled in section 2).


## 16. Addendum: rule v2.1 (2026-11-07, re-freeze as v2.1)

Written 2026-11-07T10:14Z (from `date -u`). Where this addendum and sections 1-15 differ, this addendum governs.
The v2 files (`FREEZE_TD4_A_v2.json`, `CLEARANCE_REQUEST_TD4_A_v2.json`, `PREREGISTRATION_v2.json`) are untouched;
they were never cleared and are bound in freeze A v2.1 as the superseded record.

**Decision** (coordinator; a technical correction by Claude that the reviewer will check; the researcher is told). For
2023, 2024 and 2025 the M4 firm-level classes NONAME and NONAME_AFTER_2022 no longer withhold, and MASKER_NO_SOURCE
does. For 2021-2022 nothing changes: the adjudicated key sets stay withheld, and NONAME/NONAME_AFTER_2022 stay in
(they flag no 2021-2022 input).

**Reason.** s13 NONAME tests for no CRSP name row dated on or before 2024-12-31, and it does not read the CIZ 2025
source. NONAME_AFTER_2022 tests for a first CRSP name after 2022-12-31. Both stand in for "the masker had no name to
mask for this firm". In 2023-2025 that premise fails: recently listed firms are ordinary, and masker v2 masked them
from CRSP (rows dated on or before the build's last day) or from CIZ (2025). Their text residues are covered by the
span detectors, which ran and were redacted. The round-0 2025 hits confirm this: every NONAME hit was a link firm, and
every NONAME_AFTER_2022 hit was a link firm or a focal firm first listed after 2022.

**Timing.** The correction was fixed after the redaction rounds, before any provider call and before any outcome was
read. It is recorded in `PREREGISTRATION_v2_1.json` (`inputs.redaction.v2_1_correction`), with the rule-v2 shares as a
sensitivity.

**MASKER_NO_SOURCE** (`td4v2_nosource.py`). An input is withheld, for every arm alike, when its focal firm or a shown
firm (menu partner) has no name and no ticker entry in the exact name tables that masker v2 used for that period's
build. Those tables are:
- the CRSP stocknames/names rows of the firm and of its issuer siblings (PERMNOs sharing a CUSIP6) dated on or before
  the build's last day (`MASK_V2_RECEIPT.last_day`), read with `td3_build.issuer_siblings` and `load_names`;
- the FSB1 FactSet names and US tickers linked by issuer CUSIP6 (`td3_build.factset_aliases`);
- for 2025, the bound CIZ file (`td4v2_names2025.load_ciz`; its SHA256 is checked against the 2025 receipt).

RIC roots do not count. The check is computed from these tables, not from s13; their SHA256 are in every
MASKER_NO_SOURCE file, and the CRSP and FSB1 files are now bound in the freeze.

| Period | Focal and shown firms checked | Unsourced firms | Inputs flagged | Applied |
| --- | --- | --- | --- | --- |
| 2021-22 | 3,256 | 0 | 0 | no (rule v2 unchanged) |
| 2023 | 3,061 | 0 | 0 | yes |
| 2024 | 2,823 | 0 | 0 | yes |
| 2025 | 2,608 | 0 | 0 | yes |

As a disclosure check, the same test on the link firms finds 0 unsourced link firms in 2023, 2024 and 2025. The 23
link firms behind the 2025 NONAME hits are sourced by the CIZ 2025 file.

**Withheld, v2 versus v2.1** (share of callable inputs):

| Period | Callable | v2 withheld | v2.1 withheld | Over 2% (v2.1) | Moved to sent | Redactions per moved input: mean / p50 / p95 | Moved inputs with no redaction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-22 | 20,379 | 1,184 (5.81%) | 1,184 (5.81%) | yes (all adjudicated key sets) | 0 | - | - |
| 2023 | 42,336 | 719 (1.70%) | 0 (0.00%) | no | 719 | 1.94 / 1 / 7 | 269 |
| 2024 | 45,750 | 4,010 (8.77%) | 0 (0.00%) | no | 4,010 | 1.46 / 1 / 5 | 1,722 |
| 2025 | 42,430 | 6,741 (15.89%) | 0 (0.00%) | no | 6,741 | 1.36 / 1 / 5 | 2,883 |

- 2021-22 subsample: 1,165 of 19,979 withheld; 18,814 sent.
- A moved input with no redaction had no detector match other than the firm-level classes.
- The placeholder-dependent M4 GB_* matches still count for nothing (section 6b). Counting them would withhold 11
  inputs (2023), 4 (2024) and 625 (2025).
- The sent texts are unchanged: the round-3 INPUTS of section 6b.

**Spot check of moved 2025 inputs** (`inputs_redacted/2025/SPOTCHECK_v2_1_2026-11-07T100735Z.json`, counts only, no
names stored). The sample was the first 20 v2-withheld 2025 keys by SHA256('TD4-v21-spotcheck|' + key): 274 focal and
shown firm slots and 250 articles.

For each firm, the names and tickers were taken from the masker's own tables (CRSP own and issuer-sibling rows,
FSB1, CIZ 2025). The article text of the final redacted input was then searched for:
- every distinctive name word (at least 4 characters, not a legal or generic word, not an English word);
- every 2-5 letter ticker (case-sensitive, not inside a bracket token);
- the first two words of every name;
- in a second pass, the first core word of every name, English words included, in upper and Title case.

The 114 hits and all 250 headlines were read by the builder:
- every hit is a generic word (an inflected English word that also occurs in some other firm's name, such as a
  plural noun), an initial pair matched across a word boundary, or an ordinary phrase;
- no name, name part or ticker of a focal or shown firm of these inputs is readable;
- the readable firm names in the headlines belong to firms that are not tokens of the input.

**Cost** (`COST_PROJECTION_v2_4b18b8cc.json`, off-peak DeepSeek, from the v2.1 inputs; USD):

| Period | Firm-weeks sent | DeepSeek calls | DeepSeek | Jev requests | Jev | Total | DeepSeek cumulative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-22 | 18,814 | 18,914 | 9.78 | 37,828 | 6.46 | 16.24 | 9.78 |
| 2023 | 42,336 | 42,436 | 28.15 | 84,872 | 18.18 | 46.33 | 37.93 |
| 2024 | 45,750 | 45,850 | 31.07 | 91,700 | 20.03 | 51.10 | 69.00 |
| 2025 | 42,430 | 42,530 | 27.44 | 85,060 | 17.76 | 45.20 | 96.44 |
| all | | 149,730 | 96.44 | 299,460 | 62.44 | 158.88 | |

**DeepSeek shortfall: USD 21.87.** Projected off-peak spend is USD 96.44 against USD 74.57 usable (balance USD 82.57
minus the USD 8 floor). 2025 reaches the floor after about USD 5.57 of its USD 27.44, so at least USD 21.87 must be
added before 2025, or 2025 stops at the floor. Caps (1.3 times the projection, bound in freeze A v2.1): DeepSeek USD
12.72 / 36.59 / 40.39 / 35.67 and Jev USD 8.40 / 23.63 / 26.04 / 23.09 (2021-22 / 2023 / 2024 / 2025).

**Tests, dry runs, freeze.**
- `QUALIFICATION_TD4v2_b1af9f1142a340f9aecb6dbe25b26631.json`: 100 tests (46 v1, 54 v2), 0 failures, 0 errors.
- New since the v2 freeze: 4 tests in `TestMaskerNoSource`:
  - a positive case (an unsourced firm flags its inputs);
  - a negative case (sourced by an own CRSP row, an issuer sibling's row, a FactSet name or ticker; not by a row after
    the last day);
  - a 2025-only CIZ firm that counts as sourced;
  - the v2.1 redaction result is preferred.
- Dry runs over the v2.1 inputs (2,000 records, 6,000 mock calls each, no halt, resume sends nothing): `runs_v2/`,
  2023, 2024, 2025, 10:12 UTC.
- Freeze A v2.1 binds:
  - the v2.1 code inventory;
  - every build's MASKER_NO_SOURCE and REDACTION_FINAL_v2_1 files;
  - the spot-check record and the masker name tables;
  - the superseded v2 freeze, request and pre-registration;
  - everything freeze A v2 bound.
- Clearance A for v2.1: `D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_A_v2_1.json`, schema
  `TD4_CLEARANCE_A_v2_1`, status CLEARED, `freeze_sha256` = SHA256 of `FREEZE_TD4_A_v2_1.json`, plus the fields of
  `CLEARANCE_REQUEST_TD4_A_v2_1.json`. The gate reads `PREREGISTRATION_v2_1.json`.

| File (private root) | SHA256 |
| --- | --- |
| `PREREGISTRATION_v2_1.json` | 7dd613fff76c71749fa21fafd3cced060e8f794a7b955307e28d6520c4c07a24 |
| `FREEZE_TD4_A_v2_1.json` | 55447afe1145d10a92651fb8de69cc3bccdd13fca169b08dbbb20e495c200ed6 |
| `CLEARANCE_REQUEST_TD4_A_v2_1.json` | 19128279b6c49fa7dd102cb2f9dda9458fbbdeb0c9994f59606cf761eb1066de |
| `QUALIFICATION_TD4v2_b1af9f1142a340f9aecb6dbe25b26631.json` | 1dfd53f17ee1f1cc490e2f29107b55695089c7f3137822d0cff34af8dd640168 |
| `COST_PROJECTION_v2_4b18b8cc.json` | 312a343f074664af5c91c0f320ea956b525b6185ea93204d07de794187e6b846 |
| `inputs_m2/select_2122_r3/REDACTION_FINAL_v2_1_2026-11-07T101152Z.json` | 36236be0c6a8581c5cca7d9e7b4f82b0771e980052a15f247cc128578eb0be23 |
| `inputs_m2/period_2023_r3/REDACTION_FINAL_v2_1_2026-11-07T101154Z.json` | 5e845a02d946f490c75894127d0faf5c74150553f2fe29dbacbef1c83240e567 |
| `inputs_m2/period_2024_r3/REDACTION_FINAL_v2_1_2026-11-07T101156Z.json` | c57c5dd21afe4bd3bb9969a49b93b7e610e2517a03487d35f744901e98b4aa59 |
| `inputs_m2/period_2025_r3/REDACTION_FINAL_v2_1_2026-11-07T101158Z.json` | 2c4e99f1e9052c3b246edf06790e2a07abe6bd77f77f2a5ee72a42bda191981d |
| `inputs_redacted/2025/SPOTCHECK_v2_1_2026-11-07T100735Z.json` | bde75601ae115fda5c4ade3c8f706c4dcaafc489342b9a1cb14f04dcf36d5bb6 |

Earlier MASKER_NO_SOURCE and REDACTION_FINAL_v2_1 files of 10:03-10:09Z are development runs of the same rule, and the
spot-check file of 10:05Z is a first pass with fewer search terms. All of them are superseded by the files above and
bound only as the record: `td4v2_inputs` reads the latest file of each kind.

**Code changes for v2.1** (since freeze A v2):

| File | Change |
| --- | --- |
| `td4v2_nosource.py` (new) | MASKER_NO_SOURCE: `sourced`, `flag_inputs`, `tables` (the build's name tables), `run`, `latest`, rehearsal `stand_in` |
| `td4v2_redact_final.py` | `finalize_v21` (v2.1 withheld set next to v2, moved-input redaction counts); `final_path` prefers `REDACTION_FINAL_v2_1_*`; `v2_path`; the stand-in writes a v2.1 file |
| `td4v2_spec.py` | v2.1 paths (`FREEZE_TD4_A_v2_1.json`, `CLEARANCE_REQUEST_TD4_A_v2_1.json`, `PREREGISTRATION_v2_1.json`, `TD4_CLEARANCE_A_v2_1.json`, schema `TD4_CLEARANCE_A_v2_1`), `SUPERSEDED_V2`, `RULE_VERSION`; pre-registration `supersedes`, `rule_version`, `inputs.redaction.v2_1_correction`, `masker_no_source_manifest` |
| `td4v2_freeze.py` | schema `TD4_FREEZE_A_v2_1`, `rule_version`; binds MASKER_NO_SOURCE, the v2.1 results, the spot check, the CRSP/FSB1 name tables and the superseded v2 files; build receipts require the v2.1 result and MASKER_NO_SOURCE |
| `td4v2_rehearsal.py` | sandbox stand-ins for MASKER_NO_SOURCE and the v2.1 result; sandbox `SUPERSEDED_V2` |
| `run_td4v2.py` | commands `nosource`, `redact-final-v21` |
| `td4v2_tests_redact.py` | `TestMaskerNoSource` (4 tests) |

Wall time of the v2.1 correction: 2026-11-07 10:00-10:14 UTC (about 15 minutes).

## 17. Addendum: v2.2 redaction fix round (2026-11-07)

Written 2026-11-07T18:57Z (from `date -u`). Where this addendum and sections 1-16 differ, this addendum governs. Nothing of v2 or v2.1 is overwritten: the v2.1 rounds (`inputs_redacted/`), detector folders, REDACTION_FINAL_v2_1 files, freeze, request and pre-registration stay as they were and are bound in freeze A v2.2 as the record.

**Trigger.** The TD4M3 redaction review of the v2.1 inputs (`five_session_review/td4m3/TD4M3_REDACTION_AUDIT.md`, SHA256 4b4c7b94...8a104d4) failed every period: readable name, unit, code and date residues in the reading sample (B1), an unsound placeholder waiver for the M4 GB_* classes (B2), and placeholders inside words (B3). The coordinator brief F1-F6 asked for a redaction-only fix round; the re-review draws a new sample.

**What changed** (`td4v2_redact22.py`, `td4v2_render.py`, `td4v2_check22.py`; module docstring of `td4v2_redact22` has the exact rules):

- F1 word boundaries. One definition of a word: a maximal run of Unicode letters and digits (`str.isalnum`), joined by single connectors (hyphen, en dash, apostrophes, ampersand, period, slash) between two such characters; a leading news slug (`BRIEF-`, `UPDATE 2-`, ...) is not part of the word. Every placeholder span is widened to whole words, and a word part glued to any placeholder is redacted. Month abbreviations cut out of words and date forms inside codes become the whole word.
- F2 builder locators, run in every round with the reviewers' detectors and repeated on the round's provisional text until they find nothing (at most 6 passes; not converged = still flagged): (a) own names and initialisms in any case (guard: an initialism or ticker that is an English or short common word only in upper case in a mixed-case line; generic acronyms never; an English first name word only in the firm's own article); (b) hyphenated name parts and first-letter-cut forms; (c) compounds, name fragments, codes from tickers, initialisms or name prefixes, the same code number under another prefix, long forms of short names (lower-case English compounds in mixed text and all-English hyphen compounds are kept); (d) the code, model, number or upper-case word after a role token (period codes and stop words kept), acronyms defined after a hidden name, unit company names after a role token; (e) GB_* phrases with their trigger words, frozen MA1 GB neighbour words, both neighbours of a slot that hides a generic acronym, period-code-named firms in period slots; (f) `[YEAR]-yy`, year ranges, `Month [DATE] dd`, month names next to `[DATE]`, the masker-v2 swallow case. Name tables: the receipt-bound tables of `td4v2_nosource`; word lists are built at run time from private data and never written to the public folder.
- F3 no GB waiver: every M4 GB_* hit after round 3 is still flagged; the only waiver is an M4 word hit equal to the word inside a placeholder (`literal_placeholder_hit`), listed per class below.
- F4 rule A.4 unchanged (three rounds from `inputs_m2/*_r3`, unmodified; unlocatable or still flagged after round 3 withheld for all arms; 2021-22 adjudicated sets withheld; MASKER_NO_SOURCE unchanged).
- F5 storage: rounds 1-2 are ledger-only (`LEDGER.jsonl`, `HASHES.jsonl`, `ROUND.json`); round 3 also stores `INPUTS.jsonl`; `td4v2_render.render(masker-v2 text, cumulative ledger)` rebuilds any round and `td4v2_render.open_round` feeds the detector re-runs of rounds 1-2. Outputs: `inputs_redacted_v2_2/<period>/`, detector re-runs `detector_flags/<period>_v22rd<r>_<UTC>/`.

**Per period** (sent = callable minus withheld; 2021-22 inside the 20,000 subsample; s04b classes and P6 by `td4v2_check22`, reproduced from the review's definitions; over-redaction = share of article word tokens that a ledger span covers):

| Quantity | 2021-22 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| callable | 20,379 | 42,336 | 45,750 | 42,430 |
| ledger entries round 1 / 2 / 3 | 23,066 / 69 / 0 | 122,767 / 539 / 0 | 113,519 / 426 / 6 | 101,250 / 1,439 / 84 |
| placeholders [REDACTED] / [DATE] / [TERM] | 14,999 / 1,146 / 6,990 | 83,942 / 8,721 / 30,643 | 83,287 / 6,904 / 23,760 | 76,885 / 6,104 / 19,784 |
| residual after round 1 / 2 / 3 (inputs) | 64 / 0 / 0 | 1,036 / 719 / 719 | 4,331 / 4,010 / 4,010 | 7,537 / 6,811 / 6,741 |
| withheld: adjudicated / unlocatable / still flagged / no source | 1,184 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| withheld (share of callable) | 1,184 (5.81%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| withheld in subsample (2021-22) | 1,165 (5.83%) | - | - | - |
| sent inputs scanned | 18,814 | 42,336 | 45,750 | 42,430 |
| P6: sent inputs / entries with a placeholder next to a letter or digit | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| s04b D1, sent inputs (v2.1 -> v2.2) | 54 -> 0 | 617 -> 0 | 278 -> 0 | 133 -> 0 |
| s04b D2, sent inputs (v2.1 -> v2.2) | 0 -> 0 | 206 -> 0 | 0 -> 0 | 28 -> 0 |
| s04b N1, sent inputs (v2.1 -> v2.2) | 0 -> 0 | 182 -> 0 | 0 -> 0 | 0 -> 0 |
| s04b N1b, sent inputs (v2.1 -> v2.2) | 0 -> 0 | 209 -> 0 | 9 -> 0 | 0 -> 0 |
| s04b N7, sent inputs (v2.1 -> v2.2) | 0 -> 0 | 12 -> 0 | 0 -> 0 | 5 -> 0 |
| s04b N4er, sent inputs (v2.1 -> v2.2) | 1 -> 0 | 3 -> 0 | 3 -> 0 | 7 -> 0 |
| article word tokens redacted (v2.1 -> v2.2, own sent sets) | 0.07% -> 0.14% | 0.10% -> 0.24% | 0.10% -> 0.20% | 0.11% -> 0.23% |
| same, inputs sent in both | 0.07% -> 0.14% | 0.10% -> 0.24% | 0.10% -> 0.20% | 0.11% -> 0.23% |
| literal-placeholder waivers (F3) | none | none | none | none |

**Cost** (`COST_PROJECTION_v2_2_b40885cd.json`, off-peak DeepSeek, from the v2.2 inputs; USD):

| Period | Firm-weeks sent | DeepSeek calls | DeepSeek | Jev requests | Jev | Total | DeepSeek cumulative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-22 | 18,814 | 18,914 | 9.78 | 37,828 | 6.46 | 16.24 | 9.78 |
| 2023 | 42,336 | 42,436 | 28.15 | 84,872 | 18.18 | 46.33 | 37.93 |
| 2024 | 45,750 | 45,850 | 31.07 | 91,700 | 20.03 | 51.11 | 69.01 |
| 2025 | 42,430 | 42,530 | 27.44 | 85,060 | 17.76 | 45.21 | 96.45 |
| all | | 149,730 | 96.45 | 299,460 | 62.44 | 158.89 | |

DeepSeek shortfall: USD 21.88 (projected USD 96.45 against USD 74.57 usable; first period reaching the floor: 2025).

**Label packet `labeling_v2r2`** (`PACKET_INDEX_v2r2.json`): 31 of 150 contexts have a v2.2 text that differs from the labelled text (16 labelled from masker-v2 text in labeling_v2, 15 from v2.1 text in labeling_v2r); 4 batches (S01-S04), same INSTRUCTIONS text; withheld under v2.2: 7 contexts (4 of them changed). Codex runs on it outside the package (`convert-labels-r2`, then `score-labels --labels V2,V2R,V2R2`).

**Construction parity** (`PARITY_construction_redacted_v2_2_2cf58e99.json`): 600 of 600 request bodies built from the v2.2 text equal the documented payload (equal = True).

**Tests** (`QUALIFICATION_TD4v2_1ce0c9aaf8c64bd88e9e173eef2c212b.json`): 113 tests (46 v1, 67 v2), 0 failures, 0 errors. New: 13 tests in `td4v2_tests_redact22` (word definition, month and code cuts, glued words and P6, each locator class a-f, no GB waiver, fixpoint and idempotency, ledger-only rounds rendering exactly, v2.2 result preferred).

**Dry runs over the v2.2 inputs** (mock providers, `runs_v2/dryrun_v2_2_*`):

- 2023: 2,000 records, 6,000 mock calls, halt None, resume sends 0 new calls.
- 2024: 2,000 records, 6,000 mock calls, halt None, resume sends 0 new calls.
- 2025: 2,000 records, 6,000 mock calls, halt None, resume sends 0 new calls.
- 2021-22: 2,000 records, 6,000 mock calls, halt None, resume sends 0 new calls.

**Disclosures for v2.2.**

- First attempt stopped. The first population start (13:16Z) held the whole INPUT_META_COMPACT file in memory, and the
  three round-1 processes reached about 1.0-1.1 GB, above the 1 GB cap. They were stopped at 13:23Z, before any
  ROUND.json was written. Their partial ledgers were moved, not deleted, to
  `inputs_redacted_v2_2_aborted/2026-11-07T132433Z/`. The fix streams the metadata in lockstep with INPUTS (keys
  asserted equal) and caps the compiled-locator cache; peak working set is now about 0.29-0.34 GB per process. Two
  watchdogs (free RAM below 1,500 MB; any heavy process above 1,000 MB) ran for the whole population run and never
  fired; the lowest free RAM they logged was 4,833 MB.
- Concurrency. The first slot scheme (lock directories) let a fourth heavy process start between about 15:13Z and
  15:44Z: at most 4 ran at once, and free RAM stayed above about 5 GB. From 15:39Z an admission supervisor counted
  the heavy work actually running and admitted a step only below 3. No step was affected; each step is a separate
  process with its own inputs and outputs.
- Disk. D: fell to 24.5 GiB free at 18:52Z, mostly from writes outside this task, and the runtime refused the first
  freeze write. The v2.2 outputs were then NTFS-compressed. This does not change their content hashes; the
  population rounds had already written them. D: recovered to 32.7 GiB, and the freeze was written at 18:56Z.
- Code versions of `td4v2_redact22.py`. The 2023-2025 round-1 processes ran the version with SHA256 5a317a05...6318.
  Two edits followed while round 1 was running, and each `ROUND.json` / `VERIFY22` file records the file hash at the
  time it was written (1e8641fa...9eb3, then 36df37a3...94ce). The edits change only rounds 2-3:
  - an F2 residual class from the previous round is not counted as "unlocatable", since process() re-locates it and
    the round's own residual checks it;
  - in rounds 2-3, an input that the previous re-run did not flag is not processed again. Its text is the previous
    round's fixpoint, under the same tables, patterns and locators. `idempotency22` checks the first 2,000 flagged and
    1,000 unflagged inputs of each such round.

  Round 1 gives the same output under all three versions. The 2021-22 round 1 and every later step ran 36df37a3.
- F2 residual after each round = the inputs whose fixpoint was not reached within 6 passes, plus a rescan of the first
  3,000 callable inputs with the F2 locators (`RESIDUAL_*.json` `v22_rescan`), and the frozen-GB rescan.
- Not solved by design: subsidiaries, brands and units known only from world knowledge, with no name word shared with
  the firm's tables; executive names; HQ cities. Over-redaction is measured above: whole words around matches, codes,
  and ordinary words next to a give-back slot.
- `e_slot_context` (both neighbours of a slot) applies only to hidden generic acronyms (GENERIC_ACR). This is the form
  of the 2021-22 tagline case; ordinary English tickers are not widened.

**Freeze A v2.2** (`FREEZE_TD4_A_v2_2.json`, schema `TD4_FREEZE_A_v2_2`). It binds 864 artifacts, 112 external files
and 59 code files, including:
- the v2.2 rounds, ledgers, hashes, VERIFY22 and RESIDUAL files, and the round-3 INPUTS;
- the twelve v2.2 detector re-run folders;
- REDACTION_FINAL_v2_2, CHECK22 and labeling_v2r2;
- the superseded v2 and v2.1 freezes, requests and pre-registrations, as the record.

`PREREGISTRATION_v2_2.json` changes only the redaction text (`inputs.redaction.v2_2_fix`, the F3 wording in
`still_flagged`, the human-read caveat). It adds erratum V5: the changed label-packet contexts are relabelled by Codex
(OpenAI gpt-6-astra), as third-LLM diagnostics only, not by the researcher.

The reviewer writes `D:/OptResearch_data/five_session/clearance/TD4_CLEARANCE_A_v2_2.json` with:
- schema `TD4_CLEARANCE_A_v2_2` and status CLEARED;
- `freeze_sha256` = SHA256 of `FREEZE_TD4_A_v2_2.json`;
- the fields of `CLEARANCE_REQUEST_TD4_A_v2_2.json`.

The builder wrote the request only.

| File (private root) | SHA256 |
| --- | --- |
| `PREREGISTRATION_v2_2.json` | 7a9fc3f25153f02c97ce715d1c5d43d8dd1d28d1f3fb79572a9255206f63e025 |
| `FREEZE_TD4_A_v2_2.json` | 1a5c20e56dfeab5da431dc1d381248ba89ed4af488dc501d4ecebf5347f26123 |
| `CLEARANCE_REQUEST_TD4_A_v2_2.json` | 0ae96d09aa1048761b8a2538a8931ad1fa6e5701d1966d19ac495b18d4d00ffa |
| `QUALIFICATION_TD4v2_1ce0c9aaf8c64bd88e9e173eef2c212b.json` | 0125f0efc4a4c0df480b112e4b2a905da3de4b56c10c50f748c09d33275876f4 |
| `COST_PROJECTION_v2_2_b40885cd.json` | 68f1f015b11f0fc73bc14cd4824ea069dc3daa70a591cc5de986842b4f41279a |
| `parity/PARITY_construction_redacted_v2_2_2cf58e99.json` | 3bbd5f1ca16425778a2cc060c5d62b499cba530aee079da2be98aa136e4533b1 |
| `inputs_m2/select_2122_r3/REDACTION_FINAL_v2_2_2026-11-07T174033Z.json` | 5f2e2bfce5974e2b3abae95ff7f9b3656d1e89db14974b82bf09075412d2be0c |
| `inputs_m2/period_2023_r3/REDACTION_FINAL_v2_2_2026-11-07T174714Z.json` | 131956c783588a1683bf3c9d4b4d68b642c83587f063b47dac5d713aff10dd55 |
| `inputs_m2/period_2024_r3/REDACTION_FINAL_v2_2_2026-11-07T184208Z.json` | 06a420197aff7ebd5fcd5890567cb9e9889909bbb852fe95ad0ec8d48745d604 |
| `inputs_m2/period_2025_r3/REDACTION_FINAL_v2_2_2026-11-07T175939Z.json` | 9c7b2d2345bab1c4d3f94f04880c718519b18d31b6457772d9ad3ace8dbd2904 |
| `inputs_redacted_v2_2/2122/CHECK22_2026-11-07T174257Z.json` | 889b705a63c2cfb58d1daa1151d7f574be9e65ad42b7a8fad3131afbd00e776b |
| `inputs_redacted_v2_2/2023/CHECK22_2026-11-07T175446Z.json` | 3ae58facde9ce80bce6113dacb06e53b978d64e28d479e30b5d74321a8fdafa1 |
| `inputs_redacted_v2_2/2024/CHECK22_2026-11-07T185009Z.json` | f909efa9dcdd4ba4b86caf6b7ce2e1451d2e90dbaf54cdbae7235889467a83a7 |
| `inputs_redacted_v2_2/2025/CHECK22_2026-11-07T180613Z.json` | aaceefd090ca72f023d70f362f6dcd8df361272155406574f84cebbbc8b5ef15 |
| `labeling_v2r2/PACKET_INDEX_v2r2.json` | 578f09d4a8f49e01ce8222981a9a4d3b95aa8a1e19118202148cddb270c8e23e |

The full list of the v2.2 files, with SHA256, is the freeze's `artifact_sha256`.

**Code changes for v2.2** (since freeze A v2.1):

| File | Change |
| --- | --- |
| `td4v2_redact22.py` (new) | v2.2 transform: word definition (F1), F2 locators, fixpoint `process`, GB context and slot context, ledger-only rounds, `prepare_rerun` / `residual` through the round reader, `verify_round22`, `idempotency22`, `finalize22` |
| `td4v2_render.py` (new) | `render`, `ledger_index`, `render_record`, `iter_round`, `open_round` (file-like reader for the detector copies) |
| `td4v2_check22.py` (new) | review classes D1, D2, N1, N1b, N7, N4er and P6 on sent texts; placeholders by type; over-redaction share, v2.1 against v2.2 |
| `td4v2_tests_redact22.py` (new) | 13 tests (F1, F2 a-f, F3, fixpoint, ledger-only rendering, v2.2 result preferred) |
| `td4v2_redact_final.py` | `final_path` prefers `REDACTION_FINAL_v2_2_*`; the rehearsal stand-in carries the rule version |
| `td4v2_spec.py` | v2.2 paths, `LABELS_R2`, superseded v2.1 files, `RULE_VERSION`; pre-registration redaction text, `v2_2_fix`, erratum V5, `redaction_manifest_v2_2` |
| `td4v2_freeze.py` | schema `TD4_FREEZE_A_v2_2`; binds the v2.2 rounds, CHECK22, labeling_v2r2; the build receipts require the v2.2 result |
| `td4v2_labels.py` | `redacted_packet_v22` (labeling_v2r2), `convert(revised='v2r2')`, `score(..., path_r2)` |
| `td4v2_cost.py`, `td4v2_parity.py`, `td4v2_runner.py` | v2.2 file and folder names, rule version and redaction file in the outputs |
| `run_td4v2.py` | commands `redact22`, `redact22-prepare`, `redact22-residual`, `redact22-verify`, `redact22-final`, `check22`, `packet-r2`, `convert-labels-r2`; `score-labels` takes a third template |
| `td4v2_tests.py` | runs `td4v2_tests_redact22` in `qualify` |

Wall time of v2.2: 2026-11-07 about 12:10-18:57 UTC (population rounds 13:44-18:42 UTC).

## 18. Addendum: v2.3 evaluator memory and crash-safety fix (2026-11-09)

Written 2026-11-09T07:15Z (from `date -u`). Where this addendum and sections 1-17 differ, this addendum governs. Nothing
of v2.2 is overwritten: `runs_v2/`, `selection_v2/`, the clearance folder, the earlier freezes, requests and
pre-registrations, `inputs_m2/*_r3` and `inputs_redacted_v2_2/` were not written (read only).

**Trigger.** The TD4C clearance-B review of v2.2 (`five_session_review/td4c/TD4C_AUDIT_B_v2_2.md`, SHA256
46762b99...57b4) failed on item B6-M: `td4v2_evaluate.run` kept every 2023-2025 provider output in memory and then ran
`actual_costs()` after the once-only latch; the reviewer projected 1.5-1.9 GiB committed against the 1,400,000,000-byte
job cap, so a MemoryError after the latch would have lost the single evaluation. Fix brief: `TD4_FIX_BRIEF_v2_3.md`
(F1-F8). No provider or LLM call, no 2023-2025 label or return read, no git, no deletion.

**What changed** (no number, rule, grid, confirmation logic, pre-registration or provider output changes):

- F1. `actual_costs()` runs right after `admit_b`, before any provider output is collected and before the latch
  (ledgers and inputs only); only its small result is kept.
- F2. `td4v2_slim.collect_slim` keeps, per test year, the main-pass fields that `td4_score.provider_outputs` reads
  (http_ok, status and the arm's one field, the field only for valid outputs); raw responses, re-ask rows, unsent keys
  and every other field are dropped while each line is streamed. The unchanged `td4_score.provider_outputs` builds the
  arm values from it. Same selection rule as `td4v2_runner.collect_root` (numeric session order N4, input-hash check
  N5, first HTTP-200 output, otherwise the last attempt).
- F3. Per-year keys, labels, compact outputs and eligible maps are released once used; the pooled frame is released
  after the primary result. `JOB_MEMORY_CAP` is unchanged (1,400,000,000).
- F4. The pooled `evaluate_core` result and its stop-rule readings are written to `evaluation_v2/EVALUATION_PRIMARY_v2.json`
  (bound in freeze A v2.3, `evaluation_paths`) before any descriptive part runs. Per year, the 2025 halves, S1 and
  product continuity (per year) each run in their own try/except (MemoryError included; a value with a non-finite float
  is recorded as an error, not written); the result file repeats the primary result as written and records its SHA256.
  `admit_b` refuses first when the latch, the primary file or the result exists; there is no replay path. The evaluator
  also runs `td4_env.guard()` (the disk and memory floors that `test_labels` checks) just before the latch.
- F5. Wording only: P1/P2 `comparison` says "chosen on 2021-2022"; P1 and P2 carry `confirmation_meaning`
  ("interval above zero" in an exploratory historical reproduction of 2023-2025, the TODO caveat), `exploratory: True`
  (the TD2 `paired` marker, now also on P2) and `exploratory_note`; the result label says the same. The v1 module
  `td4_evaluate.py` is unchanged; the v2 evaluator overwrites only these text fields.
- F7. Freeze A v2.3; `call` and `reask` refuse (in `run_td4v2.main` before the runtime is installed, and in
  `td4v2_gate.admit`); `select` writes `selection_v2_3/SELECTION_FREEZE_v2_3.json` and request B
  `selection_v2_3/CLEARANCE_REQUEST_TD4_B_v2_3.json`; clearances `TD4_CLEARANCE_A_v2_3.json` (schema
  `TD4_CLEARANCE_A_v2_3`) and `TD4_CLEARANCE_B_v2_3.json` (schema `TD4_CLEARANCE_B_v2_3`, `clearance_a_sha256` = SHA256 of
  clearance A v2.3). `check_a` (used by select and evaluate) also recomputes `provider_outputs_record` (per-period ledger
  digest and the SHA256 of every `runs_v2/real_<P>` file) and refuses on any change.

Code files (SHA256 old -> new; every `td4/*.py` v1 file and `td4v2_runner.py` are unchanged):

| File | Old | New | Change |
| --- | --- | --- | --- |
| `v2/td4v2_evaluate.py` | c551445d...91fc | 122bfadf...7880 | F1, F3, F4, F5 |
| `v2/td4v2_gate.py` | cfe37baf...2dee | 15c9288c...f84b | F7: call/reask closed, `runs_digests`/`check_runs_record`, once-only files first, v2.3 bindings and status |
| `v2/td4v2_spec.py` | 002367af...5610 | bced74e5...6314 | v2.3 paths, `CODE_VERSION`, `EVAL_PRIMARY`, `SELECTION_V2_RECORD`, `CLEARANCE_A_V2_2_SHA256`, memory target; `PREREGISTRATION` unchanged |
| `v2/td4v2_freeze.py` | 0c36b35a...4312 | 189339f8...999e | freeze A v2.3: v2.2 artifacts re-bound, `provider_outputs_record`, v2.3 evidence, comparison with v2.2 |
| `v2/run_td4v2.py` | fdec9062...bbd7 | e6588d12...d52b | `slim-parity`, `mem-rehearsal`; call/reask refused |
| `v2/td4v2_rehearsal.py` | 181cdd6e...9a84 | 8818f768...1851 | the dev-year rehearsal refuses in v2.3 (it starts with `admit('call')` and reads 2021-2022 labels); sandbox paths for the primary file |
| `v2/td4v2_tests.py` | 0db7c5cc...2d89 | e4393d37...f7cd | `qualify` runs the v2.3 tests |
| `v2/td4v2_tests_m2.py` | 06b312ed...4709 | 8a087e77...4dde | the R1 test also sandboxes `EVAL_PRIMARY` |
| `v2/td4v2_slim.py` (new) | - | 5764e7cc...8de0 | F2 compact outputs, parity check, memory helper |
| `v2/td4v2_memrehearsal.py` (new) | - | 1248a907...105a | F6 rehearsal |
| `v2/td4v2_tests_v23.py` (new) | - | af753b6c...cf18 | 10 tests (F1, F2 x2, F4 x2, F5, F7 x4) |

**F2 parity on real provider outputs** (`slim-parity`, no label read; rows = the year's primary feature rows, 2122 = the
subsample's ELIGIBLE rows; every arm's ordering, choice, yes_scores, status and valid compared element by element and
type by type, plus a digest; the eligible map of the v2.3 call equals the v2.2 call):

| Period | Receipt (`OUT/v2_3/`) | SHA256 | Rows | Equal | Compact store + eligible map (Python heap) | v2.2 rows (commit growth) |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | `SLIM_PARITY_2023_c222417d.json` | 1906f70f...322e | 46,432 | yes, 0 differences | 42,256,282 | 355,885,056 |
| 2024 | `SLIM_PARITY_2024_973c7199.json` | 9bd63ee0...0602 | 48,993 | yes, 0 differences | 46,439,659 | 384,503,808 |
| 2025 | `SLIM_PARITY_2025_7180c9af.json` | 383ed46a...e209 | 46,469 | yes, 0 differences | 42,335,757 | 352,145,408 |
| 2122 | `SLIM_PARITY_2122_59a80b4c.json` | 5de1c28c...852d | 20,000 | yes, 0 differences | 19,131,909 | 153,620,480 |

The qualification also compares old and new on invented edge cases and on the builder's rehearsal fixtures
(`rehearsal_v2_5e8e8422/runs_v2/real_2122` and `real_2022`, mock outputs with a re-ask session).

**F6 full-size memory rehearsal** (`mem-rehearsal`; receipt `rehearsal_mem_v2_3_913e5a66/MEMREHEARSAL_v2_3.json`, SHA256
32ce6033cdb4c94fd35dc4c8de96b8948eb677138f806b097e410d030dae3d0a, 2026-11-09T07:10:30-07:12:37Z, 126.5 s). The complete
v2.3 `td4v2_evaluate.run` ran on the real 2023-2025 provider outputs, ledgers, sent inputs, features, TD1 b and TD2
heads, in a fresh process under the real runtime (job cap 1,400,000,000 bytes, single-thread BLAS set before numpy
loads, offline audit hook). Replacements: `td4v2_outcomes.test_labels` by a synthetic-label stub (same token, year and
seal checks, same read windows, same body and the unchanged `td4_labels.label` on a SHA256-seeded synthetic CRSP-shaped
panel of 5,800 PERMNOs, 1.51-1.58 million rows a year against 1,314,536 rows in the 2022 CRSP file); sentinels on
`td4_labels._crsp/compute/development` and `selection_labels`; `admit_b` by a sandbox gate (once-only check, real
ledger-digest recomputation and model-file hashes; no clearance read). Latch, primary, result and receipt went to the
sandbox; the selection freeze read was `selection_v2/SELECTION_FREEZE_v2.json` (read only).

| Measure | Bytes | Target 1,100,000,000 |
| --- | --- | --- |
| Peak working set (process) | 798,851,072 | within |
| Peak commit (process, PeakPagefileUsage) | 761,282,560 | within |
| Peak process / job memory used (job object) | 761,282,560 / 761,282,560 | within |

Phases (working set / commit, bytes): costs 195.6M / 157.0M (the peak, about 761M, is inside `actual_costs`, at
`completion` of the largest period, before any output is collected); collect 2023 248.8M / 210.7M; 2024 314.6M / 276.7M;
2025 364.3M / 325.8M; labels and frame 2023 475.3M / 447.6M; 2024 490.4M / 461.0M; 2025 505.7M / 503.4M; pooled primary
499.4M / 496.6M; per year 500.0M / 497.1M; 2025 halves 501.7M / 500.1M; S1 504.3M / 502.1M; product continuity
437.9M / 433.0M. Order: admit_b first; costs before collection; collection before the latch; latch before the first
label; primary before every descriptive part; result after every descriptive part; no descriptive error; the result
repeats the primary. The real `evaluation_v2/` latch, primary and result paths did not exist before or after.

File-open trace (audit hook on every open, wrappers on every parquet/dataset/CSV reader; anything not allowlisted is
refused): **clean** - 0 refusals, 0 CRSP files opened, 0 reads with a label column. 118 data files were opened: 82
`runs_v2/real_<P>` session files, 16 `inputs_m2/*_r3` build files, the 4 round-3 INPUTS, TD2 features 2023/2024 and
features 2025, TD2 evidence 2023/2024 and evidence 2025, TD1 predictions 2023/2024, the 2025 manifest (22 projected
columns: keys and TD1 features, no label column), `model_frozen_v2.pkl`, `MODEL_v2.json`, `risk_menu.csv`,
`PREREGISTRATION_v2_2.json`, `SELECTION_FREEZE_v2.json`, and 2 sandbox files; 1,268 code and library opens. Stub windows:
2023 = 2023-01-01..12-31 + 2024-01-01..01-15; 2024 = 2024 + 2025-01-01..01-15; 2025 = 2025-01-01..12-31; 0 pseudo-labels
observed after 2025-12-31.

**Tests.** `QUALIFICATION_TD4v2_0f17f8798ed4408d93e9795e719ab95e.json` (SHA256
950230e7717cb5fc5b886e4aee0979428710ef7018564006769839b3a06f9453): 123 tests (46 v1, 77 v2 of which 10 v2.3), 0
failures, 0 errors, 0 skipped. New: costs before the collection and the latch; compact outputs identical (invented edge
cases; rehearsal fixtures); primary written before every descriptive part; a descriptive failure (MemoryError per year
and in S1, RuntimeError in product continuity, a non-finite value in the halves) contained with the primary identical to
a clean run; wording with unchanged numbers; call/reask refused (gate, runner, entry point); the new paths and the
unchanged pre-registration; once-only files checked first; the provider-output record detects a changed ledger.

**Freeze A v2.3** (`FREEZE_TD4_A_v2_3.json`, schema `TD4_FREEZE_A_v2_3`): 62 code files, 872 artifacts (all 864 of
v2.2 with the same SHA256, plus the v2.2 freeze and request, the qualification, the memory-rehearsal receipt and the
four parity receipts), 112 externals (all equal to v2.2), the same models, caps, question, periods and
`PREREGISTRATION_v2_2.json` (7a9fc3f2...e025). `provider_outputs_record`: clearance A v2.2 SHA256 bd0e0f67...e5d1, freeze A
v2.2 1a5c20e5...6f26123, request A v2.2 0ae96d09...0ffa; sessions 2 / 2 / 27 / 11 (2122 / 2023 / 2024 / 2025), every one
bound to clearance A v2.2, freeze A v2.2 and the unchanged runner code; ledger digests (taken 2026-11-09T07:13:22Z) 2122
8a81d564...02e5 (56,742 settle rows), 2023 aaedc8da...c184 (127,309), 2024 a1c436fb...574d (137,629), 2025
76681962...da45 (127,594) - the 2023-2025 digests equal the clearance-B review's B5 values; plus the SHA256 of every run
file. A read-only self-check of every `check_a` component against the new freeze and request (code inventory, 872
artifacts, 112 externals, pre-registration text and file, provider-output record, request bindings) passed; no
clearance-shaped file was written.

| File (private root) | SHA256 |
| --- | --- |
| `FREEZE_TD4_A_v2_3.json` | 9e35416bb098f3323e0bf9f4fd0f51f88e8a8f45f499c16bffd83d21acd35d93 |
| `CLEARANCE_REQUEST_TD4_A_v2_3.json` | 7070b892e199cab01a8318a5b30ba36a0519d8c28d58773c9b6238cf24bbe892 |
| `QUALIFICATION_TD4v2_0f17f8798ed4408d93e9795e719ab95e.json` | 950230e7717cb5fc5b886e4aee0979428710ef7018564006769839b3a06f9453 |
| `rehearsal_mem_v2_3_913e5a66/MEMREHEARSAL_v2_3.json` | 32ce6033cdb4c94fd35dc4c8de96b8948eb677138f806b097e410d030dae3d0a |

**Next steps (reviewers and runbook).** (1) Reviewer writes `TD4_CLEARANCE_A_v2_3.json` from the v2.3 request. (2)
`select` (writes `selection_v2_3/`; the numbers are deterministic, reproduced bitwise by the B review) and `request-b`
(or `select` writes it). (3) Reviewer writes `TD4_CLEARANCE_B_v2_3.json` (binds clearance A v2.3). (4) `evaluate`
once. New offline commands: `slim-parity --period P`, `mem-rehearsal`. Before `evaluate`, keep D: >= 26 GiB and C:
above 15 GiB with margin (the pre-latch guard refuses below the floors; after the latch, `test_labels` and the writers
check them again).

**Disclosures for v2.3.**

- Over the memory limits: the first development rehearsal (`rehearsal_mem_v2_3_21e16a63`, receipt SHA256 667b8a2f...e200,
  pass false) peaked at 1,266,257,920 bytes committed (working set 798,777,344): above the 1 GB per-process brief and the
  1.1 GB target, below the 1.4 GB job cap under which F6 runs. Cause: the harness imported pandas/numpy before the runtime
  set single-thread BLAS, so OpenBLAS committed per-thread buffers for 16 threads (commit 543 MiB at a 88 MiB working set
  before any work). The real `run_td4v2.py evaluate` installs the runtime before numpy loads; the harness now does the
  same and records `numpy_loaded_before_runtime` (false in the final run). The same run also stopped on two trace false
  positives (a no-op `mkdir` of the existing private root; pyarrow's internal nested `dataset()` call under a projected
  `read_table` of the 2025 manifest), both fixed; it had written only a sandbox latch.
- The qualification process committed 761,729,024 bytes (working set 284,291,072) for the same BLAS reason
  (`td4v2_tests` imports numpy before `qualify` installs the runtime; pre-existing, v2.2 qualification 761,565,184), under
  1 GB. Every other process stayed below 1 GB (parity at most 615,993,344 working set). At most 2 heavy processes ran at
  once; free RAM stayed above about 12.9 GB; C: about 18 GiB and D: about 32 GiB free throughout.
- Superseded development receipts, kept as the record (earlier code): rehearsal `rehearsal_mem_v2_3_d74cc816`
  (2bd92a80...8ca2, pass true, peaks 797,753,344 / 760,791,040) and parity `SLIM_PARITY_2023_5002a712.json`
  (288e9f78...0567) and `SLIM_PARITY_2024_7a0eda73.json` (cb872918...8b8b), both equal; the parity memory measurement was
  changed after them.
- Read before the rehearsal, metadata only: the row counts of the 2021 and 2022 CRSP files and the 2022 CRSP column types
  (to size the synthetic panel), and the column names and row counts of the 2023-2025 manifests and features (to confirm
  they hold no label column). No value was read. One ad-hoc schema probe of a 2023 responses file was blocked by the
  permission system and did not run; the response schema was taken from the runner code.
- The rehearsal gate stub does not run `check_a` (its streaming hashes include the bound 2025 CRSP snapshot, which the
  trace forbids); its memory is negligible. The 2025 b reads 18 TD1 feature columns of the 2025 manifest, including
  TD1's lagged pre-origin return features (rv1, rv5, rv22, ret1, ret5, ret22, own_downside5), as the real evaluator does;
  no label column and no CRSP return file is read.
- The sandbox result files of the rehearsals hold numbers on pseudo-labels (engineering only); the receipts keep no P1/P2
  number. Sandboxes and fixtures are kept (no deletion).
- `td4v2_rehearsal.run` (the v2 dev-year rehearsal) now refuses: under v2.3 its real-step path would start with the closed
  `admit('call')`. The full-size v2.3 rehearsal replaces it for the evaluator.
- Unchanged, to be reported with the result (B review, non-blocking): two notes on the selection outcome; the
  outcome itself is kept in the private copy of this handoff and in the private result report.

Wall time of v2.3: 2026-11-09 about 06:33-07:16 UTC.
