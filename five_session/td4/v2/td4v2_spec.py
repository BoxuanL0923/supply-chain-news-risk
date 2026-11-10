"""TD4 v2 constants, period definitions and the pre-registered design (amendment 2 of 2026-11-06).

Amendment 2 (researcher, about 12:30 EST): "let us do 2023-2025 -- don't worry about it"; the blinded packet is
labeled by a Codex agent (OpenAI) instead of humans. Design = the TODO's own split:
- models frozen: TD1 model_frozen_v2.pkl and TD2 MODEL_v2.json; no refit;
- selection on 2021-2022 by TD2's protocol (rows scored by the models fitted before 2021: TD1 selection_mlp and the
  TD2 2015-2020 training head) on a 20,000-firm-week hash-ranked subsample of the TD3-eligible population;
- test on every eligible firm-week of 2023, 2024 and 2025, evaluated once on the pooled years (clearance B).
Every label window ends by 2025-12-31; nothing dated 2026 or later. Nothing runs on import.
v2.3 (2026-11-09): code paths only (freeze, request, clearance A/B, selection folder, primary-result file); the
PREREGISTRATION dict below and PREREGISTRATION_v2_2.json are unchanged.
"""
import hashlib
import json
from pathlib import Path

import td4_spec as T

VERSION = 'TD4-provider-selectors-v2'
CODE_V2 = Path(__file__).resolve().parent
OUT = T.OUT                                   # same private root (runtime write root); v2 files are new names
RUNS = OUT / 'runs_v2'
EV = OUT / 'evaluation_v2'
LABELS = OUT / 'labeling_v2'
LABELS_R = OUT / 'labeling_v2r'                 # redacted versions of the changed packet contexts (relabels)
LABELS_R2 = OUT / 'labeling_v2r2'               # v2.2 text of the contexts whose text differs from the labelled text
# v2.3 (2026-11-09, evaluator memory and crash-safety fix after the TD4C clearance-B review item B6-M): new freeze,
# request and clearance paths; every provider output was produced under clearance A v2.2 and no further provider call
# is possible (call and reask refuse); the pre-registration v2.2 is unchanged and bound as it is.
CODE_VERSION = 'v2.3'
CLEARANCE_A = T.CLEARANCE_DIR / 'TD4_CLEARANCE_A_v2_3.json'      # v2.3; the v2, v2.1 and v2.2 requests are superseded
CLEARANCE_B = T.CLEARANCE_DIR / 'TD4_CLEARANCE_B_v2_3.json'      # binds clearance A v2.3 (clearance_a_sha256)
CLEARANCE_A_SCHEMA = 'TD4_CLEARANCE_A_v2_3'
CLEARANCE_B_SCHEMA = 'TD4_CLEARANCE_B_v2_3'
FREEZE_A = OUT / 'FREEZE_TD4_A_v2_3.json'                       # v2.3; the v2, v2.1 and v2.2 freezes stay as the record
REQUEST_A = OUT / 'CLEARANCE_REQUEST_TD4_A_v2_3.json'
PREREG = OUT / 'PREREGISTRATION_v2_2.json'                     # unchanged in v2.3 (supersedes PREREGISTRATION_v2_1.json)
SUPERSEDED_V2 = {'freeze': OUT / 'FREEZE_TD4_A_v2.json', 'request': OUT / 'CLEARANCE_REQUEST_TD4_A_v2.json',
                 'prereg': OUT / 'PREREGISTRATION_v2.json',
                 'freeze_v2_1': OUT / 'FREEZE_TD4_A_v2_1.json', 'request_v2_1': OUT / 'CLEARANCE_REQUEST_TD4_A_v2_1.json',
                 'prereg_v2_1': OUT / 'PREREGISTRATION_v2_1.json',
                 'freeze_v2_2': OUT / 'FREEZE_TD4_A_v2_2.json', 'request_v2_2': OUT / 'CLEARANCE_REQUEST_TD4_A_v2_2.json'}
RULE_VERSION = 'v2.2'                            # redaction rule version (unchanged in v2.3)
AGGREGATION_FREEZE = OUT / 'AGGREGATION_FREEZE_v1.json'          # v1 file reused unchanged (TD3 rule unchanged)
SEL = OUT / 'selection_v2_3'                     # v2.3 select writes here; selection_v2/ (v2.2 code) stays untouched
SELECTION_FREEZE = SEL / 'SELECTION_FREEZE_v2_3.json'
REQUEST_B = SEL / 'CLEARANCE_REQUEST_TD4_B_v2_3.json'
SELECTION_V2_RECORD = {'selection_freeze_v2': OUT / 'selection_v2' / 'SELECTION_FREEZE_v2.json',
                       'request_b_v2': OUT / 'selection_v2' / 'CLEARANCE_REQUEST_TD4_B_v2.json'}   # record, never written
EVAL_LATCH = EV / 'EVALUATION_STARTED_v2.json'   # one once-only latch for the TD4 v2 evaluation (none exists)
EVAL_PRIMARY = EV / 'EVALUATION_PRIMARY_v2.json'  # v2.3 F4: pooled P1/P2 and readings, written before any descriptive part
EVAL_RESULT = EV / 'EVALUATION_RESULT_v2.json'
PROVIDER_CALLS_CLOSED = True                     # v2.3: call and reask refuse (td4v2_gate.admit)
CLEARANCE_A_V2_2 = T.CLEARANCE_DIR / 'TD4_CLEARANCE_A_v2_2.json'  # every runs_v2/real_<P> session ran under it
CLEARANCE_A_V2_2_SHA256 = 'bd0e0f67c9162a72c7782e9412bed98c9d794aadcd22cbb92b2defbf8bd2e5d1'
EVAL_MEMORY_TARGET_BYTES = 1_100_000_000         # v2.3 F3: measured peak target of the evaluate step (job cap unchanged)
V23 = OUT / 'v2_3'                               # v2.3 receipts (slim-output parity)

SELECT = '2122'
TEST_YEARS = [2023, 2024, 2025]
RUN_ORDER = [SELECT, '2023', '2024', '2025']          # a later period refuses until the earlier one is complete
PERIOD_REQ_YEAR = {SELECT: 2122, '2023': 2023, '2024': 2024, '2025': 2025}   # 'year' field of the v1 request keys
SUBSAMPLE_N = 20000
LABEL_END_BOUND = T.HARD_SEAL                    # every test label window ends by 2025-12-31
SELECT_LABEL_END = T.DEV_SEAL                    # TD1 development labels (label end <= 2022-12-31)
BALANCE_USD = 82.57                              # DeepSeek balance after the top-up (coordinator, 2026-11-06)
BALANCE_NOTE = 'DeepSeek balance USD 82.57 after a top-up (coordinator message, 2026-11-06); re-read at run start'
LABEL_PACKET_N = 150
LABEL_BATCH = 10
CODEX_LABELER = 'codex-openai-gpt-6-astra'
LLM_LABEL_NOTE = ('Diagnostic labels are LLM labels from a third provider (OpenAI gpt-6-astra via Codex), not human '
                  'ground truth.')

V2_2_FIX = {
    'trigger': 'TD4M3 redaction review of the v2.1 inputs (2026-11-07): FAIL (B1 readable name, unit, code and date '
               'residues in the reading sample; B2 unsound placeholder waiver for M4 GB_*; B3 placeholders inside '
               'words). Fix round v2.2 (coordinator brief F1-F6), redaction only; every other part of the design is '
               'unchanged. Written after the v2.2 rounds and before any provider call and before any outcome is read.',
    'F1_word_boundaries': 'A word is a maximal run of Unicode letters and digits (str.isalnum) joined by single '
                          'connectors (hyphens, en dash, apostrophes, ampersand, period, slash) between two such '
                          'characters; a leading news slug (BRIEF-, UPDATE n-, ...) is not part of the word. Every '
                          '[REDACTED]/[DATE] span covers whole words, and a word or word part glued to any placeholder '
                          'is redacted, so no ledger placeholder is next to a letter or digit (P6 = 0).',
    'F2_builder_locators': 'New deterministic locators (td4v2_redact22.locate_v22), run in every round with the '
                           'reviewers\' detectors and repeated on the provisional text until they find nothing more: '
                           '(a) own names and initialisms of the focal and shown firms in any case, with the guard '
                           'that an initialism or ticker that is an English or short common word is matched only in '
                           'upper case in a mixed-case line, and the English first name word in the firm\'s own '
                           'article; (b) hyphenated name parts and first-letter-cut forms; (c) compounds, name '
                           'fragments, codes built from tickers, initialisms or prefixes, the same code number under '
                           'another prefix, and long forms of short names; (d) the code, model, number or acronym '
                           'right after a role token, acronyms defined after a hidden name, and unit company names '
                           'right after a role token; (e) give-back phrases with their trigger words, frozen GB group '
                           'neighbour words, phrase slots that hide a generic acronym, and period-code-named firms in '
                           'period slots; (f) [YEAR]-yy and year ranges, Month [DATE] dd, month names next to [DATE] '
                           'and the masker-v2 swallow case. Name tables: the receipt-bound tables of td4v2_nosource; '
                           'lists built at run time from private data and never written to the public folder.',
    'F3_no_gb_waiver': 'See still_flagged: no placeholder waiver for M4 GB_*; only literal placeholder words are waived.',
    'F4_rule_A4': 'Unchanged: at most three rounds from the masker-v2 inputs (inputs_m2/*_r3, unmodified); inputs '
                  'still flagged or unlocatable after round 3, and inputs whose F2 fixpoint was not reached, are '
                  'withheld for all arms alike; the 2021-2022 adjudicated key sets are withheld; for 2021-2022 the M4 '
                  'firm-level classes still withhold (rule v2), for 2023-2025 MASKER_NO_SOURCE replaces them (rule '
                  'v2.1, unchanged).',
    'storage': 'Rounds 1-2 are ledger-only (LEDGER.jsonl, HASHES.jsonl); round 3 stores the full INPUTS.jsonl; any '
               'round is rebuilt with td4v2_render.render(masker-v2 text, cumulative ledger); the detector re-runs '
               'read rounds 1-2 through td4v2_render.open_round. Outputs under OUT/inputs_redacted_v2_2/<period>/.',
    'checks': 'td4v2_check22: the review\'s s04b classes D1, D2, N1, N1b, N7, N4er and P6 on every sent input, '
              'placeholders by type, and the share of article word tokens redacted, v2.1 against v2.2.',
    'result': {'2023': {'article_word_tokens_redacted_share': {'v2.1': 0.001038, 'v2.2': 0.002445},
                        'callable': 42336,
                        'ledger_entries_by_round': {'1': 122767, '2': 539, '3': 0},
                        'p6_sent_inputs': 0,
                        's04b_sent_inputs': {'D1': 0, 'D2': 0, 'N1': 0, 'N1b': 0, 'N4er': 0, 'N7': 0},
                        'sent': 42336,
                        'waived_literal_placeholder_hits': {},
                        'withheld': '0 (0.00%)',
                        'withheld_reasons': {'adjudicated': 0,
                                             'masker_no_source': 0,
                                             'still_flagged_after_round_3': 0,
                                             'unlocatable': 0}},
               '2024': {'article_word_tokens_redacted_share': {'v2.1': 0.000999, 'v2.2': 0.002031},
                        'callable': 45750,
                        'ledger_entries_by_round': {'1': 113519, '2': 426, '3': 6},
                        'p6_sent_inputs': 0,
                        's04b_sent_inputs': {'D1': 0, 'D2': 0, 'N1': 0, 'N1b': 0, 'N4er': 0, 'N7': 0},
                        'sent': 45750,
                        'waived_literal_placeholder_hits': {},
                        'withheld': '0 (0.00%)',
                        'withheld_reasons': {'adjudicated': 0,
                                             'masker_no_source': 0,
                                             'still_flagged_after_round_3': 0,
                                             'unlocatable': 0}},
               '2025': {'article_word_tokens_redacted_share': {'v2.1': 0.0011, 'v2.2': 0.002253},
                        'callable': 42430,
                        'ledger_entries_by_round': {'1': 101250, '2': 1439, '3': 84},
                        'p6_sent_inputs': 0,
                        's04b_sent_inputs': {'D1': 0, 'D2': 0, 'N1': 0, 'N1b': 0, 'N4er': 0, 'N7': 0},
                        'sent': 42430,
                        'waived_literal_placeholder_hits': {},
                        'withheld': '0 (0.00%)',
                        'withheld_reasons': {'adjudicated': 0,
                                             'masker_no_source': 0,
                                             'still_flagged_after_round_3': 0,
                                             'unlocatable': 0}},
               '2122': {'article_word_tokens_redacted_share': {'v2.1': 0.000667, 'v2.2': 0.001434},
                        'callable': 20379,
                        'ledger_entries_by_round': {'1': 23066, '2': 69, '3': 0},
                        'p6_sent_inputs': 0,
                        's04b_sent_inputs': {'D1': 0, 'D2': 0, 'N1': 0, 'N1b': 0, 'N4er': 0, 'N7': 0},
                        'sent': 18814,
                        'waived_literal_placeholder_hits': {},
                        'withheld': '1,184 (5.81%)',
                        'withheld_in_subsample': '1,165 (5.83%)',
                        'withheld_reasons': {'adjudicated': 1184,
                                             'masker_no_source': 0,
                                             'still_flagged_after_round_3': 0,
                                             'unlocatable': 0}}},
}

PREREGISTRATION = {
    'schema': 'TD4_PREREGISTRATION_v2',
    'version': VERSION,
    'supersedes': 'PREREGISTRATION_v2_1.json (same design; v2.2 redaction fix round after the TD4M3 redaction review, '
                  'inputs.redaction.v2_2_fix; erratum V5 in labels_and_caveats), PREREGISTRATION_v2.json (rule v2.1 '
                  'correction of the withholding classes, inputs.redaction.v2_1_correction) and PREREGISTRATION_v1.json '
                  '(amendment 1 design)',
    'rule_version': 'v2.2',
    'experiment': 'Collaborator TODO sections 3-4: five-session log realized variance; provider selectors gate the '
                  'frozen shared ridge head; firm-only MLP benchmark.',
    'response': 'y = log(mean of the next five squared CRSP daily total returns), origin excluded, share_type 0/2, '
                'label window end <= 2025-12-31 (2023 windows may end in January 2024); rows whose label is missing '
                'and rows with y <= -20 (historical panel rule) are excluded from every arm alike.',
    'models': 'Frozen, no refit: TD1 model_frozen_v2.pkl (MLP (64, 32), TD1 hyperparameters, refit through 2022) and '
              'TD2 MODEL_v2.json (shared ridge head and per-hop heads, penalty chosen on 2021-2022, refit through '
              '2022). Never the 98 intraday slopes; one beta for every arm.',
    'selection': {
        'rows': '20,000 TD3-eligible 2021-2022 firm-weeks: the smallest SHA256("TD4-select-2122|" + key), key = '
                'TD3 record key, over the full eligible population; rows with a label (end <= 2022-12-31) only.',
        'scoring_models': 'TD2 protocol: TD1 selection_mlp (fit 2015-2020) as b and the TD2 2015-2020 training heads '
                          '(training_beta), never the through-2022 refit; training-|beta| order = TD2 '
                          'training_beta_order.',
        'grid': 'DeepSeek and Noul top-K for K in {0,1,2,3,5,all}; Noul threshold tau in {0.25,0.5,0.75}; Choice '
                'top-one-or-none; fallback {none, all} for rows with news and no valid output; references '
                'training-|beta| and random seeds 72101/72103/72109 with the TD2 K/fallback grid; lowest selection '
                'MSE wins, ties keep the first grid entry (K order 0,1,2,3,5,all, then thresholds; none before all).',
        'best_pipeline_arm': 'Among the selected DeepSeek, Jev Choice and Jev Noul arms and the all-on arm, the lowest '
                             '2021-2022 MSE (ties: this order).',
        'better_jev_arm': 'Selected Jev Choice vs selected Jev Noul: lower 2021-2022 MSE (ties: Jev Choice).',
    },
    'test': {
        'rows': 'Every eligible firm-week of 2023, 2024 and 2025 (TD2/TD1 features; provider outputs where an input '
                'was sent; otherwise the arm fallback), pooled; evaluated exactly once after clearance B.',
        'models': 'TD1 refitted MLP b (predictions_v2 for 2023-2024; the same frozen network for 2025) and TD2 '
                  'MODEL_v2 beta (refit through 2022).',
    },
    'primary_comparisons': {
        'P1': 'Best pipeline arm (chosen on 2021-2022) versus the firm-only MLP on pooled 2023-2025: paired '
              'squared-error gain (MLP minus arm).',
        'P2': 'Better Jev arm (chosen on 2021-2022) versus matched-input DeepSeek (the selected DeepSeek arm), same '
              'head, pooled 2023-2025: paired squared-error gain (DeepSeek minus Jev).',
        'inference': 'Mean paired gain, two-way clustered by firm and calendar week (TD2 statistics.clustered_mean), '
                     'two-sided normal 95% interval; confirmed only if the interval lies above zero; an UNSUPPORTED '
                     'variance is not confirmed.',
    },
    'stop_rule_readings': {
        'P1_not_confirmed': 'The frozen graph/news pipeline does not improve on the firm-only NN in this test: report '
                            'no validated total gain.',
        'P2_not_confirmed': 'Jev does not improve on matched-input DeepSeek: report no Jev advantage.',
        'P1_confirmed': 'A pipeline-versus-NN gain alone does not isolate the model\'s contribution from graph/news '
                        'access.',
    },
    'secondary_S1_descriptive': {
        'design': 'Descriptive secondary S1 (coordinator decision of 2026-11-06, written before any outcome is read): '
                  'the same frozen through-2022 models (TD1 refitted MLP b, TD2 MODEL_v2 beta), the same grids, '
                  'fallback and tie rules, but selection on 2023 instead of 2021-2022; S1-P1 (best pipeline arm '
                  'chosen on 2023 versus the firm-only MLP) and S1-P2 (better Jev arm chosen on 2023 versus the '
                  'DeepSeek arm chosen on 2023) on pooled 2024-2025, paired squared-error gains with the same '
                  'two-way clustered two-sided 95% intervals. Labelled secondary and descriptive: never confirmatory, '
                  'no stop-rule reading.',
        'also_descriptive': '(a) whether the 2021-2022 and the 2023 selections pick the same configuration per arm '
                            '(and the same best pipeline arm and better Jev arm); (b) per test year 2023, 2024 and '
                            '2025 the in-year best configuration per arm, an oracle table that is not a valid forecast.',
        'once_only': 'Computed by the evaluator in the same once-only run as the primary, from the 2023-2025 outcomes '
                     'read once for the primary; no S1 selection file is written before that run, so clearance B '
                     'binds the one selection file (SELECTION_FREEZE_v2.json) and the result file holds S1.',
    },
    'descriptive_only': 'Per-year 2023/2024/2025 results, every arm, the references, coverage, cost, H1/H2/H3 '
                        'increments (same gates; frozen main beta on exact-hop Z and per-hop refit heads), the 2025 '
                        'first-half/second-half split (origins on or before 2025-06-06 versus after), and the '
                        'diagnostics (re-ask stability; NDCG, top-one, calibration against LLM labels).',
    'inputs': {
        'masker': 'Every period (2021-2022 subsample, 2023, 2024, 2025) is built with TD4 masker v2 (td4v2_mask: the '
                  'TD3 v6 masker plus the rules required by masking reviews td4m M1-M4), the D8 v2 story-field rule '
                  'and the mis-tag-notice rule; no input built with the v1 masker is used. The round-2 reviews '
                  '(td4m2 MA1 and MA2) found masker v2 FIX REQUIRED; by the researcher\'s decisions of 2026-11-06 '
                  '(masker v3 skipped; its code is kept for the record and is not used) and 2026-11-07 (span-level '
                  'redaction instead of whole-input withholding by detector flags) the masker is not revised and the '
                  'redaction rule below is applied to its inputs.',
        'withholding': 'A build residual (strict or date/boilerplate) or a population-check problem or an exact-class '
                       'v2 scanner hit withholds the input from all three providers alike; the row then takes each '
                       'arm\'s fallback.',
        'detectors': 'The blocking-class detectors of the masking reviewers: round 1 td4m M1, M2, M3 and M4 (re-run '
                     'through the round-2 adapters that import the round-1 modules read-only) and round 2 td4m2 MA1 '
                     '(classes C1-C8, D1-D4) and MA2 (G1-G3, B1, D), run over every callable input of every period with '
                     'only the input path, article store, withheld set and output folder changed; included classes, '
                     'excluded disclosed classes, the article-part scope and the SHA256 of every script are in '
                     'detector_flags_manifest. The whole-input detector-flag rule of 2026-11-06 is superseded: its '
                     'shares (33-63% of callable inputs) went back to the researcher and were not applied.',
        'redaction': {
            'transform': 'A separate, deterministic post-processing step on the masker-v2 inputs (td4v2_redact; the '
                         'masker is unchanged), applied identically for every arm and reference selector before any '
                         'provider call and before any outcome is read: every string matched by an included detector '
                         'class is replaced in article text only (the headline and text of each PARTNER NEWS article) '
                         'by a placeholder: [REDACTED] for name residues (names, name parts, tickers, RIC parts, '
                         'initialisms, codes, glued compounds, story IDs, URL residues, mis-tag wording), [DATE] for '
                         'date, year and story-time residues, [TERM] for give-back slots (the one role-token occurrence '
                         'that a give-back detector names, replaced whole: GB, IDX, MA2 G1-G3, s41 token patterns, '
                         's41b, M2 surface, M3 P5, M4 GB_*).',
            'never_touched': 'Every other pseudonym token, the frozen header (instructions, risk menu, characteristics, '
                             'graph state), article labels and field labels; a [REDACTED]/[DATE] span is trimmed to '
                             'article text and never covers a bracket token; a [TERM] span is exactly one role token '
                             'in article text.',
            'location': 'From each detector\'s own output: the matched string and its context window are re-found by '
                        'rebuilding the detector\'s context string at every candidate position (exact match); for '
                        'the key-only classes (s41_targeted, s41b_nasdaq, ma2_s12_dates_v2) and the date classes of '
                        's11, s14, s15 and s16, the detector\'s own regular expressions (read from the run copy) are '
                        're-run and every match in article text is replaced; GB groups are re-scanned at every '
                        'occurrence; MA2 single-word give-backs are located by re-running MA2\'s raw-versus-masked word '
                        'alignment. A flagged class of an input with no located span in article text withholds the '
                        'input (this covers the M4 firm-level classes NONAME and NONAME_AFTER_2022, which flag a firm '
                        'and have no text span). The reviewers\' adjudicated 2021-2022 key sets are withheld.',
            'rounds': 'Round r = 1, 2, 3 replaces the matches of the detector run on the round r-1 inputs (round 0 = the '
                      'masker-v2 inputs) and then re-runs every detector on the round r inputs (copies with only the '
                      'INPUTS path changed; population, metadata, article store and withheld set of the masker-v2 '
                      'build). At most three rounds; every input still flagged after the round-3 re-run is withheld for '
                      'all arms alike and takes each arm\'s fallback.',
            'still_flagged': 'An input is still flagged after round r when td4v2_flags.extract (same classes, '
                             'exclusions and article-part scope) lists it in the round-r re-run, or when a frozen GB '
                             'group still has a slot in its round-r text. Population-level detectors are frozen across '
                             'rounds: an s18 GB group (permno, word, side, neighbour) kept in any round so far stays a '
                             'group in every later round and is re-scanned at every occurrence, so a group that falls '
                             'below s18\'s ratio after redaction cannot release its remaining slots. Per-class hit caps '
                             '(s10 and s12 12, s14-s16 8, s17b 30 per class and input) only bound what one round '
                             'locates from the outputs: an input with a further match is listed by the re-run and is '
                             'still flagged. The M4 firm-level classes are listed by every re-run. Rule v2.2 (F3): '
                             'there is no placeholder waiver for the M4 give-back patterns GB_*; the phrase\'s trigger '
                             'words are redacted with the slot and every GB_* hit still present after round 3 is still '
                             'flagged; the only waiver is an M4 word hit that exists only because of a placeholder\'s '
                             'own characters (the word inside [TERM], [DATE] or [REDACTED]; '
                             'td4v2_redact22.literal_placeholder_hit), listed per class with its count. The v2/v2.1 '
                             'waiver of GB_* hits on a placeholder slot (td4v2_redact_final.placeholder_dependent) '
                             'is withdrawn.',
            'ledger': 'Every replacement is recorded in a private ledger (key, article, field, offset and length in '
                      'the masker-v2 text, classes, placeholder, SHA256 of the original string, round); no plaintext '
                      'in any public file. Checks: outside the ledger spans each redacted input is byte-identical to '
                      'its masker-v2 input; pseudonym tokens outside [TERM] spans, the header and the labels are '
                      'byte-identical; redaction is idempotent; the request body is built by the unchanged TD3/TD4 '
                      'construction code from the redacted text.',
            'reported': 'Per period: callable inputs, inputs with at least one redaction, redactions per input (mean, '
                        'p50, p95), share of article characters redacted, residual flags after each round, final '
                        'withheld count and share (a share above 2% is flagged), and the share without the two M4 '
                        'firm-level classes. The rule is not changed after it is run.',
            'v2_1_correction': {
                'rule': 'MASKER_NO_SOURCE replaces the M4 firm-level classes NONAME and NONAME_AFTER_2022 as a withholding '
                        'class for 2023, 2024 and 2025: an input is withheld (every arm alike) when its focal firm or a '
                        'shown firm (menu partner) has no name and no ticker entry in the exact name tables masker v2 used '
                        'for that period\'s build (CRSP stocknames/names rows of the firm and its issuer siblings dated on '
                        'or before the build\'s last day, FSB1 FactSet names and US tickers linked by issuer CUSIP6, and '
                        'for 2025 the bound CRSP CIZ 2025 file, as recorded by MASK_V2_RECEIPT; RIC roots do not count); '
                        'computed from those tables (td4v2_nosource), not from s13. 2021-2022 is unchanged (the '
                        'adjudicated key sets stay withheld; NONAME and NONAME_AFTER_2022 stay withholding classes and '
                        'flag no 2021-2022 input). Every other part of the redaction rule is unchanged.',
                'reason': 's13 NONAME tests for no CRSP name row dated on or before 2024-12-31 and does not read the CIZ '
                          '2025 source; NONAME_AFTER_2022 tests for a first CRSP name after 2022-12-31. Both stand in for '
                          '"the masker had no name to mask for this firm". In 2023-2025 that premise fails: recently '
                          'listed firms are ordinary and masker v2 masked them from CRSP (on or before the build\'s last '
                          'day) or CIZ (2025); their text residues are covered by the span detectors, which ran and were '
                          'redacted. The two classes flag a firm, not a string, so redaction cannot clear them.',
                'timing': 'Coordinator decision (technical correction by Claude, to be checked by the reviewer; the '
                          'researcher is told) of 2026-11-07, after the redaction rounds and before any provider call '
                          'and before any outcome is read; the v2 freeze and request were never cleared and stay as '
                          'the record.',
                'result': 'MASKER_NO_SOURCE flags 0 inputs in 2021-2022 (not applied), 2023, 2024 and 2025: every focal '
                          'and shown firm has a name or ticker entry in its build\'s tables.',
                'sensitivity_rule_v2_withheld': {'2122': '1,184 (5.81%)', '2023': '719 (1.70%)',
                                                 '2024': '4,010 (8.77%)', '2025': '6,741 (15.89%)'},
                'rule_v2_1_withheld': {'2122': '1,184 (5.81%)', '2023': '0 (0.00%)', '2024': '0 (0.00%)',
                                       '2025': '0 (0.00%)'},
                'sensitivity_note': 'The rule-v2 withheld shares (firm-level classes withholding) are reported as a '
                                    'sensitivity of the input set; the rule-v2.1 input set is the one sent.'},
            'v2_2_fix': V2_2_FIX,
        },
        'names_2025': 'The 2025 inputs are masked with an additional alias source: the CRSP CIZ security-information '
                      'rows that start in 2025 (issuer names and tickers of every PERMNO; coordinator pull of '
                      '2026-11-06, file SHA256 bound in freeze A v2). A population check withholds, for every arm '
                      'alike, any 2025 input in which a 2025-only CIZ name or ticker of its focal or a shown firm is '
                      'still readable. This replaces the hand rename table and the earlier post-WRDS withholding rule; '
                      'it uses no outcome and no provider output.',
        'factset_horizon': 'The sealed 2025 FactSet graph (October 2025 snapshot) has no relationship start or end '
                           'after 2025-06-06 (review TD4C N8): for the 2025 origins after that date the graph is frozen '
                           'at 2025-06-06 and no termination is observed; this is stated with every 2025 result.',
    },
    'runner_rules': {
        'terminal_client_errors': 'A unit whose request receives a deterministic HTTP 4xx client error (400, 404, 413 '
                                  'or 422) in two separate sessions and never an HTTP 200 is settled without output '
                                  '(the arm fallback), within a cap of max(3, 0.2% of the units) per provider and '
                                  'period; above the cap the period stays incomplete and stops for review.',
        'model_violation': 'A session halted because a provider returned a model outside the allowlist can be resumed '
                           'only after a reviewer writes the override file for that session '
                           '(overrides/MODEL_OVERRIDE_<period>_<session>.json).',
        'output_matching': 'A provider output is used only when its stored input SHA256 equals the sent input\'s '
                           'SHA256 (and its key matches); the first HTTP-200 output per unit is used, sessions in '
                           'numeric order.',
    },
    'labels_and_caveats': [
        '2023-2025 is a historical reproduction of a period the earlier risk study already examined (TODO); the '
        'researcher chose to run it anyway.',
        'Both providers\' models were trained after 2025, so masked articles may be recognised (model memory).',
        'FactSet relationship dates are vendor record dates from one October 2025 snapshot, not point-in-time; its '
        'relationship horizon ends on 2025-06-06.',
        'Executive names, owned brands without a firm word and unique descriptions stay readable (disclosed TD3 '
        'limitation; counted by the v2 scanner): residual identifiability from these remains in most inputs.',
        'Masking was human-read on the round-1 samples (2024, 2025) and the round-2 samples (2021-2022); the 2023, '
        '2024 and 2025 masker-v2 inputs were not human-read beyond the round-1 samples. The v2.1 redacted inputs were '
        'read by the TD4M3 redaction review (30 sent inputs per period, FAIL); the v2.2 inputs are read again by that '
        'review with a new seed before clearance. The span-level redaction replaces what the reviewers\' detectors '
        'and the v2.2 builder locators flag; recall is not measured, so some give-backs and name residues (for '
        'example subsidiaries and brands known only from world knowledge) remain in the sent inputs. False positives '
        'are redacted too (ordinary words, prices, model numbers, whole words around a match), which removes some '
        'text for every arm alike, and a placeholder shows that something was removed.',
        'Diagnostic labels: the label-packet contexts whose text the redaction changed are relabelled by Codex '
        '(OpenAI gpt-6-astra; labeling_v2r for the v2.1 text, labeling_v2r2 for the contexts whose v2.2 text differs '
        'from the labelled text), as third-LLM diagnostics only; a changed context without its relabel and every '
        'withheld context are excluded from the diagnostics. Erratum V5: PREREGISTRATION_v2_1 said these contexts are '
        'relabelled by the researcher; they were relabelled by Codex.',
        LLM_LABEL_NOTE,
    ],
}
TERMINAL_STATUS = (400, 404, 413, 422)
TERMINAL_MIN_SESSIONS = 2
TERMINAL_CAP_FRACTION = 0.002
TERMINAL_CAP_MIN = 3
OVERRIDES = OUT / 'overrides'
FACTSET_HORIZON = '2025-06-06'


def preregistration_text():
    """The pre-registration with the detector manifest (included classes, excluded classes, SHA256 of every reviewer
    detector script; td4v2_flags.manifest(), a pure function of those files) and the redaction manifest (SHA256 of the
    transform and final modules, placeholders, rounds, firm-level classes)."""
    import td4v2_flags as FL
    import td4v2_redact as RD
    pre = dict(PREREGISTRATION)
    pre['detector_flags_manifest'] = FL.manifest()
    pre['redaction_manifest'] = {
        'transform_code_sha256': hashlib.sha256(Path(RD.__file__).read_bytes()).hexdigest(),
        'final_code_sha256': hashlib.sha256((CODE_V2 / 'td4v2_redact_final.py').read_bytes()).hexdigest(),
        'placeholders': RD.PH, 'rounds': RD.ROUNDS, 'firm_level_classes': list(RD.FIRM_LEVEL),
        'term_classes': list(RD.TERM_CLASSES) + ['M4:GB_*', 'MA1:targeted (token patterns)'],
        'output_root': str(RD.ROOT)}
    import td4v2_redact22 as R22
    import td4v2_render as RR
    import td4v2_check22 as C22
    pre['redaction_manifest_v2_2'] = {
        'transform_code_sha256': hashlib.sha256(Path(R22.__file__).read_bytes()).hexdigest(),
        'render_code_sha256': hashlib.sha256(Path(RR.__file__).read_bytes()).hexdigest(),
        'check_code_sha256': hashlib.sha256(Path(C22.__file__).read_bytes()).hexdigest(),
        'final_path_code_sha256': hashlib.sha256((CODE_V2 / 'td4v2_redact_final.py').read_bytes()).hexdigest(),
        'placeholders': R22.PH, 'rounds': R22.ROUNDS, 'fixpoint_passes': R22.ITER, 'rescan_n': R22.RESCAN_N,
        'output_root': str(R22.ROOT)}
    import td4v2_nosource as NS
    pre['masker_no_source_manifest'] = {
        'code_sha256': hashlib.sha256(Path(NS.__file__).read_bytes()).hexdigest(), 'periods_applied': ['2023', '2024', '2025'],
        'tables': 'CRSP name files (td4_spec.CRSP_NAME_FILES), FSB1 company file (td4_spec.FSB1_COMPANY), CIZ 2025 '
                  '(td4v2_names2025.CIZ, SHA256 bound in the 2025 MASK_V2_RECEIPT)'}
    return json.dumps(pre, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def preregistration_sha256():
    return hashlib.sha256(preregistration_text().encode('utf-8')).hexdigest()
