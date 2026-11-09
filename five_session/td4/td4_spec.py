"""TD4 frozen constants, rules and the pre-registered design (public file: no data, outputs, keys or article text).

TD4 is the last stage of the collaborator TODO (sections 3-4): provider selectors (DeepSeek order, Jev Choice, Jev
Noul) feeding the frozen five-session risk head, compared with the firm-only MLP.

Scope amendment of 2026-11-06 (researcher, relayed by the coordinator) replaces the original 2021-2024 design:
- selector calls only for 2024 (selection) and 2025 (confirmatory test);
- 2025 data are approved for THIS experiment only; every label window ends by 2025-12-31 and nothing dated 2026 or
  later is decoded or used;
- hyperparameters are frozen from TD1 (MLP architecture, inputs, imputation/scaling procedure, stopping rule) and TD2
  (menu, exposure dictionary, ridge penalty, article aggregation rule); no re-tuning;
- selection: MLP and head refit through 2023 with the frozen hyperparameters, 2024 scored with the selector gates;
- test: MLP and head refit through 2024, 2025 evaluated once, gated by clearance and a hashed pre-registration.

Nothing runs on import.
"""
import hashlib
import json
from pathlib import Path

VERSION = 'TD4-provider-selectors-v1'
CODE = Path(__file__).resolve().parent                    # .../five_session/td4 (public clone)
FIVE = CODE.parent                                         # .../five_session
TD1_CODE = FIVE / 'v2'
TD2_CODE = FIVE / 'td2'
TD3_CODE = FIVE / 'td3' / 'v6'
OUT = Path('D:/OptResearch_data/five_session/td4')          # private root (inputs, outputs, ledgers)
TD1 = Path('D:/OptResearch_data/five_session')
TD2 = TD1 / 'td2'
TD3 = TD1 / 'td3'
CLEARANCE_DIR = TD1 / 'clearance'
# Clearance A authorises the paid calls, the label-reading refits and the 2024 selection; clearance B (written by
# the reviewer after checking the selection freeze) authorises the once-only 2025 evaluation.
CLEARANCE_A = CLEARANCE_DIR / 'TD4_CLEARANCE_A_v1.json'
CLEARANCE_B = CLEARANCE_DIR / 'TD4_CLEARANCE_B_v1.json'
CLEARANCE_A_SCHEMA = 'TD4_CLEARANCE_A_v1'
CLEARANCE_B_SCHEMA = 'TD4_CLEARANCE_B_v1'

# Periods (amendment): selection 2024, confirmatory test 2025. Training data for the selection models end 2023,
# for the test models 2024. Hard seal for this experiment: 2025-12-31.
SELECTION_YEAR = 2024
TEST_YEAR = 2025
PERIODS = (SELECTION_YEAR, TEST_YEAR)
HARD_SEAL = '2025-12-31'
DEV_SEAL = '2022-12-31'          # TD1/TD2 development seal (parity checks only)
TD2_SEAL = '2024-12-31'          # TD1/TD2 evaluation-era seal (parity checks only)
FIRST_YEAR = 2015
TRAIN_END = {SELECTION_YEAR: 2023, TEST_YEAR: 2024}       # refit through this year (label end <= Dec 31)

NAMES = ['size', 'cash', 'cash_flow', 'leverage', 'inventory']
HOPS = (1, 2, 3)
K_GRID = [0, 1, 2, 3, 5, 'all']
NOUL_THRESHOLDS = [0.25, 0.5, 0.75]       # TD2 spec 'selection' rule (frozen grid for the Noul threshold rule)
FALLBACKS = ('none', 'all')               # TD2 fallback options; applied to rows with news but no valid output
LOCAL_SELECTORS = [('training_beta', 0), ('random', 72101), ('random', 72103), ('random', 72109)]   # TD2 references
HOP_ARMS = ['main', 1, 2, 3, '1r', '2r', '3r']                                                     # TD2 ablations
PROVIDER_ARMS = ('deepseek_order', 'jev_choice', 'jev_noul')

# Providers: questions, models and parsers are TD3 v6's, unchanged (imported from td3_spec / td3_providers).
DEEPSEEK_BALANCE_URL = 'https://api.deepseek.com/user/balance'
DEEPSEEK_BALANCE_FLOOR_USD = 8.00          # stop DeepSeek calls before the balance would fall below this
DEEPSEEK_BALANCE_READ_EVERY_CALLS = 250    # re-read the balance (read-only GET) every N settled DeepSeek calls
DEEPSEEK_BALANCE_READ_EVERY_S = 900        # ... or every 15 minutes, whichever is first
PEAK_GUARD_S = 120                         # no new DeepSeek call within 2 minutes of a peak window start
DEEPSEEK_CONCURRENCY = 6
JEV_CONCURRENCY = 3
JEV_429_BACKOFF_S = (5, 15, 45, 120, 300)  # Jev HTTP 429: retry with these waits (each attempt reserved/counted)
OTHER_RETRY_BACKOFF_S = (5, 20)            # TD3: transport error / HTTP 5xx, at most 2 extra attempts
REASK_N = 100                              # menu-order stability: fixed re-ask subset per period (identical bytes)
LABEL_PACKET_N = 150                       # blinded human-labeling packet (2024 contexts only)

# Resources (brief): one heavy process at a time, each under about 1 GB, pause below 1,500 MB available memory.
JOB_MEMORY_CAP = 1_400_000_000             # hard job cap; measured peaks are reported against the 1 GB target
MIN_AVAILABLE_MB = 1500
MIN_FREE_D_BYTES = 25 * 1024 ** 3
MIN_FREE_C_BYTES = 15 * 1024 ** 3

TD1_FREEZE_SHA256 = '05d18f9b6e2677faa4ea3a9303de199ede1bc98fde76b68f562411692c0e4a0d'
TD1_MODEL_SHA256 = '7ce4015df2fed29134363893afe1f54d713cfaf57ad99ce99acf2f8f511df8ce'
TD2_FREEZE_SHA256 = '2d383ffb4894f4054728b30f15c9ce2197a5d03537aac95033e011b66aff03c5'
TD2_MODEL_SHA256 = 'd5f0d41546450b19fa84be52657bb460ec203de6b15c303a0902d2a0be3e4426'
TD3_FREEZE_SHA256 = '3e1f12028d7acdc63939c718d287758f3c21b5108dd5ca81b25c3de3de7c1c7a'
TD3_INPUTS_SHA256 = 'b4de0b6ddfee40b0dd096c2e8bb583282e8f6ec2d104183b437f68282df2ba81'
TD3_TABLES_SHA256 = 'ba2fdc7b814faf9e5199b4054861eddf99282f4d956d9b9ad1a213243725fb8b'
TD3_QUESTION_SHA256 = '8e1284dd158ec26d1ba94880434319f0e4b0327c8226f5e6d2cb0a831132c03a'
TD3_PILOT_RUN = TD3 / 'runs' / 'real_0b596ec45502' / 'session_1'

# Local sources (inventoried in td4_inventory; the 2025 extracts are built by td4_sources).
PROCESSED = Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed')
CRSP_DAILY = Path('C:/OptResearch/data/macro/exp_crsp_daily_compact.parquet')
JKP = PROCESSED / 'jkp_full_chars.parquet'
QUARTERLY = PROCESSED / 'compustat_quarterly_full.csv'
EARNINGS = PROCESSED / 'earnings_dates.csv'
GVKEY_LINK = PROCESSED / 'gvkey_permno_link.csv'
FS_MAP = PROCESSED / 'factset_to_permno.csv'
ARTICLES = PROCESSED / 'articles_classified.parquet'
CRSP_NAME_FILES = (PROCESSED / 'crsp_stocknames.csv', PROCESSED / 'crsp_names.csv')
PANEL = Path('C:/OptResearch/results/hist_panel_2015_2024/panel.parquet')
FACTSET_RAW = Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/FactSetRevere/FactSetRevere')
FACTSET_RELATIONSHIPS = FACTSET_RAW / 'ent_scr_relationships.txt'      # direct rows (TD2 graph source, Oct 2025)
FACTSET_SUPPLY_CHAIN = FACTSET_RAW / 'ent_scr_supply_chain.txt'         # derived rows (panel universe source)
FSB5 = Path('C:/OptResearch/results/sc_risk/t7v3_20261028/model01/mech_round3/fsb/fsb5/'
            'FSB5_FSMAP_CANDIDATES_v1.csv')
FSB1_COMPANY = Path('D:/OptResearch_data/external/factset_baidu/FSB1_pre2025_v1/company_pre2025_v1.parquet')
PANEL_UNIVERSE_N = 1000                                                 # build_panel.py N_UNIVERSE

PRICES_NOTE = ('Prices are TD3 v6 constants (td3_spec.PRICES, JEV_PRICE_INPUT_PER_M); the pilot ledger measured '
               'tokens per call; peak windows 01:00-04:00 and 06:00-10:00 UTC Monday-Friday.')

# --------------------------------------------------------------------------------------------------------------------
# Pre-registration (written to a hashed file by td4_prereg before any 2025 outcome can be read; the evaluator refuses
# without it). Text is frozen here so that its hash is bound in the clearance request.
PREREGISTRATION = {
    'schema': 'TD4_PREREGISTRATION_v1',
    'version': VERSION,
    'experiment': 'Collaborator TODO sections 3-4: five-session log realized variance; provider selectors gate the '
                  'frozen shared ridge head; firm-only MLP benchmark.',
    'response': 'y = log(mean of the next five squared CRSP daily total returns), origin excluded; share_type 0/2; '
                'label window end <= 2025-12-31; rows whose label is missing are excluded from every arm alike; the '
                'historical panel rule y > -20 (halted or stale stock) is applied to the test rows as a key filter '
                'identical for all arms (the 2015-2024 panel keys already carry it).',
    'frozen_hyperparameters': {
        'td1': 'MLP (64, 32) ReLU, Adam, batch 4096, seed 41037, inputs = TD1 FEATURES plus missing indicators, '
               'median imputation and standard scaling and the target standardisation fitted on the 2015-2020 '
               'fitting rows (TD1 refit rule: preprocessing is not refitted), selected lr 0.001, alpha 0.01, '
               'stopping rule = exactly the TD1 selected 14 epochs of partial_fit on all eligible rows with label '
               'end <= Dec 31 of the last training year.',
        'td2': 'menu = TD2 risk_menu.csv (5 items); exposure dictionary = TD2 EXPOSURE_DICTIONARY_v1.json; article '
               'aggregation = TD2/TD3 rule (up to five news-bearing partners per exact hop, SHA256(cutoff, partner) '
               'priority, partner maximum lexicon score / 5, main Z = mean over hops; TD3 one most recent eligible '
               'article per shown partner in the selector input); ridge without intercept on Z scaled by the '
               '2015-2020 training RMS; penalties frozen from TD2 MODEL_v2.json (main 0.01, 1r 0.01, 2r 0.01, '
               '3r 1e5); refit residual = y minus the in-sample prediction of the MLP refitted through the same year.',
    },
    'selection': {
        'period': '2024 only (selection only; never reported as a test).',
        'models': 'MLP and heads refitted through 2023 (label end <= 2023-12-31); 2024 scored out of sample.',
        'grid': 'DeepSeek and Noul top-K for K in {0,1,2,3,5,all}; Noul threshold rule tau in {0.25,0.5,0.75} '
                '(every eligible item with yes score >= tau); Choice top-one-or-none; fallback for rows with news but '
                'no valid output (invalid, missing, refused or not called) in {none, all}; reference selectors '
                'training-|beta| (order by |standardised coefficient| of the through-2023 head) and random seeds '
                '72101/72103/72109 with TD2 K/fallback grid. All arms use the same rows, candidate values and beta.',
        'rule': 'Per arm, the K (or threshold) and fallback with the lowest 2024 MSE; ties keep the first grid entry '
                '(K order 0,1,2,3,5,all then thresholds; fallback none before all) (TD2 A10).',
        'best_pipeline_arm': 'Among the selected DeepSeek, Jev Choice, Jev Noul arms and the all-on arm (K = all, '
                             'q = 1 on every eligible item, no selector), the arm with the lowest 2024 MSE (ties: '
                             'this order).',
        'better_jev_arm': 'Between the selected Jev Choice and Jev Noul arms, the one with the lower 2024 MSE (ties: '
                          'Jev Choice).',
    },
    'test': {
        'period': '2025, evaluated exactly once after clearance B.',
        'models': 'MLP and heads refitted through 2024 (label end <= 2024-12-31) with the frozen hyperparameters.',
        'gates': 'The frozen 2024 selections applied unchanged to 2025 provider outputs.',
    },
    'primary_comparisons': {
        'P1': 'Best pipeline arm (chosen on 2024) versus the firm-only MLP: paired squared-error gain '
              '(MLP squared error minus arm squared error) on identical 2025 firm-weeks.',
        'P2': 'Better Jev arm (chosen on 2024) versus matched-input DeepSeek (the selected DeepSeek arm) with the '
              'same head: paired squared-error gain (DeepSeek squared error minus Jev squared error) on identical '
              '2025 firm-weeks.',
        'inference': 'Mean paired gain, two-way clustered by firm and calendar week (inclusion-exclusion, small-sample '
                     'factor G/(G-1) per dimension, as TD2 statistics.clustered_mean), two-sided normal 95% interval. '
                     'A gain counts as confirmed only if the interval excludes zero (positive gain = improvement). An '
                     'UNSUPPORTED (negative) variance estimate is reported as not confirmed.',
        'multiplicity': 'Two primary comparisons, each read against its own TODO stop rule; no other result is '
                        'confirmatory.',
    },
    'stop_rule_readings': {
        'P1_not_confirmed': 'The frozen graph/news pipeline does not improve on the firm-only NN in this test: report '
                            'no validated total gain.',
        'P2_not_confirmed': 'Jev does not improve on matched-input DeepSeek: report no Jev advantage.',
        'P1_confirmed': 'A pipeline-versus-NN gain alone does not isolate the model\'s contribution from graph/news '
                        'access.',
    },
    'descriptive_only': 'Every other arm, the training-|beta| and random references, coverage, cost, H1/H2/H3 '
                        'increments (frozen main beta on exact-hop Z and per-hop refit heads, same gates), the '
                        'identity-sensitivity variant if reported, and all diagnostics.',
    'labels_and_caveats': [
        '2024 results are selection-only.',
        'The 2025 result is CONFIRMATORY for this pipeline, with two caveats: (a) the TODO states that the earlier '
        'risk study had examined 2023-2025; (b) both providers\' models were trained after 2025, so masked articles '
        'may still be recognised (look-ahead through model memory cannot be excluded).',
        'FactSet relationship dates are vendor record dates from one October 2025 snapshot (not point-in-time '
        'publication dates); relationships published after the snapshot are absent for late-2025 cutoffs.',
        'Fundamentals are latest-extract values with availability lags, not certified revision-free point-in-time '
        'data.',
    ],
}


def preregistration_text():
    return json.dumps(PREREGISTRATION, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def preregistration_sha256():
    return hashlib.sha256(preregistration_text().encode('utf-8')).hexdigest()


RULES = {
    'periods': 'Selector calls for 2024 (selection) and 2025 (test) only; 2024 runs first, then 2025.',
    'eligibility': 'TD3 rule per period: TD2-style features row (primary variant), origin and cutoff in the period '
                   'year, has_news, mapped focal, at least one non-missing menu characteristic, and at least one '
                   'partner article passing TD3 re-verification. Every eligible firm-week gets one call per selector '
                   'with byte-identical input text.',
    'inputs': 'Built by the cleared TD3 v6 functions (td3_build.with_percentiles, verify_article, fetch_texts, '
              'priority, path_relations, assign_tokens, aliases_for, article_extras, article_entries, render, '
              'build_inputs; td3_mask.Masker with the frozen MASK_TABLES_v4.json). Changes are isolated to period '
              'parameters (td3_spec seal constants and source paths set per period) and one copied function '
              '(mapped_edges with the 2025 seal); see TD4_HANDOFF.md, Deviations.',
    'aggregation': 'TD3 AGGREGATION_RULE unchanged, frozen in AGGREGATION_FREEZE.json before any 2025 outcome is '
                   'read.',
    'gates': 'DeepSeek: first K eligible names of the validated ordering; an output with an invalid, duplicate or '
             'missing name is invalid (logged, never backfilled) and the row uses the fallback. Jev Choice: the '
             'chosen item if eligible (an ineligible choice activates nothing), none activates nothing; full '
             'probability map stored. Jev Noul: top K eligible items by yes score (ties by menu order) or every '
             'eligible item with yes score >= threshold. LLM outputs never enter as regressors; they only choose '
             'which deterministic characteristic interaction Z is active.',
    'head': 'One shared beta per refit year for every arm (TD2 head refitted with the frozen penalty); never '
            'refitted per selector; intraday slopes never used.',
    'no_future': 'Model inputs are TD1 features and TD2 Z built from information available strictly before the '
                 'cutoff; labels are read only by the gated refit, selection and evaluation steps.',
}
