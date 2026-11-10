"""TD2 v2 amendments, written before any risk-head fit or evaluation.

Values not redefined here are imported unchanged from spec.py (v1). No private
observations or licensed text are stored in this file.
"""
from spec import (NAMES, K_GRID, SEEDS, PARTNER_CAP, LOOKBACK_DAYS, MENU, LEXICONS,
                  RULES as RULES_V1, SLOT_SCHEMAS, VERSION as VERSION_V1)

VERSION = 'TD2-five-session-v2'
HOPS = (1, 2, 3)
VARIANTS = ('primary', 'fsb5_sensitivity')
SELECTORS = [('training_beta', 0)] + [('random', s) for s in SEEDS]
FALLBACKS = ('none', 'all')
# A1: 1e-2 .. 1e8. With about 2.7e5 training rows and unit-RMS columns the v1
# grid (max 1e3) moved coefficients by under 0.4 percent, so it could not shrink.
RIDGE_GRID = [10.0 ** k for k in range(-2, 9)]
HOP_ARMS = ['main', 1, 2, 3, '1r', '2r', '3r']
EVAL_YEARS = (2023, 2024)
MAX_LABEL_END = '2024-12-31'
FIRST_YEAR, LAST_DEV_YEAR, LAST_YEAR = 2015, 2022, 2024

AMENDMENTS = {
    'A1_ridge_grid': 'Penalty grid 1e-2..1e8 (factor 10). v1 grid max 1e3 is below 0.4% of the x\'x diagonal '
                     '(~2.7e5 training rows, unit-RMS columns) and could not shrink.',
    'A2_beta_order': 'training_beta selector orders by |standardized training coefficient| (coefficient on '
                     'training-RMS-scaled Z), ties by name. v1 ordered by the raw-unit coefficient, which depends '
                     'on characteristic units (log dollars versus ratios).',
    'A3_hop_ablation': 'Primary ablation unchanged from v1: the frozen main coefficients applied to exact-hop Z_h on '
                       'identical rows, no refit. Added secondary arms 1r,2r,3r: one ridge head per exact hop '
                       'fitted on Z_h alone with the same grid, 2021-2022 penalty choice and through-2022 refit, '
                       'identical rows and budgets.',
    'A4_news_identity': 'The frozen development RIC->PERMNO mapping (panel keys <=2022) is used for every year, '
                        'including 2023-2024, as the v1 code already did. The v1 rule text "<=2024 for evaluation" '
                        'is superseded; no evaluation-era identity keys are used.',
    'A5_evidence_store': 'Witness evidence is stored as one parquet row per selected (firm-week, variant, hop, '
                         'partner) with the v1 fields plus n_articles, instead of JSONL.',
    'A6_main_E': 'E_<m> = (E1_<m> + E2_<m> + E3_<m>)/3 is stored next to Z_<m> for coverage reporting.',
    'A7_vector_gates': 'Gates are computed in vectorized form; exact parity with selectors.gate is tested.',
    'A8_refit_penalty': 'The selected penalty is reused unchanged (sum-of-squares objective) for the through-2022 '
                        'refit, as in v1.',
    'A9_inactive_partial': 'evidence/year_2015.jsonl and evidence/year_2016.jsonl from the interrupted v1 builds '
                           'are retained untouched and are not used.',
    'A10_tie_break': 'K and fallback are chosen per selector key by minimum 2021-2022 MSE; ties keep the first '
                     'grid entry (K order 0,1,2,3,5,all; fallback none before all). For local selectors the '
                     'fallback cannot change predictions because rows without news have Z=0.',
}

RULES = dict(RULES_V1)
RULES.update({
    'news': 'Reuters article-firm tags; frozen development one-to-one RIC/PERMNO mapping (keys <=2022) for all '
            'years 2015-2024; mapping first month must precede origin month',
    'ridge': 'No intercept; Z scaled by training (2015-2020) root mean square, no centering; minimize sum of '
             'squared residuals + lambda*||scaled beta||^2; all eligible coordinates q=1 while fitting; lambda by '
             '2021-2022 MSE over 1e-2..1e8; refit through 2022 with the same lambda and scale',
    'selection': 'K in 0,1,2,3,5,all and fallback none/all selected per local selector key on 2021-2022 only, '
                 'using the 2015-2020 head and TD1 selection MLP; provider slots stay empty and unselected',
    'ablations': 'Main and exact H1,H2,H3 with the frozen main coefficients (no refit) plus per-hop refit heads '
                 '1r,2r,3r; identical firm-week rows, selectors and K for every arm',
    'evidence': 'Parquet witness rows: nodes, intermediate nodes, relationship IDs, source rows, reporting '
                'roles, article IDs, availability maxima; intermediate_product_evidence=unavailable_in_sources; '
                'physical_channel_verified=false',
})

FEATURE_INT = ['graph_partners', 'news_partners', 'selected_partners', 'unique_articles']

def feature_columns(names=NAMES):
    cols = ['firm_id', 'cutoff', 'origin_session', 'variant', 'mapped_focal', 'has_news']
    for n in names:
        cols += ['C_' + n, 'available_' + n, 'missing_' + n, 'Z_' + n, 'E_' + n]
    for h in HOPS:
        cols += [f'Z{h}_{n}' for n in names] + [f'E{h}_{n}' for n in names]
        cols += [f'H{h}_{c}' for c in FEATURE_INT]
    return cols

FORBIDDEN_INPUT_TOKENS = ('ret', 'label', 'future', 'target', 'loss')

def forbidden_columns(columns):
    """Return names that look like outcomes; used to prove no label/return enters features."""
    bad = []
    for c in columns:
        low = c.lower()
        if low in ('y', 'label_end', 'label_observed') or any(t in low for t in FORBIDDEN_INPUT_TOKENS):
            bad.append(c)
    return bad
