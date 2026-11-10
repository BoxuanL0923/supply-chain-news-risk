"""Frozen TD1 conventions; all periods are exploratory."""
VERSION = 'TD1-firm-only-64x32-v1'
FEATURES = ['size', 'cash', 'cash_flow', 'leverage', 'inventory',
            'rv1', 'rv5', 'rv22', 'ret1', 'ret5', 'ret22', 'own_downside5',
            'post_earnings', 'earnings_covered', 'earnings_scheduled',
            'market_log_rv5', 'market_log_rv22', 'industry_log_rv5']
OLD = ['rv1','rv5','rv22','own_downside5','size',
       'market_log_rv5','market_log_rv22','industry_log_rv5']
GRID = [(0.001,0.0001),(0.001,0.01),(0.0003,0.0001),(0.0003,0.01)]
MAX_EPOCHS = 80
PATIENCE = 10
MIN_DELTA = 1e-5
SEED = 41037
CLOCK_RULES = {
    'cutoff': 'Historical panel origin session at 23:59:59 America/New_York, stored UTC.',
    'daily': 'CRSP session data assumed available at 20:00 America/New_York that day; strict availability < cutoff.',
    'quarter': 'Valid rdq >= datadate, else datadate+120 calendar days; availability at 23:59:59 New York; same cutoff date excluded; max age 550 days.',
    'cash_flow': 'JKP ocf_at at month end, plus SIX calendar months, 23:59:59 New York; conservative proxy clock, no vintage/revision guarantee. Max age 550 days. common=1 and primary_sec=1.',
    'earnings': 'Realized rdq available at 23:59:59 New York; use only announcements strictly before cutoff. Scheduled flag missing: no publication-stamped forward calendar.',
    'identity': 'Static gvkey-PERMNO link retained only when globally one-to-one. Ambiguous/unmatched fundamentals missing; this is not a dated CCM identity guarantee.',
    'universe': 'Historical panel keys only; prior-year size and graph-linked selection inherited. No graph values enter inputs. Universe remains conditional on legacy graph selection.',
    'revision': 'Local Compustat/JKP files are latest extracts, not historical vintages; these are availability-lag reconstructions, not certified revision-free PIT data.',
    'label': 'Exact log(mean(next five squared CRSP daily total returns)); all five finite, positive mean; no floor, annualization, or abnormal returns. Origin excluded.',
    'splits': 'Fit cutoffs 2015-2020 and label end <=2020-12-31; select 2021-2022 and label end <=2022-12-31; refit all eligible <=2022 with end <=2022-12-31. Preprocessing remains fitted on original fitting rows.',
    'eligible': 'Origin observed, full 22-session own history finite, positive size. Missing fundamentals retained with missing indicators. Missing future labels do not change feature eligibility.',
    'old': 'CC1 log-target own HAR: eight inputs, median imputation, scaling, intercept, numerical ridge 1e-6. Sources rebuilt on same rows; size uses original panel log(ME*1e6). Cached future earnings and GARCH excluded as in CC1 primary.'
}
