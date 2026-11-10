"""Five-session labels for v2, behind v2 admission tokens.

Selection (2021-2022): the frozen TD1 development labels (v1 td4_labels.development: label end <= 2022-12-31; TD2
used the same labels for its 2021-2022 selection). Test (2023-2025): the TD1 rule recomputed from the CRSP snapshots
with v1 td4_labels.label (allowed end = 2025-12-31; a window past it is never opened), reading the year and the first
15 days of the next year only when that next year is <= 2025 (2023 windows may end in January 2024, as in TD1's and
TD2's 2023 evaluations).
"""
import hashlib

import numpy as np
import pandas as pd

import td4_spec as T
import td4_env as E
import td4v2_spec as S
import td4v2_gate as G


def selection_labels(years, token):
    import td4_labels as L
    if not G.valid(token, ('select',)):
        raise PermissionError('Selection label admission required')
    if any(y > 2022 for y in years):
        raise PermissionError('Selection labels are 2021-2022 development labels only')
    return pd.concat([L.development(y) for y in years], ignore_index=True)


def test_labels(year, keys, token):
    import td4_labels as L
    if not G.valid(token, ('evaluate',)):
        raise PermissionError('Test label admission required')
    if year not in S.TEST_YEARS:
        raise PermissionError('Not a test year: ' + str(year))
    bound = S.LABEL_END_BOUND
    if bound > T.HARD_SEAL:
        raise PermissionError('Label seal')
    E.guard()
    parts = [L._crsp(year, f'{year}-01-01', min(f'{year}-12-31', bound))]
    nxt_hi = min(f'{year + 1}-01-15', bound)
    if year + 1 <= 2025 and nxt_hi >= f'{year + 1}-01-01':
        parts.append(L._crsp(year + 1, f'{year + 1}-01-01', nxt_hi))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.share_type.isin([0, 2])].copy()
    d['date'] = pd.to_datetime(d.date)
    if d.duplicated(['permno', 'date']).any():
        raise ValueError('Duplicate return session')
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    wide = d.pivot(index='date', columns='permno', values='ret').reindex(dates)
    mapper = {hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]: p for p in wide.columns}
    rows = []
    for firm, g in keys.groupby('firm_id', sort=True):
        p = mapper.get(firm)
        r = wide[p].to_numpy(float) if p is not None else None
        for c, o in zip(g.cutoff, g.origin_session):
            t = dates.get_indexer([pd.Timestamp(o)])[0] if r is not None else -1
            if t < 0:
                rows.append((firm, c, np.nan, pd.NaT, False))
                continue
            y, end, ok = L.label(dates, r, t, bound)
            rows.append((firm, c, y, end, ok))
    return pd.DataFrame(rows, columns=['firm_id', 'cutoff', 'y', 'label_end', 'label_observed'])
