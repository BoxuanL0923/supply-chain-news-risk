"""Five-session labels behind admission tokens (TD1 rule; seal 2025-12-31 for this experiment only).

label() is TD1 v2 core.label with its clamp min(allowed_end, '2024-12-31') replaced by min(allowed_end, HARD_SEAL)
(TD4: amendment item 2). Development labels (label end <= 2022-12-31) come from the frozen TD1 manifest; every other
label needs a token from td4_gate (refit: <= 2024-12-31; select: 2024; evaluate: 2025). A window that would end
after its allowed end is never opened (its return fields are not touched).
"""
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4_gate as G

PANEL_LABEL_FLOOR = -20.0      # hist panel assemble: y > -20 (halted/stale stock), applied to every arm alike


def label(dates, returns, pos, allowed_end):
    """TD1 core.label (TD4: clamp at the 2025 seal)."""
    if pos + 5 >= len(dates):
        return np.nan, pd.NaT, False
    end = pd.Timestamp(dates[pos + 5])
    if end > pd.Timestamp(min(allowed_end, T.HARD_SEAL)):
        return np.nan, end, False
    w = np.asarray(returns[pos + 1:pos + 6], float)
    valid = len(w) == 5 and np.isfinite(w).all() and np.mean(w * w) > 0
    return (float(np.log(np.mean(w * w))) if valid else np.nan), end, bool(valid)


def _crsp(year, lo, hi):
    import td4_sources as S
    import td4_manifest as M
    return S.parquet_dated(M.crsp_path(year), ['date', 'permno', 'ret', 'share_type'], lo=lo, hi=hi).to_pandas()


def compute(year, keys, allowed_end, token):
    """Labels for (firm_id, cutoff, origin_session) rows of one origin year. Returns a frame with y, label_end,
    label_observed. Requires a token whose step admits this year."""
    allowed = {'refit': tuple(range(2022, T.TEST_YEAR)), 'select': (T.SELECTION_YEAR,),
               'evaluate': (T.TEST_YEAR,)}
    if not (isinstance(token, G.Token) and G.valid(token, tuple(allowed)) and year in allowed[token.step]):
        raise PermissionError('Label admission required for ' + str(year))
    if allowed_end > T.HARD_SEAL:
        raise PermissionError('Label window seal')
    E.guard()
    parts = [_crsp(year, f'{year}-01-01', f'{year}-12-31')]
    nxt_hi = min(f'{year + 1}-01-15', allowed_end)
    if year + 1 <= 2025 and nxt_hi >= f'{year + 1}-01-01':
        parts.append(_crsp(year + 1, f'{year + 1}-01-01', nxt_hi))
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
            if r is None:
                rows.append((firm, c, np.nan, pd.NaT, False))
                continue
            t = dates.get_indexer([pd.Timestamp(o)])[0]
            if t < 0:
                rows.append((firm, c, np.nan, pd.NaT, False))
                continue
            y, end, ok = label(dates, r, t, allowed_end)
            rows.append((firm, c, y, end, ok))
    out = pd.DataFrame(rows, columns=['firm_id', 'cutoff', 'y', 'label_end', 'label_observed'])
    if (pd.to_datetime(out.label_end.dropna()) > pd.Timestamp(allowed_end)).any() and \
            out.loc[pd.to_datetime(out.label_end) > pd.Timestamp(allowed_end), 'label_observed'].any():
        raise ValueError('Label date seal')
    return out


def development(year):
    """Frozen TD1 development labels (years 2015-2022, label end <= 2022-12-31); no token needed."""
    if not 2015 <= year <= 2022:
        raise PermissionError('Development labels are 2015-2022 only')
    f = pq.read_table(T.TD1 / f'firm_week_manifest/year_{year}.parquet',
                      columns=['firm_id', 'cutoff', 'origin_session', 'y', 'label_end', 'label_observed']).to_pandas()
    if (pd.to_datetime(f.label_end.dropna()) > pd.Timestamp(T.DEV_SEAL)).any() and f.label_observed.any():
        f.loc[pd.to_datetime(f.label_end) > pd.Timestamp(T.DEV_SEAL), ['y', 'label_observed']] = [np.nan, False]
    return f
