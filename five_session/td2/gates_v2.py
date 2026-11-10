"""Vectorized orderings and gates; exact parity with selectors.ordering/selectors.gate is tested.

Pure functions: no readers, returns, labels or provider calls.
"""
import hashlib
import numpy as np
import pandas as pd

def beta_order(names, coef_std):
    """A2: order by |standardized training coefficient|, ties by name."""
    coef = dict(zip(names, [float(v) for v in coef_std]))
    return sorted(names, key=lambda n: (-abs(coef[n]), n))

def row_key(firm_id, cutoff):
    # Same key string as v1 pipeline.predictions: f'{r.firm_id}|{r.cutoff}' with a pandas Timestamp.
    return f'{firm_id}|{pd.Timestamp(cutoff)}'

def rank_matrix(frame, names, kind, seed=0, order=None):
    """Rank (0 = first) of every menu item in each row's ordering; shape (rows, items)."""
    n, m = len(frame), len(names)
    if kind == 'training_beta':
        if order is None or sorted(order) != sorted(names) or len(set(order)) != m:
            raise ValueError('Complete global order required')
        pos = {name: i for i, name in enumerate(order)}
        return np.tile(np.array([pos[name] for name in names], dtype=np.int16), (n, 1))
    if kind != 'random':
        raise ValueError('Unknown local selector')
    ranks = np.empty((n, m), dtype=np.int16)
    for i, (firm, cutoff) in enumerate(zip(frame.firm_id.to_numpy(), frame.cutoff.to_numpy())):
        key = row_key(firm, cutoff)
        digests = [hashlib.sha256(f'{seed}|{key}|{name}'.encode()).hexdigest() for name in names]
        # v1 sorts names by digest; ties are impossible in practice but break by menu position.
        order = sorted(range(m), key=lambda j: (digests[j], j))
        for r, j in enumerate(order):
            ranks[i, j] = r
    return ranks

def gate_matrix(ranks, available, k):
    """q[i,m]=1 if item m is available and among the first K available items of row i's order."""
    ranks = np.asarray(ranks); available = np.asarray(available, dtype=bool)
    if ranks.shape != available.shape:
        raise ValueError('Shape mismatch')
    if k == 'all':
        return available.astype(float)
    k = int(k)
    if k < 0:
        raise ValueError('Negative K')
    # Number of available items that precede item m in the row order.
    before = (available[:, None, :] & (ranks[:, None, :] < ranks[:, :, None])).sum(axis=2)
    return (available & (before < k)).astype(float)

def gates(ranks, available, has_news, k, fallback):
    """Rows without news use the fallback budget (0 or all); their Z are zero for local selectors."""
    if fallback not in ('none', 'all'):
        raise ValueError('Fallback')
    q = gate_matrix(ranks, available, k)
    fb = gate_matrix(ranks, available, 'all' if fallback == 'all' else 0)
    has_news = np.asarray(has_news, dtype=bool)
    q[~has_news] = fb[~has_news]
    return q

def predict(b, z, q, beta):
    b = np.asarray(b, float); z = np.asarray(z, float)
    return b + np.sum(z * np.asarray(q, float) * np.asarray(beta, float)[None, :], axis=1)
