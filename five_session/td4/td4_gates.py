"""Provider gates q(a, i, t, m) and shared-head predictions (pure functions; no reader, label or provider call).

Every arm uses the same rows, the same candidate values Z and the same beta; only q differs:
- DeepSeek order, top-K: the first K available (non-missing) items of the validated ordering (TD2 gates_v2.gate_matrix
  on the ordering's ranks). An invalid output (invalid, duplicate or missing names), no output, or no call is never
  backfilled from a partial ordering: the row takes the arm's fallback.
- Jev Choice: the chosen item if it is available; 'none' (or an unavailable choice) activates nothing.
- Jev Noul, top-K: the K available items with the highest yes scores (ties by menu order); threshold tau: every
  available item with yes score >= tau.
- all-on: q = available (K = all, no selector). References: TD2 training-|beta| and random orders (gates_v2).
Fallback ('none' = all zero, 'all' = every available item) applies to rows with news and no valid output; rows without
news have Z = 0, so their prediction is b whatever q is.
"""
import numpy as np

import td4_spec as T
import td4_env as E

STATUS_VALID = 'valid'


def _gv():
    E.paths()
    import gates_v2
    return gates_v2


def ranks_from_orders(orders, names=T.NAMES):
    """orders: list of complete orderings (list of names) or None. Returns (ranks int16 matrix, valid mask)."""
    n, m = len(orders), len(names)
    pos = {nm: j for j, nm in enumerate(names)}
    ranks = np.tile(np.arange(m, dtype=np.int16), (n, 1))
    valid = np.zeros(n, dtype=bool)
    for i, o in enumerate(orders):
        if o is not None and sorted(o) == sorted(names) and len(set(o)) == m:
            for r, nm in enumerate(o):
                ranks[i, pos[nm]] = r
            valid[i] = True
    return ranks, valid


def ranks_from_scores(scores, names=T.NAMES):
    """scores: list of {name: yes score} or None. Rank by score descending, ties by menu order."""
    n, m = len(scores), len(names)
    ranks = np.tile(np.arange(m, dtype=np.int16), (n, 1))
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(scores):
        if isinstance(s, dict) and set(s) == set(names) and all(np.isfinite(float(s[k])) for k in names):
            order = sorted(range(m), key=lambda j: (-float(s[names[j]]), j))
            for r, j in enumerate(order):
                ranks[i, j] = r
            valid[i] = True
    return ranks, valid


def fallback_matrix(available, fallback):
    if fallback not in T.FALLBACKS:
        raise ValueError('fallback')
    a = np.asarray(available, dtype=bool)
    return a.astype(float) if fallback == 'all' else np.zeros(a.shape)


def apply_fallback(q, valid, available, fallback):
    q = np.array(q, dtype=float)
    fb = fallback_matrix(available, fallback)
    q[~valid] = fb[~valid]
    return q


def deepseek_gate(orders, available, k, fallback):
    ranks, valid = ranks_from_orders(orders)
    q = _gv().gate_matrix(ranks, available, k)
    return apply_fallback(q, valid, available, fallback), valid


def choice_gate(choices, valid, available, fallback, names=T.NAMES):
    """choices: chosen option per row (a menu name or 'none'); valid: parsed-valid mask."""
    a = np.asarray(available, dtype=bool)
    q = np.zeros(a.shape)
    pos = {nm: j for j, nm in enumerate(names)}
    for i, c in enumerate(choices):
        if valid[i] and c in pos and a[i, pos[c]]:
            q[i, pos[c]] = 1.0
    return apply_fallback(q, np.asarray(valid, bool), available, fallback), np.asarray(valid, bool)


def noul_topk_gate(scores, available, k, fallback):
    ranks, valid = ranks_from_scores(scores)
    q = _gv().gate_matrix(ranks, available, k)
    return apply_fallback(q, valid, available, fallback), valid


def noul_threshold_gate(scores, available, tau, fallback, names=T.NAMES):
    a = np.asarray(available, dtype=bool)
    _, valid = ranks_from_scores(scores)
    q = np.zeros(a.shape)
    for i, s in enumerate(scores):
        if valid[i]:
            q[i] = [1.0 if a[i, j] and float(s[nm]) >= tau else 0.0 for j, nm in enumerate(names)]
    return apply_fallback(q, valid, available, fallback), valid


def all_on_gate(available):
    return np.asarray(available, dtype=bool).astype(float)


def predict(b, z, q, beta):
    """TD2 gates_v2.predict: b + sum_m z * q * beta."""
    return _gv().predict(b, z, q, beta)


def configs(arm):
    """Selection grid per provider arm: (rule, parameter) pairs in tie-break order, crossed with FALLBACKS."""
    if arm == 'deepseek_order':
        base = [('top_k', k) for k in T.K_GRID]
    elif arm == 'jev_noul':
        base = [('top_k', k) for k in T.K_GRID] + [('threshold', t) for t in T.NOUL_THRESHOLDS]
    elif arm == 'jev_choice':
        base = [('top_one_or_none', None)]
    else:
        raise ValueError(arm)
    return [(r, p, fb) for (r, p) in base for fb in T.FALLBACKS]


def gate_for(arm, rule, param, fallback, outputs, available):
    """outputs: dict with lists 'ordering', 'choice', 'yes_scores', 'valid' aligned with the rows."""
    if arm == 'deepseek_order':
        return deepseek_gate(outputs['ordering'], available, param, fallback)
    if arm == 'jev_choice':
        return choice_gate(outputs['choice'], outputs['valid'], available, fallback)
    if arm == 'jev_noul':
        if rule == 'top_k':
            return noul_topk_gate(outputs['yes_scores'], available, param, fallback)
        return noul_threshold_gate(outputs['yes_scores'], available, param, fallback)
    raise ValueError(arm)
