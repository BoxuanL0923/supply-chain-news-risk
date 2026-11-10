"""Selector diagnostics (TODO section 4, second item): menu-order stability from a fixed re-ask subset, and a blinded
human-labeling packet with a scorer for NDCG, top-one accuracy and Jev calibration.

No label is created by any LLM. build_packet() writes 150 masked 2024 selection-period contexts (the frozen selector
input texts: no outcome, no provider output, no record key), a plain CSV template and instructions; score_labels()
computes the diagnostics once a human has filled the template. Held-out volatility loss stays primary.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

import td4_spec as T
import td4_env as E

PACKET = T.OUT / 'labeling'
REL_LEVELS = (0, 1, 2, 3)
INSTRUCTIONS = """TD4 blinded labeling packet (2024 selection-period contexts)

Each context is the exact masked input the three selectors received: a five-item risk menu of firm characteristics,
the focal firm's characteristics, its supply-chain graph state and masked partner news. No outcome, price, return or
selector output is shown, and none may be looked up.

For every packet_id, fill one row of LABEL_TEMPLATE_v1.csv:
- rel_<item> for each of size, cash, cash_flow, leverage, inventory: how relevant the characteristic is to the focal
  firm's stock return volatility over the next five trading sessions, given only the context:
  0 = not relevant, 1 = weakly, 2 = moderately, 3 = strongly relevant.
- top_choice: the single most relevant item, or none if no item is at least moderately relevant.
- labeler_id and labeled_utc (date -u). Notes are optional.
Do not try to identify the firms. Label every context before any 2025 outcome of this experiment is read.
"""


def _mods():
    E.paths()
    import td3_runtime as R
    return R


def packet_id(key):
    return 'L' + hashlib.sha256(f'TD4-label-v1|{key}'.encode()).hexdigest()[:10]


def build_packet(n=T.LABEL_PACKET_N, year=T.SELECTION_YEAR):
    import td4_inputs as I
    R = _mods()
    recs = I.callable_inputs(year)
    keys = sorted(recs, key=lambda k: hashlib.sha256(f'TD4-label-select-v1|{year}|{k}'.encode()).hexdigest())[:n]
    w = R.JsonlWriter(PACKET / 'LABEL_PACKET_v1.jsonl', R.Redactor())
    for k in keys:
        w.write({'packet_id': packet_id(k), 'context': recs[k]['input_text']})
    w.close()
    R.new_json(PACKET / 'PACKET_KEYS_v1.json', {'year': year, 'map': {packet_id(k): k for k in keys},
                                                'note': 'private; used only by the scorer'})
    buf = io.StringIO()
    cols = ['packet_id'] + [f'rel_{n_}' for n_ in T.NAMES] + ['top_choice', 'labeler_id', 'labeled_utc', 'notes']
    wr = csv.writer(buf, lineterminator='\n')
    wr.writerow(cols)
    for k in keys:
        wr.writerow([packet_id(k)] + [''] * (len(cols) - 1))
    R.new_text(PACKET / 'LABEL_TEMPLATE_v1.csv', buf.getvalue())
    R.new_text(PACKET / 'LABELING_INSTRUCTIONS_v1.txt', INSTRUCTIONS)
    return {'contexts': len(keys), 'packet': str(PACKET / 'LABEL_PACKET_v1.jsonl'),
            'template': str(PACKET / 'LABEL_TEMPLATE_v1.csv'), 'sha256': R.sha(PACKET / 'LABEL_PACKET_v1.jsonl')}


# ---------------------------------------------------------------------------------------------------------- metrics
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at(order, rel, k):
    got = dcg([rel[n_] for n_ in order[:k]])
    ideal = dcg(sorted(rel.values(), reverse=True)[:k])
    return None if ideal == 0 else got / ideal


def kendall_tau(a, b):
    pos = {n_: i for i, n_ in enumerate(b)}
    x = [pos[n_] for n_ in a]
    conc = disc = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            conc += x[i] < x[j]
            disc += x[i] > x[j]
    return (conc - disc) / (conc + disc) if conc + disc else None


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if not len(conf):
        return None, []
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= 1))
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.sum() / len(conf) * gap
            table.append({'bin': [float(edges[i]), float(edges[i + 1])], 'n': int(m.sum()),
                          'mean_conf': float(conf[m].mean()), 'accuracy': float(correct[m].mean())})
    return float(total), table


def noul_top(scores):
    return max(T.NAMES, key=lambda n_: (scores[n_], -T.NAMES.index(n_)))


def stability(year, collected):
    """Main versus re-ask outputs of the fixed re-ask subset."""
    main = {(r['key'], r['selector']): r for r in collected if r['pass'] == 'main'}
    re_ = {(r['key'], r['selector']): r for r in collected if r['pass'] == 'reask1'}
    out = {}
    pairs = [(k, main.get(k), r) for k, r in re_.items()]
    ds = [(a, b) for (k, a, b) in pairs if k[1] == 'deepseek_order' and a and a['ordering'] and b['ordering']]
    out['deepseek_order'] = {'pairs_valid': len(ds),
                             'identical_ordering': float(np.mean([a['ordering'] == b['ordering'] for a, b in ds])) if ds else None,
                             'same_top1': float(np.mean([a['ordering'][0] == b['ordering'][0] for a, b in ds])) if ds else None,
                             'mean_kendall_tau': float(np.mean([kendall_tau(a['ordering'], b['ordering']) for a, b in ds])) if ds else None}
    ch = [(a, b) for (k, a, b) in pairs if k[1] == 'jev_choice' and a and a['probabilities'] and b['probabilities']]
    out['jev_choice'] = {'pairs_valid': len(ch),
                         'same_choice': float(np.mean([a['choice'] == b['choice'] for a, b in ch])) if ch else None,
                         'mean_total_variation': float(np.mean([0.5 * sum(abs(a['probabilities'][o] - b['probabilities'][o])
                                                                         for o in a['probabilities']) for a, b in ch])) if ch else None}
    nl = [(a, b) for (k, a, b) in pairs if k[1] == 'jev_noul' and a and a['yes_scores'] and b['yes_scores']]
    out['jev_noul'] = {'pairs_valid': len(nl),
                       'same_top1': float(np.mean([noul_top(a['yes_scores']) == noul_top(b['yes_scores']) for a, b in nl])) if nl else None,
                       'mean_abs_yes_change': float(np.mean([np.mean([abs(a['yes_scores'][n_] - b['yes_scores'][n_])
                                                                       for n_ in T.NAMES]) for a, b in nl])) if nl else None}
    return out


def read_labels(path):
    rows = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if not r.get('packet_id'):
                continue
            rel = {}
            for n_ in T.NAMES:
                v = (r.get(f'rel_{n_}') or '').strip()
                if v == '':
                    rel = None
                    break
                if int(v) not in REL_LEVELS:
                    raise ValueError(f'relevance out of range for {r["packet_id"]}')
                rel[n_] = int(v)
            top = (r.get('top_choice') or '').strip()
            if top and top not in T.NAMES + ['none']:
                raise ValueError(f'top_choice invalid for {r["packet_id"]}')
            if rel is not None:
                rows[r['packet_id']] = {'rel': rel, 'top': top or None}
    return rows


def score_core(labels, outputs):
    """labels: {pid: {'rel', 'top'}}; outputs: {pid: {selector: collected row}}. Pure."""
    res = {'labeled_contexts': len(labels)}
    for sel in ('deepseek_order', 'jev_noul'):
        vals = {k: [] for k in (1, 2, 3, 5)}
        for pid, lab in labels.items():
            o = outputs.get(pid, {}).get(sel)
            order = (o or {}).get('ordering') if sel == 'deepseek_order' else (
                sorted(T.NAMES, key=lambda n_: (-o['yes_scores'][n_], T.NAMES.index(n_))) if o and o.get('yes_scores') else None)
            if order:
                for k in vals:
                    v = ndcg_at(order, lab['rel'], k)
                    if v is not None:
                        vals[k].append(v)
        res[f'ndcg_{sel}'] = {f'@{k}': (float(np.mean(v)) if v else None) for k, v in vals.items()}
        res[f'ndcg_{sel}_n'] = {f'@{k}': len(v) for k, v in vals.items()}
    acc = {'deepseek_order': [], 'jev_choice': [], 'jev_noul': []}
    brier, conf, corr, nb_conf, nb_corr, n_brier = [], [], [], [], [], []
    for pid, lab in labels.items():
        o = outputs.get(pid, {})
        if lab['top']:
            if o.get('deepseek_order', {}).get('ordering') and lab['top'] != 'none':
                acc['deepseek_order'].append(o['deepseek_order']['ordering'][0] == lab['top'])
            if o.get('jev_choice', {}).get('choice'):
                acc['jev_choice'].append(o['jev_choice']['choice'] == lab['top'])
                p = o['jev_choice']['probabilities']
                brier.append(sum((p[opt] - (1.0 if opt == lab['top'] else 0.0)) ** 2 for opt in p))
                c = o['jev_choice']['choice']
                conf.append(p[c])
                corr.append(c == lab['top'])
            if o.get('jev_noul', {}).get('yes_scores') and lab['top'] != 'none':
                acc['jev_noul'].append(noul_top(o['jev_noul']['yes_scores']) == lab['top'])
        if o.get('jev_noul', {}).get('yes_scores'):
            for n_ in T.NAMES:
                s = o['jev_noul']['yes_scores'][n_]
                y_ = 1.0 if lab['rel'][n_] >= 2 else 0.0
                n_brier.append((s - y_) ** 2)
                nb_conf.append(s)
                nb_corr.append(y_)
    res['top_one_accuracy'] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
    res['top_one_n'] = {k: len(v) for k, v in acc.items()}
    e1, t1 = ece(conf, corr)
    e2, t2 = ece(nb_conf, nb_corr)
    res['jev_choice_calibration'] = {'multiclass_brier': float(np.mean(brier)) if brier else None, 'ece_top_choice': e1,
                                     'reliability': t1, 'n': len(brier)}
    res['jev_noul_calibration'] = {'brier_vs_rel_ge_2': float(np.mean(n_brier)) if n_brier else None, 'ece': e2,
                                   'reliability': t2, 'n_item_pairs': len(n_brier)}
    return res


def score_labels(path):
    import td4_runner as RU
    R = _mods()
    if not path:
        raise ValueError('--labels path to the filled template is required')
    labels = read_labels(path)
    pmap = json.loads((PACKET / 'PACKET_KEYS_v1.json').read_text(encoding='utf-8'))['map']
    collected = RU.collect(T.SELECTION_YEAR)
    by = {}
    for r in collected:
        if r['pass'] == 'main':
            by.setdefault(r['key'], {})[r['selector']] = r
    outputs = {pid: by.get(k, {}) for pid, k in pmap.items()}
    res = score_core(labels, outputs)
    res['stability'] = {str(y): stability(y, RU.collect(y)) for y in T.PERIODS
                        if RU.period_dir(y, 'real').exists()}
    res['labels_sha256'] = R.sha(path)
    R.new_json(PACKET / f'LABEL_SCORES_{res["labels_sha256"][:8]}.json', res)
    return res
