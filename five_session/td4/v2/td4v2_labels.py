"""Diagnostic labels by a Codex agent (researcher's decision, amendment 2): packet, strict converter, scorer.

LLM_LABEL_NOTE: these are LLM labels from a third provider (OpenAI gpt-6-astra via Codex), not human ground truth;
every output written here says so. build_packet() takes about 150 contexts of the 2021-2022 selection subsample,
exactly as sent to the providers (masked frozen input text; no outcome, no selector output, no record key), and
splits them into batch files of 10, each with plain instructions and a strict JSON output schema. Claude runs Codex on
the batches outside this package. convert() validates the Codex outputs (every context and every menu item present,
integers 0-3, top choice a menu item or "none", no extra keys) and writes the v1 label template; score() runs the v1
scorer (td4_diagnostics.score_core) against the 2021-2022 selector outputs.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4v2_spec as S

INSTRUCTIONS = """TD4 diagnostic labeling batch (2021-2022 contexts). You are labeling for a research diagnostic.

Each context is a masked text: a five-item risk menu of firm characteristics (size, cash, cash_flow, leverage,
inventory), the focal firm's characteristics, its supply-chain graph state and masked partner news. Use only the text
of the context. Do not try to identify any firm, date or outcome, and do not use outside information.

For every context_id, give:
- relevance: for each of size, cash, cash_flow, leverage, inventory, an integer 0-3: how relevant that characteristic
  is to the focal firm's stock return volatility over the next five trading sessions, given only the context
  (0 = not relevant, 1 = weakly, 2 = moderately, 3 = strongly relevant);
- top_choice: the single most relevant item, or "none" if no item is at least moderately relevant.

Return only one JSON object that matches output_schema exactly: {"labels": {"<context_id>": {"relevance": {"size": n,
"cash": n, "cash_flow": n, "leverage": n, "inventory": n}, "top_choice": "<item or none>"}, ...}} with every
context_id of this batch and no other keys.
"""


def output_schema(context_ids):
    item = {'type': 'integer', 'minimum': 0, 'maximum': 3}
    one = {'type': 'object', 'additionalProperties': False, 'required': ['relevance', 'top_choice'],
           'properties': {'relevance': {'type': 'object', 'additionalProperties': False, 'required': T.NAMES,
                                        'properties': {n: item for n in T.NAMES}},
                          'top_choice': {'type': 'string', 'enum': T.NAMES + ['none']}}}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'additionalProperties': False,
            'required': ['labels'],
            'properties': {'labels': {'type': 'object', 'additionalProperties': False, 'required': list(context_ids),
                                      'properties': {c: one for c in context_ids}}}}


def context_id(key):
    return 'C' + hashlib.sha256(f'TD4-label-v2|{key}'.encode()).hexdigest()[:10]


def build_packet(n=None, batch=None):
    import td3_runtime as R
    import td4v2_inputs as IV
    n, batch = n or S.LABEL_PACKET_N, batch or S.LABEL_BATCH
    recs = IV.callable_inputs(S.SELECT)
    keys = sorted(recs, key=lambda k: hashlib.sha256(f'TD4-label-select-v2|{k}'.encode()).hexdigest())[:n]
    batches = []
    for i in range(0, len(keys), batch):
        part = keys[i:i + batch]
        ids = [context_id(k) for k in part]
        b = {'batch_id': f'B{i // batch + 1:02d}', 'labeler_note': S.LLM_LABEL_NOTE, 'instructions': INSTRUCTIONS,
             'output_schema': output_schema(ids),
             'contexts': [{'context_id': context_id(k), 'context': recs[k]['input_text']} for k in part]}
        p = R.new_json(S.LABELS / f'batch_{b["batch_id"]}.json', b, timed=False)
        batches.append({'batch_id': b['batch_id'], 'path': str(p), 'contexts': len(part), 'sha256': R.sha(p)})
    R.new_text(S.LABELS / 'INSTRUCTIONS_v2.txt', INSTRUCTIONS)
    R.new_json(S.LABELS / 'PACKET_KEYS_v2.json', {'map': {context_id(k): k for k in keys},
                                                  'note': 'private; used only by the converter and the scorer'})
    R.new_json(S.LABELS / 'PACKET_INDEX_v2.json', {'batches': batches, 'contexts': len(keys),
                                                   'expected_output_name': 'batch_<batch_id>_output.json',
                                                   'labeler': S.CODEX_LABELER, 'note': S.LLM_LABEL_NOTE})
    return {'contexts': len(keys), 'batches': len(batches), 'folder': str(S.LABELS)}


def packet_marker():
    """Marker for the coordinator: OUT/v2_PACKET_READY.json with every packet file's path and SHA256, the selection
    build it was drawn from (masker v2) and the label note."""
    import td3_runtime as R
    import td4v2_build2 as B2
    import td4v2_inputs as IV
    idx = S.LABELS / 'PACKET_INDEX_v2.json'
    if not idx.exists():
        raise PermissionError('packet not built')
    index = json.loads(idx.read_text(encoding='utf-8'))
    d = Path(B2.PERIOD_DIR[S.SELECT])
    files = {str(p): R.sha(p) for p in [S.LABELS / 'INSTRUCTIONS_v2.txt', idx] +
             [Path(b['path']) for b in index['batches']]}
    out = {'status': 'READY', 'folder': str(S.LABELS), 'batches': len(index['batches']),
           'contexts': index['contexts'], 'files_sha256': files,
           'expected_output_name': index['expected_output_name'],
           'converter': 'python -B run_td4v2.py convert-labels --labels <folder with batch_<id>_output.json>',
           'selection_build': str(d), 'selection_build_receipt_sha256': R.sha(d / 'MASK_V2_RECEIPT.json'),
           'population_check': str(IV.withheld_file(d)), 'population_check_sha256': R.sha(IV.withheld_file(d)),
           'masker': 'TD4 masker v2 (no v1-masker input)', 'labeler': S.CODEX_LABELER, 'note': S.LLM_LABEL_NOTE,
           'private': 'the batch files hold masked article text; they stay in the private root'}
    p = R.new_json(S.OUT / 'v2_PACKET_READY.json', out)
    return {'marker': str(p), 'sha256': R.sha(p), 'batches': out['batches'], 'contexts': out['contexts']}


def redacted_packet(batch=None):
    """Span-level redaction (2026-11-07): the labeling_v2 contexts whose text the redaction changed, written with their
    redacted text (the text now sent) to OUT/labeling_v2r/ in the labeling_v2 batch format (same context ids; batches
    of 10 in packet order). The researcher relabels these; Codex is not called here. Unchanged contexts keep their
    labeling_v2 labels."""
    import td3_runtime as R
    import td4_inputs as I
    import td4v2_inputs as IV
    batch = batch or S.LABEL_BATCH
    d = IV.period_dir(S.SELECT)
    pmap = json.loads((S.LABELS / 'PACKET_KEYS_v2.json').read_text(encoding='utf-8'))['map']
    index = json.loads((S.LABELS / 'PACKET_INDEX_v2.json').read_text(encoding='utf-8'))
    recs = I.load_inputs(None, IV.inputs_path(d))
    bad = IV.withheld(d)
    order, old = [], {}
    for b in index['batches']:
        for c in json.loads(Path(b['path']).read_text(encoding='utf-8'))['contexts']:
            order.append(c['context_id'])
            old[c['context_id']] = c['context']
    changed, same, rows = [], [], {}
    for cid in order:
        k = pmap[cid]
        new = recs[k]['input_text']
        rows[cid] = {'old_sha256': hashlib.sha256(old[cid].encode('utf-8')).hexdigest(),
                     'new_sha256': hashlib.sha256(new.encode('utf-8')).hexdigest(), 'withheld': k in bad}
        (changed if new != old[cid] else same).append(cid)
    batches = []
    for i in range(0, len(changed), batch):
        part = changed[i:i + batch]
        b = {'batch_id': f'R{i // batch + 1:02d}', 'labeler_note': S.LLM_LABEL_NOTE, 'instructions': INSTRUCTIONS,
             'output_schema': output_schema(part),
             'contexts': [{'context_id': cid, 'context': recs[pmap[cid]]['input_text']} for cid in part]}
        p = R.new_json(S.LABELS_R / f'batch_{b["batch_id"]}.json', b, timed=False)
        batches.append({'batch_id': b['batch_id'], 'path': str(p), 'contexts': len(part), 'sha256': R.sha(p)})
    R.new_text(S.LABELS_R / 'INSTRUCTIONS_v2r.txt', INSTRUCTIONS)
    R.new_json(S.LABELS_R / 'PACKET_KEYS_v2r.json', {'map': {cid: pmap[cid] for cid in changed},
                                                     'note': 'private; used only by the converter and the scorer'})
    out = {'source_packet': str(S.LABELS), 'redaction_file': str(IV.redaction_file(d)),
           'redaction_file_sha256': R.sha(IV.redaction_file(d)), 'contexts': len(order),
           'changed': len(changed), 'changed_context_ids': changed, 'unchanged': len(same),
           'withheld_context_ids': sorted(cid for cid in order if rows[cid]['withheld']),
           'withheld_changed': sorted(cid for cid in changed if rows[cid]['withheld']),
           'per_context': rows, 'batches': batches, 'expected_output_name': 'batch_<batch_id>_output.json',
           'labeler': 'researcher relabels the changed contexts (no Codex call by the builder)',
           'note': S.LLM_LABEL_NOTE}
    R.new_json(S.LABELS_R / 'PACKET_INDEX_v2r.json', out)
    return {k: v for k, v in out.items() if k != 'per_context'}


def redacted_packet_v22(batch=None):
    """v2.2 redaction (2026-11-07 fix round): the labeling_v2 contexts whose v2.2 sent text differs from the text that
    was already labelled (the labeling_v2r text for the contexts relabelled in v2r, else the labeling_v2 masker-v2
    text), written with their v2.2 text to OUT/labeling_v2r2/ in the same batch format (same context ids, same
    INSTRUCTIONS, batches of 10 in packet order, ids S01, S02, ...). Codex (OpenAI gpt-6-astra) relabels these outside
    this package; contexts whose text is unchanged keep their labels."""
    import td3_runtime as R
    import td4_inputs as I
    import td4v2_inputs as IV
    batch = batch or S.LABEL_BATCH
    d = IV.period_dir(S.SELECT)
    rf = json.loads(IV.redaction_file(d).read_text(encoding='utf-8'))
    if rf.get('rule_version') != 'v2.2':
        raise PermissionError('v2.2 redaction result not the binding one')
    pmap = json.loads((S.LABELS / 'PACKET_KEYS_v2.json').read_text(encoding='utf-8'))['map']
    index = json.loads((S.LABELS / 'PACKET_INDEX_v2.json').read_text(encoding='utf-8'))
    rindex = json.loads((S.LABELS_R / 'PACKET_INDEX_v2r.json').read_text(encoding='utf-8'))
    recs = I.load_inputs(None, IV.inputs_path(d))
    bad = IV.withheld(d)
    order, labelled, source = [], {}, {}
    for b in index['batches']:
        for c in json.loads(Path(b['path']).read_text(encoding='utf-8'))['contexts']:
            order.append(c['context_id'])
            labelled[c['context_id']] = c['context']
            source[c['context_id']] = 'labeling_v2/' + b['batch_id']
    for b in rindex['batches']:
        for c in json.loads(Path(b['path']).read_text(encoding='utf-8'))['contexts']:
            labelled[c['context_id']] = c['context']
            source[c['context_id']] = 'labeling_v2r/' + b['batch_id']
    changed, rows = [], {}
    for cid in order:
        new = recs[pmap[cid]]['input_text']
        rows[cid] = {'labelled_from': source[cid],
                     'labelled_sha256': hashlib.sha256(labelled[cid].encode('utf-8')).hexdigest(),
                     'v2_2_sha256': hashlib.sha256(new.encode('utf-8')).hexdigest(), 'withheld_v2_2': pmap[cid] in bad}
        if new != labelled[cid]:
            changed.append(cid)
    batches = []
    for i in range(0, len(changed), batch):
        part = changed[i:i + batch]
        b = {'batch_id': f'S{i // batch + 1:02d}', 'labeler_note': S.LLM_LABEL_NOTE, 'instructions': INSTRUCTIONS,
             'output_schema': output_schema(part),
             'contexts': [{'context_id': cid, 'context': recs[pmap[cid]]['input_text']} for cid in part]}
        p = R.new_json(S.LABELS_R2 / f'batch_{b["batch_id"]}.json', b, timed=False)
        batches.append({'batch_id': b['batch_id'], 'path': str(p), 'contexts': len(part), 'sha256': R.sha(p)})
    R.new_text(S.LABELS_R2 / 'INSTRUCTIONS_v2r2.txt', INSTRUCTIONS)
    R.new_json(S.LABELS_R2 / 'PACKET_KEYS_v2r2.json', {'map': {cid: pmap[cid] for cid in changed},
                                                       'note': 'private; used only by the converter and the scorer'})
    out = {'source_packets': [str(S.LABELS), str(S.LABELS_R)], 'redaction_file': str(IV.redaction_file(d)),
           'redaction_file_sha256': R.sha(IV.redaction_file(d)), 'rule_version': 'v2.2', 'contexts': len(order),
           'changed': len(changed), 'changed_context_ids': changed,
           'changed_from_v2': sorted(c for c in changed if source[c].startswith('labeling_v2/')),
           'changed_from_v2r': sorted(c for c in changed if source[c].startswith('labeling_v2r/')),
           'unchanged': len(order) - len(changed),
           'withheld_context_ids': sorted(cid for cid in order if rows[cid]['withheld_v2_2']),
           'withheld_changed': sorted(cid for cid in changed if rows[cid]['withheld_v2_2']),
           'per_context': rows, 'batches': batches, 'expected_output_name': 'batch_<batch_id>_output.json',
           'instructions_identical_to_v2': True, 'labeler': S.CODEX_LABELER + ' (run by the coordinator)',
           'note': S.LLM_LABEL_NOTE}
    R.new_json(S.LABELS_R2 / 'PACKET_INDEX_v2r2.json', out)
    return {k: v for k, v in out.items() if k != 'per_context'}


def packet_withheld():
    """Packet contexts whose input is withheld (population check or the redaction rule): excluded from the
    diagnostics. Returns {'withheld_context_ids': [...], 'kept': n}."""
    import td4v2_inputs as IV
    pmap = json.loads((S.LABELS / 'PACKET_KEYS_v2.json').read_text(encoding='utf-8'))['map']
    bad = IV.withheld(IV.period_dir(S.SELECT))
    out = sorted(cid for cid, k in pmap.items() if k in bad)
    return {'withheld_context_ids': out, 'withheld': len(out), 'kept': len(pmap) - len(out), 'contexts': len(pmap)}


def validate_batch(batch, output):
    """Problems (list of strings) of one Codex output against its batch; empty when valid."""
    ids = [c['context_id'] for c in batch['contexts']]
    probs = []
    if not isinstance(output, dict) or set(output) != {'labels'} or not isinstance(output.get('labels'), dict):
        return ['top level must be exactly {"labels": {...}}']
    lab = output['labels']
    if set(lab) != set(ids):
        missing, extra = sorted(set(ids) - set(lab)), sorted(set(lab) - set(ids))
        probs.append(f'context ids differ: missing {missing}, extra {extra}')
    for cid in ids:
        v = lab.get(cid)
        if not isinstance(v, dict) or set(v) != {'relevance', 'top_choice'}:
            probs.append(f'{cid}: keys must be relevance and top_choice')
            continue
        rel = v['relevance']
        if not isinstance(rel, dict) or set(rel) != set(T.NAMES):
            probs.append(f'{cid}: relevance must have exactly the five menu items')
        else:
            for nm in T.NAMES:
                x = rel[nm]
                if not (isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 3):
                    probs.append(f'{cid}: relevance {nm} not an integer 0-3')
        if v['top_choice'] not in T.NAMES + ['none']:
            probs.append(f'{cid}: top_choice invalid')
    return probs


def convert(folder, revised=False):
    """Codex outputs (batch_<id>_output.json in `folder`) -> LABEL_TEMPLATE_v2_FILLED_<tag>.csv (v1 template format).
    revised=True converts the relabels of the redacted contexts (labeling_v2r) into labeling_v2r/; revised='v2r2' the
    relabels of the v2.2 contexts (labeling_v2r2) into labeling_v2r2/."""
    import td3_runtime as R
    folder = Path(folder)
    root = S.LABELS_R2 if revised == 'v2r2' else S.LABELS_R if revised else S.LABELS
    idx_name = 'PACKET_INDEX_v2r2.json' if revised == 'v2r2' else 'PACKET_INDEX_v2r.json' if revised else \
        'PACKET_INDEX_v2.json'
    index = json.loads((root / idx_name).read_text(encoding='utf-8'))
    rows, report = [], {}
    for b in index['batches']:
        batch = json.loads(Path(b['path']).read_text(encoding='utf-8'))
        out_p = folder / f'batch_{b["batch_id"]}_output.json'
        if not out_p.exists():
            report[b['batch_id']] = ['output file missing']
            continue
        try:
            output = json.loads(out_p.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            report[b['batch_id']] = ['output is not JSON']
            continue
        probs = validate_batch(batch, output)
        report[b['batch_id']] = probs
        if probs:
            continue
        for c in batch['contexts']:
            v = output['labels'][c['context_id']]
            rows.append([c['context_id']] + [v['relevance'][n] for n in T.NAMES] +
                        [v['top_choice'], S.CODEX_LABELER, '', 'LLM label (OpenAI via Codex), not human ground truth'])
    bad = {k: v for k, v in report.items() if v}
    if bad:
        R.new_json(root / f'CONVERSION_REJECTED_{R.utc().replace(":", "")}.json',
                   {'problems': bad, 'note': S.LLM_LABEL_NOTE})
        raise ValueError('Codex outputs failed validation: ' + json.dumps(bad)[:500])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['packet_id'] + [f'rel_{n}' for n in T.NAMES] + ['top_choice', 'labeler_id', 'labeled_utc', 'notes'])
    w.writerows(rows)
    tag = R.utc().replace(':', '')
    suffix = 'r2' if revised == 'v2r2' else 'r' if revised else ''
    p = R.new_text(root / f'LABEL_TEMPLATE_v2{suffix}_FILLED_{tag}.csv', buf.getvalue())
    R.new_json(root / f'CONVERSION_RECEIPT_{tag}.json',
               {'contexts': len(rows), 'template': str(p), 'template_sha256': R.sha(p),
                'outputs_sha256': {b['batch_id']: R.sha(folder / f'batch_{b["batch_id"]}_output.json')
                                   for b in index['batches']}, 'note': S.LLM_LABEL_NOTE})
    return {'contexts': len(rows), 'template': str(p), 'note': S.LLM_LABEL_NOTE}


def score(path, path_r=None, path_r2=None):
    """Scores against the 2021-2022 selector outputs. The contexts the redaction changed (labeling_v2r) use their
    relabels (path_r, converted with revised=True); the contexts whose v2.2 text differs from the labelled text
    (labeling_v2r2, when that packet exists) use their v2.2 relabels (path_r2, converted with revised='v2r2'); a changed
    context without its relabel is excluded; withheld contexts are excluded."""
    import td3_runtime as R
    import td4_diagnostics as D
    import td4v2_runner as RV
    labels = D.read_labels(path)
    pmap = json.loads((S.LABELS / 'PACKET_KEYS_v2.json').read_text(encoding='utf-8'))['map']
    rev = json.loads((S.LABELS_R / 'PACKET_INDEX_v2r.json').read_text(encoding='utf-8'))
    changed = set(rev['changed_context_ids'])
    labels = {cid: v for cid, v in labels.items() if cid not in changed}
    if path_r:
        labels.update({cid: v for cid, v in D.read_labels(path_r).items() if cid in changed})
    idx2 = S.LABELS_R2 / 'PACKET_INDEX_v2r2.json'
    changed2 = set(json.loads(idx2.read_text(encoding='utf-8'))['changed_context_ids']) if idx2.exists() else set()
    labels = {cid: v for cid, v in labels.items() if cid not in changed2}
    if path_r2:
        labels.update({cid: v for cid, v in D.read_labels(path_r2).items() if cid in changed2})
    changed |= changed2
    wh = packet_withheld()                        # withheld contexts (population check, redaction rule) excluded
    drop = set(wh['withheld_context_ids']) | (changed - set(labels))
    labels = {cid: v for cid, v in labels.items() if cid not in drop}
    pmap = {cid: k for cid, k in pmap.items() if cid not in drop}
    by = {}
    for r in RV.collect(S.SELECT):
        if r['pass'] == 'main':
            by.setdefault(r['key'], {})[r['selector']] = r
    res = D.score_core(labels, {cid: by.get(k, {}) for cid, k in pmap.items()})
    res['withheld_contexts_excluded'] = wh
    res['redacted_contexts'] = {'changed': sorted(changed), 'relabels': str(path_r) if path_r else None,
                                'changed_v2_2': sorted(changed2), 'relabels_v2_2': str(path_r2) if path_r2 else None,
                                'excluded_without_relabel': sorted(changed - set(labels) - set(wh['withheld_context_ids']))}
    res['stability'] = {p: D.stability(p, RV.collect(p)) for p in S.RUN_ORDER if RV.period_root(p).exists()}
    res['label_source'] = S.LLM_LABEL_NOTE
    res['labels_sha256'] = R.sha(path)
    res['relabels_sha256'] = R.sha(path_r) if path_r else None
    res['relabels_v2_2_sha256'] = R.sha(path_r2) if path_r2 else None
    R.new_json(S.LABELS / f'LABEL_SCORES_v2_{res["labels_sha256"][:8]}.json', res)
    return res
