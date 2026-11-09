"""TD4 v2 span-level redaction: final inputs and withheld set per period (after the three rounds of td4v2_redact).

Kept apart from td4v2_redact so that the transform module stays byte-identical to the version that ran the rounds
(every ROUND.json, VERIFY and RESIDUAL file records its SHA256).

Withheld (every arm alike, before any provider call): the reviewers' adjudicated 2021-2022 key sets; every input with a
flagged class that could not be located in article text in any round (this includes the M4 firm-level classes NONAME
and NONAME_AFTER_2022, which have no text span); every input still flagged after the round-3 re-run. Writes
<masker-v2 build>/REDACTION_FINAL_<UTC>.json. Rule v2.1 (finalize_v21): for 2023-2025 the two M4 firm-level classes
no longer withhold and MASKER_NO_SOURCE (td4v2_nosource) does; written as REDACTION_FINAL_v2_1_<UTC>.json, which
td4v2_inputs reads when present. Reads no label, return or provider output.
"""
import json
from collections import Counter
from pathlib import Path

import td4_env as E
import td4v2_redact as RD


def final_path(d):
    """The binding redaction result of a masker-v2 build folder: the latest v2.2 file REDACTION_FINAL_v2_2_<UTC>
    (td4v2_redact22.finalize22) when one exists, else the latest rule-v2.1 file REDACTION_FINAL_v2_1_<UTC>, else the
    latest rule-v2 file REDACTION_FINAL_<UTC> (earlier files stay as the record)."""
    runs = sorted(Path(d).glob('REDACTION_FINAL_v2_2_*.json')) or \
        sorted(Path(d).glob('REDACTION_FINAL_v2_1_*.json')) or sorted(Path(d).glob('REDACTION_FINAL_2*.json'))
    if not runs:
        raise PermissionError('span-level redaction not finalised for ' + str(d))
    return runs[-1]


def v2_path(d):
    runs = sorted(Path(d).glob('REDACTION_FINAL_2*.json'))
    if not runs:
        raise PermissionError('rule-v2 redaction result missing for ' + str(d))
    return runs[-1]


_M4 = {}


def placeholder_dependent(run_dir, text_file, key, cls):
    """True when every hit of class cls for key in the detector run rests on a redaction placeholder, so the match is
    the redaction itself and not a residual: the M4 give-back patterns (GB_*) accept any bracket token in their token
    slot, and a slot that holds [REDACTED], [DATE] or [TERM] ('Logone [REDACTED]') names no firm; an M4 word hit that
    is the word inside a placeholder ('TERM' of '[TERM]') likewise. Firm-level classes and every other family are
    never placeholder-dependent (text_file is the text the run scanned; kept for the record)."""
    if not cls.startswith('M4:') or cls in RD.FIRM_LEVEL:
        return False
    run_dir = str(run_dir)
    if run_dir not in _M4:
        idx = {}
        for h in RD._jsonl(Path(run_dir) / 'ma1' / 's13_m4_hits.jsonl'):
            idx.setdefault(h['key'], []).append(h)
        _M4[run_dir] = idx
    hs = [h for h in _M4[run_dir].get(key, []) if 'M4:' + h['class'] == cls]
    if not hs:
        return False
    for h in hs:
        if h.get('token') in RD.PLACEHOLDERS:
            continue
        w = '[' + (h.get('match') or '') + ']'
        if w in RD.PLACEHOLDERS and w in h.get('ctx', ''):
            continue
        return False
    return True


def _q(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def finalize(period):
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    import td4v2_spec as S
    period = str(period)
    d = FL._dir(period)
    rounds = [json.loads((RD.round_dir(period, r) / 'ROUND.json').read_text(encoding='utf-8'))
              for r in range(1, RD.ROUNDS + 1)]
    resid = [json.loads(RD._residual_file(period, r).read_text(encoding='utf-8')) for r in range(1, RD.ROUNDS + 1)]
    verify = sorted(RD.round_dir(period, RD.ROUNDS).glob('VERIFY_*.json'))
    if not verify or not json.loads(verify[-1].read_text(encoding='utf-8'))['clean']:
        raise PermissionError('round-3 invariant check missing or not clean')
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    callable_keys = []
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            k = json.loads(line)['key']
            if k not in pop:
                callable_keys.append(k)
    cset = set(callable_keys)
    n_call = len(callable_keys)
    adj = set(rounds[0]['adjudicated_withheld']) & cset
    # placeholder-dependent flags: a detector match that rests on a redaction placeholder is the redaction itself
    srcs = [Path(d) / 'INPUTS.jsonl'] + [RD.round_dir(period, r) / 'INPUTS.jsonl' for r in range(1, RD.ROUNDS + 1)]
    unl_by_round = []
    pdep = Counter()
    for i, r in enumerate(rounds):                     # round i+1 located the hits of the run on round i's text
        keep = {}
        for k, cs in r['unlocated'].items():
            cs2 = [c for c in cs if not placeholder_dependent(r['detector_run'], srcs[i], k, c)]
            pdep.update(f'round {i + 1} unlocated {c}' for c in cs if c not in cs2)
            if cs2:
                keep[k] = cs2
        unl_by_round.append(keep)
    by_key3 = {}
    for c, ks in resid[-1]['by_class_keys'].items():
        for k in ks:
            by_key3.setdefault(k, []).append(c)
    still3_cls = {}
    for k, cs in by_key3.items():
        cs2 = [c for c in cs if not placeholder_dependent(resid[-1]['run_dir'], srcs[RD.ROUNDS], k, c)]
        pdep.update(f'round 3 still flagged {c}' for c in cs if c not in cs2)
        if cs2:
            still3_cls[k] = cs2
    unloc = set().union(*[set(u) for u in unl_by_round]) & cset
    still3 = set(still3_cls) & cset
    unloc_raw = set().union(*[set(r['unlocated']) for r in rounds]) & cset
    still3_raw = set(by_key3) & cset
    firm_level = set()
    for u in unl_by_round:
        firm_level |= {k for k, cs in u.items() if any(c in RD.FIRM_LEVEL for c in cs)}
    firm_level |= {k for k, cs in still3_cls.items() if any(c in RD.FIRM_LEVEL for c in cs)}
    firm_level &= cset
    withheld = adj | unloc | still3
    alt = set(adj)                      # the same rule if the two M4 firm-level classes were not withholding classes
    for u in unl_by_round:
        alt |= {k for k, cs in u.items() if any(c not in RD.FIRM_LEVEL for c in cs)}
    alt |= {k for k, cs in still3_cls.items() if any(c not in RD.FIRM_LEVEL for c in cs)}
    alt &= cset
    raw_withheld = adj | unloc_raw | still3_raw          # without the placeholder-dependence rule (record)
    per_key, kinds, chars, ledgers = Counter(), Counter(), 0, {}
    for r in range(1, RD.ROUNDS + 1):
        lp = RD.round_dir(period, r) / 'LEDGER.jsonl'
        ledgers[str(lp)] = R.sha(lp)
        for e in RD._jsonl(lp):
            if e['key'] in cset:
                per_key[e['key']] += 1
                chars += e['length']
                kinds[e['placeholder']] += 1
    counts_all = [per_key.get(k, 0) for k in callable_keys]
    counts_red = [c for c in counts_all if c]
    art_chars = rounds[0]['article_chars_callable']
    sub = None
    if period == str(S.SELECT):
        keep = IV.subsample(S.SELECT)
        sc = [k for k in callable_keys if k in keep]
        sub = {'subsample_callable': len(sc), 'withheld_in_subsample': len(withheld & set(sc)),
               'withheld_share_subsample': round(len(withheld & set(sc)) / max(1, len(sc)), 5),
               'inputs_with_redaction_in_subsample': sum(1 for k in sc if per_key.get(k))}
    final_inputs = RD.round_dir(period, RD.ROUNDS) / 'INPUTS.jsonl'
    res = {'period': period, 'rule': 'span-level redaction, three rounds (td4v2_redact; PREREGISTRATION_v2 redaction)',
           'build_dir': str(d), 'population_check_file': str(IV.withheld_file(d)),
           'final_inputs': str(final_inputs), 'final_inputs_sha256': R.sha(final_inputs),
           'callable': n_call, 'inputs_with_redaction': len(counts_red),
           'inputs_with_redaction_share': round(len(counts_red) / n_call, 5),
           'redactions_per_input_all': {'mean': round(sum(counts_all) / n_call, 3), 'p50': _q(counts_all, 0.5),
                                        'p95': _q(counts_all, 0.95)},
           'redactions_per_redacted_input': {'mean': round(sum(counts_red) / max(1, len(counts_red)), 3),
                                             'p50': _q(counts_red, 0.5), 'p95': _q(counts_red, 0.95)},
           'replacements_by_placeholder': dict(kinds), 'chars_replaced_v2': chars, 'article_chars_v2': art_chars,
           'article_char_share_redacted': round(chars / art_chars, 6) if art_chars else None,
           'flagged_before_round_1': rounds[0]['flagged_inputs_entering'],
           'residual_after_round': {str(r['round']): r['still_flagged_inputs'] for r in resid},
           'residual_share_after_round': {str(r['round']): r['still_flagged_share'] for r in resid},
           'residual_by_class_after_round_3': resid[-1]['by_class_inputs'],
           'unlocated_by_round': {str(r['round']): r['unlocated_inputs'] for r in rounds},
           'unlocated_by_class_by_round': {str(r['round']): r['unlocated_by_class'] for r in rounds},
           'adjudicated_withheld': len(adj), 'unlocated_withheld': len(unloc), 'still_flagged_after_round_3': len(still3),
           'firm_level_flag_inputs': len(firm_level),
           'withheld_keys': sorted(withheld), 'withheld_inputs': len(withheld),
           'withheld_share': round(len(withheld) / n_call, 5), 'over_2pct': len(withheld) / n_call > 0.02,
           'placeholder_dependent_flags': dict(sorted(pdep.items())),
           'withheld_if_placeholder_dependent_flags_counted': len(raw_withheld),
           'withheld_share_if_placeholder_dependent_flags_counted': round(len(raw_withheld) / n_call, 5),
           'withheld_without_firm_level_classes': len(alt),
           'withheld_share_without_firm_level_classes': round(len(alt) / n_call, 5), 'subsample': sub,
           'rounds': [{k: v for k, v in r.items() if k not in ('unlocated', 'adjudicated_withheld')} for r in rounds],
           'residual_files': [str(RD._residual_file(period, r)) for r in range(1, RD.ROUNDS + 1)],
           'residual_files_sha256': [R.sha(RD._residual_file(period, r)) for r in range(1, RD.ROUNDS + 1)],
           'detector_reruns': [r['run_dir'] for r in resid], 'ledgers_sha256': ledgers,
           'verify_file': str(verify[-1]), 'verify_sha256': R.sha(verify[-1]),
           'redact_code_sha256': R.sha(Path(RD.__file__)), 'code_sha256': R.sha(Path(__file__)),
           'note': 'withheld for every arm alike before any provider call; no outcome or provider output is read',
           **R.peak()}
    out = Path(d) / f'REDACTION_FINAL_{R.utc().replace(":", "")}.json'
    R.new_json(out, res)
    return {k: v for k, v in res.items() if k not in ('withheld_keys', 'rounds')}


def finalize_v21(period):
    """Rule v2.1 (coordinator technical correction, 2026-11-07, before any provider call and any outcome): for 2023,
    2024 and 2025 the M4 firm-level classes NONAME and NONAME_AFTER_2022 no longer withhold and MASKER_NO_SOURCE
    (td4v2_nosource) does; 2021-2022 is unchanged. The text, the rounds, the placeholder-dependence rule and every other
    withholding reason are those of rule v2. Writes <build>/REDACTION_FINAL_v2_1_<UTC>.json with the v2 numbers next to
    the v2.1 numbers and the redaction counts of the inputs that move from withheld to sent."""
    E.paths()
    import td3_runtime as R
    import td4v2_flags as FL
    import td4v2_inputs as IV
    import td4v2_nosource as NS
    import td4v2_spec as S
    period = str(period)
    d = FL._dir(period)
    v2p = v2_path(d)
    v2 = json.loads(v2p.read_text(encoding='utf-8'))
    nsp = NS.latest(d)
    ns = json.loads(nsp.read_text(encoding='utf-8'))
    rounds = [json.loads((RD.round_dir(period, r) / 'ROUND.json').read_text(encoding='utf-8'))
              for r in range(1, RD.ROUNDS + 1)]
    resid = json.loads(RD._residual_file(period, RD.ROUNDS).read_text(encoding='utf-8'))
    pop = set(json.loads(IV.withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    callable_keys = []
    with (Path(d) / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            k = json.loads(line)['key']
            if k not in pop:
                callable_keys.append(k)
    cset = set(callable_keys)
    v2w = set(v2['withheld_keys'])
    ns_keys = set(ns['flagged_keys']) & cset
    if period == str(S.SELECT):
        withheld = set(v2w)                                  # 2021-2022: rule v2 unchanged
        reasons = {'rule_v2_unchanged': len(withheld)}
    else:
        srcs = [Path(d) / 'INPUTS.jsonl'] + [RD.round_dir(period, r) / 'INPUTS.jsonl' for r in range(1, RD.ROUNDS + 1)]
        text_unloc, text_still = set(), set()
        for i, r in enumerate(rounds):
            for k, cs in r['unlocated'].items():
                if any(c not in RD.FIRM_LEVEL and not placeholder_dependent(r['detector_run'], srcs[i], k, c)
                       for c in cs):
                    text_unloc.add(k)
        for c, ks in resid['by_class_keys'].items():
            if c in RD.FIRM_LEVEL:
                continue
            for k in ks:
                if not placeholder_dependent(resid['run_dir'], srcs[RD.ROUNDS], k, c):
                    text_still.add(k)
        adj = set(rounds[0]['adjudicated_withheld'])
        withheld = (adj | text_unloc | text_still | ns_keys) & cset
        reasons = {'adjudicated': len(adj & cset), 'unlocatable_text_classes': len(text_unloc & cset),
                   'still_flagged_text_classes_after_round_3': len(text_still & cset),
                   'masker_no_source': len(ns_keys)}
    moved = v2w - withheld
    added = withheld - v2w
    per_key = {}
    for r in range(1, RD.ROUNDS + 1):
        for e in RD._jsonl(RD.round_dir(period, r) / 'LEDGER.jsonl'):
            if e['key'] in moved:
                per_key[e['key']] = per_key.get(e['key'], 0) + 1
    mc = [per_key.get(k, 0) for k in sorted(moved)]
    n_call = len(callable_keys)
    sub = None
    if period == str(S.SELECT):
        keep = IV.subsample(S.SELECT)
        sc = [k for k in callable_keys if k in keep]
        sub = {'subsample_callable': len(sc), 'withheld_in_subsample': len(withheld & set(sc)),
               'withheld_share_subsample': round(len(withheld & set(sc)) / max(1, len(sc)), 5)}
    keepf = ('period', 'build_dir', 'population_check_file', 'final_inputs', 'final_inputs_sha256', 'callable',
             'inputs_with_redaction', 'inputs_with_redaction_share', 'redactions_per_input_all',
             'redactions_per_redacted_input', 'replacements_by_placeholder', 'chars_replaced_v2', 'article_chars_v2',
             'article_char_share_redacted', 'flagged_before_round_1', 'residual_after_round',
             'residual_share_after_round', 'residual_by_class_after_round_3', 'unlocated_by_round',
             'unlocated_by_class_by_round', 'placeholder_dependent_flags', 'residual_files', 'residual_files_sha256',
             'detector_reruns', 'ledgers_sha256', 'verify_file', 'verify_sha256', 'redact_code_sha256')
    res = {**{k: v2[k] for k in keepf if k in v2},
           'rule': 'span-level redaction, three rounds; rule v2.1: for 2023-2025 MASKER_NO_SOURCE replaces the M4 '
                   'firm-level classes NONAME and NONAME_AFTER_2022 as a withholding class (PREREGISTRATION_v2_1)',
           'rule_version': 'v2.1', 'v2_file': str(v2p), 'v2_file_sha256': R.sha(v2p),
           'masker_no_source_file': str(nsp), 'masker_no_source_sha256': R.sha(nsp),
           'masker_no_source_applied': period != str(S.SELECT), 'masker_no_source_flagged': len(ns_keys),
           'withheld_reasons_v2_1': reasons,
           'withheld_keys': sorted(withheld), 'withheld_inputs': len(withheld),
           'withheld_share': round(len(withheld) / n_call, 5), 'over_2pct': len(withheld) / n_call > 0.02,
           'v2_withheld_inputs': len(v2w), 'v2_withheld_share': v2['withheld_share'],
           'moved_to_sent': len(moved), 'added_to_withheld': len(added),
           'moved_redactions': {'inputs': len(mc), 'mean': round(sum(mc) / len(mc), 3) if mc else None,
                                'p50': _q(mc, 0.5), 'p95': _q(mc, 0.95), 'with_no_redaction': sum(1 for c in mc if not c)},
           'subsample': sub, 'code_sha256': R.sha(Path(__file__)),
           'note': 'withheld for every arm alike before any provider call; no outcome or provider output is read',
           **R.peak()}
    out = Path(d) / f'REDACTION_FINAL_v2_1_{R.utc().replace(":", "")}.json'
    R.new_json(out, res)
    return {k: v for k, v in res.items() if k not in ('withheld_keys',)}


def stand_in(d):
    """Rehearsal only: an identity redaction (the sandbox build's own INPUTS, nothing withheld) for a sandbox build under
    OUT/rehearsal_v2_<id>/ (the reviewers' detectors are not run on development data; an engineering check)."""
    import td3_runtime as R
    import td4v2_spec as S
    if 'rehearsal_v2_' not in str(d):
        raise PermissionError('stand-in redaction only in a rehearsal sandbox')
    p = Path(d) / 'INPUTS.jsonl'
    tag = S.RULE_VERSION.replace('.', '_')
    return R.new_json(Path(d) / f'REDACTION_FINAL_{tag}_{R.utc().replace(":", "")}.json',
                      {'final_inputs': str(p), 'final_inputs_sha256': R.sha(p), 'withheld_keys': [],
                       'withheld_inputs': 0, 'rule_version': S.RULE_VERSION, 'rehearsal_stand_in': True})
