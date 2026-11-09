"""Inputs sent to the providers per v2 period (2122 selection subsample, 2023, 2024, 2025): masker-v2 builds after the
pre-registered span-level redaction (td4v2_redact; researcher decision 2026-11-07).

Every period is built with masker v2 (td4v2_build2.PERIOD_DIR; every directory used must carry a MASK_V2_RECEIPT.json;
no input built with the v1 masker is read). The text sent is the round-3 output of the span-level redaction named by
the build's latest REDACTION_FINAL_<UTC>.json (td4v2_redact_final; its file SHA256 is checked). Sent inputs = those
texts minus the keys withheld by td4v2_verify2.population (WITHHELD_POPULATION_M2*.json, required) minus the keys the
redaction rule withholds (adjudicated 2021-2022 key sets, unlocatable flags, inputs still flagged after round 3). The
whole-input detector-flag rule (td4v2_flags DETECTOR_FLAGS_M2*.json) is superseded and no longer withholds. The
selection period is further restricted to the 20,000-firm-week subsample (the first 20,000 TD3-eligible candidates in
SHA256('TD4-select-2122|' + key) order). Paths are read from td4v2_build2 at call time (the rehearsal points them at a
sandbox).
"""
import json
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

import td4_spec as T
import td4v2_spec as S


def period_dir(period):
    import td4v2_build2 as B2
    key = S.SELECT if str(period) == str(S.SELECT) else int(period)
    d = Path(B2.PERIOD_DIR[key])
    if not (d / 'BUILD.json').exists():
        raise PermissionError('masker-v2 build missing: ' + str(d))
    rec = d / 'MASK_V2_RECEIPT.json'
    if not rec.exists() or json.loads(rec.read_text(encoding='utf-8')).get('v1_masker_inputs_used') is not False:
        raise PermissionError('not a masker-v2 build: ' + str(d))
    return d


def period_dirs(period):
    return [period_dir(period)]


def withheld_file(d):
    """The latest population check of the build (a re-run writes WITHHELD_POPULATION_M2_<UTC>.json)."""
    runs = sorted(Path(d).glob('WITHHELD_POPULATION_M2_*.json'))
    p = runs[-1] if runs else Path(d) / 'WITHHELD_POPULATION_M2.json'
    if not p.exists():
        raise PermissionError('Population check (td4v2_verify2) not run for ' + str(d))
    return p


def flags_file(d):
    """The detector-flag file of the build (td4v2_flags.run; superseded by the redaction rule, kept as the record)."""
    import td4v2_flags as FL
    p = FL.flags_path(d)
    if not p.exists():
        raise PermissionError('Detector-flag rule (td4v2_flags) not run for ' + str(d))
    return p


def redaction_file(d):
    """The build's latest span-level redaction result (td4v2_redact_final.finalize)."""
    import td4v2_redact_final as RF
    return RF.final_path(d)


def inputs_path(d):
    """The redacted INPUTS file named by the redaction result, after checking its SHA256."""
    import hashlib
    rec = json.loads(redaction_file(d).read_text(encoding='utf-8'))
    p = Path(rec['final_inputs'])
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    if h.hexdigest() != rec['final_inputs_sha256']:
        raise PermissionError('redacted inputs changed after the redaction result was written: ' + str(p))
    return p


def withheld(d):
    """Population-check keys and the keys withheld by the redaction rule (both withheld from every arm alike)."""
    out = set(json.loads(withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    out |= set(json.loads(redaction_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    return out


def subsample_frame(d=None):
    """The selection subsample: the first SUBSAMPLE_N TD3-eligible candidates (built, or withheld by a build residual)
    in hash order. Returns the ELIGIBLE rows with columns rk (record key) and rank."""
    import td4v2_build as BV
    d = Path(d or period_dir(S.SELECT))
    e = pq.read_table(d / 'ELIGIBLE.parquet').to_pandas()
    e['rk'] = [BV.record_key(a, b) for a, b in zip(e.firm_id, e.cutoff)]
    e['rank'] = [BV.select_rank(k) for k in e.rk]
    e = e.sort_values('rank', kind='stable')
    td3_eligible = e[(e.status == 'input_built') | e.status.str.startswith('withheld_')]
    n = S.SUBSAMPLE_N
    sub = td3_eligible.iloc[:n]
    if len(sub) < n:
        raise ValueError(f'Only {len(sub)} TD3-eligible candidates built; raise CANDIDATES')
    return sub


def subsample(period):
    """Record keys of the selection subsample (None for test periods)."""
    if str(period) != str(S.SELECT):
        return None
    return set(subsample_frame().rk)


def callable_inputs(period):
    import td4_inputs as I
    d = period_dir(period)
    recs = I.load_inputs(None, inputs_path(d))
    bad = withheld(d)
    keep = subsample(period)
    return {k: v for k, v in recs.items() if k not in bad and (keep is None or k in keep)}


def eligible_map(period, keys=None):
    """(firm_id, cutoff) -> record key for the inputs sent in the period."""
    keys = set(callable_inputs(period)) if keys is None else keys
    e = pq.read_table(period_dir(period) / 'ELIGIBLE.parquet').to_pandas()
    e = e[(e.status == 'input_built') & e.key.isin(keys)]
    if e.key.duplicated().any():
        raise ValueError('Duplicate key in a build')
    return {(a, pd.Timestamp(c)): k for a, c, k in zip(e.firm_id, e.cutoff, e.key)}


def counts(period):
    """Counts for the handoff: eligible, built, withheld at build, population-withheld, redaction-withheld, sent."""
    d = period_dir(period)
    b = json.loads((d / 'BUILD.json').read_text(encoding='utf-8'))['counts']
    pop = set(json.loads(withheld_file(d).read_text(encoding='utf-8'))['withheld_keys'])
    red = json.loads(redaction_file(d).read_text(encoding='utf-8'))
    rw = set(red['withheld_keys'])
    out = {'build': b, 'population_withheld': len(pop), 'redaction_withheld': len(rw),
           'withheld_union': len(pop | rw), 'sent': len(callable_inputs(period)), 'dir': str(d),
           'redaction_file': str(redaction_file(d)), 'final_inputs': red.get('final_inputs')}
    if str(period) == str(S.SELECT):
        sub = subsample_frame()
        out['subsample'] = int(len(sub))
        out['subsample_status'] = {k: int(v) for k, v in sub.status.value_counts().items()}
    return out
