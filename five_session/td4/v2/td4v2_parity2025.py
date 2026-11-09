"""N9 (TD4C): 2025 source and feature parity re-run with the frozen code, with the code hashes in every receipt.

The 2025 extracts were built before the last edits of td4_sources.py and td4_features.py. These checks regenerate,
with the current (frozen) v1 code, the 2025 graph and link extracts in full and the 2025 features and evidence for a
fixed, evenly spaced set of origins (memory-bounded, like td4_features.parity2024), and require equality with the
bound files. Every receipt records the SHA256 of the code files it ran. No label or return is read.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E

CODE_FILES = ('td4_sources.py', 'td4_features.py', 'td4_manifest.py', 'td4_parity.py', 'td4_spec.py', 'td4_env.py')


def code_hashes():
    out = {n: hashlib.sha256((T.CODE / n).read_bytes()).hexdigest() for n in CODE_FILES}
    out['v2/td4v2_parity2025.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return out


def graph2025(**_):
    import td4_sources as S
    t, excluded = S.graph_extract(T.HARD_SEAL)
    mine = t.to_pandas()
    del t
    bound = pq.read_table(S.TD2SRC / 'graph_october_sealed_2025.parquet', columns=list(mine.columns)).to_pandas()
    from td4_parity import _frames_equal
    res = _frames_equal(mine, bound)
    return {'graph': res, 'excluded': excluded, 'equal': res['equal']}


def links2025(**_):
    import td4_sources as S
    from td4_parity import _frames_equal
    mine = S.links_extract(T.HARD_SEAL)
    bound = pq.read_table(S.TD2SRC / 'factset_links_2025.parquet').to_pandas()
    cols = [c for c in mine.columns if c in bound.columns]
    res = _frames_equal(mine[cols].sort_values(cols).reset_index(drop=True),
                        bound[cols].sort_values(cols).reset_index(drop=True))
    return {'links': res, 'equal': res['equal']}


def features2025(limit=None, **_):
    """td4_features copies on n evenly spaced 2025 origins versus the bound features/evidence year_2025 files."""
    import td4_features as F
    from td4_parity import _frames_equal
    engine, fast_builder, build_v2, decoder_v3 = F._td2()
    f = F.manifest_features(2025)
    origins = sorted(f.origin_session.unique())
    n = limit or 8
    pick = [origins[int(i)] for i in np.linspace(0, len(origins) - 1, n)]
    news = F.news_for_year(2025)
    graph, links, suspects = F.sources_for(T.HARD_SEAL, min_day=str(pd.Timestamp(min(pick)).date()))
    idmap = build_v2.firm_index(links)
    feats, evs = [], []
    for o in pick:
        g = f[f.origin_session == o]
        stories = decoder_v3.week_stories(news, o)
        a, b = F.origin_rows(o, g.cutoff.iloc[0], g, T.NAMES, stories, graph, links, suspects, idmap,
                             seal=T.HARD_SEAL)
        feats += a
        evs += b
        E.guard()
    del graph, links, news
    mine_f = build_v2.to_table(feats, build_v2.feature_schema(T.NAMES)).to_pandas()
    mine_e = build_v2.to_table(evs, build_v2.EVIDENCE_SCHEMA).to_pandas()
    del feats, evs
    sel = [('origin_session', 'in', [pd.Timestamp(o) for o in pick])]
    ff = pq.read_table(F.FEATURES / 'year_2025.parquet', filters=sel).to_pandas()
    fe = pq.read_table(F.EVIDENCE / 'year_2025.parquet', filters=sel).to_pandas()
    keys = ['firm_id', 'cutoff', 'variant']
    res_f = _frames_equal(mine_f, ff[mine_f.columns], sort=keys)
    for col in ('nodes', 'intermediate_nodes', 'relationship_ids', 'relationship_source_rows', 'reporting_roles',
                'article_ids'):
        mine_e[col] = mine_e[col].map(lambda v: tuple(v))
        fe[col] = fe[col].map(lambda v: tuple(v))
    res_e = _frames_equal(mine_e, fe[mine_e.columns], sort=keys + ['hop', 'partner'])
    return {'origins': [str(pd.Timestamp(o).date()) for o in pick], 'features': res_f, 'evidence': res_e,
            'equal': res_f['equal'] and res_e['equal']}


def _v1(name):
    def f(limit=None, **_):
        import td4_parity as P1
        checks = {**P1.CHECKS, **P1._late_checks()}
        return checks[name](limit=limit)
    return f


CHECKS = {'graph2025_frozen_code': graph2025, 'links2025_frozen_code': links2025,
          'features2025_frozen_code': features2025}
# the v1 parity checks re-run with the frozen code (TD4C N9: several v1 receipts predate the last code edits)
for _n in ('graph2024', 'links2024', 'td1snap2024', 'news2024', 'manifest2024', 'features2024', 'features2024_td4src',
           'panel2022', 'panel2024'):
    CHECKS[_n + '_frozen_code'] = _v1(_n)


def run(name, limit=None):
    import td3_runtime as R
    t0 = time.perf_counter()
    res = {'check': name, 'version': 'TD4-provider-selectors-v2', **CHECKS[name](limit=limit),
           'code_sha256': code_hashes(), 'elapsed_seconds': time.perf_counter() - t0, **R.peak()}
    tag = R.sha_text(json.dumps(res, sort_keys=True, default=str))[:8]
    R.new_json(T.OUT / 'parity' / f'PARITY_{name}_{tag}.json', res)
    return {k: v for k, v in res.items() if k in ('check', 'equal', 'elapsed_seconds', 'peak_working_set_bytes')}
