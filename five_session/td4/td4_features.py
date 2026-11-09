"""TD2 exposures E, Z = C x E and witness evidence for the 2025 firm-weeks (features only; no return or label).

TD2's tested primitives are used unchanged: engine.news_pool / day_clock, fast_builder.graph_at / exposure,
build_v2.feature_schema / EVIDENCE_SCHEMA / to_table / coverage / firm_index, decoder_v3.week_stories. Two functions are
copies with one change each (marked TD4:): mapped_edges (fast_builder.mapped_edges with the 2024-12-31 seal literal
replaced by a parameter) and origin_rows (build_v2.origin_rows calling that copy). parity2024 runs the copies on 2024
origins (TD2 sources, and again with the TD4 2025-sealed sources) and compares every feature and evidence value with
TD2 features_v2/evidence_v2.
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E
import td4_sources as S
import td4_manifest as M

FEATURES = T.OUT / 'features'
EVIDENCE = T.OUT / 'evidence'


def _td2():
    E.paths()
    import engine
    import fast_builder
    import build_v2
    import decoder_v3
    return engine, fast_builder, build_v2, decoder_v3


def mapped_edges(graph, links, origin, suspects=None, seal=T.HARD_SEAL):
    """fast_builder.mapped_edges (TD4: seal parameter instead of the 2024-12-31 literal; otherwise unchanged)."""
    engine, _, _, _ = _td2()
    day = str(pd.Timestamp(origin).date())
    if day > seal:
        raise ValueError('Graph date seal')
    eligible = graph[graph.start < day].drop_duplicates(['id', 'start'], keep='first').drop_duplicates('id', keep='last')
    eligible = eligible[eligible.end.isna() | (eligible.end > day)]
    lm = links[(links.start < day) & (links.end >= day)].drop_duplicates(['entity', 'permno'])
    if suspects:
        lm = lm[[(r.entity, int(r.permno)) not in suspects for r in lm.itertuples()]]
    lm = lm[~lm.duplicated('entity', keep=False)]
    lookup = dict(zip(lm.entity, lm.permno))
    eligible = eligible.assign(a=eligible.src.map(lookup), b=eligible.tgt.map(lookup))
    eligible = eligible[eligible.a.notna() & eligible.b.notna() & (eligible.a != eligible.b)]
    clocks = {s: engine.day_clock(s) for s in eligible.start.unique()}
    return [{'id': r.id, 'a': int(r.a), 'b': int(r.b), 'start': r.start, 'end': r.end,
             'available_at': clocks[r.start], 'rownum': int(r.rownum), 'typ': r.typ} for r in eligible.itertuples()]


def origin_rows(origin, cutoff, g, names, stories, graph, links, suspects, idmap, seal=T.HARD_SEAL):
    """build_v2.origin_rows (TD4: calls the seal-parameter mapped_edges copy; otherwise unchanged)."""
    engine, fast_builder, build_v2, _ = _td2()
    HOPS, FEATURE_INT, VARIANTS = (1, 2, 3), build_v2.FEATURE_INT, build_v2.VARIANTS
    origin = pd.Timestamp(origin)
    cutoff = pd.Timestamp(cutoff)
    pool = engine.news_pool(stories, cutoff, origin)
    features = []
    evidence = []
    for variant in VARIANTS:
        bad = suspects if variant == 'fsb5_sensitivity' else None
        adj = fast_builder.graph_at(mapped_edges(graph, links, origin, bad, seal=seal), cutoff, origin)   # TD4
        for r in g.itertuples(index=False):
            root = idmap.get(r.firm_id, -1)
            chars = {n: float(getattr(r, n)) for n in names}
            avail = {n: getattr(r, n + '__available_at') for n in names}
            values, witness = fast_builder.exposure(root, cutoff, origin, chars, avail, adj, pool, names)
            row = {'firm_id': r.firm_id, 'cutoff': cutoff, 'origin_session': origin, 'variant': variant,
                   'mapped_focal': root != -1}
            for n in names:
                row['C_' + n] = chars[n]
                a = avail[n]
                row['available_' + n] = pd.NaT if pd.isna(a) else pd.Timestamp(a)
            for v in values:
                h, n = v['hop'], v['name']
                row[f'Z{h}_{n}'] = v['Z']
                row[f'E{h}_{n}'] = v['E']
                row['missing_' + n] = bool(v['characteristic_missing'])
                for c in FEATURE_INT:
                    row[f'H{h}_{c}'] = int(v[c])
            for n in names:
                row['Z_' + n] = sum(row[f'Z{h}_{n}'] for h in HOPS) / 3
                row['E_' + n] = sum(row[f'E{h}_{n}'] for h in HOPS) / 3
            row['has_news'] = any(row[f'H{h}_selected_partners'] > 0 for h in HOPS)
            features.append(row)
            for w in witness:
                edge_at = pd.Timestamp(w['max_edge_available_at'])
                news_at = pd.Timestamp(w['max_news_available_at'])
                if edge_at >= cutoff or news_at >= cutoff:
                    raise ValueError('Witness availability not strictly before cutoff')
                if w['physical_channel_verified'] is not False:
                    raise ValueError('Physical channel must never be asserted')
                evidence.append({'firm_id': r.firm_id, 'cutoff': cutoff, 'origin_session': origin,
                                 'variant': variant, 'hop': int(w['hop']), 'partner': int(w['partner']),
                                 'nodes': [int(x) for x in w['nodes']],
                                 'intermediate_nodes': [int(x) for x in w['intermediate_nodes']],
                                 'relationship_ids': [str(x) for x in w['relationship_ids']],
                                 'relationship_source_rows': [int(x) for x in w['relationship_source_rows']],
                                 'reporting_roles': [str(x) for x in w['reporting_roles']],
                                 'max_edge_available_at': edge_at,
                                 'article_ids': [str(x) for x in w['article_ids']], 'n_articles': len(w['article_ids']),
                                 'max_news_available_at': news_at,
                                 'intermediate_product_evidence': w['intermediate_product_evidence'],
                                 'physical_channel_verified': False})
    return features, evidence


def alive_filter(t, min_day):
    """Keep every version of the relationship IDs that have at least one version with a null end or an end after
    min_day (see sources_for)."""
    import pyarrow.compute as pc
    alive = pc.or_kleene(pc.is_null(t['end']), pc.fill_null(pc.greater(t['end'], min_day), False))
    ids = pc.unique(t.filter(alive)['id'])
    return t.filter(pc.is_in(t['id'], value_set=ids))


def sources_for(seal, min_day=None):
    """(graph, links, suspects) sealed at `seal`: TD2's own files for 2024-12-31, TD4's extracts for 2025-12-31.

    min_day (memory only, exact): relationship IDs none of whose versions can be active on or after min_day (every
    version has an end on or before min_day) are dropped before conversion. mapped_edges keeps, per ID, the latest
    version starting before the origin and then requires end > origin; for such an ID that version ends on or before
    min_day <= origin, so the ID never yields an edge for origins on or after min_day."""
    import pyarrow.compute as pc
    if seal == T.TD2_SEAL:
        gp, lp = T.TD2 / 'sources/graph_october_sealed.parquet', T.TD2 / 'sources/factset_links.parquet'
    elif seal == T.HARD_SEAL:
        gp, lp = S.TD2SRC / 'graph_october_sealed_2025.parquet', S.TD2SRC / 'factset_links_2025.parquet'
    else:
        raise ValueError('seal')
    t = pq.read_table(gp, columns=['id', 'date', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'])
    t = t.filter(pc.less_equal(t['date'], seal))
    if min_day is not None:
        t = alive_filter(t, min_day)
    graph = t.to_pandas()
    del t
    graph = graph.rename(columns={'date': 'start'})
    links = pq.read_table(lp, columns=['entity', 'permno', 'date', 'end']).to_pandas()
    links = links[links.date <= seal].rename(columns={'date': 'start'})
    suspects = {tuple(x) for x in json.loads((T.TD2 / 'sources/identity_sensitivity_v1.json')
                                             .read_text(encoding='utf-8'))['links']}
    return graph, links, suspects


def news_for_year(year):
    """decoder_v3.news_for_year: partition year-1 from Dec 20 plus partition year (TD2 or TD4 news scores)."""
    cols = ['date', 'alt_id', 'firms', 'first_months', 'available_at'] + T.NAMES
    chunks = []
    for y in (year - 1, year):
        lo = pd.Timestamp(f'{year - 1}-12-20') if y < year else pd.Timestamp(f'{year}-01-01')
        f = pq.read_table(S.news_path(y), columns=cols).to_pandas()
        chunks.append(f[(f.date >= lo) & (f.date <= pd.Timestamp(f'{y}-12-31'))])
    return pd.concat(chunks, ignore_index=True)


def manifest_features(year):
    cols = ['firm_id', 'cutoff', 'origin_session', 'eligible_features'] + T.NAMES + [n + '__available_at' for n in T.NAMES]
    f = pq.read_table(M.manifest_path(year), columns=cols).to_pandas()
    return f[f.eligible_features].copy()


def build(year):
    """Streaming E/Z/evidence build for 2025 (exclusive files; refuses to overwrite)."""
    import td3_runtime as R
    engine, fast_builder, build_v2, decoder_v3 = _td2()
    if year != T.TEST_YEAR:
        raise PermissionError('TD4 builds 2025 features only (2015-2024 are TD2 partitions)')
    start = time.perf_counter()
    final, evpath = FEATURES / f'year_{year}.parquet', EVIDENCE / f'year_{year}.parquet'
    for p in (final, evpath):
        if p.exists():
            raise FileExistsError(p)
    f = manifest_features(year)
    graph, links, suspects = sources_for(T.HARD_SEAL, min_day=str(pd.Timestamp(f.origin_session.min()).date()))
    idmap = build_v2.firm_index(links)
    news = news_for_year(year)
    if build_v2.forbidden_columns(list(f.columns)):
        raise ValueError('Manifest projection contains outcome-like columns')
    if f.duplicated(['firm_id', 'cutoff']).any():
        raise ValueError('Duplicate manifest firm-week')
    fschema = build_v2.feature_schema(T.NAMES)
    rows = 0
    with E.private(final).open('xb') as fh, E.private(evpath).open('xb') as eh:
        fw = pq.ParquetWriter(fh, fschema, compression='zstd')
        ew = pq.ParquetWriter(eh, build_v2.EVIDENCE_SCHEMA, compression='zstd')
        for origin, g in f.groupby('origin_session', sort=True):
            E.guard()
            cutoffs = g.cutoff.unique()
            if len(cutoffs) != 1:
                raise ValueError('Cutoff inconsistent within origin')
            stories = decoder_v3.week_stories(news, origin)
            feats, ev = origin_rows(origin, cutoffs[0], g, T.NAMES, stories, graph, links, suspects, idmap)
            table = build_v2.to_table(feats, fschema)
            fw.write_table(table)
            rows += table.num_rows
            if ev:
                ew.write_table(build_v2.to_table(ev, build_v2.EVIDENCE_SCHEMA))
            print('TD4_FEATURES', year, str(pd.Timestamp(origin).date()), rows, flush=True)
        fw.close()
        ew.close()
    frame = pq.read_table(final).to_pandas()
    if len(frame) != 2 * len(f):
        raise ValueError('Row count mismatch after build')
    info = {'version': T.VERSION, 'year': year, 'coverage': build_v2.coverage(frame, T.NAMES),
            'manifest_eligible_rows': int(len(f)), 'feature_rows': int(len(frame)), 'feature_sha256': R.sha(final),
            'evidence_sha256': R.sha(evpath), 'labels_or_returns_read': False,
            'elapsed_seconds': time.perf_counter() - start, **R.peak()}
    R.new_json(T.OUT / f'FEATURES_{year}_{info["feature_sha256"][:8]}.json', info)
    return {k: v for k, v in info.items() if k != 'coverage'}


def parity2024(limit=None, seal=None, **_):
    """TD4 copies on a fixed sample of 2024 origins versus TD2 features_v2/evidence_v2, with TD2's 2024-sealed sources
    (seal 2024-12-31) or with TD4's 2025-sealed sources (seal 2025-12-31), which must give the same 2024 rows."""
    from td4_parity import _frames_equal
    engine, fast_builder, build_v2, decoder_v3 = _td2()
    seal = seal or T.TD2_SEAL
    f = manifest_features(2024)
    origins = sorted(f.origin_session.unique())
    n = limit or 6
    pick = [origins[int(i)] for i in np.linspace(0, len(origins) - 1, n)]
    news = news_for_year(2024)
    out = {'origins': [str(pd.Timestamp(o).date()) for o in pick], 'seal': seal}
    graph, links, suspects = sources_for(seal, min_day=str(pd.Timestamp(min(pick)).date()))
    idmap = build_v2.firm_index(links)
    feats, evs = [], []
    for o in pick:
        g = f[f.origin_session == o]
        stories = decoder_v3.week_stories(news, o)
        a, b = origin_rows(o, g.cutoff.iloc[0], g, T.NAMES, stories, graph, links, suspects, idmap, seal=seal)
        feats += a
        evs += b
    del graph, links, news
    mine_f = build_v2.to_table(feats, build_v2.feature_schema(T.NAMES)).to_pandas()
    mine_e = build_v2.to_table(evs, build_v2.EVIDENCE_SCHEMA).to_pandas()
    del feats, evs
    sel = [('origin_session', 'in', [pd.Timestamp(o) for o in pick])]
    ff = pq.read_table(T.TD2 / 'features_v2/year_2024.parquet', filters=sel).to_pandas()
    fe = pq.read_table(T.TD2 / 'evidence_v2/year_2024.parquet', filters=sel).to_pandas()
    keys = ['firm_id', 'cutoff', 'variant']
    res_f = _frames_equal(mine_f, ff[mine_f.columns], sort=keys)
    for col in ('nodes', 'intermediate_nodes', 'relationship_ids', 'relationship_source_rows', 'reporting_roles',
                'article_ids'):
        mine_e[col] = mine_e[col].map(lambda v: tuple(v))
        fe[col] = fe[col].map(lambda v: tuple(v))
    res_e = _frames_equal(mine_e, fe[mine_e.columns], sort=keys + ['hop', 'partner'])
    out.update(features=res_f, evidence=res_e, equal=res_f['equal'] and res_e['equal'])
    return out


def parity2024_td4src(limit=None, **_):
    return parity2024(limit=limit, seal=T.HARD_SEAL)
