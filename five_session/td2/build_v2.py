"""Streaming E/Z builder for every eligible TD1 firm-week (features only; no returns or labels).

Uses the tested v1 primitives (engine.news_pool, fast_builder.mapped_edges/graph_at/exposure)
and writes features_v2/, evidence_v2/ and coverage_v2/ under the private root.
"""
import hashlib
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import engine
import fast_builder
import io_v2 as io
from runtime_v3 import guard, sha, new_json, peak
from spec_v2 import NAMES, HOPS, VARIANTS, FEATURE_INT, feature_columns, forbidden_columns, VERSION

def feature_schema(names=NAMES):
    fields = []
    for c in feature_columns(names):
        if c in ('firm_id', 'variant'):
            t = pa.string()
        elif c in ('cutoff', 'origin_session') or c.startswith('available_'):
            t = pa.timestamp('ns')
        elif c in ('mapped_focal', 'has_news') or c.startswith('missing_'):
            t = pa.bool_()
        elif any(c.endswith('_' + x) for x in FEATURE_INT) and c[0] == 'H':
            t = pa.int64()
        else:
            t = pa.float64()
        fields.append(pa.field(c, t))
    return pa.schema(fields)

EVIDENCE_SCHEMA = pa.schema([
    ('firm_id', pa.string()), ('cutoff', pa.timestamp('ns')), ('origin_session', pa.timestamp('ns')),
    ('variant', pa.string()), ('hop', pa.int64()), ('partner', pa.int64()),
    ('nodes', pa.list_(pa.int64())), ('intermediate_nodes', pa.list_(pa.int64())),
    ('relationship_ids', pa.list_(pa.string())), ('relationship_source_rows', pa.list_(pa.int64())),
    ('reporting_roles', pa.list_(pa.string())), ('max_edge_available_at', pa.timestamp('ns')),
    ('article_ids', pa.list_(pa.string())), ('n_articles', pa.int64()),
    ('max_news_available_at', pa.timestamp('ns')), ('intermediate_product_evidence', pa.string()),
    ('physical_channel_verified', pa.bool_())])

def firm_index(links):
    """Opaque TD1 firm IDs for mapped PERMNOs (same hash as TD1 build.py)."""
    return {hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]: int(p)
            for p in pd.unique(links.permno)}

def origin_rows(origin, cutoff, g, names, stories, graph, links, suspects, idmap):
    """All feature and evidence rows for one origin. g holds manifest feature columns only.

    Any label-like column present in g is ignored by construction (never referenced).
    """
    origin = pd.Timestamp(origin); cutoff = pd.Timestamp(cutoff)
    pool = engine.news_pool(stories, cutoff, origin)
    features = []; evidence = []
    for variant in VARIANTS:
        bad = suspects if variant == 'fsb5_sensitivity' else None
        adj = fast_builder.graph_at(fast_builder.mapped_edges(graph, links, origin, bad), cutoff, origin)
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
                row[f'Z{h}_{n}'] = v['Z']; row[f'E{h}_{n}'] = v['E']
                row['missing_' + n] = bool(v['characteristic_missing'])
                for c in FEATURE_INT:
                    row[f'H{h}_{c}'] = int(v[c])
            for n in names:
                row['Z_' + n] = sum(row[f'Z{h}_{n}'] for h in HOPS) / 3
                row['E_' + n] = sum(row[f'E{h}_{n}'] for h in HOPS) / 3
            row['has_news'] = any(row[f'H{h}_selected_partners'] > 0 for h in HOPS)
            features.append(row)
            for w in witness:
                edge_at = pd.Timestamp(w['max_edge_available_at']); news_at = pd.Timestamp(w['max_news_available_at'])
                if edge_at >= cutoff or news_at >= cutoff:
                    raise ValueError('Witness availability not strictly before cutoff')
                if w['physical_channel_verified'] is not False:
                    raise ValueError('Physical channel must never be asserted')
                evidence.append({'firm_id': r.firm_id, 'cutoff': cutoff, 'origin_session': origin,
                    'variant': variant, 'hop': int(w['hop']), 'partner': int(w['partner']),
                    'nodes': [int(x) for x in w['nodes']], 'intermediate_nodes': [int(x) for x in w['intermediate_nodes']],
                    'relationship_ids': [str(x) for x in w['relationship_ids']],
                    'relationship_source_rows': [int(x) for x in w['relationship_source_rows']],
                    'reporting_roles': [str(x) for x in w['reporting_roles']], 'max_edge_available_at': edge_at,
                    'article_ids': [str(x) for x in w['article_ids']], 'n_articles': len(w['article_ids']),
                    'max_news_available_at': news_at,
                    'intermediate_product_evidence': w['intermediate_product_evidence'],
                    'physical_channel_verified': False})
    return features, evidence

def to_table(rows, schema):
    frame = pd.DataFrame(rows, columns=schema.names)
    for f in schema:
        if pa.types.is_timestamp(f.type):
            frame[f.name] = pd.to_datetime(frame[f.name]).astype('datetime64[ns]')
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)

def coverage(frame, names):
    out = []
    for variant, g in frame.groupby('variant', sort=True):
        allmissing = np.logical_and.reduce([g['missing_' + n].to_numpy() for n in names])
        out.append({'variant': variant, 'rows': int(len(g)), 'mapped_focal': int(g.mapped_focal.sum()),
            'firm_weeks_with_news': int(g.has_news.sum()),
            'missing_characteristics': {n: int(g['missing_' + n].sum()) for n in names},
            'rows_all_characteristics_missing': int(allmissing.sum()),
            'nonzero_Z_main': {n: int((g['Z_' + n] != 0).sum()) for n in names},
            'hop_coverage': {str(h): {'rows_with_graph_partner': int((g[f'H{h}_graph_partners'] > 0).sum()),
                'rows_with_news_partner': int((g[f'H{h}_news_partners'] > 0).sum()),
                'rows_with_selected_partner': int((g[f'H{h}_selected_partners'] > 0).sum()),
                'selected_partner_packets': int(g[f'H{h}_selected_partners'].sum()),
                'unique_articles_sum': int(g[f'H{h}_unique_articles'].sum()),
                'nonzero_Z': {n: int((g[f'Z{h}_{n}'] != 0).sum()) for n in names}} for h in HOPS}})
    return out

def build_year(year, names=None):
    start = time.perf_counter(); guard()
    if not 2015 <= year <= 2024:
        raise ValueError('Build year 2015-2024 only')
    names = list(NAMES if names is None else names)
    final = io.FEATURES / f'year_{year}.parquet'; evpath = io.EVIDENCE / f'year_{year}.parquet'
    covpath = io.COVERAGE / f'year_{year}.json'
    for p in (final, evpath, covpath):
        if p.exists():
            raise FileExistsError(p)
    graph, links, suspects = io.graph_sources()
    idmap = firm_index(links)
    news = io.news_for_year(year)
    f = io.manifest_features(year)
    if forbidden_columns(list(f.columns)):
        raise ValueError('Manifest projection contains outcome-like columns')
    f = f[f.eligible_features].copy()
    if f.duplicated(['firm_id', 'cutoff']).any():
        raise ValueError('Duplicate manifest firm-week')
    fschema = feature_schema(names); rows = 0; summary = []
    with io.parquet_stream(final, fschema) as fw, io.parquet_stream(evpath, EVIDENCE_SCHEMA) as ew:
        for origin, g in f.groupby('origin_session', sort=True):
            guard(); cutoffs = g.cutoff.unique()
            if len(cutoffs) != 1:
                raise ValueError('Cutoff inconsistent within origin')
            stories = io.week_stories(news, origin)
            feats, ev = origin_rows(origin, cutoffs[0], g, names, stories, graph, links, suspects, idmap)
            table = to_table(feats, fschema); fw.write_table(table); rows += table.num_rows
            if ev:
                ew.write_table(to_table(ev, EVIDENCE_SCHEMA))
            print('BUILD_V2', year, str(pd.Timestamp(origin).date()), rows, flush=True)
    frame = io.read_features(year, [c for c in fschema.names if c != 'origin_session'])
    if len(frame) != 2 * len(f):
        raise ValueError('Row count mismatch after build')
    new_json(covpath, {'version': VERSION, 'year': year, 'coverage': coverage(frame, names),
        'manifest_eligible_rows': int(len(f)), 'feature_rows': int(len(frame)),
        'feature_sha256': sha(final), 'evidence_sha256': sha(evpath),
        'intermediate_product_evidence_observed': 0, 'physical_channel_verified_rows': 0,
        'labels_or_returns_read': False, 'elapsed_seconds': time.perf_counter() - start, **peak()})
    print('BUILD_V2_DONE', year, str(final), flush=True)
