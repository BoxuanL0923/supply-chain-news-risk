"""Selector inputs for every eligible 2024 and 2025 firm-week, built with the cleared TD3 v6 code.

Unchanged TD3 v6 functions do the work: td3_build.with_percentiles, load_evidence, load_news, verify_article,
ric_map, fetch_texts, priority, path_relations, assign_tokens, aliases_for, article_extras, article_entries, render,
build_inputs, name_sources, install_tables; td3_mask.Masker with the frozen MASK_TABLES_v4.json; td3_scan for the
audit sample. TD3's frozen input header (TD3-input-v7) and question texts are kept byte for byte.

Isolated changes (each justified in TD4_HANDOFF.md, Deviations):
D1 period parameters: td3_spec YEARS, FIRST_DAY, LAST_DAY, LAST_MONTH, NEWS_YEARS and ARTICLE_LO are set per period
   (TD3 hard-codes 2021-2022), and the Sources object points at the period's feature, evidence and news files.
D2 eligible(): a copy of td3_build.eligible with its hard-coded 2021-2022 year checks replaced by the period year.
D3 edge_maps(): a copy of td3_build.edge_maps calling td4_features.mapped_edges (seal parameter) instead of
   fast_builder.mapped_edges (2024-12-31 literal), so that 2025 origins can be mapped.
D4 every eligible firm-week instead of a 200-firm-week sample, processed in (cutoff, firm_id) order in chunks; the
   archive texts are read from a period article store (archive rows of the period's evidence article IDs, physical
   order preserved, so fetch_texts' first-physical-row rule is unchanged).
D5 aliases_for is memoised per PERMNO inside the process (it depends on the origin only through the metadata field
   comnams_after_origin, recomputed per call); input texts are unchanged (pilot parity).
D6 build_inputs is called per firm-week; a strict or date masking residual (which aborts a TD3 build) withholds that
   firm-week from all three providers alike and is recorded (its row then uses the fallback).
D7 full TD3-style metadata (aliases, raw texts, replacement logs, sweep rows) is kept only for a fixed audit sample;
   every input keeps compact metadata.
parity_pilot rebuilds the 200 TD3 pilot firm-weeks with the 2021-2022 parameters and requires byte-identical texts.
"""
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import td4_spec as T
import td4_env as E

CHUNK = 1500
AUDIT_N = 300                  # fixed audit sample per period (full metadata, scanner, sweep)
INPUTS = T.OUT / 'inputs'


def _mods():
    E.paths()
    import td3_spec as S3
    import td3_build as B
    import td3_mask
    import td3_scan
    return S3, B, td3_mask, td3_scan


# ------------------------------------------------------------------------------------------------- period parameters
def period_params(years):
    years = tuple(sorted(years))
    lo, hi = years[0], years[-1]
    return {'YEARS': years, 'FIRST_DAY': f'{lo}-01-01', 'LAST_DAY': f'{hi}-12-31', 'LAST_MONTH': hi * 100 + 12,
            'NEWS_YEARS': tuple(range(lo - 1, hi + 1)), 'ARTICLE_LO': f'{lo - 1}-12-20'}


def set_period(years):
    """D1: set the TD3 seal constants for the period (module attributes read at call time by td3_build/td3_scan)."""
    S3, B, _, _ = _mods()
    p = period_params(years)
    if p['LAST_DAY'] > T.HARD_SEAL:
        raise PermissionError('TD4 seal')
    for k, v in p.items():
        setattr(S3, k, v)
    return p


def feature_path(year):
    return (T.TD2 / f'features_v2/year_{year}.parquet') if year <= 2024 else (T.OUT / f'features/year_{year}.parquet')


def evidence_path(year):
    return (T.TD2 / f'evidence_v2/year_{year}.parquet') if year <= 2024 else (T.OUT / f'evidence/year_{year}.parquet')


def store_path(tag):
    return INPUTS / f'article_store_{tag}.parquet'


def sources(years, store):
    import td4_sources as S
    _, B, _, _ = _mods()
    seal = f'{max(years)}-12-31'
    if max(years) <= 2024:
        gp, lp = T.TD2 / 'sources/graph_october_sealed.parquet', T.TD2 / 'sources/factset_links.parquet'
    else:
        gp, lp = S.TD2SRC / 'graph_october_sealed_2025.parquet', S.TD2SRC / 'factset_links_2025.parquet'
    p = period_params(years)
    return B.Sources(menu=T.TD2 / 'risk_menu.csv', features={y: feature_path(y) for y in years},
                     evidence={y: evidence_path(y) for y in years},
                     news={y: S.news_path(y) for y in p['NEWS_YEARS']}, graph=gp, links=lp,
                     ric_identity=T.TD2 / 'sources/reuters_identity_development_v1.json', articles=store,
                     crsp_names=tuple(T.CRSP_NAME_FILES), td2_bound=False, factset=T.FSB1_COMPANY,
                     english=Path('C:/Users/Boxuan/AppData/Roaming/nltk_data/corpora/words/en'),
                     tables=T.TD3 / 'tables_v6/MASK_TABLES_v4.json'), seal


# -------------------------------------------------------------------------------------------------- copied functions
def eligible(frame, years):
    """D2: td3_build.eligible with the period years in place of the 2021-2022 literals (otherwise unchanged)."""
    S3, _, _, _ = _mods()
    lo, hi = pd.Timestamp(S3.FIRST_DAY), pd.Timestamp(S3.LAST_DAY)
    o, c = frame.origin_session, frame.cutoff
    period = (o >= lo) & (o <= hi) & (c >= lo) & (c < hi + pd.Timedelta(days=1)) & (c > o)
    anychar = ~np.logical_and.reduce([frame['missing_' + n].to_numpy(bool) for n in S3.NAMES])
    e = frame[period & frame.has_news.astype(bool) & frame.mapped_focal.astype(bool) & anychar]
    if (e.origin_session.dt.year > max(years)).any() or (e.cutoff.dt.year > max(years)).any() or \
            (e.origin_session.dt.year < min(years)).any():
        raise PermissionError('Row outside the period reached the eligible frame')
    if e.duplicated(['firm_id', 'cutoff']).any():
        raise ValueError('Duplicate firm-week')
    return e.reset_index(drop=True)


def edge_maps(sources_, accepted, seal):
    """D3: td3_build.edge_maps calling td4_features.mapped_edges (seal parameter); otherwise unchanged."""
    _, B, _, _ = _mods()
    import td4_features as F
    rel_ids = sorted({rid for a in accepted for p in a['partners'] for rid in p['row'].relationship_ids})
    graph = B._concat(list(B.sealed_batches(sources_.graph, ['id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'],
                                            'date', '1900-01-01', seal,
                                            isin=('id', pa.array(rel_ids, type=pa.string())))),
                      ['date', 'id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum']).rename(columns={'date': 'start'})
    links = B._concat(list(B.sealed_batches(sources_.links, ['entity', 'permno', 'end'], 'date', '1900-01-01', seal)),
                      ['date', 'entity', 'permno', 'end']).rename(columns={'date': 'start'})
    maps = {}
    for origin in sorted({pd.Timestamp(a['cand'].origin_session) for a in accepted}):
        maps[origin] = {e['id']: e for e in F.mapped_edges(graph, links, origin, seal=seal)}
    return maps, links


from collections import OrderedDict
_ALIAS_CACHE = OrderedDict()
ALIAS_CACHE_MAX = 4000


def memoised_aliases(original):
    """D5: cache aliases_for per PERMNO (bounded LRU); comnams_after_origin (metadata only) is recomputed for each
    origin. aliases_for reads only the firm's own CRSP/FactSet rows, its issuer siblings and near-name index entries
    keyed by its own name anchors, so the value does not depend on which other firms share the name sources."""
    S3, _, _, _ = _mods()

    def wrapper(permno, origin, ns):
        key = int(permno)
        if key in _ALIAS_CACHE:
            _ALIAS_CACHE.move_to_end(key)
        else:
            _ALIAS_CACHE[key] = original(permno, origin, ns)
            if len(_ALIAS_CACHE) > ALIAS_CACHE_MAX:
                _ALIAS_CACHE.popitem(last=False)
        al, comnams, _after, fs, scn = _ALIAS_CACHE[key]
        day = pd.Timestamp(origin).strftime('%Y-%m-%d')
        rows = [x for x in ns['names'].get(key, []) if x[0] <= S3.LAST_DAY]
        after = sorted({c for d, c, _ in rows if c and d > day})
        return copy.deepcopy(al), list(comnams), after, list(fs), list(scn)
    wrapper.__wrapped__ = original
    return wrapper


def performance_shims(td3_mask):
    """D5 (performance only; outputs unchanged, checked by parity_pilot): td3_mask.variant_regex, a pure function of
    (alias, acronym) that the masker and loose_scan call for every alias of every article, is memoised; Python's
    regex cache is enlarged so that loose_scan's per-alias patterns are compiled once instead of on every article."""
    import functools
    import re as _re
    if not hasattr(td3_mask.variant_regex, 'cache_info'):
        td3_mask.variant_regex = functools.lru_cache(maxsize=200000)(td3_mask.variant_regex)
    _re._MAXCACHE = max(_re._MAXCACHE, 8000)


def trim_caches(td3_mask):
    """Memory only (outputs unchanged): empty the pure caches between chunks once they are large (td3_mask's compiled
    regex cache, which TD3 itself clears in set_tables/clear_tables, the variant_regex memo, Python's regex cache)."""
    import re as _re
    if len(td3_mask._COMPILED) > 20000:
        td3_mask._COMPILED.clear()
    if hasattr(td3_mask.variant_regex, 'cache_info') and td3_mask.variant_regex.cache_info().currsize > 40000:
        td3_mask.variant_regex.cache_clear()
    if len(getattr(_re, '_cache', {})) > 6000:
        _re.purge()


# ---------------------------------------------------------------------------------------------------- article store
def build_store(years, frame_keys, tag):
    """Archive rows (date, alt_id, rics, headline, body_short) whose alt_id appears in the primary evidence rows of
    the given firm-weeks, dated in [ARTICLE_LO, LAST_DAY], in the archive's physical order (private file)."""
    S3, B, _, _ = _mods()
    path = store_path(tag)
    if path.exists():
        return path, None
    ids = set()
    for y in years:
        pf = pq.ParquetFile(evidence_path(y))
        for b in pf.iter_batches(batch_size=65536, columns=['firm_id', 'cutoff', 'variant', 'article_ids'],
                                 use_threads=False):
            f = b.to_pandas()
            f = f[f.variant == 'primary']
            if frame_keys is not None:
                f = f[[k in frame_keys for k in zip(f.firm_id, f.cutoff)]]
            for lst in f.article_ids:
                ids.update(lst)
            E.guard()
    value_set = pa.array(sorted(ids), type=pa.string())
    pf = pq.ParquetFile(T.ARTICLES, pre_buffer=False, buffer_size=65536)
    schema = pa.schema([('date', pf.schema_arrow.field('date').type), ('alt_id', pa.string()),
                        ('rics', pf.schema_arrow.field('rics').type), ('headline', pf.schema_arrow.field('headline').type),
                        ('body_short', pf.schema_arrow.field('body_short').type)])
    rows = 0
    with E.private(path).open('xb') as fh:
        w = pq.ParquetWriter(fh, schema, compression='zstd')
        for raw in B.sealed_batches(T.ARTICLES, ['alt_id', 'rics', 'headline', 'body_short'], 'date', S3.ARTICLE_LO,
                                    S3.LAST_DAY, isin=('alt_id', value_set), batch_size=8192):
            t = pa.Table.from_pandas(raw[['date', 'alt_id', 'rics', 'headline', 'body_short']], schema=schema,
                                     preserve_index=False)
            w.write_table(t)
            rows += t.num_rows
        w.close()
    return path, {'wanted_ids': len(ids), 'store_rows': rows}


# ------------------------------------------------------------------------------------------------------- chunk logic
def accept_chunk(sources_, chunk, ricmap, engine, stats):
    """td3_build.select_sample's per-chunk logic for every candidate of the chunk (no sample cut-off)."""
    _, B, _, _ = _mods()
    ev = B.load_evidence(sources_, chunk)
    news = B.load_news(sources_, {a for lst in ev.article_ids for a in lst})
    verdict = {}
    for w in ev.itertuples(index=False):
        for aid in w.article_ids:
            verdict[(w.firm_id, w.cutoff, int(w.partner), aid)] = B.verify_article(aid, w.partner, w.origin_session,
                                                                                  w.cutoff, news)
    wanted = {k[3]: news[k[3]] for k, v in verdict.items() if v is None}
    texts, tstatus = B.fetch_texts(sources_, wanted, ricmap, engine.score)
    groups = {k: g for k, g in ev.groupby(['firm_id', 'cutoff'], sort=False)}
    accepted, rejected = [], []
    for cand in chunk.itertuples(index=False):
        stats['examined'] += 1
        g = groups.get((cand.firm_id, cand.cutoff))
        if g is None or not len(g):
            stats['rejected']['no_evidence_rows'] += 1
            rejected.append((cand, 'no_evidence_rows'))
            continue
        for h in (1, 2, 3):
            if int((g.hop == h).sum()) != int(getattr(cand, f'H{h}_selected_partners')):
                raise ValueError('Evidence rows differ from TD2 selected-partner counts')
        g = g.assign(_p=[B.priority(cand.cutoff, p) for p in g.partner])
        ordered = g.sort_values(['hop', '_p'], kind='stable')
        if list(ordered.index) != list(g.sort_values('hop', kind='stable').index):
            stats['partner_order_equal_td2'] = False
        partners = []
        for w in ordered.itertuples(index=False):
            good = []
            for aid in w.article_ids:
                reason = verdict[(w.firm_id, w.cutoff, int(w.partner), aid)]
                if reason is None:
                    reason = tstatus.get(aid)
                if reason is None:
                    good.append((news[aid]['date'], aid))
                else:
                    stats['article_rejects'][reason] += 1
            good.sort(key=lambda t: (-t[0].value, t[1]))
            partners.append({'row': w, 'article': good[0][1] if good else None})
        if not any(p['article'] for p in partners):
            stats['rejected']['no_verified_article'] += 1
            rejected.append((cand, 'no_verified_article'))
            continue
        stats['partners_without_article'] += sum(p['article'] is None for p in partners)
        accepted.append({'cand': cand, 'partners': partners, 'news': news, 'texts': texts})
    stats['text_status'].update('ok' if v is None else v for v in tstatus.values())
    return accepted, rejected


def compact_meta(meta, record):
    return {'key': meta['key'], 'firm_id': meta['firm_id'], 'cutoff': meta['cutoff'],
            'origin_session': meta['origin_session'], 'focal_permno': meta['focal_permno'],
            'shown': [{k: s[k] for k in ('hop', 'permno', 'token', 'role', 'article', 'edge_age_days')}
                      for s in meta['shown']],
            'links': meta['links'], 'articles': [{'alt_id': x['alt_id'], 'tokens': x['tokens'],
                                                  'days_before': x['days_before'], 'news_day': x['news_day'],
                                                  'truncated': x['truncated']} for x in meta['articles']],
            'replacements_by_token': meta['replacements_by_token'], 'collisions': len(meta['collisions']),
            'loose_hits': len(meta['loose_hits']), 'input_sha256': record['input_sha256'],
            'input_bytes': record['input_bytes']}


def audit_rank(firm_id, cutoff, year):
    return hashlib.sha256(f'TD4-audit-v1|{year}|{firm_id}|{pd.Timestamp(cutoff).isoformat()}'.encode()).hexdigest()


# --------------------------------------------------------------------------------------------------------- the build
def run_build(years, out_dir, frame_filter=None, order=None, audit_n=AUDIT_N, limit=None, tag=None,
              chunk=CHUNK, text_hook=None):
    """Build inputs for all eligible firm-weeks of `years` (or the subset frame_filter, in `order`)."""
    import td3_runtime as R
    import td4_features as F
    S3, B, td3_mask, td3_scan = _mods()
    start = time.perf_counter()
    out_dir = Path(out_dir)
    targets = {k: out_dir / f for k, f in (('inputs', 'INPUTS.jsonl'), ('meta', 'INPUT_META_COMPACT.jsonl'),
                                           ('audit', 'AUDIT_META.jsonl'), ('sweep', 'AUDIT_SWEEP.jsonl'),
                                           ('withheld', 'WITHHELD.jsonl'), ('eligible', 'ELIGIBLE.parquet'),
                                           ('receipt', 'BUILD.json'))}
    for p in targets.values():
        if p.exists():
            raise FileExistsError(p)
    params = set_period(years)
    engine, _, _, _ = F._td2()
    store_tag = tag or f'{min(years)}_{max(years)}'
    frame = B.with_percentiles(pd.concat([B.features_year(sources(years, None)[0], y) for y in years],
                                         ignore_index=True))
    elig = eligible(frame, years)
    if frame_filter is not None:
        elig = frame_filter(elig)
    if order is None:
        elig = elig.sort_values(['cutoff', 'firm_id'], kind='stable').reset_index(drop=True)
    else:
        elig = order(elig)
    if limit:
        elig = elig.iloc[:limit]
    keys = set(zip(elig.firm_id, elig.cutoff))
    store, store_info = build_store(years, keys, store_tag)
    src, seal = sources(years, store)
    tables_sha = B.install_tables(src)
    if tables_sha != T.TD3_TABLES_SHA256:
        raise ValueError('Frozen TD3 mask table differs')
    menu = S3.menu_rows(src.menu)
    ricmap = B.ric_map(src.ric_identity)
    B.aliases_for = memoised_aliases(B.aliases_for.__wrapped__ if hasattr(B.aliases_for, '__wrapped__')
                                     else B.aliases_for)
    performance_shims(td3_mask)
    ranked = sorted((audit_rank(a, b, max(years)), a, b) for a, b in keys)
    audit_keys = {(a, b) for _, a, b in ranked[:audit_n]}
    stats = {'examined': 0, 'rejected': Counter(), 'article_rejects': Counter(), 'text_status': Counter(),
             'partners_without_article': 0, 'partner_order_equal_td2': True, 'withheld': Counter()}
    eligible_rows, n_inputs, bytes_ = [], 0, []
    parts = out_dir / 'parts'
    kinds = ('inputs', 'meta', 'audit', 'sweep', 'withheld')
    final_parts = []
    for ci, c0 in enumerate(range(0, len(elig), chunk)):
        E.guard()
        done_path = parts / f'DONE_{ci:04d}.json'
        part = elig.iloc[c0:c0 + chunk]
        if done_path.exists():                       # resume: a finished chunk is verified and reused, never rebuilt
            d = json.loads(done_path.read_text(encoding='utf-8'))
            if d['first_key'] != [part.firm_id.iloc[0], pd.Timestamp(part.cutoff.iloc[0]).isoformat()] or \
                    d['rows'] != len(part):
                raise ValueError('Chunk boundaries differ from the finished chunk ' + str(ci))
            for k, (fp, h) in d['files'].items():
                if R.sha(fp) != h:
                    raise ValueError('Finished chunk file changed: ' + fp)
            for k in ('rejected', 'article_rejects', 'text_status', 'withheld'):
                stats[k].update(d['stats'][k])
            stats['examined'] += d['stats']['examined']
            stats['partners_without_article'] += d['stats']['partners_without_article']
            stats['partner_order_equal_td2'] &= d['stats']['partner_order_equal_td2']
            eligible_rows += [dict(r, cutoff=pd.Timestamp(r['cutoff'])) for r in d['eligible_rows']]
            n_inputs += d['n_inputs']
            bytes_ += d['bytes']
            final_parts.append(d['files'])
            print('TD4_INPUTS resume chunk', ci, 'reused', flush=True)
            continue
        attempt = 1 + len(list(parts.glob(f'INPUTS_{ci:04d}_a*.jsonl')))
        files = {k: parts / f'{k.upper()}_{ci:04d}_a{attempt}.jsonl' for k in kinds}
        writers = {k: R.JsonlWriter(files[k], R.Redactor()) for k in kinds}
        cst = {'examined': 0, 'rejected': Counter(), 'article_rejects': Counter(), 'text_status': Counter(),
               'partners_without_article': 0, 'partner_order_equal_td2': True, 'withheld': Counter()}
        c_rows, c_bytes, c_n = [], [], 0
        accepted, rejected = accept_chunk(src, part, ricmap, engine, cst)
        if text_hook is not None and accepted:        # D8 repair pass only (td4_repair); never in the main builds
            cst['text_hook_replacements'] = 0
            for aid, t in accepted[0]['texts'].items():
                t['headline'], n1 = text_hook(t['headline'])
                t['body'], n2 = text_hook(t['body'])
                cst['text_hook_replacements'] += n1 + n2
        for cand, why in rejected:
            c_rows.append({'firm_id': cand.firm_id, 'cutoff': cand.cutoff, 'status': why, 'key': None})
        if accepted:
            maps, links = edge_maps(src, accepted, seal)
            # name sources for the firms this chunk can show (focal + path nodes), as td3_build.build does for its
            # sample; aliases depend only on the firm, not on the set (see memoised_aliases), so per-chunk sources
            # bound the memory without changing any alias
            permnos = {int(x) for a in accepted for p in a['partners'] for x in p['row'].nodes}
            ns = B.name_sources(src, permnos, ricmap)
        for a in accepted:
            sweep = []
            cand = a['cand']
            try:
                records, metas = B.build_inputs(menu, [a], maps, links, ns, sweep)
            except ValueError as err:
                msg = str(err)
                kind = 'strict_residual' if 'Strict masking residual' in msg else (
                    'date_residual' if 'Date/boilerplate residual' in msg else 'build_error')
                cst['withheld'][kind] += 1
                writers['withheld'].write({'firm_id': cand.firm_id, 'cutoff': pd.Timestamp(cand.cutoff).isoformat(),
                                           'reason': kind, 'message': msg[:200]})
                c_rows.append({'firm_id': cand.firm_id, 'cutoff': cand.cutoff, 'status': 'withheld_' + kind,
                               'key': None})
                continue
            c_n += 1
            rank = n_inputs + c_n
            r, m = records[0], metas[0]
            r['sample_rank'] = m['sample_rank'] = rank
            r['period'] = int(max(years)) if len(years) == 1 else 'pilot'
            writers['inputs'].write(r)
            writers['meta'].write(compact_meta(m, r))
            if (cand.firm_id, cand.cutoff) in audit_keys:
                writers['audit'].write(m)
                for row in sweep:
                    row['rank'] = rank
                    row['row_id'] = B.sweep_row_id(row)
                    writers['sweep'].write(row)
            c_rows.append({'firm_id': cand.firm_id, 'cutoff': cand.cutoff, 'status': 'input_built', 'key': r['key']})
            c_bytes.append(r['input_bytes'])
        for w in writers.values():
            w.close()
        done = {'chunk': ci, 'rows': len(part), 'first_key': [part.firm_id.iloc[0],
                                                              pd.Timestamp(part.cutoff.iloc[0]).isoformat()],
                'n_inputs_before': n_inputs, 'n_inputs': c_n, 'bytes': c_bytes,
                'files': {k: (str(fp), R.sha(fp)) for k, fp in files.items()},
                'stats': {k: (dict(v) if isinstance(v, Counter) else v) for k, v in cst.items()},
                'eligible_rows': [dict(r, cutoff=pd.Timestamp(r['cutoff']).isoformat()) for r in c_rows]}
        R.new_json(done_path, done)
        for k in ('rejected', 'article_rejects', 'text_status', 'withheld'):
            stats[k].update(cst[k])
        stats['examined'] += cst['examined']
        stats['partners_without_article'] += cst['partners_without_article']
        stats['partner_order_equal_td2'] &= cst['partner_order_equal_td2']
        eligible_rows += c_rows
        n_inputs += c_n
        bytes_ += c_bytes
        final_parts.append(done['files'])
        trim_caches(td3_mask)
        print('TD4_INPUTS', store_tag, c0 + len(part), len(elig), n_inputs, round(time.perf_counter() - start),
              R.peak()['peak_working_set_bytes'], flush=True)
    for k in kinds:                                   # merge the finished chunk files in chunk order
        with E.private(targets[k]).open('xb') as out:
            for files in final_parts:
                with open(files[k][0], 'rb') as fin:
                    for line in fin:
                        out.write(line)
    ef = pd.DataFrame(eligible_rows, columns=['firm_id', 'cutoff', 'status', 'key'])
    E.write_parquet(targets['eligible'], ef)
    receipt = {'version': T.VERSION, 'years': list(years), 'params': params, 'input_version': S3.INPUT_VERSION,
               'question_sha256': S3.question_bundle(menu)[1], 'mask_tables_sha256': tables_sha,
               'counts': {'primary_rows': int(len(frame)), 'eligible_td2_unit': int(len(elig)),
                          'examined': stats['examined'], 'inputs_built': n_inputs,
                          'rejected': dict(stats['rejected']), 'withheld': dict(stats['withheld']),
                          'article_rejects': dict(stats['article_rejects']),
                          'archive_text_status': dict(stats['text_status']),
                          'shown_partners_without_article': stats['partners_without_article'],
                          'partner_priority_equals_td2_order': stats['partner_order_equal_td2'],
                          'audit_sample': len(audit_keys)},
               'input_bytes': {'mean': float(np.mean(bytes_)) if bytes_ else None,
                               'median': float(np.median(bytes_)) if bytes_ else None,
                               'p90': float(np.percentile(bytes_, 90)) if bytes_ else None},
               'store': str(store), 'store_info': store_info,
               'outputs_sha256': {k: R.sha(p) for k, p in targets.items() if k != 'receipt'},
               'labels_or_returns_read': False, 'provider_calls': 0,
               'elapsed_seconds': time.perf_counter() - start, **R.peak()}
    R.new_json(targets['receipt'], receipt)
    return receipt


def build(year, limit=None):
    if year not in T.PERIODS:
        raise PermissionError('Selector inputs are built for 2024 and 2025 only')
    out = INPUTS / f'period_{year}' if not limit else INPUTS / f'period_{year}_limit{limit}'
    r = run_build((year,), out, limit=limit, tag=str(year) if not limit else f'{year}_limit{limit}')
    return {'counts': r['counts'], 'input_bytes': r['input_bytes'], 'elapsed_seconds': r['elapsed_seconds'],
            'peak_working_set_bytes': r['peak_working_set_bytes']}


class LazyRecord(dict):
    """A frozen input record whose input_text is read from INPUTS.jsonl on access (by byte offset) and re-checked
    against its sha256, so that a period's ~0.6 GB of texts is never held in memory at once."""

    def __getitem__(self, k):
        if k == 'input_text' and not dict.__contains__(self, 'input_text'):
            import hashlib
            with open(dict.__getitem__(self, '_path'), 'rb') as f:
                f.seek(dict.__getitem__(self, '_offset'))
                r = json.loads(f.readline().decode('utf-8'))
            if r['key'] != dict.__getitem__(self, 'key') or \
                    hashlib.sha256(r['input_text'].encode('utf-8')).hexdigest() != dict.__getitem__(self, 'input_sha256'):
                raise ValueError('Input text changed on disk')
            return r['input_text']
        return dict.__getitem__(self, k)


def repair_dir(year):
    return INPUTS / f'repair_{year}'


def withheld_population(year):
    """Keys withheld after the build because the population checks (td4_verify) found a TD3 verify problem in their
    text (for example a role token glued to a hyphen and digits, TD3C4 class D2), in the main build and in the D8
    repair build. Required for real runs."""
    out = set()
    for d in (INPUTS / f'period_{year}', repair_dir(year)):
        p = d / f'WITHHELD_POPULATION_{year}.json'
        if d.exists() and (d / 'BUILD.json').exists():
            if not p.exists():
                raise PermissionError('Population check not run for ' + str(d) + ' (run_td4.py verify --year)')
            out |= set(json.loads(p.read_text(encoding='utf-8'))['withheld_keys'])
    if not (INPUTS / f'period_{year}' / f'WITHHELD_POPULATION_{year}.json').exists():
        raise PermissionError('Population check not run for ' + str(year))
    return out


REPAIR_RANK_OFFSET = 1_000_000


def callable_inputs(year, path=None):
    """The inputs sent to the providers: every input of the main build plus the D8 repair build (repaired firm-weeks
    were withheld in the main build, so keys never collide), except the population-check withheld keys. Repaired
    inputs get sample_rank + 1,000,000 (they are called after the main ones)."""
    recs = load_inputs(year, path)
    if path is None:
        rp = repair_dir(year) / 'INPUTS.jsonl'
        if rp.exists():
            for k, v in load_inputs(year, rp).items():
                if k in recs:
                    raise ValueError('Repaired key also in the main build: ' + k)
                dict.__setitem__(v, 'sample_rank', REPAIR_RANK_OFFSET + dict.__getitem__(v, 'sample_rank'))
                recs[k] = v
        bad = withheld_population(year)
    else:
        bad = set()
    return {k: v for k, v in recs.items() if k not in bad}


def load_inputs(year, path=None):
    """Frozen inputs of a period: {key: LazyRecord(key, sample_rank, input_sha256, input_bytes)}; every text hash is
    verified once while streaming the file."""
    import hashlib
    path = Path(path or INPUTS / f'period_{year}' / 'INPUTS.jsonl')
    recs = {}
    with path.open('rb') as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            r = json.loads(line.decode('utf-8'))
            if hashlib.sha256(r['input_text'].encode('utf-8')).hexdigest() != r['input_sha256']:
                raise ValueError('Input text hash mismatch')
            if r['key'] in recs:
                raise ValueError('Duplicate input key')
            recs[r['key']] = LazyRecord(key=r['key'], sample_rank=r['sample_rank'], input_sha256=r['input_sha256'],
                                        input_bytes=r['input_bytes'], _path=str(path), _offset=off)
    return recs


# ------------------------------------------------------------------------------------------------------------ audit
def audit_metas(path, year, max_n):
    """At most max_n full-metadata records, chosen by the TD4 audit hash rank (two streaming passes, bounded memory)."""
    ranks = []
    with Path(path).open('rb') as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                m = json.loads(line.decode('utf-8'))
                ranks.append((audit_rank(m['firm_id'], m['cutoff'], year), off))
    keep = sorted(off for _, off in sorted(ranks)[:max_n])
    out = []
    with Path(path).open('rb') as f:
        for off in keep:
            f.seek(off)
            out.append(json.loads(f.readline().decode('utf-8')))
    return out, len(ranks)


def audit_sample(year, period_dir=None, store_tag=None, max_n=AUDIT_N):
    """Independent TD3 scanner and the strict residual re-check over the period's audit sample (full metadata; at
    most max_n records by audit hash rank)."""
    import td3_runtime as R
    S3, B, td3_mask, td3_scan = _mods()
    d = Path(period_dir) if period_dir else INPUTS / f'period_{year}'
    set_period((year,))
    src, _ = sources((year,), store_path(store_tag or str(year)))
    B.install_tables(src)
    metas, n_meta = audit_metas(d / 'AUDIT_META.jsonl', year, max_n)
    english = B.load_english(src.english)
    ricmap = B.ric_map(src.ric_identity)
    hits, summary, adjacency = td3_scan.scan_build(metas, src.crsp_names, src.factset, ricmap, english)
    keys = ('names', 'common', 'tickers', 'tickers_extra', 'rics', 'embedded', 'fuzzy', 'products', 'code_prefixes',
            'near', 'unit_prefixes')
    problems = Counter()
    for m in metas:
        order = ['[FOCAL]'] + [s['token'] for s in m['shown']] + [x['token'] for x in m['links']]
        order += [t for t in m['aliases'] if t not in order]
        entries = [(t, {k: m['aliases'][t].get(k, {} if k == 'products' else []) for k in keys})
                   for t in order if t in m['aliases']]
        for x in m['articles']:
            remasker = B.Masker(B.article_entries(entries, x['owner_terms'], x.get('article_extras')), english)
            h2, b2, *_ = remasker.mask_article(x['headline_raw'], x['body_raw'])
            if (h2, b2) != (x['headline'], x['body']):
                problems['remask_differs'] += 1
            remasker.set_active_acronyms(x.get('active_acronyms'))
            if remasker.residual(x['headline']) + remasker.residual(x['body']):
                problems['strict_residual'] += 1
            text = x['headline'] + '\n' + x['body']
            if B.re.search(r'(?<![\d.,])(?:19|20)\d\d(?!\d)', text):
                problems['year_in_article_text'] += 1
            if td3_mask.URL_RESIDUAL.search(text):
                problems['url_in_article_text'] += 1
    w = R.JsonlWriter(d / f'AUDIT_SCAN_HITS_{R.utc().replace(":", "")}.jsonl', R.Redactor())
    for h in hits:
        w.write(h)
    w.close()
    out = {'year': year, 'audit_inputs': len(metas), 'audit_meta_records': n_meta,
           'articles': sum(len(m['articles']) for m in metas),
           'scan_summary': summary, 'scan_hits': len(hits), 'problems': dict(problems),
           'note': 'scan hits are candidates for the reviewer to adjudicate (TD3 practice); no LLM labels them'}
    R.new_json(d / f'AUDIT_SAMPLE_{year}.json', out)
    return out


# ------------------------------------------------------------------------------------------------------ pilot parity
def parity_pilot(limit=None, **_):
    """The 200 TD3 pilot firm-weeks rebuilt with the 2021-2022 parameters; texts must equal build_v7 byte for byte."""
    import td3_runtime as R
    pilot = pd.read_csv(T.TD3 / 'build_v7/PILOT_SAMPLE.csv')
    pilot['cutoff'] = pd.to_datetime(pilot.cutoff)
    if limit:
        pilot = pilot[pilot.sample_rank <= limit]
    want = {(a, pd.Timestamp(b)): r for a, b, r in zip(pilot.firm_id, pilot.cutoff, pilot.sample_rank)}

    def only(elig):
        return elig[[(a, pd.Timestamp(b)) in want for a, b in zip(elig.firm_id, elig.cutoff)]]

    def by_rank(elig):
        return elig.assign(_r=[want[(a, pd.Timestamp(b))] for a, b in zip(elig.firm_id, elig.cutoff)]) \
            .sort_values('_r').drop(columns='_r').reset_index(drop=True)
    tag = R.utc().replace(':', '').replace('-', '')
    out_dir = INPUTS / f'parity_pilot_{tag}'
    # chunks of 60 (four chunks): exercises per-chunk name sources and the alias cache across chunks
    rec = run_build((2021, 2022), out_dir, frame_filter=only, order=by_rank, audit_n=0, tag=f'pilot_{tag}', chunk=60)
    frozen = {r['key']: r for r in R.read_jsonl(T.TD3 / 'build_v7/INPUTS.jsonl')}
    mine = {r['key']: r for r in R.read_jsonl(out_dir / 'INPUTS.jsonl')}
    same = sum(1 for k, r in mine.items() if k in frozen and r['input_text'] == frozen[k]['input_text'])
    return {'pilot_firm_weeks': len(want), 'rebuilt': len(mine), 'byte_identical_texts': same,
            'missing': len(set(frozen) - set(mine)) if not limit else None,
            'equal': same == len(want) == len(mine), 'build_counts': rec['counts'], 'chunk_size': 60,
            'elapsed_seconds_build': rec['elapsed_seconds'], 'peak_working_set_bytes_build': rec['peak_working_set_bytes']}
