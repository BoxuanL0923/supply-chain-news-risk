"""TD3 pilot sample and frozen selector inputs (offline). v6 (TD3F): build_v7 inputs (sample unchanged since v1).

v6 (review TD3C5 D1): per firm 'unit_prefixes' (tickers and initialisms, td3_mask.unit_prefixes) for the
initialism-unit rule; verify counts its residual ('d1c5_initialism_unit') and checks the TD3C5 finding.

v5 (review TD3C4): code prefixes per firm (td3_mask.code_prefix_terms; D2 firm codes), near names from the CRSP and
FactSet name histories (near_name_index, near_names; D2), the raw truncated article text kept in the private metadata
so that verify re-masks every article and recomputes the sweep with its hidden words, glued tokens as sweep rows (D1),
and verify checks for glued tokens, house numbers before a street-word token, firm codes, token-hyphen-digit remnants,
near names and the TD3C4 findings.

v4 (review TD3C3): the frozen private mask table (td3_spec.MASK_TABLES) is loaded and hash-bound; firm-owned product,
unit and joint-product phrases and their acronyms are absorbed per article (E1, Masker.mask_article); FactSet-only
single-word variants must be distinctive (N4); the adjacency sweep (ADJACENCY_SWEEP.jsonl: every role token next to
a capitalised word, with the hidden raw text from a raw-versus-masked word alignment) and its row-by-row review; verify
re-checks the E1 product adjacency, the E2 date patterns, the TD3C3 findings and the sweep.


Reads hash-bound TD2 artifacts (features_v2, evidence_v2, news_scores, sources, risk_menu), the Reuters archive
and CRSP name files. Feature/evidence rows are decoded for 2021-2022 only; news and article rows for
2020-12-20..2022-12-31 only (rows after the seal that share a decoded Arrow batch are dropped by the date mask
before conversion and are never used or stored). No label, return, loss or TD1 manifest is read. The independent
residual scanner (td3_scan) also reads the FactSet FSB1 company names (start_ <= 2022-12-31); they never enter an
input. v3 (TD3BC D1): issuer siblings (CRSP CUSIP6), issuer-linked FactSet names and tickers, per-article issuer RICs
from the archive rics field, a replacement log per article (private metadata for the adjacency review), the adjacency
report and a verify step that re-checks every v3 rule.
"""
import csv
import difflib
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import td3_spec as S
import td3_scan
import td3_mask
from td3_mask import (Masker, name_variants, owner_terms, truncate, loose_scan, common_aliases, date_residuals,
                      normalize, word_initials, embedded_terms, fuzzy_words, firm_products, GENERIC, LEGAL,
                      ACRONYM_STOP, PLACE_WORDS, DATE_PATTERNS, URL_RESIDUAL, code_prefix_terms, STREET_WORDS,
                      FORM_WORDS, unit_prefixes)
from td3_runtime import guard, sha, sha_text, new_json, new_text, JsonlWriter, Redactor, peak, read_jsonl

FRAME_COLS = (['firm_id', 'cutoff', 'origin_session', 'variant', 'mapped_focal', 'has_news']
              + [f'{p}_{n}' for n in S.NAMES for p in ('C', 'missing', 'available')]
              + [f'H{h}_{c}' for h in S.HOPS for c in ('graph_partners', 'news_partners', 'selected_partners')])
EV_COLS = ['firm_id', 'cutoff', 'variant', 'hop', 'partner', 'nodes', 'intermediate_nodes', 'relationship_ids',
           'reporting_roles', 'article_ids', 'max_edge_available_at', 'max_news_available_at']
NEWS_COLS = ['alt_id', 'firms', 'first_months', 'available_at'] + S.NAMES


@dataclass(frozen=True)
class Sources:
    menu: Path
    features: dict
    evidence: dict
    news: dict
    graph: Path
    links: Path
    ric_identity: Path
    articles: Path
    crsp_names: tuple
    td2_bound: bool
    factset: Path = None       # FactSet FSB1 company names: masking name source and scanner name source
    english: Path = None       # frozen English word list (title-case ticker rule)
    tables: Path = None        # v4: frozen private mask table (common names, firm products, fixed phrases)


def real_sources():
    t = S.TD2
    return Sources(menu=S.TD2_MENU, features={y: t / f'features_v2/year_{y}.parquet' for y in S.YEARS},
                   evidence={y: t / f'evidence_v2/year_{y}.parquet' for y in S.YEARS},
                   news={y: t / f'news_scores/year_{y}.parquet' for y in S.NEWS_YEARS},
                   graph=t / 'sources/graph_october_sealed.parquet', links=t / 'sources/factset_links.parquet',
                   ric_identity=t / 'sources/reuters_identity_development_v1.json', articles=S.ARTICLE,
                   crsp_names=tuple(S.CRSP_NAME_FILES), td2_bound=True, factset=S.FACTSET_COMPANY,
                   english=S.ENGLISH_WORDS, tables=S.MASK_TABLES)


def install_tables(sources):
    """v4 (E1, N6): load the frozen private mask table into td3_mask (validated). A TD2-bound (real) build refuses
    without the file; a synthetic fixture without a table path runs with empty tables. Returns the table sha256."""
    path = sources.tables
    if path is None:
        if sources.td2_bound:
            raise PermissionError('The frozen mask table is required: ' + str(S.MASK_TABLES))
        td3_mask.clear_tables()
        return None
    path = Path(path)
    if not path.exists():
        raise PermissionError('Mask table missing: ' + str(path))
    digest = sha(path)
    td3_mask.set_tables(json.loads(path.read_text(encoding='utf-8')), sha256=digest, source=str(path))
    return digest


def td2_modules():
    """TD2 pure modules, imported read-only. The TD2 folder is appended last so stdlib and TD3 names win."""
    path = str(S.TD2_CODE)
    if path not in sys.path:
        sys.path.append(path)
    import engine
    import fast_builder
    for mod in (engine, fast_builder):
        if Path(mod.__file__).resolve().parent != S.TD2_CODE.resolve():
            raise ImportError('TD2 module resolved outside the TD2 code folder')
    return engine, fast_builder


def td2_code_hashes():
    return {str(S.TD2_CODE / n): sha(S.TD2_CODE / n) for n in S.TD2_CODE_USED}


def code_hashes(folder=S.CODE):
    return {str(p): sha(p) for p in sorted(Path(folder).glob('*.py'))}


def source_hashes(sources):
    paths = [sources.menu, *sources.features.values(), *sources.evidence.values(), *sources.news.values(),
             sources.graph, sources.links, sources.ric_identity]
    if sources.td2_bound:
        paths.append(S.TD2_COST)
    out = {str(Path(p)): sha(p) for p in paths}
    if sources.td2_bound:
        if sha(S.TD2_FREEZE) != S.TD2_FREEZE_SHA256:
            raise ValueError('TD2 FREEZE_v2.json changed')
        bound = json.loads(S.TD2_FREEZE.read_text(encoding='utf-8'))['artifact_sha256']
        for p, h in out.items():
            if bound.get(p) != h:
                raise ValueError('TD2 artifact differs from FREEZE_v2.json: ' + p)
    return out


def external_hashes(sources):
    out = {str(Path(p)): sha(p) for p in sources.crsp_names}
    out[str(Path(sources.articles))] = sha(sources.articles)
    for p in (sources.factset, sources.english, sources.tables):
        if p is not None and Path(p).exists():
            out[str(Path(p))] = sha(p)
    return out


def load_english(path):
    """Frozen English word list (lower case); empty when no path is given (synthetic fixtures)."""
    if path is None:
        return frozenset()
    with Path(path).open(encoding='utf-8') as f:
        return frozenset(w.strip().lower() for w in f if w.strip())


def load_cusips(paths, permnos):
    """8-digit CUSIPs (ncusip and cusip) of each PERMNO from CRSP stocknames rows with namedt <= 2022-12-31."""
    out = {int(p): set() for p in permnos}
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if 'ncusip' not in (reader.fieldnames or []) and 'cusip' not in (reader.fieldnames or []):
                continue
            for r in reader:
                day = (r.get('namedt') or '')[:10]
                if not day or day > S.LAST_DAY:
                    continue
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                if p in out:
                    out[p].update(c.strip()[:8] for c in (r.get('ncusip'), r.get('cusip')) if c and c.strip())
    return out


def issuer_siblings(paths, permnos):
    """v3 (TD3BC D1 item 1): other PERMNOs of the same issuer, i.e. sharing a 6-character issuer CUSIP in any CRSP row
    with namedt <= 2022-12-31 (share classes, preferred and debt securities, predecessors). Returns
    ({permno: sorted sibling PERMNOs}, {permno: set of issuer CUSIP6})."""
    by6, mine6 = {}, {int(p): set() for p in permnos}
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if 'ncusip' not in (reader.fieldnames or []) and 'cusip' not in (reader.fieldnames or []):
                continue
            for r in reader:
                day = (r.get('namedt') or '')[:10]
                if not day or day > S.LAST_DAY:
                    continue
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                for c in (r.get('ncusip'), r.get('cusip')):
                    c = (c or '').strip()
                    if len(c) >= 6:
                        by6.setdefault(c[:6], set()).add(p)
                        if p in mine6:
                            mine6[p].add(c[:6])
    sib = {}
    for p, cs in mine6.items():
        s = set()
        for c in cs:
            s |= by6.get(c, set())
        s.discard(p)
        sib[p] = sorted(s)
    return sib, mine6


US_TICKER_EXCHANGES = ('', 'NYS', 'NAS', 'ASE', 'OTC', 'USA', 'PNK', 'BAT', 'ARC', 'NMS', 'BATS')


def factset_aliases(path, cusip6):
    """v3 (TD3BC scan method): FactSet company ids linked by issuer CUSIP6 (any row with start_ <= 2022-12-31 whose
    cusip starts with a firm's CRSP CUSIP6), then every name and every US-listed ticker (2-5 capital letters) of those
    ids in rows with start_ <= 2022-12-31, including rows with a blank CUSIP. Rows after the seal are dropped in Arrow
    before conversion. Returns ({permno: names}, {permno: tickers}); empty when no FactSet file is configured."""
    names = {p: set() for p in cusip6}
    tickers = {p: set() for p in cusip6}
    if path is None:
        return names, tickers
    guard()
    by6 = {}
    for p, cs in cusip6.items():
        for c in cs:
            by6.setdefault(c, set()).add(p)
    if not by6:
        return names, tickers
    pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=65536)
    seal = pa.scalar(dt.date.fromisoformat(S.LAST_DAY), type=pa.date32())
    v6 = pa.array(sorted(by6), type=pa.string())
    ids = {}
    for b in pf.iter_batches(batch_size=65536, columns=['id', 'cusip', 'start_'], use_threads=False):
        b = b.filter(pc.less_equal(b.column('start_'), seal))
        t = b.filter(pc.is_in(pc.utf8_slice_codeunits(b.column('cusip'), 0, 6), value_set=v6))
        for i, c in zip(t.column('id').to_pylist(), t.column('cusip').to_pylist()):
            ids.setdefault(i, set()).update(by6.get(c[:6], ()))
    if not ids:
        return names, tickers
    vid = pa.array(sorted(ids), type=pa.string())
    for b in pf.iter_batches(batch_size=65536, columns=['id', 'name', 'ticker', 'ticker_exchange', 'start_'],
                             use_threads=False):
        b = b.filter(pc.less_equal(b.column('start_'), seal))
        t = b.filter(pc.is_in(b.column('id'), value_set=vid))
        for i, n, tk, ex in zip(*(t.column(k).to_pylist() for k in ('id', 'name', 'ticker', 'ticker_exchange'))):
            for p in ids[i]:
                if n and n.strip():
                    names[p].add(' '.join(n.split()))
                tk = (tk or '').strip()
                suffix = (ex or '').strip().rsplit('-', 1)[-1] if '-' in (ex or '') else ''
                if re.fullmatch(r'[A-Z]{2,5}', tk) and suffix in US_TICKER_EXCHANGES:
                    tickers[p].add(tk)
    return names, tickers


def factset_names(path, cusips):
    """v2 (kept for reference; v3 uses factset_aliases). FactSet company names (start_ <= 2022-12-31) by PERMNO via
    8-digit CUSIP; rows after the seal are dropped in Arrow before conversion."""
    out = {p: set() for p in cusips}
    if path is None:
        return out
    guard()
    by_cusip = {}
    for p, cs in cusips.items():
        for c in cs:
            by_cusip.setdefault(c, set()).add(p)
    if not by_cusip:
        return out
    pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=65536)
    value_set = pa.array(sorted(by_cusip), type=pa.string())
    seal = pa.scalar(dt.date.fromisoformat(S.LAST_DAY), type=pa.date32())
    for b in pf.iter_batches(batch_size=65536, columns=['cusip', 'name', 'start_'], use_threads=False):
        t = b.filter(pc.and_(pc.less_equal(b.column('start_'), seal), pc.is_in(b.column('cusip'), value_set=value_set)))
        for c, n in zip(t.column('cusip').to_pylist(), t.column('name').to_pylist()):
            if n and n.strip():
                for p in by_cusip[c]:
                    out[p].add(' '.join(n.split()))
    return out


def sealed_batches(path, columns, datecol, lo, hi, isin=None, batch_size=4096):
    """Date-sealed projection; refuses any upper bound after 2022-12-31."""
    guard()
    if hi > S.LAST_DAY or lo > hi:
        raise PermissionError('TD3 date seal: nothing after 2022-12-31 is read')
    pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=65536)
    typ = pf.schema_arrow.field(datecol).type

    def scalar(v):
        if pa.types.is_string(typ) or pa.types.is_large_string(typ):
            return pa.scalar(v, type=typ)
        if pa.types.is_date(typ):
            return pa.scalar(dt.date.fromisoformat(v), type=typ)
        return pa.scalar(pd.Timestamp(v), type=typ)
    ix = pf.schema_arrow.get_field_index(datecol)
    low, high = scalar(lo), scalar(hi)
    cols = list(dict.fromkeys([datecol] + list(columns)))
    for group in range(pf.num_row_groups):
        stat = pf.metadata.row_group(group).column(ix).statistics
        if stat is not None and stat.has_min_max and (pa.scalar(stat.max, type=typ).as_py() < low.as_py()
                                                      or pa.scalar(stat.min, type=typ).as_py() > high.as_py()):
            continue
        for raw in pf.iter_batches(batch_size=batch_size, row_groups=[group], columns=cols, use_threads=False):
            dates = raw.column(raw.schema.get_field_index(datecol))
            mask = pc.and_(pc.greater_equal(dates, low), pc.less_equal(dates, high))
            if isin is not None:
                mask = pc.and_(mask, pc.is_in(raw.column(raw.schema.get_field_index(isin[0])), value_set=isin[1]))
            safe = raw.filter(mask)
            if safe.num_rows:
                yield safe.to_pandas()
        guard()


def _concat(chunks, columns):
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)


def features_year(sources, year):
    if year not in S.YEARS:
        raise PermissionError('TD3 reads 2021-2022 features only')
    f = _concat(list(sealed_batches(sources.features[year], FRAME_COLS, 'origin_session', f'{year}-01-01',
                                    f'{year}-12-31')), FRAME_COLS)
    return f[f.variant == 'primary'].reset_index(drop=True)


def with_percentiles(frame):
    frame = frame.copy()
    for n in S.NAMES:
        v = frame['C_' + n].where(~frame['missing_' + n].astype(bool))
        g = v.groupby(frame.cutoff)
        frame['pct_' + n] = np.round(100 * (g.rank(method='average') - 0.5) / g.transform('count'))
        frame['universe_' + n] = g.transform('count')
    return frame


def eligible(frame):
    lo, hi = pd.Timestamp(S.FIRST_DAY), pd.Timestamp(S.LAST_DAY)
    o, c = frame.origin_session, frame.cutoff
    period = (o >= lo) & (o <= hi) & (c >= lo) & (c < hi + pd.Timedelta(days=1)) & (c > o)
    anychar = ~np.logical_and.reduce([frame['missing_' + n].to_numpy(bool) for n in S.NAMES])
    e = frame[period & frame.has_news.astype(bool) & frame.mapped_focal.astype(bool) & anychar]
    if (e.origin_session.dt.year > 2022).any() or (e.cutoff.dt.year > 2022).any() or \
            (e.origin_session.dt.year < 2021).any():
        raise PermissionError('Row outside 2021-2022 reached the eligible frame')
    if e.duplicated(['firm_id', 'cutoff']).any():
        raise ValueError('Duplicate firm-week')
    return e.reset_index(drop=True)


def _iso(ts):
    return pd.Timestamp(ts).isoformat()


def sample_key(firm_id, cutoff):
    return hashlib.sha256(f'TD3-pilot-v1|{S.SEED}|{firm_id}|{_iso(cutoff)}'.encode()).hexdigest()


def reask_key(firm_id, cutoff):
    return hashlib.sha256(f'TD3-reask-v1|{S.SEED}|{firm_id}|{_iso(cutoff)}'.encode()).hexdigest()


def record_key(firm_id, cutoff):
    return hashlib.sha256(f'TD3-key|{firm_id}|{_iso(cutoff)}'.encode()).hexdigest()[:16]


def load_evidence(sources, cands):
    out = []
    for year in sorted(set(cands.origin_session.dt.year)):
        if year not in S.YEARS:
            raise PermissionError('TD3 reads 2021-2022 evidence only')
        sub = cands[cands.origin_session.dt.year == year]
        keys = set(zip(sub.firm_id, sub.cutoff))
        for b in sealed_batches(sources.evidence[year], EV_COLS, 'origin_session', f'{year}-01-01', f'{year}-12-31',
                                isin=('firm_id', pa.array(sorted(set(sub.firm_id)), type=pa.string()))):
            b = b[b.variant == 'primary']
            b = b[[k in keys for k in zip(b.firm_id, b.cutoff)]]
            if len(b):
                out.append(b)
    return _concat(out, ['origin_session'] + EV_COLS)


def load_news(sources, ids):
    rows = {}
    value_set = pa.array(sorted(ids), type=pa.string())
    for year in S.NEWS_YEARS:
        lo = S.ARTICLE_LO if year == S.NEWS_YEARS[0] else f'{year}-01-01'
        for b in sealed_batches(sources.news[year], NEWS_COLS, 'date', lo, f'{year}-12-31', isin=('alt_id', value_set)):
            for r in b.to_dict('records'):
                if r['alt_id'] in rows:
                    raise ValueError('Duplicate news alt_id across partitions')
                rows[r['alt_id']] = {'date': pd.Timestamp(r['date']), 'firms': [int(x) for x in r['firms']],
                                     'first_months': [int(x) for x in r['first_months']],
                                     'available_at': pd.Timestamp(r['available_at']),
                                     'scores': {n: float(r[n]) for n in S.NAMES}}
    return rows


def verify_article(aid, partner, origin, cutoff, news):
    """None when eligible; otherwise the rejection reason (ARTICLE_RULE)."""
    r = news.get(aid)
    if r is None:
        return 'not_in_news'
    origin = pd.Timestamp(origin)
    if not origin - pd.Timedelta(days=S.LOOKBACK_DAYS) <= r['date'] < origin:
        return 'outside_window_or_origin_day'
    if not r['available_at'] < pd.Timestamp(cutoff):
        return 'available_not_before_cutoff'
    month = origin.year * 100 + origin.month
    if int(partner) not in [p for p, m in zip(r['firms'], r['first_months']) if m < month]:
        return 'partner_not_tagged_before_origin_month'
    return None


def ric_map(path):
    return {k: (int(v[0]), int(v[1])) for k, v in json.loads(Path(path).read_text(encoding='utf-8'))['mapping'].items()}


def fetch_texts(sources, wanted, ricmap, score):
    """Archive text of the record TD2 scored: first physical row with a mapped RIC on the news_scores day."""
    found = {}
    if wanted:
        value_set = pa.array(sorted(wanted), type=pa.string())
        for b in sealed_batches(sources.articles, ['alt_id', 'rics', 'headline', 'body_short'], 'date',
                                S.ARTICLE_LO, S.LAST_DAY, isin=('alt_id', value_set)):
            for r in b.to_dict('records'):
                aid = r['alt_id']
                if aid in found or r['date'][:10] != wanted[aid]['date'].strftime('%Y-%m-%d') or not r['rics']:
                    continue
                bindings = sorted({ricmap[t.strip()] for t in r['rics'].split(',') if t.strip() in ricmap})
                if not bindings:
                    continue
                found[aid] = {'headline': r['headline'] or '', 'body': r['body_short'] or '',
                              'firms': [p for p, m in bindings], 'first_months': [m for p, m in bindings],
                              'rics': sorted({t.strip() for t in r['rics'].split(',') if t.strip()})}
    status = {}
    for aid, rec in wanted.items():
        t = found.get(aid)
        if t is None:
            status[aid] = 'text_not_found'
        elif t['firms'] != rec['firms'] or t['first_months'] != rec['first_months']:
            status[aid] = 'text_firms_mismatch'
        else:
            sc = score(t['headline'][:S.HEADLINE_CHARS] + t['body'][:S.BODY_CHARS])
            status[aid] = None if all(abs(sc[n] - rec['scores'][n]) <= 1e-12 for n in S.NAMES) else 'text_score_mismatch'
    return found, status


def priority(cutoff, partner):
    return hashlib.sha256(f'{pd.Timestamp(cutoff)}|{int(partner)}'.encode()).hexdigest()


def select_sample(sources, elig, n, ricmap, engine):
    order = elig.assign(_k=[sample_key(a, b) for a, b in zip(elig.firm_id, elig.cutoff)]).sort_values('_k')
    accepted, stats = [], {'examined': 0, 'chunks': 0, 'rejected': Counter(), 'article_rejects': Counter(),
                           'text_status': Counter(), 'partners_without_article': 0, 'partner_order_equal_td2': True}
    pos = 0
    while len(accepted) < n and pos < len(order):
        chunk = order.iloc[pos:pos + S.CANDIDATE_BATCH]
        pos += len(chunk)
        stats['chunks'] += 1
        ev = load_evidence(sources, chunk)
        news = load_news(sources, {a for lst in ev.article_ids for a in lst})
        verdict = {}
        for w in ev.itertuples(index=False):
            for aid in w.article_ids:
                verdict[(w.firm_id, w.cutoff, int(w.partner), aid)] = verify_article(aid, w.partner, w.origin_session,
                                                                                    w.cutoff, news)
        wanted = {k[3]: news[k[3]] for k, v in verdict.items() if v is None}
        texts, tstatus = fetch_texts(sources, wanted, ricmap, engine.score)
        groups = {k: g for k, g in ev.groupby(['firm_id', 'cutoff'], sort=False)}
        for cand in chunk.itertuples(index=False):
            stats['examined'] += 1
            g = groups.get((cand.firm_id, cand.cutoff))
            if g is None or not len(g):
                stats['rejected']['no_evidence_rows'] += 1
                continue
            for h in S.HOPS:
                if int((g.hop == h).sum()) != int(getattr(cand, f'H{h}_selected_partners')):
                    raise ValueError('Evidence rows differ from TD2 selected-partner counts')
            g = g.assign(_p=[priority(cand.cutoff, p) for p in g.partner])
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
                continue
            stats['partners_without_article'] += sum(p['article'] is None for p in partners)
            accepted.append({'cand': cand, 'partners': partners, 'news': news, 'texts': texts})
            if len(accepted) == n:
                break
        stats['text_status'].update('ok' if v is None else v for v in tstatus.values())
    return accepted, stats


def edge_maps(sources, accepted, fast_builder):
    rel_ids = sorted({rid for a in accepted for p in a['partners'] for rid in p['row'].relationship_ids})
    graph = _concat(list(sealed_batches(sources.graph, ['id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'], 'date',
                                        '1900-01-01', S.LAST_DAY, isin=('id', pa.array(rel_ids, type=pa.string())))),
                    ['date', 'id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum']).rename(columns={'date': 'start'})
    links = _concat(list(sealed_batches(sources.links, ['entity', 'permno', 'end'], 'date', '1900-01-01', S.LAST_DAY)),
                    ['date', 'entity', 'permno', 'end']).rename(columns={'date': 'start'})
    maps = {}
    for origin in sorted({pd.Timestamp(a['cand'].origin_session) for a in accepted}):
        maps[origin] = {e['id']: e for e in fast_builder.mapped_edges(graph, links, origin)}
    return maps, links


def path_relations(nodes, rel_ids, edges, cutoff, with_times=False):
    """rels[k] = role of node k+1 relative to node k. CUSTOMER rows: source supplier, target customer; SUPPLIER rows:
    source customer, target supplier (FSB3 role convention). with_times: also the edges' available_at values."""
    rels, times = [], []
    for u, v, rid in zip(nodes[:-1], nodes[1:], rel_ids):
        e = edges.get(rid)
        if e is None or {int(e['a']), int(e['b'])} != {int(u), int(v)}:
            raise ValueError('Witness edge not reproduced at origin')
        if pd.Timestamp(e['available_at']) >= pd.Timestamp(cutoff):
            raise ValueError('Edge availability not before cutoff')
        if e['typ'] not in ('CUSTOMER', 'SUPPLIER'):
            raise ValueError('Unexpected relationship type')
        supplier, customer = (int(e['a']), int(e['b'])) if e['typ'] == 'CUSTOMER' else (int(e['b']), int(e['a']))
        rels.append('SUPPLIER' if int(v) == supplier else 'CUSTOMER')
        times.append(pd.Timestamp(e['available_at']))
    return (rels, times) if with_times else rels


def load_names(paths, permnos):
    out = {int(p): [] for p in permnos}
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                day = (r.get('namedt') or '')[:10]
                if not day or day > S.LAST_DAY:
                    continue  # rows dated after the seal are discarded on read
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                if p in out:
                    out[p].append((day, (r.get('comnam') or '').strip(), (r.get('ticker') or '').strip()))
    return out


def _not_english(word, english):
    w = word.lower()
    return w not in english and not (w.endswith('s') and w[:-1] in english)


def name_anchors(comnams):
    """v5 (TD3C4 D2): first significant name word of each CRSP name (letters of an acronym such as 'Q.R.T.'), >= 3
    letters, not a generic or place word: the words whose near forms are near names."""
    out = set()
    for c in comnams:
        words = normalize(c)
        if words and words[0] == 'THE':
            words = words[1:]
        if words:
            a = words[0].replace('.', '').replace('&', '')
            if len(a) >= 3 and a.isalpha() and a.lower() not in GENERIC and a.lower() not in PLACE_WORDS:
                out.add(a)
    return out


def _near_entry(name):
    """(first word letters, normalized two-word phrase) of a name with >= 2 words whose second word is a word of >= 3
    letters that is not a corporate-form word; None otherwise."""
    words = normalize(name)
    if words and words[0] == 'THE':
        words = words[1:]
    if len(words) < 2:
        return None
    first = words[0].replace('.', '').replace('&', '')
    second = words[1]
    if not first.isalpha() or len(first) < 2 or not second.isalpha() or len(second) < 3 or second in FORM_WORDS:
        return None
    return first, words[0] + ' ' + second


def near_name_index(sources, anchors):
    """v5 (TD3C4 D2): every CRSP comnam (both files, namedt <= 2022-12-31) and FactSet company name (start_ <=
    2022-12-31, any company id) whose first word is an anchor minus its last one letter (>= 2 left) or two letters
    (>= 3 left), or an anchor plus one or two letters. Returns {first word: {phrase: source}}."""
    minus = {a[:-k] for a in anchors for k in (1, 2) if len(a) - k >= (2 if k == 1 else 3)}
    wanted2 = {w[:2] for w in minus} | {a[:2] for a in anchors}
    index = {}

    def consider(name, source):
        u = name.upper()
        if u.startswith('THE '):
            u = u[4:]
        if re.sub(r'[^A-Z]', '', u[:8])[:2] not in wanted2:
            return
        e = _near_entry(name)
        if e is None:
            return
        f, phrase = e
        if f in minus or f[:-1] in anchors or (len(f) >= 5 and f[:-2] in anchors):
            index.setdefault(f, {}).setdefault(phrase, source)
    for path in sources.crsp_names:
        with Path(path).open(encoding='utf-8-sig', newline='') as fh:
            for r in csv.DictReader(fh):
                day = (r.get('namedt') or '')[:10]
                if day and day <= S.LAST_DAY and r.get('comnam'):
                    consider(r['comnam'], 'crsp:' + r['comnam'].strip())
    if sources.factset is not None:
        guard()
        pf = pq.ParquetFile(sources.factset, pre_buffer=False, buffer_size=65536)
        seal = pa.scalar(dt.date.fromisoformat(S.LAST_DAY), type=pa.date32())
        for b in pf.iter_batches(batch_size=65536, columns=['name', 'start_'], use_threads=False):
            b = b.filter(pc.less_equal(b.column('start_'), seal))
            for n in b.column('name').to_pylist():
                if n:
                    consider(n, 'factset:' + ' '.join(n.split()))
    plus = {}
    for f in index:
        for k in (1, 2):
            if len(f) - k >= 3:
                plus.setdefault(f[:-k], []).append(f)
    return {'by_first': index, 'plus': {k: sorted(v) for k, v in plus.items()}}


def near_names(anchors, index, own_variants, english):
    """v5 (TD3C4 D2): near names of one firm: phrases of the index whose first word F is an anchor minus one or two
    trailing letters or plus one or two letters, F not an English word (>= 3 letters), generic, place or common
    acronym, and the phrase not already one of the firm's own name variants. Returns [{'phrase', 'anchor',
    'relation', 'source'}]."""
    if not index:
        return []
    by_first, plus = index['by_first'], index['plus']
    own = {v.upper() for v in own_variants}
    out = {}
    for a in sorted(anchors):
        cands = [(a[:-k], f'minus_{k}') for k in (1, 2) if len(a) - k >= (2 if k == 1 else 3)]
        cands += [(f, 'plus_' + str(len(f) - len(a))) for f in plus.get(a, ())]
        for f, rel in cands:
            if (len(f) >= 3 and td3_mask._is_english(f, english)) or f.lower() in GENERIC or \
                    f.lower() in PLACE_WORDS or f in ACRONYM_STOP:
                continue
            for phrase, src in sorted(by_first.get(f, {}).items()):
                if phrase.upper() in own or phrase in out:
                    continue
                out[phrase] = {'phrase': phrase, 'anchor': a, 'relation': rel, 'source': src}
    return [out[k] for k in sorted(out)]


def aliases_for(permno, origin, ns):
    """Masking aliases of one firm. ns: name sources from name_sources() ('names', 'fsnames', 'fstickers', 'english',
    'rics_by_permno', 'siblings').
    v2 (C1 item 6): every CRSP name/ticker row with namedt <= 2022-12-31 (load_names drops later rows), including
    rows dated after the origin, every RIC with first month <= 2022-12, and FactSet names (start_ <= 2022-12-31).
    Masking only removes text. Extra single-token variants: a first name word of >= 3 letters that equals one of the
    firm's tickers (ZEQ THERAPEUTICS / ZEQ); the next word after a leading initialism of <= 2 letters when it is not an
    English word (H Q BRINWOOD -> BRINWOOD, not F 7 NETWORKS -> NETWORKS); and the initials of the name's words (3-6
    letters, not English, not ACRONYM_STOP: all-caps short names).
    v3 (TD3BC D1): issuer siblings (other PERMNOs sharing the issuer CUSIP6) give their names, RICs and tickers; FactSet
    names and tickers come from the issuer-linked FactSet ids; sibling and FactSet-only tickers are 'tickers_extra'
    (restricted ticker rules); 'embedded' and 'fuzzy' terms feed the embedded-name and misspelling checks. Place words
    are never single-token variants."""
    names, rics_by_permno = ns['names'], ns['rics_by_permno']
    english = ns.get('english', frozenset())
    permno = int(permno)
    day = pd.Timestamp(origin).strftime('%Y-%m-%d')
    rows = [x for x in names.get(permno, []) if x[0] <= S.LAST_DAY]
    sibs = [int(q) for q in ns.get('siblings', {}).get(permno, [])]
    srows = [x for q in sibs for x in names.get(q, []) if x[0] <= S.LAST_DAY]
    comnams = sorted({c for _, c, _ in rows if c})
    sib_comnams = sorted({c for _, c, _ in srows if c} - set(comnams))
    fs = sorted(ns.get('fsnames', {}).get(permno, set()))
    rics = sorted({r for r, m in rics_by_permno.get(permno, []) if m <= S.LAST_MONTH})
    sib_rics = sorted({r for q in sibs for r, m in rics_by_permno.get(q, []) if m <= S.LAST_MONTH} - set(rics))
    tickers = {t for _, _, t in rows if t} | {r.split('.')[0] for r in rics if r.split('.')[0]}
    extra = ({t for _, _, t in srows if t} | {r.split('.')[0] for r in sib_rics if r.split('.')[0]} |
             set(ns.get('fstickers', {}).get(permno, set()))) - tickers

    def variants_of(c):
        out = set(name_variants(c))
        words = normalize(c)
        if words and words[0] == 'THE':
            words = words[1:]
        first = words[0] if words else ''
        if len(first) >= 3 and first.isalpha() and first in tickers and first.lower() not in GENERIC and \
                first.lower() not in PLACE_WORDS:
            out.add(first)
        if len(words) >= 2 and len(first.replace('.', '').replace('&', '')) <= 2 and not first.isalpha():
            nxt = words[1]
            if len(nxt) >= 4 and nxt.isalpha() and nxt.lower() not in GENERIC and nxt not in LEGAL and \
                    nxt.lower() not in PLACE_WORDS and _not_english(nxt, english):
                out.add(nxt)
        init = word_initials(c)
        if 3 <= len(init) <= 6 and init not in ACRONYM_STOP and _not_english(init, english) and \
                init.lower() not in GENERIC:
            out.add(''.join(ch + '.' for ch in init))
        return out
    variants = set()
    for c in comnams + sib_comnams:
        variants |= variants_of(c)
    fs_only = set()
    for c in fs:
        fs_only |= variants_of(c)
    fs_only -= variants
    # v4 (TD3C3 N4): a single-word variant that comes only from a FactSet-linked name (a subsidiary or other entity of
    # the issuer) must be distinctive: not an English word or inflection, generic word, place word, legal word or given
    # name ("GOLDEN" from "Golden Zorbex Holdings LLC" is dropped; "GOLDEN ZORBEX" stays). Acronym forms are kept.
    dropped_fs = sorted(v for v in fs_only if ' ' not in v and v.isalpha() and
                        not td3_mask._distinctive(v, english, 4))
    variants |= fs_only - set(dropped_fs)
    after = sorted({c for d, c, _ in rows if c and d > day})
    al = {'names': sorted(variants), 'common': sorted(common_aliases(comnams + sib_comnams)),
          'tickers': sorted(tickers), 'tickers_extra': sorted(extra), 'rics': sorted(set(rics) | set(sib_rics))}
    al['embedded'] = [list(x) for x in embedded_terms(al, english)]
    al['fuzzy'] = fuzzy_words(al, english)
    al['products'] = firm_products(comnams + sib_comnams)      # v4 (E1)
    al['factset_single_words_dropped'] = dropped_fs              # v4 (N4), metadata only
    # v5 (TD3C4 D2): firm-code prefixes (tickers, initialisms, leading-letter abbreviations, table) and near names
    codes = code_prefix_terms(comnams + sib_comnams, fs, sorted(tickers | extra), english)
    al['code_prefixes'] = sorted(codes)
    al['code_prefix_rules'] = codes                              # metadata only
    al['unit_prefixes'] = unit_prefixes(codes)                   # v6 (TD3C5 D1)
    near = near_names(name_anchors(comnams + sib_comnams), ns.get('near_index'),
                      set(variants) | set(al['common']), english)
    al['near'] = [x['phrase'] for x in near]
    al['near_names'] = near                                      # metadata only
    return al, comnams, after, fs, sib_comnams


HALT_HEADLINE = re.compile(r'^(?:NYSE|NASDAQ|NYSE AMERICAN|NYSE ARCA|NYSE MKT|AMEX)\b[^\n]*?\b(?:IMBALANCE|HALT|'
                           r'PAUSE|RESUM\w*)\b', re.I)


def article_extras(rics, headline, toks, entries):
    """v3 (TD3BC D1 item 1): per-article issuer RICs. (a) A RIC in the archive rics field that is not already a masked
    firm's RIC and whose root starts with a masked firm's ticker or RIC root (>= 2 characters; at most 3 more
    characters, e.g. a preferred-share suffix) takes that firm's token (longest prefix wins). (b) Every <RIC> in an
    exchange order-imbalance or trading-halt headline takes the token of the firm the article was chosen for, unless it
    is another masked firm's RIC. Returns {token: {'rics': [...], 'tickers_extra': [...]}}."""
    known = {r: tok for tok, al in entries for r in al.get('rics', ())}
    roots = []
    for tok, al in entries:
        for t in set(al.get('tickers', ())) | set(al.get('tickers_extra', ())):
            if len(t) >= 2:
                roots.append((t, tok))
    extras = {}

    def add(tok, ric):
        e = extras.setdefault(tok, {'rics': set(), 'tickers_extra': set()})
        e['rics'].add(ric)
        root = ric.split('.')[0]
        if re.fullmatch(r'[A-Z]{2,6}', root):
            e['tickers_extra'].add(root)
    for r in rics or ():
        if r in known:
            continue
        root = r.split('.')[0]
        best = max(((t, tok) for t, tok in roots if root.startswith(t) and 0 < len(root) - len(t) <= 3 and
                    (len(t) >= 3 or re.fullmatch(r'_?[a-z]{1,2}', root[len(t):]))),
                   key=lambda x: len(x[0]), default=None)
        if best is not None:
            add(best[1], r)
    if HALT_HEADLINE.match(headline or '') and toks:
        for m in re.finditer(r'<\s*([A-Za-z0-9._]{1,16})\s*>', headline):
            if known.get(m.group(1), toks[0]) == toks[0]:
                add(toks[0], m.group(1))
    return {tok: {k: sorted(v) for k, v in e.items()} for tok, e in extras.items()}


def assign_tokens(focal, partners, rels):
    tokens, counters, shown = {int(focal): '[FOCAL]'}, Counter(), []
    for p in partners:
        w = p['row']
        role = '_OF_'.join(reversed(rels[(int(w.hop), int(w.partner))]))
        counters[role] += 1
        token = f'[{role}_{counters[role]}]'
        if int(w.partner) in tokens:
            raise ValueError('Firm shown twice')
        tokens[int(w.partner)] = token
        words = [r.lower() for r in reversed(rels[(int(w.hop), int(w.partner))])]
        shown.append({'hop': int(w.hop), 'permno': int(w.partner), 'token': token, 'role': role,
                      'description': ' of a '.join(words) + ' of [FOCAL]', 'nodes': [int(x) for x in w.nodes],
                      'article': p['article']})
    for p in partners:
        for node in p['row'].intermediate_nodes:
            if int(node) not in tokens:
                counters['LINK'] += 1
                tokens[int(node)] = f'[LINK_{counters["LINK"]}]'
    for s in shown:
        s['path'] = [tokens[n] for n in s['nodes']]
    return tokens, shown


def article_entries(entries, owners, extras=None):
    """Masker entries for one article: owner tokens also carry their owner-rule terms; v3: per-article issuer RICs and
    their roots (article_extras)."""
    extras = extras or {}
    out = []
    for tok, al in entries:
        e = extras.get(tok, {})
        out.append((tok, {**al, 'names': sorted(set(al['names']) | set(owners.get(tok, ()))),
                          'rics': sorted(set(al.get('rics', ())) | set(e.get('rics', ()))),
                          'tickers_extra': sorted(set(al.get('tickers_extra', ())) | set(e.get('tickers_extra', ())))}))
    return out


def _ny_date(ts):
    return pd.Timestamp(ts).tz_localize('UTC').tz_convert('America/New_York').date()


def _fmt(name, v):
    return f'{v:.2f}' if name == 'size' else f'{v:.3f}'


def _days(n):
    return f'{n} day' if n == 1 else f'{n} days'


def _firms(n):
    return f'{n} linked firm' if n == 1 else f'{n} linked firms'


def render(menu, cand, shown, articles):
    origin = pd.Timestamp(cand.origin_session)
    lines = [f'FROZEN INPUT ({S.INPUT_VERSION})',
             'All information below was available before the forecast cutoff. The forecast date is the last trading '
             'session before the cutoff; the risk horizon is the next five trading sessions. Calendar dates and years '
             'are withheld; ages are calendar days before the forecast date. Names and tickers of the focal firm and '
             'of every firm shown in the supply-chain graph are replaced by role tokens; [NEWS_AGENCY] stands for the '
             'news agency that published the articles.',
             '', 'RISK MENU (firm characteristics)']
    lines += [f"{i}. {m['name']}: {m['economic_meaning']}. Risk channel: {m['risk_channel']}."
              for i, m in enumerate(menu, 1)]
    lines += ['', 'FOCAL FIRM [FOCAL]: CHARACTERISTICS',
              'Latest value available before the cutoff, with its percentile among the study-universe firms that have '
              'a value at this cutoff (0 = lowest, 100 = highest).']
    for n in S.NAMES:
        if bool(getattr(cand, 'missing_' + n)):
            lines.append(f'- {n}: missing (no value available before the cutoff)')
        else:
            age = max(0, (origin.date() - _ny_date(getattr(cand, 'available_' + n))).days)
            lines.append(f"- {n}: {_fmt(n, float(getattr(cand, 'C_' + n)))} (percentile "
                         f"{int(getattr(cand, 'pct_' + n))} of {int(getattr(cand, 'universe_' + n))} firms; value dated "
                         f'{_days(age)} before the forecast date)')
    lines += ['', 'SUPPLY-CHAIN GRAPH STATE',
              'Reported customer and supplier relationships active before the cutoff. A path is a reported business '
              'link, not a verified physical product flow. Up to five firms with recent news are shown per exact hop. '
              'Link age: days from the date of the newest relationship record on the path (vendor record date, not a '
              'verified publication date) to the forecast date.']
    for h in S.HOPS:
        items = []
        for s in (x for x in shown if x['hop'] == h):
            text = f"{s['token']} ({s['description']}"
            if h > 1:
                text += '; path ' + ' -> '.join(s['path'])
            text += f"; link age {_days(s['edge_age_days'])}"
            if s['article'] is None:
                text += '; no eligible article'
            items.append(text + ')')
        lines.append(f"Exact hop {h}: {_firms(int(getattr(cand, f'H{h}_graph_partners')))}, "
                     f"{int(getattr(cand, f'H{h}_news_partners'))} with news in the 7 days before the forecast date. "
                     f"Shown: {'; '.join(items) if items else 'none'}.")
    lines.append('Token key: [SUPPLIER_k] supplies [FOCAL]; [CUSTOMER_k] buys from [FOCAL]; a token X_OF_Y_k names a '
                 'firm that is the X of a Y of [FOCAL], read from the left; [LINK_k] is an intermediate firm on a path '
                 'that is not itself shown.')
    lines += ['', 'PARTNER NEWS',
              'One article per shown firm: its most recent eligible article, published 1 to 7 days before the '
              f'forecast date. Articles: {len(articles)}.']
    for i, a in enumerate(articles, 1):
        lines += ['', f"Article {i} | firm: {', '.join(a['tokens'])} | published {_days(a['days_before'])} before "
                      'the forecast date', f"Headline: {a['headline']}", f"Text: {a['body']}"]
    return '\n'.join(lines) + '\n'


ROLE_TOKEN = re.compile(r'\[(?:FOCAL|LINK_\d+|(?:SUPPLIER|CUSTOMER)(?:_OF_(?:SUPPLIER|CUSTOMER))*_\d+)\]')
_ALIGN_TOK = re.compile(r"\[[A-Z0-9_]+\]|[^\W_]+(?:['’][^\W_]+)*|[^\w\s]")
_SWEEP_WORD = r"[A-Za-z0-9À-ž][\w'’&.]*"
_SWEEP_SEP_AFTER = re.compile(r"(?:['’]s)?(?:[ \t]*[®™])*(?:[ \t]+|[\-–/])")
_SWEEP_SEP_BEFORE = re.compile(r"(?:[ \t]+|[\-–/])$")


def _firm_class(token):
    return 'focal' if token == '[FOCAL]' else 'link' if token.startswith('[LINK_') else 'shown'


def _cap(word):
    """Title-case or upper-case word, a number, or a camel-case word with an upper-case letter inside (Zorbphone,
    ZORB, 720, iZorb, oneZorb)."""
    return bool(word) and (word[0].isupper() or word[0].isdigit() or any(c.isupper() for c in word[1:]))


def _is_and(word):
    return word.lower() in ('and', '&')


_GLUED_AFTER = re.compile(r'[^\W_]+')
_GLUED_BEFORE = re.compile(r'[^\W_]+$')


def sweep_side(text, start, end):
    """v4 (E1 item 4): the neighbour words of a role token at text[start:end]. Returns (after_words, before_words),
    each a list of up to three words or None when that side is not a sweep row: the token is immediately followed
    (optional possessive or signs, then a space, hyphen, dash or slash) or preceded (a space, hyphen, dash or slash)
    by a word starting with an upper-case letter or a digit (a number neighbour such as a house number counts), or by
    and/& plus such a word. v5 (TD3C4 D1): a token glued to letters or digits (no separator) is always a row on that
    side, with the glued word first."""
    words_rx = r'\[[A-Z0-9_]+\]|&|' + _SWEEP_WORD
    after = None
    g = _GLUED_AFTER.match(text, end)
    m = _SWEEP_SEP_AFTER.match(text, end)
    if g:                                        # v5: glued to a letter or digit
        after = [g.group(0)] + re.findall(words_rx, text[g.end():g.end() + 120])[:2]
    elif m:
        rest = text[m.end():m.end() + 120]
        first = re.match(r'&|' + _SWEEP_WORD, rest)
        if first:
            w = first.group(0)
            second = re.match(r'[ \t]+(' + _SWEEP_WORD + ')', rest[first.end():])
            if (not _is_and(w) and _cap(w)) or (_is_and(w) and second and _cap(second.group(1))):
                after = re.findall(words_rx, rest)[:3]
    before = None
    left = text[max(0, start - 120):start]
    g = _GLUED_BEFORE.search(left)
    m = _SWEEP_SEP_BEFORE.search(left)
    if g:                                        # v5: glued to a letter or digit
        before = re.findall(words_rx, left[:g.start()])[-2:] + [g.group(0)]
    elif m:
        lw = left[:m.start()]
        w1 = re.search('(&|' + _SWEEP_WORD + ')$', lw)
        if w1:
            w = w1.group(1)
            prev = re.search('(' + _SWEEP_WORD + r')[ \t]+$', lw[:w1.start()])
            if (not _is_and(w) and _cap(w)) or (_is_and(w) and prev and _cap(prev.group(1))):
                before = re.findall(words_rx, lw)[-3:]
    return after, before


def glued_sides(text, start, end):
    """v5 (TD3C4 D1): which sides of the token at text[start:end] are glued to a letter or digit."""
    return {'after': bool(_GLUED_AFTER.match(text, end)),
            'before': bool(_GLUED_BEFORE.search(text[max(0, start - 60):start]))}


def _align_hidden(raw, masked):
    """{start offset of a role token in masked: (hidden raw text, shared)} from a word-level difflib alignment."""
    rt = [(m.group(0), m.start(), m.end()) for m in _ALIGN_TOK.finditer(raw)]
    mt = [(m.group(0), m.start(), m.end()) for m in _ALIGN_TOK.finditer(masked)]
    sm = difflib.SequenceMatcher(None, [t[0] for t in rt], [t[0] for t in mt], autojunk=False)
    out = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        toks = [j for j in range(j1, j2) if ROLE_TOKEN.fullmatch(mt[j][0])]
        if not toks:
            continue
        hidden = raw[rt[i1][1]:rt[i2 - 1][2]] if i2 > i1 else ''
        shared = len(toks) > 1 or (j2 - j1) > len(toks)
        for j in toks:
            out[mt[j][1]] = (hidden, shared)
    return out


def sweep_positions(masked):
    """(start, token, side) of every sweep row of a masked text (no raw text needed; verify recomputes these)."""
    out = []
    for m in ROLE_TOKEN.finditer(masked):
        after, before = sweep_side(masked, m.start(), m.end())
        if after is not None or before is not None:
            out.append((m.start(), m.group(0), 'both' if after and before else 'after' if after else 'before'))
    return out


def sweep_rows(raw, masked, field):
    """v4 (E1 item 4): adjacency-sweep rows of one article field with the hidden raw text of each token."""
    rows = []
    hidden = None
    for start, token, side in sweep_positions(masked):
        if hidden is None:
            hidden = _align_hidden(raw, masked)
        end = start + len(token)
        after, before = sweep_side(masked, start, end)
        h, shared = hidden.get(start, ('', True))
        gl = glued_sides(masked, start, end)
        rows.append({'field': field, 'start': start, 'token': token, 'side': side,
                     'before_words': before or [], 'after_words': after or [], 'hidden': h, 'hidden_shared': shared,
                     'glued': [k for k in ('before', 'after') if gl[k]],                     # v5 (D1)
                     'number_neighbour': [k for k, ws in (('before', before), ('after', after))
                                          if ws and re.fullmatch(r'\d[\d,.]*[A-Za-z]{0,2}', ws[-1 if k == 'before' else 0])],
                     'context': masked[max(0, start - 70):end + 70]})
    return rows


def sweep_text_problems(raw, masked, field):
    """v5 (TD3C4 D1): (sweep rows recomputed from the raw text, problems) of one article field: a 1-6 digit house
    number right before a token whose hidden raw text ends in a street word ('d1_house_number_street'), and a token
    glued to a letter or digit ('d1_glued_sweep_row')."""
    problems = Counter()
    rows = sweep_rows(raw, masked, field)
    for row in rows:
        last = re.findall(r'[A-Za-z]+', row['hidden'])
        if row['before_words'] and re.fullmatch(r'\d{1,6}[A-Za-z]?', row['before_words'][-1]) and last and \
                last[-1].upper() in STREET_WORDS:
            problems['d1_house_number_street'] += 1
        if row['glued']:
            problems['d1_glued_sweep_row'] += 1
    return rows, problems


def sweep_row_id(r):
    return hashlib.sha256(f"{r['rank']}|{r['alt_id']}|{r['field']}|{r['start']}|{r['token']}".encode()).hexdigest()[:20]


def build_inputs(menu, accepted, maps, links, ns, sweep=None):
    idmap = {hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]: int(p) for p in pd.unique(links.permno)}
    records, metas = [], []
    sweep = [] if sweep is None else sweep
    for rank, a in enumerate(accepted, 1):
        cand, origin, cutoff = a['cand'], pd.Timestamp(a['cand'].origin_session), pd.Timestamp(a['cand'].cutoff)
        focal = idmap.get(cand.firm_id)
        if focal is None:
            raise ValueError('Focal firm not in the FactSet identity links')
        edges = maps[origin]
        rels, ages, edge_checks = {}, {}, Counter()
        for p in a['partners']:
            w = p['row']
            if int(w.nodes[0]) != focal or int(w.nodes[-1]) != int(w.partner) or len(w.nodes) != int(w.hop) + 1:
                raise ValueError('Witness path does not start at the focal firm')
            rel, times = path_relations(list(w.nodes), list(w.relationship_ids), edges, cutoff, with_times=True)
            rels[(int(w.hop), int(w.partner))] = rel
            newest = pd.Timestamp(w.max_edge_available_at)
            edge_checks['equal' if max(times) == newest else 'differs'] += 1
            if newest >= cutoff:
                raise ValueError('Edge availability not before cutoff')
            ages[(int(w.hop), int(w.partner))] = max(0, (origin.date() - _ny_date(newest)).days)
        tokens, shown = assign_tokens(focal, a['partners'], rels)
        for s in shown:
            s['edge_age_days'] = ages[(s['hop'], s['permno'])]
        entries, comnams, firm_aliases = [], {}, {}
        for permno, token in tokens.items():
            al, cn, after, fs, scn = aliases_for(permno, origin, ns)
            entries.append((token, al))
            comnams[token] = cn
            firm_aliases[token] = {'permno': permno, **al, 'comnams': cn, 'comnams_after_origin': after,
                                   'factset_names': fs, 'sibling_permnos': ns.get('siblings', {}).get(permno, []),
                                   'sibling_comnams': scn}
        english = ns.get('english', frozenset())
        collisions = Masker(entries, english).collisions
        by_article = {}
        for s in shown:
            if s['article'] is not None:
                by_article.setdefault(s['article'], []).append(s['token'])
        articles, residual, loose, replaced, date_left = [], 0, [], Counter(), Counter()
        for aid, toks in by_article.items():
            t, rec = a['texts'][aid], a['news'][aid]
            owners = {tok: sorted(owner_terms(comnams[tok])) for tok in toks if owner_terms(comnams[tok])}
            extras = article_extras(t.get('rics', []), t['headline'], toks, entries)
            masker = Masker(article_entries(entries, owners, extras), english)
            head_raw, body_raw = truncate(t['headline'], S.HEADLINE_CHARS), truncate(t['body'], S.BODY_CHARS)
            head, body, c1, c2, l1, l2 = masker.mask_article(head_raw, body_raw)      # v4 (E1 acronyms)
            log = [{'field': 'headline', **e} for e in l1] + [{'field': 'body', **e} for e in l2]
            replaced.update(c1)
            replaced.update(c2)
            hits = masker.residual(head) + masker.residual(body)
            residual += len(hits)
            date_left.update(date_residuals(head) + date_residuals(body))
            loose += loose_scan(head + '\n' + body, entries, comnams)
            for field, raw, masked in (('headline', head_raw, head), ('body', body_raw, body)):
                sweep.extend(dict(r, rank=rank, key=record_key(cand.firm_id, cutoff), alt_id=aid,
                                  firm_class=_firm_class(r['token']))
                             for r in sweep_rows(raw, masked, field))
            articles.append({'alt_id': aid, 'tokens': toks, 'days_before': int((origin - rec['date']).days),
                             'news_day': rec['date'].strftime('%Y-%m-%d'),
                             'news_available_at': rec['available_at'].isoformat(), 'owner_terms': owners,
                             'article_extras': extras, 'active_acronyms': masker.active_acronyms(),
                             'headline': head, 'body': body, 'archive_rics': t.get('rics', []),
                             'headline_raw': head_raw, 'body_raw': body_raw,      # v5: private; verify re-masks
                             'replacement_log': log,
                             'truncated': head_raw != t['headline'] or body_raw != t['body']})
        if residual:
            raise ValueError(f'Strict masking residual for sample rank {rank}: {residual} hits')
        if date_left:
            raise ValueError(f'Date/boilerplate residual for sample rank {rank}: {dict(date_left)}')
        text = render(menu, cand, shown, articles)
        if cand.firm_id in text:
            raise ValueError('Firm identifier in input text')
        key = record_key(cand.firm_id, cutoff)
        records.append({'key': key, 'sample_rank': rank, 'input_version': S.INPUT_VERSION,
                        'input_sha256': sha_text(text), 'input_bytes': len(text.encode('utf-8')), 'input_text': text})
        metas.append({'key': key, 'sample_rank': rank, 'firm_id': cand.firm_id, 'cutoff': cutoff.isoformat(),
                      'origin_session': origin.isoformat(), 'focal_permno': focal, 'shown': shown,
                      'links': [{'permno': p, 'token': t} for p, t in tokens.items() if t.startswith('[LINK_')],
                      'aliases': firm_aliases, 'collisions': collisions,
                      'articles': [{k: v for k, v in x.items()} for x in articles],
                      'replacements_by_token': dict(replaced), 'strict_residual_hits': residual,
                      'date_residual_hits': int(sum(date_left.values())), 'edge_age_checks': dict(edge_checks),
                      'loose_hits': loose, 'focal_token_in_articles': any('[FOCAL]' in x['headline'] + x['body']
                                                                         for x in articles)})
    return records, metas


def _csv_text(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _stats(values):
    v = np.asarray(values, float)
    return {'n': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)),
            'max': float(v.max()), 'min': float(v.min())} if len(v) else {'n': 0}


def name_sources(sources, permnos, ricmap):
    """v3: CRSP rows of the masked firms and their issuer siblings, issuer-linked FactSet names and tickers, the English
    word list and RICs by PERMNO (all rows dated on or before 2022-12-31)."""
    siblings, cusip6 = issuer_siblings(sources.crsp_names, permnos)
    allp = set(int(p) for p in permnos) | {q for v in siblings.values() for q in v}
    fsnames, fstickers = factset_aliases(sources.factset, cusip6)
    rics_by_permno = {}
    for ric, (p, m) in ricmap.items():
        rics_by_permno.setdefault(p, []).append((ric, m))
    names = load_names(sources.crsp_names, allp)
    anchors = name_anchors({c for rows in names.values() for _, c, _ in rows if c})
    return {'names': names, 'fsnames': fsnames, 'fstickers': fstickers,
            'english': load_english(sources.english), 'rics_by_permno': rics_by_permno, 'siblings': siblings,
            'near_index': near_name_index(sources, anchors)}            # v5 (TD3C4 D2)


def build(out_dir, sources=None, n=S.SAMPLE_N, reask_n=S.REASK_N):
    start = time.perf_counter()
    sources = sources or real_sources()
    out_dir = Path(out_dir)
    targets = [out_dir / x for x in ('PILOT_SAMPLE.csv', 'INPUTS.jsonl', 'INPUT_META.jsonl', 'SCAN_HITS.jsonl',
                                     'MASK_AUDIT.json', 'BUILD.json', 'ADJACENCY.jsonl', 'ADJACENCY_SWEEP.jsonl')]
    for p in targets:
        if p.exists():
            raise FileExistsError(p)
    engine, fast_builder = td2_modules()
    hashes = source_hashes(sources)
    tables_sha = install_tables(sources)                 # v4 (E1, N6): frozen private mask table
    menu = S.menu_rows(sources.menu)
    frame = with_percentiles(pd.concat([features_year(sources, y) for y in S.YEARS], ignore_index=True))
    elig = eligible(frame)
    ricmap = ric_map(sources.ric_identity)
    accepted, stats = select_sample(sources, elig, n, ricmap, engine)
    if len(accepted) < n:
        raise ValueError(f'Only {len(accepted)} eligible firm-weeks passed re-verification')
    maps, links = edge_maps(sources, accepted, fast_builder)
    permnos = set()
    for a in accepted:
        for p in a['partners']:
            permnos.update(int(x) for x in p['row'].nodes)
    ns = name_sources(sources, permnos, ricmap)
    english = ns['english']
    sweep = []
    records, metas = build_inputs(menu, accepted, maps, links, ns, sweep)
    for r in sweep:
        r['row_id'] = sweep_row_id(r)
    if len({r['row_id'] for r in sweep}) != len(sweep):
        raise ValueError('Duplicate adjacency-sweep row id')
    reask = sorted(records, key=lambda r: reask_key(metas[r['sample_rank'] - 1]['firm_id'],
                                                    metas[r['sample_rank'] - 1]['cutoff']))[:reask_n]
    reask_rank = {r['key']: i for i, r in enumerate(reask, 1)}
    for r, m in zip(records, metas):
        r['reask'] = m['reask'] = r['key'] in reask_rank
        r['reask_rank'] = m['reask_rank'] = reask_rank.get(r['key'])
    sample_rows = [{'sample_rank': m['sample_rank'], 'key': m['key'], 'firm_id': m['firm_id'], 'cutoff': m['cutoff'],
                    'origin_session': m['origin_session'], 'year': pd.Timestamp(m['origin_session']).year,
                    'reask': m['reask'], 'reask_rank': m['reask_rank'] or ''} for m in metas]
    if any(pd.Timestamp(m['origin_session']).year not in S.YEARS or pd.Timestamp(m['cutoff']).year not in S.YEARS
           for m in metas):
        raise PermissionError('Sampled row outside 2021-2022')
    hits, scan_summary, adjacency = td3_scan.scan_build(metas, sources.crsp_names, sources.factset, ricmap, english)
    red = Redactor()
    new_text(targets[0], _csv_text(sample_rows, list(sample_rows[0])), red)
    for path, rows in ((targets[1], records), (targets[2], metas), (targets[3], hits), (targets[6], adjacency),
                       (targets[7], sweep)):
        w = JsonlWriter(path, red)
        for row in rows:
            w.write(row)
        w.close()
    loose_kinds = Counter(h['kind'] for m in metas for h in m['loose_hits'])
    firms = [t for m in metas for t in m['aliases'].values()]
    audit = {'version': S.VERSION, 'inputs': len(records), 'articles': sum(len(m['articles']) for m in metas),
             'replacements_total': int(sum(sum(m['replacements_by_token'].values()) for m in metas)),
             'inputs_with_focal_token_in_articles': int(sum(m['focal_token_in_articles'] for m in metas)),
             'strict_residual_hits_total': int(sum(m['strict_residual_hits'] for m in metas)),
             'date_residual_hits_total': int(sum(m['date_residual_hits'] for m in metas)),
             'alias_collisions_total': int(sum(len(m['collisions']) for m in metas)),
             'loose_indicator_hits_total': int(sum(len(m['loose_hits']) for m in metas)),
             'loose_indicator_hits_by_kind': dict(loose_kinds),
             'inputs_with_loose_indicator_hits': int(sum(bool(m['loose_hits']) for m in metas)),
             'firms_without_crsp_name': int(sum(1 for t in firms if not t['comnams'])),
             'firm_entries_with_factset_names': int(sum(1 for t in firms if t['factset_names'])),
             'firm_entries_with_names_after_origin': int(sum(1 for t in firms if t['comnams_after_origin'])),
             'firm_entries_with_common_names': int(sum(1 for t in firms if t['common'])),
             'firm_entries_with_issuer_siblings': int(sum(1 for t in firms if t.get('sibling_permnos'))),
             'firm_entries_with_extra_tickers': int(sum(1 for t in firms if t.get('tickers_extra'))),
             'replacements_by_kind': dict(Counter(e['kind'] for m in metas for x in m['articles']
                                                  for e in x.get('replacement_log', ()))),
             'articles_with_issuer_ric_extras': int(sum(1 for m in metas for x in m['articles']
                                                        if x.get('article_extras'))),
             'firms_masked_total': len(firms),
             'edge_age_checks': dict(sum((Counter(m['edge_age_checks']) for m in metas), Counter())),
             'articles_truncated': int(sum(x['truncated'] for m in metas for x in m['articles'])),
             'input_bytes': _stats([r['input_bytes'] for r in records]),
             'articles_per_input': _stats([len(m['articles']) for m in metas]),
             'independent_scan': scan_summary,
             'articles_with_active_acronyms': int(sum(1 for m in metas for x in m['articles']
                                                      if x.get('active_acronyms'))),
             'firm_entries_with_products': int(sum(1 for t in firms if (t.get('products') or {}).get('keys'))),
             'factset_single_words_dropped': int(sum(len(t.get('factset_single_words_dropped', ())) for t in firms)),
             'adjacency_sweep_rows': len(sweep),
             'adjacency_sweep_rows_by_class': dict(Counter(r['firm_class'] for r in sweep)),
             'adjacency_sweep_rows_glued': int(sum(1 for r in sweep if r.get('glued'))),          # v5 (D1)
             'adjacency_sweep_rows_number_neighbour': int(sum(1 for r in sweep if r.get('number_neighbour'))),
             'house_numbers_absorbed': int(sum(1 for m in metas for x in m['articles']
                                               for e in x.get('replacement_log', ()) if e.get('house_number_absorbed'))),
             'firm_entries_with_near_names': int(sum(1 for t in firms if t.get('near'))),         # v5 (D2)
             'firm_entries_with_code_prefixes': int(sum(1 for t in firms if t.get('code_prefixes'))),
             'near_name_index_first_words': len((ns.get('near_index') or {}).get('by_first', {})),
             'note': ('loose indicator hits are possible residuals from a broader case-insensitive scan; independent '
                      'scan hits (SCAN_HITS.jsonl) are candidates adjudicated in SCAN_ADJUDICATION.json before '
                      'verify')}
    new_json(targets[4], audit, red)
    years = Counter(r['year'] for r in sample_rows)
    receipt = {'version': S.VERSION, 'exploratory': True, 'rules': S.rules(),
               'question_sha256': S.question_bundle(menu)[1], 'menu': menu,
               'td2_artifact_sha256': hashes, 'external_source_sha256': external_hashes(sources),
               'td2_code_sha256': td2_code_hashes(), 'td3_code_sha256': code_hashes(),
               'td2_freeze_sha256': S.TD2_FREEZE_SHA256 if sources.td2_bound else None,
               'english_words': len(english),
               'mask_tables_sha256': tables_sha, 'mask_table_state': td3_mask.table_state(),
               'adjacency_sweep_rows': len(sweep),
               'counts': {'primary_rows_2021_2022': int(len(frame)), 'eligible_firm_weeks': int(len(elig)),
                          'eligible_by_year': {str(k): int(v) for k, v in elig.origin_session.dt.year.value_counts()
                                               .sort_index().items()},
                          'examined': stats['examined'], 'accepted': len(accepted), 'chunks': stats['chunks'],
                          'rejected': dict(stats['rejected']), 'article_rejects': dict(stats['article_rejects']),
                          'archive_text_status': dict(stats['text_status']),
                          'shown_partners_without_article': stats['partners_without_article'],
                          'partner_priority_equals_td2_order': stats['partner_order_equal_td2'],
                          'sample_by_year': {str(k): v for k, v in sorted(years.items())},
                          'distinct_firms': len({m['firm_id'] for m in metas}), 'reask': len(reask)},
               'independent_scan': scan_summary,
               'outputs_sha256': {str(p): sha(p) for p in targets[:5] + targets[6:]},
               'labels_or_returns_read': False, 'network': False, 'provider_calls': 0,
               'elapsed_seconds': time.perf_counter() - start, **peak()}
    new_json(targets[5], receipt, red)
    return receipt


def adjudication_check(hits, path):
    """Every scan hit needs a verdict; any confirmed leak is a problem. Returns (problems Counter, summary)."""
    problems, summary = Counter(), {'path': str(path), 'present': Path(path).exists()}
    if not Path(path).exists():
        problems['scan_adjudication_missing'] += 1
        return problems, summary
    adj = json.loads(Path(path).read_text(encoding='utf-8')).get('adjudications', {})
    ids = {h['hit_id'] for h in hits}
    if set(adj) != ids:
        problems['scan_adjudication_ids_differ'] += 1
    verdicts = Counter()
    for h in hits:
        v = (adj.get(h['hit_id']) or {}).get('verdict')
        verdicts[(h['firm_class'], v)] += 1
        if v not in ('false_positive', 'confirmed_leak'):
            problems['scan_hit_without_verdict'] += 1
        elif v == 'confirmed_leak':
            problems['confirmed_leak_' + ('focal_or_shown' if h['firm_class'] in ('focal', 'shown') else 'link')] += 1
    summary.update(hits=len(hits), adjudicated=len(adj), verdicts={f'{a}:{b}': n for (a, b), n in verdicts.items()},
                   sha256=sha(path))
    return problems, summary


_YEAR_TIME = re.compile(r'(?<![\d.,])(?:19|20)\d\d(?=\d{1,2}:\d\d)')
_RIC_IN_BRACKETS = re.compile(r'[<(]\s*([A-Z0-9]{1,8})(?:_?[a-z]{1,2})?\.[A-Z]{1,3}[a-z]?\s*[>)]')


def review_findings_check(records):
    """The TD3C 79 leak strings and the TD3BC confirmed findings must be absent from the same-rank input. A TD3BC
    recoverable-mask item counts as present when its masked phrase still occurs (for an item without a role-token
    bracket, e.g. a headquarters-town phrase, when a role token still sits right before the phrase)."""
    problems, out = Counter(), {}
    by_rank = {r['sample_rank']: r['input_text'] for r in records}
    for name, path in (('td3c', S.REVIEW_LEAKS), ('td3bc', S.REVIEW_FINDINGS_BC)):
        res = {'path': str(path), 'checked': 0, 'present': []}
        if Path(path).exists():
            for it in json.loads(Path(path).read_text(encoding='utf-8'))['items']:
                res['checked'] += 1
                text, string = by_rank.get(it['rank'], ''), it['string']
                if it.get('class', '').startswith('recoverable_overmask') and '[' not in string and ']' not in string:
                    present = re.search(r'\]\s*,?\s*' + re.escape(string), text) is not None
                else:
                    present = string in text
                if present:
                    res['present'].append({'rank': it['rank'], 'string': string})
                    problems[f'review_{name}_item_present'] += 1
        out[name] = res
    # v4: TD3C3 E1 (every masked unit-compound phrase of the evidence file must be gone from the same-rank input) and
    # E2 (every listed date-typo remnant that matches the audit's month-typo pattern must be gone)
    res = {'path': str(S.REVIEW_FINDINGS_C3), 'checked': 0, 'present': []}
    if Path(S.REVIEW_FINDINGS_C3).exists():
        for it in json.loads(Path(S.REVIEW_FINDINGS_C3).read_text(encoding='utf-8')):
            res['checked'] += 1
            if it['masked_text'].strip() in by_rank.get(it['rank'], ''):
                res['present'].append({'rank': it['rank'], 'token': it['token']})
                problems['review_td3c3_e1_item_present'] += 1
    out['td3c3_e1'] = res
    res = {'path': str(S.REVIEW_DATE_TYPOS_C3), 'checked': 0, 'present': []}
    audit_month_typo = re.compile(r'\b[A-Za-z]{1,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,\s*\[YEAR\]')
    if Path(S.REVIEW_DATE_TYPOS_C3).exists():
        for line in Path(S.REVIEW_DATE_TYPOS_C3).read_text(encoding='utf-8').splitlines():
            m = re.match(r"\s*(\d+)\s*\|\s*'(.*?)'\s*\|", line)
            if not m or not audit_month_typo.fullmatch(m.group(2)):
                continue
            res['checked'] += 1
            if m.group(2) in by_rank.get(int(m.group(1)), ''):
                res['present'].append({'rank': int(m.group(1)), 'string': m.group(2)})
                problems['review_td3c3_e2_item_present'] += 1
    out['td3c3_e2'] = res
    # v5: TD3C4 D1, D2 and N-b items (private evidence file transcribed from TD3C4_AUDIT.md): every listed string must
    # be absent from the same-rank input (case-insensitive)
    res = {'path': str(S.REVIEW_FINDINGS_C4), 'checked': 0, 'present': []}
    if Path(S.REVIEW_FINDINGS_C4).exists():
        res['sha256'] = sha(S.REVIEW_FINDINGS_C4)
        for it in json.loads(Path(S.REVIEW_FINDINGS_C4).read_text(encoding='utf-8'))['items']:
            res['checked'] += 1
            if it['string'].lower() in by_rank.get(it['rank'], '').lower():
                res['present'].append({'rank': it['rank'], 'class': it['class']})
                problems['review_td3c4_item_present'] += 1
    out['td3c4'] = res
    # v6: the TD3C5 D1 item (private evidence file transcribed from TD3C5_AUDIT.md), absent from its input
    res = {'path': str(S.REVIEW_FINDINGS_C5), 'checked': 0, 'present': []}
    if Path(S.REVIEW_FINDINGS_C5).exists():
        res['sha256'] = sha(S.REVIEW_FINDINGS_C5)
        for it in json.loads(Path(S.REVIEW_FINDINGS_C5).read_text(encoding='utf-8'))['items']:
            res['checked'] += 1
            if it['string'].lower() in by_rank.get(it['rank'], '').lower():
                res['present'].append({'rank': it['rank'], 'class': it['class']})
                problems['review_td3c5_item_present'] += 1
    out['td3c5'] = res
    return problems, out


def sweep_review_check(path, sweep_path, sweep, metas):
    """v4 (E1 item 4): ADJACENCY_SWEEP_REVIEW.json must cover every row of the current ADJACENCY_SWEEP.jsonl with a
    verdict ('safe' with a reason, or 'gives_back'), record zero rows that give the hidden word back, and the sweep's
    token positions must equal those found again in the masked texts of INPUT_META.jsonl."""
    problems, summary = Counter(), {'path': str(path), 'present': Path(path).exists()}
    expected = set()
    for m in metas:
        for x in m['articles']:
            for field in ('headline', 'body'):
                for start, token, side in sweep_positions(x[field]):
                    expected.add((m['sample_rank'], x['alt_id'], field, start, token, side))
    found = {(r['rank'], r['alt_id'], r['field'], r['start'], r['token'], r['side']) for r in sweep}
    if found != expected:
        problems['adjacency_sweep_positions_differ'] += 1
    summary.update(rows=len(sweep), recomputed_rows=len(expected))
    if not Path(path).exists():
        problems['adjacency_sweep_review_missing'] += 1
        return problems, summary
    rev = json.loads(Path(path).read_text(encoding='utf-8'))
    if rev.get('sweep_sha256') != sha(sweep_path):
        problems['adjacency_sweep_review_stale'] += 1
    verdicts = rev.get('verdicts', {})
    ids = {r['row_id'] for r in sweep}
    if set(verdicts) != ids:
        problems['adjacency_sweep_review_ids_differ'] += 1
    counts = Counter()
    for i in ids:
        v = verdicts.get(i) or {}
        counts[v.get('verdict')] += 1
        if v.get('verdict') not in ('safe', 'gives_back') or (v.get('verdict') == 'safe' and not v.get('reason')):
            problems['adjacency_sweep_row_without_verdict'] += 1
        elif v.get('verdict') == 'gives_back':
            problems['adjacency_sweep_gives_back'] += 1
    if rev.get('gives_back', 1) != 0:
        problems['adjacency_sweep_gives_back_recorded'] += 1
    summary.update(sha256=sha(path), verdicts=dict(counts), reason_classes=rev.get('reason_classes'))
    return problems, summary


def adjacency_review_check(path, adjacency_path):
    """The builder's adjacency review (ADJACENCY_REVIEW.json) must cover this build's ADJACENCY.jsonl and record zero
    masks that give the hidden word away."""
    problems, summary = Counter(), {'path': str(path), 'present': Path(path).exists()}
    if not Path(path).exists():
        problems['adjacency_review_missing'] += 1
        return problems, summary
    rev = json.loads(Path(path).read_text(encoding='utf-8'))
    if rev.get('adjacency_sha256') != sha(adjacency_path):
        problems['adjacency_review_stale'] += 1
    if rev.get('inferable_masks', 1) != 0 or rev.get('open_items'):
        problems['adjacency_inferable_masks'] += 1
    summary.update(sha256=sha(path), inferable_masks=rev.get('inferable_masks'),
                   rows_reviewed=rev.get('rows_reviewed'), neighbour_word_types=rev.get('neighbour_word_types'))
    return problems, summary


def verify(out_dir, sources=None):
    """Independent re-check of a finished build; writes VERIFY.json (exclusive). v6 (TD3C5): no readable firm ticker or
    initialism followed by a corporate-unit word remains (d1c5_initialism_unit) and the TD3C5 finding is absent.
    v5 (TD3C4): every article is re-masked
    from its stored raw text and must reproduce the input; the sweep rows and their hidden words are recomputed from the
    raw text and must equal the sweep file; no token is glued to a letter or digit (D1); no house number sits before a
    token whose hidden text ends in a street word (D1); no firm code of a masked firm, no token-hyphen-digit remnant and
    no near name remains (D2); the TD3C4 findings are absent. v2: date/boilerplate patterns (C2),
    edge ages (R4), a re-run of the independent scanner and its adjudication (C1), the TD3C leak strings. v3 (TD3BC):
    the strict residual with every v3 rule (context guard, embedded names, misspellings, issuer RICs), an issuer-RIC
    check against the archive rics field, no <RIC> left in exchange halt or imbalance headlines, no RIC in brackets
    whose root is a masked firm's ticker, years glued to a time, a URL check independent of the masker's TLD list, the
    TD3BC findings, a re-run of the adjacency report and the builder's adjacency review. v4 (TD3C3): the frozen mask
    table must be the one bound in BUILD.json; the strict residual includes the E1 checks (a role token followed by a
    product phrase of its own firm, preceded by its own prefix phrase, joined to its joint-product partner, or an
    active acronym of the article), counted separately; the E2 month-typo and day-range patterns (independently of
    date_residuals); the TD3C3 E1 and E2 findings; the adjacency sweep and its row-by-row review."""
    sources = sources or real_sources()
    out_dir = Path(out_dir)
    records = read_jsonl(out_dir / 'INPUTS.jsonl')
    metas = read_jsonl(out_dir / 'INPUT_META.jsonl')
    build_receipt = json.loads((out_dir / 'BUILD.json').read_text(encoding='utf-8'))
    problems = Counter()
    tables_sha = install_tables(sources)
    if tables_sha != build_receipt.get('mask_tables_sha256'):
        problems['mask_tables_differ_from_build'] += 1
    for p, h in build_receipt['outputs_sha256'].items():
        if sha(p) != h:
            problems['output_hash_changed'] += 1
    if len(records) != len(metas) or len({r['key'] for r in records}) != len(records):
        problems['record_count_or_key'] += 1
    english = load_english(sources.english)
    news = load_news(sources, {x['alt_id'] for m in metas for x in m['articles']})
    cands = pd.DataFrame([{'firm_id': m['firm_id'], 'cutoff': pd.Timestamp(m['cutoff']),
                           'origin_session': pd.Timestamp(m['origin_session'])} for m in metas])
    ev = load_evidence(sources, cands) if len(cands) else pd.DataFrame()
    newest = {(r.firm_id, pd.Timestamp(r.cutoff), int(r.hop), int(r.partner)): pd.Timestamp(r.max_edge_available_at)
              for r in ev.itertuples(index=False)}
    date_hits, ric_checks = Counter(), Counter()
    keys = ('names', 'common', 'tickers', 'tickers_extra', 'rics', 'embedded', 'fuzzy', 'products', 'code_prefixes',
            'near', 'unit_prefixes')
    e1_kinds = ('product_after_token', 'prefix_before_token', 'joint_partner_adjacent', 'product_acronym')
    v5_kinds = {'glued_token': 'd1_glued_token', 'firm_code': 'd2_firm_code', 'near_name': 'd2_near_name',
                'initialism_unit': 'd1c5_initialism_unit'}
    recomputed_sweep = set()
    remask_checked = 0
    e2_month_typo = re.compile(r'\b(?!(?i:' + '|'.join(td3_mask.NON_MONTH_STOP) + r')\b)[A-Za-z]{1,9}\.?\s+\d{1,2}'
                               r'(?:st|nd|rd|th)?,\s*\[YEAR\]')
    e2_day_range = re.compile(r'\[DATE\]\s*(?:-|–|to|and|through)\s*\d{1,2}\b(?![.,:]\d)', re.I)
    for r, m in zip(records, metas):
        if r['key'] != m['key'] or sha_text(r['input_text']) != r['input_sha256']:
            problems['input_hash'] += 1
        o, c = pd.Timestamp(m['origin_session']), pd.Timestamp(m['cutoff'])
        if o.year not in S.YEARS or c.year not in S.YEARS or not c > o:
            problems['period'] += 1
        for s in m['shown']:
            t = newest.get((m['firm_id'], c, int(s['hop']), int(s['permno'])))
            if t is None or t >= c or s['edge_age_days'] != max(0, (o.date() - _ny_date(t)).days):
                problems['edge_age'] += 1
            if f"{s['token']} (" not in r['input_text'] or f"link age {_days(s['edge_age_days'])}" not in r['input_text']:
                problems['edge_age_not_rendered'] += 1
        for x in m['articles']:
            for token in x['tokens']:
                permno = next(s['permno'] for s in m['shown'] if s['token'] == token)
                if verify_article(x['alt_id'], permno, o, c, news) is not None:
                    problems['article_not_eligible'] += 1
            if news[x['alt_id']]['available_at'] >= c or news[x['alt_id']]['date'] >= o:
                problems['article_after_cutoff'] += 1
        # v5: masker precedence order as in the build (focal, shown partners in order, link firms); the metadata is
        # written with sorted keys, so the order of m['aliases'] is not the precedence order
        order = ['[FOCAL]'] + [s['token'] for s in m['shown']] + [x['token'] for x in m['links']]
        order += [t for t in m['aliases'] if t not in order]
        entries = [(t, {k: m['aliases'][t].get(k, {} if k == 'products' else []) for k in keys})
                   for t in order if t in m['aliases']]
        firm_rics = {ric for _, a in entries for ric in a['rics']}
        roots = {t for _, a in entries for t in list(a['tickers']) + list(a['tickers_extra']) if len(t) >= 2}
        for x in m['articles']:
            # v5: re-mask the stored raw (truncated) text with the stored entries; it must reproduce the masked text
            remasker = Masker(article_entries(entries, x['owner_terms'], x.get('article_extras')), english)
            if 'headline_raw' not in x or 'body_raw' not in x:
                problems['raw_text_missing'] += 1
            else:
                remask_checked += 1
                h2, b2, *_ = remasker.mask_article(x['headline_raw'], x['body_raw'])
                if (h2, b2) != (x['headline'], x['body']) or \
                        remasker.active_acronyms() != (x.get('active_acronyms') or {}):
                    problems['remask_differs'] += 1
                # v5: the sweep rows recomputed from the raw and masked text, with hidden words
                for field in ('headline', 'body'):
                    rows, sp = sweep_text_problems(x[field + '_raw'], x[field], field)
                    problems.update(sp)
                    for row in rows:
                        recomputed_sweep.add((m['sample_rank'], x['alt_id'], field, row['start'], row['token'],
                                              row['side'], row['hidden']))
            masker = Masker(article_entries(entries, x['owner_terms'], x.get('article_extras')), english)
            masker.set_active_acronyms(x.get('active_acronyms'))
            res = masker.residual(x['headline']) + masker.residual(x['body'])
            if res:
                problems['strict_residual'] += 1
            for h in res:
                if h['kind'] in e1_kinds:
                    problems['e1_' + h['kind']] += 1
                if h['kind'] in v5_kinds:
                    problems[v5_kinds[h['kind']]] += 1
            text = x['headline'] + '\n' + x['body']
            # v5 (TD3C4 D1/D2): no role token glued to a letter or digit, and no token followed by a hyphen or dash and
            # a digit (the remnant of a code masked only at its prefix)
            if re.search(r'[^\W_]' + ROLE_TOKEN.pattern + '|' + ROLE_TOKEN.pattern + r'[^\W_]', text):
                problems['d1_glued_token_text'] += 1
            if re.search(ROLE_TOKEN.pattern + r'[\-–]\d', text):
                problems['d2_token_code_remnant'] += 1
            if e2_month_typo.search(text):
                problems['e2_month_typo_date'] += 1
            if e2_day_range.search(text):
                problems['e2_day_range_remnant'] += 1
            if x['headline'] not in r['input_text'] or x['body'] not in r['input_text']:
                problems['article_text_not_in_input'] += 1
            for name in date_residuals(x['headline']) + date_residuals(x['body']):
                date_hits[name] += 1
            if re.search(r'(?<![\d.,])(?:19|20)\d\d(?!\d)', text) or _YEAR_TIME.search(text):
                problems['year_in_article_text'] += 1
            if URL_RESIDUAL.search(text):
                problems['url_in_article_text'] += 1
            # issuer RICs of the archive rics field attributable to a masked firm must not appear in the text
            for ric in x.get('archive_rics', ()):
                root = ric.split('.')[0]
                own = ric in firm_rics or ric in {q for e in (x.get('article_extras') or {}).values()
                                                  for q in e.get('rics', ())} or \
                    any(root.startswith(t) and len(t) >= 3 and 0 <= len(root) - len(t) <= 3 for t in roots)
                if own:
                    ric_checks['archive_rics_attributed'] += 1
                    if re.search(r'(?<![A-Za-z0-9])' + re.escape(ric) + r'(?![A-Za-z0-9])', text):
                        problems['issuer_ric_in_text'] += 1
            if HALT_HEADLINE.match(x['headline']) and re.search(r'<\s*(?!\[)[^<>\s]+\s*>', x['headline']):
                problems['ric_left_in_halt_headline'] += 1
            for mm in _RIC_IN_BRACKETS.finditer(text):
                ric_checks['ric_like_in_brackets'] += 1
                if mm.group(1) in roots:
                    problems['masked_firm_ric_in_brackets'] += 1
        if m['firm_id'] in r['input_text']:
            problems['firm_id_in_input'] += 1
        if re.search(r'\d{4}-\d{2}-\d{2}', r['input_text']) or re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', r['input_text']):
            problems['numeric_date_in_input'] += 1
        if re.search(r'STORY_(?:NUMBER|DATE|TIME)', r['input_text']):
            problems['story_fields_in_input'] += 1
    for name, k in date_hits.items():
        problems['date_residual_' + name] += k
    hits = read_jsonl(out_dir / 'SCAN_HITS.jsonl') if (out_dir / 'SCAN_HITS.jsonl').exists() else []
    adjacency = read_jsonl(out_dir / 'ADJACENCY.jsonl') if (out_dir / 'ADJACENCY.jsonl').exists() else []
    ricmap = ric_map(sources.ric_identity)
    rerun, scan_summary, adj_rerun = td3_scan.scan_build(metas, sources.crsp_names, sources.factset, ricmap, english)
    if sorted(h['hit_id'] for h in rerun) != sorted(h['hit_id'] for h in hits):
        problems['scan_rerun_differs'] += 1
    if adj_rerun != adjacency:
        problems['adjacency_rerun_differs'] += 1
    adj_problems, adj_summary = adjudication_check(rerun, out_dir / 'SCAN_ADJUDICATION.json')
    problems.update(adj_problems)
    ar_problems, ar_summary = adjacency_review_check(out_dir / 'ADJACENCY_REVIEW.json', out_dir / 'ADJACENCY.jsonl')
    problems.update(ar_problems)
    sweep_path = out_dir / 'ADJACENCY_SWEEP.jsonl'
    sweep = read_jsonl(sweep_path) if sweep_path.exists() else []
    if not sweep_path.exists():
        problems['adjacency_sweep_missing'] += 1
    sw_problems, sw_summary = sweep_review_check(out_dir / 'ADJACENCY_SWEEP_REVIEW.json', sweep_path, sweep, metas)
    problems.update(sw_problems)
    # v5: the sweep file's rows (positions, side and hidden words) equal those recomputed from the raw text
    if {(r['rank'], r['alt_id'], r['field'], r['start'], r['token'], r['side'], r['hidden']) for r in sweep} != \
            recomputed_sweep:
        problems['adjacency_sweep_hidden_differs'] += 1
    sw_summary.update(recomputed_with_hidden=len(recomputed_sweep), remask_checked_articles=remask_checked,
                      glued_rows=sum(1 for r in sweep if r.get('glued')),
                      number_neighbour_rows=sum(1 for r in sweep if r.get('number_neighbour')))
    review = {'td3c': {'checked': 0}, 'td3bc': {'checked': 0}, 'td3c3_e1': {'checked': 0}, 'td3c3_e2': {'checked': 0},
              'td3c4': {'checked': 0}, 'td3c5': {'checked': 0}}
    if sources.td2_bound:
        rv_problems, review = review_findings_check(records)
        problems.update(rv_problems)
    n_reask = sum(bool(r['reask']) for r in records)
    passed = not problems
    if sources.td2_bound:
        passed = passed and len(records) == S.SAMPLE_N and n_reask == S.REASK_N and \
            review['td3c']['checked'] > 0 and review['td3bc']['checked'] > 0 and \
            review['td3c3_e1']['checked'] > 0 and review['td3c3_e2']['checked'] > 0 and \
            review['td3c4']['checked'] > 0 and review['td3c5']['checked'] > 0
    result = {'version': S.VERSION, 'records': len(records), 'reask': n_reask, 'problems': dict(problems),
              'passed': passed, 'inputs_sha256': sha(out_dir / 'INPUTS.jsonl'),
              'build_sha256': sha(out_dir / 'BUILD.json'), 'independent_scan': scan_summary,
              'scan_adjudication': adj_summary, 'adjacency_review': ar_summary, 'review_findings': review,
              'adjacency_sweep': sw_summary, 'mask_tables_sha256': tables_sha,
              'ric_checks': dict(ric_checks),
              'date_patterns_checked': [n for n, _, _ in DATE_PATTERNS] + ['story_fields', 'story_ids', 'url',
                                                                          'apostrophe_year2', 'url_independent',
                                                                          'year_time_independent',
                                                                          'e2_month_typo_independent',
                                                                          'e2_day_range_independent',
                                                                          'e1_product_after_token',
                                                                          'remask_from_raw', 'sweep_hidden_recomputed',
                                                                          'd1_glued_token', 'd1_house_number_street',
                                                                          'd2_firm_code', 'd2_token_code_remnant',
                                                                          'd2_near_name', 'td3c4_findings',
                                                                          'd1c5_initialism_unit', 'td3c5_findings']}
    new_json(out_dir / 'VERIFY.json', result)
    return result
