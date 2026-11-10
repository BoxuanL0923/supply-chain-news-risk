"""TD4 v2 input builds with masker v2 (td4v2_mask) for every period: the 2021-2022 selection subsample and the 2023,
2024 and 2025 test years. No input built with the v1 masker is used by v2.

The v1 build path is reused (td4_inputs.run_build: chunked, resumable, the TD3 v6 render and frozen questions), with
these v2 replacements installed at run time (the TD3 and v1 files are unchanged):
M2-1 td3_build.build_inputs -> build_inputs_v2: a copy of TD3 v6 build_inputs in which each article is masked by
     td4v2_mask.MaskerV2 with the article's own tokens, the strict residual is MaskerV2.residual and the date residual
     is td4v2_mask.date_residuals_v2 (a residual withholds the firm-week from all providers alike, as in v1 D6).
M2-2 td3_build.aliases_for -> aliases_for_v2: TD3 v6 aliases_for plus td4v2_mask.add_v2 (v2 variants; pure function of
     the firm's names and tickers, the frozen word tables and the private v2 table; memoised per PERMNO by run_build).
M2-3 td4_inputs.accept_chunk -> accept_chunk_v2: a copy in which a mis-tag withdrawal notice is not an eligible shown
     article (the partner's next most recent eligible article is shown instead; TD2 features are unchanged).
M2-4 every build applies the D8 v2 story-field rule (td4v2_mask.story_fields_v2, equal length, before truncation) to
     every archive text; there is no separate repair build.
Outputs: OUT/inputs_m2/<period>/ (INPUTS.jsonl, INPUT_META_COMPACT.jsonl, AUDIT_META.jsonl, WITHHELD.jsonl,
ELIGIBLE.parquet, BUILD.json, MASK_V2_RECEIPT.json). Article stores of v1 are reused by tag (they hold archive rows of
the period's evidence article IDs and no masked text).
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import td4_spec as T
import td4_env as E

M2_ROOT = T.OUT / 'inputs_m2'
# the final masker-v2 builds (inputs_m2/select_2122, period_2023 and period_2024 hold stopped first attempts made
# with an earlier masker-v2 revision; they are never read)
PERIOD_DIR = {'2122': M2_ROOT / 'select_2122_r3', 2023: M2_ROOT / 'period_2023_r3', 2024: M2_ROOT / 'period_2024_r3',
              2025: M2_ROOT / 'period_2025_r3'}
STORE_TAG = {'2122': 'select_2122', 2023: '2023', 2024: '2024', 2025: '2025'}
YEARS = {'2122': (2021, 2022), 2023: (2023,), 2024: (2024,), 2025: (2025,)}
TABLES_V2 = T.OUT / 'tables_v2' / 'MASK_TABLES_TD4_v2.json'
_STATE = {'td3_aliases_for': None, 'installed': None}


def code_files():
    here = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('td4v2_*.py'))}


def install(years):
    """Load the frozen word tables and the private v2 table, then install the v2 replacements (M2-1..M2-3)."""
    E.paths()
    import td3_build as B
    import td4_inputs as I
    import td4v2_mask as MV
    import td4v2_words as W
    last_day = f'{max(years)}-12-31'
    if last_day > T.HARD_SEAL:
        raise PermissionError('TD4 seal')
    common, common_sha = W.load_common()
    if not W.df_file(last_day).exists():
        W.build_df(last_day)
    df, df_sha = W.load_df(last_day)
    tables = MV.validate_tables_v2(json.loads(TABLES_V2.read_text(encoding='utf-8')))
    for k, e in tables['renames'].items():
        if e['dated'] > T.HARD_SEAL:
            raise PermissionError('rename dated after the seal')
    MV.install(common, df, tables, last_day)
    if _STATE['td3_aliases_for'] is None:
        f = B.aliases_for
        while hasattr(f, '__wrapped__'):
            f = f.__wrapped__
        _STATE['td3_aliases_for'] = f
    B.aliases_for = aliases_for_v2
    B.build_inputs = build_inputs_v2
    I.accept_chunk = accept_chunk_v2
    I._ALIAS_CACHE.clear()                       # the per-PERMNO alias memo depends on the period's word tables
    info = {'mask_version': MV.VERSION, 'common_words_sha256': common_sha, 'crsp_word_df_sha256': df_sha,
            'tables_v2_sha256': hashlib.sha256(TABLES_V2.read_bytes()).hexdigest(), 'last_day': last_day,
            'code_sha256': code_files(), 'common_min': MV.COMMON_MIN, 'df_max_nonfirst': MV.DF_MAX_NONFIRST}
    _STATE['installed'] = info
    return info


def aliases_for_v2(permno, origin, ns):
    """M2-2: TD3 v6 aliases_for + td4v2_mask.add_v2."""
    import td4v2_mask as MV
    al, comnams, after, fs, scn = _STATE['td3_aliases_for'](permno, origin, ns)
    al = MV.add_v2(al, comnams, fs, scn, ns.get('english', frozenset()), permno=int(permno))
    return al, comnams, after, fs, scn


def accept_chunk_v2(sources_, chunk, ricmap, engine, stats):
    """M2-3: td4_inputs.accept_chunk with mis-tag withdrawal notices rejected as shown articles (reason
    'mistag_notice'); otherwise unchanged."""
    import td3_build as B
    import td4v2_mask as MV
    ev = B.load_evidence(sources_, chunk)
    news = B.load_news(sources_, {a for lst in ev.article_ids for a in lst})
    verdict = {}
    for w in ev.itertuples(index=False):
        for aid in w.article_ids:
            verdict[(w.firm_id, w.cutoff, int(w.partner), aid)] = B.verify_article(aid, w.partner, w.origin_session,
                                                                                  w.cutoff, news)
    wanted = {k[3]: news[k[3]] for k, v in verdict.items() if v is None}
    texts, tstatus = B.fetch_texts(sources_, wanted, ricmap, engine.score)
    for aid, t in texts.items():                                   # v2: mis-tag notices are not shown
        if tstatus.get(aid) is None and MV.mistag(t.get('headline'), t.get('body')):
            tstatus[aid] = 'mistag_notice'
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


def build_inputs_v2(menu, accepted, maps, links, ns, sweep=None):
    """M2-1: td3_build.build_inputs (TD3 v6) with td4v2_mask.MaskerV2 per article (own tokens = the tokens the article
    was chosen for), MaskerV2.residual and date_residuals_v2; everything else unchanged."""
    import td3_build as B
    import td3_spec as S
    import td4v2_mask as MV
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
            rel, times = B.path_relations(list(w.nodes), list(w.relationship_ids), edges, cutoff, with_times=True)
            rels[(int(w.hop), int(w.partner))] = rel
            newest = pd.Timestamp(w.max_edge_available_at)
            edge_checks['equal' if max(times) == newest else 'differs'] += 1
            if newest >= cutoff:
                raise ValueError('Edge availability not before cutoff')
            ages[(int(w.hop), int(w.partner))] = max(0, (origin.date() - B._ny_date(newest)).days)
        tokens, shown = B.assign_tokens(focal, a['partners'], rels)
        for s in shown:
            s['edge_age_days'] = ages[(s['hop'], s['permno'])]
        entries, comnams, firm_aliases = [], {}, {}
        for permno, token in tokens.items():
            al, cn, after, fs, scn = B.aliases_for(permno, origin, ns)
            entries.append((token, al))
            comnams[token] = cn
            firm_aliases[token] = {'permno': permno, **al, 'comnams': cn, 'comnams_after_origin': after,
                                   'factset_names': fs, 'sibling_permnos': ns.get('siblings', {}).get(permno, []),
                                   'sibling_comnams': scn}
        english = ns.get('english', frozenset())
        collisions = MV.MaskerV2(entries, english).collisions
        by_article = {}
        for s in shown:
            if s['article'] is not None:
                by_article.setdefault(s['article'], []).append(s['token'])
        articles, residual, loose, replaced, date_left = [], 0, [], Counter(), Counter()
        res_kinds = Counter()
        for aid, toks in by_article.items():
            t, rec = a['texts'][aid], a['news'][aid]
            owners = {tok: sorted(B.owner_terms(comnams[tok])) for tok in toks if B.owner_terms(comnams[tok])}
            extras = B.article_extras(t.get('rics', []), t['headline'], toks, entries)
            masker = MV.MaskerV2(B.article_entries(entries, owners, extras), english, own=toks)
            head_raw, body_raw = B.truncate(t['headline'], S.HEADLINE_CHARS), B.truncate(t['body'], S.BODY_CHARS)
            head, body, c1, c2, l1, l2 = masker.mask_article(head_raw, body_raw)
            log = [{'field': 'headline', **e} for e in l1] + [{'field': 'body', **e} for e in l2]
            replaced.update(c1)
            replaced.update(c2)
            hits = masker.residual(head) + masker.residual(body)
            residual += len(hits)
            res_kinds.update(h['kind'] for h in hits)
            date_left.update(MV.date_residuals_v2(head) + MV.date_residuals_v2(body))
            loose += B.loose_scan(head + '\n' + body, entries, comnams)
            for field, raw, masked in (('headline', head_raw, head), ('body', body_raw, body)):
                sweep.extend(dict(r, rank=rank, key=B.record_key(cand.firm_id, cutoff), alt_id=aid,
                                  firm_class=B._firm_class(r['token']))
                             for r in B.sweep_rows(raw, masked, field))
            articles.append({'alt_id': aid, 'tokens': toks, 'days_before': int((origin - rec['date']).days),
                             'news_day': rec['date'].strftime('%Y-%m-%d'),
                             'news_available_at': rec['available_at'].isoformat(), 'owner_terms': owners,
                             'article_extras': extras, 'active_acronyms': masker.active_acronyms(),
                             'headline': head, 'body': body, 'archive_rics': t.get('rics', []),
                             'headline_raw': head_raw, 'body_raw': body_raw,
                             'replacement_log': log,
                             'truncated': head_raw != t['headline'] or body_raw != t['body']})
        if residual:
            raise ValueError(f'Strict masking residual for sample rank {rank}: {residual} hits {dict(res_kinds)}')
        if date_left:
            raise ValueError(f'Date/boilerplate residual for sample rank {rank}: {dict(date_left)}')
        text = B.render(menu, cand, shown, articles)
        if cand.firm_id in text:
            raise ValueError('Firm identifier in input text')
        key = B.record_key(cand.firm_id, cutoff)
        records.append({'key': key, 'sample_rank': rank, 'input_version': S.INPUT_VERSION,
                        'input_sha256': B.sha_text(text), 'input_bytes': len(text.encode('utf-8')), 'input_text': text})
        metas.append({'key': key, 'sample_rank': rank, 'firm_id': cand.firm_id, 'cutoff': cutoff.isoformat(),
                      'origin_session': origin.isoformat(), 'focal_permno': focal, 'shown': shown,
                      'links': [{'permno': p, 'token': t} for p, t in tokens.items() if t.startswith('[LINK_')],
                      'aliases': firm_aliases, 'collisions': collisions,
                      'articles': [{k: v for k, v in x.items()} for x in articles],
                      'replacements_by_token': dict(replaced), 'strict_residual_hits': residual,
                      'date_residual_hits': int(sum(date_left.values())), 'edge_age_checks': dict(edge_checks),
                      'loose_hits': loose, 'focal_token_in_articles': any('[FOCAL]' in x['headline'] + x['body']
                                                                         for x in articles),
                      'mask_version': MV.VERSION})
    return records, metas


# ------------------------------------------------------------------------------------------------------- builds
def _select_filter():
    import td4v2_build as BV
    return lambda e: BV._order(e).iloc[:BV.CANDIDATES]


def build(period, out_dir=None, frame_filter=None, order=None, audit_n=300, limit=None, chunk=None, tag=None):
    """Build one period with masker v2. period: '2122', 2023, 2024 or 2025."""
    import td4_inputs as I
    import td3_runtime as R
    import td4v2_mask as MV
    period = '2122' if str(period) == '2122' else int(period)
    years = YEARS[period]
    info = install(years)
    out_dir = Path(out_dir or PERIOD_DIR[period])
    if frame_filter is None and period == '2122':
        frame_filter = _select_filter()
    kw = {} if chunk is None else {'chunk': chunk}
    rec = I.run_build(years, out_dir, frame_filter=frame_filter, order=order, audit_n=audit_n, limit=limit,
                      tag=tag or STORE_TAG[period], text_hook=MV.story_fields_v2, **kw)
    receipt = {'period': str(period), **info, 'build_receipt_sha256': R.sha(out_dir / 'BUILD.json'),
               'd8_rule': 'td4v2_mask.story_fields_v2', 'mistag_rule': 'td4v2_mask.MISTAG',
               'v1_masker_inputs_used': False}
    R.new_json(out_dir / 'MASK_V2_RECEIPT.json', receipt)
    return {'counts': rec['counts'], 'elapsed_seconds': rec['elapsed_seconds'],
            'peak_working_set_bytes': rec['peak_working_set_bytes'], 'out_dir': str(out_dir)}


def keys_filter(pairs):
    want = {(a, pd.Timestamp(b)) for a, b in pairs}
    return lambda e: e[[(a, pd.Timestamp(b)) in want for a, b in zip(e.firm_id, e.cutoff)]]
