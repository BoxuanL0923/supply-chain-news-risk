"""Population checks and audit samples for the masker-v2 builds (td4v2_build2; every period).

population(period): over every input of the build,
  (a) the v1/TD3 text checks (header, firm id, article count, year, year glued to a time, URL, numeric date, story
      fields, glued role token, token-hyphen-digit; td4_verify regexes), plus the v2 story-field stubs and clock times
      with seconds (TD4C B1);
  (b) the independent v2 scanner (td4v2_scan) on every article of the sent text.
An input with an (a) problem or a hit of an exact scanner class (td4v2_scan.WITHHOLD_CLASSES) is withheld from all three
providers alike (fallback). Other scanner hits are counted by class and kept for the reviewers (SCAN_HITS_M2.jsonl,
private). Executive names next to a token are counted (disclosed limitation).
audit(period): the hash-ranked audit sample (full metadata, at most 300 inputs): every article is re-masked from its
raw text with MaskerV2 (equality required), the strict v2 residual and v2 date residual are re-run, and the TD3 scanner
and the v2 scanner hits are written for the reviewers.
"""
import json
import re
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E


def _setup(period):
    import td4v2_build2 as B2
    import td3_build as B
    import td4v2_words as W
    period = '2122' if str(period) == '2122' else int(period)
    years = B2.YEARS[period]
    info = B2.install(years)
    english = B.load_english(Path('C:/Users/Boxuan/AppData/Roaming/nltk_data/corpora/words/en'))
    common, _ = W.load_common()
    tables = json.loads(B2.TABLES_V2.read_text(encoding='utf-8'))
    return period, years, B2.PERIOD_DIR[period], english, common, tables, info


def population(period, d=None):
    E.paths()
    import td3_runtime as R
    import td3_build as B
    import td3_mask
    import td3_spec as S3
    import td4_verify as V1
    import td4v2_mask as MV
    import td4v2_scan as SC
    period, years, d0, english, common, tables, info = _setup(period)
    d = Path(d or d0)
    if not (d / 'BUILD.json').exists():
        raise PermissionError('build not finished: ' + str(d))
    role = B.ROLE_TOKEN.pattern
    glued = re.compile(r'[^\W_]' + role + '|' + role + r'[^\W_]')
    remnant = re.compile(role + r'[\-–]\d')
    metas = {}
    perms = set()
    for m in R.read_jsonl(d / 'INPUT_META_COMPACT.jsonl'):
        metas[m['key']] = m
        perms |= set(SC.tok_map(m).values())
    terms = SC.build_terms(perms, f'{max(years)}-12-31', english, common, tables.get('renames'))
    problems, scan_classes, bad, n, articles = Counter(), Counter(), set(), 0, 0
    execs = 0
    sfx = '' if not any((d / n).exists() for n in ('POPULATION_CHECK_M2.json', 'SCAN_HITS_M2.jsonl')) else \
        '_' + R.utc().replace(':', '')                 # re-run (or after a stopped run): new files
    hits_w = R.JsonlWriter(d / f'SCAN_HITS_M2{sfx}.jsonl', R.Redactor())
    with (d / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            text, key = r['input_text'], r['key']
            meta = metas.get(key)
            if not text.startswith(f'FROZEN INPUT ({S3.INPUT_VERSION})'):
                problems['header'] += 1
                bad.add(key)
            if meta is None or meta['firm_id'] in text:
                problems['no_compact_meta' if meta is None else 'firm_id_in_input'] += 1
                bad.add(key)
                continue
            body = '\n'.join(x for x in text.split('\n') if x.startswith('Headline: ') or x.startswith('Text: '))
            k = text.count('\nArticle ')
            articles += k
            if k != len(meta['articles']):
                problems['article_count_differs_from_meta'] += 1
                bad.add(key)
            for name, rx, src in (('year_in_article_text', V1.YEAR, body), ('year_glued_to_time', V1.YEAR_TIME, body),
                                  ('url_in_article_text', td3_mask.URL_RESIDUAL, body),
                                  ('numeric_date_in_input', V1.NUM_DATE, text),
                                  ('story_fields_in_input', V1.STORY, text),
                                  ('story_stub_v2', MV.STORY_STUB, body), ('clock_seconds_v2', MV.CLOCK_SECONDS, body),
                                  ('glued_role_token', glued, body), ('token_hyphen_digit', remnant, body)):
                if rx.search(src):
                    problems[name] += 1
                    bad.add(key)
            hs = SC.scan_input(text, meta, terms)
            cls = {h['class'] for h in hs}
            scan_classes.update(cls)
            if 'S15_executive_next_to_token' in cls:
                execs += 1
            if cls & set(SC.WITHHOLD_CLASSES):
                bad.add(key)
                for c in cls & set(SC.WITHHOLD_CLASSES):
                    problems['scan_' + c] += 1
            for h in hs:
                hits_w.write({'key': key, **h})
            if n % 2000 == 0:
                E.guard()
    hits_w.close()
    names2025 = None
    if str(period) == '2025':                          # 2025-only CIZ names/tickers of focal or shown firms
        metas.clear()                                  # memory: the 2025 check streams its own metadata
        terms = None
        import td4v2_rename as RN
        chk = RN.check_inputs(d, english)
        bad |= set(chk['withheld_keys'])
        problems['scan_S16_2025_only_name_readable'] = len(chk['withheld_keys'])
        R.new_json(d / f'NAMES2025_CHECK_M2{sfx}.json', chk)               # private (matched names)
        names2025 = {'permnos_with_2025_only_names': chk['permnos_with_2025_only_names'],
                     'inputs_withheld': len(chk['withheld_keys'])}
    out = {'period': str(period), 'dir': str(d), 'inputs': n, 'articles': articles, 'problems': dict(problems),
           'names2025_check': names2025,
           'withheld_inputs': len(bad), 'scan_inputs_by_class': dict(sorted(scan_classes.items())),
           'executive_next_to_token_inputs': execs, 'mask_v2': info,
           'rule': 'an input with a text problem or an exact-class scanner hit is withheld from all three providers '
                   'alike and takes the fallback; other scanner hits are candidates for the reviewers',
           'withhold_classes': list(SC.WITHHOLD_CLASSES), **R.peak()}
    R.new_json(d / f'POPULATION_CHECK_M2{sfx}.json', out)
    R.new_json(d / f'WITHHELD_POPULATION_M2{sfx}.json', {'dir': str(d), 'withheld_keys': sorted(bad),
                                                   'problems': dict(problems)})
    return out


def audit_rank_m2(firm_id, cutoff, period):
    import hashlib
    import pandas as pd
    return hashlib.sha256(f'TD4-audit-m2|{period}|{firm_id}|{pd.Timestamp(cutoff).isoformat()}'.encode()).hexdigest()


def audit(period, d=None, max_n=300):
    """Fresh hash-ranked audit sample (rank SHA256('TD4-audit-m2|<period>|firm_id|cutoff')) of the build's AUDIT_META
    records; re-mask, residuals and both scanners."""
    E.paths()
    import td3_runtime as R
    import td3_build as B
    import td3_scan
    import td3_mask
    import td4_inputs as I
    import td4v2_mask as MV
    import td4v2_scan as SC
    period, years, d0, english, common, tables, info = _setup(period)
    d = Path(d or d0)
    I.set_period(years)
    ranks = []
    with (d / 'AUDIT_META.jsonl').open('rb') as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                m = json.loads(line.decode('utf-8'))
                ranks.append((audit_rank_m2(m['firm_id'], m['cutoff'], period), off))
    keep = sorted(off for _, off in sorted(ranks)[:max_n])
    metas = []
    with (d / 'AUDIT_META.jsonl').open('rb') as f:
        for off in keep:
            f.seek(off)
            metas.append(json.loads(f.readline().decode('utf-8')))
    src, _ = I.sources(years, I.store_path('x'))
    if B.install_tables(src) != T.TD3_TABLES_SHA256:          # the frozen TD3 table, as in the build
        raise ValueError('Frozen TD3 mask table differs')
    ricmap = B.ric_map(src.ric_identity)
    hits3, summary3, _adj = td3_scan.scan_build(metas, src.crsp_names, src.factset, ricmap, english)
    keys = ('names', 'common', 'tickers', 'tickers_extra', 'rics', 'embedded', 'fuzzy', 'products', 'code_prefixes',
            'near', 'unit_prefixes', 'v2')
    problems = Counter()
    perms = set()
    for m in metas:
        perms |= {int(a['permno']) for a in m['aliases'].values()}
    terms = SC.build_terms(perms, f'{max(years)}-12-31', english, common, tables.get('renames'))
    w = R.JsonlWriter(d / f'AUDIT_SCAN_HITS_M2_{R.utc().replace(":", "")}.jsonl', R.Redactor())
    for h in hits3:
        w.write({'scanner': 'td3_scan', **h})
    v2hits = Counter()
    for m in metas:
        order = ['[FOCAL]'] + [s['token'] for s in m['shown']] + [x['token'] for x in m['links']]
        order += [t for t in m['aliases'] if t not in order]
        entries = [(t, {k: m['aliases'][t].get(k, {} if k == 'products' else []) for k in keys})
                   for t in order if t in m['aliases']]
        tm = {t: int(a['permno']) for t, a in m['aliases'].items()}
        for x in m['articles']:
            mk = MV.MaskerV2(B.article_entries(entries, x['owner_terms'], x.get('article_extras')), english,
                             own=x['tokens'])
            h2, b2, *_ = mk.mask_article(x['headline_raw'], x['body_raw'])
            if (h2, b2) != (x['headline'], x['body']):
                problems['remask_differs'] += 1
            mk.set_active_acronyms(x.get('active_acronyms'))
            if mk.residual(x['headline']) + mk.residual(x['body']):
                problems['strict_residual_v2'] += 1
            if MV.date_residuals_v2(x['headline']) + MV.date_residuals_v2(x['body']):
                problems['date_residual_v2'] += 1
            for field in ('headline', 'body'):
                for cls, term, ctx in SC.scan_text(x[field], tm, terms, set(x['tokens'])):
                    v2hits[cls] += 1
                    w.write({'scanner': 'td4v2_scan', 'key': m['key'], 'alt_id': x['alt_id'], 'field': field,
                             'class': cls, 'term': term, 'context': ctx})
    w.close()
    out = {'period': str(period), 'audit_inputs': len(metas), 'audit_meta_records': len(ranks),
           'articles': sum(len(m['articles']) for m in metas), 'rank': "SHA256('TD4-audit-m2|<period>|firm_id|cutoff')",
           'td3_scan_summary': summary3, 'td3_scan_hits': len(hits3), 'v2_scan_hits_by_class': dict(v2hits),
           'problems': dict(problems), 'mask_v2': info,
           'note': 'scan hits are candidates for the reviewers to adjudicate (TD3 practice); no LLM labels them'}
    R.new_json(d / f'AUDIT_SAMPLE_M2_{period}_{R.utc().replace(":", "")}.json', out)
    return out
