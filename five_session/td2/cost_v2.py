"""DeepSeek ordering-call token and cost ESTIMATE from eligible firm-weeks with news and masked prompts.

No provider call, SDK, credential or network access. Article text is never placed in a prompt
here: each article is replaced by 'x' repeated to its measured truncated length.
"""
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import io_v2 as io
from runtime_v3 import OUT, guard, new_json, peak, sha
from spec_v2 import NAMES, MENU, HOPS, VERSION

LEDGER = Path('D:/OptResearch_data/sc_risk/o4_llm_loop/deepseek_costs.csv')
PRICE_SOURCE = ('Project memory note deepseek-usage.md (official pricing page checked by the project on '
                '2026-10-28): deepseek-v4-flash off-peak USD per million tokens 0.003 cache hit, 0.15 cache '
                'miss, 0.60 output; peak hours cost twice off-peak. Not re-verified here (no network).')
PRICES = {'off_peak': {'input_miss': 0.15, 'input_hit': 0.003, 'output': 0.60},
          'peak': {'input_miss': 0.30, 'input_hit': 0.006, 'output': 1.20}}
DOC_TOKENS_PER_CHAR = 0.30  # commonly cited DeepSeek guidance for English text; upper bound here
SAMPLE = 400
ANSWER = json.dumps({'ordering': NAMES})

def header(menu=MENU):
    lines = ["Task: order the frozen firm-characteristic menu by relevance to the focal firm's risk over the "
             "next five trading sessions, using only the dated information below; all of it was available "
             "before the cutoff. Return JSON {\"ordering\": [...]} listing every menu name exactly once. Do not "
             "add, drop or repeat names. Make no causal claims.", 'Menu:']
    lines += [f"- {m['name']}: {m['economic_meaning']}; channel: {m['risk_channel']}" for m in menu]
    return '\n'.join(lines) + '\n'

def masked_prompt(chars, hops, articles, menu=MENU):
    """chars: {name: float or None}; hops: {h: {'graph': int, 'news': int, 'partners': [(label, roles, n_mid)]}};
    articles: [(date_str, partner_label, length)]. Returns the masked prompt string."""
    parts = [header(menu), 'Focal firm: FIRM_A. Cutoff: YYYY-MM-DD.', 'Characteristics as of cutoff: ' +
             ', '.join(f'{n}=' + ('missing' if v is None or not np.isfinite(v) else f'{v:+.4g}') for n, v in chars.items())]
    for h in HOPS:
        info = hops.get(h, {'graph': 0, 'news': 0, 'partners': []})
        shown = '; '.join(f'{lab} (roles {">".join(roles)}; intermediate firms {mid})' for lab, roles, mid in info['partners'])
        parts.append(f"Exact hop {h}: {info['graph']} linked firms, {info['news']} with news. Shown: {shown or 'none'}")
    parts.append('Articles (masked):')
    parts += [f'[{d}] {lab}: ' + 'x' * int(n) for d, lab, n in articles]
    return '\n'.join(parts) + '\n'

def ledger_calibration(path=LEDGER):
    """tokens/byte from ledger rows whose prompt file is Markdown (file SIZE only, content not read)."""
    ratios = []; flash = []
    if not path.exists():
        return {'available': False}
    with path.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            t = int(float(r['prompt_tokens'] or 0)); p = r['prompt_file']
            if t > 0 and p.endswith('.md') and os.path.exists(p):
                ratios.append(t / os.path.getsize(p))
            if 'flash' in r['model'] and t > 0:
                flash.append(int(float(r['completion_tokens'] or 0)))
    return {'available': True, 'ledger_sha256': sha(path), 'markdown_rows': len(ratios),
            # The td2 folder has a local statistics.py, so the stdlib module is not used here.
            'tokens_per_byte_median': float(np.median(ratios)) if ratios else None,
            'flash_rows': len(flash), 'flash_completion_tokens_median': float(np.median(flash)) if flash else None}

def sample_keys(frame, n=SAMPLE):
    keyed = frame.assign(_k=[hashlib.sha256(f'TD2-cost|{a}|{pd.Timestamp(b)}'.encode()).hexdigest()
                             for a, b in zip(frame.firm_id, frame.cutoff)])
    return keyed.sort_values('_k').head(n).drop(columns='_k')

def article_lengths(ids, lo, hi):
    """Truncated text length (headline[:200]+body_short[:1000], earliest day per ID). Only lengths are kept."""
    out = {}; first = {}; wanted = list(ids)
    for batch in io.dec.parquet(io.dec.ARTICLE, ['alt_id', 'headline', 'body_short'], 'date', lo, hi):
        sub = batch[batch.alt_id.isin(wanted)]
        for r in sub.itertuples(index=False):
            day = r.date[:10]
            if r.alt_id not in first or day < first[r.alt_id]:
                first[r.alt_id] = day
                out[r.alt_id] = len((r.headline or '')[:200] + (r.body_short or '')[:1000])
        guard()
    return out

def call_counts(years):
    counts = {}
    cols = ['firm_id', 'cutoff', 'variant', 'has_news'] + ['missing_' + n for n in NAMES]
    cols += [f'E{h}_{n}' for h in HOPS for n in NAMES]
    for y in years:
        f = io.read_features(y, cols)
        p = f[f.variant == 'primary']; s = f[f.variant == 'fsb5_sensitivity']
        anychar = ~np.logical_and.reduce([p['missing_' + n].to_numpy() for n in NAMES])
        m = p.merge(s, on=['firm_id', 'cutoff'], suffixes=('', '_s'), validate='one_to_one')
        e = [f'E{h}_{n}' for h in HOPS for n in NAMES]
        differs = (m[e].to_numpy() != m[[c + '_s' for c in e]].to_numpy()).any(axis=1)
        counts[str(y)] = {'eligible_firm_weeks': int(len(p)), 'with_news': int(p.has_news.sum()),
                          'with_news_and_characteristic': int((p.has_news.to_numpy() & anychar).sum()),
                          'sensitivity_context_differs': int(differs.sum())}
    return counts

def prompt_sizes(years=(2021, 2022)):
    cols = ['firm_id', 'cutoff', 'variant', 'has_news'] + ['C_' + n for n in NAMES] + ['missing_' + n for n in NAMES]
    cols += [f'H{h}_{c}' for h in HOPS for c in ('graph_partners', 'news_partners')]
    frames = [io.read_features(y, cols) for y in years]
    f = pd.concat(frames, ignore_index=True); f = f[(f.variant == 'primary') & f.has_news]
    s = sample_keys(f); keys = set(zip(s.firm_id, pd.to_datetime(s.cutoff)))
    ev = []
    for y in years:
        for batch in io.dec.parquet(io.EVIDENCE / f'year_{y}.parquet', ['firm_id', 'cutoff', 'variant', 'hop',
                'partner', 'reporting_roles', 'intermediate_nodes', 'article_ids'], 'origin_session',
                f'{y}-01-01', f'{y}-12-31'):
            b = batch[batch.variant == 'primary']
            b = b[[k in keys for k in zip(b.firm_id, pd.to_datetime(b.cutoff))]]
            if len(b): ev.append(b)
    ev = pd.concat(ev, ignore_index=True)
    ids = set(a for lst in ev.article_ids for a in lst)
    lengths = article_lengths(ids, f'{min(years) - 1}-12-20', f'{max(years)}-12-31')
    sizes = {'all_articles': [], 'one_article_per_partner': []}; nart = {'all_articles': [], 'one_article_per_partner': []}
    for r in s.itertuples(index=False):
        e = ev[(ev.firm_id == r.firm_id) & (pd.to_datetime(ev.cutoff) == pd.Timestamp(r.cutoff))].sort_values(['hop', 'partner'])
        chars = {n: (None if getattr(r, 'missing_' + n) else float(getattr(r, 'C_' + n))) for n in NAMES}
        hops = {h: {'graph': int(getattr(r, f'H{h}_graph_partners')), 'news': int(getattr(r, f'H{h}_news_partners')),
                    'partners': []} for h in HOPS}
        allart = {}; onearts = []
        for j, w in enumerate(e.itertuples(index=False)):
            lab = f'P{w.hop}.{j + 1}'
            hops[int(w.hop)]['partners'].append((lab, list(w.reporting_roles), len(w.intermediate_nodes)))
            arts = sorted(w.article_ids)
            for a in arts:
                allart.setdefault(a, lab)
            if arts:
                onearts.append((arts[0], lab))
        for policy, chosen in [('all_articles', list(allart.items())), ('one_article_per_partner', onearts)]:
            arts = [('YYYY-MM-DD', lab, lengths.get(a, 1200)) for a, lab in chosen]
            sizes[policy].append(len(masked_prompt(chars, hops, arts).encode('utf-8'))); nart[policy].append(len(arts))
    return {'sample_rows': int(len(s)), 'sample_years': list(years), 'articles_with_measured_length': len(lengths),
            'articles_needed': len(ids), 'missing_length_assumed_1200_chars': len(ids - set(lengths)),
            'prompt_bytes': {p: {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)),
                                 'max': int(max(v))} for p, v in sizes.items()},
            'articles_per_prompt': {p: {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'max': int(max(v))}
                                    for p, v in nart.items()},
            'header_bytes': len(header().encode('utf-8'))}

def dollars(calls, in_tokens, out_tokens, cached_tokens, price):
    miss = max(in_tokens - cached_tokens, 0)
    return calls * (miss * price['input_miss'] + cached_tokens * price['input_hit'] + out_tokens * price['output']) / 1e6

def estimate_table(calls, mean_bytes, header_bytes, ratios, outputs):
    rows = []
    for rname, ratio in ratios.items():
        tin = mean_bytes * ratio; thead = header_bytes * ratio
        for oname, tout in outputs.items():
            for pname, price in PRICES.items():
                for cname, cached in [('no_cache', 0.0), ('header_cached', thead)]:
                    rows.append({'tokens_per_byte': rname, 'output': oname, 'price_window': pname, 'cache': cname,
                        'input_tokens_per_call': tin, 'output_tokens_per_call': tout,
                        'total_input_tokens': calls * tin, 'total_output_tokens': calls * tout,
                        'usd': dollars(calls, tin, tout, cached, price)})
    return rows

def run():
    start = time.perf_counter(); guard()
    target = OUT / 'TOKEN_COST_ESTIMATE_v2.json'
    if target.exists():
        raise FileExistsError(target)
    counts = call_counts(range(2015, 2025))
    sizes = prompt_sizes()
    cal = ledger_calibration()
    ratios = {'measured_ledger_median': cal.get('tokens_per_byte_median') or 0.25, 'doc_0.30_per_char': DOC_TOKENS_PER_CHAR}
    outputs = {'non_thinking_answer_only': len(ANSWER.encode('utf-8')) * DOC_TOKENS_PER_CHAR,
               'thinking_ledger_flash_median': float(cal.get('flash_completion_tokens_median') or 0)}
    sets = {'selection_2021_2022': ['2021', '2022'], 'evaluation_2023_2024': ['2023', '2024'],
            'selection_plus_evaluation': ['2021', '2022', '2023', '2024']}
    tables = {}
    for sname, ys in sets.items():
        calls = sum(counts[y]['with_news_and_characteristic'] for y in ys)
        tables[sname] = {'calls': calls, 'sensitivity_extra_calls': sum(counts[y]['sensitivity_context_differs'] for y in ys),
            'by_policy': {p: estimate_table(calls, sizes['prompt_bytes'][p]['mean'], sizes['header_bytes'], ratios, outputs)
                          for p in sizes['prompt_bytes']}}
    new_json(target, {'version': VERSION, 'estimate_only': True, 'actual_provider_calls': 0,
        'unit': 'one DeepSeek ordering call per primary firm-week with news and at least one eligible characteristic',
        'counts_by_year': counts, 'masked_prompt_measurement': sizes, 'tokenizer_calibration': cal,
        'tokens_per_byte_used': ratios, 'output_tokens_per_call_used': outputs, 'answer_template': ANSWER,
        'prices_usd_per_million': PRICES, 'price_source': PRICE_SOURCE, 'estimates': tables,
        'excluded': 'retries, malformed-answer repeats, Jev calls, the fsb5 sensitivity calls (counted separately), '
                    'provider-side framing tokens; prompt template is a sizing draft, not the frozen provider question',
        'elapsed_seconds': time.perf_counter() - start, **peak()})
    print('TOKEN_COST_ESTIMATE_V2', str(target), json.dumps({k: v['calls'] for k, v in tables.items()}), flush=True)
