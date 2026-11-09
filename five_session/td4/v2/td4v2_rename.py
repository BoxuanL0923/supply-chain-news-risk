"""2025-only names: population check of the 2025 inputs against the CRSP CIZ 2025 name file (td4v2_names2025.CIZ).

A 2025-only name or ticker of a PERMNO is a CIZ 2025 issuer name or ticker that the local CRSP name files (namedt
<= 2024-12-31) do not carry for it. check_inputs() reports, for every 2025 input, whether such a name (its TD3 name
variants of two or more words, or a distinctive single word) or ticker (3-6 letters, not an English word, upper case,
whole word) of the
input's focal or a shown firm is readable in the articles. td4v2_verify2 withholds those inputs for every arm alike.
This replaces the earlier post-WRDS withholding rule (coordinator, 2026-11-06 19:30Z): the CIZ names are now an alias
source of the 2025 build, so this check should find nothing. No outcome or provider output is used.
"""
import csv
import json
import re
from pathlib import Path

import td4_spec as T
import td4_env as E


def _local_names():
    seen = {}
    for path in T.CRSP_NAME_FILES:
        with Path(path).open(encoding='utf-8-sig', newline='') as fh:
            for r in csv.DictReader(fh):
                d = (r.get('namedt') or '')[:10]
                if not d or d > '2024-12-31':
                    continue
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                s = seen.setdefault(p, (set(), set()))
                if r.get('comnam'):
                    s[0].add(' '.join(r['comnam'].upper().split()))
                if r.get('ticker'):
                    s[1].add(r['ticker'].strip().upper())
    return seen


def new_2025_names():
    """{permno: (2025-only names, 2025-only tickers)} from the CIZ file."""
    import td4v2_names2025 as N25
    rows, _ = N25.load_ciz()
    local = _local_names()
    out = {}
    for p, rs in rows.items():
        ln, lt = local.get(p, (set(), set()))
        names = {n for _, n, _ in rs if n and n not in ln}
        ticks = {t for _, _, t in rs if t and t not in lt}
        if names or ticks:
            out[p] = (names, ticks)
    return out


def _patterns(names, tickers, english):
    import td3_mask as M
    pats = []
    for c in names:
        for v in M.name_variants(c):
            letters = re.sub(r'[^A-Za-z0-9]', '', v)
            if (' ' in v and len(letters) >= 5) or (' ' not in v and M._distinctive(v.replace('.', ''), english, 5)):
                pats.append(re.compile(M.variant_regex(v), re.I))
    for t in tickers:                            # 3-6 letters, not an English word (a 2-letter or word ticker is
        if re.fullmatch(r'[A-Z]{3,6}', t) and t.lower() not in english:      # mostly ordinary upper-case text)
            pats.append(re.compile(r'(?<![A-Za-z0-9.$])' + re.escape(t) + r'(?![A-Za-z0-9])'))
    return pats


def check_inputs(d, english):
    """{'withheld_keys': [...], 'hits': [...], 'permnos_with_2025_only_names': n} for the 2025 build folder d."""
    E.paths()
    import td3_runtime as R
    import td4v2_scan as SC
    d = Path(d)
    new = new_2025_names()
    pats = {p: _patterns(v[0], v[1], english) for p, v in new.items()}
    wanted = set(pats)
    firms_by_key = {}                    # streamed (memory): only the inputs whose focal or shown firm has a 2025 name
    with (d / 'INPUT_META_COMPACT.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                firms = {int(m['focal_permno'])} | {int(s['permno']) for s in m['shown']}
                if firms & wanted:
                    firms_by_key[m['key']] = frozenset(firms & wanted)
    withheld, hits = set(), []
    with (d / 'INPUTS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            todo = firms_by_key.get(r['key'])
            if not todo:
                continue
            arts = '\n'.join(h + '\n' + b for _, h, b in SC.articles(r['input_text']))
            for p in todo:
                for rx in pats[p]:
                    mm = rx.search(arts)
                    if mm:
                        withheld.add(r['key'])
                        hits.append({'key': r['key'], 'permno': p, 'match': mm.group(0)})
                        break
    return {'permnos_with_2025_only_names': len(new), 'withheld_keys': sorted(withheld), 'hits': hits}
