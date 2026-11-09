"""Frozen word tables for masker v2 (private outputs; built once, hash-bound in the v2 freeze).

COMMON_WORDS_v2.json: lower-case words (letters only, >= 2) and how often each is written in lower case in the
headlines and texts of the TD4 2021-2022 selection article store (archive rows dated 2020-12-20..2022-12-31). A word
written in lower case at least td4v2_mask.COMMON_MIN times is a common English word for the v2 rules.
CRSP_WORD_DF_v2.json: for every word of a CRSP company name (namedt <= the seal), the number of distinct PERMNOs whose
names carry it (the non-first-word distinctiveness rule). One table per seal year.
No label, return or provider output is read.
"""
import hashlib
import json
import re
import csv
from collections import Counter
from pathlib import Path

import td4_spec as T
import td4_env as E

WORDS_DIR = T.OUT / 'tables_v2'
COMMON_FILE = WORDS_DIR / 'COMMON_WORDS_v2.json'
STORE_2122 = T.OUT / 'inputs' / 'article_store_select_2122.parquet'
_LOWER = re.compile(r"(?<![A-Za-z0-9'’\-])([a-z]{2,30})(?![A-Za-z0-9'’])")


def build_common(store=STORE_2122, out=COMMON_FILE, min_keep=3):
    import pyarrow.parquet as pq
    import td3_runtime as R
    if Path(out).exists():
        return json.loads(Path(out).read_text(encoding='utf-8'))['meta']
    pf = pq.ParquetFile(store)
    cnt, rows, dmax = Counter(), 0, ''
    for b in pf.iter_batches(batch_size=4096, columns=['date', 'headline', 'body_short'], use_threads=False):
        d = b.to_pydict()
        for day, h, t in zip(d['date'], d['headline'], d['body_short']):
            dmax = max(dmax, str(day)[:10])
            cnt.update(_LOWER.findall((h or '') + '\n' + (t or '')))
            rows += 1
        E.guard()
    if dmax > '2022-12-31':
        raise PermissionError('store row after 2022-12-31')
    words = {w: c for w, c in sorted(cnt.items()) if c >= min_keep}
    meta = {'source': str(store), 'source_sha256': R.sha(store), 'rows': rows, 'max_date': dmax,
            'words_kept': len(words), 'min_keep': min_keep, 'rule': 'lower-case whole words, letters only, 2-30'}
    R.new_json(out, {'meta': meta, 'counts': words})
    return meta


def load_common(path=COMMON_FILE, min_count=None):
    import td4v2_mask as MV
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    k = MV.COMMON_MIN if min_count is None else min_count
    return frozenset(w for w, c in d['counts'].items() if c >= k), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def df_file(last_day):
    return WORDS_DIR / f'CRSP_WORD_DF_v2_{last_day[:4]}.json'


def build_df(last_day):
    """Distinct PERMNOs per CRSP name word (both CRSP name files, namedt <= last_day)."""
    import td3_runtime as R
    import td3_mask as M
    out = df_file(last_day)
    if out.exists():
        return json.loads(out.read_text(encoding='utf-8'))['meta']
    by_word = {}
    for path in T.CRSP_NAME_FILES:
        with Path(path).open(encoding='utf-8-sig', newline='') as fh:
            for r in csv.DictReader(fh):
                day = (r.get('namedt') or '')[:10]
                if not day or day > last_day or not r.get('comnam'):
                    continue
                try:
                    p = int(float(r['permno']))
                except (TypeError, ValueError):
                    continue
                for w in set(M.normalize(r['comnam'])):
                    by_word.setdefault(w, set()).add(p)
    df = {w: len(v) for w, v in sorted(by_word.items())}
    meta = {'last_day': last_day, 'files': [str(p) for p in T.CRSP_NAME_FILES],
            'files_sha256': [R.sha(p) for p in T.CRSP_NAME_FILES], 'words': len(df)}
    R.new_json(out, {'meta': meta, 'df': df})
    return meta


def load_df(last_day):
    p = df_file(last_day)
    return json.loads(p.read_text(encoding='utf-8'))['df'], hashlib.sha256(p.read_bytes()).hexdigest()
