"""TD2 v2 private readers/writers. Raw parquet access stays here and in decoder_v3.

Nothing runs on import. Evaluation-era returns require an admission token that only
evaluation_v3.admit() can create after the hash-bound clearance check.
"""
import contextlib
import json
import os
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import decoder_v3 as dec
from runtime_v3 import OUT, TD1, guard, sha, new_json
from spec_v2 import NAMES, MAX_LABEL_END

FEATURES = OUT / 'features_v2'
EVIDENCE = OUT / 'evidence_v2'
COVERAGE = OUT / 'coverage_v2'
NEWS = OUT / 'news_scores'
NEWS_EVAL_RECEIPT = OUT / 'NEWS_EVAL_v2.json'
_evaluation_token = None

read_frame = dec.read_frame
json_read = dec.json_read
LABEL_COLUMNS = ['y', 'label_end', 'fit_included', 'selection_included', 'refit_included']

def manifest_features(year):
    """Feature-only projection of the frozen TD1 manifest (hash checked by decoder_v3)."""
    if not 2015 <= year <= 2024:
        raise ValueError('Manifest years 2015-2024 only')
    return dec.manifest(year, labels=False)

def manifest_labels(year):
    """Development labels; decoder_v3 refuses years after 2022 before any read."""
    return dec.manifest(year, labels=True)

def prepared_check():
    prepared = json_read(OUT / 'PREPARED_v1.json')
    for p, h in prepared['source_sha256'].items():
        if sha(p) != h:
            raise ValueError('Prepared source changed: ' + p)
    return prepared

def graph_sources():
    """Sealed October graph, FactSet links and FSB5 sensitivity links (all <=2024-12-31)."""
    prepared_check()
    graph = read_frame(OUT / 'sources/graph_october_sealed.parquet',
                       ['id', 'end', 'status', 'typ', 'src', 'tgt', 'rownum'], 'date', '1900-01-01',
                       MAX_LABEL_END).rename(columns={'date': 'start'})
    links = read_frame(OUT / 'sources/factset_links.parquet', ['entity', 'permno', 'end'], 'date',
                       '1900-01-01', MAX_LABEL_END).rename(columns={'date': 'start'})
    suspects = {tuple(x) for x in json_read(OUT / 'sources/identity_sensitivity_v1.json')['links']}
    return graph, links, suspects

def build_eval_news():
    """Feature-only news scores for 2023 and 2024 with the frozen development identities."""
    if NEWS_EVAL_RECEIPT.exists():
        raise FileExistsError('Evaluation-era news scores already built')
    prepared_check()
    devmap = json_read(OUT / 'sources/reuters_identity_development_v1.json')['mapping']
    seen = set(); counts = {}
    for y in range(2014, 2025):
        guard(); p = NEWS / f'year_{y}.parquet'
        if not p.exists():
            if y < 2023:
                raise PermissionError('Development news partition missing: ' + str(p))
            dec.build_news(y, devmap, seen)
        f = read_frame(p, ['alt_id'], 'date', f'{y}-01-01', f'{y}-12-31')
        seen.update(f.alt_id); counts[str(y)] = len(f)
    hashes = {str(NEWS / f'year_{y}.parquet'): sha(NEWS / f'year_{y}.parquet') for y in (2023, 2024)}
    new_json(NEWS_EVAL_RECEIPT, {'news_counts': counts, 'news_sha256': hashes,
        'identity': 'frozen development mapping sources/reuters_identity_development_v1.json (keys <=2022)',
        'labels_read': False})
    return counts

def news_check(year):
    if year >= 2023:
        receipt = json_read(NEWS_EVAL_RECEIPT)
        for p, h in receipt['news_sha256'].items():
            if sha(p) != h:
                raise ValueError('Evaluation-era news changed: ' + p)

def news_for_year(year):
    news_check(year)
    return dec.news_for_year(year)

week_stories = dec.week_stories

@contextlib.contextmanager
def parquet_stream(final, schema):
    """Exclusive partial file renamed to the final name only after a clean close.

    Partial files from interrupted runs are left in place (no deletion) and never read.
    """
    final = Path(final).resolve(); guard()
    if not final.is_relative_to(OUT.resolve()):
        raise ValueError('Private output root')
    if final.exists():
        raise FileExistsError(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + f'.partial_{uuid.uuid4().hex}')
    stream = partial.open('xb')
    writer = pq.ParquetWriter(stream, schema, compression='zstd')
    try:
        yield writer
        writer.close(); stream.close()
    except BaseException:
        try:
            writer.close()
        finally:
            stream.close()
        raise
    if final.exists():
        raise FileExistsError(final)
    os.rename(partial, final)

def write_frame(path, frame):
    dec.write_frame(path, frame)

def read_features(year, columns, root=None):
    root = FEATURES if root is None else Path(root)
    return read_frame(root / f'year_{year}.parquet', columns, 'origin_session', f'{year}-01-01', f'{year}-12-31')

def read_evidence(year, columns, root=None):
    root = EVIDENCE if root is None else Path(root)
    return read_frame(root / f'year_{year}.parquet', columns, 'origin_session', f'{year}-01-01', f'{year}-12-31')

def schema_names(path):
    return pq.ParquetFile(path).schema_arrow.names

def admitted_returns(td1_root, year, token):
    """CRSP daily returns for evaluation labels; refuses without the admission token."""
    if token is None or token is not _evaluation_token:
        raise PermissionError('No evaluation admission')
    if year not in (2023, 2024):
        raise PermissionError('Evaluation years only')
    cols = ['date', 'permno', 'ret', 'share_type']
    td1_root = Path(td1_root)
    f = read_frame(td1_root / f'inputs/crsp_{year}.parquet', cols, 'date', f'{year}-01-01', f'{year}-12-31')
    if year == 2023:
        f = pd.concat([f, read_frame(td1_root / 'inputs/crsp_2024.parquet', cols, 'date', '2024-01-01',
                                     '2024-01-15')], ignore_index=True)
    return f

def baseline_frame(td1_root, year):
    return read_frame(Path(td1_root) / f'predictions_v2/year_{year}.parquet', ['firm_id', 'cutoff', 'b'],
                      'origin_session', f'{year}-01-01', f'{year}-12-31')
