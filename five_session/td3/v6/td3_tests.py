"""TD3 v6 (TD3F) offline qualification: synthetic fixtures and mocked providers only.

v6 (review TD3C5 D1): tests for the initialism-unit rule (masker, residual, scanner, limitation example), the v6 paths
and names, and the must_read field of the manual-read record.

v5 (review TD3C4): tests for D1 (glued tokens take the whole word; house numbers before a street phrase; glued and
number-neighbour sweep rows; verify re-masks from the stored raw text), D2 (firm codes from tickers, initialisms,
leading letters and the table, with the stop lists; near names from the name histories; the scanner's firm-code,
near-prefix, glued and house-number checks), R1 (every invented example of the limitation text), N-a (ordinal
dates), N-b (press-release IDs), N-c (pass order), N-e (25 GiB guard), N-f (invented test strings only) and the v5
manual-read rule.

v4 (review TD3C3): tests for E1 (firm-owned product, unit, prefix and joint-product phrases absorbed into the token,
acronyms in the same article, the residual and verify checks, the table schema and the adjacency sweep), E2 (month
typos and day-range remnants), N1 (person surnames masked), N2 (no place guard), N3 (fixed-phrase guards restricted
to index and price phrases), N4 (FactSet-only single-word variants must be distinctive), N5 (probability-sum
tolerance 0.03 = 6 options x 0.005) and N6 (invented names only: every firm-specific table is private, the tests
install invented tables, and the strings flagged by the reviews occur nowhere in the code).

Run through `python -B run_td3.py qualify`, which installs the offline runtime first (every socket event refused).
Fixtures use invented firms and texts and are written under D:/OptResearch_data/five_session/td3/fixtures/<uuid>/
(never deleted). No real provider, key, label or return is used, and every firm name, headline, story ID and text in
this file is invented (m2). v2 tests (C1-C5, R1-R5) are kept and adapted; v3 adds tests for D1 (issuer RICs and
siblings, embedded names, misspellings, the context guard, place words, bylines, the adjacency report and the v3
scanner), D2 (years glued to a time, generic short URLs), D3 (deepseek-flash with thinking disabled), D4 (one
provider failing), m1 (reasoning fields), m2 (no review-flagged strings in the code) and m4 (no re-ask after a model
refusal without a new clearance).
"""
import contextlib
import hashlib
import io
import json
import re
import socket
import sys
import unittest
import uuid
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import td3_spec as S
import td3_runtime as R
import td3_mask as M
import td3_build as B
import td3_scan as SC
import td3_providers as P
import td3_ledger as L
import td3_runner as RU
import td3_report as RP
import td3_freeze as FZ

RUN_ROOT = S.FIXTURES / uuid.uuid4().hex
SENT_DS = 'sk-SENTINEL-DEEPSEEK-0123456789abcdef'
SENT_JEV = 'ts-SENTINEL-JEV-fedcba9876543210'
PEAK_CLOCK = '2026-11-05T02:00:00Z'      # Thursday, inside the 01:00-04:00 UTC peak window
OFFPEAK_CLOCK = '2026-11-07T12:00:00Z'   # Saturday

MENU_TEXT = [('size', 'Log dollar market equity', 'Scale interacted with partner activity and disruption news'),
             ('cash', 'Cash divided by total assets', 'Liquidity buffer against partner payment or operating disruption'),
             ('cash_flow', 'Operating cash flow divided by assets', 'Internal funding capacity'),
             ('leverage', 'Short and long term debt divided by assets', 'Financing sensitivity to partner credit'),
             ('inventory', 'Inventory divided by total assets', 'Inventory exposure to partner shortage')]
MENU = [{'name': n, 'economic_meaning': m, 'risk_channel': c} for n, m, c in MENU_TEXT]
COUNTS = {'2021': 1000, '2022': 1000, '2023': 1200, '2024': 1300}

F1, P1, P2, P3, P4, P5, F2, P6, F3, F4, F5, F6 = (10001, 20001, 20002, 20003, 20004, 20005, 10002, 20006, 10003,
                                                   10004, 10005, 10006)
O1, C1 = pd.Timestamp('2021-03-05'), pd.Timestamp('2021-03-06 04:59:59')
O2, C2 = pd.Timestamp('2022-06-10'), pd.Timestamp('2022-06-11 03:59:59')
O3, C3 = pd.Timestamp('2023-01-06'), pd.Timestamp('2023-01-07 04:59:59')   # 2023 row inside the 2022 file
O4, C4 = pd.Timestamp('2022-12-30'), pd.Timestamp('2023-01-01 04:59:59')   # cutoff in 2023
# (permno, ticker, comnam, namedt, cusip). ZENTRIQ is P2's rename after the 2021 origin but before the 2022 seal.
NAMES_ROWS = [(F1, 'ZRBX', 'ZORBLAX INDUSTRIES INC', '2010-01-01', '11111101'),
              (F1, 'LTRN', 'LATERNAME CORP', '2025-03-01', '11111101'),
              (P1, 'QTL', 'QUINTRELL CORP', '2010-01-01', '22222201'),
              (P1, 'QTL', 'QUINTRELL NEWCO CORP', '2021-06-01', '22222201'),
              (P2, 'F', 'MAVRICK HOLDINGS INC', '2010-01-01', '33333301'),
              (P2, 'ZNTQ', 'ZENTRIQ CORP', '2021-09-01', '33333301'),
              (P3, 'BLND', 'BLENDOR TECHNOLOGIES CO', '2010-01-01', '44444401'),
              (P4, 'KPW', 'KAPPAWIRE CORP', '2010-01-01', '55555501'),
              (P5, 'OSTX', 'OSTRANIX PLC', '2010-01-01', '66666601'),
              (F2, 'VLTG', 'VELTRIDGE INC', '2010-01-01', '77777701'),
              (P6, 'NRVN', 'NORVANE CORP', '2010-01-01', '88888801')]
# (cusip, name, start_, FactSet id, ticker, ticker_exchange); the blank-CUSIP row of id FS1 is linked through the id.
FACTSET_ROWS = [('88888801', 'Nrvx Global Holdings, Inc.', '2015-01-01', 'FS8', 'NRVX', 'NRVX-NAS'),
                ('88888801', 'Postsealname Corp', '2023-06-01', 'FS8', 'PSTL', 'PSTL-NYS'),   # after the seal
                ('11111101', 'Zorblax Industries, Inc.', '2012-01-01', 'FS1', 'ZRBX', 'ZRBX-NAS'),
                ('', 'Zorblaxian Holdings', '2016-01-01', 'FS1', 'ZBXH', ''),
                ('11111101', 'Zorblax Frankfurt Listing', '2016-01-01', 'FS1', 'ZBXF', 'ZBXF-FRA')]
RICS = {'ZRBX.O': [F1, 201001], 'QTL.N': [P1, 201001], 'F.N': [P2, 201001], 'BLND.N': [P3, 201001],
        'KPW.N': [P4, 201001], 'OSTX.N': [P5, 201001], 'VLTG.O': [F2, 201001], 'NRVN.O': [P6, 201001],
        'BAD.N': [F3, 201001], 'EDGE.N': [F4, 201001]}
FORBIDDEN_NAMES = ['Zorblax', 'ZORBLAX', 'Quintrell', 'QUINTRELL', 'Mavrick', 'MAVRICK', 'Blendor', 'Kappawire',
                   'Ostranix', 'Veltridge', 'Norvane', 'Zentriq', 'Nrvx', 'Newco', 'Zorblaxian', 'Zorbcloud']
FORBIDDEN_TICKERS = ['ZRBX', 'QTL', 'F.N', 'BLND', 'KPW', 'OSTX', 'VLTG', 'NRVN', 'Zntq', 'ZNTQ', 'BLNDP', 'QZS']
TEXTS = {
    'A1': ('Quintrell Corp cuts orders from Zorblax',
           'NEW YORK, March 3 (Reuters) - Quintrell Corp (QTL.N) said on March 3, 2021 it would cut orders from '
           'Zorblax Industries Inc <ZRBX.O>, its main supplier. Quintrell\'s shares fell 3% while ZRBX slid. QUINTRELL '
           'CORP told investors (NYSE: QTL) that $QTL holders should expect lower demand. Mavrick Holdings (F.N) and '
           'Kappawire Corp also commented; a grade F rating was unchanged. Quintrell Zorbcloud Services grew and QZS '
           'margins rose. (Reporting by Jane Doe; Editing by John Roe)'),
    'A1_DUP': ('SECONDROW duplicate', 'SECONDROW same-day later physical row must not be used'),
    'A1b': ('Quintrell demand outlook', 'Quintrell Newco expects demand to recover; inventory at the Zorblaxian unit '
                                        'remains high.'),
    'A_ORIGIN': ('FUTUREMARKER origin day', 'FUTUREMARKER published on the origin day'),
    'A_FUT': ('FUTUREMARKER after origin', 'FUTUREMARKER published after the origin'),
    'A_OLD': ('FUTUREMARKER old', 'FUTUREMARKER published eight days before'),
    'A2': ('BRIEF-Mavrick Holdings warns on supply; Zntq Units', 'Mavrick Holdings Inc (F.N) warned of inventory '
           'shortages. Mavrick\'s chief executive said deliveries to Zorblax would slip, and Blendor Technologies Co was '
           'affected; the parent will be renamed Zentriq. Summit County officials agreed. Details: '
           'https://www.mavrick.com/news/zorblax-delay and ir@mavrick.com; link bit.ly/9Qwe12 STORY_DATE:[14/03/2021] '
           'STORY_TIME:[11:22:33] guidance FY22 and Q3\'21 [nZZ9TEST01]'),
    'A_LATE': ('FUTUREMARKER late', 'FUTUREMARKER available only after the cutoff'),
    'A3': ('FUTUREMARKER tagged late', 'FUTUREMARKER firm mapping starts in the origin month'),
    'A3b': ('Blendor expands capacity', 'Blendor Technologies expands capacity; BLND rose, the preferred <BLNDP.N> '
            'held and Kappawire said production would rise.'),
    'A5': ('Ostranix liquidity', 'Ostranix PLC (OSTX.N) arranged new financing; credit lines widened. Trial of '
           'OSTX-204 started at 12 Ostranix Way.'),
    'B6': ('Norvane orders', 'Norvane Corp (NRVN.O) raised orders from Veltridge Inc; VLTG gained. SUMMIT SAID '
           'Summit expects a summit with suppliers. Nrvx Global said Reuters reported it. Postsealname stays.'),
    'C_BAD': ('BADYEARMARKER', 'BADYEARMARKER 2023 article'),
    'D_EDGE': ('EDGEMARKER', 'EDGEMARKER article for a cutoff in 2023'),
}
# m2: SHA256 digests, lengths and first characters of the sample-derived strings that the TD3BC review found in the
# v2 tests and handoff; the strings themselves are not repeated anywhere in this repository.
M_FLAGGED = [('ee9dab4f8c930768a52260e97ae37ea0248589f1c4f2c5e6869a474e33a04740', 18, 'A'),
             ('6e6f2a127f77db1f08e5dffe0302a3645913602904cd51061c987fd7f7ce26a5', 9, 'T'),
             ('76d7042a1228d8395c7301d6f483af8f54f4bb0423f8c2b47e87823e8052819c', 24, 'P'),
             ('dc0c62b607ea8f5696148b560a4f6ef372992c064c24a1c1904325d9bd64a11c', 10, 'n'),
             ('42d600bb77013bab44133e51c3cc5be4fd2ff7a64266ff26c2ee7ff72fa17f6a', 7, '3'),
             ('e8fd04fc92fd98b153ffaa7855b277625a8e2d8bccd3db8861d81404f5f6f8fb', 10, 'n'),
             ('7748f5bdcc51ea0e27c6bb3230f8984a111992870b11da553149757488ea160d', 10, 'n'),
             ('506d173130b42e372ae5616927ed04d1af7ff83d3efba6d2ae34c2b7a189805e', 15, 'A'),
             ('f1256e1727938da2714113aad58850e356cf4a00245f313332a6903eb7bc31e3', 7, 'T'),
             ('3bd1d560e68fdbd54ed4403860878c5c650ed51be3273bd30aa0a059965ee378', 17, 'M'),
             ('39dcc7983651eb9b7d427bfed1e03916e38bffd68e30891c785c4fe6ee57f09b', 21, 'S'),
             ('31ec9681fdd01775d50a7919a57189c0d187e8a857ec95ce8082b9946d023565', 26, 'T'),
             ('c8ae9ced1aa78c4e7ce4db920bc01589c1663b0beb52c92cd31729732822b7cf', 32, 'A'),
             ('7b3e8b27b00a025514b9d91b5f6589312aa7ac81e329036562b3b944b6ccb7d7', 22, 'C')]


def firm_id(p):
    return hashlib.sha256(f'TD1-firm:{int(p)}'.encode()).hexdigest()[:24]


def write_parquet(path, frame, schema):
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    with R.new_bytes_handle(path) as f:
        pq.write_table(table, f)


def _csv(rows, cols):
    buf = io.StringIO()
    import csv
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    w.writerows(rows)
    return buf.getvalue()


def write_factset(path, rows):
    schema = pa.schema([('id', pa.string()), ('cusip', pa.string()), ('name', pa.string()), ('ticker', pa.string()),
                        ('ticker_exchange', pa.string()), ('start_', pa.date32())])
    cols = ['id', 'cusip', 'name', 'ticker', 'ticker_exchange', 'start_']
    write_parquet(path, pd.DataFrame([{'id': i, 'cusip': c, 'name': n, 'ticker': t, 'ticker_exchange': e,
                                       'start_': dt.date.fromisoformat(d)} for c, n, d, i, t, e in rows], columns=cols),
                  schema)


ENGLISH = frozenset({'dim', 'all', 'see', 'bet', 'eye', 'basis', 'in', 'the', 'pro', 'play', 'hot', 'spot', 'ox',
                     'sign', 'up', 'golden', 'state'})
_FIX = {}
# v4 (N6): tables are private; tests install invented tables only.
EMPTY_TABLES = {'schema': M.TABLE_SCHEMA, 'common_names': {}, 'firm_products': {}, 'fixed_phrases': {},
                'code_prefixes': {}}
FIXTURE_TABLES = {'schema': M.TABLE_SCHEMA, 'common_names': {}, 'fixed_phrases': {}, 'code_prefixes': {},
                  'firm_products': {'QUINTRELL': {'products': [{'phrase': 'Zorbcloud Services', 'acronym': 'QZS'}],
                                                  'source': 'invented fixture'},
                                    'OSTRANIX': {'products': [{'phrase': 'Way', 'capitalized': True}],
                                                 'source': 'invented fixture (v5 house number)'}}}
INVENTED_TABLES = {
    'schema': M.TABLE_SCHEMA,
    'code_prefixes': {'ZORBLEX ONLINE': {'prefixes': ['QX'], 'source': 'invented'}},
    'common_names': {'Q V B FINANCIAL': {'aliases': ['Quorn Valley Bank'], 'source': 'invented'}},
    'firm_products': {
        'ZORBLEX ONLINE': {'products': [{'phrase': 'Cloudrift Services'}, {'phrase': 'Parcel & Freight', 'acronym': 'ZPX'},
                                        {'phrase': 'Go', 'capitalized': True},
                                        {'phrase': 'Huddle', 'not_before': r'\s+(?:up|with)\b'}],
                           'prefixes': ['Delivered by'], 'source': 'invented'},
        'QUINTARA CORP': {'products': [{'phrase': '720'}], 'source': 'invented'},
        'ZELNTECH SE': {'joint_partners': [{'partner': 'Quillco'}], 'source': 'invented'}},
    'fixed_phrases': {
        'ZINDEX': {'guards': [[None, r'^\s*(?:Composite|Jones)\b'], [r'S&P\s*500\s*,\s*$', None],
                              [r'\bthe\s+$', r'^\s+(?:rose|fell)\b']], 'source': 'invented index word'},
        'ZARGET': {'guards': [[r'\b(?:price|earnings)\s+$', None], [None, r'^\s+(?:price|prices)\b'],
                              [None, r'^\s+(?:of|to)\s+[$]?\d']], 'source': 'invented price word'}}}


@contextlib.contextmanager
def use_tables(tables):
    """Install a table for the duration of a test and restore the previous one."""
    saved = (dict(M.COMMON_NAMES), dict(M.FIXED_PHRASES), dict(M.FIRM_PRODUCTS), dict(M.CODE_PREFIXES),
             dict(M._TABLE_STATE))
    M.set_tables(tables if tables is not None else EMPTY_TABLES, source='test')
    try:
        yield
    finally:
        for d, s in zip((M.COMMON_NAMES, M.FIXED_PHRASES, M.FIRM_PRODUCTS, M.CODE_PREFIXES, M._TABLE_STATE), saved):
            d.clear()
            d.update(s)
        M._COMPILED.clear()


def adjacency_review(build_dir):
    """Fixture stand-in for the builder's adjacency review (zero inferable masks)."""
    return R.new_json(Path(build_dir) / 'ADJACENCY_REVIEW.json',
                      {'adjacency_sha256': R.sha(Path(build_dir) / 'ADJACENCY.jsonl'), 'inferable_masks': 0,
                       'open_items': [], 'rows_reviewed': 'all (fixture)'}, timed=False)


def sweep_review(build_dir, verdict='safe'):
    """Fixture stand-in for the builder's adjacency-sweep review (every row adjudicated)."""
    path = Path(build_dir) / 'ADJACENCY_SWEEP.jsonl'
    rows = R.read_jsonl(path)
    return R.new_json(Path(build_dir) / ('ADJACENCY_SWEEP_REVIEW.json' if verdict == 'safe' else
                                         f'ADJACENCY_SWEEP_REVIEW_{verdict}.json'),
                      {'sweep_sha256': R.sha(path), 'gives_back': 0 if verdict == 'safe' else len(rows),
                       'verdicts': {r['row_id']: {'verdict': verdict, 'reason': 'fixture'} for r in rows}},
                      timed=False)


def fixture():
    """Synthetic sources + one build (cached per process)."""
    if _FIX:
        return _FIX
    engine, _ = B.td2_modules()
    src = RUN_ROOT / 'src'
    R.new_text(src / 'risk_menu.csv', _csv([(n, n, m, c, 'synthetic', 'True', 'synthetic') for n, m, c in MENU_TEXT],
                                           ['name', 'firm_column', 'economic_meaning', 'risk_channel', 'asof_source',
                                            'eligible', 'reason']))
    fschema = pa.schema([(c, pa.string() if c in ('firm_id', 'variant') else pa.timestamp('ns')
                          if c in ('cutoff', 'origin_session') or c.startswith('available_') else pa.bool_()
                          if c in ('mapped_focal', 'has_news') or c.startswith('missing_') else pa.int64()
                          if c.startswith('H') else pa.float64()) for c in B.FRAME_COLS])

    def frow(p, o, c, has_news=True, h=(0, 0, 0), allmissing=False, variant='primary'):
        r = {'firm_id': firm_id(p), 'cutoff': c, 'origin_session': o, 'variant': variant, 'mapped_focal': True,
             'has_news': has_news}
        for i, n in enumerate(S.NAMES):
            r['C_' + n] = np.nan if allmissing else 20.0 + i + p % 7 if n == 'size' else 0.1 * (i + 1) + (p % 5) / 50
            r['missing_' + n] = allmissing
            r['available_' + n] = pd.NaT if allmissing else o - pd.Timedelta(days=30 * i)
        for k, hop in enumerate(S.HOPS):
            r[f'H{hop}_graph_partners'] = h[k] + 3
            r[f'H{hop}_news_partners'] = h[k] + 1
            r[f'H{hop}_selected_partners'] = h[k]
        return r
    f21 = pd.DataFrame([frow(F1, O1, C1, h=(2, 1, 1)), frow(F1, O1, C1, h=(2, 1, 1), variant='fsb5_sensitivity'),
                        frow(F5, O1, C1, has_news=False), frow(F6, O1, C1, allmissing=True, h=(1, 0, 0))])
    f22 = pd.DataFrame([frow(F2, O2, C2, h=(1, 0, 0)), frow(F3, O3, C3, h=(1, 0, 0)), frow(F4, O4, C4, h=(1, 0, 0))])
    write_parquet(src / 'features_2021.parquet', f21[B.FRAME_COLS], fschema)
    write_parquet(src / 'features_2022.parquet', f22[B.FRAME_COLS], fschema)
    eschema = pa.schema([('firm_id', pa.string()), ('cutoff', pa.timestamp('ns')), ('origin_session', pa.timestamp('ns')),
                         ('variant', pa.string()), ('hop', pa.int64()), ('partner', pa.int64()),
                         ('nodes', pa.list_(pa.int64())), ('intermediate_nodes', pa.list_(pa.int64())),
                         ('relationship_ids', pa.list_(pa.string())), ('reporting_roles', pa.list_(pa.string())),
                         ('article_ids', pa.list_(pa.string())), ('max_edge_available_at', pa.timestamp('ns')),
                         ('max_news_available_at', pa.timestamp('ns'))])

    def erow(p, o, c, hop, partner, nodes, rels, arts, variant='primary'):
        return {'firm_id': firm_id(p), 'cutoff': c, 'origin_session': o, 'variant': variant, 'hop': hop,
                'partner': partner, 'nodes': nodes, 'intermediate_nodes': nodes[1:-1], 'relationship_ids': rels,
                'reporting_roles': ['X'] * len(rels), 'article_ids': arts,
                'max_edge_available_at': o - pd.Timedelta(days=900 + 10 * hop),
                'max_news_available_at': o - pd.Timedelta(days=1)}
    e21 = pd.DataFrame([erow(F1, O1, C1, 1, P1, [F1, P1], ['CUST-1'], ['A1', 'A1b', 'A_ORIGIN', 'A_FUT', 'A_OLD']),
                        erow(F1, O1, C1, 1, P2, [F1, P2], ['SUPPL-2'], ['A2', 'A_LATE']),
                        erow(F1, O1, C1, 2, P3, [F1, P2, P3], ['SUPPL-2', 'CUST-3'], ['A3', 'A3b']),
                        erow(F1, O1, C1, 3, P5, [F1, P1, P4, P5], ['CUST-1', 'SUPPL-4', 'CUST-5'], ['A5']),
                        erow(F1, O1, C1, 1, P1, [F1, P1], ['CUST-1'], ['A_FUT'], variant='fsb5_sensitivity')])
    e22 = pd.DataFrame([erow(F2, O2, C2, 1, P6, [F2, P6], ['CUST-6'], ['B6']),
                        erow(F3, O3, C3, 1, P6, [F3, P6], ['CUST-6'], ['C_BAD']),
                        erow(F4, O4, C4, 1, P6, [F4, P6], ['CUST-6'], ['D_EDGE'])])
    write_parquet(src / 'evidence_2021.parquet', e21, eschema)
    write_parquet(src / 'evidence_2022.parquet', e22, eschema)
    nschema = pa.schema([('date', pa.timestamp('us')), ('alt_id', pa.string()), ('firms', pa.list_(pa.int64())),
                         ('first_months', pa.list_(pa.int64())), ('available_at', pa.timestamp('us'))] +
                        [(n, pa.float64()) for n in S.NAMES])

    def nrow(aid, day, firms, months=None, available=None, text_key=None):
        h, b = TEXTS[text_key or aid]
        sc = engine.score(h[:200] + b[:1000])
        return {'date': pd.Timestamp(day), 'alt_id': aid, 'firms': firms, 'first_months': months or [201001] * len(firms),
                'available_at': pd.Timestamp(available) if available else engine.day_clock(day), **sc}
    n20 = pd.DataFrame([nrow('Z0', '2020-12-21', [P6], text_key='B6')])
    n21 = pd.DataFrame([nrow('A1', '2021-03-03', [P1]), nrow('A1b', '2021-03-01', [P1]),
                        nrow('A_ORIGIN', '2021-03-05', [P1]), nrow('A_FUT', '2021-03-08', [P1]),
                        nrow('A_OLD', '2021-02-25', [P1]), nrow('A2', '2021-03-04', [P2]),
                        nrow('A_LATE', '2021-03-04', [P2], available='2021-03-07 05:00:00'),
                        nrow('A3', '2021-03-02', [P3], months=[202103]), nrow('A3b', '2021-03-02', [P3]),
                        nrow('A5', '2021-03-01', [P5])])
    n22 = pd.DataFrame([nrow('B6', '2022-06-08', [P6]), nrow('C_BAD', '2023-01-04', [P6]),
                        nrow('D_EDGE', '2022-12-28', [P6])])
    for y, fr in ((2020, n20), (2021, n21), (2022, n22)):
        write_parquet(src / f'news_{y}.parquet', fr, nschema)
    aschema = pa.schema([('alt_id', pa.string()), ('date', pa.string()), ('year', pa.int64()), ('headline', pa.string()),
                         ('body_short', pa.string()), ('is_scr', pa.bool_()), ('has_firm', pa.bool_()),
                         ('rics', pa.string()), ('finbert_scr', pa.int64()), ('finbert_prob', pa.float64())])
    arts = [('A2', '2021-03-02', '', 'A2'), ('A1', '2021-03-03', 'QTL.N', 'A1'), ('A1', '2021-03-03', 'QTL.N', 'A1_DUP'),
            ('A1b', '2021-03-01', 'QTL.N', 'A1b'), ('A_ORIGIN', '2021-03-05', 'QTL.N', 'A_ORIGIN'),
            ('A_FUT', '2021-03-08', 'QTL.N', 'A_FUT'), ('A_OLD', '2021-02-25', 'QTL.N', 'A_OLD'),
            ('A2', '2021-03-04', 'F.N', 'A2'), ('A_LATE', '2021-03-04', 'F.N', 'A_LATE'),
            ('A3', '2021-03-02', 'BLND.N', 'A3'), ('A3b', '2021-03-02', 'BLND.N,BLNDP.N', 'A3b'),
            ('A5', '2021-03-01', 'OSTX.N', 'A5'), ('B6', '2022-06-08', 'NRVN.O', 'B6'),
            ('D_EDGE', '2022-12-28', 'NRVN.O', 'D_EDGE'), ('C_BAD', '2023-01-04', 'NRVN.O', 'C_BAD')]
    write_parquet(src / 'articles.parquet', pd.DataFrame([{'alt_id': a, 'date': d, 'year': int(d[:4]),
                                                           'headline': TEXTS[t][0], 'body_short': TEXTS[t][1],
                                                           'is_scr': True, 'has_firm': True, 'rics': r,
                                                           'finbert_scr': 0, 'finbert_prob': 0.5}
                                                          for a, d, r, t in arts]), aschema)
    gschema = pa.schema([('id', pa.string()), ('date', pa.string()), ('end', pa.string()), ('status', pa.string()),
                         ('typ', pa.string()), ('src', pa.string()), ('tgt', pa.string()), ('rownum', pa.int64())])
    g = [('CUST-1', 'CUSTOMER', F1, P1), ('SUPPL-2', 'SUPPLIER', F1, P2), ('CUST-3', 'CUSTOMER', P2, P3),
         ('SUPPL-4', 'SUPPLIER', P1, P4), ('CUST-5', 'CUSTOMER', P5, P4), ('CUST-6', 'CUSTOMER', F2, P6)]
    write_parquet(src / 'graph.parquet', pd.DataFrame([{'id': i, 'date': '2015-01-01', 'end': None, 'status': 'active',
                                                        'typ': t, 'src': f'E{a}', 'tgt': f'E{b}', 'rownum': k}
                                                       for k, (i, t, a, b) in enumerate(g)]), gschema)
    lschema = pa.schema([('entity', pa.string()), ('permno', pa.int64()), ('date', pa.string()), ('end', pa.string())])
    write_parquet(src / 'links.parquet', pd.DataFrame([{'entity': f'E{p}', 'permno': p, 'date': '2010-01-01',
                                                        'end': '2024-12-31'}
                                                       for p in (F1, P1, P2, P3, P4, P5, F2, P6, F3, F4, F5, F6)]),
                  lschema)
    R.new_text(src / 'ric_identity.json', json.dumps({'mapping': RICS}))
    R.new_text(src / 'crsp_stocknames.csv', _csv([(p, cu, cu, t, c, d, '2030-12-31') for p, t, c, d, cu in NAMES_ROWS],
                                                 ['permno', 'ncusip', 'cusip', 'ticker', 'comnam', 'namedt', 'nameenddt']))
    R.new_text(src / 'crsp_names.csv', _csv([(P5, '2012-01-01', '2015-01-01', 'OSTRA HOLDINGS', 'OSTX', 3),
                                             (P6, '2012-01-01', '2016-01-01', 'SUMMIT HOLDINGS', 'NRVN', 3)],
                                            ['permno', 'namedt', 'nameenddt', 'comnam', 'ticker', 'exchcd']))
    write_factset(src / 'factset_company.parquet', FACTSET_ROWS)
    R.new_text(src / 'mask_tables.json', json.dumps(FIXTURE_TABLES))
    sources = B.Sources(menu=src / 'risk_menu.csv',
                        features={2021: src / 'features_2021.parquet', 2022: src / 'features_2022.parquet'},
                        evidence={2021: src / 'evidence_2021.parquet', 2022: src / 'evidence_2022.parquet'},
                        news={y: src / f'news_{y}.parquet' for y in S.NEWS_YEARS}, graph=src / 'graph.parquet',
                        links=src / 'links.parquet', ric_identity=src / 'ric_identity.json',
                        articles=src / 'articles.parquet',
                        crsp_names=(src / 'crsp_stocknames.csv', src / 'crsp_names.csv'), td2_bound=False,
                        factset=src / 'factset_company.parquet', english=None, tables=src / 'mask_tables.json')
    build_dir = RUN_ROOT / 'build'
    with use_tables(EMPTY_TABLES):
        receipt = B.build(build_dir, sources, n=2, reask_n=1)
    hits = R.read_jsonl(build_dir / 'SCAN_HITS.jsonl')
    R.new_json(build_dir / 'SCAN_ADJUDICATION.json',
               {'adjudications': {h['hit_id']: {'verdict': 'false_positive', 'reason': 'fixture'} for h in hits}},
               timed=False)
    adjacency_review(build_dir)
    sweep_review(build_dir)
    _FIX.update(sources=sources, build_dir=build_dir, receipt=receipt, hits=hits,
                records=R.read_jsonl(build_dir / 'INPUTS.jsonl'), metas=R.read_jsonl(build_dir / 'INPUT_META.jsonl'))
    return _FIX


def keyring(red):
    return RU.KeyRing(red, {'deepseek': SENT_DS, 'jev': SENT_JEV})


def all_text_under(root):
    return '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in Path(root).rglob('*') if p.is_file())


def aliases(comnam, tickers=(), english=frozenset(), extra_names=(), tickers_extra=(), rics=(), fs_names=(),
            near=()):
    names = set(M.name_variants(comnam)) | set(extra_names)
    al = {'names': sorted(names), 'common': sorted(M.common_aliases([comnam])), 'tickers': list(tickers),
          'tickers_extra': list(tickers_extra), 'rics': list(rics)}
    al['embedded'] = M.embedded_terms(al, english)
    al['fuzzy'] = M.fuzzy_words(al, english)
    al['products'] = M.firm_products([comnam])
    codes = M.code_prefix_terms([comnam], list(fs_names), list(tickers) + list(tickers_extra), english)
    al['code_prefixes'] = sorted(codes)
    al['unit_prefixes'] = M.unit_prefixes(codes)                       # v6 (TD3C5 D1)
    al['near'] = list(near)
    return al


def mask_one(comnam, text, tickers=(), english=frozenset(), token='[X]', extra_names=(), tables=None, **kw):
    with use_tables(tables):
        return M.Masker([(token, aliases(comnam, tickers, english, extra_names, **kw))], english).mask(text)[0]


class TestMasking(unittest.TestCase):
    def masker(self):
        return M.Masker([('[FOCAL]', {'names': M.name_variants('ZORBLAX INDUSTRIES INC'), 'tickers': ['ZRBX'],
                                      'rics': ['ZRBX.O']}),
                         ('[SUPPLIER_1]', {'names': M.name_variants('MAVRICK HOLDINGS INC'), 'tickers': ['F'],
                                           'rics': ['F.N']}),
                         ('[CUSTOMER_1]', {'names': M.name_variants('GENERAL WIDGET CO'), 'tickers': ['IT', 'GWX']}),
                         ('[LINK_1]', {'names': M.name_variants('3Q CO'), 'tickers': ['QQQX']})])

    def test_names_and_tickers_removed(self):
        text = ('Zorblax Industries Inc. (ZRBX.O) and ZORBLAX said; Zorblax\'s ZRBX fell. $ZRBX (NASDAQ: ZRBX) '
                'Mavrick Holdings (F.N) and Mavrick\'s unit; (NYSE: F) and (F). General Widget gained; GWX rose; '
                '3Q and QQQX; (NYSE: IT).')
        out, counts = self.masker().mask(text)
        for bad in ['Zorblax', 'ZORBLAX', 'ZRBX', 'Mavrick', 'F.N', '(F)', 'NYSE: F)', 'General Widget', 'GWX', '3Q',
                    'QQQX', 'NYSE: IT']:
            self.assertNotIn(bad, out)
        for tok in ['[FOCAL]', '[SUPPLIER_1]', '[CUSTOMER_1]', '[LINK_1]']:
            self.assertIn(tok, out)
        self.assertEqual(self.masker().residual(out), [])
        self.assertGreater(sum(counts.values()), 10)

    def test_generic_and_lowercase_and_acronyms_kept(self):
        out, _ = self.masker().mask('General demand rose; a grade F rating; IT spending; zorblax is lower case.')
        self.assertIn('General demand', out)
        self.assertIn('grade F rating', out)
        self.assertIn('IT spending', out)
        self.assertIn('zorblax is lower case', out)

    def test_dates_bylines_and_truncation(self):
        out, _ = self.masker().mask('LONDON, March 3 (Reuters) - Sales on Jan. 5, 2021 and 14 February 2022 rose in '
                                    '2020. (Reporting by A B; Editing by C D)')
        for bad in ('2021', '2020', 'Reporting by', 'LONDON', 'Reuters'):
            self.assertNotIn(bad, out)
        self.assertIn('[DATE]', out)
        self.assertIn('[YEAR]', out)
        self.assertEqual(M.truncate('Shares of Zorblax rose', 12), 'Shares of ...')
        self.assertEqual(M.truncate('abc def', 50), 'abc def')

    def test_urls_emails_and_owner_terms(self):
        out, _ = self.masker().mask('See https://blogs.zorblax.com/x and www.zorblax.com or ir@zorblax.com today.')
        self.assertEqual(out, 'See [URL] and [URL] or [EMAIL] today.')
        self.assertEqual(M.owner_terms(['ALPHA QUORN LINES INC', 'ACME CORP']), {'ALPHA'})
        self.assertEqual(M.owner_terms(['GENERAL QUORN CO']), set())
        self.assertEqual(M.owner_terms(['TEXAS QUORN CO', 'COPPER QUORN INC']), set())    # v3: place/commodity words
        m = M.Masker([('[SUPPLIER_1]', {'names': sorted(M.name_variants('ALPHA QUORN LINES INC') | {'ALPHA'})})])
        out, _ = m.mask('ALPHA HAS ANNOUNCED a fare; Alpha Quorn Lines said; the alpha version shipped.')
        self.assertEqual(out, '[SUPPLIER_1] HAS ANNOUNCED a fare; [SUPPLIER_1] said; the alpha version shipped.')

    def test_collision_keeps_first_token(self):
        m = M.Masker([('[FOCAL]', {'names': ['ACME WIDGETS']}), ('[SUPPLIER_1]', {'names': ['ACME WIDGETS']})])
        out, _ = m.mask('Acme Widgets said')
        self.assertEqual(out, '[FOCAL] said')
        self.assertEqual(len(m.collisions), 1)

    def test_ticker_inside_token_word_not_rematched(self):
        m = M.Masker([('[FOCAL]', {'names': ['LINKCO SYSTEMS'], 'tickers': ['LINK']}),
                      ('[CUSTOMER_1]', {'tickers': ['CUSTX']})])
        out, _ = m.mask('Linkco Systems and CUSTX traded')
        self.assertEqual(out, '[FOCAL] and [CUSTOMER_1] traded')
        self.assertEqual(m.residual(out), [])

    def test_replacement_log(self):
        m = self.masker()
        m.mask('Zorblax said')
        self.assertEqual([(e['token'], e['match']) for e in m.log], [('[FOCAL]', 'Zorblax')])


class TestMaskingLeakClassesV2(unittest.TestCase):
    """C1 item 9: one synthetic test per leak class found by the TD3C review (invented firm names)."""

    def assertMasked(self, comnam, text, expected, **kw):
        self.assertEqual(mask_one(comnam, text, **kw), expected)

    def test_spaced_initials(self):
        self.assertMasked('Q R T SYSTEMS INC', 'BRIEF-QRT Systems Receives Order; Q.R.T. Systems; Qrt said; QRT SYSTEMS',
                          'BRIEF-[X] Receives Order; [X]; [X] said; [X]')
        al = B.aliases_for(P3, '2021-03-05', {'names': {P3: [('2010-01-01', 'H Q BRINWOOD CORP', 'HQB')]},
                                              'rics_by_permno': {}})[0]
        self.assertEqual(M.Masker([('[X]', al)]).mask('Fire At HQ Brinwood Facility; H.Q. Brinwood said; '
                                                     'Brinwood\'s plant')[0], 'Fire At [X] Facility; [X] said; [X] plant')
        self.assertMasked('V W ZELTRON ENTERPRISES INC NEW', 'VW Zeltron Reports; V.W. Zeltron Enterprises Inc : *',
                          '[X] Reports; [X] : *')
        self.assertEqual(sorted(M.name_variants('Q R T SYSTEMS INC')), ['Q.R.T.', 'Q.R.T. SYSTEMS', 'Q.R.T. SYSTEMS INC'])
        self.assertMasked('C P I GLORBEX INC', 'CPI data rose; CPI Glorbex fell', 'CPI data rose; [X] fell')

    def test_title_case_tickers_and_initialisms(self):
        self.assertMasked('K Q Z INC', 'Kqz wins contract; KQZ INC said', '[X] wins contract; [X] said')
        self.assertEqual(mask_one('ZUFRAX FINANCIAL GROUP', 'Loan Arranged By Quorn And Zfgx', tickers=['ZFGX']),
                         'Loan Arranged By Quorn And [X]')
        self.assertEqual(mask_one('ZOOT CORP', 'Linkage (Zoot) and (Cnxq) rose', tickers=['CNXQ']),
                         'Linkage ([X]) and ([X]) rose')
        self.assertEqual(mask_one('DIMRAN INC', 'Dim prices; shares of DIM fell', tickers=['DIM'],
                                  english=frozenset({'dim'})), 'Dim prices; shares of [X] fell')

    def test_ampersand_forms(self):
        self.assertMasked('Q T & Q INC', 'QT&Q said; Qt&q', '[X] said; [X]')
        self.assertMasked('K & V BANK CORP', 'K&V Bank said', '[X] said')
        self.assertMasked('BRAXTON & WILLOUGHBY ENTERPRISES INC', 'B&W Zorbal; Braxton and Willoughby',
                          '[X] Zorbal; [X]')
        self.assertNotIn('S.&P.', M.name_variants('S & P ZORTEX INC'))
        self.assertMasked('S & P ZORTEX INC', 'S&P 500 fell; S&P Zortex rose', 'S&P 500 fell; [X] rose')

    def test_possessives(self):
        self.assertMasked('GLORPINOS BAKERY INC', "Glorpino's Bakery Q2; GLORPINO'S BAKERY; Glorpino’s said",
                          '[X] Q2; [X]; [X] said')
        self.assertMasked('PEOPLES ZENITH FINANCIAL INC', "People's Zenith rose", '[X] rose')

    def test_abbreviations_and_bank_of(self):
        self.assertMasked('BANK QUORVILLE QUE', 'Bank of Quorville reports', '[X] reports')
        self.assertMasked('HOUSE OF ZEPHYR RESORT & ENT CO', 'House of Zephyr Resort & Entertainment Co - CFO quits',
                          '[X] - CFO quits')
        self.assertMasked('ZARNOW RESH CORP', 'BRIEF-Zarnow Research Corporation Reports', 'BRIEF-[X] Reports')
        self.assertMasked('ZEMBLAR FACTORY OUTLET CENTERS IN', 'Zemblar Factory Outlet Centers Inc : * PRICES',
                          '[X] : * PRICES')

    def test_common_short_names(self):
        t = {**EMPTY_TABLES, 'common_names': {'QUORNWELL EDISTON': {'aliases': ['Q Ediston', 'QuEd'], 'source': 'invented'},
                                              'BRANDLE QUIXON': {'aliases': ['BQX'], 'source': 'invented'}}}
        with use_tables(t):
            self.assertEqual(M.common_aliases(['QUORNWELL EDISTON INC']), {'Q Ediston', 'QuEd'})
        self.assertMasked('QUORNWELL EDISTON INC', 'Q Ediston said; QuEd rose', '[X] said; [X] rose', tables=t)
        self.assertMasked('BRANDLE QUIXON & CO', "BQX's unit; Brandle Quixon; a bqx degree",
                          "[X] unit; [X]; a bqx degree", tables=t)

    def test_inverted_person_names(self):
        self.assertMasked('ZOLDEN STEVEN LTD', 'Steven Zolden Ltd said', '[X] said')
        self.assertMasked('TRAVOLT WALT CO', 'Walt Travolt Co executives; Travolt', '[X] executives; [X]')
        self.assertMasked('PLUMMER J M CO', 'J.M. Plummer Co. said; JM Plummer', '[X] said; [X]')
        self.assertMasked('BLENTON W R & CO DEL NEW', 'W R Blenton & Co; W.R. Blenton', '[X]; [X]')
        self.assertMasked('QUILLAN H & R INC', 'H&R Quillan said', '[X] said')

    def test_renames_after_origin_and_factset_names(self):
        ns = {'names': {P2: [('2010-01-01', 'MAVRICK HOLDINGS INC', 'F'), ('2021-09-01', 'ZENTRIQ CORP', 'ZNTQ')]},
              'rics_by_permno': {P2: [('F.N', 201001), ('ZNTQ.N', 202201)]},
              'fsnames': {P2: {'Mavrix Global Holdings, Inc.'}}}
        al, cn, after, fs, scn = B.aliases_for(P2, '2021-03-05', ns)
        self.assertIn('ZENTRIQ', al['names'])
        self.assertIn('ZNTQ', al['tickers'])
        self.assertIn('ZNTQ.N', al['rics'])
        self.assertEqual(after, ['ZENTRIQ CORP'])
        self.assertIn('MAVRIX GLOBAL', al['names'])
        out = M.Masker([('[S]', al)]).mask('Zentriq (ZNTQ) and Mavrix Global Holdings, Inc. said')[0]
        self.assertEqual(out, '[S] [S] and [S] said')            # '(ZNTQ)' is replaced with its parentheses
        late = {'names': {P2: [('2023-02-01', 'POSTSEAL CORP', 'PSTL')]}, 'rics_by_permno': {}}
        self.assertEqual(B.aliases_for(P2, '2021-03-05', late)[1], [])

    def test_news_agency_as_subject(self):
        out = mask_one('ZORBEX REUTERS CORP', 'Reuters to name Jane Placeholder; Zorbex Reuters said; told REUTERS')
        self.assertEqual(out, '[NEWS_AGENCY] to name Jane Placeholder; [X] said; told [NEWS_AGENCY]')
        out = M.Masker([]).mask('Reuters Foundation said, Reuters reported')[0]
        self.assertEqual(out, '[NEWS_AGENCY] Foundation said, [NEWS_AGENCY] reported')
        self.assertEqual(mask_one('ZORVATED MATERIALS INC', 'Zorvated Material is up; Zorvated Materials fell'),
                         '[X] is up; [X] fell')

    def test_trailing_name_words_absorbed_and_ticker_first_word(self):
        m = M.Masker([('[Y]', {'names': M.name_variants('ZEPHA OUTLET CENTERS INC') | {'ZEPHA'}})])
        self.assertEqual(m.mask('Zepha Outlet Centers, Inc. said; Zepha Group co-founder')[0],
                         '[Y] said; [Y] Group co-founder')
        al = B.aliases_for(P1, '2021-03-05', {'names': {P1: [('2010-01-01', 'ZEQ THERAPEUTICS INC', 'ZEQ')]},
                                              'rics_by_permno': {}})[0]
        self.assertIn('ZEQ', al['names'])
        # v5 (TD3C4 D2): a code whose prefix is the firm's first name word is masked whole (no "[X]-102" remnant)
        self.assertEqual(M.Masker([('[X]', al)]).mask('Collaboration For Zeq-102; zeq sera')[0],
                         'Collaboration For [X]; zeq sera')

    def test_new_york_not_masked_for_city_named_firm(self):
        self.assertNotIn('NEW YORK', M.name_variants('NEW YORK ZORBANE CORP'))
        self.assertEqual(mask_one('NEW YORK ZORBANE CORP', 'New York Zorbane signs; New York Stock Exchange'),
                         '[X] signs; New York Stock Exchange')
        self.assertIn('AIR SERVICES', M.name_variants('AIR SERVICES & CHEMICALS INC'))
        self.assertEqual(mask_one('AIR SERVICES & CHEMICALS INC', 'Air Services to invest; air services demand'),
                         '[X] to invest; air services demand')


class TestManualReadFixesV2(unittest.TestCase):
    """Classes found by the v2 builder's manual read (invented names)."""

    def test_word_initials_and_next_word_filter(self):
        self.assertEqual(M.word_initials('QUANWOOD REAL ESTATE INVT TR'), 'QREIT')
        self.assertEqual(M.word_initials('NEW ZORQ COMMUNITY BANCORP INC'), 'NZCB')
        eng = frozenset({'network', 'real', 'estate'})
        names = {P1: [('2010-01-01', 'QUANWOOD REAL ESTATE INVT TR', 'QWD')], P2: [('2010-01-01', 'F 7 NETWORKS INC', 'FFVN')],
                 P3: [('2010-01-01', 'H Q BRINWOOD CORP', 'HQB')]}
        ns = {'names': names, 'rics_by_permno': {}, 'english': eng}
        a1 = B.aliases_for(P1, '2021-03-05', ns)[0]
        self.assertIn('Q.R.E.I.T.', a1['names'])
        self.assertEqual(M.Masker([('[X]', a1)], eng).mask('BRIEF-Qreit Wins Award; QREIT WINS')[0],
                         'BRIEF-[X] Wins Award; [X] WINS')
        a2 = B.aliases_for(P2, '2021-03-05', ns)[0]
        self.assertIn('F.7.', a2['names'])
        self.assertNotIn('NETWORKS', a2['names'])               # plural of an English word: not a single-word variant
        self.assertEqual(M.Masker([('[X]', a2)], eng).mask('BRIEF-F7 Collaborates; Quorn Networks rose')[0],
                         'BRIEF-[X] Collaborates; Quorn Networks rose')
        self.assertIn('BRINWOOD', B.aliases_for(P3, '2021-03-05', ns)[0]['names'])

    def test_upper_case_dates_story_ids_and_accents(self):
        out = M.Masker([]).mask('ON MAY 19, SIGNED; DATED AS OF FEB 11, 2021; AS OF 30 APRIL, WHEN; '
                                'Source text for Eikon: [nQZ9AbCdEfG] and [ID:nXy7TeSt12 ...')[0]
        for bad in ('MAY 19', 'FEB 11', '30 APRIL', 'nQZ9AbCdEfG', 'nXy7TeSt12'):
            self.assertNotIn(bad, out)
        self.assertEqual(M.date_residuals(out), [])
        self.assertTrue(M.date_residuals('PAYABLE ON OCTOBER 14'))
        self.assertIn('[nothing]', M.Masker([]).mask('a [nothing] here')[0])     # bracketed words without digits kept
        self.assertEqual(mask_one('ZORVELEZ INTERNATIONAL INC', 'ZORVELĒZ INTERNATIONAL REPORTS; Zorvelēz said'),
                         '[X] REPORTS; [X] said')

    def test_single_letter_ticker_and_rule_order(self):
        m = M.Masker([('[X]', {'names': [], 'tickers': ['U']}), ('[Y]', {'names': ['Q.X.Z.'], 'rics': ['QXZ.N']})])
        self.assertEqual(m.mask('Statement On U.S. Rules; (U.N) and <U.N>; <QXZ.N> rose; QXZ fell')[0],
                         'Statement On U.S. Rules; ([X]) and <[X]>; <[Y]> rose; [Y] fell')

    def test_table_and_absorption(self):
        with use_tables(INVENTED_TABLES):          # v4 (N6): invented table entries only
            self.assertEqual(M.common_aliases(['Q V B FINANCIAL GROUP']), {'Quorn Valley Bank'})
        self.assertEqual(mask_one('ZAISER ALUMINUM CORP', 'ZAISER ALUMINIUM CORPORATION REPORTS'), '[X] REPORTS')
        self.assertNotIn('TIME', M.name_variants('TIME ZORBING RESOURCES INC'))
        self.assertNotIn('CORN', M.name_variants('CORN ZORBEX INTL INC'))
        self.assertNotIn('AIR TRANSPORT', M.name_variants('AIR TRANSPORT ZORB GROUP INC'))


class TestDatesAndBoilerplateV2(unittest.TestCase):
    """C2: numeric dates, two-digit and fiscal years, story boilerplate and bare domains (invented strings)."""

    def test_patterns_masked_and_detected(self):
        raw = ("STORY_NUMBER:[XQZ0TEST1] STORY_DATE:[14/03/2021] STORY_TIME:[11:22:33] FY22 GAAP; Q3'21 close; "
               "Q2 FY23 vs Q2 FY20; 3Q21; H1 22 results; Q1 20 cents; Source link: bloom.bg/9Zz0Qq -- On 3/14/21, "
               "2021-03-14 and 14.03.2021, FY2022, Q4-2021, Sept '21, Fall '22. [nZZ9TEST01] Phase 2/3, 24/7.")
        out = M.Masker([]).mask(raw)[0]
        for bad in ('STORY_', '14/03', '3/14/21', '2021', 'FY22', "Q3'21", 'FY23', 'FY20', '3Q21', 'H1 22',
                    'bloom.bg', 'nZZ9TEST01', "Sept '21", "Fall '22", '14.03'):
            self.assertNotIn(bad, out)
        for keep in ('Q1 20 cents', 'Phase 2/3', '24/7'):
            self.assertIn(keep, out)
        self.assertEqual(M.date_residuals(out), [])
        for bad in ('STORY_DATE:[14/03/2021]', 'FY22', "Q3'21", 'bloom.bg/9Zz0Qq', '1/2/21', '2021', "Sept '21",
                    '[nZZ9TEST01]'):
            self.assertTrue(M.date_residuals('x ' + bad + ' y'), bad)


class TestV3IssuerRics(unittest.TestCase):
    """D1 item 1: other RICs and tickers of the same issuer."""

    def test_article_extras_prefix_and_halt_headline(self):
        entries = [('[A]', {'tickers': ['QRXA'], 'tickers_extra': [], 'rics': ['QRXA.O']}),
                   ('[B]', {'tickers': ['ZDE'], 'tickers_extra': [], 'rics': ['ZDE.N']})]
        ex = B.article_extras(['QRXA.O', 'QRXAP.O', 'ZZOTHER.L'], 'BRIEF-headline', ['[A]'], entries)
        self.assertEqual(ex, {'[A]': {'rics': ['QRXAP.O'], 'tickers_extra': ['QRXAP']}})
        ex = B.article_extras(['ZDE.N', 'ZDG.N'], 'NYSE ORDER IMBALANCE <ZDG.N> 12345.0 SHARES ON SELL SIDE', ['[B]'],
                              entries)
        self.assertEqual(ex['[B]']['rics'], ['ZDG.N'])
        al = {**entries[1][1], 'names': []}
        m = M.Masker(B.article_entries([('[B]', al)], {}, ex))
        self.assertEqual(m.mask('NYSE ORDER IMBALANCE <ZDG.N> 12345.0 SHARES ON SELL SIDE')[0],
                         'NYSE ORDER IMBALANCE <[B]> 12345.0 SHARES ON SELL SIDE')
        m = M.Masker(B.article_entries([('[A]', {**entries[0][1], 'names': []})], {},
                                       B.article_extras(['QRXA.O', 'QRXAP.O'], 'x', ['[A]'], entries)))
        self.assertEqual(m.mask('NASDAQ TRADE HALT <QRXAP.O> PAUSE; QRXAP fell')[0],
                         'NASDAQ TRADE HALT <[A]> PAUSE; [A] fell')
        self.assertEqual(B.article_extras(['GMBX.L'], 'x', ['[C]'], [('[C]', {'tickers': ['GM'], 'rics': []})]), {})

    def test_ric_after_role_token(self):
        out = mask_one('ZELWICK GROUP INC', 'Filing by ZELWICK GROUP INC <9876.T>: notice; Zelwick (QZW.L) rose')
        self.assertEqual(out, 'Filing by [X] <[X]>: notice; [X] ([X]) rose')

    def test_issuer_siblings_and_factset_ids(self):
        root = RUN_ROOT / ('sib_' + uuid.uuid4().hex[:6])
        crsp = root / 'crsp_stocknames.csv'
        R.new_text(crsp, _csv([(30001, '90000001', '90000001', 'QDE', 'QUOREX ENERGY CO', '2010-01-01', '2030-12-31'),
                               (30002, '90000002', '90000002', 'QDG', 'QUOREX ENERGY CO', '2012-01-01', '2030-12-31'),
                               (30003, '91111101', '91111101', 'ZZZ', 'OTHER FIRM INC', '2010-01-01', '2030-12-31')],
                              ['permno', 'ncusip', 'cusip', 'ticker', 'comnam', 'namedt', 'nameenddt']))
        sib, c6 = B.issuer_siblings((crsp,), [30001, 30003])
        self.assertEqual(sib, {30001: [30002], 30003: []})
        fs = root / 'fs.parquet'
        write_factset(fs, [('90000001', 'Quorex Energy Company', '2011-01-01', 'Q1', 'QDE', 'QDE-NYS'),
                           ('', 'Quorex Electric Co.', '2012-01-01', 'Q1', 'QDA', 'QDA-NYS'),
                           ('', 'Quorex Later Name', '2023-02-01', 'Q1', 'QLN', 'QLN-NYS'),
                           ('90000001', 'Quorex Germany Listing', '2012-01-01', 'Q1', 'QDX', 'QDX-FRA')])
        names, ticks = B.factset_aliases(fs, c6)
        self.assertEqual(names[30001], {'Quorex Energy Company', 'Quorex Electric Co.', 'Quorex Germany Listing'})
        self.assertEqual(ticks[30001], {'QDE', 'QDA'})            # the Frankfurt ticker and the 2023 row are excluded
        ns = {'names': B.load_names((crsp,), [30001, 30002]), 'rics_by_permno': {30002: [('QDG.N', 201201)]},
              'siblings': sib, 'fsnames': names, 'fstickers': ticks}
        al = B.aliases_for(30001, '2021-03-05', ns)[0]
        self.assertIn('QDG.N', al['rics'])
        self.assertIn('QDG', al['tickers_extra'])
        self.assertIn('QDA', al['tickers_extra'])
        out = M.Masker([('[D]', al)]).mask('NYSE ORDER IMBALANCE <QDG.N>; (NYSE: QDA); QDG fell')[0]
        self.assertEqual(out, 'NYSE ORDER IMBALANCE <[D]>; (NYSE: [D]); [D] fell')
        al2 = {'names': [], 'tickers': [], 'tickers_extra': ['TRY', 'CG'], 'rics': []}
        out = M.Masker([('[E]', al2)], frozenset({'try'})).mask('WILL TRY TO; CG rose; (NYSE: CG)')[0]
        self.assertEqual(out, 'WILL TRY TO; CG rose; (NYSE: [E])')   # restricted rules for sibling/FactSet tickers


class TestV3EmbeddedAndMisspellings(unittest.TestCase):
    """D1 items 2 and 3."""

    def test_embedded_names(self):
        eng = ENGLISH
        out = mask_one('Q R X RESORTS INTL', "BRIEF-BetQRX Adds Markets; BETQRX'S ONLINE; QRX said; QRXP traded; "
                                             "Reimbursement For Qrxpro; QRXPRO DEVICE", english=eng, tickers=['QRX'])
        self.assertEqual(out, 'BRIEF-[X] Adds Markets; [X] ONLINE; [X] said; [X] traded; Reimbursement For [X]; '
                              '[X] DEVICE')
        out = mask_one('ZELWICKO PHARMACEUTICALS INC', 'BRIEF-Zelwickon Reduces Risk; ZELWICKON TREATMENT; ZelwickoBio',
                       english=eng)
        self.assertEqual(out, 'BRIEF-[X] Reduces Risk; [X] TREATMENT; [X]')
        # not embedded: acronym ending a token after one letter, a non-English rest, no camel boundary, middle position
        out = mask_one('S B Q TELECOM INC', 'HSBQ hires staff; SBQTESTIM data', english=eng, tickers=['SBQ'])
        self.assertEqual(out, 'HSBQ hires staff; SBQTESTIM data')
        # v5 (TD3C4 R1): a name word of >= 5 letters plus up to four letters is masked whole ("Mobixxeye"); a name
        # word plus a longer English word is not ("Mobixxweapons", with "weapon" in the word list)
        out = mask_one('MOBIXX OIL CORP', 'Mobixxeye rose; HOTSPOT network; Mobixxweapons seized', english=eng |
                       {'weapon'}, tickers=['OTSP'])
        self.assertEqual(out, '[X] rose; HOTSPOT network; Mobixxweapons seized')
        self.assertTrue(M.embedded_hit('BetQRX', 'QRX', 'acro'))
        self.assertFalse(M.embedded_hit('Starqrxmine', 'QRX', 'acro'))

    def test_misspellings(self):
        eng = ENGLISH
        self.assertEqual(mask_one('TORVANDEL ENERGY CORP', 'a leak at the Torvanndel Plant; TORVNDEL said', english=eng),
                         'a leak at the [X] Plant; [X] said')
        self.assertEqual(mask_one('QUENTRAXIS INC', 'Says Quintraxis Signed A Loan', english=eng),
                         'Says [X] Signed A Loan')
        # a 6-letter word counts only an inserted or deleted letter; lower case and other first letters never count
        self.assertEqual(mask_one('ZORFEL INC', 'ZACH ZORPEL, FOUNDER; Zorfels said; zorfell; Xorfel', english=eng),
                         'ZACH ZORPEL, FOUNDER; [X] said; zorfell; Xorfel')
        self.assertTrue(M.lev1('TORVEL', 'TORVVEL'))
        self.assertFalse(M.lev1('TORVEL', 'TORVEL'))
        self.assertFalse(M.lev1('TORVEL', 'TERVVEL'))


class TestV3ContextGuard(unittest.TestCase):
    """D1 item 4 (c), v4 (N1-N3, N6): no mask may make the hidden word inferable; invented firm names and an invented
    fixed-phrase table (the real table is private)."""

    def test_fixed_phrases_left_as_written(self):
        eng = ENGLISH
        t = INVENTED_TABLES
        self.assertEqual(mask_one('ZINDEX ZORB INC', '* Zindex, S&P 500 slip; S&P 500, Zindex slip 1%; Zindex Jones fell; '
                                  'the Zindex rose; Zindex Zorb said', english=eng, tables=t),
                         '* [X], S&P 500 slip; S&P 500, Zindex slip 1%; Zindex Jones fell; the Zindex rose; [X] said')
        self.assertEqual(mask_one('ZARGET ZORB CORP', 'cuts price Zarget; Zarget Price raised; Zarget of $50 set; '
                                  'BRIEF-Zarget Lands Order', english=eng, tables=t),
                         'cuts price Zarget; Zarget Price raised; Zarget of $50 set; BRIEF-[X] Lands Order')
        # without the table the same words are masked everywhere (the guards live only in the private table)
        self.assertEqual(mask_one('ZARGET ZORB CORP', 'cuts price Zarget', english=eng), 'cuts price [X]')

    def test_n3_guards_restricted_to_index_and_price_phrases(self):
        """N3: an index or price word that is the masked firm's name is masked outside the index or price phrase."""
        eng = ENGLISH
        t = INVENTED_TABLES
        self.assertEqual(mask_one('ZINDEX ZORB INC', 'The fire at the Zindex plant in Zorbport; The Zindex said it would '
                                  'delist the stock; a Zindex spokesperson said', english=eng, tables=t),
                         'The fire at the [X] plant in Zorbport; The [X] said it would delist the stock; a [X] '
                         'spokesperson said')
        self.assertEqual(mask_one('ZARGET ZORB CORP', 'Zarget to hire 100,000 workers; the Zarget store in Ohio closed; '
                                  'a Zarget spokesperson said', english=eng, tables=t),
                         '[X] to hire 100,000 workers; the [X] store in Ohio closed; a [X] spokesperson said')

    def test_particles_hq_towns_and_no_person_or_place_guard(self):
        eng = ENGLISH
        self.assertEqual(mask_one('SIGN ZORBA GLOBAL', 'STORES OFFER SIGN-UP BONUS; SIGN REPORTS', english=eng),
                         'STORES OFFER SIGN-UP BONUS; [X] REPORTS')
        self.assertEqual(mask_one('ZORBEXA CORP', 'ZORBEXA-IN Q2 PLANS', english=eng), '[X]-IN Q2 PLANS')
        # v4 (N1): a person who shares the firm word is masked at the surname and at every later mention
        self.assertEqual(mask_one('BRANWICK FOODS CO', 'Judge said Maria Branwick alleged; Branwick said', english=eng),
                         'Judge said Maria [X] alleged; [X] said')
        self.assertEqual(mask_one('BRANWICK FOODS CO', "Maria Branwick sued. Branwick's lawyers said she was misled",
                                  english=eng), "Maria [X] sued. [X] lawyers said she was misled")
        self.assertEqual(mask_one('BRANWICK FOODS CO', 'Branwick said; Branwick Foods rose', english=eng),
                         '[X] said; [X] rose')
        # v4 (N2): no place guard; a firm word after a place word is masked
        self.assertEqual(mask_one('ZELTRANIX CORP', 'fund India Zeltranix backs startups; Zeltranix rose', english=eng),
                         'fund India [X] backs startups; [X] rose')
        self.assertEqual(mask_one('TERVAND FURNITURE INDUSTRIES INC',
                                  'The Tervand, Ohio-based company; based in Tervand, Va., the firm; Tervand Furniture rose',
                                  english=eng),
                         'The Ohio-based company; based in Va., the firm; [X] rose')
        self.assertNotIn('SACRAMENTO', M.name_variants('SACRAMENTO ZORB EXCHANGE CORP'))
        self.assertIn('SACRAMENTO ZORB', M.name_variants('SACRAMENTO ZORB EXCHANGE CORP'))
        self.assertNotIn('PENNSYLVANIA', M.name_variants('PENNSYLVANIA ZORB TRUST'))
        self.assertNotIn('GRUPO', M.name_variants('GRUPO ZORBEX SAB'))
        self.assertNotIn('CORPORATE', M.name_variants('CORPORATE ZORB PROPERTIES TRUST'))
        self.assertEqual(mask_one('SACRAMENTO ZORB EXCHANGE CORP', 'manager in Sacramento expects; Sacramento Zorb rose'),
                         'manager in Sacramento expects; [X] rose')

    def test_n1_n2_probe_sentences(self):
        """The TD3C3 guard probes, with invented firms (N1, N2)."""
        self.assertEqual(mask_one('ZELWICK TECHNOLOGIES INC', 'Zelwick plans to trim 5,000 roles. Founder '
                                  'Michael Zelwick called the plan painful. Zelwick stock slid 4%.', tickers=['ZLWK']),
                         '[X] plans to trim 5,000 roles. Founder Michael [X] called the plan painful. '
                         '[X] stock slid 4%.')
        self.assertEqual(mask_one('ZELWICK TECHNOLOGIES INC', 'BRIEF-Zelwick Cuts Outlook; Chairman Michael Zelwick To Retire'),
                         'BRIEF-[X] Cuts Outlook; Chairman Michael [X] To Retire')
        self.assertEqual(mask_one('QUORVEX CORP', 'Fire at Texas Quorvex refinery forces evacuation; Quorvex said no injuries.'),
                         'Fire at Texas [X] refinery forces evacuation; [X] said no injuries.')
        self.assertEqual(mask_one('QUORVEX CORP', 'Explosion At Georgia Quorvex Plant Injures Two'),
                         'Explosion At Georgia [X] Plant Injures Two')

    def test_english_word_tickers_in_upper_case_text(self):
        eng = frozenset({'dim', 'ox'})
        self.assertEqual(mask_one('DIMRAN INC', 'SEES DIM SINGLE DIGIT GROWTH; shares of DIM fell', tickers=['DIM'],
                                  english=eng), 'SEES DIM SINGLE DIGIT GROWTH; shares of [X] fell')
        self.assertEqual(mask_one('OXBRIDGE ZARTS INC', "- ZED QUILL, OX'S HEAD OF DESIGN, IS LEAVING OX",
                                  tickers=['OX'], english=eng),
                         "- ZED QUILL, [X]'S HEAD OF DESIGN, IS LEAVING [X]")

    def test_bylines_and_contact_blocks_removed(self):
        out = M.Masker([]).mask('Jet demand rises By Jane Placeholder and Zed Quill OSLO/ROME, May 1 (Reuters) - '
                                'Orders rose. ((jane.p@example.com; +44 (0) 20 7000 0000; Twitter @zquill)) Done')[0]
        self.assertEqual(out, 'Orders rose. Done')            # repeated headline + dateline at the start (v2 rule)
        out = M.Masker([]).mask('Up 3%. Jet demand rises By Jane Placeholder and Zed Quill OSLO/ROME, May 1 '
                                '(Reuters) - Orders rose.')[0]
        self.assertEqual(out, 'Up 3%. Jet demand rises OSLO/ROME, [DATE] Orders rose.')
        out = M.Masker([]).mask('x By Sam Quill and Rob Zed SAO PAULO, Apr 9 (Reuters Breakingviews) - y')[0]
        self.assertEqual(out, 'x SAO PAULO, [DATE] y')

    def test_residual_respects_guard(self):
        with use_tables(INVENTED_TABLES):
            m = M.Masker([('[X]', aliases('ZINDEX ZORB CORP'))])
            out = m.mask('the Zindex Composite said; Zindex rose')[0]
            self.assertEqual(out, 'the Zindex Composite said; [X] rose')
            self.assertEqual(m.residual(out), [])
            self.assertTrue(m.residual('Zindex rose'))
        m = M.Masker([('[X]', aliases('BRANWICK FOODS CO'))])
        out = m.mask('Judge said Maria Branwick alleged; Branwick said she was unsure')[0]
        self.assertEqual(out, 'Judge said Maria [X] alleged; [X] said she was unsure')     # v4 (N1)
        self.assertEqual(m.residual(out), [])
        self.assertTrue(m.residual('Maria Branwick alleged'))


class TestV3DatesAndUrls(unittest.TestCase):
    """D2: years glued to a time and short URLs with any TLD."""

    def test_year_time_and_short_urls(self):
        out = M.Masker([]).mask('3 May 202211:05 a.m. Totals in euros; Source: (shorturl.at/AB129); see qz.xy/abc')[0]
        self.assertEqual(out, '[DATE] [YEAR] 11:05 a.m. Totals in euros; Source: ([URL]); see [URL]')
        self.assertIn('year_time', M.date_residuals('x 202112:32 p.m.'))
        self.assertIn('url', M.date_residuals('Source: (shorturl.at/AB129)'))
        self.assertTrue(M.URL_RESIDUAL.search('see abc.zq/x1'))
        self.assertFalse(M.URL_RESIDUAL.search('U.S./China trade, e.g./i.e., Inc./Ltd'))
        self.assertEqual(M.Masker([]).mask('U.S./China trade at 12:32 and 1,20112:30')[0],
                         'U.S./China trade at 12:32 and 1,20112:30')


class TestV4FirmProducts(unittest.TestCase):
    """E1 (TD3C3): firm-owned product, unit, prefix and joint-product phrases (invented table and firms)."""

    def masker(self, *firms, english=frozenset()):
        return M.Masker([(tok, aliases(c, tickers=t, english=english)) for tok, c, t in firms], english)

    def test_product_phrase_absorbed(self):
        with use_tables(INVENTED_TABLES):
            m = self.masker(('[A]', 'ZORBLEX ONLINE INC', ()), ('[B]', 'QUINTARA CORP', ()))
            out = m.mask('QTRLY ZORBLEX CLOUDRIFT SERVICES INCOME ROSE; BRIEF-Zorblex Cloudrift Services - Maker Picks '
                         "Zorblex; Zorblex's Cloudrift Services unit; Zorblex® Parcel and Freight™ grew; Zorblex-Go "
                         'store; Zorblex Go opens; customers use Zorblex Huddle and Quintara 720; Zorblex Huddle Up With '
                         'X; Zorblex go ahead; Zorblex Huddle with peers; parcels Delivered by Zorblex arrive')[0]
            self.assertEqual(out, "QTRLY [A] INCOME ROSE; BRIEF-[A] - Maker Picks [A]; [A] unit; [A] grew; [A] store; "
                                  '[A] opens; customers use [A] and [B]; [A] Huddle Up With X; [A] go ahead; [A] Huddle '
                                  'with peers; parcels [A] arrive')
            self.assertEqual(m.residual(out), [])
            kinds = [e['kind'] for e in m.log]
            self.assertIn('product_absorb', kinds)
            self.assertIn('prefix_absorb', kinds)

    def test_acronyms_masked_when_phrase_absorbed_in_same_article(self):
        with use_tables(INVENTED_TABLES):
            m = self.masker(('[A]', 'ZORBLEX ONLINE INC', ()))
            h, b, *_ = m.mask_article('Rival Selected ZCS As Cloud Provider; ZPX volumes up',
                                      'QTRLY ZORBLEX CLOUDRIFT SERVICES INCOME; ZCS NET SALES UP; ZORBLEX PARCEL & FREIGHT')
            self.assertEqual(h, 'Rival Selected [A] As Cloud Provider; [A] volumes up')
            self.assertEqual(b, 'QTRLY [A] INCOME; [A] NET SALES UP; [A]')
            self.assertEqual(m.active_acronyms(), {'[A]': ['ZCS', 'ZPF', 'ZPX']})
            self.assertEqual(m.residual(h) + m.residual(b), [])
            self.assertTrue(any(x['kind'] == 'product_acronym' for x in m.residual('ZCS rose')))
            # no absorbed phrase in the article: the acronym is ordinary text and stays
            h, b, *_ = m.mask_article('ZCS volumes up', 'Zorblex said ZCS rose')
            self.assertEqual((h, b), ('ZCS volumes up', '[A] said ZCS rose'))
            self.assertEqual(m.active_acronyms(), {})

    def test_joint_product_partner_replaced(self):
        with use_tables(INVENTED_TABLES):
            m = self.masker(('[C]', 'ZELNTECH SE', ()))
            out = m.mask('a third dose of the Quillco and Zelntech vaccine; the Quillco-Zelntech shot; the '
                         'Zelntech/Quillco product; vaccine from Zelntech and partner Quillco Inc, researchers said; '
                         'Quillco & Zelntech said')[0]
            self.assertEqual(out, 'a third dose of the [C] vaccine; the [C] shot; the [C] product; vaccine from [C], '
                                  'researchers said; [C] said')
            self.assertEqual(m.residual(out), [])
            self.assertTrue(any(x['kind'] == 'joint_partner_adjacent' for x in m.residual('the Quillco and [C] vaccine')))
            # the partner masked as another firm: both tokens stay (no readable partner phrase)
            m2 = M.Masker([('[P]', aliases('QUILLCO INC')), ('[C]', aliases('ZELNTECH SE'))])
            self.assertEqual(m2.mask('the Quillco and Zelntech vaccine')[0], 'the [P] and [C] vaccine')

    def test_verify_style_check_fails_on_token_followed_by_phrase(self):
        with use_tables(INVENTED_TABLES):
            m = self.masker(('[A]', 'ZORBLEX ONLINE INC', ()))
            hits = m.residual('BRIEF-[A] Cloudrift Services Picks; Delivered by [A]')
            self.assertEqual(sorted({h['kind'] for h in hits}), ['prefix_before_token', 'product_after_token'])
            self.assertEqual(m.residual('BRIEF-[A] Picks [B] Cloudrift Services'), [])   # a different firm's token

    def test_firm_products_whole_word_prefix(self):
        with use_tables(INVENTED_TABLES):
            self.assertEqual(M.firm_products(['ZORBLEX ONLINE INC'])['keys'], ['ZORBLEX ONLINE'])
            self.assertEqual(M.firm_products(['ZORBLEX ONLINEWORKS INC'])['keys'], [])
            self.assertEqual(M.firm_products(['ZORBLEX ONLINE'])['keys'], ['ZORBLEX ONLINE'])
            acr = {p['phrase']: p['acronyms'] for p in M.firm_products(['ZORBLEX ONLINE INC'])['products']}
            self.assertEqual(acr['Cloudrift Services'], ['ZCS'])
            self.assertEqual(acr['Parcel & Freight'], ['ZPF', 'ZPX'])
            self.assertEqual(acr['Go'], [])

    def test_table_schema_and_source_notes(self):
        M.validate_tables(INVENTED_TABLES)
        bad = json.loads(json.dumps(INVENTED_TABLES))
        bad['firm_products']['QUINTARA CORP']['source'] = ''
        with self.assertRaises(ValueError):
            M.validate_tables(bad)
        bad = json.loads(json.dumps(INVENTED_TABLES))
        bad['firm_products']['quintara corp'] = bad['firm_products'].pop('QUINTARA CORP')
        with self.assertRaises(ValueError):
            M.validate_tables(bad)
        with self.assertRaises(ValueError):
            M.validate_tables({**INVENTED_TABLES, 'schema': 'other'})
        bad = json.loads(json.dumps(INVENTED_TABLES))
        bad['fixed_phrases']['ZINDEX']['guards'] = [[None, None]]
        with self.assertRaises(ValueError):
            M.validate_tables(bad)
        bad = json.loads(json.dumps(INVENTED_TABLES))                   # v5: code prefixes are upper-case letters
        bad['code_prefixes']['ZORBLEX ONLINE']['prefixes'] = ['q1']
        with self.assertRaises(ValueError):
            M.validate_tables(bad)
        bad = json.loads(json.dumps(INVENTED_TABLES))
        del bad['code_prefixes']
        with self.assertRaises(ValueError):
            M.validate_tables(bad)

    def test_public_code_holds_no_table_content(self):
        """N6: every firm-specific table is private; the module defaults are empty."""
        src = (S.CODE / 'td3_mask.py').read_text(encoding='utf-8')
        for name in ('COMMON_NAMES', 'FIXED_PHRASES', 'FIRM_PRODUCTS', 'CODE_PREFIXES'):
            self.assertIn(f'\n{name} = {{}}\n', src)
        with use_tables(None):
            self.assertEqual((len(M.COMMON_NAMES), len(M.FIXED_PHRASES), len(M.FIRM_PRODUCTS), len(M.CODE_PREFIXES)),
                             (0, 0, 0, 0))

    def test_real_build_requires_the_table(self):
        src = B.Sources(menu=None, features={}, evidence={}, news={}, graph=None, links=None, ric_identity=None,
                        articles=None, crsp_names=(), td2_bound=True, tables=None)
        with self.assertRaises(PermissionError):
            B.install_tables(src)
        with self.assertRaises(PermissionError):
            B.install_tables(B.Sources(**{**src.__dict__, 'tables': RUN_ROOT / 'no_table.json'}))


class TestV4DatesE2(unittest.TestCase):
    """E2 (TD3C3): month typos before '<day>, [YEAR]' and day-range remnants (invented strings)."""

    def test_month_typo_and_day_range(self):
        # v5 (TD3C4 N-f): invented strings only (no window of a sampled article)
        out = M.Masker([]).mask('ZEBRA LOANS MATURE NEAR QZO 7, 2022 * Decembr 14, 2021 memo; Phase 3, '
                                '2022 data; from March 2 And 29, 2021 the pilot ran; valid Feb 3-16, 2022 only; '
                                'IN OSLO, Feb 5-19, 2022; LAGGED Jan 27 -0.31 -0.28; [DATE] - 12,000 crates')[0]
        self.assertEqual(out, 'ZEBRA LOANS MATURE NEAR [DATE], [YEAR] * [DATE], [YEAR] memo; Phase 3, '
                              '[YEAR] data; from [DATE], [YEAR] the pilot ran; valid [DATE], [YEAR] only; '
                              'IN OSLO, [DATE], [YEAR]; LAGGED [DATE] -0.31 -0.28; [DATE] - 12,000 crates')
        self.assertEqual(M.date_residuals(out), [])
        self.assertIn('month_typo', M.date_residuals('MATURE NEAR QZO 7, [YEAR]'))
        self.assertIn('day_range', M.date_residuals('Games, [DATE]-20, [YEAR] closed'))
        self.assertIn('day_range', M.date_residuals('between [DATE] AND 31, FOR A'))
        self.assertEqual(M.date_residuals('Phase 3, [YEAR]; No. 6, [YEAR]; [DATE] -0.79'), [])

    def test_month_typo_pattern_equals_audit_pattern(self):
        audit = re.compile(r'\b[A-Za-z]{1,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,\s*\[YEAR\]')
        rx = dict((n, r) for n, r, _ in M.DATE_PATTERNS)['month_typo']
        for s in ('OCO 1, [YEAR]', 'Sepember 30th, [YEAR]', 'Ja. 5,[YEAR]', 'x 1, [YEAR]'):
            self.assertTrue(audit.search(s) and rx.search(s), s)
        for s in ('Phase 3, [YEAR]', 'Form 10, [YEAR]', 'no 6, [YEAR]'):
            self.assertTrue(audit.search(s))
            self.assertFalse(rx.search(s), s)            # words on the frozen non-month stop list


class TestV4FactSetSingleWordsN4(unittest.TestCase):
    """N4 (TD3C3): single-word variants that come only from FactSet-linked names must be distinctive."""

    def test_english_first_word_of_factset_name_dropped(self):
        ns = {'names': {P1: [('2010-01-01', 'QUORVEX CORP', 'QVX')]}, 'rics_by_permno': {},
              'fsnames': {P1: {'Golden Quorvex Holdings LLC', 'Zarvellian Quorvex Ltd'}}, 'english': ENGLISH}
        al = B.aliases_for(P1, '2021-03-05', ns)[0]
        self.assertNotIn('GOLDEN', al['names'])
        self.assertIn('GOLDEN QUORVEX', al['names'])
        self.assertIn('ZARVELLIAN', al['names'])                 # a distinctive FactSet-only first word is kept
        self.assertEqual(al['factset_single_words_dropped'], ['GOLDEN'])
        out = M.Masker([('[X]', al)], ENGLISH).mask('The Golden State Warriors beat Boston; Golden Quorvex rose; '
                                                    'Quorvex fell')[0]
        self.assertEqual(out, 'The Golden State Warriors beat Boston; [X] rose; [X] fell')
        # the same word from a CRSP name stays a variant
        ns2 = {'names': {P1: [('2010-01-01', 'GOLDENVALE CORP', 'GVX')]}, 'rics_by_permno': {}, 'english': ENGLISH}
        self.assertIn('GOLDENVALE', B.aliases_for(P1, '2021-03-05', ns2)[0]['names'])


class TestV4AdjacencySweep(unittest.TestCase):
    """E1 item 4: the adjacency sweep lists every token next to a capitalised word with its hidden raw text."""

    def test_sweep_rows_and_hidden_words(self):
        raw = 'BRIEF-Zorblex Cloudrift Picks Quillco and Zelntech; the zorblex unit; Pfizzle and Zelntech vaccine (Zelntech).'
        masked = ('BRIEF-[SUPPLIER_1] Cloudrift Picks Quillco and [CUSTOMER_2]; the zorblex unit; Pfizzle and '
                  '[CUSTOMER_2] vaccine ([CUSTOMER_2]).')
        rows = B.sweep_rows(raw, masked, 'headline')
        got = [(r['token'], r['side'], r['hidden'], r['before_words'], r['after_words']) for r in rows]
        self.assertEqual(got, [('[SUPPLIER_1]', 'both', 'Zorblex', ['BRIEF'], ['Cloudrift', 'Picks', 'Quillco']),
                               ('[CUSTOMER_2]', 'before', 'Zelntech', ['Picks', 'Quillco', 'and'], []),
                               ('[CUSTOMER_2]', 'before', 'Zelntech', ['unit', 'Pfizzle', 'and'], [])])
        self.assertEqual(B.sweep_positions(masked), [(r['start'], r['token'], r['side']) for r in rows])

    def test_sweep_side_rules(self):
        for text, side in [('[X] Reports', 'after'), ('[X] reports', None), ('BRIEF-[X] said', 'before'),
                           ('* [X] SAYS', 'after'), ('[X] and Quillco', 'after'), ('[X] and quillco', None),
                           ('Quillco and [X]', 'before'), ('quillco and [X]', None), ('[X] 720', 'after'),
                           ('U.S. [X] card', 'before'), ('[DATE] [X], said', None), ("[X]'s Cloudrift unit", 'after'),
                           ('[X], Quillco', None), ('Quillco/[X]', 'before'), ('[X] & Co', 'after'),
                           ('[LINK_3] Reports', 'after'), ('[NEWS_AGENCY] Reports', None),
                           # v5 (TD3C4 D1): glued tokens and number neighbours are rows
                           ('Trial Of [X]515 In', 'both'), ('of [X]515 in', 'after'), ('of zq[X] in', 'before'),
                           ('Site: 455 [X] Source', 'both'), ('site 455 [X] source', 'before')]:
            text = text.replace('[X]', '[SUPPLIER_OF_CUSTOMER_1]')
            pos = B.sweep_positions(text)
            self.assertEqual(pos[0][2] if pos else None, side, text)

    def test_sweep_review_check(self):
        root = RUN_ROOT / ('sweep_' + uuid.uuid4().hex[:6])
        metas = [{'sample_rank': 1, 'articles': [{'alt_id': 'a', 'headline': 'BRIEF-[FOCAL] Reports', 'body': 'none'}]}]
        rows = [{'rank': 1, 'alt_id': 'a', 'field': 'headline', 'start': 6, 'token': '[FOCAL]', 'side': 'both'}]
        for r in rows:
            r['row_id'] = B.sweep_row_id(r)
        sweep = R.new_text(root / 'ADJACENCY_SWEEP.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
        ok = R.new_json(root / 'ok.json', {'sweep_sha256': R.sha(sweep), 'gives_back': 0,
                                           'verdicts': {rows[0]['row_id']: {'verdict': 'safe', 'reason': 'verb'}}},
                        timed=False)
        self.assertEqual(B.sweep_review_check(ok, sweep, rows, metas)[0], {})
        bad = R.new_json(root / 'bad.json', {'sweep_sha256': R.sha(sweep), 'gives_back': 1,
                                             'verdicts': {rows[0]['row_id']: {'verdict': 'gives_back'}}}, timed=False)
        self.assertIn('adjacency_sweep_gives_back', B.sweep_review_check(bad, sweep, rows, metas)[0])
        stale = R.new_json(root / 'stale.json', {'sweep_sha256': '0' * 64, 'gives_back': 0,
                                                 'verdicts': {rows[0]['row_id']: {'verdict': 'safe', 'reason': 'x'}}},
                           timed=False)
        self.assertIn('adjacency_sweep_review_stale', B.sweep_review_check(stale, sweep, rows, metas)[0])
        noreason = R.new_json(root / 'noreason.json', {'sweep_sha256': R.sha(sweep), 'gives_back': 0,
                                                       'verdicts': {rows[0]['row_id']: {'verdict': 'safe'}}}, timed=False)
        self.assertIn('adjacency_sweep_row_without_verdict', B.sweep_review_check(noreason, sweep, rows, metas)[0])
        metas2 = [{'sample_rank': 1, 'articles': [{'alt_id': 'a', 'headline': 'BRIEF-[FOCAL] Reports; [FOCAL] Unit',
                                                   'body': 'none'}]}]
        self.assertIn('adjacency_sweep_positions_differ', B.sweep_review_check(ok, sweep, rows, metas2)[0])
        self.assertIn('adjacency_sweep_review_missing', B.sweep_review_check(root / 'none.json', sweep, rows, metas)[0])


LIMITATION_TABLE = {
    'schema': M.TABLE_SCHEMA, 'common_names': {}, 'fixed_phrases': {},
    'firm_products': {'ZORBEX': {'products': [{'phrase': 'Zorbcloud Services'}, {'phrase': 'Way', 'capitalized': True}],
                                 'joint_partners': [{'partner': 'Quillco'}], 'source': 'invented (limitation examples)'}},
    'code_prefixes': {'ZORBEX': {'prefixes': ['QX'], 'source': 'invented (limitation examples)'}}}
V5_ENGLISH = ENGLISH | {'cut', 'costs', 'launch', 'shares', 'deal', 'data', 'study', 'trial', 'zeta', 'cloud', 'weapon',
                        'seized', 'target'}


class TestV5Corrections(unittest.TestCase):
    """TD3C4 D1, D2, R1, N-a to N-e, with invented firms, texts and tables only."""

    def limitation_maskers(self):
        idx = {'by_first': {}, 'plus': {}}
        f, phrase = B._near_entry('ZORB LOGISTICS INC')
        idx['by_first'].setdefault(f, {})[phrase] = 'crsp:ZORB LOGISTICS INC'
        near = B.near_names(B.name_anchors(['ZORBEX INC']), idx, set(M.name_variants('ZORBEX INC')), V5_ENGLISH)
        x = aliases('ZORBEX INC', tickers=['ZRBX'], english=V5_ENGLISH, fs_names=['ZorBex Inc.'],
                    near=[n['phrase'] for n in near])
        y = aliases('Q R T BIO INC', tickers=['QRTB'], english=V5_ENGLISH)
        return lambda: M.Masker([('[X]', x), ('[Y]', y)], V5_ENGLISH)

    def test_limitation_examples_r1(self):
        """R1: every invented example of LIMITATION_CONTEXT_IDENTIFICATION holds, and the text quotes each one."""
        with use_tables(LIMITATION_TABLE):
            make = self.limitation_maskers()
            for firm, raw, expected in S.LIMITATION_EXAMPLES:
                m = make()
                out = m.mask(raw)[0]
                self.assertEqual(out, expected, raw)
                self.assertEqual(m.residual(out), [], raw)
        for quoted in ('"Zorbexon"', '"Zorbexpro"', '"ZRBXP"', '"ZetaZRBX"', '"ZRBX1181"', '"Zrbx1181"', '"ZRBX-1181"',
                       '"ZORBEX1181"', '"Zorbex-1181"', '"ZOR-09"', '"ZB-12"', '"QX-6137"', '"KQ-17"', '"Q.R.T.515"',
                       '"signed by Project ZB Parent"', '"ZB Ventures sold its stake"', '"ZRBX Ventures rose"',
                       '"Zorb Logistics said"', '"455 Zorbex Way"', '"Zorbex and Quillco said on Monday"',
                       '"Zorbex and Quillmark said"', '"Zorbexweapons"', '"Founder Jane Zorbex said"'):
            self.assertIn(quoted, S.LIMITATION_CONTEXT_IDENTIFICATION)

    def test_glued_tokens_d1(self):
        with use_tables(EMPTY_TABLES):
            m = M.Masker([('[Y]', aliases('Q R T BIO INC', tickers=['QRTB']))])
            self.assertEqual(m.mask('Trial Of Q.R.T.515 In Adults; Q.R.T.12AB data')[0],
                             'Trial Of [Y] In Adults; [Y] data')
            self.assertTrue(any(e['kind'] == 'glued_word' for e in m.log))
            self.assertIn('glued_token', {h['kind'] for h in m.residual('Trial Of [Y]515 In')})
            self.assertIn('glued_token', {h['kind'] for h in m.residual('pre[Y] In')})
            self.assertEqual(m.residual("[Y]'s unit; [Y]-led; ([Y])"), [])

    def test_house_number_only_with_street_phrase_d1(self):
        with use_tables(LIMITATION_TABLE):
            m = self.limitation_maskers()()
            self.assertEqual(m.mask('Site address: 455 Zorbex Way Source: county; 12B Zorbex Way, east')[0],
                             'Site address: [X] Source: county; [X], east')
            self.assertTrue(all(e.get('house_number_absorbed') for e in m.log if e['kind'] == 'product_absorb'))
        with use_tables(INVENTED_TABLES):                # '720' is a model phrase, not a street: the number stays
            m = M.Masker([('[B]', aliases('QUINTARA CORP'))])
            self.assertEqual(m.mask('Unit 5 Quintara 720 opened')[0], 'Unit 5 [B] opened')
        t = '[SUPPLIER_1]'
        rows, problems = B.sweep_text_problems('Site: 455 Zorbex Way Source', f'Site: 455 {t} Source', 'body')
        self.assertEqual(problems.get('d1_house_number_street'), 1)
        self.assertEqual(rows[0]['hidden'], 'Zorbex Way')
        rows, problems = B.sweep_text_problems('Site: 455 Zorbex Way Source', f'Site: {t} Source', 'body')
        self.assertEqual(dict(problems), {})
        rows, problems = B.sweep_text_problems('Trial Of QRT515 In', f'Trial Of {t}515 In', 'headline')
        self.assertEqual(problems.get('d1_glued_sweep_row'), 1)
        self.assertEqual(rows[0]['glued'], ['after'])
        rows, _ = B.sweep_text_problems('sold 12 Zorbex trucks', f'sold 12 {t} trucks', 'body')
        self.assertEqual(rows[0]['number_neighbour'], ['before'])

    def test_code_prefix_terms_d2(self):
        p = M.code_prefix_terms(['QOLVANE THERAPEUTICS INC'], ['QolVane Therapeutics, Inc.'], ['QLVN'])
        self.assertEqual(p['QLVN'], ['ticker'])
        self.assertIn('initialism', p['QT'])             # CRSP name initials
        self.assertIn('initialism', p['QVT'])            # camel-case parts of the FactSet name
        for k in ('QO', 'QOL', 'QOLV', 'QOLVANE'):
            self.assertIn('leading', p[k])
        self.assertNotIn('THE', p)
        with use_tables(INVENTED_TABLES):
            self.assertEqual(M.code_prefix_terms(['ZORBLEX ONLINE INC'])['QX'], ['table'])
        self.assertNotIn('MID', M.code_prefix_terms(['MIDLORA INC']))       # date word on CODE_PREFIX_STOP
        self.assertNotIn('AMERICAN', M.code_prefix_terms(['AMERICAN ZORB CO']))   # generic first word
        self.assertIn('AZ', M.code_prefix_terms(['AMERICAN ZORB CO']))

    def test_firm_codes_masked_whole_d2(self):
        eng = V5_ENGLISH
        with use_tables(EMPTY_TABLES):
            m = M.Masker([('[Q]', aliases('QOLVANE THERAPEUTICS INC', tickers=['QLVN'], english=eng,
                                          fs_names=['QolVane Therapeutics, Inc.'])),
                          ('[M]', aliases('MIDLORA INC', tickers=['MDLR'], english=eng)),
                          ('[B]', aliases('BAAZ CORP', tickers=['BAAZ'], english=eng))], eng)
            out = m.mask('Phase 2 Of QVT-388 Began; QLVN1181 and Qlvn-1181/2 data; QOL-05 trial; QT-12 dose; KQ-17 '
                         'kept; CO2 kept; Opens In Mid-2021; rated Baa2 by an agency; qvt-388 lower; MDLR-3 pump')[0]
            self.assertEqual(out, 'Phase 2 Of [Q] Began; [Q] and [Q] data; [Q] trial; [Q] dose; KQ-17 kept; CO2 kept; '
                                  'Opens In Mid-[YEAR]; rated Baa2 by an agency; qvt-388 lower; [M] pump')
            self.assertEqual(m.residual(out), [])
            self.assertIn('firm_code', {h['kind'] for h in m.residual('Trial Of QVT-388 In')})

    def test_near_names_from_name_histories_d2(self):
        root = RUN_ROOT / ('near_' + uuid.uuid4().hex[:6])
        crsp = R.new_text(root / 'crsp_stocknames.csv', _csv(
            [(40001, '1', '1', 'QRTE', 'Q R T ENERGY CO', '2010-01-01', '2030-12-31'),
             (40002, '2', '2', 'QRM', 'Q R LOGISTICS INC', '2021-07-01', '2030-12-31'),
             (40003, '3', '3', 'ZRB', 'ZORBE PARTNERS LP', '2012-01-01', '2030-12-31'),
             (40004, '4', '4', 'ZRBA', 'ZORBEXA LABS INC', '2012-01-01', '2030-12-31'),
             (40005, '5', '5', 'ZRBL', 'ZORBEXALOT HOLDINGS INC', '2012-01-01', '2030-12-31'),
             (40006, '6', '6', 'TGD', 'TARGETED ZORB INC', '2012-01-01', '2030-12-31'),
             (40007, '7', '7', 'QRL', 'Q R LATE CORP', '2023-03-01', '2030-12-31')],
            ['permno', 'ncusip', 'cusip', 'ticker', 'comnam', 'namedt', 'nameenddt']))
        fs = root / 'factset.parquet'
        write_factset(fs, [('', 'QR Logistics, LLC', '2021-07-01', 'N1', '', ''),
                           ('', 'QR Postseal Inc.', '2023-01-05', 'N2', '', '')])
        src = B.Sources(menu=None, features={}, evidence={}, news={}, graph=None, links=None, ric_identity=None,
                        articles=None, crsp_names=(crsp,), td2_bound=False, factset=fs)
        anchors = B.name_anchors(['Q R T ENERGY CO', 'ZORBEX INC', 'TARGET ZORB CORP'])
        self.assertEqual(anchors, {'QRT', 'ZORBEX', 'TARGET'})
        idx = B.near_name_index(src, anchors)
        near = {a: [n['phrase'] for n in B.near_names({a}, idx, set(), V5_ENGLISH)] for a in sorted(anchors)}
        self.assertEqual(near['QRT'], ['Q.R. LOGISTICS', 'QR LOGISTICS'])     # minus one letter; rows after the
        self.assertEqual(near['ZORBEX'], ['ZORBE PARTNERS', 'ZORBEXA LABS'])  # seal never read; plus 4 letters no
        self.assertEqual(near['TARGET'], [])                                  # an English first word is no near name
        al = aliases('Q R T ENERGY CO', tickers=['QRTE'], near=near['QRT'])
        out = M.Masker([('[X]', al)]).mask('QR Logistics raised debt; Q.R. Logistics fell; qr midstream lower')[0]
        self.assertEqual(out, '[X] raised debt; [X] fell; qr midstream lower')

    def test_ordinal_dates_n_a(self):
        out = M.Masker([]).mask('on the 30th of September 2021, the 1st of May, 4th July and September the 30th; '
                                'the 2nd of the pilots')[0]
        self.assertEqual(out, 'on the [DATE], the [DATE], [DATE] and [DATE]; the 2nd of the pilots')
        self.assertEqual(M.date_residuals(out), [])
        self.assertIn('ordinal_day_month', M.date_residuals('the 30th of September'))
        # v5: a year cut to three digits after a masked date
        self.assertEqual(M.Masker([]).mask('ZEBRA PAYOUT NEAR 12 MARCH 202')[0], 'ZEBRA PAYOUT NEAR [DATE] [YEAR]')
        self.assertEqual(M.Masker([]).mask('due 12 March 205 units shipped')[0], 'due [DATE] 205 units shipped')
        self.assertIn('year_fragment', M.date_residuals('ON [DATE] 202.'))

    def test_project_code_names(self):
        with use_tables(EMPTY_TABLES):
            m = M.Masker([('[Q]', aliases('QOLVANE SOFTWARE INC', tickers=['QLVN']))])
            out = m.mask('Bid Signed By Project Qs Parent; PROJECT QS MERGER SUB LLC signed; Project Kuiper; '
                         'project qs parent')[0]
            self.assertEqual(out, 'Bid Signed By [Q]; [Q] LLC signed; Project Kuiper; project qs parent')
            self.assertIn('firm_code', {h['kind'] for h in m.residual('paid Project QS Parent')})

    def test_story_ids_and_alert_numbers_n_b(self):
        out = M.Masker([]).mask('Source text for Eikon: [Pn7QzX4qa] More; see [Pn7Qz ...')[0]
        self.assertEqual(out, 'Source text for Eikon: More; see ...')
        self.assertIn('story_ids', M.date_residuals('Source text for Eikon: [Pn7QzX4qa]'))
        out = M.Masker([]).mask('retracted items: QZX012AB QZX012AC QZX012A9 end; code QZX012AB alone')[0]
        self.assertEqual(out, 'retracted items: end; code QZX012AB alone')

    def test_pass_order_n_c(self):
        t = {**EMPTY_TABLES, 'firm_products': {'ZELNTECH SE': {'products': [{'phrase': 'Air', 'capitalized': True}],
                                                               'joint_partners': [{'partner': 'Quillco'}],
                                                               'source': 'invented'}}}
        with use_tables(t):
            m = M.Masker([('[C]', aliases('ZELNTECH SE'))])
            out = m.mask('the Zelntech Air Quillco shot was approved')[0]
            self.assertEqual(out, 'the [C] shot was approved')
            self.assertEqual(m.residual(out), [])

    def test_guard_floor_n_e(self):
        self.assertEqual(S.MIN_FREE_BYTES, 25 * 1024 ** 3)
        self.assertIn('MIN_FREE_BYTES // 1024 ** 3', (S.CODE / 'td3_runtime.py').read_text(encoding='utf-8'))


class TestV6UnitNames(unittest.TestCase):
    """TD3C5 D1: a unit named with the firm's ticker or initialism and a corporate-unit word (invented firms)."""

    def test_unit_prefixes_and_masking(self):
        codes = M.code_prefix_terms(['QARVEL INDUSTRIES INC'], ['QarVel Industries, Inc.'], ['QVLX'])
        up = M.unit_prefixes(codes)
        self.assertIn('QI', up)                       # initialism of the CRSP name
        self.assertIn('QVLX', up)                     # ticker
        self.assertNotIn('QA', up)                    # a leading-letter prefix only: not a unit prefix
        with use_tables(EMPTY_TABLES):
            m = M.Masker([('[Q]', aliases('QARVEL INDUSTRIES INC', tickers=['QVLX'], fs_names=['QarVel Industries, Inc.']))])
            out = m.mask('* QARVEL INDUSTRIES INC - UNIT QI VENTURES NO LONGER HOLDS SHARES; Qi Capital rose; QVLX Labs '
                         "said; QI Ventures' stake; qi ventures lower; QI Morgan kept; QA Ventures kept; QIX Capital kept")[0]
            # the ticker is masked first by the ticker rule, so its unit word stays after the token ("[Q] Labs")
            self.assertEqual(out, "* [Q] - UNIT [Q] NO LONGER HOLDS SHARES; [Q] rose; [Q] Labs said; [Q]' stake; qi "
                                  'ventures lower; QI Morgan kept; QA Ventures kept; QIX Capital kept')
            self.assertTrue(any(e['kind'] == 'initialism_unit' for e in m.log))
            self.assertEqual(m.residual(out), [])
            self.assertIn('initialism_unit', {h['kind'] for h in m.residual('UNIT QI VENTURES SOLD')})
        # prefixes on the stop lists are never unit prefixes ("US Bank", "AI Labs")
        self.assertNotIn('US', M.unit_prefixes({'US': ['initialism'], 'AI': ['ticker'], 'QZ': ['leading']}))
        self.assertEqual(M.unit_prefixes({'US': ['initialism'], 'AI': ['ticker'], 'QZ': ['leading']}), [])

    def test_unit_words_frozen(self):
        for w in ('Ventures', 'Capital', 'Holdings', 'Group', 'Partners', 'Labs', 'Energy', 'Solutions', 'Bank'):
            self.assertIn(w, M.UNIT_WORDS)
        self.assertNotIn('Morgan', M.UNIT_WORDS)


class TestBuildFixture(unittest.TestCase):
    def setUp(self):
        self.fx = fixture()

    def test_sample_years_and_rows(self):
        metas = self.fx['metas']
        self.assertEqual(sorted(m['focal_permno'] for m in metas), [F1, F2])
        for m in metas:
            self.assertIn(pd.Timestamp(m['origin_session']).year, S.YEARS)
            self.assertIn(pd.Timestamp(m['cutoff']).year, S.YEARS)
        self.assertNotIn(firm_id(F3), json.dumps(metas))
        self.assertNotIn(firm_id(F4), json.dumps(metas))
        sample = (self.fx['build_dir'] / 'PILOT_SAMPLE.csv').read_text(encoding='utf-8')
        self.assertNotIn('2023-', sample)
        self.assertEqual(sum(bool(r['reask']) for r in self.fx['records']), 1)

    def test_no_2023_reads(self):
        with self.assertRaises(PermissionError):
            B.features_year(self.fx['sources'], 2023)
        with self.assertRaises(PermissionError):
            list(B.sealed_batches(self.fx['sources'].articles, ['alt_id'], 'date', '2022-01-01', '2023-01-31'))
        fs = B.factset_names(self.fx['sources'].factset, {P6: {'88888801'}})
        self.assertEqual(fs[P6], {'Nrvx Global Holdings, Inc.'})       # the 2023 FactSet row is dropped
        names, ticks = B.factset_aliases(self.fx['sources'].factset, {P6: {'888888'}, F1: {'111111'}})
        self.assertEqual(names[P6], {'Nrvx Global Holdings, Inc.'})
        self.assertNotIn('PSTL', ticks[P6])
        self.assertIn('Zorblaxian Holdings', names[F1])               # blank-CUSIP row of the same FactSet id
        self.assertEqual(ticks[F1], {'ZRBX', 'ZBXH'})                  # the Frankfurt ticker is excluded
        crsp = SC.load_crsp(self.fx['sources'].crsp_names)
        self.assertNotIn('LATERNAME CORP', crsp[F1]['names'])

    def test_post_cutoff_text_never_enters_inputs(self):
        text = '\n'.join(r['input_text'] for r in self.fx['records'])
        for marker in ('FUTUREMARKER', 'SECONDROW', 'BADYEARMARKER', 'EDGEMARKER'):
            self.assertNotIn(marker, text)
        rej = self.fx['receipt']['counts']['article_rejects']
        self.assertEqual(rej.get('outside_window_or_origin_day'), 3)
        self.assertEqual(rej.get('available_not_before_cutoff'), 1)
        self.assertEqual(rej.get('partner_not_tagged_before_origin_month'), 1)
        for m in self.fx['metas']:
            o, c = pd.Timestamp(m['origin_session']), pd.Timestamp(m['cutoff'])
            for a in m['articles']:
                self.assertLess(pd.Timestamp(a['news_day']), o)
                self.assertLess(pd.Timestamp(a['news_available_at']), c)
                self.assertTrue(1 <= a['days_before'] <= 7)

    def test_most_recent_article_and_aggregation(self):
        m = next(x for x in self.fx['metas'] if x['focal_permno'] == F1)
        chosen = {s['permno']: s['article'] for s in m['shown']}
        self.assertEqual(chosen, {P1: 'A1', P2: 'A2', P3: 'A3b', P5: 'A5'})
        self.assertEqual(len(m['articles']), 4)

    def test_masking_in_inputs(self):
        for r in self.fx['records']:
            for bad in FORBIDDEN_NAMES + FORBIDDEN_TICKERS + ['Jane Doe', 'LATERNAME', 'Ostra', 'mavrick.com', 'https',
                                                              'bit.ly', 'STORY_', 'FY22', "Q3'21", 'nZZ9TEST01',
                                                              'Reuters']:
                self.assertNotIn(bad, r['input_text'])
            self.assertIn('[FOCAL]', r['input_text'])
        f1 = next(m for m in self.fx['metas'] if m['focal_permno'] == F1)
        f2 = next(m for m in self.fx['metas'] if m['focal_permno'] == F2)
        a2 = next(a for a in f1['articles'] if a['alt_id'] == 'A2')
        a3b = next(a for a in f1['articles'] if a['alt_id'] == 'A3b')
        b6 = next(a for a in f2['articles'] if a['alt_id'] == 'B6')
        self.assertIn('Summit County', a2['body'])          # not an owner article for a Summit firm: kept
        self.assertIn('[URL]', a2['body'])
        self.assertIn('[EMAIL]', a2['body'])
        self.assertIn('<[CUSTOMER_OF_SUPPLIER_1]>', a3b['body'])      # v3: issuer RIC of the archive rics field
        self.assertEqual(a3b['article_extras'], {'[CUSTOMER_OF_SUPPLIER_1]': {'rics': ['BLNDP.N'],
                                                                             'tickers_extra': ['BLNDP']}})
        self.assertEqual(a3b['archive_rics'], ['BLND.N', 'BLNDP.N'])
        self.assertTrue(any(e['token'] == '[CUSTOMER_OF_SUPPLIER_1]' for e in a3b['replacement_log']))
        self.assertEqual(b6['owner_terms'], {'[CUSTOMER_1]': ['SUMMIT']})
        self.assertNotIn('Summit', b6['body'])              # owner rule in the article chosen for that firm
        self.assertNotIn('SUMMIT', b6['body'])
        self.assertIn('a summit with suppliers', b6['body'])  # lower case kept
        self.assertIn('[NEWS_AGENCY] reported', b6['body'])
        self.assertIn('Postsealname stays', b6['body'])      # the FactSet row dated after the seal is never read
        self.assertEqual(self.fx['receipt']['counts']['accepted'], 2)
        m = next(x for x in self.fx['metas'] if x['focal_permno'] == F1)
        self.assertNotIn('LATERNAME CORP', m['aliases']['[FOCAL]']['comnams'])     # dated after the seal
        self.assertIn('QUINTRELL NEWCO CORP', m['aliases']['[CUSTOMER_1]']['comnams'])   # after origin, before seal
        self.assertEqual(m['aliases']['[CUSTOMER_1]']['comnams_after_origin'], ['QUINTRELL NEWCO CORP'])
        self.assertIn('Zorblax Industries, Inc.', m['aliases']['[FOCAL]']['factset_names'])
        self.assertIn('Zorblaxian Holdings', m['aliases']['[FOCAL]']['factset_names'])
        self.assertEqual(m['strict_residual_hits'], 0)
        self.assertEqual(m['date_residual_hits'], 0)
        a1 = next(a for a in m['articles'] if a['alt_id'] == 'A1')       # v4 (E1): product phrase and acronym
        self.assertIn('[CUSTOMER_1] grew and [CUSTOMER_1] margins rose', a1['body'])
        self.assertEqual(a1['active_acronyms'], {'[CUSTOMER_1]': ['QZS']})
        self.assertTrue(any(e['kind'] == 'product_absorb' for e in a1['replacement_log']))
        self.assertEqual(self.fx['receipt']['mask_table_state']['firm_products'], 2)
        a5 = next(a for a in m['articles'] if a['alt_id'] == 'A5')       # v5 (D1, D2): firm code and house number
        self.assertIn('Trial of [SUPPLIER_OF_SUPPLIER_OF_CUSTOMER_1] started at [SUPPLIER_OF_SUPPLIER_OF_CUSTOMER_1].',
                      a5['body'])
        self.assertTrue(a5['body_raw'].endswith('12 Ostranix Way.'))    # the raw text is kept for verify
        self.assertTrue(any(e.get('house_number_absorbed') for e in a5['replacement_log']))
        self.assertTrue(any(e['kind'] == 'firm_code' for e in a5['replacement_log']))
        self.assertEqual(self.fx['receipt']['mask_tables_sha256'],
                         R.sha(self.fx['sources'].tables))

    def test_role_tokens_follow_fsb3_direction(self):
        m = next(x for x in self.fx['metas'] if x['focal_permno'] == F1)
        tok = {s['permno']: s['token'] for s in m['shown']}
        self.assertEqual(tok, {P1: '[CUSTOMER_1]', P2: '[SUPPLIER_1]', P3: '[CUSTOMER_OF_SUPPLIER_1]',
                               P5: '[SUPPLIER_OF_SUPPLIER_OF_CUSTOMER_1]'})
        self.assertEqual(m['links'], [{'permno': P4, 'token': '[LINK_1]'}])
        text = next(r for r in self.fx['records'] if r['key'] == m['key'])['input_text']
        self.assertIn('[FOCAL] -> [CUSTOMER_1] -> [LINK_1] -> [SUPPLIER_OF_SUPPLIER_OF_CUSTOMER_1]', text)
        self.assertIn('[CUSTOMER_1] (customer of [FOCAL]; link age', text)

    def test_edge_ages_rendered_r4_and_grammar_r5(self):
        m = next(x for x in self.fx['metas'] if x['focal_permno'] == F1)
        text = next(r for r in self.fx['records'] if r['key'] == m['key'])['input_text']
        for s in m['shown']:
            expect = (O1.date() - B._ny_date(O1 - pd.Timedelta(days=900 + 10 * s['hop']))).days
            self.assertEqual(s['edge_age_days'], expect)
            self.assertIn(f"link age {expect} days", text)
        self.assertIn('Link age: days from the date of the newest relationship record', text)
        self.assertNotIn('2015', text)
        self.assertEqual(B._firms(1), '1 linked firm')
        self.assertEqual(B._firms(2), '2 linked firms')

    def _verify(self):
        if 'verify' not in self.fx:
            self.fx['verify'] = B.verify(self.fx['build_dir'], self.fx['sources'])
        return self.fx['verify']

    def test_cli_verify_summary_keys(self):
        import re
        src = (S.CODE / 'run_td3.py').read_text(encoding='utf-8')
        block = src[src.index("print('VERIFY'"):src.index("flush=True)", src.index("print('VERIFY'"))]
        keys = re.findall(r"'([a-z_]+)'", block)
        self.assertTrue(keys)
        self.assertEqual([k for k in keys if k not in self._verify()], [])

    def test_independent_scan_and_verify(self):
        self.assertTrue(self.fx['receipt']['independent_scan']['factset_used'])
        self.assertTrue((self.fx['build_dir'] / 'ADJACENCY.jsonl').exists())
        result = self._verify()
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['scan_adjudication']['hits'], len(self.fx['hits']))
        self.assertEqual(result['adjacency_review']['inferable_masks'], 0)
        self.assertEqual(result['adjacency_sweep']['verdicts'], {'safe': result['adjacency_sweep']['rows']})
        self.assertGreater(result['adjacency_sweep']['rows'], 0)
        self.assertEqual(result['mask_tables_sha256'], R.sha(self.fx['sources'].tables))
        sweep = R.read_jsonl(self.fx['build_dir'] / 'ADJACENCY_SWEEP.jsonl')
        self.assertTrue(all(r['hidden'] for r in sweep), [r for r in sweep if not r['hidden']])
        bad = sweep_review(self.fx['build_dir'], verdict='gives_back')
        self.assertIn('adjacency_sweep_gives_back', B.sweep_review_check(
            bad, self.fx['build_dir'] / 'ADJACENCY_SWEEP.jsonl', sweep, self.fx['metas'])[0])
        self.assertGreater(result['ric_checks'].get('archive_rics_attributed', 0), 0)
        for r in self.fx['records']:
            self.assertNotRegex(r['input_text'], r'(?<![\d.,])(?:19|20)\d\d(?!\d)(?! linked)')
            self.assertNotIn(str(F1), r['input_text'])
            self.assertNotIn(firm_id(F1), r['input_text'])

    def test_build_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            B.build(self.fx['build_dir'], self.fx['sources'], n=2, reask_n=1)

    def test_verify_v5_remask_and_sweep_hidden(self):
        """v5: verify re-masks every article from its stored raw text and recomputes the sweep's hidden words; a copy
        of the build whose stored raw text differs fails both checks."""
        result = self._verify()
        self.assertTrue(result['passed'], result)
        self.assertGreater(result['adjacency_sweep']['remask_checked_articles'], 0)
        self.assertEqual(result['adjacency_sweep']['recomputed_with_hidden'], result['adjacency_sweep']['rows'])
        src, dst = self.fx['build_dir'], RUN_ROOT / ('tampered_' + uuid.uuid4().hex[:6])
        for p in sorted(src.iterdir()):
            if p.is_file() and p.name not in ('VERIFY.json', 'INPUT_META.jsonl'):
                R.new_text(dst / p.name, p.read_text(encoding='utf-8'))
        metas = R.read_jsonl(src / 'INPUT_META.jsonl')
        r0 = next(r for r in R.read_jsonl(src / 'ADJACENCY_SWEEP.jsonl') if r['hidden'])
        for m in metas:
            for x in m['articles']:
                if (m['sample_rank'], x['alt_id']) == (r0['rank'], r0['alt_id']):
                    f = r0['field'] + '_raw'
                    x[f] = x[f].replace(r0['hidden'], 'Q' + r0['hidden'][1:], 1)    # the hidden word altered
        R.new_text(dst / 'INPUT_META.jsonl', ''.join(json.dumps(m) + '\n' for m in metas))
        bad = B.verify(dst, self.fx['sources'])
        self.assertFalse(bad['passed'])
        self.assertIn('remask_differs', bad['problems'])
        self.assertIn('adjacency_sweep_hidden_differs', bad['problems'])


class TestIndependentScanner(unittest.TestCase):
    """The second scanner flags each leak class from its own name sources and rules (not the masking rules)."""

    def _scan(self, rows, fs_rows, text, archive_rics=(), tokens=()):
        root = RUN_ROOT / ('scan_' + uuid.uuid4().hex[:6])
        crsp = root / 'crsp_stocknames.csv'
        R.new_text(crsp, _csv([(p, cu, cu, t, c, d, '2030-12-31') for p, t, c, d, cu in rows],
                              ['permno', 'ncusip', 'cusip', 'ticker', 'comnam', 'namedt', 'nameenddt']))
        fs = root / 'factset.parquet'
        write_factset(fs, fs_rows)
        als = {f'[T{i}]': {'permno': p} for i, (p, *_rest) in enumerate(rows)}
        metas = [{'sample_rank': 1, 'key': 'k1', 'aliases': als,
                  'articles': [{'alt_id': 'x1', 'headline': 'none', 'body': text, 'archive_rics': list(archive_rics),
                                'tokens': list(tokens)}]}]
        return SC.scan_build(metas, (crsp,), fs, {}, ENGLISH), metas, crsp, fs

    def test_scanner_flags_leaks(self):
        rows = [(30001, 'QVZX', 'Q V Z X SYSTEMS INC', '2010-01-01', '90000001'),
                (30002, 'GLMQ', 'GLORMIQOS BAKERY INC', '2010-01-01', '90000002'),
                (30003, 'ZLX', 'ZYLBORANTH CORP AMERICA HLDGS', '2010-01-01', '90000003'),
                (30004, 'KBZ', 'K B Z INC', '2010-01-01', '90000004'),
                (30005, 'BNZQ', 'BANK NEW ZORQUE MELLOWAN CORP', '2010-01-01', '90000005'),
                (30006, 'ZZZ', 'ZEBULON QUARTZ INC', '2010-01-01', '90000006')]
        fs_rows = [('90000005', 'The Bank of New Zorque Mellowan Corp.', '2010-01-01', 'B5', 'BNZQ', ''),
                   ('90000006', 'JPZebulon Quartz & Co.', '2010-01-01', 'B6', 'ZZZ', '')]
        text = ("QVZX Systems rose; Glormiqō's said; Zylborcorp gained; Kbz fell; BNZM unit; JP Zebulon said; "
                "plain words only")
        (hits, summ, adj), metas, crsp, fs = self._scan(rows, fs_rows, text)
        got = {(h['token'], h['check']) for h in hits}
        self.assertIn(('[T0]', 'ngram'), got)                 # "qvzx systems" (collapsed initials)
        self.assertIn(('[T1]', 'rare_token'), got)            # Glormiqō's -> (accent fold) glormiqos
        self.assertIn(('[T2]', 'portmanteau'), got)           # Zylborcorp = zylbor(anth) + corp
        self.assertIn(('[T3]', 'ticker_title'), got)          # Kbz
        self.assertIn(('[T4]', 'acronym'), got)               # BNZM from FactSet name initials
        self.assertIn(('[T5]', 'joined_text_tokens'), got)    # JP Zebulon -> jpzebulon
        self.assertTrue(summ['factset_used'])
        metas[0]['articles'][0]['body'] = "[T0] rose; [T1] said; [T2] gained; [T3] fell; [T4] unit; [T5] said; plain"
        self.assertEqual(SC.scan_build(metas, (crsp,), fs, {}, ENGLISH)[0], [])

    def test_scanner_v3_fuzzy_checks(self):
        rows = [(31001, 'TVQ', 'TORVANDEL ENERGY CORP', '2010-01-01', '91000001'),
                (31002, 'QRX', 'Q R X RESORTS INTL', '2010-01-01', '91000002'),
                (31003, 'QRXA', 'QORAXIN SYSTEMS INC', '2010-01-01', '91000003'),
                (31004, 'GRLF', 'GRELFORD INC', '2010-01-01', '91000004'),
                (31005, 'NWZ', 'NORTHWIND ZEPHYR TECHNOLOGIES', '2010-01-01', '91000005')]
        text = ('a leak at the Torvanndel plant; BRIEF-BetQRX Adds; NASDAQ TRADE HALT <QRXAP.O>; '
                'the Northwind Zephyrs Technology unit; Grelforde said')
        (hits, summ, adj), *_ = self._scan(rows, [], text, archive_rics=['QRXAP.O'], tokens=['[T2]'])
        got = {(h['token'], h['check']) for h in hits}
        self.assertIn(('[T0]', 'fuzzy_token'), got)           # Torvanndel ~ torvandel
        self.assertIn(('[T1]', 'embedded'), got)              # BetQRX
        self.assertIn(('[T2]', 'ric_residual'), got)          # QRXAP.O (root starts with QRXA)
        self.assertIn(('[T4]', 'fuzzy_phrase'), got)          # northwind zephyrs technology ~ northwind zephyr
        self.assertIn(('[T3]', 'fuzzy_token'), got)           # Grelforde ~ grelford

    def test_scanner_name_echo(self):
        """TD3C2 manual read: a visible name word in the same text as that firm's token is reported for review."""
        rows = [(32001, 'BRWQ', 'BRANWICK FOODS CO', '2010-01-01', '92000001')]
        (hits, summ, adj), *_ = self._scan(rows, [], 'Judge said Maria Branwick alleged; [T0] said she was unsure')
        got = {(h['token'], h['check']) for h in hits}
        self.assertIn(('[T0]', 'rare_token'), got)
        self.assertIn(('[T0]', 'name_echo'), got)
        (hits, summ, adj), *_ = self._scan(rows, [], 'Judge said Maria Branwick alleged; Branwick said she was unsure')
        got = {(h['token'], h['check']) for h in hits}
        self.assertIn(('[T0]', 'rare_token'), got)
        self.assertNotIn(('[T0]', 'name_echo'), got)

    def test_scanner_v5_checks(self):
        """v5 (TD3C4): firm codes, near prefixes, glued tokens and house numbers, from the scanner's own rules."""
        rows = [(33001, 'QLVN', 'QOLVANE THERAPEUTICS INC', '2010-01-01', '93000001'),
                (33002, 'QRTE', 'Q R T ENERGY CO', '2010-01-01', '93000002'),
                (33003, 'QOVX', 'QOVAXIN PHARMACEUTICALS INC', '2010-01-01', '93000003')]
        fs_rows = [('93000001', 'QolVane Therapeutics, Inc.', '2010-01-01', 'Q1', 'QLVN', '')]
        text = 'Phase 2 of QVT-388 began; QR Logistics notes; QVX7731 dosed; [T0]515 study; 455 [T1] Source'
        (hits, summ, adj), *_ = self._scan(rows, fs_rows, text)
        got = {(h['token'], h['check'], h['term']) for h in hits}
        self.assertIn(('[T0]', 'firm_code', 'QVT'), got)          # camel-case initials of the FactSet name
        self.assertIn(('[T1]', 'near_prefix', 'QR'), got)         # QRT minus one letter + a capitalised word
        self.assertIn(('[T2]', 'firm_code', 'QVX'), got)          # a ticker subsequence (QOVX -> QVX)
        self.assertIn(('[T0]', 'glued_token', '[T0]515'), got)
        self.assertIn(('[T1]', 'house_number', '455 [T1]'), got)

    def test_scanner_v6_initialism_unit(self):
        """v6 (TD3C5 D1): a ticker or name initialism followed by a corporate-unit word, from the scanner's own rules."""
        rows = [(34001, 'QVLX', 'QARVEL INDUSTRIES INC', '2010-01-01', '94000001')]
        text = 'UNIT QI VENTURES SOLD; Qi Capital rose; QI Morgan said; qi ventures lower; [T0] said'
        (hits, summ, adj), *_ = self._scan(rows, [], text)
        got = [(h['token'], h['check'], h['term'], h['matched']) for h in hits if h['check'] == 'initialism_unit']
        self.assertEqual(sorted(got), [('[T0]', 'initialism_unit', 'QI', 'QI VENTURES'),
                                       ('[T0]', 'initialism_unit', 'QI', 'Qi Capital')])

    def test_adjacency_report(self):
        rows = SC.adjacency('BRIEF-[X] Reports; JP [Y], and [Z]')
        self.assertEqual(rows, [('[X]', 'BRIEF-', 'Reports'), ('[Y]', 'JP', 'and'), ('[Z]', 'and', '')])

    def test_adjudication_and_reviews(self):
        hits = [{'hit_id': 'a', 'firm_class': 'shown'}, {'hit_id': 'b', 'firm_class': 'link'}]
        root = RUN_ROOT / ('adj_' + uuid.uuid4().hex[:6])
        p1 = R.new_json(root / 'ok.json', {'adjudications': {'a': {'verdict': 'false_positive'},
                                                             'b': {'verdict': 'false_positive'}}}, timed=False)
        self.assertEqual(B.adjudication_check(hits, p1)[0], {})
        p2 = R.new_json(root / 'leak.json', {'adjudications': {'a': {'verdict': 'confirmed_leak'},
                                                               'b': {'verdict': 'false_positive'}}}, timed=False)
        self.assertEqual(B.adjudication_check(hits, p2)[0].get('confirmed_leak_focal_or_shown'), 1)
        p3 = R.new_json(root / 'missing.json', {'adjudications': {'a': {'verdict': 'false_positive'}}}, timed=False)
        self.assertIn('scan_adjudication_ids_differ', B.adjudication_check(hits, p3)[0])
        self.assertIn('scan_adjudication_missing', B.adjudication_check(hits, root / 'none.json')[0])
        adj = R.new_text(root / 'ADJACENCY.jsonl', '{"a": 1}\n')
        good = R.new_json(root / 'rev_ok.json', {'adjacency_sha256': R.sha(adj), 'inferable_masks': 0}, timed=False)
        self.assertEqual(B.adjacency_review_check(good, adj)[0], {})
        bad = R.new_json(root / 'rev_bad.json', {'adjacency_sha256': R.sha(adj), 'inferable_masks': 2}, timed=False)
        self.assertIn('adjacency_inferable_masks', B.adjacency_review_check(bad, adj)[0])
        stale = R.new_json(root / 'rev_stale.json', {'adjacency_sha256': '0' * 64, 'inferable_masks': 0}, timed=False)
        self.assertIn('adjacency_review_stale', B.adjacency_review_check(stale, adj)[0])
        # v5 manual-read rule: every input no reviewer read end to end and every changed input, read by readers that
        # are not the builder, changed inputs read in their final content, every reported item adjudicated
        reviewer = list(range(1, 181))
        m = {'inputs_sha256': 'h', 'reads': {'reader_A': list(range(181, 191)), 'reader_B': list(range(191, 201)) + [5]},
             'confirmed_focal_or_shown_names': 0, 'inferable_masks': 0, 'reviewer_read_sets': {'earlier': reviewer},
             'changed_inputs': [5], 'final_content_read': [5], 'must_read': [5],
             'reported_items': [{'rank': 5, 'adjudication': 'fixed: code masked'}]}
        self.assertEqual(FZ.manual_read_check(m, 'h'), [])
        # v6: the audit-listed inputs (must_read) are listed and read in their final content
        self.assertTrue(FZ.manual_read_check({k: v for k, v in m.items() if k != 'must_read'}, 'h'))
        self.assertTrue(FZ.manual_read_check({**m, 'must_read': [5, 7]}, 'h'))
        self.assertTrue(FZ.manual_read_check({**m, 'reads': {'reader_A': list(range(181, 201))}}, 'h'))  # changed 5
        self.assertTrue(FZ.manual_read_check({**m, 'reads': {'reader_A': list(range(182, 201)) + [5]}}, 'h'))  # 181
        self.assertTrue(FZ.manual_read_check({**m, 'reads': {'TD3E builder': list(range(181, 201)) + [5]}}, 'h'))
        self.assertTrue(FZ.manual_read_check({**m, 'final_content_read': []}, 'h'))
        self.assertTrue(FZ.manual_read_check({**m, 'inferable_masks': 1}, 'h'))
        self.assertTrue(FZ.manual_read_check({**m, 'reported_items': [{'rank': 5}]}, 'h'))
        self.assertTrue(FZ.manual_read_check({k: v for k, v in m.items() if k != 'changed_inputs'}, 'h'))
        self.assertTrue(FZ.manual_read_check({**m, 'inputs_sha256': 'other'}, 'h'))


class TestProviders(unittest.TestCase):
    def setUp(self):
        self.s = P.selectors(MENU)

    def ds(self, content, status=200, usage=None, model='deepseek-flash', extra=None, details=None):
        u = usage or {'prompt_tokens': 100, 'completion_tokens': 20}
        if details is not None:
            u = {**u, 'completion_tokens_details': details}
        body = {'model': model, 'choices': [{'message': {'content': content, **(extra or {})},
                                             'finish_reason': 'stop'}], 'usage': u}
        sel = self.s['deepseek_order']
        return sel.apply_model_check(status, sel.parse(status, json.dumps(body).encode()))

    def test_bodies_share_identical_input(self):
        text = 'FROZEN INPUT sample with [FOCAL]\nline two'
        ds = json.loads(self.s['deepseek_order'].body(text))
        ch = json.loads(self.s['jev_choice'].body(text))
        nl = json.loads(self.s['jev_noul'].body(text))
        self.assertTrue(ds['messages'][1]['content'].endswith('\n\n' + text))
        self.assertEqual(ch['state'], text)
        self.assertEqual(nl['state'], text)
        self.assertEqual(ch['model'], 'jev-1.14.0')
        self.assertIn('none', ch['questions']['risk_choice']['criteria'])
        self.assertEqual(ch['questions']['risk_choice']['type'], 'choice')
        self.assertEqual(sorted(nl['questions']), sorted('applies_' + n for n in S.NAMES))
        self.assertTrue(all(q['type'] == 'noul' for q in nl['questions'].values()))
        self.assertEqual(self.s['deepseek_order'].body(text), self.s['deepseek_order'].body(text))

    def test_deepseek_request_d3(self):
        ds = json.loads(self.s['deepseek_order'].body('x'))
        self.assertEqual(ds['model'], 'deepseek-flash')
        self.assertEqual(ds['thinking'], {'type': 'disabled'})
        self.assertEqual((ds['temperature'], ds['response_format'], ds['stream'], ds['max_tokens']),
                         (0, {'type': 'json_object'}, False, S.DEEPSEEK_MAX_TOKENS))
        bundle, _ = S.question_bundle(MENU)
        self.assertEqual((bundle['deepseek_model'], bundle['deepseek_thinking']), ('deepseek-flash', {'type': 'disabled'}))
        self.assertEqual(S.model_checks()['deepseek_returned_model_allowlist'], ['deepseek-flash'])
        self.assertEqual(RU.clearance_bindings()['deepseek_thinking'], {'type': 'disabled'})
        ok = json.dumps({'ordering': ['cash', 'size', 'leverage', 'inventory', 'cash_flow']})
        self.assertEqual(self.ds(ok, model='deepseek-chat')['model_violation'], 'deepseek_model_not_allowed')

    def test_deepseek_valid(self):
        r = self.ds(json.dumps({'ordering': ['cash', 'size', 'leverage', 'inventory', 'cash_flow']}))
        self.assertEqual(r['status'], 'valid')
        self.assertEqual(r['ordering'], ['cash', 'size', 'leverage', 'inventory', 'cash_flow'])
        self.assertIsNone(r['model_violation'])

    def test_deepseek_malformed_logged_not_backfilled(self):
        cases = {'not json': ['invalid_json'], json.dumps({'order': []}): ['missing_ordering'],
                 json.dumps({'ordering': ['cash', 'cash', 'size', 'leverage', 'inventory']}): ['duplicate_names',
                                                                                              'missing_names'],
                 json.dumps({'ordering': ['cash', 'Size', 'leverage', 'inventory', 'cash_flow']}): ['invalid_names',
                                                                                                   'missing_names'],
                 json.dumps({'ordering': ['cash', 'size']}): ['missing_names']}
        for content, issues in cases.items():
            r = self.ds(content)
            self.assertEqual(r['status'], 'invalid', content)
            self.assertIsNone(r['ordering'])
            self.assertEqual(r['issues'], issues, content)
            if r['ordering_filtered'] is not None:
                self.assertLess(len(r['ordering_filtered']), 5)
        self.assertEqual(self.s['deepseek_order'].parse(500, b'oops')['status'], 'http_error')
        self.assertEqual(self.s['deepseek_order'].parse(200, b'<html>')['status'], 'malformed_body')

    def test_deepseek_returned_model_checks_c5(self):
        ok = json.dumps({'ordering': ['cash', 'size', 'leverage', 'inventory', 'cash_flow']})
        self.assertEqual(self.ds(ok, model='deepseek-flash')['status'], 'valid')
        for kw, reason in [({'model': 'deepseek-v4-pro'}, 'deepseek_model_not_allowed'),
                           ({'model': None}, 'deepseek_model_not_allowed'),
                           ({'details': {'reasoning_tokens': 150}}, 'deepseek_reasoning_tokens'),
                           ({'extra': {'reasoning_content': 'Let me think'}}, 'deepseek_reasoning_content')]:
            r = self.ds(ok, **kw)
            self.assertEqual(r['status'], 'model_not_allowed', kw)
            self.assertEqual(r['model_violation'], reason)
            self.assertEqual(r['content_status'], 'valid')
            self.assertIsNone(r['ordering'])
        self.assertEqual(self.ds(ok, extra={'reasoning_content': None})['status'], 'valid')

    def test_reasoning_fields_m1(self):
        ok = json.dumps({'ordering': ['cash', 'size', 'leverage', 'inventory', 'cash_flow']})
        for kw, reason in [({'details': {'reasoning_tokens': '12'}}, 'deepseek_reasoning_tokens_not_numeric'),
                           ({'details': {'reasoning_tokens': True}}, 'deepseek_reasoning_tokens_not_numeric'),
                           ({'details': {'reasoning_tokens': [1]}}, 'deepseek_reasoning_tokens_not_numeric'),
                           ({'extra': {'reasoning_content': ''}}, 'deepseek_reasoning_content'),
                           ({'extra': {'reasoning_content': {'a': 1}}}, 'deepseek_reasoning_content'),
                           ({'extra': {'reasoning_content': 0}}, 'deepseek_reasoning_content')]:
            r = self.ds(ok, **kw)
            self.assertEqual((r['status'], r['model_violation']), ('model_not_allowed', reason), kw)
        for kw in ({'details': {'reasoning_tokens': 0}}, {'details': {'reasoning_tokens': None}}, {'details': {}}):
            self.assertEqual(self.ds(ok, **kw)['status'], 'valid', kw)

    def jev(self, sel, answers, status=200, model='jev-1.14.0'):
        body = {'model': model, 'answers': answers, 'usage': {'input_tokens': 10, 'output_tokens': 2}}
        s = self.s[sel]
        return s.apply_model_check(status, s.parse(status, json.dumps(body).encode()))

    def test_jev_choice(self):
        probs = {'size': 0.1, 'cash': 0.2, 'cash_flow': 0.1, 'leverage': 0.4, 'inventory': 0.1, 'none': 0.1}
        r = self.jev('jev_choice', {'risk_choice': {'choice': 'leverage', 'probabilities': probs, 'confidence': 0.4}})
        self.assertEqual((r['status'], r['choice']), ('valid', 'leverage'))
        self.assertEqual(r['probabilities'], probs)
        r = self.jev('jev_choice', {'risk_choice': {'choice': 'none', 'probabilities': probs, 'confidence': 0.1}})
        self.assertEqual((r['status'], r['choice']), ('valid', 'none'))
        self.assertIn('choice_not_argmax', r['notes'])
        for answers, issue in [({'risk_choice': {'choice': 'bananas', 'probabilities': probs}}, 'invalid_choice'),
                               ({'risk_choice': {'choice': 'cash'}}, 'missing_probabilities'),
                               ({'risk_choice': {'choice': 'cash', 'probabilities': {'cash': 1.0}}}, 'probability_keys'),
                               ({'risk_choice': {'choice': 'cash', 'probabilities': {**probs, 'cash': 0.9}}},
                                'probability_sum'),
                               ({}, 'missing_answer')]:
            r = self.jev('jev_choice', answers)
            self.assertEqual(r['status'], 'invalid')
            self.assertIn(issue, r['issues'])
            self.assertIsNone(r['choice'])
            self.assertIsNone(r['probabilities'])
        r = self.s['jev_choice'].parse(200, json.dumps({'answers': [1, 2]}).encode())
        self.assertEqual(r['status'], 'malformed_body')

    def test_jev_probability_sum_tolerance_c3(self):
        base = {'size': 0.1, 'cash': 0.2, 'cash_flow': 0.1, 'leverage': 0.4, 'inventory': 0.1}
        # v4 (N5): tolerance 0.03 = 6 options x 0.005 (two-decimal rounding), inclusive
        for none, valid in ((0.11, True), (0.09, True), (0.12, True), (0.13, True), (0.07, True), (0.14, False),
                            (0.06, False)):
            probs = {**base, 'none': none}
            r = self.jev('jev_choice', {'risk_choice': {'choice': 'leverage', 'probabilities': probs}})
            self.assertEqual(r['status'] == 'valid', valid, none)
            self.assertAlmostEqual(r['probability_sum_raw'], sum(probs.values()))
            if valid:
                self.assertIn('probability_sum_rounded', r['notes'])
                self.assertEqual(r['probabilities'], probs)          # stored as returned, never renormalized
            else:
                self.assertIn('probability_sum', r['issues'])
        exact = self.jev('jev_choice', {'risk_choice': {'choice': 'leverage', 'probabilities': {**base, 'none': 0.1}}})
        self.assertNotIn('probability_sum_rounded', exact['notes'])

    def test_jev_two_decimal_rounding_never_invalid_n5(self):
        """N5: any distribution over the 6 options, each value rounded to the two decimals the API uses, is valid."""
        self.assertGreaterEqual(S.PROBABILITY_SUM_TOLERANCE, S.CHOICE_OPTIONS * 0.005)
        rng = np.random.default_rng(73001)
        options = S.NAMES + [S.NONE_OPTION]
        worst = 0.0
        for _ in range(3000):
            w = rng.dirichlet(np.full(len(options), 0.3))
            probs = {o: round(float(x), 2) for o, x in zip(options, w)}
            worst = max(worst, abs(sum(probs.values()) - 1))
            r = self.jev('jev_choice', {'risk_choice': {'choice': max(probs, key=probs.get), 'probabilities': probs}})
            self.assertEqual(r['status'], 'valid', (probs, sum(probs.values())))
        self.assertLess(worst, 0.03)
        edge = {'size': 0.165, 'cash': 0.165, 'cash_flow': 0.165, 'leverage': 0.165, 'inventory': 0.165, 'none': 0.175}
        rounded = {k: round(v - 0.0049, 2) for k, v in edge.items()}          # every value rounded down
        r = self.jev('jev_choice', {'risk_choice': {'choice': 'none', 'probabilities': rounded}})
        self.assertEqual(r['status'], 'valid')
        self.assertAlmostEqual(r['probability_sum_raw'], sum(rounded.values()))

    def test_jev_returned_model_check_c5(self):
        ok = {f'applies_{n}': {'noul': 0.5} for n in S.NAMES}
        self.assertEqual(self.jev('jev_noul', ok)['status'], 'valid')
        for model in ('jev-1.15.0', 'jev-latest', None):
            r = self.jev('jev_noul', ok, model=model)
            self.assertEqual(r['status'], 'model_not_allowed', model)
            self.assertEqual(r['model_violation'], 'jev_model_not_allowed')
            self.assertIsNone(r['yes_scores'])

    def test_jev_noul(self):
        ok = {f'applies_{n}': {'noul': 0.1 * (i + 1)} for i, n in enumerate(S.NAMES)}
        r = self.jev('jev_noul', ok)
        self.assertEqual(r['status'], 'valid')
        self.assertAlmostEqual(r['yes_scores']['inventory'], 0.5)
        bad = dict(ok)
        del bad['applies_cash']
        bad['applies_size'] = {'noul': True}
        bad['applies_leverage'] = {'noul': 1.5}
        r = self.jev('jev_noul', bad)
        self.assertEqual(r['status'], 'invalid')
        self.assertIsNone(r['yes_scores'])
        self.assertEqual(r['missing_answers'], ['cash'])
        self.assertEqual(sorted(r['invalid_scores']), ['leverage', 'size'])
        self.assertNotIn('cash', r['yes_scores_partial'])

    def test_redirects_never_followed_r2(self):
        t = P.UrllibTransport(timeout=1)
        handlers = [h for h in t.opener.handlers if isinstance(h, P._NoRedirect)]
        self.assertEqual(len(handlers), 1)
        self.assertIsNone(handlers[0].redirect_request(None, None, 302, 'Found', {}, 'http://api.deepseek.com/x'))
        import urllib.request
        self.assertFalse(any(type(h) is urllib.request.HTTPRedirectHandler for h in t.opener.handlers))


class TestLedgerAndCaps(unittest.TestCase):
    def test_window_and_prices(self):
        self.assertEqual(L.window('2026-11-05T02:00:00Z'), 'peak')
        self.assertEqual(L.window('2026-11-05T05:00:00Z'), 'off_peak')
        self.assertEqual(L.window('2026-11-05T09:59:00Z'), 'peak')
        self.assertEqual(L.window('2026-11-07T02:00:00Z'), 'off_peak')
        u = {'prompt_tokens': 1000, 'completion_tokens': 100, 'prompt_cache_hit_tokens': 400}
        self.assertAlmostEqual(L.deepseek_usd(u, 'off_peak'), (600 * 0.15 + 400 * 0.003 + 100 * 0.60) / 1e6)
        self.assertAlmostEqual(L.deepseek_usd(u, 'peak'), 2 * L.deepseek_usd(u, 'off_peak'))
        self.assertIsNone(L.deepseek_usd({}, 'peak'))
        self.assertEqual(S.DEEPSEEK_RETURNED_MODEL_PRICES, {'deepseek-flash': S.PRICES})

    def test_jev_pricing_c4(self):
        self.assertEqual(S.JEV_PRICE_INPUT_PER_M, 0.042)
        self.assertAlmostEqual(L.jev_usd({'input_tokens': 1_000_000, 'output_tokens': 5000}), 0.042)
        self.assertIsNone(L.jev_usd({}))
        self.assertAlmostEqual(L.jev_upper_usd(b'x' * 10000), (10000 * 0.5 + 64) * 0.042 / 1e6)
        root = RUN_ROOT / ('jev_' + uuid.uuid4().hex[:6])
        led = L.Ledger(root / 'ledger.jsonl', R.Redactor())
        seq, upper = led.reserve('jev', 'jev_choice', {}, b'x' * 10000)
        self.assertAlmostEqual(upper, L.jev_upper_usd(b'x' * 10000))
        self.assertAlmostEqual(led.usd(), upper)                         # unsettled reserve counts at its bound
        row = led.settle(seq, 'jev', 'jev_choice', {}, OFFPEAK_CLOCK, OFFPEAK_CLOCK, 0.1, 200,
                         {'input_tokens': 2000, 'output_tokens': 3}, 'jev-1.14.0', 'valid', upper)
        self.assertAlmostEqual(row['usd_estimate'], 2000 * 0.042 / 1e6)
        self.assertEqual(row['usd_basis'], 'reported_usage')
        self.assertAlmostEqual(led.usd(), 2000 * 0.042 / 1e6)
        led.close()
        led2 = L.Ledger(root / 'ledger2.jsonl', R.Redactor(), prior_usd=S.MAX_USD - 1e-7)   # Jev under the USD cap
        with self.assertRaises(L.CapStop):
            led2.reserve('jev', 'jev_noul', {}, b'x' * 10000)
        self.assertEqual(led2.rows, [])
        led2.close()
        v1_row = {'kind': 'reserve', 'call_seq': 1, 'provider': 'jev', 'selector': 'jev_choice', 'unit': {},
                  'usd_upper': None, 'body_bytes': 1000}
        self.assertAlmostEqual(L.summarize([v1_row])[0], (1000 * 0.5 + 64) * 0.042 / 1e6)

    def test_precall_cap_refuses_without_row(self):
        root = RUN_ROOT / ('cap_' + uuid.uuid4().hex[:6])
        led = L.Ledger(root / 'ledger.jsonl', R.Redactor(), prior_usd=4.9999)
        with self.assertRaises(L.CapStop):
            led.reserve('deepseek', 'deepseek_order', {}, b'x' * 10000)
        self.assertEqual(led.rows, [])
        led2 = L.Ledger(root / 'ledger2.jsonl', R.Redactor(), prior_jev=600)
        with self.assertRaises(L.CapStop):
            led2.reserve('jev', 'jev_choice', {}, b'x')
        led.close()
        led2.close()

    def _execute(self, ledger_kwargs, responder=None, clock=PEAK_CLOCK):
        fx = fixture()
        root = RUN_ROOT / ('exec_' + uuid.uuid4().hex[:6])
        red = R.Redactor()
        recs = fx['records']
        led = L.Ledger(root / 'ledger.jsonl', red, **ledger_kwargs)
        w = R.JsonlWriter(root / 'responses.jsonl', red)
        tr = P.MockTransport(responder)
        summ = RU.execute(RU.plan(recs), {r['key']: r for r in recs}, P.selectors(MENU), tr, keyring(red), led, w, red,
                          'test', 'test', 'session_1', sleep=lambda s: None, clock=lambda: clock, progress_every=0)
        led.close()
        w.close()
        return summ, led, tr, R.read_jsonl(root / 'ledger.jsonl')

    def test_usd_cap_stops_calls(self):
        recs = sorted(fixture()['records'], key=lambda r: r['sample_rank'])
        upper = L.deepseek_upper_usd(P.selectors(MENU)['deepseek_order'].body(recs[0]['input_text']))

        def heavy(url, headers, body):   # reported usage equal to the pre-call bound (worst case)
            status, raw = P.mock_response(url, headers, body)
            data = json.loads(raw)
            if url == S.DEEPSEEK_URL:
                data['usage'] = {'prompt_tokens': len(body) * S.UPPER_TOKENS_PER_BYTE + S.UPPER_FRAMING_TOKENS,
                                 'completion_tokens': S.DEEPSEEK_MAX_TOKENS}
            return status, json.dumps(data).encode()
        summ, led, tr, rows = self._execute({'max_usd': 1.5 * upper}, heavy)
        self.assertTrue(summ['halt'].startswith('cap_usd'))
        reserves = [r for r in rows if r['kind'] == 'reserve']
        self.assertEqual(len(tr.calls), len(reserves))
        self.assertLessEqual(led.usd(), 1.5 * upper)
        self.assertEqual(sum(c['url'] == S.DEEPSEEK_URL for c in tr.calls), 1)
        self.assertTrue(all(r['usd_upper'] > 0 for r in reserves))         # Jev reserves carry a USD bound too
        running = 0.0
        settled = {r['call_seq']: r for r in rows if r['kind'] == 'settle'}
        for r in reserves:
            self.assertLessEqual(running + r['usd_upper'], 1.5 * upper)
            running += settled[r['call_seq']]['usd_estimate']

    def test_jev_count_cap(self):
        summ, led, tr, rows = self._execute({'max_jev': 3})
        self.assertTrue(summ['halt'].startswith('cap_jev_requests'))
        self.assertEqual(sum(c['url'] == S.JEV_URL for c in tr.calls), 3)

    def test_retry_then_success_and_halt_on_401(self):
        state = {'n': 0}

        def flaky(url, headers, body):
            state['n'] += 1
            return (503, b'busy') if state['n'] == 1 else P.mock_response(url, headers, body)
        summ, led, tr, rows = self._execute({}, flaky)
        self.assertIsNone(summ['halt'])
        self.assertEqual(len(tr.calls), len(RU.plan(fixture()['records'])) + 1)
        self.assertEqual(sum(r['kind'] == 'reserve' for r in rows), len(tr.calls))
        summ, led, tr, rows = self._execute({}, lambda u, h, b: (401, b'{"error":"unauthorized"}'))
        self.assertEqual(summ['halt'], 'http_401_deepseek')
        self.assertEqual(len(tr.calls), 1)

    def test_schema_mismatch_halts(self):
        def other_schema(url, headers, body):
            if url == S.JEV_URL:
                return 200, json.dumps({'results': {}, 'usage': {'input_tokens': 5, 'output_tokens': 1}}).encode()
            return P.mock_response(url, headers, body)
        summ, led, tr, rows = self._execute({}, other_schema)
        self.assertEqual(summ['halt'], 'schema_mismatch_jev_choice')
        self.assertEqual(sum(c['url'] == S.JEV_URL and b'risk_choice' in c['body'] for c in tr.calls), 3)

    def test_structural_invalid_halts_c3(self):
        def bad_sum(url, headers, body):
            status, raw = P.mock_response(url, headers, body)
            data = json.loads(raw)
            if url == S.JEV_URL and 'risk_choice' in data['answers']:
                data['answers']['risk_choice']['probabilities'] = {o: 0.5 for o in S.NAMES + ['none']}
            return status, json.dumps(data).encode()
        summ, led, tr, rows = self._execute({}, bad_sum)
        self.assertEqual(summ['halt'], 'structural_invalid_probability_sum_jev_choice')
        self.assertEqual(sum(c['url'] == S.JEV_URL and b'risk_choice' in c['body'] for c in tr.calls), 3)
        self.assertIsNone(RU.structural_halt([{'status': 'invalid', 'issues': ['invalid_names']}] * 3))
        self.assertIsNone(RU.structural_halt([{'status': 'invalid', 'issues': ['probability_sum']},
                                              {'status': 'valid', 'issues': []},
                                              {'status': 'invalid', 'issues': ['probability_sum']}]))

    def test_model_violation_halts_after_first_response_c5(self):
        def wrong_jev(url, headers, body):
            status, raw = P.mock_response(url, headers, body)
            data = json.loads(raw)
            if url == S.JEV_URL:
                data['model'] = 'jev-1.15.0'
            return status, json.dumps(data).encode()
        summ, led, tr, rows = self._execute({}, wrong_jev)
        self.assertEqual(summ['halt'], 'model_not_allowed_jev_choice:jev_model_not_allowed')
        self.assertEqual(sum(c['url'] == S.JEV_URL for c in tr.calls), 1)

        def thinking(url, headers, body):
            status, raw = P.mock_response(url, headers, body)
            data = json.loads(raw)
            if url == S.DEEPSEEK_URL:
                data['model'] = 'deepseek-v4-pro'
                data['usage']['completion_tokens_details'] = {'reasoning_tokens': 150}
            return status, json.dumps(data).encode()
        summ, led, tr, rows = self._execute({}, thinking)
        self.assertEqual(summ['halt'], 'model_not_allowed_deepseek_order:deepseek_model_not_allowed')
        self.assertEqual(len(tr.calls), 1)

    def test_transport_error_retried_and_counted(self):
        def down(url, headers, body):
            raise P.TransportError('URLError: gaierror')
        summ, led, tr, rows = self._execute({}, down)
        self.assertEqual(summ['attempts'], len(tr.calls))
        self.assertEqual(sum(r['kind'] == 'reserve' for r in rows), len(tr.calls))
        # both providers down: Jev reaches 3 consecutive failed units first (ds, jc, jn, ds, jc)
        self.assertEqual(summ['halt'], 'failure_unit_limit_jev')
        self.assertEqual(len(tr.calls), 3 * 5)

    def test_persistent_5xx_counts_as_failure_r1(self):
        summ, led, tr, rows = self._execute({}, lambda u, h, b: (503, b'busy'))
        self.assertEqual(summ['halt'], 'failure_unit_limit_jev')
        self.assertEqual(len(tr.calls), 3 * 5)
        summ, led, tr, rows = self._execute({}, lambda u, h, b: (429, b'slow down'))
        self.assertEqual(summ['halt'], 'failure_unit_limit_jev')

    def test_single_provider_outage_halts_d4(self):
        def ds_down(url, headers, body):
            return (503, b'busy') if url == S.DEEPSEEK_URL else P.mock_response(url, headers, body)
        summ, led, tr, rows = self._execute({}, ds_down)
        self.assertEqual(summ['halt'], 'failure_unit_limit_deepseek')
        self.assertEqual(sum(c['url'] == S.DEEPSEEK_URL for c in tr.calls), 3 * 3)   # 3 units x 3 attempts
        self.assertEqual(sum(c['url'] == S.JEV_URL for c in tr.calls), 4)            # Jev units in between succeed

        def jev_down(url, headers, body):
            return (503, b'busy') if url == S.JEV_URL else P.mock_response(url, headers, body)
        summ, led, tr, rows = self._execute({}, jev_down)
        self.assertEqual(summ['halt'], 'failure_unit_limit_jev')
        self.assertEqual(sum(c['url'] == S.JEV_URL for c in tr.calls), 3 * 3)
        self.assertEqual(sum(c['url'] == S.DEEPSEEK_URL for c in tr.calls), 2)

    def test_redirect_status_halts_r2(self):
        summ, led, tr, rows = self._execute({}, lambda u, h, b: (302, b''))
        self.assertEqual(summ['halt'], 'http_redirect_302_deepseek')
        self.assertEqual(len(tr.calls), 1)


class TestRunnerSessions(unittest.TestCase):
    def test_keys_never_written_or_printed(self):
        fx = fixture()

        def echo(url, headers, body):
            if url == S.JEV_URL and b'applies_' in body:
                raise P.TransportError('failure for ' + headers['Authorization'])
            status, raw = P.mock_response(url, headers, body)
            data = json.loads(raw)
            data['echo'] = headers['Authorization']
            return status, json.dumps(data).encode()
        red = R.Redactor()
        run_dir = RUN_ROOT / 'runs' / ('keys_' + uuid.uuid4().hex[:6])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            RU.session(run_dir, 'keys', 'test', fx['build_dir'], P.MockTransport(echo), keyring(red), red, (5.0, 600),
                       (0.0, 0), MENU, COUNTS, 1.0, sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        blob = all_text_under(run_dir) + out.getvalue()
        self.assertNotIn(SENT_DS, blob)
        self.assertNotIn(SENT_JEV, blob)
        self.assertIn('[REDACTED]', blob)
        self.assertGreater(red.count, 0)

    def test_load_key_process_scope_silent(self):
        import os
        out = io.StringIO()
        os.environ['TD3_TEST_ONLY_KEY'] = SENT_DS
        try:
            with contextlib.redirect_stdout(out):
                self.assertEqual(R.load_key('TD3_TEST_ONLY_KEY'), SENT_DS)
        finally:
            del os.environ['TD3_TEST_ONLY_KEY']
        self.assertEqual(out.getvalue(), '')

    def test_malformed_logged_and_reported(self):
        fx = fixture()

        def bad(url, headers, body):
            req = json.loads(body)
            if url == S.DEEPSEEK_URL:
                return 200, json.dumps({'model': 'deepseek-flash', 'choices': [{'message': {
                    'content': json.dumps({'ordering': ['cash', 'cash', 'bogus']})}}],
                    'usage': {'prompt_tokens': 50, 'completion_tokens': 9}}).encode()
            if 'risk_choice' in req['questions']:
                return 200, json.dumps({'model': 'jev-1.14.0', 'answers': {'risk_choice': {'choice': 'bananas'}},
                                        'usage': {'input_tokens': 5, 'output_tokens': 1}}).encode()
            return 200, json.dumps({'model': 'jev-1.14.0', 'answers': {'applies_size': {'noul': 0.3}},
                                    'usage': {'input_tokens': 5, 'output_tokens': 1}}).encode()
        red = R.Redactor()
        run_dir = RUN_ROOT / 'runs' / ('bad_' + uuid.uuid4().hex[:6])
        with contextlib.redirect_stdout(io.StringIO()):
            sess, summ, report = RU.session(run_dir, 'bad', 'test', fx['build_dir'], P.MockTransport(bad), keyring(red),
                                            red, (5.0, 600), (0.0, 0), MENU, COUNTS, 1.0, sleep=lambda s: None,
                                            clock=lambda: OFFPEAK_CLOCK)
        inv = report['invalid_counts']
        self.assertEqual(inv['deepseek_order_pass1']['status'], {'invalid': 2})
        self.assertEqual(inv['jev_choice_pass1']['issues'].get('invalid_choice'), 2)
        self.assertEqual(inv['jev_noul_pass1']['issues'].get('missing_answers'), 2)
        self.assertEqual(summ['halt'], 'structural_invalid_invalid_choice_jev_choice')   # third invalid_choice
        csv_text = (sess / 'selections.csv').read_text(encoding='utf-8')
        self.assertNotIn('cash>', csv_text)
        for r in R.read_jsonl(sess / 'responses.jsonl'):
            self.assertEqual(r['parsed']['status'], 'invalid')
            self.assertIsNotNone(r['response_raw'])

    def test_reask_identical_body_and_report(self):
        fx = fixture()
        red = R.Redactor()
        run_dir = RUN_ROOT / 'runs' / ('ok_' + uuid.uuid4().hex[:6])
        with contextlib.redirect_stdout(io.StringIO()):
            sess, summ, report = RU.session(run_dir, 'ok', 'test', fx['build_dir'], P.MockTransport(), keyring(red),
                                            red, (5.0, 600), (0.0, 0), MENU, COUNTS, 0.5, sleep=lambda s: None,
                                            clock=lambda: OFFPEAK_CLOCK)
        rows = R.read_jsonl(sess / 'responses.jsonl')
        by = {}
        for r in rows:
            by.setdefault((r['unit']['key'], r['unit']['selector']), set()).add(r['request']['body_sha256'])
        self.assertTrue(all(len(v) == 1 for v in by.values()))
        self.assertEqual(len(rows), 9)
        self.assertIsNone(summ['halt'])
        self.assertTrue(all(r['request']['model'] in ('deepseek-flash', 'jev-1.14.0') for r in rows))
        st = report['stability_reask']
        self.assertEqual(st['deepseek_order']['identical_ordering_rate'], 1.0)
        self.assertEqual(st['jev_choice']['total_variation_mean'], 0.0)
        proj = report['projection']['sets']['selection_plus_evaluation']
        ds1 = [r['parsed']['usage']['prompt_tokens'] for r in rows
               if r['unit']['selector'] == 'deepseek_order' and r['unit']['pass'] == 1]
        self.assertAlmostEqual(report['projection']['deepseek_mean_input_tokens'], float(np.mean(ds1)))
        self.assertAlmostEqual(proj['deepseek_usd_off_peak_no_cache'],
                               2250 * (np.mean(ds1) * 0.15 + S.ESTIMATE_OUTPUT_TOKENS * 0.60) / 1e6)
        self.assertAlmostEqual(proj['projected_calls_per_selector'], 0.5 * 4500)
        self.assertAlmostEqual(proj['jev_requests'], 4500)
        jin = [r['parsed']['usage']['input_tokens'] for r in rows
               if r['unit']['selector'] != 'deepseek_order' and r['unit']['pass'] == 1]
        self.assertAlmostEqual(proj['jev_usd'], 2250 * 2 * np.mean(jin) * 0.042 / 1e6)
        self.assertAlmostEqual(proj['total_usd_off_peak_no_cache'], proj['deepseek_usd_off_peak_no_cache'] + proj['jev_usd'])
        tot = report['tokens_and_cost']['totals']
        self.assertAlmostEqual(tot['jev_usd_estimate'], sum(r['usd_estimate'] for r in rows
                                                            if r['unit']['selector'] != 'deepseek_order'))
        self.assertGreater(tot['jev_usd_estimate'], 0)
        self.assertTrue(report['mock_providers'])


class TestClearanceAndRealGate(unittest.TestCase):
    def make(self, name, code_dir=S.CODE, **over):
        fx = fixture()
        root = RUN_ROOT / name
        inputs = fx['build_dir'] / 'INPUTS.jsonl'
        q = S.question_bundle(MENU)[1]
        code = {str(p): R.sha(p) for p in sorted(Path(code_dir).glob('*.py'))}
        freeze = {'schema': S.FREEZE_SCHEMA, 'code_sha256': code, 'inputs_path': str(inputs),
                  'inputs_sha256': R.sha(inputs), 'build_dir': str(fx['build_dir']), 'question_sha256': q,
                  'mask_tables_sha256': 'f' * 64, **RU.clearance_bindings()}
        fpath = R.new_json(root / 'FREEZE.json', freeze, timed=False)
        c = {'schema': S.CLEARANCE_SCHEMA, 'status': 'CLEARED', 'freeze_path': str(fpath),
             'freeze_sha256': R.sha(fpath), 'code_sha256': code, 'inputs_sha256': freeze['inputs_sha256'],
             'question_sha256': q, 'mask_tables_sha256': 'f' * 64, **RU.clearance_bindings()}
        c.update(over)
        cpath = R.new_json(root / 'TD3_CLEARANCE_v6.json', c, timed=False)
        return cpath, RUN_ROOT

    def test_clearance_variants(self):
        self.assertEqual(S.CLEARANCE.name, 'TD3_CLEARANCE_v6.json')
        cpath, out = self.make('clr_ok')
        self.assertEqual(RU.clearance_check(cpath, out, S.CODE, MENU)[:2], (True, []))
        self.assertFalse(RU.clearance_check(RUN_ROOT / 'absent.json', out, S.CODE, MENU)[0])
        for name, over in [('clr_req', {'request_only': True}), ('clr_status', {'status': 'REQUESTED'}),
                           ('clr_schema', {'schema': 'TD3_CLEARANCE_v5'}),
                           ('clr_tables', {'mask_tables_sha256': '0' * 64}),
                           ('clr_code', {'code_sha256': {'x.py': '0' * 64}}), ('clr_cap', {'max_usd': 50.0}),
                           ('clr_inputs', {'inputs_sha256': '0' * 64}), ('clr_freeze', {'freeze_sha256': '0' * 64}),
                           ('clr_model', {'jev_model': 'jev-latest'}), ('clr_jevprice', {'jev_price_input_per_m': 0}),
                           ('clr_allow', {'deepseek_returned_model_allowlist': ['deepseek-v4-pro']}),
                           ('clr_tol', {'probability_sum_tolerance': 0.02}),
                           ('clr_ds', {'deepseek_model': 'deepseek-chat'}),
                           ('clr_think', {'deepseek_thinking': {'type': 'enabled'}})]:
            cpath, out = self.make(name, **over)
            self.assertFalse(RU.clearance_check(cpath, out, S.CODE, MENU)[0], name)

    def test_extra_code_file_blocks(self):
        code_dir = RUN_ROOT / 'codecopy'
        for p in sorted(S.CODE.glob('*.py')):
            R.new_text(code_dir / p.name, p.read_text(encoding='utf-8'))
        cpath, out = self.make('clr_copy', code_dir=code_dir)
        self.assertTrue(RU.clearance_check(cpath, out, code_dir, MENU)[0])
        R.new_text(code_dir / 'extra.py', '# added\n')
        ok, reasons, _ = RU.clearance_check(cpath, out, code_dir, MENU)
        self.assertFalse(ok)
        self.assertTrue(any('code_sha256' in r for r in reasons))

    def test_real_run_refused_without_clearance(self):
        with self.assertRaises(PermissionError):
            RU.run_real(runs_root=RUN_ROOT / 'runs_none', clearance_path=RUN_ROOT / 'no_clearance.json',
                        out=RUN_ROOT, menu=MENU, counts=COUNTS, transport=P.MockTransport(),
                        keyring=keyring(R.Redactor()))
        with self.assertRaises(PermissionError):   # no injected transport and offline runtime
            RU.run_real(runs_root=RUN_ROOT / 'runs_none', clearance_path=RUN_ROOT / 'no_clearance.json')

    def test_real_run_with_mocks_prior_spend_and_resume(self):
        cpath, out = self.make('clr_real')
        runs = RUN_ROOT / 'runs_real'
        red = R.Redactor()
        tr = P.MockTransport()
        with contextlib.redirect_stdout(io.StringIO()):
            sess, summ, report = RU.run_real(runs_root=runs, clearance_path=cpath, out=out, menu=MENU, counts=COUNTS,
                                             transport=tr, keyring=keyring(red), redactor=red,
                                             sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        self.assertIsNone(summ['halt'])
        self.assertEqual(len(tr.calls), 9)
        run_id = sess.parent.name
        with self.assertRaises(PermissionError):   # a second fresh run is refused
            RU.run_real(runs_root=runs, clearance_path=cpath, out=out, menu=MENU, counts=COUNTS,
                        transport=P.MockTransport(), keyring=keyring(R.Redactor()))
        tr2 = P.MockTransport()
        with contextlib.redirect_stdout(io.StringIO()):
            sess2, summ2, _ = RU.run_real(resume=run_id, runs_root=runs, clearance_path=cpath, out=out, menu=MENU,
                                          counts=COUNTS, transport=tr2, keyring=keyring(R.Redactor()),
                                          sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        self.assertEqual(len(tr2.calls), 0)
        self.assertEqual(summ2['units_skipped_done'], 9)
        # a prior ledger at 4.999 USD blocks the next DeepSeek call before any transport call
        R.new_text(runs / 'real_prior' / 'session_1' / 'ledger.jsonl', json.dumps(
            {'kind': 'reserve', 'call_seq': 1, 'provider': 'deepseek', 'selector': 'deepseek_order', 'unit': {},
             'usd_upper': 4.999, 'body_bytes': 1}) + '\n')
        R.new_text(runs / 'real_prior' / 'session_1' / 'responses.jsonl', '')
        tr3 = P.MockTransport()
        with contextlib.redirect_stdout(io.StringIO()):
            sess3, summ3, _ = RU.run_real(resume='real_prior', runs_root=runs, clearance_path=cpath, out=out,
                                          menu=MENU, counts=COUNTS, transport=tr3, keyring=keyring(R.Redactor()),
                                          sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        self.assertTrue(summ3['halt'].startswith('cap_usd'))
        self.assertEqual(len(tr3.calls), 0)

    def test_run_lock_r3(self):
        cpath, out = self.make('clr_lock')
        runs = RUN_ROOT / 'runs_lock'
        held = R.RunLock(runs / S.RUN_LOCK.name).acquire()
        try:
            with self.assertRaises(PermissionError):
                R.RunLock(runs / S.RUN_LOCK.name).acquire()
            tr = P.MockTransport()
            with self.assertRaises(PermissionError):
                RU.run_real(runs_root=runs, clearance_path=cpath, out=out, menu=MENU, counts=COUNTS, transport=tr,
                            keyring=keyring(R.Redactor()), sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
            self.assertEqual(len(tr.calls), 0)
            self.assertFalse(any(runs.glob('real_*')))
        finally:
            held.release()
        with R.RunLock(runs / S.RUN_LOCK.name):
            pass                                    # released locks can be taken again

    def test_model_refusal_needs_new_clearance_m4(self):
        cpath, out = self.make('clr_m4')
        runs = RUN_ROOT / 'runs_m4'

        def wrong(url, headers, body):
            status, raw = P.mock_response(url, headers, body)
            data = json.loads(raw)
            if url == S.DEEPSEEK_URL:
                data['model'] = 'deepseek-v4-pro'
            return status, json.dumps(data).encode()
        with contextlib.redirect_stdout(io.StringIO()):
            sess, summ, _ = RU.run_real(runs_root=runs, clearance_path=cpath, out=out, menu=MENU, counts=COUNTS,
                                        transport=P.MockTransport(wrong), keyring=keyring(R.Redactor()),
                                        sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        self.assertTrue(summ['halt'].startswith('model_not_allowed_deepseek_order'))
        run_id = sess.parent.name
        self.assertEqual(RU.model_refusals(runs / run_id)[0]['clearance_sha256'], R.sha(cpath))
        tr = P.MockTransport()
        with self.assertRaises(PermissionError):           # same clearance: refused, no call
            RU.run_real(resume=run_id, runs_root=runs, clearance_path=cpath, out=out, menu=MENU, counts=COUNTS,
                        transport=tr, keyring=keyring(R.Redactor()), sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        self.assertEqual(len(tr.calls), 0)
        cpath2, _ = self.make('clr_m4_new', note='new clearance after review')    # a new clearance file
        tr2 = P.MockTransport()
        with contextlib.redirect_stdout(io.StringIO()):
            sess2, summ2, _ = RU.run_real(resume=run_id, runs_root=runs, clearance_path=cpath2, out=out, menu=MENU,
                                          counts=COUNTS, transport=tr2, keyring=keyring(R.Redactor()),
                                          sleep=lambda s: None, clock=lambda: OFFPEAK_CLOCK)
        self.assertIsNone(summ2['halt'])
        self.assertGreater(len(tr2.calls), 0)
        self.assertEqual(json.loads(tr2.calls[0]['body'])['model'], 'deepseek-flash')

    def test_model_not_allowed_units_not_done(self):
        root = RUN_ROOT / ('resume_' + uuid.uuid4().hex[:6])
        rows = [{'http_status': 200, 'unit': {'key': 'k', 'selector': 'jev_choice', 'pass': 1},
                 'parsed': {'status': 'model_not_allowed'}},
                {'http_status': 200, 'unit': {'key': 'k', 'selector': 'jev_noul', 'pass': 1},
                 'parsed': {'status': 'invalid'}}]
        R.new_text(root / 'session_1' / 'responses.jsonl', '\n'.join(json.dumps(r) for r in rows) + '\n')
        self.assertEqual(RU.done_units(root), {('k', 'jev_noul', 1)})
        self.assertEqual(RU.model_refusals(root), [{'session': 'session_1', 'refusals': 1, 'clearance_sha256': None}])


class TestReportMetrics(unittest.TestCase):
    def test_agreement_and_stability(self):
        def row(ds, choice, probs, yes):
            return {'deepseek_order': {'status': 'valid', 'value': ds, 'choice': None, 'issues': []},
                    'jev_choice': {'status': 'valid', 'value': probs, 'choice': choice, 'issues': []},
                    'jev_noul': {'status': 'valid', 'value': yes, 'choice': None, 'issues': []}}
        order = ['cash', 'size', 'leverage', 'inventory', 'cash_flow']
        probs = {'size': 0.3, 'cash': 0.4, 'cash_flow': 0.0, 'leverage': 0.2, 'inventory': 0.1, 'none': 0.0}
        yes = {'size': 0.8, 'cash': 0.9, 'cash_flow': 0.1, 'leverage': 0.7, 'inventory': 0.2}
        rows = {('k1', 1): row(order, 'cash', probs, yes), ('k2', 1): row(order[::-1], 'none', probs, yes),
                ('k1', 2): row(order, 'cash', probs, yes)}
        a = RP.agreement(rows, ['k1', 'k2'])
        self.assertEqual(a['deepseek_order__vs__jev_choice']['n_both_valid_non_none'], 1)
        self.assertEqual(a['deepseek_order__vs__jev_choice']['top1_agreement'], 1.0)
        self.assertEqual(a['deepseek_order__vs__jev_noul']['n_both_valid_non_none'], 2)
        self.assertEqual(a['deepseek_order__vs__jev_noul']['top1_agreement'], 0.5)
        sp = a['deepseek_order__vs__jev_noul']['rank_correlation']['spearman']
        self.assertAlmostEqual(sp['mean'], 0.0)
        st = RP.stability(rows, ['k1'])
        self.assertEqual(st['deepseek_order']['identical_ordering_rate'], 1.0)
        self.assertEqual(st['jev_noul']['max_abs_yes_change'], 0.0)


class TestOfflineAndScope(unittest.TestCase):
    def test_network_blocked(self):
        if R.mode() != 'offline':
            self.fail('offline runtime not installed; refusing to attempt any socket call')
        with self.assertRaises(PermissionError):
            socket.getaddrinfo('api.deepseek.com', 443)
        with self.assertRaises(P.TransportError):
            P.UrllibTransport(timeout=1).post(S.DEEPSEEK_URL, {'Authorization': 'Bearer x'}, b'{}')

    def test_host_allowlist(self):
        self.assertTrue(R.host_allowed('api.deepseek.com'))
        self.assertTrue(R.host_allowed(b'api.typesafe.ai'))
        self.assertFalse(R.host_allowed('example.com'))
        self.assertFalse(R.host_allowed('api.deepseek.com.evil.net'))

    def test_writes_outside_private_root_refused(self):
        if R.mode() != 'offline':
            self.fail('runtime not installed')
        with self.assertRaises(PermissionError):
            open(S.CODE / 'should_not_exist.txt', 'x')
        with self.assertRaises(PermissionError):
            open(RUN_ROOT / 'overwrite.txt', 'w')

    def test_no_label_or_return_sources_referenced(self):
        tokens = ['firm_week_manifest', 'crsp_20', 'predictions_v2', 'admitted_returns', 'model_frozen',
                  'label_end', 'EVALUATION_RESULT']
        for p in sorted(S.CODE.glob('*.py')):
            if p.name == 'td3_tests.py':
                continue
            text = p.read_text(encoding='utf-8')
            for t in tokens:
                self.assertNotIn(t, text, f'{p.name}: {t}')

    def test_review_flagged_strings_absent_m2(self):
        """m2/N6: the sample-derived strings flagged by the TD3BC and TD3C3 reviews (kept here only as SHA256 digests)
        occur in no v4 code file."""
        for p in sorted(S.CODE.glob('*.py')):
            text = p.read_text(encoding='utf-8')
            for digest, n, first in M_FLAGGED:
                for i in range(len(text) - n + 1):
                    if text[i] == first and hashlib.sha256(text[i:i + n].encode('utf-8')).hexdigest() == digest:
                        self.fail(f'{p.name} contains a review-flagged string')

    def test_utc_from_date(self):
        self.assertRegex(R.utc(), r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')

    def test_v6_paths_and_versions(self):
        self.assertEqual(S.CODE.name, 'v6')
        self.assertEqual(S.TD2_CODE.name, 'td2')
        self.assertTrue((S.TD2_CODE / 'engine.py').exists())
        self.assertEqual((S.FREEZE_NAME, S.CLEARANCE_REQUEST_NAME), ('FREEZE_TD3_v6.json', 'CLEARANCE_REQUEST_TD3_v6.json'))
        self.assertEqual(S.CLEARANCE.name, 'TD3_CLEARANCE_v6.json')
        self.assertEqual(S.CLEARANCE_SCHEMA, 'TD3_CLEARANCE_v6')
        self.assertEqual(S.INPUT_VERSION, 'TD3-input-v7')
        self.assertIn('FREEZE_TD3_v5.json', S.SUPERSEDED_FREEZES)
        self.assertTrue(str(S.REVIEW_FINDINGS_C5).startswith(str(S.OUT)))
        self.assertEqual(S.PROBABILITY_SUM_TOLERANCE, 0.03)
        self.assertEqual((S.DEEPSEEK_MODEL, S.DEEPSEEK_THINKING, S.JEV_MODEL), ('deepseek-flash', {'type': 'disabled'},
                                                                               'jev-1.14.0'))
        self.assertEqual((S.MAX_USD, S.MAX_JEV_REQUESTS), (5.00, 600))
        self.assertTrue(S.model_checks()['deepseek_allowlist_prices_equal_PRICES'])
        for n in ('FREEZE_TD3_v1.json', 'FREEZE_TD3_v2.json', 'FREEZE_TD3_v3.json', 'FREEZE_TD3_v4.json'):
            self.assertIn(n, S.SUPERSEDED_FREEZES)
        self.assertEqual(S.MIN_FREE_BYTES, 25 * 1024 ** 3)                  # v5 (TD3C4 N-e)
        self.assertTrue(str(S.REVIEW_FINDINGS_C4).startswith(str(S.OUT)))
        self.assertTrue(str(S.MASK_TABLES).startswith(str(S.OUT)))         # the table is private, never in the code
        self.assertFalse(str(S.MASK_TABLES).startswith(str(S.CODE)))


def qualify():
    if R.mode() != 'offline':
        raise RuntimeError('qualify requires the offline runtime (run through run_td3.py)')
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    classes = sorted({t.__class__.__name__ for t in _iter(suite)})
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    names = lambda items: [str(t) for t, _ in items]
    report = {'version': S.VERSION, 'tests': result.testsRun,
              'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'failed_tests': names(result.failures) + names(result.errors), 'classes': classes,
              'fixture_root': str(RUN_ROOT), 'mocked_providers_only': True, 'network': 'refused by audit hook',
              'code_sha256': {str(p): R.sha(p) for p in sorted(S.CODE.glob('*.py'))}, **R.peak()}
    path = R.new_json(S.OUT / f'{S.QUALIFICATION_PREFIX}{uuid.uuid4().hex}.json', report)
    print(stream.getvalue(), flush=True)
    print('TEST_COUNTS', {k: report[k] for k in ('tests', 'passed', 'failures', 'errors', 'skipped')}, flush=True)
    print('QUALIFICATION_REPORT', path, flush=True)
    return 0 if report['passed'] == report['tests'] else 1


def _iter(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _iter(t)
        else:
            yield t
