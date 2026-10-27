"""Q1 label-free, offline design study. Run with python -m ...prompt_gd_design.

Only host metadata, the explicitly authorized cached 120-case feature bank,
and published aggregate estimates are inputs. Never constructs an LLM client.
Registration is exclusive/immutable and precedes all RNG-based experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm

from . import prompt_gd_identify_v2 as v2

REPO = Path(__file__).resolve().parents[3]
OUT = Path('D:/OptResearch_data/agent_q1/run_v2_design')
INTRA = Path('C:/OptResearch/results/sc_risk/intraday')
WORK = Path('D:/OptResearch_data/sc_risk')
CACHE = Path('C:/OptResearch/results/sc_risk/cache')
PROCESSED = Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed')
HOP3 = Path('D:/OptResearch_data/agent_q1/hop3_run')
REPORT = REPO/'docs/Q1_DESIGN_REPORT.md'
COPY = Path('C:/OptResearch/results/sc_risk/agent_q1/Q1_DESIGN_REPORT.md')
SEED = 20261027
GRID = (50, 100, 200, 500, 1000, 2000, 3500, 5000)
RS = (.05, .1, .2)
TARGETS = (.10, .20)
FULL_NAMES = ('lambda2', 'lambda3', 'q_hub', 'q_RA', 'q_clust', 'w')
STARTS = v2.old.STARTS[:, [0, 1, 2, 3, 4, 6]]
READS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def read_json(path):
    path = Path(path)
    if any(t in path.name.lower() for t in ('labels_', 'predictions', 'evaluation')):
        raise ValueError('Outcome/label artifact forbidden')
    READS.append({'path': str(path), 'kind': 'json'})
    return json.loads(path.read_text(encoding='utf-8'))


def metadata(path, columns, filters=None):
    """Positive column allowlist: no return/content-label columns can be read."""
    import pyarrow.parquet as pq
    allowed = {'alt_id', 'ts', 'candidate_rics', 'ric', 'permno', 'date', 'c',
               'bin', 's', 'rel', 'sign', 'neg', 'month', 'asof',
               'placebo1', 'placebo2', 'placebo3', 'first_created'}
    if not columns or set(columns)-allowed or '2025' in Path(path).name:
        raise ValueError('Non-metadata or 2025 intraday read forbidden')
    READS.append({'path': str(path), 'columns': columns, 'filters': str(filters)})
    return pq.read_table(path, columns=columns, filters=filters,
                         use_threads=False).to_pandas()


def peak_memory():
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    x = PMC(); x.cb = ctypes.sizeof(x)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(x),x.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    n = max(x.PeakWorkingSetSize, x.PeakPagefileUsage)
    if n <= 0:raise RuntimeError('Memory measurement failed')
    if n >= 2*1024**3:
        raise MemoryError('2 GiB limit')
    return int(n)


def protocol():
    return {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'simulations_before_registration': 0, 'new_label_reads': 0,
        'seed': SEED, 'r': RS, 'n_grid': GRID, 'targets': TARGETS,
        'replications': 10, 'starts': STARTS.tolist(),
        'criterion': {'five_start_coordinate_range': .05, 'agree_replications': 8,
                      'every_parameter_MAE': list(TARGETS), 'D2_power': .8,
                      'alpha_one_sided': .05},
        'minimum': 'smallest passing REGISTERED GRID n, never an integer/global minimum; no monotonicity assumed',
        'optimizer': {'method': 'L-BFGS-B', 'bounds': [0, 1], 'maxiter': 4000,
                      'maxfun': 100000, 'maxls': 50, 'maxcor': 10,
                      'gtol': 1e-7, 'ftol': 1e-12,
                      'objective': 'MSE/label-free midpoint mean square; no intercept/penalty'},
        'daily_truth': 'V2 midpoint [lambda2,lambda3,q_clust,w]=[.5,.5,.5,.5]; other gates q=.5',
        'D1': 'hop2 ratio of historical direct/two-hop linked coefficients; hop3 exact-distance3/direct ratio; thresholds unestimated, q=.5 engineering assumptions; separate SE and +/-0.2 threshold stress',
        'D2': 'one-sided known-sigma Gaussian score test with all other parameters including w fixed; exact alternative difference linear in tested hop weight; alpha .05 separately, Holm not claimed',
        'D3': {'candidate_n': 12000, 'construction_seed': [SEED, 11],
               'host_templates': '120 cached ten-ticker cases plus eligible hop3 pools/relations; compact snapshots, no text or outcomes',
               'LLM_imputation': 'resample entire (relevance,direction,materiality) tuple conditional on exact hop from 120 cache; independent across rows; fields never sampled separately',
               'selection': 'D-optimal greedy determinant increments at midpoint; unit-box coordinates, tiny numerical ridge only for design; no repeats in finite candidate ordering',
               'noise': 'fixed sigma from UNSELECTED reference at each r, never recalibrate after selecting',
               'caveat': 'optimal selector sees imputed LLM fields; optimistic two-stage screening, host-only selector and no-improvement comparison also reported'},
        'expanded_random_and_host_only': 'simulate expanded-random as support-matched control; host-only ranking uses template-mean Jacobian second moments and cannot see individual imputed fields',
        'D4': 'lambda2=lambda3; both truths .5; merged derivative is SUM; random and D3',
        'D5': 'D3+w-only and D3+fixed thresholds+hop3,w; D1 thresholds+D2 hop tests; no adaptive choice',
        'simulation': {'rng': 'SeedSequence([SEED,40,design_index,r_index,n_index,rep]); D2 [SEED,41,hop,r_index,n_index,rep]',
                       'noise': 'daily std(unselected score)*sqrt(r^-2-1), independent Gaussian',
                       'baseline': '120-case empirical support sampled with replacement above 120; draw count is NOT distinct observed cases',
                       'expanded': 'without replacement from synthetic 12000 bank through 5000; host case IDs repeated, so projected fresh-case equivalent only',
                       'D1_stress_n': 5000, 'stress_seed': '[SEED,42,scenario,r_index,rep]'},
        'local_information': {'recovery': 'max diagonal inverse(E[J J^T])*sigma^2*(sqrt(2/pi)/MAE_target)^2; Gaussian expected MAE, NOT five-start certificate',
                              'power': '(z_.95+z_.8)^2*sigma^2/E[(f_alt-f_null)^2]',
                              'singular': 'infinite/not identified, never pseudo-inverse finite sample sizes',
                              'scope': 'local, correct model, iid/frozen-design approximation beyond 5000; selection depletion explicitly reported'},
        'D6': {'years': list(range(2015, 2025)),
               'metadata_only': True, 'timestamps': 'reuse O2 articles timestamp/candidate-RIC metadata and N1 mapping; no content columns',
               'match_primary': '[bin start-30 minutes, bin end), earliest article; strict pre-bin subset separately; <=bin-end equality excluded',
               'sensitivity': 'all earlier same-day articles also counted; news_bins/news_days crosscheck only (original cache does not cover full decade)',
               'partners': 'O3 complete matched partner metadata; exact shortest graph distance grouped 1/2/3; previous-month undirected graph, focal responder degree/clustering',
               'model': 'unsigned same-bin response, set FE; score(relevance*direction*materiality)*L*gates; balanced original up/down slopes; per-set linked-minus-three-placebo contrast',
               'information': 'event sums J^T Sigma^-1 J; Sigma=(4/3)*[(1-rho)I+rho*11T], rho=.25 primary, 0 and .75 sensitivity',
               'signal': 'M3 balanced premium .0396-ish read from JSON, scale to imputed per-row RMS; r here attenuates transferable article signal, not daily return correlation; r=1 archive envelope also reported',
               'gradient_check': 'map archived RA and log-degree partial slopes to observed feature SD, compare against prompt local gradients; no threshold is learned from a linear slope',
               'simulation': '10 replications x 5 starts at n=50,500,5000, cluster-correlated Gaussian row noise; matched-set contrast is exact FWL for constant article-score controls',
               'model_windows': {'gpt-oss-20b': '2024-07-01', 'Qwen3.8-27B': None,
                                 'frozen Claude Sonnet 4.5': '2025-08-01',
                                 'frozen Claude Opus 4.6': '2025-09-01'},
               'lookahead': 'user-specified cutoffs are assumptions, not verified online; pre-cutoff text requires no-text recall probe; clean window necessary not sufficient for contamination freedom',
               'calls': 'one per DISTINCT selected article with a bundle of partner requests, not one per row; full-window exact counts and smallest deterministic selected-prefix counts'},
        'limits': {'RAM_bytes': 2*1024**3, 'network_calls': 0, 'LLM_calls': 0,
                   'evaluation_labels': 'forbidden', 'intraday_2025': 'forbidden'},
        'protected_hashes': {n: sha(Path(__file__).with_name(n)) for n in
                             ('prompt_gd.py', 'hop_ablation.py', 'q1_run_v2.py', 'prompt_gd_identify_v2.py')},
        'input_hashes': {str(p): sha(p) for p in (v2.old.DATA/'cases.json', v2.old.CACHE)},
    }


def register(out=OUT):
    p = Path(out)/'PROTOCOL_DESIGN.json'
    if p.exists():
        return read_json(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    value = protocol()
    save(p, value, exclusive=True)
    return value


def check_registration(out=OUT):
    p = Path(out)/'PROTOCOL_DESIGN.json'
    if not p.exists():
        raise RuntimeError('Register before any simulation')
    reg = read_json(p)
    for n, digest in reg['protected_hashes'].items():
        if sha(Path(__file__).with_name(n)) != digest:
            raise RuntimeError('Protected source changed')
    for n, digest in reg['input_hashes'].items():
        if sha(n) != digest:
            raise RuntimeError('Cached evidence changed')
    return reg


def estimate(entry, b='estimate', t='t_week'):
    value = entry[b]
    tv = entry.get(t)
    return {'estimate': value, 'SE': abs(value/tv) if tv else None, 't': tv}


def calibration(out):
    m3 = read_json(INTRA/'m3_horserace/M3_RESULTS.json')
    grpath = INTRA/'GRAPHTOOLS_2015+2016+2017+2018+2019+2020+2021_to_2022_2023_2024.json'
    gr = read_json(grpath)['targets']['sig_b0']
    train = '2015+2016+2017+2018+2019+2020+2021'
    linked = gr['labels'][train]
    direct = [estimate(linked['L_'+s], 'b', 't') for s in ('supplier', 'customer')]
    two = [estimate(linked['L_'+s], 'b', 't') for s in ('cocust', 'cosup', 'cust2', 'sup2')]
    b1 = np.mean([x['estimate'] for x in direct]); se1 = np.mean([x['SE'] for x in direct])
    b2 = np.mean([x['estimate'] for x in two]); se2 = np.mean([x['SE'] for x in two])
    d3 = estimate(gr['single']['d3'][train], 'b', 't')
    ratio = lambda b, se: {'estimate': float(np.clip(b/b1, 0, 1)),
                          'SE': float(se/abs(b1)+abs(b)*se1/b1**2),
                          'unclipped': float(b/b1)}
    # L1 delta SE bound permits unknown covariance; it is NOT an exact SE.
    hop2, hop3 = ratio(b2, se2), ratio(d3['estimate'], d3['SE'])
    terms = m3['H2']['2015-2024']['models']['optimizer']['slope']
    weekly = read_json(Path('C:/OptResearch/results/sc_risk/step4b/COCUST_RA.json'))
    o5 = read_json(INTRA/'o5_exposure_tail/O5_RESULTS.json')
    g2 = read_json(INTRA/'g2_ra/G2_RESULTS.json')
    # Reports are read as evidence too, without copying their large bodies.
    reports = [INTRA/'m3_horserace/M3_REPORT.md', INTRA/'g2_ra/G2_REPORT.md',
               INTRA/'g3_news_links/G3_REPORT.md', INTRA/'g1_clocks/G1_REPORT.md',
               INTRA/'g4_multiplex/G4_REPORT.md', INTRA/'g5_shared/G5_REPORT.md',
               INTRA/'g6_activation/G6_REPORT.md', INTRA/'o5_exposure_tail/O5_REPORT.md',
               INTRA/'news_role/NEWSROLE_REPORT.md', WORK/'intraday/text_jumps/TJ_REPORT.md',
               INTRA/'o3_multistep/O3_REPORT.md']
    evidence = []
    for p in reports:
        content = p.read_text(encoding='utf-8')
        evidence.append({'path': str(p), 'sha256': sha(p), 'characters_read': len(content)})
    result = {'sources': evidence, 'graph_tools_source': str(grpath),
              'direct_linked_premia': direct, 'two_hop_linked_premia': two,
              'direct_mean': float(b1), 'direct_SE_bound': float(se1),
              'two_mean': float(b2), 'two_SE_bound': float(se2),
              'hop3_exact_distance_indicator': d3, 'lambda2': hop2, 'lambda3': hop3,
              'ratio_uncertainty': 'L1 delta bound, unknown covariance; historical different-support regressions, structural transport uncertainty omitted',
              'balanced_linked_premium': estimate(m3['sign_balance']['2015-2024']['slope']),
              'N_events_archive': m3['sanity']['N_events'],
              'N_sets_archive': m3['sign_balance']['2015-2024']['N'],
              'M3_RA': estimate(terms['ra']), 'M3_responder_logdegree': estimate(terms['ldeg_r']),
              'M3_shared_supplier_x_resp_hub': estimate(terms['ss*resp_hub']),
              'graph_tools_single_RA': estimate(gr['single']['ra'][train], 'b', 't'),
              'graph_tools_single_logdegree': estimate(gr['single']['ldeg_r'][train], 'b', 't'),
              'walk3_NOT_hop3': estimate(gr['single']['p3'][train], 'b', 't'),
              'clustering': {'estimate': None, 'SE': None, 'threshold_identified': False,
                             'reason': 'No prompt-style logistic local-clustering threshold estimate in inspected archives. sc is shared CUSTOMERS, not clustering.'},
              'threshold_mapping': {k: {'q': .5, 'SE': None, 'status': 'engineering midpoint, not calibrated estimate'}
                                    for k in ('hub', 'RA', 'clustering', 'materiality')},
              'weekly_2021_22': {k: v.get('2021-22') for k, v in weekly['summary'].items()},
              'weekly_SE': None, 'weekly_uncertainty': '10 graph-rewire rank p; not a sampling SE; use only 2021-22 summary, no 2025 raw data',
              'O5_model_flags_weights': dict(zip(('dependence','few','physical','old','ra_top','shared2','corroborated'), o5['model']['weights'])),
              'O5_RA_cutoffs': o5['model']['ra_cuts'], 'O5_weight_SE': None,
              'O5_reason': 'Ridge predictor weights/top5 cutoffs are not logistic gate thresholds or unpenalized SEs',
              'G2_gamma': g2.get('executive_summary', {}),
              'strict_full_D1_identified': False,
              'calibrated_truth': [hop2['estimate'], hop3['estimate'], .5, .5, .5, .5]}
    save(Path(out)/'calibration.json', result)
    return result


class Graph:
    """Small host-only graph; no returns, model fields or date inference."""
    def __init__(self, edges, universe=None):
        self.adj = defaultdict(set); self.inc = Counter(); self.out = Counter()
        for a, b in set(tuple(e) for e in edges):
            if a != b:
                self.adj[a].add(b); self.adj[b].add(a)
                self.out[a] += 1; self.inc[b] += 1
        self.universe = list(universe or self.adj)
        self.degrees = np.sort([self.inc[n]+self.out[n] for n in self.universe])
        self._node = {}; self._ra = {}

    def node(self, n):
        if n not in self._node:
            adj = self.adj[n]; d = len(adj)
            cl = sum(len(adj & self.adj[k]) for k in adj)/(d*(d-1)) if d > 1 else 0.
            pct = 100*np.searchsorted(self.degrees, self.inc[n]+self.out[n], side='right')/len(self.degrees)
            self._node[n] = (float(pct), float(cl))
        return self._node[n]

    def distribution(self, target):
        if target not in self._ra:
            if len(self._ra) >= 128:
                self._ra.pop(next(iter(self._ra)))
            sums = defaultdict(float)
            for k in self.adj[target]:
                inv = 1/len(self.adj[k])
                for n in self.adj[k]:
                    if n != target:
                        sums[n] += inv
            vals = np.sort([sums.get(n, 0.) for n in self.universe if n != target])
            self._ra[target] = (sums, vals)
        return self._ra[target]

    def fields(self, target, other):
        degree, clust = self.node(target)
        sums, vals = self.distribution(target)
        ra = sums.get(other, 0.)
        pct = 100*np.searchsorted(vals, ra, side='right')/len(vals) if len(vals) else 0.
        return {'degree_percentile': degree, 'clustering': clust,
                'ra': float(ra), 'ra_percentile': float(pct),
                'log_degree':math.log1p(self.inc[target]+self.out[target])}


def snapshot_templates(out, cases):
    """Copy only compact host fields from files being written by another job."""
    dest = Path(out)/'host_templates.json'
    if dest.exists():
        return read_json(dest)['templates']
    templates = [{'id': c['case_id'], 'pool': 'ten_ticker',
                  'rows': [{k: e[k] for k in ('source_id', 'hop', 'graph')}
                           for e in c['candidates']]} for c in cases]
    graphpath = HOP3/'graph_snapshot.json'
    blob = graphpath.read_bytes(); digest = hashlib.sha256(blob).hexdigest()
    data = json.loads(blob); del blob
    univ = next(iter(data['monthly_universe'].values()))
    graph = Graph([(e['source'], e['target']) for e in data['edges']], univ)
    targets = data['targets']; del data; gc.collect()
    manifest = [{'path': str(graphpath), 'sha256_of_copy': digest}]
    relations = {}
    # Restrict reads to exactly the six declared pool paths, never predictions.
    for date in ('2025-10-01','2025-10-08','2025-10-15','2025-10-22','2025-10-30','2025-11-06'):
        p = HOP3/'pools'/f'{date}.json'
        blob = p.read_bytes(); digest = hashlib.sha256(blob).hexdigest()
        day = json.loads(blob); del blob
        manifest.append({'path': str(p), 'sha256_of_copy': digest,
                         'bytes': p.stat().st_size})
        for cid, pool in day['pools'].items():
            ticker = pool['case']['ticker']
            if ticker not in relations:
                rp = HOP3/'relations'/f'{ticker}.json'
                relblob = rp.read_bytes()
                relation = json.loads(relblob)['relations']
                manifest.append({'path': str(rp), 'sha256_of_copy': hashlib.sha256(relblob).hexdigest()})
                by_perm = defaultdict(list)
                for rel in relation:
                    for pm in rel['permnos']:
                        by_perm[int(pm)].append((rel['hops'], rel['entity']))
                relations[ticker] = by_perm
            target = targets[ticker][0]
            rows = [{'source_id': sid, 'hop': 0, 'graph': {}} for sid in pool['own_ids'][:2]]
            for sid in pool['network_ids']:
                doc = day['documents'][sid]
                options = [r for pm in doc['ids'] for r in relations[ticker].get(int(pm), [])]
                if options:
                    hop, other = min(options)
                    rows.append({'source_id': sid, 'hop': int(hop), 'graph': graph.fields(target, other)})
            if rows:
                templates.append({'id': cid, 'pool': 'hop3', 'rows': rows})
        del day; gc.collect(); peak_memory()
    save(dest, {'templates': templates, 'manifest': manifest,
                'note': 'No article text, return outcome, prediction or other job output copied; source files untouched'})
    return templates


class Bank:
    """V2 graph gates, arrays (case,row); the four graph coordinates plus w.

    Can also retain hub/RA to diagnose D6's six-parameter alternative.
    """
    def __init__(self, original, maps, fixed=None, transform=None, group=None, rho=0.):
        self.b = original; self.maps = maps
        self.fixed = np.full(6, .5) if fixed is None else np.asarray(fixed, float)
        self.transform = np.eye(6)[:, [0, 1, 4, 5]] if transform is None else np.asarray(transform, float)
        self.p = self.transform.shape[1]
        self.names = []
        for col in self.transform.T:
            self.names.append('+'.join(FULL_NAMES[i] for i in np.flatnonzero(col)))
        self.n = len(original.base); self.group = group; self.rho = rho
        self._pre = None
        # Fixed hub/RA/materiality can be cached exactly.
        if not np.any(self.transform[2:4]):
            qh, qr = self.fixed[2:4]
            h = v2.sigmoid((original.degree-maps[0].at(qh)[0])/maps[0].temperature)
            ra = v2.sigmoid((original.ra_pct-maps[1].at(qr)[0])/maps[1].temperature)*original.ra_positive
            m = v2.sigmoid((original.mat-maps[3].at(.5)[0])/maps[3].temperature)
            self._pre = original.base*np.where(original.neighbour, m*(1-.5*h)*(1+.25*ra), 1.)

    def full(self, x):
        return self.fixed + self.transform@(np.asarray(x)-.5)

    def score(self, x):
        f = self.full(x); b = self.b
        th, dth = self.maps[2].at(f[4])
        c = v2.sigmoid((b.clust-th)/self.maps[2].temperature)
        C = np.where(b.neighbour, 1-.5*c, 1.)
        L = np.where(b.h2, f[0], np.where(b.h3, f[1], 1.))
        if self._pre is None:
            thh, dh = self.maps[0].at(f[2]); thr, dr = self.maps[1].at(f[3])
            h = v2.sigmoid((b.degree-thh)/self.maps[0].temperature)
            r = v2.sigmoid((b.ra_pct-thr)/self.maps[1].temperature)*b.ra_positive
            H = np.where(b.neighbour, 1-.5*h, 1.)
            R = np.where(b.neighbour, 1+.25*r, 1.)
            m = np.where(b.neighbour, v2.sigmoid((b.mat-self.maps[3].at(.5)[0])/self.maps[3].temperature), 1.)
            pre = b.base*m*H*R
            jh = (b.base*L*m*R*C*b.neighbour*.5*h*(1-h)*dh/self.maps[0].temperature).sum(1)*f[5]
            jr = (b.base*L*m*H*C*b.neighbour*(-.25*r*(1-r))*dr/self.maps[1].temperature).sum(1)*f[5]
        else:
            pre = self._pre; jh = jr = np.zeros(self.n)
        raw = pre*C
        s = (raw*L).sum(1)
        j = np.column_stack(((raw*b.h2).sum(1)*f[5], (raw*b.h3).sum(1)*f[5], jh, jr,
            (pre*L*b.neighbour*.5*c*(1-c)*dth/self.maps[2].temperature).sum(1)*f[5], s))
        return s*f[5], j@self.transform

    def subset(self, indices):
        obj = object.__new__(type(self.b))
        for name, val in vars(self.b).items():
            setattr(obj, name, val[indices] if isinstance(val, np.ndarray) and val.shape[0] == self.n else val)
        return Bank(obj, self.maps, self.fixed, self.transform)


def bank_from_rows(rows, maps, **kw):
    return Bank(v2.old.Evidence(rows), maps, **kw)


def transform(indices, merged=False):
    t = np.eye(6)[:, indices]
    if merged:
        t[1, 0] = 1
    return t


def candidates(templates, features, maps, out):
    path = Path(out)/'candidate_arrays.npz'
    if path.exists():
        z = np.load(path)
        obj = object.__new__(v2.old.Evidence)
        for k in z.files:
            if k != 'template': setattr(obj, k, z[k])
        return Bank(obj, maps), z['template']
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 11]))
    empirical = {h: [] for h in range(4)}
    for rows in features.values():
        for e in rows[:8]:
            empirical[e['hop']].append((e['relevance'], e['direction'], e['materiality']))
    allrows, ti = [], []
    for i in range(12000):
        idx = i % len(templates); template = templates[idx]
        own = [r for r in template['rows'] if r['hop'] == 0]
        neighbour = [r for r in template['rows'] if r['hop'] > 0]
        selected = own[:2]
        if neighbour:
            selected += [neighbour[j] for j in rng.choice(len(neighbour), min(8-len(selected), len(neighbour)), replace=False)]
        rows = []
        for row in selected:
            joint = empirical[row['hop']][rng.integers(len(empirical[row['hop']]))]
            rows.append({**row, **dict(zip(('relevance','direction','materiality'), joint))})
        allrows.append(rows); ti.append(idx)
    bank = bank_from_rows(allrows, maps)
    np.savez_compressed(path, **vars(bank.b), template=np.array(ti))
    save(Path(out)/'imputation.json', {'empirical_joint_by_hop': empirical,
         'candidate_count': 12000, 'host_templates': len(templates),
         'host_ids': len({t['id'] for t in templates}),
         'distinct_joint_feature_banks': len({tuple((r['source_id'],r['relevance'],r['direction'],r['materiality']) for r in rows) for rows in allrows}),
         'no_new_observed_cases': True, 'conditional_on_hop': True,
         'row_dependence_ignored': True, 'same_article_across_targets_may_differ': True})
    return bank, np.array(ti)


def d_order(j, maximum=5000):
    """Greedy D-optimal without replacement; prior J only, no outcomes."""
    j = np.asarray(j); gram = j.T@j/len(j)
    ridge = max(np.trace(gram)*1e-9, 1e-20)
    inv = np.eye(j.shape[1])/ridge
    used = np.zeros(len(j), bool); order = []
    for _ in range(min(maximum, len(j))):
        leverage = np.einsum('ij,jk,ik->i', j, inv, j)
        leverage[used] = -np.inf
        k = int(np.argmax(leverage)); u = inv@j[k]
        inv -= np.outer(u, u)/(1+j[k]@u)
        inv = (inv+inv.T)/2
        used[k] = True; order.append(k)
        if len(order) % 100 == 0:
            inv = np.linalg.inv(j[order].T@j[order]+ridge*np.eye(j.shape[1]))
    return np.array(order)


def host_order(j, template_index, maximum=5000):
    """D-optimal expected information per host template; no individual score."""
    ids=np.unique(template_index);p=j.shape[1]
    blocks=np.array([j[template_index==k].T@j[template_index==k]/np.sum(template_index==k) for k in ids])
    queues=[np.flatnonzero(template_index==k) for k in ids]
    used=np.zeros(len(ids),int);G=np.eye(p)*max(np.trace(j.T@j/len(j))*1e-9,1e-20)
    order=[]
    for _ in range(min(maximum,len(j))):
        _,values=np.linalg.slogdet(G[None]+blocks)
        values[used>=np.array([len(q) for q in queues])]=-np.inf
        k=int(np.argmax(values));order.append(int(queues[k][used[k]]));used[k]+=1;G+=blocks[k]
    return np.array(order)


def information(j, sigma, target, groups=None, rho=0.):
    j = np.asarray(j, float)
    if groups is None:
        gram = j.T@j/len(j)
    else:
        gram = np.zeros((j.shape[1], j.shape[1]))
        for g in np.unique(groups):
            rows = j[groups == g]; m = len(rows); su = rows.sum(0)
            gram += (rows.T@rows-rho*np.outer(su, su)/(1-rho+rho*m))/(1-rho)
        gram /= len(np.unique(groups))
    eig = np.linalg.eigvalsh(gram)
    if eig[0] <= max(1e-24, eig[-1]*1e-12):
        return {'n': None, 'eigenvalues': eig.tolist(), 'SE_at_n1': None, 'condition': None}
    se = sigma*np.sqrt(np.diag(np.linalg.inv(gram)))
    return {'n': int(math.ceil(max(se)**2*2/math.pi/target**2)),
            'eigenvalues': eig.tolist(), 'SE_at_n1': se.tolist(),
            'condition': float(eig[-1]/eig[0])}


def fit_five(bank, y, groups=None, rho=0., starts=None, row_weights=None):
    starts = STARTS@bank.transform if starts is None else starts
    starts = np.clip(starts, .1, .9)  # merged start is mean, not a sum
    if any(np.count_nonzero(c)>1 for c in bank.transform.T):
        starts = (STARTS@bank.transform)/np.sum(bank.transform, axis=0)
    scale = max(float(np.mean(bank.score(np.full(bank.p, .5))[0]**2)), 1e-20)
    if groups is not None:
        _, gid = np.unique(groups, return_inverse=True)
        counts = np.bincount(gid)
    def objective(x):
        s, j = bank.score(x); residual = s-y
        if groups is not None and rho:
            sums = np.bincount(gid, weights=residual)
            wr = (residual-rho*sums[gid]/(1-rho+rho*counts[gid]))/(1-rho)
        else:
            wr = residual
        if row_weights is not None:
            wr = wr*row_weights
        return float(residual@wr/len(y)/scale), 2*j.T@wr/len(y)/scale
    fits = []
    for initial in starts:
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(0., 1.)]*bank.p, options={'maxiter':4000,'maxfun':100000,
                'maxls':50,'maxcor':10,'gtol':1e-7,'ftol':1e-12})
        loss, grad = objective(result.x)
        fits.append({'theta': result.x.tolist(), 'loss': loss, 'iterations': int(result.nit),
                     'success': bool(result.success), 'message': str(result.message),
                     'projected_gradient': float(np.max(abs(result.x-np.clip(result.x-grad,0,1))))})
    best = min(fits, key=lambda f:f['loss'])
    agree = bool(np.max(np.ptp([f['theta'] for f in fits], axis=0)) <= .05)
    return best, agree, fits


def simulation(bank, r, n, seedparts, sigma, order=None, truth=None):
    truths = np.full(bank.p, .5) if truth is None else np.array(truth)
    planted = bank.score(truths)[0]
    errors, agrees, runs, supports = [], [], [], []
    for rep in range(10):
        rng = np.random.default_rng(np.random.SeedSequence([SEED,40,*seedparts,rep]))
        idx = order[:n] if order is not None else rng.choice(bank.n,n,replace=n>bank.n)
        sampled = bank.subset(idx)
        y = planted[idx]+rng.normal(0,sigma,n)
        best, agree, fits = fit_five(sampled,y)
        errors.append(abs(np.array(best['theta'])-truths)); agrees.append(agree)
        runs.append(fits); supports.append(len(np.unique(idx)))
    mae = np.mean(errors,axis=0)
    return {'r':r,'n':n,'names':bank.names,'agreement_share':float(np.mean(agrees)),
            'MAE':mae.tolist(),'passes':{str(t):bool(np.mean(agrees)>=.8 and np.all(mae<=t)) for t in TARGETS},
            'unique_bank_rows_mean':float(np.mean(supports)), 'fits':runs}


def decision_default(bank):
    x=np.full(bank.p,.5)
    for name,value in zip(('lambda2','lambda3'),v2.pg.DEFAULT[:2]):
        x[bank.names.index(name)]=value
    return x


def power_curve(bank, r, sigma, hop, seedtag, order=None, nuisance_w=False):
    x = decision_default(bank); alt = bank.score(x)[0]
    col = bank.names.index('lambda'+str(hop)); x[col]=0
    null=bank.score(x)[0]
    delta = alt-null
    pop = delta if order is None else delta[order]
    nullpop=null if order is None else null[order]
    if nuisance_w:pop=pop-nullpop*(pop@nullpop)/max(nullpop@nullpop,1e-30)
    nlocal = int(math.ceil((norm.ppf(.95)+norm.ppf(.8))**2*sigma**2/np.mean(pop*pop))) if np.any(pop) else None
    cells=[]
    for ni,n in enumerate(GRID):
        reject=[]; size=[];conditional=[]
        for rep in range(10):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,41,hop,seedtag,ni,rep]))
            idx=order[:n] if order is not None else rng.choice(bank.n,n,replace=n>bank.n)
            contrast=delta[idx]
            if nuisance_w:
                a=null[idx];contrast=contrast-a*(a@contrast)/max(a@a,1e-30)
            noncentral=np.linalg.norm(contrast)/sigma
            z=rng.normal();reject.append(bool(z+noncentral>norm.ppf(.95)))
            size.append(bool(z>norm.ppf(.95)))
            conditional.append(float(norm.cdf(noncentral-norm.ppf(.95))))
        exact=float(np.mean(conditional))
        cells.append({'n':n,'power_10':float(np.mean(reject)),'analytic_power':exact,'null_size_10':float(np.mean(size)),
                      'pass':bool(np.mean(reject)>=.8 and exact>=.8)})
    first=next((c['n'] for c in cells if c['pass']),None)
    return {'hop':hop,'r':r,'cells':cells,'minimum_grid_n':first,'local_n':nlocal,
            'nuisance_w':nuisance_w,
            'alternative_hop_weights':v2.pg.DEFAULT[:2].tolist(),
            'method':'one-sided exact Gaussian score test; unknown-w contrast projected orthogonal to null scale regressor; no boundary Wald approximation'}


def decision_experiment(base,expanded,out):
    sd0=float(np.std(base.score(decision_default(base))[0]));sdE=float(np.std(expanded.score(decision_default(expanded))[0]))
    powers=[]
    for ri,r in enumerate(RS):
        for hop in (3,2):
            x=decision_default(expanded);alt=expanded.score(x)[0];x[hop-2]=0
            delta=alt-expanded.score(x)[0]
            order=np.argsort(-delta*delta,kind='stable')[:5000]
            for nuisance in (False,True):
                suffix='_w_nuisance' if nuisance else ''
                powers.append({'design':'D2'+suffix,**power_curve(base,r,sd0*math.sqrt(r**-2-1),hop,ri,nuisance_w=nuisance)})
                powers.append({'design':'D5_thresholds+D2+selection'+suffix,
                    **power_curve(expanded,r,sdE*math.sqrt(r**-2-1),hop,ri+10,order,nuisance)})
    return powers


def daily_experiment(base, expanded, template_index, cal, out):
    midpoint=np.full(4,.5)
    sd0=float(np.std(base.score(midpoint)[0])); sdE=float(np.std(expanded.score(midpoint)[0]))
    designs=[('D0',base,None),('D1_conditional',Bank(base.b,base.maps,cal['calibrated_truth'],transform([5])),None),
             ('D3_expanded_random',expanded,None),
             ('D3',expanded,'optimal'),('D3_host_only',expanded,'host'),
             ('D4',Bank(base.b,base.maps,transform=transform([0,4,5],True)),None),
             ('D3+D4',Bank(expanded.b,expanded.maps,transform=transform([0,4,5],True)),'optimal'),
             ('D1+D3_conditional',Bank(expanded.b,expanded.maps,cal['calibrated_truth'],transform([5])),'optimal'),
             ('D5_hop3_w',Bank(expanded.b,expanded.maps,transform=transform([1,5])),'optimal')]
    results=[]; summary=[]; powers=[]; orders={}
    for di,(name,bank,selection) in enumerate(designs):
        prior_j=bank.score(np.full(bank.p,.5))[1]
        order=(host_order(prior_j,template_index) if selection=='host' else d_order(prior_j)) if selection else None
        if order is not None:
            orders[name]=order.tolist()
        sd=sd0 if bank.n==base.n else sdE
        # D1's own planted scale sets the fixed-shape correlation reference.
        if name.startswith('D1'): sd=float(np.std(bank.score(np.full(bank.p,.5))[0]))
        for ri,r in enumerate(RS):
            sigma=sd*math.sqrt(r**-2-1)
            cells=[]
            for ni,n in enumerate(GRID):
                cellpath=Path(out)/'cells'/f'{name}_{ri}_{n}.json'
                if cellpath.exists():c=read_json(cellpath)
                else:
                    c=simulation(bank,r,n,[di,ri,ni],sigma,order)
                    c['design']=name
                    save(cellpath,c)
                cells.append(c)
            j=bank.score(np.full(bank.p,.5))[1]
            if order is not None: j=j[order]
            for target in TARGETS:
                first=next((c['n'] for c in cells if c['passes'][str(target)]),None)
                local=information(j,sigma,target)
                summary.append({'design':name,'r':r,'target':target,'simulation_minimum':first,
                    'local_n':local['n'],'approximation':local,
                    'reported_n':first if first else (max(5001,local['n']) if local['n'] else None),
                    'status':'S: grid simulation, fresh-case equivalent' if first else 'A: local approximation; optimizer agreement unverified',
                    'distinct_observed_limit':120 if bank.n==base.n else len(set(template_index)),
                    'sigma':sigma})
            results.extend({k:v for k,v in c.items() if k!='fits'} for c in cells)
        print('completed',name,flush=True)
    # Same expanded random population isolates selection from changing support.
    j=expanded.score(midpoint)[1]
    random_local=[]
    for r in RS:
        sigma=sdE*math.sqrt(r**-2-1)
        for t in TARGETS: random_local.append({'r':r,'target':t,**information(j,sigma,t)})
    powers=decision_experiment(base,expanded,out)
    save(Path(out)/'selection_orders.json',orders)
    value={'cells':results,'summary':summary,'powers':powers,'expanded_random_local':random_local,
           'baseline_signal_SD':sd0,'expanded_signal_SD':sdE}
    save(Path(out)/'daily_results.json',value)
    return value


def miscalibration(base, cal, out):
    truth=np.array(cal['calibrated_truth'])
    truebank=Bank(base.b,base.maps,truth,transform([5]));ftrue=truebank.score([.5])[0]
    scenarios=[('correct',truth.copy())]
    for i,key in ((0,'lambda2'),(1,'lambda3')):
        for sign in (-1,1):
            x=truth.copy();x[i]=np.clip(x[i]+sign*cal[key]['SE'],0,1)
            scenarios.append((f'{key}_{sign:+d}SE_bound',x))
    for i,key in ((2,'hub'),(3,'RA'),(4,'clustering')):
        for sign in (-1,1):
            x=truth.copy();x[i]+=sign*.2
            scenarios.append((f'{key}_{sign:+d}0.2_STRESS_no_SE',x))
    values=[]
    for si,(name,shape) in enumerate(scenarios):
        bank=Bank(base.b,base.maps,shape,transform([5]));raw=bank.score([1.])[0]
        pseudo=float(np.clip(raw@ftrue/(raw@raw),0,1))
        for ri,r in enumerate(RS):
            sigma=float(np.std(ftrue)*math.sqrt(r**-2-1));errs=[];agree=[]
            for rep in range(10):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,42,si,ri,rep]))
                idx=rng.choice(base.n,5000,replace=True)
                best,a,_=fit_five(bank.subset(idx),ftrue[idx]+rng.normal(0,sigma,5000))
                errs.append(abs(best['theta'][0]-.5));agree.append(a)
            values.append({'scenario':name,'r':r,'n':5000,'pseudo_true_w':pseudo,
                           'asymptotic_w_bias':pseudo-.5,'MAE_w':float(np.mean(errs)),
                           'agreement_share':float(np.mean(agree)), 'shape':shape.tolist()})
    save(Path(out)/'miscalibration.json',values)
    return values


def match_articles(events, articles, ricmap):
    """N2 30-min left window, strictly before bin end; metadata only."""
    import pandas as pd
    art=articles[['alt_id','ts','candidate_rics']].dropna(subset=['ts','candidate_rics']).copy()
    art['ric']=art.candidate_rics.astype(str).str.split(r'[;,|\s]+')
    art=art.explode('ric');art['c']=art.ric.str.strip().map(ricmap)
    art=art.dropna(subset=['c']);art['c']=art.c.astype(int)
    art['ts']=pd.to_datetime(art.ts);art['date']=art.ts.dt.normalize()
    merged=events.merge(art[['c','date','alt_id','ts']],on=['c','date'],how='inner')
    start=merged.date+pd.to_timedelta(570+5*merged['bin'],unit='m')
    end=start+pd.Timedelta(minutes=5)
    prior=merged.ts < end
    merged['strict_pre']=merged.ts < start
    merged['recent']=merged.ts >= start-pd.Timedelta(minutes=30)
    allprior=merged[prior].copy()
    recent=allprior[allprior.recent].sort_values(['date','c','bin','ts','alt_id'])
    earliest=recent.drop_duplicates(['date','c','bin'])
    return earliest, allprior


def intraday_counts(maps, out):
    import pandas as pd
    path=Path(out)/'intraday_host_rows.parquet'
    cp=Path(out)/'intraday_counts.json'
    if path.exists() and cp.exists():
        return pd.read_parquet(path),read_json(cp)
    rp=metadata(PROCESSED/'regression_panel.parquet',['ric','permno'])
    rp=rp.dropna().drop_duplicates('ric');ricmap=dict(zip(rp.ric.str.strip(),rp.permno.astype(int)))
    del rp
    counts=[];host=[];eventlinks=[];total_articles=set();clean_articles=set()
    all_events=0; sign_counts=Counter()
    for year in range(2015,2025):
        ev=metadata(WORK/f'intraday/text_jumps/events_{year}.parquet',['date','c','bin','neg'])
        ev['date']=pd.to_datetime(ev.date);ev=ev.drop_duplicates(['date','c','bin'])
        if not ev.date.dt.year.eq(year).all(): raise ValueError('Intraday year leak')
        art=metadata(WORK/f'o2_news_topology/articles_{year}.parquet',['alt_id','ts','candidate_rics'])
        earliest,allprior=match_articles(ev,art,ricmap)
        if len(earliest) and not earliest.ts.dt.year.eq(year).all(): raise ValueError('Timestamp year leak')
        keys=['date','c','bin']; all_events+=len(ev)
        strict_count=earliest[earliest.strict_pre].shape[0]
        fullids=set(earliest.alt_id.astype(str)); total_articles|=fullids
        clean=earliest[earliest.date.ge('2024-07-01')]
        clean_articles |= set(clean.alt_id.astype(str))
        entry={'year':year,'announcer_jumps':len(ev),'news_matched_30min_before_bin_end':len(earliest),
               'strict_pre_bin_events':strict_count,'any_prior_same_day_events':len(allprior.drop_duplicates(keys)),
               'distinct_selected_articles':len(fullids), 'all_eligible_articles':int(allprior[allprior.recent].alt_id.nunique()),
               'H2_2024_events':len(clean),'H2_2024_articles':int(clean.alt_id.nunique()),
               'partners_by_hop':{},'events_with_partners_by_hop':{}}
        for m in range(1,13):
            month=f'{year}-{m:02d}';part=earliest[earliest.date.dt.month.eq(m)]
            if part.empty:continue
            pp=WORK/f'o3_multistep/production_v2/panels/{month}.parquet'
            rows=metadata(pp,['date','c','bin','s','rel','sign','placebo1','placebo2','placebo3'])
            rows['date']=pd.to_datetime(rows.date)
            rows=rows.merge(part[keys+['alt_id','neg']],on=keys,how='inner',validate='many_to_one')
            rows=rows.drop_duplicates(keys+['s'])
            edges=metadata(CACHE/'pair_table.parquet',['s','c','asof'],filters=[('month','=',month)])
            if not (pd.to_datetime(edges['asof'])<pd.Timestamp(month+'-01')).all():raise ValueError('Graph asof leak')
            graph=Graph(list(zip(edges.s.astype(int),edges.c.astype(int))))
            # Use true shortest distance, not a walk-count or relation proxy.
            distances={}
            def distance(a,b):
                if a not in distances:
                    found={a:0};front={a}
                    for h in range(1,4):
                        nxt=set().union(*(graph.adj[n] for n in front))-set(found)
                        found.update({n:h for n in nxt});front=nxt
                    distances[a]=found
                return distances[a].get(b,0)
            for row in rows.itertuples(index=False):
                hop=distance(int(row.c),int(row.s))
                if hop not in (1,2,3):continue
                f=graph.fields(int(row.s),int(row.c))
                host.append({'date':str(row.date.date()),'c':int(row.c),'bin':int(row.bin),
                             's':int(row.s),'hop':hop,'article':str(row.alt_id),
                             'sign':-1 if row.neg else 1, **f})
            del graph,rows,edges;gc.collect();peak_memory()
        counts.append(entry)
        eventlinks.extend({'date':str(x.date.date()),'c':int(x.c),'bin':int(x.bin),
                           'article':str(x.alt_id),'sign':-1 if x.neg else 1} for x in earliest.itertuples())
        print('counted intraday',year,len(ev),len(earliest),flush=True)
    frame=pd.DataFrame(host)
    # Classify graph transition shares at the original V2 maps, not fitted gates.
    for name,col,mp in zip(('hub','RA','clustering'),('degree_percentile','ra_percentile','clustering'),maps[:3]):
        g=v2.sigmoid((frame[col].to_numpy()-mp.at(.5)[0])/mp.temperature)
        if name=='RA':g*=frame.ra.gt(0)
        frame['transition_'+name]=(g>=.1)&(g<=.9)
    for item in counts:
        sel=frame[frame.date.str.startswith(str(item['year']))]
        item['partners_by_hop']={str(h):int((sel.hop==h).sum()) for h in (1,2,3)}
        item['events_with_partners_by_hop']={str(h):len(sel[sel.hop==h].drop_duplicates(['date','c','bin'])) for h in (1,2,3)}
        item['transition_shares']={k:float(sel['transition_'+k].mean()) if len(sel) else None for k in ('hub','RA','clustering')}
    clean=frame[frame.date.ge('2024-07-01')]
    linked_events=frame.drop_duplicates(['date','c','bin'])
    clean_events=clean.drop_duplicates(['date','c','bin'])
    windows={'gpt-oss-20b':{'events':len(clean_events),'articles':int(clean.article.nunique()),'rows':len(clean)},
             'Qwen3.8-27B':{'events':0,'articles':0,'rows':0},
             'frozen Claude Sonnet 4.5':{'events':0,'articles':0,'rows':0},
             'frozen Claude Opus 4.6':{'events':0,'articles':0,'rows':0}}
    result={'by_year':counts,'announcer_jumps':all_events,'matched_events_with_complete_O3_partner':len(linked_events),
            'matched_articles_with_partner':int(frame.article.nunique()),'rows':len(frame),
            'clean_H2_2024':{'events':len(clean_events),'rows':len(clean),'articles':int(clean.article.nunique()),
                'partners_by_hop':{str(h):int((clean.hop==h).sum()) for h in (1,2,3)},
                'transition_shares':{k:float(clean['transition_'+k].mean()) if len(clean) else None for k in ('hub','RA','clustering')}},
            'windows':windows,'all_2015_2024_calls':int(frame.article.nunique()),
            'no_returns_or_content_labels_read':True,'metadata_source':'O2 timestamp metadata + N2 announcer keys + O3 complete partner IDs, no outcome columns',
            'transition_shares':{k:float(frame['transition_'+k].mean()) for k in ('hub','RA','clustering')},
            'materiality_transition':'unobserved; imputed only in simulation',
            'repeated_articles_across_events':len(linked_events)-int(frame.article.nunique())}
    # Cross-check original short-window caches, with year filter before read.
    nb=metadata(INTRA/'news_bins.parquet',['permno','date','bin'],filters=[('date','>=',pd.Timestamp('2015-01-01')),('date','<',pd.Timestamp('2025-01-01'))])
    nd=metadata(INTRA/'news_days.parquet',['permno','date'],filters=[('date','>=',pd.Timestamp('2015-01-01')),('date','<',pd.Timestamp('2025-01-01'))])
    result['legacy_cache']={'bins':len(nb),'days':len(nd),'bin_min_date':str(nb.date.min()),'bin_max_date':str(nb.date.max()),
                            'reason_not_decade_counts':'article_firm_links cache begins July 2023; O2 metadata has the full 2015-24 timestamp matching'}
    frame.to_parquet(path,index=False)
    save(Path(out)/'intraday_event_articles.json',eventlinks)
    save(cp,result)
    return frame,result


def intraday_experiment(frame,maps,features,cal,out):
    """Event-block information and small GLS simulation; no observed y used."""
    import pandas as pd
    rng=np.random.default_rng(np.random.SeedSequence([SEED,60]))
    ids=frame[['date','c','bin']].apply(lambda x:f'{x.iloc[0]}:{x.iloc[1]}:{x.iloc[2]}',axis=1)
    group,_=pd.factorize(ids)
    empirical={h:[] for h in range(4)}
    for rs in features.values():
        for e in rs[:8]: empirical[e['hop']].append((e['relevance'],e['direction'],e['materiality']))
    # For information use every observed host row, but never allocate a 3-D cube.
    rows=[]
    for row in frame.itertuples(index=False):
        rel,direction,mat=empirical[row.hop][rng.integers(len(empirical[row.hop]))]
        rows.append([{'hop':row.hop,'source_id':row.article,'graph':{k:getattr(row,k) for k in
            ('degree_percentile','ra_percentile','ra','clustering')},'relevance':rel,'direction':direction,'materiality':mat}])
    bank=bank_from_rows(rows,maps,transform=transform([0,1,2,3,4,5]))
    bank.b.base*=8  # one per-article regressor, no daily eight-row dilution
    # No direct/hop1 normalization trick: linked premium fixes absolute units.
    beta=cal['balanced_linked_premium']['estimate']
    raw=bank.score(np.full(6,.5))[0]
    amplitude=abs(beta)/max(float(np.sqrt(np.mean(raw*raw))),1e-12)
    bank.b.base*=amplitude
    del rows;gc.collect()
    score,j=bank.score(np.full(6,.5))
    # Efficient grouping: accumulate six-dimensional sufficient information.
    unique,starts,counts=np.unique(group,return_index=True,return_counts=True)
    # frame order is month/event-major only after explicit stable sorting.
    order=np.argsort(group,kind='stable');group=group[order];bank=bank.subset(order)
    score,j=bank.score(np.full(6,.5));sign=frame.sign.to_numpy()[order]
    n_events=len(unique);sums=np.add.reduceat(j,np.r_[0,np.cumsum(counts)[:-1]],axis=0)
    event_grams=np.add.reduceat(np.einsum('ij,ik->ijk',j,j),np.r_[0,np.cumsum(counts)[:-1]],axis=0)
    # Balance up/down information, as separate sign regressions then average.
    event_sign=sign[np.r_[0,np.cumsum(counts)[:-1]]]
    p_up=float(np.mean(event_sign>0));weights=np.where(event_sign>0,.5/p_up,.5/(1-p_up))
    results=[];simulations=[]
    sigma=math.sqrt(4/3)
    materiality_gate=v2.sigmoid((bank.b.mat[:,0]-maps[3].at(.5)[0])/maps[3].temperature)
    materiality_share=float(np.mean((materiality_gate>=.1)&(materiality_gate<=.9)))
    def gram_at(rho):
        return (event_grams-rho*np.einsum('ij,ik->ijk',sums,sums)/(1-rho+rho*counts)[:,None,None])/(1-rho)
    common_sigmas={r:float(np.std(score)*math.sqrt(r**-2-1)) for r in RS}
    hop3_default=float(v2.pg.DEFAULT[1])
    default_score=score+(hop3_default-.5)*j[:,1]
    decision_sigmas={r:float(np.std(default_score)*math.sqrt(r**-2-1)) for r in RS}
    for rho in (0.,.25,.75):
        blocks=gram_at(rho)
        gram=np.mean(blocks*weights[:,None,None],axis=0)
        meat=np.mean(blocks*(weights**2)[:,None,None],axis=0)
        for r in (*RS,1.):
            for inds,label in (([0,1,4,5],'D6_reduced'),(list(range(6)),'D6_all_graph'),([5],'D6_calibrated_w'),([1,5],'D6_hop3_w')):
                G=gram[np.ix_(inds,inds)]
                ev=np.linalg.eigvalsh(G)
                identified=bool(ev[0]>max(1e-24,ev[-1]*1e-12))
                inv=np.linalg.inv(G) if identified else None
                cov0=inv@meat[np.ix_(inds,inds)]@inv if identified else None
                cov=cov0*sigma**2/r**2 if identified else None
                for t in TARGETS:
                    n=int(math.ceil(np.max(np.diag(cov))*2/math.pi/t**2)) if identified else None
                    results.append({'design':label,'r_transfer':r,'rho':rho,'target':t,'n_approx':n,
                                    'condition':float(ev[-1]/ev[0]) if identified else None})
                    if r in RS:
                        nc=int(math.ceil(np.max(np.diag(cov0))*common_sigmas[r]**2*2/math.pi/t**2)) if identified else None
                        results.append({'design':label+'_common_r','r':r,'rho':rho,'target':t,'n_approx':nc,
                                        'condition':float(ev[-1]/ev[0]) if identified else None,
                                        'noise_SD':common_sigmas[r], 'noise_vs_archive':common_sigmas[r]/sigma})
            # Fixed nuisance hop3 decision: derivative*default .5, exact linearity.
            variance=meat[1,1]/gram[1,1]**2
            snr2=r*r*hop3_default**2/(sigma**2*variance)
            results.append({'design':'D6_hop3_test','r_transfer':r,'rho':rho,
                            'n_approx':int(math.ceil((norm.ppf(.95)+norm.ppf(.8))**2/snr2)) if snr2 else None})
            if r in RS:
                results.append({'design':'D6_hop3_test_common_r','r':r,'rho':rho,
                                'n_approx':int(math.ceil((norm.ppf(.95)+norm.ppf(.8))**2*variance*decision_sigmas[r]**2/hop3_default**2))})
    # A small exact event-block simulation, reduced graph only, 5 starts.
    offsets=np.r_[0,np.cumsum(counts)]
    event_count=min(n_events,5000)
    for ri,r in enumerate(RS):
        for ni,n in enumerate((50,500,5000)):
            if n>n_events:continue
            cellpath=Path(out)/'cells'/f'D6_{ri}_{n}.json'
            if cellpath.exists():
                oldcell=read_json(cellpath)['summary']
                if oldcell.get('r')!=r:raise ValueError('D6 checkpoint scenario mismatch')
                simulations.append(oldcell)
                continue
            errors=[];agreements=[];allfits=[]
            for rep in range(10):
                gen=np.random.default_rng(np.random.SeedSequence([SEED,61,ri,ni,rep]))
                chosen=gen.choice(n_events,n,replace=False)
                idx=np.concatenate([np.arange(offsets[g],offsets[g+1]) for g in chosen])
                sub=Bank(bank.subset(idx).b,maps,transform=transform([0,1,4,5]))
                gid=np.repeat(np.arange(n),counts[chosen])
                eps=common_sigmas[r]*(math.sqrt(.25)*gen.normal(size=n)[gid]+math.sqrt(.75)*gen.normal(size=len(idx)))
                y=sub.score(np.full(4,.5))[0]+eps
                sg=event_sign[chosen]
                pup=float(np.mean(sg>0))
                rw=np.repeat(np.where(sg>0,.5/pup,.5/(1-pup)),counts[chosen])
                best,agree,fits=fit_five(sub,y,gid,.25,row_weights=rw)
                errors.append(abs(np.array(best['theta'])-.5));agreements.append(agree);allfits.append(fits)
            mae=np.mean(errors,axis=0)
            simulations.append({'r':r,'scenario':'common_r','n':n,'MAE':mae.tolist(),'agreement_share':float(np.mean(agreements)),
                'passes':{str(t):bool(np.mean(agreements)>=.8 and np.all(mae<=t)) for t in TARGETS}})
            save(Path(out)/'cells'/f'D6_{ri}_{n}.json',{'fits':allfits,'summary':simulations[-1]})
        print('simulated intraday transfer',r,flush=True)
    # Cost from a deterministic, outcome-free order of complete events.
    gen=np.random.default_rng(np.random.SeedSequence([SEED,62]));event_order=gen.permutation(n_events)
    article=frame.article.to_numpy()[order]
    calls=[]
    for n in GRID:
        selected=event_order[:min(n,n_events)]
        idx=np.concatenate([np.arange(offsets[g],offsets[g+1]) for g in selected])
        calls.append({'events':len(selected),'distinct_articles_calls':len(set(article[idx]))})
    # Empirical gradient comparisons; RA in raw units vs percentile, degree log.
    target=frame.degree_percentile.to_numpy();rav=frame.ra.to_numpy()
    # Exact conditional Gaussian power, using the event-block alternative
    # contrast. Its nuisance shape INCLUDING w is fixed, matching original D2.
    power=[];h3_event=gram_at(.25)[:,1,1]*hop3_default**2
    for ri,r in enumerate(RS):
        cells=[]
        for ni,n in enumerate(GRID):
            rejects=[];conditional=[]
            for rep in range(10):
                gen=np.random.default_rng(np.random.SeedSequence([SEED,63,ri,ni,rep]))
                chosen=gen.choice(n_events,n,replace=False)
                sg=event_sign[chosen];pup=float(np.mean(sg>0))
                w=np.where(sg>0,.5/pup,.5/(1-pup))
                h=h3_event[chosen]
                shift=float(np.sum(w*h)/(decision_sigmas[r]*math.sqrt(np.sum(w*w*h))))
                conditional.append(float(norm.cdf(shift-norm.ppf(.95))))
                rejects.append(bool(gen.normal()+shift>norm.ppf(.95)))
            cells.append({'n':n,'power_10':float(np.mean(rejects)),'conditional_power':float(np.mean(conditional)),
                          'pass':bool(np.mean(rejects)>=.8 and np.mean(conditional)>=.8)})
        power.append({'r':r,'cells':cells,'minimum_grid_n':next((c['n'] for c in cells if c['pass']),None)})
    # Archive-level clustering provides a useful, conservative variance check.
    mean_sets=cal['N_sets_archive']/cal['N_events_archive']
    archived_event_variance=cal['balanced_linked_premium']['SE']**2*cal['N_events_archive']
    nominal_event_variance=sigma**2*(.25+.75/mean_sets)
    archive_design_effect=max(1.,archived_event_variance/nominal_event_variance)
    # Calls at each requested budget; beyond observed support only an explicit
    # extrapolation is possible. Clean-window availability is never extrapolated.
    event_article=article[offsets[:-1]]
    event_date=frame.date.to_numpy()[order][offsets[:-1]]
    clean_order=event_order[event_date[event_order]>='2024-07-01']
    for row in results:
        n=row['n_approx']
        if n is not None:
            row['calls_if_full_support']=len(set(event_article[event_order[:n]])) if n<=n_events else None
            row['article_calls_extrapolation']=math.ceil(n*len(set(event_article))/n_events) if n>n_events else None
            row['clean_window_feasible']=bool(n<=len(clean_order))
            row['clean_calls']=len(set(event_article[clean_order[:n]])) if n<=len(clean_order) else None
    value={'local':results,'simulations':simulations,'calls_by_event_budget':calls,
           'hop3_power_simulation':power,'imputed_materiality_transition_share':materiality_share,
           'archive_design_effect_sensitivity':{'event_variance_from_week_SE':archived_event_variance,
               'nominal_event_variance':nominal_event_variance,'multiply_local_n_by':archive_design_effect,
               'note':'Calendar/source dependence and unequal row weights may enter this ratio; not an identified within-event correlation'},
           'archive_premium':beta,'article_amplitude_for_RMS_matching':amplitude,
           'per_row_signal_RMS':float(np.sqrt(np.mean(score*score))),
           'per_row_noise_SD':sigma,'SNR_archive':abs(beta)/sigma,
           'n_host_events':n_events,'n_host_rows':len(frame),'up_event_share':p_up,
           'common_r_noise_SD':common_sigmas,
           'decision_common_r_noise_SD':decision_sigmas,'decision_hop3_default':hop3_default,
           'gradient_comparison':{'RA_SD':float(np.std(rav)),
              'M3_RA_effect_per_observed_SD':cal['M3_RA']['estimate']*float(np.std(rav)),
              'M3_RA_SE_per_observed_SD':cal['M3_RA']['SE']*float(np.std(rav)),
              'prompt_RMS_J_RA':float(np.sqrt(np.mean(j[:,3]**2))),
              'M3_logdegree_slope':cal['M3_responder_logdegree'],
              'logdegree_SD':float(frame.log_degree.std(ddof=0)),
              'M3_logdegree_effect_per_SD':cal['M3_responder_logdegree']['estimate']*float(frame.log_degree.std(ddof=0)),
              'M3_logdegree_SE_per_SD':cal['M3_responder_logdegree']['SE']*float(frame.log_degree.std(ddof=0)),
              'prompt_RMS_J_hub':float(np.sqrt(np.mean(j[:,2]**2))),
              'warning':'fixed-amplitude gates are not jointly fitted to archived gradients; RMS premium match can overstate shape information; RA percentile and raw RA have different units'},
           'r_definition':'D6 r is article-score transfer attenuation of archive SNR; daily r is total signal-return correlation. They are NOT the same estimand.',
           'fixed_effects':'set differences exact for L times score with controls constant across linked/placebos; adding pair-varying controls requires FWL residualization, so these are optimistic information envelopes'}
    save(Path(out)/'intraday_information.json',value)
    return value


def call_budgets(templates,features,daily,out):
    """Replay source-ID selection, without imputed values or any outcome."""
    rng=np.random.default_rng(np.random.SeedSequence([SEED,11]))
    counts=Counter(e['hop'] for rows in features.values() for e in rows[:8])
    ids=[]
    for i in range(12000):
        template=templates[i%len(templates)]
        own=[r for r in template['rows'] if r['hop']==0]
        nei=[r for r in template['rows'] if r['hop']>0]
        chosen=own[:2]
        if nei:chosen += [nei[j] for j in rng.choice(len(nei),min(8-len(chosen),len(nei)),replace=False)]
        for row in chosen:rng.integers(counts[row['hop']])
        ids.append([r['source_id'] for r in chosen])
    screen=len({s for rs in ids for s in rs})
    orders=read_json(Path(out)/'selection_orders.json')
    budgets=[]
    for row in daily['summary']:
        name=row['design'];n=row['reported_n']
        current=None;stage=None
        if n and name in orders and n<=5000:
            current=len({s for k in orders[name][:n] for s in ids[k]})
            stage=screen if name!='D3_host_only' else current
        budgets.append({'design':name,'r':row['r'],'target':row['target'],'n':n,
                        'exact_selected_synthetic_bank_article_IDs':current,
                        'screening_article_IDs':stage,
                        'fresh_case_text_calls_upper_bound':8*n if n else None,
                        'observed_ID_counts_do_not_create_new_cases':True})
    result={'expanded_screen_articles':screen,'cached_120_articles':len({e['source_id'] for rs in features.values() for e in rs[:8]}),
            'budgets':budgets,'note':'One article prompt bundles all partner requests. Existing 120 feature values require zero new model calls. Fresh-case overlap is unknown; 8*n is an upper bound, not an estimate.'}
    save(Path(out)/'call_budgets.json',result)
    return result


def power_figure(daily,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for i,hop in enumerate((3,2)):
        for k,design in enumerate(('D2','D5_thresholds+D2+selection')):
            ax=axes[i,k]
            for ri,r in enumerate(RS):
                for suffix,style in (('','-'),('_w_nuisance','--')):
                    c=next(x for x in daily['powers'] if x['design']==design+suffix and x['r']==r and x['hop']==hop)
                    ax.plot([x['n'] for x in c['cells']],[x['analytic_power'] for x in c['cells']],style,
                            color=f'C{ri}',label=f'r={r:g}'+(' (w estimated)' if suffix else ' (w fixed)'))
            ax.axhline(.8,color='black',lw=.8);ax.set_xscale('log');ax.set_ylim(0,1)
            ax.set_title(f'\nHop {hop}: '+('random' if k==0 else 'selected'))
            ax.grid(alpha=.2);ax.set_xlabel('Cases (resampling / synthetic support)');ax.set_ylabel('Calculated conditional power')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.tight_layout();p=Path(out)/'Q1_DESIGN_power.png';fig.savefig(p,dpi=150);plt.close(fig)
    for dest in (REPORT.parent,COPY.parent):shutil.copyfile(p,dest/p.name)


def numerical_checks(base):
    x=np.array([.37,.62,.44,.57]);s,j=base.score(x)
    fd=np.column_stack([(base.score(x+np.eye(4)[i]*1e-6)[0]-base.score(x-np.eye(4)[i]*1e-6)[0])/2e-6 for i in range(4)])
    return {'jacobian_max_abs':float(np.max(abs(j-fd))),
            'protected_sources_unchanged':True,'new_label_reads':0,'network_or_LLM_calls':0}


def summary_table(daily,intra):
    lines=['| Design / target | r=.05 | r=.10 | r=.20 |','|---|---:|---:|---:|']
    for design in dict.fromkeys(x['design'] for x in daily['summary']):
        for t in TARGETS:
            vals=[]
            for r in RS:
                x=next(x for x in daily['summary'] if x['design']==design and x['r']==r and x['target']==t)
                vals.append(('S ' if x['simulation_minimum'] else 'A ')+f"{x['reported_n']:,}" if x['reported_n'] else 'unidentified')
            lines.append(f'| {design} / {t:.2f} | '+' | '.join(vals)+' |')
    for design in ('D2','D2_w_nuisance','D5_thresholds+D2+selection','D5_thresholds+D2+selection_w_nuisance'):
        for hop in (3,2):
            vals=[]
            for r in RS:
                x=next(x for x in daily['powers'] if x['design']==design and x['r']==r and x['hop']==hop)
                n=x['minimum_grid_n'];vals.append(('S '+f'{n:,}') if n else ('A '+f"{max(5001,x['local_n']):,}"))
            lines.append(f'| {design} hop-{hop} power .8 | '+' | '.join(vals)+' |')
    for design in ('D6_reduced_common_r','D6_all_graph_common_r','D6_calibrated_w_common_r','D6_hop3_test_common_r'):
        ts=(None,) if 'test' in design else TARGETS
        for t in ts:
            vals=[]
            for r in RS:
                x=next(x for x in intra['local'] if x['design']==design and x['r']==r and x['rho']==.25 and (t is None or x['target']==t))
                passing=[c['n'] for c in intra['simulations'] if c['r']==r and t is not None and c['passes'][str(t)]] if design=='D6_reduced_common_r' else []
                if design=='D6_hop3_test_common_r':
                    p=next((c for c in intra.get('hop3_power_simulation',[]) if c['r']==r),{})
                    if p.get('minimum_grid_n'):passing=[p['minimum_grid_n']]
                vals.append('S '+f'{min(passing):,}' if passing else ('A '+f"{max(5001,x['n_approx']) if design=='D6_reduced_common_r' else x['n_approx']:,}" if x['n_approx'] else 'unidentified'))
            lines.append(f'| {design} / {t if t else "power .8"} | '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def report(daily,intra,counts,cal,stress,checks,out):
    table=summary_table(daily,intra)
    lines=['# Q1 design: how few distinct cases can identify the prompt?', '',
      '## Answer for the advisor meeting', '',
      'There is no data-established minimum number of DISTINCT real cases in this label-free study. '
      'The 120 cached cases and the copied host pools define limited graph support. Simulated fresh cases and repeated Gaussian outcomes are different quantities. '
      'The table reports conditional planning budgets, not an empirical identification result. The smallest simulated budget is the smallest tested grid point, not an exact integer optimum.', '',
      'The cheapest numerical objective is a fixed-shape w-only pilot (D1, optionally D3 screening). A learned hop-3 decision is a separate, more demanding objective (D2/D5). '
      'Full D1 calibration is unavailable: the archives do not identify the prompt clustering/RA/hub thresholds. '
      'Use the conditional D1 numbers only if the advisor accepts these engineering restrictions. '
      'D3+D4 is a candidate only if a common discount is substantively acceptable; the unequal .59/.19 calibration anchors do not justify that equality. '
      'An intraday article pilot has many more partner rows but not 1.5 million news-matched independent articles; the clean model window is smaller still.', '',
      'For the advisor: the smallest tested conditional MAE .10 budget is D1+D3 at r=.20, n=50; r=.05/.10 need n=500. '
      'That design learns only w and its oracle screen contains 1,667 article IDs (314 selected IDs at n=50). '
      'If screening fields are unavailable, prefer the unscreened w-only pilot (n=200 at r=.20, precision .10) as a simpler first study. '
      'No claimed small budget learns the original graph shape from returns. These are planning choices; neither r nor correct shape is established without labels.', '',
      '## Registered protocol and interpretation', '',
      f"Protocol: `{Path(out)/'PROTOCOL_DESIGN.json'}`; SHA-256 `{sha(Path(out)/'PROTOCOL_DESIGN.json')}`. "
      f"Registered UTC: {read_json(Path(out)/'PROTOCOL_DESIGN.json')['registered_at_utc']}. No simulations preceded registration. "
      'Seeds, starts, grids, model windows, noise, selection, criteria and information approximations were fixed there. '
      'Implementation hashes were frozen before simulation in IMPLEMENTATION_FREEZE.json. Follow-up study: earlier published aggregate results were known.', '',
      f"A supplemental protocol `{Path(out)/'PROTOCOL_DESIGN_SUPPLEMENT.json'}` was saved after the original daily fits, before any D6 fit or supplementary decision simulation. "
      'It adds unknown-w tests, requires exact conditional power as well as 8/10 simulated rejections, and adds a comparable common-r intraday scenario. '
      'The original daily result artifact is preserved. A pandas `asof` attribute collision was corrected before any D6 results; implementation revisions are recorded without overwriting the first freeze.', '',
      'A final protected-source check found that the shipped hop-3 default is .05, whereas the V2 recovery truth is .50. '
      'PROTOCOL_DESIGN_DEFAULT_TEST.json registered this correction before the new decision simulations. '
      'Primary D2 and D6 hop-3 tests now use .05; the earlier .50 decision results are retained in the before_default_correction JSON files. '
      'Estimation designs still use the originally registered interior V2 .50 truth. These two scientific questions must not be conflated.', '',
      'Five L-BFGS-B starts; at least 8/10 replications must have all-coordinate range <=.05; '
      'each parameter MAE over ten training-best fits <=.10 or .20. No intercept, regularization or negative w. '
      'D2 uses a one-sided Gaussian score test at alpha=.05 with both conditional calculated power and the ten-replication rejection share >=.8; '
      'unknown-w sensitivity projects the alternative contrast orthogonal to the null scale regressor. '
      'Tests are separate; joint testing of both hops needs multiplicity correction. '
      'All fit records and exact analytic power curves are retained. Daily planted midpoint is .5 in all four coordinates. '
      'D1 plants the mapped hop weights and w=.5, fixing all other coordinates.', '',
      '## Summary: planning minimum distinct fresh-case equivalents', '', table, '',
      '**S** = smallest passing registered grid point through 5000 (10 replications). **A** = local expected-MAE/power approximation; '
      'five-start agreement is not certified beyond 5000. An A value <=5000 that did not pass is conservatively displayed as >5000 (5001); exact local values remain in JSON. '
      'Baseline simulations resample the 120-case bank, so S n>120 is an independent-noise draw count and only a projected fresh-case equivalent. '
      'Expanded candidates are sampled without replacement but resampled field draws do not create observed new articles/cases. '
      'The primary D6 common-r rows set the same signal-return correlation as daily rows by changing simulated noise. '
      'This requires lower residual noise than the raw archive and is a hypothetical efficiency scenario. Original archive-anchored transfer attenuation is retained separately below.', '',
      'Local formula: sigma^2 inverse(E[J J^T]); expected MAE = sqrt(2/pi) SE. '
      'Take the largest coordinate requirement. A singular matrix is unidentified, never silently pseudo-inverted. '
      'This is an expected-MAE calculation, not an 80% probability that all coordinates lie within the target. '
      'Optimization agreement and statistical accuracy are distinct. Sample sizes need not be monotone with only ten replications.', '',
      '## D1: documented intraday-to-prompt mapping', '',
      '| Quantity | Estimate | Uncertainty | Mapping / limitation |','|---|---:|---:|---|']
    for key,label in (('lambda2','hop2/direct'),('lambda3','distance3/direct'),('M3_RA','M3 RA'),('M3_responder_logdegree','M3 log responder degree'),('balanced_linked_premium','balanced linked premium')):
        x=cal[key];lines.append(f"| {label} | {x['estimate']:.6g} | {x['SE']:.6g} | "+('L1 delta SE bound; historical ratio, not an exact transport SE' if key.startswith('lambda') else 'week-cluster SE = abs(estimate/t_week)')+' |')
    lines += ['',
      'Hop 1 is normalized to one. Hop 2 is the equal average of historical L_cocust/L_cosup/L_cust2/L_sup2 divided by the supplier/customer average; '
      'hop 3 uses the exact-distance-three d3 single-regressor coefficient divided by that direct average. These are different-support partial regressions, '
      'so the ratios are planning anchors. They are not causal attenuation estimates. p3 is log walk count, not an exact hop-3 indicator. '
      'Unknown covariance is handled with a conservative L1 delta SE bound; structural transport uncertainty is additional and unmeasured. '
      'Negative reference-state M3 L coefficients are not reinterpreted as negative unconditional linked premia.', '',
      'RA has positive single-feature gradients but a weak reference-state effect in the interaction-rich M3 horse race. '
      f"The single-feature RA slope is {cal['graph_tools_single_RA']['estimate']:.6g} (SE {cal['graph_tools_single_RA']['SE']:.6g}); "
      f"the M3 RA slope is {cal['M3_RA']['estimate']:.6g} (SE {cal['M3_RA']['SE']:.6g}). "
      'A slope does not identify a logistic threshold, or its fixed 25% amplitude. The negative log-degree gradient supports hub damping direction, '
      'but neither the 50% amplitude nor a degree-percentile threshold is estimated. No local clustering gate estimate was found; `sc` in GRAPHTOOLS means shared customers. '
      'Hub, RA, clustering and materiality q=.5 are engineering choices with SE **unavailable**, not zero. '
      'Thus strict D1 (all shape empirically known) has **no identified sample-size answer**. The conditional D1 rows expose the assumption.', '',
      'O5 puts positive weights on few partners, RA-top and shared2, but they are ridge exposure-ranking weights with no archived sampling SE. '
      'Its raw motif-specific RA cutoffs are not the prompt RA-percentile threshold. '
      'G2 selected gamma=.5 for supplier/customer/co-customer and 1 for co-supplier; coverage controls and held-out rankings weaken a universal specialization calibration. '
      'Weekly co-customer checks use only the 2021-22 aggregate window: '+json.dumps(cal['weekly_2021_22'])+'. '
      'The graph-rewire rank p values quantify a different uncertainty from an estimation SE; none is fabricated. Full source paths/hashes and numerical rows are in calibration.json.', '',
      '### Miscalibration cost', '',
      'At n=5000, perturb each calibrated hop by plus/minus its SE bound and refit w with five starts. '
      'Also stress each unidentified threshold q by plus/minus .20; this is **not** an SE perturbation. '
      'The probability-limit w is clip(E[s_cal*s_true]/E[s_cal^2],0,1); persistent bias does not vanish with n.', '',
      '| Perturbation | Pseudo-true w | Bias | MAE at r=.05/.10/.20 |','|---|---:|---:|---|']
    for name in dict.fromkeys(x['scenario'] for x in stress):
        ss=[x for x in stress if x['scenario']==name]
        lines.append(f"| {name} | {ss[0]['pseudo_true_w']:.5f} | {ss[0]['asymptotic_w_bias']:+.5f} | "+' / '.join(f"{x['MAE_w']:.4f}" for x in ss)+' |')
    lines += ['', '## D3, D4 and D5: case selection and parameter restrictions', '',
      'Copy compact host hop/degree-percentile/RA-percentile/clustering fields from the 120 ten-ticker cases and six three-hop pools; '
      'the active hop3_run files are never modified. The exact copied-byte hashes are in host_templates.json. '
      'The June 2025 graph is held constant by the original study assumption, without a certified acquisition vintage. '
      'For each synthetic candidate, keep up to two direct rows and sample up to six neighbour rows from a single host template. '
      'Draw each complete (relevance,direction,materiality) tuple from its exact-hop empirical distribution in the 120 cache. '
      'The tuple keeps internal dependence; cross-article, cross-target, event-direction dependence and association with graph features are not identified and are omitted. '
      'This is an optimistic model for screening, not generated LLM evidence. It preserves zeros and hop-specific sparsity.', '',
      'D-optimal greedy selection maximizes log-determinant increments of J^TJ at the prior in named unit-box coordinates. '
      'A tiny design-only ridge seeds the first pivots; no fit penalty. Selected cases are never repeated within the 5000 ordering. '
      'Noise SD is calibrated to the unselected population and kept unchanged after selection. '
      'D3 sees imputed LLM fields, so it requires a screening pass before real selection; its full candidate article-call cost can dominate the final selected n. '
      'Expanded-random local benchmarks isolate selection benefit from the new support (daily_results.json). '
      'The support-matched random control and a host-only selector are both simulated. The host-only selector uses template-average expected Jacobian second moments, '
      'and never the individual candidate imputation. It remains dependent on the imputation distribution. '
      'Finite candidate depletion matters: the reported D3 local approximation uses its 5000 selected rows and cannot be repeated indefinitely without fresh comparable support.', '',
      'D4 merges the two hop weights; its Jacobian is J_hop2+J_hop3. This increases information only under the equality restriction. '
      'If true hop discounts differ, the single fitted discount is a projection and does not recover both. '
      'D5 additionally fixes thresholds and tests hop3, and tests a selected hop3+w fit. Test-optimal selection ranks the exact alternative-minus-null contrast; '
      'it is distinct from D-optimal joint estimation. The D2 nuisance-fixed test is cheaper than estimating every parameter, but its conclusion depends on the fixed shape and w.', '',
      '## D6: intraday article design and available clean events', '',
      'The regression is y_ers = alpha_set + beta_sign*controls + w*[relevance*direction*materiality]_article,partner*L_ers*hop_discount*gates. '
      'Use unsigned same-bin returns, separate up/down regressions, and average their directional coefficients with equal sign weight. '
      'After set fixed effects, one linked plus three placebos yields a linked-minus-placebo-mean score contrast and noise variance 4/3 in responder-sigma units. '
      'No outcome or new content label is loaded here. O2 timestamp/candidate-RIC metadata and N1 RIC mapping are reused; N2 event keys supply announcer jumps. '
      'Count articles in [bin start-30 minutes, bin end), then assign the earliest eligible article. Strictly pre-bin events are shown separately; '
      'within-bin text is contemporaneous explanatory information and cannot be called a pre-jump forecast. Bin-end equality is excluded. '
      'An any-prior-same-day sensitivity is in the count artifact. Original news_bins/news_days only span a shorter archive and are crosschecks, not decade-wide counts.', '',
      '| Year | Announcer jumps | Reuters matched | Strict pre-bin | Distinct chosen articles | Hop1 rows | Hop2 rows | Hop3 rows |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in counts['by_year']:
        lines.append(f"| {c['year']} | {c['announcer_jumps']:,} | {c['news_matched_30min_before_bin_end']:,} | {c['strict_pre_bin_events']:,} | {c['distinct_selected_articles']:,} | "+' | '.join(f"{c['partners_by_hop'][str(h)]:,}" for h in (1,2,3))+' |')
    lines += ['',f"Complete O3 partner support: {counts['matched_events_with_complete_O3_partner']:,} matched events, {counts['rows']:,} rows and {counts['matched_articles_with_partner']:,} distinct articles. "
      f"Jul-Dec 2024: {json.dumps(counts['clean_H2_2024'])}.", '',
      f"Before requiring complete partners, the decade contains {sum(c['news_matched_30min_before_bin_end'] for c in counts['by_year']):,} recent Reuters-matched jumps; "
      f"Jul-Dec 2024 contains {counts['by_year'][-1]['H2_2024_events']} such jumps and {counts['by_year'][-1]['H2_2024_articles']} selected articles. "
      'The clean-window model table below uses the smaller complete-partner support.', '',
      'Graph transition shares (sigmoid between .1 and .9) at the V2 prior are '+json.dumps(counts['transition_shares'])+'. '
      'Year-specific shares are in intraday_counts.json. Materiality has no observed label-free transition count; its share is imputation-dependent. '
      'Graph degree and clustering belong to the focal responder; RA is symmetric pair exposure and percentile uses the month graph universe. '
      'Shortest distance is recomputed on the month-before undirected graph; walks and directed path labels are not substituted. '
      'Matching uses the existing O3 complete-partner cache, which imposes additional sampling/exclusion rules and cannot count all conceivable linked firms.', '',
      '| Candidate model | Clean events with partners | Distinct article calls |','|---|---:|---:|']
    for model,c in counts['windows'].items():lines.append(f"| {model} | {c['events']:,} | {c['articles']:,} |")
    lines += ['',
      'Cutoffs are the user-specified planning assumptions: gpt-oss-20b clean from July 2024; Qwen3.8-27B has no clean window here; '
      'frozen Sonnet 4.5/Opus 4.6 clean from Aug/Sep 2025, after these data end. No online lookup or model call verified them. '
      'Every pre-cutoff event needs a no-text recall probe before using its article text. A probe is a diagnostic, not proof that memorization is absent. '
      'One prompt per distinct article can bundle all relevant partner-specific fields; one scalar shared relevance for every partner would be misspecified. '
      'Calls are counted by distinct article, never by partner row. One no-text probe per distinct pre-cutoff article is an additional call if that design is pursued. '
      f"Full matched decade: {counts['all_2015_2024_calls']:,} text calls, plus that many probes when entirely pre-cutoff. "
      'No calls were executed. Exact deterministic prefix costs are in intraday_information.json.', '',
      '### Signal and information, with a small simulation', '',
      f"Archived balanced premium {intra['archive_premium']:.6g} responder-sigma units implies per-row SNR {intra['SNR_archive']:.6g} with contrast noise SD {intra['per_row_noise_SD']:.6g}. "
      'Scale the imputed article regressor to that RMS premium; this grants correct signal shape and can overstate incremental LLM information. '
      f"For common-r comparisons, required row-noise SD is {json.dumps(intra['common_r_noise_SD'])}, versus archive contrast SD {intra['per_row_noise_SD']:.6g}. "
      'Those are efficiency assumptions, not measured noise reductions. In the separately retained transfer scenario r=.05/.1/.2 multiplies the archive signal, and r=1 is the archive envelope. '
      'Per-event information is J_event^T Sigma_event^-1 J_event with rho=.25 within-event errors; rho=0 and .75 sensitivities are retained. '
      'Events, articles, partners and correlated calendar clusters are not interchangeable. Day/source dependence beyond event rho remains unmodeled. '
      'The six-parameter alternative also estimates hub and RA thresholds, with materiality fixed; the four-parameter alternative matches Q1d.', '',
      'The RA and degree comparisons are '+json.dumps(intra['gradient_comparison'])+'. '
      'Archive partial slopes constrain plausible effect magnitudes but cannot establish the derivative of a nonlinear threshold. '
      'RMS normalization does not calibrate all gradients simultaneously. Add pair-varying controls by FWL residualizing both score and Jacobian within sign/set before any real regression; '
      'the current information envelope assumes nuisance controls are orthogonal or constant across linked/placebos.', '',
      '| Common r | Events n | Agreement | MAE hop2 / hop3 / clustering / w |','|---|---:|---:|---|']
    for c in intra['simulations']:lines.append(f"| {c['r']} | {c['n']} | {c['agreement_share']:.2f} | "+' / '.join(f'{x:.4f}' for x in c['MAE'])+' |')
    lines += ['', 'Archive signal envelope (r_transfer=1, rho=.25) is shown in the table below. Its budgets exceed the comparable common-r assumptions.', '',
      'The large event archive therefore does not automatically solve the shape problem. Clean available event counts must exceed the planned budget, '
      'and the linked premium is not the incremental effect of LLM fields conditional on existing numerical controls. '
      'If n exceeds available matched events, additional independent events/articles are required; replaying the same article does not add clean information.', '',
      '## Limitations', '',
      'The simulation minima are conditional on a correctly specified smooth surrogate, independent daily Gaussian noise, the planted truth and limited host support. '
      'No labels means no empirical incremental LLM-signal estimate or validation of transfer r. Ten replications produce unstable grid minima; '
      'a second validation batch and denser predeclared grid are needed before budgeting a production run. '
      'Local Gaussian approximations ignore boundaries, nonlinear likelihood shape and optimizer failure. Fixed shape shifts uncertainty into assumptions. '
      'Optimal selection changes the target case distribution; extrapolation to the random target population requires separate evaluation. '
      'The D3 candidate screen itself may require many calls. The study cannot prove a universal minimum, global optimality, causal propagation or contamination freedom.', '',
      '## Possible errors and checks performed', '',
      f"- Analytic score/Jacobian finite-difference maximum absolute error {checks['jacobian_max_abs']:.3g}. Exact equivalence to V2 and merged Jacobian tested.",
      '- Tests cover the information formula/singular rejection, greedy selection/no replacement, null-vs-alternative test power, fixed-noise selection, immutable preregistration, strict bin-end matching, metadata allowlist, 2025 guard and clustered GLS algebra.',
      '- Graph asof dates are checked before event month; timestamp/years bounded to 2015-24. No intraday 2025 rows or returns were opened. Sign balance and available clean-window counts are explicit.',
      '- Ratio SEs are conservative bounds without covariance, not fabricated exact SEs; unavailable clustering/threshold/weekly SEs remain unavailable. Shared customers are not clustering.',
      '- Distinct observed host IDs, synthetic feature banks, independent outcome draws, articles and partner rows are recorded separately. Resampling never creates observed distinct cases.',
      '- Registration hash, protected-source hashes and cached input hashes checked after simulation. No label-loading helper is called. Metadata column projections exclude content labels and return outcomes. No package install, process termination, external write or network/LLM call.',
      f"- Peak process memory: {checks['peak_memory_bytes']/1024**2:.1f} MiB, below 2 GiB. Serial fits, one native BLAS thread. Existing unrelated changes preserved; .git untouched.",
      '- All fit diagnostics distinguish solver success, coordinate agreement and recovery. A relative-loss stop is not a projected-gradient proof. All S/A labels expose where no nonlinear simulation passed.', '',
      '- The original memory API returned zero because of an incorrect 64-bit handle signature. This was fixed, failure now raises, a positive-memory unit test was added, and the full intraday simulations were replayed under the working guard. '
      'An optional later host-snapshot memory replay stopped when the other job moved a pool file; the original copied host metadata was retained and all design results use that saved copy.', '',
      '## Files and reproduction', '',
      'New repository files: `source/code/news_network_50_2025/prompt_gd_design.py`, `examples/test_q1_design.py`, `docs/Q1_DESIGN_REPORT.md`, `docs/Q1_DESIGN_power.png`. Changes left uncommitted. '
      'Run with the supplied interpreter: `python -m source.code.news_network_50_2025.prompt_gd_design register`, then `... run`. '
      'Tests: `python -m unittest examples.test_q1_design`. All data artifacts under `D:/OptResearch_data/agent_q1/run_v2_design/`.', '',
      'Set `$env:Q1_SEAL_DATE="2025-11-03"` in PowerShell before running the study, and use `C:/OptResearch/.venv_dspy/Scripts/python.exe` for each command. '
      'Raw 2025 intraday inputs are never needed. Saved compact host snapshots allow replay even if the active hop-3 job moves its files.', '',
      f'Report copy: `{COPY}`.', '']
    budgets=read_json(Path(out)/'call_budgets.json')
    insertion=['### Selection benefit and actual article-call accounting','',
        'At MAE .10 and r=.20, expanded-random local n is 55,817, D3 is 24,910 (2.24 times fewer), and host-only selection is 39,831 (1.40 times fewer). '
        'D3+D4 local n is 24,765: merging barely improves the clustering bottleneck. Its earlier .20 grid pass is a coarse Monte Carlo result, not a robust 2.5-fold information gain.','',
        f"The full 12,000-candidate screen contains {budgets['expanded_screen_articles']:,} distinct article IDs. The cached 120 cases contain {budgets['cached_120_articles']:,}; reusing their fields costs zero new calls. "
        'An oracle D3 screen would need partner-specific article fields for the full candidate screen; host-only selection can score just the final selected articles. '
        'For future independent cases, article overlap is unknown, so eight articles per case is only a call upper bound. Repeated article IDs do not create fresh cases.','',
        '| Selected conditional w-only design, MAE .10 | Cases | Distinct selected article IDs | Full screening calls |','|---|---:|---:|---:|']
    for c in budgets['budgets']:
        if c['design']=='D1+D3_conditional' and c['target']==.1:
            insertion.append(f"| r={c['r']} | {c['n']:,} | {c['exact_selected_synthetic_bank_article_IDs']:,} | {c['screening_article_IDs']:,} |")
    insertion += ['','Exact source-ID replay and every daily-design budget are in call_budgets.json. '
        'D2/D5 projected fresh-case budgets have at most 8*n text calls, plus any candidate-screen calls; no exact future article count can be inferred.','',
        '### D2 power curves through n=5000','',
        'The curves show calculated conditional power. Dashed curves estimate w as a nuisance; solid curves fix w. '
        'A passing decision cell also requires at least eight simulated rejections in ten. No daily test meets both requirements by n=5000.','',
        '![Daily hop-test power](Q1_DESIGN_power.png)','']
    lines[lines.index('## D6: intraday article design and available clean events'):lines.index('## D6: intraday article design and available clean events')]=insertion
    d6extra=['### Archive-calibrated event budgets and clean-window feasibility','',
        '| Archive-signal design | Target | Local events | Article calls within current archive | Fits clean 517-event window? |',
        '|---|---:|---:|---:|---|']
    for c in intra['local']:
        if c.get('r_transfer')==1 and c['rho']==.25:
            if c['design'] in ('D6_reduced','D6_calibrated_w','D6_hop3_w','D6_hop3_test'):
                calls=str(c['calls_if_full_support']) if c['calls_if_full_support'] is not None else 'unavailable; needs fresh events'
                d6extra.append(f"| {c['design']} | {c.get('target','power .8')} | {c['n_approx']:,} | {calls} | {c['clean_window_feasible']} |")
    d6extra += ['',
        'The archive-calibrated shape requirement is about 44,077 events at MAE .10 and 11,020 at .20, exceeding the 8,468 matched-event archive. '
      f"The fixed-shape hop-3 test at the shipped .05 default needs about {next(x['n_approx'] for x in intra['local'] if x['design']=='D6_hop3_test' and x.get('r_transfer')==1 and x['rho']==.25):,} events, exceeding both the full matched archive and the clean 517-event window. "
        'A w-only pilot at precision .20 needs about 143 events under this envelope and fits the clean window; .10 needs about 570 and exceeds it. '
        'These are local approximations, with nuisance shape assumed correct.','',
        'The archived week-cluster SE implies an additional variance sensitivity: '+json.dumps(intra['archive_design_effect_sensitivity'])+'. '
        'Multiplying local sample sizes by this factor is a conservative calendar/source-dependence diagnostic. It is not an identified event correlation.','',
        f"Materiality transition share in the imputed article rows is {intra['imputed_materiality_transition_share']:.4f}; this is simulated, while the other three gate shares above use observed host fields. "
        'Hop weights and w have no transition gate.','',
        '| Common-r hop-3 test | Minimum tested events | Calculated power at that n | Simulated power |',
        '|---|---:|---:|---:|']
    for c in intra.get('hop3_power_simulation',[]):
        n=c['minimum_grid_n'];cell=next((x for x in c['cells'] if x['n']==n),None)
        if cell:d6extra.append(f"| {c['r']} | {n} | {cell['conditional_power']:.3f} | {cell['power_10']:.2f} |")
        else:d6extra.append(f"| {c['r']} | none through 5000 | NA | NA |")
    d6extra += ['', 'Intraday small simulations used n=50,500,5000 for the four-parameter model; optional six-parameter/w-only rows are information calculations only. '
        'The D6 power experiment and its .05-default correction were separately registered after the primary fits. '
        'The common-r .20 shape design needs 5,000 events at precision .10 (about 4,935 calls in the deterministic archive prefix), which exceeds the clean window. '
        'A .20 precision target passes at 500 events, but its r=.20 requires roughly six times lower residual noise SD than the raw archive. '
        'The earlier .50 hop-3 archive test required about 1,793 events; reducing the alternative to the shipped .05 multiplies that local budget by 100 at fixed noise.','']
    lines[lines.index('## Limitations'):lines.index('## Limitations')]=d6extra
    REPORT.write_text('\n'.join(lines),encoding='utf-8');COPY.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(REPORT,COPY)
    (Path(out)/'SUMMARY_TABLE.md').write_text(table+'\n',encoding='utf-8')
    print(table,flush=True);print('Report:',REPORT,flush=True);print('Report copy:',COPY,flush=True)


def run(out=OUT):
    from threadpoolctl import threadpool_limits
    out=Path(out);started=time.perf_counter();check_registration(out)
    freeze={'module':sha(__file__),'tests':sha(REPO/'examples/test_q1_design.py'),
            'protocol':sha(out/'PROTOCOL_DESIGN.json')}
    fp=out/'IMPLEMENTATION_FREEZE.json'
    if fp.exists():
        if read_json(fp)!=freeze:
            revision=out/('IMPLEMENTATION_REVISION_'+freeze['module'][:12]+'.json')
            if not revision.exists():save(revision,{'at_utc':datetime.now(timezone.utc).isoformat(),
                'reason':'Registered supplement, pandas graph-date bug fix, balanced-WLS sandwich and report corrections; original daily fits retained',
                'previous':read_json(fp),'current':freeze},exclusive=True)
    else:save(fp,freeze,exclusive=True)
    with threadpool_limits(limits=1):
        cases,features=v2.old.load_evidence()
        original=v2.old.Evidence([features[c['case_id']] for c in cases])
        diag,reduced=v2.diagnose(original)
        maps=reduced.maps;base=Bank(original,maps)
        cal=calibration(out);templates=snapshot_templates(out,cases)
        expanded,ti=candidates(templates,features,maps,out)
        checks=numerical_checks(base)
        # All simulation work is behind both registration and code freeze.
        if (out/'daily_results.json').exists():
            daily=read_json(out/'daily_results.json');daily['powers']=decision_experiment(base,expanded,out)
            save(out/'daily_results.json',daily)
        else:daily=daily_experiment(base,expanded,ti,cal,out)
        stress=read_json(out/'miscalibration.json') if (out/'miscalibration.json').exists() else miscalibration(base,cal,out)
        call_budgets(templates,features,daily,out)
        power_figure(daily,out)
        del expanded;gc.collect()
        frame,counts=intraday_counts(maps,out)
        intra=intraday_experiment(frame,maps,features,cal,out)
        check_registration(out)
        measured=peak_memory()
        previous=read_json(out/'checks.json') if (out/'checks.json').exists() else {}
        checks.update(peak_memory_bytes=max(measured,previous.get('peak_memory_bytes',0)),
                      peak_memory_this_run_bytes=measured,elapsed_seconds=time.perf_counter()-started)
        old_audit=read_json(out/'metadata_read_audit.json') if (out/'metadata_read_audit.json').exists() else []
        save(out/'checks.json',checks);save(out/'metadata_read_audit.json',old_audit+READS)
        report(daily,intra,counts,cal,stress,checks,out)
    return daily


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('register','run'))
    p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    if a.command=='register':
        value=register(a.out);print(a.out/'PROTOCOL_DESIGN.json');print(value['registered_at_utc'])
    else:run(a.out)


if __name__=='__main__':main()
