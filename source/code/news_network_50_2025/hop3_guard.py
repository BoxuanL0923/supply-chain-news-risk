"""qwen3.8-27b development-only filesystem, case, resource and label gates."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

MODEL = 'qwen3.8-27b'
# Local data and run artifacts are operator inputs, never repository content.
PROJECT = Path(os.environ.get('HOP3_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
LIVE = Path(os.environ.get('HOP3_LIVE_ROOT', PROJECT/'results/news_network_50_2025')).resolve()
EXPOSURE = Path(os.environ.get('HOP3_EXPOSURE_ROOT', PROJECT/'results/news_exposure_2025')).resolve()
REUTERS_INDEX = Path(os.environ.get('HOP3_REUTERS_INDEX_ROOT', PROJECT/'results/news_event_risk_2025/reuters_index')).resolve()
ROOT = Path(os.environ.get('HOP3_RUN_ROOT', PROJECT/'results/hop3_run')).resolve()
REPORT = Path(os.environ.get('HOP3_REPORT_PATH', ROOT/'reports/HOP3_ABLATION_REPORT.md')).resolve()
FACTSET = Path(os.environ.get('HOP3_FACTSET_ROOT', PROJECT/'data/factset')).resolve()
DATES = ('2025-10-01','2025-10-08','2025-10-15','2025-10-22','2025-10-30','2025-11-06')
SPLITS = dict(zip(DATES, ['train']*4+['valid']*2))
SEAL = '2025-11-14'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'model': MODEL, **value}
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def check_case(c):
    if c.get('date') not in SPLITS or c.get('split') != SPLITS[c['date']]:
        raise ValueError('Evaluation case or unexpected development date forbidden')
    if c.get('asof','')[:10] != c['date']:
        raise ValueError('Case clock mismatch')
    future = c.get('next_sessions', [])
    if len(future) != 5 or future != sorted(set(future)) or any(not c['date'] < d < SEAL for d in future):
        raise ValueError('Evaluation target window forbidden')
    if set(c) & {'y','label','labels','outcome','residual'}:
        raise ValueError('Outcome in prediction case')
    return c

class Guard:
    """Python audit gate, including Arrow opens via explicit guarded loader.

    Only the six dated live pools may supply cases. Mixed cases/controls/calendar
    artifacts are blocked. Native Arrow is called only after filename + freeze
    checks because Python audit hooks do not intercept Arrow C++ file I/O.
    """
    def __init__(self):
        self.labels_allowed = False
        self.opened = set()
    def check_path(self, path, writing=False):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        p = str(Path(os.fsdecode(path)).resolve()).replace('\\','/').lower()
        name = p.rsplit('/',1)[-1]
        def under(root):
            prefix = str(root.resolve()).replace('\\','/').lower().rstrip('/')
            return p == prefix or p.startswith(prefix+'/')
        live = under(LIVE) or '/results/news_network_50_2025/' in p
        exposure = under(EXPOSURE) or '/results/news_exposure_2025/' in p
        if 'labels_test' in name or 'labels_sealed' in name or 'matched_test' in name or 'labels_evaluation' in name:
            raise PermissionError('Evaluation/mixed labels forbidden: '+p)
        if live:
            if writing:
                raise PermissionError('Live cache is read only')
            if '/pools/' in p and name.removesuffix('.json') not in DATES:
                raise PermissionError('Evaluation pool forbidden')
            if name in {'cases.json','controls.parquet','calendar_features.json','anchor.json', 'summary.json','matched_validation.csv'}:
                raise PermissionError('Mixed/live outcome-derived artifact forbidden: '+p)
        if exposure:
            if writing:raise PermissionError('Live exposure cache is read only')
            if name in {'cases.json','controls.parquet','calendar_features.json','anchor.json','summary.json'}:
                raise PermissionError('Mixed exposure cases/outcome-derived artifact forbidden: '+p)
        if (under(PROJECT/'code') or under(Path(__file__).resolve().parents[1])) and writing:
            raise PermissionError('Live code is read only')
        if name.startswith('labels_') and not self.labels_allowed:
            raise PermissionError('Labels forbidden before complete durable prediction freeze')
        if name.startswith('labels_') and name not in {'labels_train.parquet','labels_valid.parquet'}:
            raise PermissionError('Unexpected label artifact')
        if live:
            self.opened.add(p)
    def install(self):
        def hook(event, args):
            if event == 'open':
                mode = args[1]
                flags = args[2] or 0
                writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
                self.check_path(args[0], writing)
            if event == 'socket.connect':
                address = args[1]
                if not isinstance(address,tuple) or address[:2] != ('127.0.0.1',8081):
                    raise PermissionError('Only authorized loopback model endpoint permitted')
        sys.addaudithook(hook)

def memory():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    m = MEMORYSTATUSEX(); m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('Memory status unavailable')
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p=PMC(); p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb):
        raise OSError('Process memory information unavailable')
    return {'free_commit_gb':m.avail_page/2**30,'rss_gb':p.WorkingSetSize/2**30,'peak_rss_gb':p.PeakWorkingSetSize/2**30}

def resources(stage):
    while True:
        m=memory(); print(MODEL,stage,m,flush=True)
        if m['peak_rss_gb'] >= 3:
            raise MemoryError('Process RAM exceeded 3 GB')
        if m['free_commit_gb'] >= 5:
            return m
        time.sleep(60)

def verify_freeze(out, cases):
    if (len(cases)!=300 or len({c['case_id'] for c in cases})!=300
            or {s:sum(c['split']==s for c in cases) for s in ('train','valid')}!={'train':200,'valid':100}):
        raise ValueError('A full 200-train/100-valid freeze is required before label access')
    for c in cases:check_case(c)
    protocol=read(out/'PROTOCOL.json')
    if protocol['cases_hash']!=digest(cases):raise ValueError('Cases differ from frozen protocol')
    frozen=read(out/'prediction_freeze.json')
    seal=read(out/'prediction_freeze.sha256.json')
    if sha(out/'prediction_freeze.json') != seal['sha256'] or frozen['cases_hash'] != digest(cases):
        raise ValueError('Prediction freeze mismatch')
    if frozen['protocol_sha256']!=sha(out/'PROTOCOL.json'):
        raise ValueError('Prediction protocol changed')
    expected={(c['case_id'],mode,arm) for c in cases for mode in ('all_paths','continuity_only')
              for arm in ('OWN','H1','H2','H3','H3_GRAPH')}
    actual={(r['case_id'],r['mode'],r['arm']) for r in frozen['predictions']}
    if actual != expected or len(actual) != len(frozen['predictions']):
        raise ValueError('Incomplete or duplicate prediction freeze')
    for item in frozen['predictions']:
        if sha(out/'predictions'/item['file']) != item['sha256']:
            raise ValueError('Frozen prediction bytes changed')
    return frozen

def load_labels(guard, out, cases, split):
    if split not in {'train','valid'}:
        raise ValueError('Evaluation labels forbidden')
    for c in cases: check_case(c)
    verify_freeze(out,cases)
    guard.labels_allowed=True
    path=LIVE/f'labels_{split}.parquet'
    guard.check_path(path)
    label_manifest=read(out.parent/'label_source_manifest.json')
    if sha(path)!=label_manifest['expected_sha256'][path.name]:
        raise ValueError('Development label bytes changed from the pre-model source manifest')
    import pandas as pd
    frame=pd.read_parquet(path)
    allowed={c['case_id'] for c in cases if c['split']==split}
    if set(frame.case_id) != allowed or frame.case_id.duplicated().any() or set(frame['split']) != {split}:
        raise ValueError('Unexpected label cases')
    if any(str(d)[:10] not in SPLITS for d in frame.date):
        raise ValueError('Evaluation label date forbidden')
    return frame[['case_id','y']]
