"""TD4 process environment: import paths, runtime contract, memory pause, exclusive private writes.

The TD3 v6 runtime (td3_runtime.install) is reused unchanged with the TD4 private root as its only write root and a
lower job memory cap (td3_runtime.JOB_MEMORY_CAP is a module constant read when the job is created). TD1 v2 modules
are loaded under private names with a temporary `spec` shim, because TD1 and TD2 both ship a module called spec.py.
Nothing runs on import.
"""
import ctypes
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import td4_spec as T

_PATHS_DONE = [False]
_TD1 = {}


def paths():
    """TD4 first, then TD3 v6, TD2 appended last (TD3's own convention: stdlib and TD3 names win)."""
    if _PATHS_DONE[0]:
        return
    sys.dont_write_bytecode = True
    for p in (str(T.CODE), str(T.TD3_CODE)):
        if p not in sys.path:
            sys.path.insert(0 if p == str(T.CODE) else 1, p)
    if str(T.TD2_CODE) not in sys.path:
        sys.path.append(str(T.TD2_CODE))
    _PATHS_DONE[0] = True


def install(mode='offline'):
    """Install the TD3 runtime contract (job memory cap, audit hook, exclusive writes under the TD4 root only)."""
    paths()
    import td3_runtime as R
    R.JOB_MEMORY_CAP = T.JOB_MEMORY_CAP
    T.OUT.mkdir(parents=True, exist_ok=True)
    R.install(mode, write_roots=(T.OUT,))
    guard()
    return R


def guard():
    """D: >= 25 GiB and C: >= 15 GiB free; waits while available physical memory is below 1,500 MB."""
    d = shutil.disk_usage('D:/').free
    c = shutil.disk_usage('C:/').free
    if d < T.MIN_FREE_D_BYTES:
        raise RuntimeError(f'STOP: D: free {d} bytes below 25 GiB; delete nothing')
    if c < T.MIN_FREE_C_BYTES:
        raise RuntimeError(f'STOP: C: free {c} bytes below 15 GiB; delete nothing')
    wait_memory()
    return d


def available_mb():
    class MEMSTAT(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024 ** 2


def wait_memory(log=None, max_wait_s=6 * 3600):
    """Pause (sleep in 30 s steps) while available physical memory is below MIN_AVAILABLE_MB."""
    waited = 0
    while available_mb() < T.MIN_AVAILABLE_MB:
        if log:
            log(f'MEMORY_PAUSE available {available_mb():.0f} MB < {T.MIN_AVAILABLE_MB}; waiting')
        time.sleep(30)
        waited += 30
        if waited > max_wait_s:
            raise RuntimeError('STOP: available memory stayed below the floor for 6 hours')
    return waited


def td1(name):
    """Load a TD1 v2 module (spec, core) under the name td1_<name>; core's `from spec import` gets TD1's spec."""
    if name in _TD1:
        return _TD1[name]
    if name not in ('spec', 'core'):
        raise ValueError('Only TD1 spec and core are loaded by TD4')
    path = T.TD1_CODE / f'{name}.py'
    saved = sys.modules.get('spec')
    try:
        if name == 'core':
            sys.modules['spec'] = td1('spec')
        mod_spec = importlib.util.spec_from_file_location('td1_' + name, path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    finally:
        if name == 'core':
            if saved is None:
                sys.modules.pop('spec', None)
            else:
                sys.modules['spec'] = saved
    _TD1[name] = mod
    return mod


def td2_statistics():
    """TD2 statistics.py (paired, clustered_mean) under the name td2_statistics (stdlib 'statistics' would win)."""
    if 'statistics' in _TD1:
        return _TD1['statistics']
    mod_spec = importlib.util.spec_from_file_location('td2_statistics', T.TD2_CODE / 'statistics.py')
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    _TD1['statistics'] = mod
    return mod


def private(path):
    path = Path(path).resolve()
    if not path.is_relative_to(T.OUT.resolve()):
        raise ValueError('TD4 outputs must stay under the TD4 private root: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(path, frame):
    """Exclusive-create parquet (zstd) under the private root; never overwrites."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    guard()
    path = private(path)
    table = frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame, preserve_index=False)
    with path.open('xb') as f:
        pq.write_table(table, f, compression='zstd')
    return path


def log_line(path, text):
    """Append-free log: one exclusive file per call site is not practical for long runs, so long runs print to
    stdout (redirected by the caller) and write JSON receipts with new_json."""
    print(text, flush=True)
