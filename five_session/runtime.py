"""Offline TD1 resource and append-only artifact rules (standard library only)."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

CODE = Path(__file__).resolve().parent
OUT = Path('D:/OptResearch_data/five_session')
CAP = 2_500_000_000  # below the task's decimal 2.6 GB ceiling
MIN_FREE = 40 * 1024**3
_job = None

def guard():
    free = shutil.disk_usage('D:/').free
    if free < MIN_FREE:
        raise RuntimeError(f'STOP: D: free bytes {free} below 40 GiB; delete nothing')
    return free

def limit():
    global _job
    guard()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if os.name != 'nt':
        raise RuntimeError('TD1 production runner requires its Windows memory job')
    from ctypes import wintypes as w
    class BASIC(ctypes.Structure):
        _fields_ = [('a', ctypes.c_int64), ('b', ctypes.c_int64), ('flags', w.DWORD),
                    ('min', ctypes.c_size_t), ('max', ctypes.c_size_t), ('active', w.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(x, ctypes.c_uint64) for x in ['r','w','o','rb','wb','ob']]
    class EXT(ctypes.Structure):
        _fields_ = [('basic', BASIC), ('io', IO), ('process', ctypes.c_size_t),
                    ('job', ctypes.c_size_t), ('peakprocess', ctypes.c_size_t), ('peakjob', ctypes.c_size_t)]
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateJobObjectW.restype = w.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    k.GetCurrentProcess.restype = w.HANDLE
    _job = k.CreateJobObjectW(None, None)
    e = EXT(); e.basic.flags = 0x100 | 0x200; e.process = CAP; e.job = CAP
    if not _job or not k.SetInformationJobObject(_job, 9, ctypes.byref(e), ctypes.sizeof(e)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(_job, k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
            guard()
    return h.hexdigest()

def utc():
    # The command is run immediately before obtaining any wall-clock time for writing.
    return subprocess.check_output(['C:/Program Files/Git/usr/bin/date.exe', '-u',
                                    '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def new_json(path, value):
    guard()
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Data/output must stay in TD1 output root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def peak_bytes():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (x, ctypes.c_size_t) for x in ['peak','working','qp','p','qn','n','page','peakpage']]
    p = PMC(); p.cb = ctypes.sizeof(p)
    k = ctypes.WinDLL('kernel32'); k.GetCurrentProcess.restype = ctypes.c_void_p
    fn = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not fn(k.GetCurrentProcess(), ctypes.byref(p), p.cb):
        raise ctypes.WinError()
    return {'peak_working_set_bytes': p.peak, 'peak_private_bytes': p.peakpage}

def freeze_path():
    amended=OUT/'MODEL_FREEZE_v2.json'
    return amended if amended.exists() else OUT/'MODEL_FREEZE.json'
