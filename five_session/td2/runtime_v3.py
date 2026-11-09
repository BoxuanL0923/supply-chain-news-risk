"""Offline, append-only TD2 resource contract. No work happens on import."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
OUT = Path('D:/OptResearch_data/five_session/td2')
TD1 = Path('D:/OptResearch_data/five_session')
CAP = 2_400_000_000
MIN_FREE = 40 * 1024**3
_job = None

def guard():
    free = shutil.disk_usage('D:/').free
    if free < MIN_FREE:
        raise RuntimeError(f'STOP: D: free {free} bytes below 40 GiB; delete nothing')
    return free

def limit():
    global _job
    guard()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                      TMP=str(OUT), TEMP=str(OUT))
    if os.name != 'nt':
        raise RuntimeError('Windows memory job required')
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
    def audit(event, args):
        if event.startswith('socket.') or event in ('os.system', 'os.startfile'):
            raise PermissionError('TD2 offline contract')
        if event == 'subprocess.Popen':
            executable, argv = args[:2]
            if 'c:/program files/git/usr/bin/date.exe' not in str(argv).lower() or '-u' not in str(argv):
                raise PermissionError('Only date -u subprocess is permitted')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            text = str(args[0]).lower().replace('\\','/')
            if any(x in text for x in ('verified_security_v2', '/idsens2/', '/ids2/', '/.git/', 'wrds.lock')):
                raise PermissionError('Protected path')
            mode, flags = args[1:3]
            writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if writing:
                resolved = Path(args[0]).resolve()
                if not resolved.is_relative_to(OUT.resolve()):
                    raise PermissionError('Runtime writes must stay in TD2 private output root')
                if not (flags & os.O_EXCL):
                    raise PermissionError('Exclusive creation required; no overwrite')
    sys.addaudithook(audit)

def sha(path):
    guard(); h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()

def utc():
    return subprocess.check_output(['C:/Program Files/Git/usr/bin/date.exe', '-u',
                                    '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def new_json(path, value, timed=True):
    guard(); path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Output outside TD2 private root')
    path.parent.mkdir(parents=True, exist_ok=True)
    if timed:
        value = {**value, 'written_utc': utc()}
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def peak():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (x, ctypes.c_size_t) for x in ['peak','working','qp','p','qn','n','page','peakpage']]
    p=PMC(); p.cb=ctypes.sizeof(p)
    k=ctypes.WinDLL('kernel32'); k.GetCurrentProcess.restype=ctypes.c_void_p
    fn=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not fn(k.GetCurrentProcess(),ctypes.byref(p),p.cb):
        raise ctypes.WinError()
    return {'peak_working_set_bytes':p.peak,'peak_private_bytes':p.peakpage}
