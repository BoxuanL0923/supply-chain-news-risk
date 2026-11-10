"""TD3 runtime contract: memory cap, offline/online audit hook, exclusive private writes, keys and redaction.
v5 (TD3C4 N-e): the D: free-space guard uses td3_spec.MIN_FREE_BYTES (25 GiB).

Nothing runs on import. No key value is ever printed, logged or written: keys live only in memory, every
record and console line passes through Redactor.scrub, and request headers are never stored.
"""
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from td3_spec import OUT, ALLOWED_HOSTS, JOB_MEMORY_CAP, MIN_FREE_BYTES, DATE_EXE

_state = {'mode': None, 'job': None, 'roots': None}
_DELETE_EVENTS = ('os.remove', 'os.rmdir', 'os.rename', 'os.replace', 'os.truncate', 'shutil.rmtree',
                  'shutil.move', 'os.link', 'os.symlink', 'os.chmod')
_EXEC_EVENTS = ('os.system', 'os.startfile', 'os.exec', 'os.spawn', 'os.posix_spawn', 'os.fork', 'os.forkpty')
_RESOLVE_EVENTS = ('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyname_ex')


def mode():
    return _state['mode']


def guard():
    """v5 (TD3C4 N-e): refuse below td3_spec.MIN_FREE_BYTES (25 GiB, the project's standing D: floor)."""
    free = shutil.disk_usage('D:/').free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f'STOP: D: free {free} bytes below {MIN_FREE_BYTES // 1024 ** 3} GiB; delete nothing')
    return free


def host_allowed(host):
    if isinstance(host, bytes):
        host = host.decode('ascii', 'replace')
    return isinstance(host, str) and host.lower().rstrip('.') in ALLOWED_HOSTS


def _within(path, roots):
    try:
        resolved = Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError, OSError):
        return False
    return any(resolved == r or resolved.is_relative_to(r) for r in roots)


def _date_argv(argv):
    """True only for `date.exe -u ...` (Windows passes the command line as one string to the audit event)."""
    if isinstance(argv, (str, bytes)):
        text = os.fsdecode(argv)
        m = re.match(r'^\s*"([^"]+)"\s+(\S+)', text) or re.match(r'^\s*(\S+)\s+(\S+)', text)
        if not m:
            return False
        exe, first = m.group(1), m.group(2)
    elif isinstance(argv, (list, tuple)) and len(argv) >= 2:
        exe, first = os.fsdecode(argv[0]), os.fsdecode(argv[1])
    else:
        return False
    return Path(exe).as_posix().lower() == DATE_EXE.lower() and first == '-u'


def _job_limit():
    from ctypes import wintypes as w

    class BASIC(ctypes.Structure):
        _fields_ = [('a', ctypes.c_int64), ('b', ctypes.c_int64), ('flags', w.DWORD), ('min', ctypes.c_size_t),
                    ('max', ctypes.c_size_t), ('active', w.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', w.DWORD), ('scheduling', w.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(x, ctypes.c_uint64) for x in ['r', 'w', 'o', 'rb', 'wb', 'ob']]

    class EXT(ctypes.Structure):
        _fields_ = [('basic', BASIC), ('io', IO), ('process', ctypes.c_size_t), ('job', ctypes.c_size_t),
                    ('peakprocess', ctypes.c_size_t), ('peakjob', ctypes.c_size_t)]
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateJobObjectW.restype = w.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    k.GetCurrentProcess.restype = w.HANDLE
    job = k.CreateJobObjectW(None, None)
    e = EXT()
    e.basic.flags = 0x100 | 0x200  # process and job memory limits
    e.process = JOB_MEMORY_CAP
    e.job = JOB_MEMORY_CAP
    if not job or not k.SetInformationJobObject(job, 9, ctypes.byref(e), ctypes.sizeof(e)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def install(run_mode, write_roots=(OUT,)):
    """Install once per process. offline: every socket event is refused. online: only DNS resolution of the two
    provider hosts is allowed (urllib resolves before connecting; no proxy handler is used)."""
    if _state['mode'] is not None:
        raise RuntimeError('TD3 runtime already installed')
    if run_mode not in ('offline', 'online'):
        raise ValueError('mode must be offline or online')
    if os.name != 'nt':
        raise RuntimeError('Windows memory job required')
    guard()
    sys.dont_write_bytecode = True
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    _state['job'] = _job_limit()
    roots = tuple(Path(r).resolve() for r in write_roots)

    def audit(event, args):
        if event.startswith('socket.'):
            if run_mode == 'offline':
                raise PermissionError('TD3 offline contract: network disabled')
            if event in _RESOLVE_EVENTS and not host_allowed(args[0]):
                raise PermissionError('TD3 online contract: host not allowed')
            if event in ('socket.gethostbyaddr', 'socket.sethostname', 'socket.bind'):
                raise PermissionError('TD3 online contract: socket operation not allowed')
            return
        if event in _EXEC_EVENTS:
            raise PermissionError('TD3: process execution not allowed')
        if event == 'subprocess.Popen':
            if not _date_argv(args[1]):
                raise PermissionError('TD3: only the date -u subprocess is permitted')
            return
        if event in _DELETE_EVENTS:
            raise PermissionError('TD3: delete, rename, truncate and overwrite are forbidden')
        if event == 'os.mkdir':
            if not _within(args[0], roots):
                raise PermissionError('TD3: directories only under the TD3 private root')
            return
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            fmode, flags = args[1], args[2] or 0
            writing = (isinstance(fmode, str) and any(c in fmode for c in 'wax+')) or \
                bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            if writing:
                if not _within(args[0], roots):
                    raise PermissionError('TD3: writes only under the TD3 private root')
                if not flags & os.O_EXCL:
                    raise PermissionError('TD3: exclusive creation required; no overwrite or append')
    sys.addaudithook(audit)
    _state['mode'] = run_mode
    _state['roots'] = roots


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


_UTC = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def utc():
    """UTC timestamp from `date -u` (project rule: run date -u immediately before writing a timestamp)."""
    out = subprocess.check_output([DATE_EXE, '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
    if not _UTC.match(out):
        raise RuntimeError('Unexpected date -u output')
    return out


def peak():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (x, ctypes.c_size_t) for x in ['peak', 'working', 'qp', 'p', 'qn', 'n', 'page', 'peakpage']]
    p = PMC()
    p.cb = ctypes.sizeof(p)
    k = ctypes.WinDLL('kernel32')
    k.GetCurrentProcess.restype = ctypes.c_void_p
    fn = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not fn(k.GetCurrentProcess(), ctypes.byref(p), p.cb):
        raise ctypes.WinError()
    return {'peak_working_set_bytes': p.peak, 'peak_private_bytes': p.peakpage}


class Redactor:
    """Holds secret values in memory only; replaces any occurrence before text leaves the process."""

    def __init__(self):
        self._secrets = []
        self.count = 0

    def add(self, value):
        if isinstance(value, str) and len(value) >= 8 and value not in self._secrets:
            self._secrets.append(value)

    def scrub(self, text):
        for s in self._secrets:
            if s in text:
                self.count += text.count(s)
                text = text.replace(s, '[REDACTED]')
        return text


def load_key(var):
    """Process environment first, then the Windows User scope (HKCU\\Environment). Never printed."""
    value = os.environ.get(var)
    if not value and os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                value, _ = winreg.QueryValueEx(k, var)
        except OSError:
            value = None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f'{var} is not set at process or Windows User scope')
    return value.strip()


def _private(path):
    path = Path(path).resolve()
    roots = _state['roots'] or (OUT.resolve(),)
    if not any(path == r or path.is_relative_to(r) for r in roots):
        raise ValueError('Output outside the TD3 private root')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dumps(value):
    return json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False, sort_keys=False, default=str)


def new_json(path, value, redactor=None, timed=True):
    guard()
    path = _private(path)
    if timed:
        value = {**value, 'written_utc': utc()}
    text = dumps(value)
    if redactor is not None:
        text = redactor.scrub(text)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return path


def new_text(path, text, redactor=None):
    guard()
    path = _private(path)
    if redactor is not None:
        text = redactor.scrub(text)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return path


def new_bytes_handle(path):
    guard()
    return _private(path).open('xb')


class JsonlWriter:
    """Exclusive-create JSONL; each line is scrubbed, flushed and fsynced before the next call."""

    def __init__(self, path, redactor):
        guard()
        self.path = _private(path)
        self.redactor = redactor
        self._f = self.path.open('x', encoding='utf-8', newline='\n')

    def write(self, record):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, default=str)
        line = self.redactor.scrub(line)
        self._f.write(line + '\n')
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        if not self._f.closed:
            self._f.close()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class RunLock:
    """R3: exclusive real-run lock. The lock file is created once (exclusive create, never deleted) and an OS
    byte-range lock (msvcrt.locking, non-blocking) is held on it for the life of the run; the OS releases it when the
    process ends, so a crash leaves no stale lock. A second run or resume refuses while the lock is held."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = None

    def acquire(self):
        import msvcrt
        guard()
        target = _private(self.path)
        try:
            with target.open('x', encoding='utf-8', newline='\n') as f:
                f.write('TD3 real-run lock file (byte-range locked while a run is active; never deleted)\n')
        except FileExistsError:
            pass
        self._f = target.open('rb')
        try:
            msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self._f.close()
            self._f = None
            raise PermissionError('STOP: another TD3 real run or resume holds the run lock') from None
        return self

    def release(self):
        if self._f is not None:
            import msvcrt
            try:
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._f.close()
            self._f = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def say(redactor, *parts):
    """Console output passes through the redactor."""
    text = ' '.join(str(p) for p in parts)
    print(redactor.scrub(text) if redactor is not None else text, flush=True)
