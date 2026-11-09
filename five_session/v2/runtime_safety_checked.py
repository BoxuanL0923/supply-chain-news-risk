"""Task-local offline guard. Install after the Windows memory cap, before science."""
import os
from pathlib import Path
import platform
import socket
import sys
from runtime import OUT, guard

READ_EVENTS = set()
platform._uname_cache = platform.uname_result('Windows', 'td1b-offline', '11', '10.0', 'AMD64')
socket.gethostname = lambda: 'td1b-offline'
sys.dont_write_bytecode = True

def audit_hook(event, args):
    if event.startswith('socket.') or event in ('os.remove', 'os.rmdir', 'os.rename'):
        raise RuntimeError('TD1B forbids network/deletion/renaming: ' + event)
    if event == 'subprocess.Popen':
        argv = args[1]
        import subprocess
        expected = ['C:/Program Files/Git/usr/bin/date.exe', '-u', '+%Y-%m-%dT%H:%M:%SZ']
        # Windows audits the serialized command line; POSIX audits argv.
        if argv != expected and argv != subprocess.list2cmdline(expected):
            raise RuntimeError('Only date -u is permitted as a child process')
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    name = str(path).replace('\\', '/').lower()
    if any(s in name for s in ('/verified_security_v2/', '/idsens2/', '/ids2/', 'wrds.lock')):
        raise RuntimeError('Protected path: ' + str(path))
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing:
        guard()
        if not path.is_relative_to(OUT.resolve()):
            raise RuntimeError('Runtime writes must stay under the private root: ' + str(path))
        if path.exists():
            raise FileExistsError('Preserve existing file: ' + str(path))
    if path.is_relative_to(OUT.resolve()):
        if '2025' in path.name or '/evaluation/' in name or '/evaluation_v2/' in name:
            raise RuntimeError('Forbidden data: ' + str(path))
        READ_EVENTS.add(str(path))
    if name.endswith(('.parquet', '.csv')) and not path.is_relative_to(OUT.resolve()):
        raise RuntimeError('Only bounded private snapshots may be opened: ' + str(path))

sys.addaudithook(audit_hook)

def seal_labels():
    import core
    original = core.label
    def development_only(dates, returns, pos, allowed_end):
        if str(dates[pos])[:4] >= '2023':
            raise AssertionError('TD1B cannot compute any evaluation label')
        return original(dates, returns, pos, min(allowed_end, '2022-12-31'))
    core.label = development_only
