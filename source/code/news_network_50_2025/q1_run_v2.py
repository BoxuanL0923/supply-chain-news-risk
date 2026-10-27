"""Sequential background runner, v2 (2026-10-26): full development panel (80 train / 40 valid), seal 2025-11-03.

Never kills children or manages llama-server. One Python worker and one local
model request at a time. Stops the pipeline on a nonzero child exit.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
from .prompt_gd import MODEL, write


def memory_mb(pid):
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
            ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle: return None
    try:
        c = Counters(); c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb): return None
        return c.peak / 1024**2
    finally: kernel.CloseHandle(handle)


def main():
    root = Path("D:/OptResearch_data/agent_q1")
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "Q1_SEAL_DATE": "2025-11-03", "Q1_SPSA_TRAIN": "20", "Q1_SPSA_VALID": "40",
                       "Q1_SPSA_STARTS": "3", "Q1_SPSA_ALPHAS": "0.602", "Q1_SPSA_ITERS": "10",
                       "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONUNBUFFERED": "1"}
    stages = []
    data, run = root/"data_v2", root/"run_v2"
    io = ["--data", str(data), "--out", str(run)]
    if not (data/"development_manifest.json").exists():
        stages.append(("prepare", ["-m", "news_network_50_2025.q1_prepare_v2"]))
    stages += [("features", ["-m", "news_network_50_2025.prompt_gd", "cache-features", *io]),
               ("gd", ["-m", "news_network_50_2025.prompt_gd", "gd", *io]),
               ("spsa", ["-m", "news_network_50_2025.prompt_gd", "spsa", *io]),
               ("smoke", ["-m", "news_network_50_2025.hop_ablation", "--q1-validation-data", str(data), "--out", str(root/"smoke_v2")]),
               ("report", ["-m", "news_network_50_2025.prompt_gd", "convergence-report", "--out", str(run)])]
    results = []
    for name, args in stages:
        started = time.time(); peak = 0.
        print(MODEL, "START", name, flush=True)
        with (root/(name+"_v2.stdout.log")).open("a", encoding="utf-8") as log, (root/(name+"_v2.stderr.log")).open("a", encoding="utf-8") as err:
            child = subprocess.Popen([sys.executable, "-u", *args], env=env, stdout=log, stderr=err)
            while child.poll() is None:
                peak = max(peak, memory_mb(child.pid) or 0)
                write(root/"status_v2.json", {"model": MODEL, "stage": name, "pid": child.pid,
                      "elapsed_seconds": time.time()-started, "peak_working_set_mb": peak, "completed": results})
                time.sleep(2)
        result = {"stage": name, "exit_code": child.returncode, "seconds": time.time()-started, "peak_working_set_mb": peak}
        results.append(result)
        write(root/"run_status_v2.json", {"model": MODEL, "completed": results, "done": child.returncode != 0 or name == "report"})
        print(MODEL, "END", result, flush=True)
        if child.returncode: raise SystemExit(child.returncode)


if __name__ == "__main__": main()
