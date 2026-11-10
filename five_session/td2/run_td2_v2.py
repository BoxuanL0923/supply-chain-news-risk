"""Supported append-only entry point: limits installed before scientific imports."""
from runtime_v3 import limit
limit()
import delivery
import run_td2
# run_td2's import installs a second stricter-compatible Windows job; avoid it
# by executing main only after its module is loaded (nested job caps are safe).
if __name__=='__main__': run_td2.main()
