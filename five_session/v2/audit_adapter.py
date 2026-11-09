"""Redirect the preserved independent traps to append-only private fixtures."""
from pathlib import Path
import uuid
import runtime
from runtime import sha as hash_file
REVIEW = runtime.OUT/'repair_v2'/('traps_'+uuid.uuid4().hex)
REVIEW.mkdir(parents=True)
def write_json(name, value):
    runtime.new_json(REVIEW/name, value)
