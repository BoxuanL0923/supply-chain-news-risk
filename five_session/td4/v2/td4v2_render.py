"""TD4 v2.2 ledger rendering: any redaction round is rebuilt from the masker-v2 text and the cumulative ledger.

render(v2_text, entries) applies ledger entries (offset, end, placeholder) in masker-v2 coordinates. Rounds 1 and 2 of
v2.2 are stored as ledgers only (LEDGER.jsonl and hashes); the final round also stores the full INPUTS.jsonl text.
open_round() gives a file-like reader of the rendered JSONL records (bytes or text lines, record order and fields of
the masker-v2 INPUTS, input_text / input_sha256 / input_bytes of the rendered text), so the reviewers' detector copies
can read a ledger-only round without a stored text file (their INPUTS open call is the only line changed besides
paths). Reviewers can rebuild any round with render(). Reads no label, return or provider output.
"""
import hashlib
import json
from pathlib import Path


def render(v2_text, entries):
    """The redacted text: masker-v2 text with the placeholders of the (non-overlapping) entries (offset, end,
    placeholder, ...) put at their masker-v2 offsets."""
    out, i = [], 0
    for e in sorted(entries, key=lambda x: (x[0], x[1])):
        s, t, ph = e[0], e[1], e[2]
        if s < i:
            raise ValueError('overlapping ledger entries')
        out.append(v2_text[i:s])
        out.append(ph)
        i = t
    out.append(v2_text[i:])
    return ''.join(out)


def _jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ledger_index(ledgers):
    """key -> sorted [(offset, end, placeholder)] over the given ledger files (cumulative rounds)."""
    idx = {}
    for p in ledgers:
        for e in _jsonl(p):
            idx.setdefault(e['key'], []).append((e['offset'], e['offset'] + e['length'], e['placeholder']))
    for k in idx:
        idx[k].sort()
    return idx


def render_record(rec, entries, round_no=None):
    """A masker-v2 INPUTS record with the rendered text and its hash fields (the masker-v2 hash kept as
    v2_input_sha256)."""
    r = dict(rec)
    text = render(rec['input_text'], entries) if entries else rec['input_text']
    r['v2_input_sha256'] = rec['input_sha256']
    r['input_text'] = text
    r['input_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
    r['input_bytes'] = len(text.encode('utf-8'))
    if round_no is not None:
        r['redaction_round'] = round_no
    return r


def iter_round(v2_inputs, ledgers, round_no=None):
    """Rendered records of a round, in masker-v2 INPUTS order."""
    idx = ledger_index(ledgers)
    with Path(v2_inputs).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield render_record(rec, idx.get(rec['key'], []), round_no)


class open_round:
    """File-like reader of a rendered round: iterate lines (bytes for mode 'rb', str for 'r'); context manager."""

    def __init__(self, v2_inputs, ledgers, mode='rb', round_no=None):
        self.v2_inputs, self.ledgers, self.mode, self.round_no = v2_inputs, list(ledgers), mode, round_no
        self._it = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._it = None
        return False

    def close(self):
        self._it = None

    def __iter__(self):
        for r in iter_round(self.v2_inputs, self.ledgers, self.round_no):
            s = json.dumps(r, ensure_ascii=False) + '\n'
            yield s.encode('utf-8') if 'b' in self.mode else s

    def readline(self):
        if self._it is None:
            self._it = iter(self)
        return next(self._it, b'' if 'b' in self.mode else '')
