"""Cost ledger and hard caps. v2 (TD3B, C4): Jev is priced and sits under the same USD cap as DeepSeek. v3, v4: unchanged
except this docstring (prices re-confirmed on the official pages; deepseek-flash is the only priced DeepSeek model).

Before every attempt a 'reserve' row is written and fsynced with the attempt's pre-call USD upper bound: DeepSeek
request bytes x 0.5 tokens/byte + framing at the peak cache-miss price plus max_tokens output at the peak output
price; Jev request bytes x 0.5 tokens/byte + framing at USD 0.042 per million input tokens (output free), plus one
request against the Jev request cap. After the attempt a 'settle' row records tokens, returned model, latency and
the USD estimate from reported usage (the bound when usage is missing). Cumulative USD = settled estimates +
reserves never settled (crash-safe) + all earlier real runs, DeepSeek and Jev together. A call is refused when
cumulative USD + its upper bound would exceed MAX_USD, or when the Jev count would exceed MAX_JEV_REQUESTS.
"""
import datetime as dt
from pathlib import Path
import td3_spec as S
from td3_runtime import JsonlWriter, read_jsonl


def window(utc_iso):
    t = dt.datetime.strptime(utc_iso, '%Y-%m-%dT%H:%M:%SZ')
    if t.weekday() < 5 and any(a <= t.hour < b for a, b in S.PEAK_WINDOWS_UTC):
        return 'peak'
    return 'off_peak'


def deepseek_upper_usd(body):
    tokens_in = len(body) * S.UPPER_TOKENS_PER_BYTE + S.UPPER_FRAMING_TOKENS
    p = S.PRICES['peak']
    return (tokens_in * p['input_miss'] + S.DEEPSEEK_MAX_TOKENS * p['output']) / 1e6


def jev_upper_usd(body):
    tokens_in = len(body) * S.UPPER_TOKENS_PER_BYTE + S.UPPER_FRAMING_TOKENS
    return tokens_in * S.JEV_PRICE_INPUT_PER_M / 1e6


def jev_usd(usage):
    """USD from reported Jev usage (input tokens priced; output free); None when input tokens are missing."""
    if not isinstance(usage, dict) or 'input_tokens' not in usage:
        return None
    return (usage['input_tokens'] * S.JEV_PRICE_INPUT_PER_M + usage.get('output_tokens', 0) * S.JEV_PRICE_OUTPUT_PER_M) / 1e6


def upper_usd(provider, body):
    return deepseek_upper_usd(body) if provider == 'deepseek' else jev_upper_usd(body)


def deepseek_usd(usage, price_window):
    """USD from reported usage; None when prompt or completion tokens are missing."""
    if not isinstance(usage, dict) or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
        return None
    p = S.PRICES[price_window]
    hit = usage.get('prompt_cache_hit_tokens', 0) or 0
    miss = usage.get('prompt_cache_miss_tokens', usage['prompt_tokens'] - hit)
    return (miss * p['input_miss'] + hit * p['input_hit'] + usage['completion_tokens'] * p['output']) / 1e6


def summarize(rows):
    """(usd, jev_requests, attempts) from one ledger's rows; USD covers both providers. A v1 Jev reserve row has no
    usd_upper (Jev was unpriced in v1); it is then valued at its body-size bound."""
    reserved, settled, jev = {}, {}, 0
    for r in rows:
        if r['kind'] == 'reserve':
            reserved[r['call_seq']] = r
            if r['provider'] == 'jev':
                jev += 1
        elif r['kind'] == 'settle':
            settled[r['call_seq']] = r
    usd = 0.0
    for seq, r in reserved.items():
        s = settled.get(seq)
        if s is not None and s.get('usd_estimate') is not None:
            usd += s['usd_estimate']
        elif r.get('usd_upper') is not None:
            usd += r['usd_upper']
        else:
            usd += (r.get('body_bytes', 0) * S.UPPER_TOKENS_PER_BYTE + S.UPPER_FRAMING_TOKENS) * S.JEV_PRICE_INPUT_PER_M / 1e6
    return usd, jev, len(reserved)


def prior_spend(runs_root=S.RUNS):
    """Sum over every session ledger of every real run (dry runs and fixtures never count)."""
    usd, jev, ledgers = 0.0, 0, []
    root = Path(runs_root)
    if root.exists():
        for p in sorted(root.glob('real_*/session_*/ledger.jsonl')):
            u, j, _ = summarize(read_jsonl(p))
            usd, jev = usd + u, jev + j
            ledgers.append(str(p))
    return usd, jev, ledgers


class CapStop(Exception):
    pass


class Ledger:
    def __init__(self, path, redactor, max_usd=S.MAX_USD, max_jev=S.MAX_JEV_REQUESTS, prior_usd=0.0, prior_jev=0):
        self.max_usd, self.max_jev = max_usd, max_jev
        self.prior_usd, self.prior_jev = prior_usd, prior_jev
        self.rows = []
        self.seq = 0
        self.writer = JsonlWriter(path, redactor)

    def usd(self):
        return self.prior_usd + summarize(self.rows)[0]

    def jev(self):
        return self.prior_jev + summarize(self.rows)[1]

    def reserve(self, provider, selector, unit, body):
        upper = upper_usd(provider, body)
        if provider == 'jev' and self.jev() + 1 > self.max_jev:
            raise CapStop(f'cap_jev_requests: {self.jev()} + 1 > {self.max_jev}')
        if self.usd() + upper > self.max_usd:
            raise CapStop(f'cap_usd: cumulative {self.usd():.6f} + upper bound {upper:.6f} > {self.max_usd:.2f}')
        self.seq += 1
        row = {'kind': 'reserve', 'call_seq': self.seq, 'provider': provider, 'selector': selector, 'unit': unit,
               'usd_upper': upper, 'body_bytes': len(body)}
        self.rows.append(row)
        self.writer.write(row)
        return self.seq, upper

    def settle(self, seq, provider, selector, unit, utc_start, utc_end, latency_s, http_status, usage,
               model_returned, status, upper):
        price_window = window(utc_start)
        est = deepseek_usd(usage, price_window) if provider == 'deepseek' else jev_usd(usage)
        basis = 'reported_usage' if est is not None else 'upper_bound_no_usage'
        est = est if est is not None else upper
        row = {'kind': 'settle', 'call_seq': seq, 'provider': provider, 'selector': selector, 'unit': unit,
               'utc_start': utc_start, 'utc_end': utc_end, 'latency_s': latency_s, 'http_status': http_status,
               'usage': usage, 'model_returned': model_returned, 'status': status, 'price_window': price_window,
               'usd_estimate': est, 'usd_basis': basis}
        self.rows.append(row)
        row = {**row, 'cumulative_usd_after': self.usd(), 'jev_requests_after': self.jev()}
        self.writer.write(row)
        return row

    def close(self):
        self.writer.close()
