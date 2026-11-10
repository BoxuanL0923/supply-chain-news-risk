"""Provider calls for one period (2024 first, then 2025): resumable, idempotent, off-peak DeepSeek, balance floor.

Requests, parsers and returned-model checks are TD3 v6's (td3_providers.selectors: DeepSeekOrder, JevChoice, JevNoul;
UrllibTransport; MockTransport); prices and cost formulas are TD3 v6's (td3_ledger). New here:
- keyed ledger: every request has req_key = SHA256(TD4|year|pass|selector|input key|input sha256|question sha256|model);
  a 'reserve' row (upper bound) is fsynced before the attempt and a 'settle' row (HTTP status, tokens, returned model,
  USD, price window, latency, response sha256) after it, in one ledger per session; on resume, a req_key with an
  HTTP-200 settle row (valid or invalid content) is never re-sent (no re-billing); a reserve without a settle (crash)
  is counted at its upper bound and the request is re-sent once;
- DeepSeek balance floor: the balance is read (GET /user/balance, read-only) at session start and every 250 calls or
  15 minutes; before each DeepSeek attempt the projected balance min(last read - spend since that read, first read -
  session spend) minus the attempt's upper bound must stay >= USD 8.00, else the session stops cleanly
  (DEEPSEEK_BALANCE_FLOOR) and can be resumed after a top-up;
- DeepSeek only off-peak: no DeepSeek attempt starts in 01:00-04:00 or 06:00-10:00 UTC Monday-Friday or within two
  minutes before such a window (workers wait); Jev is not time-restricted;
- Jev HTTP 429: retried with waits 5, 15, 45, 120, 300 s (each attempt reserved and counted); per-period Jev USD and
  request caps from the frozen cost projection; the returned model must be jev-1.14.0 in every response (TD3 check);
- concurrency: 6 DeepSeek and 3 Jev worker threads; a halt (HTTP 401/402/403, a redirect, a third other 4xx per
  provider, three consecutive failed units per provider, the TD3 structural probe, a disallowed returned model, a cap)
  stops new attempts, lets in-flight attempts settle, and writes the session summary.
Keys come only from DEEPSEEK_API_KEY and TYPESAFE_API_KEY (process or Windows User scope), are held in memory, and every
written line passes the TD3 redactor.
"""
import datetime as dt
import hashlib
import json
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import td4_spec as T
import td4_env as E
import td4_gate as G

RUNS = T.OUT / 'runs'


def _mods():
    E.paths()
    import td3_spec as S3
    import td3_providers as P
    import td3_ledger as LG
    import td3_runtime as R
    import td3_runner as RN
    return S3, P, LG, R, RN


# ------------------------------------------------------------------------------------------------------------ clocks
class RealClock:
    def now(self):
        return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)

    def iso(self):
        return self.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    def sleep(self, s):
        time.sleep(s)


class SimClock(RealClock):
    """Simulated UTC clock for dry runs and tests: sleep advances time instantly."""

    def __init__(self, start='2026-11-07T12:00:00'):
        self.t = dt.datetime.fromisoformat(start)
        self.lock = threading.Lock()
        self.slept = 0.0

    def now(self):
        with self.lock:
            return self.t

    def sleep(self, s):
        with self.lock:
            self.t += dt.timedelta(seconds=s)
            self.slept += s

    def advance(self, s):
        self.sleep(s)


def in_peak(t):
    return t.weekday() < 5 and any(a <= t.hour < b for a, b in T_PEAK())


def T_PEAK():
    S3 = _mods()[0]
    return S3.PEAK_WINDOWS_UTC


def deepseek_window_ok(t, guard=T.PEAK_GUARD_S):
    return not in_peak(t) and not in_peak(t + dt.timedelta(seconds=guard))


# ------------------------------------------------------------------------------------------------- balance and keys
def read_balance(key, timeout=30):
    """GET https://api.deepseek.com/user/balance (read-only); returns USD total_balance as float."""
    import ssl
    import urllib.request
    _, P, _, _, _ = _mods()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), P._NoRedirect(),
                                         urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    req = urllib.request.Request(T.DEEPSEEK_BALANCE_URL, headers={'Authorization': 'Bearer ' + key,
                                                                  'Accept': 'application/json'}, method='GET')
    with opener.open(req, timeout=timeout) as r:
        data = json.loads(r.read(100000).decode('utf-8'))
    return parse_balance(data)


def parse_balance(data):
    infos = data.get('balance_infos') if isinstance(data, dict) else None
    usd = [x for x in (infos or []) if isinstance(x, dict) and x.get('currency') == 'USD']
    if len(usd) != 1:
        raise ValueError('DeepSeek balance response has no single USD entry')
    return float(usd[0]['total_balance'])


class Keyring:
    def __init__(self, redactor, keys):
        self.keys = keys
        self.redactor = redactor            # the redactor that holds the key values (used for every written line)
        for v in keys.values():
            redactor.add(v)

    @classmethod
    def from_environment(cls, redactor):
        _, _, _, R, _ = _mods()
        S3 = _mods()[0]
        return cls(redactor, {p: R.load_key(v) for p, v in S3.KEY_VARS.items()})

    def get(self, provider):
        return self.keys[provider]


# ------------------------------------------------------------------------------------------------------ ledger state
def req_key(year, pas, selector, key, input_sha, question_sha, model):
    return hashlib.sha256(f'TD4|{year}|{pas}|{selector}|{key}|{input_sha}|{question_sha}|{model}'.encode()).hexdigest()[:32]


class CapStop(Exception):
    pass


def iter_jsonl(path):
    """Stream a JSONL file one record at a time (ledgers and responses can hold hundreds of thousands of rows)."""
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prior_state(period_dir):
    """Across every earlier session of the period: completed req_keys, USD per provider (settled estimates; unsettled
    reserves at their bound), Jev request count."""
    _, _, _, R, _ = _mods()
    done, usd, jev_req = {}, Counter(), 0
    reserves, settles = {}, {}
    for p in sorted(Path(period_dir).glob('session_*/ledger.jsonl')):
        for r in iter_jsonl(p):
            k = (p.parent.name, r['req_key'], r['attempt'])
            if r['kind'] == 'reserve':
                reserves[k] = r
                if r['provider'] == 'jev':
                    jev_req += 1
            elif r['kind'] == 'settle':
                settles[k] = r
                if r['http_status'] == 200 and r['status'] != 'model_not_allowed':
                    done[r['req_key']] = r
    for k, r in reserves.items():
        s = settles.get(k)
        usd[r['provider']] += s['usd_estimate'] if s is not None and s.get('usd_estimate') is not None else r['usd_upper']
    return done, dict(usd), jev_req


class Budget:
    def __init__(self, caps, prior_usd, prior_jev, balance_fn, clock):
        self.lock = threading.Lock()
        self.caps = caps
        self.usd = Counter(prior_usd)
        self.jev_requests = prior_jev
        self.balance_fn = balance_fn
        self.clock = clock
        self.first_balance = self.last_balance = balance_fn()
        self.spend_since_read = 0.0
        self.session_ds_spend = 0.0
        self.calls_since_read = 0
        self.read_at = time.monotonic()
        self.reads = [self.first_balance]

    def projected_balance(self):
        return min(self.last_balance - self.spend_since_read, self.first_balance - self.session_ds_spend)

    def maybe_reread(self):
        due = self.calls_since_read >= T.DEEPSEEK_BALANCE_READ_EVERY_CALLS or \
            time.monotonic() - self.read_at >= T.DEEPSEEK_BALANCE_READ_EVERY_S
        if due:
            b = self.balance_fn()
            with self.lock:
                self.last_balance, self.spend_since_read, self.calls_since_read = b, 0.0, 0
                self.read_at = time.monotonic()
                self.reads.append(b)

    def reserve(self, provider, upper):
        with self.lock:
            if provider == 'deepseek':
                if self.projected_balance() - upper < T.DEEPSEEK_BALANCE_FLOOR_USD:
                    raise CapStop(f'DEEPSEEK_BALANCE_FLOOR: projected {self.projected_balance():.4f} - bound '
                                  f'{upper:.6f} < {T.DEEPSEEK_BALANCE_FLOOR_USD:.2f}')
                if self.usd['deepseek'] + upper > self.caps['deepseek_max_usd']:
                    raise CapStop(f"DEEPSEEK_USD_CAP: {self.usd['deepseek']:.4f} + {upper:.6f} > "
                                  f"{self.caps['deepseek_max_usd']}")
                self.spend_since_read += upper
                self.session_ds_spend += upper
                self.calls_since_read += 1
            else:
                if self.jev_requests + 1 > self.caps['jev_max_requests']:
                    raise CapStop(f"JEV_REQUEST_CAP: {self.jev_requests} + 1 > {self.caps['jev_max_requests']}")
                if self.usd['jev'] + upper > self.caps['jev_max_usd']:
                    raise CapStop(f"JEV_USD_CAP: {self.usd['jev']:.4f} + {upper:.6f} > {self.caps['jev_max_usd']}")
                self.jev_requests += 1
            self.usd[provider] += upper

    def settle(self, provider, upper, est):
        with self.lock:
            self.usd[provider] += est - upper
            if provider == 'deepseek':
                self.spend_since_read += est - upper
                self.session_ds_spend += est - upper


# --------------------------------------------------------------------------------------------------------- execution
class Session:
    def __init__(self, year, period_dir, records, plan, transport, keyring, budget, redactor, clock, run_mode,
                 binding, menu, question_sha, concurrency=None):
        S3, P, LG, R, RN = _mods()
        self.S3, self.P, self.LG, self.R, self.RN = S3, P, LG, R, RN
        self.year, self.records, self.plan = year, records, plan
        self.transport, self.keyring, self.budget = transport, keyring, budget
        self.redactor, self.clock, self.run_mode = redactor, clock, run_mode
        self.sels = P.selectors(menu)
        self.question_sha = question_sha
        n = 1 + max([int(p.name.split('_')[1]) for p in Path(period_dir).glob('session_*')], default=0)
        self.dir = Path(period_dir) / f'session_{n}'
        R.new_json(self.dir / 'SESSION_STARTED.json',
                   {'version': T.VERSION, 'year': year, 'run_mode': run_mode, 'session': self.dir.name,
                    'units_planned': len(plan), 'binding': binding, 'caps': budget.caps,
                    'balance_first_read_usd': budget.first_balance, 'question_sha256': question_sha,
                    'deepseek': {'url': S3.DEEPSEEK_URL, 'model': S3.DEEPSEEK_MODEL, 'thinking': S3.DEEPSEEK_THINKING},
                    'jev': {'url': S3.JEV_URL, 'model': S3.JEV_MODEL}, 'model_checks': S3.model_checks(),
                    'concurrency': concurrency or {'deepseek': T.DEEPSEEK_CONCURRENCY, 'jev': T.JEV_CONCURRENCY}},
                   redactor)
        self.ledger = R.JsonlWriter(self.dir / 'ledger.jsonl', redactor)
        self.responses = R.JsonlWriter(self.dir / 'responses.jsonl', redactor)
        self.wlock = threading.Lock()
        self.slock = threading.Lock()
        self.stop = threading.Event()
        self.halt = None
        self.counts = Counter()
        self.recent = {s: [] for s in S3.SELECTORS}
        self.failure_run = Counter()
        self.client_errors = Counter()
        self.concurrency = concurrency or {'deepseek': T.DEEPSEEK_CONCURRENCY, 'jev': T.JEV_CONCURRENCY}

    def write(self, writer, row):
        with self.wlock:
            writer.write(row)

    def set_halt(self, reason):
        with self.slock:
            if self.halt is None:
                self.halt = reason
        self.stop.set()

    def wait_off_peak(self):
        while not deepseek_window_ok(self.clock.now()):
            if self.stop.is_set():
                return False
            self.counts['deepseek_peak_waits'] += 1
            self.clock.sleep(60)
        return True

    def attempt(self, unit, attempt):
        sel = self.sels[unit['selector']]
        rec = self.records[unit['key']]
        body = sel.body(rec['input_text'])
        upper = self.LG.upper_usd(sel.provider, body)
        if sel.provider == 'deepseek':
            if not self.wait_off_peak():
                return None
            try:
                self.budget.maybe_reread()
            except Exception as e:              # sanitized: exception type only (never a header or key)
                self.set_halt('deepseek_balance_read_failed_' + type(e).__name__)
                return None
        try:
            self.budget.reserve(sel.provider, upper)
        except CapStop as e:
            self.set_halt(str(e))
            return None
        t0 = self.clock.iso()
        self.write(self.ledger, {'kind': 'reserve', 'req_key': unit['req_key'], 'attempt': attempt,
                                 'provider': sel.provider, 'selector': sel.name, 'key': unit['key'],
                                 'pass': unit['pass'], 'usd_upper': upper, 'body_bytes': len(body), 'utc': t0})
        status = raw = terr = None
        p0 = time.perf_counter()
        try:
            status, raw = self.transport.post(sel.url, sel.headers(self.keyring.get(sel.provider)), body)
        except self.P.TransportError as e:
            terr = str(e)
        except Exception as e:
            terr = 'unexpected_' + type(e).__name__
        latency = time.perf_counter() - p0
        t1 = self.clock.iso()
        truncated = raw is not None and len(raw) > self.S3.MAX_RESPONSE_BYTES
        if terr is not None:
            parsed = {'selector': sel.name, 'status': 'transport_error', 'issues': ['transport_error'], 'usage': {},
                      'model_returned': None}
        elif truncated:
            parsed = {'selector': sel.name, 'status': 'response_too_large', 'issues': ['response_too_large'],
                      'usage': {}, 'model_returned': None}
        else:
            parsed = sel.apply_model_check(status, sel.parse(status, raw))
        window = self.LG.window(t0)
        est = self.LG.deepseek_usd(parsed.get('usage', {}), window) if sel.provider == 'deepseek' else \
            self.LG.jev_usd(parsed.get('usage', {}))
        basis = 'reported_usage' if est is not None else 'upper_bound_no_usage'
        est = upper if est is None else est
        self.budget.settle(sel.provider, upper, est)
        # the response row is written before the settle row: a crash between them leaves a stored response and an
        # unsettled reserve (re-sent once on resume, counted at its bound), never a settled request without output
        self.write(self.responses, {'req_key': unit['req_key'], 'attempt': attempt, 'key': unit['key'],
                                    'selector': sel.name, 'pass': unit['pass'], 'year': self.year,
                                    'input_sha256': rec['input_sha256'], 'body_sha256': self.R.sha_bytes(body),
                                    'http_status': status, 'transport_error': terr, 'utc_start': t0,
                                    'response_raw': raw[:self.S3.MAX_RESPONSE_BYTES].decode('utf-8', 'replace')
                                    if raw is not None else None, 'parsed': parsed, 'usd_estimate': est,
                                    'price_window': window})
        self.write(self.ledger, {'kind': 'settle', 'req_key': unit['req_key'], 'attempt': attempt,
                                 'provider': sel.provider, 'selector': sel.name, 'key': unit['key'],
                                 'pass': unit['pass'], 'utc_start': t0, 'utc_end': t1, 'latency_s': latency,
                                 'http_status': status, 'usage': parsed.get('usage', {}),
                                 'model_returned': parsed.get('model_returned'), 'status': parsed['status'],
                                 'price_window': window, 'usd_estimate': est, 'usd_basis': basis,
                                 'response_sha256': self.R.sha_bytes(raw) if raw is not None else None})
        self.counts['attempts_' + sel.provider] += 1
        return status, terr, parsed

    def run_unit(self, unit):
        if self.stop.is_set():
            return
        sel = self.sels[unit['selector']]
        jev = sel.provider == 'jev'
        waits = T.JEV_429_BACKOFF_S if jev else T.OTHER_RETRY_BACKOFF_S
        other_waits = T.OTHER_RETRY_BACKOFF_S
        attempt, n429, nother = 0, 0, 0
        while True:
            if self.stop.is_set():
                return
            attempt += 1
            out = self.attempt(unit, attempt)
            if out is None:
                return
            status, terr, parsed = out
            if status == 200:
                with self.slock:
                    if parsed['status'] == 'model_not_allowed':
                        self.halt = self.halt or f"model_not_allowed_{sel.name}:{parsed.get('model_violation')}"
                        self.stop.set()
                        return
                    self.recent[sel.name].append(parsed)
                    self.recent[sel.name] = self.recent[sel.name][-self.S3.STRUCTURAL_PROBE_N:]   # bounded memory
                    reason = self.RN.structural_halt(self.recent[sel.name])
                    self.failure_run[sel.provider] = 0
                    self.counts['done_' + sel.name] += 1
                    self.counts[f"{sel.name}_{parsed['status']}"] += 1
                if reason:
                    self.set_halt(f'{reason}_{sel.name}')
                return
            if status == 429 and jev and n429 < len(waits):
                n429 += 1
                self.clock.sleep(waits[n429 - 1])
                continue
            if (terr is not None or status in self.S3.RETRY_STATUS) and nother < len(other_waits):
                nother += 1
                self.clock.sleep(other_waits[nother - 1])
                continue
            if status in self.S3.HALT_STATUS:
                self.set_halt(f'http_{status}_{sel.provider}')
                return
            if status is not None and 300 <= status < 400:
                self.set_halt(f'http_redirect_{status}_{sel.provider}')
                return
            if status is not None and 400 <= status < 500 and status not in self.S3.RETRY_STATUS:
                with self.slock:
                    self.client_errors[sel.provider] += 1
                    n = self.client_errors[sel.provider]
                if n >= self.S3.CLIENT_ERROR_LIMIT:
                    self.set_halt(f'client_error_limit_{sel.provider}')
                return
            with self.slock:
                self.failure_run[sel.provider] += 1
                n = self.failure_run[sel.provider]
                self.counts['failed_units_' + sel.provider] += 1
            if n >= self.S3.FAILURE_UNIT_LIMIT:
                self.set_halt(f'failure_unit_limit_{sel.provider}')
            return

    def execute(self):
        units = {'deepseek': [u for u in self.plan if self.sels[u['selector']].provider == 'deepseek'],
                 'jev': [u for u in self.plan if self.sels[u['selector']].provider == 'jev']}
        pools = {p: ThreadPoolExecutor(max_workers=self.concurrency[p], thread_name_prefix='td4_' + p)
                 for p in units}
        futures = []
        try:
            for p, us in units.items():
                for u in us:
                    futures.append(pools[p].submit(self.run_unit, u))
            done = 0
            for f in futures:
                f.result()
                done += 1
                if done % 2000 == 0:
                    self.R.say(self.redactor, 'TD4_PROGRESS', self.year, done, '/', len(futures),
                               json.dumps({k: round(v, 4) for k, v in self.budget.usd.items()}),
                               'balance_proj', round(self.budget.projected_balance(), 4), 'halt', self.halt)
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
            self.ledger.close()
            self.responses.close()
        summary = {'version': T.VERSION, 'year': self.year, 'session': self.dir.name, 'run_mode': self.run_mode,
                   'units_planned': len(self.plan), 'halt': self.halt, 'counts': dict(self.counts),
                   'client_errors': dict(self.client_errors), 'usd_period_after': dict(self.budget.usd),
                   'jev_requests_period_after': self.budget.jev_requests,
                   'balance_reads_usd': self.budget.reads,
                   'balance_projected_end_usd': self.budget.projected_balance(), 'redactions': self.redactor.count,
                   **self.R.peak()}
        self.R.new_json(self.dir / 'SESSION_SUMMARY.json', summary, self.redactor)
        return summary


# --------------------------------------------------------------------------------------------------------- planning
def reask_keys(records, year, n=T.REASK_N):
    return sorted(records, key=lambda k: hashlib.sha256(f'TD4-reask-v1|{year}|{k}'.encode()).hexdigest())[:n]


def make_plan(year, records, question_sha, model_for, reask=False):
    S3 = _mods()[0]
    keys = sorted(records, key=lambda k: records[k]['sample_rank'])
    pas = 'main'
    if reask:
        keys = reask_keys(records, year)
        pas = 'reask1'
    plan = []
    for k in keys:
        for s in S3.SELECTORS:
            plan.append({'key': k, 'selector': s, 'pass': pas,
                         'req_key': req_key(year, pas, s, k, records[k]['input_sha256'], question_sha, model_for[s])})
    return plan


def period_dir(year, mode):
    return RUNS / f'{mode}_{year}'


def run_session(year, records, transport, keyring, balance_fn, clock, caps, run_mode, binding, reask=False,
                menu=None, limit=None, concurrency=None, root=None):
    S3, P, LG, R, RN = _mods()
    menu = menu or S3.menu_rows(T.TD2 / 'risk_menu.csv')
    question_sha = S3.question_bundle(menu)[1]
    model_for = {'deepseek_order': S3.DEEPSEEK_MODEL, 'jev_choice': S3.JEV_MODEL, 'jev_noul': S3.JEV_MODEL}
    pdir = Path(root) if root else period_dir(year, run_mode)
    done, prior_usd, prior_jev = prior_state(pdir)
    plan = make_plan(year, records, question_sha, model_for, reask=reask)
    if reask:
        main_done = {req_key(year, 'main', s, k, records[k]['input_sha256'], question_sha, model_for[s])
                     for k in records for s in S3.SELECTORS} & set(done)
        need = {u['key'] for u in plan}
        if not all(req_key(year, 'main', s, k, records[k]['input_sha256'], question_sha, model_for[s]) in main_done
                   for k in need for s in S3.SELECTORS):
            raise PermissionError('Re-ask runs only after the main pass of the re-ask keys is complete')
    if limit:
        plan = plan[:limit * len(S3.SELECTORS)]
    todo = [u for u in plan if u['req_key'] not in done]
    budget = Budget(caps, prior_usd, prior_jev, balance_fn, clock)
    sess = Session(year, pdir, records, todo, transport, keyring, budget, keyring.redactor, clock, run_mode,
                   {**binding, 'skipped_done': len(plan) - len(todo)}, menu, question_sha, concurrency)
    summary = sess.execute()
    summary['skipped_done_before'] = len(plan) - len(todo)
    return sess.dir, summary


# ---------------------------------------------------------------------------------------------------- entry points
def run_real(year, reask=False):
    """Clearance A step. 2025 calls refuse until the 2024 main pass is complete (2024 first, then 2025)."""
    import td4_inputs as I
    S3, P, LG, R, RN = _mods()
    if year not in T.PERIODS:
        raise PermissionError('Periods 2024 and 2025 only')
    token, freeze = G.admit('reask' if reask else 'call')
    if R.mode() != 'online':
        raise PermissionError('online runtime required')
    records = I.callable_inputs(year)
    if year == T.TEST_YEAR and not reask:
        prev = I.callable_inputs(T.SELECTION_YEAR)
        st = completion(T.SELECTION_YEAR, prev)
        if st['main_complete'] is not True:
            raise PermissionError('STOP: run 2024 to completion before 2025 (' + json.dumps(st) + ')')
    lock = R.RunLock(RUNS / 'TD4_RUN.lock').acquire()
    try:
        redactor = R.Redactor()
        keyring = Keyring.from_environment(redactor)
        caps = freeze['caps'][str(year)]
        pdir, summary = run_session(year, records, P.UrllibTransport(), keyring,
                                    lambda: read_balance(keyring.get('deepseek')), RealClock(), caps, 'real',
                                    {'clearance_sha256': token.clearance_sha, 'freeze_sha256': G.sha(G.FREEZE_A)},
                                    reask=reask)
    finally:
        lock.release()
    return {'session_dir': str(pdir), **{k: summary[k] for k in ('halt', 'counts', 'usd_period_after',
                                                                 'balance_projected_end_usd', 'skipped_done_before')},
            'completion': completion(year, records)}


def completion(year, records, mode='real'):
    S3 = _mods()[0]
    menu = S3.menu_rows(T.TD2 / 'risk_menu.csv')
    q = S3.question_bundle(menu)[1]
    model_for = {'deepseek_order': S3.DEEPSEEK_MODEL, 'jev_choice': S3.JEV_MODEL, 'jev_noul': S3.JEV_MODEL}
    done, usd, jev = prior_state(period_dir(year, mode))
    need = [req_key(year, 'main', s, k, r['input_sha256'], q, model_for[s]) for k, r in records.items()
            for s in S3.SELECTORS]
    have = sum(1 for k in need if k in done)
    return {'year': year, 'units': len(need), 'completed': have, 'main_complete': have == len(need),
            'usd_by_provider': usd, 'jev_requests': jev}


def dry_run(year, limit=None, inputs_path=None):
    """Mock providers over the real frozen inputs of a period (no network, simulated clock, mock balance)."""
    import td4_inputs as I
    import td4_cost as C
    S3, P, LG, R, RN = _mods()
    records = I.callable_inputs(year) if inputs_path is None else I.load_inputs(year, path=inputs_path)
    if limit:
        keep = sorted(records, key=lambda k: records[k]['sample_rank'])[:limit]
        records = {k: records[k] for k in keep}
    redactor = R.Redactor()
    keyring = Keyring(redactor, {'deepseek': 'sk-mock-deepseek-0000', 'jev': 'mock-jev-key-0000'})
    clock = SimClock('2026-11-09T00:59:00')          # a Monday 00:59 UTC: DeepSeek must wait for 04:00, Jev runs
    caps = C.caps_for(year, len(records))
    root = RUNS / f'dryrun_{year}_{uuid.uuid4().hex[:8]}'
    transport = P.MockTransport()
    pdir, summary = run_session(year, records, transport, keyring, lambda: 62.57, clock, caps, 'dryrun',
                                {'dry_run': True}, root=root, concurrency={'deepseek': 4, 'jev': 2})
    done, usd, jev = prior_state(root)
    out = {'year': year, 'records': len(records), 'mock_calls': len(transport.calls), 'halt': summary['halt'],
           'usd_by_provider_mock_tokens': usd, 'jev_requests': jev, 'counts': summary['counts'],
           'simulated_peak_wait_s': clock.slept, 'run_dir': str(root)}
    resume_dir, resume = run_session(year, records, transport, keyring, lambda: 62.57, clock, caps, 'dryrun',
                                     {'dry_run': True, 'resume_check': True}, root=root,
                                     concurrency={'deepseek': 4, 'jev': 2})
    out['resume_check'] = {'skipped_done_before': resume['skipped_done_before'],
                           'new_mock_calls': len(transport.calls) - out['mock_calls']}
    R.new_json(root / 'DRYRUN_SUMMARY.json', out)
    return out


# ---------------------------------------------------------------------------------------------- collected outputs
def collect(year, mode='real', root=None):
    """Final parsed output per (key, selector, pass): the HTTP-200 attempt (valid or invalid content); otherwise the
    last attempt's status (no output). Returns a list of dicts."""
    _, _, _, R, _ = _mods()
    pdir = Path(root) if root else period_dir(year, mode)
    best = {}
    for p in sorted(pdir.glob('session_*/responses.jsonl')):
        for r in iter_jsonl(p):
            r.pop('response_raw', None)                  # raw bodies stay on disk; parsed fields are kept
            k = (r['key'], r['selector'], r['pass'])
            ok = r['http_status'] == 200 and r['parsed'].get('status') != 'model_not_allowed'
            prev = best.get(k)
            if prev is None or (ok and not prev['_ok']) or (ok == prev['_ok'] and not prev['_ok']):
                pr = r['parsed']
                best[k] = {'_ok': ok, 'usd_estimate': r.get('usd_estimate'), 'input_sha256': r.get('input_sha256'),
                           'parsed': {f: pr.get(f) for f in ('status', 'ordering', 'ordering_raw', 'choice',
                                                             'probabilities', 'yes_scores', 'issues',
                                                             'model_returned', 'usage')}}
    out = []
    for (key, sel, pas), r in best.items():
        pr = r['parsed']
        out.append({'key': key, 'selector': sel, 'pass': pas, 'http_ok': r['_ok'], 'status': pr.get('status'),
                    'ordering': pr.get('ordering'), 'ordering_raw': pr.get('ordering_raw'),
                    'choice': pr.get('choice'), 'probabilities': pr.get('probabilities'),
                    'yes_scores': pr.get('yes_scores'), 'issues': pr.get('issues'),
                    'model_returned': pr.get('model_returned'), 'usage': pr.get('usage'),
                    'usd_estimate': r.get('usd_estimate'), 'input_sha256': r.get('input_sha256')})
    return out
