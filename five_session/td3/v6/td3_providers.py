"""Selector requests, strict response parsers and HTTP transports. v4 (TD3D).

v4 (TD3C3 N5): the Jev Choice probability-sum tolerance is td3_spec.PROBABILITY_SUM_TOLERANCE = 0.03 (6 options x
0.005, so that rounding each value to two decimals never makes a valid answer invalid); the raw sum is stored as
before. Otherwise unchanged from v3.

v3: the DeepSeek request names deepseek-flash and sends "thinking": {"type": "disabled"} (official API reference,
read 2026-11-05; thinking is otherwise on by default) with temperature 0 and JSON output (D3); any reasoning_content
that is not null, or a reasoning_tokens count that is not null and is non-numeric or positive, is a model violation
(m1).

v2: Jev Choice probabilities must sum to 1 within +/-0.02 (C3; the raw sum is stored); every HTTP-200 response is
checked against the frozen returned-model rules (C5: DeepSeek model in the priced allowlist, no reasoning tokens or
reasoning_content; Jev model jev-1.14.0) and a violation marks the response model_not_allowed, which halts the run;
redirects are never followed (R2).

Request bodies are deterministic functions of (frozen input text, frozen question), so a re-ask sends identical
bytes. Parsers never repair, normalize or backfill: invalid content is recorded with its issues and the raw
response, and the validated field stays None. Keys are passed only as header values at call time and are never
part of a body, record or exception message.
"""
import json
import math
import ssl
import urllib.error
import urllib.request
import td3_spec as S


class TransportError(Exception):
    """Sanitized transport failure (exception type and reason only)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """R2: never follow a redirect (urllib would copy the Authorization header); the 3xx reaches the runner."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _decode(raw):
    if raw is None:
        return None, 'no_body'
    try:
        return json.loads(raw.decode('utf-8')), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 'malformed_body'


class Selector:
    name = provider = url = model = None

    def __init__(self, menu):
        self.menu = menu
        self.names = [m['name'] for m in menu]

    def headers(self, key):
        return {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'Accept': 'application/json',
                'User-Agent': 'td3-selector-pilot/1'}

    def body(self, input_text):
        return json.dumps(self.payload(input_text), ensure_ascii=False, separators=(',', ':')).encode('utf-8')

    def base(self, status, raw):
        out = {'selector': self.name, 'status': None, 'issues': [], 'notes': [], 'usage': {}, 'model_returned': None}
        if status != 200:
            out['status'] = 'http_error'
            return out, None
        data, err = _decode(raw)
        if err or not isinstance(data, dict):
            out['status'] = 'malformed_body'
            out['issues'].append(err or 'body_not_object')
            return out, None
        out['model_returned'] = data.get('model') if isinstance(data.get('model'), str) else None
        return out, data

    def model_violation(self, parsed):
        """C5: reason string when an HTTP-200 response breaks the frozen returned-model rules, else None."""
        return None

    def apply_model_check(self, status, parsed):
        """Called by the runner on every parsed response; a violation turns the status into model_not_allowed."""
        if status != 200 or parsed.get('status') in (None, 'http_error'):
            return parsed
        if parsed.get('status') == 'malformed_body' and parsed.get('model_returned') is None:
            return parsed        # no model reported in an unusable body: left to the structural probe
        reason = self.model_violation(parsed)
        parsed['model_violation'] = reason
        if reason is not None:
            parsed['content_status'] = parsed['status']
            parsed['status'] = 'model_not_allowed'
            parsed['issues'] = list(parsed.get('issues', [])) + [reason]
            for k in ('ordering', 'choice', 'probabilities', 'yes_scores'):
                if k in parsed:
                    parsed[k] = None
        return parsed


class DeepSeekOrder(Selector):
    name, provider, url, model = 'deepseek_order', 'deepseek', S.DEEPSEEK_URL, S.DEEPSEEK_MODEL

    def payload(self, input_text):
        return {'model': self.model,
                'messages': [{'role': 'system', 'content': S.SYSTEM_DEEPSEEK},
                             {'role': 'user', 'content': S.deepseek_question(self.menu) + '\n\n' + input_text}],
                'thinking': dict(S.DEEPSEEK_THINKING),
                'temperature': S.DEEPSEEK_TEMPERATURE, 'max_tokens': S.DEEPSEEK_MAX_TOKENS,
                'response_format': {'type': 'json_object'}, 'stream': False}

    def parse(self, status, raw):
        out, data = self.base(status, raw)
        out.update(ordering=None, ordering_raw=None, ordering_filtered=None, invalid_names=[], duplicate_names=[],
                   missing_names=[], finish_reason=None, system_fingerprint=None, reasoning_content_present=False,
                   reasoning_tokens_raw=None, reasoning_tokens_reported=False)
        if data is None:
            return out
        u = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        details = u.get('completion_tokens_details') if isinstance(u.get('completion_tokens_details'), dict) else {}
        out['usage'] = {k: u.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens',
                                              'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens') if _num(u.get(k))}
        if _num(details.get('reasoning_tokens')):
            out['usage']['reasoning_tokens'] = details['reasoning_tokens']
        if details.get('reasoning_tokens') is not None:          # m1: kept raw, whatever its type
            out['reasoning_tokens_reported'] = True
            out['reasoning_tokens_raw'] = details['reasoning_tokens'] if isinstance(
                details['reasoning_tokens'], (int, float, str, bool)) else repr(details['reasoning_tokens'])
        out['system_fingerprint'] = data.get('system_fingerprint') if isinstance(data.get('system_fingerprint'), str) \
            else None
        try:
            choice = data['choices'][0]
            content = choice['message']['content']
            out['finish_reason'] = choice.get('finish_reason')
            rc = choice['message'].get('reasoning_content')
            out['reasoning_content_present'] = rc is not None      # m1: any non-null value, empty string included
        except (KeyError, IndexError, TypeError, AttributeError):
            out['status'] = 'malformed_body'
            out['issues'].append('no_message_content')
            return out
        if not isinstance(content, str):
            out['status'] = 'malformed_body'
            out['issues'].append('content_not_text')
            return out
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            out['status'] = 'invalid'
            out['issues'].append('invalid_json')
            return out
        if not isinstance(obj, dict) or not isinstance(obj.get('ordering'), list):
            out['status'] = 'invalid'
            out['issues'].append('missing_ordering')
            return out
        if set(obj) - {'ordering'}:
            out['notes'].append('extra_keys')
        out['ordering_raw'] = obj['ordering']
        clean = []
        for x in obj['ordering']:
            if not isinstance(x, str) or x not in self.names:
                out['invalid_names'].append(x if isinstance(x, str) else repr(x))
            elif x in clean:
                out['duplicate_names'].append(x)
            else:
                clean.append(x)
        out['missing_names'] = [n for n in self.names if n not in clean]
        for field in ('invalid_names', 'duplicate_names', 'missing_names'):
            if out[field]:
                out['issues'].append(field)
        out['ordering_filtered'] = clean          # invalid/duplicate entries removed; nothing appended
        out['status'] = 'invalid' if out['issues'] else 'valid'
        out['ordering'] = clean if out['status'] == 'valid' else None
        return out

    def model_violation(self, parsed):
        if parsed.get('model_returned') not in S.DEEPSEEK_RETURNED_MODEL_PRICES:
            return 'deepseek_model_not_allowed'
        if S.DEEPSEEK_RETURNED_MODEL_PRICES[parsed['model_returned']] != S.PRICES:
            return 'deepseek_model_price_differs'
        if parsed.get('reasoning_tokens_reported'):
            raw = parsed.get('reasoning_tokens_raw')
            if not _num(raw):
                return 'deepseek_reasoning_tokens_not_numeric'
            if raw > 0:
                return 'deepseek_reasoning_tokens'
        if parsed.get('reasoning_content_present'):
            return 'deepseek_reasoning_content'
        return None


class JevBase(Selector):
    provider, url, model = 'jev', S.JEV_URL, S.JEV_MODEL

    def payload(self, input_text):
        return {'state': input_text, 'model': self.model, 'questions': self.questions()}

    def jev_base(self, status, raw):
        out, data = self.base(status, raw)
        if data is None:
            return out, None
        u = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        out['usage'] = {k: u.get(k) for k in ('input_tokens', 'output_tokens') if _num(u.get(k))}
        answers = data.get('answers')
        if not isinstance(answers, dict):
            out['status'] = 'malformed_body'
            out['issues'].append('answers_not_object')
            return out, None
        extra = set(answers) - set(self.questions())
        if extra:
            out['notes'].append('extra_answer_keys')
        return out, answers

    def model_violation(self, parsed):
        return None if parsed.get('model_returned') == S.JEV_MODEL else 'jev_model_not_allowed'


class JevChoice(JevBase):
    name = 'jev_choice'

    def questions(self):
        return S.jev_choice_question(self.menu)

    def parse(self, status, raw):
        out, answers = self.jev_base(status, raw)
        options = self.names + [S.NONE_OPTION]
        out.update(choice=None, probabilities=None, choice_raw=None, probabilities_raw=None, confidence=None,
                   probability_sum_raw=None)
        if answers is None:
            return out
        a = answers.get(S.CHOICE_KEY)
        if not isinstance(a, dict):
            out['status'] = 'invalid'
            out['issues'].append('missing_answer')
            return out
        choice, probs = a.get('choice'), a.get('probabilities')
        out['choice_raw'], out['probabilities_raw'] = choice, probs
        out['confidence'] = a.get('confidence') if _num(a.get('confidence')) else None
        if out['confidence'] is None:
            out['notes'].append('confidence_missing_or_not_numeric')
        if not isinstance(choice, str) or choice not in options:
            out['issues'].append('invalid_choice')
        if not isinstance(probs, dict):
            out['issues'].append('missing_probabilities')
        else:
            if set(probs) != set(options):
                out['issues'].append('probability_keys')
            if not all(_num(v) and 0 <= v <= 1 for v in probs.values()):
                out['issues'].append('probability_values')
            else:
                total = float(sum(probs.values()))
                out['probability_sum_raw'] = total
                if abs(total - 1) > S.PROBABILITY_SUM_TOLERANCE + 1e-9:   # +/-0.03 inclusive (float slack)
                    out['issues'].append('probability_sum')
                elif abs(total - 1) > 1e-9:
                    out['notes'].append('probability_sum_rounded')   # stored as returned; never renormalized
        out['status'] = 'invalid' if out['issues'] else 'valid'
        if out['status'] == 'valid':
            out['choice'] = choice
            out['probabilities'] = {o: float(probs[o]) for o in options}
            top = max(options, key=lambda o: (probs[o], -options.index(o)))
            if top != choice:
                out['notes'].append('choice_not_argmax')
        return out


class JevNoul(JevBase):
    name = 'jev_noul'

    def questions(self):
        return S.jev_noul_questions(self.menu)

    def parse(self, status, raw):
        out, answers = self.jev_base(status, raw)
        out.update(yes_scores=None, yes_scores_partial={}, missing_answers=[], invalid_scores=[], raw_answers=None)
        if answers is None:
            return out
        out['raw_answers'] = {k: answers.get(k) for k in self.questions()}
        for n in self.names:
            a = answers.get(S.NOUL_PREFIX + n)
            if not isinstance(a, dict) or 'noul' not in a:
                out['missing_answers'].append(n)
            elif _num(a['noul']) and 0 <= a['noul'] <= 1:
                out['yes_scores_partial'][n] = float(a['noul'])
            else:
                out['invalid_scores'].append(n)
        for field in ('missing_answers', 'invalid_scores'):
            if out[field]:
                out['issues'].append(field)
        out['status'] = 'invalid' if out['issues'] else 'valid'
        if out['status'] == 'valid':
            out['yes_scores'] = dict(out['yes_scores_partial'])
        return out


def selectors(menu):
    return {s.name: s for s in (DeepSeekOrder(menu), JevChoice(menu), JevNoul(menu))}


class UrllibTransport:
    """Direct HTTPS POST (no proxy handler, certificate verification on). Returns (status, body bytes)."""

    def __init__(self, timeout=S.HTTP_TIMEOUT_S):
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
                                                  urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def post(self, url, headers, body):
        if url not in (S.DEEPSEEK_URL, S.JEV_URL):
            raise TransportError('URL not allowed')
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                return r.status, r.read(S.MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            try:
                data = e.read(S.MAX_RESPONSE_BYTES + 1)
            except Exception:
                data = b''
            return e.code, data
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionError, OSError) as e:
            reason = getattr(e, 'reason', None)
            raise TransportError(f'{type(e).__name__}: {type(reason).__name__ if reason is not None else ""}') from None


class MockTransport:
    """Deterministic offline provider used by the dry run and the tests. Every response carries "mock": true and
    reports the frozen models (deepseek-flash, jev-1.14.0) so that the C5 checks pass."""

    def __init__(self, responder=None):
        self.calls = []
        self.responder = responder or mock_response

    def post(self, url, headers, body):
        self.calls.append({'url': url, 'body': body, 'has_auth': headers.get('Authorization', '').startswith('Bearer ')})
        return self.responder(url, headers, body)


def _h(*parts):
    import hashlib
    return int(hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / 16 ** 12


def mock_response(url, headers, body):
    req = json.loads(body.decode('utf-8'))
    n_in = int(len(body) * S.ESTIMATE_TOKENS_PER_BYTE)
    if url == S.DEEPSEEK_URL:
        text = req['messages'][1]['content']
        order = sorted(S.NAMES, key=lambda n: _h('ds', text, n))
        resp = {'mock': True, 'model': S.DEEPSEEK_MODEL,
                'choices': [{'message': {'content': json.dumps({'ordering': order})},
                                              'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': n_in, 'completion_tokens': S.ESTIMATE_OUTPUT_TOKENS,
                          'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': n_in}}
    else:
        state, qs = req['state'], req['questions']
        answers = {}
        for q, spec in qs.items():
            if spec['type'] == 'choice':
                w = {o: _h('ch', state, o) + 1e-9 for o in spec['criteria']}
                tot = sum(w.values())
                probs = {o: v / tot for o, v in w.items()}
                answers[q] = {'choice': max(probs, key=probs.get), 'probabilities': probs, 'confidence': 0.5}
            else:
                answers[q] = {'noul': _h('nl', state, q)}
        resp = {'mock': True, 'model': S.JEV_MODEL, 'answers': answers,
                'usage': {'input_tokens': n_in, 'output_tokens': 10}}
    return 200, json.dumps(resp).encode('utf-8')
