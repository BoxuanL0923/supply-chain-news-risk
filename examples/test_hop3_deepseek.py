"""deepseek-v4-pro transport, budget, final-content and resume checks; offline."""
import copy
import json
import io
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from news_network_50_2025 import hop3_deepseek as ds


class DeepSeekTests(unittest.TestCase):
    def envelope(self):
        return {'model': ds.MODEL, 'request_sha256': '', 'server_model': ds.MODEL,
                'received_at': '2026-10-26T00:00:00Z', 'seconds': 1,
                'envelope': {'model': ds.MODEL, 'usage': {'prompt_tokens': 1000,
                    'prompt_cache_hit_tokens': 300, 'prompt_cache_miss_tokens': 700,
                    'completion_tokens': 200, 'completion_tokens_details': {'reasoning_tokens': 150}},
                    'choices': [{'finish_reason': 'stop', 'message': {
                        'reasoning_content': '{"score":999,"evidence_ids":["forged"]}',
                        'content': '{"score":0.2,"evidence_ids":["s"],"reason":"limited evidence"}'}}]}}

    def test_parse_only_final_content(self):
        self.assertEqual(ds.parse_answer(self.envelope(), ['s'])['score'], .2)
        for change in ('model', 'length', 'nan'):
            saved = self.envelope()
            if change == 'model': saved['server_model'] = 'other'
            if change == 'length': saved['envelope']['choices'][0]['finish_reason'] = 'length'
            if change == 'evidence': saved['envelope']['choices'][0]['message']['content'] = '{"score":1,"evidence_ids":["x"],"reason":"x"}'
            if change == 'nan': saved['envelope']['choices'][0]['message']['content'] = '{"score":NaN,"evidence_ids":[],"reason":"x"}'
            with self.assertRaises(ValueError): ds.parse_answer(saved, ['s'])

    def test_price_includes_all_reasoning_and_cache_discount(self):
        u = self.envelope()['envelope']['usage']
        self.assertAlmostEqual(ds.token_cost(u), (700*.435 + 300*.0435 + 200*.87)/1e6)
        u['prompt_cache_hit_tokens'] = 500
        with self.assertRaises(ValueError): ds.token_cost(u)

    def test_only_unique_exact_source_prefix_may_be_resolved(self):
        saved = self.envelope()
        saved['envelope']['choices'][0]['message']['content'] = '{"score":0.1,"evidence_ids":["RTRS:2025-10:12"],"reason":"short ID"}'
        answer = ds.parse_answer(saved, ['RTRS:2025-10:12:abcd'])
        self.assertEqual(answer['evidence_ids'], ['RTRS:2025-10:12:abcd'])
        self.assertEqual(answer['score'], .1)
        for allowed in (['RTRS:2025-10:12:a', 'RTRS:2025-10:12:b'], ['RTRS:2025-10:123:a']):
            invalid = ds.parse_answer(saved, allowed)
            self.assertEqual(invalid['score'], .1)
            self.assertEqual(invalid['evidence_ids'], [])
            self.assertEqual(invalid['citation_validation_status'], 'FLAGGED')
            self.assertEqual(invalid['unresolved_evidence_ids'], ['RTRS:2025-10:12'])
        saved['envelope']['choices'][0]['message']['content'] = '{"score":0.1,"evidence_ids":["RTRS:2025-11:12:abcd"],"reason":"month typo"}'
        self.assertEqual(ds.parse_answer(saved, ['RTRS:2025-10:12:abcd'])['evidence_ids'], ['RTRS:2025-10:12:abcd'])
        invalid = ds.parse_answer(saved, ['RTRS:2025-10:12:forged'])
        self.assertEqual(invalid['evidence_ids'], [])
        self.assertEqual(invalid['citation_validation_status'], 'FLAGGED')

    def test_reporting_without_full_freeze_cannot_read_labels(self):
        from news_network_50_2025 import hop3_deepseek_report as reporting
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(ds, 'OUT', root), patch.object(ds.gates, 'load_labels', side_effect=AssertionError('labels forbidden')):
                ds.write(root/'PILOT_PROJECTION.json', {'projected_full_usd': 14})
                with self.assertRaisesRegex(ValueError, 'Full prediction freeze'):
                    reporting.report(None)
                ds.write(root/'PILOT_PROJECTION.json', {'projected_full_usd': 16})
                with patch.object(reporting, 'budget_stop_report', return_value='sealed'):
                    self.assertEqual(reporting.report(None), 'sealed')

    def test_cluster_correction_uses_effective_rank_with_redundant_controls(self):
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_cluster
        from examples.test_hop3_v2 import InferenceTests
        from news_network_50_2025.hop3_analysis import design, rank, regress
        frame = InferenceTests().frame()
        frame['rv22'] = 0.0
        frame['calendar_unknown'] = frame.b
        x = design(frame)
        fit = sm.OLS(frame.y.to_numpy(float), x).fit()
        covariance = cov_cluster(fit, frame.date.to_numpy(), use_correction=False)
        factor = 6/5 * (len(frame)-1)/(len(frame)-rank(x))
        expected = np.sqrt(covariance[-1, -1]*factor)
        actual = regress(frame)['coefficients']['H3_minus_H2']['date_cluster']['se']
        self.assertAlmostEqual(actual, expected, places=12)

    def test_prompt_limit_and_reasoning_budget(self):
        with self.assertRaises(ValueError): ds.request_body('x'*24000)
        body = ds.request_body('safe')
        self.assertEqual(body['model'], 'deepseek-v4-pro')
        self.assertGreaterEqual(body['max_tokens'], 16000)
        self.assertEqual(body['thinking'], {'type': 'enabled'})

    def test_cache_reconciles_cost_without_duplicate_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(ds, 'OUT', root), patch.object(ds, 'LEDGER', root/'cost.csv'):
                payload = {'prompt': 'safe', 'evidence_ids': ['s']}
                body = ds.request_body(payload['prompt'])
                key = ds.gates.digest(body)
                saved = self.envelope()
                saved.update(request=body, request_sha256=key)
                ds.write(root/'responses'/(key+'.json'), saved)
                with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden in test')):
                    first = ds.call(payload, None)
                    second = ds.call(payload, None)
                self.assertEqual(first, second)
                self.assertEqual(len(ds.rows()), 1)
                self.assertAlmostEqual(ds.spent(), ds.token_cost(saved['envelope']['usage']))

    def test_unresolved_request_is_never_repeated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(ds, 'OUT', root), patch.object(ds, 'LEDGER', root/'cost.csv'):
                body = ds.request_body('safe')
                ds.write(root/'requests'/(ds.gates.digest(body)+'.json'), {'request': body, 'reserved_usd': .02})
                with self.assertRaises(RuntimeError): ds.reconcile()
                with self.assertRaises(RuntimeError): ds.call({'prompt': 'safe'}, None)

    def test_hard_cap_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(ds, 'OUT', root), patch.object(ds, 'LEDGER', root/'cost.csv'), patch.object(ds, 'spent', return_value=14.999):
                with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden in test')):
                    with self.assertRaisesRegex(RuntimeError, 'budget cap'):
                        ds.call({'prompt': 'safe'}, None)
                self.assertFalse((root/'requests').exists())

    def test_explicit_429_rejection_retries_and_charges_only_received_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self.envelope()
            error = urllib.error.HTTPError(ds.URL, 429, 'rate limit', {}, io.BytesIO(b'{"error":{"type":"rate_limit_error","message":"Too many requests"}}'))
            class Opener:
                calls = 0
                def open(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == 1: raise error
                    return io.StringIO(json.dumps(saved['envelope']))
            opener = Opener()
            with patch.object(ds, 'OUT', root), patch.object(ds, 'LEDGER', root/'cost.csv'), patch('urllib.request.build_opener', return_value=opener), patch.dict(ds.os.environ, {'DEEPSEEK_API_KEY': 'offline-test-placeholder'}), patch.object(ds.time, 'sleep'):
                response = ds.call({'prompt': 'safe', 'evidence_ids': ['s']}, None)
                self.assertEqual(opener.calls, 2)
                self.assertEqual(len(ds.rows()), 1)
                self.assertAlmostEqual(ds.spent(), ds.token_cost(saved['envelope']['usage']))
                state = ds.gates.read(root/'requests'/(response['request_sha256']+'.json'))
                self.assertEqual(state['status'], 'RESPONSE_RECEIVED')
                self.assertEqual(state['reserved_usd'], 0)
                self.assertEqual(state['attempts'][0]['http_status'], 429)
                self.assertEqual(len(list((root/'http_errors').glob('*.json'))), 1)

    def test_402_blocks_further_calls_and_preserves_zero_cost_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            error = urllib.error.HTTPError(ds.URL, 402, 'balance', {}, io.BytesIO(b'{"error":{"type":"insufficient_balance_error","message":"Insufficient Balance"}}'))
            class Opener:
                calls = 0
                def open(self, *args, **kwargs):
                    self.calls += 1
                    raise error
            opener = Opener()
            with patch.object(ds, 'OUT', root), patch.object(ds, 'LEDGER', root/'cost.csv'), patch('urllib.request.build_opener', return_value=opener), patch.dict(ds.os.environ, {'DEEPSEEK_API_KEY': 'offline-test-placeholder'}):
                with self.assertRaisesRegex(RuntimeError, 'HTTP 402'): ds.call({'prompt': 'safe'}, None)
                with self.assertRaisesRegex(RuntimeError, 'account balance blocked'): ds.call({'prompt': 'safe'}, None)
                self.assertEqual(opener.calls, 1)
                self.assertEqual(ds.spent(), 0)
                request = ds.gates.read(next((root/'requests').glob('*.json')))
                self.assertEqual(request['status'], 'REJECTED_402_NO_INFERENCE')
                self.assertEqual(request['reserved_usd'], 0)
                ds.reconcile()

    def test_pilot_has_four_distinct_requests_per_arm(self):
        index, unique = [], {}
        for a in ds.ARMS:
            for n in range(12):
                file = a+str(n)
                unique[file] = {'prompt': 'x'*(n+1)}
                index.append({'arm': a, 'input': file, 'case_id': str(n), 'mode': 'all_paths'})
        selection = ds.pilot_selection(index, unique)
        self.assertEqual(len({s['input'] for s in selection}), 20)
        for a in ds.ARMS:
            self.assertEqual(sum(s['arm'] == a for s in selection), 4)


if __name__ == '__main__':
    unittest.main()
