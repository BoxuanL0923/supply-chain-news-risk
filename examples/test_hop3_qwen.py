"""Offline Qwen-only transport, freeze, and failure-path checks."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from news_network_50_2025 import hop3_qwen as q


def response(score=.2, ids=None):
    return {'model': q.MODEL, 'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'score': score, 'evidence_ids': ids or [], 'reason': 'Dated evidence only.'})}}],
        'usage': {'prompt_tokens': 120, 'completion_tokens': 30}}


def saved(raw=None):
    return {'server_model': q.MODEL, 'envelope': raw or response()}


class QwenTests(unittest.TestCase):
    def test_fresh_preparation_without_deepseek_history(self):
        from datetime import date, timedelta
        from news_network_50_2025.hop3_v2 import make_prompt, POLICY, GRAPH_POLICY
        from news_network_50_2025.config import CONTROLS
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); prep=root/'hop_ablation_v2'; out=prep/'qwen'
            with patch.object(q,'ROOT',root),patch.object(q,'PREP',prep),patch.object(q,'OUT',out),patch.object(q.gates,'resources'):
                cases=[]; records=[]
                for day in q.gates.DATES:
                    sessions=[]; d=date.fromisoformat(day)
                    while len(sessions)<5:
                        d+=timedelta(days=1)
                        if d.weekday()<5:sessions.append(d.isoformat())
                    for i in range(50):
                        c={'case_id':f'TOY{i}:{day}','ticker':f'TOY{i}','company':'Synthetic firm',
                           'date':day,'split':q.gates.SPLITS[day],'asof':day+'T21:00:00Z','next_sessions':sessions,'b':-8}
                        cases.append(c)
                        prompt=make_prompt(c,{'excerpt_chars':100,'controls':dict.fromkeys(CONTROLS,0),'calendar':{}},[])
                        key=q.gates.digest(prompt); name=key+'.json'
                        q.write(prep/'inputs'/name,{'prompt':prompt,'prompt_sha256':key,'evidence_ids':[],'excerpt_chars':100})
                        records.extend({'case_id':c['case_id'],'mode':m,'arm':a,'input':name,
                                        'input_sha256':q.gates.sha(prep/'inputs'/name)} for m in q.MODES for a in q.ARMS)
                q.write(root/'cases.json',{'cases':cases})
                q.write(root/'data_manifest.json',{'prefix_support_complete':True,'outputs':{}})
                for name in ('scope_audit.json','continuity_summary.json','continuity_source_audit.json'):
                    q.write(prep/name,{'records':[]})
                q.write(prep/'full_graph_radius_audit.json',{'mismatches':[],
                        'data_manifest_sha256':q.gates.sha(root/'data_manifest.json')})
                q.write(prep/'PROTOCOL.json',{'fixed_policy':POLICY,'graph_policy':GRAPH_POLICY,'cases_hash':q.gates.digest(cases)})
                q.write(prep/'input_index.json',{'records':records})
                q.write(prep/'OFFLINE_READY.json',{'protocol_sha256':q.gates.sha(prep/'PROTOCOL.json'),
                        'inputs_sha256':q.gates.sha(prep/'input_index.json')})
                index,unique=q.prepare()
                self.assertEqual(len(index),3000)
                self.assertEqual(len(unique),300)
                self.assertEqual(q.gates.read(out/'DEEPSEEK_PRESERVED.json')['distinct_responses'],0)
                self.assertIsNone(q.gates.read(out/'MODEL_SWITCH.json')['from'])
                self.assertFalse((out/'responses').exists())

    def test_model_context_and_no_thinking(self):
        b = q.request_body('short')
        self.assertEqual(b['model'], 'qwen3.8-27b')
        self.assertEqual(b['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(b['max_tokens'], 7000)
        self.assertEqual(b['seed'], 20261026)
        self.assertLess(len('short'.encode()) + 1024 + b['max_tokens'], 32768)
        with self.assertRaises(ValueError):
            q.request_body('x' * (24000 - 1024))

    def test_finite_score_and_final_only(self):
        raw = response()
        raw['choices'][0]['message']['reasoning_content'] = '{"score":999}'
        self.assertEqual(q.parse_answer(saved(raw), [])['score'], .2)
        for score in (True, float('nan'), float('inf'), '0.2'):
            with self.assertRaises(ValueError):
                q.parse_answer(saved(response(score)), [])

    def test_wrong_model_and_truncation(self):
        raw = response()
        raw['model'] = 'deepseek-v4-pro'
        with self.assertRaises(ValueError):
            q.parse_answer(saved(raw), [])
        raw = response()
        raw['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            q.parse_answer(saved(raw), [])

    def test_actual_context_usage(self):
        for field, value in [('prompt_tokens', 24000), ('completion_tokens', 7001), ('prompt_tokens', -1)]:
            raw = response()
            raw['usage'][field] = value
            with self.assertRaises(ValueError):
                q.parse_answer(saved(raw), [])

    def test_citation_quality_never_drops_score(self):
        ids = ['RTRS:202510:10:abc']
        answer = q.parse_answer(saved(response(ids=['RTRS:202510:10', 'invented'])), ids)
        self.assertEqual(answer['score'], .2)
        self.assertEqual(answer['evidence_ids'], ids)
        self.assertEqual(answer['unresolved_evidence_ids'], ['invented'])
        self.assertTrue(answer['citation_validation_issues'])

    def test_ambiguous_alias_is_not_verified(self):
        answer = q.parse_answer(saved(response(ids=['RTRS:202510:10'])), ['RTRS:202510:10:abc', 'RTRS:202510:10:def'])
        self.assertEqual(answer['evidence_ids'], [])
        self.assertEqual(answer['unresolved_evidence_ids'], ['RTRS:202510:10'])

    def test_raw_cache_before_parse_and_no_duplicate_call(self):
        with tempfile.TemporaryDirectory() as folder:
            client = q.QwenClient(Path(folder))
            with patch.object(q, 'check_server_log'), patch.object(q.gates, 'resources'), patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(response()).encode())) as op:
                first = client.call('short')
                second = client.call('short')
            self.assertEqual(first, second)
            self.assertEqual(op.call_count, 1)
            self.assertEqual(len(list((Path(folder) / 'raw_responses').glob('*'))), 1)

    def test_malformed_raw_json_still_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            client = q.QwenClient(Path(folder))
            with patch.object(q, 'check_server_log'), patch.object(q.gates, 'resources'), patch.object(client.opener, 'open', return_value=io.BytesIO(b'invalid JSON')):
                with self.assertRaises(ValueError):
                    client.call('short')
            self.assertEqual(next((Path(folder) / 'raw_responses').glob('*')).read_bytes(), b'invalid JSON')
            with self.assertRaises(RuntimeError):
                client.call('short')

    def test_connection_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            client = q.QwenClient(Path(folder))
            with patch.object(q, 'check_server_log'), patch.object(q.gates, 'resources'), patch.object(client.opener, 'open', side_effect=urllib.error.URLError('connection refused')) as op:
                with self.assertRaises(urllib.error.URLError):
                    client.call('short')
                with self.assertRaises(RuntimeError):
                    client.call('short')
            self.assertEqual(op.call_count, 1)

    def test_raw_response_tamper_blocks_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            client = q.QwenClient(Path(folder))
            with patch.object(q, 'check_server_log'), patch.object(q.gates, 'resources'), patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(response()).encode())):
                client.call('short')
            next((Path(folder) / 'raw_responses').glob('*')).write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                client.call('short')

    def test_cuda_detection(self):
        with patch.object(q, 'server_log_tail', return_value=(1, 'CUDA error: out of memory')):
            with self.assertRaises(RuntimeError):
                q.check_server_log()

    def test_evaluation_and_incomplete_label_gate(self):
        guard = q.QwenGuard()
        with self.assertRaises(ValueError):
            q.load_labels(guard, 'evaluation')
        with patch.object(q.gates, 'read', return_value={'cases': []}), patch.object(q.gates, 'verify_freeze', side_effect=ValueError('incomplete')):
            with self.assertRaises(ValueError):
                q.load_labels(guard, 'train')
        self.assertFalse(guard.labels_allowed)

    def test_done_written_after_failure_and_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            done = out / 'done.json'
            with patch.object(q, 'OUT', out), patch.object(q, 'DONE', done), patch.object(q, 'SERVER_LOG', out / 'missing.log'), patch.object(q, 'predict', side_effect=RuntimeError('connection refused')), patch.object(q, 'failure_report') as failure_report:
                with self.assertRaises(RuntimeError):
                    q.run(q.QwenGuard())
            result = q.gates.read(done)
            self.assertEqual(result['status'], 'failure')
            self.assertEqual(result['counts']['responses'], 0)
            self.assertTrue((out / 'RUN_FAILURE.json').exists())
            failure_report.assert_called_once()

    def test_failure_report_has_no_label_or_fake_mse(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            case = {'case_id':'TOY:2025-10-30','split':'valid','date':'2025-10-30',
                    'asof':'2025-10-30T21:00:00Z','next_sessions':['2025-10-31','2025-11-03','2025-11-04','2025-11-05','2025-11-06']}
            with patch.object(q, 'OUT', out), patch.object(q, 'TARGET', out/'published.md'), patch.object(q.gates, 'read', return_value={'cases':[case]}):
                q.failure_report({'message':'truncated'},q.QwenGuard())
            summary = json.loads((out/'summary.json').read_text())
            self.assertIsNone(summary['raw_residual_mse'])
            self.assertFalse(summary['train_valid_labels_opened'])
            header = (out/'matched_validation.csv').read_text().splitlines()[0].split(',')
            self.assertNotIn('y',header)
            self.assertTrue((out/'REPORT.md').exists())

    def test_post_freeze_failure_reports_actual_label_state(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            guard = q.QwenGuard();guard.labels_allowed=True
            with patch.object(q,'OUT',out),patch.object(q,'TARGET',out/'published.md'):
                q.failure_report({'message':'analysis failed'},guard)
            summary = json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['train_valid_labels_opened'])
            self.assertEqual(summary['status'],'ANALYSIS_FAILURE_AFTER_FREEZE')


if __name__ == '__main__':
    unittest.main()
