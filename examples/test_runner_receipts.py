"""Actual Runner/Toolkit wiring, gate execution, and static/runtime label guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples.test_shared_supplier import capability, event
from examples import gate_skills_synthetic as toy
from news_network_50_2025.graph_router import digest
from news_network_50_2025.label_firewall import boundary, check_payload, development_labels, static_verify
from news_network_50_2025.runner_receipts import training_gate, verify_case_trace
from news_network_50_2025.toolkit import Toolkit
from news_network_50_2025.workflow import Runner
from news_network_50_2025.optimize import choose_with_gates


class FakeClient:
    def __init__(self):
        self.native_calls = 0
        self.prompts = []

    def native(self, prompt, capfile):
        self.native_calls += 1
        cap = json.loads(Path(capfile).read_text())
        kit = Toolkit(cap)
        if cap['network']:
            kit.call('exposure_paths')
            kit.call('search_events', query='outage')
        ids = ['supplier', 'peer'] if cap['network'] else ['own']
        for sid in ids:
            kit.call('inspect_event', source_id=sid)
        return dict(answer=json.dumps(dict(events=[event(s) for s in ids], missing=[])),
                    request_sha256='fake-native', trace=dict(calls=kit.trace, reads=kit.reads,
                    inspected_paths=list(kit.inspected_paths), router_events=kit.router_events))

    def call(self, prompt, **kwargs):
        self.prompts.append(prompt)
        evidence = json.loads(prompt.split('\nINPUT\n', 1)[1].split('\nReturn JSON', 1)[0])['evidence']
        ids = [e['source_id'] for arm in evidence.values() for e in arm['events']]
        return dict(answer=json.dumps(dict(score=0.1, evidence_ids=ids[:1], reason='Synthetic test')),
                    request_sha256='fake-score')


def runner_fixture(directory, *, enabled=True, documents=True):
    cap = capability()
    case = dict(cap['case'], date='2020-02-01', b=-5., next_sessions=[], state={'numerical_aids': {}})
    own = dict(cap['documents']['supplier'], source_id='own', ids=[1], story_id='own-story')
    docs = dict(cap['documents'], own=own) if documents else {}
    pool = dict(case=case, controls={}, calendar={}, coverage={}, own_packet_hash='own-hash',
                own_ids=['own'] if documents else [], network_ids=['supplier', 'peer'] if documents else [],
                history_ids=[], relations=cap['relations'], graph_firm_id='i', graph_snapshot=cap['graph_snapshot'])
    saved = dict(pools={case['case_id']: pool}, documents=docs)
    client = FakeClient()
    return Runner(Path(directory), client, [case], {case['case_id']: -5.}, 'test-manifest',
                  enable_graph_router=enabled), case, saved, client


class RunnerReceiptTests(unittest.TestCase):
    def test_preflight_keeps_evaluation_bytes_sealed_even_for_hash_checks(self):
        from news_network_50_2025.run import preflight
        from news_event_risk_data_2025 import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [dict(case_id=str(n), ticker='TOY') for n in range(600)]
            (root/'cases.json').write_text(json.dumps(cases))
            (root/'labels_test.parquet').write_text('unopened evaluation canary')
            (root/'data_manifest.json').write_text(json.dumps(dict(outputs={
                'cases.json': file_hash(root/'cases.json'), 'labels_test.parquet': 'seal-from-preparation'})))
            with boundary(training=True) as access:
                result, manifest = preflight(root, True)
            self.assertEqual(len(result), 600)
            self.assertEqual(access['accesses'], [])
            self.assertIn('gate4_skill_sources', manifest)
            protocol = json.loads((root/'PROTOCOL.json').read_text())
            self.assertEqual(protocol['gate4']['sealed_outputs_not_opened'], {'labels_test.parquet': 'seal-from-preparation'})

    def test_enabled_execute_finishes_before_evaluation_loader(self):
        from news_network_50_2025 import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, case, saved, client = runner_fixture(directory)
            cases = [dict(case, case_id=split+'-case', split=split) for split in ('train', 'valid', 'test')]
            pool = saved['pools'][case['case_id']]
            saved['pools'] = {c['case_id']: dict(pool, case=c) for c in cases}
            manifest = {'synthetic': True}
            (root/'toolkit_review.json').write_text(json.dumps(dict(approved=True, manifest=digest(manifest))))
            (root/'labels_test.parquet').write_text('unopened evaluation canary')
            client.budget = SimpleNamespace(start=lambda: None, remaining_seconds=lambda: 0)
            client.limits = {}
            def matched_only_train(*args):
                self.assertEqual(args[-1], 'train')
                return 'synthetic training frame'
            with patch.object(run, 'preflight', return_value=(cases, manifest)), \
                 patch.object(run, 'acceptance'), patch.object(run, 'resume_manifest', return_value=manifest), \
                 patch.object(run, 'anchor', return_value={c['case_id']: -5. for c in cases}), \
                 patch.object(run, 'NativeClient', return_value=client), patch.object(run, 'resource_limits', return_value={}), \
                 patch.object(run, 'matched', side_effect=matched_only_train) as loader, \
                 patch('news_network_50_2025.metrics.fit', return_value={'coefficient': [0.]}), \
                 patch('news_network_50_2025.workflow.pool_file', return_value=saved):
                run.execute(root, enable_graph_router=True)
            self.assertEqual(loader.call_count, 1)
            receipt = json.loads((root/'gate4_verification.json').read_text())
            self.assertTrue(receipt['runtime_verified'])
            self.assertFalse(receipt['test_labels_accessed'])
            self.assertFalse((root/'summary.json').exists())

    def test_training_selection_cannot_promote_without_gate_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [dict(case_id='v1', date='2020-01-01', split='valid'),
                     dict(case_id='v2', date='2020-01-08', split='valid')]
            runner = Runner(Path(directory), None, cases, {}, 'synthetic', enable_graph_router=True)
            candidates = [dict(policy='Seed policy', scores=dict(v1=10., v2=10.)),
                          dict(policy='Candidate policy', scores=dict(v1=0., v2=0.))]
            chosen, receipt = choose_with_gates(runner, candidates, dict(v1=0., v2=0.), cases, cases)
            self.assertEqual(chosen['policy'], 'Seed policy')
            self.assertTrue(receipt['gate_blocked'])
            self.assertEqual(receipt['acceptance_gate_receipt']['decision'], 'NOT_RUN')

    def test_real_runner_retrieval_selection_trace_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, case, saved, client = runner_fixture(directory)
            with patch('news_network_50_2025.workflow.pool_file', return_value=saved):
                result = runner.score(case, 'Fixed synthetic policy', True)
                calls = client.native_calls
                repeated = runner.score(case, 'Fixed synthetic policy', True)
            self.assertEqual(calls, 2)
            self.assertEqual(client.native_calls, calls)
            self.assertEqual(result, repeated)
            trace = result['skill_python_trace']
            self.assertTrue(verify_case_trace(trace))
            names = [e['event'] for e in trace['events']]
            self.assertIn('router_decision', names)
            self.assertIn('tool_call', names)
            self.assertIn('select_context', names)
            routes = [e['receipt'] for e in trace['events'] if e['event'] == 'router_decision']
            self.assertEqual(max(p['hops'] for p in routes[0]['paths']), 1)
            self.assertEqual(max(p['hops'] for p in routes[-1]['paths']), 2)
            self.assertFalse(trace['test_labels_accessed'])
            bad = copy.deepcopy(trace); bad['case_id'] = 'other'; bad['sha256'] = digest({k:v for k,v in bad.items() if k != 'sha256'})
            with self.assertRaises(ValueError): verify_case_trace(bad)
            self.assertTrue(list((Path(directory)/'case_traces').glob('*.json')))

    def test_empty_pool_still_calls_python_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, case, saved, _ = runner_fixture(directory, documents=False)
            with patch('news_network_50_2025.workflow.pool_file', return_value=saved):
                result = runner.score(case, 'Synthetic policy')
            self.assertTrue(verify_case_trace(result['skill_python_trace']))

    def test_enabled_mode_rejects_legacy_snapshot_and_nested_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, case, saved, _ = runner_fixture(directory)
            saved['pools'][case['case_id']].pop('graph_snapshot')
            with patch('news_network_50_2025.workflow.pool_file', return_value=saved):
                with self.assertRaisesRegex(ValueError, 'snapshot'): runner.capability(case, True)
                runner.enable_graph_router = False
                with patch.dict('os.environ', SC_ENABLE_GRAPH_ROUTER='1'):
                    legacy = runner.capability(case, True)
                    self.assertNotIn('graph_snapshot', legacy)
                    self.assertFalse(Toolkit(legacy).router_receipts)
            with self.assertRaises(ValueError): Toolkit(dict(capability(), controls={'nested': {'y': 1}}))
            runner.enable_graph_router = True
            with self.assertRaises(ValueError): runner.score(dict(case, state={'y': 10}), 'policy')

    def test_runtime_canaries_and_training_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('labels_test.parquet', 'evaluation_labels.csv', 'labels_train.parquet'):
                (root/name).write_text('canary')
            for name in ('labels_test.parquet', 'evaluation_labels.csv', 'labels_train.parquet'):
                with boundary(), self.assertRaisesRegex(ValueError, 'Label file'):
                    (root/name).read_bytes()
            with boundary(training=True):
                with self.assertRaises(ValueError): development_labels(root, 'test')
                self.assertEqual((root/'labels_train.parquet').read_text(), 'canary')
                with boundary(), self.assertRaises(ValueError): (root/'labels_train.parquet').read_bytes()
            with self.assertRaises(ValueError): check_payload({'state': [{'outcome': 1}]})
            self.assertTrue(static_verify()['verified'])

    def test_static_verifier_detects_injected_label_access(self):
        from news_network_50_2025 import label_firewall
        root = Path(label_firewall.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for name in ('graph_projection.py', 'graph_router.py', 'graph_skills.py', 'domain_signals.py',
                         'runner_receipts.py', 'toolkit.py', 'workflow.py', 'optimize.py', 'context_selection.py', 'run.py'):
                (dest/name).write_bytes((root/name).read_bytes())
            with (dest/'workflow.py').open('a') as stream:
                stream.write('\ndef injected(row):\n    return row["y"]\n')
            with self.assertRaisesRegex(ValueError, 'Static label firewall'): static_verify(dest)

    def test_training_gates_execute_and_bind_to_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs = root/'inputs'; toy.write(inputs)
            config = json.loads((inputs/'acceptance.json').read_text())
            policy = 'Synthetic proposed policy'
            cases = [dict(case_id='training-case', split='train')]
            bundle = dict(policy_sha256=digest(policy), case_ids=['training-case'], config=config)
            path = inputs/'bundle.json'; path.write_text(json.dumps(bundle))
            receipt = training_gate(root, policy, cases, path)
            self.assertTrue(receipt['accepted'])
            self.assertEqual(set(receipt['events'][-1]['receipt']['payload']['receipts']),
                             {'validation', 'sign_stability', 'industry_peer', 'rewire_placebo'})
            self.assertEqual(training_gate(root, policy, cases, path)['binding'], receipt['binding'])
            runner = Runner(root, None, cases, {}, 'test', enable_graph_router=True,
                            acceptance_bundles={digest(policy): path})
            self.assertTrue(runner.accept_candidate(policy, cases)['accepted'])
            # A fresh process must restore/recheck gate receipts on cache reuse.
            decision_dir = root/'optimization'/'NET'; decision_dir.mkdir(parents=True)
            (decision_dir/'decision.json').write_text(json.dumps(dict(policy=policy, gate4_enabled=True,
                selection=dict(changed=True, acceptance_gate_receipt=receipt))))
            fresh = Runner(root, None, cases, {}, 'test', enable_graph_router=True,
                           acceptance_bundles={digest(policy): path})
            from news_network_50_2025.optimize import optimize
            self.assertEqual(optimize(fresh, cases, True, None)['policy'], policy)
            self.assertTrue(fresh.policy_receipts[digest(policy)]['accepted'])
            self.assertFalse(runner.accept_candidate('Missing bundle', cases)['accepted'])
            with self.assertRaises(ValueError): training_gate(root, policy, [dict(case_id='training-case', split='test')], path)
            with self.assertRaises(ValueError): training_gate(root, 'Other policy', cases, path)
            # A changed input cannot reuse a previously passing ledger row.
            import csv
            with (inputs/'validation.csv').open('w', newline='') as stream:
                rows = toy.loss_rows(12.); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            self.assertFalse(training_gate(root, policy, cases, path)['accepted'])
            with self.assertRaisesRegex(ValueError, 'inputs changed'): runner.accept_candidate(policy, cases)


if __name__ == '__main__':
    unittest.main()
