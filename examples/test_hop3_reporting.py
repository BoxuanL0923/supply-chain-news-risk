"""Cache-only coverage/cost arithmetic, request reuse, alignment and corruption."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from news_network_50_2025.config import CONTROLS
from news_network_50_2025.hop3_analysis import cached_hop_report, cached_report_markdown


def build_cache(root):
    def save(name, value):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    index, frozen, rows = [], [], []
    for n, day in enumerate(('2025-10-01', '2025-10-08', '2025-10-30', '2025-11-06')):
        case = 'synthetic-' + str(n)
        row = dict(case_id=case, date=day, split='train' if n < 2 else 'valid', anchor=0., y=float(n),
                   **{c: float(n) for c in CONTROLS}, **{'candidate_exact_h'+str(h): 1 for h in (1,2,3)})
        for mode in ('all_paths', 'continuity_only'):
            for arm in ('OWN', 'H1', 'H2', 'H3', 'H3_GRAPH'):
                radius = 0 if arm == 'OWN' else int(arm[1])
                if mode == 'continuity_only': radius = min(radius, 1)
                hops = [0] + [h for h in range(1, radius+1) if h == 1 or n % 2 == 0]
                evidence = [dict(source_id=f'{case}-h{h}', available_at='2025-09-01T00:00:00Z', hop=h) for h in hops]
                prompt = 'Synthetic prompt\nINPUT\n' + json.dumps(dict(case_id=case, asof=day+'T20:00:00Z', evidence=evidence))
                request = dict(model='qwen3.8-27b', messages=[dict(role='user', content=prompt)])
                key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
                input_name = key+'.json'
                inp_hash = save('inputs/'+input_name, dict(prompt=prompt, evidence_ids=[e['source_id'] for e in evidence]))
                response_hash = save('responses/'+key+'.json', dict(request=request, request_sha256=key,
                    seconds=2., envelope=dict(usage=dict(prompt_tokens=100*len(evidence), completion_tokens=10))))
                score = sum(hops)*.1
                pred = dict(case_id=case, mode=mode, arm=arm, score=score, input=input_name,
                            input_sha256=inp_hash, request_sha256=key, response_file_sha256=response_hash)
                file = case+'-'+mode+'-'+arm+'.json'
                pred_hash = save('predictions/'+file, pred)
                frozen.append(dict(case_id=case, mode=mode, arm=arm, file=file, sha256=pred_hash))
                index.append(dict(case_id=case, mode=mode, arm=arm, input=input_name, input_sha256=inp_hash))
                row[arm if mode == 'all_paths' else 'continuity_'+arm] = score
        rows.append(row)
    freeze_hash = save('prediction_freeze.json', dict(predictions=frozen))
    save('prediction_freeze.sha256.json', dict(sha256=freeze_hash))
    save('input_index.json', dict(records=index))
    pd.DataFrame(rows).to_csv(root/'matched_development_qwen3.8-27b.csv', index=False)


class CachedReportingTests(unittest.TestCase):
    def test_missingness_cost_loss_and_shared_request_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build_cache(root)
            report = cached_hop_report(root)
            self.assertEqual(report['new_model_calls'], 0)
            self.assertEqual(len(report['rows']), 12)
            rows = [r for r in report['rows'] if r['mode']=='all_paths' and r['split']=='valid']
            self.assertEqual([r['share_any_evidence'] for r in rows], [1., .5, .5])
            self.assertEqual([r['candidate_share_any'] for r in rows], [1., 1., 1.])
            self.assertEqual(rows[1]['cost']['model_calls'], 2)
            self.assertEqual(rows[1]['additional_unique_requests'], 1)
            self.assertEqual(rows[1]['cost']['articles_retrieved'], 5)
            self.assertEqual(rows[1]['cost']['tokens'], 520)
            self.assertEqual(rows[1]['additional_request_seconds'], 2.)
            self.assertIsNone(rows[1]['retrieval_seconds'])
            for r in report['rows']:
                if r['mode']=='continuity_only' and r['exact_hop'] > 1:
                    self.assertEqual(r['share_any_evidence'], 0)
                    self.assertEqual(r['additional_unique_requests'], 0)
                    self.assertEqual(r['raw_mse_gain_vs_previous'], 0)
            self.assertLess(report['unique_run_cost']['model_calls'], 40)
            self.assertIn('Nested gain', cached_report_markdown(report))

    def test_corrupt_response_and_incomplete_design_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build_cache(root)
            path = next((root/'responses').glob('*.json'))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): cached_hop_report(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build_cache(root)
            index = json.loads((root/'input_index.json').read_text())
            index['records'].pop()
            (root/'input_index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'design'): cached_hop_report(root)

    def test_evaluation_case_cannot_enter_cached_development_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build_cache(root)
            path = root/'matched_development_qwen3.8-27b.csv'
            frame = pd.read_csv(path); frame.loc[0, 'split'] = 'test'; frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'development cases'): cached_hop_report(root)


if __name__ == '__main__':
    unittest.main()
