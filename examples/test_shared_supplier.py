"""Fork direction, source identity/clocks and conservative finance interpretation."""
import copy
import unittest

from news_network_50_2025.domain_signals import finance_graph_signal
from news_network_50_2025.graph_projection import bound_paths, project_path
from news_network_50_2025.context_selection import select_context, _model_metadata
from news_network_50_2025.toolkit import Toolkit
from news_network_50_2025.workflow import validate_evidence
from news_exposure_2025.graph import Graph


def capability():
    edges = [dict(id='si', source='s', target='i', type='supplies', vintage='2020-01-01',
                  valid_from='2020-01-01', revenue_pct=None),
             dict(id='sj', source='s', target='j', type='supplies', vintage='2020-01-01',
                  valid_from='2020-01-01', revenue_pct=None)]
    relations = [dict(entity='s', permnos=[2], path=['i', 's'], hops=1, vintage='2020-01-01',
                      steps=[dict(edge_id='si', neighbor_role='supplier')]),
                 dict(entity='j', permnos=[3], path=['i', 's', 'j'], hops=2, vintage='2020-01-01',
                      steps=[dict(edge_id='si', neighbor_role='supplier'), dict(edge_id='sj', neighbor_role='customer')])]
    docs = {name: dict(source_id=name, ids=[n], headline='Synthetic input outage',
            available_at='2020-01-30T12:00:00Z', body='Synthetic source describes a production interruption.',
            story_id='story-' + name) for name, n in [('supplier', 2), ('peer', 3)]}
    return dict(case=dict(case_id='toy-fork', ticker='TOY', company='Toy', asof='2020-02-01T12:00:00Z',
                          split='train'), network=True, documents=docs, relations=relations,
                controls={}, calendar={}, coverage={}, graph_firm_id='i', enable_graph_router=True,
                graph_snapshot=dict(nodes=['i', 's', 'j'], edges=edges, vintage='2020-01-01',
                                    monthly_universe={'2020-02': ['i', 's', 'j']}))


def event(sid):
    return dict(source_id=sid, page=0, fact='Synthetic fact at the source endpoint',
                interpretation='Retrieval hypothesis', qualifications='Product evidence missing',
                event_type='supplier disruption', status='ongoing')


class SharedSupplierTests(unittest.TestCase):
    def test_explicit_input_fork_and_graph_projection(self):
        cap = capability()
        path = project_path(cap['relations'][1], cap['case']['asof'], cap['graph_snapshot'])
        self.assertEqual(path['role_motif'], 'shared_supplier_fork')
        self.assertEqual(path['fork']['input_edges'], [['s', 'i'], ['s', 'j']])
        data = dict(nodes={n: dict(permnos=[p]) for n, p in [('i', 1), ('s', 2), ('j', 3)]},
                    targets={'TOY': ['i']}, edges=cap['graph_snapshot']['edges'])
        peer = next(p for p in Graph(data).paths('TOY', hops=2) if p['entity'] == 'j')
        self.assertEqual(peer['role_motif'], 'shared_supplier_fork')
        self.assertEqual([s['source'] for s in peer['steps']], ['s', 's'])
        for n in (0, 1):
            bad = copy.deepcopy(cap['graph_snapshot'])
            e = bad['edges'][n]
            e['source'], e['target'] = e['target'], e['source']
            with self.assertRaisesRegex(ValueError, 'direction'):
                project_path(cap['relations'][1], cap['case']['asof'], bad)

    def test_peer_news_is_not_supplier_news(self):
        cap = capability()
        peer = bound_paths(cap['relations'], cap['documents']['peer'], cap['case']['asof'], cap['graph_snapshot'])
        supplier = bound_paths(cap['relations'], cap['documents']['supplier'], cap['case']['asof'], cap['graph_snapshot'])
        self.assertEqual([(p['entity'], p['news_binding']) for p in peer], [('j', 'peer_endpoint')])
        self.assertEqual([p['entity'] for p in supplier], ['s'])
        kit = Toolkit(cap)
        self.assertIn('error', kit.call('inspect_event', source_id='peer'))
        kit.call('exposure_paths')
        kit.call('search_events', query='outage')
        kit.call('inspect_event', source_id='supplier')
        kit.call('inspect_event', source_id='peer')
        trace = dict(calls=kit.trace, reads=kit.reads, inspected_paths=list(kit.inspected_paths))
        forged = dict(event('peer'), source_endpoint_bindings=[dict(entity='s')], graph_statistics={'ra': 999})
        row = validate_evidence(dict(events=[forged], missing=[]), trace, cap)['events'][0]
        self.assertEqual(row['source_endpoint_bindings'], [dict(entity='j', news_binding='peer_endpoint')])
        signal = finance_graph_signal(dict(row, _scope='network_supplement'), cap['case']['asof'])
        self.assertEqual(signal['role_motif'], 'shared_supplier_fork')
        self.assertEqual(signal['path_pattern'], 'shared_supplier_fork')
        self.assertEqual(signal['graph_role'], 'shared_supplier_peer')
        self.assertEqual(signal['finance_hypothesis'], 'common_input_possible_dated_supplier_disruption')
        self.assertEqual(signal['channel_status'], 'path_only')
        from news_network_50_2025.product_continuity import validate_continuity
        serial = validate_continuity(cap['relations'][1], cap['case']['asof'],
                                     edges={e['id']: e for e in cap['graph_snapshot']['edges']})
        self.assertEqual(serial['role_motif'], 'shared_supplier_fork')
        self.assertEqual(serial['status'], 'path_only')
        self.assertNotIn('direction_change', serial['reasons'])
        self.assertEqual(serial['intermediates'], [])

    def test_graph_and_news_clocks_fail_closed(self):
        for key in ('vintage', 'valid_from'):
            cap = capability()
            cap['graph_snapshot']['edges'][1][key] = cap['case']['asof']
            with self.assertRaises(ValueError): Toolkit(cap)
        cap = capability()
        cap['graph_snapshot']['edges'][1]['valid_to'] = cap['case']['asof']
        with self.assertRaises(ValueError): Toolkit(cap)
        cap = capability()
        cap['relations'][1]['vintage'] = cap['case']['asof']
        with self.assertRaises(ValueError): Toolkit(cap)
        for available in ('2020-02-01T12:00:00Z', '2020-02-01T08:00:00-05:00'):
            cap = capability(); cap['documents']['peer']['available_at'] = available
            with self.assertRaises(ValueError): Toolkit(cap)

    def test_duplicate_story_once_and_missing_product_stays_path_only(self):
        cap = capability(); kit = Toolkit(cap)
        kit.call('exposure_paths'); kit.call('search_events', query='outage')
        kit.call('inspect_event', source_id='supplier'); kit.call('inspect_event', source_id='peer')
        trace = dict(calls=kit.trace, reads=kit.reads, inspected_paths=list(kit.inspected_paths))
        rows = validate_evidence(dict(events=[event('peer'), event('peer')], missing=[]), trace, cap)['events']
        self.assertEqual(len(rows), 1)
        row = dict(rows[0], _scope='network_supplement', channel_receipt={'status': 'conditional_channel'})
        signal = finance_graph_signal(row, cap['case']['asof'])
        self.assertEqual(signal['evidence_gap'], 'supplier_event_and_focal_product_exposure')
        self.assertEqual(signal['channel_status'], 'path_only')
        selected, receipt = select_context([row, dict(row, source_id='duplicate')], ['peer', 'duplicate'],
                                          asof=cap['case']['asof'], max_events=8)
        self.assertEqual(len(selected), 1)
        self.assertEqual(receipt['excluded_duplicate_stories'], ['duplicate'])

    def test_path_multiplicity_is_never_dependence_or_redundancy(self):
        cap = capability()
        row = dict(source_id='peer', available_at=cap['documents']['peer']['available_at'],
                   _scope='network_supplement', event_type='supplier disruption',
                   verified_paths=[cap['relations'][1]])
        one = finance_graph_signal(row, cap['case']['asof'])
        many = finance_graph_signal(dict(row, verified_paths=[cap['relations'][1]] * 40), cap['case']['asof'])
        self.assertEqual({k:v for k,v in one.items() if k != 'path_count_bin'},
                         {k:v for k,v in many.items() if k != 'path_count_bin'})
        self.assertFalse(any('dependence' in k or 'redundancy' in k for k in many))
        metadata, _ = _model_metadata([row], cap['case']['asof'])
        self.assertEqual(metadata[0]['news_binding'], 'peer_endpoint')
        self.assertNotIn('s', {v for v in metadata[0].values() if isinstance(v, str)})


if __name__ == '__main__':
    unittest.main()
