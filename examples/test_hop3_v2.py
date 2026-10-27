"""qwen3.8-27b offline discipline and product-continuity tests (unittest)."""
import copy
import unittest
from pathlib import Path
from news_network_50_2025.product_continuity import validate_continuity
from news_network_50_2025.hop3_guard import Guard,check_case,LIVE,EXPOSURE,PROJECT

class ContinuityTests(unittest.TestCase):
    def fixture(self,hops=3):
        chain=list('ABCD')[:hops+1]
        edges={str(i):{'id':str(i),'source':b,'target':a,'type':'supplies','valid_from':'2025-01-01','vintage':'2025-06-06'}
               for i,(a,b) in enumerate(zip(chain,chain[1:]))}
        path={'hops':hops,'path':chain,'vintage':'2025-06-06','steps':[{'edge_id':str(i),'neighbor_role':'supplier'} for i in range(hops)]}
        quote='D supplies silicon to C; C uses silicon in chips supplied to B; B uses chips in servers supplied to A.'
        source={'s':{'body':quote,'available_at':'2025-09-01'}}
        common={'source_id':'s','quote':quote,'available_at':'2025-09-01'}
        items=['servers','chips','silicon'][:hops]
        facts=[{**common,'kind':'edge_product','edge_id':str(i),'supplier':e['source'],'customer':e['target'],'item':items[i]} for i,e in enumerate(edges.values())]
        facts+=[{**common,'kind':'intermediate_flow','entity':chain[i],'edge_ids':[str(i-1),str(i)],'input_item':items[i],'output_item':items[i-1]} for i in range(1,hops)]
        facts+=[{**common,'kind':'affected_item','entity':chain[-1],'item':items[-1]}]
        return path,dict(edges=edges,facts=facts,inspected_sources=source,affected_item=items[-1])
    def test_complete_two_and_three(self):
        for h in (2,3):
            p,k=self.fixture(h);self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'continuity_verified')
    def test_each_intermediate_required(self):
        p,k=self.fixture(); k['facts']=[f for f in k['facts'] if not (f['kind']=='intermediate_flow' and f['entity']=='C')]
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_different_products_need_explicit_transformation(self):
        p,k=self.fixture();k['facts'][3]['input_item']='unrelated plastic'
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_future_source_rejected(self):
        p,k=self.fixture();k['inspected_sources']['s']['available_at']='2025-10-01'
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_invented_quote_rejected(self):
        p,k=self.fixture();k['facts'][0]['quote']='This sentence was never in the inspected source.'
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_topology_and_revenue_alone_insufficient(self):
        p,k=self.fixture();k['facts']=[]
        for e in k['edges'].values():e['revenue_pct']=99
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_shared_search_keywords_are_not_flow_receipts(self):
        p,k=self.fixture()
        k['facts']=[{**k['facts'][0],'kind':'relationship_keywords','keywords':['chips','servers','silicon']}]
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_competitor_and_direction_mixed_rejected(self):
        for kind in ('competitor','reverse'):
            p,k=self.fixture()
            if kind=='competitor':k['edges']['1']['type']='competitor'
            else:k['edges']['1']['source'],k['edges']['1']['target']='B','C'
            self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_expired_edge(self):
        p,k=self.fixture();k['edges']['0']['valid_to']='2025-09-30'
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_wrong_branch_receipt(self):
        p,k=self.fixture();k['facts'][3]['edge_ids']=['0','8']
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')
    def test_endpoint_event_required(self):
        p,k=self.fixture();k['facts']=[f for f in k['facts'] if f['kind']!='affected_item']
        self.assertEqual(validate_continuity(p,'2025-10-01',**k)['status'],'path_only')

class GuardTests(unittest.TestCase):
    def test_subset_freeze_cannot_unlock_labels(self):
        from news_network_50_2025.hop3_guard import verify_freeze
        with self.assertRaises(ValueError):verify_freeze(Path('unused_missing_directory'),[])
    def test_label_and_evaluation_files(self):
        g=Guard()
        for name in ('labels_test.parquet','labels_sealed.parquet','labels_train.parquet','cases.json','pools/2025-11-14.json'):
            with self.assertRaises(PermissionError):g.check_path(LIVE/name)
        g.check_path(LIVE/'pools/2025-11-06.json')
        g.labels_allowed=True
        with self.assertRaises(PermissionError):g.check_path(LIVE/'labels_test.parquet')
    def test_eval_case_and_window(self):
        c={'date':'2025-11-06','split':'valid','asof':'2025-11-06T21:15:00Z','next_sessions':['2025-11-07','2025-11-10','2025-11-11','2025-11-12','2025-11-13']}
        check_case(c)
        for changes in ({'split':'test'},{'date':'2025-11-14'},{'y':1.0},{'next_sessions':c['next_sessions'][:-1]+['2025-11-14']}):
            with self.assertRaises(ValueError):check_case({**c,**changes})
    def test_live_write_block(self):
        g=Guard()
        with self.assertRaises(PermissionError):g.check_path(PROJECT/'code/news_network_50_2025/runtime.py',True)
        with self.assertRaises(PermissionError):g.check_path(EXPOSURE/'graph.json',True)
        with self.assertRaises(PermissionError):g.check_path(EXPOSURE/'cases.json')
    def test_configured_roots_with_nonstandard_directory_names(self):
        import tempfile
        from unittest.mock import patch
        from news_network_50_2025 import hop3_guard as g
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(g,'LIVE',root/'private_panel'),patch.object(g,'EXPOSURE',root/'private_graph'):
                guard=g.Guard()
                for path,writing in [(g.LIVE/'cases.json',False),(g.LIVE/'pools/2025-11-14.json',False),
                                     (g.LIVE/'graph.json',True),(g.EXPOSURE/'cases.json',False)]:
                    with self.assertRaises(PermissionError):guard.check_path(path,writing)
                guard.check_path(g.LIVE/'pools/2025-11-06.json')
    def test_prompt_cap_before_transport(self):
        from news_network_50_2025.runtime import QwenClient
        with self.assertRaises(ValueError):QwenClient(Path('unused_missing_directory')).call('x'*24000)

class InferenceTests(unittest.TestCase):
    def frame(self):
        import numpy as np
        import pandas as pd
        from news_network_50_2025.config import CONTROLS
        rng=np.random.default_rng(871)
        f=pd.DataFrame(rng.normal(size=(300,len(CONTROLS))),columns=CONTROLS)
        f['OWN']=rng.normal(size=300);f['H1']=rng.normal(size=300)
        d2=rng.normal(size=300);d3=rng.normal(size=300)
        f['H2']=f.H1+d2;f['H3']=f.H2+d3
        f['y']=2+.3*f.b+.5*f.OWN+.6*f.H1-.2*d2+.7*d3+rng.normal(scale=.05,size=300)
        f['date']=np.repeat(['2025-10-01','2025-10-08','2025-10-15','2025-10-22','2025-10-30','2025-11-06'],50)
        return f
    def test_known_coefficient_and_cluster_degrees(self):
        from news_network_50_2025.hop3_analysis import regress
        f=self.frame();result=regress(f)
        g=result['coefficients']['H3_minus_H2']
        self.assertTrue(g['identified']);self.assertAlmostEqual(g['estimate'],.7,delta=.015)
        self.assertEqual(g['date_cluster']['df'],5)
        self.assertGreater(result['gamma3_smallest_symmetric_equivalence_bound']['HC3'],abs(g['estimate']))
        valid=regress(f.iloc[200:],f.iloc[:200])
        self.assertLess(valid['nested'][-1]['valid_mse'],.01)
    def test_zero_increment_is_unidentified(self):
        from news_network_50_2025.hop3_analysis import regress
        f=self.frame();f['H3']=f.H2
        result=regress(f)
        self.assertIsNone(result['coefficients']['H3_minus_H2']['estimate'])
        self.assertIsNone(result['gamma3_smallest_symmetric_equivalence_bound']['HC3'])
        self.assertIsNone(result['power']['grid'][0]['HC3_cases_for_80pct_power'])

class HostFallbackTests(unittest.TestCase):
    def test_exact_radius_survives_old_enumeration_cap(self):
        from news_exposure_2025.graph import Graph
        from news_network_50_2025.hop3_shortest import shortest_pool_paths
        data={'nodes':{n:{'name':n,'permnos':[i]} for i,n in enumerate('ABCDE')},
              'targets':{'A':['A']},'audit':{'graph_date':'2025-06-06'},
              'edges':[{'id':str(i),'source':a,'target':b,'type':'supplies','revenue_pct':None}
                       for i,(a,b) in enumerate([('A','B'),('A','C'),('B','D'),('C','E'),('E','D')])]}
        graph=Graph(data)
        old=graph.pool_paths('A',3,limit=1)
        self.assertEqual(next(p['hops'] for p in old if p['entity']=='E'),3)
        exact=shortest_pool_paths(graph,'A',3)
        self.assertEqual(next(p['hops'] for p in exact if p['entity']=='E'),2)
        self.assertEqual({p['entity'] for p in exact},set('BCDE'))
    def test_shortest_witness_prefers_mapped_intermediate(self):
        from news_exposure_2025.graph import Graph
        from news_network_50_2025.hop3_shortest import shortest_pool_paths
        data={'nodes':{n:{'name':n,'permnos':[] if n=='B' else [i]} for i,n in enumerate('ABCE')},
              'targets':{'A':['A']},'audit':{'graph_date':'2025-06-06'},
              'edges':[{'id':str(i),'source':a,'target':b,'type':'supplies','revenue_pct':None}
                       for i,(a,b) in enumerate([('A','B'),('A','C'),('B','E'),('C','E')])]}
        rows=shortest_pool_paths(Graph(data),'A')
        self.assertEqual(next(r['path'] for r in rows if r['entity']=='E'),['A','C','E'])
    def test_support_reads_unlock_exact_prefixes(self):
        from news_network_50_2025.runtime import QwenClient
        relations=[{'entity':n,'permnos':[i],'hops':i,'path':list('ABCD')[:i+1],
                    'steps':[{'neighbor_role':'supplier'}]*i,'vintage':'2025-06-06'} for i,n in enumerate('BCD',1)]
        docs={n:{'source_id':n,'ids':[i],'headline':'Dated operating fact','body':'A dated inspected source passage.',
                 'available_at':'2025-09-01'} for i,n in reversed(list(enumerate('BCD',1)))}
        cap={'case':{'asof':'2025-10-01'},'network':True,'documents':docs,'relations':relations,
             'calendar':{},'controls':{},'coverage':{}}
        client=QwenClient(Path('unused_missing_directory'))
        result=client.native('',cap)['host_evidence']
        self.assertEqual({e['source_id']:e['hop'] for e in result['events']},{'B':1,'C':2,'D':3})
        self.assertIn(['A','B','C'],result['trace']['inspected_paths'])
        cap['documents']={k:v for k,v in docs.items() if k!='B'}
        self.assertEqual(client.native('',cap)['host_evidence']['events'],[])
    def test_own_excerpt_is_identical_across_radius_arms(self):
        import json
        from news_network_50_2025.hop3_v2 import make_prompt
        from news_network_50_2025.config import CONTROLS
        c={'case_id':'toy','ticker':'A','company':'Firm A','asof':'2025-10-01','next_sessions':[], 'b':-8}
        p={'excerpt_chars':100,'controls':dict.fromkeys(CONTROLS,0),'calendar':{'status':'unknown'}}
        own={'source_id':'own','headline':'Own report','body':'Own fact '*100,'available_at':'2025-09-30','hop':0}
        net={'source_id':'net','headline':'Neighbor report','body':'Neighbor fact '*100,'available_at':'2025-09-30',
             'hop':3,'relation_pattern':'supplier_to_supplier_to_supplier','continuity_status':'path_only','graph':{'hub':False}}
        a=json.loads(make_prompt(c,p,[own]).split('\nINPUT\n')[1])
        b=json.loads(make_prompt(c,p,[own,net],True).split('\nINPUT\n')[1])
        self.assertEqual(a['evidence'][0],b['evidence'][0])

if __name__=='__main__':unittest.main()
