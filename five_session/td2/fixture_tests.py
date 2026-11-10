"""Invented fixtures only: unittest emits no private source payload."""
import copy
import unittest
import numpy as np
import pandas as pd
from engine import graph_at,exact_paths,news_pool,exposure,feature_packet,day_clock,ridge,score
from selectors import gate,choice_gate,ordering,validate_slot,noul_gate
from spec import NAMES,SLOT_SCHEMAS
from statistics import clustered_mean,paired

class Fixture(unittest.TestCase):
    def setUp(self):
        self.origin=pd.Timestamp('2020-06-12'); self.cutoff=pd.Timestamp('2020-06-13 03:59:59')
        self.edges=[dict(id=str(j),a=j,b=j+1,start='2020-01-01',end=None,
            available_at=day_clock('2020-01-01'),rownum=j,typ='CUSTOMER') for j in range(1,4)]
        self.story=dict(alt_id='synthetic',date=pd.Timestamp('2020-06-11'),
            available_at=day_clock('2020-06-11'),firms=[2,3,4],scores=score('debt shortage demand production cash'))
        self.chars={n:float(i+1) for i,n in enumerate(NAMES)}
        self.avail={n:day_clock('2020-05-01') for n in NAMES}
    def args(self):
        return (1,self.cutoff,self.origin,self.chars,self.avail,
            graph_at(self.edges,self.cutoff,self.origin),news_pool([self.story],self.cutoff,self.origin),NAMES)
    def test_future_edge_rejected(self):
        self.edges[0]['available_at']=self.cutoff
        with self.assertRaises(ValueError): graph_at(self.edges,self.cutoff,self.origin)
    def test_future_edge_start_rejected(self):
        self.edges[0]['start']='2021-01-01'
        with self.assertRaises(ValueError): graph_at(self.edges,self.cutoff,self.origin)
    def test_future_news_rejected(self):
        self.story['available_at']=self.cutoff
        with self.assertRaises(ValueError): news_pool([self.story],self.cutoff,self.origin)
    def test_origin_day_news_rejected(self):
        self.story['date']=self.origin
        with self.assertRaises(ValueError): news_pool([self.story],self.cutoff,self.origin)
    def test_characteristic_future_rejected(self):
        self.avail['cash']=self.cutoff
        with self.assertRaises(ValueError): exposure(*self.args())
    def test_missing_characteristic_zero(self):
        self.chars['cash']=np.nan
        r,_=exposure(*self.args())
        self.assertTrue(all(v['Z']==0 and v['characteristic_missing'] for v in r if v['name']=='cash'))
    def test_none(self): self.assertEqual(choice_gate(NAMES,'none',NAMES).sum(),0)
    def test_gate_mapping(self):
        order=list(reversed(NAMES)); g=gate(NAMES,order,2)
        self.assertEqual(dict(zip(NAMES,g)),dict(size=0,cash=0,cash_flow=0,leverage=1,inventory=1))
    def test_missing_does_not_consume_budget(self):
        self.assertEqual(gate(NAMES,NAMES,1,['cash']).tolist(),[0,1,0,0,0])
    def test_k_zero(self): self.assertEqual(gate(NAMES,NAMES,0).sum(),0)
    def test_k_all(self): self.assertEqual(gate(NAMES,NAMES,'all').sum(),5)
    def test_invalid_order_no_backfill(self):
        with self.assertRaises(ValueError): gate(NAMES,['cash']*5,2)
    def test_exact_hops(self):
        paths=exact_paths(graph_at(self.edges,self.cutoff,self.origin),1)
        self.assertEqual([set(paths[h]) for h in (1,2,3)],[{2},{3},{4}])
    def test_shortest_distance_excludes_longer_walk(self):
        e=copy.deepcopy(self.edges); e.append(dict(e[0],a=1,b=3,id='shortcut'))
        paths=exact_paths(graph_at(e,self.cutoff,self.origin),1)
        self.assertIn(3,paths[1]); self.assertNotIn(3,paths[2])
    def test_intermediates_and_physical_claim(self):
        _,e=exposure(*self.args()); h3=[r for r in e if r['hop']==3][0]
        self.assertEqual(h3['intermediate_nodes'],[2,3]); self.assertFalse(h3['physical_channel_verified'])
    def test_z_product(self):
        r,_=exposure(*self.args())
        for v in r: self.assertAlmostEqual(v['Z'],v['C']*v['E'])
    def test_duplicate_article_no_gain(self):
        r,_=exposure(*self.args()); args=list(self.args())
        args[-2]=news_pool([self.story,self.story],self.cutoff,self.origin)
        self.assertEqual(r,exposure(*args)[0])
    def test_article_max_not_sum(self):
        s=copy.deepcopy(self.story); s['alt_id']='second'
        args=list(self.args()); args[-2]=news_pool([self.story,s],self.cutoff,self.origin)
        self.assertEqual([r['E'] for r in exposure(*args)[0]],[r['E'] for r in exposure(*self.args())[0]])
    def test_empty_news_zero(self):
        args=list(self.args()); args[-2]={}
        self.assertEqual(sum(r['Z'] for r in exposure(*args)[0]),0)
    def test_empty_graph_same_rows(self):
        args=list(self.args()); args[-3]={}
        self.assertEqual(len(exposure(*args)[0]),15)
    def test_future_returns_schema_poison(self):
        packet=dict(firm_id='x',cutoff=self.cutoff,origin_session=self.origin,C=self.chars,
                    available_at=self.avail,E=self.chars,news_ids=[],graph_evidence=[])
        for field in ['y','ret','future_returns','label_end','r_t_plus_1']:
            with self.assertRaises(ValueError): feature_packet(dict(packet,**{field:object()}),NAMES)
    def test_no_return_argument_feature_api(self):
        import inspect
        self.assertFalse(any('ret' in p or 'label' in p for p in inspect.signature(exposure).parameters))
    def test_rng_repeatable(self):
        self.assertEqual(ordering(NAMES,'random','x',72101),ordering(NAMES,'random','x',72101))
    def test_beta_training_order(self):
        self.assertEqual(ordering(NAMES,'training_beta','x',beta=dict(zip(NAMES,[1,-5,3,2,0])))[0],'cash')
    def test_empty_slots_checked(self):
        for kind,s in SLOT_SCHEMAS.items(): self.assertTrue(validate_slot(s,kind,NAMES))
    def test_invalid_choice_probabilities(self):
        s=copy.deepcopy(SLOT_SCHEMAS['jev_choice']); s['records']=[dict(firm_id='x',cutoff='t',input_sha256='a'*64,
            receipt_id='r',choice='none',probabilities=dict.fromkeys(NAMES+['none'],.8))]
        with self.assertRaises(ValueError): validate_slot(s,'jev_choice',NAMES)
    def test_noul_threshold(self):
        self.assertEqual(noul_gate(NAMES,dict(zip(NAMES,[.1,.9,.2,.8,.3])),NAMES,threshold=.5).sum(),2)
    def test_ridge_is_new_and_no_intercept(self):
        x=np.eye(5); b=ridge(x,np.arange(5.),1.)
        np.testing.assert_allclose(b,np.arange(5.)/2)
    def test_two_way_calendar_week(self):
        result=clustered_mean([1,2,3,4],['a','b','a','b'],['2020-01-06','2020-01-07','2020-01-13','2020-01-14'])
        self.assertEqual(result['clusters_calendar_week'],2)
    def test_paired_gain_sign(self):
        r=paired([1,2,3,4],[0,0,0,0],[1,2,3,4],['a','b','a','b'],['2020-01-06','2020-01-07','2020-01-13','2020-01-14'])
        self.assertGreater(r['mean'],0)

def run():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Fixture)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    return {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'passed':result.testsRun-len(result.failures)-len(result.errors),'synthetic_only':True}
