"""Exact parity with pure fixture engine on independently generated graph shapes."""
import unittest
import numpy as np
import pandas as pd
import engine
import fast_builder
from spec import NAMES

class FastParity(unittest.TestCase):
    def test_exposures_and_all_witnesses_match(self):
        rng=np.random.default_rng(731); cutoff=pd.Timestamp('2020-06-13 03:59:59'); origin=pd.Timestamp('2020-06-12')
        for seed in range(12):
            edges=[]
            for a in range(1,20):
                for b in range(a+1,20):
                    if rng.random()<.12:
                        edges.append(dict(id=f'{a}:{b}',a=a,b=b,start='2020-01-01',end=None,
                            available_at=engine.day_clock('2020-01-01'),rownum=a*20+b,typ='CUSTOMER'))
            adj=engine.graph_at(edges,cutoff,origin)
            pool={p:{str(p):dict(alt_id=str(p),scores=engine.score('debt demand shortage cash'),
                available_at=engine.day_clock('2020-06-11'))} for p in range(2,20,2)}
            chars=dict.fromkeys(NAMES,1.); availability=dict.fromkeys(NAMES,engine.day_clock('2020-01-01'))
            for root in range(1,20):
                expected=engine.exposure(root,cutoff,origin,chars,availability,adj,pool,NAMES)
                actual=fast_builder.exposure(root,cutoff,origin,chars,availability,fast_builder.Graph(adj),pool,NAMES)
                self.assertEqual(actual,expected)
    def test_mapped_edge_projection_matches(self):
        import decoder
        graph=pd.DataFrame([dict(id='x',start='2018-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='b',rownum=1)])
        links=pd.DataFrame([('a',1,'2010-01-01','2024-12-31'),('b',2,'2010-01-01','2024-12-31')],columns=['entity','permno','start','end'])
        self.assertEqual(fast_builder.mapped_edges(graph,links,'2020-01-01'),decoder.mapped_edges(graph,links,'2020-01-01'))

def run_fast():
    r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(FastParity))
    return {'tests':r.testsRun,'passed':r.testsRun-len(r.errors)-len(r.failures),'errors':len(r.errors),'failures':len(r.failures)}
