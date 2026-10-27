"""Offline algebra and access-boundary tests, python -m unittest examples.test_q1_design."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from source.code.news_network_50_2025 import prompt_gd_design as d


def fixture():
    rows=[]
    for i in range(12):
        features=[]
        for h in range(4):
            features.append({'source_id':f'{i}-{h}','hop':h,'relevance':.2+.06*i,
                'direction':(-1)**(i+h),'materiality':.1+.04*(i+h),
                'graph':{'degree_percentile':65+2*i,'ra_percentile':60+3*i,
                         'ra':.02*(i+1),'clustering':.005+.005*i}})
        rows.append(features)
    original=d.v2.old.Evidence(rows)
    mask=original.neighbour & original.present
    maps=[d.v2.Quantile(x[mask]) for x in (original.degree,original.ra_pct,original.clust,original.mat)]
    return original,maps


class ScoreTests(unittest.TestCase):
    def test_v2_equivalence(self):
        original,maps=fixture()
        v=d.v2.Evidence(original,maps,[0,1,4,6]);b=d.Bank(original,maps)
        x=np.array([.3,.7,.4,.6])
        for left,right in zip(v.score(x),b.score(x)):
            np.testing.assert_allclose(left,right,atol=1e-14,rtol=1e-13)

    def test_all_graph_jacobian(self):
        original,maps=fixture();b=d.Bank(original,maps,transform=np.eye(6));x=np.linspace(.3,.7,6)
        _,j=b.score(x)
        fd=np.column_stack([(b.score(x+np.eye(6)[i]*1e-6)[0]-b.score(x-np.eye(6)[i]*1e-6)[0])/2e-6 for i in range(6)])
        np.testing.assert_allclose(j,fd,atol=1e-10,rtol=1e-6)

    def test_merged_jacobian_is_sum(self):
        original,maps=fixture();b=d.Bank(original,maps);m=d.Bank(original,maps,transform=d.transform([0,4,5],True))
        s,j=b.score([.4,.4,.6,.7]);sm,jm=m.score([.4,.6,.7])
        np.testing.assert_allclose(s,sm)
        np.testing.assert_allclose(j[:,0]+j[:,1],jm[:,0])

    def test_decision_uses_shipped_hop3_default(self):
        original,maps=fixture();b=d.Bank(original,maps)
        x=d.decision_default(b)
        self.assertEqual(x[1],.05)
        null=x.copy();null[1]=0
        np.testing.assert_allclose(b.score(x)[0]-b.score(null)[0],.05*b.score(x)[1][:,1],atol=1e-15)

    def test_w_only_closed_form_and_five_starts(self):
        original,maps=fixture();b=d.Bank(original,maps,transform=d.transform([5]))
        s=b.score([.63])[0];best,agree,fits=d.fit_five(b,s)
        self.assertTrue(agree);self.assertEqual(len(fits),5)
        self.assertAlmostEqual(best['theta'][0],.63,places=6)

    def test_cluster_GLS_inverse(self):
        original,maps=fixture();b=d.Bank(original,maps,transform=d.transform([5]))
        y=b.score([.66])[0]+np.linspace(-.005,.006,12)
        groups=np.repeat(np.arange(4),3);rho=.25
        best,agree,_=d.fit_five(b,y,groups,rho)
        inv=np.linalg.inv((1-rho)*np.eye(3)+rho*np.ones((3,3)))
        raw=b.score([1])[0].reshape(4,3);yy=y.reshape(4,3)
        expected=sum(x@inv@z for x,z in zip(raw,yy))/sum(x@inv@x for x in raw)
        self.assertAlmostEqual(best['theta'][0],np.clip(expected,0,1),places=6)


class InformationTests(unittest.TestCase):
    def test_information_scale(self):
        j=np.array([[1.,0.],[0.,2.]])
        a=d.information(j,2.,.1);b=d.information(j,1.,.2)
        self.assertIsNotNone(a['n']);self.assertAlmostEqual(a['n']/b['n'],16,delta=.1)
        self.assertIsNone(d.information(np.ones((10,2)),1.,.1)['n'])

    def test_d_optimal_unique_and_high_leverage(self):
        j=np.array([[.1,0.],[0.,.1],[2.,0.],[0.,2.],[1.,1.]])
        order=d.d_order(j,4)
        self.assertEqual(len(set(order)),4)
        self.assertIn(order[0],(2,3))
        G=j[order[:2]].T@j[order[:2]]
        self.assertGreater(np.linalg.det(G),.1)

    def test_host_order_template_average(self):
        j=np.array([[1.,0.],[0.,2.],[.2,.2],[.1,.1]])
        order=d.host_order(j,np.array([0,0,1,1]),4)
        self.assertEqual(len(set(order)),4)
        self.assertEqual(order[0],0) # the first ID in the informative template

    def test_power_threshold(self):
        # Exact fixed-nuisance test; alpha=.05 and n formula imply .8.
        effect=.02;sigma=1.
        n=(d.norm.ppf(.95)+d.norm.ppf(.8))**2*sigma**2/effect**2
        self.assertAlmostEqual(d.norm.cdf(np.sqrt(n)*effect/sigma-d.norm.ppf(.95)),.8)
        self.assertAlmostEqual(d.norm.cdf(-d.norm.ppf(.95)),.05)


class AccessAndClockTests(unittest.TestCase):
    def test_memory_measurement_positive(self):
        self.assertGreater(d.peak_memory(),1024**2)

    def test_no_outcome_projection_or_2025(self):
        for cols in (['sig_b0'],['relevance'],['label_confidence'],['ret']):
            with self.assertRaises(ValueError):d.metadata('does-not-exist.parquet',cols)
        with self.assertRaises(ValueError):d.metadata('events_2025.parquet',['date'])

    def test_immutable_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(d,'protocol',return_value={'seed':1}):first=d.register(temp)
            with patch.object(d,'protocol',side_effect=AssertionError('must not overwrite')):
                second=d.register(temp)
            self.assertEqual(first,second)
            with self.assertRaises(FileExistsError):d.save(Path(temp)/'PROTOCOL_DESIGN.json',{},exclusive=True)

    def test_missing_registration_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):d.check_registration(temp)

    def test_timestamp_strict_end(self):
        events=pd.DataFrame({'date':pd.to_datetime(['2024-07-01']),'c':[1],'bin':[6],'neg':[0]})
        # bin 6 starts 10:00 and ends 10:05. Only 09:30 <= ts < 10:05.
        articles=pd.DataFrame({'alt_id':['tooearly','pre','within','end','late'],
            'ts':pd.to_datetime(['2024-07-01 09:29:59','2024-07-01 09:59:00','2024-07-01 10:04:59','2024-07-01 10:05:00','2024-07-01 10:06:00']),
            'candidate_rics':['A.N']*5})
        earliest,allprior=d.match_articles(events,articles,{'A.N':1})
        self.assertEqual(earliest.alt_id.tolist(),['pre'])
        self.assertEqual(set(allprior.alt_id),{'tooearly','pre','within'})

    def test_graph_clustering_and_RA(self):
        g=d.Graph([(1,2),(2,3),(1,3),(3,4)])
        self.assertEqual(g.node(1)[1],1.)
        f=g.fields(1,2);self.assertAlmostEqual(f['ra'],1/3)
        self.assertAlmostEqual(g.node(3)[1],1/3)


if __name__=='__main__':unittest.main()
