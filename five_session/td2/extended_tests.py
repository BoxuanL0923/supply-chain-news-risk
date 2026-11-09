"""Regression fixtures for production-shaped CSV/news values and bounded output."""
import copy
import io
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import decoder
import evaluation
from engine import score,exposure,graph_at,news_pool,day_clock
from selectors import validate_slot
from spec import NAMES,SLOT_SCHEMAS

class Adapter(unittest.TestCase):
    def test_json_mapping_lists_are_accepted(self):
        batch=pd.DataFrame([dict(date='2020-02-03',alt_id='s',rics='RIC.A',headline='debt',body_short='shortage')])
        saved=[]
        with patch.object(decoder,'parquet',return_value=iter([batch])),patch.object(decoder,'write_frame',side_effect=lambda p,f:saved.append(f.copy())):
            self.assertEqual(decoder.build_news(2020,{'RIC.A':[123,201901]},set()),1)
        self.assertEqual(saved[0].iloc[0].firms,[123])
    def test_duplicate_story_earliest_day_wins(self):
        batch=pd.DataFrame([dict(date=day,alt_id='s',rics='R',headline=text,body_short='') for day,text in [('2020-02-03','debt'),('2020-02-01','demand')]])
        saved=[]
        with patch.object(decoder,'parquet',return_value=iter([batch])),patch.object(decoder,'write_frame',side_effect=lambda p,f:saved.append(f.copy())):
            decoder.build_news(2020,{'R':[123,201901]},set())
        self.assertEqual(str(saved[0].iloc[0].date.date()),'2020-02-01')
    def test_seen_story_not_reintroduced(self):
        batch=pd.DataFrame([dict(date='2020-02-01',alt_id='s',rics='R',headline='debt',body_short='')])
        saved=[]
        with patch.object(decoder,'parquet',return_value=iter([batch])),patch.object(decoder,'write_frame',side_effect=lambda p,f:saved.append(f.copy())):
            self.assertEqual(decoder.build_news(2020,{'R':[123,201901]},{'s'}),0)
    def test_large_bounded_fsb5_field(self):
        text='bridge_start,entity_id,A_PERMNO,intervals\n2017-01-01,a,123,'+('x'*150000)+'\n'
        with patch.object(decoder,'FS5') as source:
            source.open.return_value=io.StringIO(text)
            self.assertEqual(decoder.sensitivity_links(),{('a',123)})
    def test_ended_latest_version_not_resurrected(self):
        graph=pd.DataFrame([dict(id='x',start='2018-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='b',rownum=1),
            dict(id='x',start='2019-01-01',end='2019-12-31',status='valid',typ='CUSTOMER',src='a',tgt='b',rownum=2)])
        links=pd.DataFrame([('a',1,'2010-01-01','2024-12-31'),('b',2,'2010-01-01','2024-12-31')],columns=['entity','permno','start','end'])
        self.assertEqual(decoder.mapped_edges(graph,links,'2020-01-01'),[])
    def test_later_version_not_selected_early(self):
        graph=pd.DataFrame([dict(id='x',start='2018-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='b',rownum=1),
            dict(id='x',start='2021-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='c',rownum=2)])
        links=pd.DataFrame([(n,p,'2010-01-01','2024-12-31') for p,n in enumerate(['a','b','c'],1)],columns=['entity','permno','start','end'])
        self.assertEqual(decoder.mapped_edges(graph,links,'2020-01-01')[0]['b'],2)
    def test_article_mapping_month_must_precede_origin(self):
        news=pd.DataFrame([dict(date=pd.Timestamp('2020-06-01'),alt_id='s',firms=[1,2],first_months=[202006,202005],available_at=day_clock('2020-06-01'),**dict.fromkeys(NAMES,1.))])
        self.assertEqual(decoder.week_stories(news,'2020-06-05')[0]['firms'],[2])
    def test_label_exact_five_synthetic_2020(self):
        previous=decoder._evaluation_token; token=object(); decoder._evaluation_token=token
        try:
            dates=pd.bdate_range('2020-01-01',periods=6)
            value,end=evaluation.label(dates,np.array([100.,.01,.02,.03,.04,.05]),0,token)
            self.assertAlmostEqual(value,np.log(np.mean(np.arange(1,6)**2*.0001)))
            self.assertEqual(end,dates[5])
        finally: decoder._evaluation_token=previous
    def test_article_score_bounded(self):
        self.assertTrue(all(0<=v<=1 for v in score('debt '*10000).values()))
    def test_no_news_zero_falls_back_to_baseline(self):
        from pipeline import predictions
        f=pd.DataFrame([dict(firm_id='synthetic',cutoff='2020-01-01',b=3.,has_news=False,
            **{'missing_'+n:False for n in NAMES},**{'Z_'+n:0. for n in NAMES})])
        m=dict(names=NAMES,beta=[99.]*5,training_beta=dict.fromkeys(NAMES,1.))
        for k in [0,1,2,3,5,'all']:
            np.testing.assert_equal(predictions(f,m,'training_beta',k,0),[3.])
    def test_unknown_noul_item_rejected(self):
        s=copy.deepcopy(SLOT_SCHEMAS['jev_noul'])
        s['records']=[dict(firm_id='x',cutoff='2020-01-01',input_sha256='a'*64,receipt_id='r',yes_scores={'not_a_characteristic':.9})]
        with self.assertRaises(ValueError): validate_slot(s,'jev_noul',NAMES)
    def test_identical_rows_across_hops_and_K(self):
        from pipeline import predictions
        rows=[]
        for i in range(4):
            r=dict(firm_id=str(i),cutoff='2020-01-01',b=0.,has_news=True)
            r.update({'missing_'+n:False for n in NAMES})
            for prefix in ['Z_','Z1_','Z2_','Z3_']: r.update({prefix+n:float(i) for n in NAMES})
            rows.append(r)
        f=pd.DataFrame(rows); m=dict(names=NAMES,beta=[1.]*5,training_beta=dict.fromkeys(NAMES,1.))
        for hop in ['main',1,2,3]:
            for k in [0,1,2,3,5,'all']: self.assertEqual(len(predictions(f,m,'random',k,1,hop)),4)

def run_extended():
    r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Adapter))
    return {'tests':r.testsRun,'passed':r.testsRun-len(r.errors)-len(r.failures),
            'errors':len(r.errors),'failures':len(r.failures),'synthetic_only':True}
