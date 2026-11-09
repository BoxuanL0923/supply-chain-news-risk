"""Fresh TD1C traps. No real evaluation outcomes and no original source reads."""
import audit_adapter as audit
import contextlib
import copy
import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import sources
import build
import core
import evaluate
from spec import FEATURES

ROOT = audit.REVIEW/'traps'
ROOT.mkdir()

class Bounded:
    def __init__(self, data, last): self.data=np.asarray(data); self.last=last; self.shape=self.data.shape
    def __getitem__(self,key):
        primary=key[0] if isinstance(key,tuple) else key
        indices=np.arange(len(self.data))[primary]
        if np.size(indices) and np.max(indices)>self.last:
            raise AssertionError('Post-cutoff input access')
        return self.data[key]

class Traps(unittest.TestCase):
    def test_01_own_history_cannot_index_future(self):
        dates=pd.bdate_range('2021-01-01',periods=80)
        r=np.linspace(-.02,.03,80); me=np.full(80,12.)
        a=core.own_inputs(dates,Bounded(r,40),Bounded(me,40),40)
        b=core.own_inputs(dates,r,me,40)
        self.assertEqual(a,b)

    def test_02_common_history_cannot_index_future(self):
        rng=np.random.default_rng(71)
        r=rng.normal(0,.02,(80,60)); me=rng.uniform(1,10,(80,60)); ff=np.ones((80,60))
        a=build.common_all(Bounded(r,40),Bounded(me,40),Bounded(ff,40),40)
        b=build.common_all(r,me,ff,40)
        for k in a: np.testing.assert_array_equal(a[k],b[k])

    def test_03_label_boundary_before_any_return_access(self):
        class Bomb:
            def __getitem__(self,key): raise AssertionError('Forbidden label return read')
        dates=pd.bdate_range('2022-12-12',periods=25)
        t=dates.get_indexer([pd.Timestamp('2022-12-29')])[0]
        self.assertFalse(core.label(dates,Bomb(),t,'2022-12-31')[2])

    def test_04_label_exact_last_five_of_2022(self):
        dates=pd.bdate_range('2022-12-12',periods=15)
        t=len(dates)-6; r=np.linspace(-.04,.03,len(dates))
        actual,end,valid=core.label(dates,r,t,'2022-12-31')
        expected=np.log(sum(float(r[t+k])**2 for k in range(1,6))/5)
        self.assertTrue(valid); self.assertAlmostEqual(actual,expected,14)

    def fixture(self,year,mode='base'):
        dates=pd.bdate_range(f'{year}-01-02',periods=60)
        origin=dates[35]; cut=core.clock(origin)
        ids=np.arange(100,160); rng=np.random.default_rng(19)
        d=pd.DataFrame({'date':np.repeat(dates,60),'permno':np.tile(ids,60),
            'ret':rng.normal(0,.02,3600),'me':rng.uniform(5,20,3600),'ff49':1.,'share_type':0})
        q=pd.DataFrame({'permno':[100,100,100],
            'availability':[cut-pd.Timedelta(days=1),cut,cut+pd.Timedelta(days=1)],
            'datadate':[origin-pd.Timedelta(days=n) for n in (100,90,80)],
            'availability_rule':['rdq_end_of_day']*3,'cash':[.1,99.,999.],
            'inventory':[.2,99.,999.],'leverage':[.3,99.,999.]})
        cf=pd.DataFrame({'permno':[100,100,100],
            'availability':[cut-pd.Timedelta(days=1),cut,cut+pd.Timedelta(days=1)],
            'eom':[origin-pd.Timedelta(days=n) for n in (250,240,230)],'ocf_at':[.4,99.,999.]})
        e=pd.DataFrame({'permno':[100,100,100],
                       'availability':[core.clock(dates[10]),cut,cut+pd.Timedelta(days=1)]})
        if mode=='future_poison':
            d.loc[d.date>origin,['ret','me','ff49']]=[9e9,0.,49.]
            for f,cols in [(q,['cash','inventory','leverage']),(cf,['ocf_at'])]:
                f.loc[f.availability>=cut,cols]=-9e9
            e.loc[e.availability>=cut,'availability']+=pd.Timedelta(days=800)
        empty=d.iloc[:0].copy()
        def reader(name,**kwargs): return d.copy() if name==f'crsp_{year}.parquet' else empty.copy()
        output=ROOT/f'build_{year}_{mode}'
        # No target after 2022 can execute, even for invented returns.
        old_label=build.label
        def sealed_label(dates,returns,t,allowed_end):
            if pd.Timestamp(dates[t]).year>=2023:
                raise AssertionError('Attempt to compute an evaluation target in test')
            return old_label(dates,returns,t,allowed_end)
        with patch('sources.OUT',ROOT), patch('build.MANIFEST',output), patch('build.read_input',side_effect=reader), patch('build.label',side_effect=sealed_label):
            build.build_year(year,pd.DataFrame({'date':[origin],'permno':[100]}),q,e,cf)
        import pyarrow.parquet as pq
        f=pq.read_table(output/f'year_{year}.parquet').to_pandas()
        return f.iloc[0],f.columns

    def test_05_manifest_future_mutations_do_not_change_features(self):
        a,_=self.fixture(2021); b,_=self.fixture(2021,'future_poison')
        columns=FEATURES+[c+'__available_at' for c in FEATURES]+[c+'__missing' for c in FEATURES]
        pd.testing.assert_series_equal(a[columns],b[columns])

    def test_06_equal_and_future_accounting_and_earnings_excluded(self):
        a,_=self.fixture(2020)
        self.assertEqual([a.cash,a.inventory,a.leverage,a.cash_flow],[.1,.2,.3,.4])
        self.assertEqual(a.post_earnings,0.)
        self.assertTrue(a.earnings_covered)

    def test_07_2023_manifest_cannot_call_label(self):
        _,columns=self.fixture(2023)
        self.assertFalse(set(columns)&{'y','label_end','label_observed'})

    def test_08_2024_manifest_cannot_call_label(self):
        _,columns=self.fixture(2024)
        self.assertFalse(set(columns)&{'y','label_end','label_observed'})

    def test_09_quarter_missing_rdq_and_same_day_release(self):
        q=pd.DataFrame({'gvkey':['1','2','3'],'datadate':['2021-03-31']*3,
            'atq':['100']*3,'cheq':['10']*3,'invtq':['20']*3,'dlcq':['2']*3,'dlttq':['3']*3})
        e=pd.DataFrame({'gvkey':['1','2','3'],'datadate':['2021-03-31']*3,
                        'rdq':['2021-05-01','','2021-03-01']})
        with patch('build.read_input',side_effect=lambda name,**kw:q.copy() if name=='quarters.parquet' else e.copy()), patch('sources.read_keys',return_value=pd.DataFrame({'gvkey':[1,2,3],'permno':[101,102,103]})):
            accounting,_=build.accounting()
        q=accounting.set_index('permno')
        self.assertEqual(q.loc[101,'availability'],core.clock('2021-05-01'))
        for p in (102,103):
            self.assertEqual(q.loc[p,'availability'],core.clock(pd.Timestamp('2021-03-31')+pd.Timedelta(days=120)))

    def test_10_evaluate_run_absent_clearance_never_reads_or_labels(self):
        with patch('evaluate.CLEARANCE',ROOT/'missing_clearance.json'), patch('sources.dated',side_effect=AssertionError('Payload read')), patch('core.label',side_effect=AssertionError('Evaluation label call')):
            with self.assertRaises(PermissionError): evaluate.run()

    def make_gate(self,name,bad_field=None):
        root=ROOT/name; code=root/'code'; code.mkdir(parents=True)
        py=code/'dummy.py'; py.write_text('# invented code only\n')
        files={}
        for field in ['manifest_sha256','input_sha256','prediction_sha256','selection_sha256']:
            p=root/(field+'.bin'); p.write_bytes(b'invented, no outcomes')
            files[field]={str(p):audit.hash_file(p)}
        files['code_sha256']={str(py):audit.hash_file(py)}
        model=root/'model_frozen_v2.pkl'; model.write_bytes(b'invented, not a pickle')
        f=dict(files,version='synthetic-review',model_sha256=audit.hash_file(model))
        if bad_field:
            if bad_field=='model_sha256': f[bad_field]='0'*64
            else: f[bad_field][next(iter(f[bad_field]))]='0'*64
        fp=root/'freeze.json'; fp.write_text(json.dumps(f))
        cp=root/'clearance.json'; cp.write_text(json.dumps(dict(f,schema='TD1_CLEARANCE_v2',status='CLEARED',freeze_sha256=audit.hash_file(fp))))
        return root,code,fp,cp

    def test_11_every_hash_checked_before_payload(self):
        for field in ['code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256','model_sha256']:
            root,code,fp,cp=self.make_gate(field,field)
            with patch('evaluate.OUT',root), patch('evaluate.CODE',code), patch('evaluate.CLEARANCE',cp), patch('evaluate.freeze_path',return_value=fp), patch('sources.dated',side_effect=AssertionError('Payload read')), patch('core.label',side_effect=AssertionError('Evaluation label call')):
                with self.assertRaises(PermissionError): evaluate.run()
            self.assertFalse((root/'EVALUATION_STARTED.json').exists())

    def test_12_once_only_refuses_before_hash_or_payload(self):
        root,code,fp,cp=self.make_gate('once_only')
        (root/'EVALUATION_STARTED.json').write_text('{}')
        with patch('evaluate.OUT',root), patch('evaluate.CODE',code), patch('evaluate.CLEARANCE',cp), patch('evaluate.freeze_path',side_effect=AssertionError('Read after replay stop')), patch('sources.dated',side_effect=AssertionError('Payload read')):
            with self.assertRaises(PermissionError): evaluate.run()

    def test_13_csv_future_dates_never_index_payload(self):
        class PoisonRow:
            def __getitem__(self,i):
                if i==0: return '2025-01-01'
                raise AssertionError('Forbidden future CSV payload access')
        p=ROOT/'csv_parser_stub.csv'; p.write_text('unused')
        with patch('sources.csv.reader',return_value=iter([['datadate','amount'],PoisonRow()])):
            self.assertTrue(sources.csv_dated(p,['datadate','amount']).empty)

    def test_14_unadmitted_panel_outcome_rejected_before_reader(self):
        with patch('sources.pq.read_schema',side_effect=AssertionError('Native decoder reached')):
            with self.assertRaises(ValueError): sources.dated(sources.PANEL,['date','permno','y'])

    def test_15_forbidden_year_rejected_before_decoder(self):
        with patch('sources.pq.read_schema',side_effect=AssertionError('Native decoder reached')):
            with self.assertRaises(ValueError): sources.dated(ROOT/'unopened.parquet',['date','ret'],hi='2025-01-01')

    def test_16_all_feature_inputs_contain_no_news_or_topology(self):
        self.assertEqual(FEATURES,['size','cash','cash_flow','leverage','inventory','rv1','rv5','rv22',
            'ret1','ret5','ret22','own_downside5','post_earnings','earnings_covered','earnings_scheduled',
            'market_log_rv5','market_log_rv22','industry_log_rv5'])

log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Traps)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
with (audit.REVIEW/'lookahead_traps.log').open('x',encoding='utf-8') as stream: stream.write(log.getvalue())
facts={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
       'real_evaluation_labels_read_or_computed':False,**audit.runtime.peak_bytes()}
audit.write_json('lookahead_traps_result.json',facts)
print(facts)
if not result.wasSuccessful(): print(log.getvalue())
raise SystemExit(not result.wasSuccessful())
