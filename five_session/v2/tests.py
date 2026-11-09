"""Local invented-data tests. Fixtures persist privately; no deletion or providers."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import pandas as pd
import sources as io
from runtime import OUT, guard, CAP, MIN_FREE, new_json, utc, peak_bytes
from core import clock, require_past, own_inputs, label, gate_values, prepare_fit, transform, old_fit, old_predict, paired
from spec import FEATURES, OLD
from evaluate import validate_clearance, admit

class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=OUT/'fixtures'/uuid.uuid4().hex; cls.path.mkdir(parents=True)
        cls.dates=pd.bdate_range('2020-01-01',periods=40)
        cls.r=np.linspace(-.03,.04,40); cls.me=np.full(40,10.)
        cls.cut=clock(cls.dates[25])

    def test_01_future_feature_rejected(self):
        with self.assertRaises(ValueError): require_past([self.cut+pd.Timedelta(seconds=1)],self.cut)
    def test_02_equal_cutoff_rejected(self):
        with self.assertRaises(ValueError): require_past([self.cut],self.cut)
    def test_03_unknown_availability_rejected(self):
        with self.assertRaises(ValueError): require_past([pd.NaT],self.cut)
    def test_04_past_feature_accepted(self):
        require_past([self.cut-pd.Timedelta(seconds=1)],self.cut)
    def test_05_future_news_rejected(self):
        with self.assertRaises(ValueError): require_past([self.cut+pd.Timedelta(days=1)],self.cut,'news version')
    def test_06_future_edge_rejected(self):
        with self.assertRaises(ValueError): require_past([self.cut+pd.Timedelta(days=1)],self.cut,'edge publication')
    def test_07_future_five_returns_do_not_affect_inputs(self):
        before=own_inputs(self.dates,self.r,self.me,25)
        r=self.r.copy(); r[26:31]=[999,-999,np.nan,np.inf,-np.inf]
        self.assertEqual(before,own_inputs(self.dates,r,self.me,25))
    def test_08_future_size_does_not_affect_inputs(self):
        a=own_inputs(self.dates,self.r,self.me,25); m=self.me.copy(); m[26:]=9999
        self.assertEqual(a,own_inputs(self.dates,self.r,m,25))
    def test_09_target_exact_five_subsequent(self):
        y,end,ok=label(self.dates,self.r,25,'2022-12-31')
        self.assertTrue(ok); self.assertAlmostEqual(y,np.log(np.mean(self.r[26:31]**2)),places=14)
        self.assertEqual(end,self.dates[30])
    def test_10_origin_return_not_in_target(self):
        a=label(self.dates,self.r,25,'2022-12-31'); r=self.r.copy(); r[25]=1e9
        self.assertEqual(a,label(self.dates,r,25,'2022-12-31'))
    def test_11_label_after_boundary_not_read(self):
        class Bomb:
            def __getitem__(self,key): raise AssertionError('Future label fields touched')
        y,end,ok=label(self.dates,Bomb(),25,'2019-12-31')
        self.assertFalse(ok); self.assertTrue(np.isnan(y))
    def test_12_insufficient_future_sessions(self):
        self.assertFalse(label(self.dates,self.r,38,'2022-12-31')[2])
    def test_13_missing_future_return_not_zero(self):
        r=self.r.copy(); r[27]=np.nan
        self.assertFalse(label(self.dates,r,25,'2022-12-31')[2])
    def test_14_zero_target_not_floored(self):
        self.assertFalse(label(self.dates,np.zeros(40),25,'2022-12-31')[2])
    def test_15_missing_characteristic_retained(self):
        x=np.array([[1.,np.nan],[3.,np.nan]])
        p=prepare_fit(x); z=transform(x,p)
        self.assertTrue(np.isfinite(z).all()); np.testing.assert_equal(z[:,-1],1.)
    def test_16_scaler_train_only(self):
        x=np.array([[0.,1.],[2.,3.]])
        p=prepare_fit(x); q=copy.deepcopy(p); transform(np.array([[1e8,np.nan]]),p)
        for k in p: np.testing.assert_array_equal(p[k],q[k])
    def test_17_training_medians(self):
        p=prepare_fit(np.array([[1.,np.nan],[3.,4.],[5.,6.]]))
        np.testing.assert_array_equal(p['median'],[3.,5.])
    def test_18_none_gate(self):
        self.assertEqual(gate_values({'cash':2.},{'cash':self.cut},self.cut,'none'),{'cash':0.})
    def test_19_gate_maps_exact_characteristic(self):
        out=gate_values({'cash':2.,'inventory':3.},{'cash':self.cut-pd.Timedelta(days=1)},self.cut,'cash')
        self.assertEqual(out,{'cash':2.,'inventory':0.})
    def test_20_gate_missing_characteristic(self):
        out=gate_values({'cash':np.nan},{'cash':pd.NaT},self.cut,'cash')
        self.assertEqual(out,{'cash':0.})
    def test_21_invalid_gate(self):
        with self.assertRaises(ValueError): gate_values({'cash':2.},{},self.cut,'leverage')
    def test_22_date_filter_before_payload(self):
        path=self.path/'mixed.parquet'
        io.write_parquet(path,pd.DataFrame({'date':pd.to_datetime(['2024-12-31','2025-01-01']),
                                           'payload':['ok','POISON_NEVER_ACCESS']}))
        f=io.dated(path,['date','payload'])
        self.assertEqual(f.payload.tolist(),['ok'])
    def test_23_future_upper_bound_rejected(self):
        with self.assertRaises(ValueError): io.dated(self.path/'never_open.parquet',[],hi='2025-01-01')
    def test_24_csv_dates_drop_before_payload(self):
        path=self.path/'mixed.csv'
        path.write_text('datadate,rdq,amount\n2024-01-01,2024-02-01,4\n2025-01-01,2025-02-01,poison\n2024-12-31,2025-01-01,poison\n')
        f=io.csv_dated(path,['datadate','rdq','amount'],second_date='rdq')
        self.assertEqual(f.amount.tolist(),['4'])
    def test_25_old_baseline_analytic(self):
        x=np.arange(50.,dtype=float)[:,None]; y=2*x[:,0]+1
        m=old_fit(x,y); np.testing.assert_allclose(old_predict(x,m),y,atol=2e-6)
    def test_26_paired_losses_same_rows(self):
        report=paired(np.array([1.,2.]),np.array([1.,2.]),np.array([0.,0.]),['a','b'],['w1','w2'])
        self.assertEqual(report['n'],2); self.assertEqual(report['mlp_mse'],0.)
    def test_27_nonfinite_pair_rejected(self):
        with self.assertRaises(ValueError): paired(np.array([1.]),np.array([np.nan]),np.array([0.]),['a'],['w'])
    def test_28_clearance_hash_mismatch(self):
        f={k:{} for k in ['code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256']}
        f.update(version='v',model_sha256='m')
        c=dict(f,schema='TD1_CLEARANCE_v2',status='CLEARED',freeze_sha256='wrong')
        with self.assertRaises(PermissionError): validate_clearance(c,f,'right')
    def test_29_each_clearance_binding_checked(self):
        f={k:{} for k in ['code_sha256','manifest_sha256','input_sha256','prediction_sha256','selection_sha256']}
        f.update(version='v',model_sha256='m')
        c=dict(f,schema='TD1_CLEARANCE_v2',status='CLEARED',freeze_sha256='right')
        validate_clearance(c,f,'right')
        for key in f:
            bad=copy.deepcopy(c); bad[key]='altered'
            with self.assertRaises(PermissionError): validate_clearance(bad,f,'right')
    def test_30_absent_clearance_no_payload(self):
        with patch('evaluate.CLEARANCE',self.path/'absent.json'):
            with patch('sources.dated',side_effect=AssertionError('payload read')):
                with self.assertRaises(PermissionError): admit()
    def test_31_append_only_outputs(self):
        path=self.path/'append.json'; new_json(path,{'x':1})
        with self.assertRaises(FileExistsError): new_json(path,{'x':2})
    def test_32_private_output_root(self):
        with self.assertRaises(ValueError): new_json(Path(__file__).parent/'not_allowed.json',{})
    def test_33_resource_limit(self):
        self.assertLess(CAP,2_600_000_000); self.assertGreaterEqual(guard(),MIN_FREE)
    def test_34_mlp_architecture_and_scalar(self):
        from fit import model
        m=model(.001,.0001); rng=np.random.default_rng(1)
        m.partial_fit(rng.normal(size=(40,36)),rng.normal(size=40))
        self.assertEqual([v.shape for v in m.coefs_],[(36,64),(64,32),(32,1)])
        self.assertEqual(m.activation,'relu')
    def test_35_common_inputs_future_invariance(self):
        from build import common_all
        r=np.tile(self.r[:,None],(1,60)); me=np.full_like(r,10.); ff=np.ones_like(r)
        a=common_all(r,me,ff,25); r[26:]=9999; me[26:]=0; ff[26:]=49
        b=common_all(r,me,ff,25)
        for key in a: np.testing.assert_equal(a[key],b[key])
    def test_36_vectorized_common_matches_reference(self):
        from build import common_all, common_inputs
        rng=np.random.default_rng(4); r=rng.normal(0,.01,(40,60)); me=rng.uniform(1,4,(40,60)); ff=np.ones_like(r)
        a=common_all(r,me,ff,25); b=common_inputs(r,me,ff,25,3)
        for k in b: self.assertAlmostEqual(a[k][3],b[k],places=13)
    def test_37_clock_dst_and_strict_report_day(self):
        self.assertEqual(clock('2020-07-01').hour,3)
        self.assertEqual(clock('2020-01-01').hour,4)
        with self.assertRaises(ValueError): require_past([clock('2020-07-01')],clock('2020-07-01'))
    def test_38_feature_allowlist(self):
        self.assertFalse(any('news' in c or 'supplier' in c or 'customer' in c for c in FEATURES))
        self.assertEqual(len(FEATURES),18); self.assertEqual(len(OLD),8)
    def test_39_string_date_snapshot(self):
        path=self.path/'strings.parquet'
        io.write_parquet(path,pd.DataFrame({'datadate':['2024-12-30','2025-01-01'],'x':[1,999]}))
        f=io.dated(path,['x'],date_col='datadate')
        self.assertEqual(f.x.tolist(),[1])
    def test_40_quarter_report_date_and_fallback(self):
        import build
        q=pd.DataFrame({'gvkey':['1','2'],'datadate':['2015-03-31','2015-03-31'],
            'atq':['100','200'],'cheq':['10','20'],'invtq':['5','6'],'dlcq':['2',''], 'dlttq':['3','4']})
        e=pd.DataFrame({'gvkey':['1','2'],'datadate':['2015-03-31','2015-03-31'],'rdq':['2015-05-01','']})
        with patch('build.read_input',side_effect=lambda n,**kw:q.copy() if n=='quarters.parquet' else e.copy()):
            with patch('sources.read_keys',return_value=pd.DataFrame({'permno':[101,102],'gvkey':[1,2]})):
                values,_=build.accounting()
        values=values.set_index('permno')
        self.assertEqual(values.loc[101,'availability'],clock('2015-05-01'))
        self.assertEqual(values.loc[102,'availability'],clock(pd.Timestamp('2015-03-31')+pd.Timedelta(days=120)))
        self.assertTrue(np.isnan(values.loc[102,'leverage']))
    def test_41_manifest_integration_never_calls_evaluation_label(self):
        import build
        for year in [2015,2023]:
            dates=pd.bdate_range(f'{year}-01-01',periods=40); ids=np.arange(100,160)
            d=pd.DataFrame({'date':np.repeat(dates,60),'permno':np.tile(ids,40),
                'ret':np.tile(np.linspace(-.01,.02,40), (60,1)).T.ravel(),
                'me':10.,'ff49':1.,'share_type':0})
            empty=d.iloc[:0].copy()
            q=pd.DataFrame(columns=['permno','availability','datadate','availability_rule','cash','inventory','leverage'])
            e=pd.DataFrame(columns=['permno','availability'])
            cf=pd.DataFrame(columns=['permno','availability','eom','ocf_at'])
            keys=pd.DataFrame({'date':[dates[25]],'permno':[100]})
            def reader(name,**kwargs):
                return d.copy() if name==f'crsp_{year}.parquet' else empty.copy()
            with patch('build.read_input',side_effect=reader), patch('build.MANIFEST',self.path/f'manifest_{year}'):
                if year==2023:
                    with patch('build.label',side_effect=AssertionError('Evaluation label called')):
                        result=build.build_year(year,keys,q,e,cf)
                else: result=build.build_year(year,keys,q,e,cf)
            path=self.path/f'manifest_{year}/year_{year}.parquet'
            names=io.schema_names(path)
            self.assertEqual('y' in names,year<=2022); self.assertEqual(result['rows'],1)
    def test_42_legacy_panel_keys_only(self):
        with self.assertRaises(ValueError): io.dated(io.PANEL,['date','permno','y'])
    def test_43_reloaded_training_layout_exact(self):
        rng=np.random.default_rng(9)
        x=np.ascontiguousarray(rng.normal(size=(500,18)))
        p=prepare_fit(x); reloaded=np.asfortranarray(x)
        q=prepare_fit(np.ascontiguousarray(reloaded))
        for k in p: np.testing.assert_array_equal(p[k],q[k])

def run_and_record():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Fixture)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'synthetic_only':True,**peak_bytes()}
    report['utc']=utc(); new_json(OUT/('SYNTHETIC_TESTS_'+uuid.uuid4().hex+'.json'),report)
    if not result.wasSuccessful(): raise SystemExit(1)
    return report
