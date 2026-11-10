"""Fail-closed gates and production adapter checks using invented objects only."""
import ast
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import decoder
import evaluation
from runtime_v3 import CODE,OUT
from spec import VERSION

def synthetic_binding():
    f={'version':VERSION,'evaluation_years':[2023,2024],'maximum_label_end':'2024-12-31',
        **{k:{'synthetic':'hash'} for k in evaluation.BINDINGS}}
    c=dict(f,schema='TD2_CLEARANCE_v1',status='CLEARED',freeze_sha256='syntheticfreeze')
    return c,f

class Boundary(unittest.TestCase):
    def test_valid_clearance_shape(self):
        c,f=synthetic_binding(); evaluation.validate_clearance(c,f,'syntheticfreeze')
    def test_absent_clearance_before_source_read(self):
        with patch.object(evaluation,'CLEARANCE',OUT/'fixtures/NONEXISTENT_CLEARANCE.json'),patch.object(decoder,'admitted_returns',side_effect=AssertionError('payload read')):
            with self.assertRaises(PermissionError): evaluation.run()
    def test_all_hash_bindings_rejected_if_different(self):
        for field in evaluation.BINDINGS+['freeze_sha256']:
            c,f=synthetic_binding(); c[field]='tampered'
            with self.assertRaises(PermissionError): evaluation.validate_clearance(c,f,'syntheticfreeze')
    def test_scope_cannot_include_2025(self):
        c,f=synthetic_binding(); c['evaluation_years']=[2023,2024,2025]
        with self.assertRaises(PermissionError): evaluation.validate_clearance(c,f,'syntheticfreeze')
    def test_label_refuses_without_token(self):
        with self.assertRaises(PermissionError): evaluation.label(None,None,None,None)
    def test_return_decoder_refuses_without_token(self):
        with self.assertRaises(PermissionError): decoder.admitted_returns(2023,None)
    def test_dev_manifest_rejects_eval_label_before_read(self):
        with patch.object(decoder,'verify_td1',side_effect=AssertionError('source read')):
            with self.assertRaises(PermissionError): decoder.manifest(2023,True)
    def test_2025_build_rejected(self):
        import pipeline
        with patch.object(pipeline,'require_synthetic'),patch.object(pipeline,'read_menu',return_value=['size']):
            with self.assertRaises(ValueError): pipeline.build_year(2025)
    def test_mapping_respects_asof_and_sensitivity(self):
        graph=pd.DataFrame([dict(id='ok',start='2019-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='b',rownum=1),
            dict(id='future',start='2021-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='b',rownum=2)])
        links=pd.DataFrame([('a',1,'2010-01-01','2024-12-31'),('b',2,'2010-01-01','2024-12-31')],columns=['entity','permno','start','end'])
        self.assertEqual([r['id'] for r in decoder.mapped_edges(graph,links,'2020-01-01')],['ok'])
        self.assertEqual(decoder.mapped_edges(graph,links,'2020-01-01',{('b',2)}),[])
    def test_mapping_ambiguity_dropped(self):
        graph=pd.DataFrame([dict(id='ok',start='2019-01-01',end=None,status='missing',typ='CUSTOMER',src='a',tgt='b',rownum=1)])
        links=pd.DataFrame([('a',1,'2010-01-01','2024-12-31'),('a',3,'2010-01-01','2024-12-31'),('b',2,'2010-01-01','2024-12-31')],columns=['entity','permno','start','end'])
        self.assertEqual(decoder.mapped_edges(graph,links,'2020-01-01'),[])
    def test_row_join_refuses_duplicates(self):
        from pipeline import checked_join
        f=pd.DataFrame([dict(firm_id='a',cutoff='t')]); labels=pd.concat([f,f])
        with self.assertRaises(pd.errors.MergeError): checked_join(f,labels)
    def test_production_no_network_import(self):
        forbidden={'requests','httpx','urllib','socket','wrds','openai','anthropic'}
        for p in CODE.glob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(forbidden.intersection(a.name.split('.')[0] for a in node.names),p.name)
                if isinstance(node,ast.ImportFrom): self.assertNotIn((node.module or '').split('.')[0],forbidden,p.name)
    def test_raw_market_readers_only_in_decoder(self):
        for name in ['engine.py','selectors.py','pipeline.py','evaluation.py']:
            tree=ast.parse((CODE/name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Attribute):
                    self.assertNotIn(node.attr,{'read_parquet','read_csv','ParquetFile','read_table','connect','read_feather'},name)
    def test_clipped_label_window_rejected_before_returns(self):
        class Poison:
            def __getitem__(self,key): raise AssertionError('future returns touched')
        previous=decoder._evaluation_token; token=object(); decoder._evaluation_token=token
        try:
            dates=pd.bdate_range('2024-12-30',periods=6)
            self.assertEqual(evaluation.label(dates,Poison(),0,token),(None,None))
        finally: decoder._evaluation_token=previous

def run_boundary():
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Boundary))
    return {'tests':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors),
            'failures':len(result.failures),'errors':len(result.errors),'synthetic_only':True}
