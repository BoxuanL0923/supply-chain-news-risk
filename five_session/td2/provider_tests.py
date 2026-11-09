import copy
import unittest
import pandas as pd
import numpy as np
from spec import NAMES,SLOT_SCHEMAS
from provider_harness import gates,tune_selection,slot_lookup,context_hash

class Provider(unittest.TestCase):
    def frame(self,year=2021):
        return pd.DataFrame([dict(firm_id='synthetic',cutoff=f'{year}-06-04',origin_session=f'{year}-06-04',
            label_end=f'{year}-06-11',b=0.,y=1.,**{'missing_'+n:False for n in NAMES},**{'Z_'+n:1. for n in NAMES})])
    def slot(self):
        s=copy.deepcopy(SLOT_SCHEMAS['deepseek_order']); s['model_version']='synthetic-pinned'
        s['records']=[dict(firm_id='synthetic',cutoff='2021-06-04',input_sha256='a'*64,receipt_id='synthetic',ordering=NAMES)]
        return s
    def test_empty_pending_not_invented_parameters(self):
        r=tune_selection(self.frame(),NAMES,[1]*5,'deepseek_order',SLOT_SCHEMAS['deepseek_order'],{})
        self.assertIsNone(r['selected'])
    def test_tuning_2023_refused_before_score(self):
        with self.assertRaises(PermissionError): tune_selection(self.frame(2023),NAMES,[1]*5,'deepseek_order',self.slot(),{})
    def test_tuning_label_end_crossing_refused(self):
        f=self.frame(); f['label_end']='2023-01-03'
        with self.assertRaises(PermissionError): tune_selection(f,NAMES,[1]*5,'deepseek_order',self.slot(),{})
    def test_hash_mismatch_refused(self):
        with self.assertRaises(ValueError): slot_lookup(self.slot(),'deepseek_order',NAMES,{('synthetic','2021-06-04 00:00:00'):'b'*64})
    def test_complete_order_to_features(self):
        q,missing=gates(self.frame(),NAMES,'deepseek_order',self.slot(),2,None,'none',{('synthetic','2021-06-04 00:00:00'):'a'*64})
        self.assertEqual(q.tolist(),[[1,1,0,0,0]]); self.assertFalse(missing)
    def test_missing_record_fallback_is_explicit(self):
        q,missing=gates(self.frame(),NAMES,'deepseek_order',SLOT_SCHEMAS['deepseek_order'],2,None,'none',{})
        self.assertEqual(q.sum(),0); self.assertEqual(len(missing),1)
    def test_selects_on_selection_only(self):
        r=tune_selection(self.frame(),NAMES,[1]*5,'deepseek_order',self.slot(),{('synthetic','2021-06-04 00:00:00'):'a'*64})
        self.assertEqual(r['selected']['K'],1)
    def test_placeholder_version_cannot_accept_records(self):
        s=self.slot(); s['model_version']='REQUIRED'
        with self.assertRaises(ValueError): slot_lookup(s,'deepseek_order',NAMES,{('synthetic','2021-06-04 00:00:00'):'a'*64})

def run_provider():
    r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Provider))
    return {'tests':r.testsRun,'passed':r.testsRun-len(r.errors)-len(r.failures),'errors':len(r.errors),'failures':len(r.failures)}
