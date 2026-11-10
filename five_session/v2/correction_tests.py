"""Regression tests for the comparator repair and active artifact gate."""
import copy
import json
import pickle
import unittest
import uuid
from unittest.mock import patch
import numpy as np
import pandas as pd
from runtime import OUT, CODE, sha, new_json, peak_bytes, freeze_path
import core
import sources
import evaluate
from validate_correction import independent_statistics, assert_preprocessing, extreme_selection_missingness_trap

class Correction(unittest.TestCase):
    def fixture(self):
        x = np.array([[1.,np.nan],[3.,4.],[5.,8.],[7.,12.]])
        p = independent_statistics(x)
        return x, {'selection_old': {'preprocess': copy.deepcopy(p)}, 'old': {'preprocess': copy.deepcopy(p)}}

    def test_01_extreme_selection_missingness(self):
        self.assertTrue(extreme_selection_missingness_trap())

    def test_02_independent_statistics_never_calls_prepare_fit(self):
        x, b = self.fixture()
        with patch('core.prepare_fit', side_effect=AssertionError('Refitted preprocessing')):
            self.assertEqual(assert_preprocessing(x, b), 6)

    def test_03_each_selection_statistic_rejected_independently(self):
        x, b = self.fixture()
        for k in ('median','mean','scale'):
            damaged = copy.deepcopy(b); damaged['selection_old']['preprocess'][k][0] += 1
            with self.assertRaisesRegex(ValueError, 'independent 2015-2020'):
                assert_preprocessing(x, damaged)

    def test_04_each_final_statistic_rejected_exactly(self):
        x, b = self.fixture()
        for k in ('median','mean','scale'):
            damaged = copy.deepcopy(b)
            damaged['old']['preprocess'][k][0] = np.nextafter(damaged['old']['preprocess'][k][0], np.inf)
            with self.assertRaisesRegex(ValueError, 'differs from selection'):
                assert_preprocessing(x, damaged)

    def test_05_reuse_copies_without_fitting(self):
        x, b = self.fixture(); p = b['selection_old']['preprocess']
        with patch('core.prepare_fit', side_effect=AssertionError('Refitted preprocessing')):
            m = core.old_fit(x, np.arange(len(x),dtype=float), preprocess=p)
        for k in p:
            np.testing.assert_array_equal(m['preprocess'][k],p[k])
            self.assertFalse(np.shares_memory(m['preprocess'][k],p[k]))

    def test_06_active_artifact_paths(self):
        self.assertEqual(freeze_path(), OUT/'MODEL_FREEZE_v3.json')
        self.assertEqual(evaluate.CLEARANCE, OUT/'clearance/TD1_CLEARANCE_v2.json')
        with patch('sources.pickle', create=True):
            with patch('pathlib.Path.open', side_effect=FileNotFoundError) as opened:
                with self.assertRaises(FileNotFoundError): sources.read_frozen_model()
                self.assertEqual(opened.call_args.args, ('rb',))
        self.assertIn('model_frozen_v2.pkl', (CODE/'sources.py').read_text())
        self.assertIn('predictions_v2/year_', (CODE/'evaluate.py').read_text())

    def test_07_old_clearance_schema_rejected(self):
        f={k:{} for k in evaluate.BINDINGS}; f.update(version='synthetic',model_sha256='m')
        c=dict(f,schema='TD1_CLEARANCE_v1',status='CLEARED',freeze_sha256='h')
        with self.assertRaises(PermissionError): evaluate.validate_clearance(c,f,'h')

    def test_08_real_bundle_preserves_training_preprocessing(self):
        freeze=json.loads(freeze_path().read_text())
        self.assertEqual(sha(OUT/'model_frozen_v2.pkl'),freeze['model_sha256'])
        b=sources.read_frozen_model()
        for k in ('median','mean','scale'):
            np.testing.assert_array_equal(b['old']['preprocess'][k], b['selection_old']['preprocess'][k])

def run():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Correction))
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'evaluation_labels_read_or_computed':False,**peak_bytes()}
    new_json(OUT/'repair_v2'/('CORRECTION_TESTS_'+uuid.uuid4().hex+'.json'),report)
    if not result.wasSuccessful(): raise SystemExit(1)
    return report
