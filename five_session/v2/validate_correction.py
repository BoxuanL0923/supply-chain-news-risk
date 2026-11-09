"""Independent training statistics and the extreme-selection/missingness trap."""
import numpy as np

def independent_statistics(x):
    # Deliberately do not call core.prepare_fit. Preserve the fitter's layout.
    med = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    filled = np.where(np.isfinite(x), x, med)
    scale = np.std(filled, axis=0)
    scale[scale == 0] = 1.
    return {'median': med, 'mean': filled.mean(axis=0), 'scale': scale}

def assert_preprocessing(training, bundle):
    expected = independent_statistics(training)
    for key in ('median', 'mean', 'scale'):
        if not np.array_equal(expected[key], bundle['selection_old']['preprocess'][key]):
            raise ValueError('Selection comparator preprocessing differs from independent 2015-2020 statistics: ' + key)
        if not np.array_equal(bundle['old']['preprocess'][key], bundle['selection_old']['preprocess'][key]):
            raise ValueError('Final comparator preprocessing differs from selection comparator: ' + key)
    return 6

def independent_beta(x, y, p):
    filled = np.where(np.isfinite(x), x, p['median'])
    z = np.column_stack([np.ones(len(x)), (filled-p['mean'])/p['scale']])
    ridge = np.diag(np.r_[0., np.full(x.shape[1], 1e-6)])
    return np.linalg.solve(z.T@z + ridge, z.T@y)

def extreme_selection_missingness_trap():
    from core import old_fit
    train = np.array([[1.,np.nan],[3.,4.],[5.,8.],[7.,12.]])
    initial = old_fit(train, np.array([1.,2.,3.,4.]))
    extreme = np.array([[1e6,1e9],[np.nan,-1e9],[5e6,1e8]])
    joined = np.vstack([train, extreme]); response = np.arange(len(joined), dtype=float)
    fixed = old_fit(joined, response, preprocess=initial['preprocess'])
    unsafe = old_fit(joined, response)
    assert all(np.array_equal(fixed['preprocess'][k], initial['preprocess'][k]) for k in initial['preprocess'])
    assert any(not np.array_equal(unsafe['preprocess'][k], initial['preprocess'][k]) for k in initial['preprocess'])
    assert all(not np.shares_memory(fixed['preprocess'][k], initial['preprocess'][k]) for k in initial['preprocess'])
    np.testing.assert_array_equal(fixed['beta'], independent_beta(joined, response, initial['preprocess']))
    return True
