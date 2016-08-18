import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from spydiff import clean_difmap
from from_fits import create_model_from_fits_file


# TODO: Add optional ``rewrite`` argument to ``UVData.save``
# TODO: Use only positive weighted data for CV
class KFoldCV(object):
    def __init__(self, fname, k, basename='cv'):
        self.fname = fname
        self.uvdata = UVData(fname)
        self.k = k
        self.basename = basename
        self.test_fname = "{}_test.FITS".format(basename)
        self.train_fname = "{}_train.FITS".format(basename)
        self.baseline_folds = None
        self.create_folds()

    def create_folds(self):
        baseline_folds = dict()
        for bl, indxs in self.uvdata._indxs_baselines.items():
            kfold = KFold(np.count_nonzero(indxs), self.k, shuffle=True)
            baseline_folds[bl] = kfold
        self.baseline_folds = baseline_folds

    def __iter__(self):
        for i in xrange(self.k):
            test_indxs = list()
            train_indxs = list()
            for bl, kfold in self.baseline_folds.items():
                itrain, itest = list(kfold)[i]
                train_indxs.extend(itrain)
                test_indxs.extend(itest)
            train_data = self.uvdata.hdu.data[train_indxs]
            test_data = self.uvdata.hdu.data[test_indxs]
            self.uvdata.save(self.test_fname, test_data, rewrite=True)
            self.uvdata.save(self.train_fname, train_data, rewrite=True)

            yield self.train_fname, self.test_fname


seed = 42
cc_pars = list()
cv_scores = dict()
for cc_par in cc_pars:
    kfold = KFoldCV('some.fits', 50, seed)
    cv = list()
    for tr_fname, ts_fname in kfold:
        clean_difmap(kfold.train_fname, 'trained_model.FITS', cc_par)
        tr_model = create_model_from_fits_file('trained_model.FITS')
        ts_uvdata = UVData(ts_fname)
        cv.append(ts_uvdata.cv_score(tr_model))
    cv_scores[cc_par] = (np.mean(cv), np.std(cv))
