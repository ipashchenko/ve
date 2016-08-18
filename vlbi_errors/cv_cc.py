import copy
import numpy as np
from uv_data import UVData
from sklearn.cv import KFold
from spydiff import clean_difmap
from from_fits import create_model_from_fits_file


class KFoldCV(object):
    def __init__(self, fname, k, basename='cv'):
        self.fname = fname
        self.uvdata = UVData(fname)
        self.k = k
        self.create_folds()
        
    def create_folds(self):
        baseline_folds = dict()
        for bl, indxs in self.uvdata._index_baselines.items():
            kfold = KFold(np.nonzero(indx), self.k, shuffle=True)
            baseline_folds[bl] = kfold
        self.baseline_folds = baseline_folds
            
    def __iter__(self):
        for i in xrange(self.k):
            test_indxs = list()
            train_indxs = list()
            for bl, kfold in self.baseline_folds.items():
                train, test = kfold[i]
                train_indxs.extend(train)
                test_indxs.extend(test)
            train_data = self.uvdata[train]
            test_data = self.uvdata[test]
            self.uvdata.save('cv_train.fits', train_data)
            self.uvdata.save('cv_test.fits', test_data)
                    
            yield train_fname, test_fname
            
           
cv_scores = dict()
for cc_par in cc_pars: 
    kfold = KFoldCV('some.fits', 50, seed)
    cv = list()
    for tr_fname, ts_fname in kfold:
        clean_difmap(tr_fname, tr_model_fn, cc_par)
        tr_model = create_model_from_fits_file(tr_model_fn)
        ts_uvdata = UVData(ts_fname)
        cv.append(ts_uvdata.cv_score(tr_model))
    cv_scores[cc_par] = (np.mean(cv), np.std(cv))
    
    