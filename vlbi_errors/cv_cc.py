import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from spydiff import clean_n
from from_fits import create_model_from_fits_file
from utils import to_boolean_array


# TODO: Add optional ``rewrite`` argument to ``UVData.save``
# TODO: Use only positive weighted data for CV
class KFoldCV(object):
    def __init__(self, fname, k, basename='cv', seed=None):
        self.fname = fname
        self.uvdata = UVData(fname)
        self.k = k
        self.seed = seed
        self.basename = basename
        self.test_fname = "{}_test.FITS".format(basename)
        self.train_fname = "{}_train.FITS".format(basename)
        self.baseline_folds = None
        self.create_folds()

    def create_folds(self):
        baseline_folds = dict()
        for bl, indxs in self.uvdata._indxs_baselines.items():
            print "Baseline {} has {} samples".format(bl,
                                                      np.count_nonzero(indxs))
            try:
                kfold = KFold(np.count_nonzero(indxs), self.k, shuffle=True,
                              random_state=self.seed)
                baseline_folds[bl] = list()
                for train, test in kfold:
                    tr = to_boolean_array(np.nonzero(indxs)[0][train], len(indxs))
                    te = to_boolean_array(np.nonzero(indxs)[0][test], len(indxs))
                    baseline_folds[bl].append((tr, te))
            # When ``k`` more then number of baseline samples
            except ValueError:
                pass
        self.baseline_folds = baseline_folds

    def __iter__(self):
        for i in xrange(self.k):
            train_indxs = np.zeros(len(self.uvdata.hdu.data))
            test_indxs = np.zeros(len(self.uvdata.hdu.data))
            for bl, kfolds in self.baseline_folds.items():
                itrain, itest = kfolds[i]
                # itrain = to_boolean_array(itrain)
                train_indxs = np.logical_or(train_indxs, itrain)
                test_indxs = np.logical_or(test_indxs, itest)
            train_data = self.uvdata.hdu.data[train_indxs]
            test_data = self.uvdata.hdu.data[test_indxs]
            self.uvdata.save(self.test_fname, test_data, rewrite=True)
            self.uvdata.save(self.train_fname, train_data, rewrite=True)

            yield self.train_fname, self.test_fname


if __name__ == '__main__':
    # 45.19953388864762 = min(scores) + sigma_min
    # cc_pars = np.linspace(100, 300, 2)
    cc_pars = [100, 1000, 3000, 5000, 10000, 15000, 20000, 30000]
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/clean_n'
    uv_fits = '/home/ilya/data/3c273/1226+023.x.2006_06_15.uvf'
    cv_scores = dict()
    n_folds = 10
    for niter in cc_pars:
        print "Using niter = {}".format(niter)
        kfold = KFoldCV(uv_fits, n_folds)
        cv = list()
        for j, (tr_fname, ts_fname) in enumerate(kfold):
            clean_n(kfold.train_fname, 'trained_model_{}.FITS'.format(niter), 'I',
                    (512, 0.15), niter=niter, path_to_script=path_to_script,
                    show_difmap_output=True)
            tr_model = create_model_from_fits_file('trained_model_{}.FITS'.format(niter))
            ts_uvdata = UVData(ts_fname)
            score = ts_uvdata.cv_score(tr_model)
            print "{} of {} gives {}".format(j+1, n_folds, score)
            cv.append(score)
        cv_scores[niter] = (np.nanmean(cv), np.nanstd(cv))
        print "CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv))

    print cv_scores
    n = cv_scores.keys()
    scores = [cv_scores[i][0] for i in n]
    errors = [cv_scores[i][1] for i in n]
    import matplotlib.pyplot as plt
    plt.errorbar(n, scores, errors, fmt='.k')
    min_score = min(scores)
    min_error = errors[scores.index(min_score)]
    s = min_score + min_error
    plt.axhline(s)
    plt.show()
