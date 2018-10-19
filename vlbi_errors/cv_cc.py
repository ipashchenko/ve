import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from spydiff import clean_n
from from_fits import create_model_from_fits_file
from utils import to_boolean_array


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
            print("saving data to {} and {}".format(self.test_fname,
                                                    self.train_fname))
            self.uvdata.save(self.test_fname, test_data, rewrite=True)
            self.uvdata.save(self.train_fname, train_data, rewrite=True)

            yield self.train_fname, self.test_fname


if __name__ == '__main__':
    from cv_model import score as score_
    # 45.19953388864762 = min(scores) + sigma_min
    cc_pars = np.linspace(100, 2000, 5)
    cc_pars = [50, 75, 100, 125, 150, 200, 300, 500, 750, 1000, 1500, 2500, 5000, 10000]
    path_to_script = '/home/ilya/github/ve/difmap/clean_n'
    # path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw_n'
    # uv_fits = '/home/ilya/data/3c273/1226+023.x.2006_06_15.uvf'
    uv_fits = '/home/ilya/data/cv_cc/0055+300.u.2006_02_12.uvf'
    # uv_fits = '/home/ilya/data/check_cv_misha/1226+023.X1.2010_01_26.UV_CAL'
    # windows = '/home/ilya/data/check_cv_misha/1226+023.X1.2010_01_26.win'
    cv_scores = dict()
    n_folds = 10
    for niter in cc_pars:
        print("Using niter = {}".format(niter))
        kfold = KFoldCV(uv_fits, n_folds)
        cv = list()
        for j, (tr_fname, ts_fname) in enumerate(kfold):
            clean_n(kfold.train_fname, 'trained_model_{}.FITS'.format(niter), 'i',
                    (1024, 0.1), niter=niter, path_to_script=path_to_script,
                    show_difmap_output=True)
            tr_model = create_model_from_fits_file('trained_model_{}.FITS'.format(niter))
            ts_uvdata = UVData(ts_fname)
            # score = ts_uvdata.cv_score(tr_model)
            score = score_(ts_fname, 'trained_model_{}.FITS'.format(niter))
            print("{} of {} gives {}".format(j+1, n_folds, score))
            cv.append(score)
        cv_scores[niter] = (np.nanmean(cv), np.nanstd(cv))
        print("CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv)))

    print(cv_scores)
    n = cv_scores.keys()
    scores = [cv_scores[i][0] for i in n]
    errors = [cv_scores[i][1] for i in n]

    import matplotlib
    label_size = 20
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    #ax.semilogy()
    #ax.semilogx()
    scores = np.array(scores)
    errors = np.array(errors)
    ax.errorbar(n, 1000*scores, 1000*errors, fmt='.k')
    ax.set_xlim([90, None])
    ax.set_ylim([57.5, 70])
    min_score = 1000*min(scores)
    # min_error = errors[scores.index(min_score)]
    s = min_score
    ax.axhline(s)
    ax.set_xlabel(r"$N_{\rm CC}$")
    # ax.xaxis.set_ticks(np.arange(start, end, 0.712123))
    ax.set_ylabel(r"RMSE, mJy")
    fig.show()
    fig.savefig('/home/ilya/data/boot/cv_cc.pdf',
                bbox_inches='tight', format='pdf', dpi=600)
    # plt.savefig('/home/ilya/Dropbox/papers/boot/new_pics/cv_cc.svg',
    #             bbox_inches='tight', format='svg', dpi=1200)
