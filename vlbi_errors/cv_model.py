import os
import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from utils import to_boolean_array, mask_boolean_with_boolean
from spydiff import import_difmap_model, modelfit_difmap
from model import Model


# TODO: Use only positive weighted data for CV
class KFoldCV(object):
    def __init__(self, fname, k, basename='cv', seed=None, baselines=None):
        self.fname = fname
        self.uvdata = UVData(fname)
        self.k = k
        self.seed = seed
        self.basename = basename
        self.test_fname = "{}_test.FITS".format(basename)
        self.train_fname = "{}_train.FITS".format(basename)
        self.baseline_folds = None
        self.create_folds(baselines)

    def create_folds(self, baselines=None):
        baseline_folds = dict()
        if baselines is None:
            baselines = self.uvdata.baselines
        for bl in baselines:
            bl_indxs = self.uvdata._indxs_baselines[bl]
            print "Baseline {} has {} samples".format(bl,
                                                      np.count_nonzero(bl_indxs))
            bl_indxs_pw = self.uvdata.pw_indxs_baseline(bl, average_bands=True,
                                                        stokes=['RR', 'LL'],
                                                        average_stokes=True)
            bl_indxs = mask_boolean_with_boolean(bl_indxs, bl_indxs_pw)
            print "Baseline {} has {} samples with positive weight".format(bl,
                                                      np.count_nonzero(bl_indxs))

            try:
                kfold = KFold(np.count_nonzero(bl_indxs), self.k, shuffle=True,
                              random_state=self.seed)
                baseline_folds[bl] = list()
                for train, test in kfold:
                    tr = to_boolean_array(np.nonzero(bl_indxs)[0][train], len(bl_indxs))
                    te = to_boolean_array(np.nonzero(bl_indxs)[0][test], len(bl_indxs))
                    baseline_folds[bl].append((tr, te))
            # When ``k`` more then number of baseline samples
            except ValueError:
                pass

        # Add all other baselines data w/o folding - all data to train & nothing
        # to test
        rest_baselines = list(self.uvdata.baselines)
        for bl in baselines:
            print "removing baseline {}".format(bl)
            rest_baselines.remove(bl)
        for bl in rest_baselines:
            baseline_folds[bl] = list()
        for bl in rest_baselines:
            print "Adding baseline {} to folds".format(bl)
            bl_indxs = self.uvdata._indxs_baselines[bl]
            for k in range(self.k):
                baseline_folds[bl].append((bl_indxs, np.zeros(len(bl_indxs),
                                                              dtype=bool)))

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


def cv_model(dfm_model_files, uv_fits, K=10, dfm_model_dir=None, baselines=None,
             dfm_niter=50):
    if dfm_model_dir is None:
        dfm_model_dir = os.getcwd()
    mdl_dict = {i: mdl_file for (i, mdl_file) in enumerate(dfm_model_files)}
    mdl_comps = [import_difmap_model(mdl_file, dfm_model_dir) for mdl_file in
                 dfm_model_files]
    models = [Model(stokes='I')] * len(dfm_model_files)
    for model, comps in zip(models, mdl_comps):
        model.add_components(*comps)

    cv_scores = dict()
    n_folds = K
    for i in mdl_dict:
        kfold = KFoldCV(uv_fits, n_folds, baselines=baselines)
        cv = list()
        for j, (tr_fname, ts_fname) in enumerate(kfold):
            modelfit_difmap(kfold.train_fname, mdl_dict[i],
                            'trained_model_{}.mdl'.format(i),
                            mdl_path=dfm_model_dir,
                            niter=dfm_niter)
            tr_comps = import_difmap_model('trained_model_{}.mdl'.format(i))
            tr_model = Model(stokes='I')
            tr_model.add_components(*tr_comps)
            ts_uvdata = UVData(ts_fname)
            score = ts_uvdata.cv_score(tr_model)
            print "{} of {} gives {}".format(j+1, n_folds, score)
            cv.append(score)
        cv_scores[i] = (np.nanmean(cv), np.nanstd(cv)/np.sqrt(K))
        # print "CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv)/np.sqrt(K))
        print "CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv))

    return cv_scores


if __name__ == '__main__':

    dfm_mdl_files = ['0235+164_L_delta_fitted.mdl',
                     '0235+164_L.mdl']
    # dfm_mdl_files = ['0235+164.u1.2008_09_02_delta_fitted.mdl',
    #                  '0235+164.u1.2008_09_02_cgauss_fitted.mdl',
    #                  '0235+164.u1.2008_09_02.mdl']
    uv_fits = '/home/ilya/code/vlbi_errors/pet/0235+164_L.uvf_difmap'
    # uv_fits = '/home/ilya/code/vlbi_errors/bin_u/0235+164.u1.2008_09_02.uvf_difmap'
    cv_scores = cv_model(dfm_mdl_files, uv_fits, baselines=None, K=5,
                         dfm_model_dir='/home/ilya/code/vlbi_errors/pet',
                         dfm_niter=50)
    a = np.array(cv_scores.values())
    y = a[:, 0]
    yerr = a[:, 1]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.errorbar(np.arange(len(dfm_mdl_files))+1, y, yerr, lw=2)
    plt.xlim([0.9, len(dfm_mdl_files) + 0.1])
    plt.xlabel("Model number")
    plt.ylabel("CV score")
    plt.xticks(range(len(dfm_mdl_files)))
    plt.show()


    # cv_scores_ = list()
    # for i in range(10):
    #     cv_scores = cv_model(dfm_mdl_files, uv_fits, baselines=[774, 1546], K=10,
    #                          dfm_model_dir='/home/ilya/code/vlbi_errors/bin_c1')
    #     cv_scores_.append(cv_scores)
    # print cv_scores_
    # import matplotlib.pyplot as plt
    # plt.figure()
    # a = np.array(cv_scores_.values())[..., 0].T
    # for ar in a:
    #     plt.plot(np.arange(len(dfm_mdl_files)) +
    #              np.random.normal(0, 0.03, size=3), ar, '.k', lw=2)
    # plt.xlim([-0.1, len(dfm_mdl_files) -0.9])
    # plt.xlabel("Model number")
    # plt.ylabel("CV score, lower - better")
    # plt.xticks(range(len(dfm_mdl_files)))
    # plt.show()


