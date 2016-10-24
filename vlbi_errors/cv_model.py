import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from spydiff import clean_n
from from_fits import create_model_from_fits_file
from utils import to_boolean_array, mask_boolean_with_boolean


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


if __name__ == '__main__':
    from spydiff import import_difmap_model, modelfit_difmap
    from model import Model
    mdl_dir = '/home/ilya/Dropbox/Ilya/0235_bk150_uvf_data/models_from_Sanya_new/'
    uvfits_dir = '/home/ilya/Dropbox/Ilya/0235_bk150_uvf_data/'
    # mdl_file_1 = '0235_q_single_core.mdl'
    # mdl_file_2 = '0235_q_2comp_ellipse_circ.mdl'
    mdl_files = ['mod_q1_1c.mdl', 'mod_q1_1e.mdl', 'mod_q1_2cc.mdl',
                 'mod_q1_2ce.mdl', 'mod_q1_2ec.mdl', 'mod_q1_2ee.mdl']
    mdl_dict = {i: mdl_file for (i, mdl_file) in enumerate(mdl_files)}
    # comps1 = import_difmap_model(mdl_file_1, mdl_dir)
    # comps2 = import_difmap_model(mdl_file_2, mdl_dir)
    mdl_comps = [import_difmap_model(mdl_file, mdl_dir) for mdl_file in
                 mdl_files]
    models = [Model(stokes='I') for mdl_file in mdl_files]
    for model, comps in zip(models, mdl_comps):
        model.add_components(*comps)
    # model1 = Model(stokes='I')
    # model2 = Model(stokes='I')
    # model1.add_components(*comps1)
    # model2.add_components(*comps2)
    # models = [model1, model2]
    import os
    uv_fits = '0235+164.q1.2008_09_02.uvf_difmap'
    cv_scores = dict()
    n_folds = 10
    for i in mdl_dict:
        kfold = KFoldCV(os.path.join(uvfits_dir, uv_fits), n_folds,
                        baselines=[774, 1546])
        cv = list()
        for j, (tr_fname, ts_fname) in enumerate(kfold):
            modelfit_difmap(kfold.train_fname, mdl_dict[i],
                            'trained_model_{}.mdl'.format(i), mdl_path=mdl_dir,
                            niter=50)
            tr_comps = import_difmap_model('trained_model_{}.mdl'.format(i))
            tr_model = Model(stokes='I')
            tr_model.add_components(*tr_comps)
            ts_uvdata = UVData(ts_fname)
            score = ts_uvdata.cv_score(tr_model)
            print "{} of {} gives {}".format(j+1, n_folds, score)
            cv.append(score)
        cv_scores[i] = (np.nanmean(cv), np.nanstd(cv))
        print "CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv))

    print cv_scores
    # n = cv_scores.keys()
    # scores = [cv_scores[i][0] for i in n]
    # errors = [cv_scores[i][1] for i in n]
    # import matplotlib.pyplot as plt
    # plt.errorbar(n, scores, errors, fmt='.k')
    # min_score = min(scores)
    # min_error = errors[scores.index(min_score)]
    # s = min_score + min_error
    # plt.axhline(s)
    # plt.show()
