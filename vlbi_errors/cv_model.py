import os
import glob
import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from utils import to_boolean_array, mask_boolean_with_boolean
from spydiff import import_difmap_model, modelfit_difmap, clean_difmap
from model import Model
from from_fits import create_model_from_fits_file
import matplotlib.pyplot as plt


# TODO: Use only positive weighted data for CV
class KFoldCV(object):
    def __init__(self, uv_fits_path, k, basename='cv', seed=None, baselines=None):
        self.uv_fits_path = uv_fits_path
        self.uvdata = UVData(uv_fits_path)
        self.k = k
        self.seed = seed
        self.basename = basename
        self.test_fname_base = "{}_test".format(basename)
        self.train_fname_base = "{}_train".format(basename)
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
                    tr = to_boolean_array(np.nonzero(bl_indxs)[0][train],
                                          len(bl_indxs))
                    te = to_boolean_array(np.nonzero(bl_indxs)[0][test],
                                          len(bl_indxs))
                    baseline_folds[bl].append((tr, te))
            # When ``k`` more then number of baseline samples
            except ValueError:
                pass

        # Add all other baselines data w/o folding - all data to train & nothing
        # to test
        rest_baselines = list(self.uvdata.baselines)
        for bl in baselines:
            rest_baselines.remove(bl)
        for bl in rest_baselines:
            baseline_folds[bl] = list()
        for bl in rest_baselines:
            bl_indxs = self.uvdata._indxs_baselines[bl]
            for k in range(self.k):
                baseline_folds[bl].append((bl_indxs, np.zeros(len(bl_indxs),
                                                              dtype=bool)))

        self.baseline_folds = baseline_folds

    def create_train_test_data(self, outdir=None):
        if outdir is None:
            outdir = os.getcwd()
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
            self.uvdata.save(os.path.join(outdir, self.test_fname_base + '_{}.fits'.format(i)),
                             test_data, rewrite=True)
            self.uvdata.save(os.path.join(outdir, self.train_fname_base + '_{}.fits'.format(i)),
                             train_data, rewrite=True)

    def cv_score(self, initial_dfm_model_path=None, data_dir=None, niter=100,
                 path_to_script=None, mapsize_clean=None):
        if data_dir is None:
            data_dir = os.getcwd()
        train_uv_fits_paths = sorted(glob.glob(os.path.join(data_dir,
                                                            self.train_fname_base+'*')))
        test_uv_fits_paths = sorted(glob.glob(os.path.join(data_dir,
                                                           self.test_fname_base+'*')))
        cv_scores = list()
        train_scores = list()
        if initial_dfm_model_path is not None:
            for i, (train_uv_fits_path, test_uv_fits_path) in enumerate(zip(train_uv_fits_paths,
                                                                            test_uv_fits_paths)):
                out_mdl_fname = 'train_{}.mdl'.format(i)
                dfm_model_dir, dfm_model_fname = os.path.split(initial_dfm_model_path)
                modelfit_difmap(train_uv_fits_path, dfm_model_fname,
                                out_mdl_fname, niter=niter,
                                path=data_dir, mdl_path=dfm_model_dir,
                                out_path=data_dir)
                cv_scores.append(score(test_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))
                train_scores.append(score(train_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))
        else:
            for i, (train_uv_fits_path, test_uv_fits_path) in enumerate(zip(train_uv_fits_paths,
                                                                            test_uv_fits_paths)):
                out_mdl_fname = 'train_{}.fits'.format(i)
                clean_difmap(train_uv_fits_path, out_mdl_fname, 'I',
                             mapsize_clean, data_dir, path_to_script,
                             outpath=data_dir, show_difmap_output=True)
                cv_scores.append(score(test_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))
                train_scores.append(score(train_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))

        return cv_scores, train_scores

# def cv_model(dfm_model_files, uv_fits, K=10, dfm_model_dir=None, baselines=None,
#              dfm_niter=50):
#     if dfm_model_dir is None:
#         dfm_model_dir = os.getcwd()
#     mdl_dict = {i: mdl_file for (i, mdl_file) in enumerate(dfm_model_files)}
#     mdl_comps = [import_difmap_model(mdl_file, dfm_model_dir) for mdl_file in
#                  dfm_model_files]
#     models = [Model(stokes='I')] * len(dfm_model_files)
#     for model, comps in zip(models, mdl_comps):
#         model.add_components(*comps)
#
#     cv_scores = dict()
#     n_folds = K
#     for i in mdl_dict:
#         kfold = KFoldCV(uv_fits, n_folds, baselines=baselines)
#         cv = list()
#         for j, (tr_fname, ts_fname) in enumerate(kfold):
#             tr_uvdata = UVData(tr_fname)
#             ts_uvdata = UVData(ts_fname)
#             fig = tr_uvdata.uvplot()
#             ts_uvdata.uvplot(fig=fig, color='r')
#             fig.show()
#             modelfit_difmap(kfold.train_fname, mdl_dict[i],
#                             'trained_model_{}.mdl'.format(i),
#                             mdl_path=dfm_model_dir,
#                             niter=dfm_niter)
#             tr_comps = import_difmap_model('trained_model_{}.mdl'.format(i))
#             tr_model = Model(stokes='I')
#             tr_model.add_components(*tr_comps)
#             ts_uvdata = UVData(ts_fname)
#             score = ts_uvdata.cv_score(tr_model)
#             print "{} of {} gives {}".format(j+1, n_folds, score)
#             cv.append(score)
#         cv_scores[i] = (np.nanmean(cv), np.nanstd(cv))
#         # cv_scores[i] = (np.nanmean(cv), np.nanstd(cv)/np.sqrt(K))
#         # print "CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv)/np.sqrt(K))
#         print "CV gives {} +/- {}".format(np.nanmean(cv), np.nanstd(cv))
#
#     return np.array(cv_scores)


def score(uv_fits_path, mdl_path):
    """
    Returns rms of model on given uv-data for stokes 'I".
    
    :param uv_fits_path: 
        Path to uv-fits file.
    :param mdl_path: 
        Path to difmap model text file or FITS-file with CLEAN model.
    :return: 
        Per-point rms between given data and model evaluated at given data
        points.
    """
    uvdata = UVData(uv_fits_path)
    uvdata_model = UVData(uv_fits_path)
    try:
        model = create_model_from_fits_file(mdl_path)
    except IOError:
        dfm_mdl_dir, dfm_mdl_fname = os.path.split(mdl_path)
        comps = import_difmap_model(dfm_mdl_fname, dfm_mdl_dir)
        model = Model(stokes='I')
        model.add_components(*comps)
    uvdata_model.substitute([model])
    uvdata_diff = uvdata - uvdata_model
    i_diff = 0.5 * (uvdata_diff.uvdata_weight_masked[..., 0] +
                    uvdata_diff.uvdata_weight_masked[..., 1])
    factor = np.count_nonzero(i_diff)
    # factor = np.count_nonzero(~uvdata_diff.uvdata_weight_masked.mask[:, :, :2])
    # squared_diff = uvdata_diff.uvdata_weight_masked[:, :, :2] * \
    #                uvdata_diff.uvdata_weight_masked[:, :, :2].conj()
    squared_diff = i_diff * i_diff.conj()
    return np.sqrt(float(np.sum(squared_diff)) / factor)

if __name__ == '__main__':
    data_dir = '/home/ilya/silke'
    epoch = '2017_01_28'
    original_model_fname = '2017_01_28us'
    original_model_path = os.path.join(data_dir, original_model_fname)
    from mojave import mojave_uv_fits_fname
    uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
    uv_fits_path = os.path.join(data_dir, uv_fits_fname)
    kfold = KFoldCV(uv_fits_path, 5)
    kfold.create_train_test_data(outdir=data_dir)
    cv_scores = kfold.cv_score(original_model_path, data_dir=data_dir)

    # dfm_mdl_files = ['k1mod1.mdl']
    # # uv_fits = '/home/ilya/code/vlbi_errors/bin_q/0235+164.q1.2008_09_02.uvf_difmap'
    # # uv_fits = '/home/ilya/code/vlbi_errors/bin_u/0235+164.u1.2008_09_02.uvf_difmap'
    # uv_fits = '/home/ilya/Dropbox/0235/tmp/to_compare/0235+164.k1.2008_09_02.uvf_difmap'
    # cv_scores = cv_model(dfm_mdl_files, uv_fits, baselines=None, K=5,
    #                      dfm_model_dir='/home/ilya/Dropbox/0235/tmp/to_compare',
    #                      dfm_niter=50)
    # a = np.array(cv_scores.values())
    # y = a[:, 0]
    # yerr = a[:, 1]
    # label_size = 12
    # import matplotlib
    # matplotlib.rcParams['xtick.labelsize'] = label_size
    # matplotlib.rcParams['ytick.labelsize'] = label_size
    # matplotlib.rcParams['axes.titlesize'] = label_size
    # matplotlib.rcParams['axes.labelsize'] = label_size
    # matplotlib.rcParams['font.size'] = 12
    # matplotlib.rcParams['legend.fontsize'] = 12

    # import matplotlib.pyplot as plt
    # plt.figure()
    # # plt.errorbar(np.arange(len(dfm_mdl_files))+1, y, yerr, lw=2)
    # plt.plot(np.arange(len(dfm_mdl_files))+1, y)
    # # plt.xlim([0.9, len(dfm_mdl_files) + 0.1])
    # plt.xlabel("Model number")
    # plt.ylabel("CV score")
    # # plt.xticks(range(len(dfm_mdl_files)))
    # plt.show()
    # # plt.savefig('/home/ilya/Dropbox/papers/boot/new_pics/cv_cc.eps',
    # #             bbox_inches='tight', format='eps', dpi=500)
    # # plt.savefig('/home/ilya/Dropbox/papers/boot/new_pics/cv_cc.svg',
    # #             bbox_inches='tight', format='svg', dpi=500)


    # # cv_scores_ = list()
    # # for i in range(10):
    # #     cv_scores = cv_model(dfm_mdl_files, uv_fits, baselines=[774, 1546], K=10,
    # #                          dfm_model_dir='/home/ilya/code/vlbi_errors/bin_c1')
    # #     cv_scores_.append(cv_scores)
    # # print cv_scores_
    # # import matplotlib.pyplot as plt
    # # plt.figure()
    # # a = np.array(cv_scores_.values())[..., 0].T
    # # for ar in a:
    # #     plt.plot(np.arange(len(dfm_mdl_files)) +
    # #              np.random.normal(0, 0.03, size=3), ar, '.k', lw=2)
    # # plt.xlim([-0.1, len(dfm_mdl_files) -0.9])
    # # plt.xlabel("Model number")
    # plt.ylabel("CV score, lower - better")
    # plt.xticks(range(len(dfm_mdl_files)))
    # plt.show()


