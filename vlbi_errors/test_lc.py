import os
import numpy as np
from cv_model import KFoldCV
from mojave import mojave_uv_fits_fname
from uv_data import UVData
import matplotlib.pyplot as plt


def learning_curve(uv_fits_path, fracs, K, initial_dfm_model_path=None,
                   n_iter=100, mapsize_clean=(512, 0.1), path_to_script=None,
                   n_splits=10, data_dir=None, ls_cv='-',
                   ls_train='-', plot=False):

    uvdata = UVData(uv_fits_path)
    cv_means = dict()
    train_means = dict()

    try:
        assert len(fracs)+1 == len(n_iter)
    except TypeError:
        n_iter = [n_iter]*len(fracs+1)

    for n, frac in zip(n_iter[:-1], fracs):
        cv_means[frac] = list()
        train_means[frac] = list()
        for i in range(n_splits):
            uv_frac_path = os.path.join(data_dir, 'frac_{}.fits'.format(frac))
            uvdata.save_fraction(uv_frac_path, frac,
                                 random_state=np.random.randint(0, 1000))
            kfold = KFoldCV(uv_frac_path, K, seed=np.random.randint(0, 1000))
            kfold.create_train_test_data(outdir=data_dir)
            cv_scores, train_scores = kfold.cv_score(initial_dfm_model_path=initial_dfm_model_path,
                                                     data_dir=data_dir,
                                                     niter=n,
                                                     mapsize_clean=mapsize_clean,
                                                     path_to_script=path_to_script)
            cv_means[frac].append(np.mean(cv_scores))
            train_means[frac].append(np.mean(train_scores))

    # CV-score for full data
    cv_means[1.0] = list()
    train_means[1.0] = list()
    for i in range(n_splits):
        kfold = KFoldCV(uv_fits_path, K, seed=np.random.randint(0, 1000))
        kfold.create_train_test_data(outdir=data_dir)
        cv_scores, train_scores = kfold.cv_score(initial_dfm_model_path=initial_dfm_model_path,
                                                 data_dir=data_dir,
                                                 niter=n_iter[-1],
                                                 mapsize_clean=mapsize_clean,
                                                 path_to_script=path_to_script)
        cv_means[1.0].append(np.mean(cv_scores))
        train_means[1.0].append(np.mean(train_scores))

    if plot:
        fig, axes = plt.subplots()
        axes.errorbar(sorted(cv_means.keys()),
                      y=[np.mean(cv_means[frac]) for frac in sorted(cv_means.keys())],
                      yerr=[np.std(cv_means[frac]) for frac in sorted(cv_means.keys())],
                      label='CV', ls=ls_cv)
        axes.errorbar(sorted(train_means.keys()),
                      y=[np.mean(train_means[frac]) for frac in sorted(train_means.keys())],
                      yerr=[np.std(train_means[frac]) for frac in sorted(train_means.keys())],
                      label='Train', ls=ls_train)
        axes.legend()
        axes.set_xlabel("Frac. of training data")
        axes.set_ylabel("RMSE")
        fig.show()

    return cv_means, train_means


if __name__ == '__main__':
    from mojave import download_mojave_uv_fits, get_mojave_mdl_file
    from spydiff import modelfit_difmap
    # source = '0059+581'
    # epoch = '2005_03_05'
    # epoch_ = '2005-03-05'
    # source = '0016+731'
    # epoch = '2005_09_05'
    # epoch_ = '2005-09-05'
    # source = '0106+013'
    # epoch = '2007_06_03'
    # epoch_ = '2007-06-03'
    # source = '0109+224'
    # epoch = '2007_07_03'
    # epoch_ = '2007-07-03'
    # source = '0224+671'
    # epoch = '2007_08_09'
    # epoch_ = '2007-08-09'
    # source = '0430+052'
    # epoch = '2007_08_16'
    # epoch_ = '2007-08-16'
    # source = '0552+398'
    # epoch = '2006_07_07'
    # epoch_ = '2006-07-07'
    # source = '0716+714'
    # epoch = '2007_09_06'
    # epoch_ = '2007-09-06'
    # source = '1807+698'
    # epoch = '2007_07_03'
    # epoch_ = '2007-07-03'
    source = '0336-019'
    epoch = '2010_10_25'
    epoch_ = '2010-10-25'
    data_dir = '/home/ilya/github/vlbi_errors/examples/LC'
    data_dir = os.path.join(data_dir, source, epoch)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # download_mojave_uv_fits(source, [epoch], download_dir=data_dir)
    path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'

    uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
    uv_fits_path = os.path.join(data_dir, uv_fits_fname)
    # get_mojave_mdl_file('/home/ilya/Dropbox/papers/boot/new_pics/mojave_mod_first/asu.tsv',
    #                     source, epoch_, outfile='initial.mdl', outdir=data_dir)
    uvdata = UVData(uv_fits_path)
    # modelfit_difmap(uv_fits_fname, 'initial.mdl',
    #                 'initial.mdl', niter=300,
    #                 path=data_dir, mdl_path=data_dir,
    #                 out_path=data_dir)
    original_model_path = os.path.join(data_dir, 'initial.mdl')
    from spydiff import import_difmap_model, clean_difmap
    comps = import_difmap_model(original_model_path)
    from automodel import plot_clean_image_and_components
    path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'

    # clean_difmap(uv_fits_path, os.path.join(data_dir, 'cc.fits'), 'I',
    #              (1024, 0.1), path=data_dir, path_to_script=path_to_script,
    #              outpath=data_dir)
    from from_fits import create_clean_image_from_fits_file
    ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                             'cc.fits'))
    plot_clean_image_and_components(ccimage, comps,
                                    outname=os.path.join(data_dir, "model_image.png"))
    # # LC for CLEAN model
    # cv_means_cc, train_means_cc =\
    #     learning_curve(uv_fits_path, (0.25, 0.5, 0.75), K=5,
    #                    mapsize_clean=(1024, 0.2), path_to_script=path_to_script,
    #                    n_splits=20, data_dir=data_dir, ls_cv='-', ls_train='--')
    # LC for direct model
    cv_means_m, train_means_m = \
        learning_curve(uv_fits_path, (0.125, 0.25, 0.5, 0.75), K=5,
                       initial_dfm_model_path=original_model_path,
                       n_iter=[1000, 500, 250, 100, 100], n_splits=1,
                       data_dir=data_dir, ls_cv='-.', ls_train=':')

    fig, axes = plt.subplots()
    # axes.errorbar(sorted(cv_means_cc.keys()),
    #               y=[np.mean(cv_means_cc[frac]) for frac in sorted(cv_means_cc.keys())],
    #               yerr=[np.std(cv_means_cc[frac]) for frac in sorted(cv_means_cc.keys())],
    #               label='CV CLEAN', ls='--')
    # axes.errorbar(sorted(train_means_cc.keys()),
    #               y=[np.mean(train_means_cc[frac]) for frac in sorted(train_means_cc.keys())],
    #               yerr=[np.std(train_means_cc[frac]) for frac in sorted(train_means_cc.keys())],
    #               label='Train CLEAN', ls='-')
    axes.errorbar(sorted(cv_means_m.keys()),
                  y=[np.mean(cv_means_m[frac]) for frac in sorted(cv_means_m.keys())],
                  yerr=[np.std(cv_means_m[frac]) for frac in sorted(cv_means_m.keys())],
                  label='CV model', ls=':')
    axes.errorbar(sorted(train_means_m.keys()),
                  y=[np.mean(train_means_m[frac]) for frac in sorted(train_means_m.keys())],
                  yerr=[np.std(train_means_m[frac]) for frac in sorted(train_means_m.keys())],
                  label='Train model', ls='-.')
    axes.legend(loc='upper right')
    axes.set_xlabel("Frac. of training data")
    axes.set_ylabel("RMSE")
