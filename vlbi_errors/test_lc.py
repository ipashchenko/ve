import os
import numpy as np
from cv_model import KFoldCV
from mojave import mojave_uv_fits_fname
from uv_data import UVData



source = '1514-241'
epoch = '2006_04_28'
data_dir = '/home/ilya/Dropbox/papers/boot/bias/new'
path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'

uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
uv_fits_path = os.path.join(data_dir, uv_fits_fname)
original_model_path = os.path.join(data_dir, 'initial.mdl')
uvdata = UVData(uv_fits_path)



# data_dir = '/home/ilya/silke'
# epoch = '2017_01_28'
# original_model_fname = '2017_01_28us'
# original_model_path = os.path.join(data_dir, original_model_fname)
# uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
# uv_fits_path = os.path.join(data_dir, uv_fits_fname)
# uvdata = UVData(uv_fits_path)

cv_means = dict()
train_means = dict()
for frac in (0.25, 0.5, 0.75):
    cv_means[frac] = list()
    train_means[frac] = list()
    for i in range(10):
        uv_frac_path = os.path.join(data_dir, 'frac_{}.fits'.format(frac))
        uvdata.save_fraction(uv_frac_path, frac, random_state=np.random.randint(0, 1000))
        kfold = KFoldCV(uv_frac_path, 5, seed=np.random.randint(0, 1000))
        kfold.create_train_test_data(outdir=data_dir)
        cv_scores, train_scores = kfold.cv_score(initial_dfm_model_path=None,
                                                 data_dir=data_dir, niter=50,
                                                 mapsize_clean=(512, 0.1),
                                                 path_to_script=path_to_script)
        cv_means[frac].append(np.mean(cv_scores))
        train_means[frac].append(np.mean(train_scores))

# CV-score for full data
cv_means[1.0] = list()
train_means[1.0] = list()
for i in range(10):
    kfold = KFoldCV(uv_fits_path, 5, seed=np.random.randint(0, 1000))
    kfold.create_train_test_data(outdir=data_dir)
    cv_scores, train_scores = kfold.cv_score(initial_dfm_model_path=None,
                                             data_dir=data_dir,
                                             niter=50, mapsize_clean=(1024, 0.1),
                                             path_to_script=path_to_script)
    cv_means[1.0].append(np.mean(cv_scores))
    train_means[1.0].append(np.mean(train_scores))





import matplotlib.pyplot as plt
plt.figure()
plt.errorbar(sorted(cv_means.keys()),
             y=[np.mean(cv_means[frac]) for frac in sorted(cv_means.keys())],
             yerr=[np.std(cv_means[frac]) for frac in sorted(cv_means.keys())],
             label='CV')
plt.errorbar(sorted(train_means.keys()),
             y=[np.mean(train_means[frac]) for frac in sorted(train_means.keys())],
             yerr=[np.std(train_means[frac]) for frac in sorted(train_means.keys())],
             label='Train')
# plt.errorbar((0.15, 0.25, 0.5, 0.75, 0.85, 0.95), train_means, train_stds, lw=2, label='train')
plt.legend()
plt.xlabel("Frac. of training data")
plt.ylabel("RMSE")
plt.show()
