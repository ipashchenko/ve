import os
import numpy as np
from mcmc_difmap_model import fit_model_with_mcmc
from uv_data import UVData
from spydiff import import_difmap_model, modelfit_difmap
from model import Model


data_dir = '/home/ilya/code/vlbi_errors/bin_c1/'
uv_fits = '0235+164.c1.2008_09_02.uvf_difmap'
mdl_file = '0235+164.c1.2008_09_02.mdl'

uvdata = UVData(os.path.join(data_dir, uv_fits))

original_comps = import_difmap_model(mdl_file, data_dir)
lnpost, sampler = fit_model_with_mcmc(os.path.join(data_dir, uv_fits),
                                      os.path.join(data_dir, mdl_file),
                                      samples_file='samples_of_mcmc.txt',
                                      outdir='/home/ilya/code/vlbi_errors/bin_c1/')
samples = sampler.flatchain[::10, :]

# Create a sample of models with parameters from posterior distribution
models = list()
for i, s in enumerate(samples[np.random.randint(len(samples), size=100)]):
    model = Model(stokes='I')
    j = 0
    for orig_comp in original_comps:
        comp = orig_comp.__class__(*(s[j: j + orig_comp.size]))
        model.add_component(comp)
        j += orig_comp.size
    models.append(model)

cv_scores = list()
for model in models:
    cv_scores.append(uvdata.cv_score(model, baselines=[774, 1546]))

np.savetxt(os.path.join(data_dir, 'cv_scores_eg.txt'), np.array(cv_scores))

# # Now check delta
# modelfit_difmap(uv_fits, '0235+164.c1.2008_09_02_cgauss.mdl',
#                 '0235+164.c1.2008_09_02_cgauss_fitted_fitted.mdl', niter=100,
#                 path=data_dir, mdl_path=data_dir, out_path=data_dir)
