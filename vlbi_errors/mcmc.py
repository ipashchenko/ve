# Test LS_estimates
import os
import sys
import numpy as np
from components import CGComponent, EGComponent
from uv_data import UVData
from model import Model
from spydiff import clean_difmap
from from_fits import create_clean_image_from_fits_file
try:
    from scipy.optimize import minimize, fmin
except ImportError:
    sys.exit("install scipy for ml estimation")
import scipy as sp
from stats import LnLikelihood, LnPost
from image import find_bbox
from image import plot as iplot
from image_ops import rms_image
import emcee
import corner
import matplotlib.pyplot as plt


data_dir = '/home/ilya/Dropbox/Ilya/0235_bk150_uvf_data/'
# q - 43GHz, k - 23.8GHz, u - 15GHz, ...
uv_fname = '0235+164.q1.2008_09_02.uvf_difmap'

# Clean uv-data
clean_difmap(uv_fname, 'cc.fits', 'I', (1024, 0.03), path=data_dir,
             path_to_script='/home/ilya/code/vlbi_errors/difmap/final_clean_nw',
             outpath=data_dir, show_difmap_output=True)

image = create_clean_image_from_fits_file(os.path.join(data_dir, 'cc.fits'))
rms = rms_image(image)
blc, trc = find_bbox(image.image, 2.*rms, delta=int(image._beam.beam[0]))
# Plot image
iplot(image.image, x=image.x, y=image.y, min_abs_level=3. * rms,
      outfile='clean_image', outdir=data_dir, blc=blc, trc=trc, beam=image.beam,
      show_beam=True)


uvdata = UVData(os.path.join(data_dir, uv_fname))
# # Create model
# cg1 = CGComponent(3.0, 0.0, 0.0, 0.5)
# cg2 = CGComponent(1.0, 0.0, 0.0, 0.5)
# mdl = Model(stokes='I')
# mdl.add_components(cg1, cg2)
# # Create log of likelihood function
# lnlik = LnLikelihood(uvdata, mdl, average_freq=True, amp_only=False)
# # Nelder-Mead simplex algorithm
# # p_ml = fmin(lambda p: -lnlik(p), mdl.p)
# # Various methods of minimization (some require jacobians)
# # TODO: Implement analitical grad of likelihood (it's gaussian)
# # fit = minimize(lambda p: -lnlik(p), mdl.p, method='L-BFGS-B',
# #                options={'maxiter': 30000, 'maxfev': 1000000, 'xtol': 0.001,
# #                         'ftol': 0.001, 'approx_grad': True},
# #                bounds=[(0., 2), (None, None), (None, None), (0., +np.inf),
# #                        (0., 1.), (None, None),
# #                        (0., 2), (None, None), (None, None), (0., 5),
# #                        (0., 1), (None, None), (None, None), (0., 20)])
# fit = minimize(lambda p: -lnlik(p), mdl.p, method='L-BFGS-B',
#                options={'maxiter': 30000, 'maxfev': 1000000, 'xtol': 0.00001,
#                         'ftol': 0.00001, 'approx_grad': True},
#                bounds=[(0., 5), (None, None), (None, None), (0., +np.inf),
#                        (0., 5), (None, None), (None, None), (0., +np.inf)])
# if fit['success']:
#     print "Succesful fit!"
#     p_ml = fit['x']
#     print p_ml


# Create several components
eg1 = EGComponent(4.0, 0.0, 0.0, 0.1, 0.4, 1.5)
eg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 10.], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),),
              e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
              bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
# Create model
mdl1 = Model(stokes='I')
# Add components to model
mdl1.add_component(eg1)
# Create posterior for data & model
lnpost = LnPost(uvdata, mdl1)
ndim = mdl1.size
nwalkers = 50
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
p_std1 = [0.1, 0.01, 0.01, 0.01, 0.01, 0.01]
p0 = emcee.utils.sample_ball(mdl1.p, p_std1, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, lnp, _ = sampler.run_mcmc(pos, 300)

# Plot corner
fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
fig.set_size_inches(13.5, 13.5)
corner.corner(sampler.flatchain[::10, :], fig=fig,
              labels=[r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$e$', r'$bpa$'],
              show_titles=True, title_kwargs={'fontsize': 16},
              quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize': 16},
              title_fmt=".3f")
fig.savefig('corner_1component.png', bbox_inches='tight', dpi=200)

# Find mode of parameter distributions
p_map = pos[np.argmax(lnp)]

# Now fitting two components
eg1 = EGComponent(*p_map)
cg2 = CGComponent(0.2, 0.0, -0.2, 0.2)
eg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 10.], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),),
              e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
              bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 1.5], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0.01, 20.], dict(),))
# Create model
mdl2 = Model(stokes='I')
# Add components to model
mdl2.add_component(eg1)
mdl2.add_component(cg2)
# Create posterior for data & model
lnpost = LnPost(uvdata, mdl2)
ndim = mdl2.size
nwalkers = 100
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
p_std1 = [0.1, 0.01, 0.01, 0.005, 0.01, 0.005]
p_std2 = [0.01, 0.01, 0.01, 0.01]
p0 = emcee.utils.sample_ball(mdl2.p, p_std1 + p_std2, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 100)
####
sampler.reset()
pos, lnp, _ = sampler.run_mcmc(pos, 300)

# Plot corner
fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
fig.set_size_inches(13.5, 13.5)
corner.corner(sampler.flatchain[::10, :], fig=fig,
              labels=[r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$e$', r'$bpa$',
                      r'$flux$', r'$x$', r'$y$', r'$bmaj$'],
              show_titles=True, title_kwargs={'fontsize': 14},
              quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize': 14},
              title_fmt=".3f")
fig.savefig('corner_2components.png', bbox_inches='tight', dpi=200)

p_ = [4.18100886e+00, -3.91663159e-03, 4.15828600e-02, 1.16368974e-01,
      3.91873876e-01,  1.49900603e+00, ]
mdl = Model(stokes='I')
eg1 = EGComponent(*(p_map[:6]))
cg2 = CGComponent(*(p_map[6:]))
mdl.add_components(eg1, cg2)
fig = uvdata.uvplot(stokes='I')
mdl.uvplot(uv=uvdata.uv, fig=fig)
fig.savefig('2component_mdl_vs_data.png', bbox_inches='tight', dpi=200)

# Adding third component
mdl = Model(stokes='I')
eg1 = EGComponent(*(p_map[:6]))
cg2 = CGComponent(*(p_map[6:]))
cg3 = CGComponent(0.05, 0.29, 0.75, 0.2)
eg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 10.], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),),
              e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
              bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 1.5], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0.03, 10.], dict(),))
cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 1.0], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0.03, 10.], dict(),))
mdl.add_components(eg1, cg2, cg3)
# Create posterior for data & model
lnpost = LnPost(uvdata, mdl)
ndim = mdl.size
nwalkers = 100
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
p_std1 = [0.1, 0.01, 0.01, 0.01, 0.01, 0.01]
p_std2 = [0.03, 0.03, 0.03, 0.03]
p_std3 = [0.03, 0.03, 0.03, 0.03]
p0 = emcee.utils.sample_ball(mdl.p, p_std1 + p_std2 + p_std3, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, lnp, _ = sampler.run_mcmc(pos, 300)

# Plot corner
fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
fig.set_size_inches(13.5, 13.5)
corner.corner(sampler.flatchain[::10, :], fig=fig,
              labels=[r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$e$', r'$bpa$',
                      r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$flux$', r'$x$',
                      r'$y$', r'$bmaj$'],
              show_titles=True, title_kwargs={'fontsize': 10},
              quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize': 10},
              title_fmt=".3f", max_n_ticks=3)
fig.savefig('corner_3components.png', bbox_inches='tight', dpi=200)

# Overplot data and model
p_map = pos[np.argmax(lnp)]
# In [128]: p_map
# Out[128]:
# array([  4.01536730e+00,  -4.24002619e-03,   3.67696324e-02,
#          9.22227342e-02,   4.81077919e-01,   1.50883915e+00,
#          1.73122933e-01,   1.25605087e-02,   2.19525868e-01,
#          3.84424150e-02,   2.00000000e-03,   2.60000000e-01,
#          7.00000000e-01,   1.00000000e-01])
# Adding third component
mdl = Model(stokes='I')
eg1 = EGComponent(*(p_map[:6]))
cg2 = CGComponent(*(p_map[6:10]))
cg3 = CGComponent(*(p_map[10:]))
eg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 10.], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0, 2.], dict(),),
              e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
              bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 1.5], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0.01, 10.], dict(),))
cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0.01, 10.], dict(),))
mdl.add_components(eg1, cg2, cg3)
# Create posterior for data & model
lnpost = LnPost(uvdata, mdl)
ndim = mdl.size
nwalkers = 100
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
p_std1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
p_std2 = [0.01, 0.01, 0.01, 0.01]
p_std3 = [0.0001, 0.01, 0.01, 0.01]
p0 = emcee.utils.sample_ball(mdl.p, p_std1 + p_std2 + p_std3, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, lnp, _ = sampler.run_mcmc(pos, 300)

# p_map = list()
# for i in range(ndim):
#     counts, bin_values = np.histogram(sampler.flatchain[::10, i], bins=100)
#     p_map.append(bin_values[counts == counts.max()][0])
mdl = Model(stokes='I')
eg1 = EGComponent(*(p_map[:6]))
cg2 = CGComponent(*(p_map[6:10]))
cg3 = CGComponent(*(p_map[10:]))
mdl.add_components(eg1, cg2, cg3)
fig = uvdata.uvplot(stokes='I')
mdl.uvplot(uv=uvdata.uv, fig=fig)
fig.savefig('3component_mdl_vs_data.png', bbox_inches='tight', dpi=200)
