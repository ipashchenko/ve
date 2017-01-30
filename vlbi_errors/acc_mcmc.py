import os
from uv_data import UVData
from model import Model
from stats import LnPost
from components import EGComponent
import emcee
import corner
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np



data_dir = '/home/ilya/Dropbox/ACC/BK/simulations/0952'
uv_fits = '0952_15GHz_BK.fits'
eg = EGComponent(10., 0., 0., 0.3, 0.5, 1.)
eg.add_prior(flux=(sp.stats.uniform.logpdf, [0., 30.], dict(),),
             bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),),
             e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
             bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
model = Model(stokes='I')
model.add_component(eg)

uvdata = UVData(os.path.join(data_dir, uv_fits))

lnpost = LnPost(uvdata, model, use_V=True, average_freq=True)
ndim = model.size
nwalkers = 200
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
p_std = [1., 0.1, 0.1, 0.03, 0.05, 0.2]
p0 = emcee.utils.sample_ball(model.p, p_std, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 100)
print "Acceptance fraction at burnin: {}".format(sampler.acceptance_fraction)
sampler.reset()
pos, lnp, _ = sampler.run_mcmc(pos, 500)
p_map = sampler.flatchain[np.argmax(sampler.lnprobability)]
sampler.reset()
p_std = [0.1, 0.01, 0.01, 0.01, 0.01, 0.01]
p0 = emcee.utils.sample_ball(p_map, p_std, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 500)
# Plot corner
fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
fig.set_size_inches(19.5, 19.5)
corner.corner(sampler.flatchain[::10, :], fig=fig,
              labels=[r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$e$', r'$bpa$'],
              show_titles=True, title_kwargs={'fontsize': 16},
              quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize': 16},
              title_fmt=".3f")
fig.savefig(os.path.join(data_dir, '0235_BK_15GHz_corner.png'),
            bbox_inches='tight', dpi=200)
