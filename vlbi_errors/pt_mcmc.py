from uv_data import UVData
from spydiff import import_difmap_model
from stats import LnLikelihood, LnPrior
from model import Model
from emcee import PTSampler, utils
import scipy as sp
import numpy as np


mdl_dir = '/home/ilya/vlbi_errors/bin_q'
mdl_file = '0235+164.q1.2008_09_02.mdl'
uvfits = '/home/ilya/vlbi_errors/bin_q/0235+164.q1.2008_09_02.uvf_difmap'

uvdata = UVData(uvfits)
comps = import_difmap_model(mdl_file, mdl_dir)
eg = comps[0]
eg.add_prior(flux=(sp.stats.uniform.logpdf, [0., 10.], dict(),),
             bmaj=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
             x=(sp.stats.uniform.logpdf, [-1., 1.], dict(),),
             y=(sp.stats.uniform.logpdf, [-1., 1.], dict(),),
             e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
             bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
model = Model(stokes='I')
model.add_component(eg)
ndim = model.size
nwalkers = 32
ntemps = 20


lnlik = LnLikelihood(uvdata, model)
lnpr = LnPrior(model)

p0 = utils.sample_ball(model.p, [0.3, 0.1, 0.1, 0.003, 0.03, 0.1],
                       size=ntemps*nwalkers).reshape((ntemps, nwalkers, ndim))

betas = np.exp(np.linspace(0, -(ntemps - 1) * 0.5 * np.log(2), ntemps))
ptsampler = PTSampler(ntemps, nwalkers, ndim, lnlik, lnpr, betas=betas)


# Burning in
print "Burnin"
for p, lnprob, lnlike in ptsampler.sample(p0, iterations=1000):
    pass
ptsampler.reset()

print "Production"
for p, lnprob, lnlike in ptsampler.sample(p, lnprob0=lnprob,
                                          lnlike0=lnlike,
                                          iterations=10000, thin=10):
    pass

# # 0-temperature chain
# mu0 = np.mean(np.mean(ptsampler.chain[0,...], axis=0), axis=0)
# # array([ 3.49944349, -0.00425058,  0.02648386,  0.06396026,  0.58487231,
# #         1.57089506])
#
# # (11427.220089952611, 5.9308987859385525)
# import corner
# samples = ptsampler.flatchain[0, ::3, 3:5]
fig = corner.corner(samples, range=((0, 0.15), (0, 1)),
                    labels=[r'$bmaj$', r'$e$'], show_titles=True,
                    title_kwargs={'fontsize': 16}, quantiles=[0.16, 0.5, 0.84],
                    label_kwargs={'fontsize': 16}, title_fmt=".3f")
fig.savefig('_corner.png', bbox_inches='tight', dpi=200)
