import emcee
import triangle
import scipy as sp
from uv_data import UVData
from components import CGComponent
from model import Model
from stats import LnPost


if __name__ == '__main__':
    uvdata = UVData('J0006-0623_X_2008_07_09_pus_vis.fits')
    uv = uvdata.uvw[:, :2]
    # Create several components
    cg1 = CGComponent(2.0, 0.0, 0.0, 0.1)
    # cg2 = CGComponent(1.0, 0.1, -1.05, 1.18)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 2.], dict(),))
    # cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #                bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    # Create model
    mdl1 = Model(stokes='I')
    # Add components to model
    mdl1.add_component(cg1)
    # mdl1.add_component(cg2)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl1)
    # lnpr = LnPrior(mdl1)
    # lnlik = LnLikelihood(uvdata, mdl1)
    # model.uvplot(uv = uv)
    # model.ft(uv=uv)
    ndim = mdl1.size
    nwalkers = 50
    # p0 = mdl1.p
    # cov = np.zeros(ndim * ndim).reshape((ndim, ndim,))
    # cov[0, 0] = 0.1
    # cov[1, 1] = 0.1
    # cov[2, 2] = 0.1
    # cov[3, 3] = 0.1
    # sampler = emcee.MHSampler(cov, ndim, lnpost)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    p_std1 = [0.01, 0.01, 0.01, 0.01]
    # p_std2 = [0.003, 0.01, 0.01, 0.01]
    p0 = emcee.utils.sample_ball(mdl1.p, p_std1, size=nwalkers)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 700)
    uvdata.uvplot_model(mdl1, stokes='RR', style='a&p')
    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])
