import emcee
import triangle
import scipy as sp
import numpy as np
from from_fits import create_uvdata_from_fits_file
from components import CGComponent
from model import Model, CCModel
from stats import LnPost


if __name__ == '__main__':
    uv_fname = 'J0005+3820_X_1998_06_24_fey_vis.fits'
    map_fname = 'J0005+3820_X_1998_06_24_fey_map.fits'
    uvdata = create_uvdata_from_fits_file(uv_fname)
    # Create several components
    cg1 = CGComponent(1.0, 0.0, 0.0, 1.)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 10.], dict(),))
    # Create model
    mdl1 = Model(stokes='RR')
    # Add components to model
    mdl1.add_component(cg1)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl1)
    ndim = mdl1.size
    nwalkers = 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    p_std1 = [0.1, 1., 1., 0.1]
    p0 = emcee.utils.sample_ball(mdl1.p, p_std1, size=nwalkers)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    p_map = np.max(sampler.flatchain[::20, :], axis=0)

    # Overplot data and model
    mdl = Model(stokes='RR')
    # cg = CGComponent(1.441, 0.76, 0.65, 3.725)
    cg = CGComponent(*p_map)
    mdl.add_component(cg)
    uvdata.uvplot(stokes='RR')
    mdl.uvplot(uv=uvdata.uv)

    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])

    # Now fitting two components
    cg1 = CGComponent(*p_map)
    cg2 = CGComponent(0.1, 0.0, 0.0, 5.0)
    # TODO: use Jeffrey's prior p(\sigma) ~ 1/\sigma
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 10.], dict(),))
    cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),))
    # Create model
    mdl2 = Model(stokes='RR')
    # Add components to model
    mdl2.add_component(cg1)
    mdl2.add_component(cg2)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl2)
    ndim = mdl2.size
    nwalkers = 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    p_std1 = [0.1, 0.3, 0.3, 0.1]
    p_std2 = [0.1, 1., 1., 0.1]
    p0 = emcee.utils.sample_ball(mdl2.p, p_std1 + p_std2, size=nwalkers)
    pos, prob, state = sampler.run_mcmc(p0, 300)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    # Overplot data and model
    mdl = Model(stokes='RR')
    cg1 = CGComponent(0.415, 0.41, -0.05, 1.940)
    cg2 = CGComponent(0.130, -8.35, 3.40, 2.350)
    mdl.add_component(cg1)
    mdl.add_component(cg2)
    uvdata.uvplot(stokes='RR')
    mdl.uvplot(uv=uvdata.uv)
    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])

    mdl_image = mdl.make_image(map_fname)
    mdl_image.plot(min_rel_level=1.)
    # Plot cc-map
    ccmodel = CCModel()
    ccmodel.add_cc_from_fits(map_fname)
    ccimage = ccmodel.make_image(map_fname)
    ccimage.plot(min_rel_level=0.25)
