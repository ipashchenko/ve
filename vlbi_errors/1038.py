import emcee
import triangle
import scipy as sp
import numpy as np
from from_fits import create_uvdata_from_fits_file
from components import CGComponent
from model import Model, CCModel
from stats import LnPost


if __name__ == '__main__':
    uv_fname = '1633+382.l22.2010_05_21.uvf'
    map_fname = '1633+382.l22.2010_05_21.icn.fits'
    uvdata = create_uvdata_from_fits_file(uv_fname)
    # Create several components
    cg1 = CGComponent(1.0, 0.0, 0.0, 1.)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 10.], dict(),))
    # Create model
    mdl1 = Model(stokes='I')
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
    sampler.run_mcmc(pos, 300)
    p_map = np.max(sampler.flatchain[::10, :], axis=0)

    # Overplot data and model
    mdl = Model(stokes='I')
    # cg = CGComponent(1.441, 0.76, 0.65, 3.725)
    cg = CGComponent(*p_map)
    mdl.add_component(cg)
    uvdata.uvplot(stokes='I')
    mdl.uvplot(uv=uvdata.uv)

    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])

    # Now fitting two components
    cg1 = CGComponent(*p_map)
    cg2 = CGComponent(0.5, 0.0, 0.0, 2.0)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 10.], dict(),))
    cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 1.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),))
    # Create model
    mdl2 = Model(stokes='I')
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
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 300)

    # Overplot data and model
    p_map = np.max(sampler.flatchain[::10, :], axis=0)
    mdl = Model(stokes='I')
    cg1 = CGComponent(1.4, -0.4, 0.1, 1.25)
    cg2 = CGComponent(0.33, 3.3, -0.3, 1.55)
    mdl.add_components(cg1, cg2)
    uvdata.uvplot(stokes='I')
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

    # Add third component
    cg1 = CGComponent(1.4, -0.4, 0.1, 1.25)
    cg2 = CGComponent(0.33, 3.3, -0.3, 1.55)
    cg3 = CGComponent(0.2, -10.0, 0.0, 2.0)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    # Create model
    mdl3 = Model(stokes='I')
    # Add components to model
    mdl3.add_components(cg1, cg2, cg3)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl3)
    ndim = mdl3.size
    nwalkers = 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    p_std1 = [0.1, 0.3, 0.3, 0.1]
    p_std2 = [0.1, 1., 1., 0.1]
    p_std3 = [0.1, 1., 1., 0.1]
    p0 = emcee.utils.sample_ball(mdl3.p, p_std1 + p_std2 + p_std3,
                                 size=nwalkers)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 300)

    # Check results
    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])
    mdl = Model(stokes='I')
    cg1 = CGComponent(1.23, -0.51, 0.21, 0.88)
    cg2 = CGComponent(0.42, 2.9, -0.6, 1.55)
    cg3 = CGComponent(0.2, -8.0, -2.8, 1.98)
    mdl.add_components(cg1, cg2, cg3)
    uvdata.uvplot(stokes='I')
    mdl.uvplot(uv=uvdata.uv)

    mdl_image = mdl.make_image(map_fname)
    mdl_image.plot(min_rel_level=1.)
    # Plot cc-map
    ccmodel = CCModel()
    ccmodel.add_cc_from_fits(map_fname)
    ccimage = ccmodel.make_image(map_fname)
    ccimage.plot(min_rel_level=0.025)

    # Add forth component
    cg1 = CGComponent(1.23, -0.51, 0.21, 0.88)
    cg2 = CGComponent(0.42, 2.9, -0.6, 1.55)
    cg3 = CGComponent(0.2, -8.0, -2.8, 1.98)
    cg4 = CGComponent(0.2, -15.0, 0.0, 2.0)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    cg4.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 9.], dict(),))
    # Create model
    mdl4 = Model(stokes='I')
    # Add components to model
    mdl4.add_components(cg1, cg2, cg3, cg4)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl4)
    ndim = mdl4.size
    nwalkers = 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    p_std1 = [0.1, 0.3, 0.3, 0.1]
    p_std2 = [0.1, 1., 1., 0.1]
    p_std3 = [0.1, 1., 1., 0.1]
    p_std4 = [0.1, 1., 1., 0.1]
    p0 = emcee.utils.sample_ball(mdl4.p, p_std1 + p_std2 + p_std3 + p_std4,
                                 size=nwalkers)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 300)

    # Check results
    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])
    mdl = Model(stokes='I')
    cg1 = CGComponent(1.16, -0.61, 0.26, 0.72)
    cg2 = CGComponent(0.48, 2.9, -0.6, 1.25)
    cg3 = CGComponent(0.03, -7.4, -1.2, 1.95)
    cg4 = CGComponent(0.005, -7.5, -0, 2.15)
    mdl.add_components(cg1, cg2, cg3, cg4)
    uvdata.uvplot(stokes='I')
    mdl.uvplot(uv=uvdata.uv)

    mdl_image = mdl.make_image(map_fname)
    mdl_image.plot(min_rel_level=0.001)
    # Plot cc-map
    ccmodel = CCModel()
    ccmodel.add_cc_from_fits(map_fname)
    ccimage = ccmodel.make_image(map_fname)
    ccimage.plot(min_rel_level=0.025)

    # Add firth component
    cg1 = CGComponent(1.0, -0.51, 0.21, 0.88)
    cg2 = CGComponent(0.75, 2.9, -0.6, 1.55)
    cg3 = CGComponent(0.2, -8.0, -2.8, 1.98)
    cg4 = CGComponent(0.05, -20.0, 0.0, 2.0)
    cg5 = CGComponent(0.01, -75.0, 100.0, 25.0)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.5], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    cg4.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.2], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 15.], dict(),))
    cg5.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.2], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 90.], dict(),))
    # Create model
    mdl5 = Model(stokes='I')
    # Add components to model
    mdl5.add_components(cg1, cg2, cg3, cg4, cg5)
    mdl_image = mdl5.make_image(map_fname)
    mdl_image.plot(min_rel_level=0.001)
    # Plot cc-map
    ccmodel = CCModel()
    ccmodel.add_cc_from_fits(map_fname)
    ccimage = ccmodel.make_image(map_fname)
    ccimage.plot(min_rel_level=0.025)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl5)
    ndim = mdl5.size
    nwalkers = 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    p_std1 = [0.1, 0.3, 0.3, 0.1]
    p_std2 = [0.05, 1., 1., 0.1]
    p_std3 = [0.01, 1., 1., 0.1]
    p_std4 = [0.01, 1., 1., 1]
    p_std5 = [0.01, 1., 1., 1]
    p0 = emcee.utils.sample_ball(mdl5.p, p_std1 + p_std2 + p_std3 + p_std4 +
                                 p_std5, size=nwalkers)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 300)

    # Check results
    fig = triangle.corner(sampler.flatchain[::10, :4],
                          labels=["$flux$", "$y$", "$x$", "$maj$"])
    mdl = Model(stokes='I')
    cg1 = CGComponent(1.16, -0.61, 0.26, 0.72)
    cg2 = CGComponent(0.48, 2.9, -0.6, 1.25)
    cg3 = CGComponent(0.03, -7.4, -1.2, 1.95)
    cg4 = CGComponent(0.005, -7.5, -0, 2.15)
    cg5 = CGComponent(0.005, -7.5, -0, 2.15)
    mdl.add_components(cg1, cg2, cg3, cg4, cg5)
    uvdata.uvplot(stokes='I')
    mdl.uvplot(uv=uvdata.uv)

    mdl_image = mdl.make_image(map_fname)
    mdl_image.plot(min_rel_level=0.001)
    # Plot cc-map
    ccmodel = CCModel()
    ccmodel.add_cc_from_fits(map_fname)
    ccimage = ccmodel.make_image(map_fname)
    ccimage.plot(min_rel_level=0.025)
