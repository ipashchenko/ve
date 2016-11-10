import os
import numpy as np
from uv_data import UVData
from model import Model
from components import CGComponent, EGComponent, DeltaComponent
from spydiff import import_difmap_model
import scipy as sp
from stats import LnLikelihood, LnPrior
import emcee
import corner
import matplotlib.pyplot as plt


def fit_model_with_ptmcmc(uv_fits, mdl_file, outdir=None, nburnin=1000,
                          nproduction=10000, nwalkers=50,
                          samples_file=None, stokes='I', use_weights=False,
                          ntemps=10, thin=10):

    # Initialize ``UVData`` instance
    uvdata = UVData(uv_fits)

    # Load difmap model
    mdl_dir, mdl_fname = os.path.split(mdl_file)
    comps = import_difmap_model(mdl_fname, mdl_dir)
    # Sort components by distance from phase center
    comps = sorted(comps, key=lambda x: np.sqrt(x.p[1]**2 + x.p[2]**2))

    # Cycle for components, add prior and calculate std for initial position of
    # walkers: 3% of flux for flux, 1% of size for position, 3% of size for
    # size, 0.01 for e, 0.01 for bpa
    p0_dict = dict()
    for comp in comps:
        print comp
        if isinstance(comp, EGComponent):
            # flux_high = 2 * comp.p[0]
            flux_high = 10. * comp.p[0]
            # bmaj_high = 4 * comp.p[3]
            bmaj_high = 4.
            if comp.size == 6:
                comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),),
                               bmaj=(sp.stats.uniform.logpdf, [0, bmaj_high], dict(),),
                               e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
                               bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
                p0_dict[comp] = [0.03 * comp.p[0],
                                 0.01 * comp.p[3],
                                 0.01 * comp.p[3],
                                 0.03 * comp.p[3],
                                 0.01,
                                 0.01]
            elif comp.size == 4:
                # flux_high = 2 * comp.p[0]
                flux_high = 10. * comp.p[0]
                # bmaj_high = 4 * comp.p[3]
                bmaj_high = 4.
                comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),),
                               bmaj=(sp.stats.uniform.logpdf, [0, bmaj_high], dict(),))
                p0_dict[comp] = [0.03 * comp.p[0],
                                 0.01 * comp.p[3],
                                 0.01 * comp.p[3],
                                 0.03 * comp.p[3]]
            else:
                raise Exception("Gauss component should have size 4 or 6!")
        elif isinstance(comp, DeltaComponent):
            flux_high = 5 * comp.p[0]
            comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),))
            p0_dict[comp] = [0.03 * comp.p[0],
                             0.01,
                             0.01]
        else:
            raise Exception("Unknown type of component!")

    # Construct labels for corner and truth values (of difmap models)
    labels = list()
    truths = list()
    for comp in comps:
        truths.extend(comp.p)
        if isinstance(comp, EGComponent):
            if comp.size == 6:
                labels.extend([r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$e$', r'$bpa$'])
            elif comp.size == 4:
                labels.extend([r'$flux$', r'$x$', r'$y$', r'$bmaj$'])
            else:
                raise Exception("Gauss component should have size 4 or 6!")
        elif isinstance(comp, DeltaComponent):
            labels.extend([r'$flux$', r'$x$', r'$y$'])
        else:
            raise Exception("Unknown type of component!")

    # Create model
    mdl = Model(stokes=stokes)
    # Add components to model
    mdl.add_components(*comps)

    # Create likelihood for data & model
    lnlik = LnLikelihood(uvdata, mdl, use_weights=use_weights,)
    lnpr = LnPrior(mdl)
    ndim = mdl.size

    # Initialize pool of walkers
    p_std = list()
    for comp in comps:
        p_std.extend(p0_dict[comp])
    print "Initial std of parameters: {}".format(p_std)
    p0 = emcee.utils.sample_ball(mdl.p, p_std,
                                 size=ntemps*nwalkers).reshape((ntemps,
                                                                nwalkers, ndim))
    betas = np.exp(np.linspace(0, -(ntemps - 1) * 0.5 * np.log(2), ntemps))
    # Initialize sampler
    ptsampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlik, lnpr,
                                betas=betas)

    # Burning in
    print "Burnin"
    for p, lnprob, lnlike in ptsampler.sample(p0, iterations=nburnin):
        pass
    print "Acceptance fraction for initial burning: ", ptsampler.acceptance_fraction
    ptsampler.reset()

    print "Production"
    for p, lnprob, lnlike in ptsampler.sample(p, lnprob0=lnprob, lnlike0=lnlike,
                                              iterations=nproduction,
                                              thin=thin):
        pass
    print "Acceptance fraction for production: ", ptsampler.acceptance_fraction

    # Plot corner
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
    fig.set_size_inches(14.5, 14.5)

    # Choose fontsize
    if len(comps) <= 2:
        fontsize = 16
    elif 2 < len(comps) <= 4:
        fontsize = 13
    else:
        fontsize = 11

    # Use zero-temperature chain
    samples = ptsampler.flatchain[0, :, :]

    corner.corner(samples, fig=fig, labels=labels,
                  truths=truths, show_titles=True,
                  title_kwargs={'fontsize': fontsize},
                  quantiles=[0.16, 0.5, 0.84],
                  label_kwargs={'fontsize': fontsize}, title_fmt=".3f")
    fig.savefig(os.path.join(outdir, 'corner_mcmc_x.png'), bbox_inches='tight',
                dpi=200)
    if not samples_file:
        samples_file = 'mcmc_samples.txt'
    print "Saving thinned samples to {} file...".format(samples_file)
    np.savetxt(samples_file, samples)
    return lnpost, ptsampler

if __name__ == '__main__':
    uv_fits = '/home/ilya/code/vlbi_errors/pet/0235+164_X.uvf_difmap'
    mdl_file = '/home/ilya/code/vlbi_errors/pet/0235+164_X.mdl'
    lnpost, sampler = fit_model_with_ptmcmc(uv_fits, mdl_file, nwalkers=100,
                                            samples_file='samples_of_mcmc.txt',
                                            outdir='/home/ilya/code/vlbi_errors/pet',
                                            stokes='RR')
