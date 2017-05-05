import os
import numpy as np
from uv_data import UVData
from model import Model
from components import CGComponent, EGComponent, DeltaComponent
from spydiff import import_difmap_model
import scipy as sp
from stats import LnPost
import emcee
import corner
import matplotlib.pyplot as plt


def fit_model_with_mcmc(uv_fits, mdl_file, outdir=None, nburnin_1=100,
                        nburnin_2=300, nproduction=500, nwalkers=50,
                        samples_file=None, stokes='I', use_weights=False):

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
            flux_high = 2 * comp.p[0]
            try:
                bmaj_high = 4 * comp.p[3]
            except IndexError:
                pass
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
                flux_high = 2 * comp.p[0]
                bmaj_high = 4 * comp.p[3]
                comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),),
                               bmaj=(sp.stats.uniform.logpdf, [0, bmaj_high], dict(),))
                p0_dict[comp] = [0.03 * comp.p[0],
                                 0.01 * comp.p[3],
                                 0.01 * comp.p[3],
                                 0.03 * comp.p[3]]
            elif comp.size == 3:
                flux_high = 2 * comp.p[0]
                comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),))
                p0_dict[comp] = [0.03 * comp.p[0],
                                 0.01,
                                 0.01]
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
            elif comp.size == 3:
                labels.extend([r'$flux$', r'$x$', r'$y$'])
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
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl, use_weights=use_weights)
    ndim = mdl.size

    # Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)

    # Initialize pool of walkers
    p_std = list()
    for comp in comps:
        p_std.extend(p0_dict[comp])
    print "Initial std of parameters: {}".format(p_std)
    p0 = emcee.utils.sample_ball(mdl.p, p_std, size=nwalkers)
    print p0[0]

    # Run initial burnin
    pos, prob, state = sampler.run_mcmc(p0, nburnin_1)
    print "Acceptance fraction for initial burning: ", sampler.acceptance_fraction
    sampler.reset()
    # Run second burning
    pos, lnp, _ = sampler.run_mcmc(pos, nburnin_2)
    print "Acceptance fraction for second burning: ", sampler.acceptance_fraction
    sampler.reset()
    pos, lnp, _ = sampler.run_mcmc(pos, nproduction)
    print "Acceptance fraction for production: ", sampler.acceptance_fraction

    # # Plot corner
    # fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
    # fig.set_size_inches(14.5, 14.5)

    # # Choose fontsize
    # if len(comps) <= 2:
    #     fontsize = 16
    # elif 2 < len(comps) <= 4:
    #     fontsize = 13
    # else:
    #     fontsize = 11

    # if plot_corner:
    #     corner.corner(sampler.flatchain[::10, :], fig=fig, labels=labels,
    #                   truths=truths, show_titles=True,
    #                   title_kwargs={'fontsize': fontsize},
    #                   quantiles=[0.16, 0.5, 0.84],
    #                   label_kwargs={'fontsize': fontsize}, title_fmt=".3f")
    #     fig.savefig(os.path.join(outdir, 'corner.png'), bbox_inches='tight',
    #                 dpi=200)
    if not samples_file:
        samples_file = 'mcmc_samples.txt'
    print "Saving thinned samples to {} file...".format(samples_file)
    np.savetxt(os.path.join(outdir, samples_file), sampler.flatchain[::, :])
    return sampler, labels, truths


def plot_comps(components_to_plot, samples, original_dfm_mdl,
               title_fontsize=12, label_fontsize=12,
               outdir=None, outfname=None, fig=None, limits=None):
    if outfname is None:
        outfname = str(components_to_plot) + '_corner.pdf'
    if outdir is None:
        outdir = os.getcwd()

    n_samples, n_dim = np.shape(samples)
    indxs = np.zeros(n_dim)
    components_to_plot = sorted(components_to_plot)

    # Load difmap model
    mdl_dir, mdl_fname = os.path.split(original_dfm_mdl)
    comps = import_difmap_model(mdl_fname, mdl_dir)
    # Sort components by distance from phase center
    comps = sorted(comps, key=lambda x: np.sqrt(x.p[1]**2 + x.p[2]**2))

    # Construct labels for corner and truth values (of difmap models)
    labels = list()
    truths = list()
    j = 0
    for i, comp in enumerate(comps):
        if i in components_to_plot:
            truths.extend(comp.p)
            indxs[j: j+comp.size] = np.ones(comp.size)
            if isinstance(comp, EGComponent):
                if comp.size == 6:
                    labels.extend([r'$flux$', r'$x$', r'$y$', r'$bmaj$', r'$e$', r'$bpa$'])
                elif comp.size == 4:
                    labels.extend([r'$flux$', r'$x$', r'$y$', r'$bmaj$'])
                elif comp.size == 3:
                    labels.extend([r'$flux$', r'$x$', r'$y$'])
                else:
                    raise Exception("Gauss component should have size 4 or 6!")
            elif isinstance(comp, DeltaComponent):
                labels.extend([r'$flux$', r'$x$', r'$y$'])
            else:
                raise Exception("Unknown type of component!")
        j += comp.size

    samples = samples[:, np.array(indxs, dtype=bool)]

    ndim = len(truths)
    if fig is None:
        fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
        fig.set_size_inches(14.5, 14.5)

    fig = corner.corner(samples, fig=fig, labels=labels,
                        truths=truths,
                        hist_kwargs={'normed': True,
                                     'histtype': 'step',
                                     'stacked': True,
                                     'ls': 'dashdot'},
                        smooth=0.5,
                        # show_titles=True,
                        title_kwargs={'fontsize': title_fontsize},
                        # quantiles=[0.16, 0.5, 0.84],
                        label_kwargs={'fontsize': label_fontsize},
                        title_fmt=".4f",
                        max_n_ticks=4,
                        plot_datapoints=False,
                        plot_contours=True,
                        levels=[0.68, 0.95],
                        bins=20,
                        range=limits,
                        fill_contours=True, color='green')
    # fig.tight_layout()
    # fig.savefig(os.path.join(outdir, outfname), bbox_inches='tight',
    #             dpi=200, format='pdf')
    return fig


if __name__ == '__main__':
    # data_dir = '/home/ilya/Dropbox/papers/boot/new_pics/corner/1807+698'
    # uv_fits = os.path.join(data_dir, '1807+698.u.2007_07_03.uvf')
    # mdl_file = os.path.join(data_dir, 'dfm_original_model_refitted.mdl')
    # data_dir = '/home/ilya/code/vlbi_errors/examples/LC/0552+398/2006_07_07'
    # uv_fits = os.path.join(data_dir, '0552+398.u.2006_07_07.uvf')
    # mdl_file = os.path.join(data_dir, 'initial.mdl')
    # mdl_file_rf = os.path.join(data_dir, 'initial_refitted.mdl')
    data_dir = '/home/ilya/Dropbox/papers/boot/new_pics/corner/new/parametric/1807+698'
    uv_fits = os.path.join(data_dir, '1807+698.u.2007_07_03.uvf')
    mdl_file_rf = os.path.join(data_dir, 'new2.mdl')
    # from spydiff import modelfit_difmap
    # modelfit_difmap('0552+398.u.2006_07_07.uvf', 'initial.mdl',
    #                 'initial_refitted.mdl', niter=300,
    #                 path=data_dir, mdl_path=data_dir,
    #                 out_path=data_dir)
    sampler, labels, truths = fit_model_with_mcmc(uv_fits, mdl_file_rf,
                                                  nburnin_2=100,
                                                  nproduction=300,
                                                  nwalkers=200,
                                                  samples_file='samples_of_mcmc.txt',
                                                  outdir=data_dir)

    samples = sampler.flatchain[::10, :]
    limits_0_1 = [(0.672, 0.696), (-0.0805, -0.0745), (-0.018, -0.015),
                  (0.215, 0.236), (0.22, 0.33), (2.85, 2.92), (0.24, 0.266),
                  (0.242, 0.26), (0.06, 0.07), (0.128, 0.16)]
    limits_10_11 = [(0.002, 0.010), (7.2, 8.4), (1, 1.8), (0.8, 2.4), (0.01, 0.0185),
              (12.2, 13.2), (2.6, 3.2), (1.8, 3.0)]
    fig = plot_comps([0, 1], samples, mdl_file_rf, outdir=data_dir, label_fontsize=12, title_fontsize=12)
    # fig = plot_comps([8, 9], samples, mdl_file, outdir=data_dir, label_fontsize=12, title_fontsize=12)

    # from bootstrap import bootstrap_uvfits_with_difmap_model
    # fig = bootstrap_uvfits_with_difmap_model(uv_fits, mdl_file_rf,
    #                                          boot_dir=data_dir, n_boot=300,
    #                                          clean_after=False,
    #                                          use_kde=True,
    #                                          use_v=False,
    #                                          out_plot_file=os.path.join(data_dir, 'plot_boot.pdf'),
    #                                          niter=100,
    #                                          bootstrapped_uv_fits=None)
