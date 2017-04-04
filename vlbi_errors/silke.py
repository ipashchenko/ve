import os
import json
import time
import numpy as np
import scipy as sp
import corner
import nestle
from utils import _function_wrapper
from collections import OrderedDict
from functools import partial
from uv_data import UVData
from stats import LnLikelihood
from model import Model
from spydiff import import_difmap_model


def hypercube_full(u, ppfs):
    assert len(u) == len(ppfs)
    return [ppf(u_) for ppf, u_ in zip(ppfs, u)]


def hypercube_partial(ppfs):
    return partial(hypercube_full, ppfs=ppfs)


def fit_model_with_nestle(uv_fits, model_file, components_priors, outdir=None,
                          **nestle_kwargs):
    """
    :param uv_fits:
        Path to uv-fits file with self-calibrated visibilities.
    :param model_file:
        Path to file with difmap model.
    :param components_priors:
        Components prior's ppf. Close to phase center component goes first.
        Iterable of dicts with keys - name of the parameter and values -
        (callable, args, kwargs,) where args & kwargs - additional arguments to
        callable. Each callable is called callable.ppf(p, *args, **kwargs).
        Thus callable should has ``ppf`` method.

        Example of prior on single component:
            {'flux': (scipy.stats.uniform.ppf, [0., 10.], dict(),),
             'bmaj': (scipy.stats.uniform.ppf, [0, 5.], dict(),),
             'e': (scipy.stats.beta.ppf, [alpha, beta], dict(),)}
        First key will result in calling: scipy.stats.uniform.ppf(u, 0, 10) as
        value from prior for ``flux`` parameter.
    :param outdir: (optional)
        Directory to output results. If ``None`` then use cwd. (default:
        ``None``)
    :param nestle_kwargs: (optional)
        Any arguments passed to ``nestle.sample`` function.

    :return
        Results of ``nestle.sample`` work on that model.
    """
    if outdir is None:
        outdir = os.getcwd()

    mdl_file = model_file
    uv_data = UVData(uv_fits)
    mdl_dir, mdl_fname = os.path.split(mdl_file)
    comps = import_difmap_model(mdl_fname, mdl_dir)

    # Sort components by distance from phase center
    comps = sorted(comps, key=lambda x: np.sqrt(x.p[1]**2 + x.p[2]**2))

    ppfs = list()
    labels = list()
    for component_prior in components_priors:
        for comp_name in ('flux', 'x', 'y', 'bmaj', 'e', 'bpa'):
            try:
                ppfs.append(_function_wrapper(*component_prior[comp_name]))
                labels.append(comp_name)
            except KeyError:
                pass

    for ppf in ppfs:
        print(ppf.args)

    hypercube = hypercube_partial(ppfs)

    # Create model
    mdl = Model(stokes=stokes)
    # Add components to model
    mdl.add_components(*comps)
    loglike = LnLikelihood(uv_data, mdl)
    time0 = time.time()
    result = nestle.sample(loglikelihood=loglike, prior_transform=hypercube,
                           ndim=mdl.size, npoints=50, method='multi',
                           callback=nestle.print_progress, **nestle_kwargs)
    print("Time spent : {}".format(time.time()-time0))
    samples = nestle.resample_equal(result.samples, result.weights)
    # Save re-weighted samples from posterior to specified ``outdir``
    # directory
    np.savetxt(os.path.join(outdir, 'samples.txt'), samples)
    fig = corner.corner(samples, show_titles=True, labels=labels,
                        quantiles=[0.16, 0.5, 0.84], title_fmt='.3f')
    # Save corner plot os samples from posterior to specified ``outdir``
    # directory
    fig.savefig(os.path.join(outdir, "corner.png"), bbox_inches='tight',
                dpi=200)

    return result


if __name__ == '__main__':
    data_dir = '/home/ilya/code/vlbi_errors/silke/'
    outdir = data_dir
    uv_fits = os.path.join(data_dir, '0851+202.u.2004_11_05.uvf')
    mdl_file = os.path.join(data_dir, '1.mod.2004_11_05')
    stokes = 'I'

    components_priors = list()
    components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 5], {}),
                                 'x': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'y': (sp.stats.uniform.ppf, [-0.5, 1], {})})
    components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'y': (sp.stats.uniform.ppf, [-0.5, 1], {})})
    components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                              'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                              'y': (sp.stats.uniform.ppf, [-1, 2], {})})
    components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-2, 4], {}),
                                 'y': (sp.stats.uniform.ppf, [-2, 4], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                              'x': (sp.stats.uniform.ppf, [-5, 10], {}),
                              'y': (sp.stats.uniform.ppf, [-5, 10], {}),
                              'bmaj': (sp.stats.uniform.ppf, [0, 2], {})})
    components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                              'x': (sp.stats.uniform.ppf, [-6, 12], {}),
                              'y': (sp.stats.uniform.ppf, [-6, 12], {}),
                              'bmaj': (sp.stats.uniform.ppf, [0, 3], {})})
    results = fit_model_with_nestle(uv_fits, mdl_file, components_priors,
                                    outdir=outdir)


data_dir = '/home/ilya/code/vlbi_errors/silke'
# uv_fits = '0851+202.u.2012_11_11.uvf'
uv_fits = '0851+202.u.2004_11_05.uvf'
# mdl_fname = '2.mod.2012_11_11'
mdl_fname = '1.mod.2004_11_05'
uv_data = UVData(os.path.join(data_dir, uv_fits))
comps = import_difmap_model(mdl_fname, data_dir)
model = Model(stokes='I')
model.add_components(*comps)

fig = uv_data.uvplot(style='a&p')
uv_data.substitute([model])
uv_data.uvplot(color='r', fig=fig, phase_range=[-0.2, 0.2])

