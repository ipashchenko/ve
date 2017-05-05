import os
import json
import time
import numpy as np
import scipy as sp
import corner
from uv_data import UVData
from spydiff import import_difmap_model
from model import Model
from stats import LnLikelihood
import nestle
from utils import _function_wrapper
from collections import OrderedDict
from functools import partial


def hypercube_full(u, ppfs):
    assert len(u) == len(ppfs)
    return [ppf(u_) for ppf, u_ in zip(ppfs, u)]


def hypercube_partial(ppfs):
    return partial(hypercube_full, ppfs=ppfs)


def check_resolved(uv_fits, mdl_dict, components_priors, outdir=None,
                   **nestle_kwargs):
    """
    :param uv_fits:
        Path to uv-fits file with self-calibrated visibilities.
    :param mdl_dict:
        Dictionary with keys - component types ('pt', 'cg', 'el') and values -
        paths to difmap-style model files.
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
        Dictionary with keys - model types and values - results of
        ``nestle.sample`` work on that model.
    """
    if outdir is None:
        outdir = os.getcwd()

    evidences = {}
    result_dict = OrderedDict()

    for comp_type in ('pt', 'cg', 'el'):
        print("\nWorking on component type: {}\n".format(comp_type))
        mdl_file = mdl_dict[comp_type]
        uv_data = UVData(uv_fits)
        mdl_dir, mdl_fname = os.path.split(mdl_file)
        comps = import_difmap_model(mdl_fname, mdl_dir)

        # Sort components by distance from phase center
        comps = sorted(comps, key=lambda x: np.sqrt(x.p[1]**2 + x.p[2]**2))

        ppfs = list()
        labels = list()
        components_prior = components_priors[comp_type]
        for component_prior in components_prior:
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
        np.savetxt(os.path.join(outdir, '{}_samples.txt'.format(comp_type)),
                   samples)
        fig = corner.corner(samples, show_titles=True, labels=labels,
                            quantiles=[0.16, 0.5, 0.84], title_fmt='.3f')
        # Save corner plot os samples from posterior to specified ``outdir``
        # directory
        fig.savefig(os.path.join(outdir, "{}_corner.png".format(comp_type)),
                    bbox_inches='tight', dpi=200)
        result_dict[comp_type] = result
        evidences[comp_type] = (result['logz'], result['logzerr'])

    with open(os.path.join(outdir, 'logz_logzerr.json'), 'w') as fo:
        json.dump(evidences, fo)

    return result_dict


if __name__ == '__main__':
    data_dir = '/home/ilya/code/vlbi_errors/bin_q/'
    # data_dir = '/home/ilya/Dropbox/0235/tmp/2ilya'
    outdir = '/home/ilya/Dropbox/0235/tmp/evidence/Q'
    uv_fits = os.path.join(data_dir, '0235+164.q1.2008_09_02.uvf_difmap')
    mdl_file_pt = 'qmodel_point.mdl'
    mdl_file_cg = 'qmodel_circ.mdl'
    mdl_file_el = '0235+164.q1.2008_09_02.mdl'
    mdl_dict = {'pt': mdl_file_pt, 'cg': mdl_file_cg, 'el': mdl_file_el}
    for key, value in mdl_dict.items():
        mdl_dict.update({key: os.path.join(data_dir, value)})
    stokes = 'I'

    pt_components_priors = list()
    pt_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 4], {}),
                                 'x': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'y': (sp.stats.uniform.ppf, [-0.5, 1], {})})
    pt_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 2], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    pt_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    pt_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 2], {})})

    cg_components_priors = list()
    cg_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 4], {}),
                                 'x': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'y': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    cg_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 2], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    cg_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    cg_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 2], {})})

    el_components_priors = list()
    el_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 4], {}),
                                 'x': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'y': (sp.stats.uniform.ppf, [-0.5, 1], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'e': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'bpa': (sp.stats.uniform.ppf, [0, np.pi], {})})
    el_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 2], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    el_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 1], {})})
    el_components_priors.append({'flux': (sp.stats.uniform.ppf, [0, 1], {}),
                                 'x': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'y': (sp.stats.uniform.ppf, [-1, 2], {}),
                                 'bmaj': (sp.stats.uniform.ppf, [0, 2], {})})
    components_priors = {'pt': pt_components_priors,
                         'cg': cg_components_priors,
                         'el': el_components_priors}
    results = check_resolved(uv_fits, mdl_dict, components_priors,
                             outdir=outdir)
