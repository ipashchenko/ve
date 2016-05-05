import os
import glob
try:
    import corner as triangle
except ImportError:
    triangle = None
import numpy as np
from uv_data import UVData
from components import DeltaComponent, CGComponent, EGComponent
from bootstrap import CleanBootstrap
from model import Model
from spydiff import modelfit_difmap, import_difmap_model
import matplotlib.pyplot as plt
from utils import hdi_of_mcmc


path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
data_dir = '/home/ilya/sandbox/modelfit/2models'
uv_fname = '1226+023.q1.2009_08_16.uvp'
mdl_fname = '1226+023.q1.2009_08_16.mdl'
outname = 'boot_uv'
n = 300


if __name__ == '__main__':

    uvdata = UVData(os.path.join(data_dir, uv_fname))
    model = Model(stokes='I')
    comps = import_difmap_model(mdl_fname, data_dir)
    model.add_components(*comps)
    boot = CleanBootstrap([model], uvdata)
    curdir = os.getcwd()
    os.chdir(data_dir)
    boot.run(n=n, nonparametric=True, outname=[outname, '.fits'])
    os.chdir(curdir)

    # # Radplot uv-data and model
    # comps = import_difmap_model(mdl_fname, data_dir)
    # uvdata.uvplot(style='a&p')
    # uvdata.substitute([model])
    # uvdata.uvplot(style='a&p', sym='.r')

    # # Radplot residuals
    # uvdata_ = create_uvdata_from_fits_file(os.path.join(data_dir, uv_fname_cc))
    # res_uvdata = uvdata_ - uvdata
    # res_uvdata.uvplot(style='re&im')

    booted_uv_paths = glob.glob(os.path.join(data_dir, outname + "*"))
    # Modelfit bootstrapped uvdata
    for booted_uv_path in booted_uv_paths:
        path, booted_uv_file = os.path.split(booted_uv_path)
        i = booted_uv_file.split('_')[-1].split('.')[0]
        modelfit_difmap(booted_uv_file, mdl_fname, mdl_fname + '_' + i,
                        path=path, mdl_path=data_dir, out_path=data_dir)

    # Load models and plot
    params = list()
    booted_mdl_paths = glob.glob(os.path.join(data_dir, mdl_fname + "_*"))
    for booted_mdl_path in booted_mdl_paths:
        path, booted_mdl_file = os.path.split(booted_mdl_path)
        comps = import_difmap_model(booted_mdl_file, path)
        comps_params = list()
        for comp in comps:
            if isinstance(comp, CGComponent):
                comps_params.append(np.array(list(comp.p) + [0., 0.]))
            elif isinstance(comp, EGComponent) and not isinstance(comp,
                                                                  CGComponent):
                comps_params.append(comp.p)
            elif isinstance(comp, DeltaComponent):
                comps_params.append(np.array(list(comp.p) + [0., 0., 0.]))
        params.append(comps_params)
    # (#comp, 6, #boot)
    params = np.dstack(params)

    # Get params of initial model used for bootstrap
    comps = import_difmap_model(mdl_fname, data_dir)
    params0 = list()
    extents = list()
    for comp in comps:
        if isinstance(comp, CGComponent):
            params0.append(np.array(list(comp.p) + [0., 0.]))
        if isinstance(comp, EGComponent) and not isinstance(comp, CGComponent):
            params0.append(comp.p)
        elif isinstance(comp, DeltaComponent):
            params0.append(np.array(list(comp.p) + [0., 0., 0.]))
    for comp in comps:
        if isinstance(comp, CGComponent):
            extents.append([1., 1., 1., 1., (-0.5, 0.5), (-0.5, 0.5)])
        if isinstance(comp, EGComponent) and not isinstance(comp, CGComponent):
            extents.append([1., 1., 1., 1., 1., 1.])
        elif isinstance(comp, DeltaComponent):
            extents.append([1., 1., 1., (-0.5, 0.5), (-0.5, 0.5),
                            (-0.5, 0.5)])
    extents = [val for sublist in extents for val in sublist]
    # (#comps, #params)
    params0 = np.vstack(params0)

    n_pars_true = len(params0.flatten())
    labels = {0: "flux", 1: "x", 2: "y", 3: "FWHM"}
    # labels = {0: "flux", 1: "x", 2: "y"}
    if triangle:
        # Show one component
        fig, axes = plt.subplots(nrows=4, ncols=4)
        triangle.corner(params.reshape((n_pars_true, n)).T[:, 90:94],
                        extents=extents[90:94],
                        # labels=labels,
                        truths=params0.reshape(n_pars_true)[90:94], fig=fig)
        fig.savefig('sampling_distributions_16.png', bbox_inches='tight',
                    dpi=300)
    else:
        print "Install ``triangle`` for corner-plots"

    labels = {0: "flux", 1: "x", 2: "y", 3: "FWHM"}
    # Print 65-% intervals (1 sigma)
    for i in range(12):
        print "Component #{}".format(i + 1)
        for j in range(4):
            low, high = hdi_of_mcmc(params[i, j, :], cred_mass=0.65)
            print "65% hdi of {0}: [{1:.4f}, {2:.4f}]".format(labels[j], low,
                                                              high)
    # Print 95-% intervals (2 sigma)
    for i in range(12):
        print "Component #{}".format(i + 1)
        for j in range(4):
            low, high = hdi_of_mcmc(params[i, j, :])
            print "95% hdi of {0}: [{1:.4f}, {2:.4f}]".format(labels[j], low,
                                                              high)
