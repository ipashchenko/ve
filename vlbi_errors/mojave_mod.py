import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import hdi_of_mcmc
from mojave import download_mojave_uv_fits, mojave_uv_fits_fname
from spydiff import import_difmap_model, modelfit_difmap
from uv_data import UVData
from model import Model
from bootstrap import CleanBootstrap
from components import DeltaComponent, CGComponent, EGComponent

try:
    import corner as triangle
except ImportError:
    triangle = None

base_dir = '/home/ilya/vlbi_errors/mojave_mod'
n_boot = 200
outname = 'boot_uv'
names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
         'bpa']
df = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
                   names=names, dtype={key: str for key in names},
                   index_col=False)

# Mow for all sources get the latest epoch and create directory for analysis
for source in df['source'].unique():
    epochs = df.loc[df['source'] == source]['epoch']
    last_epoch_ = list(epochs)[-1]
    last_epoch = last_epoch_.replace('-', '_')
    data_dir = os.path.join(base_dir, source, last_epoch)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    try:
        download_mojave_uv_fits(source, epochs=[last_epoch], bands=['u'],
                                download_dir=data_dir)
    except:
        open('problem_download_from_mojave', 'a').close()
        continue
    uv_fits_fname = mojave_uv_fits_fname(source, 'u', last_epoch)

    # Create instance of Model and bootstrap uv-data
    dfm_model_fname = 'dfmp_original_model.mdl'
    fn = open(os.path.join(data_dir, dfm_model_fname), 'w')
    model_df = df.loc[np.logical_and(df['source'] == source,
                                     df['epoch'] == last_epoch_)]
    for (flux, r, pa, bmaj, e, bpa) in np.asarray(model_df[['flux', 'r', 'pa',
                                                            'bmaj', 'e',
                                                            'bpa']]):
        print flux, r, pa, bmaj, e, bpa
        if not r.strip(' '):
            r = '0.0'
        if not pa.strip(' '):
            pa = '0.0'

        if not bmaj.strip(' '):
            bmaj = '0.0'
        if not e.strip(' '):
            e = "1.0"

        if np.isnan(float(bpa)):
            bpa = "0.0"
        else:
            bpa = bpa + 'v'

        if bmaj == '0.0':
            type_ = 0
            bpa = "0.0"
        else:
            bmaj = bmaj + 'v'
            type_ = 1
        fn.write("{}v {}v {}v {} {} {} {} {} {}".format(flux, r, pa, bmaj, e, bpa, type_, "0", "0\n"))
    fn.close()

    try:
        comps = import_difmap_model(dfm_model_fname, data_dir)
    except ValueError:
        open('problem_import_difmap_model', 'a').close()
        continue
    uvdata = UVData(os.path.join(data_dir, uv_fits_fname))
    model = Model(stokes='I')
    model.add_components(*comps)
    try:
        boot = CleanBootstrap([model], uvdata)
    # If uv-data contains only one Stokes parameter (e.g. `0838+133`)
    except IndexError:
        open('problem_bootstrapping', 'a').close()
        continue
    curdir = os.getcwd()
    os.chdir(data_dir)
    boot.run(n=n_boot, nonparametric=True, outname=[outname, '.fits'])
    os.chdir(curdir)

    booted_uv_paths = sorted(glob.glob(os.path.join(data_dir, outname + "*")))
    # Modelfit bootstrapped uvdata
    for booted_uv_path in booted_uv_paths:
        path, booted_uv_file = os.path.split(booted_uv_path)
        i = booted_uv_file.split('_')[-1].split('.')[0]
        modelfit_difmap(booted_uv_file, dfm_model_fname,
                        dfm_model_fname + '_' + i,
                        path=path, mdl_path=data_dir, out_path=data_dir)

    # Load models and plot
    params = list()
    booted_mdl_paths = glob.glob(os.path.join(data_dir, dfm_model_fname + "_*"))
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
    comps = import_difmap_model(dfm_model_fname, data_dir)
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
        try:
            triangle.corner(params.reshape((n_pars_true, n_boot)).T[:, :4],
                            # extents=extents[:4],
                            # labels=labels,
                            truths=params0.reshape(n_pars_true)[:4], fig=fig,
                            plot_contours=False)
            fig.savefig(os.path.join(data_dir, '{}_{}_core.png'.format(source, last_epoch)),
                        bbox_inches='tight', dpi=300)
        except ValueError:
            print "Failed to plot..."
    else:
        print "Install ``corner`` for corner-plots"

    labels = {0: "flux", 1: "x", 2: "y", 3: "FWHM"}
    # Print 65-% intervals (1 sigma)
    errors_fname = '68_{}_{}.txt'.format(source, last_epoch)
    fn = open(os.path.join(data_dir, errors_fname), 'w')
    for i in range(len(comps)):
        print "Component #{}".format(i + 1)
        for j in range(4):
            low, high = hdi_of_mcmc(params[i, j, :], cred_mass=0.68)
            fn.write("{} ".format(high - low))
            print "68% hdi of {0}: [{1:.4f}, {2:.4f}]".format(labels[j], low,
                                                              high)
        fn.write("\n")
    fn.close()
    # Print 95-% intervals (2 sigma)
    errors_fname = '95_{}_{}.txt'.format(source, last_epoch)
    fn = open(os.path.join(data_dir, errors_fname), 'w')
    for i in range(len(comps)):
        print "Component #{}".format(i + 1)
        for j in range(4):
            low, high = hdi_of_mcmc(params[i, j, :])
            fn.write("{} ".format(high - low))
            print "95% hdi of {0}: [{1:.4f}, {2:.4f}]".format(labels[j], low,
                                                              high)
        fn.write("\n")
    fn.close()

    # Cleaning up
    for booted_uv_path in booted_uv_paths:
        print("Removing file {}".format(booted_uv_path))
        os.unlink(booted_uv_path)
