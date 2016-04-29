import glob
import os
import pandas as pd
import numpy as np
from utils import hdi_of_mcmc
from image_ops import rms_image
from mojave import download_mojave_uv_fits, mojave_uv_fits_fname
from spydiff import import_difmap_model, modelfit_difmap, clean_difmap
from uv_data import UVData
from model import Model
from bootstrap import CleanBootstrap
from components import DeltaComponent, CGComponent, EGComponent
from from_fits import create_image_from_fits_file
import matplotlib
matplotlib.use('Agg')

try:
    import corner as triangle
except ImportError:
    triangle = None

base_dir = '/home/ilya/vlbi_errors/mojave_mod'
n_boot = 30
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

    # Clean uv-data
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    clean_difmap(uv_fits_fname, 'original_cc.fits', 'I', (1024, 0.1),
                 path=data_dir, path_to_script=path_to_script,
                 outpath=data_dir)
    image = create_image_from_fits_file(os.path.join(data_dir,
                                                     'original_cc.fits'))
    rms = rms_image(image)

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
        fn.write("{}v {}v {}v {} {} {} {} {} {}".format(flux, r, pa, bmaj, e,
                                                        bpa, type_, "0", "0\n"))
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

    # Get params of initial model used for bootstrap
    comps = import_difmap_model(dfm_model_fname, data_dir)
    comps_params0 = {i: [] for i in range(len(comps))}
    for i, comp in enumerate(comps):
        comps_params0[i].extend(list(comp.p))

    # Load models and plot
    booted_mdl_paths = glob.glob(os.path.join(data_dir, dfm_model_fname + "_*"))
    comps_params = {i: [] for i in range(len(comps))}
    for booted_mdl_path in booted_mdl_paths:
        path, booted_mdl_file = os.path.split(booted_mdl_path)
        comps = import_difmap_model(booted_mdl_file, path)
        for i, comp in enumerate(comps):
            comps_params[i].extend(list(comp.p))

    n_pars = sum([len(comp) for comp in comps])
    labels = {0: "flux", 1: "x", 2: "y", 3: "FWHM"}

    # Optionally plot
    if triangle:
        lens = list(np.cumsum([len(comp) for comp in comps]))
        lens.insert(0, 0)
        for i, comp in enumerate(comps):
            # Show one component
            figure, axes = matplotlib.pyplot.subplots(nrows=len(comp), ncols=len(comp))
            triangle.corner(np.array(comps_params[i]).reshape((n_boot,
                                                               len(comp))),
                            labels=[r"${}$".format(lab) for lab in
                                    comp._parnames],
                            truths=comps_params0[i],
                            title_kwargs={"fontsize": 12},
                            quantiles=[0.16, 0.5, 0.84], fig=figure)
            figure.gca().annotate("Source {}, component {}".format(source, i),
                                  xy=(0.5, 1.0), xycoords="figure fraction",
                                  xytext=(0, -5), textcoords="offset points",
                                  ha="center", va="top")
            figure.savefig(os.path.join(data_dir,
                                        '{}_{}_comp{}.png'.format(source,
                                                                  last_epoch,
                                                                  i)),
                           bbox_inches='tight', dpi=300)
            #except ValueError:
            #    print "Failed to plot..."
    else:
        print "Install ``corner`` for corner-plots"

    # Print 65-% intervals (1 sigma)
    for i, comp in enumerate(comps):
        errors_fname = '68_{}_{}_comp{}_rms_{}.txt'.format(source, last_epoch, i,
                                                       "{0:.5f}".format(rms))
        fn = open(os.path.join(data_dir, errors_fname), 'w')
        print "Component #{}".format(i + 1)
        for j in range(len(comp)):
            low, high, mean, median = hdi_of_mcmc(np.array(comps_params[i]).reshape((n_boot,
                                                                                     len(comp))).T[j],
                                                  cred_mass=0.68,
                                                  return_mean_median=True)
            fn.write("{} {} {} {} {}".format(comp.p[j], low, high, mean,
                                             median))
            fn.write("\n")
        fn.close()
    # Print 95-% intervals (2 sigma)
    for i, comp in enumerate(comps):
        errors_fname = '95_{}_{}_comp{}_rms_{}.txt'.format(source, last_epoch, i,
                                                       "{0:.5f}".format(rms))
        fn = open(os.path.join(data_dir, errors_fname), 'w')
        print "Component #{}".format(i + 1)
        for j in range(len(comp)):
            low, high, mean, median = hdi_of_mcmc(np.array(comps_params[i]).reshape((n_boot,
                                                                                     len(comp))).T[j],
                                                  cred_mass=0.95,
                                                  return_mean_median=True)
            fn.write("{} {} {} {} {}".format(comp.p[j], low, high, mean,
                                             median))
            fn.write("\n")
        fn.close()

    # Cleaning up
    for booted_uv_path in booted_uv_paths:
        print("Removing file {}".format(booted_uv_path))
        os.unlink(booted_uv_path)
