import os
import pandas as pd
import numpy as np
import copy
import glob
from mojave import get_mojave_mdl_file
from mojave import download_mojave_uv_fits
from mojave import convert_mojave_epoch_to_float
from mojave import mojave_uv_fits_fname
from uv_data import UVData
from spydiff import import_difmap_model, modelfit_difmap
from model import Model
from bootstrap import CleanBootstrap


def create_sample(original_uv_fits, original_mdl_file, outdir=None,
                  n_sample=100, stokes='I'):
    """
    Create `sample` from `true` or `model` source

    :param outdir: (optional)
        Directory to store intermediate results. If ``None`` then use CWD.
        (default: ``None``)
    :param n_sample: (optional)
        Number of `samples` from infinite population to consider in coverage
        analysis of intervals. Here `samples` - observations of known source
        with different realisations of noise with known parameters. (default:
         ``100``)
    :param stokes: (optional)
        Stokes parameter to use. If ``None`` then use ``I``. (default: ``None``)
    """
    original_uv_data = UVData(original_uv_fits)
    noise = original_uv_data.noise()
    path, _ = os.path.split(original_mdl_file)
    comps = import_difmap_model(original_mdl_file, path)
    original_model = Model(stokes=stokes)
    original_model.add_components(*comps)

    # Substitute uv-data with original model and create `model` uv-data
    print("Substituting `original` uv-data with difmap model...")
    model_uv_data = copy.deepcopy(original_uv_data)
    model_uv_data.substitute([original_model])

    # Create `sample` uv-data
    # Add noise to `model` uv-data ``n_cov`` times and get ``n_cov`` `samples`
    # from population
    sample_uv_fits_paths = list()
    print("Creating {} `samples` from population".format(n_sample))
    for i in range(n_sample):
        sample_uv_data = copy.deepcopy(model_uv_data)
        sample_uv_data.noise_add(noise)
        sample_uv_fits_path = os.path.join(outdir,
                                           'sample_uv_{}.uvf'.format(str(i + 1).zfill(3)))
        sample_uv_data.save(sample_uv_fits_path)
        sample_uv_fits_paths.append(sample_uv_fits_path)

    # Fitting in difmap each `sample` FITS-file
    print("Fitting `samples` uv-data")
    for uv_fits_path in sample_uv_fits_paths:
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        j = uv_fits_fname.split('.')[0].split('_')[-1]
        print("Fitting {} sample uv-data to"
              " {}".format(uv_fits_path,
                           os.path.join(outdir,
                                        'sample_model_{}.mdl'.format(j))))
        modelfit_difmap(uv_fits_fname, original_mdl_file,
                        'sample_model_{}.mdl'.format(j), path=uv_fits_dir,
                        mdl_path=uv_fits_dir, out_path=uv_fits_dir)

    sample_mdl_paths = sorted(glob.glob(os.path.join(outdir,
                                                         'sample_model_*.mdl')))
    sample_uv_fits_paths = sorted(glob.glob(os.path.join(outdir,
                                                         'sample_uv_*.uvf')))
    return sample_uv_fits_paths, sample_mdl_paths


def bootstrap_uvfits_with_difmap_model(uv_fits_path, dfm_model_path,
                                       nonparametric=False, use_kde=False,
                                       use_v=False, n_boot=100, stokes='I',
                                       boot_dir=None, recenter=True,
                                       pairs=False, niter=100,
                                       bootstrapped_uv_fits=None,
                                       additional_noise=None,
                                       boot_mdl_outname_base="bootstrapped_model"):
    dfm_model_dir, dfm_model_fname = os.path.split(dfm_model_path)
    comps = import_difmap_model(dfm_model_fname, dfm_model_dir)
    if boot_dir is None:
        boot_dir = os.getcwd()
    if bootstrapped_uv_fits is None:
        uvdata = UVData(uv_fits_path)
        model = Model(stokes=stokes)
        model.add_components(*comps)
        boot = CleanBootstrap([model], uvdata, additional_noise=additional_noise)
        os.chdir(boot_dir)
        boot.run(nonparametric=nonparametric, use_kde=use_kde, recenter=recenter,
                 use_v=use_v, n=n_boot, pairs=pairs)
        bootstrapped_uv_fits = sorted(glob.glob(os.path.join(boot_dir,
                                                             'bootstrapped_data*.fits')))
    for j, bootstrapped_fits in enumerate(bootstrapped_uv_fits):
        modelfit_difmap(bootstrapped_fits, dfm_model_fname,
                        '{}_{}.mdl'.format(boot_mdl_outname_base, j),
                        path=boot_dir, mdl_path=dfm_model_dir,
                        out_path=boot_dir, niter=niter)
    booted_mdl_paths = glob.glob(os.path.join(boot_dir, '{}*'.format(boot_mdl_outname_base)))

    # Clean uv_fits
    for file_ in bootstrapped_uv_fits:
        os.unlink(file_)
    logs = glob.glob(os.path.join(boot_dir, "*.log*"))
    for file_ in logs:
        os.unlink(file_)
    comms = glob.glob(os.path.join(boot_dir, "*commands*"))
    for file_ in comms:
        os.unlink(file_)

    return booted_mdl_paths


def calculate_coverage_of_difmap_model(original_dfm_model,
                                       bootstrapped_dfm_models_for_sample,
                                       alpha=0.68):
    from utils import hdi_of_mcmc
    original_comps = import_difmap_model(original_dfm_model)
    true_params = dict()
    boot_params = dict()
    coverage_params = dict()
    for i, comp in enumerate(original_comps):
        true_params[i] = dict()
        boot_params[i] = dict()
        coverage_params[i] = dict()
        for j, parname in enumerate(comp.parnames):
            true_params[i].update({parname: comp.p[j]})
            boot_params[i].update({parname: list()})
            coverage_params[i].update({parname: list()})
    for k, bootstrapped_dfm_models in enumerate(bootstrapped_dfm_models_for_sample):
        boot_params_ = boot_params.copy()
        for bootstrapped_dfm_model in bootstrapped_dfm_models:
            comps = import_difmap_model(bootstrapped_dfm_model)
            for i, comp in enumerate(comps):
                for j, parname in enumerate(comp.parnames):
                    boot_params_[i][parname].append(comp.p[j])
        for i, comp in enumerate(comps):
            for j, parname in enumerate(comp.parnames):
                low, high = hdi_of_mcmc(boot_params_[i][parname])
                coverage_params[i][parname].append(low < true_params[i][parname] < high)
    return true_params, boot_params, coverage_params
# def coverage_of_model(original_uv_fits, original_mdl_file, outdir=None,
#                       n_cov=100, n_boot=300, sample_model_paths=None,
#                       sample_uv_fits_paths=None):
#     """
#     Conduct coverage analysis of uv-data & model
#
#     :param original_uv_fits:
#         Self-calibrated uv-fits file.
#     :param original_mdl_file:
#         Difmap txt-file with model.
#     :param outdir:
#         Output directory to store results.
#     :param n_cov:
#         Number of samples to create.
#     """
#     if sample_model_paths is None or sample_uv_fits_paths is None:
#         # Create sample of 100 uv-fits data & models
#         sample_uv_fits_paths, sample_model_paths = create_sample(original_uv_fits,
#                                                                  original_mdl_file,
#                                                                  outdir=outdir,
#                                                                  n_sample=n_cov)
#
#     # For each sample uv-fits & model find 1) conventional errors & 2) bootstrap
#     # errors
#     for j, (sample_uv_fits_path, sample_mdl_path) in enumerate(zip(sample_uv_fits_paths,
#                                                                    sample_model_paths)):
#         sample_uv_fits, dir = os.path.split(sample_uv_fits_path)
#         sample_mdl_file, dir = os.path.split(sample_mdl_path)
#         try:
#             comps = import_difmap_model(sample_mdl_file, dir)
#         except ValueError:
#             print('Problem import difmap model')
#         model = Model(stokes='I')
#         model.add_components(*comps)
#
#         # Find errors by using bootstrap
#         # FT model to uv-plane
#         uvdata = UVData(sample_uv_fits_path)
#         try:
#             boot = CleanBootstrap([model], uvdata)
#         # If uv-data contains only one Stokes parameter (e.g. `0838+133`)
#         except IndexError:
#             print('Problem bootstrapping')
#         curdir = os.getcwd()
#         os.chdir(dir)
#         outname = ['sample_uv_boot', '.uvf']
#         boot.run(n=n_boot, outname=outname, nonparametric=False, use_v=False,
#                  use_kde=False)
#         os.chdir(curdir)
#         booted_uv_paths = sorted(glob.glob(os.path.join(outdir, outname + "*")))
#
#         # Modelfit bootstrapped uvdata
#         for booted_uv_path in booted_uv_paths:
#             path, booted_uv_file = os.path.split(booted_uv_path)
#             i = booted_uv_file.split('_')[-1].split('.')[0]
#             modelfit_difmap(booted_uv_file, original_mdl_file,
#                             original_mdl_file + '_' + str(i),
#                             path=source_dir, mdl_path=source_dir,
#                             out_path=source_dir)
#
#         # Get params of initial model used for bootstrap
#         comps = import_difmap_model(dfm_model_fname, data_dir)
#         comps_params0 = {i: [] for i in range(len(comps))}
#         for i, comp in enumerate(comps):
#             comps_params0[i].extend(list(comp.p))
#
#         # Load bootstrap models
#         booted_mdl_paths = glob.glob(os.path.join(data_dir, dfm_model_fname + "_*"))
#         comps_params = {i: [] for i in range(len(comps))}
#         for booted_mdl_path in booted_mdl_paths:
#             path, booted_mdl_file = os.path.split(booted_mdl_path)
#             comps = import_difmap_model(booted_mdl_file, path)
#             for i, comp in enumerate(comps):
#                 comps_params[i].extend(list(comp.p))
#
#         # Print 65-% intervals (1 sigma)
#         for i, comp in enumerate(comps):
#             errors_fname = '68_{}_{}_comp{}.txt'.format(source, last_epoch, i)
#             fn = open(os.path.join(data_dir, errors_fname), 'w')
#             print "Component #{}".format(i + 1)
#             for j in range(len(comp)):
#                 low, high, mean, median = hdi_of_mcmc(np.array(comps_params[i]).reshape((n_boot,
#                                                                                          len(comp))).T[j],
#                                                       cred_mass=0.68,
#                                                       return_mean_median=True)
#                 fn.write("{} {} {} {} {}".format(comp.p[j], low, high, mean,
#                                                  median))
#                 fn.write("\n")
#             fn.close()


base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mojave_chisq'

names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
         'bpa']
df1 = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
                    names=names, dtype={key: str for key in names},
                    index_col=False)
names = ['source', 'id', 'flux', 'r', 'pa', 'n_mu', 't_mid', 's_ra', 's_dec']
df2 = pd.read_table(os.path.join(base_dir, 'asu_chisq_all.tsv'), sep='\t',
                    header=None, names=names, dtype={key: str for key in names},
                    index_col=False, skiprows=1)


source_dict = dict()
for source in df2['source'].unique():
    # source_epochs = get_epochs_for_source(source)
    df = df1.loc[df1['source'] == source]
    source_epochs = df['epoch'].values
    source_epochs_ = list()
    for epoch in source_epochs:
        try:
            epoch_ = convert_mojave_epoch_to_float(epoch)
        except ValueError:
            print("Can't convert epoch : {} for source {}".format(epoch, source))
            continue
        source_epochs_.append(epoch_)
    source_dict[source] = list()
    df = df2.loc[df2['source'] == source]
    for index, row in df.iterrows():
        if row['n_mu'] != 'a':
            id = int(row['id'])
            flux = float(row['flux'])
            r = float(row['r'])
            pa = float(row['pa'])
            t_mid = float(row['t_mid'])
            closest_epoch = source_epochs[np.argmin(abs(np.array(source_epochs_)
                                                        - t_mid))]
            sigma_pos = np.hypot(float(row['s_ra']), float(row['s_dec']))
            source_dict[source].append([id, flux, r, pa, t_mid, closest_epoch,
                                        sigma_pos])

for source in sorted(df2['source'].unique()):
    if len(source_dict[source]) > 5:
        epoch = source_dict[source][0][5]
        source_dir = os.path.join(base_dir, source, epoch)
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)

        get_mojave_mdl_file(os.path.join(base_dir, 'asu.tsv'),
                            source, epoch, outdir=source_dir)
        epoch_ = "{}_{}_{}".format(*epoch.split('-'))
        download_mojave_uv_fits(source, epochs=[epoch_],
                                download_dir=source_dir, bands=['u'])
        fname = mojave_uv_fits_fname(source, 'u', epoch_)
        uvdata = UVData(os.path.join(source_dir, fname))
        print(uvdata.stokes)
        if 'RR' not in uvdata.stokes or 'LL' not in uvdata.stokes:
            continue

        # Refit difmap model
        modelfit_difmap(fname, "{}_{}.mdl".format(source, epoch),
                        "{}_{}.mdl".format(source, epoch), niter=200,
                        path=source_dir, mdl_path=source_dir,
                        out_path=source_dir, show_difmap_output=True)

        # Create sample of 100 artificial data sets
        sample_uv_fits_paths, sample_mdl_paths =\
            create_sample(os.path.join(source_dir, fname),
            os.path.join(source_dir, "{}_{}.mdl".format(source, epoch)),
            outdir=source_dir, n_sample=100)

        for i, (sample_uv_fits_path, sample_mdl_path) in enumerate(zip(sample_uv_fits_paths, sample_mdl_paths)):
            # Create bootstrapped model files
            booted_mdl_paths =\
                bootstrap_uvfits_with_difmap_model(sample_uv_fits_path,
                sample_mdl_path, boot_dir=source_dir, n_boot=100,
                boot_mdl_outname_base="booted_mdl_for_sample_{}".format(i))
