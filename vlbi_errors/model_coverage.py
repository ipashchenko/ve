import copy
import matplotlib
matplotlib.use('Agg')
label_size = 8
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
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
from from_fits import create_clean_image_from_fits_file


def download_mojave_models(base_dir):
    """
    Download MOJAVE models & uv-fits files.
    :param base_dir:
        Directory with MOJAVE table.
    """
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
            open('problem_download_from_mojave_{}_{}'.format(source, last_epoch), 'a').close()
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
            fn.write("{}v {}v {}v {} {} {} {} {} {}".format(flux, r, pa, bmaj, e,
                                                            bpa, type_, "0", "0\n"))
        fn.close()


# TODO: Extend to polarization (Q & U together)
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
    print("Substituting `original` uv-data with CLEAN model...")
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


def coverage_of_model(original_uv_fits, original_mdl_file, outdir=None,
                      n_cov=100, n_boot=300, mapsize=(1024, 0.1),
                      path_to_script=None):
    """
    Conduct coverage analysis of uv-data & model

    :param original_uv_fits:
        Self-calibrated uv-fits file.
    :param original_mdl_file:
        Difmap txt-file with model.
    :param outdir:
        Output directory to store results.
    :param n_cov:
        Number of samples to create.
    """
    # Create sample of 100 uv-fits data & models
    sample_uv_fits_paths, sample_model_paths = create_sample(original_uv_fits,
                                                             original_mdl_file,
                                                             outdir=outdir,
                                                             n_sample=n_cov)

    # For each sample uv-fits & model find 1) conventional errors & 2) bootstrap
    # errors
    for j, (sample_uv_fits_path, sample_mdl_path) in enumerate(zip(sample_uv_fits_paths,
                                                                   sample_model_paths)):
        sample_uv_fits, dir = os.path.split(sample_uv_fits_path)
        sample_mdl_file, dir = os.path.split(sample_mdl_path)
        try:
            comps = import_difmap_model(sample_mdl_file, dir)
        except ValueError:
            print('Problem import difmap model')
        model = Model(stokes='I')
        model.add_components(*comps)

        # Find errors by using Fomalont way
        # 1. Clean uv-data
        clean_difmap(sample_uv_fits, 'sample_cc_{}.fits'.format(j), 'I',
                     mapsize, path=dir, path_to_script=path_to_script,
                     outpath=dir)
        # 2. Get beam
        ccimage = create_clean_image_from_fits_file(os.path.join(dir,
                                                                 'sample_cc_{}.fits'.format(j)))
        beam = ccimage.beam_image

        # 2. Subtract components convolved with beam
        ccimage.substract_model(model)

        # Find errors by using Lee way
        # a) fit uv-data and find model
        # b) CLEAN uv-data
        # c) substract model from CLEAN image
        # d) find errors
        pass

        # Find errors by using bootstrap
        # FT model to uv-plane
        uvdata = UVData(sample_uv_fits_path)
        try:
            boot = CleanBootstrap([model], uvdata)
        # If uv-data contains only one Stokes parameter (e.g. `0838+133`)
        except IndexError:
            print('Problem bootstrapping')
        curdir = os.getcwd()
        os.chdir(dir)
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

        # Load bootstrap models
        booted_mdl_paths = glob.glob(os.path.join(data_dir, dfm_model_fname + "_*"))
        comps_params = {i: [] for i in range(len(comps))}
        for booted_mdl_path in booted_mdl_paths:
            path, booted_mdl_file = os.path.split(booted_mdl_path)
            comps = import_difmap_model(booted_mdl_file, path)
            for i, comp in enumerate(comps):
                comps_params[i].extend(list(comp.p))

        # Print 65-% intervals (1 sigma)
        for i, comp in enumerate(comps):
            errors_fname = '68_{}_{}_comp{}.txt'.format(source, last_epoch, i)
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

    # For source in sources with component close to core
    # 1. Find residuals or estimate noise
    # 2. N times add resampled residuals (or just gaussian noise) to model and
    # create N new datasets
    # 3. Fit them using difmap.
    # 4. Find errors using Fomalont, Yee and using bootstrap. Check coverage.
    base_dir = '/home/ilya/vlbi_errors/model_cov'
    n_boot = 300
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
            open('problem_download_from_mojave_{}_{}'.format(source, last_epoch), 'a').close()
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
            fn.write("{}v {}v {}v {} {} {} {} {} {}".format(flux, r, pa, bmaj, e,
                                                            bpa, type_, "0", "0\n"))
        fn.close()
