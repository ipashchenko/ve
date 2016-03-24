import copy
import os
import glob
import numpy as np
from spydiff import clean_difmap
from uv_data import UVData
from from_fits import (create_model_from_fits_file, create_image_from_fits_file)
from bootstrap import CleanBootstrap
from utils import (hdi_of_mcmc, get_fits_image_info, mas_to_rad, bc_endpoint)


def bootstrap_uv_fits(uv_fits_path, cc_fits_paths, n, outpath=None,
                      outname=None):
    """
    Function that bootstraps uv-data in user-specified FITS-files and
    FITS-files with clean components.

    :param uv_fits_path:
        Path to fits file with self-calibrated uv-data.
    :param cc_fits_paths:
        Iterable of paths to files with CC models.
    :param n:
        Number of bootstrap realizations.
    :param outpath: (optional)
        Directory to save bootstrapped uv-data FITS-files. If ``None``
        then use CWD. (default: ``None``)
    :param outname: (optional)
        How to name bootstrapped uv-data FITS-files. If ``None`` then
        use default for ``Bootstap.run`` method. (default: ``None``)

    """

    uvdata = UVData(uv_fits_path)

    models = list()
    for cc_fits_path in cc_fits_paths:
        ccmodel = create_model_from_fits_file(cc_fits_path)
        models.append(ccmodel)

    boot = CleanBootstrap(models, uvdata)
    if outpath is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    curdir = os.getcwd()
    os.chdir(outpath)
    boot.run(n=n, outname=outname)
    os.chdir(curdir)


def boot_ci(boot_cc_fits_paths, original_cc_fits_path, alpha=0.68):
    """
    Calculate bootstrap CI.

    :param boot_cc_fits_paths:
        Iterable of paths to bootstrapped CC FITS-files.
    :param original_cc_fits_path:
        Path to original CC FITS-file.
    :return:
        Two numpy arrays with low and high CI borders for each pixel.

    """
    original_image = create_image_from_fits_file(original_cc_fits_path)
    boot_images = list()
    for boot_cc_fits_path in boot_cc_fits_paths:
        print("Reading image from {}".format(boot_cc_fits_path))
        image = create_image_from_fits_file(boot_cc_fits_path)
        boot_images.append(image.image)

    images_cube = np.dstack(boot_images)
    boot_ci = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(boot_ci):
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=alpha)
        boot_ci[x, y] = hdi[1] - hdi[0]

    hdi_low = original_image.image - boot_ci / 2.
    hdi_high = original_image.image + boot_ci / 2.

    return hdi_low, hdi_high


def boot_ci_bc(boot_cc_fits_paths, original_cc_fits_path, alpha=0.68):
    """
    Calculate bootstrap CI.

    :param boot_cc_fits_paths:
        Iterable of paths to bootstrapped CC FITS-files.
    :param original_cc_fits_path:
        Path to original CC FITS-file.
    :return:
        Two numpy arrays with low and high CI borders for each pixel.

    """
    alpha = 0.5 * (1. - alpha)
    original_image = create_image_from_fits_file(original_cc_fits_path)
    boot_images = list()
    for boot_cc_fits_path in boot_cc_fits_paths:
        print("Reading image from {}".format(boot_cc_fits_path))
        image = create_image_from_fits_file(boot_cc_fits_path)
        boot_images.append(image.image)

    images_cube = np.dstack(boot_images)
    boot_ci_0 = np.zeros(np.shape(images_cube[:, :, 0]))
    boot_ci_1 = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(boot_ci_0):
        boot_ci_0[x, y] = bc_endpoint(images_cube[x, y, :],
                                      original_image.image[x, y], alpha)
        boot_ci_1[x, y] = bc_endpoint(images_cube[x, y, :],
                                      original_image.image[x, y], 1. - alpha)

    return boot_ci_0, boot_ci_1


def create_coverage_map(original_uv_fits_path, ci_type,
                        original_cc_fits_path=None, imsize=None,
                        outdir=None, n_boot=200, path_to_script=None,
                        alpha=0.68, n_cov=100, n_rms=1., stokes='I',
                        boot_cc_fits_paths=None, sample_cc_fits_paths=None):
    """
    Conduct coverage analysis of image pixels flux CI. Find number of times
    when CI of `observed` value contains values of `samples`.

    :param original_uv_fits_path:
        Path to original FITS-file with uv-data.
    :param ci_type:
        Type of CI to test. ``boot`` or ``rms``. If ``boot`` then use residuals
        bootstrap CI. If ``rms`` then use Hovatta corrected image rms CI.
    :param original_cc_fits_path: (optional)
        Path to original FITS-file with CC model. If ``None`` then use
        ``imsize`` parameter to get `original` CC model from
        ``original_uv_fits_path``. (default: ``None``)
    :param imsize: (optional)
        Image parameters (image size [pix], pixel size [mas]) to use
        when doing first CC with ``original_cc_fits_path = None``. (default:
        ``None``)
    :param outdir: (optional)
        Directory to store intermediate results. If ``None`` then use CWD.
        (default: ``None``)
    :param n_boot: (optional)
        Number of bootstrap replications to use when calculating bootstrap CI
        for ``ci_type = boot`` option when ``boot_cc_fits_paths`` hasn't
        specified. (default: ``200``)
    :param path_to_script: (optional)
        Path to Dan Homan's script for final clean. If ``None`` then use CWD.
        (default: ``None``)
    :param alpha: (optional)
        Level of significance when calculating bootstrap CI for ``ci_type =
        boot`` case. E.g. ``0.68`` corresponds to `1 \sigma`. (default:
        ``0.68``)
    :param n_cov: (optional)
        Number of `samples` from infinite population to consider in coverage
        analysis of intervals. Here `samples` - observations of known source
        with different realisations of noise with known parameters. (default:
         ``100``)
    :param n_rms: (optional)
        Number of rms to use in ``ci_type = rms`` case. (default: ``1.``)
    :param stokes: (optional)
        Stokes parameter to use. If ``None`` then use ``I``. (default: ``None``)
    :param boot_cc_fits_paths: (optional)
        If ``ci_type = boot`` then this parameter could specify paths to cleaned
        bootstrapped uv-data.
    :param sample_cc_fits_paths: (optional)
        Path to FITS-files with CLEAN models of `sample` uv-data. If ``None``
        then create ``n_cov`` `sample` uv-data from noise of `original` uv-data
        and `original` CLEAN model. (default: ``None``)

    :return:
        Coverage map. Each pixel contain frequency of times when samples from
        population hit inside CI for given pixel.

    """

    # If not given `original` CLEAN model - get it by cleaning `original`
    # uv-data
    if original_cc_fits_path is None:
        print("No `original` CLEAN model specified! Will CLEAN `original`"
              " uv-data.")
        if imsize is None:
            raise Exception("Specify ``imsize``")
        uv_fits_dir, uv_fits_fname = os.path.split(original_uv_fits_path)
        print("Cleaning `original` uv-data to"
              " {}".format(os.path.join(outdir, 'cc.fits')))
        clean_difmap(uv_fits_fname, 'cc.fits', stokes, imsize, path=uv_fits_dir,
                     path_to_script=path_to_script, outpath=outdir)
        original_cc_fits_path = os.path.join(outdir, 'cc.fits')

    original_uv_data = UVData(original_uv_fits_path)
    noise = original_uv_data.noise()
    original_model = create_model_from_fits_file(original_cc_fits_path)
    # Find images parameters for cleaning if necessary
    if imsize is None:
        print("Getting image parameters from `original`"
              " CLEAN FITS file {}.".format(original_cc_fits_path))
        image_params = get_fits_image_info(original_cc_fits_path)
        imsize = (image_params['imsize'][0],
                  abs(image_params['pixsize'][0]) / mas_to_rad)

    # Substitute uv-data with original model and create `model` uv-data
    print("Substituting original uv-data with CLEAN model...")
    model_uv_data = copy.deepcopy(original_uv_data)
    model_uv_data.substitute([original_model])

    # Add noise to `model` uv-data to get `observed` uv-data
    observed_uv_data = copy.deepcopy(model_uv_data)
    observed_uv_data.noise_add(noise)
    observed_uv_fits_path = os.path.join(outdir, 'observed_uv.uvf')
    if os.path.isfile(observed_uv_fits_path):
        os.unlink(observed_uv_fits_path)
    print("Adding noise to `model` uv-data to get `observed` uv-data...")
    observed_uv_data.save(fname=observed_uv_fits_path)

    observed_cc_fits_path = os.path.join(outdir, 'observed_cc.fits')
    if os.path.isfile(observed_cc_fits_path):
        os.unlink(observed_cc_fits_path)
    # Clean `observed` uv-data to get `observed` image and model
    print("Cleaning `observed` uv-data to `observed` CLEAN model...")
    clean_difmap('observed_uv.uvf', 'observed_cc.fits',
                 original_model.stokes, imsize, path=outdir,
                 path_to_script=path_to_script, outpath=outdir)
    # Get `observed` model and image
    observed_model = create_model_from_fits_file(observed_cc_fits_path)
    observed_image = create_image_from_fits_file(observed_cc_fits_path)

    # Testing coverage of bootstrapped CI
    if ci_type == 'boot':
        # Bootstrap and clean only when necessary
        if boot_cc_fits_paths is None:
            # Bootstrap `observed` uv-data with `observed` model
            boot = CleanBootstrap([observed_model], observed_uv_data)
            cwd = os.getcwd()
            path_to_script = path_to_script or cwd
            os.chdir(outdir)
            print("Bootstrapping uv-data with {} replications".format(n_boot))
            boot.run(outname=['observed_uv_boot', '.uvf'], n=n_boot)
            os.chdir(cwd)

            boot_uv_fits_paths = sorted(glob.glob(os.path.join(outdir,
                                                               'observed_uv_boot*.uvf')))
            # Clean each bootstrapped uv-data
            for i, uv_fits_path in enumerate(boot_uv_fits_paths):
                uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
                print("Cleaning {} bootstrapped observed"
                      " uv-data to {}".format(uv_fits_path,
                                              os.path.join(outdir,
                                                           'observed_cc_boot_{}.fits'.format(i + 1))))
                clean_difmap(uv_fits_fname, 'observed_cc_boot_{}.fits'.format(i + 1),
                             original_model.stokes, imsize, path=uv_fits_dir,
                             path_to_script=path_to_script, outpath=outdir)

            boot_cc_fits_paths = glob.glob(os.path.join(outdir,
                                                        'observed_cc_*.fits'))

        # Calculate bootstrap CI
        # hdi_low, hdi_high = boot_ci_bc(boot_cc_fits_paths,
        #                                observed_cc_fits_path, alpha=alpha)
        hdi_low, hdi_high = boot_ci(boot_cc_fits_paths, observed_cc_fits_path,
                                    alpha=alpha)
    elif ci_type == 'rms':
        # Calculate ``n_rms`` CI
        rms = observed_image.rms(region=(50, 50, 50, None))
        rms = np.sqrt(rms ** 2. + (1.5 * rms ** 2.) ** 2.)
        hdi_low = observed_image.image - rms
        hdi_high = observed_image.image + rms
    else:
        raise Exception("CI intervals must be `boot` or `rms`!")

    # Create `sample` uv-data and clean it only when necessary
    if sample_cc_fits_paths is None:
        # Add noise to `model` uv-data ``n_cov`` times and get ``n_cov``
        # `samples` from population
        sample_uv_fits_paths = list()
        for i in range(n_cov):
            sample_uv_data = copy.deepcopy(model_uv_data)
            sample_uv_data.noise_add(noise)
            sample_uv_fits_path = os.path.join(outdir,
                                               'samle_uv_{}.uvf'.format(i + 1))
            sample_uv_data.save(sample_uv_fits_path)
            sample_uv_fits_paths.append(sample_uv_fits_path)

        # Clean each `sample` FITS-file
        for i, uv_fits_path in enumerate(sample_uv_fits_paths):
            uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
            print("Cleaning {} sample uv-data to"
                  " {}".format(uv_fits_path,
                               os.path.join(outdir,
                                            'sample_cc_{}.fits'.format(i + 1))))
            clean_difmap(uv_fits_fname, 'sample_cc_{}.fits'.format(i + 1),
                         original_model.stokes, imsize, path=uv_fits_dir,
                         path_to_script=path_to_script, outpath=outdir)

        sample_cc_fits_paths = glob.glob(os.path.join(outdir,
                                                      'sample_cc_*.fits'))

    sample_images = list()
    for sample_cc_fits_path in sample_cc_fits_paths:
        image = create_image_from_fits_file(sample_cc_fits_path)
        sample_images.append(image.image)

    # For each pixel check how often flux in `sample` images lies in CI derived
    # for observed image.
    cov_array = np.zeros((imsize[0], imsize[0]), dtype=float)
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(cov_array):
        for image in sample_images:
            cov_array[x, y] += float(np.logical_and(hdi_low[x, y] < image[x, y],
                                                    image[x, y] < hdi_high[x,
                                                                           y]))

    return cov_array / n_cov


def create_sample(original_uv_fits_path, original_cc_fits_path=None,
                  imsize=None, outdir=None, path_to_script=None,
                  n_sample=100, stokes='I'):
    """
    Create `sample` from `true` or `model` source

    :param original_uv_fits_path:
        Path to original FITS-file with uv-data.
    :param original_cc_fits_path: (optional)
        Path to original FITS-file with CC model. If ``None`` then use
        ``imsize`` parameter to get `original` CC model from
        ``original_uv_fits_path``. (default: ``None``)
    :param imsize: (optional)
        Image parameters (image size [pix], pixel size [mas]) to use
        when doing first CC with ``original_cc_fits_path = None``. (default:
        ``None``)
    :param outdir: (optional)
        Directory to store intermediate results. If ``None`` then use CWD.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to Dan Homan's script for final clean. If ``None`` then use CWD.
        (default: ``None``)
    :param n_sample: (optional)
        Number of `samples` from infinite population to consider in coverage
        analysis of intervals. Here `samples` - observations of known source
        with different realisations of noise with known parameters. (default:
         ``100``)
    :param stokes: (optional)
        Stokes parameter to use. If ``None`` then use ``I``. (default: ``None``)

    :return:
        Creates FITS-files with uv-data and CLEAN models of `sample`.
    """

    # If not given `original` CLEAN model - get it by cleaning `original`
    # uv-data
    if original_cc_fits_path is None:
        print("No `original` CLEAN model specified! Will CLEAN `original`"
              " uv-data.")
        if imsize is None:
            raise Exception("Specify ``imsize``")
        uv_fits_dir, uv_fits_fname = os.path.split(original_uv_fits_path)
        original_cc_fits_path = os.path.join(outdir, 'original_cc.fits')
        print("Cleaning `original` uv-data to {}".format(original_cc_fits_path))
        clean_difmap(uv_fits_fname, 'original_cc.fits', stokes, imsize,
                     path=uv_fits_dir, path_to_script=path_to_script,
                     outpath=outdir)

    original_uv_data = UVData(original_uv_fits_path)
    noise = original_uv_data.noise()
    original_model = create_model_from_fits_file(original_cc_fits_path)
    # Find images parameters for cleaning if necessary
    if imsize is None:
        print("Getting image parameters from `original`"
              " CLEAN FITS file {}.".format(original_cc_fits_path))
        image_params = get_fits_image_info(original_cc_fits_path)
        imsize = (image_params['imsize'][0],
                  abs(image_params['pixsize'][0]) / mas_to_rad)

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

    # Clean each `sample` FITS-file
    print("CLEANing `samples` uv-data")
    for uv_fits_path in sample_uv_fits_paths:
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        j = uv_fits_fname.split('.')[0].split('_')[-1]
        print("Cleaning {} sample uv-data to"
              " {}".format(uv_fits_path,
                           os.path.join(outdir,
                                        'sample_cc_{}.fits'.format(j))))
        clean_difmap(uv_fits_fname, 'sample_cc_{}.fits'.format(j),
                     original_model.stokes, imsize, path=uv_fits_dir,
                     path_to_script=path_to_script, outpath=outdir)

    sample_cc_fits_paths = sorted(glob.glob(os.path.join(outdir,
                                                         'sample_cc_*.fits')))
    sample_uv_fits_paths = sorted(glob.glob(os.path.join(outdir,
                                                         'sample_uv_*.uvf')))
    return sample_uv_fits_paths, sample_cc_fits_paths


def create_coverage_map_classic(original_uv_fits_path, ci_type,
                                original_cc_fits_path=None, imsize=None,
                                outdir=None, n_boot=200, path_to_script=None,
                                alpha=0.68, n_cov=100, n_rms=1., stokes='I',
                                sample_cc_fits_paths=None,
                                sample_uv_fits_paths=None):
    """
    Conduct coverage analysis of image pixels flux CI. Find number of times
    when CI of `sample` values contains `true` value.

    :param original_uv_fits_path:
        Path to original FITS-file with uv-data.
    :param ci_type:
        Type of CI to test. ``boot`` or ``rms``. If ``boot`` then use residuals
        bootstrap CI. If ``rms`` then use Hovatta corrected image rms CI.
    :param original_cc_fits_path: (optional)
        Path to original FITS-file with CC model. If ``None`` then use
        ``imsize`` parameter to get `original` CC model from
        ``original_uv_fits_path``. (default: ``None``)
    :param imsize: (optional)
        Image parameters (image size [pix], pixel size [mas]) to use
        when doing first CC with ``original_cc_fits_path = None``. (default:
        ``None``)
    :param outdir: (optional)
        Directory to store intermediate results. If ``None`` then use CWD.
        (default: ``None``)
    :param n_boot: (optional)
        Number of bootstrap replications to use when calculating bootstrap CI
        for ``ci_type = boot`` option when ``boot_cc_fits_paths`` hasn't
        specified. (default: ``200``)
    :param path_to_script: (optional)
        Path to Dan Homan's script for final clean. If ``None`` then use CWD.
        (default: ``None``)
    :param alpha: (optional)
        Level of significance when calculating bootstrap CI for ``ci_type =
        boot`` case. E.g. ``0.68`` corresponds to `1 \sigma`. (default:
        ``0.68``)
    :param n_cov: (optional)
        Number of `samples` from infinite population to consider in coverage
        analysis of intervals. Here `samples` - observations of known source
        with different realisations of noise with known parameters. (default:
         ``100``)
    :param n_rms: (optional)
        Number of rms to use in ``ci_type = rms`` case. (default: ``1.``)
    :param stokes: (optional)
        Stokes parameter to use. If ``None`` then use ``I``. (default: ``None``)
    :param boot_cc_fits_paths: (optional)
        If ``ci_type = boot`` then this parameter could specify paths to cleaned
        bootstrapped uv-data.
    :param sample_uv_fits_paths: (optional)
        Path to FITS-files with `sample` uv-data. If ``None`` then create
        ``n_cov`` `sample` uv-data from noise of `original` uv-data and
        `original` CLEAN model. (default: ``None``)
    :param sample_cc_fits_paths: (optional)
        Path to FITS-files with CLEAN models of `sample` uv-data. If ``None``
        then create ``n_cov`` `sample` uv-data from noise of `original` uv-data
        and `original` CLEAN model. (default: ``None``)

    :return:
        Coverage map. Each pixel contain frequency of times when CI for samples
        from population contain `true` value for given pixel.

    """
    if original_cc_fits_path is None:
        print("No `original` CLEAN model specified! Will CLEAN `original`"
              " uv-data.")
        if imsize is None:
            raise Exception("Specify ``imsize``")
        uv_fits_dir, uv_fits_fname = os.path.split(original_uv_fits_path)
        print("Cleaning `original` uv-data to"
              " {}".format(os.path.join(outdir, 'original_cc.fits')))
        clean_difmap(uv_fits_fname, 'original_cc.fits', stokes, imsize,
                     path=uv_fits_dir, path_to_script=path_to_script,
                     outpath=outdir)
        original_cc_fits_path = os.path.join(outdir, 'original_cc.fits')

    # Find images parameters for cleaning if necessary
    if imsize is None:
        print("Getting image parameters from `original`"
              " CLEAN FITS file {}.".format(original_cc_fits_path))
        image_params = get_fits_image_info(original_cc_fits_path)
        imsize = (image_params['imsize'][0],
                  abs(image_params['pixsize'][0]) / mas_to_rad)

    # This is `true` values. Will check how often they arise in `sample` CIs.
    original_image = create_image_from_fits_file(original_cc_fits_path)
    # If `sample` data doesn't ready - create it!
    if sample_uv_fits_paths is None:
        # Create `sample`
        sample_uv_fits_paths, sample_cc_fits_paths =\
            create_sample(original_uv_fits_path,
                          original_cc_fits_path=original_cc_fits_path,
                          imsize=imsize, outdir=outdir,
                          path_to_script=path_to_script, n_sample=n_cov,
                          stokes=stokes)

    # For each pixel check how often CI of `sample` images contains `model`]
    # values.
    print("Creating coverage array")
    cov_array = np.zeros((imsize[0], imsize[0]), dtype=float)
    # Testing coverage of bootstrapped CI
    # For each `sample` uv-data and model generate bootstrap CI
    print sample_uv_fits_paths
    print sample_cc_fits_paths
    for sample_cc_fits_path, sample_uv_fits_path in zip(sample_cc_fits_paths,
                                                        sample_uv_fits_paths):
        print("Sample : {}".format(sample_cc_fits_path))
        if ci_type == 'boot':
            n__ = sample_uv_fits_path.split('.')[0].split('_')[-1]
            n_ = sample_cc_fits_path.split('.')[0].split('_')[-1]
            assert n_ == n__
            print("Bootstrapping sample uv-data {} with sample model {} using"
                  " {} replications".format(sample_uv_fits_path,
                                            sample_cc_fits_path, n_boot))

            print("Removing old bootstrapped files...")
            boot_cc_fits_paths = glob.glob(os.path.join(outdir,
                                                        'sample_cc_boot_*.fits'))
            for rmfile in boot_cc_fits_paths:
                print("Removing CC file {}".format(rmfile))
                os.unlink(rmfile)
            boot_uv_fits_paths = \
                sorted(glob.glob(os.path.join(outdir, 'sample_uv_boot_*.uvf')))
            for rmfile in boot_uv_fits_paths:
                print("Removing UV file {}".format(rmfile))
                os.unlink(rmfile)

            bootstrap_uv_fits(sample_uv_fits_path, [sample_cc_fits_path],
                              n_boot, outpath=outdir, outname=['sample_uv_boot',
                                                               '.uvf'])

            boot_uv_fits_paths =\
                sorted(glob.glob(os.path.join(outdir, 'sample_uv_boot_*.uvf')))

            # Clean each bootstrapped uv-data
            for i, uv_fits_path in enumerate(boot_uv_fits_paths):
                uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
                j = uv_fits_fname.split('.')[0].split('_')[-1]
                cc_fname = 'sample_cc_boot_{}.fits'.format(j)
                print("Cleaning {} bootstrapped sample"
                      " uv-data to {}".format(uv_fits_path,
                                              os.path.join(outdir, cc_fname)))
                clean_difmap(uv_fits_fname, cc_fname, stokes, imsize,
                             path=uv_fits_dir, path_to_script=path_to_script,
                             outpath=outdir)

            boot_cc_fits_paths = glob.glob(os.path.join(outdir,
                                                        'sample_cc_boot_*.fits'))

            # Calculate bootstrap CI for current `sample`
            # hdi_low, hdi_high = boot_ci_bc(boot_cc_fits_paths,
            #                                observed_cc_fits_path, alpha=alpha)
            hdi_low, hdi_high = boot_ci(boot_cc_fits_paths, sample_cc_fits_path,
                                        alpha=alpha)
        elif ci_type == 'rms':
            # Calculate ``n_rms`` CI
            print("Calculating rms...")
            sample_image = create_image_from_fits_file(sample_cc_fits_path)
            rms = sample_image.rms(region=(25, 25, 25, None))
            rms = np.sqrt(rms ** 2. + (0.0 * rms ** 2.) ** 2.)
            hdi_low = sample_image.image - n_rms * rms
            hdi_high = sample_image.image + n_rms * rms
        else:
            raise Exception("CI intervals must be `boot` or `rms`!")

        # Check if `model` value falls in current `sample` CI
        print("Calculating hits of `true` values for"
              " {}".format(sample_cc_fits_path))
        for (x, y), value in np.ndenumerate(cov_array):
            cov_array[x, y] += float(np.logical_and(hdi_low[x, y] < original_image.image[x, y],
                                                    original_image.image[x, y] < hdi_high[x, y]))

    return cov_array / n_cov


if __name__ == '__main__':
    base_dir = '/home/ilya/code/vlbi_errors/examples/L/'
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    from mojave import (download_mojave_uv_fits, mojave_uv_fits_fname)
    source = '1038+064'
    epochs = ['2010_05_21']
    bands = ['l18']
    # download_mojave_uv_fits(source, epochs=epochs, bands=bands,
    #                          download_dir=base_dir)

    # # Working
    # original_uv_fits_path = os.path.join(base_dir, '2230+114.x.2006_02_12.uvf')
    # original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    # sample_cc_fits_paths = glob.glob(os.path.join(base_dir, 'sample_cc_*.fits'))
    # boot_cc_fits_paths = glob.glob(os.path.join(base_dir, 'observed_cc_*.fits'))
    # coverage_map = create_coverage_map(original_uv_fits_path, ci_type='rms',
    #                                    original_cc_fits_path=original_cc_fits_path,
    #                                    outdir=base_dir,
    #                                    sample_cc_fits_paths=sample_cc_fits_paths,
    #                                    boot_cc_fits_paths=boot_cc_fits_paths,
    #                                    path_to_script=path_to_script,
    #                                    n_cov=100, n_rms=2., stokes='I',
    #                                    n_boot=200,
    #                                    alpha=0.95)

    # Classic coverage analysis
    fname = mojave_uv_fits_fname(source, bands[0], epochs[0])
    original_uv_fits_path = os.path.join(base_dir, fname)
    # sample_uv_fits_paths, sample_cc_fits_paths =\
    #     create_sample(original_uv_fits_path, imsize=(256, 1.), outdir=base_dir,
    #                   path_to_script=path_to_script)
    sample_cc_fits_paths = sorted(glob.glob(os.path.join(base_dir,
                                                         'sample_cc_*.fits')))
    # sample_cc_fits_paths = None
    sample_uv_fits_paths = sorted(glob.glob(os.path.join(base_dir,
                                                         'sample_uv_*.uvf')))
    # sample_uv_fits_paths = None
    original_cc_fits_path = os.path.join(base_dir, 'original_cc.fits')
    # original_cc_fits_path = None
    coverage_map =\
        create_coverage_map_classic(original_uv_fits_path, ci_type='rms',
                                    original_cc_fits_path=original_cc_fits_path,
                                    imsize=(256, 1.), outdir=base_dir,
                                    path_to_script=path_to_script,
                                    sample_cc_fits_paths=sample_cc_fits_paths,
                                    sample_uv_fits_paths=sample_uv_fits_paths,
                                    n_rms=3., alpha=0.95, n_boot=100)
