import copy
import os
import glob
import numpy as np
from spydiff import clean_difmap
from uv_data import UVData
from from_fits import (create_model_from_fits_file, create_image_from_fits_file)
from bootstrap import CleanBootstrap
from utils import (hdi_of_mcmc, get_fits_image_info, mas_to_rad, bc_endpoint)


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
    Conduct coverage analysis of image pixels flux CI.

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
        Stokes parameter to use. (default: ``None``)
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
        print("Cleaning original uv-data to {}".format(os.path.join(outdir,
                                                                    'cc.fits')))
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
                                                           'observed_cc_{}.fits'.format(i + 1))))
                clean_difmap(uv_fits_fname, 'observed_cc_{}.fits'.format(i + 1),
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


if __name__ == '__main__':
    base_dir = '/home/ilya/code/vlbi_errors/examples/'
    # from mojave import download_mojave_uv_fits
    # source = '2230+114'
    # epochs = ['2006_02_12']
    # bands = ['x']
    # download_mojave_uv_fits(source, epochs=epochs, bands=bands,
    #                         download_dir=base_dir)

    original_uv_fits_path = os.path.join(base_dir, '2230+114.x.2006_02_12.uvf')
    original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    sample_cc_fits_paths = glob.glob(os.path.join(base_dir, 'sample_cc_*.fits'))
    boot_cc_fits_paths = glob.glob(os.path.join(base_dir, 'observed_cc_*.fits'))
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    coverage_map = create_coverage_map(original_uv_fits_path, ci_type='rms',
                                       original_cc_fits_path=original_cc_fits_path,
                                       outdir=base_dir,
                                       sample_cc_fits_paths=sample_cc_fits_paths,
                                       boot_cc_fits_paths=boot_cc_fits_paths,
                                       path_to_script=path_to_script,
                                       n_cov=100, n_rms=2., stokes='I',
                                       n_boot=200,
                                       alpha=0.95)
