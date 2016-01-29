import copy
import os
import glob
import numpy as np
from spydiff import clean_difmap
from uv_data import UVData
from from_fits import (create_model_from_fits_file, create_image_from_fits_file)
from bootstrap import CleanBootstrap
from utils import (hdi_of_mcmc, get_fits_image_info, mas_to_rad)


def cov_analysis_image(uv_fits_path, n_boot, cc_fits_path=None, imsize=None,
                       path_to_script=None, stokes='I', outdir=None,
                       cred_mass=0.95, mask=False, nmask=1.):
    """
    Function that runs coverage analysis of bootstrap CIs using
    user-specified FITS-file with uv-data and optional CLEAN model.

    :param uv_fits_path:
        Path to FITS-file with uv-data.
    :param n_boot:
        Number of bootstrap replications to use when calculating CIs.
    :param cc_fits_path: (optional)
        Path to FITS-file with CLEAN models. This models will
        be used as model for power analysis. If ``None`` then CLEAN uv-data
        first and use result as real model for calculating coverage.
        (default: ``None``)
    :param imsize: (optional)
        Image parameters (image size [pix], pixel size [mas]) to use
        when doing first CLEAN with ``cc_fits_path = None``.
    :param path_to_script: (optional)
        Path to directory with ``difmap`` final CLEANing script. If ``None``
        then CWD. (default: ``None``)
    :param stokes: (optional)
        Stokes parameter to deal with. (default: ``I``)
    :param outdir: (optional)
        Directory to store files. If ``None`` then use CWD. (default:
        ``None``)
    :param cred_mass:  (optional)
        Credibility mass of CI to check.
    :param mask: (optional)
        Use mask? (default: ``False``)
    :param nmask: (optional)
        If using mask then how many std to mask? ``1`` - ``65%``, ``2`` -
        ``95%``, ``3`` - ``99%``. (default: ``1``)

    """
    if cc_fits_path is None:
        if imsize is None:
            raise Exception("Specify ``imszie``")
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        print("Cleaning original uv-data to {}".format(os.path.join(outdir,
                                                                    'cc.fits')))
        clean_difmap(uv_fits_fname, 'cc.fits', stokes, imsize, path=uv_fits_dir,
                     path_to_script=path_to_script, outpath=outdir)
        cc_fits_path = os.path.join(outdir, 'cc.fits')

    uvdata = UVData(uv_fits_path)
    ccmodel = create_model_from_fits_file(cc_fits_path)
    bt = CleanBootstrap([ccmodel], uvdata)
    cwd = os.getcwd()
    os.chdir(outdir)
    print("Bootstrapping uv-data with {} replications".format(n_boot))
    bt.run(outname=['uv_boot', '.uvf'], n=n_boot)
    os.chdir(cwd)

    if imsize is None:
        image_params = get_fits_image_info(cc_fits_path)
        imsize = (image_params['imsize'][0],
                  abs(image_params['pixsize'][0]) / mas_to_rad)

    uv_fits_paths = sorted(glob.glob(os.path.join(outdir, 'uv_boot*.uvf')))
    for i, uv_fits_path in enumerate(uv_fits_paths):
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        print("Cleaning {} bootstrapped"
              " uv-data to {}".format(uv_fits_path,
                                      os.path.join(outdir,
                                                   'cc_{}.fits'.format(i + 1))))
        clean_difmap(uv_fits_fname, 'cc_{}.fits'.format(i + 1), stokes, imsize,
                     path=uv_fits_dir, path_to_script=path_to_script,
                     outpath=outdir)

    boot_cc_fits_paths = glob.glob(os.path.join(outdir, 'cc_*.fits'))
    boot_ci, coverages = cov_analysis_image_boot(boot_cc_fits_paths,
                                                 cc_fits_path,
                                                 cred_mass=cred_mass,
                                                 base_dir=outdir)
    return boot_ci, coverages


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


def coverage_map_boot(original_uv_fits_path, ci_type,
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
        image_params = get_fits_image_info(original_cc_fits_path)
        imsize = (image_params['imsize'][0],
                  abs(image_params['pixsize'][0]) / mas_to_rad)

    # Substitute uv-data with original model and create `model` uv-data
    model_uv_data = copy.deepcopy(original_uv_data)
    model_uv_data.substitute([original_model])

    # Add noise to `model` uv-data to get `observed` uv-data
    observed_uv_data = copy.deepcopy(model_uv_data)
    observed_uv_data.noise_add(noise)
    observed_uv_fits_path = os.path.join(outdir, 'observed_uv.uvf')
    observed_uv_data.save(observed_uv_fits_path)

    # Clean `observed` uv-data to get `observed` image and model
    clean_difmap('observed_uv.uvf', 'observed_cc.fits',
                 original_model.stokes, imsize, path=outdir,
                 path_to_script=path_to_script, outpath=outdir)
    # Get `observed` model and image
    observed_cc_fits_path = os.path.join(outdir, 'observed_cc.fits')
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
            cov_array[x, y] += float(np.logical_and(hdi_low < image,
                                                    image < hdi_high))

    return cov_array / n_cov


def cov_analysis_image_boot(boot_cc_fits_paths, original_cc_fits_path,
                            cred_mass=0.68,
                            base_dir='/home/ilya/code/vlbi_errors/examples',
                            mask=None, nmask=1.):
    # Original image
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
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=cred_mass)
        boot_ci[x, y] = hdi[1] - hdi[0]

    hdi_low = original_image.image - boot_ci / 2.
    hdi_high = original_image.image + boot_ci / 2.

    if mask:
        print("Using mask with {} RMS".format(nmask))
        rms = original_image.rms(region=(50, 50, 50, None))
        mask = original_image.image < nmask * rms
    else:
        mask = None

    coverages = list()
    for boot_image in boot_images:
        coverage_map = np.logical_and(hdi_low < boot_image,
                                      boot_image < hdi_high)
        coverage_map = np.ma.array(coverage_map, mask=mask)
        print("Shape of coverage map - {}".format(coverage_map.shape))
        print("Size of coverage map - {}".format(coverage_map.size))
        print("Number of non-masked elements in coverage map -"
              " {}".format(np.ma.count(coverage_map)))
        print("Number of covered (nonzero) non-masked elements -"
              " {}".format(len(coverage_map.nonzero()[0])))
        coverage = len(coverage_map.nonzero()[0]) /\
                   float(np.ma.count(coverage_map))
        print("Coverage = {}".format(coverage))
        coverages.append(coverage)
    save_file = os.path.join(base_dir, 'boot_ci_{}.txt'.format(cred_mass))
    print("Saving bootstrap CI to {}".format(save_file))
    np.savetxt(save_file, boot_ci)

    return boot_ci, coverages


def cov_analysis_image_old(cc_fits_dir, cc_glob='cc_*.fits',
                           original_cc_file='cc.fits', mask=True, nmask=1.):
    original_image = create_image_from_fits_file(os.path.join(cc_fits_dir,
                                                              original_cc_file))
    if mask:
        print("Using mask with {} RMS".format(nmask))
        rms = original_image.rms(region=(50, 50, 50, None))
        mask = original_image.image < nmask * rms
    else:
        mask = None

    cc_fits_paths = glob.glob(os.path.join(cc_fits_dir, cc_glob))
    coverages = list()
    for cc_fits_path in cc_fits_paths:
        print("Checking {}".format(cc_fits_path))
        image = create_image_from_fits_file(cc_fits_path)
        rms = image.rms(region=(50, 50, 50, None))
        print("RMS = {}".format(rms))
        rms = np.sqrt(rms ** 2. + (1.5 * rms) ** 2.)
        hdi_low = image.image - rms
        hdi_high = image.image + rms
        coverage_map = np.logical_and(hdi_low < original_image.image,
                                      original_image.image < hdi_high)
        coverage_map = np.ma.array(coverage_map, mask=mask)
        print("Shape of coverage map - {}".format(coverage_map.shape))
        print("Size of coverage map - {}".format(coverage_map.size))
        print("Number of non-masked elements in coverage map - {}".format(np.ma.count(coverage_map)))
        print("Number of covered (nonzero) non-masked elements -"
              " {}".format(len(coverage_map.nonzero()[0])))
        coverage = len(coverage_map.nonzero()[0]) / float(np.ma.count(coverage_map))
        print("Coverage = {}".format(coverage))
        coverages.append(coverage)
    return coverages


def calculate_coverage_boot(boot_cc_fits_paths, original_cc_fits_path,
                            nmasks=None, cred_mass=0.95):
    coverages = list()
    for nmask in nmasks:
        if nmask:
            nmask = nmask
            mask = True
        else:
            mask = False
        boot_ci, coverage = cov_analysis_image_boot(boot_cc_fits_paths,
                                                    original_cc_fits_path,
                                                    mask=mask, nmask=nmask,
                                                    cred_mass=cred_mass)
        coverages.append((np.mean(coverage), np.std(coverage)))
    return coverages


def calculate_coverage_old(cc_fits_dir, cc_glob='cc_*.fits',
                           original_cc_file='cc.fits', nmasks=None):
    coverages = list()
    for nmask in nmasks:
        if nmask:
            nmask = nmask
            mask = True
        else:
            mask = False
        coverage = cov_analysis_image_old(cc_fits_dir, cc_glob='cc_*.fits',
                                          original_cc_file='cc.fits',
                                          mask=mask, nmask=nmask)
        coverages.append((np.mean(coverage), np.std(coverage)))
    return coverages


if __name__ == '__main__':
    base_dir = '/home/ilya/code/vlbi_errors/examples/'
    # source = '2230+114'
    # epochs = ['2006_02_12']
    # bands = ['x']
    # from mojave import download_mojave_uv_fits
    # download_mojave_uv_fits(source, epochs=epochs, bands=bands,
    #                         download_dir=base_dir)
    # uv_fits_path = os.path.join(base_dir, '2230+114.x.2006_02_12.uvf')
    # # cc_fits_path = os.path.join(base_dir, 'cc.fits')
    # cc_fits_path = None
    # path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    # n_boot = 200
    # imsize = (1024, 0.1)
    # outdir = base_dir
    # hdi_low, hdi_high, coverage, coverage_map =\
    #     cov_analysis_image(uv_fits_path, n_boot, cc_fits_path=cc_fits_path,
    #                        path_to_script=path_to_script, outdir=base_dir,
    #                        cred_mass=0.65, imsize=imsize)
    # np.savetxt(os.path.join(base_dir, 'hdi_low_65.txt'), hdi_low)
    # np.savetxt(os.path.join(base_dir, 'hdi_high_65.txt'), hdi_high)
    # np.savetxt(os.path.join(base_dir, 'coverage_map_65.txt'), coverage_map)
    # print coverage

    # coverages = cov_analysis_image_old(base_dir, mask=True, nmask=85.)
    # # 0.82 for mask=1, n_non_masked = 100899
    # # 0.62 for mask=2, n_non_masked = 26773
    # # 0.57 for mask=3, n_non_masked = 17730
    # # 0.56 for mask=4, n_non_masked = 14505
    # # 0.56 for mask=5, n_non_masked = 13326
    # # 0.50 for mask=15, n_non_masked = 9793
    # # 0.47 for mask=25, n_non_masked = 7966
    # # 0.47 for mask=35, n_non_masked = 6347
    # # 0.42 for mask=55, n_non_masked = 4769
    # # 0.38 for mask=85, n_non_masked = 3582
    # # 0.94 for mask=False, n_non_masked = 1048576 (full)
    # print coverages, np.mean(coverages)


    # boot_cc_fits_paths = glob.glob(os.path.join(base_dir, 'cc_*.fits'))
    # original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    # boot_ci, coverages = cov_analysis_image_boot(boot_cc_fits_paths,
    #                                              original_cc_fits_path,
    #                                              mask=True, nmask=85.)
    # # 0.84 for mask=1, n_non_masked = 100899
    # # 0.85 for mask=2, n_non_masked = 26773
    # # 0.85 for mask=3, n_non_masked = 17730
    # # 0.85 for mask=4, n_non_masked = 14505
    # # 0.85 for mask=5, n_non_masked = 13326
    # # 0.85 for mask=15, n_non_masked = 9793
    # # 0.85 for mask=25, n_non_masked = 7966
    # # 0.85 for mask=35, n_non_masked = 6347
    # # 0.84 for mask=55, n_non_masked = 4769
    # # 0.84 for mask=85, n_non_masked = 3582
    # # 0.84 for mask=False, n_non_masked = 1048576 (full)
    # print coverages, np.mean(coverages)

    # boot_cc_fits_paths = glob.glob(os.path.join(base_dir, 'cc_*.fits'))
    # original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    # boot_ci, coverages = cov_analysis_image_boot(boot_cc_fits_paths,
    #                                              original_cc_fits_path,
    #                                              mask=False, nmask=1.,
    #                                              cred_mass=0.95)
    # # 0.4 for mask=1, n_non_masked = 100899
    # # 0.85 for mask=2, n_non_masked = 26773
    # # 0.85 for mask=3, n_non_masked = 17730
    # # 0.85 for mask=4, n_non_masked = 14505
    # # 0.85 for mask=5, n_non_masked = 13326
    # # 0.85 for mask=15, n_non_masked = 9793
    # # 0.85 for mask=25, n_non_masked = 7966
    # # 0.85 for mask=35, n_non_masked = 6347
    # # 0.84 for mask=55, n_non_masked = 4769
    # # 0.84 for mask=85, n_non_masked = 3582
    # # 0.91 for mask=False, n_non_masked = 1048576 (full)
    # print coverages, np.mean(coverages)

    # boot_cc_fits_paths = glob.glob(os.path.join(base_dir, 'cc_*.fits'))
    # original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    # nmasks = [0., 1., 2., 3., 4., 5., 15., 25., 35., 55., 85.]
    # coverages = calculate_coverage_boot(boot_cc_fits_paths,
    #                                     original_cc_fits_path,
    #                                     nmasks=nmasks, cred_mass=0.68)
    # print coverages

    # nmasks = [0., 1., 2., 3., 4., 5., 15., 25., 35., 55., 85.]
    # coverages = calculate_coverage_old(base_dir, cc_glob='cc_*.fits',
    #                                    original_cc_file='cc.fits',
    #                                    nmasks=nmasks)
    # print coverages

    original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    original_uv_fits_path = os.path.join(base_dir, '2230+114.x.2006_02_12.uvf')
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    coverage_map = coverage_map_boot(original_cc_fits_path, ci_type='rms',
                                     outdir=base_dir,
                                     path_to_script=path_to_script, n_cov=100,
                                     n_rms=1.)
