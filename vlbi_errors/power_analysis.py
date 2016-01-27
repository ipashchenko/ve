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
                       cred_mass=0.95):
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
    images = list()
    for boot_cc_fits_path in boot_cc_fits_paths:
        print("Reading image from {}".format(boot_cc_fits_path))
        image = create_image_from_fits_file(boot_cc_fits_path)
        images.append(image.image)

    images_cube = np.dstack(images)
    hdi_low = np.zeros(np.shape(images_cube[:, :, 0]))
    hdi_high = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(hdi_low):
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=cred_mass)
        hdi_low[x, y] = hdi[0]
        hdi_high[x, y] = hdi[1]

    # Original image
    image = create_image_from_fits_file(cc_fits_path)
    coverage_map = np.logical_and(hdi_low < image.image, image.image < hdi_high)
    coverage = np.count_nonzero(coverage_map) / float(coverage_map.size)

    return hdi_low, hdi_high, coverage, coverage_map


def cov_analysis_image_boot(boot_cc_fits_paths, original_cc_fits_path,
                            cred_mass=0.65):
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

    # Original image
    original_image = create_image_from_fits_file(original_cc_fits_path)
    hdi_low = original_image.image - boot_ci
    hdi_high = original_image.image + boot_ci

    coverages = list()
    for boot_image in boot_images:
        coverage_map = np.logical_and(hdi_low < boot_image,
                                      boot_image < hdi_high)
        coverage = np.count_nonzero(coverage_map) / float(coverage_map.size)
        print("Coverage = {}".format(coverage))
        coverages.append(coverage)

    return boot_ci, coverages


def cov_analysis_image_old(cc_fits_dir, cc_glob='cc_*.fits',
                           original_cc_file='cc.fits'):
    original_image = create_image_from_fits_file(os.path.join(cc_fits_dir,
                                                              original_cc_file))
    cc_fits_paths = glob.glob(os.path.join(cc_fits_dir, cc_glob))
    coverages = list()
    for cc_fits_path in cc_fits_paths:
        print("Checking {}".format(cc_fits_path))
        image = create_image_from_fits_file(cc_fits_path)
        rms = image.rms(region=(50, 50, 50, None))
        # rms = np.sqrt(rms ** 2. + (1.5 * rms) ** 2.)
        hdi_low = image.image - rms
        hdi_high = image.image + rms
        coverage_map = np.logical_and(hdi_low < original_image.image,
                                      original_image.image < hdi_high)
        coverage = np.count_nonzero(coverage_map) / float(coverage_map.size)
        print("Coverage = {}".format(coverage))
        coverages.append(coverage)
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

    # coverages = cov_analysis_image_old(base_dir)
    # print coverages, np.mean(coverages)


    boot_cc_fits_paths = glob.glob(os.path.join(base_dir, 'cc_*.fits'))
    original_cc_fits_path = os.path.join(base_dir, 'cc.fits')
    boot_ci, coverages = cov_analysis_image_boot(boot_cc_fits_paths,
                                                 original_cc_fits_path)
    print coverages, np.mean(coverages)
