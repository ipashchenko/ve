#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import json
import numpy as np
from from_fits import (create_model_from_fits_file,
                       create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from uv_data import UVData
from bootstrap import CleanBootstrap
from utils import (mas_to_rad, degree_to_rad, get_fits_image_info)
from spydiff import clean_difmap
from images import Images
from image_ops import rms_image, pol_mask, analyze_rotm_slice


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


def clean_uv_fits(uv_fits_path, out_fits_path, stokes, beam=None,
                  mapsize_clean=None,
                  mapsize_fits_path=None, pixsize_fits_path=None,
                  pixel_per_beam=None, mapsize=None,
                  beamsize_fits_path=None, mapsize_restore=None,
                  path_to_script=None, shift=None):
    """
    Clean uv-data fits files with image and clean parameters that are chosen in
    several possible ways.

    :param uv_fits_path:
        Path to uv-data fits-file.
    :param out_fits_path:
        Path to output fits-file with CLEAN image.
    :param stokes:
        Iterable of stokes parameters.
    :param beam: (optional)
        Beam parameter for cleaning (bmaj, bmin, bpa). If ``None`` then use
        naitive beam from uv-data and difmap cleaning script. (default:
        ``None``)
    :param pixel_per_beam: (optional)
        Number of pixels per beam. Used for alternative way to choose pixel
        size. If ``None`` then don't use this option for choice.
        (default: ``None``)
    :param mapsize_clean: (optional)
        Parameters of map for cleaning (map size, pixel size). If ``None``
        then use those of map in ``map_fits_path`` image fits file.
        (default: ``None``)
    :param mapsize_fits_path: (optional)
        Path to fits file with image which physical size will be used.
        If ``None`` then don't choose image size in this way. (default:
        ``None``)
    :param pixsize_fits_path: (optional)
        Path to fits file with image which pixel size will be used.
        If ``None`` then don't choose pixel size in this way. (default:
        ``None``)
    :param beamsize_fits_path: (optional)
        Path to fits file with image which beam parameters will be used.
        If ``None`` then don't choose beam parameters in this way. (default:
        ``None``)
    :param mapsize: (optional)
        Size of map to use when only beam and pixel information is supplied (via
        ``beam``, ``pixel_per_beam`` arguments). If ``None`` then don't specify
        map size with this option.
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use ``mapsize_clean``. (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param shift: (optional)
        Shift to apply. (mas, mas). If ``None`` then don't apply shift.
        (default: ``None``)

    :note:
        Image, pixel & beam specification uses this sequence. If ``beam`` is not
        supplied and ``beamsize_fits_path`` image is not supplied then beam
        parameters native to uv-data and difmap cleaning script are used. Image
        size information comes from this sequence: ``mapsize_clean``,
        ``mapsize_fits_path``, ``mapsize``. If ``pixel_per_beam``
        is used with any of arguments that supplied image size, then image size
        is altered to keep physical image size the same. Pixel is chosen in
        this sequence: ``pixel_per_beam``, ``pixsize_fits_path``,
        ``mapsize_clean``.

    """
    stokes = list(stokes)
    stokes = [stoke.upper() for stoke in stokes]
    # Now ``I`` goes first
    stokes.sort()

    curdir = os.getcwd()

    # Choosing beam
    beam_pars = None
    if beam is not None:
        beam_pars = beam
    if beam_pars is None and beamsize_fits_path is not None:
        map_info = get_fits_image_info(beamsize_fits_path)
        beam_pars = (map_info['bmaj'] / mas_to_rad,
                     map_info['bmin'] / mas_to_rad,
                     map_info['bpa'] / degree_to_rad)

    # Choosing image parameters
    map_pars = None
    if mapsize_clean is not None:
        map_pars = mapsize_clean
    if map_pars is None and mapsize_fits_path is not None:
        map_info = get_fits_image_info(mapsize_fits_path)
        map_pars = (map_info['imsize'][0],
                    abs(map_info['pixsize'][0]) / mas_to_rad)

    # Choosing pixel parameters
    pixsize = None
    if pixel_per_beam is not None:
        pixsize = beam_pars[0] / pixel_per_beam

    if pixsize is None and pixsize_fits_path is not None:
        map_info = get_fits_image_info(pixsize_fits_path)
        pixsize = abs(map_info['pixsize'][0]) / mas_to_rad
    # Correcting image size when pixel size has changed
    imsize = map_pars[0] * abs(map_pars[1]) / abs(pixsize)
    print(imsize)
    powers = np.array([float(imsize) / (2 ** i) for i in range(15)])
    print(powers)
    imsize = 2**(list(np.array(powers <= 1, dtype=int)).index(1))

    map_pars = (imsize, pixsize)

    print "Selected image and pixel size: {}".format(map_pars)

    for stoke in stokes:
        print "Cleaning stokes {}", stoke
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        out_fits_dir, out_fits_fname = os.path.split(out_fits_path)
        print("Cleaning {} to {} stokes {}, mapsize_clean {}, beam_restore"
              " {} with shift {}".format(uv_fits_fname,
                                         os.path.join(out_fits_path,
                                                      out_fits_fname),
                                         stokes, map_pars, beam_pars, shift))
        #clean_difmap(fname=uv_fits_fname,
        #             outfname=out_fits_fname,
        #             stokes=stoke, mapsize_clean=map_pars,
        #             path=uv_fits_dir,
        #             path_to_script=path_to_script,
        #             beam_restore=beam_pars,
        #             outpath=out_fits_dir)
    os.chdir(curdir)


# FIXME: This finds only dr that minimize std for shift - radius dependence
# TODO: use iterables of shifts and sizes as arguments. UNIX-way:)
def find_shift(image1, image2, max_shift, shift_step, min_shift=0,
               max_mask_r=None, mask_step=5):
    """
    Find shift between two images using our heruistic.

    :param image1:
        Instance of ``BasicImage`` class.
    :param image2:
        Instance of ``BasicImage`` class.
    :param max_shift:
        Maximum size of shift to check [pxl].
    :param shift_step:
        size of shift changes step [pxl].
    :param min_shift: (optional)
        Minimum size of shift to check [pxl]. (default: ``0``)
    :param max_mask_r: (optional)
        Maximum size of mask to apply. If ``None`` then use maximum possible.
        (default: ``None``)
    :param mask_step: (optional)
        Size of mask size changes step. (default: ``5``)
    :return:
        Array of shifts.
    """
    shift_dict = dict()

    # Iterating over difference of mask sizes
    for dr in range(min_shift, max_shift, shift_step):
        shift_dict[dr] = list()

        # Iterating over mask sizes
        for r in range(0, max_mask_r, mask_step):
            r1 = r
            r2 = r + dr
            shift = image1.cross_correlate(image2,
                                           region1=(image1.x_c, image1.y_c, r1,
                                                    None),
                                           region2=(image2.x_c, image2.y_c, r2,
                                                    None))
            shift_dict[dr].append(shift)

    for key, value in shift_dict.items():
        value = np.array(value)
        shifts = np.sqrt(value[:, 0] ** 2. + value[:, 1] ** 2.)
        shift_dict.update({key: np.std(shifts)})

    # Searching for mask size difference that has minimal std in shifts
    # calculated for different mask sizes
    return sorted(shift_dict, key=lambda _: shift_dict[_])[0]


def analyze_source(uv_fits_paths, n_boot, imsizes=None, common_imsize=None,
                   common_beam=None, find_shifts=False, outdir=None,
                   path_to_script=None, clear_difmap_logs=True,
                   rotm_slices=None):
    """
    Function that uses multifrequency self-calibration data for in-depth
    analysis.

    :param uv_fits_paths:
        Iterable of paths to self-calibrated uv-data FITS-files.
    :param n_boot:
        Number of bootstrap replications to use in analysis.
    :param imsizes: (optional)
        Iterable of image parameters (imsize, pixsize) that should be used for
        CLEANing of uv-data if no CLEAN-images are supplied. Should be sorted in
        increasing frequency order. If ``None`` then specify parameters by CLEAN
        images. (default: ``None``)
    :param common_imsize: (optional)
        Image parameters that will be used in making common size images for
        multifrequency analysis. If ``None`` then use physical image size of
        lowest frequency and pixel size of highest frequency. (default:
        ``None``)
    :param outdir: (optional)
        Output directory. This directory will be used for saving picture, data,
        etc. If ``None`` then use CWD. (default: ``None``)
    :param path_to_script: (optional)
        Path ot difmap CLEAN script. If ``None`` then use CWD. (default:
        ``None``)

    :notes:
        Workflow:
        1) Чистка с родным разрешением всех N диапазонов и получение родных
        моделей I, Q, U.
        2) Выбор общей ДН из N возможных
        3) (Опционально) Выбор uv-tapering
        4) Чистка uv-данных для всех диапазонов с (опционально применяемым
            uv-tapering) общей ДН
        5) Оценка сдвига ядра
        6) Создание B наборов из N многочастотных симулированных данных
            используя родные модели
        7) (Опционально) Чистка B наборов из N многочастотных симданных с
            родным разрешением для получения карт ошибок I для каждой из N
            частот
        8) Чистка B наборов из N многочастотных симданных для всех диапазонов с
            (опционально применяемым uv-tapering) общей ДН
        9) Оценка ошибки определения сдвига ядра
        10) Оценка RM и ее ошибки
        11) Оценка alpha и ее ошибки

    """

    # Fail early
    if imsizes is None:
        raise Exception("Provide imsizes argument!")
    if common_imsize is not None:
        print("Using common image size {}".format(common_imsize))
    else:
        raise Exception("Provide common_imsize argument!")

    # Setting up the output directory
    if outdir is None:
        outdir = os.getcwd()
    print("Using output directory {}".format(outdir))
    os.chdir(outdir)

    # Assume input self-calibrated uv-data FITS files have different frequencies
    n_freq = len(uv_fits_paths)
    print("Using {} frequencies".format(n_freq))

    # Assuming full multifrequency analysis
    stokes = ('I', 'Q', 'U')

    # Container for original self-calibrated uv-data
    uv_data_dict = dict()
    # Container for original self-calibrated uv-data FITS-file paths
    uv_fits_dict = dict()
    for uv_fits_path in uv_fits_paths:
        uvdata = UVData(uv_fits_path)
        # Mark frequencies by total band center [Hz] for consistency with image.
        uv_data_dict.update({uvdata.band_center: uvdata})
        uv_fits_dict.update({uvdata.band_center: uv_fits_path})

    # Lowest frequency goes first
    freqs = sorted(uv_fits_dict.keys())
    print("Frequencies are: {}".format(freqs))
    # Assert we have original map parameters for all frequencies
    assert len(imsizes) == n_freq

    # Container for original CLEAN-images of self-calibrated uv-data
    cc_image_dict = dict()
    # Container for paths to FITS-files with original CLEAN-images of
    # self-calibrated uv-data
    cc_fits_dict = dict()
    # Container for original CLEAN-image's beam parameters
    cc_beam_dict = dict()
    for freq in freqs:
        cc_image_dict.update({freq: dict()})
        cc_fits_dict.update({freq: dict()})
        cc_beam_dict.update({freq: dict()})

    # 1.
    # Clean original uv-data with specified map parameters
    print("1. Clean original uv-data with specified map parameters...")
    imsizes_dict = dict()
    for i, freq in enumerate(freqs):
        imsizes_dict.update({freq: imsizes[i]})
    for freq in freqs:
        uv_fits_path = uv_fits_dict[freq]
        uv_dir, uv_fname = os.path.split(uv_fits_path)
        for stoke in stokes:
            outfname = '{}_{}_cc.fits'.format(freq, stoke)
            outpath = os.path.join(outdir, outfname)
            clean_difmap(uv_fname, outfname, stoke, imsizes_dict[freq],
                         path=uv_dir, path_to_script=path_to_script,
                         outpath=outdir)
            cc_fits_dict[freq].update({stoke: os.path.join(outdir,
                                                           outfname)})
            image = create_clean_image_from_fits_file(outpath)
            cc_image_dict[freq].update({stoke: image})
            if stoke == 'I':
                cc_beam_dict.update({freq: image.beam})

    # Containers for images and paths to FITS files with common size images
    cc_cs_image_dict = dict()
    cc_cs_fits_dict = dict()
    # 2.
    # Choose common beam size
    print("2. Choosing common beam size...")
    if common_beam is None:
        common_beam = cc_beam_dict[freqs[0]]
    print("Using common beam [mas, mas, deg] : {}".format(common_beam))

    # 3.
    # Optionally uv-tapering uv-data
    print("3. Optionally uv-tapering uv-data...")
    print("skipping...")

    # 4.
    # Clean original uv-data with common map parameters
    print("4. Clean original uv-data with common map parameters...")
    for freq in freqs:
        cc_cs_image_dict.update({freq: dict()})
        cc_cs_fits_dict.update({freq: dict()})

        uv_fits_path = uv_fits_dict[freq]
        uv_dir, uv_fname = os.path.split(uv_fits_path)
        for stoke in stokes:
            outfname = 'cs_{}_{}_cc.fits'.format(freq, stoke)
            outpath = os.path.join(outdir, outfname)
            # clean_difmap(uv_fname, outfname, stoke, common_imsize,
            #              path=uv_dir, path_to_script=path_to_script,
            #              outpath=outdir, show_difmap_output=False)
            cc_cs_fits_dict[freq].update({stoke: os.path.join(outdir,
                                                              outfname)})
            image = create_image_from_fits_file(outpath)
            cc_cs_image_dict[freq].update({stoke: image})

    # 5.
    # Optionally find shifts between original CLEAN-images
    print("5. Optionally find shifts between original CLEAN-images...")
    if find_shifts:
        print("Determining images shift...")
        shift_dict = dict()
        freq_1 = freqs[0]
        image_1 = cc_image_dict[freq_1]['I']

        for freq_2 in freqs[1:]:
            image_2 = cc_image_dict[freq_2]['I']
            # Coarse grid of possible shifts
            shift = find_shift(image_1, image_2, 100, 5, max_mask_r=200,
                               mask_step=5)
            # More accurate grid of possible shifts
            print("Using fine grid for accurate estimate")
            coarse_grid = range(0, 100, 5)
            idx = coarse_grid.index(shift)
            if idx > 0:
                min_shift = coarse_grid[idx - 1]
            else:
                min_shift = 0
            shift = find_shift(image_1, image_2, coarse_grid[idx + 1], 1,
                               min_shift=min_shift, max_mask_r=200,
                               mask_step=5)

            shift_dict.update({str((freq_1, freq_2,)): shift})

        # Dumping shifts to json file in target directory
        with open(os.path.join(outdir, "shifts_original.json"), 'w') as fp:
            json.dump(shift_dict, fp)
    else:
        print("skipping...")

    # 6.
    # Bootstrap self-calibrated uv-data with CLEAN-models
    print("6. Bootstrap self-calibrated uv-data with CLEAN-models...")
    uv_boot_fits_dict = dict()
    for freq, uv_fits_path in uv_fits_dict.items():
        # cc_fits_paths = [cc_fits_dict[freq][stoke] for stoke in stokes]
        # bootstrap_uv_fits(uv_fits_path, cc_fits_paths, n_boot, outpath=outdir,
        #                   outname=('boot_{}'.format(freq), '_uv.fits'))
        files = glob.glob(os.path.join(outdir, 'boot_{}*.fits'.format(freq)))
        uv_boot_fits_dict.update({freq: sorted(files)})

    # 7.
    # Optionally clean bootstrap replications with original restoring beams and
    # map sizes to get error estimates for original resolution maps of I, PPOL,
    # FPOL, ...
    print("7. Optionally clean bootstrap replications with original restoring"
          " beams and map sizes...")
    print("skipping...")

    # 8.
    # Optionally clean bootstrap replications with common restoring beams and
    # map sizes
    print("8. Optionally clean bootstrap replications with common restoring"
          " beams and map sizes...")
    cc_boot_fits_dict = dict()
    for freq in freqs:
        cc_boot_fits_dict.update({freq: dict()})
        uv_fits_paths = uv_boot_fits_dict[freq]
        for stoke in stokes:
            for i, uv_fits_path in enumerate(uv_fits_paths):
                uv_dir, uv_fname = os.path.split(uv_fits_path)
                outfname = 'boot_{}_{}_cc_{}.fits'.format(freq, stoke,
                                                          str(i + 1).zfill(3))
                # clean_difmap(uv_fname, outfname, stoke, common_imsize,
                #              path=uv_dir, path_to_script=path_to_script,
                #              outpath=outdir, show_difmap_output=False)
            files = sorted(glob.glob(os.path.join(outdir,
                                                  'boot_{}_{}_cc_*.fits'.format(freq, stoke))))
            cc_boot_fits_dict[freq].update({stoke: files})

    # 9. Optionally estimate RM map and it's error
    print("9. Optionally estimate RM map and it's error...")
    original_cs_images = Images()
    for freq in freqs:
        for stoke in stokes:
            original_cs_images.add_images(cc_cs_image_dict[freq][stoke])

    # Find rough mask for creating bootstrap images of RM, alpha, ...
    print("Finding rough mask for creating bootstrap images of RM, alpha, ...")
    cs_mask = pol_mask({stoke: cc_cs_image_dict[freqs[-1]][stoke] for
                        stoke in stokes}, n_sigma=3.)

    rotm_image, _ = original_cs_images.create_rotm_image(mask=cs_mask)

    boot_images = Images()
    fnames = sorted(glob.glob(os.path.join(data_dir, "boot_*_*_cc_*.fits")))
    for freq in freqs:
        for stoke in stokes:
                boot_images.add_from_fits(cc_boot_fits_dict[freq][stoke])
    boot_rotm_images = boot_images.create_rotm_images(mask=cs_mask)
    s_rotm_image = boot_rotm_images.create_error_image(cred_mass=0.95)

    if rotm_slices is not None:
        fnames = ['rotm_slice_spread_{}.png'.format(i + 1) for i in
                  range(len(rotm_slices))]
        for rotm_slice, fname in zip(rotm_slices, fnames):
            analyze_rotm_slice(rotm_slice, rotm_image, boot_rotm_images,
                               outdir=outdir, outfname=fname)


    # # Calculate simulataneous confidence bands
    # # Bootstrap slices
    # slices = list()
    # for image in rotm_images_sym.images:
    #     slice_ = image.slice((216, 276), (296, 276))
    #     slices.append(slice_[~np.isnan(slice_)])

    # # Find means
    # obs_slice = rotm_image_sym.slice((216, 276), (296, 276))
    # x = np.arange(216, 296, 1)
    # x = x[~np.isnan(obs_slice)]
    # obs_slice = obs_slice[~np.isnan(obs_slice)]
    # # Find sigmas
    # slices_ = [arr.reshape((1, len(obs_slice))) for arr in slices]
    # sigmas = hdi_of_arrays(slices_).squeeze()
    # means = np.mean(np.vstack(slices), axis=0)
    # diff = obs_slice - means
    # # Move bootstrap curves to original simulated centers
    # slices_ = [slice_ + diff for slice_ in slices]
    # # Find low and upper confidence band
    # low, up = create_sim_conf_band(slices_, obs_slice, sigmas,
    #                                alpha=conf_band_alpha)

    # # Plot confidence bands and model values
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x, low[::-1], 'g')
    # ax.plot(x, up[::-1], 'g')
    # [ax.plot(x, slice_[::-1], 'r', lw=0.15) for slice_ in slices_]
    # ax.plot(x, obs_slice[::-1], '.k')
    # # Plot ROTM model
    # ax.plot(np.arange(216, 296, 1),
    #         rotm_grad_value * (np.arange(216, 296, 1) - 256.)[::-1] +
    #         rotm_value_0)
    # fig.savefig(os.path.join(data_dir, 'rotm_slice_spread.png'),
    #             bbox_inches='tight', dpi=200)
    # plt.close()


    if clear_difmap_logs:
        print("Removing difmap log-files...")
        difmap_logs = glob.glob(os.path.join(outdir, "difmap.log*"))
        for difmpa_log in difmap_logs:
            os.unlink(difmpa_log)



if __name__ == '__main__':
    source = '2230+114'
    epoch = '2006_02_12'
    base_dir = '/home/ilya/code/vlbi_errors/examples/mojave/sources'
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    mapsize_dict = {'x': (512, 0.1), 'y': (512, 0.1), 'j': (512, 0.1),
                    'u': (512, 0.1)}
    mapsize_common = (512, 0.1)
    data_dir = os.path.join(base_dir, source, epoch)
    bands = ['x', 'y', 'j', 'u']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    from mojave import download_mojave_uv_fits
    download_mojave_uv_fits(source, epochs=[epoch], bands=bands,
                            download_dir=data_dir)
    from mojave import mojave_uv_fits_fname
    uv_fits_fnames = [mojave_uv_fits_fname(source, band, epoch) for band in
                      bands]
    uv_fits_paths = [os.path.join(data_dir, uv_fits_fname) for uv_fits_fname in
                     uv_fits_fnames]

    analyze_source(uv_fits_paths, n_boot=30, outdir=data_dir,
                   path_to_script=path_to_script, imsizes=mapsize_dict.values(),
                   common_imsize=mapsize_common, find_shifts=False)
