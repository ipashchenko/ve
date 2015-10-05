import os
import glob
import numpy as np
from data_io import get_fits_image_info
from from_fits import (create_uvdata_from_fits_file,
                       create_ccmodel_from_fits_file,
                       create_clean_image_from_fits_file)
from bootstrap import CleanBootstrap
from utils import mas_to_rad, find_card_from_header
from spydiff import clean_difmap


def boot_uv_fits_with_cc_fits(uv_fits_fname, cc_fits_fnames, n, uvpath=None,
                              ccpath=None, outname=None, outpath=None,
                              nonparametric=True):
    """
    Function that bootstraps UV-data in user-specified UV-FITS files and FITS
    files with CCs.

    :param uv_fits_fname:
        UV-FITS file name.
    :param cc_fits_fnames:
        Iterable of file names with CCs.
    :param n:
        Number of bootstrap replications to create.
    :param uvpath: (optional)
        Path to directory with UV-FITS file. If ``None`` then use cwd. (default:
        ``None``)
    :param ccpath: (optional)
        Iterable of paths to directories with Clean images FITS-files. If not
        ``None`` then length must be equal to length of ``cc_fits_fnames``, else
        use cwd for each of FITS-file. (default: ``None``)
    :param outname: (optional)
        Base of output name for bootstrapped data. If ``None`` then use
        ``uv_fits_fname`` for constructing ``outname`` by omitting extension.
        (default: ``None``)
    :param outpath: (optional)
        Output directory for saving bootstrapped FITS-files. If ``None`` then
        use ``uvpath``. (default: ``None``)
    :param nonparametric (optional):
        If ``True`` then use actual residuals between model and data. If
        ``False`` then use gaussian noise fitted to actual residuals for
        parametric bootstrapping. (default: ``False``)
    """
    if ccpath is not None:
        if len(ccpath) > 1:
            assert len(cc_fits_fnames) == len(ccpath)
    else:
        ccpath = [None] * len(cc_fits_fnames)

    if uvpath is not None:
        uv_fits_fname = os.path.join(uvpath, uv_fits_fname)
    uvdata = create_uvdata_from_fits_file(uv_fits_fname)

    models = list()
    for cc_fits_fname, ccpath_ in zip(cc_fits_fnames, ccpath):
        if ccpath_ is not None:
            cc_fits_fname = os.path.join(ccpath_, cc_fits_fname)
        stokes = get_fits_image_info(cc_fits_fname)[-2].upper()
        ccmodel = create_ccmodel_from_fits_file(cc_fits_fname, stokes=stokes)
        models.append(ccmodel)

    boot = CleanBootstrap(models, uvdata)
    if outpath is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    curdir = os.getcwd()
    os.chdir(outpath)
    if outname is None:
        outname = uv_fits_fname.split('.')[0]
    boot.run(n=n, outname=[outname, '.fits'])
    os.chdir(curdir)


def find_core_shift(uv_fits_paths, cc_image_paths, r_mask=None,
                    path_to_script=None, pixels_per_beam=None, imsize=None,
                    do_bootstrap=False, n_boot=50, nonparametric=True,
                    data_dir=None):
    """
    Function that calculates core shift between two frequencies.

    :param uv_fits_paths:
        Iterable of 2 paths to UV-FITS files.
    :param cc_image_paths:
        Iterable of 2 paths to Clean images FITS files.
    :param r_mask: (optional)
        Radius of circular mask used to mask the optically thick core. If
        ``None`` then find radius automatically. (default: ``None``)
    :param path_to_script: (optional)
        Path to difmap autoclean-script. If ``None`` then use cwd. (default:
        ``None``)
    :param pixels_per_beam: (optional)
        Number of pixels per low frequency beam. If ``None`` then just use
        pixel size of high frequency map. (default: ``None``)
    :param imsize: (optional)
        Image size to use [pixels]. If ``None`` then use physical image size of
        low frequency map with pixel size of high frequency map rounded to
        nearest power of ``2`` to the biggest value. (default: ``None``)
    :param do_bootstrap: (optional)
        Do bootstrap for uncertainty calculation? (default: ``False``)
    :param n_boot: (optional)
        How many bootstrap replications to use? (default: ``50``)
    :param nonparametric (optional):
        If ``True`` then use actual residuals between model and data. If
        ``False`` then use gaussian noise fitted to actual residuals for
        parametric bootstrapping. (default: ``False``)
    :param data_dir: (optional)
        Path to directory where to store clean image FITS files and bootstrapped
        UV-FITS files. If ``None`` then use cwd. (default: ``None``)

    :return:
    """

    map_info_0 = get_fits_image_info(cc_image_paths[0])
    map_info_1 = get_fits_image_info(cc_image_paths[1])
    # 0 - lower frequency
    if map_info_0[-1] < map_info_1[-1]:
        map_info_l = map_info_0
        map_info_h = map_info_1
        high_freq_map = cc_image_paths[1]
        low_freq_map = cc_image_paths[0]
    else:
        map_info_l = map_info_1
        map_info_h = map_info_0
        high_freq_map = cc_image_paths[0]
        low_freq_map = cc_image_paths[1]

    bmaj_l = map_info_l[3][0] / mas_to_rad
    bmin_l = map_info_l[3][1] / mas_to_rad
    bpa_l = map_info_l[3][2]
    pixsize_l = map_info_l[-3][0] / mas_to_rad
    pixsize_h = map_info_h[-3][0] / mas_to_rad
    imsize_h = map_info_h[0][0]
    imsize_l = map_info_l[0][0]
    beam_restore = (bmaj_l, bmin_l, bpa_l)

    # If we told to use some pixel size (in units of low frequency beam)
    if pixels_per_beam is not None:
        pixsize = bmaj_l / pixels_per_beam
    else:
        pixsize = pixsize_h
        # If we don't told to use some image size we construct it to keep
        # physical image size as in low frequency map
        if imsize is None:
            imsize = imsize_l * pixsize_l / pixsize
            powers = [2 ** i / imsize for i in range(15)]
            indx = len(powers) - powers[::-1].index(0) - 1
            imsize = 2 ** indx

    # Chosen image & pixel sizes
    mapsize_clean = (imsize, pixsize)
    map_center = imsize / 2

    uvdata_dict = dict()
    # Clean both UV data with new image parameters
    for uvdata_path in uv_fits_paths:
        uvdata_dir, uvdata_file = os.path.split(uvdata_path)
        print "Cleaning uv file {}".format(uvdata_path)
        uvdata = create_uvdata_from_fits_file(uvdata_path)
        freq_card = find_card_from_header(uvdata._io.hdu.header,
                                          value='FREQ')[0]
        # Frequency in Hz
        freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        uvdata_dict.update({freq: uvdata_path})
        print "restoring with beam ", beam_restore
        print "Cleaning with map parama ", mapsize_clean
        clean_difmap(uvdata_file, "shift_i_{}_cc.fits".format(freq),
                     stokes='i', mapsize_clean=mapsize_clean, path=uvdata_dir,
                     path_to_script=path_to_script, beam_restore=beam_restore,
                     outpath=data_dir)

    # Find where high and low frequency UV data
    low_freq, high_freq = sorted(uvdata_dict)
    h_uvdata_path = uvdata_dict[high_freq]
    l_uvdata_path = uvdata_dict[low_freq]

    images = glob.glob(os.path.join(data_dir, 'shift_i_*_cc.fits'))
    images = sorted(images)
    ccimage_l = create_clean_image_from_fits_file(images[0])
    ccimage_h = create_clean_image_from_fits_file(images[1])

    if r_mask is None:
        # Find mask radius that gaves maximum shift length
        shifts = dict()
        for r in range(0, map_center / 4, 10):
            region = (map_center, map_center, r, None)
            shift = ccimage_h.cross_correlate(ccimage_l, region1=region,
                                              region2=region)
            shifts.update({r: shift})
        r_mask = max(shifts.iterkeys(), key=lambda k: shifts[k])

    region = (map_center, map_center, r_mask, None)
    shift = ccimage_h.cross_correlate(ccimage_l, region1=region,
                                      region2=region)
    print "Shift found: {}".format(shift)

    if do_bootstrap:
        # Bootstrap clean images
        print "Bootstrapping uv-data and original clean models..."
        for uvdata_path, cc_fits_path in zip([h_uvdata_path, l_uvdata_path],
                                             [high_freq_map, low_freq_map]):
            uvdata_dir, uvdata_file = os.path.split(uvdata_path)
            cc_dir, cc_file = os.path.split(cc_fits_path)
            print "Bootstrapping uv file {}".format(uvdata_path)
            print "With cc-model {}".format(cc_fits_path)
            uvdata = create_uvdata_from_fits_file(uvdata_path)
            freq_card = find_card_from_header(uvdata._io.hdu.header,
                                              value='FREQ')[0]
            # Frequency in Hz
            freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
            print "Bootstrapping frequency {}".format(freq)
            boot_uv_fits_with_cc_fits(uvdata_file, [cc_file], n_boot,
                                      uvpath=uvdata_dir, ccpath=[cc_dir],
                                      outpath=data_dir,
                                      outname='boot_{}'.format(freq),
                                      nonparametric=nonparametric)

        # Clean bootstrapped uv-data
        for uvdata_path in glob.glob(os.path.join(data_dir, 'boot_*')):
            uvdata_dir, uvdata_file = os.path.split(uvdata_path)
            i_boot = uvdata_file.split('_')[-1].split('.')[0]
            uvdata = create_uvdata_from_fits_file(uvdata_path)
            freq_card = find_card_from_header(uvdata._io.hdu.header,
                                              value='FREQ')[0]
            freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
            clean_difmap(uvdata_path, "shift_{}_{}_cc.fits".format(freq,
                                                                   i_boot), 'i',
                         mapsize_clean, path=uvdata_dir,
                         path_to_script=path_to_script,
                         beam_restore=beam_restore, outpath=data_dir)

        # Caclulate shifts for all pairs of bootstrapped clean maps
        shifts = list()
        for i_boot in range(1, n_boot + 1):
            cc_fnames_glob = 'shift_*_{}_cc.fits'.format(i_boot)
            cc_fnames = glob.glob(os.path.join(data_dir, cc_fnames_glob))
            print "Caculating shifts for bootstrap realization {}".format(i_boot)
            cc_fnames = sorted(cc_fnames)
            ccimage_l = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                       cc_fnames[0]))
            ccimage_h = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                       cc_fnames[1]))
            shift = ccimage_h.cross_correlate(ccimage_l, region1=region, region2=region)
            print "Found shift {}".format(shift)
            shifts.append(shift)

        shifts = np.vstack(shifts)

    return shift, shifts, r_mask
