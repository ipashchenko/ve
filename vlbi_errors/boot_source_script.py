import os
from from_fits import (create_uvdata_from_fits_file,
                       create_ccmodel_from_fits_file)
from bootstrap import CleanBootstrap
from data_io import get_fits_image_info
from utils import (mas_to_rad, degree_to_rad)
from spydiff import clean_difmap


def bootstrap_uv_fits(uv_fits_path, cc_fits_paths, n, outpath=None,
                      outname=None):
    """
    Function that bootstraps UV-data in user-specified UV-FITS files and FITS
    files with CC-models.

    :param uv_fits_path:
        Path to fits file with self-calibrated uv-data.
    :param cc_fits_paths:
        Iterable of paths to files with CC models.
    :param outpath:

    :param boot_kwargs:
    """

    uvdata = create_uvdata_from_fits_file(uv_fits_path)

    models = list()
    for cc_fits_path in cc_fits_paths:
        # FIXME: I can infer ``stokes`` from FITS-file!
        stokes = get_fits_image_info(cc_fits_path)[-2].upper()
        ccmodel = create_ccmodel_from_fits_file(cc_fits_path, stokes=stokes)
        models.append(ccmodel)

    boot = CleanBootstrap(models, uvdata)
    if outpath is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    curdir = os.getcwd()
    os.chdir(outpath)
    boot.run(n=n, outname=[outname, '.fits'])
    os.chdir(curdir)


def clean_uv_fits(uv_fits_paths, out_fits_paths, stokes, beam=None,
                  mapsize_clean=None,
                  mapsize_fits_path=None, pixsize_fits_path=None,
                  pixel_per_beam=None, mapsize=None,
                  beamsize_fits_path=None, mapsize_restore=None,
                  path_to_script=None):
    """
    Clean uv-data fits files with image and clean parameters that are chosen in
    several possible ways.

    :param uv_fits_paths:
        Iterable of paths to uv-data fits-files.
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

    :note:
        Image, pixel & beam specification uses this sequence. If ``beam`` is not
        supplied and ``beamsize_fits_path`` image is not supplied then beam
        parameters naitive to uv-data and difmap cleaning script are used. Image
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
        beam_pars = (map_info[3][0] / mas_to_rad, map_info[3][1] / mas_to_rad,
                     map_info[3][2] / degree_to_rad)

    # Choosing image parameters
    map_pars = None
    if mapsize_clean is not None:
        map_pars = mapsize_clean
    if map_pars is None and mapsize_fits_path is not None:
        map_info = get_fits_image_info(mapsize_fits_path)
        map_pars = (map_info[0][0], map_info[-3][0] / mas_to_rad)

    # Choosing pixel parameters
    pixsize = None
    if pixel_per_beam is not None:
        pixsize = beam_pars[0] / pixel_per_beam

    if pixsize is None and pixsize_fits_path is not None:
        map_info = get_fits_image_info(pixsize_fits_path)
        pixsize = map_info[-3][0] / mas_to_rad
    # Correcting image size when pixel size has changed
    imsize = map_pars[0] * abs(map_pars[1]) / pixsize
    powers = [imsize // (2 ** i) for i in range(15)]
    imsize = 2 ** powers.index(0)

    map_pars = (imsize, pixsize)

    print "Selected image and pixel size: {}".format(map_pars)

    for uv_fits_path, out_fits_path in zip(uv_fits_paths, out_fits_paths):
        for stoke in stokes:
            print "Cleaning stokes {}", stoke
            uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
            out_fits_dir, out_fits_fname = os.path.split(out_fits_path)
            clean_difmap(fname=uv_fits_fname,
                         outfname=out_fits_fname,
                         stokes=stoke, mapsize_clean=map_pars,
                         path=uv_fits_dir,
                         path_to_script=path_to_script,
                         beam_restore=beam_pars,
                         outpath=out_fits_dir)
    os.chdir(curdir)
