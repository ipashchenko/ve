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


def bootstrap_uv_fits(uv_fits_path, cc_fits_paths, n, outpath=None,
                      outname=None):
    """
    Function that bootstraps uv-data in user-specified FITS-files and
    FITS-files with clean components.

    :param uv_fits_path:
        Path to fits file with self-calibrated uv-data.
    :param cc_fits_paths:
        Iterable of paths to files with CC models.
    :param outpath:

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
    boot.run(n=n, outname=[outname, '.fits'])
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
        print("Using dr={}".format(dr))
        shift_dict[dr] = list()

        # Iterating over mask sizes
        for r in range(0, max_mask_r, mask_step):
            print("Using r={}".format(r))
            r1 = r
            r2 = r + dr
            shift = image1.cross_correlate(image2,
                                           region1=(image1.x_c, image1.y_c, r1,
                                                    None),
                                           region2=(image2.x_c, image2.y_c, r2,
                                                    None))
            print("Got shift {}".format(shift))
            shift_dict[dr].append(shift)

    for key, value in shift_dict.items():
        value = np.array(value)
        shifts = np.sqrt(value[:, 0] ** 2. + value[:, 1] ** 2.)
        shift_dict.update({key: np.std(shifts)})

    # Searching for mask size difference that has minimal std in shifts
    # calculated for different mask sizes
    return sorted(shift_dict, key=lambda _: shift_dict[_])[0]


def analyze_source(uv_fits_paths, n_boot, cc_fits_paths=None, imsizes=None,
                   common_imsize=None, common_beam=None, find_shifts=False,
                   bootstrap_shift_error=False, add_shift=False, outdir=None,
                   path_to_script=None, stokes=None):
    """
    Function that uses multifrequency self-calibration data for in-depth
    analysis.

    :param uv_fits_paths:
        Iterable of paths to self-calibrated uv-data FITS-files.
    :param n_boot:
        Number of bootstrap replications to use in analysis.
    :param cc_fits_paths: (optional)
        Iterable of dicts with paths to FITS-files with CLEAN-images. Each
        dictionary should include stokes parameters ``I``[, ``Q``, ``U``]. If
        ``None`` then use use-specified image parameters. (default: ``None``)
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
    :param common_beam: (optional)
        Beam parameters that will be used in making common size images for
        multifrequency analysis. If ``None`` then use beam of lowest frequency.
        (default: ``None``)
    :param find_shifts: (optional)
        Find shifts between multifrequency images? (default: ``False``)
    :param bootstrap_shift_error: (optional)
        Use bootstrap to estimate shift image error. (default: ``False``)
    :param add_shift: (optional)
        Add shift while CLEANing bootstrap replications? (default: ``False``)
    :param outdir: (optional)
        Output directory. This directory will be used for saving picture, data,
        etc. If ``None`` then use CWD. (default: ``None``)
    :param path_to_script: (optional)
        Path ot difmap CLEAN script. If ``None`` then use CWD. (default:
        ``None``)
    :param stokes: (optional)
        Iterable of stokes parameters to CLEAN original self-calibrated uv-data.
        If ``None`` then ('I', 'Q', 'U'). (default: ``None``)

    :notes:
        Function uses this workflow:

        1. CLEAN uv-data in specified FITS-files (``uv_fits_paths``) with
        parameters specified in ``imsizes`` argument. If ``cc_fits_paths`` are
        specified then it supposed that it is the result of such CLEAN.

        #. Bootstrap uv-data with obtained CLEAN-models using ``n_boot``
        bootstrap realizations.

        #. Find shift between all possible frequencies. Optionally finds
        bootstrap error estimate of found shift values.

        #. CLEAN bootstrapped data (optionally with added shifts relative to
        lowest frequency) with image parameters (imsize, pixsize) specified by
        ``common_imsize`` or (if it is ``None``) by lowest and highest frequency
        CLEAN images.

        #. Create ROTM images from CLEANed bootstrapped data.

    """
    # Setting up the output directory
    if outdir is None:
        outdir = os.getcwd()
    # Assume input self-calibrated uv-data FITS files have different frequencies
    n_freq = len(uv_fits_paths)
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

    # If CLEAN-images are supplied then use them to fill the containers
    if cc_fits_paths is not None:
        imsizes_dict = dict()
        print("Exploring supplied CLEAN images")
        for cc_fits_path in cc_fits_paths:
            image = create_clean_image_from_fits_file(cc_fits_path)
            freq = image.freq
            stoke = image.stokes
            if freq not in freqs:
                raise Exception("Frequency of CLEAN image not known from"
                                " UV-data!")
            cc_image_dict[freq].update({stoke: image})
            cc_fits_dict[freq].update({stoke: cc_fits_path})
            if stoke == 'I':
                cc_beam_dict[freq].update({stoke: image.beam})
        for freq in freqs:
            image = cc_image_dict[freq][stoke]
            imsizes_dict.update({freq: (image.imsize[0], abs(image.pixsize[0] /
                                                             mas_to_rad))})

        assert set(uv_fits_dict.keys()).issubset(cc_fits_dict.keys())
        # Define stokes from existing CLEAN-images
        stokes = cc_image_dict[freqs[0]].keys()

    # If no CLEAN-images are supplied then CLEAN original data with
    # use-specified parameters
    elif imsizes is not None:
        assert len(imsizes) == n_freq
        if stokes is None:
            stokes = ('I', 'Q', 'U')
        imsizes_dict = dict()
        for i, freq in enumerate(freqs):
            imsizes_dict.update({freq: imsizes[i]})
        for freq in freqs:
            uv_fits_path = uv_fits_dict[freq]
            uv_dir, uv_fname = os.path.split(uv_fits_path)
            for stoke in stokes:
                outfname = '{}_{}_cc.fits'.format(freq, stoke)
                outpath = os.path.join(outdir, outfname)
                clean_difmap(uv_fname, outfname, stoke, imsizes[freq],
                             path=uv_dir, path_to_script=path_to_script,
                             outpath=outdir, show_difmap_output=True)
                cc_fits_dict[freq].update({stoke: os.path.join(outdir,
                                                               outfname)})
                image = create_image_from_fits_file(outpath)
                cc_image_dict[freq].update({stoke: image})
                if stoke == 'I':
                    cc_beam_dict[freq].update({stoke: image.beam})
    else:
        raise Exception("Provide ``cc_fits_paths`` of ``imsizes``")

    # Now CLEAN uv-data using ``common_imsize`` or parameters of CLEAN-image
    # from lowest and highest frequencies
    # Container for common sized CLEAN-images of self-calibrated uv-data
    cc_cs_image_dict = dict()
    cc_cs_fits_dict = dict()
    if common_imsize is None:
        imsize_low, pixsize_low = imsizes_dict[freqs[0]]
        imsize_high, pixsize_high = imsizes_dict[freqs[-1]]
        imsize = imsize_low * pixsize_low / pixsize_high
        powers = [imsize // (2 ** i) for i in range(15)]
        imsize = 2 ** powers.index(0)
        common_imsize = (imsize, pixsize_high)
    for freq in freqs:
        cc_cs_image_dict.update({freq: dict()})
        cc_cs_fits_dict.update({freq: dict()})

        uv_fits_path = uv_fits_dict[freq]
        uv_dir, uv_fname = os.path.split(uv_fits_path)
        for stoke in stokes:
            outfname = 'cs_{}_{}_cc.fits'.format(freq, stoke)
            outpath = os.path.join(outdir, outfname)
            clean_difmap(uv_fname, outfname, stoke, common_imsize,
                         path=uv_dir, path_to_script=path_to_script,
                         outpath=outdir, show_difmap_output=True)
            cc_fits_dict[freq].update({stoke: os.path.join(outdir,
                                                           outfname)})
            image = create_image_from_fits_file(outpath)
            cc_image_dict[freq].update({stoke: image})

    # Bootstrap self-calibrated uv-data with CLEAN-models
    for freq, uv_fits_path in uv_fits_dict:
        cc_fits_paths = cc_fits_dict[freq].keys()
        bootstrap_uv_fits(uv_fits_path, cc_fits_paths, n_boot, outpath=outpath,
                          outname=('boot_{}'.format(freq), '_uv.fits'))

    # Optionally find shifts between original CLEAN-images
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
                               min_shift=min_shift, max_mask_r=200, mask_step=5)

            shift_dict.update({(freq_1, freq_2,): shift})

        # Dumping shifts to json file in target directory
        json.dump(shift_dict, os.path.join(outpath, "shifts_original.json"))


if __name__ == '__main__':
    pass
