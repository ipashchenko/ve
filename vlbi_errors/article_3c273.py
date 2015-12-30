import glob
import os
import shutil
import numpy as np
import matplotlib
# For saving images without plotting
matplotlib.use('Agg')
from uv_data import UVData
from from_fits import (create_model_from_fits_file,
                       get_fits_image_info)
from bootstrap import CleanBootstrap
from spydiff import clean_difmap
from utils import mas_to_rad, degree_to_rad, hdi_of_arrays, find_card_from_header
from images import Images
from sim_func import simulate_grad
from image import plot


def im_fits_fname(source, band, epoch, stokes='i', ext='fits'):
    if band in ('x', 'y', 'j') and stokes == 'i':
        stokes = 'i_0.1'
    if stokes == 'i':
        stokes = 'icn'
    return source + '.' + band + '.' + epoch + '.' + stokes + '.' + ext


def boot_im_fits_fname(freq, stokes):
    return 'simul_{}_{}_cc.fits'.format(stokes, freq)


def boot_uv_fits_fname(freq):
    return 'simul_uv_{}_Hz.fits'.format(freq)


def uv_fits_fname(source, band, epoch, ext='uvf'):
    return source + '.' + band + '.' + epoch + '.' + ext


def uv_fits_path(source, band, epoch, base_path=None):
    """
    Function that returns path to uv-file for given source, epoch and band.

    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    """
    return os.path.join(base_path, source + '/' + epoch + '/' + band.upper() +
                        '/uv/')


def im_fits_path(source, band, epoch, stoke, base_path=None):
    """
    Function that returns path to im-file for given source, epoch, band and
    stokes parameter.

    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    """
    return os.path.join(base_path, source + '/' + epoch + '/' + band.upper() +
                        '/im/' + stoke.upper() + '/')


# FIXME: results in changing cwd to ``base_path``
def create_dirtree(sources, epochs, bands, stokes, base_path=None):
    """
    Function that creates directory tree for observations.

    :param sources:
        Iterable of sources names.
    :param epochs:
        Iterable of sources epochs.
    :param bands:
        Iterable of bands.
    :param stokes:
        Iterable of stokes parameters.
    :param base_path:
        Path to root directory of directory tree.

    """
    stokes = [stoke.upper() for stoke in stokes]
    bands = [band.upper() for band in bands]
    if base_path is None:
        base_path = os.getcwd()
    elif not base_path.endswith("/"):
        base_path += "/"
    curdir = os.getcwd()
    os.chdir(base_path)

    for source in sources:
        os.mkdir(source)
        os.chdir(source)
        for epoch in epochs:
            os.mkdir(epoch)
            os.chdir(epoch)
            for band in bands:
                os.mkdir(band)
                os.chdir(band)
                os.mkdir('uv')
                os.mkdir('im')
                os.chdir('im')
                for dir in stokes + ['ALPHA', 'IPOL', 'FPOL', 'RM']:
                    os.mkdir(dir)
                os.chdir(os.path.join(os.path.pardir, os.curdir))
                os.chdir(os.path.join(os.path.pardir, os.curdir))
            os.chdir(os.path.join(os.path.pardir, os.curdir))
        os.chdir(os.path.join(os.path.pardir, os.curdir))

    os.chdir(curdir)


def put_uv_files_to_dirs(sources, epochs, bands, base_path=None, ext="uvf",
                         uv_files_path=None):
    """
    :param sources:
        Iterable of sources names.
    :param epochs:
        Iterable of sources epochs.
    :param bands:
        Iterable of bands.
    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    :param uv_files_path: (optional)
        Path to directory with uv-files. If ``None`` then use current directory.
        (default: ``None``)
    """
    # bands = [band.upper() for band in bands]
    if base_path is None:
        base_path = os.getcwd()
    if uv_files_path is None:
        uv_files_path = os.getcwd()

    # Circle through sources, epochs and bands and copy files to directory tree.
    for source in sources:
        for epoch in epochs:
            for band in bands:
                fname = uv_fits_fname(source, band, epoch, ext=ext)
                outpath = uv_fits_path(source, band, epoch, base_path=base_path)
                try:
                    print "fname {}".format(os.path.join(uv_files_path, fname))
                    shutil.copyfile(os.path.join(uv_files_path, fname),
                                    os.path.join(outpath, 'sc_uv.fits'))
                    print "Copied file ", fname
                    print "from ", uv_files_path, " to ", outpath
                except IOError:
                    print "No such file ", fname, " in ", uv_files_path


def put_im_files_to_dirs(sources, epochs, bands, stokes, base_path=None,
                         ext="fits", im_files_path=None):
    """
    :param sources:
        Iterable of sources names.
    :param epochs:
        Iterable of sources epochs.
    :param bands:
        Iterable of bands.
    :param stokes:
        Iterable of stokes parameters.
    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    :param im_files_path: (optional)
        Path to directory with im-files. If ``None`` then use current directory.
        (default: ``None``)

    """
    # stokes = [stoke.upper() for stoke in stokes]
    # bands = [band.upper() for band in bands]
    if base_path is None:
        base_path = os.getcwd()
    if im_files_path is None:
        im_files_path = os.getcwd()

    # Circle through sources, epochs and bands and copy files to directory tree.
    for source in sources:
        for epoch in epochs:
            for band in bands:
                for stoke in stokes:
                    fname = im_fits_fname(source, band, epoch, stoke, ext=ext)
                    outpath = im_fits_path(source, band, epoch, stoke,
                                           base_path=base_path)
                    try:
                        print "fname {}".format(os.path.join(im_files_path,
                                                             fname))
                        shutil.copyfile(os.path.join(im_files_path, fname),
                                        os.path.join(outpath, 'cc.fits'))
                        print "Copied file ", fname
                        print "from ", im_files_path, " to ", outpath
                    except IOError:
                        print "No such file ", fname, " in ", im_files_path


def generate_boot_data(sources, epochs, bands, stokes, n_boot=10,
                       base_path=None):
    """
    :param sources:
        Iterable of sources names.
    :param epochs:
        Iterable of sources epochs.
    :param bands:
        Iterable of bands.
    :param stokes:
        Iterable of stokes parameters.
    :param n_boot: (optional)
        Number of bootstrap replications to create. (default: ``10``)
    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.

    """
    if base_path is None:
        base_path = os.getcwd()
    elif not base_path.endswith("/"):
        base_path += "/"

    curdir = os.getcwd()
    print "Generating bootstrapped data..."
    for source in sources:
        print " for source ", source
        for epoch in epochs:
            print " for epoch ", epoch
            for band in bands:
                print " for band ", band
                uv_path = uv_fits_path(source, band.upper(), epoch,
                                       base_path=base_path)
                uv_fname = uv_path + 'sc_uv.fits'
                if not os.path.isfile(uv_fname):
                    print "...skipping absent file ", uv_fname
                    continue
                print "  Using uv-file (data): ", uv_fname
                uvdata = UVData(uv_fname)
                models = list()
                for stoke in stokes:
                    print "  Adding model with stokes parameter ", stoke
                    map_path = im_fits_path(source, band, epoch, stoke,
                                            base_path=base_path)
                    map_fname = map_path + 'cc.fits'
                    print "  from CC-model file ", map_fname
                    ccmodel = create_model_from_fits_file(map_fname,
                                                          stokes=stoke.upper())
                    models.append(ccmodel)
                boot = CleanBootstrap(models, uvdata)
                os.chdir(uv_path)
                boot.run(n=n_boot, outname=['boot', '.fits'])
    os.chdir(curdir)


def generate_simulated_boot_data(freqs, stokes, n_boot=10, base_path=None):
    """
    :param freqs:
        Iterable of frequencies [Hz]. Lowest frequency goes first.
    :param stokes:
        Iterable of stokes parameters.
    :param n_boot: (optional)
        Number of bootstrap replications to create. (default: ``10``)
    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.

    """
    if base_path is None:
        base_path = os.getcwd()

    curdir = os.getcwd()
    print "Generating bootstrapped data..."
    for freq in freqs:
        print " for freq", freq
        uv_fname = boot_uv_fits_fname(freq)
        print "  Using uv-file (data): ", uv_fname
        uvdata = UVData(os.path.join(base_path, uv_fname))
        models = list()
        for stoke in stokes:
            print "  Adding model with stokes parameter ", stoke
            map_fname = boot_im_fits_fname(freq, stoke)
            print "  from CC-model file ", map_fname
            ccmodel = create_model_from_fits_file(os.path.join(base_path,
                                                               map_fname))
            models.append(ccmodel)
        boot = CleanBootstrap(models, uvdata)
        os.chdir(base_path)
        boot.run(n=n_boot, outname=['simul_uv_boot_{}'.format(freq), '.fits'])
    os.chdir(curdir)


def clean_boot_data(sources, epochs, bands, stokes, base_path=None,
                    path_to_script=None, pixels_per_beam=None, imsize=None):
    """
    :param sources:
        Iterable of sources names.
    :param epochs:
        Iterable of sources epochs.
    :param bands:
        Iterable of bands.
    :param stokes:
        Iterable of stokes parameters.
    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param beam: (optional)
        Beam parameter for cleaning (bmaj, bmin, bpa). If ``None`` then use
        naitive beam. (default: ``None``)
    :param mapsize_clean: (optional)
        Parameters of map for cleaning (map size, pixel size). If ``None``
        then use those of map in map directory (not bootstrapped).
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use ``mapsize_clean``. (default: ``None``)

    """
    if base_path is None:
        base_path = os.getcwd()
    elif not base_path.endswith("/"):
        base_path += "/"

    stokes = list(stokes)
    # Now ``I`` goes first
    stokes.sort()

    curdir = os.getcwd()
    print "Cleaning bootstrapped and original data..."
    for source in sources:
        print " for source ", source
        for epoch in epochs:
            print " for epoch ", epoch
            stoke = 'i'
            # Find ``mapsize`` using highest frequency data
            band = bands[-1]

            map_path = im_fits_path(source, band, epoch, stoke,
                                    base_path=base_path)
            try:
                map_info = get_fits_image_info(map_path + 'cc.fits')
            except IOError:
                continue
            # FIXME: dirty hack - check images!
            # 3c273
            # mapsize_clean = (1024, map_info[-3][0] / mas_to_rad)
            # 1633
            mapsize_clean = (map_info[0][0], map_info[-3][0] / mas_to_rad)
            # Find ``beam_restore`` using lowest frequency data
            band = bands[0]
            map_path = im_fits_path(source, band, epoch, stoke,
                                    base_path=base_path)
            map_info = get_fits_image_info(map_path + 'cc.fits')
            beam_restore = (map_info[3][0] / mas_to_rad,
                            map_info[3][1] / mas_to_rad,
                            map_info[3][2] / degree_to_rad)
            # If we told to use some pixel size (in units of low frequency beam)
            # if pixels_per_beam is not None:
            #     pixsize = beam_restore[0] / pixels_per_beam
            # else:
            #     pixsize = mapsize_clean[1]
            #     # If we don't told to use some image size we construct it to keep
            #     # physical image size as in low frequency map
            #     if imsize is None:
            #         # imsize = imsize_low * pix_size_low / new_pixsize
            #         imsize = map_info[0][0] * (map_info[-3][0] /
            #                                    mas_to_rad) / pixsize
            #         powers = [imsize // (2 ** i) for i in range(15)]
            #         indx = powers.index(0)
            #         imsize = 2 ** indx

            # Chosen image & pixel sizes
            # mapsize_clean = (imsize, pixsize)
            print "Common mapsize: {}".format(mapsize_clean)
            print "Common beam: {}".format(beam_restore)

            for band in bands:
                print " for band ", band
                uv_path = uv_fits_path(source, band.upper(), epoch,
                                       base_path=base_path)
                n = len(glob.glob(uv_path + '*boot*_*.fits'))
                if n == 0:
                    print "skippin source {}, epoch {}, band {}".format(source,
                                                                        epoch,
                                                                        band)
                    continue
                # Cleaning bootstrapped data & restore with low resolution
                for i in range(n):
                    uv_fname = uv_path + 'boot_' + str(i + 1) + '.fits'
                    if not os.path.isfile(uv_fname):
                        print "...skipping absent file ", uv_fname
                        continue
                    print "  Using uv-file ", uv_fname
                    # Sort stokes with ``I`` first and use it's beam
                    for stoke in stokes:
                        print "  working with stokes parameter ", stoke
                        map_path = im_fits_path(source, band, epoch, stoke,
                                                base_path=base_path)
                        clean_difmap(fname='boot_' + str(i + 1) + '.fits',
                                     outfname='cc_' + str(i + 1) + '.fits',
                                     stokes=stoke, mapsize_clean=mapsize_clean,
                                     path=uv_path,
                                     path_to_script=path_to_script,
                                     mapsize_restore=None,
                                     beam_restore=beam_restore,
                                     outpath=map_path,
                                     show_difmap_output=True)
                # Cleaning original data & restore with low_freq resolution
                for stoke in stokes:
                    print "  working with stokes parameter ", stoke
                    map_path = im_fits_path(source, band, epoch, stoke,
                                            base_path=base_path)
                    clean_difmap(fname='sc_uv.fits',
                                 outfname='cc_orig.fits',
                                 stokes=stoke, mapsize_clean=mapsize_clean,
                                 path=uv_path,
                                 path_to_script=path_to_script,
                                 mapsize_restore=None,
                                 beam_restore=beam_restore,
                                 outpath=map_path)
    os.chdir(curdir)


def clean_sumulated_boot_data(freqs, stokes, base_path=None,
                              path_to_script=None, pixels_per_beam=None,
                              imsize=None):
    """
    :param freqs:
        Iterable of frequencies [Hz]. Lowest frequency goes first.
    :param stokes:
        Iterable of stokes parameters.
    :param base_path: (optional)
        Path to route of directory with bootstrapped uv-data and
        cleaned_bootstrapped cc-models. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param beam: (optional)
        Beam parameter for cleaning (bmaj, bmin, bpa). If ``None`` then use
        naitive beam. (default: ``None``)
    :param mapsize_clean: (optional)
        Parameters of map for cleaning (map size, pixel size). If ``None``
        then use those of map in map directory (not bootstrapped).
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use ``mapsize_clean``. (default: ``None``)

    """
    if base_path is None:
        base_path = os.getcwd()

    stokes = list(stokes)
    # Now ``I`` goes first
    stokes.sort()

    curdir = os.getcwd()
    print "Cleaning bootstrapped and original data..."
    stoke = 'i'
    # Find ``mapsize`` using highest frequency data
    freq = freqs[-1]
    print freq, stoke
    hfreq_map_path = os.path.join(base_path, boot_im_fits_fname(freq, stoke))
    print hfreq_map_path
    map_info = get_fits_image_info(hfreq_map_path)
    # FIXME: dirty hack - check images!
    # 3c273
    # mapsize_clean = (1024, map_info[-3][0] / mas_to_rad)
    # 1633
    mapsize_clean = (map_info[0][0], map_info[-3][0] / mas_to_rad)
    # Find ``beam_restore`` using lowest frequency data
    freq = freqs[0]
    map_info = get_fits_image_info(os.path.join(base_path,
                                                boot_im_fits_fname(freq,
                                                                   stoke)))
    beam_restore = (map_info[3][0] / mas_to_rad,
                    map_info[3][1] / mas_to_rad,
                    map_info[3][2] / degree_to_rad)
    # If we told to use some pixel size (in units of low frequency beam)
    # if pixels_per_beam is not None:
    #     pixsize = beam_restore[0] / pixels_per_beam
    # else:
    #     pixsize = mapsize_clean[1]
    #     # If we don't told to use some image size we construct it to keep
    #     # physical image size as in low frequency map
    #     if imsize is None:
    #         # imsize = imsize_low * pix_size_low / new_pixsize
    #         imsize = map_info[0][0] * (map_info[-3][0] /
    #                                    mas_to_rad) / pixsize
    #         powers = [imsize // (2 ** i) for i in range(15)]
    #         indx = powers.index(0)
    #         imsize = 2 ** indx

    # Chosen image & pixel sizes
    # mapsize_clean = (imsize, pixsize)
    print "Common mapsize: {}".format(mapsize_clean)
    print "Common beam: {}".format(beam_restore)

    for freq in freqs:
        print " for freq", freq
        n = len(glob.glob(base_path + 'simul_uv_boot_{}_*.fits'.format(freq)))
        print "n = {}".format(n)
        # Cleaning bootstrapped data & restore with low resolution
        for i in range(n):
            uv_fname = os.path.join(base_path,
                                    'simul_uv_boot_{}_{}.fits'.format(freq,
                                                                      i + 1))
            print "  Using uv-file ", uv_fname
            # Sort stokes with ``I`` first and use it's beam
            for stoke in stokes:
                print "  working with stokes parameter ", stoke
                clean_difmap(fname=uv_fname,
                             outfname='simul_boot_cc_{}_{}_{}.fits'.format(freq, stoke, i + 1),
                             stokes=stoke, mapsize_clean=mapsize_clean,
                             path=base_path,
                             path_to_script=path_to_script,
                             mapsize_restore=None,
                             beam_restore=beam_restore,
                             outpath=base_path,
                             show_difmap_output=False)
        # Cleaning original data & restore with low_freq resolution
        for stoke in stokes:
            print " Cleaning original data with low freq resolution"
            print "  working with stokes parameter ", stoke
            clean_difmap(fname='simul_uv_{}_Hz.fits'.format(freq),
                         outfname='simul_boot_cc_orig_{}_{}.fits'.format(freq, stoke),
                         stokes=stoke, mapsize_clean=mapsize_clean,
                         path=base_path,
                         path_to_script=path_to_script,
                         mapsize_restore=None,
                         beam_restore=beam_restore,
                         outpath=base_path)
    os.chdir(curdir)


def create_images_from_boot_images(source, epoch, bands, stokes,
                                   base_path=None):
    """
    :param source:
        Source name.
    :param epoch:
        Sources epoch.
    :param bands:
        Iterable of bands.
    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    """

    print "Stacking bootstrapped images..."
    images = Images()
    for band in bands:
        print " for band ", band
        for stoke in stokes:
            map_path = im_fits_path(source, band, epoch, stoke,
                                    base_path=base_path)
            images.add_from_fits(wildcard=os.path.join(map_path,
                                                       'cc_*[0-9].fits'))

    return images


def create_images_from_simulated_boot_images(freqs, stokes, base_path=None):

    print "Stacking bootstrapped images..."
    images = Images()
    for freq in freqs:
        print " for freq", freq
        for stoke in stokes:
            glob_name = 'simul_boot_cc_{}_{}_*[0-9].fits'.format(freq, stoke)
            print glob_name
            images.add_from_fits(wildcard=os.path.join(base_path, glob_name))

    return images


def is_embraced(curve, means, widths):
    """
    Function that checks if 1D curve is embraced by 1D band.
    :param curve:
        Iterable of curve's points.
    :param means:
        Iterable of band's means.
    :param widths:
        Iterable of widths (sigmas).
    :return:
        ``True`` if curve is embraced by band.
    """
    curve = np.array(curve)
    means = np.array(means)
    widths = np.array(widths)
    assert len(curve) == len(means) == len(widths)
    diff = np.abs(curve - means)

    return np.alltrue(diff < widths)


def count_contained(curves, means, widths):
    """
    Count haw many curves are contained inside given band.
    :param curves:
        Iterable of numpy 1D arrays with curves to count.
    :param means:
        Iterable of band's means.
    :param widths:
        Iterable of widths (sigmas).
    :return:
        Number of curves within band.
    """
    i = 0
    for curve in curves:
        if is_embraced(curve, means, widths):
            i += 1
    return i


def create_sim_conf_band(curves, means, widths, alpha=0.95, delta=0.01):
    """
    Function that builds simultanious confidence band.
    :param curves:
        Iterable of numpy 1D arrays with curves to count.
    :param means:
        Iterable of band's means.
    :param widths:
        Iterable of widths (sigmas).
    :param alpha: (optional)
        Number in (0., 1.) - ``1 - significance`` of CB. (default: ``0.95``)
    :param delta: (optional)
        Step of widening of band. (default: ``0.01``)
    :return:
        Two 1D numpy arrays with low and upper alpha-level simultaneous
        confidence bands.
    """
    n = count_contained(curves, means, widths)
    f = 1
    while n < len(curves) * alpha:
        f += delta
        n = count_contained(curves, means, f * widths)
    return means - f * widths, means + f * widths


if __name__ == '__main__':

    n_boot = 100

    # MOJAVE
    # C - 4.6&5GHz, X - 8.11&8.43GHz, U - 15.4GHz
    # Bands must be sorted with lowest frequency first
    # 8.1, 8.4, 12.1, 15
    # bands = ['x', 'y', 'j', 'u']
    # stokes = ['i', 'q', 'u']
    # # 1633
    bands = ['l22', 'l21', 'l20', 'l18']
    stokes = ['i', 'q', 'u']

    # 3c273
    # sources = ['1226+023']
    # epochs = ['2006_03_09', '2006_06_15']
    # 2230+114
    # sources = ['2230+114']
    # epochs = ['2006_02_12']
    # 1633+382
    sources = ['1633+382']
    epochs = ['2010_05_21']

    # Directories that contain data for loading in project
    # 3c273
    # uv_data_dir = '/home/ilya/data/3c273'
    # im_data_dir = '/home/ilya/data/3c273'
    # Path to project's root directory
    # base_path = '/home/ilya/data/3c273'
    # # 2230+114
    # uv_data_dir = '/home/ilya/data/2230'
    # im_data_dir = '/home/ilya/data/2230'
    # # Path to project's root directory
    # base_path = '/home/ilya/data/2230'
    # # 1633+382
    uv_data_dir = '/home/ilya/data/1633'
    im_data_dir = '/home/ilya/data/1633'
    base_path = '/home/ilya/data/1633'


    path_to_script = '/home/ilya/data/final_clean_nw'

    # create_dirtree(sources, epochs, bands, stokes, base_path=base_path)
    # put_uv_files_to_dirs(sources, epochs, bands, base_path=base_path,
    #                      ext="uvf", uv_files_path=uv_data_dir)
    # put_im_files_to_dirs(sources, epochs, bands, stokes, base_path=base_path,
    #                      ext="fits", im_files_path=im_data_dir)

    # # Now clean Q&U[&I] stokes with native resolution to use this models for
    # # bootstrap
    # for source in sources:
    #     print "Cleaning source {}".format(source)
    #     for epoch in epochs:
    #         print "epoch {}".format(epoch)
    #         for band in bands:
    #             print "band {}".format(band)
    #             uv_path = uv_fits_path(source, band.upper(), epoch,
    #                                    base_path=base_path)
    #             # Find mapsize from I
    #             map_path = im_fits_path(source, band, epoch, stoke='i',
    #                                     base_path=base_path)
    #             try:
    #                 map_info = get_fits_image_info(map_path + 'cc.fits')
    #             except IOError:
    #                 print "Haven't found map for band {}".format(band)
    #                 # FIXME: OK for close frequencies
    #                 # continue
    #             mapsize_clean = (map_info[0][0], map_info[-3][0] / mas_to_rad)
    #             print "mapsize {}".format(mapsize_clean)
    #             for stoke in ('i', 'q', 'u'):
    #                 if stoke not in stokes:
    #                     print "Skipping stokes {}".format(stoke)
    #                     continue
    #                 print "Stokes {}".format(stoke)
    #                 outpath = im_fits_path(source, band, epoch, stoke,
    #                                        base_path=base_path)

    #                 clean_difmap('sc_uv.fits', 'cc.fits', stoke, mapsize_clean,
    #                              path=uv_path, path_to_script=path_to_script,
    #                              outpath=outpath, show_difmap_output=True)

    # generate_boot_data(sources, epochs, bands, stokes, n_boot=n_boot,
    #                    base_path=base_path)
    # clean_boot_data(sources, epochs, bands, stokes, base_path=base_path,
    #                 path_to_script=path_to_script)

    # Workflow for one source
    # 3c273
    # epoch = '2006_03_09'
    # source = '1226+023'
    # 2230+114
    # epoch = '2006_02_12'
    # source = '2230+114'
    # 1633+382
    epoch = '2010_05_21'
    source = '1633+382'

    # ==========================================================================
    # Find bootstrap error-map of stokes 'I'
    # # 3c273
    # im_fits_path_ = im_fits_path(source, 'x', epoch, 'i', base_path=base_path)
    # 1633
    # im_fits_path_ = im_fits_path(source, 'l22', epoch, 'i', base_path=base_path)
    # image = create_image_from_fits_file(os.path.join(im_fits_path_, 'cc.fits'))
    # Find rms
    # 1633
    # rms = image.rms(region=(50, 50, 50, None))
    # 3c273
    # rms = image.rms(region=(100, 100, 100, None))
    # print "imsize {}".format(image.imsize)
    # print "RMS = {}".format(rms)

    # # 3c273
    # images = create_images_from_boot_images(source, epoch, ['x'], ['i'],
    #                                         base_path=base_path)
    # 1633
    # images = create_images_from_boot_images(source, epoch, ['l22'], ['i'],
    #                                         base_path=base_path)
    # 1633
    # images.pixels_histogram(stokes='I', region=(256, 256, 0, None),
    #                         outdir=base_path, outname="pixel_hist_1.png")
    # images.pixels_histogram(stokes='I', region=(285, 260, 4, None), mode='sum',
    #                         outdir=base_path, outname="pixel_hist_sum.png")
    # error_image = images.create_error_image()

    # color_mask = image.image < 2. * rms
    # # 3c273
    # plot(contours=image.image, colors=error_image.image / rms,
    #      colors_mask=color_mask, min_rel_level=0.75, x=image.x[0],
    #      y=image.y[:, 0], outfile='i_error', outdir=base_path, blc=(450, 300),
    #      trc=(800, 600))
    # 1633
    # plot(contours=image.image, colors=error_image.image / rms,
    #      colors_mask=color_mask, min_rel_level=0.05, x=image.x[0],
    #      y=image.y[:, 0], outfile='i_error', outdir=base_path, blc=(200, 200),
    #      trc=(450, 400))
    # # ==========================================================================

    # # ==========================================================================
    # # Find core shift between each pair of frequencies
    # low_band = 'x'
    # high_band = 'j'
    # im_fits_path_low = im_fits_path(source, low_band, epoch, stoke='i',
    #                                 base_path=base_path)
    # im_fits_path_high = im_fits_path(source, high_band, epoch, stoke='i',
    #                                  base_path=base_path)
    # image_low = create_image_from_fits_file(os.path.join(im_fits_path_low,
    #                                                      'cc_orig.fits'))
    # image_high = create_image_from_fits_file(os.path.join(im_fits_path_high,
    #                                                       'cc_orig.fits'))
    # shifts_orig = list()
    # for r in range(0, 100, 5):
    #     region = (image_low.imsize[0] / 2, image_low.imsize[0] / 2, r, None)
    #     shift_orig = image_low.cross_correlate(image_high, region1=region,
    #                                            region2=region)
    #     shifts_orig.append(shift_orig)
    # shifts_orig = np.vstack(shifts_orig)
    # shifts_orig = shifts_orig[:, 0] + 1j * shifts_orig[:, 1]

    # # Find bootstrapped distribution of shifts
    # shifts_dict_boot = dict()
    # for j in range(1, n_boot+1):
    #     print "Finding shifts for bootstrap images #{}".format(j)
    #     image_low = create_image_from_fits_file(os.path.join(im_fits_path_low,
    #                                                          'cc_{}.fits'.format(j)))
    #     image_high = create_image_from_fits_file(os.path.join(im_fits_path_high,
    #                                                           'cc_{}.fits'.format(j)))
    #     shift_boot = list()
    #     for r in range(0, 100, 5):
    #         region = (image_low.imsize[0] / 2, image_low.imsize[0] / 2, r, None)
    #         shift = image_low.cross_correlate(image_high, region1=region,
    #                                           region2=region)
    #         shift_boot.append(shift)
    #     shift_boot = np.vstack(shift_boot)
    #     shift_boot = shift_boot[:, 0] + 1j * shift_boot[:, 1]

    #     shifts_dict_boot.update({j: shift_boot})

    # from cmath import polar
    # polar = np.vectorize(polar)

    # # Plot all shifts
    # matplotlib.pyplot.figure()
    # for i, shifts in shifts_dict_boot.items():
    #     matplotlib.pyplot.plot(range(0, 100, 5), polar(shifts)[0], '.k')
    # matplotlib.pyplot.plot(range(0, 100, 5), polar(shifts_orig)[0], '.r')
    # matplotlib.pyplot.xlabel("R of mask, [pix]")
    # matplotlib.pyplot.ylabel("shift value, [pix]")
    # matplotlib.pyplot.savefig(os.path.join(base_path,
    #                           "shifts_{}_{}_{}_{}.png".format(source, epoch,
    #                                                           low_band,
    #                                                           high_band)),
    #                           bbox_inches='tight', dpi=200)
    # matplotlib.pyplot.close()
    # hist_shifts = [polar(shifts)[0][11] for i, shifts in
    #                shifts_dict_boot.items()]
    # matplotlib.pyplot.hist(hist_shifts, bins=15, normed=True)
    # matplotlib.pyplot.axvline(polar(shifts_orig)[0][11], lw=2, color='r')
    # matplotlib.pyplot.savefig(os.path.join(base_path,
    #                           "shift_{}_{}_{}_{}.png".format(source, epoch,
    #                                                          low_band,
    #                                                          high_band)),
    #                           bbox_inches='tight', dpi=200)
    # matplotlib.pyplot.close()
    # ==========================================================================

    # # ==========================================================================
    # # For each frequency create mask based on PPOL distribution
    # print "For each frequency create mask based on PPOL distribution"
    # ppol_error_images_dict = dict()
    # pang_error_images_dict = dict()
    # ppol_images_dict = dict()
    # pang_images_dict = dict()
    # ppol_masks_dict = dict()
    # for band in bands:
    #     images_ = create_images_from_boot_images(source, epoch, [band], stokes,
    #                                              base_path=base_path)
    #     ppol_images = Images()
    #     pang_images = Images()
    #     ppol_images.add_images(images_.create_pol_images())
    #     pang_images.add_images(images_.create_pang_images())
    #     ppol_error_image = ppol_images.create_error_image(cred_mass=0.99)
    #     pang_error_image = pang_images.create_error_image(cred_mass=0.68)
    #     ppol_error_images_dict.update({band: ppol_error_image})
    #     pang_error_images_dict.update({band: pang_error_image})
    #     images_ = Images()
    #     for stoke in stokes:
    #         map_path = im_fits_path(source, band, epoch, stoke,
    #                                 base_path=base_path)
    #         images_.add_from_fits(wildcard=os.path.join(map_path,
    #                                                     'cc_orig.fits'))
    #     ppol_image = images_.create_pol_images()[0]
    #     ppol_images_dict.update({band: ppol_image})
    #     mask = ppol_image.image < ppol_error_image.image
    #     ppol_masks_dict.update({band: mask})

    # # Create overall mask for PPOL flux
    # print "Create overall mask for PPOL flux"
    # masks = [np.array(mask, dtype=int) for mask in ppol_masks_dict.values()]
    # ppol_mask = np.zeros(masks[0].shape, dtype=int)
    # for mask in masks:
    #     ppol_mask += mask
    # ppol_mask[ppol_mask != 0] = 1
    # # Save mask to disk
    # np.savetxt(os.path.join(base_path, "1633_ppol_mask_99.txt"), ppol_mask)
    # ==========================================================================

    # ==========================================================================
    # Create bootstrap ROTM images with calculated mask
    # print "Create bootstrap ROTM images with calculated mask"
    # ppol_mask = np.loadtxt(os.path.join(base_path, "1633_ppol_mask_95.txt"))
    # rotm_images_list = list()
    # for i in range(1, n_boot + 1):
    #     images = Images()
    #     for band in bands:
    #         for stoke in stokes:
    #             map_path = im_fits_path(source, band, epoch, stoke,
    #                                     base_path=base_path)
    #             fname = os.path.join(map_path, "cc_{}.fits".format(i))
    #             images.add_from_fits(fnames=[fname])
    #     rotm_image, s_rotm_image = images.create_rotm_image(mask=ppol_mask)
    #     rotm_images_list.append(rotm_image)

    # import pickle
    # f = open(os.path.join(base_path, '1633_rm_images_list_95.pkl'), 'wb')
    # print "Dumping to file bootstrapped ROTM images..."
    # pickle.dump(rotm_images_list, f)
    # f.close()


    ## Stack ROTM images
    #rotm_images_boot = Images()
    #rotm_images_boot.add_images(rotm_images_list)
    #fig = plt.figure()
    #for image in rotm_images_boot.images:
    #    plt.plot(np.arange(500, 550, 1), image.slice((550, 500), (550, 550)),
    #             '.k')

    # # Plot I, ROTM image
    # i_path = im_fits_path(source, bands[-1], epoch, 'i', base_path=base_path)
    # i_image = create_clean_image_from_fits_file(os.path.join(i_path,
    #                                                          'cc_orig.fits'))
    # # Create original ROTM image
    # rotm_images = Images()
    # for band in bands:
    #     for stoke in stokes:
    #         map_path = im_fits_path(source, band, epoch, stoke,
    #                                 base_path=base_path)
    #         fname = os.path.join(map_path, "cc_orig.fits")
    #         images.add_from_fits(fnames=[fname])
    # s_pang_arrays = [pang_error_images_dict[band].image for band in bands]
    # rotm_image, s_rotm_image = images.create_rotm_image(s_pang_arrays=s_pang_arrays,
    #                                                     mask=ppol_mask)
    # plot(contours=i_image.image, colors=rotm_image.image[::-1, ::-1],
    #      min_rel_level=0.5, x=i_image.x[0], y=i_image.y[:, 0])
    # ==========================================================================

    # TODO: Use it for 1633 with 512x512 map size
    # ==========================================================================
    # Simulate gradient
    grad_value = 50.
    rm_value_0 = 100.
    noise_factor = 6.
    # # Width  of jet in beams
    width = 0.5
    # # Length of jet in beams
    length = 4.
    # # How much model pixels in units of high frequency map pixels to use
    k = 2
    high_freq_map = os.path.join(im_fits_path(source, bands[-1], epoch, 'i',
                                              base_path=base_path), 'cc.fits')
    # # high_freq_map = os.path.join(base_path, '1226+023.u.2006_03_09.icn.fits')
    low_freq_map = os.path.join(im_fits_path(source, bands[0], epoch, 'i',
                                             base_path=base_path), 'cc.fits')
    # # low_freq_map = os.path.join(base_path, '1226+023.x.2006_03_09.i_0.1.fits')
    uvdata_files = [os.path.join(uv_fits_path(source, band, epoch,
                                              base_path=base_path),
                                 'sc_uv.fits') for band in bands]
    # # uvdata_files = [os.path.join(base_path, fname) for fname in
    # #                 ('1226+023.x.2006_03_09.uvf', '1226+023.y.2006_03_09.uvf',
    # #                  '1226+023.j.2006_03_09.uvf', '1226+023.u.2006_03_09.uvf')]
    cc_flux = 0.20
    outpath = os.path.join(base_path,
                           'simdata_n{}_w{}_v{}/'.format(noise_factor, width,
                                                         grad_value))
    # if not os.path.exists(outpath):
    #     os.makedirs(outpath)
    # print "Simulating gradient data..."
    # simulate_grad(low_freq_map, high_freq_map, uvdata_files, cc_flux=cc_flux,
    #               outpath=outpath, grad_value=grad_value, width=width,
    #               length=length, k=k, noise_factor=noise_factor,
    #               rm_value_0=rm_value_0)
    # # ==========================================================================

    # ==========================================================================
    # Clean simulated uv-data
    print "Cleaning simulated uv-data..."
    map_info_l = get_fits_image_info(low_freq_map)
    map_info_h = get_fits_image_info(high_freq_map)
    beam_restore = map_info_l[3]
    beam_restore_ = (beam_restore[0] / mas_to_rad, beam_restore[1] / mas_to_rad,
                     beam_restore[2])
    mapsize_clean = (map_info_h[0][0],
                     map_info_h[-3][0] / mas_to_rad)
    for uvpath in glob.glob(os.path.join(outpath, "simul_uv_*")):
        uvdir, uvfile = os.path.split(uvpath)
        print "Cleaning uv file {}".format(uvpath)
        uvdata = UVData(uvpath)
        freq_card = find_card_from_header(uvdata._io.hdu.header,
                                          value='FREQ')[0]
        # Frequency in Hz
        freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        for stoke in ('i', 'q', 'u'):
            print "Stokes {}".format(stoke)
            clean_difmap(uvfile, "simul_{}_{}_cc.fits".format(stoke, freq),
                         stoke, mapsize_clean, path=uvdir,
                         path_to_script=path_to_script,
                         beam_restore=beam_restore_, outpath=outpath)
    # ==========================================================================

    # ==========================================================================
    # Create image of ROTM for simulated cleaned data
    print "Creating image of simulated ROTM"
    images = Images()
    images.add_from_fits(wildcard=os.path.join(outpath, "simul_*_cc.fits"))
    # Create mask for faster calculation
    # mask = np.ones((1024, 1024), dtype=int)
    mask = np.ones((512, 512), dtype=int)
    mask[100:400, 100:400] = 0
    rotm_image, s_rotm_image = images.create_rotm_image(mask=mask)
    matplotlib.pyplot.matshow(rotm_image.image)
    # Plot slice
    matplotlib.pyplot.errorbar(np.arange(210, 302, 1),
                               rotm_image.slice((240, 210), (240, 302)),
                               s_rotm_image.slice((240, 210), (240, 302)),
                               fmt='.k')
    # Plot real ROTM grad values
    (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
     stokes_l, freq_l) = get_fits_image_info(low_freq_map)
    # Jet width in pixels
    jet_width = width * bmaj_l / abs(rotm_image.pixsize[0])

    # Analytical gradient in real image (didn't convolved)
    def rm(x, y, grad_value, rm_value_0=0.0):
        k = grad_value / (bmaj_l/abs(rotm_image.pixsize[0]))
        return k * x + rm_value_0

    matplotlib.pyplot.plot(np.arange(210, 302, 1),
                           rm(np.arange(210, 302, 1) - rotm_image.pixref[1],
                              None, grad_value, rm_value_0=rm_value_0))
    matplotlib.pyplot.axvline(rotm_image.pixref[1] - jet_width / 2.)
    matplotlib.pyplot.axvline(rotm_image.pixref[1] + jet_width / 2.)
    # ==========================================================================

    # ==========================================================================
    # Generate bootstrapped-data
    print "Generating bootstrapped data from simulated data..."
    freqs = list()
    uv_fits_paths = glob.glob(os.path.join(outpath, 'simul_uv*_Hz.fits'))
    for uv_fits_path in uv_fits_paths:
        _, uv_fits_fname = os.path.split(uv_fits_path)
        freq = uv_fits_fname.split('_')[2]
        freqs.append(freq)
    freqs.sort()
    generate_simulated_boot_data(freqs, ('i', 'q', 'u'), n_boot=n_boot,
                                 base_path=outpath)
    clean_sumulated_boot_data(freqs, ('i', 'q', 'u'), base_path=outpath,
                              path_to_script=path_to_script)
    # ==========================================================================

    # ==========================================================================
    # For each frequency create mask of simulated maps based on PPOL
    # distribution
    print "For each frequency create mask based on PPOL distribution"
    ppol_error_images_dict = dict()
    pang_error_images_dict = dict()
    ppol_images_dict = dict()
    pang_images_dict = dict()
    ppol_masks_dict = dict()
    for freq in freqs:
        print "freq {}".format(freq)
        images_ = create_images_from_simulated_boot_images([freq],
                                                           ('i', 'q', 'u'),
                                                           base_path=outpath)
        ppol_images = Images()
        # pang_images = Images()
        ppol_images.add_images(images_.create_pol_images())
        # pang_images.add_images(images_.create_pang_images())
        ppol_error_image = ppol_images.create_error_image(cred_mass=0.99)
        # pang_error_image = pang_images.create_error_image(cred_mass=0.68)
        ppol_error_images_dict.update({freq: ppol_error_image})
        # pang_error_images_dict.update({freq: pang_error_image})
        images_ = Images()
        for stoke in stokes:
            orig_map_name = 'simul_{}_{}_cc.fits'.format(stoke, freq)
            images_.add_from_fits(wildcard=os.path.join(outpath,
                                                        orig_map_name))
        ppol_image = images_.create_pol_images()[0]
        ppol_images_dict.update({freq: ppol_image})
        mask = ppol_image.image < ppol_error_image.image
        ppol_masks_dict.update({freq: mask})

    # Create overall mask for PPOL flux
    print "Create overall mask for PPOL flux"
    masks = [np.array(mask, dtype=int) for mask in ppol_masks_dict.values()]
    ppol_mask = np.zeros(masks[0].shape, dtype=int)
    for mask in masks:
        ppol_mask += mask
    ppol_mask[ppol_mask != 0] = 1
    # Save mask to disk
    print "Saving ppol mask on disk..."
    np.savetxt(os.path.join(outpath, "ppol_mask_99.txt"), ppol_mask)
    # ==========================================================================

    # # ==========================================================================
    # # Create bootstrap ROTM images with calculated mask
    # print "Create bootstrap silumated ROTM images with calculated mask"
    ppol_mask = np.loadtxt(os.path.join(outpath, "ppol_mask_99.txt"))
    rotm_images_list = list()
    for i in range(1, n_boot + 1):
        images = Images()
        for freq in freqs:
            for stoke in stokes:
                fname = os.path.join(outpath,
                                     'simul_boot_cc_{}_{}_{}.fits'.format(freq,
                                                                          stoke,
                                                                          i))
                images.add_from_fits(fnames=[fname])
        rotm_image, s_rotm_image = images.create_rotm_image(mask=ppol_mask)
        rotm_images_list.append(rotm_image)

    import pickle
    f = open(os.path.join(outpath, 'rm_images_list_99.pkl'), 'wb')
    print "Dumping to file bootstrapped simulated ROTM images..."
    pickle.dump(rotm_images_list, f)
    f.close()
    # ==========================================================================

    # ==========================================================================
    # Stack ROTM images
    rotm_images_boot = Images()
    rotm_images_boot.add_images(rotm_images_list)
    fig = matplotlib.pyplot.figure()
    slices = list()
    for image in rotm_images_boot.images:
        matplotlib.pyplot.plot(np.arange(210, 302, 1) +
                               np.random.normal(0, 0.05, 92),
                               image.slice((228, 210), (228, 302)), 'k', lw=0.1)
        slice_ = image.slice((228, 210), (228, 302))
        slices.append(slice_[~np.isnan(slice_)])
    # Plot real ROTM grad values
    (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
     stokes_l, freq_l) = get_fits_image_info(low_freq_map)
    # Jet width in pixels
    jet_width = width * bmaj_l / abs(rotm_image.pixsize[0])

    # Analytical gradient in real image (didn't convolved)
    def rm(x, y, grad_value, rm_value_0=0.0):
        k = grad_value / (bmaj_l/abs(rotm_image.pixsize[0]))
        # return k * x + rm_value_0
        return -k * x + rm_value_0

    matplotlib.pyplot.plot(np.arange(210, 302, 1),
                           rm(np.arange(210, 302, 1) - rotm_image.pixref[1],
                              None, grad_value, rm_value_0=rm_value_0))
    matplotlib.pyplot.axvline(rotm_image.pixref[1] - jet_width / 2.)
    matplotlib.pyplot.axvline(rotm_image.pixref[1] + jet_width / 2.)

    # Plot simulated RM slice with CB
    # Create image of ROTM for simulated cleaned data
    print "Creating image of simulated ROTM"
    images = Images()
    images.add_from_fits(wildcard=os.path.join(outpath, "simul_*_cc.fits"))
    rotm_image, s_rotm_image = images.create_rotm_image(mask=ppol_mask)
    obs_slice = rotm_image.slice((228, 210), (228, 302))


    x = np.arange(210, 302, 1)
    x = x[~np.isnan(obs_slice)]


    obs_slice = obs_slice[~np.isnan(obs_slice)]

    print "Find 68% cred. intervals from bootstrapped slices"
    slices_ = [arr.reshape((1, len(obs_slice))) for arr in slices]
    sigmas = hdi_of_arrays(slices_).squeeze()

    print "Find means of bootstrapped distribution"
    means = np.mean(np.vstack(slices), axis=0)
    print "Move each bootstrap slice on diff between bootstrap means and " \
          "original slice"
    diff = obs_slice - means
    # Move bootstrap curves to original simulated centers
    slices_ = [slice_ + diff for slice_ in slices]

    # Find low and upper confidence band
    low, up = create_sim_conf_band(slices_, obs_slice, sigmas, alpha=0.99)

    # Plot slice with confidence band
    sl_len = len(obs_slice)
    matplotlib.pyplot.plot(x, low[::-1], 'g')
    matplotlib.pyplot.plot(x, up[::-1], 'g')
    [matplotlib.pyplot.plot(x, slice_[::-1], 'r', lw=0.15) for slice_ in
     slices_]
    matplotlib.pyplot.plot(x, obs_slice[::-1], '.k')
    matplotlib.pyplot.plot(x, rm(x - rotm_image.pixref[1], None, grad_value,
                                 rm_value_0=100))
    matplotlib.pyplot.axvline(rotm_image.pixref[1] - jet_width / 2.)
    matplotlib.pyplot.axvline(rotm_image.pixref[1] + jet_width / 2.)
    matplotlib.pyplot.savefig(os.path.join(base_path, "RM_slice_CB.png"),
                              bbox_inches='tight', dpi=200)

    # # Plot I, ROTM image
    # i_path = im_fits_path(source, bands[-1], epoch, 'i', base_path=base_path)
    # i_image = create_clean_image_from_fits_file(os.path.join(i_path,
    #                                                          'cc_orig.fits'))
    # # Create original ROTM image
    # rotm_images = Images()
    # for band in bands:
    #     for stoke in stokes:
    #         map_path = im_fits_path(source, band, epoch, stoke,
    #                                 base_path=base_path)
    #         fname = os.path.join(map_path, "cc_orig.fits")
    #         images.add_from_fits(fnames=[fname])
    # s_pang_arrays = [pang_error_images_dict[band].image for band in bands]
    # rotm_image, s_rotm_image = images.create_rotm_image(s_pang_arrays=s_pang_arrays,
    #                                                     mask=ppol_mask)
    # plot(contours=i_image.image, colors=rotm_image.image[::-1, ::-1],
    #      min_rel_level=0.5, x=i_image.x[0], y=i_image.y[:, 0])
    # ==========================================================================
