import os
import shutil
from from_fits import (create_uvdata_from_fits_file,
                       create_ccmodel_from_fits_file)
from bootstrap import CleanBootstrap

# TODO: We need to get RM map and it's uncertainty for each source.
# Input: calibrated visibilities, CLEAN models in "naitive" resolution.
# Maps on higher frequencies are made by:
#     1) convolving clean model with low-frequency beam
#     2) cleaning uv-data using low-frequency beam and parameters of
#         low-frequency CC-maps.
# I think i should use 2) because output of bootstrap - set of resampled
# uv-data - and i should use "naitive" CC-model for resampling.
# Then shift all low frequency CC-maps by specified shift.

# FIXME: Actually, this shift should be calculated between sets of resampled
# imaged data to obtain the distribution of shifts.


# C - 4.6&5GHz, X - 8.11&8.43GHz, U - 15.4GHz
bands = ['c1', 'c2', 'x1', 'x2', 'u1']
epochs = ['2007_03_01', '2007_04_30', '2007_05_03', '2007_06_01']
sources = ['0148+274',
           '0342+147',
           '0425+048',
           '0507+179',
           '0610+260',
           '0839+187',
           '0923+392',
           '0952+179',
           '1004+141',
           '1011+250',
           '1049+215',
           '1219+285',
           '1226+023',
           '1406-076',
           '1458+718',
           '1642+690',
           '1655+077',
           '1803+784',
           '1830+285',
           '1845+797',
           '2201+315',
           '2320+506']

stokes = ['i', 'q', 'u']


def im_fits_fname(source, band, epoch, stokes, ext='fits'):
    return source + '.' + band + '.' + epoch + '.' + stokes + '.' + ext


def uv_fits_fname(source, band, epoch, ext='fits'):
    return source + '.' + band + '.' + epoch + '.' + ext


def uv_fits_path(source, band, epoch, base_path=None):
    """
    Function that returns path to uv-file for given source, epoch and band.

    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    """
    return base_path + source + '/' + epoch + '/' + band + '/uv/'


def im_fits_path(source, band, epoch, stoke, base_path=None):
    """
    Function that returns path to im-file for given source, epoch, band and
    stokes parameter.

    :param base_path: (optional)
        Path to route of directory tree. If ``None`` then use current directory.
        (default: ``None``)
    """
    return base_path + source + '/' + epoch + '/' + band.upper() + '/im/' +\
           stoke.upper() + '/'


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
        base_path = base_path + "/"
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
                for dir in stokes + ['alpha', 'pol', 'fpol', 'rm']:
                    os.mkdir(dir)
                os.chdir(os.path.join(os.path.pardir, os.curdir))
                os.chdir(os.path.join(os.path.pardir, os.curdir))
            os.chdir(os.path.join(os.path.pardir, os.curdir))
        os.chdir(os.path.join(os.path.pardir, os.curdir))


def put_uv_files_to_dirs(sources, epochs, bands, base_path=None, ext="PINAL",
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
    bands = [band.upper() for band in bands]
    if base_path is None:
        base_path = os.getcwd()
    elif not base_path.endswith("/"):
        base_path = base_path + "/"
    if uv_files_path is None:
        uv_files_path = os.getcwd()
    elif not uv_files_path.endswith("/"):
        uv_files_path = uv_files_path + "/"

    # Circle through sources, epochs and bands and copy files to directory tree.
    for source in sources:
        for epoch in epochs:
            for band in bands:
                fname = uv_fits_fname(source, band, epoch, ext="PINAL")
                outpath = uv_fits_path(source, band, epoch, base_path=base_path)
                try:
                    shutil.copyfile(uv_files_path + fname,
                                    outpath + 'sc_uv.fits')
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
    elif not base_path.endswith("/"):
        base_path = base_path + "/"
    if im_files_path is None:
        im_files_path = os.getcwd()
    elif not im_files_path.endswith("/"):
        im_files_path = im_files_path + "/"

    # Circle through sources, epochs and bands and copy files to directory tree.
    for source in sources:
        for epoch in epochs:
            for band in bands:
                for stoke in stokes:
                    fname = im_fits_fname(source, band, epoch, stoke, ext=ext)
                    outpath = im_fits_path(source, band, epoch, stoke,
                                           base_path=base_path)
                    try:
                        shutil.copyfile(im_files_path + fname,
                                        outpath + 'cc.fits')
                        print "Copied file ", fname
                        print "from ", im_files_path, " to ", outpath
                    except IOError:
                        print "No such file ", fname, " in ", im_files_path


def generate_boot_data(sources, epochs, bands, stokes, base_path=None):
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

    """
    if base_path is None:
        base_path = os.getcwd()
    elif not base_path.endswith("/"):
        base_path = base_path + "/"

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
                print "  Using uv-file ", uv_fname
                uvdata = create_uvdata_from_fits_file(uv_fname)
                models = list()
                for stoke in stokes:
                    print "  working with stokes parameter ", stoke
                    map_path = im_fits_path(source, band, epoch, stoke,
                                            base_path=base_path)
                    map_fname = map_path + 'cc.fits'
                    print "  Using CC-model file ", map_fname
                    ccmodel = create_ccmodel_from_fits_file(map_fname,
                                                            stokes=stoke.upper())
                    models.append(ccmodel)
                boot = CleanBootstrap(models, uvdata)
                os.chdir(uv_path)
                boot.run(n=10, outname=['boot', ''])
    os.chdir(curdir)


if __name__ == '__main__':

    # Directories that contain data for loading in project
    uv_data_dir = '/home/ilya/Dropbox/Zhenya/to_ilya/uv/'
    im_data_dir = '/home/ilya/Dropbox/Zhenya/to_ilya/clean_images/'
    # Path to project's root directory
    base_path = '/home/ilya/code/vlbi_errors/vlbi_errors/test/'

    create_dirtree(sources, epochs, bands, stokes, base_path=base_path)
    put_uv_files_to_dirs(sources, epochs, bands, base_path=base_path,
                         ext="PINAL", uv_files_path=uv_data_dir)
    put_im_files_to_dirs(sources, epochs, bands, stokes, base_path=base_path,
                         ext="fits", im_files_path=im_data_dir)
    generate_boot_data(sources, epochs, bands, stokes, base_path=base_path)
