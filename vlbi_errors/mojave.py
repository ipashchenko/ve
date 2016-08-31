import os
import urllib
import BeautifulSoup
import urllib2
import fnmatch
import numpy as np
import sys

from spydiff import clean_difmap
from from_fits import create_image_from_fits_file, \
    create_clean_image_from_fits_file
from uv_data import UVData


# TODO: check connection to MOJAVE servers
mojave_multifreq_url = "http://www.cv.nrao.edu/2cmVLBA/data/multifreq/"
# Path to u-frequency file: dir/source/epoch/fname
mojave_u_url = "http://www.cv.nrao.edu/2cmVLBA/data/"
mojave_l_url = "http://www.cv.nrao.edu/MOJAVELBAND"

download_dir = '/home/ilya/code/vlbi_errors/examples/mojave'
# x - 8.1, y - 8.4, j - 12.1, u - 15 GHz
pixel_size_dict = {'x': None, 'y': None, 'j': None, 'u': None}
# epoch format: YYYY-MM-DD
mojave_bands = ['x', 'y', 'j', 'u', 'l18', 'l20', 'l21', 'l22']
l_bands = ['l18', 'l20', 'l21', 'l22']


def mojave_uv_fits_fname(source, band, epoch, ext='uvf'):
    return source + '.' + band + '.' + epoch + '.' + ext


def get_all_mojave_sources(use_db='u'):
    url_dict = {'u': mojave_u_url, 'multifreq': mojave_multifreq_url}
    request = urllib2.Request(url_dict[use_db])
    response = urllib2.urlopen(request)
    soup = BeautifulSoup.BeautifulSoup(response)

    sources = list()
    if use_db == 'u':
        for a in soup.findAll('a'):
            if fnmatch.fnmatch(a['href'], "*+*") or fnmatch.fnmatch(a['href'],
                                                                    "*-*"):
                source = str(a['href'].strip('/'))
                sources.append(source)
    if use_db == 'multifreq':
        for a in soup.findAll('a'):
            if 'uvf' in a['href']:
                fname = a['href']
                sources.append(str(fname.split('.')[0]))
    return sorted(set(sources))


def get_epochs_for_source(source, use_db='u'):
    url_dict = {'u': os.path.join(mojave_u_url, source),
                'multifreq': mojave_multifreq_url}
    request = urllib2.Request(url_dict[use_db])
    response = urllib2.urlopen(request)
    soup = BeautifulSoup.BeautifulSoup(response)

    epochs = list()
    if use_db == 'u':
        for a in soup.findAll('a'):
            if fnmatch.fnmatch(a['href'], "*_*_*"):
                epoch = str(a['href'].strip('/'))
                epochs.append(epoch)
    if use_db == 'multifreq':
        for a in soup.findAll('a'):
            if source in a['href'] and 'uvf' in a['href']:
                fname = a['href']
                epochs.append(str(fname.split('.')[2]))
    return sorted(set(epochs))


def download_mojave_uv_fits(source, epochs=None, bands=None, download_dir=None):
    """
    Download FITS-files with self-calibrated uv-data from MOJAVE server.

    :param source:
        Source name [B1950].
    :param epochs: (optional)
        Iterable of epochs to download [YYYY-MM-DD]. If ``None`` then download
        all. (default: ``None``)
    :param bands: (optional)
        Iterable bands to download ('x', 'y', 'j' or 'u'). If ``None`` then
        download all available bands for given epochs. (default: ``None``)
    :param download_dir: (optional)
        Local directory to save files. If ``None`` then use CWD. (default:
        ``None``)
    """
    if bands is None:
        bands = mojave_bands
    else:
        assert set(bands).issubset(mojave_bands)

    if 'u' in bands:
        # Finding epochs in u-band data
        request = urllib2.Request(os.path.join(mojave_u_url, source))
        response = urllib2.urlopen(request)
        soup = BeautifulSoup.BeautifulSoup(response)

        available_epochs = list()
        for a in soup.findAll('a'):
            if fnmatch.fnmatch(a['href'], "*_*_*"):
                epoch = str(a['href'].strip('/'))
                available_epochs.append(epoch)

        if epochs is not None:
            if not set(epochs).issubset(available_epochs):
                raise Exception(" No epochs {} in MOJAVE data."
                                " Available are {}".format(epochs,
                                                           available_epochs))
        else:
            epochs = available_epochs

        # Downloading u-band data
        u_url = os.path.join(mojave_u_url, source)
        for epoch in epochs:
            fname = mojave_uv_fits_fname(source, 'u', epoch)
            url = os.path.join(u_url, epoch, fname)
            print("Downloading file {}".format(fname))
            path = os.path.join(download_dir, fname)
            if os.path.isfile(path):
                print("File {} does exist in {}."
                      " Skipping...".format(fname, download_dir))
                continue
            urllib.urlretrieve(url, path)

    # Downloading (optionally) x, y & j-band data
    request = urllib2.Request(mojave_multifreq_url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup.BeautifulSoup(response)

    download_list = list()
    for a in soup.findAll('a'):
        if source in a['href'] and '.uvf' in a['href']:
            fname = a['href']
            epoch = fname.split('.')[2]
            band = fname.split('.')[1]
            if band in bands:
                if epochs is None:
                    download_list.append(os.path.join(mojave_multifreq_url,
                                                      fname))
                else:
                    if epoch in epochs:
                        download_list.append(os.path.join(mojave_multifreq_url,
                                                          fname))
    for url in download_list:
        fname = os.path.split(url)[-1]
        print("Downloading file {}".format(fname))
        path = os.path.join(download_dir, fname)
        if os.path.isfile(path):
            print("File {} does exist in {}."
                  " Skipping...".format(fname, download_dir))
            continue
        urllib.urlretrieve(url, os.path.join(download_dir, fname))

    # Downloading (optionally) l-band data
    if 'l18' in bands or 'l20' in bands or 'l21' in bands or 'l22' in bands:
        request = urllib2.Request(os.path.join(mojave_l_url, source))
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError:
            print("No L-bands data available")
            return
        soup = BeautifulSoup.BeautifulSoup(response)

        available_epochs = list()
        for a in soup.findAll('a'):
            if fnmatch.fnmatch(a['href'], "*_*_*"):
                epoch = str(a['href'].strip('/'))
                available_epochs.append(epoch)

        if epochs is not None:
            if not set(epochs).issubset(available_epochs):
                raise Exception(" No epochs {} in MOJAVE data")
        else:
            epochs = available_epochs

        # Downloading l-band data
        l_url = os.path.join(mojave_l_url, source)
        for epoch in epochs:
            for band in bands:
                if band in l_bands:
                    fname = mojave_uv_fits_fname(source, band, epoch)
                    url = os.path.join(l_url, epoch, fname)
                    print("Downloading file {}".format(fname))
                    path = os.path.join(download_dir, fname)
                    if os.path.isfile(path):
                        print("File {} does exist in {}."
                              " Skipping...".format(fname, download_dir))
                        continue
                    urllib.urlretrieve(url, os.path.join(download_dir, fname))


def get_stacked_map(source, mojave_dir=None, out_dir=None, imsize=(512, 0.1),
                    path_to_script=None, epochs_slice=None):
    """
    Functions that returns stacked image of given source using MOJAVE 15 GHz
    data downloading it directly from MOJAVE DB or getting it from
    user-specified directory.

    :param source:
        Source name [B1950].
    :param mojave_dir: (optional)
        Path to directory with MOJAVE 15 GHz data. If ``None`` then download
        from MOJAVE DB. (default: ``None``)
    :param out_dir: (optional)
        Directory where to store files. If ``None`` then use CWD. (default:
        ``None``)
    :param imsize: (optional)
        Tuple of image size [pix], pixel size [mas] to put to difmap. (default:
        ``(512, 0.1)``)
    :param path_to_script: (optional)
        Path to difmap CLEANing script. If ``None`` then use CWD. (default:
        ``None``)
    :param epochs_slice: (optional)
        Slice of epochs sorted to process. If ``None`` then use all available
        epochs. (default: ``None``)

    :return:
        Numpy 2D array with stacked image.
    """
    if path_to_script is None:
        path_to_script = os.getcwd()
    epochs = get_epochs_for_source(source)
    if epochs_slice is not None:
        epochs = epochs[::-1][epochs_slice]
    if out_dir is None:
        out_dir = os.getcwd()
    elif not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if mojave_dir is None:
        download_mojave_uv_fits(source, epochs, bands=['u'],
                                download_dir=out_dir)
    else:
        out_dir = mojave_dir
    beams_dict = dict()

    print "Output directory : {}".format(out_dir)

    # First clean and restore with native beam
    epoch_stokes_dict = dict()
    for epoch in sorted(epochs):
        print "Cleaning epoch {} with naitive restoring beam".format(epoch)
        uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
        uvdata = UVData(os.path.join(out_dir, uv_fits_fname))
        uv_stokes = uvdata.stokes
        if 'RR' in uv_stokes and 'LL' in uv_stokes:
            stokes = 'I'
        elif 'RR' in uv_stokes and 'LL' not in uv_stokes:
            stokes = 'RR'
        elif 'LL' in uv_stokes and 'RR' not in uv_stokes:
            stokes = 'LL'
        else:
            continue
        epoch_stokes_dict.update({epoch: stokes})
        im_fits_fname = "{}_{}_{}_{}.fits".format(source, 'U', epoch, stokes)
        print "Difmap params: "
        print "uv_fits_fname : {}".format(uv_fits_fname)
        print "im_fits_fname : {}".format(im_fits_fname)
        print "path : {}".format(out_dir)
        print "outpath: {}".format(out_dir)
        clean_difmap(uv_fits_fname, im_fits_fname, stokes, imsize,
                     path=out_dir, path_to_script=path_to_script,
                     outpath=out_dir)
        ccimage = create_clean_image_from_fits_file(os.path.join(out_dir,
                                                                 im_fits_fname))
        beam = ccimage.beam
        print "Beam for epoch {} : {} [mas, mas, deg]".format(epoch, beam)
        beams_dict.update({epoch: (beam[0], beam[0], 0)})

    circ_beam = np.mean([beam[0] for beam in beams_dict.values()])

    # Now clean and restore with circular beam
    images = list()
    for epoch in sorted(epochs):
        stokes = epoch_stokes_dict[epoch]
        uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
        im_fits_fname = "{}_{}_{}_{}_circ.fits".format(source, 'U', epoch,
                                                       stokes)
        print "Difmap params: "
        print "uv_fits_fname : {}".format(uv_fits_fname)
        print "im_fits_fname : {}".format(im_fits_fname)
        print "path : {}".format(out_dir)
        print "outpath: {}".format(out_dir)
        clean_difmap(uv_fits_fname, im_fits_fname, stokes, imsize,
                     path=out_dir, path_to_script=path_to_script,
                     outpath=out_dir, beam_restore=(circ_beam, circ_beam, 0))
        image = create_image_from_fits_file(os.path.join(out_dir,
                                                         im_fits_fname))
        images.append(image.image.copy())

    images = np.dstack(images)
    return np.mean(images, axis=2)


if __name__ == '__main__':
    # source = '2230+114'
    # source = '1055+018'
    # epochs = ['2010_02_03']
    # epochs = ['2006_02_12']
    # epochs = ['2006_03_09', '2006_06_15']
    # bands = ['l18']
    # bands = None
    # download_mojave_uv_fits(source, epochs=epochs, bands=bands,
    #                         download_dir=download_dir)
    sources = get_all_mojave_sources(use_db='multifreq')
    source_epoch_dict = dict()
    for source in sources:
        print("Querying source {}".format(source))
        epochs = get_epochs_for_source(source, use_db='multifreq')
        source_epoch_dict.update({source: sorted(epochs)[-1]})
    print source_epoch_dict

    source_uv_dict = dict()
    from uv_data import UVData
    for source in sources:
        download_mojave_uv_fits(source, epochs=[source_epoch_dict[source]],
                                bands=['x'], download_dir=download_dir)
        fname = mojave_uv_fits_fname(source, 'x', source_epoch_dict[source])
        source_uv_dict.update({source:
                                   UVData(os.path.join(download_dir, fname))})

    source_dec_dict = dict()
    for source in sources:
        for sign in ('+', '-'):
            if sign in source:
                dec = source.split(sign)[1]
                break
        source_dec_dict.update({source: float(sign + dec)})
    source_dec_list = sorted(source_dec_dict, key=lambda x: source_dec_dict[x])

    grid_sources = list()
    grid_dec = np.arange(-300, 1100, 200)
    for dec in grid_dec:
        print("Searching close to {} declination source".format(dec))
        dec_diffs = abs(np.array(source_dec_dict.values()) - dec)
        indx = np.where(dec_diffs == min(dec_diffs))[0]
        print("Index: {}".format(indx))
        min_dec = source_dec_dict.values()[indx]
        grid_sources.append([source for source, dec in source_dec_dict.items()
                             if dec == min_dec][0])
