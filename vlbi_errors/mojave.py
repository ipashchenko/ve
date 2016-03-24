import os
import urllib
import BeautifulSoup
import urllib2
import fnmatch

mojave_multifreq_url = "http://www.cv.nrao.edu/2cmVLBA/data/multifreq/"
# Path to u-frequency file: dir/source/epoch/fname
mojave_u_url = "http://www.cv.nrao.edu/2cmVLBA/data/"
mojave_l_url = "http://www.cv.nrao.edu/MOJAVELBAND"

download_dir = '/home/ilya/code/vlbi_errors/examples/'
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
                epoch = str(a['href'].strip('/'))
                sources.append(epoch)
    if use_db == 'multifreq':
        for a in soup.findAll('a'):
            if 'uvf' in a['href']:
                fname = a['href']
                sources.append(str(fname.split('.')[0]))
    return sorted(set(sources))


def get_epochs_for_source(source, use_db='u'):
    pass


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
        response = urllib2.urlopen(request)
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
    print sources
