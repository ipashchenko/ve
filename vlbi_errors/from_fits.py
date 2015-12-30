import warnings
import numpy as np
import pyfits as pf
from model import Model
from utils import (degree_to_mas, degree_to_rad, find_card_from_header,
                   stokes_dict, AbsentHduExtensionError)
from components import DeltaComponent
from image import Image, CleanImage


def create_model_from_fits_file(fname, ver=1):
    """
    Function that creates instance of ``CleanModel`` from FITS-file with CLEAN
    model
    :param fname:
    :param ver:
    :return:
    """
    hdulist = pf.open(fname)
    return create_model_from_hdulist(hdulist, ver)


# FIXME: Implement adding gaussians
def create_model_from_hdulist(hdulist, ver=1):

    image_params = get_fits_image_info_from_hdulist(hdulist)
    ccmodel = Model(stokes=image_params['stokes'])
    hdu = get_hdu_from_hdulist(hdulist, extname='AIPS CC', ver=ver)
    # TODO: Need this when dealing with IDI UV_DATA extension binary table
    # dtype = build_dtype_for_bintable_data(hdu.header)
    dtype = hdu.data.dtype
    data = np.zeros(hdu.header['NAXIS2'], dtype=dtype)
    for name in data.dtype.names:
        data[name] = hdu.data[name]

    for flux, x, y in zip(data['FLUX'], data['DELTAX'] * degree_to_mas,
                          data['DELTAY'] * degree_to_mas):
        # We keep positions in mas
        component = DeltaComponent(flux, -x, -y)
        ccmodel.add_component(component)
    return ccmodel


def create_clean_image_from_fits_file(fname, ver=1):
    """
    Create instance of ``CleanImage`` from FITS-file of CLEAN image.
    :param fname:
    :return:
        Instance of ``CleanImage``.
    """
    hdulist = pf.open(fname)
    return create_clean_image_from_hdulist(hdulist, ver)


def create_clean_image_from_hdulist(hdulist, ver=1):
    """
    Create instance of ``CleanImage`` from instance of ``Pyfits.HDUList``.
    :param hdulist:
        Instance of ``PyFits.HDUList``.
    :return:
        Instance of ``CleanImage``.
    """
    image_params = get_fits_image_info_from_hdulist(hdulist)

    imsize = image_params['imsize']
    pixref = image_params['pixref']
    pixrefval = image_params['pixrefval']
    pixsize = image_params['pixsize']
    stokes = image_params['stokes']
    freq = image_params['freq']
    # bmaj, deg in rad
    bmaj = image_params['bmaj']
    bmin = image_params['bmin']
    bpa = image_params['bpa']

    model = create_model_from_hdulist(hdulist, stokes=stokes, ver=ver)
    if bmaj is None:
        raise Exception("Can't find Beam info!")
    ccimage = CleanImage(imsize=imsize, pixref=pixref, pixrefval=pixrefval,
                         pixsize=pixsize, bmaj=bmaj, bmin=bmin,
                         bpa=bpa/degree_to_rad, stokes=stokes, freq=freq)
    ccimage.add_model(model)
    image = create_image_from_hdulist(hdulist)
    ccimage._residuals = image.image - ccimage.cc_image
    ccimage._image_original = image.image
    return ccimage


def create_image_from_fits_file(fname):
    """
    Create instance of ``CleanImage`` from FITS-file of CLEAN image.
    :param fname:
    :return:
        Instance of ``CleanImage``.
    """
    hdulist = pf.open(fname)
    return create_image_from_hdulist(hdulist)


def create_image_from_hdulist(hdulist):
    """
    Create instance of ``Image`` from instance of ``PyFits.HDUList``.
    :param hdulist:

    :return:
        Instance of ``CleanImage``.
    """
    image_params = get_fits_image_info_from_hdulist(hdulist)
    image = Image(image_params['imsize'], image_params['pixref'],
                  image_params['pixrefval'], image_params['pixsize'],
                  image_params['stokes'], image_params['freq'])
    pr_hdu = get_hdu_from_hdulist(hdulist)
    image.image = pr_hdu.data.squeeze()
    return image


def get_fits_image_info(fname):
    """
    Returns image parameters from FITS-file.

    :param fname:
        Fits-file name.

    :return:
        Dictionary with following information:
        ``imsize`` [pix, pix] - size of image,
        ``pixref`` [pix, pix] - reference pixel numbers,
        ``pixrefval`` [rad, rad] - value of coordinates at reference pixels,
        ``(bmaj, bmin, bpa,)`` [rad, rad, rad] - beam parameters (if any). If no
        beam parameters found => ``(None, None, None,)``,
        ``pixsize`` [rad, rad]- size of pixel dimensions,
        ``stokes`` (I, Q, U or V) - stokes parameter that image does describe,
        ``freq`` [Hz] - sky frequency.

    """
    hdulist = pf.open(fname)
    return get_fits_image_info_from_hdulist(hdulist)


def get_fits_image_info_from_hdulist(hdulist):
    """
    Returns image parameters from instance of ``PyFits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.

    :return:
        Dictionary with following information:
        ``imsize`` [pix, pix] - size of image,
        ``pixref`` [pix, pix] - reference pixel numbers,
        ``pixrefval`` [rad, rad] - value of coordinates at reference pixels,
        ``(bmaj, bmin, bpa,)`` [rad, rad, rad] - beam parameters (if any). If no
        beam parameters found => ``(None, None, None,)``,
        ``pixsize`` [rad, rad]- size of pixel dimensions,
        ``stokes`` (I, Q, U or V) - stokes parameter that image does describe,
        ``freq`` [Hz] - sky frequency.

    """
    bmaj, bmin, bpa = None, None, None
    pr_header = hdulist[0].header
    imsize = (pr_header['NAXIS1'], pr_header['NAXIS2'],)
    pixref = (int(pr_header['CRPIX1']), int(pr_header['CRPIX2']),)
    pixrefval = (pr_header['CRVAL1'] * degree_to_rad,
                 pr_header['CRVAL2'] * degree_to_rad,)
    pixsize = (pr_header['CDELT1'] * degree_to_rad,
               pr_header['CDELT2'] * degree_to_rad,)
    # Find stokes info
    stokes_card = find_card_from_header(pr_header, value='STOKES')[0]
    indx = stokes_card.keyword[-1]
    stokes = stokes_dict[pr_header['CRVAL' + indx]]
    # Find frequency info
    freq_card = find_card_from_header(pr_header, value='FREQ')[0]
    indx = freq_card.keyword[-1]
    freq = pr_header['CRVAL' + indx]

    try:
        # BEAM info in ``AIPS CG`` table
        idx = hdulist.index_of('AIPS CG')
        data = hdulist[idx].data
        bmaj = float(data['BMAJ']) * degree_to_rad
        bmin = float(data['BMIN']) * degree_to_rad
        bpa = float(data['BPA']) * degree_to_rad
    # In Petrov's data it in PrimaryHDU header
    except KeyError:
        try:
            bmaj = pr_header['BMAJ'] * degree_to_rad
            bmin = pr_header['BMIN'] * degree_to_rad
            bpa = pr_header['BPA'] * degree_to_rad
        except KeyError:
            # In Denise data it is in PrimaryHDU ``HISTORY``
            # TODO: Use ``pyfits.header._HeaderCommentaryCards`` interface if
            # any
            for line in pr_header['HISTORY']:
                if 'BMAJ' in line and 'BMIN' in line and 'BPA' in line:
                    bmaj = float(line.split()[3]) * degree_to_rad
                    bmin = float(line.split()[5]) * degree_to_rad
                    bpa = float(line.split()[7]) * degree_to_rad
        if not (bmaj and bmin and bpa):
            warnings.warn("Beam info absent!")

    return {"imsize": imsize, "pixref": pixref, "pixrefval": pixrefval,
            "bmaj": bmaj, "bmin": bmin, "bpa": bpa, "pixsize": pixsize,
            "stokes": stokes, "freq": freq}


def get_hdu(fname, extname=None, ver=1):
    """
    Function that returns instance of ``PyFits.HDU`` class with specified
    extension and version from specified file.

    :param fname:
        Path to FITS-file.

    :param extname: (optional)
        Header's extension. If ``None`` then return first from
        ``PyFits.HDUList``. (default: ``None``)

    :param ver: (optional)
        Version of ``HDU`` with specified extension. (default: ``1``)

    :return:
        Instance of ``PyFits.HDU`` class.
    """

    hdulist = pf.open(fname)
    return get_hdu_from_hdulist(hdulist, extname, ver)


def get_hdu_from_hdulist(hdulist, extname=None, ver=1):
    """
    Function that returns instance of ``PyFits.HDU`` class with specified
    extension and version from instance of ``PyFits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.

    :param extname: (optional)
        Header's extension. If ``None`` then return first from
        ``PyFits.HDUList``. (default: ``None``)

    :param ver: (optional)
        Version of ``HDU`` with specified extension. (default: ``1``)

    :return:
        Instance of ``PyFits.HDU`` class.

    """
    if extname:
        try:
            indx = hdulist.index_of((extname, ver,))
            hdu = hdulist[indx]
        except:
            raise AbsentHduExtensionError('No {} binary table'
                                          ' found'.format(extname))
    else:
        hdu = hdulist[0]

    return hdu
