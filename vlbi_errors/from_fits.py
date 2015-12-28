import numpy as np
import pyfits as pf
from model import CCModel
from utils import degree_to_mas, degree_to_rad
from components import DeltaComponent
from data_io import get_fits_image_info_from_hdulist, get_hdu_from_hdulist
from image import Image, CleanImage


def create_ccmodel_from_fits_file(fname, ver=1):
    """
    Function that creates instance of ``CleanModel`` from FITS-file with CLEAN
    model
    :param fname:
    :param ver:
    :return:
    """
    hdulist = pf.open(fname)
    return create_ccmodel_from_hdulist(hdulist, ver)


def create_ccmodel_from_hdulist(hdulist, ver=1):

    image_params = get_fits_image_info_from_hdulist(hdulist)
    ccmodel = CCModel(stokes=image_params['stokes'])
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

    ccmodel = create_ccmodel_from_hdulist(hdulist, stokes=stokes, ver=ver)
    if bmaj is None:
        raise Exception("Can't find Beam info!")
    ccimage = CleanImage(imsize=imsize, pixref=pixref, pixrefval=pixrefval,
                         pixsize=pixsize, bmaj=bmaj, bmin=bmin,
                         bpa=bpa/degree_to_rad, stokes=stokes, freq=freq)
    ccimage.add_model(ccmodel)
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
