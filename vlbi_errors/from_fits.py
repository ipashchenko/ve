import pyfits as pf
from model import CCModel
from utils import degree_to_mas
from components import DeltaComponent
from data_io import BinTable, get_fits_image_info
from new_image import Image, CleanImage


def create_ccmodel_from_fits_file(fname, stokes='I', ver=1):
    ccmodel = CCModel(stokes=stokes)
    cc = BinTable(fname, extname='AIPS CC', ver=ver)
    adds = cc.load()
    for flux, x, y in zip(adds['FLUX'], adds['DELTAX'] * degree_to_mas,
                          adds['DELTAY'] * degree_to_mas):
        # We keep positions in mas
        component = DeltaComponent(flux, x, y)
        ccmodel.add_component(component)
    return ccmodel


# TESTED w Petrov's data
def create_clean_image_from_fits_file(fname, stokes='I', ver=1):
    """
    Create instance of ``CleanImage`` from FITS-file of CLEAN image.
    :param fname:
    :return:
        Instance of ``CleanImage``.
    """
    ccmodel = create_ccmodel_from_fits_file(fname, stokes=stokes, ver=ver)
    imsize, pixref, (bmaj, bmin, bpa,), pixsize = get_fits_image_info(fname)
    if bmaj is None:
        raise Exception("Can't find Beam info!")
    ccimage = CleanImage(imsize, pixref, pixsize, bmaj, bmin, bpa)
    ccimage.add_model(ccmodel)
    return ccimage


# FIXME: This is quite useless function actually. Remove?
# TODO: There must be subclass of IO.PyFitsIO class for loading images
def create_image_from_fits_file(fname):
    """
    Create instance of ``CleanImage`` from FITS-file of CLEAN image.
    :param fname:
    :return:
        Instance of ``CleanImage``.
    """
    imsize, pixref, (bmaj, bmin, bpa,), pixsize = get_fits_image_info(fname)
    image = Image(imsize, pixref, pixsize)
    # FIXME: THIS IS BAD!!!
    image_hdu = pf.open(fname)[0]
    image._image = image_hdu.data.squeeze()
    return image
