from model import CCModel
from utils import degree_to_mas
from components import DeltaComponent
from data_io import Groups, IDI, BinTable, get_hdu, get_fits_image_info
from uv_data import UVData
from image import Image, CleanImage


def create_uvdata_from_fits_file(fname, structure='UV'):
    """
    Helper function for loading FITS-files.

        :param fname:
            Path to FITS-file.

        :param structure (optional):
            Structure of FITS-file. ``UV`` or ``IDI``. (default: ``UV``)

        :return:
            Instance of ``UVData`` class for the specified FITS-file.
    """

    assert(structure in ['UV', 'IDI'])

    structures = {'UV': Groups(), 'IDI': IDI()}
    uvdata = UVData(io=structures[structure])
    uvdata.load(fname)

    return uvdata


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
    imsize, pixref, pixrefval, (bmaj, bmin, bpa,), pixsize =\
        get_fits_image_info(fname)
    if bmaj is None:
        raise Exception("Can't find Beam info!")
    ccimage = CleanImage(imsize, pixref, pixrefval, pixsize, bmaj, bmin, bpa)
    ccimage._fname = fname
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
    imsize, pixref, pixrefval, (bmaj, bmin, bpa,), pixsize =\
        get_fits_image_info(fname)
    image = Image(imsize, pixref, pixrefval, pixsize)
    # FIXME: THIS IS BAD!!!
    image_hdu = get_hdu(fname)
    image._image = image_hdu.data.squeeze()
    return image
