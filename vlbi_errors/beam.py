from scipy import signal
from data_io import get_fits_image_info
from utils import gaussianBeam


# * Resolution could depend on position in map or frequency. Should i associate
# instance of ``Beam`` with each pixel in ``BasicImage`` instance in such cases?
# * Beam is connected with ``CCModel`` instance naturally by deconvolution
# process? NO! Delta functions can be used in modelling without CLEAN!
def create_clean_beam_from_fits(fname):
    """
    Create instance of ``CleanBeam`` from FITS-file of CLEAN image.
    """
    imsize, pixref, (bmaj, bmin, bpa), pixsize, stokes, freq =\
        get_fits_image_info(fname)
    return CleanBeam(bmaj / abs(pixsize[0]), bmin / abs(pixsize[0]), bpa,
                     imsize)


class Beam(object):
    """
    Basic class that represents point spread function.
    """
    def __init__(self):
        self.image = None

    # Convolve with any object that has ``image`` attribute
    def convolve(self, image_like):
        return signal.fftconvolve(self.image, image_like.image, mode='same')


class DirtyBeam(Beam):
    """
    Class that represents point spread function.
    """
    def fit_central_part(self):
        """
        Fit central part of dirty beam and return instance of ``CleanBeam``.
        """
        raise NotImplementedError


# TODO: Refactor class - beam should't depend on any size.
class CleanBeam(Beam):
    """
    Class that represents central part of point spread function.

    :param bmaj:
        Beam major axis [pxl].
    :param bmin:
        Beam minor axis [pxl].
    :param bpa:
        Beam positional angle [deg].
    :param size:
        Size of beam image [pxl].
    """
    def __init__(self, bmaj=None, bmin=None, bpa=None, size=None):
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.size = size
        self.image = gaussianBeam(self.size[0], self.bmaj, self.bmin,
                                  self.bpa + 90., self.size[1])
    def __eq__(self, other):
        """
        Compares current instance of ``CleanBeam`` class with other instance.
        """
        return (self.bmja == other.bmaj and self.bmin == other.bmin and
                self.bpa == other.bpa)

    def __ne__(self, other):
        """
        Compares current instance of ``CleanBeam`` class with other instance.
        """
        return (self.bmja != other.bmaj or self.bmin != other.bmin or
                self.bpa != other.bpa)
