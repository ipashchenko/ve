from vlbi_errors.data_io import get_fits_image_info


# * Resolution could depend on position in map or frequency. Should i associate
# instance of ``Beam`` with each pixel in ``Image`` instance in such cases?
# * Beam is connected with ``CCModel`` instance naturally by deconvolution
# process? NO! Delta functions can be used in modelling without CLEAN!
class Beam(object):
    """
    Basic class that represents point spread function.
    """
    pass


class DirtyBeam(Beam):
    """
    Class that represents point spread function.
    """
    def fit_central_part(self):
        """
        Fit central part of dirty beam and return instance of ``CleanBeam``.
        """
        pass


class CleanBeam(object):
    """
    Class that represents central part of point spread function.
    """
    def __init__(self, bmaj=None, bmin=None, bpa=None):
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa

    def from_fits(self, fname):
        imsize, pixref, (bmaj, bmin, bpa), pixsize = get_fits_image_info(fname)
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa

