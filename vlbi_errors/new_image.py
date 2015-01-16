import numpy as np
from scipy import signal
from data_io import get_fits_image_info
from utils import gaussianBeam, create_grid
from vlbi_errors.model import Model

try:
    import pylab
except ImportError:
    pylab = None


def create_clean_image_from_fits_file(fname):
    """
    Create instance of ``CleanImage`` from FITS-file of CLEAN image.
    :param fname:
    :return:
        Instance of ``CleanImage``.
    """
    pass


class Image(object):
    """
    Class that represents images.
    """
    def __init__(self, imsize=None, pixref=None, pixsize=None):
        self.imsize = imsize
        self.dx, self.dy = pixsize
        self.x_c, self.y_c = pixref
        # Create flux array
        self._image = np.zeros(self.imsize, dtype=float)
        # Create coordinate arrays
        x, y = create_grid(self.imsize)
        x = x - self.x_c
        y = y - self.y_c
        x = x * self.dx
        y = y * self.dy
        self.x = x
        self.y = y

    @property
    def image(self):
        """
        Shorthand for image array.
        """
        return self._image

    @image.setter
    def image(self, image):
        self._image += image.image

    def __eq__(self, image):
        """
        Compares current instance of ``ImageModel`` class with other instance.
        """
        return (self.imsize == image.imsize and self.pixsize == image.pixsize)

    def __sum__(self, image):
        """
        Sums current instance of ``Image`` class with other instance.
        """
        raise NotImplementedError

    # Convolve with any object that has ``image`` attribute
    def convolve(self, image_like):
        """
        Convolve ``Image`` array with ``Beam`` instance.
        """
        return signal.fftconvolve(self._image, image_like.image, mode='same')

    def add_component(self, component):
        component.add_to_image(self)

    def add_model(self, model):
        model.add_to_image(self)

    def add_noise(self, std, df=None):
        size = self.imsize[0] * self.imsize[1]
        if df is None:
            rvs = np.random.normal(loc=0., scale=std, size=size)
        else:
            raise NotImplementedError
        rvs = rvs.reshape(self.imsize)
        self.image_grid += rvs

    def cross_correlate(self, image, region1=(None, None, None, None),
                        region2=(None, None, None, None)):
        """
        Cross-correlates image with another image.

        Computes normalized cross-correlation of images.

        :param image:
            Instance of image class.
        :param region1:
            Region to EXCLUDE in current instance of ``Image``.
            Or (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
            None,).
        :param region2:
            Region to EXCLUDE in ``image``. Or (blc[0], blc[1], trc[0], trc[1],)
            or (center[0], center[1], r, None,).
        :return:
            (dx, dy,) tuple of shifts (subpixeled) in each direction.
        """
        pass

    def plot(self, blc=None, trc=None, clim=None, cmap=None):
        """
        Plot image.
        """
        if not pylab:
            raise Exception("Install matplotlib for plotting!")
        if blc or trc:
            path_to_plot = self._array[blc[0]:trc[0], blc[1]:trc[1]]
            imgplot = pylab.imshow(path_to_plot)
            if cmap:
                try:
                    imgplot.set_cmap(cmap)
                except:
                    # Show wo ``cmap`` set, print availbale ``cmap``s.
                    pass
            if clim:
                # TODO: Warn if ``clim`` is out of range for image.
                imgplot.set_clim(clim)



class CleanImage(Image):
    """
    Class that represents image made using CLEAN algorithm.
    """
    pass


class MemImage(Image, Model):
    """
    Class that represents image made using MEM algorithm.
    """
    pass
