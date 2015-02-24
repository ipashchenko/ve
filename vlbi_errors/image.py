import numpy as np
from scipy import signal
from utils import create_grid, mask_region, fitgaussian
from beam import CleanBeam
from fft_routines import fft_convolve2d

try:
    import pylab
except ImportError:
    pylab = None


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

    # TODO: Sometimes we need to add/substract convolved images. So subclasses
    # should implement property with convolution.
    @property
    def image(self):
        """
        Shorthand for image array.
        """
        return self._image

    # TODO: Am i need it? Should i compare instances before setting?
    @image.setter
    def image(self, image):
        self._image = image.image

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

    def __sub__(self, image):
        """
        Substruct from current instance of ``Image`` class other instance.
        """
        raise NotImplementedError

    def __div__(self, image):
        """
        Divide current instance of ``Image`` class on other instance.
        """
        raise NotImplementedError

    # Convolve with any object that has ``image`` attribute
    def convolve(self, image_like):
        """
        Convolve ``Image`` array with image-like instance.
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
        self._image += rvs

    # TODO: Should i compare images before?
    def cross_correlate(self, image, region1=None, region2=None):
        """
        Cross-correlates current instance of ``Image`` with another instance.

        Computes normalized cross-correlation of images.

        :param image:
            Instance of image class.
        :param region1 (optional):
            Region to EXCLUDE in current instance of ``Image``.
            Or (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
            None,). Default ``None``.
        :param region2 (optional):
            Region to EXCLUDE in ``image``. Or (blc[0], blc[1], trc[0], trc[1],)
            or (center[0], center[1], r, None,). Default ``None``.
        :return:
            (dx, dy,) tuple of shifts (subpixeled) in each direction.
        """
        if region1 is not None:
            image1 = mask_region(self.image, region1)
        if region2 is not None:
            image2 = mask_region(image.image, region2)
        # Cross-correlate images
        shift_array = fft_convolve2d(image1, image2)
        params = fitgaussian(shift_array)
        return tuple(params[1: 3])

    # TODO: fix BLC,TRC to display expected behavior. Use data_io.get_image()
    def plot(self, blc=None, trc=None, clim=None, cmap=None):
        """
        Plot image.

        :note:
            ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
            converted to python-like zero-indexing.
        """
        if not pylab:
            raise Exception("Install matplotlib for plotting!")
        if blc or trc:
            blc = blc or (1, 1,)
            trc = trc or self.imsize
            part_to_plot = self.image[blc[0] - 1: trc[0] - 1, blc[1] - 1: trc[1]
                                      - 1]
        else:
            part_to_plot = self.image
        imgplot = pylab.imshow(part_to_plot)
        if cmap:
            try:
                imgplot.set_cmap(cmap)
            except:
                # Show wo ``cmap`` set, print availbale ``cmap``s.
                pass
        if clim:
            # TODO: Warn if ``clim`` is out of range for image.
            imgplot.set_clim(clim)


# FIXME: THIS DOESN't KEEP RESIDUALS! ONLY CCs!
class CleanImage(Image):
    """
    Class that represents image made using CLEAN algorithm.
    """
    def __init__(self, imsize=None, pixref=None, pixsize=None, bmaj=None,
                 bmin=None, bpa=None):
        super(CleanImage, self).__init__(imsize, pixref, pixsize)
        # TODO: What if pixsize has different sizes???
        self.beam = CleanBeam(bmaj / abs(pixsize[0]), bmin / abs(pixsize[0]),
                              bpa, imsize)

    @property
    def image(self):
        """
        Shorthand for CLEAN image.
        """
        return signal.fftconvolve(self._image, self.beam.image, mode='same')


#class MemImage(Image, Model):
#    """
#    Class that represents image made using MEM algorithm.
#    """
#    pass
