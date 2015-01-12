import glob
import numpy as np
from scipy import signal
from data_io import get_fits_image_info
from model import CCModel
from utils import gaussianBeam

try:
    import pylab
except ImportError:
    pylab = None


# TODO: Add operation of comparing of ImageGrid instances.
class ImageGrid(object):
    """
    Class that represents image grid (array) representation of models.
    """
    def __init__(self, fname=None, imsize=None, pixref=None, pixsize=None):
        if fname:
            self.from_image(fname)
        else:
            self.imsize = imsize
            self.dx, self.dy = pixsize
            self.x_c, self.y_c = pixref
        self.image_grid = np.zeros(self.imsize, dtype=float)

    def from_image(self, fname):
        imsize, pixref, (bmaj, bmin, bpa,), pixsize = get_fits_image_info(fname)
        self.imsize = imsize
        self.dx, self.dy = pixsize
        self.x_c, self.y_c = pixref

    def add_component(self, component):
        component.add_to_image_grid(self)

    def add_noise(self, std, df=None):
        pass


# TODO: Add operation of comparing of Image instances.
class Image(object):
    """
    Class that implements images.
    :param imsize:
    :param pixsize:
    :param beam:
    :param stokes:
    """
    def __init__(self, imsize=None, pixsize=None, pixref=None, bmaj=None,
                 bmin=None, bpa=None, stokes=None):
        self.imsize = imsize
        self.pixsize = pixsize
        self.pixref = pixref
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.stokes = stokes

    def __eq__(self, image):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        if (self.imsize == image.imsize and self.pixsize == image.pixsize and
                self.bmaj == image.bmaj and self.bmin == image.bmin and
                self.bpa == image.bpa):
            return True
        else:
            return False

    def add_from_array(self, data, pixsize=None, bmaj=None, bmin=None, bpa=None,
                       stokes=None):
        self.image = np.atleast_2d(data)
        self.imsize = (self._image.shape)
        self.pixsize = pixsize
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa

    def add_from_txt(self, fname, stokes='I'):
        """
        Load image from text file.

        :param fname:
            Text file with image data.
        :param stokes (optional):
        """
        pass

    def add_cc_from_fits(self, fname, stokes='I'):
        """
        Load image from FITS file.

        :param fname:
            FITS file with image data.
        :param stokes (optional):
        """
        model = CCModel()
        model.add_cc_from_fits(fname, stoke=stokes)
        image_grid = ImageGrid(imsize=self.imsize, pixref=self.pixref,
                               pixsize=self.pixsize)
        model.add_to_image_grid(image_grid)
        gaussian_beam = gaussianBeam(self.imsize[0], self.bmaj, self.bmin,
                                     self.bpa, self.imsize[1])
        self.image = signal.fftconvolve(image_grid.image_grid, gaussian_beam,
                                        mode='same')

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


class ImageSet(object):
    """
    Class that implements collection of images.
    """
    def __init__(self, referenced_image=None, imsize=None, pixsize=None,
                 beam=None, stokes=None):
        """
        :param reference:
            Instance of ``Image`` class, used for setting parameters of images.
        """
        self.images = list()
        if referenced_image is not None:
            self.referenced_image = referenced_image
            self.imsize = referenced_image.imsize
            self.pixsize = referenced_image.pixsize
            self.beam = referenced_image.beam
            self.stokes = referenced_image.stokes
        elif imsize and pixsize and beam and stokes:
            self.imsize = imsize
            self.pixsize = pixsize
            self.beam = beam
            self.stokes = stokes
        else:
            # Use first added image to initialize reference parameters
            # (``imsize``, ``pixsize``, ``beam``, ``stokes``)
            pass
        # Keeping image data in 3D-cube.
        self._3d_array = np.empty((len(self.images), self.pixsize[0],
                                   self.pixsize[1]))

    def add_from_fits(self, wildcard, stokes='I'):
        """
        Load images from FITS files.

        :param wildcard:
            Wildcard used for ``glob.glob`` to select FITS files with images.

        :param stokes (optional):
        """
        fnames = glob.glob(wildcard)
        for fname in fnames:
            image = ImageGrid()
            image.add_from_fits(fname, stokes=stokes)
            self.images.append(image)

    def add_from_txt(self, wildcard, stokes='I'):
        """
        Load images from text files.

        :param wildcard:
            Wildcard used for ``glob.glob`` to select txt-files with images.

        :param stokes (optional):
        """
        fnames = glob.glob(wildcard)
        for fname in fnames:
            image = ImageGrid()
            image.add_from_txt(fname, stokes=stokes)
            self.images.append(image)

    def pixels_histogram(self, region=None, sum=None):
        """
        :param region (optional):
            Region where to calculate histograms. Or (blc[0], blc[1], trc[0],
            trc[1],) or (center[0], center[1], r, None,).
        :param sum (optional):
            Calculate sum of pixels in region?
        """
        # First, create mask
        for i, image in enumerate(self.images):
            self._3d_array[i] = image.data

    def cross_correlate_with_image(self, image,
                                   region1=(None, None, None, None),
                                   region2=(None, None, None, None)):
        """
        Cross-correlate with one image.

        :param image:
            Instance of ``Image`` class.
        :param region1:
            Region to EXCLUDE in current instance images. Or (blc[0], blc[1],
            trc[0], trc[1],) or (center[0], center[1], r, None,).
        :param region2:
            Region to EXCLUDE in ``image``. Or (blc[0], blc[1], trc[0], trc[1],)
            or (center[0], center[1], r, None,).
        :return:
            Tuple of tuples (dx, dy,) of shifts (subpixeled) in each direction.
        """
        pass

    def cross_correlate_with_set_of_images(self, images,
                                           region1=(None, None, None, None),
                                           region2=(None, None, None, None)):
        """
        Cross-correlate with set of images.

        :param images:
            Instance of ``ImageSet`` class.
        :param region1:
            Region to EXCLUDE in current instance images. Or (blc[0], blc[1],
            trc[0], trc[1],) or (center[0], center[1], r, None,).
        :param region2:
            Region to EXCLUDE in images of ``images``. Or (blc[0], blc[1],
            trc[0], trc[1],) or (center[0], center[1], r, None,).
        :return:
            Tuple of tuples (dx, dy,) of shifts (subpixeled) in each direction.
        """
        pass
