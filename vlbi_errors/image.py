__author__ = 'ilya'

import glob
import numpy as np
try:
    import pylab
except ImportError:
    pylab = None


class Image(object):
    """
    Class that implements images.
    :param imsize:
    :param pixsize:
    :param beam:
    :param stokes:
    """
    def __init__(self, imsize=None, pixsize=None, beam=None, stokes=None):
        self.imsize = imsize
        self.pixsize = pixsize
        self.beam = beam
        self.stokes = stokes

    def add_from_txt(self, fname, stokes='I'):
        """
        Load image from text file.

        :param fname:
            Text file with image data.
        :param stokes (optional):
        """
        pass

    def add_from_fits(self, fname, stokes='I'):
        """
        Load image from FITS file.

        :param fname:
            FITS file with image data.

        :param stokes (optional):
        """
        pass

    def cross_corelate(self, image):
        """
        Cross-correlates image with another image.

        :param image:
            Instance of image class.
        """
        pass
        
    def plot(self, blc=None, trc=None, clim=None, cmap=None):
        """
        Plot image.
        """
        if not pylab:
            raise Exception("Install matplotlib for plotting!")
        if blc or trc:
            patch_to_plot = self._array[blc[0]:trc[0], blc[1]:trc[1]]
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
            image = Image()
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
            image = Image()
            image.add_from_txt(fname, stokes=stokes)
            self.images.append(image)

    def pixels_histogram(self):
        for i, image in enumerate(self.images):
            self._3_darray[i] = image.data
