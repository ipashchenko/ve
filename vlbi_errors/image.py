__author__ = 'ilya'


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


class ImageSet(object):
    """
    Class that implements collection of images.
    """
    def __init__(self):
        pass


