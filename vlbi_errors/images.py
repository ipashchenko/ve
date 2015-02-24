import numpy as np
import glob
from from_fits import create_image_from_fits_file
from utils import mask_region


class Images(object):
    """
    Class that handle set of images.
    """
    def __init__(self, images=None):
        if images:
            self.images = images
            self._cube = self.create_cube()
        else:
            self.images = list()
            self._cube = None

    def add_image(self, image):
        self.images.append(image)
        self._cube = self.create_cube()

    def delete_image(self, image):
        self.images.remove(image)
        self._cube = self.create_cube()

    def create_cube(self, region=None):
        return np.dstack(tuple(image.image for image in self.images))

    def add_from_fits(self, wildcard, stokes='I'):
        """
        Load images from FITS files.

        :param wildcard:
            Wildcard used for ``glob.glob`` to select FITS files with images.

        :param stokes (optional):
        """
        fnames = glob.glob(wildcard)
        for fname in fnames:
            image = create_image_from_fits_file(fname)
            self.images.append(image)
        self._cube = self.create_cube()

    def pixels_histogram(self, region=None):
        """
        :param region (optional):
            Region where to calculate histograms. Or (blc[0], blc[1], trc[0],
            trc[1],) or (center[0], center[1], r, None,).
        :param sum (optional):
            Calculate sum of pixels in region?
        """
        # First, create mask
        cube = self.create_cube(region=region)


