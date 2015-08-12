import numpy as np
import glob
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from utils import (mask_region, mas_to_rad, hdi_of_mcmc)
from image import BasicImage, CleanImage


class Images(object):
    """
    Class that handle set of images that can be stacked.

    :note:
        Instance of ``Images`` class should contain only images that can be
        stacked - that is images with equal ``imsize``, ``pixsize``. It is
        responsibility of user to supply it images that have same resolution
        (``CleanImage`` instances), Stokes parameters, etc.
    """
    def __init__(self):
        # Container of ``Image`` instances
        self._images = list()
        # Stacked images
        self._cube = None

    # TODO: Implement option for stacking only region(s) of images
    # TODO: Sort somehow by Stokes parameters & frequencies in methods that
    # need it (getting RM, apec. index maps)
    def _create_cube(self):
        self._cube = np.dstack(tuple(image.image for image in self._images))

    def compare_images_by_param(self, param):
        if param not in self._images[0].__dict__:
            raise Exception("No " + param + " attribute at Image instance to"
                                           " compare!")
        attr_values = list()
        for image in self._images:
            attr_values.append(image.__getattribute__(param))
            assert len(set(attr_values)) == 1, ("Check " + param + " for " +
                                                image)

    def add_from_fits(self, fnames=None, wildcard=None):
        """
        Load images from user-specified FITS files.

        :param fnames: (optional)
            Iterable of FITS-file names.
        :param wildcard: (optional)
            Wildcard used for ``glob.glob`` to select FITS-files with images.

        """
        if fnames is None:
            fnames = glob.glob(wildcard)
        if len(fnames) < 2:
            raise Exception("Need at least 2 images")

        for fname in fnames:
            # FIXME: When use clean_image & when just image?
            print "Processing ", fname
            image = create_image_from_fits_file(fname)
            if self._images:
                assert image == self._images[-1], "Adding image with different" \
                                                  "parameters!"
            self._images.append(image)

    def create_error_map(self):
        """
        Method that creates an error map for current collection of instances.
        """
        # Check that collection of images isn't empty
        if not self._images:
            raise Exception("First, add some images to instance!")
        # Check additionally, that images has the same Stokes parameters and
        # frequency
        self.compare_images_by_param("stokes")
        self.compare_images_by_param("freq")

        # Now can safely create cube
        self._create_cube()

        img = self._images[0]
        hdis = np.zeros(np.shape(self._cube[:, :, 0]))
        for (x, y), value in np.ndenumerate(hdis):
            hdi = hdi_of_mcmc(self._cube[x, y, :], cred_mass=0.68)
            hdis[x, y] = hdi[1] - hdi[0]
        # Create basic image and add map of error
        image = BasicImage(imsize=img.imsize, pixref=img.pixref,
                           pixrefval=img.pixrefval, pixsize=img.pixsize)
        image.image = hdis
        return image

    def pixels_histogram(self, region=None):
        """
        :param region (optional):
            Region where to calculate histograms. Or (blc[0], blc[1], trc[0],
            trc[1],) or (center[0], center[1], r, None,).
        :param sum (optional):
            Calculate sum of pixels in region?
        """
        # First, create mask
        pass

    def rotm(self):
        pass


if __name__ == '__main__':
    boot_dir = '/home/ilya/code/vlbi_errors/data/zhenya/ccbots/'
    images = Images()
    images.add_from_fits(wildcard=boot_dir + "cc_*.fits")
    error_map = images.create_error_map()
