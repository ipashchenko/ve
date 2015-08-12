import numpy as np
import glob
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from utils import (mask_region, mas_to_rad, hdi_of_mcmc)
from data_io import get_fits_image_info
from image import BasicImage, CleanImage

# TODO: Should work with any ``BasicImage`` subclass instance
# TODO: Use fixed ``alpha_low = 2.5``, ``nu_cut``, ``S_cut`` & ``alpha_high``
# parameters. S_nu ~ nu ** alpha
def alpha(add_residuals=True, *fnames):
    if len(fnames) < 2:
        raise Exception("Need at least 2 images")
    mapsizes = list()
    stokes = list()
    beams = list()
    freqs = list()
    image_dict = dict()
    for fname in fnames:
        map_info = get_fits_image_info(fname)
        beam = map_info[3]
        beams.append(beam)
        mapsize = (map_info[0][0], map_info[-3][0] / mas_to_rad)
        mapsizes.append(mapsize)
        freq = map_info[-1]
        freqs.append(freq)
        stoke = map_info[-2]
        stokes.append(stoke)
        # Assertions on consistency
        assert (len(set(beams)) == 1)
        assert (len(set(mapsizes)) == 1)
        assert (len(set(stokes)) == 1)
        assert (len(set(freqs)) == len(fnames))
        ccimage = create_clean_image_from_fits_file(fname, stokes=stoke)
        if add_residuals:
            image = ccimage.image_w_residuals
        else:
            image = ccimage.image
        image_dict.update({freq: image})


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
    def create_cube(self):
        return np.dstack(tuple(image.image for image in self._images))

    def compare_images_by_param(self, param):
        if param not in CleanImage.__dict__:
            raise Exception("No", param, " attribute at Image instances!")
        attr_values = list()
        for image in self._images:
            attr_values.append(image.__getattribute__(param))
            assert(len(set(attr_values)) == 1, "Check ", param, " for ", image)

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
            image = create_image_from_fits_file(fname)
            if self._images:
                assert(image == self._images[-1],
                       "Adding image with different parameters!")
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

        img = self._images[0]
        hdis = np.zeros(np.shape(self._cube[:,:,0]))
        for (x,y), value in np.ndenumerate(hdis):
            hdis[x,y] = hdi_of_mcmc(self._cube[x,y,:])
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

