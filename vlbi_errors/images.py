import numpy as np
import glob
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from utils import (mask_region, mas_to_rad, hdi_of_mcmc, flatten,
                   nested_dict_itervalue)
from image import BasicImage, CleanImage
from collections import defaultdict


class Images(object):
    """
    Class that handle set of images that can be stacked.

    :note:
        Instance of ``Images`` class should contain only images that can be
        stacked - that is images with equal ``imsize``, ``pixsize``. It is
        responsibility of user to supply it images that have same resolution
        (``CleanImage`` instances), Stokes parameters, etc.

        Use cases:
            - create error map from images of bootstrapped uv-data (single
            Stokes parameter, single frequency)
            - create map of complex polarization (I, Q & U Stokes parameters,
            single frequency)
            - create map of spectral index (I Stokes parameter, several
            frequencies)
            - create rotation measure map (Q & U Stokes parameters, several
            frequencies)
    """
    def __init__(self):
        # Container of ``Image`` instances
        self._images_dict = defaultdict(lambda: defaultdict(list))
        # Pickleable solution:
        # from functools import partial
        # self._images_dict = defaultdict(partial(defaultdict, list))
        # Stacked images
        self._images_cube = None

    @property
    def images(self):
        return list(flatten(nested_dict_itervalue(self._images_dict)))

    @property
    def freqs(self):
        return self._images_dict.keys()

    def stokeses(self, freq):
        return self._images_dict[freq].keys()

    def _create_cube(self, stokes=None, freq=None):
        self._images_cube = np.dstack(tuple(image.image for image in
                                      self._images_dict[freq][stokes]))

    def compare_images_by_param(self, param, freq_stokes_dict=None):
        """
        Method that compares images in ``self._images_dict`` by value of
        user-specified parameter.

        :param param:
            Parameter to compare.
        :param freq_stokes_dict:
            Dictionary with {frequency: Stokes parameters} which select what
            images to compare.

        """
        # If no frequencies are supplied => use all available
        if freq_stokes_dict is None:
            freqs = self.freqs
        else:
            freqs = freq_stokes_dict.keys()
        images = list()
        for freq in freqs:
            # If no Stokes parameters are supplied => use all available for each
            # available frequency
            if freq_stokes_dict is None:
                stokeses = self._images_dict[freq].keys()
            for stokes in stokeses:
                images.extend(self._images_dict[freq][stokes])

        attr_values = list()
        for image in images:
            try:
                attr_values.append(image.__getattribute__(param))
            except AttributeError:
                raise Exception("No " + param + " attribute at Image instance"
                                                " to compare!")

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

        # Here we check that images we are collecting are equal
        previous_image = None
        for fname in fnames:
            # FIXME: When use clean_image & when just image?
            print "Processing ", fname
            image = create_image_from_fits_file(fname)
            if previous_image:
                assert image == previous_image, "Adding image with different " \
                                                "basic parameters!"
            freq = image.freq
            stokes = image.stokes
            self._images_dict[freq][stokes].append(image)

    def create_error_map(self, freq=None, stokes=None, cred_mass=0.68):
        """
        Method that creates an error map for current collection of instances.
        """
        # Check that collection of images isn't empty
        if not self.images:
            raise Exception("First, add some images to instance!")

        # If no frequency is supplied => check that instance contains images of
        # only one frequency and use it. Otherwise - raise Exception
        if freq is None:
            freqs = self.freqs
            if len(freqs) > 1:
                raise Exception("Choose what frequency images to use!")
            else:
                freq = freqs[0]
        # If no Stokes parameter is specified => check that chosen frequency
        # contains images of only one Stokes parameter. Otherwise - raise
        # Exception
        if stokes is None:
            stokeses = self.stokeses(freq)
            if len(stokeses) > 1:
                raise Exception("Choose what Stokes parameter images to use!")
            else:
                stokes = stokeses[0]

        # Now can safely create cube
        self._create_cube(stokes, freq)

        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        img = self._images_dict[freq][stokes][0]
        hdis = np.zeros(np.shape(self._images_cube[:, :, 0]))
        for (x, y), value in np.ndenumerate(hdis):
            hdi = hdi_of_mcmc(self._images_cube[x, y, :], cred_mass=cred_mass)
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
