import math
import numpy as np
import glob
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from utils import (mask_region, mas_to_rad, hdi_of_mcmc, flatten,
                   nested_dict_itervalue)
from image import BasicImage, CleanImage
from collections import defaultdict
from scipy.optimize import leastsq


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


def rotm_map(freqs, chis, s_chis):
    """
    Function that calculates Rotation Measure map.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of 2D numpy arrays with polarization positional angles [rad].
    :param s_chis:
        Iterable of 2D numpy arrays with polarization positional angles
        uncertainties estimates [rad].
    :return:
        Tuple of 2D numpy array with values of Rotation Measure [rad/m**2] and
        2D numpy array with uncertainties map [rad/m**2].

    """
    chi_cube = np.dstack(chis)
    s_chi_cube = np.dstack(s_chis)
    rotm_array = np.zeros(np.shape(chi_cube[:, :, 0]))
    s_rotm_array = np.zeros(np.shape(chi_cube[:, :, 0]))
    for (x, y), value in np.ndenumerate(rotm_array):
        p, pcov = rotm(freqs, chi_cube[x, y, :], s_chi_cube[x, y, :])
        rotm_array[x, y] = p[0]
        s_rotm_array[x, y] = math.sqrt(pcov[0, 0])

    return rotm_array, s_rotm_array


def rotm(freqs, chis, s_chis, p0=None):
    """
    Function that calculates Rotation Measure.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of polarization positional angles [rad].
    :param s_chis:
        Iterable of polarization positional angles uncertainties estimates
        [rad].
    :param p0:
        Starting value for minimization (RM [rad/m**2], PA_zero_lambda [rad]).
    :return:
        Tuple of numpy array of (RM [rad/m**2], PA_zero_lambda [rad]) and 2D
        numpy array of covariance matrix.

    """

    if p0 is None:
        p0 = [0., 0.]

    p0 = np.array(p0)
    freqs = np.array(freqs)
    chis = np.array(chis)
    s_chis = np.array(s_chis)

    def rm_model(p, freqs):
        lambdasq = (3. * 10 ** 8 / freqs) ** 2
        return p[0] * lambdasq + p[1]

    def weighted_residuals(p, freqs, chis, s_chis):
        return (chis - rm_model(p, freqs)) / s_chis

    func, args = weighted_residuals, (freqs, chis, s_chis,)
    fit = leastsq(func, p0, args=args, full_output=True)
    (p, pcov, infodict, errmsg, ier) = fit

    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)

    if (len(chis) > len(p0)) and pcov is not None:
        # Residual variance
        s_sq = (func(p, *args) ** 2.).sum() / (len(chis) - len(p0))
        pcov *= s_sq
    else:
        pcov = np.nan

    return p, pcov


if __name__ == '__main__':
    boot_dir = '/home/ilya/code/vlbi_errors/data/zhenya/ccbots/'
    images = Images()
    images.add_from_fits(wildcard=boot_dir + "cc_*.fits")
    error_map = images.create_error_map()
