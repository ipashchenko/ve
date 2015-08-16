import math
import numpy as np
import glob
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from utils import (mask_region, mas_to_rad, hdi_of_mcmc, flatten,
                   nested_dict_itervalue, mask_region)
from image import BasicImage, Image
from collections import defaultdict
from scipy.optimize import leastsq
from matplotlib.pyplot import hist, bar, show


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

    def pixels_histogram(self, stokes=None, freq=None, region=None, mask=None,
                         mode='mean'):
        """
        Method that creates histogram of pixel values for use-specified pixel or
        region of pixels.

        :param region (optional):
            Region where to calculate histograms. Or (blc[0], blc[1], trc[0],
            trc[1],) or (center[0], center[1], r, None,).
        :param mask: (optional)
            2D numpy array that can be converted to boolen. Mask that specifies
            what pixel to use.
        :param mode: (optional)
            Operation on selected region - 'mean' or 'sum'. (default: 'mean')

        """
        mode_dict = {'mean': np.mean, 'sum': np.sum}
        if mode not in mode_dict:
            raise Exception("Use mean, median or sum for ``mode``!")
        # If ``_image_cube`` haven't created yet - create it now
        if self._images_cube is None:
            # Check that collection of images isn't empty
            if not self.images:
                raise Exception("First, add some images to instance!")

            # If no frequency is supplied => check that instance contains images
            # of only one frequency and use it. Otherwise - raise Exception
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
                    raise Exception("Choose what Stokes parameter images to"
                                    " use!")
                else:
                    stokes = stokeses[0]

            # Now can safely create cube
            self._create_cube(stokes, freq)

        cube = self._images_cube
        if mask is None:
            mask = np.ones(cube[:, :, 0].shape)
            if region is not None:
                mask[mask_region(mask, region).mask] = 0
            else:
                mask = np.zeros(cube[:, :, 0].shape)

        mask = np.resize(mask, (cube.shape[2], mask.shape[0], mask.shape[1]))
        mask = mask.T
        values = np.ma.array(cube, mask=mask)
        values = values.reshape((cube.shape[0] * cube.shape[1], cube.shape[2]))
        values = mode_dict[mode](values, axis=0)
        from knuth_hist import histogram
        counts, edges = histogram(values)
        lower_d = np.resize(edges, len(edges) - 1)
        bar(lower_d, counts, width=np.diff(lower_d)[0], linewidth=2, color='w')
        show()

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
            previous_image = self._images_dict[freq][stokes][-1]

    def add_image(self, image):
        """
        Method that adds instance of ``Image`` class to self.

        :param image:
            Instance of ``Image`` class.
        """
        freq = image.freq
        stokes = image.stokes
        # Check that if any images are already in they have same parameters
        try:
            assert image == self._images_dict[freq][stokes][-1]
        except IndexError:
            pass
        self._images_dict[freq][stokes].append(image)

    def add_images(self, images):
        """
        Method that adds instances of ``Image`` class to self.

        :param images:
            Iterable of ``Image`` class instances.
        """
        for image in images:
            self.add_image(image)

    def create_error_image(self, freq=None, stokes=None, cred_mass=0.68):
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

    # FIXME: Implement option for many (equal number) of Q & U images for each
    # frequency like in ``Images.create_pang_images``
    def create_rotm_image(self, s_pang_arrays=None, freqs=None, mask=None, n=0):
        """
        Method that creates image of Rotation Measure for current collection of
        instances.

        :param s_pang_arrays:
            Iterable of 2D numpy arrays with uncertainty estimates of
            Polarization Angle. Number of arrays must be equal to number of
            frequencies used in rotm calculation.
        :param freqs: (optional)
             What frequences to use. If ``None`` then use all available in
             instance's containter. (default: ``None``)
        :param s_pang_arrays: (optional)
            Iterable of 2D numpy arrays with uncertainty estimates of
            Polarization Angle. Number of arrays must be equal to number of
            frequencies used in rotm calculation. If ``None`` then don't use
            errors in minimization. (default: ``None``)
        :param mask: (optional)
            Mask to be applied to arrays before calculation. If ``None`` then
            don't apply mask. Note that ``mask`` must have dimensions of only
            one image, that is it should be 2D array.
        :param n: (optional)
            Sequence number of Q & U images to use for each frequency. (default:
            ``0``)

        :return:
           Tuple of two ``Image`` instances with Rotation Measure values and
           it's uncertainties estimates.

        """
        required_stokeses = ('Q', 'U')

        # Check that collection of images isn't empty
        if not self.images:
            raise Exception("First, add some images to instance!")

        # Choose frequencies
        if freqs is None:
            freqs = self.freqs
        if len(freqs) < 2:
            raise Exception("Not enough frequencies for RM calculation!")

        # Check that all frequencies have Q & U maps
        for freq in freqs:
            stokeses = self.stokeses(freq)
            for stokes in required_stokeses:
                if stokes not in stokeses:
                    raise Exception("No stokes " + stokes + " parameter for " +
                                    freq + " frequency!")

        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        img = self._images_dict[freq][stokes][0]

        # Create container for Polarization Angle maps
        pang_arrays = list()
        # Fill it with pang arrays - one array for each frequency
        for freq in freqs:
            q_images = self._images_dict[freq]['Q']
            u_images = self._images_dict[freq]['U']
            # Check that we got the same number of ``Q`` and ``U`` images
            if len(q_images) != len(u_images):
                raise Exception("Different # of Q & U images for " + str(freq) +
                                " MHz!")
            pang_arrays.append(pang_map(q_images[n].image, u_images[n].image,
                                        mask=mask))

        # Calculate Rotation Measure array and write it to ``BasicImage``
        # isntance
        rotm_array, s_rotm_array = rotm_map(freqs, pang_arrays, s_pang_arrays,
                                            mask=mask)
        rotm_image = Image(imsize=img.imsize, pixref=img.pixref,
                           pixrefval=img.pixrefval, pixsize=img.pixsize,
                           freq=freqs, stokes='ROTM')
        rotm_image.image = rotm_array
        s_rotm_image = Image(imsize=img.imsize, pixref=img.pixref,
                             pixrefval=img.pixrefval, pixsize=img.pixsize,
                             freq=freqs, stokes='ROTM')
        s_rotm_image.image = s_rotm_array

        return rotm_image, s_rotm_image

    def create_pang_images(self, freq=None, mask=None):
        """
        Method that creates Polarization Angle images for current collection of
        image instances.

        :param freq: (optional)
             What frequency to use. If ``None`` then assume that only one
             frequency is present in instance. (default: ``None``)
        :param mask: (optional)
            Mask to be applied to arrays before calculation. If ``None`` then
            don't apply mask. Note that ``mask`` must have dimensions of only
            one image, that is it should be 2D array.

        :return:
            List of ``Image`` instances with Polarization Angle maps.

        """
        required_stokeses = ('Q', 'U')

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

        # Check that used frequency has Q & U maps
        stokeses = self.stokeses(freq)
        for stokes in required_stokeses:
            if stokes not in stokeses:
                raise Exception("No stokes " + stokes + " parameter for " +
                                freq + " frequency!")

        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        q_images = self._images_dict[freq]['Q']
        u_images = self._images_dict[freq]['U']
        # Check that we got the same number of ``Q`` and ``U`` images
        if len(q_images) != len(u_images):
            raise Exception("Number of Q & U images for " + str(freq) +
                            " differs!")
        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        img = self._images_dict[freq][stokes][0]
        # Create container for pang-images
        pang_images = list()
        for q_image, u_image in zip(q_images, u_images):
            pang_array = pang_map(q_image.image, u_image.image, mask=mask)
            # Create basic image and add ``pang_array``
            pang_image = Image(imsize=img.imsize, pixref=img.pixref,
                               pixrefval=img.pixrefval, pixsize=img.pixsize,
                               freq=img.freq, stokes='PANG')
            pang_image.image = pang_array
            pang_images.append(pang_image)

        return pang_images


    def create_pol_images(self, freq=None, mask=None):
        """
        Method that creates Polarization Flux images for current collection of
        image instances.

        :param freq: (optional)
             What frequency to use. If ``None`` then assume that only one
             frequency is present in instance. (default: ``None``)
        :param mask: (optional)
            Mask to be applied to arrays before calculation. If ``None`` then
            don't apply mask. Note that ``mask`` must have dimensions of only
            one image, that is it should be 2D array.

        :return:
            List of ``Image`` instances with Polarization Flux maps.

        """
        required_stokeses = ('Q', 'U')

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

        # Check that used frequency has Q & U maps
        stokeses = self.stokeses(freq)
        for stokes in required_stokeses:
            if stokes not in stokeses:
                raise Exception("No stokes " + stokes + " parameter for " +
                                freq + " frequency!")

        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        q_images = self._images_dict[freq]['Q']
        u_images = self._images_dict[freq]['U']
        # Check that we got the same number of ``Q`` and ``U`` images
        if len(q_images) != len(u_images):
            raise Exception("Number of Q & U images for " + str(freq) +
                            " differs!")
        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        img = self._images_dict[freq][stokes][0]
        # Create container for pang-images
        pol_images = list()
        for q_image, u_image in zip(q_images, u_images):
            pol_array = pol_map(q_image.image, u_image.image, mask=mask)
            # Create basic image and add ``pang_array``
            pol_image = Image(imsize=img.imsize, pixref=img.pixref,
                              pixrefval=img.pixrefval, pixsize=img.pixsize,
                              freq=img.freq, stokes='PPOL')
            pol_image.image = pol_array
            pol_images.append(pol_image)

        return pol_images


def rotm_map(freqs, chis, s_chis=None, mask=None):
    """
    Function that calculates Rotation Measure map.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of 2D numpy arrays with polarization positional angles [rad].
    :param s_chis: (optional)
        Iterable of 2D numpy arrays with polarization positional angles
        uncertainties estimates [rad].
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask. Note that ``mask`` must have dimensions of only one image,
        that is it should be 2D array.

    :return:
        Tuple of 2D numpy array with values of Rotation Measure [rad/m**2] and
        2D numpy array with uncertainties map [rad/m**2].

    """
    if s_chis is not None:
        assert len(freqs) == len(chis) == len(s_chis)
    else:
        assert len(freqs) == len(chis)

    chi_cube = np.dstack(chis)
    if s_chis is not None:
        s_chi_cube = np.dstack(s_chis)
    rotm_array = np.empty(np.shape(chi_cube[:, :, 0]))
    s_rotm_array = np.empty(np.shape(chi_cube[:, :, 0]))
    rotm_array[:] = np.nan
    s_rotm_array[:] = np.nan

    if mask is None:
        mask = np.zeros(rotm_array.shape)

    for (x, y), value in np.ndenumerate(rotm_array):
        # If pixel should be masked then just pass by and leave NaN as value
        if mask[x, y]:
            continue

        if s_chis is not None:
            p, pcov = rotm(freqs, chi_cube[x, y, :], s_chi_cube[x, y, :])
        else:
            p, pcov = rotm(freqs, chi_cube[x, y, :])

        if pcov is not np.nan:
            rotm_array[x, y] = p[0]
            s_rotm_array[x, y] = math.sqrt(pcov[0, 0])
        else:
            rotm_array[x, y] = p[0]
            s_rotm_array[x, y] = np.nan

    return rotm_array, s_rotm_array


def pang_map(q_array, u_array, mask=None):
    """
    Function that calculates Polarization Angle map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Polarization Angle values [rad].

    :note:
        ``q_array`` & ``u_array`` must have the same units (e.g. [Jy/beam])

    """
    q_array = np.atleast_2d(q_array)
    u_array = np.atleast_2d(u_array)
    assert q_array.shape == u_array.shape

    if mask is not None:
        q_array = np.ma.array(q_array, mask=mask, fill_value=np.nan)
        u_array = np.ma.array(u_array, mask=mask, fill_value=np.nan)

    return 0.5 * np.arctan2(q_array, u_array)


def cpol_map(q_array, u_array, mask=None):
    """
    Function that calculates Complex Polarization map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Complex Polarization values.

    :note:
        ``q_array`` & ``u_array`` must have the same units (e.g. [Jy/beam]),
        then output array will have the same units.

    """
    q_array = np.atleast_2d(q_array)
    u_array = np.atleast_2d(u_array)
    assert q_array.shape == u_array.shape

    if mask is not None:
        q_array = np.ma.array(q_array, mask=mask, fill_value=np.nan)
        u_array = np.ma.array(u_array, mask=mask, fill_value=np.nan)

    return q_array  + 1j * u_array


def pol_map(q_array, u_array, mask=None):
    """
    Function that calculates Polarization Flux map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Polarization Flux values.

    :note:
        ``q_array`` & ``u_array`` must have the same units (e.g. [Jy/beam])

    """
    cpol_array = cpol_map(q_array, u_array, mask=mask)
    return np.sqrt(cpol_array * cpol_array.conj()).real


def fpol_map(q_array, u_array, i_array, mask=None):
    """
    Function that calculates Fractional Polarization map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param i_array:
        Numpy 2D array of Stokes I values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Fractional Polarization values.

    :note:
        ``q_array``, ``u_array`` & ``i_array`` must have the same units (e.g.
        [Jy/beam])

    """
    cpol_array = cpol_map(q_array, u_array, mask=mask)
    return np.sqrt(cpol_array * cpol_array.conj()).real / i_array


def rotm(freqs, chis, s_chis=None, p0=None):
    """
    Function that calculates Rotation Measure.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of polarization positional angles [rad].
    :param s_chis: (optional)
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

    if s_chis is not None:
        assert len(freqs) == len(chis) == len(s_chis)
    else:
        assert len(freqs) == len(chis)

    p0 = np.array(p0)
    freqs = np.array(freqs)
    chis = np.array(chis)
    if s_chis is not None:
        s_chis = np.array(s_chis)

    def rotm_model(p, freqs):
        lambdasq = (3. * 10 ** 8 / freqs) ** 2
        return p[0] * lambdasq + p[1]

    def weighted_residuals(p, freqs, chis, s_chis):
        return (chis - rotm_model(p, freqs)) / s_chis

    def residuals(p, freqs, chis):
        return chis - rotm_model(p, freqs)

    if s_chis is None:
        func, args = residuals, (freqs, chis,)
    else:
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


def hdi_of_images(images, cred_mass=0.68):
    """
    Function that calculates a width of highest density interval for each pixel
    using user supplied images.
    :param images:
        Iterable of images.
    :param cred_mass: (optional)
        Credibility mass. (default: ``0.68``)
    :return:
        Numpy 2D array with
    """
    images = [np.atleast_2d(image) for image in images]
    # Check that images have the same shape
    assert len(set([image.shape for image in images])) == 1

    images_cube = np.dstack(tuple(image for image in images))
    hdis = np.zeros(np.shape(images_cube[:, :, 0]))
    for (x, y), value in np.ndenumerate(hdis):
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=cred_mass)
        hdis[x, y] = hdi[1] - hdi[0]
    return hdis


if __name__ == '__main__':
    import os
    data_dir = '/home/ilya/vlbi_errors/0148+274/2007_03_01/'
    # Directory with fits-images of bootstrapped data
    i_dir_c1 = data_dir + 'C1/im/I/'
    q_dir_c1 = data_dir + 'C1/im/Q/'
    u_dir_c1 = data_dir + 'C1/im/U/'
    q_dir_c2 = data_dir + 'C2/im/Q/'
    u_dir_c2 = data_dir + 'C2/im/U/'
    q_dir_x1 = data_dir + 'X1/im/Q/'
    u_dir_x1 = data_dir + 'X1/im/U/'
    q_dir_x2 = data_dir + 'X2/im/Q/'
    u_dir_x2 = data_dir + 'X2/im/U/'
    original_cc_fits_file = 'cc.fits'

    # # Testing ``Images.create_error_image``
    # print "Testing ``Images.create_error_image`` method..."
    images = Images()
    images.add_from_fits(wildcard=i_dir_c1 + "cc_*.fits")
    i_error_map = images.create_error_image()
    # images = Images()
    # images.add_from_fits(wildcard=q_dir_c1 + "cc_*.fits")
    # q_error_map = images.create_error_image()
    # images = Images()
    # images.add_from_fits(wildcard=u_dir_c1 + "cc_*.fits")
    # u_error_map = images.create_error_image()

    # # Test rm-creating functions
    # print "Testing rm-creating functions..."
    # chis = [np.zeros(100, dtype=float).reshape((10, 10)) + 2.3,
    #         np.zeros(100, dtype=float).reshape((10, 10)) + 1.3,
    #         np.zeros(100, dtype=float).reshape((10, 10)) + 0.8]
    # s_chis = [np.zeros(100, dtype=float).reshape((10, 10)) + 0.3,
    #           np.zeros(100, dtype=float).reshape((10, 10)) + 0.3,
    #           np.zeros(100, dtype=float).reshape((10, 10)) + 0.3]
    # freqs = np.array([1.4 * 10 ** 9, 5. * 10 ** 9, 8.4 * 10 ** 9])
    # rotm_array_no_s, s_rotm_array_no_s = rotm_map(freqs, chis)
    # rotm_array, s_rotm_array = rotm_map(freqs, chis, s_chis)

    # mask = np.zeros(100).reshape((10, 10))
    # mask[3, 3] = 1
    # ma_rotm_array_no_s, ma_s_rotm_array_no_s = rotm_map(freqs, chis, mask=mask)
    # ma_rotm_array, ma_s_rotm_array = rotm_map(freqs, chis, s_chis, mask=mask)

    # # Testing ``pang_map`` function
    # print "Testing ``pang_map`` function..."
    # q_array = np.zeros(100, dtype=float).reshape((10, 10)) + 2.3
    # u_array = np.zeros(100, dtype=float).reshape((10, 10)) + 0.3
    # chi_array = pang_map(q_array, u_array)

    # mask = np.zeros(100).reshape((10, 10))
    # mask[3, 3] = 1
    # ma_chi_array = pang_map(q_array, u_array, mask=mask)

    # # Testing ``cpol_map`` function
    # print "Testing ``cpol_map`` function..."
    # q_array = np.zeros(100, dtype=float).reshape((10, 10)) + 2.3
    # u_array = np.zeros(100, dtype=float).reshape((10, 10)) + 0.3
    # cpol_array = cpol_map(q_array, u_array)

    # mask = np.zeros(100).reshape((10, 10))
    # mask[3, 3] = 1
    # ma_cpol_array = cpol_map(q_array, u_array, mask=mask)

    # # Testing ``fpol_map`` function
    # print "Testing ``fpol_map`` function..."
    # q_array = np.zeros(100, dtype=float).reshape((10, 10)) + 2.3
    # u_array = np.zeros(100, dtype=float).reshape((10, 10)) + 0.3
    # i_array = np.zeros(100, dtype=float).reshape((10, 10)) + 5.3
    # fpol_array = fpol_map(q_array, u_array, i_array)

    # mask = np.zeros(100).reshape((10, 10))
    # mask[3, 3] = 1
    # ma_fpol_array = fpol_map(q_array, u_array, i_array, mask=mask)

    # # Testing ``Images.create_pang_images``
    # print "Testing ``Images.create_pang_images``..."
    # # Testing one pair of Q & U images
    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
    #                      os.path.join(u_dir_c1, 'cc.fits')])
    # pang_images = images.create_pang_images()
    # # Testing two pairs of Q & U images
    # images = Images()
    # fnames = [os.path.join(q_dir_c1, 'cc_{}.fits'.format(i)) for i in range(1, 11)]
    # fnames += [os.path.join(u_dir_c1, 'cc_{}.fits'.format(i)) for i in
    #            range(1, 11)]
    # images.add_from_fits(fnames)
    # pang_images_10 = images.create_pang_images()

    # # Testing ``Images.create_pol_images``
    # print "Testing ``Images.create_pol_images``..."
    # # Testing one pair of Q & U images
    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
    #                              os.path.join(u_dir_c1, 'cc.fits')])
    # pol_images = images.create_pol_images()
    # # Testing ten pairs of Q & U images
    # images = Images()
    # fnames = [os.path.join(q_dir_c1, 'cc_{}.fits'.format(i)) for i in range(1, 11)]
    # fnames += [os.path.join(u_dir_c1, 'cc_{}.fits'.format(i)) for i in
    #            range(1, 11)]
    # images.add_from_fits(fnames)
    # pol_images_10 = images.create_pol_images()

    # # Testing making error-map for polarization flux images
    # print "Testing making error-map for polarization flux images..."
    # images = Images()
    # images.add_images(pol_images_10)
    # pol_error_image = images.create_error_image()

    # # Testing ``Images.create_rotm_image``
    # print "Testing ``Images.create_rotm_image``..."
    # images = Images()
    # s_pang_arrays = [np.zeros(512 * 512, dtype=float).reshape((512, 512)) + 0.1]
    # s_pang_arrays *= 4
    # # Only one of Q & U at each frequency
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc_1.fits'),
    #                              os.path.join(u_dir_c1, 'cc_1.fits'),
    #                              os.path.join(q_dir_c2, 'cc_1.fits'),
    #                              os.path.join(u_dir_c2, 'cc_1.fits'),
    #                              os.path.join(q_dir_x1, 'cc_1.fits'),
    #                              os.path.join(u_dir_x1, 'cc_1.fits'),
    #                              os.path.join(q_dir_x2, 'cc_1.fits'),
    #                              os.path.join(u_dir_x2, 'cc_1.fits')])

    # mask = np.ones(512 * 512).reshape((512, 512))
    # mask[200:400, 200:400] = 0
    # rotm_image, s_rotm_image = images.create_rotm_image(s_pang_arrays,
    #                                                     mask=mask)
    # rotm_image_no_s, s_rotm_image_no_s = images.create_rotm_image(mask=mask)

    # # Testing blanking ROTM images...
    # print "Testing blanking of ROTM images..."
    # # Blanking mask should be based on polarization flux. One should create
    # # bootstrapped realization of polarization flux images and find error on it.
    # # Then when calculating ROTM use only pixels with POL > error.

    # # TODO: Create mask stacking masks from all bands
    # # First, create polarization flux (PPOL) error image using high frequency
    # # data
    # images = Images()
    # fnames = [os.path.join(q_dir_x2, 'cc_{}.fits'.format(i)) for i in
    #           range(1, 101)]
    # fnames += [os.path.join(u_dir_x2, 'cc_{}.fits'.format(i)) for i in
    #            range(1, 101)]
    # images.add_from_fits(fnames)
    # pol_images_x2_100 = images.create_pol_images()
    # images = Images()
    # images.add_images(pol_images_x2_100)
    # pol_error_image = images.create_error_image(cred_mass=0.99)
    # # Now create mask using model PPOL image and error PPOL image on highest
    # # frequency
    # images = Images()
    # q_dir_ = data_dir + 'X2/im/Q/'
    # u_dir_ = data_dir + 'X2/im/U/'
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
    #                              os.path.join(u_dir_c1, 'cc.fits')])
    # pol_image = images.create_pol_images()[0]
    # mask = pol_image.image < pol_error_image.image
    # # Now make ROTM image with this mask
    # images = Images()
    # # FIXME: Must be imaged of Q&U with the same parameters from original
    # # uv-data
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc_1.fits'),
    #                              os.path.join(u_dir_c1, 'cc_1.fits'),
    #                              os.path.join(q_dir_c2, 'cc_1.fits'),
    #                              os.path.join(u_dir_c2, 'cc_1.fits'),
    #                              os.path.join(q_dir_x1, 'cc_1.fits'),
    #                              os.path.join(u_dir_x1, 'cc_1.fits'),
    #                              os.path.join(q_dir_x2, 'cc_1.fits'),
    #                              os.path.join(u_dir_x2, 'cc_1.fits')])
    # x2masked_rotm_image_no_s, x2masked_s_rotm_image_no_s =\
    #     images.create_rotm_image(mask=mask)

    # # Testing uncertainties estimates for ROTM maps
    # print "Testing uncertainties estimates for ROTM images..."
    # Error on ROTM can be calculated in several ways. First, create Q, U images
    # from bootstrapped uv-data and make ROTM image for each. Then stack them
    # and find error in each pixel or find ``p-value`` of any feature. Second,
    # it can be created using uncertainties of PANG images created from
    # bootstrapped uv-data.
    # Concerning error of PANG-calibration. It can be used in both ways. In
    # first approach - just add random number from PARN error distribution to
    # each PANG map, made from bootstrapped uv-data. In second approach - just
    # add in quadrature estimated PA-calibration error to PANG error images at
    # each frequency.
