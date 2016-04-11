import os
import numpy as np
import glob
from collections import defaultdict
from matplotlib.pyplot import (bar, show, savefig, axvline, xlabel, ylabel,
                               close)
from knuth_hist import histogram
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from utils import (mask_region, mas_to_rad, hdi_of_mcmc, flatten,
                   nested_dict_itervalue, mask_region)
from image import BasicImage, Image, CleanImage, plot
from image_ops import add_dterm_evpa, pol_map, fpol_map, pang_map, rotm_map


# TODO: Option for using only CCc w/o residuals for building images
# TODO: Option for saving ``Images`` instance. One freq/stokes images can be
# saved in multidimensional data part of ``ImageHDU``. If many stokes/freqs =>
# use several ``HDUs``.
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
        return sorted(self._images_dict.keys())

    def stokeses(self, freq):
        return self._images_dict[freq].keys()

    def _create_cube(self, stokes=None, freq=None):
        self._images_cube = np.dstack(tuple(image.image for image in
                                      self._images_dict[freq][stokes]))

    def _save_cube(self, stokes=None, freq=None):
        """
        Method that saves optionally modified images cube back to image
        instances.
        """
        if self._images_cube is None:
            raise Exception("No available images cube to save!")
        for i, array in enumerate(self._images_cube.T):
            self._images_dict[freq][stokes][i].image = array

    def slice(self, pix1, pix2, stokes=None, freq=None):
        """
        Method that returns slice of images along line.

        :param x1:
            Iterable of cordinates of first pixel.
        :param x2:
            Iterable of cordinates of second pixel.
        :return:
            Numpy array of image values for given slice.
        """
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
        # Make slice for each image in ``self`` and stack it
        slices = list()
        for image in self._images_dict[freq][stokes]:
            slices.append(image.slice(pix1, pix2))
        return np.vstack(slices).T

    def apply_pixelwise(self, func, stokes=None, freq=None):
        """
        Method that applies user specified callable to each pixel of stacked
        image pixel by pixel.

        :param func:
            Callable that accept slice of stacked images along image's number
            axis and return optionally modified slice.
        """
        self._create_cube(stokes=stokes, freq=freq)
        for (x, y), value in self._images_cube[..., 0]:
            self._images_cube[x, y, ...] = func(self._images_cube[x, y, ...])
        self._save_cube(stokes=stokes, freq=freq)

    def pixels_histogram(self, stokes=None, freq=None, region=None, mask=None,
                         mode='mean', outdir=None, outname=None,
                         hdi_lines=0.95):
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
        probs, edges = histogram(values, density=True)
        lower_d = np.resize(edges, len(edges) - 1)
        bar(lower_d, probs, width=np.diff(lower_d)[0], linewidth=2, color='w')
        low, high = hdi_of_mcmc(values, cred_mass=hdi_lines)
        axvline(low, color='b', lw=3)
        axvline(high, color='b', lw=3)
        xlabel("Flux Density, [Jy/beam]")
        ylabel("Density")
        show()
        if outname:
            outdir = outdir or os.getcwd()
            savefig(os.path.join(outdir, outname), bbox_inches='tight', dpi=200)
            close()

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
            fnames = sorted(glob.glob(wildcard))
        if len(fnames) < 1:
            raise Exception("Need at least 1 image")

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

    # FIXME: If ``create_cube`` is ``False`` then we don't nee ``freq`` &
    # ``stokes``
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

        # FIXME: raise ValueError in apply...
        # # For PANG images pre-process angles
        # if stokes == 'PANG':
        #     self.apply_pixelwise(unwrap_, stokes='PANG', freq=freq)

        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        img = self._images_dict[freq][stokes][0]
        hdis = np.zeros(np.shape(self._images_cube[:, :, 0]))
        for (x, y), value in np.ndenumerate(hdis):
            hdi = hdi_of_mcmc(self._images_cube[x, y, :], cred_mass=cred_mass)
            hdis[x, y] = hdi[1] - hdi[0]
        # Create ``Image`` instance and add map of error
        image = Image()
        image._construct(imsize=img.imsize, pixsize=img.pixsize,
                         pixref=img.pixref, stokes=stokes,
                         freq=img.freq, pixrefval=img.pixrefval)
        image.image = hdis
        return image

    # FIXME: Implement option for many (equal number) of Q & U images for each
    # frequency like in ``Images.create_pang_images``
    # TODO: Option for plotting PANG vs. wavelength squared for pixels
    def create_rotm_image(self, s_pang_arrays=None, freqs=None, mask=None, n=0,
                          outfile=None, outdir=None, ext='png'):
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
        img = self._images_dict[freq][stokes][n]

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
        rotm_array, s_rotm_array, chisq_array = rotm_map(freqs, pang_arrays,
                                                         s_pang_arrays,
                                                         mask=mask,
                                                         outfile=outfile,
                                                         outdir=outdir, ext=ext)
        rotm_image = Image()
        rotm_image._construct(imsize=img.imsize, pixsize=img.pixsize,
                              pixref=img.pixref, stokes='ROTM',
                              freq=tuple(freqs), pixrefval=img.pixrefval)
        rotm_image.image = rotm_array
        # FIXME: use ``tuple`` for frequency container cause it is hashable
        s_rotm_image = Image()
        s_rotm_image._construct(imsize=img.imsize, pixsize=img.pixsize,
                                pixref=img.pixref, stokes='ROTM',
                                freq=tuple(freqs), pixrefval=img.pixrefval)
        s_rotm_image.image = s_rotm_array

        return rotm_image, s_rotm_image

    def create_rotm_images(self, s_pang_arrays=None, freqs=None, mask=None):
        """
        Method that creates ROTM images from series of bootstrapped data.

        :param s_pang_arrays:
        :param freqs:
        :param mask:
        :return:
            Instance of ``Images`` class with calculated ROTM images.
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
        n_replications = None
        for freq in freqs:
            q_images = self._images_dict[freq]['Q']
            u_images = self._images_dict[freq]['U']
            # Check that we got the same number of ``Q`` and ``U`` images
            if len(q_images) != len(u_images):
                raise Exception("Different # of Q & U images for " + str(freq) +
                                " MHz!")
            if n_replications is None:
                n_replications = len(q_images)
            else:
                if n_replications != len(q_images):
                    raise Exception("Each frequency must contains the same # of"
                                    " Q & U images!")

        # For each replication create ROTM map and add it to ``Images`` instance
        images = Images()
        for i in range(n_replications):
            print "Creating {} image of {} replications".format(i + 1,
                                                                n_replications)
            rotm_image, s_rotm_image = self.create_rotm_image(s_pang_arrays,
                                                              freqs, mask, i)
            images.add_image(rotm_image)

        return images

    # TODO: Implement ``create_pang_image`` & use it to implement this method as
    # in ``create_rotm_images``
    # ``Images`` instance can be easily obtained from list of ``Image``
    # instances
    def create_pang_images(self, freq=None, mask=None, convolved=True):
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
            if convolved:
                pang_array = pang_map(q_image.image, u_image.image, mask=mask)
            else:
                pang_array = pang_map(q_image._image, u_image._image, mask=mask)

            # Create basic image and add ``pang_array``
            pang_image = Image(imsize=img.imsize, pixref=img.pixref,
                               pixrefval=img.pixrefval, pixsize=img.pixsize,
                               freq=img.freq, stokes='PANG')
            if convolved:
                pang_image.image = pang_array
            else:
                pang_image._image = pang_array
            pang_images.append(pang_image)

        return pang_images

    # TODO: Implement ``create_pol_image`` & use it to implement this method as
    # in ``create_rotm_images``
    def create_pol_images(self, freq=None, mask=None, convolved=True):
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
        :param convolved: (optional)
            Use convolved images in construction? (default: ``True``)

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
            if convolved:
                pol_array = pol_map(q_image.image, u_image.image, mask=mask)
            else:
                pol_array = pol_map(q_image._image, u_image._image, mask=mask)
            # Create image and add ``pol_array``
            pol_image = Image()
            pol_image._construct(imsize=img.imsize, pixsize=img.pixsize,
                                 pixref=img.pixref, stokes='PPOL',
                                 freq=img.freq, pixrefval=img.pixrefval)
            if convolved:
                pol_image.image = pol_array
            else:
                pol_image._image = pol_array
            pol_images.append(pol_image)

        return pol_images

    # TODO: Implement ``create_pol_image`` & use it to implement this method as
    # in ``create_rotm_images``
    def create_fpol_images(self, freq=None, mask=None):
        """
        Method that creates Fractional Polarization images for current
        collection of image instances.

        :param freq: (optional)
             What frequency to use. If ``None`` then assume that only one
             frequency is present in instance. (default: ``None``)
        :param mask: (optional)
            Mask to be applied to arrays before calculation. If ``None`` then
            don't apply mask. Note that ``mask`` must have dimensions of only
            one image, that is it should be 2D array.

        :return:
            List of ``Image`` instances with Fractional Polarization maps.

        """
        required_stokeses = ('I', 'Q', 'U')

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

        # Check that used frequency has I, Q & U maps
        stokeses = self.stokeses(freq)
        for stokes in required_stokeses:
            if stokes not in stokeses:
                raise Exception("No stokes " + stokes + " parameter for " +
                                freq + " frequency!")

        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        i_images = self._images_dict[freq]['I']
        q_images = self._images_dict[freq]['Q']
        u_images = self._images_dict[freq]['U']
        # Check that we got the same number of I, Q & U images
        if len(i_images) != len(q_images) or len(q_images) != len(u_images):
            raise Exception("Number of I, Q & U images for " + str(freq) +
                            " differs!")
        # Get some image from stacked to use it parameters for saving output. It
        # doesn't matter what image - they all are checked to have the same
        # basic parameters
        img = self._images_dict[freq][stokes][0]
        # Create container for fpol-images
        fpol_images = list()
        for i_image, q_image, u_image in zip(i_images, q_images, u_images):
            fpol_array = fpol_map(q_image.image, u_image.image, i_image.image,
                                  mask=mask)
            # Create basic image and add ``fpol_array``
            fpol_image = Image(imsize=img.imsize, pixref=img.pixref,
                               pixrefval=img.pixrefval, pixsize=img.pixsize,
                               freq=img.freq, stokes='FPOL')
            fpol_image.image = fpol_array
            fpol_images.append(fpol_image)

        return fpol_images

if __name__ == '__main__':
    pass
    data_dir = '/home/ilya/Dropbox/4vlbi_errors/DENISE/'
    d_term_std = 0.002
    evpa_std = 2.
    import os
    # For each frequency create mask based on PPOL distribution
    ppol_error_images_dict = dict()
    ppol_images_dict = dict()
    ppol_masks_dict = dict()
    for freq in ('l18', 'l20', 'l21', 'l22'):
        images = Images()
        images.add_from_fits(wildcard=os.path.join(data_dir,
                                                   '1038+064.{}.boot_*.*cn.fits'.format(freq)))
        ppol_images = Images()
        ppol_images.add_images(images.create_pol_images())
        ppol_error_image = ppol_images.create_error_image(cred_mass=0.95)
        ppol_error_images_dict.update({freq: ppol_error_image})
        images = Images()
        images.add_from_fits(wildcard=os.path.join(data_dir,
                                                   '1038+064.{}.common.*cn.fits'.format(freq)))
        ppol_image = images.create_pol_images()[0]
        ppol_images_dict.update({freq: ppol_image})
        mask = ppol_image.image < ppol_error_image.image
        print mask
        ppol_masks_dict.update({freq: mask})

    # Create overall mask for PPOL flux
    masks = [np.array(mask, dtype=int) for mask in ppol_masks_dict.values()]
    ppol_mask = np.zeros(masks[0].shape, dtype=int)
    for mask in masks:
        ppol_mask += mask
    ppol_mask[ppol_mask != 0] = 1

    # Create ROTM images with calculated mask
    images = Images()
    images.add_from_fits(wildcard=os.path.join(data_dir,
                                               '1038+064.l*.boot_*.*cn.fits'))
    rotm_images_list = images.create_rotm_images(mask=ppol_mask)

    # import os
    # # data_dir = '/home/ilya/vlbi_errors/0148+274/2007_03_01/'
    # data_dir = '/home/ilya/vlbi_errors/0952+179/2007_04_30/'
    # # Directory with fits-images of bootstrapped data
    # i_dir_c1 = data_dir + 'C1/im/I/'
    # i_dir_c2 = data_dir + 'C2/im/I/'
    # i_dir_x1 = data_dir + 'X1/im/I/'
    # i_dir_x2 = data_dir + 'X2/im/I/'
    # q_dir_c1 = data_dir + 'C1/im/Q/'
    # u_dir_c1 = data_dir + 'C1/im/U/'
    # q_dir_c2 = data_dir + 'C2/im/Q/'
    # u_dir_c2 = data_dir + 'C2/im/U/'
    # q_dir_x1 = data_dir + 'X1/im/Q/'
    # u_dir_x1 = data_dir + 'X1/im/U/'
    # q_dir_x2 = data_dir + 'X2/im/Q/'
    # u_dir_x2 = data_dir + 'X2/im/U/'
    # # original_cc_fits_file = 'cc.fits'

    # # Testing ``Images.create_error_image``
    # print "Testing ``Images.create_error_image`` method..."
    # images = Images()
    # images.add_from_fits(wildcard=i_dir_c1 + "cc_*.fits")
    # i_error_map = images.create_error_image()
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
    # pang_image = images.create_pang_images()[0]
    # # Testing many of Q & U images
    # images = Images()
    # fnames = [os.path.join(q_dir_c1, 'cc_{}.fits'.format(i)) for i in
    #           range(1, 201)]
    # fnames += [os.path.join(u_dir_c1, 'cc_{}.fits'.format(i)) for i in
    #            range(1, 201)]
    # images.add_from_fits(fnames)
    # pang_images_list_200 = images.create_pang_images()

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
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc_orig.fits'),
    #                              os.path.join(u_dir_c1, 'cc_orig.fits'),
    #                              os.path.join(q_dir_c2, 'cc_orig.fits'),
    #                              os.path.join(u_dir_c2, 'cc_orig.fits'),
    #                              os.path.join(q_dir_x1, 'cc_orig.fits'),
    #                              os.path.join(u_dir_x1, 'cc_orig.fits'),
    #                              os.path.join(q_dir_x2, 'cc_orig.fits'),
    #                              os.path.join(u_dir_x2, 'cc_orig.fits')])

    # mask = np.ones(512 * 512).reshape((512, 512))
    # mask[200:400, 200:400] = 0
    # rotm_image, s_rotm_image = images.create_rotm_image(s_pang_arrays,
    #                                                     mask=mask)
    # rotm_image_no_s, s_rotm_image_no_s = images.create_rotm_image(mask=mask)

    # Testing ``Images.create_rotm_image`` from bootstrapped data
    # print "Testing ``Images.create_rotm_image`` from bootstrapped data..."

    # Testing blanking ROTM images...
    # print "Testing blanking of ROTM images..."
    # Blanking mask should be based on polarization flux. One should create
    # bootstrapped realization of polarization flux images and find error on it.
    # Then when calculating ROTM use only pixels with POL > error.

    # First, create polarization flux (PPOL) error image for each frequency
    # data
    # band_dir = {'c1': {'i': i_dir_c1, 'q': q_dir_c1, 'u': u_dir_c1},
    #             'c2': {'i': i_dir_c2, 'q': q_dir_c2, 'u': u_dir_c2},
    #             'x1': {'i': i_dir_x1, 'q': q_dir_x1, 'u': u_dir_x1},
    #             'x2': {'i': i_dir_x2, 'q': q_dir_x2, 'u': u_dir_x2}}
    # mask_last = None
    # fnames = list()
    # images_allbands_50 = Images()
    # print "Constructing Images instance for all bands for all bootstrapped" \
    #       " data..."
    # for band in ('c1', 'c2', 'x1', 'x2'):
    #     print ""
    #     fnames = [os.path.join(band_dir[band]['q'], 'cc_{}.fits'.format(i)) for
    #               i in range(1, 51)]
    #     fnames += [os.path.join(band_dir[band]['u'], 'cc_{}.fits'.format(i)) for
    #                i in range(1, 51)]
    #     fnames += [os.path.join(band_dir[band]['i'], 'cc_{}.fits'.format(i)) for
    #                i in range(1, 51)]
    #     images_allbands_50.add_from_fits(fnames)

    # for i, band in enumerate(('c1', 'c2', 'x1', 'x2')):
    #     print "Creating mask for {}-band PPOL and I image".format(band)
    #     i_error_image = images_allbands_50.create_error_image(stokes='I',
    #                                                            freq=images_allbands_50.freqs[i],
    #                                                            cred_mass=0.95)
    #     pol_images_50 = images_allbands_50.create_pol_images(freq=images_allbands_50.freqs[i])
    #     images = Images()
    #     images.add_images(pol_images_50)
    #     pol_error_image = images.create_error_image(cred_mass=0.95)
    #     images = Images()
    #     images.add_from_fits(fnames=[os.path.join(band_dir[band]['q'],
    #                                               'cc_orig.fits'),
    #                                  os.path.join(band_dir[band]['u'],
    #                                               'cc_orig.fits')])
    #     pol_image = images.create_pol_images()[0]
    #     i_image = create_clean_image_from_fits_file(os.path.join(band_dir[band]['i'],
    #                                                              'cc_orig.fits'))
    #     mask_pol = pol_image.image < pol_error_image.image
    #     mask_i = i_image.image < i_error_image.image
    #     mask = np.logical_or(mask_pol, mask_i)
    #     if mask_last is not None:
    #         mask = np.logical_or(mask, mask_last)
    #     mask_last = mask.copy()

    # # Now make ROTM image with this mask
    # print "Constructing Images instance with original data..."
    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc_orig.fits'),
    #                              os.path.join(u_dir_c1, 'cc_orig.fits'),
    #                              os.path.join(q_dir_c2, 'cc_orig.fits'),
    #                              os.path.join(u_dir_c2, 'cc_orig.fits'),
    #                              os.path.join(q_dir_x1, 'cc_orig.fits'),
    #                              os.path.join(u_dir_x1, 'cc_orig.fits'),
    #                              os.path.join(q_dir_x2, 'cc_orig.fits'),
    #                              os.path.join(u_dir_x2, 'cc_orig.fits')])
    # print "Creating original ROTM image with constructed mask..."
    # masked_rotm_image_no_s, masked_s_rotm_image_no_s =\
    #     images.create_rotm_image(mask=mask)

    # # # Testing uncertainties estimates for ROTM maps
    # # print "Testing uncertainties estimates for ROTM images..."
    # # Error on ROTM can be calculated in several ways. First, create Q, U images
    # # from bootstrapped uv-data and make ROTM image for each. Then stack them
    # # and find error in each pixel or find ``p-value`` of any feature. Second,
    # # it can be created using uncertainties of PANG images created from
    # # bootstrapped uv-data.
    # # Concerning error of PANG-calibration. It can be used in both ways. In
    # # first approach - just add random number from PANG error distribution to
    # # each PANG map, made from bootstrapped uv-data. In second approach - just
    # # add in quadrature estimated PA-calibration error to PANG error images at
    # # each frequency.

    # # Create ROTM image for each of the bootstrapped Q & U image
    # print "Creating ROTM images of bootstrapped data with constructed mask..."
    # # Now create 50 ROTM images with mask basked on PPOL & I bootstrapped data
    # rotm_images_50 = images_allbands_50.create_rotm_images(mask=mask)
    # print "Creating ERROR ROTM image from bootstrapped ROTM images..."
    # rotm_error_50 = rotm_images_50.create_error_image()

    # print "Creating ONE ROTM image from original data + bootstrapped PANG" \
    #       "errors"
    # # Now create 50 PANG images with mask basked on PPOL & I bootstrapped data
    # bands = ['c1', 'c2', 'x1', 'x2']
    # pang_50_bands = dict()
    # for i, freq in enumerate(images_allbands_50.freqs):
    #     images = Images()
    #     print "Creating boot PANG images for band {}".format(bands[i])
    #     images.add_images(images_allbands_50.create_pang_images(freq=freq,
    #                                                             mask=mask))
    #     pang_50_bands.update({bands[i]: images})

    # pang_error_maps = dict()
    # # Create PANG error maps for each band
    # #TODO: Use Q&U error images for creating PANG error maps
    # for band, pang_images in pang_50_bands.iteritems():
    #     print "Creating PANG error image for band {}".format(band)
    #     pang_error_maps.update({band: pang_images.create_error_image()})

    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc_orig.fits'),
    #                              os.path.join(u_dir_c1, 'cc_orig.fits'),
    #                              os.path.join(q_dir_c2, 'cc_orig.fits'),
    #                              os.path.join(u_dir_c2, 'cc_orig.fits'),
    #                              os.path.join(q_dir_x1, 'cc_orig.fits'),
    #                              os.path.join(u_dir_x1, 'cc_orig.fits'),
    #                              os.path.join(q_dir_x2, 'cc_orig.fits'),
    #                              os.path.join(u_dir_x2, 'cc_orig.fits')])
    # print "Creating original ROTM image with constructed mask..."
    # masked_rotm_image_w_s, masked_s_rotm_image_w_s = \
    #     images.create_rotm_image(s_pang_arrays=[pang_error_maps[band].image for
    #                                             band in bands],
    #                              mask=mask)

    # # Creating original C1 image w new resolution
    # i_image = create_clean_image_from_fits_file(os.path.join(band_dir['c1']['i'],
    #                                                          'cc_orig.fits'))

    # # Testing RM +/-pi*n stuff
    # c1_q_image = images._images_dict[4608458750.0]['Q'][0]
    # c1_u_image = images._images_dict[4608458750.0]['U'][0]
    # c2_q_image = images._images_dict[5003458750.0]['Q'][0]
    # c2_u_image = images._images_dict[5003458750.0]['U'][0]
    # x1_q_image = images._images_dict[8108458750.0]['Q'][0]
    # x1_u_image = images._images_dict[8108458750.0]['U'][0]
    # x2_q_image = images._images_dict[8429458750.0]['Q'][0]
    # x2_u_image = images._images_dict[8429458750.0]['U'][0]

    # c1_pang_array = pang_map(c1_q_image.image, c1_u_image.image, mask=mask)
    # c2_pang_array = pang_map(c2_q_image.image, c2_u_image.image, mask=mask)
    # x1_pang_array = pang_map(x1_q_image.image, x1_u_image.image, mask=mask)
    # x2_pang_array = pang_map(x2_q_image.image, x2_u_image.image, mask=mask)

    # c1_pang_error_array = pang_error_maps['c1'].image
    # c2_pang_error_array = pang_error_maps['c2'].image
    # x1_pang_error_array = pang_error_maps['x1'].image
    # x2_pang_error_array = pang_error_maps['x2'].image

    # # rotm_array, s_rotm_array = rotm_map(images.freqs, [c1_pang_array,
    # #                                                    c2_pang_array,
    # #                                                    x1_pang_array,
    # #                                                    x2_pang_array],
    # #                                     s_chis=[c1_pang_error_array,
    # #                                             c2_pang_error_array,
    # #                                             x1_pang_error_array,
    # #                                             x2_pang_error_array],
    # #                                     mask=mask,
    # #                                     outfile='LinearFit',
    # #                                     outdir='/home/ilya/vlbi_errors/')


    # print "Average BOOTSTRAPPED ROTM  images..."
    # average_ROTM = np.mean(np.dstack(tuple(image.image for image in
    #                                        rotm_images_50.images)), axis=2)
    # # Plot original ROTM (LSQ fit w/o errors)
    # # plot(contours=i_image.image_w_residuals,
    #      colors=masked_rotm_image_w_s.image,
    #      x=i_image.x[0, :], y=i_image.y[:, 0], blc=(245, 245), trc=(280, 315),
    #      min_abs_level=0.0005, colors_mask=mask,
    #      plot_title="0952+179 ROTM w s",
    #      # color_clim=[0, 120])
    #      outfile='0952+179_ROTM_w_s',
    #      outdir='/home/ilya/vlbi_errors/')
