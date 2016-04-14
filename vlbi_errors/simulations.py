import os
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils import get_fits_image_info, degree_to_rad, hdi_of_arrays
from image import BasicImage, Image
from images import Images
from utils import (mask_region, mas_to_rad, find_card_from_header, create_grid,
                   gaussian, gaussian_beam)
from image_ops import (pol_map, pang_map, rotm_map, spix_map, image_slice)
from from_fits import create_clean_image_from_fits_file
from model import Model
from uv_data import UVData
from components import DeltaComponent, ImageComponent
from image_ops import pang_map, pol_map
from spydiff import clean_difmap
from image import plot as iplot
from conf_bands import create_sim_conf_band


def alpha(imsize, center, y0, k=0.5, const=-0.5):
    """
    Function that defines model of spectral index distribution.

    :param imsize:
        Image size along x & y directions [pixels, pixels].
    :param center:
        Center of image in x & y [pixel #, pixel #].
    :param y0:
        Pixel number along jet direction with sigmoid midpoint.
    :param k: (optional)
        Steepness of the sigmoid. (default: ``0.5``)
    :param const: (optional)
        Value of `base` spectral index. (default: ``-0.5``)

    :return:
        Numpy 2D array with spectral index distribution.
    """
    # Along jet is x
    y, x = create_grid(imsize)
    x -= center[0]
    y -= center[1]
    alpha = -1. / (1. + np.exp(-k * (y - y0))) - const
    return alpha


def rotm(imsize, center, grad_value=5., rm_value_0=0.0):
    """
    Function that defines model of ROTM gradient distribution.

    :param imsize:
        Image size along x & y directions [pixels, pixels].
    :param center:
        Center of image in x & y [pixel #, pixel #].
    :param grad_value:
        Value of gradient [rad/m/m/pixel].
    :param rm_value_0: (optional)
        Value of ROTM at center [rad/m/m]. (default: ``0.0``)

    :return:
        Numpy 2D array with rotation measure distribution with gradient.
    """
    # Transverse to jet is x
    y, x = create_grid(imsize)
    x -= center[0]
    y -= center[1]
    return grad_value * x + rm_value_0


def create_jet_model_image(width, j_length, cj_length, max_flux, imsize,
                           center, gauss_peak=0.001, dist_from_core=24,
                           gauss_bmaj=10, gauss_e=1., gauss_bpa=0., cut=0.0001):
    """
    Function that returns image of jet.

    :param width:
        Width of jet [pxl].
    :param j_length:
        Length of jet [pxl].
    :param cj_length:
        Length of contr-jet [pxl].
    :param max_flux:
        Peak flux at maximum [pxl].
    :param imsize:
        Tuple of image size.
    :param center:
        Tuple of image center.
    :return:
        2D numpy array with jet image.
    """
    x, y = create_grid(imsize)
    x -= center[0]
    y -= center[1]
    max_flux = float(max_flux)
    along = np.where(x > 0, -(max_flux / j_length) * x,
                     -(max_flux / cj_length ** 2.) * x ** 2.)
    perp = -(max_flux / (width / 2) ** 2.) * y ** 2.
    image = max_flux + along + perp
    image[image < 0] = 0

    # Jet feature
    if gauss_peak:
        gaussian_ = gaussian(gauss_peak, dist_from_core, 0, gauss_bmaj, gauss_e,
                             gauss_bpa)
        image += gaussian_(x, y)
        image[image < cut] = 0.

    return image


# TODO: Models could consist of many components - ``stokes_models`` shouls be
# ``Model`` instances, not numpy arrays
class ModelGenerator(object):
    """
    # TODO: Rename to ``stokes_model_images``
    :param stokes_models:
        Model of Stokes parameters distribution. Dictionary with keys - Stokes
        parameters and values - image of Stokes distribution used as model (that
        is 2D numpy arrays of fluxes). Possible values: ``I``, ``Q``, ``U``,
        ``V``, ``PPOL``, ``PANG``, ``FPOL``.
    :param x:
        Iterable of x-coordinates [rad].
    :param y:
        Iterable of y-coordinates [rad].
    :param freq: (optional)
        Frequency at which models [Hz]. If ``None`` then +infinity. (default:
        ``None``)
    :param alpha: (optional)
        2D array of spectral index distribution. If ``None`` then use uniform
        distribution with zero value. (default: ``None``)
    :param rotm: (optional)
        2D array of rotation measure distribution. If ``None`` then use uniform
        distribution with zero value. (default: ``None``)
    """
    def __init__(self, stokes_models, x, y, freq=None, alpha=None,
                 rotm=None):
        self.stokes_models = stokes_models
        images = stokes_models.values()
        if not images:
            raise Exception("Need at least one model")
        # TODO: Can model images have different parameters (size, pixels)?
        shape = images[0].shape
        self.image_shape = shape
        self._x = x
        self._y = y
        if freq is None:
            self.freq = +np.inf
            self.lambda_sq = 0.
        else:
            self.freq = freq
            self.lambda_sq = (3. * 10 ** 8 / freq) ** 2
        self.alpha = alpha
        if alpha is None:
            self.alpha = np.zeros(shape, dtype=float)
        self.rotm = rotm
        if rotm is None:
            self.rotm = np.zeros(shape, dtype=float)

    def _get_chi_0(self):
        if 'PANG' in self.stokes:
            chi_0 = self.stokes_models['PANG']
        elif 'Q' in self.stokes and 'U' in self.stokes:
            chi_0 = pang_map(self.stokes_models['Q'], self.stokes_models['U'])
        else:
            raise Exception("No Pol. Angle information available!")
        return chi_0

    def create_models(self, stokes_models=None):
        """
        Create instances of ``Model`` class using current model images.

        :return:
            Dictionary of ``Model`` instances.
        """
        models = dict()
        stokes_models = stokes_models or self.stokes_models
        for stokes, image in stokes_models.items():
            model = Model(stokes=stokes)
            image_component = ImageComponent(image, self._x, self._y)
            model.add_components(image_component)
            models.update({stokes: model})
        return models

    def create_models_for_frequency(self, freq):
        """
        Create instance of ``Model`` class for given frequency.

        :param freq:
            Frequency at which evaluate and return models [Hz].

        :return:
            Dictionary of ``Model`` isntances.
        """
        stokes_models = self.move_for_freq(freq)
        return self.create_models(stokes_models)

    def move_for_freq(self, freq):
        """
        Update images of models using current instance ``alpha_func`` and
        ``rotm_func`` attributes.

        :param freq:
            Frequency to move to [Hz].

        :return:
            Updated dictionary with model images.
        """
        if 'Q' in self.stokes and 'U' not in self.stokes:
            raise Exception('Need to update both Q & U simultaneously!')
        if 'U' in self.stokes and 'Q' not in self.stokes:
            raise Exception('Need to update both Q & U simultaneously!')
        if 'PPOL' in self.stokes and 'PANG' not in self.stokes:
            raise Exception('Need to update both PPOL & PANG simultaneously!')
        if 'PANG' in self.stokes and 'PPOL' not in self.stokes:
            raise Exception('Need to update both PPOL & PANG simultaneously!')

        stokes_models = self.stokes_models.copy()
        if 'I' in self.stokes:
            i_image = self._move_i_to_freq(self.stokes_models['I'], freq)
            stokes_models.update({'I': i_image})
        if 'FPOL' in self.stokes:
            ppol_image = i_image * self.stokes_models['FPOL']
        # Now convert Q&U or PPOL&PANG
        if 'Q' in self.stokes and 'U' in self.stokes:
            q_image, u_image = self._move_qu_to_freq(self.stokes_models['Q'],
                                                     self.stokes_models['U'],
                                                     freq)
            stokes_models.update({'Q': q_image, 'U': u_image})

        return stokes_models

    def _move_i_to_freq(self, image_i, freq):
        return image_i.copy() * (freq / self.freq) ** self.alpha

    def _move_qu_to_freq(self, image_q, image_u, freq):
        # First move stokes Q&U to other frequency
        # image_q = self._move_i_to_freq(self.stokes_models['Q'], freq)
        # image_u = self._move_i_to_freq(self.stokes_models['U'], freq)
        image_pol = pol_map(image_q.copy(), image_u.copy())
        image_pang = pang_map(image_q.copy(), image_u.copy())
        lambda_sq = (3. * 10 ** 8 / freq) ** 2
        q_image = image_pol * np.cos(2. * (image_pang +
                                           self.rotm.copy() * (lambda_sq -
                                                        self.lambda_sq)))
        u_image = image_pol * np.sin(2. * (image_pang +
                                           self.rotm.copy() * (lambda_sq -
                                                        self.lambda_sq)))
        return q_image, u_image

    @property
    def stokes(self):
        return self.stokes_models.keys()

    # TODO: Models could consists of several components. Don't use
    # ``Model._compnents``
    def get_stokes_images(self, frequency, i_cut_frac=None, beam=None,
                          mask_before_convolve=True):
        """
        Get stokes images for given frequency.

        :param frequency:
            Frequency on which to calculate models [Hz]
        :param i_cut_frac: (optional)
            Fraction of stokes `I` intensity to create mask. If ``None`` then
            don't mask images.
        :param beam: (optional)
            Iterable of 3 beam parameters - bmaj [mas], bmean [mas], bpa [deg]
            to optionally convolve images
        :param mask_before_convolve: (optional)
            Boolean. Mask maps with ``i_cut_frac`` before or after convolution?

        :return:
            Dictionary with keys specifying stokes and values - 2D numpy arrays
            (optionally masked) with images.
        """
        models = self.create_models_for_frequency(frequency)
        i_image = models['I']._components[0].image
        if i_cut_frac is not None:
            do_mask = True
            mask = i_image < i_cut_frac * np.max(i_image)
        else:
            do_mask = False
        if beam:
            beam_image = gaussian_beam(self.image_shape[0], beam[0], beam[1],
                                       beam[2] + 90., self.image_shape[1])
        result = dict()
        for stokes in self.stokes:
            image = models[stokes]._components[0].image
            if do_mask and mask_before_convolve:
                image = np.ma.array(image, mask=mask)
            if beam:
                image = signal.fftconvolve(image, beam_image, mode='same')
                if do_mask and not mask_before_convolve:
                    image = np.ma.array(image, mask=mask)
            result.update({stokes: image})

        return result

    def get_pol_maps(self, frequency, i_cut_frac=None, beam=None,
                     mask_before_convolve=True):
        assert 'Q' in self.stokes
        assert 'U' in self.stokes
        images = self.get_stokes_images(frequency, i_cut_frac=i_cut_frac,
                                        beam=beam,
                                        mask_before_convolve=mask_before_convolve)
        pola = pang_map(images['Q'], images['U'])
        poli = pol_map(images['Q'], images['U'])
        return {'PPOL': poli, 'PANG': pola}

    def create_i_mask(self, frequency, i_cut_frac, beam=None,
                      mask_before_convolve=True):
        """
        Create mask base on stokes `I` flux on specified frequency.

        :param frequency:
            Frequency on which to calculate models [Hz]
        :param i_cut_frac:
            Fraction of stokes `I` intensity to create mask.
        :param beam: (optional)
            Iterable of 3 beam parameters - bmaj [mas], bmean [mas], bpa [deg]
            to optionally convolve images
        :param mask_before_convolve:
            Boolean. Mask maps with ``i_cut_frac`` before or after convolution?

        :return:
            Mask array.
        """
        image = self.get_stokes_images(frequency, beam=beam)['I']
        if mask_before_convolve:
            mask = image < i_cut_frac * np.max(image)
        else:
            beam_image = gaussian_beam(self.size[0], beam[0], beam[1],
                                       beam[2] + 90., self.size[1])
            image = signal.fftconvolve(image, beam_image, mode='same')
            mask = image < i_cut_frac * np.max(image)

        return mask

    def get_rotm_map(self, frequencies, i_cut_frac=None, beam=None,
                     mask_before_convolve=True, rotm_mask=None):
        """
        Calculate model ROTM map (optionally for convolved `Q` & `U` images) for
        a user-specified frequencies.

        :param frequencies:
            Iterable of frequencies [Hz].
        :param i_cut_frac: (optional)
            Fraction of stokes `I` intensity to create mask. If ``None`` then
            don't mask images.
        :param beam: (optional)
            Iterable of 3 beam parameters - bmaj [mas], bmean [mas], bpa [deg]
            to optionally convolve images
        :param mask_before_convolve:
            Boolean. Mask maps with ``i_cut_frac`` before or after convolution?
        :param rotm_mask: (optional)
            Mask used when creating ROTM map. If ``None`` the don't use mask.
            (default: ``None``)

        :return:
            Output of ``images_ops.rotm_map`` for model for given frequencies.
            Tuple of 2D numpy array with values of Rotation Measure [rad/m**2],
            2D numpy array with uncertainties map [rad/m**2] and 2D numpy array
            with chi squared values of linear fit.
        """
        from collections import OrderedDict
        images = OrderedDict()
        for frequency in sorted(frequencies):
            images[frequency] = self.get_pol_maps(frequency,
                                                  i_cut_frac=i_cut_frac,
                                                  beam=beam,
                                                  mask_before_convolve=mask_before_convolve)['PANG']
        return rotm_map(frequencies, images.values(), mask=rotm_mask)


    def get_spix_map(self, frequencies, i_cut_frac=None, beam=None,
                     mask_before_convolve=True, spix_mask=None):
        """
        Calculate model SPIX map (optionally for convolved `I` images) for a
        user-specified frequencies.

        :param frequencies:
            Iterable of frequencies [Hz].
        :param i_cut_frac: (optional)
            Fraction of stokes `I` intensity to create mask. If ``None`` then
            don't mask images.
        :param beam: (optional)
            Iterable of 3 beam parameters - bmaj [mas], bmean [mas], bpa [deg]
            to optionally convolve images
        :param mask_before_convolve:
            Boolean. Mask maps with ``i_cut_frac`` before or after convolution?
        :param spix_mask: (optional)
            Mask used when creating SPIX map. If ``None`` the don't use mask.
            (default: ``None``)

        :return:
            Output of ``images_ops.spix_map`` for model for given frequency.
        """
        from collections import OrderedDict
        images = OrderedDict()
        for frequency in sorted(frequencies):
            images[frequency] = self.get_stokes_images(frequency,
                                                       i_cut_frac=i_cut_frac,
                                                       beam=beam,
                                                       mask_before_convolve=mask_before_convolve)['I']
        return spix_map(frequencies, images.values(), mask=spix_mask)


class Simulation(object):
    """
    Basic class that handles simulations of VLBI observations.

    :param observed_uv:
        Instance of ``UVData`` class with observed uv-data.

    """
    def __init__(self, observed_uv):
        self.observed_uv = observed_uv
        self.simulated_uv = None
        self.models = dict()
        self.observed_noise = observed_uv.noise()
        self._noise = None

    @property
    def frequency(self):
        """
        Shortcut to frequency in Hz.
        :return:
        """
        return self.observed_uv.band_center

    def add_true_model(self, model):
        """
        Add `true` model.

        :param model:
            Instance of ``Model`` class.
        """
        self.models.update({model.stokes: model})

    def add_true_models(self, models):
        """
        Add `true` models.

        :param models:
            Iterable of ``Model`` class instances.
        """
        for model in models:
            self.add_true_model(model)

    def simulate(self):
        """
        Simulate uv-data.
        """
        print "Simulating..."
        self.simulated_uv = copy.deepcopy(self.observed_uv)
        self.simulated_uv.substitute(self.models.values())
        self.simulated_uv.noise_add(self.noise)
        print "Simulations finished"

    def save_fits(self, fname):
        if self.simulated_uv is not None:
            print "Saving to {}".format(fname)
            self.simulated_uv.save(fname=fname)
        else:
            raise Exception("First, simulate uv-data.")

    @property
    def noise(self):
        if self._noise is None:
            return self.observed_noise
        else:
            return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise


class MFSimulation(object):
    """
    Class that handles simulations of multifrequency VLBI observations.

    :param observed_uv:
        Iterable of ``UVData`` instances with simultaneous multifrequency
        uv-data of the same source at the same epoch.
    :param model_generator:
        Instance of ``ModelGenerator`` class.

    """
    def __init__(self, observed_uv, model_generator):
        self.observed_uv = sorted(observed_uv, key=lambda x: x.frequency)
        self.simulations = [Simulation(uv) for uv in self.observed_uv]
        self.model_generator = model_generator
        self.add_true_models()

    @property
    def stokes(self):
        return self.model_generator.stokes

    def add_true_models(self):
        for simulation in self.simulations:
            frequency = simulation.frequency
            models = self.model_generator.create_models_for_frequency(frequency)
            simulation.add_true_models(models.values())

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def save_fits(self, fnames_dict):
        for simulation in self.simulations:
            frequency = simulation.frequency
            simulation.save_fits(fnames_dict[frequency])

    @property
    def freqs(self):
        return [simulation.frequency for simulation in self.simulations]


# TODO: This function only simulates. To analyzes simulations use other funcs
# TODO: Use ``image_ops.pol_map`` function to create mask for ROTM calculation
# TODO: Add function to create realistic FPOL & Q/U distribution
# TODO: Make ``mapsize_dict`` the required argument
# TODO: Implement possibility of using several slice to analyze ROTM gradients
def simulate(source, epoch, bands, n_sample=3, n_rms=5., max_jet_flux=0.01,
             qu_fraction=0.1, model_freq=20. * 10 ** 9, rotm_clim_sym=None,
             spix_clim_sym=None, rotm_grad_value=40., rotm_value_0=200.,
             path_to_script=None, base_dir=None, mapsize_common=None,
             mapsize_dict=None, rotm_slice=((216, 276), (296, 276)), n_beam=0,
             download_mojave=False, conf_band_alpha=0.95,
             rotm_clim_model=None, spix_clim_model=None, n_freq_model=-1):
    """
    :param source:
        Source name [B1950].
    :param epoch:
        Epoch [YYYY_MM_DD].
    :param bands:
        Iterable of bands letters sorted in increasing in frequency.
    :param n_sample:
        Number of bootstrap replications to create.
    :param n_rms:
        Number of PPOL image rms to use in constructing ROTM mask.
    :param max_jet_flux:
        Maximum flux in jet model image.
    :param qu_fraction:
        Fraction of Q & U flux in model image.
    :param model_freq:
        Frequency [Hz] of model image.
    :param rotm_clim_sym: (optional)
        Range of ROTM values to show in simulated image. If ``None`` then show
        all range. (default: ``None``)
    :param rotm_clim_model: (optional)
        Range of ROTM values to show in model image. If ``None`` then show
        all range. (default: ``None``)
    :param rotm_grad_value: (optional)
        Value of model ROTM gradient [rad/m/m/pixel]. Pixel is that of model
        image. (default: ``40.0``)
    :param rotm_value_0: (optional)
        Value of model ROTM at central along-jet slice. (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param base_dir: (optional)
        Directory that will contain ``source`` catalog with visibility &
        simulations data. If ``None`` then use CWD. (default: ``None``)
    :param mapsize_common: (optional)
        Common map parameters (map size [] & pixel size [mas]) that will be used
        for creating maps of ROTM. If ``None`` the use some unimplemented logic
        to choose. (default: ``None``)
    :param mapsize_dict:
        Dictionary of `band: map parameters` values that will be used for
        cleaning with native resolution.
    :param rotm_slice:
        Iterable of coordinates for slice that will be analyzed for presense of
        ROTM gradients.
    :param n_beam: (optional)
        Number of band to use for common beam size. E.g. ``n_beam = 0`` and
        ``bands = ['x', 'y', 'j', 'u']`` then beam of ``x`` band will be used.
    :param download_mojave: (optional)
        Download original self-calibrated uv-data from MOJAVE web DB? (default:
        ``False``)
    :param conf_band_alpha: (optional)
        Significance of the simultaneous confidence bands constructed for ROTM
        slices. (default: ``0.95``)
    :param n_freq_model: (optional)
        Sequence number of frequency in sorted frequency list to display stokes
        `I` contours overlayed with beam-convolved ROTM or SPIX images. Default
        corresponds to highest frequency. (default: ``-1``)

    """

    if base_dir is None:
        base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, source)
    stokes = ['I', 'Q', 'U']
    # Download uv-data from MOJAVE web DB optionally
    if download_mojave:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        from mojave import download_mojave_uv_fits
        download_mojave_uv_fits(source, epochs=[epoch], bands=bands,
                                download_dir=data_dir)

    # Clean in original resolution (image size, beam)
    from mojave import mojave_uv_fits_fname
    for band in (bands[0], bands[-1]):
        uv_fits_fname = mojave_uv_fits_fname(source, band, epoch)
        print "Cleaning {} with native resolution".format(uv_fits_fname)
        for stoke in stokes:
            if stoke == 'I':
                print "stokes {}".format(stoke)
                cc_fits_fname = "{}_{}_{}_{}_naitive_cc.fits".format(source,
                                                                     epoch,
                                                                     band,
                                                                     stoke)
                clean_difmap(uv_fits_fname, cc_fits_fname, stoke,
                             mapsize_dict[band], path=data_dir,
                             path_to_script=path_to_script, outpath=data_dir)

    # Choose common image parameters for ROTM calculations
    cc_fits_fname_high = "{}_{}_{}_{}_naitive_cc.fits".format(source, epoch,
                                                              bands[-1], 'I')
    cc_fits_fname_high = os.path.join(data_dir, cc_fits_fname_high)
    cc_fits_fname_beam = "{}_{}_{}_{}_naitive_cc.fits".format(source, epoch,
                                                              bands[n_beam],
                                                              'I')
    cc_fits_fname_beam = os.path.join(data_dir, cc_fits_fname_beam)

    # Get common beam from lowest frequency
    map_info = get_fits_image_info(cc_fits_fname_beam)
    beam_common = (map_info['bmaj'] / mas_to_rad, map_info['bmin'] / mas_to_rad,
                   map_info['bpa'] / degree_to_rad)
    print "Common beam [mas, mas, deg]: ", beam_common
    # Common beam in pixels
    pix_size_common = abs(map_info['pixsize'][0])
    beam_common_pxl = (beam_common[0] * mas_to_rad / pix_size_common,
                       beam_common[1] * mas_to_rad / pix_size_common,
                       beam_common[2])
    print"Common beam [pix, pix, deg] : ", beam_common_pxl

    # Choose image on highest frequency for jet model construction
    map_info = get_fits_image_info(cc_fits_fname_high)
    imsize_high = (map_info['imsize'][0], abs(map_info['pixsize'][0]) /
                   mas_to_rad)
    image_high = create_clean_image_from_fits_file(cc_fits_fname_high)
    x = image_high.x
    y = image_high.y

    observed_uv_fits = glob.glob(os.path.join(data_dir,
                                              '{}*.uvf'.format(source)))

    # Don't count simulated data as observed i any
    for path in observed_uv_fits:
        if 'sym' in os.path.split(path)[-1]:
            observed_uv_fits.remove(path)

    observed_uv = [UVData(fits_file) for fits_file in observed_uv_fits]
    # Create jet model, ROTM & SPIX images
    jet_image = create_jet_model_image(30, 60, 10, max_jet_flux,
                                       (imsize_high[0], imsize_high[0]),
                                       (imsize_high[0] / 2, imsize_high[0] / 2),
                                       gauss_peak=0.001, dist_from_core=20,
                                       cut=0.0002)
    rotm_image = rotm((imsize_high[0], imsize_high[0]),
                      (imsize_high[0] / 2, imsize_high[0] / 2),
                      grad_value=rotm_grad_value, rm_value_0=rotm_value_0)
    alpha_image = alpha((imsize_high[0], imsize_high[0]),
                        (imsize_high[0] / 2, imsize_high[0] / 2), 0.)

    # # Optionally plot models
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.matshow(jet_image)
    # fig1.savefig(os.path.join(data_dir, 'jet_model_image.png'),
    #              bbox_inches='tight', dpi=200)
    # plt.close()

    stokes_models = {'I': jet_image, 'Q': qu_fraction * jet_image,
                     'U': qu_fraction * jet_image}
    mod_generator = ModelGenerator(stokes_models, x, y, rotm=rotm_image,
                                   alpha=alpha_image, freq=model_freq)
    rm_simulation = MFSimulation(observed_uv, mod_generator)

    # Mapping from frequencies to FITS file names
    fnames_dict = dict()
    os.chdir(data_dir)
    for freq in rm_simulation.freqs:
        fnames_dict.update({freq: str(freq) + '_' + 'sim.uvf'})
    rm_simulation.simulate()
    rm_simulation.save_fits(fnames_dict)

    # CLEAN uv-fits with simulated data
    for freq in rm_simulation.freqs:
        uv_fits_fname = fnames_dict[freq]
        print "Cleaning {}".format(uv_fits_fname)
        for stokes in rm_simulation.stokes:
            # # Clean stokes I only for highest frequency. Use it for rms calc.
            # if stokes == 'I' and freq != rm_simulation.freqs[-1]:
            #     continue
            print "Stokes {}".format(stokes)
            cc_fits_fname = str(freq) + '_' + stokes + '.fits'
            clean_difmap(uv_fits_fname, cc_fits_fname, stokes,
                         mapsize_common, path=data_dir,
                         path_to_script=path_to_script, outpath=data_dir,
                         beam_restore=beam_common)

    # Create simulated ROTM & SPIX image
    from images import Images
    sym_images = Images()
    fnames = glob.glob(os.path.join(data_dir, "*.0_*.fits"))
    sym_images.add_from_fits(fnames=fnames)

    # Create masks for ROTM and SPIX
    rotm_mask = None
    from from_fits import create_image_from_fits_file
    # looping through simulated cleaned images
    for i_fname in sorted(glob.glob(os.path.join(data_dir, '*_I.fits'))):
        i_image = create_image_from_fits_file(i_fname)
        r_rms = mapsize_common[0] / 10
        rms = i_image.rms(region=(r_rms, r_rms, r_rms, None))
        freq = i_image.freq
        ppol_image = sym_images.create_pol_images(freq=freq)[0]
        mask = ppol_image.image < n_rms * rms
        if rotm_mask is None:
            rotm_mask = mask
        else:
            rotm_mask = np.logical_or(rotm_mask, mask)

    spix_mask = None
    # looping through simulated cleaned images
    for i_fname in sorted(glob.glob(os.path.join(data_dir, '*_I.fits'))):
        i_image = create_image_from_fits_file(i_fname)
        r_rms = mapsize_common[0] / 10
        rms = i_image.rms(region=(r_rms, r_rms, r_rms, None))
        mask = i_image.image < n_rms * rms
        if spix_mask is None:
            spix_mask = mask
        else:
            spix_mask = np.logical_or(spix_mask, mask)

    # Optionally plot ROTM mask
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(rotm_mask)
    fig.savefig(os.path.join(data_dir, 'rotm_mask.png'),
                bbox_inches='tight', dpi=200)
    plt.close()
    # Optionally plot SPIX mask
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(spix_mask)
    fig.savefig(os.path.join(data_dir, 'spix_mask.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    print "Calculating simulated ROTM image"
    rotm_image_sym, s_rotm_image_sym = \
        sym_images.create_rotm_image(mask=rotm_mask)
    print "Calculating simulated SPIX image"
    spix_image_sym, s_spix_image_sym = \
        sym_images.create_spix_image(mask=rotm_mask)

    # Plotting simulated high-freq stokes I contours with RM values.
    iplot(i_image.image, rotm_image_sym.image, x=i_image.x, y=i_image.y,
          min_abs_level=3. * rms, colors_mask=rotm_mask,
          outfile='rotm_image_sym', outdir=data_dir, color_clim=rotm_clim_sym,
          blc=(210, 200), trc=(350, 320), beam=beam_common,
          colorbar_label='RM, [rad/m/m]', slice_points=((-2, -4), (-2, 4)),
          show=False, show_beam=True)
    # Plotting simulated high-freq stokes I contours with SPIX values.
    iplot(i_image.image, spix_image_sym.image, x=i_image.x, y=i_image.y,
          min_abs_level=3. * rms, colors_mask=spix_mask,
          outfile='spix_image_sym', outdir=data_dir, color_clim=spix_clim_sym,
          blc=(210, 200), trc=(350, 320), beam=beam_common,
          colorbar_label='SPIX', show=False, show_beam=True,
          slice_points=((4, 0), (-6, 0)))

    # Plotting convolved model of ROTM
    freqs = sorted([float(freq) for freq in rm_simulation.freqs])
    freq = freqs[n_freq_model]
    i_image_mod = mod_generator.get_stokes_images(freq, i_cut_frac=0.01,
                                                  beam=beam_common_pxl)['I']
    rotm_image_mod, _, _ = mod_generator.get_rotm_map(freqs,
                                                      rotm_mask=rotm_mask,
                                                      beam=beam_common_pxl)
    iplot(i_image_mod, rotm_image_mod, x=i_image.x, y=i_image.y,
          min_abs_level=3. * rms, colors_mask=rotm_mask,
          outfile='rotm_image_model_convolved', outdir=data_dir, blc=(210, 200),
          trc=(350, 320), beam=beam_common, colorbar_label='RM, [rad/m/m]',
          slice_points=((-2, -4), (-2, 4)), show=False, show_beam=True)

    # Plotting non-convolved (generating) model of ROTM
    # TODO: Should i use original model array for ROTM (not I cause frequency
    # changed)? Original array of ROTM model ``rotm_image`` could have much
    # smaller pixel - so probably not.
    i_image_mod_nc = mod_generator.get_stokes_images(freq, i_cut_frac=0.01)['I']
    mask_nc = mod_generator.create_i_mask(freq, i_cut_frac=0.01)
    rotm_image_mod_nc, _, _ = mod_generator.get_rotm_map(freqs,
                                                         rotm_mask=mask_nc)
    iplot(i_image_mod_nc, rotm_image_mod_nc, x=i_image.x, y=i_image.y,
          min_abs_level=0.005 * np.max(i_image_mod_nc), colors_mask=mask_nc,
          outfile='rotm_image_model_original', outdir=data_dir,
          color_clim=rotm_clim_model, blc=(210, 200), trc=(350, 320),
          colorbar_label='RM, [rad/m/m]', slice_points=((-2, -4), (-2, 4)),
          show=False)

    # Plotting convolved model of SPIX
    spix_image_mod, _, _ = mod_generator.get_spix_map(freqs,
                                                      spix_mask=spix_mask,
                                                      beam=beam_common_pxl)
    iplot(i_image_mod, spix_image_mod, x=i_image.x, y=i_image.y,
          min_abs_level=3. * rms, colors_mask=spix_mask,
          outfile='spix_image_model_convolved', outdir=data_dir,
          color_clim=spix_clim_sym, blc=(210, 200), trc=(350, 320),
          beam=beam_common, colorbar_label='SPIX', show=False, show_beam=True,
          slice_points=((4, 0), (-6, 0)))

    # Plot non-convolved (generating) model of SPIX
    # TODO: Should i use original model array for SPIX? (not I cause frequency
    # changed)? Original array of SPIX model ``alpha_image`` could have much
    # smaller pixel - so probably not.
    spix_image_mod_nc, _, _ = mod_generator.get_spix_map(freqs,
                                                         spix_mask=mask_nc)
    iplot(i_image_mod_nc, spix_image_mod_nc, x=i_image.x, y=i_image.y,
          min_abs_level=0.005 * np.max(i_image_mod_nc), colors_mask=mask_nc,
          outfile='spix_image_model_original', outdir=data_dir,
          color_clim=spix_clim_model, blc=(210, 200), trc=(350, 320),
          colorbar_label='SPIX', show=False, slice_points=((4, 0), (-6, 0)))

    # Plotting simulated slices with model values (convolved and non-convolved)
    # Simulated ROTM slice & model value
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Simulated values with formal errorbars
    ax.errorbar(np.arange(216, 296, 1),
                rotm_image_sym.slice((216, 276), (296, 276)),
                s_rotm_image_sym.slice((216, 276), (296, 276)), fmt='.k')
    # ax.plot(np.arange(216, 296, 1),
    #         rotm_grad_value * (np.arange(216, 296, 1) - 256.) + rotm_value_0)
    # Model convolved values
    ax.plot(np.arange(216, 296, 1),
            image_slice(rotm_image_mod, (216, 276), (296, 276)), 'g')
    # Model non-convolved values
    ax.plot(np.arange(216, 296, 1),
            image_slice(rotm_image, (216, 276), (296, 276)), 'r')
    ax.set_xlabel("pixel")
    ax.set_ylabel("ROTM, [rad/m/m]")
    fig.savefig(os.path.join(data_dir, 'rotm_slice.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    # Simulated SPIX slice & model value
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Simulated values with formal errorbars
    ax.errorbar(np.arange(216, 336, 1),
                spix_image_sym.slice((256, 216), (256, 336)),
                s_spix_image_sym.slice((256, 216), (256, 336)),
                fmt='.k')
    # Model convolved values
    ax.plot(np.arange(216, 336, 1),
            image_slice(spix_image_mod, (256, 216), (256, 336)), 'g')
    # Model non-convolved values
    ax.plot(np.arange(216, 336, 1),
            image_slice(alpha_image, (256, 216), (256, 336)), 'r')
    # ax.plot(np.arange(216, 296, 1),
    #         rotm_grad_value * (np.arange(216, 296, 1) - 256.) + rotm_value_0)
    ax.set_xlabel("pixel")
    ax.set_ylabel("SPIX")
    fig.savefig(os.path.join(data_dir, 'spix_slice.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    # Creating sample
    for i in range(n_sample):
        print "Creating sample {} of {}".format(i + 1, n_sample)
        fnames_dict_i = fnames_dict.copy()
        fnames_dict_i.update({freq: name + '_' + str(i + 1).zfill(3) for
                              freq, name in fnames_dict.items()})
        rm_simulation.simulate()
        rm_simulation.save_fits(fnames_dict_i)

    # CLEAN uv-fits with simulated sample data
    for freq in rm_simulation.freqs:
        for i in range(n_sample):
            uv_fits_fname = fnames_dict[freq] + '_' + str(i + 1).zfill(3)
            print "Cleaning {}".format(uv_fits_fname)
            for stokes in rm_simulation.stokes:
                if stokes != 'I':
                    print "Stokes {}".format(stokes)
                    cc_fits_fname = str(freq) + '_' + stokes + '_{}.fits'.format(str(i + 1).zfill(3))
                    clean_difmap(uv_fits_fname, cc_fits_fname, stokes,
                                 mapsize_common, path=data_dir,
                                 path_to_script=path_to_script,
                                 outpath=data_dir, beam_restore=beam_common)

    # Create ROTM images of simulated sample
    sym_images = Images()
    fnames = sorted(glob.glob(os.path.join(data_dir, "*.0_*_*.fits")))
    sym_images.add_from_fits(fnames)
    rotm_images_sym = sym_images.create_rotm_images(mask=rotm_mask)

    # Calculate simultaneous confidence bands
    # Bootstrap slices
    slices = list()
    for image in rotm_images_sym.images:
        slice_ = image.slice((216, 276), (296, 276))
        slices.append(slice_[~np.isnan(slice_)])

    # Find means
    obs_slice = rotm_image_sym.slice((216, 276), (296, 276))
    x = np.arange(216, 296, 1)
    x = x[~np.isnan(obs_slice)]
    obs_slice = obs_slice[~np.isnan(obs_slice)]
    # Find sigmas
    slices_ = [arr.reshape((1, len(obs_slice))) for arr in slices]
    sigmas = hdi_of_arrays(slices_).squeeze()
    means = np.mean(np.vstack(slices), axis=0)
    diff = obs_slice - means
    # Move bootstrap curves to original simulated centers
    slices_ = [slice_ + diff for slice_ in slices]
    # Find low and upper confidence band
    low, up = create_sim_conf_band(slices_, obs_slice, sigmas,
                                   alpha=conf_band_alpha)

    # Plot confidence bands and model values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, low[::-1], 'g')
    ax.plot(x, up[::-1], 'g')
    [ax.plot(x, slice_[::-1], 'r', lw=0.15) for slice_ in slices_]
    ax.plot(x, obs_slice[::-1], '.k')
    # Model convolved values
    ax.plot(np.arange(216, 296, 1),
            image_slice(rotm_image_mod, (216, 276), (296, 276)), 'g')
    # Model non-convolved values
    ax.plot(np.arange(216, 296, 1),
            image_slice(rotm_image, (216, 276), (296, 276)), 'r')
    ax.set_xlabel("pixel")
    ax.set_ylabel("ROTM, [rad/m/m]")
    fig.savefig(os.path.join(data_dir, 'rotm_slice_spread.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    # Create SPIX images of simulated sample
    spix_images_sym = sym_images.create_spix_images(mask=spix_mask)

    # Calculate simultaneous confidence bands
    # Bootstrap slices
    slices = list()
    for image in spix_images_sym.images:
        slice_ = image.slice((256, 216), (256, 336))
        slices.append(slice_[~np.isnan(slice_)])

    # Find means
    obs_slice = spix_image_sym.slice((256, 216), (256, 336))
    x = np.arange(216, 336, 1)
    x = x[~np.isnan(obs_slice)]
    obs_slice = obs_slice[~np.isnan(obs_slice)]
    # Find sigmas
    slices_ = [arr.reshape((1, len(obs_slice))) for arr in slices]
    sigmas = hdi_of_arrays(slices_).squeeze()
    means = np.mean(np.vstack(slices), axis=0)
    diff = obs_slice - means
    # Move bootstrap curves to original simulated centers
    slices_ = [slice_ + diff for slice_ in slices]
    # Find low and upper confidence band
    low, up = create_sim_conf_band(slices_, obs_slice, sigmas,
                                   alpha=conf_band_alpha)

    # Plot confidence bands and model values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, low[::-1], 'g')
    ax.plot(x, up[::-1], 'g')
    [ax.plot(x, slice_[::-1], 'r', lw=0.15) for slice_ in slices_]
    ax.plot(x, obs_slice[::-1], '.k')
    # Model convolved values
    ax.plot(np.arange(216, 336, 1),
            image_slice(spix_image_mod, (256, 216), (256, 336)), 'g')
    # Model non-convolved values
    ax.plot(np.arange(216, 336, 1),
            image_slice(alpha_image, (256, 216), (256, 336)), 'r')
    ax.set_xlabel("pixel")
    ax.set_ylabel("SPIX")
    fig.savefig(os.path.join(data_dir, 'spix_slice_spread.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == '__main__':

    ############################################################################
    # Test simulate
    from mojave import get_epochs_for_source
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    base_dir = '/home/ilya/code/vlbi_errors/examples/mojave/0d005'
    # sources = ['1514-241', '1302-102', '0754+100', '0055+300', '0804+499',
    #            '1749+701', '0454+844']
    mapsize_dict = {'x': (512, 0.1), 'y': (512, 0.1), 'j': (512, 0.1),
                    'u': (512, 0.1)}
    mapsize_common = (512, 0.1)
    source_epoch_dict = dict()
    # for source in sources:
    #     epochs = get_epochs_for_source(source, use_db='multifreq')
    #     print "Found epochs for source {}".format(source)
    #     print epochs
    #     source_epoch_dict.update({source: epochs[-1]})
    # for source in sources:
    #     print "Simulating source {}".format(source)
    #     simulate(source, source_epoch_dict[source], ['x', 'y', 'j', 'u'],
    #              n_sample=3, rotm_clim=[-200, 200],
    #              path_to_script=path_to_script, mapsize_dict=mapsize_dict,
    #              mapsize_common=mapsize_common, base_dir=base_dir,
    #              rotm_value_0=0., max_jet_flux=0.005)

    source = '1055+018'
    epoch = '2006_11_10'
    simulate(source, epoch, ['x', 'y', 'j', 'u'],
             n_sample=5, max_jet_flux=0.003, rotm_clim_sym=[-300, 300],
             rotm_clim_model=[-900, 900],
             path_to_script=path_to_script, mapsize_dict=mapsize_dict,
             mapsize_common=mapsize_common, base_dir=base_dir,
             rotm_value_0=0., rotm_grad_value=60., n_rms=3.,
             download_mojave=False, spix_clim_sym=[-1, 1])

    # ############################################################################
    # # Test for ModelGenerator
    # # Create jet model, ROTM & alpha images
    # imsize = (512, 512)
    # center = (256, 256)
    # # from `y`  band
    # pixsize = 4.848136191959676e-10
    # x, y = create_grid(imsize)
    # x -= center[0]
    # y -= center[1]
    # x *= pixsize
    # y *= pixsize
    # max_jet_flux = 0.01
    # qu_fraction = 0.1
    # rotm_grad_value = 40.
    # rotm_value_0 = 0.
    # model_freq = 20. * 10. ** 9.
    # jet_image = create_jet_model_image(30, 60, 10, max_jet_flux,
    #                                    (imsize[0], imsize[0]),
    #                                    (imsize[0] / 2, imsize[0] / 2),
    #                                    gauss_peak=0.001, dist_from_core=20,
    #                                    cut=0.0002)
    # rotm_image = rotm((imsize[0], imsize[0]),
    #                   (imsize[0] / 2, imsize[0] / 2),
    #                   grad_value=rotm_grad_value, rm_value_0=rotm_value_0)
    # alpha_image = alpha((imsize[0], imsize[0]),
    #                     (imsize[0] / 2, imsize[0] / 2), 0.)
    # stokes_models = {'I': jet_image, 'Q': qu_fraction * jet_image,
    #                  'U': qu_fraction * jet_image}
    # mod_generator = ModelGenerator(stokes_models, x, y, rotm=rotm_image,
    #                                alpha=alpha_image, freq=model_freq)
    # images = mod_generator.get_stokes_images(frequency=8.*10**9.,
    #                                          i_cut_frac=0.01)
    # pol_images = mod_generator.get_pol_maps(frequency=8.*10**9.,
    #                                         i_cut_frac=0.01)
    # frequencies = [5. * 10 ** 9, 8. * 10 ** 9, 12. * 10 ** 9, 15. * 10 ** 9]
    # mask = mod_generator.create_i_mask(frequencies[-1], i_cut_frac=0.01)
    # rm_map, s_rm_map, chsq_map = mod_generator.get_rotm_map(frequencies,
    #                                                         rotm_mask=mask)
    # sp_map, s_sp_map, chsq_map = mod_generator.get_spix_map(frequencies,
    #                                                         spix_mask=mask)

