import os
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import get_fits_image_info, degree_to_rad
from image import BasicImage, Image
from images import Images
from utils import mask_region, mas_to_rad, find_card_from_header, create_grid
from from_fits import create_clean_image_from_fits_file
from model import Model
from uv_data import UVData
from components import DeltaComponent, ImageComponent
from image_ops import pang_map, pol_map
from spydiff import clean_difmap
from image import plot as iplot


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
                           center):
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
    return image


# TODO: First must be fractional polarization and PANG. From fractional
# polarization and stokes I at given frequency one can calculate PPOL.
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

    def _get_chi(self):
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
            List of ``Model`` instances.
        """
        models = list()
        stokes_models = stokes_models or self.stokes_models
        for stokes, image in stokes_models.items():
            model = Model(stokes=stokes)
            image_component = ImageComponent(image, self._x, self._y)
            model.add_components(image_component)
            models.append(model)
        return models

    def create_models_for_frequency(self, freq):
        """
        Create instance of ``Model`` class for given frequency.

        :param freq:
            Frequency at which evaluate and return models [Hz].

        :return:
            List of ``Model`` isntances.
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
        # if 'I' in self.stokes:
        #     i_image = self._move_i_to_freq(self.stokes_models['I'], freq)
        #     stokes_models.update({'I': i_image})
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
        return self.observed_uv.frequency

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
            simulation.add_true_models(models)

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def save_fits(self, fnames_dict):
        for simulation in self.simulations:
            frequency = simulation.frequency
            simulation.save_fits(fnames_dict[frequency])

    @property
    def freqs(self):
        return [uvdata.frequency for uvdata in self.observed_uv]


# TODO: Define cc_flux from original image and beam (for I & Q, U independ.)
# FIXME: ``Image`` instances creation - changed API
# TODO: Derive abstractions
# FIXME: With working class that implements model = image we can skip using
# ``Image`` subclasses keeping coordinates information
def simulate_grad(low_freq_map, high_freq_map, uvdata_files, cc_flux,
                  outpath, grad_value, width, length, k, noise_factor=1.,
                  rm_value_0=0.0):
    """
    Function that simulates ROTM gradients in uv-data.

    :param low_freq_map:
        Path to FITS-file with clean map for lowest frequency.
    :param high_freq_map:
        Path to FITS-file with clean map for highest frequency.
    :param uvdata_files:
        Iterable of paths to FITS-files with uv-data.
    :param cc_flux:
        Flux density of CC-components that will model the polarization
        [Jy/pixel]. That CCs will be used in Q & U clean models. Being convolved
        with beam it increases the maximum flux density by factor
        ``pi * beam_width ** 2`` and converts to units [Jy/beam].
    :param outpath:
        Path where to save all resulting files.
    :param grad_value:
        Value of ROTM gradient [rad/m/m/beam], where ``beam`` - major axis of
        low-frequency beam.
    :param width:
        Width of model jet in units of the lowest frequency beam's major axis.
    :param length:
        Length of model jet in units of the lowest frequency beam's major axis.
    :param k:
        How many model image pixels should be in one highest frequency image
        pixel?
    :param noise_factor:
        This enhanced noise that is added to model uv-data from those that in
        original uv-data.
    :param rm_value_0: (optional)
        Value of ROTM at image center [rad/m/m]. (default: ``0.0``)

    :return:
        Creates FITS-files with uv-data where uv-data is substituted by data
        with ROTM gradient.

    """

    h_freq_image_info = get_fits_image_info(high_freq_map)
    l_freq_image_info = get_fits_image_info(low_freq_map)
    imsize_h = h_freq_image_info['imsize']
    pixref_h = h_freq_image_info['pixref']
    pixrefval_h = h_freq_image_info['pixrefval']
    bmaj_h = h_freq_image_info['bmaj']
    bmin_h = h_freq_image_info['bmin']
    bpa_h = h_freq_image_info['bpa']
    pixsize_h = h_freq_image_info['pixsize']
    stokes_h = h_freq_image_info['stokes']
    freq_h = h_freq_image_info['freq']

    imsize_l = l_freq_image_info['imsize']
    pixref_l = l_freq_image_info['pixref']
    pixrefval_l = l_freq_image_info['pixrefval']
    bmaj_l = l_freq_image_info['bmaj']
    bmin_l = l_freq_image_info['bmin']
    bpa_l = l_freq_image_info['bpa']
    pixsize_l = l_freq_image_info['pixsize']
    stokes_l = l_freq_image_info['stokes']
    freq_l = l_freq_image_info['freq']

    # new pixsize [rad]
    pixsize = (abs(pixsize_h[0]) / k, abs(pixsize_h[1]) / k)
    # new imsize
    x1 = imsize_l[0] * abs(pixsize_l[0]) / abs(pixsize[0])
    x2 = imsize_l[1] * abs(pixsize_l[1]) / abs(pixsize[1])
    imsize = (int(x1 - x1 % 2),
              int(x2 - x2 % 2))
    # new pixref
    pixref = (imsize[0]/2, imsize[1]/2)
    # FIXME: Should i use ellipse beam for comparing model with results?
    # Beam width (of low frequency map) in new pixels
    beam_width = bmaj_l / abs(pixsize[0])

    # Jet's parameters in new pixels
    jet_width = width * bmaj_l / abs(pixsize[0])
    jet_length = length * bmaj_l / abs(pixsize[0])

    # Construct image with new parameters
    image = BasicImage(imsize=imsize, pixsize=pixsize, pixref=pixref)

    # Construct region with emission
    # TODO: Construct cone region
    jet_region = mask_region(image._image, region=(pixref[0] -
                                                   int(jet_width // 2),
                                                   pixref[1],
                                                   pixref[0] +
                                                   int(jet_width // 2),
                                                   pixref[1] + jet_length))
    jet_region = np.ma.array(image._image, mask=~jet_region.mask)

    # TODO: add decline of contrjet
    def flux(x, y, max_flux, length, width):
        """
        Function that defines model flux distribution that declines linearly
        from phase center (0, 0) along jet and parabolically across.

        :param x:
            x-coordinates on image [pixels].
        :param y:
            y-coordinates on image [pixels].
        :param max_flux:
            Flux density maximum [Jy/pixels].
        :param length:
            Length of jet [pixels].
        :param width:
            Width of jet [pixels].
        """
        return max_flux - (max_flux / length) * x - \
               (max_flux / (width / 2) ** 2.) * y ** 2.

    def rm(x, y, grad_value, rm_value_0=0.0):
        """
        Function that defines model of ROTM gradient distribution.

        :param x:
            x-coordinates on image [pixels].
        :param y:
            y-coordinates on image [pixels].
        :param grad_value:
            Value of gradient [rad/m/m/pixel].
        :param rm_value_0: (optional)
            Value of ROTM at center [rad/m/m]. (default: ``0.0``)
        """
        return grad_value * x + rm_value_0

    # Create map of ROTM
    print "Creating ROTM image with gradient..."
    image_rm = Image(imsize=imsize, pixsize=pixsize, pixref=pixref)
    # Use value for gradient ``k`` times less then original because of pixel
    # size
    image_rm._image = rm(image.x/abs(pixsize[0]), image.y/abs(pixsize[1]),
                         grad_value / beam_width, rm_value_0=rm_value_0)
    image_rm._image = np.ma.array(image_rm._image, mask=jet_region.mask)

    # Create ROTM image with size as for lowest freq. and pixel size - as for
    # highest freq. map
    save_imsize = (int(imsize_l[0] * pixsize_l[0] / pixsize_h[0]),
                   int(imsize_l[1] * pixsize_l[1] / pixsize_h[1]))
    save_pixref = (int(save_imsize[0]/2), int(save_imsize[0]/2))
    save_pixsize = pixsize_h
    save_rm = Image(imsize=save_imsize, pixsize=save_pixsize,
                    pixref=save_pixref)
    save_rm._image = rm(save_rm.x/abs(save_pixsize[0]),
                        save_rm.y/abs(save_pixsize[1]),
                        grad_value/(bmaj_l/abs(save_pixsize[0])),
                        rm_value_0=rm_value_0)
    half_width_l = int(width * bmaj_l/abs(save_pixsize[0])//2)
    jet_length_l = int(length * bmaj_l/ abs(save_pixsize[0]))
    jet_region_l = mask_region(save_rm._image,
                               region=(save_pixref[0] - half_width_l,
                                       save_pixref[1],
                                       save_pixref[0] + half_width_l,
                                       save_pixref[1] + jet_length_l))
    save_rm._image = np.ma.array(save_rm._image, mask=~jet_region_l.mask)
    print "Saving image of ROTM gradient..."
    np.savetxt(os.path.join(outpath, 'RM_grad_image.txt'), save_rm._image)

    # Create model instance and fill it with components
    model_i = Model(stokes='I')
    model_q = Model(stokes='Q')
    model_u = Model(stokes='U')

    max_flux = k * cc_flux / (np.pi * beam_width ** 2)
    # Use 10x total intensity
    comps_i = [DeltaComponent(flux(image.y[x, y] / abs(pixsize[0]),
                                   image.x[x, y] / abs(pixsize[0]),
                                   10. * max_flux,
                                   jet_length, jet_width),
                              image.x[x, y] / mas_to_rad,
                              image.y[x, y] / mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    comps_q = [DeltaComponent(flux(image.y[x, y] / abs(pixsize[0]),
                                   image.x[x, y] / abs(pixsize[0]),
                                   max_flux, jet_length, jet_width),
                              image.x[x, y] / mas_to_rad,
                              image.y[x, y] / mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    # FIXME: just ``comps_u = comps_q``
    comps_u = [DeltaComponent(flux(image.y[x, y] / abs(pixsize[0]),
                                   image.x[x, y] / abs(pixsize[0]),
                                   max_flux, jet_length, jet_width),
                              image.x[x, y] / mas_to_rad,
                              image.y[x, y] / mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]

    # FIXME: why for Q & U?
    # Keep only positive components
    comps_i = [comp for comp in comps_i if comp.p[0] > 0]
    # comps_q = [comp for comp in comps_q if comp.p[0] > 0]
    # comps_u = [comp for comp in comps_u if comp.p[0] > 0]

    print "Adding components to I,Q & U models..."
    model_i.add_components(*comps_i)
    model_q.add_components(*comps_q)
    model_u.add_components(*comps_u)

    image_i = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='I')
    image_q = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='Q')
    image_u = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='U')
    image_i.add_model(model_i)
    image_q.add_model(model_q)
    image_u.add_model(model_u)

    print "Creating PPOL image for constructing Q & U images on supplied" \
          " frequencies..."
    images = Images()
    images.add_images([image_q, image_u])
    ppol_image = images.create_pol_images(convolved=False)[0]

    # Equal Q & U results in chi_0 = pi / 4
    chi_0 = np.pi / 4
    # chi_0 = 0.

    # Loop over specified uv-data, substitute real data with fake and save to
    # specified location
    print "Now substituting ROTM gradient in real data and saving out..."
    for uvfile in uvdata_files:
        uvdata = UVData(uvfile)
        freq_card = find_card_from_header(uvdata._io.hdu.header,
                                          value='FREQ')[0]
        # Frequency in Hz
        # FIXME: Create property ``freq`` for ``UVData`` class
        freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        # Rotate PANG by multiplying polarized intensity on cos/sin
        lambda_sq = (3. * 10 ** 8 / freq) ** 2
        print "Creating Faraday Rotated arrays of Q & U for frequency {}" \
              " Hz".format(freq)
        # The same flux for all frequencies
        # image_i = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
        #                 stokes='I', freq=freq)
        # Stokes ``I`` image remains the same - only frequency changes
        image_i.stokes = 'I'
        image_i.freq = freq
        q_array = ppol_image._image * np.cos(2. * (chi_0 + image_rm._image *
                                                   lambda_sq))
        u_array = ppol_image._image * np.sin(2. * (chi_0 + image_rm._image *
                                                   lambda_sq))
        image_q = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
                        stokes='Q', freq=freq)
        image_q._image = q_array
        image_u = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
                        stokes='U', freq=freq)
        image_u._image = u_array

        model_i = Model(stokes='I')
        model_q = Model(stokes='Q')
        model_u = Model(stokes='U')
        print "Creating components of I,Q & U for frequency {} Hz".format(freq)
        comps_i = [DeltaComponent(image_i._image[x, y],
                                  image_i.x[x, y] / mas_to_rad,
                                  image_i.y[x, y] / mas_to_rad) for (x, y), value
                   in np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        comps_q = [DeltaComponent(image_q._image[x, y],
                                  image_q.x[x, y] / mas_to_rad,
                                  image_q.y[x, y] / mas_to_rad) for (x, y), value
                   in np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        comps_u = [DeltaComponent(image_u._image[x, y],
                                  image_u.x[x, y] / mas_to_rad,
                                  image_u.y[x, y] / mas_to_rad) for (x, y), value
                   in np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        print "Adding components of I, Q & U for frequency {} Hz" \
              " models".format(freq)
        model_i.add_components(*comps_i)
        model_q.add_components(*comps_q)
        model_u.add_components(*comps_u)

        # Substitute I, Q & U models to uv-data and add noise
        print "Substituting I, Q & U models in uv-data, adding noise, saving" \
              " frequency {} Hz".format(freq)
        noise = uvdata.noise(average_freq=True)
        for baseline, std in noise.items():
            noise[baseline] = noise_factor * std
        uvdata.substitute([model_i, model_q, model_u])
        uvdata.noise_add(noise)
        # Save uv-data to file
        uv_save_fname = os.path.join(outpath,
                                     'simul_uv_{}_Hz.fits'.format(freq))
        if os.path.exists(uv_save_fname):
            print "Deleting existing file: {}".format(uv_save_fname)
            os.remove(uv_save_fname)
        uvdata.save(uvdata.data, uv_save_fname)


if __name__ == '__main__':
    # Use case
    # Number of replications
    n_sample = 3
    # Number of PPOL image rms to use in constructing ROTM mask
    n_rms = 5.
    # Maximum flux in jet model image
    max_jet_flux = 0.1
    # Fraction of Q & U flux in model image
    qu_fraction = 0.1
    # Frequency [Hz] of model image
    model_freq = 1.5 * 10 ** 9
    # Range of ROTM values to show in resulting images
    rotm_clim = [-50, 50]
    # Value of model ROTM gradient [rad/m/m/pixel]. Pixel is that of high
    # frequency data
    rotm_grad_value = 5.
    # Value of model ROTM at central along-jet slice
    rotm_value_0 = 0.

    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    from mojave import (download_mojave_uv_fits, mojave_uv_fits_fname)
    source = '1055+018'
    base_dir = '/home/ilya/code/vlbi_errors/examples/mojave'
    data_dir = os.path.join(base_dir, source)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    epoch = '2006_11_10'
    bands = ['x', 'y', 'j', 'u']
    mapsize_dict = {'x': (512, 0.1), 'y': (512, 0.1), 'j': (512, 0.1),
                    'u': (512, 0.1)}
    mapsize_common = (512, 0.1)
    stokes = ['I', 'Q', 'U']
    download_mojave_uv_fits(source, epochs=[epoch], bands=bands,
                            download_dir=data_dir)

    # Clean in original resolution (image size, beam)
    for band in bands:
        uv_fits_fname = mojave_uv_fits_fname(source, band, epoch)
        print "Cleaning {} with native resolution".format(uv_fits_fname)
        for stoke in stokes:
            print "stokes {}".format(stoke)
            cc_fits_fname = "{}_{}_{}_{}_naitive_cc.fits".format(source, epoch,
                                                                 band, stoke)
            clean_difmap(uv_fits_fname, cc_fits_fname, stoke,
                         mapsize_dict[band], path=data_dir,
                         path_to_script=path_to_script, outpath=data_dir)

    # Choose common image parameters for ROTM calculations
    cc_fits_fname_high = "{}_{}_{}_{}_naitive_cc.fits".format(source, epoch,
                                                              bands[-1], 'I')
    cc_fits_fname_high = os.path.join(data_dir, cc_fits_fname_high)
    cc_fits_fname_low = "{}_{}_{}_{}_naitive_cc.fits".format(source, epoch,
                                                             bands[0], 'I')
    cc_fits_fname_low = os.path.join(data_dir, cc_fits_fname_low)

    # Get common beam from lowest frequency
    map_info = get_fits_image_info(cc_fits_fname_low)
    beam_common = (map_info['bmaj'] / mas_to_rad, map_info['bmin'] / mas_to_rad,
                   map_info['bpa'] / degree_to_rad)
    print "Common beam: ", beam_common

    # Choose image on highest frequency for jet model construction
    map_info = get_fits_image_info(cc_fits_fname_high)
    imsize_high = (map_info['imsize'][0], abs(map_info['pixsize'][0]) /
                   mas_to_rad)
    image_high = create_clean_image_from_fits_file(cc_fits_fname_high)
    x = image_high.x
    y = image_high.y

    observed_uv_fits = glob.glob(os.path.join(data_dir, '*.uvf'))
    observed_uv = [UVData(fits_file) for fits_file in observed_uv_fits]
    # Create jet model, ROTM & alpha images
    jet_image = create_jet_model_image(30, 60, 10, max_jet_flux,
                                       (imsize_high[0], imsize_high[0]),
                                       (imsize_high[0] / 2, imsize_high[0] / 2))
    rotm_image = rotm((imsize_high[0], imsize_high[0]),
                      (imsize_high[0] / 2, imsize_high[0] / 2),
                      grad_value=rotm_grad_value, rm_value_0=rotm_value_0)
    alpha_image = alpha((imsize_high[0], imsize_high[0]),
                        (imsize_high[0] / 2, imsize_high[0] / 2), 0.)

    # Optionally plot models
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.matshow(jet_image)
    fig1.savefig(os.path.join(data_dir, 'jet_model_image.png'),
                 bbox_inches='tight', dpi=200)
    plt.close()

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
            print "Stokes {}".format(stokes)
            cc_fits_fname = str(freq) + '_' + stokes + '.fits'
            clean_difmap(uv_fits_fname, cc_fits_fname, stokes, mapsize_common,
                         path=data_dir, path_to_script=path_to_script,
                         outpath=data_dir, beam_restore=beam_common)
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
                print "Stokes {}".format(stokes)
                cc_fits_fname = str(freq) + '_' + stokes + '_{}.fits'.format(str(i + 1).zfill(3))
                clean_difmap(uv_fits_fname, cc_fits_fname, stokes,
                             mapsize_common, path=data_dir,
                             path_to_script=path_to_script, outpath=data_dir,
                             beam_restore=beam_common)

    # Create ROTM image
    from images import Images
    sym_images = Images()
    fnames = glob.glob(os.path.join(data_dir, "*.0_*.fits"))
    sym_images.add_from_fits(fnames=fnames)
    from from_fits import create_image_from_fits_file
    # i_fname = os.path.join(data_dir, '1354458750.0_I.fits')
    freq_highest = sorted(sym_images.freqs, reverse=True)[0]
    i_fname = sorted(glob.glob(os.path.join(data_dir, '*_I.fits')))[0]
    i_image = create_image_from_fits_file(i_fname)
    r_rms = mapsize_common[0] / 10
    rms = i_image.rms(region=(r_rms, r_rms, r_rms, None))
    print "RMS : ", rms
    ppol_image = sym_images.create_pol_images(freq=freq_highest)[0]
    rotm_mask = ppol_image.image < n_rms * rms

    # Optionally plot ROTM mask
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(rotm_mask)
    fig.savefig(os.path.join(data_dir, 'rotm_mask.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    print "Calculating ROTM image"
    rotm_image_sym, s_rotm_image_sym =\
        sym_images.create_rotm_image(mask=rotm_mask)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ri = ax.matshow(rotm_image_sym.image, clim=rotm_clim)
    fig.colorbar(ri)
    fig.savefig(os.path.join(data_dir, 'rotm_image_sim.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    # Plotting model of ROTM
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    mrotm_image = np.ma.array(rotm_image, mask=rotm_mask)
    ri = ax.matshow(mrotm_image, clim=rotm_clim)
    fig.colorbar(ri)
    fig.savefig(os.path.join(data_dir, 'rotm_image_model.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    # Plot model of SPIX
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    malpha_image = np.ma.array(alpha_image, mask=rotm_mask)
    ri = ax.matshow(malpha_image)
    fig.colorbar(ri)
    fig.savefig(os.path.join(data_dir, 'alpha_image_model.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(np.arange(240, 270, 1),
                rotm_image_sym.slice((240, 260), (270, 260)),
                s_rotm_image_sym.slice((240, 260), (270, 260)), fmt='.k')
    ax.plot(np.arange(240, 270, 1), 5. * (np.arange(240, 270, 1) - 256.))
    fig.savefig(os.path.join(data_dir, 'rotm_slice.png'),
                bbox_inches='tight', dpi=200)
    plt.close()

    # Create ROTM images of simulated sample
    sym_images = Images()
    fnames = sorted(glob.glob(os.path.join(data_dir, "*.0_*_*.fits")))
    sym_images.add_from_fits(fnames)
    rotm_images_sym = sym_images.create_rotm_images(mask=rotm_mask)
    # Plot spread of sample values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(240, 270, 1), 5. * (np.arange(240, 270, 1) - 256.))
    for i in range(n_sample):
        print "plotting {}th slice of {}".format(i + 1, n_sample)
        jitter = np.random.normal(0, 0.03)
        ax.plot(np.arange(240, 270, 1) + jitter,
                rotm_images_sym.images[i].slice((240, 260), (270, 260)),
                '.k')
    fig.savefig(os.path.join(data_dir, 'rotm_slice_spread.png'),
                bbox_inches='tight', dpi=200)
    plt.close()
