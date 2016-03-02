import os
import copy
import numpy as np
from utils import get_fits_image_info
from image import BasicImage, Image
from images import Images
from utils import mask_region, mas_to_rad, find_card_from_header
from model import Model
from uv_data import UVData
from components import DeltaComponent, ImageComponent


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
    return max_flux - (max_flux / length) * x -\
        (max_flux / (width / 2) ** 2.) * y ** 2.


def alpha(x, y, *args, **kwargs):
    return None


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


# Decorator that returns only constant fraction of decorated function values
def fraction(frac):
    def wrapper(func, *args, **kwargs):
        return frac * func(*args, **kwargs)
    return wrapper


def create_jet_model_image(width, j_length, cj_length, max_flux, imsize, center):
    from utils import create_grid
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


class ModelGenerator(object):
    """
    Class that generates models that can be represented by images (2D arrays
    with clean component in each element).

    :param stokes_models:
        Model of Stokes parameters distribution. Dictionary with keys - Stokes
        parameters and values - image of Stokes distribution used as model (that
        is 2D numpy arrays of fluxes).
    :param freq: (optional)
        Frequency at which models [GHz]. If ``None`` then infinity.
    :param alpha_func: (optional)
        Callable of spectral index distribution. The same signature as for
        ``stokes_func``. If ``None`` then use uniform distribution with zero
        value.
    :param rotm_func: (optional)
        Callable of rotation measure distribution. The same signature as for
        ``stokes_func``. If ``None`` then use uniform distribution with zero
        value.
    """
    def __init__(self, stokes_models, x, y, freq=None, alpha_func=None,
                 rotm_func=None):
        self.stokes_models = stokes_models
        self._x = x
        self._y = y
        if freq is None:
            self.freq = +np.inf
        else:
            self.freq = freq
        if alpha_func is None:
            self.alpha_func = lambda x, y: 0.0
        if rotm_func is None:
            self.rotm_func = lambda x, y: 0.0

    def create_model(self, freq, region=None, *args, **kwargs):
        """
        Create instance of ``Model`` class for given frequency.
        :param freq:
        :param region:
        :param args:
        :param kwargs:
        :return:
        """
        models = list()
        stokes_models = self.update_for_freq(freq)
        for stokes, image in stokes_models:
            model = Model(stokes=stokes)
            image_component = ImageComponent(image, self._x, self._y)
            model.add_components(image_component)
            models.append(model)
        return models

    def update_for_freq(self, freq):
        """
        Update instance of ``Model`` class in ``self`` using current instance's
        ``alpha_func`` and ``rotm_func`` attributes.

        :param freq:
            Frequency to update to [GHz].
        :return:
            Instance of ``Model`` class.
        """
        if 'Q' in self.stokes and 'U' not in self.stokes:
            raise Exception('Need to update both Q & U simultaneously!')
        if 'U' in self.stokes and 'Q' not in self.stokes:
            raise Exception('Need to update both Q & U simultaneously!')

        model_i = [model for model in self.models if model.stokes == 'I']
        models_qu = [model for model in models if model.stokes == 'Q'
                     or model.stokes == 'U']
        model_i = self.update_i_freq(model_i, freq)
        models_qu = self.update_qu_freq(models_qu, freq)
        return model_i.extend(models_qu)

    def update_i_freq(self, model, freq):
        pass

    def update_qu_freq(self, models, freq):
        pass

    @property
    def stokes(self):
        return self.stokes_models.keys()


# TODO: Generating model for simulation - task of other class/function.
class Simulation(object):
    """
    Basic Abstract class that handles simulations of VLBI observations.

    :param observed_uv:
        Instance of ``UVData`` class with observed uv-data.

    """
    def __init__(self, observed_uv):
        self.observed_uv = observed_uv
        self.simulated_uv = None
        self.models = dict()
        self._observed_noise = None
        self._noise = None

    @property
    def frequency(self):
        return self.observed_uv.frequency

    def add_true_model(self, model):
        self.models[model.stokes].append(model)

    def simulate(self):
        self.simulated_uv = copy.deepcopy(self.observed_uv)
        self.simulated_uv.substitute(self.models.values())
        self.simulated_uv.noise_add(self.noise)

    def save_fits(self, fname):
        if self.simulated_uv is not None:
            self.simulated_uv.save(fname=fname)
        else:
            raise Exception("First, simulate uv-data.")

    @property
    def noise(self):
        if self._noise is None:
            return self._observed_noise
        else:
            return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise


# TODO: Model of flux should be on infinite frequency or at highest?
class MFSimulation(object):
    """
    Class that handles simulations of multifrequency VLBI observations.

    :param observed_uv:
        Iterable of ``UVData`` instances with simultaneous multifrequency
        uv-data of the same source.
    :param model_generator:
        Instance of ``ModelGenerator`` class.

    """
    def __init__(self, observed_uv, model_generator):
        self.observed_uv = sorted(observed_uv, key=lambda x: x.frequency)
        self.simulations = [Simulation(uv) for uv in self.observed_uv]
        self.model_generator = model_generator
        self.add_true_model()

    def add_true_model(self):
        for simulation in self.simulations:
            frequency = simulation.frequency
            simulation.add_true_model(self.model_generator.update_for_freq(frequency))

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def save_fits(self, fnames_dict):
        for simulation in self.simulations:
            frequency = simulation.frequency
            simulation.save_fits(fnames_dict[frequency])


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
    # Iterable of ``UVData`` instances with simultaneous multifrequency uv-data
    # of the same source
    observed_uv = list()
    # Mapping from frequencies to FITS file names
    fnames_dict = dict()
    stokes_func = {'I': flux, 'Q': fraction(0.1)(flux),
                   'U': fraction(0.1)(flux)}
    mod_generator = ModelGenerator(stokes_func, rotm_func=rm, alpha_func=alpha)
    rm_simulation = MFSimulation(observed_uv, mod_generator)
    for i in xrange(100):
        fnames_dict_i = fnames_dict.copy()
        fnames_dict.update({key: value + '_' + str(i + 1) for key, value in
                            fnames_dict.items()})
        rm_simulation.simulate()
        rm_simulation.save_fits(fnames_dict_i)
