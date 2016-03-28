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
        # if 'FPOL' in self.stokes:
        #     ppol_image = i_image * self.stokes_models['FPOL']
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


def simulate(source, epoch, bands, n_sample=3, n_rms=5., max_jet_flux=0.01,
             qu_fraction=0.1, model_freq=20. * 10 ** 9, rotm_clim=None,
             rotm_grad_value=40., rotm_value_0=200., path_to_script=None,
             base_dir=None, mapsize_common=None, mapsize_dict=None,
             rotm_slice=((240, 260), (270, 260))):
    """
    :param source:
        Source name.
    :param epoch:
        Epoch.
    :param bands:
        Iterable of bands.
    :param n_sample:
        Number of replications to make.
    :param n_rms:
        Number of PPOL image rms to use in constructing ROTM mask
    :param max_jet_flux:
        Maximum flux in jet model image.
    :param qu_fraction:
        Fraction of Q & U flux in model image.
    :param model_freq:
        Frequency [Hz] of model image.
    :param rotm_clim:
        Range of ROTM values to show in resulting images.
    :param rotm_grad_value:
        Value of model ROTM gradient [rad/m/m/pixel]. Pixel is that of high
        frequency data.
    :param rotm_value_0:
        Value of model ROTM at central along-jet slice.
    :param path_to_script:
    :param base_dir:
    :param mapsize_common:
    :param mapsize_dict:
    :param rotm_slice:

    :return:
    """

    from mojave import (download_mojave_uv_fits, mojave_uv_fits_fname)
    data_dir = os.path.join(base_dir, source)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
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
                                                             bands[-2], 'I')
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
    rotm_image_sym, s_rotm_image_sym = \
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
    ax.plot(np.arange(240, 270, 1),
            rotm_grad_value * (np.arange(240, 270, 1) - 256.) + rotm_value_0)
    fig.savefig(os.path.join(data_dir, 'rotm_slice.png'),
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
                print "Stokes {}".format(stokes)
                cc_fits_fname = str(freq) + '_' + stokes + '_{}.fits'.format(str(i + 1).zfill(3))
                clean_difmap(uv_fits_fname, cc_fits_fname, stokes,
                             mapsize_common, path=data_dir,
                             path_to_script=path_to_script, outpath=data_dir,
                             beam_restore=beam_common)

    # Create ROTM images of simulated sample
    sym_images = Images()
    fnames = sorted(glob.glob(os.path.join(data_dir, "*.0_*_*.fits")))
    sym_images.add_from_fits(fnames)
    rotm_images_sym = sym_images.create_rotm_images(mask=rotm_mask)
    # Plot spread of sample values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(240, 270, 1),
            rotm_grad_value * (np.arange(240, 270, 1) - 256.) + rotm_value_0)
    for i in range(n_sample):
        print "plotting {}th slice of {}".format(i + 1, n_sample)
        jitter = np.random.normal(0, 0.03)
        ax.plot(np.arange(240, 270, 1) + jitter,
                rotm_images_sym.images[i].slice((240, 260), (270, 260)),
                '.k')
    fig.savefig(os.path.join(data_dir, 'rotm_slice_spread.png'),
                bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == '__main__':

    from mojave import get_epochs_for_source
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    base_dir = '/home/ilya/code/vlbi_errors/examples/mojave'
    sources = ['1514-241', '1302-102', '0754+100', '0055+300', '0804+499',
               '1749+701', '0454+844']
    mapsize_dict = {'x': (512, 0.1), 'y': (512, 0.1), 'j': (512, 0.1),
                    'u': (512, 0.1)}
    mapsize_common = (512, 0.1)
    source_epoch_dict = dict()
    for source in sources:
        epochs = get_epochs_for_source(source, use_db='multifreq')
        print "Found epochs for source {}".format(source)
        print epochs
        source_epoch_dict.update({source: epochs[-1]})
    for source in sources:
        print "Simulating source {}".format(source)
        simulate(source, source_epoch_dict[source], ['x', 'y', 'j', 'u'],
                 n_sample=3, rotm_clim=[-200, 200],
                 path_to_script=path_to_script, mapsize_dict=mapsize_dict,
                 mapsize_common=mapsize_common, base_dir=base_dir,
                 rotm_value_0=0.)
