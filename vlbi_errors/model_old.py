import math
import numpy as np
from utils import EmptyImageFtError
from data_io import get_hdu, BinTable
from scipy import signal
from image import ImageGrid
try:
    import pylab
except ImportError:
    pylab = None

v_int = np.vectorize(int)
v_round = np.vectorize(round)


def create_grid(imsize):
    """Create meshgrid of size ``imsize``.

        :param imsize:
            Container of image dimensions
        :return:
            Meshgrid of size (imsize[0], imsize[1])
    """
    xsize, ysize = imsize
    x = np.linspace(0, xsize - 1, xsize)
    y = np.linspace(0, ysize - 1, ysize)
    x, y = np.meshgrid(x, y)
    return (x, y,)


def gaussianBeam(size_x, bmaj, bmin, bpa, size_y=None):
    """
    Generate and return a 2D Gaussian function
    of dimensions (size_x,size_y).

    See Briggs PhD (Appendix B) for details.

    :param size_x:
        Size of first dimension [pixels].
    :param bmaj:
        Beam major axis size [pixels].
    :param bmin:
        Beam minor axis size [pixels].
    :param bpa:
        Beam positional angle [deg].
    :param size_y (optional):
        Size of second dimension. Default is ``size_x``.
    :return:
        Numpy.ndarray of size (``size_x``, ``size_y``,).
    """
    size_y = size_y or size_x
    x, y = np.mgrid[-size_x: size_x + 1, -size_y: size_y + 1]
    # Constructing parameters of gaussian from ``bmaj``, ``bmin``, ``bpa``.
    a0 = 1. / (0.5 * bmaj) ** 2.
    c0 = 1. / (0.5 * bmin) ** 2.
    theta = math.pi * (bpa + 90.) / 180.
    a = math.log(2) * (a0 * math.cos(theta) ** 2. +
                       c0 * math.sin(theta) ** 2.)
    b = (-(c0 - a0) * math.sin(2. * theta)) * math.log(2.)
    c = math.log(2) * (a0 * math.sin(theta) ** 2. +
                       c0 * math.cos(theta) ** 2.)

    g = np.exp(-a * x ** 2. - b * x * y - c * y ** 2.)
    # FIXME: It is already normalized?
    #return g/g.sum()
    return g


# TODO: use IO.PyFits subclasses to i/o in this class.
# TODO: implement Image class with methods to load from FITS/txt.
# TODO: implement method that accepts Image() instance and plot model to them.
class Model(object):
    """
    Class that represents models.

    This class is used to describe model of VLBI-data in both image and
    uv-domain: clean components (delta functions), gaussians etc.
    """

    @classmethod
    def ft_delta(uvs, flux, x0, y0):
        """
        Return the Fourie Transform of delta functions defined in image plane by
        it's amplitude ``amp``, center ``x0`` & ``y0``.

        :param uvs:
            Iterable of uv-points for which calculate FT.

        :param amp:
            Amplitude or iterable of amplitudes of delta functions [Jy].

        :param x0:
            X-coordinate or iterable of X-coordinates of center [rad].

        :param y0:
            Y-coordinate or iterable of Y-coordinates of center [rad].

        :return:
            Numpy array of complex visibilities for specified points ``uvs``.
            Length of resulting array = len(uvs).
        """
        u = uvs[:, 0]
        v = uvs[:, 1]
        visibilities = (flux * np.exp(2.0 * math.pi * 1j * (u[:, np.newaxis] *
                        x0 + v[:, np.newaxis] * y0))).sum(axis=1)
        return visibilities

    # TODO: check Briggs PhD App. B
    @classmethod
    def ft_2dgaussian(uvs, amp, x0, y0, bmaj, bmin, bpa):
        """
        Return the Fourie Transform of 2D gaussian defined in image plane by
        it's amplitude ``amp``, center ``x0`` & ``y0``, major and minor axes
        ``bmaj`` & ``bmin`` and positional angle of major axis ``bpa``.

        :param uvs:
            Iterable of uv-points for which calculate FT.

        :param amp:
            Amplitude of gaussian [Jy].

        :param x0:
            X-coordinate of gaussian center [rad].

        :param y0:
            Y-coordinate of gaussian center [rad].

        :param bmaj:
            Size of major axis [rad].

        :param bmin:
            Size of min axis [rad].

        :param bpa:
            Positional angle of major axis [rad].

        :return:
            Numpy array of complex visibilities for specified points ``uvs``.
            Length of resulting array = len(uvs).
        """
        # Rotate the uv-plane on angle -bpa
        uvs_ = uvs.copy()
        uvs_[:, 0] = uvs[:, 0] * math.cos(bpa) + uvs[:, 1] * math.sin(bpa)
        uvs_[:, 1] = -uvs[:, 0] * math.sin(bpa) + uvs[:, 1] * math.cos(bpa)
        # Sequence of FT of gaussian(amp, x0=0, y0=0, bmaj, bmin) with len(ft) =
        # len(uvs)
        ft = amp * math.pi * bmaj * bmin * np.exp(-math.pi ** 2 *
                                                  (bmaj ** 2 * uvs_[:, 0] ** 2 +
                                                   bmin ** 2 * uvs_[:, 1] ** 2))
        # Multiply on phases of x0, y0 in rotated system
        x0_ = x0 * math.cos(bpa) + y0 * math.sin(bpa)
        y0_ = -x0 * math.sin(bpa) + y0 * math.cos(bpa)
        ft *= np.exp(2 * math.pi * 1j * (x0_ * uvs_[:, 0] + y0_ * uvs_[:, 1]))

        return ft

    def __init__(self):

        self.mas_to_rad = 4.8481368 * 1E-09
        self.degree_to_rad = 0.01745329
        self.imsize = None
        self.pixsize = None
        self.bmaj = None
        self.bmin = None
        self.bpa = None
        self._uvws = np.array([], dtype=[('u', float), ('v', float), ('w',
                              float)])
        self._image_stokes = {'I': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('bpa', float)]),
                              'Q': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('bpa', float)]),
                              'U': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('bpa', float)]),
                              'V': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('bpa', float)])}

        self._updated = {'I': False, 'Q': False, 'U': False, 'V': False}

        self._uv_correlations = {'RR': np.array([], dtype=complex), 'LL':
            np.array([], dtype=complex), 'RL': np.array([], dtype=complex),
            'LR': np.array([], dtype=complex)}

        self._image_grids = {'I': None, 'Q': None, 'U': None, 'V': None}

    def _ndarray_image_stokes(self, stokes='I'):
        """Just to remember:)
        """
        struct_array = self._image_stokes[stokes]
        return struct_array.view((float, len(struct_array.dtype.names)))

    def get_p(self, stokes='I'):
        """
        Shortcut for parameters of model.
        """
        p = self._image_stokes[stokes].flatten()

    # TODO: Sometimes we need to change only some parameters (circular gaussians),
    # thus we need mapping (f, x, y, b, a, bpa) <-> p.
    def set_p(self, p, stokes='I'):
        self._image_stokes[stokes] = p


    def get_uvws(self, data):
        """
        Sets ``_uvws`` attribute of self with values from UVData class instance
        ``data``.

        :param data:
            Instance of ``UVData`` class. Model visibilities will be calculated
            for (u,v)-points of this instance.
        """
        self._uvws = data.uvw

    def upvlot(self, stokes=None, style='a&p', PA=None):
        """
        Method that plots model transfered to the uv-domain vs. uv-radius
        :param style:
        :param PA:
        :return:
        """

        if not pylab:
            raise Exception('Install ``pylab`` for plotting!')
        if not stokes:
            stokes = 'I'
        # Check that there's ``stokes`` Stokes data to plot.
        if not self._image_stokes[stokes]:
            raise Exception('No stokes ' + stokes + ' to plot!')

    def ft(self, stoke='I', uvws=None):
        """
        Fourie transform model from image to uv-domain in specified points of
        uv-plane. If no uvw-points are specified, use _uvws attribute. If it is
        None, raise exception.

        :param stoke (optional):
            Stokes parameter to calculate on set of (u,v)-points using current
            model. (default: ``I``)

        :param uvws (optional):
            Set of (u,v,w)-points on which to calculate visibilities. If
            ``None`` is specified then use ``_uvws`` attriubute. If it doesn't
            contain any points then raise Exception. (default: ``None``)

        :return:
            Numpy array of visibilities of Stokes type ``stoke`` for set of
            (u,v,w)-points ``uvws``.
        """
        if not uvws:
            uvws = self._uvws
            if not uvws.size:
                raise Exception("Can't find uv-points on which to calculate"
                                " visibilities of model.")

        components = self._image_stokes[stoke]

        #TODO: check that component is not empty array. If so =>
        # raise EmptyImageFtException exception

        #flux, dx, dy, maja, mina, pa, ctype = component
        flux = components['flux']
        dx = components['dx']
        dy = components['dy']
        bmaj = components['bmaj']
        bmin = components['bmin']
        bpa = components['bpa']

        # FIXME: Should i use ``w``?
        # u, v, w must already be properly scaled
        u = uvws[:, 0]
        v = uvws[:, 1]
        #w = uvws['w']

        indxs_of_cc = np.where((flux != 0) & (bmaj == 0) & (bmin == 0) & (bpa
                      == 0))[0]
        indxs_of_gc = np.where((flux != 0) & (bmaj != 0) & (bmin != 0))[0]

        # Calculate visibilities_cc with indxs_of_cc. If indxs_of_cc is empty
        # then visibilities_cc will be zeros. So add it!
        visibilities_cc = np.zeros(len(uvws))
        if indxs_of_cc.size:
            visibilities_cc = (flux[indxs_of_cc] * np.exp(2.0 * math.pi * 1j *
                (u[:, np.newaxis] * dx[indxs_of_cc] + v[:, np.newaxis] *
                    dy[indxs_of_cc]))).sum(axis=1)

        # Calculate visibilities_gc with indxs_of_gc. If indxs_of_gc is empty
        # then visibilities_gc will be zeros. So add it!
        visibilities_gc = np.zeros(len(uvws))
        if indxs_of_gc.size:
            for indx in indxs_of_gc:
                visibilities_gc_ = self.ft_2dgaussian(uvws[:, :2],
                                                      *components[indx])
                visibilities_gc += visibilities_gc_

        return visibilities_cc + visibilities_gc

    @property
    def uv_correlations(self):
        """
        Property that updates and returns ``_uv_correlation`` attribute if
        model is updated.
        """
        if self._updated['I'] or self._updated['V']:

            if self._image_stokes['I'].size and self._image_stokes['V'].size:
                RR = self.ft(stoke='I') + self.ft(stoke='V')
                LL = self.ft(stoke='I') - self.ft(stoke='V')
            elif not self._image_stokes['V'].size and\
                                            self._image_stokes['I'].size:
                RR = self.ft(stoke='I')
                LL = RR
            elif not self._image_stokes['I'].size and\
                                            self._image_stokes['V'].size:
                RR = self.ft(stoke='V')
                LL = RR
            else:
                raise EmptyImageFtError('Not enough data for RR&LL visibility'
                                        ' calculation')
            self._uv_correlations['RR'] = RR
            self._uv_correlations['LL'] = LL

        elif self._updated['Q'] or self._updated['U']:

            if self._image_stokes['Q'].size and self._image_stokes['U'].size:
                RL = self.ft(stoke='Q') + 1j * self.ft(stoke='U')
                LR = self.ft(stoke='Q') - 1j * self.ft(stoke='U')
                # RL = FT(Q + j*U)
                # LR = FT(Q - j*U)
            else:
                raise EmptyImageFtError('Not enough data for RL&LR visibility'
                                        ' calculation')
            self._uv_correlations['RL'] = RL
            self._uv_correlations['LR'] = LR

        return self._uv_correlations

    # TODO: Check that AIPS CC binary table has dx&dy in rad!
    def add_cc_from_fits(self, fname, stoke='I', ver=1):
        """
        Adds CC components of Stokes type ``stoke`` to model from FITS-file.

        :param fname:
            Path to FITS-file with model (Clean Components CC-table).

        :param stoke (optional):
            Stokes parameter of file ``fname``. (default: ``I``)
        """
        dt = self._image_stokes[stoke].dtype
        # Fetch image parameters information
        header = get_hdu(fname).header
        self.imsize = (header['NAXIS1'], header['NAXIS2'],)
        self.pixref = (int(header['CRPIX1']), int(header['CRPIX2']),)
        self.bmaj = header['BMAJ'] * self.degree_to_rad
        self.bmin = header['BMIN'] * self.degree_to_rad
        self.bpa = header['BPA'] * self.degree_to_rad
        self.pixsize = (header['CDELT1'] * self.degree_to_rad,
                        header['CDELT2'] * self.degree_to_rad,)
        # Fetch CCs
        cc = BinTable(fname, extname='AIPS CC', ver=ver)
        adds = cc.load()
        temp = np.empty(len(adds), dtype=dt)
        temp.fill(np.NAN)
        temp['flux'] = adds['FLUX']
        temp['dx'] = adds['DELTAX'] * self.degree_to_rad
        temp['dy'] = adds['DELTAY'] * self.degree_to_rad
        # Append to _image_stokes
        self._image_stokes[stoke] = np.hstack((self._image_stokes[stoke],
                                              temp))
        self._updated[stoke] = True

        # Putting CCs to image grid
        self._image_grids[stoke] = np.zeros(self.imsize, dtype=float)
        dx, dy = self.pixsize
        x_c, y_c = self.pixref
        x_coords = v_int(v_round(adds['DELTAX'] * self.degree_to_rad / dx))
        y_coords = v_int(v_round(adds['DELTAY'] * self.degree_to_rad / dy))
        # 2 means that x_c & x_coords should be zero-indexed actually both.
        x = x_c + x_coords - 2
        y = y_c + y_coords - 2
        for i, (x_, y_,) in enumerate(zip(x, y)):
            self._image_grids[stoke][x_, y_] = adds['FLUX'][i]

    def add_from_txt(self, fname, stoke='I', style='aips'):
        """
        Adds components of Stokes type ``stoke`` to model from txt-file.

        :param fname:
            Path to text file with model.

        :param stoke (optional):
            Stokes parameter of file ``fname``. (default: ``I``)

        :param style (optional):
            Type of model specifying format. (default: ``aips``)
        """
        dt = self._image_stokes[stoke].dtype

        if style == 'aips':
            adds_ = np.loadtxt(fname)
            adds = adds_.copy()
            adds[:, 1] = self.degree_to_rad * adds_[:, 1]
            adds[:, 2] = self.degree_to_rad * adds_[:, 2]

        elif style == 'difmap':
            adds_ = np.loadtxt(fname, comments='!')
            adds = adds_.copy()
            adds[:, 1] = self.mas_to_rad * adds_[:, 1] * np.sin(adds_[:, 2] *
                                                                np.pi / 180.)
            adds[:, 2] = self.mas_to_rad * adds_[:, 1] * np.cos(adds_[:, 2] *
                                                                np.pi / 180.)
        else:
            raise Exception('style = aips or difmap')

        # If components are CCs
        if adds.shape[1] == 3:
            nans = np.empty((len(adds), 3,))
            nans.fill(np.NAN)
            adds = np.hstack((adds, nans,))
        elif adds.shape[1] == 6:
            raise NotImplementedError('Implement difmap format of gaussian'
                                      ' components')
        else:
            raise Exception

        self._image_stokes[stoke] = adds.ravel().view(dt)
        self._updated[stoke] = True

    def plot_model(self, image=None, stokes='I'):
        """
        Method that plots specified model to specified instance of ``Image``
        class.

        :param image:
            Instance of ``Image`` class on which to plot model.

        :param stokes (optional):
        """
        pass

    def make_image(self, bmaj=None, bmin=None, bpa=None, stokes='I',
                   size_x=None, size_y=None):
        """
        Method that returns instance of Image class using model data (CCs, clean
        beam, map size & pixel size).
        :param bmaj: (optional)
            Beam major axis size [rad].
        :param bmin: (optional)
            Beam minor axis size [rad].
        :param bpa: (optional)
            Beam positional angle [deg]
        :param stokes: (optional)
            Stokes parameter of image. I, Q, U or V.
        :param size_x (optional):
            Size of the first dimension [pixels]. Default is half of image size.
        :param size_y (optional):
            Size of the second dimension [pixels]. Default is half of image size.
        :return:
            Instance of image class.
        """
        if not size_x:
            size_x = int(self.imsize[0] / 2.)
        if not size_y:
            size_y = int(self.imsize[1] / 2.)
        if not bmaj:
            bmaj = self.bmaj / abs(self.pixsize[0])
        if not bmin:
            bmin = self.bmin / abs(self.pixsize[1])
        if bpa is None:
            bpa = self.bpa
        gaussian_beam = gaussianBeam(size_x, bmaj, bmin, bpa)
        cc_convolved = signal.fftconvolve(self._image_grids[stokes],
                                          gaussian_beam, mode='same')
        image = ImageGrid()
        image.add_from_array(cc_convolved, pixsize=self.pixsize, bmaj=self.bmaj,
                             bmin=self.bmin, bpa=self.bpa, stokes=stokes)
        return image

    def clear_im(self, stoke='I'):
        """
        Clear the model for stoke Stokes parameter.

        :param stoke (optional):
            Stokes parameter to clear. (default: ``I``)
        """
        self._image_stokes[stoke] = np.array([], dtype=[('flux', float),
                        ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                         float), ('bpa', float)])
        self._updated[stoke] = False

    def clear_all_im(self):
        """
        Clear model for all Stokes parameters.
        """
        for stoke in self._image_stokes.keys():
            self.clear(stoke=stoke)

    def clear_uv(self, hand=None):
        """
        Clear uv-correlations.
        """

        if not hand:
            for hand in self._uv_correlations.keys():
                self._uv_correlations[hand] = np.array([], dtype=complex)
        else:
            self._uv_correlations[hand] = np.array([], dtype=complex)


if __name__ == '__main__':

    model = Model()
    model.add_cc_from_fits('0003-066.j.2006_07_07.i_0.1.fits')
    image = model.make_image()
