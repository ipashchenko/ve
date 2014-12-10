import math
import numpy as np
import scipy as sp
from utils import degree_to_mas, mas_to_rad, gaussianBeam, _function_wrapper,\
    vcomplex, gaussian, EmptyImageFtError
from image import ImageGrid
from scipy import signal
from data_io import BinTable, get_fits_image_info
from uv_data import open_fits
from stats import LnPost

try:
    import pylab
except ImportError:
    pylab = None


def uv_correlations(uvdata, modeli=None, modelq=None, modelu=None, modelv=None,
                    modelrr=None, modelll=None):
    """
    Function that accepts models of stokes parameters in image plane and returns
    cross-correlations (whatever possible) for given instance of ``UVData``
    class.
    """
    # Dictionary with keys - 'RR', 'LL', ... and values - correlations
    uv_correlations = dict()
    if modeli or modelv:
        if modeli and modelv:
            RR = modeli.ft(uv) + modelv.ft(uv)
            LL = modeli.ft(uv) - modelv.ft(uv)
        elif not modelv and modeli:
            RR = modeli.ft(uv)
            LL = RR
        elif not modeli and modelv:
            RR = modelv.ft(uv)
            LL = RR
        else:
            raise EmptyImageFtError('Not enough data for RR&LL visibility'
                                    ' calculation')
        # Setting up parallel hands correlations
        uv_correlations.update({'RR': RR})
        uv_correlations.update({'LL': LL})

    else:
        if modelrr or modelll:
            RR = modelrr.ft(uv)
            LL = modelll.ft(uv)
            # Setting up parallel hands correlations
            uv_correlations.update({'RR': RR})
            uv_correlations.update({'LL': LL})

    if modelq or modelu:
        if modelq and modelu:
            RL = modelq.ft(uv) + 1j * modelu.ft(uv)
            LR = modelq.ft(uv) - 1j * modelu.ft(uv)
            # RL = FT(Q + j*U)
            # LR = FT(Q - j*U)
            # Setting up cross hands correlations
            uv_correlations.update({'RL': RL})
            uv_correlations.update({'LR': LR})
        else:
            raise EmptyImageFtError('Not enough data for RL&LR visibility'
                                    ' calculation')

    return uv_correlations


# TODO: move create_image_grid to ``Component`` subclasses. Thus, we can mix
# different components in one model and call their own methods. ``Model`` should
# have method, specifying image grid parameters, construct 2D numpy array and
# call component's method ``add_to_image_grid`` to add to this general 2D array.
class Model(object):
    """
    Basic class that represents general functionality of models.
    """
    def __init__(self, stokes=None):
        self._components = list()
        self._uv = None
        self.stokes = stokes
        self._image_grid = None

    def __add__(self, other):
        self._components.extend(other._components)

    def add_component(self, component):
        self._components.append(component)

    def add_components(self, *components):
        for component in components:
            self._components.append(component)

    def remove_component(self, component):
        self._components.remove(component)

    def remove_components(self, *components):
        for component in components:
            self._components.remove(component)

    def clear_components(self):
        self._components = list()

    def clear_uv(self):
        self._uv = None

    def ft(self, uv=None):
        """
        Returns FT of model's components at specified points of uv-plane.
        """
        if uv is None:
            uv = self._uv
        ft = np.zeros(len(uv), dtype=complex)
        for component in self._components:
            ft += component.ft(uv)
        return ft

    def get_uv(self, uvdata):
        """
        Sets ``_uv`` attribute of self with values from UVData class instance
        ``uvdata``.

        :param uvdata:
            Instance of ``UVData`` class. Model visibilities will be calculated
            for (u,v)-points of this instance.
        """
        self._uv = uvdata.uvw[:, :2]

    def uvplot(self, uv=None, style='a&p', sym='.r'):
        """
        Plot FT of model vs uv-radius.
        """
        if uv is None:
            uv = self._uv
        ft = self.ft(uv=uv)

        if style == 'a&p':
            a1 = np.angle(ft)
            a2 = np.real(np.sqrt(ft * np.conj(ft)))
        elif style == 're&im':
            a1 = ft.real
            a2 = ft.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        uv_radius = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        pylab.subplot(2, 1, 1)
        pylab.plot(uv_radius, a2, sym)
        if style == 'a&p':
            pylab.ylim([0., 1.3 * max(a2)])
        pylab.subplot(2, 1, 2)
        pylab.plot(uv_radius, a1, sym)
        if style == 'a&p':
            pylab.ylim([-math.pi, math.pi])
        pylab.show()

    @property
    def p(self):
        """
        Shortcut for parameters of model.
        """
        p = list()
        for component in self._components:
            p.extend(component.p)
        return p

    @p.setter
    def p(self, p):
        for component in self._components:
            component.p = p[:component.size]
            p = p[component.size:]

    @property
    def size(self):
        return len(self.p)

    def add_to_image_grid(self, image_grid):
        """
        Add model to instances of ``ImagePlane`` subclasses.

        :param image_grid:
            Instance of ``ImagePlane`` subclass.
        """
        for component in self._components:
            image_grid.add_component(component)

    def make_image(self, fname=None, imsize=None, bmaj=None, bmin=None,
                   bpa=None, size_x=None, size_y=None):
        """
        Method that returns instance of Image class using model data (CCs, clean
        beam, map size & pixel size).

        :param fname: (optional)
            Fits file of image to get image parameters.
        :param bmaj: (optional)
            Beam major axis size [rad].
        :param bmin: (optional)
            Beam minor axis size [rad].
        :param bpa: (optional)
            Beam positional angle [deg]
        :param size_x (optional):
            Size of the first dimension [pixels]. Default is half of image size.
        :param size_y (optional):
            Size of the second dimension [pixels]. Default is half of image
            size.
        :return:
            Instance of ``Image`` class.
        """
        # If we got fits-file then get parameters of image from it
        if fname:
            imsize, pixref, (bmaj, bmin, bpa,), pixsize =\
                get_fits_image_info(fname)

        # First create image grid with components
        image_grid = ImageGrid(imsize=imsize, pixref=pixref, pixsize=pixsize)
        # Putting model components to image grid
        self.add_to_image_grid(image_grid)

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
        cc_convolved = signal.fftconvolve(image_grid.image_grid, gaussian_beam,
                                          mode='same')
        image = ImageGrid()
        image.add_from_array(cc_convolved, pixsize=self.pixsize, bmaj=self.bmaj,
                             bmin=self.bmin, bpa=self.bpa)
        return image

    # def create_image_grid(self, imsize, pixref, bmaj, bmin, bpa, pixsize):
    #     raise NotImplementedError()


# TODO: add ft_all method for speed up
class CCModel(Model):
    """
    Class that represents clean components model.
    """
    def add_cc_from_fits(self, fname, ver=1):
        """
        Adds CC components to model from FITS-file.

        :param fname:
            Path to FITS-file with model (Clean Components CC-table).
        """
        cc = BinTable(fname, extname='AIPS CC', ver=ver)
        adds = cc.load()
        for flux, x, y in zip(adds['FLUX'], adds['DELTAX'] * degree_to_mas,
                              adds['DELTAY'] * degree_to_mas):
            # We keep positions in mas
            component = DeltaComponent(flux, x, y)
            self.add_component(component)

    # def create_image_grid(self, imsize, pixref, bmaj, bmin, bpa, pixsize):
    #     image_grid = np.zeros(imsize, dtype=float)
    #     dx, dy = pixsize
    #     x_c, y_c = pixref
    #     x_coords = list()
    #     y_coords = list()
    #     flux_list = list()
    #     for components in self._components:
    #         flux, r, theta = self.p
    #         x = r * math.cos(theta)
    #         y = r * math.sin(theta)
    #         x_coords.append(int(round(x / dx)))
    #         y_coords.append(int(round(y / dy)))
    #         flux_list.append(flux)
    #     x_coords = np.asarray(x_coords)
    #     y_coords = np.asarray(y_coords)
    #     # 2 means that x_c & x_coords should be zero-indexed actually both.
    #     x = x_c + x_coords - 2
    #     y = y_c + y_coords - 2
    #     for i, (x_, y_,) in enumerate(zip(x, y)):
    #         image_grid[x_, y_] = flux_list[i]


class Component(object):
    """
    Basic class that implements single component of model.
    """
    def __init__(self):
        self._p = None
        self._parnames = ['flux', 'x', 'y']
        self._fixed = np.array([False, False, False])
        self._lnprior = dict()

    def add_prior(self, **lnprior):
        """
        Add prior for some parameters.
        :param lnprior:
            Kwargs with keys - name of the parameter and values - (callable,
            args, kwargs,) where args & kwargs - additional arguments to
            callable. Each callable is called callable(p, *args, **kwargs).
        Example:
        {'flux': (scipy.stats.uniform, [0., 10.], dict(),),
         'bmaj': (scipy.stats.uniform, [0, 5.], dict(),),
         'e': (scipy.stats.beta.logpdf, [alpha, beta], dict(),)}
        First key will result in calling: scipy.stats.norm.logpdf(x, mu, s) as
        prior for ``flux`` parameter.
        """
        for key, value in lnprior.items():
            if key in self._parnames:
                func, args, kwargs = value
                self._lnprior.update({key: _function_wrapper(func, args,
                                                             kwargs)})
            else:
                raise Exception("Uknown parameter name: " + str(key))

    # TODO: properties must return only free parameters!
    @property
    def p(self):
        """
        Shortcut for parameters of model.
        """
        return self._p[np.logical_not(self._fixed)]

    @p.setter
    def p(self, p):
        # print "Setting comp's parameters with ", p
        # print "Before: ", self._p
        self._p[np.logical_not(self._fixed)] = p[:]
        # print "After: ", self._p

    def ft(self, uv):
        """
        Method that returns Fourier Transform of component in given points of
        uv-plane.
        :param uv:
            2D-numpy array of uv-coordinates with shape (#data, 2,)
        :return:
            Numpy array (length = length(uv)) with complex visibility values.
        """
        raise NotImplementedError("Method must me implemented in subclasses!")

    def add_to_image_grid(self, image_grid):
        """
        Add component to image plane.

        :param image_grid:
            Instance of ``ImagePlane`` class
        """
        raise NotImplementedError("Method must me implemented in subclasses!")

    def uvplot(self, uv, style='a&p', sym='.k'):
        """
        Plot FT of component vs uv-radius.
        """
        ft = self.ft(uv)

        if style == 'a&p':
            a1 = np.angle(ft)
            a2 = np.real(np.sqrt(ft * np.conj(ft)))
        elif style == 're&im':
            a1 = ft.real
            a2 = ft.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        uv_radius = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        pylab.subplot(2, 1, 1)
        pylab.plot(uv_radius, a2, sym)
        pylab.subplot(2, 1, 2)
        pylab.plot(uv_radius, a1, sym)
        if style == 'a&p':
            pylab.ylim([-math.pi, math.pi])
        pylab.show()

    @property
    def lnpr(self):
        if not self._lnprior.keys():
            print "No priors specified"
            return 0
        lnprior = list()
        for key, value in self._lnprior.items():
            # Index of ``key`` in ``_parnames`` the same as in ``_p``
            par = self._p[self._parnames.index(key)]
            lnprior.append(value(par))
        return sum(lnprior)


class EGComponent(Component):
    """
    Class that implements elliptical gaussian component.
    """
    def __init__(self, flux, x, y, bmaj, e, bpa, fixed=None):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component phase center [mas].
        :param y:
            Y-coordinate of component phase center [mas].
        :param bmaj:
            Std of component size [mas].
        :param e:
            Minor-to-major axis ratio.
        :param bpa:
            Positional angle of major axis. Angle counted from x-axis of image
            plane counter clockwise [rad].
        :param fixed optional:
            If not None then it is iterable of parameter's names that are fixed.
        """
        super(EGComponent, self).__init__()
        self._parnames.extend(['bmaj', 'e', 'bpa'])
        self._fixed = np.concatenate((self._fixed,
                                      np.array([False, False, False]),))
        self._p = np.array([flux, x, y, bmaj, e, bpa])
        self.size = 6
        if fixed is not None:
            for par in fixed:
                if par not in self._parnames:
                    raise Exception('Uknown parameter ' + str(par) + ' !')
                self._fixed[self._parnames.index(par)] = True

    def ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.

        :note:

            The value of the Fourier transform of gaussian function (Wiki):

            g(x, y) = A*exp[-(a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2)]  (1)

            where:

                a = cos(\theta)**2/(2*std_x**2)+sin(\theta)**2/(2*std_y**2)
                b = sin(2*\theta)/(2*std_x**2)-sin(2*\theta)/(2*std_y**2)
                (corresponds to rotation counter clockwise)
                c = sin(\theta)**2/(2*std_x**2)+cos(\theta)**2/(2*std_y**2)

            For x0=0, y0=0 in point u,v of uv-plane is (Briggs Thesis):

            2*pi*A*(4*a*c-b**2)**(-1/2)*exp[(4*pi**2/(4*a*c-b**2))*
                    (-c*u**2+b*u*v-a*v**2)] (2)

            As we parametrize the flux as full flux of gaussian (that is flux at
            zero (u,v)-spacing), then change coefficient in front of exponent to
            A.

            Shift of (x0, y0) in image plane corresponds to phase shift in
            uv-plane:

            ft(x0,y0) = ft(x0=0,y0=0)*exp(-2*pi*(u*x0+v*y0))
        """
        try:
            flux, x0, y0, bmaj, e, bpa = self._p
        # If we call method inside ``CGComponent``
        except ValueError:
            flux, x0, y0, bmaj = self._p
            e = 1.
            bpa = 0.

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad
        bmaj *= mas_to_rad

        u = uv[:, 0]
        v = uv[:, 1]
        # Construct parameter of gaussian function (1)
        std_x = bmaj
        std_y = e * bmaj
        a = math.cos(bpa) ** 2. / (2. * std_x ** 2.) + \
            math.sin(bpa) ** 2. / (2. * std_y ** 2.)
        b = math.sin(2. * bpa) / (2. * std_x ** 2.) - \
            math.sin(2. * bpa) / (2. * std_y ** 2.)
        c = math.sin(bpa) ** 2. / (2. * std_x ** 2.) + \
            math.cos(bpa) ** 2. / (2. * std_y ** 2.)
        # Calculate the value of FT in point (u,v) for x0=0,y0=0 case using (2)
        k = (4. * a * c - b ** 2.)
        ft = flux * np.exp((4. * math.pi ** 2. / k) * (-c * u ** 2. +
                                                       b * u * v - a * v ** 2.))
        ft = vcomplex(ft)
        # If x0=!0 or y0=!0 then shift phase accordingly
        if x0 or y0:
            ft *= np.exp(-2. * math.pi * 1j * (u * x0 + v * y0))
        return ft

    def add_to_image_grid(self, image_grid):
        """
        Add component to given instance of ``ImagePlane`` class.
        """
        # Cellsize
        dx, dy = image_grid.dx, image_grid.dy
        # Center of image
        x_c, y_c = image_grid.x_c, image_grid.y_c

        # Parameters of component
        try:
            flux, x0, y0, bmaj, e, bpa = self._p
        # If we call method inside ``CGComponent``
        except ValueError:
            flux, x0, y0, bmaj = self._p
            e = 1.
            bpa = 0.

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad
        bmaj *= mas_to_rad

        # Construct parameter of general gaussian function
        std_x = bmaj
        std_y = e * bmaj
        # Amplitude of gaussian component
        amp = flux / (2. * math.pi * bmaj ** 2. * e)

        # Create gaussian function of (x, y) with given parameters
        gaussf = gaussian(amp, x0, y0, std_x, std_y, bpa=bpa)

        # Calculating angular distances of cells from center of component
        # from cell numbers to relative distances
        # arrays with elements from 1 to imsize
        x, y = np.mgrid[1: image_grid.imsize[0] + 1,
                        1: image_grid.imsize[1] + 1]
        # from -imsize/2 to imsize/2
        x = x - x_c
        y = y - y_c
        # the same in rads
        x = x * dx
        y = y * dy
        # relative to component center
        x = x - x0
        y = y - y0

        # Creating grid with component's flux at each cell
        fluxes = gaussf(x, y)

        # Adding component's flux to image grid
        image_grid.image_grid += fluxes


class CGComponent(EGComponent):
    """
    Class that implements circular gaussian component.
    """
    def __init__(self, flux, x, y, bmaj, fixed=None):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component phase center [mas].
        :param y:
            Y-coordinate of component phase center [mas].
        :param bmaj:
            Std of component size [mas].
        """
        super(CGComponent, self).__init__(flux, x, y, bmaj, e=1., bpa=0.,
                                          fixed=fixed)
        self._fixed = self._fixed[:-2]
        self._parnames.remove('e')
        self._parnames.remove('bpa')
        self._p = self._p[:-2]
        self.size = 4


class DeltaComponent(Component):
    """
    Class that implements delta-function component.
    """
    def __init__(self, flux, x, y, fixed=None):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component phase center [mas].
        :param y:
            Y-coordinate of component phase center [mas].
        """
        super(DeltaComponent, self).__init__()
        self._p = np.array([flux, x, y])
        self.size = 3
        if fixed is not None:
            for par in fixed:
                if par not in self._parnames:
                    raise Exception('Uknown parameter ' + str(par) + ' !')
                self._fixed[self._parnames.index(par)] = True

    def ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.
        """
        flux, x0, y0 = self._p

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad

        u = uv[:, 0]
        v = uv[:, 1]
        visibilities = (flux * np.exp(2.0 * math.pi * 1j *
                                      (u[:, np.newaxis] * x0 +
                                       v[:, np.newaxis] * y0))).sum(axis=1)
        return visibilities

    def add_to_image_grid(self, image_grid):
        """
        Add component to given instance of ``ImagePlane`` class.
        """
        dx, dy = image_grid.dx, image_grid.dy
        x_c, y_c = image_grid.x_c, image_grid.y_c

        flux, x0, y0 = self._p

        # There's ONE place to convert them
        x0 *= mas_to_rad
        y0 *= mas_to_rad

        x_coords = int(round(x0 / dx))
        y_coords = int(round(y0 / dy))
        # 2 means that x_c & x_coords should be zero-indexed actually both.
        x = x_c + x_coords - 2
        y = y_c + y_coords - 2
        image_grid.image_grid[x, y] += flux


if __name__ == "__main__":

    ## TESTING plotting cc-models on uv-data
    # cc_fits_file = '1038+064.l22.2010_05_21.icn.fits'
    # uv_fits_file = '1038+064.l22.2010_05_21.uvf'
    # ccmodel = CCModel(stokes='I')
    # ccmodel.add_cc_from_fits(cc_fits_file)
    # uvdata = open_fits(uv_fits_file)
    # uv = uvdata.uvw[:, :2]
    # ft = ccmodel.ft(uv)
    # ccmodel.uvplot(uv)

    # TESTING fitting gaussian components to uv-data
    # # Load uv-data
    # uvdata = open_fits('1308+326.U1.2009_08_28.UV_CAL')
    # uv = uvdata.uvw[:, :2]
    # # Create several components
    # cg1 = CGComponent(2.44, 0.02, -0.02, 0.10)
    # cg2 = CGComponent(0.041, 0.71, -1.05, 1.18)
    # cg3 = CGComponent(0.044, 2.60, -3.20, 0.79)
    # cg4 = CGComponent(0.021, 1.50, -5.60, 2.08)
    # cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 1.], dict(),))
    # cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    # cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    # cg4.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    # # Create model
    # mdl1 = Model(stokes='I')
    # # Add components to model
    # mdl1.add_component(cg1)
    # mdl1.add_component(cg2)
    # mdl1.add_component(cg3)
    # mdl1.add_component(cg4)
    # # Create posterior for data & model
    # lnpost = LnPost(uvdata, mdl1)
    # lnpr = LnPrior(mdl1)
    # lnlik = LnLikelihood(uvdata, mdl1)
    # # model.uvplot(uv = uv)
    # # model.ft(uv=uv)
    # import emcee
    # ndim = mdl1.size
    # nwalkers = 100
    # # p0 = mdl1.p
    # # cov = np.zeros(ndim * ndim).reshape((ndim, ndim,))
    # # cov[0, 0] = 0.1
    # # cov[1, 1] = 0.1
    # # cov[2, 2] = 0.1
    # # cov[3, 3] = 0.1
    # # sampler = emcee.MHSampler(cov, ndim, lnpost)
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    # p_std1 = [0.01, 0.01, 0.01, 0.01]
    # p_std2 = [0.003, 0.01, 0.01, 0.01]
    # p0 = emcee.utils.sample_ball(mdl1.p, p_std1 + p_std2 * 3, size=nwalkers)
    # pos, prob, state = sampler.run_mcmc(p0, 100)
    # sampler.reset()
    # sampler.run_mcmc(pos, 700)
    # # image_grid = ImageGrid(fname='J0005+3820_S_1998_06_24_fey_map.fits')
    # # print image_grid.dx, image_grid.dy, image_grid.imsize, image_grid.x_c,\
    # #     image_grid.y_c
    # # print image_grid.image_grid
    # # eg = EGComponent(1., 5., 5., 3., 0.5, 1.)
    # # print eg.p
    # # print eg._p
    # # eg.add_to_image_grid(image_grid)
    # # print image_grid.image_grid



    # With sparse RA data
    # TESTING fitting gaussian components to uv-data
    # Load uv-data
    uvdata = open_fits('0716+714_raes03dp_C_LL_uva.fits')
    uvdata.data['uvw'] *= 10 ** 9
    uv = uvdata.uvw[:, :2]
    # Create several components
    cg1 = CGComponent(1., 0.0, 0.0, 0.05)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 2.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 1.], dict(),))
    # Create model
    mdl1 = Model(stokes='RR')
    # Add components to model
    mdl1.add_component(cg1)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl1)
    import emcee
    ndim = mdl1.size
    nwalkers = 50
    p0 = mdl1.p
    cov = np.zeros(ndim * ndim).reshape((ndim, ndim,))
    cov[0, 0] = 0.1
    cov[1, 1] = 0.01
    cov[2, 2] = 0.01
    cov[3, 3] = 0.01
    sampler = emcee.MHSampler(cov, ndim, lnpost)
    pos, prob, state = sampler.run_mcmc(p0, 1000)
    #sampler.reset()
    #sampler.run_mcmc(pos, 5000)
    # # image_grid = ImageGrid(fname='J0005+3820_S_1998_06_24_fey_map.fits')
    # # print image_grid.dx, image_grid.dy, image_grid.imsize, image_grid.x_c,\
    # #     image_grid.y_c
    # # print image_grid.image_grid
    # # eg = EGComponent(1., 5., 5., 3., 0.5, 1.)
    # # print eg.p
    # # print eg._p
    # # eg.add_to_image_grid(image_grid)
    # # print image_grid.image_grid
