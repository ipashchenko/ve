#!/usr/bin python
# -*- coding: utf-8 -*-

import math
import numpy as np
from utils import gaussianBeam
from image import Image
from scipy import signal
from data_io import BinTable, get_hdu
from uv_data import open_fits

try:
    import pylab
except ImportError:
    pylab = None

vcomplex = np.vectorize(complex)
v_int = np.vectorize(int)
v_round = np.vectorize(round)

mas_to_rad = 4.8481368 * 1E-09
degree_to_rad = 0.01745329


def ln_uniform(x, a, b):
    assert(a < b)
    if not a < x < b:
        return -np.inf
    return -math.log(b - a)


def is_sorted(lst):
    return (sorted(lst) == lst)


def get_fits_image_info(self, fname):
    header = get_hdu(fname).header
    imsize = (header['NAXIS1'], header['NAXIS2'],)
    pixref = (int(header['CRPIX1']), int(header['CRPIX2']),)
    bmaj = header['BMAJ'] * degree_to_rad
    bmin = header['BMIN'] * degree_to_rad
    bpa = header['BPA'] * degree_to_rad
    pixsize = (header['CDELT1'] * degree_to_rad,
               header['CDELT2'] * degree_to_rad,)
    return imsize, pixref, (bmaj, bmin, bpa,), pixsize


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
        self._p = None
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

    def uvplot(self, uv=None, style='a&p'):
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
        pylab.plot(uv_radius, a2, '.k')
        if style == 'a&p':
            pylab.ylim([0., 1.3 * max(a2)])
        pylab.subplot(2, 1, 2)
        pylab.plot(uv_radius, a1, '.k')
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
        # Putting components to image grid
        image_grid = ImagePlane(imsize, pixref, pixsize)
        for component in self._components:
            image_grid.add_component(component)

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
        image = Image()
        image.add_from_array(cc_convolved, pixsize=self.pixsize, bmaj=self.bmaj,
                             bmin=self.bmin, bpa=self.bpa)
        return image

    # def create_image_grid(self, imsize, pixref, bmaj, bmin, bpa, pixsize):
    #     raise NotImplementedError()


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
        for flux, x, y in zip(adds['FLUX'], adds['DELTAX'] * degree_to_rad,
                              adds['DELTAY'] * degree_to_rad):
            r = math.sqrt(x ** 2. + y ** 2.)
            theta = math.atan2(y / x)
            component = DeltaComponent(flux, r, theta)
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
        self._parnames = ['flux', 'r', 'theta']
        self._lnprior = dict()

    def add_prior(self, **lnprior):
        """
        Add prior for some parameters.
        :param lnprior:
            Kwargs with keys - name of the parameter and values - (callable,
            args, kwargs,) where args & kwargs - additional arguments to
            callable. Each callable is called callable(p, *args, **kwargs).
        Example:
        {'flux': (scipy.stats.norm.logpdf, [mu, s], dict(),),
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

    @property
    def size(self):
        return len(self.p)

    @property
    def p(self):
        """
        Shortcut for parameters of model.
        """
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

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
        pass

    def uvplot(self, uv=None, style='a&p'):
        """
        Plot FT of component vs uv-radius.
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
        pylab.plot(uv_radius, a2, '.k')
        pylab.subplot(2, 1, 2)
        pylab.plot(uv_radius, a1, '.k')
        if style == 'a&p':
            pylab.ylim([-math.pi, math.pi])
        pylab.show()

    @property
    def lnpr(self):
        print "Calculating lnprior for component ", self
        if not self._lnprior.keys():
            print "No priors specified"
            return 0
        lnprior = list()
        for key, value in self._lnprior.items():
            lnprior.append(value(self.__getattribute__(key)))
        return sum(lnprior)


class EGComponent(Component):
    """
    Class that implements elliptical gaussian component.
    """
    def __init__(self, flux, r, theta, bmaj, e, bpa, **lnprior):
        """
        :param flux:
            Flux of component [Jy].
        :param r:
            Distance of component form phase center [rad].
        :param theta:
            Angle counted from x-axis of image plane counter clockwise [rad].
        :param bmaj:
            Std of component size [rad].
        :param e:
            Minor-to-major axis ratio.
        :param bpa:
            Positional angle of major axis. Angle counted from x-axis of image
            plane counter clockwise [rad].

        :note:
            This is nonstandard convention on ``theta``.
        """
        super(EGComponent, self).__init__()
        self._parnames.extend(['bmaj', 'e', 'bpa'])
        self.flux = flux
        self.r = r
        self.theta = theta
        self.bmaj = bmaj
        self.e = e
        self.bpa = bpa
        self._p = [flux, r, theta, bmaj, e, bpa]

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
            flux, r, theta, bmaj, e, bpa = self.p
        # If we call method inside ``CGComponent``
        except ValueError:
            flux, r, theta, bmaj = self.p
            e = 1.
            bpa = 0.

        x0 = r * math.cos(theta)
        y0 = r * math.sin(theta)
        u = uv[:, 0]
        v = uv[:, 1]
        # Construct parameter of gaussian function (1)
        std_x = bmaj
        std_y = e * bmaj
        bpa = self.bpa
        a = math.cos(bpa) ** 2. / (2. * std_x ** 2.) + \
            math.sin(bpa) ** 2. / (2. * std_y ** 2.)
        b = math.sin(2. * bpa) / (2. * std_x ** 2.) - \
            math.sin(2. * bpa) / (2. * std_y ** 2.)
        c = math.sin(bpa) ** 2. / (2. * std_x ** 2.) + \
            math.cos(bpa) ** 2. / (2. * std_y ** 2.)
        # Calculate the value of FT in point (u,v) for x0=0,y0=0 case using (2)
        k = (4. * a * c - b ** 2.)
        ft = self.flux * np.exp((4. * math.pi ** 2. / k) * (-c * u ** 2. +
                                                            b * u * v -
                                                            a * v ** 2.))
        ft = vcomplex(ft)
        # If x0=!0 or y0=!0 then shift phase accordingly
        if x0 or y0:
            ft *= np.exp(-2. * math.pi * 1j * (u * x0 + v * y0))
        return ft


class CGComponent(EGComponent):
    """
    Class that implements circular gaussian component.
    """
    def __init__(self, flux, r, theta, bmaj):
        """
        :param flux:
            Flux of component [Jy].
        :param r:
            Distance of component form phase center [rad].
        :param theta:
            Angle counted from x-axis of image plane counter clockwise [rad].
        :param bmaj:
            Std of component size [rad].

        :note:
            This is nonstandard convention on ``theta``.
        """
        super(CGComponent, self).__init__(flux, r, theta, bmaj, e=1., bpa=0.)
        self._parnames.remove('e')
        self._parnames.remove('bpa')
        self._p = [flux, r, theta, bmaj]


class DeltaComponent(Component):
    """
    Class that implements delta-function component.
    """
    def __init__(self, flux, r, theta):
        """
        :param flux:
            Flux of component [Jy].
        :param r:
            Distance form phase center [rad].
        :param theta:
            Angle counted from x-axis of image plane counter clockwise [rad].

        :note:
            This is nonstandard convention on ``theta``.
        """
        super(DeltaComponent, self).__init__()
        self.flux = flux
        self.r = r
        self.theta = theta
        self._p = [flux, r, theta]

    def ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.
        """
        flux, r, theta = self.p
        x0 = r * math.cos(theta)
        y0 = r * math.sin(theta)
        u = uv[:, 0]
        v = uv[:, 1]
        visibilities = (self.flux * np.exp(2.0 * math.pi * 1j *
                                           (u[:, np.newaxis] * x0 +
                                            v[:, np.newaxis] * y0))).sum(axis=1)
        return visibilities

    def add_to_image_grid(self, image_grid):
        """
        Add component to given instance of ``ImagePlane`` class.
        """
        dx, dy = image_grid.dx, image_grid.dy
        x_c, y_c = image_grid.x_c, image_grid.y_c

        flux, r, theta = self.p
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        x_coords = int(round(x / dx))
        y_coords = int(round(y / dy))
        # 2 means that x_c & x_coords should be zero-indexed actually both.
        x = x_c + x_coords - 2
        y = y_c + y_coords - 2
        image_grid.image_grid[x, y] += flux


# TODO: add ``add_noise`` method
class ImagePlane(object):
    """
    Class that represents models in image plane.
    """
    def __init__(self, fname=None, imsize=None, pixref=None, pixsize=None):
        if not fname:
            self.from_image(fname)
        else:
            self.imsize = imsize
            self.dx, self.dy = pixsize
            self.x_c, self.y_c = pixref
        self.image_grid = np.zeros(self.imsize, dtype=float)

    def from_image(self, fname):
        imsize, pixref, (bmaj, bmin, bpa,), pixsize = get_fits_image_info(fname)
        self.imsize = imsize
        self.dx, self.dy = pixsize
        self.x_c, self.y_c = pixref

    def add_component(self, component):
        component.add_to_image_grid(self)


class LnLikelihood(object):
    def __init__(self, uvdata, model, average_freq=True):
        error = uvdata.error(average_freq=average_freq)
        self.model = model
        self.uv = uvdata.uvw[:, :2]
        stokes = model.stokes
        if average_freq:
            if stokes == 'I':
                self.uvdata = 0.5 * (uvdata.uvdata_freq_averaged[:, 0] +
                                     uvdata.uvdata_freq_averaged[:, 1])
                self.error = 0.5 * np.sqrt(error[:, 0] ** 2. +
                                           error[:, 1] ** 2.)
            elif stokes == 'RR':
                self.uvdata = uvdata.uvdata_freq_averaged[:, 0]
                self.error = error[:, 0]
            elif stokes == 'LL':
                self.uvdata = uvdata.uvdata_freq_averaged[:, 1]
                self.error = error[:, 1]
            else:
                raise Exception("Working with only I, RR or LL!")
        else:
            if stokes == 'I':
                self.uvdata = 0.5 * (uvdata.uvdata[:, 0] + uvdata.uvdata[:, 1])
            elif stokes == 'RR':
                self.uvdata = uvdata.uvdata[:, 0]
            elif stokes == 'LL':
                self.uvdata = uvdata.uvdata[:, 1]
            else:
                raise Exception("Working with only I, RR or LL!")

    def __call__(self, p):
        """
        Returns ln of likelihood for data and model with parameters ``p``.
        :param p:
        :return:
        """
        # Data visibilities and noise
        data = self.uvdata
        error = self.error
        # Model visibilities at uv-points of data
        self.model.p = p
        model_data = self.model.ft(self.uv)
        # ln of data likelihood
        lnlik = -0.5 * np.log(2. * math.pi * error ** 2.) -\
            (data - model_data) * (data - model_data).conj() /\
            (2. * error ** 2.)
        lnlik = lnlik.real
        return lnlik.sum()


class LnPrior(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, p):
        self.model.p = p
        distances = list()
        for component in self.model._components:
            distances.append(component.r)
        if not is_sorted(distances):
            print "Components are not sorted."
            return -np.inf
        else:
            print "Components are sorted. OK!"
        lnpr = list()
        # FIXME: pass only component's own parameters!
        for component in self.model._components:
            print "Passing to component ", component
            print "parameters : ", p[:component.size]
            component.p = p[:component.size]
            p = p[component.size:]
            print "Got lnprior for component : ", component.lnpr
            lnpr.append(component.lnpr)

        return sum(lnpr)


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    and ``kwargs``are also included.
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("vlbi_errors: Exception while calling your prior pdf:")
            print(" params:", x)
            print(" args:", self.args)
            print(" kwargs:", self.kwargs)
            print(" exception:")
            traceback.print_exc()
            raise

if __name__ == "__main__":

    # Load uv-data
    uvdata = open_fits('0642+449.l18.2010_05_21.uvf')
    uv = uvdata.uvw[:, :2]
    mas_to_rad = 4.85 * 10 ** (-9)
    # Create several components
    c1 = DeltaComponent(.3, mas_to_rad*0, 0.)
    c2 = EGComponent(1., mas_to_rad*1.0, 0., mas_to_rad*1., .5, 2.)
    c3 = CGComponent(1., mas_to_rad*1.0, 0.9, mas_to_rad*1.)
    # Create model
    model = Model(stokes='I')
    # Add components to model
    model.add_component(c1)
    model.add_component(c2)
    model.add_component(c3)
    # Create likelihood for data & model
    lnlik = LnLikelihood(uvdata, model)
    p = model.p
    print "Ln of likelihood: "
    lnlik(p)
    # model.uvplot(uv = uv)
    # model.ft(uv=uv)
