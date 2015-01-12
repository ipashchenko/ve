import math
import numpy as np
from utils import _function_wrapper, mas_to_rad, vcomplex, gaussian

try:
    import pylab
except ImportError:
    pylab = None


class Component(object):
    """
    Basic class that implements single component of model.
    """
    def __init__(self):
        self._p = None
        self._parnames = ['flux', 'x', 'y']
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
        Add component to instances of ``ImagePlane`` subclasses.

        :param image_grid:
            Instance of ``ImagePlane`` subclass.
        """
        raise NotImplementedError("Method must me implemented in subclasses!")

    def uvplot(self, uv, pa=None, style='a&p', sym='.r'):
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
    def __init__(self, flux, x, y, bmaj, e, bpa):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component center [mas].
        :param y:
            Y-coordinate of component center [mas].
        :param bmaj:
            Std of component size [mas].
        :param e:
            Minor-to-major axis ratio.
        :param bpa:
            Positional angle of major axis. Angle counted from x-axis of image
            plane counter clockwise [rad].
        """
        super(EGComponent, self).__init__()
        self._parnames.extend(['bmaj', 'e', 'bpa'])
        self._p = [flux, x, y, bmaj, e, bpa]

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
            flux, x0, y0, bmaj, e, bpa = self.p
        # If we call method inside ``CGComponent``
        except ValueError:
            flux, x0, y0, bmaj = self.p
            e = 1.
            bpa = 0.

        # Convert to radians
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
                                                       b * u * v -
                                                       a * v ** 2.))
        ft = vcomplex(ft)
        # If x0=!0 or y0=!0 then shift phase accordingly
        if x0 or y0:
            ft *= np.exp(-2. * math.pi * 1j * (u * x0 + v * y0))
        return ft

    def add_to_image_grid(self, image_grid):
        """
        Add component to instances of ``ImagePlane`` subclasses.

        :param image_grid:
            Instance of ``ImagePlane`` subclass.
        """
        x, y = image_grid.x, image_grid.y
        dx, dy = image_grid.dx, image_grid.dy
        x_c, y_c = image_grid.x_c, image_grid.y_c

        flux, x_comp, y_comp, bmaj, e, bpa = self.p
        gauss_flux = gaussian(flux, x_comp, y_comp, bmaj, e, bpa)
        image_grid.image_grid += flux


class CGComponent(EGComponent):
    """
    Class that implements circular gaussian component.
    """
    def __init__(self, flux, x, y, bmaj):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component center [mas].
        :param y:
            Y-coordinate of component center [mas].
        :param bmaj:
            Std of component size [mas].
        """
        super(CGComponent, self).__init__(flux, x, y, bmaj, e=1., bpa=0.)
        self._parnames.remove('e')
        self._parnames.remove('bpa')
        self._p = [flux, x, y, bmaj]

    def add_to_image_grid(self, image_grid):
        """
        Add component to instances of ``ImagePlane`` subclasses.

        :param image_grid:
            Instance of ``ImagePlane`` subclass.
        """
        pass


class DeltaComponent(Component):
    """
    Class that implements delta-function component.
    """
    def __init__(self, flux, x, y):
        """
        :param flux:
            Flux of component [Jy].
        :param x:
            X-coordinate of component center [mas].
        :param y:
            Y-coordinate of component center [mas].
        """
        super(DeltaComponent, self).__init__()
        self._p = [flux, x, y]

    def ft(self, uv):
        """
        Return the Fourier Transform of component for given uv-points.
        :param uv:
            2D numpy array of uv-points for which to calculate FT.
        :return:
            Numpy array of complex visibilities for specified points of
            uv-plane. Length of the resulting array = length of ``uv`` array.
        """
        flux, x0, y0 = self.p

        # Convert to radians
        x0 *= mas_to_rad
        y0 *= mas_to_rad

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

        flux, x, y = self.p
        x_coords = int(round(x / dx))
        y_coords = int(round(y / dy))
        # 2 means that x_c & x_coords should be both zero-indexed.
        x = x_c + x_coords - 2
        y = y_c + y_coords - 2
        image_grid.image_grid[x, y] += flux
