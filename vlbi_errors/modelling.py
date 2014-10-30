#!/usr/bin python
# -*- coding: utf-8 -*-

import math
import numpy as np
try:
    import pylab
except ImportError:
    pylab = None

vcomplex = np.vectorize(complex)


class GCModel(object):
    """
    Class that implements model consisting of several components.
    """
    def __init__(self, stokes=None):
        self._components = list()
        self._p = None
        self._uv = None
        self.stokes = stokes

    def __add__(self, other):
        self._components.extend(other._components)

    def add_component(self, component):
        self._components.append(component)

    def remove_component(self, component):
        self._components.remove(component)

    def clear(self):
        self._components =list()

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
        p_ = self.p
        for component in self._components:
            component.p = p_[:component.size]
            p_ = p_[component.size:]


class Component(object):
    """
    Basic class that implements single component of model.
    """
    def __init__(self):
        self._p = None

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


class EGComponent(Component):
    """
    Class that implements elliptical gaussian component.
    """
    def __init__(self, flux, r, theta, bmaj, e, bpa):
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

            2*pi*A*(4*a*c-b**2)**(-1/2)*exp[(4*pi**2/(4*a*c-b**2))*(-c*u**2+b*u*v-a*v**2)] (2)

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
