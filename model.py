#!/usr/bin python
# -*- coding: utf-8 -*-

import math
import numpy as np
from utils import EmptyImageFtError


class Model(object):
    """
    Class that represents models used to describe model VLBI-data in both image
    and uv-domain: clean components (delta functions), gaussians etc.
    """

    def __init__(self):

        self._uvws = np.array([], dtype=[('u', float), ('v', float), ('w',
                              float)])
        # TODO: should _stokes & _correlations be recarrays? Model could
        # contain different number of components in different stokes. But it's
        # furier transform MUST contain equal number of visibilities in each
        # stokes. But sometimes not.
        self._image_stokes = {'I': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('pa', float), ('vary', bool, (6,))]),
                              'Q': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('pa', float)]),
                              'U': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('pa', float)]),
                              'V': np.array([], dtype=[('flux', float),
            ('dx', float), ('dy', float), ('bmaj', float), ('bmin',
                float), ('pa', float)])}

        self._updated = {'I': False, 'Q': False, 'U': False, 'V': False}

        self._uv_correlations = {'RR': np.array([], dtype=complex), 'LL':
            np.array([], dtype=complex), 'RL': np.array([], dtype=complex),
            'LR': np.array([], dtype=complex)}

        self._parameters = None

    @property
    def parameters(self):
        """
        Shortcut for acscesing variable parameters.
        """

        for stoke in ['I', 'Q', 'U', 'V']:
            pass

    @parameters.setter
    def parameters(self, p):
        pass

    def get_uvws(self, data):
        """
        Sets ``_uvws`` attribute of self with values from Data class instance
        ``data``.
        """
        self._uvws = data._uvws

    def ft(self, stoke='I', uvws=None):
        """Fourie transform model from image to uv-domain in specified
           points of uv-plane. If no uvw-points are specified, use _uvws
           attribute. If it is None, raise exception.
        """

        if not uvws:
            uvws = self._uvws

        components = self._image_stokes[stoke]

        #TODO: check that component is not empty array. If so =>
        # raise EmptyImageFtException exception

        #flux, dx, dy, maja, mina, pa, ctype = component
        flux = components.flux
        dx = components.dx
        dy = components.dy
        bmaj = components.bmaj
        bmin = components.bmin
        bpa = components.bpa

        # uvw must already be properly scaled
        u = uvws.u
        v = uvws.v
        #w = uvws.w

        indxs_of_cc = np.where(flux != 0 & bmaj == 0 & bmin == 0 & bpa == 0)
        #indxs_of_gc = np.where(flux != 0 & bmaj != 0 & bmin != 0)

        visibilities_cc = (flux[indxs_of_cc] * np.exp(2.0 * math.pi * 1j *
                        (u[:, np.newaxis] * dx[indxs_of_cc] + v[:, np.newaxis] *
                        dy[indxs_of_cc]))).sum(axis=1)

        #TODO: implement it
        visibilities_gc = None
        visibilities_gc_cc = np.concatenate((visibilities_cc, visibilities_gc), axis=1)
        visibilities = np.sum(visibilities_gc_cc, axis=0)

        return visibilities

    @property
    def uv_correlations(self):

        if self._updated['I'] or self._updated['V']:

            if self._image_stokes['I'] and self._image_stokes['V']:
                # RR = FT(I + V)
                # LL = FT(I - V)
                pass
            if not self._image_stokes['V'] and self._image_stokes['I']:
                # RR = FT(I)
                # LL = RR
                pass
            if not self._image_stokes['I'] and self._image_stokes['V']:
                pass
                # RR = FT(V)
                # LL = RR
            else:
                raise EmptyImageFtError('Not enough data for RR&LL visibility calculation')

        elif self._updated['Q'] or self._updated['U']:

            if self._image_stokes['Q'] and self._image_stokes['U']:
                # RL = FT(Q + j*U)
                # LR = FT(Q - j*U)
                pass
            else:
                raise EmptyImageFtError('Not enough data for RL&LR visibility calculation')

        return self._uv_correlations

    def add_from_fits(self, fname, stoke='I'):
        """
        Adds CC from image FITS-file.
        """
        self._updated[stoke] = True

    def add_from_txt(self, fname, stoke='I'):
        """
        Adds components of Stokes type ``stoke`` to model from txt-file.
        """
        adds = np.loadtxt(fname, unpack=True)
        self._image_stokes[stoke].append(adds, axis=1)
        self._updated[stoke] = True

    def clear(self, stoke='I'):
        """
        Clear the model for stoke Stokes parameter.
        """
        self._stokes[stoke] = None
        self._updated[stoke] = False

    def clear_all(self):
        """
        Clear model for all Stokes parameters.
        """
        for stoke in self._stokes.keys():
            self.clear(stoke=stoke)

    def __call__(self, params):
        """
        Return visibilities at self._uvws for model with params.
        """
        pass
