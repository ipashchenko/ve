#!/usr/bin python2
# -*- coding: utf-8 -*-

import numpy as np
import pyfits as pf


class Model(object):
    """
    Class that represents models used to describe VLBI-data in both image
    and uv-domain: clean components (delta functions), gaussians etc.
    """

    def __init__(self):

        self._uvws = np.array([], dtype=[('u', float64), ('v', float64), ('w', float64))
        #TODO: should _stokes & _correlations be recarrays? Model could contain different number of components
        # in different stokes. But it's furier transform MUST contain equal
        # number of visibilities in each stokes. But sometimes not.
        self._image_stokes = {'I': np.array([], dtype=[('flux', float64),
            ('dx', float64), ('dy', float64), ('bmaj', float64), ('bmin',
                float64), ('pa', float64)]),
                              'Q': np.array([], dtype=[('flux', float64),
            ('dx', float64), ('dy', float64), ('bmaj', float64), ('bmin',
                float64), ('pa', float64)]),
                              'U': np.array([], dtype=[('flux', float64),
            ('dx', float64), ('dy', float64), ('bmaj', float64), ('bmin',
                float64), ('pa', float64)]),
                              'V': np.array([], dtype=[('flux', float64),
            ('dx', float64), ('dy', float64), ('bmaj', float64), ('bmin',
                float64), ('pa', float64)])}

        self._updated = {'I': False, 'Q': False, 'U': False, 'V': False}

        self._uv_corelations = {'RR': np.array([], dtype=complex64), 'LL':
            np.array([], dtype=complex64), 'RL': np.array([], dtype=complex64),
            'LR': np.array([], dtype=complex64)}

    def ft(self, stoke='I', uvws=None):
    #TODO: how to substitute data to model only on one baseline?
    # uvws must be for that baseline => i need method in Data() to select
    # view of subdata
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
        w = uvws.w

        indxs_of_cc = np.where(flux != 0 && bmaj == 0 && bmin == 0 && bpa == 0)
        indxs_of_gc = np.where(flux != 0 && bmaj != 0 && bmin != 0)

        visibilities_cc = (flux[indxs_of_cc] * np.exp(2.0 * math.pi * 1j *\
        (u[:,newaxis] * dx[indxs_of_cc] + v[:,newaxis] * dy[indxs_of_cc]))).sum(axis=1)

        #TODO: implement it
        visibilities_gc = None
        visibilities_gc_cc = np.concatenate((visibilities_cc, visibilities_gc), axis=1)
        visibilities = np.sum(visibilities_gc_cc, axis=0)

    @property
    def uv_correlations(self):

        if not len(self._uv_correlations['RR']) or self._updated['I'] or self._updated['V']:

            if self._image_stokes['I'] and self._image_stokes['V']:
                #RR = FT(I + V)
            if not self._image_stokes['V'] and self._image_stokes['I']:
                #RR = FT(I)
            if not self._image_stokes['I'] and self._image_stokes['V']:
                #RR = FT(V)
            else:
                raise Exception

        elif not len(self._uv_correlations['LL']) or self._updated['I'] or self._updated['V']:

            if self._image_stokes['I'] and self._image_stokes['V']
                #LL = FT(I - V)
            if not self._image_stokes['V'] and self._image_stokes['I']:
                #LL = RR
            if not self._image_stokes['I'] and self._image_stokes['V']:
                #LL = RR
            else:
                raise Exception

        elif not len(self._uv_correlations['RL']) or self._updated['Q'] or self._updated['U']:

            if self._image_stokes['Q'] and self._image_stokes['U']
                #RL = FT(Q + j*U)
            else:
                raise Exception

        elif not len(self._uv_correlations['LR']) or self._updated['Q'] or self._updated['U']:

            if self._image_stokes['Q'] and self._image_stokes['U']
                #LR = FT(Q - j*U)
            else:
                raise Exception

        return self._uv_correlations

    def add_from_fits(self, fname, stoke='I'):
        """
        Adds CC from image FITS-file.
        """
        pass

    def add_from_txt(self, fname, stoke='I'):
        """
        Adds components of Stokes type ``stoke`` to model from txt-file.
        """
        adds = np.loadtxt(fname, unpack=True)
        self._image_stokes[stoke].append(adds, axis=1)
        self._image_modified[stoke] = True

    def clear(self, stoke='I'):
        """
        Clear the model for stoke Stokes parameter.
        """
        self._stokes[stoke] = None

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


