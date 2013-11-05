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
        # TODO: should _stokes & _correlations be structured arrays? Model
        # could contain different number of components in different stokes. But
        # it's furier transform MUST contain equal number of visibilities in
        # each stokes. But sometimes not.
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

    def get_uvws(self, data):
        """
        Sets ``_uvws`` attribute of self with values from Data class instance
        ``data``.
        """
        self._uvws = data.uvw

    def ft(self, stoke='I', uvws=None):
        """
        Fourie transform model from image to uv-domain in specified points of
        uv-plane. If no uvw-points are specified, use _uvws attribute. If it is
        None, raise exception.
        """

        if not uvws:
            uvws = self._uvws
            if not uvws.size:
                raise Exception()

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

        visibilities_cc = (flux[indxs_of_cc] * np.exp(2.0 * math.pi * 1j *
            (u[:, np.newaxis] * dx[indxs_of_cc] + v[:, np.newaxis] *
                dy[indxs_of_cc]))).sum(axis=1)

        # TODO: just calculate visibilities_gc with indxs_of_gc. If indxs_of_gc
        # is empty then visibilities_gc will be zeros. So add it!
        if indxs_of_gc.size:
            raise NotImplementedError('Implement FT of gaussians in ft()')
        else:
            visibilities = visibilities_cc

        return visibilities

    @property
    def uv_correlations(self):

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
                raise EmptyImageFtError('Not enough data for RR&LL visibility\
                        calculation')
            self._uv_correlations['RR'] = RR
            self._uv_correlations['LL'] = LL

        elif self._updated['Q'] or self._updated['U']:

            if self._image_stokes['Q'].size and self._image_stokes['U'].size:
                RL = self.ft(stoke='Q') + 1j * self.ft(stoke='U')
                LR = self.ft(stoke='Q') - 1j * self.ft(stoke='U')
                # RL = FT(Q + j*U)
                # LR = FT(Q - j*U)
            else:
                raise EmptyImageFtError('Not enough data for RL&LR visibility\
                        calculation')
            self._uv_correlations['RL'] = RL
            self._uv_correlations['LR'] = LR

        return self._uv_correlations

    def add_from_fits(self, fname, stoke='I'):
        """
        Adds CC from image FITS-file.
        """
        raise NotImplementedError('Implement loading of CC-table of FITS-file')
        self._updated[stoke] = True

    def add_from_txt(self, fname, stoke='I'):
        """
        Adds components of Stokes type ``stoke`` to model from txt-file.
        """
        adds = np.loadtxt(fname)
        dt = self._image_stokes[stoke].dtype
        self._image_stokes[stoke] = adds.ravel().view(dt)
        self._updated[stoke] = True

    def clear_im(self, stoke='I'):
        """
        Clear the model for stoke Stokes parameter.
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
