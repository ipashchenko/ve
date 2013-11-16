#!/usr/bin python2
# -*- coding: utf-8 -*-

import copy
import numpy as np
import pylab as plt
#import new_data as newd
from data_io import AN
from utils import baselines_2_ants


def open_gains(fname, snver=1):
    """
    Helper function for instantiating and loading antenna gains from
    FITS-files.

    Inputs:

        fname [str] - FTIS-file name,
        snver - version of AN-table.

    Outputs:

        instance of Gains class.
    """

    gains = Gains()
    gains.load(fname, snver=snver)

    return gains


class Gains(object):
    """
    Class that represents the set of complex antenna gains.
    """

    def __init__(self, _io=AN()):

        self._io = _io
        self._data = None

    def load(self, fname, snver=1):
        """
        Loads gains from AIPS SN binary table extension of FITS-file.
        Assumes:
            Reference antena is the same
        """

        self._data = self._io.load(fname, snver=snver)

    def save(self, fname, snver=None):
        """
        Saves gains to AIPS SN binary table extension of FITS-file.
        """
        self._io.save(fname)

    def __mul__(self, other):
        """
        Multiply self on another instance of Gains.
        """

        self_copy = copy.deepcopy(self)

        if isinstance(other, Gains):
            for t in set(0.5 * (self._data['start'] + self._data['stop'])):
                # Indexes of all entries of self._data array wich have ``t``
                indxs_self = np.where(0.5 * (self._data['start'] +
                                      self._data['stop']) == t)[0]
                for ant in self._data[indxs_self]['antenna']:
                    # Indexes of self._data array wich have ``t`` and ``ant``
                    indx = np.where((0.5 * (self._data['start'] +
                                    self._data['stop']) == t) &
                                    (self._data['antenna'] == ant))[0]
                    self_copy._data['gains'][indx] =\
                            self._data[indx]['gains'] *\
                            other.find_gains_for_antenna(t, ant)
        else:
            raise Exception('Gains instances can be multiplied only on\
                    instances of Gains class!')

        return self_copy

    def __div__(self, other):
        """
        Divide self on another instance of Gains class.
        """

        self_copy = copy.deepcopy(self)

        if isinstance(other, Gains):
            for t in set(0.5 * (self._data['start'] + self._data['stop'])):
                # Indexes of all entries of self._data array wich have ``t``
                indxs_self = np.where(0.5 * (self._data['start'] +
                                      self._data['stop']) == t)[0]
                for ant in self._data[indxs_self]['antenna']:
                    # Indexes of self._data array wich have ``t`` and ``ant``
                    indx = np.where((0.5 * (self._data['start'] +
                                    self._data['stop']) == t) &
                                    (self._data['antenna'] == ant))[0]
                    self_copy._data['gains'][indx] =\
                            self._data[indx]['gains'] /\
                            other.find_gains_for_antenna(t, ant)
        else:
            raise Exception('Gains instances can be divided only by\
                    instances of Gains class!')

        return self_copy

    def find_gains_for_antenna(self, t, ant):
        """
        Given time ``t`` and antenna ``ant`` this method finds indxs of
        ``gains`` array of ``_data`` structured array of ``self`` for ``ant`` &
        containing ``t`` and returns array of gains for antenna ``ant`` for
        time moment ``t`` with shape (#if, #pol).
        """

        # Find indx of gains entries containing ``t`` & ``ant1``
        indx = np.where((t <= self._data['stop']) & (t >= self._data['start'])
                & (self._data['antenna'] == ant))[0]

        # Shape (#if, #pol)
        gains = np.squeeze(self._data[indx]['gains'])

        return gains

    def find_gains_for_baseline(self, t, bl):
        """
        Given time ``t`` and baseline ``bl`` this method finds indxs of
        ``gains`` array of ``_data`` structured array of ``self``for ``ant1``,
        containing ``t`` and ``ant2``, containing ``t`` and returns array of
        gains (gain1 * gain2^*) for baseline ``bl`` for time moment ``t`` with
        shape (#stokes, #if), where ant1 < ant2.
        """

        ant1, ant2 = baselines_2_ants([bl])

        # Find indx of gains entries containing ``t`` with ant1 & ant2
        indx1 = np.where((t <= self._data['stop']) & (t >= self._data['start'])
                & (self._data['antenna'] == ant1))[0]
        indx2 = np.where((t <= self._data['stop']) & (t >= self._data['start'])
                & (self._data['antenna'] == ant2))[0]

        # Now each gains# has shape (#if, #pol)
        gains1 = np.squeeze(self._data[indx1]['gains'])
        gains2 = np.squeeze(self._data[indx2]['gains'])

        # ``gains12`` has shape (#stokes, #nif)
        gains12 = np.asarray([gains1[:, 0] * np.conjugate(gains2[:, 0]),
                              gains1[:, 1] * np.conjugate(gains2[:, 1]),
                              gains1[:, 0] * np.conjugate(gains2[:, 1]),
                              gains1[:, 1] * np.conjugate(gains2[:, 0])])

        return gains12

    # TODO: convert time to datetime format and use date2num for plotting
    def tplot(self, antenna=None, IF=None, pol=None):
        """
        Method that plots gains for given antennas vs. time.
        """

        if not antenna:
            raise Exception

        if not IF:
            raise Exception('Choose IF # to display: from ' + str(1) + ' to ' +
                             str(np.shape(self._data['gains']))[1])

        if not pol:
            raise Exception('Choose pol. to display: L or R')

        antenna_data = self._data[np.where(self._data['antenna'] == antenna)]
        # TODO: i need function choose parameters
        #smth. like data = self._choose_data(antenna=antenna, IF=IF, pol=None)
        times = 0.5 * (antenna_data['start'] + antenna_data['stop'])

        if pol == 'R':
            data = antenna_data['gains'][:, IF, 0]
        elif pol == 'L':
            data = antenna_data['gains'][:, IF, 1]
        else:
            raise Exception('``pol`` parameter should be ``R`` or ``L``!')

        angles = np.angle(data)
        amplitudes = np.real(np.sqrt(data * np.conj(data)))

        plt.subplot(2, 1, 1)
        plt.plot(times, amplitudes, '.k')
        plt.subplot(2, 1, 2)
        plt.plot(times, angles, '.k')
        plt.show()


class Absorber(object):
    """
    Class that absorbs gains from series of FITS-files into one instance of
    Gains class.
    """

    def __init__(self):

        self._absorbed_gains = Gains()
        self.fnames = list()

    def absorb_one(self, fname):

        gain = Gains()
        gain.load(fname)

        # if no any gains in
        if self._absorbed_gains._data is None:
            self._absorbed_gains = gain
        else:
            self._absorbed_gains = self._absorbed_gains * gain

        self.fnames.append(fname)

    def absorb(self, fnames):

        for fname in fnames:
            self.absorb_one(fname)

    def exclude_one(self, fname):

        if not fname in self.fnames:
            raise Exception('There is no gains absorbed yet!')

        gain = Gains()
        gain.load(fname)
        self.absorbed_gains = self.absorbed_gains / gain
        self.fnames.remove(fname)

    @property
    def absorbed_gains(self):
        return self._absorbed_gains

    @absorbed_gains.setter
    def absorbed_gains(self, val):
        self._absorbed_gains = val

    @property
    def _data(self):
        return self._absorbed_gains._data

    def __mul__(self, data):
        """
        Multiplicate ``data`` instance of ``Data`` class on absorbed gains.
        """
        #if not isinstance(data, newd.Data):
        #    raise Exception
        return data.__mul__(self.absorbed_gains)
