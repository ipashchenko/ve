#!/usr/bin python2
# -*- coding: utf-8 -*-

import copy
import numpy as np
try:
    import pylab as plt
except ImportError:
    pylab = None
from utils import baselines_2_ants, get_hdu


def open_gains(fname, snver=1):
    """
    Helper function for instantiating and loading complex antenna gains from
    FITS-files.

    :param fname:
        Path to FTIS-file.

    :param snver (optional):
        Version of SN-table with complex antenna gain information. (default:
        ``1``)

    :return:
        Instance of ``Gains`` class.
    """

    hdu = get_hdu(fname, extname='AIPS SN', ver=snver)

    nif = hdu.header['NO_IF']
    npol = hdu.header['NO_POL']
    nant = hdu.header['NO_ANT']
    # set ``nif'' from dtype of hdu.data
    _data = np.zeros(hdu.header['NAXIS2'], dtype=[('start', '<f8'),
                                                  ('stop', '<f8'),
                                                  ('antenna', 'int'),
                                                  ('gains', 'complex',
                                                   (nif, npol,)),
                                                  ('weights', '<f8',
                                                   (nif, npol,))])

    time = hdu.data['TIME']
    dtime = hdu.data['TIME INTERVAL']
    antenna = hdu.data['ANTENNA NO.']

    # Constructing `gains` field
    rgains = hdu.data['REAL1'] + 1j * hdu.data['IMAG1']
    # => (466, 8)
    lgains = hdu.data['REAL2'] + 1j * hdu.data['IMAG2']
    rgains = np.expand_dims(rgains, axis=2)
    # => (466, 8, 1)
    lgains = np.expand_dims(lgains, axis=2)
    gains = np.dstack((rgains, lgains))
    # => (466, 8, 2)

    # Constructing `weights` field
    rweights = hdu.data['WEIGHT 1']
    # => (466, 8)
    lweights = hdu.data['WEIGHT 2']
    rweights = np.expand_dims(rweights, axis=2)
    # => (466, 8, 1)
    lweights = np.expand_dims(lweights, axis=2)
    weights = np.dstack((rweights, lweights))
    # => (466, 8, 2)

    # Filling structured array by fields
    _data['start'] = time - 0.5 * dtime
    _data['stop'] = time + 0.5 * dtime
    _data['antenna'] = antenna
    _data['gains'] = gains
    _data['weights'] = weights

    gains = list()
    for ant in set(_data['antenna']):
        idx = _data['antenna'] == ant
        gains.append(GainCurve(ant, nif, npol, _data[idx][['start', 'stop',
                                                           'gains',
                                                           'weights']]))
    return gains


# TODO: Gains could be considered as parameters in self-calibration. Connect it
# somehow to ``Model`` via abstract class. Actually, ``GainCurve`` class could
# be part of ``Antenna`` class that keeps all calibration information for single
# antenna in VLBI experiment.
class GainCurve(object):
    """
    Class that represents gain curve for single antenna.
    """
    def __init__(self, ant, nif, npol, data):
        """
        :param ant:
        :param data:
            Numpy structured array.
        :return:
        """
        self.ant = ant
        self.n_if = nif
        self.n_pol = npol
        self._data = data
        self._t = 0.5 * (data['stop'] + data['start'])

    # FIXME: Raise exception if for some times in other instance there's no gains
    def __mul__(self, other):
        """
        Multiply gains of current instance of ``GainCurve`` class to gains
        values of other instance.

        :param other:
            Instance of ``GainCurve``.
        :return:
            Instance of ``GainCurve`` with gains values that are product of
            gains from both instances.

        :note:
            It only has sense to multiply gains of one antenna. If it is not the
            case - exception is raised. Multiplying gains of different antennas
            is implemented in ``UVData.__mul__`` method.
        """
        self_copy = copy.deepcopy(self)

        if not isinstance(other, GainCurve):
            raise Exception
        if not self.ant == other.ant:
            raise Exception
        # We need gains values at this time moments
        t = self._t
        indx = self._get_indx(t)
        self_copy._data['gains'][indx] = self(t) * other(t)

        return self_copy

    def _get_indx(self, t):
        """
        Get indexes of structured array with given times.
        :param t:
            Iterable of time values.
        :return:
            Numpy bool array.
        """
        t = np.array(t)
        a = (t[:, np.newaxis] <= self._data['stop']) & (t[:, np.newaxis] >=
                                                        self._data['start'])
        return np.array([np.where(row)[0][0] for row in a])

    def __call__(self, t):
        """
        Returns values of gains for given times.

        :param t:
            Iterable of time values.
        :return:
            Complex numpy array (#t, #IF, #pol)
        """
        idx = self._get_indx(t)
        if not idx.any():
            gains = np.empty((len(t), self.n_if, self.n_pol))
            gains[:] = np.nan
        else:
            # Shape (#t, #if, #pol)
            gains = self._data[idx]['gains']
        return gains

    def r(self, t):
        return self.__call__(t)[:, :, 0]

    def l(self, t):
        return self.__call__(t)[:, :, 1]

    def tplot(self, IF=None, pol=None):
        """
        Method that plots complex antenna gains vs. time.

        :param IF:
            IF number for which plot gains.

        :param pol:
            Polarization for which plot gains. ``R`` or ``L``.
        """

        if not IF:
            raise Exception('Choose IF # to display: from ' + str(1) + ' to ' +
                            str(np.shape(self._data['gains'])[1]))
        else:
            IF -= 1

        if not pol:
            raise Exception('Choose pol. to display: L or R')

        times = self._t

        if pol == 'R':
            data = self._data['gains'][:, IF, 0]
        elif pol == 'L':
            data = self._data['gains'][:, IF, 1]
        else:
            raise Exception('``pol`` parameter should be ``R`` or ``L``!')

        angles = np.angle(data)
        amplitudes = np.real(np.sqrt(data * np.conj(data)))

        plt.subplot(2, 1, 1)
        plt.plot(times, amplitudes, '.k')
        plt.subplot(2, 1, 2)
        plt.plot(times, angles, '.k')
        plt.show()


class GainsOfExperiment(object):
    def __init__(self):
        self.gains = dict()

    def from_fits_file(self, fname, snver=1):
        gains_list = open_gains(fname, snver=snver)
        for gains in gains_list:
            self.gains.update({gains.ant: gains})

    @property
    def antennas(self):
        return self.gains.keys()

    def find_gains_for_antenna(self, ant, t):
        return self.gains[ant](t)

    def find_gains_for_baseline(self, bl, t):
        ant1, ant2 = baselines_2_ants([bl])
        # (N, #if, #pol)
        gains1 = self.find_gains_for_antenna(ant1, t)
        gains2 = self.find_gains_for_antenna(ant2, t)
        gains12 = np.asarray([gains1[..., 0] * np.conjugate(gains2[..., 0]),
                              gains1[..., 1] * np.conjugate(gains2[..., 1]),
                              gains1[..., 0] * np.conjugate(gains2[..., 1]),
                              gains1[..., 1] * np.conjugate(gains2[..., 0])])
        return gains12


class Gains(object):
    """
    Class that represents complex antenna gains from single VLBI experiment.
    """
    def __init__(self, fname, snver=1):
        hdu = get_hdu(fname, extname='AIPS SN', ver=snver)

        nif = hdu.header['NO_IF']
        npol = hdu.header['NO_POL']
        nant = hdu.header['NO_ANT']
        self.nif = nif
        self.npol = npol
        # set ``nif'' from dtype of hdu.data
        _data = np.zeros(hdu.header['NAXIS2'], dtype=[('start', '<f8'),
                                                      ('stop', '<f8'),
                                                      ('antenna', 'int'),
                                                      ('gains', 'complex',
                                                       (nif, npol,)),
                                                      ('weights', '<f8',
                                                       (nif, npol,))])

        time = hdu.data['TIME']
        dtime = hdu.data['TIME INTERVAL']
        antenna = hdu.data['ANTENNA NO.']

        # Constructing `gains` field
        rgains = hdu.data['REAL1'] + 1j * hdu.data['IMAG1']
        # => (466, 8)
        lgains = hdu.data['REAL2'] + 1j * hdu.data['IMAG2']
        rgains = np.expand_dims(rgains, axis=2)
        # => (466, 8, 1)
        lgains = np.expand_dims(lgains, axis=2)
        gains = np.dstack((rgains, lgains))
        # => (466, 8, 2)

        # Constructing `weights` field
        rweights = hdu.data['WEIGHT 1']
        # => (466, 8)
        lweights = hdu.data['WEIGHT 2']
        rweights = np.expand_dims(rweights, axis=2)
        # => (466, 8, 1)
        lweights = np.expand_dims(lweights, axis=2)
        weights = np.dstack((rweights, lweights))
        # => (466, 8, 2)

        # Filling structured array by fileds
        _data['start'] = time - 0.5 * dtime
        _data['stop'] = time + 0.5 * dtime
        _data['antenna'] = antenna
        _data['gains'] = gains
        _data['weights'] = weights

    @property
    def data(self):
        """
        Shortcut for ``self.data``.
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def gains(self):
        """
        Shortcut for ``self.data['gains']``.
        """
        return self.data['gains']

    @gains.setter
    def gains(self, gains):
        self.data['gains'] = gains

    def load(self, fname, snver=1):
        """
        Method that loads complex antenna gains from ``AIPS SN`` binary table
        extension of FITS-file.

        ..  warning:: Current implementation assumes that reference antena is
        the same
        """
        self._data = self._io.load(fname, snver=snver)

    def save(self, fname, snver=None):
        """
        Method that saves complex antenna gains to ``AIPS SN`` binary table
        extension of FITS-file.

        :param fname:
            Path to FITS-file.

        :param snver:
            Version of SN-table with complex antenna gain information.
        """
        self._io.save(fname)

    def __mul__(self, other):
        """
        Method that multiplies complex antenna gains of ``self`` on gains of
        another instance of ``Gains`` class. ``Self`` must have more time
        steps then ``other``.

        :param other:
            Instance of ``Gains`` class.

        :return:
            Instance of ``Gains`` class. Contains product of two gain sets.

        """
        self_copy = copy.deepcopy(self)

        if isinstance(other, Gains):
            for t in set(0.5 * (self._data['start'] + self._data['stop'])):
                # Indexes of all entries of self._data array wich have ``t``
                indxs_self = np.where(0.5 * (self._data['start'] +
                                      self._data['stop']) == t)[0]
                for ant in self._data[indxs_self]['antenna']:
                    # Indexes of self._data array which have ``t`` and ``ant``
                    indx = np.where((0.5 * (self._data['start'] +
                                    self._data['stop']) == t) &
                                    (self._data['antenna'] == ant))[0]
                    print (indx)
                    print (np.shape(self._data[indx]['gains'] ))
                    print (np.shape(other.find_gains_for_antenna(t, ant)))
                    self_copy._data['gains'][indx] =\
                            self._data[indx]['gains'] *\
                            other.find_gains_for_antenna(t, ant)
                    if other.find_gains_for_antenna(t, ant) is None:
                        print self_copy._data['gains'][indx]
        else:
            raise Exception('Gains instances can be multiplied only on'
                            'instances of Gains class!')

        return self_copy

    def __div__(self, other):
        """
        Method that divides complex antenna gains of ``self`` to gains of
        another instance of ``Gains`` class.

        :param other:
            Instance of ``Gains`` class.

        :return:
            Instance of ``Gains`` class. Contains division of two gain sets.
        """

        self_copy = copy.deepcopy(self)

        if isinstance(other, Gains):
            for t in set(0.5 * (self._data['start'] + self._data['stop'])):
                # Indexes of all entries of self._data array wich have ``t``
                indxs_self = np.where(0.5 * (self._data['start'] +
                                      self._data['stop']) == t)[0]
                for ant in self._data[indxs_self]['antenna']:
                    # Indexes of self._data array which have ``t`` and ``ant``
                    indx = np.where((0.5 * (self._data['start'] +
                                    self._data['stop']) == t) &
                                    (self._data['antenna'] == ant))[0]
                    self_copy._data['gains'][indx] =\
                            self._data[indx]['gains'] /\
                            other.find_gains_for_antenna(t, ant)
        else:
            raise Exception('Gains instances can be divided only by '
                            'instances of Gains class!')

        return self_copy

    def find_gains_for_antenna(self, t, ant):
        """
        Method that find complex antenna gains for given time and antenna.

        Given time ``t`` and antenna ``ant`` this method finds indxs of
        ``gains`` array of ``_data`` structured numpy.ndarray of ``self`` for
        ``ant`` & containing ``t`` and returns array of gains for antenna
        ``ant`` for time moment ``t`` with shape (#if, #pol).

        :param t:
            Time for which fetch gains.

        :param ant:
            Antenna for which fetch gains.

        :return:
            Numpy.ndarray.
        """

        # Find indx of gains entries containing ``t`` & ``ant1``
        indx = np.where((t <= self._data['stop']) & (t >= self._data['start'])
                & (self._data['antenna'] == ant))[0]

        if not indx:
            print "No gains for time and antenna: " + str(t) + ' ' + str(ant)
            gains = np.empty(np.shape(self.gains[0]))
            gains[:] = np.nan
            print "Returning"
            print gains
        else:
            # Shape (#if, #pol)
            gains = np.squeeze(self._data[indx]['gains'])

        return gains

    def find_gains_for_baseline(self, t, bl):
        """
        Method that find complex antenna gains for given time and baseline.

        Given time ``t`` and baseline ``bl`` this method finds indxs of
        ``gains`` array of ``_data`` structured array of ``self``for ``ant1``,
        containing ``t`` and ``ant2``, containing ``t`` and returns array of
        gains (gain1 * gain2^*) for baseline ``bl`` for time moment ``t`` with
        shape (#stokes, #if), where ant1 < ant2.

        :param t:
            Time for which fetch gains.

        :param bl:
            Baseline for which fetch gains.

        :return:
            Numpy.ndarray.
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
        Method that plots complex antenna gains for given antennas vs. time.

        :param antenna:
            Antenna for which plot gains.

        :param IF:
            IF number for which plot gains.

        :param pol:
            Polarization for which plot gains. ``R`` or ``L``.
        """

        if not antenna:
            raise Exception('Choose antenna number to display!')

        if not IF:
            raise Exception('Choose IF # to display: from ' + str(1) + ' to ' +
                             str(np.shape(self._data['gains'])[1]))
        else:
            IF -= 1

        if not pol:
            raise Exception('Choose pol. to display: L or R')

        antenna_data = self._data[np.where(self._data['antenna'] == antenna)]
        # TODO: i need function to choose parameters
        #smth. like data = self._choose_uvdata(antenna=antenna, IF=IF, pol=None)
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
    ``Gains`` class.
    """

    def __init__(self):

        self._absorbed_gains = Gains()
        self.fnames = list()

    def absorb_one(self, fname, snver=1):
        """
        Method that absorbs complex antenna gains from specified FITS-file.

        :param fname:
            Path to FITS-file.
        """

        gain = Gains()
        gain.load(fname, snver=snver)

        # if no any gains in
        if self._absorbed_gains._data is None:
            self._absorbed_gains = gain
        else:
            self._absorbed_gains = self._absorbed_gains * gain

        self.fnames.append(fname)

    def absorb(self, fnames):
        """
        Method that absorbes complex antenna gains from specified set of
        FITS-file.

        :param fnames:
            Iterable of paths to FITS-files.
        """

        for fname in fnames:
            try:
                self.absorb_one(fname)
            except:
                print "Failed to read in gains"
                print fname

    def exclude_one(self, fname):
        """
        Method that excludes complex antenna gains of specified FITS-file from
        absorbed gains.

        :param fname:
            Path to FITS-file.
        """

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
        Multiplicate ``data`` instance of ``UVData`` class on absorbed gains.
        """
        #if not isinstance(data, newd.UVData):
        #    raise Exception
        return data.__mul__(self.absorbed_gains)


if __name__ == '__main__':

    gains = Absorber()
    gains.absorb_one('/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS',
                     snver=2)
    import glob
    fnames = glob.glob('/home/ilya/work/vlbi_errors/fits/12*CALIB*FITS')
    fnames.remove('/home/ilya/work/vlbi_errors/fits/1226+023_CALIB_SEQ10.FITS')
    fnames.sort(reverse=True)
    gains.absorb(fnames)
