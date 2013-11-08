#!/usr/bin python2
# -*- coding: utf-8 -*-

import copy
import math
import numpy as np
import pylab as plt
from data_io import Groups, IDI
from utils import baselines_2_ants
from utils import index_of
#import gains as g

vec_complex = np.vectorize(np.complex)

# TODO: add possibility to input/output in different FITS-formats (and other
# formats too).


def open_fits(fname, structure='UV'):
    """
    Helper function for instantiating and loading FITS-files.

    Inputs:

        fname [str] - FTIS-file name,

        structure [str] ['UV','IDI'] - structure of FITS-file.
    """

    structures = {'UV': Groups(), 'IDI': IDI()}
    data = Data(io=structures[structure])
    data.load(fname)

    return data


class Data(object):
    """
    Class that represents data in uv-domain.
    """

    def __init__(self, io=None):
        """
        Parameters:

            io - instance of IO subclass

        Initializes:

            _data - container of uv-data. It is numpy structured array with
                dtype=[('uvw', '<f8', (3,)),
                      ('time', '<f8'), ('baseline', 'int'),
                      ('hands', 'complex', (nif, nstokes,)),
                      ('weights', '<f8', (nif, nstokes,))]
        """

        self._stokes_dict = {'RR': 0, 'LL': 1, 'RL': 2, 'LR': 3}
        self._io = io
        self._data = None

    def __add__(self, other):
        """
        Add to self another instance of Data.

        Input:

            data - instance of Data class. Must have ``_data`` attribute -
            structured numpy.ndarray with the same shape as self.
        """

        self_copy = copy.deepcopy(self)
        # TODO: assert equal dtype and len
        self_copy._data['hands'] = self._data['hands'] + other._data['hands']

        return self_copy

    def __sub__(self, other):
        """
        Substruct from self another instance of Data.

        Input:

            data - instance of Data class. Must have ``_data`` attribute -
            structured numpy.ndarray with the same shape as self.
        """

        print "substracting " + str(other) + " from " + str(self)
        self_copy = copy.deepcopy(self)
        # TODO: assert equal dtype and len
        self_copy._data['hands'] = self._data['hands'] - other._data['hands']
        print self_copy._data['hands']

        return self_copy

    # TODO: Do i need the possibility of multiplying on any complex number?
    def __mul__(self, gains):
        """
        Applies complex antenna gains to the visibilities of self.

        Input:

            gains - instance of Gains class.
        """

        self_copy = copy.deepcopy(self)

        # FIXME: even if gains is instance of Gains the exception is raised
        #if not isinstance(gains, g.Gains):
        #    raise Exception('Instances of Data can be multiplied only on\
        #            instances of Gains!')

        # Assert equal number of IFs
        assert(self.nif == np.shape(gains._data['gains'])[1])
        # TODO: Now we need this to calculating gain * gains*. But try to
        # exclude this assertion
        assert(self.nstokes == 4)

        for t in set(self._data['time']):

            # Find all uv-data entries with time t:
            uv_indxs = np.where(self._data['time'] == t)[0]

            # Loop through uv_indxs (different baselines with the same ``t``)
            # and multipy visibility with baseline to gain(ant1)*gain(ant2)^*
            # for ant1 & ant2 derived for this baseline.
            for uv_indx in uv_indxs:
                bl = self._data['baseline'][uv_indx]
                gains12 = gains.find_gains_for_baseline(t, bl)
                # FIXME: In substitute() ['hands'] then [indxs] does return
                # view.
                self_copy._data[uv_indx]['hands'] *= gains12.T

        return self_copy

    def zero_data(self):
        """
        Method that zeros all data.
        """

        self._data['hands'] = np.zeros(np.shape(self._data['hands']),
                              dtype=self._data['hands'].dtype)

    def load(self, fname):
        """
        Method that loads data from FITS-file.

        Inputs:

            fname - file name.
        """
        self._data = self._io.load(fname)
        self.nif = np.shape(self._data['hands'])[1]
        self.nstokes = np.shape(self._data['hands'])[2]

    # TODO: add possibility to save in format that is different from current.
    # TODO: i need indexes of saving data in original data array!
    def save(self, data, fname):
        """
        Save ``_data`` attribute of self to FITS-file.

        Inputs:

            fname - file name.
        """

        self._io.save(data, fname)

    # TODO: make possible to choose all IFs and stokes?
    # TODO: should it be a general method for choosing subsample of structured
    # array using kwargs for parameters and kwargs with dictionary values for
    # specifying dimensions of arrays in strutured array. I need it in Gains
    # class too.
    def _choose_data(self, baselines=None, IF=None, stokes=None):
        """
        Method that returns chosen data from _date structured array based on
        user specified parameters. All checks on IF and stokes are made here
        in one place. This method used by plotting methods.

        Inputs:

            baselines - one or iterable of baselines numbers,

            IF - one or iterable of IF numbers (1-#IF),

            stokes - string - any of: I, Q, U, V, RR, LL, RL, LR.

        Outputs:

            numpy.ndarray, numpy.ndarray

                where first array is array of data with shape (#N, #IF,) and second
                array is 1d-array of indexes of data in self._data structured
                array.
        """

        data = self._data

        # TODO: create generl method for retrieving indxs of structured array
        # where specified field equals to specified values
        if baselines is None:
            baselines = self.baselines
            indxs = np.arange(len(data))
        else:
            indxs = list()
            baselines_list = list()
            # If ``baselines`` is iterable
            try:
                baselines_list.extend(baselines)
            # If ``baselines`` is not iterable (int)
            except TypeError:
                baselines_list.append(baselines)

            # Check that given baseline numbers are among existing ones
            assert(set(baselines_list).issubset(self.baselines))

            # Find indexes of structured array (in zero dim) with given
            # baselines
            for baseline in baselines_list:
                indx = np.where(data['baseline'] == baseline)[0]
                indxs.extend(indx)
            indxs = np.array(np.sort(indxs))

        print "INDXS : "
        print indxs

        if IF is None:
            IF = np.arange(self.nif) + 1
        else:
            IF_list = list()
            # If ``IF`` is iterable
            try:
                IF_list.extend(IF)
            # If ``IF`` is not iterable (int)
            except TypeError:
                IF_list.append(IF)
            IF = np.array(IF_list)

        if not set(IF).issubset(np.arange(1, self.nif + 1)):
            raise Exception('Choose IF numbers to display: from ' + str(1) +
                    ' to ' + str(self.nif))
        IF -= 1
        print "IF : "
        print IF

        if stokes == 'I':
            # I = 0.5 * (RR + LL)
            result = 0.5 * (data['hands'][indxs[:, None, None], IF[:, None], 0]\
                    + data['hands'][indxs[:, None, None], IF[:, None], 1])

        elif stokes == 'V':
            # V = 0.5 * (RR - LL)
            result = 0.5 * (data['hands'][indxs[:, None, None], IF[:, None], 0]\
                    - data['hands'][indxs[:, None, None], IF[:, None], 1])

        elif stokes == 'Q':
            # V = 0.5 * (LR + RL)
            result = 0.5 * (data['hands'][indxs[:, None, None], IF[:, None], 3]\
                    + data['hands'][indxs[:, None, None], IF[:, None], 2])

        elif stokes == 'U':
            # V = 0.5 * 1j * (LR - RL)
            result = 0.5 * 1j * (data['hands'][indxs[:, None, None], IF[:,
                None], 3] - data['hands'][indxs[:, None, None], IF[:, None],
                    2])

        elif stokes in self._stokes_dict.keys():
            result = data['hands'][indxs[:, None, None], IF[:, None],
                    self._stokes_dict[stokes]]

        else:
            raise Exception('Allowed stokes parameters: I, Q, U, V, RR, LL, RL,\
                    LR')

        return result, indxs

    # TODO: convert time to datetime format and use date2num for plotting
    # TODO: make a kwarg argument - to plot in different symbols/colors
    def tplot(self, baselines=None, IF=None, stokes=None):
        """
        Method that plots uv-data for given baselines vs. time.

        Inputs:

            baselines - one or iterable of baselines numbers,

            IF - one or iterable of IF numbers (1-#IF),

            stokes - string - any of: I, Q, U, V, RR, LL, RL, LR.
        """

        # All checks are in self._choose_data()

        if not stokes:
            stokes = 'I'

        data, indxs = self._choose_data(baselines=baselines, IF=IF,
                                        stokes=stokes)
        # # of chosen IFs
        n_if = np.shape(data)[1]

        # TODO: define colors
        try:
            syms = self.__color_list[:n_if]
        except AttributeError:
            print "Define self.__color_list to show in different colors!"
            syms = ['.k'] * n_if

        # TODO: i need fuction to choose parameters
        times = self._data[indxs]['time']
        angles = np.angle(data)
        amplitudes = np.real(np.sqrt(data * np.conj(data)))

        plt.subplot(2, 1, 1)
        for _if in range(n_if):
            # TODO: plot in different colors and make a legend
            plt.plot(times, amplitudes[:, _if], syms[_if])
        plt.subplot(2, 1, 2)
        for _if in range(n_if):
            plt.plot(times, angles[:, _if], syms[_if])
            plt.ylim([-math.pi, math.pi])
        plt.show()

    def uvplot(self, baselines=None, IF=None, stokes=None):
        """
        Method that plots uv-data for given baseline vs. uv-radius.
        """

        # All checks are in self._choose_data()

        if not stokes:
            stokes = 'I'

        data, indxs = self._choose_data(baselines=baselines, IF=IF,
                                        stokes=stokes)

        # # of chosen IFs
        n_if = np.shape(data)[1]

        # TODO: define colors
        try:
            syms = self.__color_list[:n_if]
        except AttributeError:
            print "Define self.__color_list to show in different colors!"
            syms = ['.k'] * n_if

        # TODO: i need fnction choose parameters
        uvw_data = self._data[indxs]['uvw']
        uv_radius = np.sqrt(uvw_data[:, 0] ** 2 + uvw_data[:, 1] ** 2)

        angles = np.angle(data)
        amplitudes = np.real(np.sqrt(data * np.conj(data)))

        plt.subplot(2, 1, 1)
        for _if in range(n_if):
            # TODO: plot in different colors and make a legend
            plt.plot(uv_radius, amplitudes[:, _if], syms[_if])
        plt.subplot(2, 1, 2)
        for _if in range(n_if):
            plt.plot(uv_radius, angles[:, _if], syms[_if])
            plt.ylim([-math.pi, math.pi])
        plt.show()

    @property
    def baselines(self):

        return list(set(self._data['baseline']))

    @property
    def antennas(self):
        """
        Returns list of antenna numbers.
        """

        return baselines_2_ants(self.baselines)

    @property
    def uvw(self):
        """
        Shortcut for unique (u,v,w)-elements.

        Output:

            structured numpy.ndarry with fields ``u``, ``v`` & ``w``.
        """

        x = self._data['uvw']
        dt = np.dtype([('u', x.dtype), ('v', x.dtype), ('w', x.dtype)])
        result, idx, inv = np.unique(x.ravel().view(dt), return_index=True,
                return_inverse=True)

        return result

    def noise(self, split_scans=False, use_V=True):
        """
        Calculate noise for each baseline. If ``split_scans`` is True then
        calculate noise for each scan too. If ``use_V`` is True then use stokes
        V data (`RR`` - ``LL``) for computation assuming no signal in V. Else
        use succescive differences approach (Brigg's dissertation).

        Input:

            split_scans [bool]
            use_V [bool]

        Output:

            dictionary with keys - baseline numbers, values - array of noise
                std for each IF (if ``use_V``==True), or array with shape
                (4, #if) with noise std values for each if for each hand
                (RR, LL, ...).
        """

        baseline_noises = dict()
        if use_V:
            # Calculate dictionary {baseline: noise} (if split_scans is False)
            # or {baseline: [noises]} if split_scans is True.
            if not split_scans:
                for baseline in self.baselines:
                    # TODO: use extended ``choose_data`` method?
                    baseline_data = self._data[np.where(self._data['baseline']
                        == baseline)]
                    baseline_noises[baseline] =\
                    np.std(((baseline_data['hands'][..., 0] -
                        baseline_data['hands'][..., 1])).real, axis=0)
            else:
                # Use each scan
                raise NotImplementedError("Implement with split_scans = True")

        else:
            if not split_scans:
                for baseline in self.baselines:
                    # TODO: use extended ``choose_data`` method?
                    baseline_data = self._data[np.where(self._data['baseline']
                        == baseline)]
                    differences = baseline_data['hands'][:-1, ...] -\
                                baseline_data['hands'][1:, ...]
                    baseline_noises[baseline] =\
                        np.asarray([np.std((differences).real[..., i], axis=0)
                            for i in range(self.nstokes)])
            else:
                # Use each scan
                raise NotImplementedError("Implement with split_scans = True")

        return baseline_noises

    def noise_add(self, noise=None, df=None, split_scans=False):
        """
        Add standard gaussian noise with ``noise`` - mapping from baseline
        number to std of noise or to iterables of stds (if ``split_scans`` is
        True).  If df is not None, then use t-distribtion with ``df`` d.o.f.

        Inputs:

            noise - mapping from baseline number to std of noise or to
                iterables of stds (if ``split_scans``  is set to True).

            df - # of d.o.f. for standard Student t-distribution.

            split_scans [bool]
        """

        if not df:
            if not split_scans:
                for baseline, std in noise.items():
                    # TODO: use extended ``choose_data`` method?
                    baseline_data = self._data[np.where(self._data['baseline']
                        == baseline)]
                    n = np.prod(np.shape(baseline_data['hands']))
                    noise_to_add = vec_complex(np.random.normal(scale=std,
                        size=n), np.random.normal(scale=std, size=n))
                    noise_to_add = np.reshape(noise_to_add,
                            np.shape(baseline_data['hands']))
                    baseline_data['hands'] = baseline_data['hands'] +\
                                             noise_to_add
                    self._data[np.where(self._data['baseline'] == baseline)] =\
                        baseline_data

            else:
                # Use each scan
                raise NotImplementedError("Implement with split_scans = True")

        else:
            # Use t-distribution
            raise NotImplementedError("Implement with df not None")

    # TODO: TEST ME!!!
    def cv(self, q, fname):
        """
        Method that prepares training and testing samples for q-fold
        cross-validation.

        Inputs:

            q [int] - number of folds.

            fname - base name for output the results.

        Outputs:
            ``q`` pairs of files (format that of ``io``) with training and
            testing samples prepaired such that 1/``q``- part of visibilities
            from each baseline falls in testing sample and other part falls in
            training sample.
        """

        # List of lists of ``q`` blocks of each baseline
        baselines_chunks = list()

        # split data of each baseline to ``q`` blocks
        for baseline in self.baselines:
            baseline_data = self._data[np.where(self._data['baseline']
                == baseline)]
            blen = len(baseline_data)
            indxs = np.arange(blen)
            # Shuffle indexes
            np.random.shuffle(indxs)
            # indexes of ``q`` nearly equal chunks
            q_indxs = np.array_split(indxs, q)
            # ``q`` blocks for current baseline
            baseline_chunks = [baseline_data[indx] for indx in q_indxs]
            baselines_chunks.append(baseline_chunks)

        # Combine ``q`` chunks to ``q`` pairs of training & testing datasets.
        for i in range(q):
            # List of i-th chunk for testing dataset for each baseline
            testing_data = [baseline_chunks[i] for baseline_chunks in
                            baselines_chunks]
            # List of "all - i-th" chunk as training dataset for each baseline
            training_data = [baseline_chunks[:i] + baseline_chunks[i + 1:] for
                             baseline_chunks in baselines_chunks]

            # Combain testing & training samples of each baseline in one
            testing_data = np.hstack(testing_data)
            training_data = np.hstack(sum(training_data, []))
            # Save each pair of datasets to files
            # NAXIS changed!!!
            self.save(training_data, 'train' + '_' + str(i) + 'of' + str(q))
            self.save(testing_data, 'test' + '_' + str(i) + 'of' + str(q))

    def cv_score(self, model, stokes='I'):
        """
        Returns Cross-Validation score for self (as testing cv-sample) and
        model (trained on training cv-sample).

        Inputs:

            model [instance of Model class] - model to cross-validate.

            stokes [str] - string of any of stokes parameters - ['I', 'Q',
                'U', 'V'], eg. 'I', 'QU'...
        """

        baselines_cv_scores = list()

        # Calculate noise on each baseline
        # ``noise`` is dictionary with keys - baseline numbers and values -
        # numpy arrays of noise std for each IF
        noise = self.noise()
        print "Calculating noise..."
        print noise

        data_copied = copy.deepcopy(self)
        data_copied.substitute(model)
        # TODO: use __sub__() method of data
        uv_difference = self._data['hands'] - data_copied._data['hands']
        diff_array = copy.deepcopy(self._data)
        diff_array['hands'] = uv_difference

        for baseline in self.baselines:
            # square difference for each baseline, divide by baseline noise
            # and then sum for current baseline
            baseline_indxs = np.where(diff_array['baseline'] == baseline)[0]
            hands_diff = diff_array[baseline_indxs]['hands'] /\
                         noise[baseline][None, :, None]
            # Construct difference for all Stokes parameters
            diffs = dict()
            diffs.update({'I': 0.5 * (hands_diff[..., 0] + hands_diff[..., 1])})
            diffs.update({'V': 0.5 * (hands_diff[..., 0] - hands_diff[..., 1])})
            diffs.update({'Q': 0.5 * (hands_diff[..., 2] + hands_diff[..., 3])})
            diffs.update({'U': 0.5 * 1j * (hands_diff[..., 3] - hands_diff[...,
                2])})
            for stoke in stokes:
                if stoke not in 'IQUV':
                    raise Exception('Stokes parameter must be in ``IQUV``!')
                result = 0
                diff = diffs[stoke].flatten()
                diff = diff * np.conjugate(diff)
                result += float(diff.sum())
                print "cv score for stoke " + stoke + " is : " + str(result)
            baselines_cv_scores.append(result)

        return sum(baselines_cv_scores)

    def substitute(self, model, baseline=None):
        """
        Substitue data of self with visibilities of the model.
        """

        if baseline is None:
            baseline = self.baselines
        indxs = index_of(baseline, self._data['baseline'])
        n = len(indxs)
        uvws = self._data[indxs]['uvw']
        model._uvws = uvws

        for i, stoke in enumerate(['RR', 'LL', 'RL', 'LR']):
            try:
                self._data['hands'][indxs, :, i] =\
        model.uv_correlations[stoke].repeat(self.nif).reshape((n, self.nif))
            # If model doesn't have some hands => pass it
            except ValueError:
                pass
