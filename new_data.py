#!/usr/bin python2
# -*- coding: utf-8 -*-


import numpy as np
from data_io import Groups, IDI
from utils import baselines_2_ants

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

        self._io = io
        self._data = None

    def __add__(self, data):
        """
        Add data to self.

        Input:

            data - instance of Data class. Must have ``_data`` attribute -
            structured numpy.ndarray with the same shape as self.
        """
        # TODO: assert equal dtype and len
        self._data['hands'] = self._data['hands'] + data._data['hands']

    def __multiply__(self, gains):
        """
        Applies complex antenna gains to the visibilities of self.

        Input:

            gains - instance of Gains class.
        """
        pass

    def load(self, fname):
        """
        Method that loads data from FITS-file.

        Inputs:

            fname - file name.
        """
        self._data = self._io.load(fname)
        self.nif = np.shape(self._data['hands'])[1]
        self.nstokes = np.shape(self._data['hands'])[2]

    @property
    def baselines(self):

        return set(self._data['baseline'])

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

    # TODO: add possibility to save in format that is different from current.
    def save(self, fname):
        """
        Save data to FITS-file.

        Inputs:

            fname - file name.
        """

        self._io.save(fname)

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
            shuffled_indxs = np.random.shuffle(indxs)
            # indexes of ``q`` nearly equal chunks
            q_indxs = np.array_split(shuffled_indxs, q)
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
            training_data = np.hstack(training_data)
            testing_data = np.hstack(testing_data)
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

            stokes [str] - any Stokes parameter.
        """

        baselines_cv_scores = list()

        # calculate noise on each baseline
        noise = self.noise()

        model_data = self.substitue(model, stokes=stokes)
        uv_difference = self._data['hands'] - model_data['hands']
        diff_array = self._data.copy()
        diff_array['hands'] = uv_difference

        for baseline in self.baselines:
            # square difference for each baseline, divide by baseline noise
            # and then sum for current baseline
            baseline_indxs = np.where(diff_array['BASELINE'] == baseline)
            hands_diff = diff_array[baseline_indxs]['hands'] / noise[baseline]
            diff = hands_diff.flatten()
            diff *= diff
            baselines_cv_scores.append(diff)

        return sum(baselines_cv_scores)

    def substitute(self, model, baseline=None):
        """
        Substitue data of self with visibilities of the model.
        """

        indxs = np.where(self._data['BASELINE'] == baseline)
        n = len(indxs)
        uvws = self._data[indxs]['uvw']
        model._uvws = uvws

        #self._data['hands'][indxs] =\
        for i, stoke in enumerate(['RR', 'LL', 'RL', 'LR']):
            try:
                self._data['hands'][indxs][..., i] =\
        model.uv_correlations[stoke].repeat(self.nif).reshape((n, self.nif))
            # If model doesn't have some hands => pass it
            except ValueError:
                pass
        #model.uv_correlations returns dictionary of hands. Need extend this array
        #to match the shape of self._data['hands'] (N, #if, #stokes)
