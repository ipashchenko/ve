#!/usr/bin python2
# -*- coding: utf-8 -*-


import numpy as np


class Data(object):
    """
    Class that represents data in uv-domain.
    """

    def __init__(self, io=None):
        """
        Parameters:

            io - instance of IO subclass

        Initializes:

            _data - container of uv-data. It is numpy atructured array with
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
        self._data += data._data

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

    def save(self, fname):
        """
        Save data to FITS-file.

        Inputs:

            fname - file name.
        """
        #TODO: put data from self.records (recarray) to HDU data
        # if recarray is not a view of HDU.data
        self._io.save(fname)

    def noise(self, split_scans=False, use_V=True):
        """
        Calculate noise for each baseline. If ``split_scans`` is True then
        calculate noise for each scan too. If ``use_V`` is True then use stokes
        V data (`RR`` - ``LL``) for computation. Else use succescive
        differences approach (Brigg's dissertation).

        Input:

            split_scans [bool]
            use_V [bool]

        Output:

            dictionary with keys - baseline numbers, values -
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
                raise NotImplementedError("Implement with split_scans = True")

        return baseline_noises

    #TODO: implement the possibility to choose distribution for each baseline
    # and scan
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
                    noise_to_add = np.random.normal(scale=std, size=n)
                    # FIXME: add to Re and Im!
                    noise_to_add = np.reshape(noise_to_add,
                            np.shape(baseline_data['hands']))
                    baseline_data = baseline_data['hands'] + noise_to_add
                    self._data[np.where(self._data['baseline'] == baseline)] =\
                                                                    baseline_data

            else:
                raise NotImplementedError("Implement with split_scans = True")

        else:
            raise NotImplementedError("Implement with df not None")

    def cv(self, q):
        """
        Method that prepares training and testing samples for q-fold
        cross-validation.

        Inputs:

            q [int] - number of folds.
        """
        pass

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

    #TODO: how to substitute data to model only on one baseline?
    def substitute(self, model, baseline=None):
        """
        Substitue data of self with visibilities of the model.
        """

        indxs = np.where(self._data['BASELINE'] == baseline)
        uvws = self._data[indxs]['uvw']

        self._data['hands'] = model.ft(uvws).broadcast(self._data['hands'])
