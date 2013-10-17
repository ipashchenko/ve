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
                      ('hands', 'complex', (nstokes, nif,)),
                      ('weights', '<f8', (nstokes, nif,))]
        """

        self._io = io
        self._data = None

    def __add__(self, data):
        """
        Add data to self.

        Input:

            data - instance of Data class.
        """
        pass

    def __multiply__(self, gains):
        """
        Applies complex antenna gains  to the visibilities of self.

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

    def save(self, fname):
        """
        Save data to FITS-file.

        Inputs:

            fname - file name.
        """
        #TODO: put data from self.records (recarray) to HDU data
        # if recarray is not a view of HDU.data
        self._io.save(fname)

    def noise(self, split_scans=False):
        """
        Calculate noise for each baseline. If ``split_scans`` is True then
        calculate noise for each scan too.

        Inputs:

            split_scans [bool]
        """
        pass

    #TODO: implement the possibility to choose distribution for each baseline
    # and scan
    def noise_add(self, stds=None, df=None, split_scans=False):
        """
        Add standard gaussian noise with ``stds`` - mapping from baseline
        number to std of noise or to iterables of stds (if ``split_scans`` is True).
        If df is not None, then use t-distribtion with ``df`` d.o.f.

        Inputs:

            stds - mapping from baseline number to std of noise or to
                iterables of stds (if ``split_scans``  is set to True).

            df - # of d.o.f. for standard Student t-distribution.

            split_scans [bool]
        """
        pass

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
