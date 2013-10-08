#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
from fits_formats import Groups


# TODO: Never change number of baselines, antennas! It must be done in
# AIPS. Work only with existing data.


class Data(object):
    """
    Represent VLBI observational data.
    """

    def __init__(self, fits_format=Groups()):
        """
        format - instance of UV_FITS or FITS_IDI class used for io.
        """
        self._fits_format = fits_format
        self._hands = {'RR': np.array([], dtype=complex), 'LL': np.array([],
            dtype=complex), 'RL': np.array([], dtype=complex), 'LR':
            np.array([], dtype=complex)}

    def load(self, fname):
        """
        Load data from FITS-file.
        """
        self._data = self._fits_format.load(fname)

    def _data_recarray_to_hands(self):
        """
        Transfer data part of HDU recarray (N, #if [, #chan], #hands, #complex)
        to hands (RR, LL, RL, LR) with structure complex(N, #if [, #chan]).
        """

        vcomplex = np.vectorize(np.complex)

        self._hands['RR'] = vcomplex(self._data.squeeze()[..., 0, 0],
                self._data.squeeze()[..., 0, 1])
        self._hands['LL'] = vcomplex(self._data.squeeze()[..., 1, 0],
                self._data.squeeze()[..., 1, 1])
        self._hands['RL'] = vcomplex(self._data.squeeze()[..., 2, 0],
                self._data.squeeze()[..., 2, 1])
        self._hands['LR'] = vcomplex(self._data.squeeze()[..., 3, 0],
                self._data.squeeze()[..., 3, 1])

    def _hands_to_data_recarray(self):
        """
        Transfer hands (RR, LL, RL, LR) with structure complex(N, #if, #chan)
        to data part of HDU recarray (N, #if [, #chan], #hands, #complex).
        """

        self._data.squeeze()[..., 0, 0] = self._hands['RR'].real
        self._data.squeeze()[..., 0, 1] = self._hands['RR'].imag

        self._data.squeeze()[..., 1, 0] = self._hands['LL'].real
        self._data.squeeze()[..., 1, 1] = self._hands['LL'].imag

        self._data.squeeze()[..., 2, 0] = self._hands['RL'].real
        self._data.squeeze()[..., 2, 1] = self._hands['RL'].imag

        self._data.squeeze()[..., 3, 0] = self._hands['LR'].real
        self._data.squeeze()[..., 3, 1] = self._hands['LR'].imag

    def save(self, fname):
        """
        Save data to FITS-file. Data in ``hands`` attribute is transformed to
        ``data`` attribute which is the view of data part of HDU with
        visibilities. This HDU is the member of HDUList instance method
        ``writeto`` of which we are calling.
        """

        self._hands_to_data_recarray()
        self._fits_format.save(self._data, fname)

    def cross_validation(self, frac=0.1, n=100, outname=None):
        """
        Generate training and testing samples as FITS-files thats inherit
        FITS-structure of the initial data.  Save training and testing samples
        to FITS-files.
        """

        self._fits_format.cross_validation(frac=frac, n=n)
