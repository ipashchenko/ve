#!/usr/bin python2
# -*- coding: utf-8 -*-

import numpy as np
from fits_formats import IDI_FITS


class Gains(object):
    """
    Class that represents the set of complex antenna gains.
    """

    def __init__(self, fits_format=IDI_FITS()):

        self._fits_format = fits_format
        self._recarray = None

    def load(self, fname, snver=1):
        """
        Loads gains from AIPS SN binary table extension of FITS-file.
        """

        self.hdu = self._fits_format.get_hdu(fname, 'AIPS SN', version=snver)

        names = self.hdu.columns.names
        list_of_columns = [self.hdu.data.field(column) for column in names]
        recarray = np.rec.fromarrays(list_of_columns,
                dtype=self.hdu.data.dtype)
        #data = recarray['DATA'].squeeze()  #it is a view
        self._recarray = recarray

    def save(self, fname, snver=None):
        """
        Saves gains to AIPS SN binary table extension of FITS-file.
        """
        pass

    def __multiply__(self, gains):
        """
        Multiply gains of self on gains of another instance of Gains class.
        """
        pass


class Absorber(object):
    """
    Class that absorbs gains from series of FITS-files into one instance of Gains class.
    """

    def __init__(self, files):

        self.absorbed_gains = Gains()
        self.files = files

    def absorb_one(self, fname):

        gain = Gains()
        gain.load(fname)
        self.absorbed_gains *= gain

    def absorb(self):

        for fname in self.files:
            gain = Gains()
            gain.load(fname)
            self.absorbed_gains *= gain
