#!/usr/bin python
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
        
        self._recarray = self._fits_format.load(fname, extname='AIPS SN', snver=snver)

    def save(self, fname, snver=None):
        """
        Saves gains to AIPS SN binary table extension of FITS-file.
        """
        self._fits_format.save(fname)

    def __multiply__(self, obj):
        """
        Multiply self on another instance of Gains or Data class.
        """
        
        if isinstance(obj, Data):
            obj.__multiply__(self)
        elif isinstance(obj, Gains):
            pass 
        else:
            raise Exception


class Absorber(object):
    """
    Class that absorbs gains from series of FITS-files into one instance of Gains class.
    """

    def __init__(self):

        self._absorbed_gains = Gains()
        self.files = list()

    def absorb_one(self, fname):

        gain = Gains()
        gain.load(fname)
        self.absorbed_gains *= gain
        self.fnames.append(fname)

    def absorb(self, fnames):

        for fname in fnames:
            self.absorb_one(fname)
            
    def exclude_one(self, fname):
        
        if not fname in self.fnames:
            raise Exception
            
        gain = Gains()
        gain.load(fname)
        self.absorbed_gains /= gain
        self.fnames.delete(fname)
            
    @property
    def absorbed_gains(self):
        return self._absorbed_gains
        
    def __multiply__(self, data):
        if not isinstance(data, Data):
            raise Exception
        data.__multiply__(self.absorbed_gains)
