#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import pyfits as pf
from unittest import TestCase, skip
from vlbi_errors.utils import (aips_bintable_fortran_fields_to_dtype_conversion,
                               index_of)
from vlbi_errors.data_io import IO, PyFitsIO
from vlbi_errors.data_io import Groups
from vlbi_errors.uv_data import Data, open_fits


class Test_utils(TestCase):
    def test_aips_bintable_fortran_fields_to_dtype_conversion(self):

        self.assertEqual(aips_bintable_fortran_fields_to_dtype_conversion(
                        '4J'), ('>i4', 4))
        self.assertEqual(aips_bintable_fortran_fields_to_dtype_conversion(
                        'E(4,32)'), ('>f4', (4, 32)))

    def test_index_of(self):
        self.assertEqual(index_of(np.array([1, 2, 3]),
                                      np.array([1, 63, 2, 2, 4]),
                                      issubset=False)[0], np.array([0]))
        self.assertListEqual(list(index_of(np.array([1, 2, 3]),
                                           np.array([1, 63, 2, 2, 4]),
                                           issubset=False)[1]),
                             list(np.array([2, 3])))
        self.assertIsNone(index_of(np.array([1, 2, 3]),
                                   np.array([1, 63, 2, 2, 4]),
                                   issubset=False)[2])
        with self.assertRaises(AssertionError):
            index_of(np.array([1, 2, 3]), np.array([1, 63, 2, 2, 4]))

    #def test_change_shape(self):
    #    data = pf.open('PRELAST_CALIB')[0]


class Test_IO(TestCase):
    def setUp(self):
        self.io = IO()

    def test_IO(self):
        with self.assertRaises(NotImplementedError) as context:
            self.io.load()
            self.io.save()


class Test_PyFitsIO(TestCase):
    def setUp(self):
        self.groups_uv_fname = \
            '/home/ilya/work/vlbi_errors/fits/1226+023_CALIB_SEQ10.FITS'
        self.idi_uv_fname = None
        self.im_fname = \
            '/home/ilya/work/vlbi_errors/fits/1226+023_ICLN_SEQ11.FITS'

    def test_get_hdu_groups(self):
        # TODO: How to compare HDU? Compare all keys in header and parnames
        groups_hdus = pf.open(self.groups_uv_fname)
        uv = PyFitsIO()
        hdu = uv.get_hdu(self.groups_uv_fname)
        for i in range(len(groups_hdus[0].data)):
            self.assertTrue(np.alltrue(groups_hdus[0].data[i][-1] ==
                                       hdu.data[i][-1]))

@skip
class Test_Data_Groups(TestCase):
    def setUp(self):
        self.split_groups_uv_fname = \
            '/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS'
        self.sc_groups_uv_fname =\
            '/home/ilya/work/vlbi_errors/fits/1226+023_CALIB_SEQ10.FITS'
        self.im_fname =\
            '/home/ilya/work/vlbi_errors/fits/1226+023_ICLN_SEQ11.FITS'
        self.sc_uv = Data()
        self.split_uv = Data()
        self.sc_uv.load(self.sc_groups_uv_fname)
        self.split_uv.load(self.split_groups_uv_fname)

    def test_open_fits(self):
        sc_uv = open_fits(self.sc_groups_uv_fname)
        # Test that sc_uv is the same as self.sc_uv

    def test_noise(self):
        self.sc_uv.load(self.sc_groups_uv_fname)
        # Test that ``noise`` method does return dictionary with keys that are
        # arrays with right dimensions ((1,) or (nstokes, nif,)).

        noise = self.sc_uv.noise(use_V=True, average_freq=False)
        self.assertEqual(len(noise), len(self.sc_uv.baselines))
        for key in noise:
            self.assertEqual(np.shape(noise[key]), (self.sc_uv.nif,))
            self.assertEqual(noise[key].ndim, 1)
            self.assertEqual(noise[key].size, self.sc_uv.nif)

        noise = self.sc_uv.noise(use_V=False, average_freq=False)
        self.assertEqual(len(noise), len(self.sc_uv.baselines))
        for key in noise:
            self.assertEqual(np.shape(noise[key]), (self.sc_uv.nif, self.sc_uv.nstokes))
            self.assertEqual(noise[key].ndim, self.sc_uv.nif)
            self.assertEqual(noise[key].size, self.sc_uv.nif * self.sc_uv.nstokes)

        noise = self.sc_uv.noise(use_V=True, average_freq=True)
        self.assertEqual(len(noise), len(self.sc_uv.baselines))
        for key in noise:
            self.assertEqual(np.shape(noise[key]), (self.sc_uv.nif, self.sc_uv.nstokes))
            self.assertEqual(noise[key].ndim, 1)
            self.assertEqual(noise[key].size, 1)

        noise = self.sc_uv.noise(use_V=False, average_freq=True)
        self.assertEqual(len(noise), len(self.sc_uv.baselines))
        for key in noise:
            self.assertEqual(np.shape(noise[key]), (self.sc_uv.nif, self.sc_uv.nstokes))
            self.assertEqual(noise[key].ndim, 1)
            self.assertEqual(noise[key].size, self.sc_uv.nstokes)


@skip
class Test_gains(TestCase):
    def setUp(self):
        fnames = glob.glob('/home/ilya/work/vlbi_errors/fits/12*CALIB*FITS')
        fnames.sort(reverse=True)
        fnames.append('/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS')
        self.calibrated_sequence = fnames
        self.snvers = [1] * len(self.calibrated_sequence)
        self.snvers[-1] = 2
