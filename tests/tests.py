#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import pyfits as pf
from unittest import TestCase
from vlbi_errors.utils import (aips_bintable_fortran_fields_to_dtype_conversion,
                               index_of)
from vlbi_errors.data_io import IO, PyFitsIO, Groups


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
        self.groups_uv_file = \
            '/home/ilya/work/vlbi_errors/fits/1226+023_CALIB_SEQ10.FITS'
        self.idi_uv_file = None
        self.im_file = \
            '/home/ilya/work/vlbi_errors/fits/1226+023_ICLN_SEQ11.FITS'
        self.io = IO()

    def test_IO(self):
        self.assertRaises(NotImplementedError, self.io.load())
        self.assertRaises(NotImplementedError, self.io.save())

    def test_PyFitsIO(self):
        # TODO: How to compare HDU? Compare all keys in header and parnames


class Test_Data(TestCase):
    def setUp(self):
        self.self_calibrated_fname =\
            '/home/ilya/work/vlbi_errors/fits/11226+023_CALIB_SEQ10.FITS'
        self.image_fname =\
            '/home/ilya/work/vlbi_errors/fits/1226+023_ICLN_SEQ11.FITS'

class Test_gains(TestCase):
    def setUp(self):
        fnames = glob.glob('/home/ilya/work/vlbi_errors/fits/12*CALIB*FITS')
        fnames.sort(reverse=True)
        fnames.append('/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS')
        self.calibrated_sequence = fnames
        self.snvers = [1] * len(self.calibrated_sequence)
        self.snvers[-1] = 2
