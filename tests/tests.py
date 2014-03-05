#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import pyfits as pf
from unittest import TestCase
from vlbi_errors.utils import (aips_bintable_fortran_fields_to_dtype_conversion,
                               index_of)


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

