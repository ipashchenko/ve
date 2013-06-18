#!/usr/bin python2
# -*- coding: utf-8 -*-

from unittest import TestCase, main
from utils import aips_bintable_fortran_fields_to_dtype_conversion


class Test_utils(TestCase):
    def test_aips_bintable_fortran_fields_to_dtype_conversion(self):
        
        self.assertEqual(aips_bintable_fortran_fields_to_dtype_conversion(
                        '4J'), ('>i4', 4))
        self.assertEqual(aips_bintable_fortran_fields_to_dtype_conversion(
                        'E(4,32)'), ('>f4', (4, 32)))