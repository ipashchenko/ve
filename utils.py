#!/usr/bin python2
# -*- coding: utf-8 -*-

import re
import numpy as np
import string
#from itertools import permutations
from math import floor


class AbsentHduExtensionError(Exception):
    pass


class AbsentVersionOfBinTableError(Exception):
    pass


class EmptyImageFtError(Exception):
    pass

#TODO: convert utils to using arrays instead of lists


def index_of(ar1, ar2):
    """
    Find indexes of elements of ar1 in ar2. It is assumed that each entry of
    ar1 are met only one time in ar2.
    """

    indxs_ar2_sorted = np.argsort(ar2)
    ar1_pos_left = np.searchsorted(ar2[indxs_ar2_sorted], ar1, side='left')
    ar1_pos_right = np.searchsorted(ar2[indxs_ar2_sorted], ar1, side='right')

    indxs = list()
    for i in range(len(ar1_pos_left)):
        indxs.append(range(ar1_pos_left[i], ar1_pos_right[i]))

    indxs = sum(indxs, [])

    return indxs_ar2_sorted[indxs]


def _to_complex_array(struct_array, real_name, imag_name):
    """
    Method that takes structured array and names of 2 fields and returns
    complex numpy.ndarray.
    """

    assert(np.shape(struct_array[real_name]) ==\
                                            np.shape(struct_array[imag_name]))

    return struct_array[real_name] + 1j * struct_array[imag_name]


def _to_one_array(struct_array, *names):
    """
    Method that takes structured array and names of 2 (or more) fields and
    returns numpy.ndarray with expanded shape.
    """

    # TODO: add assertion on equal shapes
    # TODO: can i use struct_array[[name1, name2]] synthax?
    arrays_to_dstack = list()
    print "Got stuct_array:"
    print struct_array
    print "names:"
    print names
    for name in names:
        print "current name:"
        print name
        name_array = struct_array[name]
        print "struct_array[name]:"
        print name_array
        print "it's shape"
        print np.shape(name_array)
        print "it's ndim"
        print name_array.ndim
        name_array = np.expand_dims(name_array, axis=name_array.ndim)
        arrays_to_dstack.append(name_array)

    return np.squeeze(np.dstack(arrays_to_dstack))


def change_shape(_array, _dict1, _dict2):
    """
    Function that takes ndarray and 2 dictionaries with array's shape and
    permuted shape returnes array with shape of the permuted array.

    Inputs:
        _array [numpy.ndarray] - array to change,
        dict1 - shape of array, that will be changed,
        dict2 - dictionary of new shape. It can include more items then dict1.
    """

    dict1 = _dict1.copy()
    dict2 = _dict2.copy()
    array = _array.copy()

    for key in dict1:
        print "check " + str(key)
        if not dict1[key] == dict2[key]:
            print "positin of axis " + str(key) + " has changed"
            print "from " + str(dict1[key]) + " to " + str(dict2[key])
            array = np.swapaxes(array, dict1[key], dict2[key])
            for item in dict1.items():
                if item[1] == dict2[key]:
                    dict1[item[0]] = dict1[key]
                    dict1[key] = dict2[key]
            print "Updated dict1 is :"
            print dict1

    # Assert that altered dict1 (it's part with shapes from dict2) coincide
    # with dict2

    return array


def aips_bintable_fortran_fields_to_dtype_conversion(aips_type):
    """Given AIPS fortran format of binary table (BT) fields, returns
    corresponding numpy dtype format and shape. Examples:
    4J => array of 4 32bit integers,
    E(4,32) => two dimensional array with 4 columns and 32 rows.
    """

    intv = np.vectorize(int)
    aips_char = None
    dtype_char = None
    repeat = None
    _shape = None

    format_dict = {'L': 'bool', 'I': '>i2', 'J': '>i4', 'A': 'S',  'E': '>f4',
            'D': '>f8'}

    for key in format_dict.keys():
        if key in aips_type:
            aips_char = key

    if not aips_char:
        raise Exception("aips data format reading problem " + str(aips_type))

    try:
        dtype_char = format_dict[aips_char]
    except KeyError:
        raise Exception("no dtype counterpart for aips data format" + str(aips_char))

    try:
        repeat = int(re.search(r"^(\d+)" + aips_char,
            aips_type).groups()[0])
        if aips_char is 'A':
            dtype_char = str(repeat) + dtype_char
            repeat = 1
    except AttributeError:
        repeat = None

    if repeat is None:
        _shape = tuple(intv(string.split(re.search(r"^" + aips_char +
            "\((.+)\)$", aips_type).groups()[0], sep=',')))
    else:
        _shape = repeat

    return dtype_char, _shape


def build_dtype_for_bintable_data(header):
    """Builds dtype for recarray from header.
    """

    #substitue = {'UV--SIN': 'u', 'VV--SIN': 'v', 'WW--SIN': 'w', 'BASELINE': 'bl', 'DATE': 't'}
    #assert(header_dict['EXTNAME'] == 'UV_DATA')

   # # # of axis. 2 => matrix
   # naxis = int(header['NAXIS'])
   # # # of fields in a item
    tfields = int(header['TFIELDS'])
   # # # of Bytes in a item (sum of length of tfields elements)
   # naxis1 = int(header['NAXIS1'])
   # # # of items
   # naxis2 = int(header['NAXIS2'])
   # nrecords = naxis2

    #parameters of regular data matrix if in UV_DATA table
    try:
        maxis = int(header['MAXIS'])
    except KeyError:
        print "non UV_DATA"

    #build np.dtype format
    names = []
    formats = []
    shapes = []
    tuple_shape = []
    array_names = []

    for i in range(1, tfields + 1):
        name = header['TTYPE' + str(i)]
        if name in names:
            name = name * 2
        names.append(name)
        _format, _shape = \
            aips_bintable_fortran_fields_to_dtype_conversion(header['TFORM' + \
            str(i)])

        #building format & names for regular data matrix
        if name == 'FLUX':
            for i in range(1, maxis + 1):
                maxisi = int(header['MAXIS' + str(i)])
                if maxisi > 1:
                    tuple_shape.append(int(header['MAXIS' + str(i)]))
                    array_names.append(header['CTYPE' + str(i)])
            formats.append('>f4')
            shapes.append(tuple(tuple_shape))
            array_names = array_names
        else:
            formats.append(_format)
            shapes.append(_shape)

    print names, formats, shapes, array_names

    dtype_builder = zip(names, formats, shapes)
    dtype = [(name, _format, shape) for (name, _format, shape) in
            dtype_builder]

    return dtype, array_names


def baselines_2_ants(baselines):
    """Given list of baseline numbers (fits keyword) returns list of
    corresponding antennas.
    """
    #TODO: CHECK IF OUTPUT/INPUT IS OK!!!
    for baseline in baselines:
        baseline = abs(baseline)
        assert(baseline > 256)

    ants = list()
    for baseline in baselines:
        baseline = abs(baseline)
        ant1 = int(baseline // 256)
        ant2 = int(baseline - ant1 * 256)
        if ant1 * 256 + ant2 != baseline:
            continue
        ants.append(ant1)
        ants.append(ant2)
    ants = list(set(ants))
    ants.sort()

    return ants


#def ants_2_baselines(ants):
#    """Given several antennas returns corresponding baselines.
#    """
#
#    baselines = list()
#    ants_by2 = list(permutations(ants, 2))
#    for ant in ants_by2:
#        baseline = 256 * ant[0] + ant[1]
#        baselines.append(baseline)
#    return baselines


def ant_2_containing_baslines(ant, antennas):
    """
    Given antenna returns list of all baselines among given list with that
    antenna.
    """

    baselines = list()
    for antenna in antennas:
        if antenna < ant:
            baselines.append(256 * antenna + ant)
        elif antenna > ant:
            baselines.append(256 * ant + antenna)
        else:
            pass

    return baselines


def ants_2_baselines(ants):
    baselines = list()
    for ant in ants:
        baselines.extend(ant_2_containing_baslines(ant, ants))
    baselines = list(set(baselines))
    baselines = baselines.sort()

    return baselines


def time_frac_to_dhms(fractime):
    """Converts time in fraction of the day format to time in d:h:m:s
    format."""

    dhms = list()

    for time in fractime:
        day = int(floor(time))
        hour = int(floor(time * 24.))
        minute = int(floor(time * 1440. - hour * 60.))
        second = int(floor(time * 86400. - hour * 3600. - minute * 60.))
        dhms.append(tuple([day, hour, minute, second]))

    return dhms


def time_dhms_to_frac(dhmses):
    """Converts time in format d:h:m:s to time in parameters format =
    fraction of the day.
    """
    fractions = list()

    for dhms in dhmses:
        day, hour, minute, second = dhms
        fraction = day + hour / 24.0 + minute / (24.0 * 60.0) + \
                   second / (24.0 * 60.0 * 60.0)
        fractions.append(fraction)

    return fractions
