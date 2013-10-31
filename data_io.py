#!/usr/bin python2
# -*- coding: utf-8 -*-

import numpy as np
import pyfits as pf
from utils import AbsentHduExtensionError
from utils import build_dtype_for_bintable_data
from utils import change_shape

vec_int = np.vectorize(np.int)
vec_complex = np.vectorize(np.complex)


class IO(object):
    """
    Abstract class for I/O of different formats of interferometric data.
    Contains load and save methods.
    """

    def __init__(self, dtype):

        self._dtype = dtype

    def load(self):
        """
        Method that returns structured numpy array with specified in __init__
        dtype, where:
            dtype=[('uvw', '<f8', (3,)),
                  ('time', '<f8'), ('baseline', 'int'),
                  ('hands', 'complex', (nstokes, nif, nch)),
                  ('weights', '<f8', (nstokes, nif, nch))]
                - for visibillity data,
            dtype=[('start', '<f8'),
                   ('stop', '<f8'),
                   ('antenna', 'int'),
                   ('gains', 'complex', (nif, npol,)),
                   ('weights', '<f8', (nif, npol,))]
                - for antenna gains data.
        """

        raise NotImplementedError("Method must be implemented in subclasses")

    def save(self):
        """
        Method that transforms structured array (_data attribute of Data
        instance) to naitive format.
        """

        raise NotImplementedError("Method must be implemented in subclasses")


class PyFitsIO(IO):

    def __init__(self):
        super(PyFitsIO, self).__init__()
        # We need hdu in save()
        self.hdu = None

    def get_hdu(self, fname, extname=None, ver=1):

        hdulist = pf.open(fname)
        self.hdulist = hdulist

        if extname:
            try:
                indx = self.hdulist.index_of((extname, ver,))
                hdu = self.hdulist[indx]
            except:
                raise AbsentHduExtensionError('Haven\'t  found ' + extname
                        + ' binary table in ' + fname)

        # Get Primary HDU with UV-data in groups.
        else:
            hdu = self.hdulist[0]

        self.hdu = hdu

        return self.hdu

    def _HDU_to_data(self, hdu):
        """
        Converts Groups/BinTableHDU instances to structured array.
        """

        raise NotImplementedError('method must be implemented in subclasses')

    def _data_to_HDU(self, data, header):
        """
        Converts structured array of data part of HDU and header to
        Groups/BinTableHDU instances.
        """

        raise NotImplementedError('method must be implemented in subclasses')

    def _update_header(self, data):
        """
        Method that updates header info using data recarray.
        """

        raise NotImplementedError('method must be implemented in subclasses')

    def _to_one_array(struct_array, *names):
        """
        Method that takes structured array and names of 2 (or more) fields and
        returns numpy.ndarray with expanded shape.
        """

        # TODO: add assertion on equal shapes
        # TODO: can i use struct_array[[name1, name2]] synthax?
        arrays_to_dstack = list()
        for name in names:
            name_array = struct_array[name]
            name_array = np.expand_dims(name_array, axis=name_array.dim)
            arrays_to_dstack.append(name_array)

        return np.dstack(arrays_to_dstack)

    def _to_complex_array(struct_array, real_name, imag_name):
        """
        Method that takes structured array and names of 2 fields and returns
        complex numpy.ndarray.
        """

        assert(np.shape(struct_array[real_name]) ==\
                                             np.shape(struct_array[imag_name]))

        return struct_array[real_name] + 1j * struct_array[imag_name]


# TODO: subclass IO.PyFitsIO.IDI! SN table is a binary table (as all HDUs in IDI
# format). So there must be general method to populate self._data structured
# array using given dtype and some kwargs.
class AN(PyFitsIO):
    """
    Class that represents input/output of antenna gains data in various FITS
    format. AN table is Binary Table, so UV- and IDI- formats are the same.
    """

    def load(self, fname, snver=1):

        # R & L
        npol = 2
        hdu = self.get_hdu(fname, extname='AIPS SN', ver=snver)

        nif = hdu.data.dtype['REAL1'].shape[0]
        # set ``nif'' from dtype of hdu.data
        _data = np.zeros(hdu.header['NAXIS2'], dtype=[('start', '<f8'),
                                                        ('stop', '<f8'),
                                                        ('antenna', 'int'),
                                                        ('gains', 'complex',
                                                            (nif, npol,)),
                                                        ('weights', '<f8',
                                                            (nif, npol,))])

        time = hdu.data['TIME']
        dtime = hdu.data['TIME INTERVAL']
        antenna = hdu.data['ANTENNA NO.']

        # Constructing `gains` field
        rgains = hdu.data['REAL1'] + 1j * hdu.data['IMAG1']
        # => (466, 8)
        lgains = hdu.data['REAL2'] + 1j * hdu.data['IMAG2']
        rgains = np.expand_dims(rgains, axis=2)
        # => (466, 8, 1)
        lgains = np.expand_dims(lgains, axis=2)
        gains = np.dstack((rgains, lgains))
        # => (466, 8, 2)

        # Constructing `weights` field
        rweights = hdu.data['WEIGHT 1']
        # => (466, 8)
        lweights = hdu.data['WEIGHT 2']
        rweights = np.expand_dims(rweights, axis=2)
        # => (466, 8, 1)
        lweights = np.expand_dims(lweights, axis=2)
        weights = np.dstack((rweights, lweights))
        # => (466, 8, 2)

        # Filling structured array by fileds
        _data['start'] = time - 0.5 * dtime
        _data['stop'] = time + 0.5 * dtime
        _data['antenna'] = antenna
        _data['gains'] = gains
        _data['weights'] = weights

        self.hdu = hdu

        return _data


class Groups(PyFitsIO):
    """
    Class that represents input/output of uv-data in UV-FITS format (\"random
    groups\").
    """

    def load(self, fname):
        """
        Load data from FITS-file.
        """

        self.hdulist = pf.open(fname)
        hdu = self.get_hdu(fname)
        self.hdu = hdu

        data_of_data = dict()
        data_of_data.update({'GROUP': (0, hdu.header['GCOUNT'])})
        for i in range(2, hdu.header['NAXIS'] + 1):
            data_of_data.update({hdu.header['CTYPE' + str(i)]:
                (hdu.header['NAXIS'] - i + 1, hdu.header['NAXIS' + str(i)])})
        nstokes = data_of_data['STOKES'][1]
        nif = data_of_data['IF'][1]
        # Describe shape and dimensions of original data recarray
        self.data_of_data = data_of_data
        # Describe shape and dimensions of structured array
        self.data_of__data = {'COMPLEX': 3, 'GROUP': 0, 'STOKES': 2, 'IF': 1}
        # Number of axis with dimension=1. 3 corresponds to 'STOKES', 'IF' &
        # 'COMPLEX'
        self.ndim_ones = hdu.header['NAXIS'] - 1 - 3

        _data = np.zeros(hdu.header['GCOUNT'], dtype=[('uvw', '<f8', (3,)),
                                                      ('time', '<f8'),
                                                      ('baseline', 'int'),
                                                      ('hands', 'complex',
                                                          (nif, nstokes)),
                                                      ('weights', '<f8',
                                                          (nif, nstokes,))])

        # Swap axis and squeeze array to get complex array (nif, nstokes,)
        temp = np.swapaxes(hdu.data['DATA'], 1, data_of_data['IF'][0])
        temp = np.swapaxes(temp, 2, data_of_data['STOKES'][0])
        temp = temp.squeeze()
        hands = vec_complex(temp[..., 0], temp[..., 1])
        weights = temp[..., 2]

        u = hdu.data[hdu.header['PTYPE1']] / hdu.header['PSCAL1'] -\
            hdu.header['PZERO1']
        v = hdu.data[hdu.header['PTYPE2']] / hdu.header['PSCAL2'] -\
            hdu.header['PZERO2']
        w = hdu.data[hdu.header['PTYPE3']] / hdu.header['PSCAL3'] -\
            hdu.header['PZERO3']
        time = hdu.data[hdu.header['PTYPE4']] / hdu.header['PSCAL4'] -\
            hdu.header['PZERO4']

        # Filling structured array by fileds
        _data['uvw'] = np.column_stack((u, v, w))
        _data['time'] = time
        _data['baseline'] =\
                vec_int(hdu.data[hdu.header['PTYPE6']] / hdu.header['PSCAL6']
                        - hdu.header['PZERO6'])
        _data['hands'] = hands
        _data['weights'] = weights

        return _data

    def save(self, _data, fname):
        """
        Save modified structured array to GroupData, then saves GroupData to
        GroupsHDU. As array could be truncated, update "NAXIS" keyword of the
        header of HDU.
        """

        # constructing array (3, 20156, 4, 8,)
        temp = np.vstack((_data['hands'].real[np.newaxis, :],
                          _data['hands'].imag[np.newaxis, :],
                          _data['weights'][np.newaxis, :]))
        # Now roll axis 0 to 3rd position (3, 20156, 8, 4) => (20156, 8, 4, 3)
        temp = np.rollaxis(temp, 0, 4)

        # First, add dimensions:
        for i in range(self.ndim_ones):
            temp = np.expand_dims(temp, axis=4)
        # Now temp has shape (20156, 8, 4, 3, 1, 1, 1)

        temp = change_shape(temp, self.data_of__data, {key:
               self.data_of_data[key][0] for key in self.data_of_data.keys()})
        # => (20156, 1, 1, 8, 1, 4, 3) as 'DATA' part of recarray

        imdata = temp
        parnames = self.hdu.parnames
        pardata = list()
        for name in parnames:
            pardata.append(self.hdu.data[name])

        a = pf.GroupData(imdata, parnames=parnames, pardata=pardata,
                         bitpix=-32)
        b = pf.GroupsHDU(a)
        b.header = self.hdu.header
        # TODO: use PyFitsIO.update_header() method to update header
        # accordingly to possibly modified structured array!
        b.header['NAXIS'] = len(imdata)

        self.hdulist[0] = b
        self.hdulist.writeto(fname + '.FITS')


class IDI(PyFitsIO):
    """
    Class that represents input/output of uv-data in IDI-FITS format.
    """

    pass
