
#!/usr/bin python
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

    def load(self):
        """
        Method that returns structured numpy array with
            dtype=[('uvw', '<f8', (3,)),
                  ('time', '<f8'), ('baseline', 'int'),
                  ('hands', 'complex', (nstokes, nif, nch)),
                  ('weights', '<f8', (nstokes, nif, nch))]
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

        else:
            hdu = self.hdulist[0]

        self.hdu = hdu

        return self.hdu

    # TODO: Save structured array to recarrays HDUs (BinTable/GroupsHDU).
    # Then save HDUs to copies of HDU_Lists. Then write HDU_Lists to
   # def save(self, data, fname):
   #     hdu = self._data_to_HDU(data, header)
   #     self.hdu = hdu
   #     self.hdulist.writeto(fname)

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


class Groups(PyFitsIO):
    """
    Class that represents input/output of uv-data in UV-FITS format (\"random
    groups\").
    """

    def load(self, fname):
        """
        Load data from FITS-file.
        """

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
        self.data_of__data = {'COMPLEX': 0, 'GROUP': 1, 'STOKES': 2, 'IF': 3}
        # Number of axis with dimension=1. 3 corresponds to 'STOKES', 'IF' &
        # 'COMPLEX'
        self.ndim_ones = hdu.header['NAXIS'] - 1 - 3

        _data = np.zeros(hdu.header['GCOUNT'], dtype=[('uvw', '<f8', (3,)),
                                                      ('time', '<f8'),
                                                      ('baseline', 'int'),
                                                      ('hands', 'complex',
                                                          (nstokes, nif,)),
                                                      ('weights', '<f8',
                                                          (nstokes, nif,))])

        # Swap axis and squeeze array to get complex array (nstokes, nif,)
        temp = np.swapaxes(hdu.data['DATA'], 1, data_of_data['STOKES'][0])
        temp = np.swapaxes(temp, 2, data_of_data['IF'][0])
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
        GroupsHDU.
        # TODO: Save structured array to recarrays HDUs (BinTable/GroupsHDU).
        # Then save HDUs to copies of HDU_Lists. Then write HDU_Lists to
        """

        # constructing array (3, 20156, 4, 8,)
        temp = np.vstack((_data['hands'].real[np.newaxis, :],
                          _data['hands'].imag[np.newaxis, :],
                          _data['weights'][np.newaxis, :]))
        # Now convert temp (3, 20156, 4, 8) to 'DATA' part of recarray
        # (20156, 1, 1, 8, 1, 4, 3)

        # First, add dimensions:
        for i in range(self.ndim_ones):
            temp = temp[:, np.newaxis]
        # Now temp has shape (3, 20156, 4, 8, 1, 1, 1)

        # TODO: make function that takes ndarray and 2 dictionaries with
        # array's shape and permuted shape returnes array with shape of the
        # permuted array
        temp = change_shape(temp, self.data_of__data, self.data_of_data)

       # # Now swap axis and change data_of__data until we have the right shape
       # # Change 'GGOUP' on axis 0
       # np.swapaxes(temp, data_of__data['GROUP'], data_of_data['GROUP'][0])
       # # and change data_of__data dictionary accordingly
       # for item in data_of__data.items():
       #     if item[1] == data_of_data['GROUP'][0]:
       #         data_of__data[item[0]] = data_of__data['GROUP']

        # TODO: should i convert ``temp`` to record array?
        imdata = temp
        parnames = self.hdu.parnames
        pardata = list()
        for name in parnames:
            pardata.append(self.hdu.data[name])

        a = pf.GroupData(imdata, parnames=parnames, pardata=pardata,
                         bitpix=-32)
        b = pf.GroupsHDU(a, self.hdu.header)

        self.hdulist[0] = b
        self.hdulist.writeto(fname)
