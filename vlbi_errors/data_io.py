#!/usr/bin python2
# -*- coding: utf-8 -*-

import numpy as np
import pyfits as pf
from utils import AbsentHduExtensionError
#from utils import build_dtype_for_bintable_data
from utils import change_shape
from utils import index_of
from utils import _to_one_ndarray
#from utils import _to_complex_array


vec_int = np.vectorize(np.int)
vec_complex = np.vectorize(np.complex)


class IO(object):
    """
    Abstract class for I/O of different formats of interferometric data.
    Contains load and save methods.
    """

    #def __init__(self, dtype=None):

    #    self._dtype = dtype

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
        instance) to native format.
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
                                     (hdu.header['NAXIS'] - i + 1,
                                      hdu.header['NAXIS' + str(i)])})
        nstokes = data_of_data['STOKES'][1]
        nif = data_of_data['IF'][1]
        self.nstokes = nstokes
        self.nif = nif
        # Describe shape and dimensions of original data recarray
        self.data_of_data = data_of_data
        # Describe shape and dimensions of structured array
        self.data_of__data = {'COMPLEX': 3, 'GROUP': 0, 'STOKES': 2, 'IF': 1}

        # Dictionary with name and corresponding index of parameters in group
        par_dict = dict()
        for name in self.hdu.data.parnames:
            # It is cool that if parameter name does appear twice it is index
            # of the first appearence that are recorded in ``par_dict``
            par_dict.update({name: self.hdu.data.parnames.index(name) + 1})
        self.par_dict = par_dict

        # Number of axis with ndim=1.
        self.ndim_ones = sum([value[1] for value in data_of_data.values() if
                             value[1] == 1])

        _data = np.zeros(hdu.header['GCOUNT'], dtype=[('uvw', '<f8', (3,)),
                                                      ('time', '<f8'),
                                                      ('baseline', 'int'),
                                                      ('hands', 'complex',
                                                       (nif, nstokes)),
                                                      ('weights', '<f8',
                                                       (nif, nstokes,))])

        # Swap axis and squeeze array to get complex array (nif, nstokes,)
        # FIXME: refactor to (nstokes, nif,)? - bad idea? think about it later!
        # Now IF is has index 1.
        temp = np.swapaxes(hdu.data['DATA'], 1, data_of_data['IF'][0])
        # Now STOKES has index 2
        temp = np.swapaxes(temp, 2, data_of_data['STOKES'][0])
        # FIXME: squeeze() does remove axis with IF if it's dim = 1!
        temp = temp.squeeze()
        # Insert dimension for IF
        if self.nif == 1:
            temp = np.expand_dims(temp, axis=1)
        hands = vec_complex(temp[..., 0], temp[..., 1])
        weights = temp[..., 2]

        u = hdu.data[hdu.header['PTYPE1']] / hdu.header['PSCAL1'] - \
            hdu.header['PZERO1']
        v = hdu.data[hdu.header['PTYPE2']] / hdu.header['PSCAL2'] - \
            hdu.header['PZERO2']
        w = hdu.data[hdu.header['PTYPE3']] / hdu.header['PSCAL3'] - \
            hdu.header['PZERO3']
        # ``DATE`` can have different number among parameters
        indx_date = par_dict['DATE']
        time = hdu.data[hdu.header['PTYPE' + str(indx_date)]] /\
               hdu.header['PSCAL' + str(indx_date)] - hdu.header['PZERO' +
                                                                 str(indx_date)]

        # Filling structured array by fileds
        _data['uvw'] = np.column_stack((u, v, w))
        _data['time'] = time
        indx_bl = par_dict['BASELINE']
        _data['baseline'] = \
            vec_int(hdu.data[hdu.header['PTYPE' + str(indx_bl)]] /
                    hdu.header['PSCAL' + str(indx_bl)] -
                    hdu.header['PZERO' + str(indx_bl)])
        _data['hands'] = hands
        _data['weights'] = weights

        return _data

    def save(self, _data, fname):
        """
        Save modified structured array to GroupData, then saves GroupData to
        GroupsHDU.
        """

        # Constructing array (3, 20156, 4, 8,)
        temp = np.vstack((_data['hands'].real[np.newaxis, :],
                          _data['hands'].imag[np.newaxis, :],
                          _data['weights'][np.newaxis, :]))

        # Construct corresponding arrays of parameter values
        _data_copy = _data.copy()
        _data_copy['uvw'][:, 0] = (_data_copy['uvw'][:, 0] +
                                   self.hdu.header['PZERO1']) * self.hdu.header['PSCAL1']
        _data_copy['uvw'][:, 1] = (_data_copy['uvw'][:, 1] +
                                   self.hdu.header['PZERO2']) * self.hdu.header['PSCAL2']
        _data_copy['uvw'][:, 2] = (_data_copy['uvw'][:, 2] +
                                   self.hdu.header['PZERO3']) * self.hdu.header['PSCAL3']
        _data_copy['time'] = (_data_copy['time'] +
                              self.hdu.header['PZERO4']) * self.hdu.header['PSCAL4']
        _data_copy['baseline'] = (_data_copy['baseline'] +
                                  self.hdu.header['PZERO6']) * self.hdu.header['PSCAL6']

        # Now roll axis 0 to 3rd position (3, 20156, 8, 4) => (20156, 8, 4, 3)
        temp = np.rollaxis(temp, 0, 4)

        # First, add dimensions:
        for i in range(self.ndim_ones):
            temp = np.expand_dims(temp, axis=4)
            # Now temp has shape (20156, 8, 4, 3, 1, 1, 1)

        temp = change_shape(temp, self.data_of__data, {key:
                                                           self.data_of_data[key][0] for key in
                                                       self.data_of_data.keys()})
        # => (20156, 1, 1, 8, 1, 4, 3) as 'DATA' part of recarray

        # Write regular array data (``temp``) and corresponding parameters to
        # instances of pyfits.GroupsHDU
        imdata = temp

        # Use parameter values of saving data to find indexes of this
        # parameters in the original data entry of HDU
        if len(_data) < len(self.hdu.data):

            original_data = _to_one_ndarray(self.hdu.data, 'UU---SIN',
                                            'VV---SIN', 'WW---SIN', 'DATE',
                                            'BASELINE')
            saving_data = np.dstack((np.array(np.hsplit(_data_copy['uvw'],
                                                        3)).T, _data_copy['time'], _data_copy['baseline']))
            saving_data = np.squeeze(saving_data)
            # TODO: this is funnest workaround:)
            par_indxs = np.hstack(index_of(saving_data.sum(axis=1),
                                           original_data.sum(axis=1)))
        elif len(_data) > len(self.hdu.data):
            raise Exception('There must be equal or less visibilities to\
                            save!')
        else:
            print "Saving data - number of groups haven't changed"
            par_indxs = np.arange(len(self.hdu.data))

        parnames = self.hdu.data.parnames
        pardata = list()
        for name in parnames:
            par = self.hdu.data[name][par_indxs]
            par = (par - self.hdu.header['PZERO' + str(self.par_dict[name])]) / \
                  self.hdu.header['PSCAL' + str(self.par_dict[name])]
            pardata.append(par)

        # If two parameters for one value (like ``DATE``)
        for name in parnames:
            if parnames.count(name) == 2:
                indx_to_zero = parnames.index(name) + 1
                break
            # then zero array for second parameter with the same name
        # TODO: use dtype from ``BITPIX`` keyword
        pardata[indx_to_zero] = np.zeros(len(par_indxs), dtype=float)

        a = pf.GroupData(imdata, parnames=parnames, pardata=pardata,
                         bitpix=-32)
        b = pf.GroupsHDU(a)
        # PyFits updates header using given data (``GCOUNT``)
        b.header = self.hdu.header

        hdulist = pf.HDUList([b])
        for hdu in self.hdulist[1:]:
            hdulist.append(hdu)
        hdulist.writeto(fname + '.FITS')


class IDI(PyFitsIO):
    """
    Class that represents input/output of uv-data in IDI-FITS format.
    """

    pass
