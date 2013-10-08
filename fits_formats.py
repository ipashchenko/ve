#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import pyfits as pf
from utils import AbsentHduExtensionError
from utils import build_dtype_for_bintable_data


# TODO: methods to convert recarray (data part of HDU & header) to instances of
# Groups/BintableHDU classes. Supermethod in IO and extensions in Groups & BinTable. Also method that updates info in header if data was truncated.

class PyFitsIO(object):

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

    def load(self, fname):
        raise NotImplementedError("Method must be implemented in subclasses")

    def save(self, data, fname):
        hdu = self._recarray_data_to_HDU(data, header)
        self.hdu = hdu
        self.hdulist.writeto(fname)

    def _recarray_data_to_HDU(self, data, header):
        """
        Converts recarray of data part of HDU and header to Groups/BinTableHDU
        instances.
        """
        
        raise NotImplementedError('method must be implemented in subclasses')
        
    def _update_header(self, data):
        """
        Method that updates header info using data recarray.
        """
        
        raise NotImplementedError('method must be implemented in subclasses')
         
    def cross_validation(self, frac):

        learn_brecarrays = list()
        test_brecarrays = list()

        for baseline in set(self.recarray.BASELINE):
            brecarray = self.recarray[np.where(self.recarray.BASELINE ==
                baseline)]
            blen = len(brecarray)
            len_test = int(frac * blen)
            #indxs = np.arange(blen)
            rand_uniform = np.random.randint(0, blen - 1)
            test_indxs = rand_uniform[:len_test]
            learn_indxs = rand_uniform[len_test:]
            learn_brecarray = brecarray[learn_indxs]
            test_brecarray = brecarray[test_indxs]

            learn_brecarrays.append(learn_brecarray)
            test_brecarrays.append(test_brecarray)

        learn_brecarrays = np.concatenate(learn_brecarrays)
        test_brecarrays = np.concatenate(test_brecarray)
        
        # TODO: Exclude transforming to recarrays (FITS_record is recarray).
        

        # TODO: Save recarrays to HDUs (BinTable/GroupsHDU). Then save HDUs to
        # copies of HDU_Lists. Then write HDU_Lists to


class Groups(IO):
    """
    Class that represents input/output of uv-data in UV-FITS format (\"random
    groups\").
    """

    def load(self, fname):
        """
        Load data from FITS-file.
        """

        hdu = self.get_hdu(fname)
        names = hdu.data.columns.names
        columns = [hdu.data.field(col) for col in names]

        dt = hdu.data.dtype
        #change to general names: u, v, w, t, dt, baseline, flux

        recarray = np.rec.fromarrays(columns, dtype=hdu.data.dtype)
        self.recarray = recarray.view(dtype=dt)

        return recarray['DATA']
        # TODO: do i really need to transform record object to recarray?
        
    def save(self, data, fname):
        """
        Using modified 'DATA' part of hdu.data.
        """
        
        self.hdu.data['DATA'] = data
        self.hdulist.writeto(fname)

#    def save(self, fname):
#        """
#        Saves ``recarray`` to GroupData and then saves GroupData to GroupsHDU
#        using header (possibly modified).
#        """
#        imdata = ghdu.data.data
#        parnames = ghdu.parnames
#        pardata = list()
#        for name in parnames:
#            pardata.append(ghdu.data[name])
#
#        a = pf.GroupData(imdata, parnames=parnames, pardata=pardata,
#                         bitpix=-32)
#        b = pf.GroupsHDU(a, ghdu.header)
#
#        self.hdulist[0] = b
#        self.hdulist.writeto(fname)


class BinTable(IO):
    """
    Class that represents input/outpur of uv-data in FITS-IDI format.
    """

    def load(self, fname, extname=None):
        """
        Load data from FITS-file.
        """

        if not extname:
            extname = 'UV_DATA'

        hdu = self.get_hdu(fname, extname=extname)
        dt = build_dtype_for_bintable_data(hdu.header)

        names = hdu.column.names
        columns = [hdu.data.field(col) for col in names]
        recarray = np.rec.fromarrays(columns, dtype=dt)
        self.recarray = recarray
