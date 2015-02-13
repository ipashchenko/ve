#!/usr/bin python
# -*- coding: utf-8 -*-

import os
from from_fits import create_ccmodel_from_fits_file
from uv_data import UVData
from bootstrap import CleanBootstrap


if __name__ == '__main__':

    os.chdir('/home/ilya/code/vlbi_errors/data/misha')
    uvdata = UVData('1308+326.U1.2009_08_28.UV_CAL')
    ccmodel = create_ccmodel_from_fits_file('1308+326_ICLN.FITS')
    ccbootstrap = CleanBootstrap(ccmodel, uvdata)
    ccbootstrap.run(100)
    os.chdir('/home/ilya/code/vlbi_errors/vlbi_errors')
