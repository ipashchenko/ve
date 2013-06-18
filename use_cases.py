#!/usr/bin python
# -*- coding: utf-8 -*-


import glob
from gains import Absorber
from model import Model
from vlbi import Data
from stat import Bootstrap


if __name__ == '__main__':

    #load gains from self-calibration sequence of FITS-files
    absorber = Absorber(glob.glob('*.CALIB'))
    Absorber.absorb()
    gains = Absorber.absorbed_gains

    imodel = Model()
    imodel.add_from_txt('my_difmap_model.txt')

    model = Data()
    split_data = Data()
    model.load('SPLIT.FITS')
    split_data.load('SPLIT.FITS')
    model.substitute(imodel)
    gained_model = gains * model

    boots = Bootstrap(gained_model, split_data)
    boots.sample('BOOTSPLIT.FITS', n=200)
