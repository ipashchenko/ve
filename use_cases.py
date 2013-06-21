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
    absorber.absorb_one(glob.glob('*.SPLIT'))
    Absorber.absorb()

    imodel = Model()
    imodel.add_from_txt('my_difmap_model.txt')

    model = Data()
    split_data = Data()
    model.load('SPLIT.FITS')
    split_data.load('SPLIT.FITS')
    model.substitute(imodel)
    gained_model = Absorber.absorbed_gains * model

    # 1) Bootstrap data from self-calibrated data and
    # residuals e_ij of self-calibration model:
    # V_split_ij = g_i * g^*_j * V_ij_sc + e_ij
    boots = Bootstrap(gained_model, split_data)
    boots.sample('BOOTSPLIT.FITS', n=200)
    # generated 200 FITS-files for batch proccessing
    
    # 2) Cross-validation analysis. To prepare 200 training samples (FITS-files)
