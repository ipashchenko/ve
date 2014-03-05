#!/usr/bin python
# -*- coding: utf-8 -*-

import glob
from gains import Absorber
from model import Model
from new_data import Data


if __name__ == '__main__':

    gains = Absorber()
    fnames = glob.glob('/home/ilya/work/vlbi_errors/fits/12*CALIB*FITS')
    fnames.remove('/home/ilya/work/vlbi_errors/fits/1226+023_CALIB_SEQ10.FITS')
    fnames.sort(reverse=True)
    gains.absorb(fnames)
    gains.absorb_one('/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS',
                     snver=2)

    model = Data()
    split_data = Data()
    imodel = Model()
    imodel.add_from_txt('/home/ilya/work/vlbi_errors/fits/1226+023_CC1_SEQ11.txt')
    model.load('/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS')
    split_data.load('/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS')
    model.substitute(imodel)
   # gained_model = Absorber.absorbed_gains * model

   # # 1) Bootstrap data from self-calibrated data and
   # # residuals e_ij of self-calibration model:
   # # V_split_ij = g_i * g^*_j * V_ij_sc + e_ij
   # boots = Bootstrap(gained_model, split_data)
   # boots.sample('BOOTSPLIT.FITS', n=200)
   # # generated 200 FITS-files for batch proccessing

   # # 2) Cross-validation analysis of model from difmap modelling.
   # # load the model to cross-validate
   # model = Model()
   # model.add_from_txt('my_difmap_model.txt')

   # # create training and testing samples from data which were used to get model
   # cv = CrossValidation('SC.FITS')
   # cv.generate_samples(frac=0.1, n=100, outname='CV')
   # # batch model train_i_CV.FITS files and get model_i.txt files
   # # cross-validate models on testing samples
   # for model_file, data_file in cv_samples:
   #     test_data = Data()
   #     test_data.load()
