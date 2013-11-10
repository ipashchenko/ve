#!/usr/bin python
# -*- coding: utf-8 -*-


import glob
import gains
from model import Model
from new_data import Data
#from vlbi import Data
#from stat import Bootstrap
#from stat import CrossValidation


if __name__ == '__main__':

    #load gains from self-calibration sequence of FITS-files
    absorber = gains.Absorber()
    file_list = glob.glob('*.FITS')
    absorber.absorb(file_list)

    imodel = Model()
    imodel.add_from_txt('cc.txt')
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

   # model = Data()
   # split_data = Data()
   # model.load('SPLIT.FITS')
   # split_data.load('SPLIT.FITS')
   # model.substitute(imodel)
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
