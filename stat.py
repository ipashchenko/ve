#!/usr/bin python2
# -*- coding: utf-8 -*-

from model import Model
from gains import Absorber
from new_data import open_fits
import glob
import numpy as np


class Bootstrap(object):
    """
    Sample with replacement (if nonparametric=True) from residuals or use
    normal zeromean random variable with std estimated from the residuals
    for each baseline (or even scan).

    Inputs:

        uncal - uncalibrated uv-data. FITS-file with data not yet
            self-calibrated.
        calibs - iterable of self-calibration sequence.
        residuals - instance of Data class. Difference between
        unself-calibrated data and self-calibrated data with gains added.

    """

    def __init__(self, uncal=None, calibs=None, nonparametric=False,
                 split_scans=False, use_V=True):
        absorber = Absorber()
        absorber.absorb(calibs)
        last_calib = open_fits(calibs[-1])
        residuals = uncal - absorber * last_calib
        self.last_calib = last_calib
        self.residuals = residuals

        if not nonparametric:
            noise_residuals = residuals.noise(split_scans=split_scans,
                                              use_V=use_V)

    def sample(self, model, outname=None, nonparametric=None, n=100,
               sigma=None):
        """
        Sample from residuals with replacement or sample from normal random
        noise and adds samples to model to form n bootstrap samples.
        """
        pass


class CrossValidation(object):
    """
    Class that implements cross-validation analysis of image-plane models.
    """

    def __init__(self, data):

        self.data = data

    def run(self, modelcard=None, testcard=None, stokes='I'):
        """
        Method that cross-validate set of models obtained by modelling training
        samples on corresponding set of testing samples.

            Inputs:

                modelfiles - wildcard of file names ~ 'model_0i_0jofN.txt',
                    where model in 'model_0i_0jofN.txt' file is from modelling
                    ``0j``-th training sample ('train_0jofN.FITS') with
                    ``0i``-th model.

                testfiles - wildcard of file names ~ 'test_0jofN.FITS'.

            Output:

                list of lists [modelfilename, CV-score, sigma_cv_score].
        """

        modelfiles = glob.glob(modelcard)
        testfiles = glob.glob(testcard)
        modelfiles.sort()
        testfiles.sort()
        ntest = len(testfiles)
        nmodels = len(modelfiles) / ntest

        assert(not len(modelfiles) % float(len(testfiles)))

        print "modelfiles : " + str(modelfiles)
        print "testfiles : " + str(testfiles)

        result = list()

        for i in range(nmodels):
            print "using models " + str(modelfiles[ntest * i: ntest * (i + 1)])\
                   + " and testing sample " + str(testfiles)
            models = modelfiles[ntest * i: ntest * (i + 1)]
            cv_scores = list()
            for j, testfile in enumerate(testfiles):
                model = Model()
                model.add_from_txt(models[j], stoke=stokes)
                print "using test file " + str(testfile)
                data = open_fits(testfile)
                cv_score = data.cv_score(model, stokes=stokes)
                print "cv_score for one testing sample is " + str(cv_score)
                cv_scores.append(cv_score)

            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            print mean_cv_score, std_cv_score

            result.append(["model#" + str(i + 1), mean_cv_score, std_cv_score])

        return result


class LnLikelihood(object):
    """
    Class that implements likelihood calculation for given data.
    """

    def __init__(self, data, model):
        pass

    def __call__(self, p):
        pass

    
if __name__ == '__main__':
    
    data = open_fits('PRELAST_CALIB.FITS')
    cv = CrossValidation(data)
    test_fits_files = glob.glob('test*.FITS')
    if not test_fits_files:
        data.cv(10, 'test')
        print "Prepairing testing and training samples."
    else:
        print "Testing and training samples are ready."
    res = cv.run(modelcard = 'model_*_**of10.txt', testcard='test_*of10.FITS')    
    print res
    