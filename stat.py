#!/usr/bin python2
# -*- coding: utf-8 -*-

from model import Model
from gains import Absorber
from new_data import open_fits
import glob
import copy
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

    def __init__(self, model, uncal=None, calibs=None, nonparametric=False,
                 split_scans=False, use_V=True):
        self.model = model
        self.uncal = uncal
        self.calibs = calibs
        self.nonparametric = nonparametric
        self.split_scans = split_scans
        self.use_V = use_V

        absorber = Absorber()
        absorber.absorb(calibs)
        uncal = open_fits(uncal)
        last_calib = open_fits(calibs[-1])
        residuals = uncal - absorber * last_calib
        self.last_calib = last_calib
        self.residuals = residuals
        self.model_uv = copy.deepcopy(residuals)
        self.model_uv.substitute(model)

    def resample(self, outname=None, nonparametric=None, sigma=None,
               split_scans=False, use_V=True):
        """
        Sample from residuals with replacement or sample from normal random
        noise and adds samples to model to form n bootstrap samples.
        """

        if split_scans:
            raise NotImplementedError('Implement split_scans=True option!')

        noise_residuals = self.residuals.noise(split_scans=split_scans,
                                        use_V=use_V)
        if use_V:
            nstokes = self.residuals.nstokes
            for key, value in noise_residuals.items():
                noise_residuals[key] = (value.repeat(nstokes).reshape((len(value),
                                                                       nstokes)))
        # Now ``noise_residuals`` has shape (nstokes, nif)

        # Do resampling
        if not nonparametric:
            # Use noise_residuals as std of gaussian noise to add.
            # shape(noise_residuals[key]) = (8,4,)
            # shape(self.residuals._data[i]['hands']) = (8,4,)
            # for each baseline create (8,4,) normal random variables with specified
            # by noise_residuals[baseline] std
            for baseline in self.residuals.baselines:
                # Find data from one baseline
                indxs = np.where(self.residuals._data['baseline'] == baseline)[0]
                data_to_add_normvars = self.residuals._data[indxs]
                # Generate (len(indxs),8,4,) array of random variables ``anormvars``
                # to add:
                lnormvars = list()
                for std in noise_residuals[baseline].flatten():
                    lnormvars.append(np.random.normal(std, size=len(indxs)))
                anormvars = np.dstack(lnormvars).reshape((len(indxs), 8, 4,))
                # Add normal random variables to data on current baseline
                data_to_add_normvars._data['hands'] += anormvars
        else:
            # TODO: should i resample all stokes and IFs together? Yes
            # Bootstrap from self.residuals._data. For each baseline.
            for baseline in self.residuals.baselines:
                # Find data from one baseline
                indxs = np.where(self.residuals._data['baseline'] == baseline)[1]
                data_to_resample = self.residuals._data[indxs]
                # Resample it
                resampled_data = np.random.choice(data_to_resample,
                                                  len(data_to_resample))

                # Add to residuals.substitute(model)
                self.model_uv[indxs] = self.model_uv[indxs] + resampled_data

        self.model_uv.save(self.model_uv._data, outname)

    def run(self, outname='bootstrapped_data', n=100, nonparametric=True,
            sigma=None, split_scans=False, use_V=True):
        """
        Generate ``n`` data sets.
        """
        for i in range(n):
            outname = outname + '_' + str(i) + '.FITS'
            self.resample(outname=outname, nonparametric=nonparametric,
                        sigma=sigma, split_scans=split_scans, use_V=use_V)


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
    pass

    # Test CrossValidation
    #data = open_fits('PRELAST_CALIB.FITS')
    #cv = CrossValidation(data)
    #test_fits_files = glob.glob('test*.FITS')
    #if not test_fits_files:
    #    data.cv(10, 'test')
    #    print "Prepairing testing and training samples."
    #else:
    #    print "Testing and training samples are ready."
    #res = cv.run(modelcard='model_*_**of10.txt', testcard='test_*of10.FITS')
    #print res

    # Test Bootstrap
    #uncal = 'FIRSTS_CALIB.FITS'
    #calibs = ['PRELAST_CALIB.FITS']
    #absorber = Absorber()
    #absorber.absorb(calibs)
    #last_calib = open_fits(calibs[-1])
    #uncal = open_fits(uncal)
    #residuals = uncal - absorber * last_calib
    model = Model()
    model.add_from_txt('cc.txt')
    bootstrap = Bootstrap(model, uncal='FIRST_CALIB.FITS',
                          calibs=['PRELAST_CALIB.FITS'],
                          nonparametric=False,
                          split_scans=False,
                          use_V=True)

    print 'Done!'
    #model = Model()
    #model.add_from_txt('cc.txt')
    #bootstrap = Bootstrap(model,
    #                      uncal='FIRST_CALIB.FITS',
    #                      calibs=['PRELAST_CALIB.FITS'],
    #                      nonparametric=False,
    #                      split_scans=False,
    #                      use_V=True)

    #noise_residuals = bootstrap.residuals.noise()
    #for key, value in noise_residuals.items():
    #    noise_residuals[key] =\
    #        value.repeat(bootstrap.residuals.nstokes).reshape((len(value),
    #                                                           bootstrap.residuals.nstokes))
    #data_to_resample =\
    #    bootstrap.residuals._data[np.where(bootstrap.residuals._data['baseline']==515)]
