import math
from model_old import Model
from gains import Absorber
from uv_data import open_fits
import glob
import copy
import numpy as np
import scipy as sp
from utils import is_sorted


class AddNoise(object):
    """
    Class that implements the simplest method of errors estimates.
    """
    def __init__(self, n=100):
        pass


class CrossValidation(object):
    """
    Class that implements cross-validation analysis of image-plane models.
    """
    def __init__(self, data):
        self.data = data

    def run(self, modelcard=None, testcard=None, stokes='I'):
        """
        Method that cross-validates set of image-plane models obtained by
        modelling training samples on corresponding set of testing samples.

        :param modelfiles:
            Wildcard of file names ~ 'model_0i_0jofN.txt', where model in
            'model_0i_0jofN.txt' file is from modelling ``0j``-th training
            sample ('train_0jofN.FITS') with ``0i``-th model.

        :param testfiles:
            Wildcard of file names ~ 'test_0jofN.FITS'.

        :return:
            List of lists [modelfilename, CV-score, sigma_cv_score].
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
    def __init__(self, uvdata, model, average_freq=True, amp_only=False):
        error = uvdata.error(average_freq=average_freq)
        self.amp_only = amp_only
        self.model = model
        self.uv = uvdata.uvw[:, :2]
        stokes = model.stokes
        if average_freq:
            if stokes == 'I':
                self.uvdata = 0.5 * (uvdata.uvdata_freq_averaged[:, 0] +
                                     uvdata.uvdata_freq_averaged[:, 1])
                self.error = 0.5 * np.sqrt(error[:, 0] ** 2. +
                                           error[:, 1] ** 2.)
            elif stokes == 'RR':
                self.uvdata = uvdata.uvdata_freq_averaged[:, 0]
                self.error = error[:, 0]
            elif stokes == 'LL':
                self.uvdata = uvdata.uvdata_freq_averaged[:, 1]
                self.error = error[:, 1]
            else:
                raise Exception("Working with only I, RR or LL!")
        else:
            if stokes == 'I':
                self.uvdata = 0.5 * (uvdata.uvdata[:, 0] + uvdata.uvdata[:, 1])
            elif stokes == 'RR':
                self.uvdata = uvdata.uvdata[:, 0]
            elif stokes == 'LL':
                self.uvdata = uvdata.uvdata[:, 1]
            else:
                raise Exception("Working with only I, RR or LL!")

    def __call__(self, p):
        """
        Returns ln of likelihood for data and model with parameters ``p``.
        :param p:
        :return:
        """
        #print "calculating lnlik for ", p
        # Data visibilities and noise
        data = self.uvdata
        error = self.error
        # Model visibilities at uv-points of data
        assert(self.model.size == len(p))
        self.model.p = p[:]
        model_data = self.model.ft(self.uv)
        # ln of data likelihood
        if self.amp_only:
            model_amp = np.absolute(model_data)
            data_amp = np.absolute(data)
            # Use Rice distribution
            lnlik = np.log(model_amp) - 2. * np.log(error) -\
                    (model_amp ** 2. + data_amp ** 2.) / (2. * error ** 2.) +\
                    np.log(sp.special.iv(0.,
                                         (model_amp * data_amp / error ** 2.)))
        else:
            # Use complex normal distribution
            lnlik = -0.5 * np.log(2. * math.pi * error ** 2.) - \
                    (data - model_data) * (data - model_data).conj() / \
                    (2. * error ** 2.)
            lnlik = lnlik.real
        return lnlik.sum()


class LnPrior(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, p):
        self.model.p = p[:]
        lnpr = list()
        for component in self.model._components:
            component.p = p[:component.size]
            p = p[component.size:]
            #print "Got lnprior for component : ", component.lnpr
            lnpr.append(component.lnpr)

        return sum(lnpr)


class LnPost(object):
    def __init__(self, uvdata, model, average_freq=True):
        self.lnlik = LnLikelihood(uvdata, model, average_freq=average_freq)
        self.lnpr = LnPrior(model)

    def __call__(self, p):
        return self.lnlik(p[:]) + self.lnpr(p[:])


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
    bootstrap = Bootstrap(model,
                          uncal='/home/ilya/work/vlbi_errors/fits/1226+023_SPT-C1.FITS',
                          calibs=['PRELAST_CALIB.FITS'],
                          nonparametric=False,
                          split_scans=False,
                          use_V=True)

    #bootstrap.resample(outname='BOOT', nonparametric=False, split_scans=False,
    #                   use_V=True)

    bootstrap.run(outname='bootstrapped_data', n=10)
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
