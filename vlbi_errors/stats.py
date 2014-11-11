import math
from model_old import Model
from gains import Absorber
from uv_data import open_fits
import glob
import copy
import numpy as np
from utils import is_sorted


class AddNoise(object):
    """
    Class that implements the simplest method of errors estimates.
    """
    def __init__(self, n=100):
        pass


class Bootstrap(object):
    """
    Class that implements bootstrapping of uv-data.

    Sample with replacement (if ``nonparametric=True``) from residuals between
    model with gains and unself-calibrated uv-data or use normal zero mean
    random variable with std estimated from that residuals for each baseline (or
    even scan - but that is not implemented yet).

    :param model:
        Instance of ``Model`` class that represent model used for bootstrapping.

    :param uncal:
        Path to FITS-file with uv-data not yet self-calibrated.

    :param calibs:
        Iterable of paths to self-calibration sequence of FITS-files. That is
        used for constructing gain curves for each antenna. AIPS keep antenna
        gains solutions in each iteration of self-calibration circle in
        FITS-files that are calibrated. So in sequence of 1st, 2nd, ..., nth
        (final) files gain curve info lives in 1nd, ..., (n-1)th FITS-file.
        Sequence must be in order of self-calibration (longer solution times
        go first).
    """

    def __init__(self, model, uncal=None, calibs=None):
        self.model = model
        self.uncal = uncal
        self.calibs = calibs

        absorber = Absorber()
        absorber.absorb(calibs)
        uncal = open_fits(uncal)
        last_calib = open_fits(calibs[-1])
        residuals = uncal - absorber * last_calib
        self.residuals = residuals
        model_uv = copy.deepcopy(residuals)
        model_uv.substitute(model)
        self.model_uv = model_uv
        self.last_calib = last_calib
        self.last_calib.save(residuals._data, 'BOOT_residuals')
        self.last_calib.save(model_uv._data, 'BOOT_model_uv')

    def resample(self, outname=None, nonparametric=False, split_scans=False,
                 use_V=True):
        """
        Sample from residuals with replacement or sample from normal random
        noise and adds samples to model to form n bootstrap samples.

        :param nonparametric (optional):
            If ``True`` then use actual residuals between model with gains and
            unself-calibrated uv-data. If ``False`` then use gaussian noise fitted
            to actual residuals for parametric bootstrapping. (default: ``False``)

        : params split_scans (optional):

        : param use_V (optional):
        """

        if split_scans:
            raise NotImplementedError('Implement split_scans=True option!')

        noise_residuals = self.residuals.noise(split_scans=split_scans,
                                               use_V=use_V)
        if use_V:
            nstokes = self.residuals.nstokes
            nif = self.residuals.nif
            for key, value in noise_residuals.items():
                noise_residuals[key] =\
                    (value.repeat(nstokes).reshape((len(value), nstokes)))
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
                indxs = np.where(self.residuals._data['baseline'] ==
                                 baseline)[0]
                data_to_add_normvars = self.residuals._data[indxs]
                # Generate (len(indxs),8,4,) array of random variables
                # ``anormvars`` to add:
                lnormvars = list()
                for std in noise_residuals[baseline].flatten():
                    lnormvars.append(np.random.normal(std, size=len(indxs)))
                anormvars = np.dstack(lnormvars).reshape((len(indxs), nif,
                                                          nstokes,))
                # Add normal random variables to data on current baseline
                data_to_add_normvars['hands'] += anormvars
        else:
            # TODO: should i resample all stokes and IFs together? Yes
            # Bootstrap from self.residuals._data. For each baseline.
            for baseline in self.residuals.baselines:
                # Find data from one baseline
                indxs = np.where(self.residuals._data['baseline'] == baseline)[0]
                data_to_resample = self.residuals._data[indxs]
                # Resample it
                resampled_data = np.random.choice(data_to_resample,
                                                  len(data_to_resample))

                # Add to residuals.substitute(model)
                self.model_uv._data['hands'][indxs] =\
                    self.model_uv._data['hands'][indxs] + \
                    resampled_data['hands']

        self.last_calib.save(self.model_uv._data, outname)

    def run(self, outname='bootstrapped_data', n=10, nonparametric=True,
            split_scans=False, use_V=True):
        """
        Generate ``n`` data sets.

        """
        for i in range(n):
            outname_ = outname + '_' + str(i + 1) + '.FITS'
            self.resample(outname=outname_, nonparametric=nonparametric,
                          split_scans=split_scans, use_V=use_V)


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
    def __init__(self, uvdata, model, average_freq=True):
        error = uvdata.error(average_freq=average_freq)
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
        # Data visibilities and noise
        data = self.uvdata
        error = self.error
        # Model visibilities at uv-points of data
        self.model.p = p
        model_data = self.model.ft(self.uv)
        # ln of data likelihood
        lnlik = -0.5 * np.log(2. * math.pi * error ** 2.) - \
                (data - model_data) * (data - model_data).conj() / \
                (2. * error ** 2.)
        lnlik = lnlik.real
        return lnlik.sum()


class LnPrior(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, p):
        self.model.p = p
        distances = list()
        for component in self.model._components:
            distances.append(component.r)
        if not is_sorted(distances):
            print "Components are not sorted."
            return -np.inf
        else:
            print "Components are sorted. OK!"
        lnpr = list()
        for component in self.model._components:
            print "Passing to component ", component
            print "parameters : ", p[:component.size]
            component.p = p[:component.size]
            p = p[component.size:]
            print "Got lnprior for component : ", component.lnpr
            lnpr.append(component.lnpr)

        return sum(lnpr)


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
