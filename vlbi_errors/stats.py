import math
#from model import Model
import glob
import numpy as np
import scipy as sp
from utils import is_sorted


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

        print("modelfiles : " + str(modelfiles))
        print("testfiles : " + str(testfiles))

        result = list()

        for i in range(nmodels):
            print("Using models " +
                  str(modelfiles[ntest * i: ntest * (i + 1)]) +
                  " and testing sample " + str(testfiles))
            models = modelfiles[ntest * i: ntest * (i + 1)]
            cv_scores = list()
            for j, testfile in enumerate(testfiles):
                model = Model()
                model.add_from_txt(models[j], stoke=stokes)
                print("Using test file " + str(testfile))
                data = UVData(testfile)
                cv_score = data.cv_score(model, stokes=stokes)
                print("cv_score for one testing sample is " + str(cv_score))
                cv_scores.append(cv_score)

            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            print(mean_cv_score, std_cv_score)

            result.append(["model#" + str(i + 1), mean_cv_score, std_cv_score])

        return result


# FIXME: For ``average_freq=True`` got shitty results
class LnLikelihood(object):
    def __init__(self, uvdata, model, average_freq=True, amp_only=False,
                 use_V=False, use_weights=False):
        if use_weights and average_freq:
            error = uvdata.errors_from_weights_masked_freq_averaged
        elif use_weights and not average_freq:
            error = uvdata.errors_from_weights
        else:
            error = uvdata.error(average_freq=average_freq, use_V=use_V)
        self.amp_only = amp_only
        self.model = model
        self.data = uvdata
        stokes = model.stokes
        self.stokes = stokes
        self.average_freq = average_freq
        if average_freq:
            if stokes == 'I':
                # UVData.uvdata_freq_averaged is masked
                self.uvdata = 0.5 * (uvdata.uvdata_freq_averaged[:, 0] +
                                     uvdata.uvdata_freq_averaged[:, 1])
                self.error = np.hypot(error[:, 0], error[:, 1])/2

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
                self.uvdata = 0.5 * (uvdata.uvdata_weight_masked[..., 0] +
                                     uvdata.uvdata_weight_masked[..., 1])
                self.error = 0.5*np.hypot(error[..., 0], error[..., 1])
            elif stokes == 'RR':
                self.uvdata = uvdata.uvdata_weight_masked[..., 0]
                self.error = error[..., 0]
            elif stokes == 'LL':
                self.uvdata = uvdata.uvdata_weight_masked[..., 1]
                self.error = error[..., 1]
            else:
                raise Exception("Working with only I, RR or LL!")

    def __call__(self, p):
        """
        Returns ln of likelihood for data and model with parameters ``p``.
        :param p:
        :return:
        """
        data = self.uvdata
        error = self.error
        assert(self.model.size == len(p))
        self.model.p = p[:]
        model_data = self.model.ft(self.data.uv)
        if not self.average_freq:
            model_data = model_data[:, np.newaxis]
        if self.amp_only:
            model_amp = np.absolute(model_data)
            data_amp = np.absolute(data)
            # FIXME: double data for stokes I conjugate
            # Use Rice distribution
            lnlik = np.log(model_amp) - 2. * np.log(error) -\
                    (model_amp ** 2. + data_amp ** 2.) / (2. * error ** 2.) +\
                    np.log(sp.special.iv(0.,
                                         (model_amp * data_amp / error ** 2.)))
            result = lnlik.sum()
        else:
            # Complex difference
            diff = data - model_data
            # Real
            lnlik_real = -np.log(2.*np.pi*error**2.)-diff.real**2/(2.*error**2.)
            # Imaginary
            lnlik_imag = -np.log(2.*np.pi*error**2.)-diff.imag**2/(2.*error**2.)
            result = np.ma.sum(lnlik_real)+np.ma.sum(lnlik_imag)

        return result


class LnPrior(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, p):
        self.model.p = p[:]
        distances = list()
        for component in self.model._components:
            distances.append(np.sqrt(component.p[1] ** 2. +
                                     component.p[2] ** 2.))
        if not is_sorted(distances):
            print("Components are not sorted:(")
            return -np.inf
        lnpr = list()
        for component in self.model._components:
            # This is implemented in ``Model.p``
            # component.p = p[:component.size]
            # p = p[component.size:]
            #print "Got lnprior for component : ", component.lnpr
            lnpr.append(component.lnpr)

        return sum(lnpr)


class LnPost(object):
    def __init__(self, uvdata, model, average_freq=True, use_V=False,
                 use_weights=False):
        self.lnlik = LnLikelihood(uvdata, model, average_freq=average_freq,
                                  use_V=use_V, use_weights=use_weights)
        self.lnpr = LnPrior(model)

    def __call__(self, p):
        lnpr = self.lnpr(p[:])
        if not np.isfinite(lnpr):
            print("inf prior")
            return -np.inf
        return self.lnlik(p[:]) + lnpr


if __name__ == '__main__':
    # Test LS_estimates
    import sys
    from components import CGComponent, EGComponent
    from uv_data import UVData
    from model import Model
    try:
        from scipy.optimize import minimize, fmin
    except ImportError:
        sys.exit("install scipy for ml estimation")
    uv_fname = '/home/ilya/vlbi_errors/examples/L/1633+382/1633+382.l18.2010_05_21.uvf'
    uvdata = UVData(uv_fname)
    # Create model
    cg1 = EGComponent(1.0, -0.8, 0.2, .7, 0.5, 0)
    cg2 = CGComponent(0.8, 2.0, -.3, 2.3)
    cg3 = CGComponent(0.2, 5.0, .0, 2.)
    mdl = Model(stokes='I')
    mdl.add_components(cg1, cg2, cg3)
    # Create log of likelihood function
    lnlik = LnLikelihood(uvdata, mdl, average_freq=True, amp_only=False)
    # Nelder-Mead simplex algorithm
    p_ml = fmin(lambda p: -lnlik(p), mdl.p)
    # Various methods of minimization (some require jacobians)
    # TODO: Implement analitical grad of likelihood (it's gaussian)
    fit = minimize(lambda p: -lnlik(p), mdl.p, method='L-BFGS-B',
                   options={'maxiter': 30000, 'maxfev': 1000000, 'xtol': 0.001,
                            'ftol': 0.001, 'approx_grad': True},
                   bounds=[(0., 2), (None, None), (None, None), (0., +np.inf),
                           (0., 1.), (None, None),
                           (0., 2), (None, None), (None, None), (0., 5),
                           (0., 1), (None, None), (None, None), (0., 20)])
    if fit['success']:
        print("Succesful fit!")
        p_ml = fit['x']
        print(p_ml)
