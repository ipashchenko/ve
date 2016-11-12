import math
#from model import Model
import glob
import numpy as np
import scipy as sp
from utils import is_sorted


# FIXME: For ``average_freq=True`` got shitty results
class LnLikelihood(object):
    def __init__(self, uvdata, model, average_freq=True, amp_only=False,
                 use_V=False, use_weights=False):
        error = uvdata.error(average_freq=average_freq, use_V=use_V)
        self.amp_only = amp_only
        self.model = model
        self.data = uvdata
        stokes = model.stokes
        self.stokes = stokes
        self.average_freq = average_freq
        if average_freq:
            if stokes == 'I':
                self.uvdata = 0.5 * (uvdata.uvdata_freq_averaged[:, 0] +
                                               uvdata.uvdata_freq_averaged[:, 1])
                # self.error = 0.5 * np.sqrt(error[:, 0] ** 2. +
                #                            error[:, 1] ** 2.)
                self.error = 0.5 * (error[:, 0] +
                                              error[:, 1])
                if use_weights:
                    self.error = uvdata.errors_from_weights_masked_freq_averaged
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
                # (#, #IF)
                self.uvdata = 0.5 * (uvdata.uvdata[..., 0] + uvdata.uvdata[..., 1])
                # (#, #IF)
                # self.error = 0.5 * np.sqrt(error[..., 0] ** 2. +
                #                            error[..., 1] ** 2.)
                self.error = 0.5 * (error[..., 0] +
                                    error[..., 1])
            elif stokes == 'RR':
                self.uvdata = uvdata.uvdata[..., 0]
                self.error = error[..., 0]
            elif stokes == 'LL':
                self.uvdata = uvdata.uvdata[..., 1]
                self.error = error[..., 1]
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
        self.model.p = p[:self.model.size]
        model_data = self.model.ft(self.data.uv)
        k = 1.
        if self.stokes == 'I':
            k = 2.

        lnlik = k * (-np.log(2. * math.pi * (p[-1] + error ** 2.)) -
                     (data - model_data) * (data - model_data).conj() /
                     (2. * (p[-1] + error ** 2.)))
        lnlik = lnlik.real
        return np.ma.sum(lnlik)


class LnPrior(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, p):
        self.model.p = p[:-1]
        distances = list()
        for component in self.model._components:
            distances.append(np.sqrt(component.p[1] ** 2. +
                                     component.p[2] ** 2.))
        if not is_sorted(distances):
            print "Components are not sorted:("
            return -np.inf
        lnpr = list()
        for component in self.model._components:
            lnpr.append(component.lnpr)
        lnpr.append(sp.stats.uniform.logpdf(p[-1], 0, 2))

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
            return -np.inf
        return self.lnlik(p[:]) + lnpr


if __name__ == '__main__':
    from spydiff import import_difmap_model
    from uv_data import UVData
    from model import Model, Jitter
    uv_fits = '/home/ilya/code/vlbi_errors/bin_c1/0235+164.c1.2008_09_02.uvf_difmap'
    uvdata = UVData(uv_fits)
    # Create model
    mdl = Model(stokes='I')
    comps = import_difmap_model('0235+164.c1.2008_09_02.mdl',
                                '/home/ilya/code/vlbi_errors/bin_c1')
    comps[0].add_prior(flux=(sp.stats.uniform.logpdf, [0., 5], dict(),),
                       bmaj=(sp.stats.uniform.logpdf, [0, 1], dict(),),
                       e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
                       bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
    comps[1].add_prior(flux=(sp.stats.uniform.logpdf, [0., 3], dict(),),
                       bmaj=(sp.stats.uniform.logpdf, [0, 5], dict(),))
    mdl.add_components(*comps)

    # Create log of likelihood function
    lnlik = LnLikelihood(uvdata, mdl)
    lnpr = LnPrior(mdl)
    lnpost = LnPost(uvdata, mdl)
    p = mdl.p + [0.04]
    print lnpr(p)
    print lnlik(p)
    print lnpost(p)


    import emcee
    sampler = emcee.EnsembleSampler(100, len(p), lnpost)
    p0 = emcee.utils.sample_ball(p, [0.1, 0.01, 0.01, 0.01, 0.03, 0.01, 0.1, 0.01, 0.01, 0.1] + [0.001],
                                 size=100)
    pos, lnp, _ = sampler.run_mcmc(p0, 100)
    print "Acceptance fraction for initial burning: ", sampler.acceptance_fraction
    sampler.reset()
    # Run second burning
    pos, lnp, _ = sampler.run_mcmc(pos, 300)
    print "Acceptance fraction for second burning: ", sampler.acceptance_fraction
    sampler.reset()
    pos, lnp, _ = sampler.run_mcmc(pos, 500)
    print "Acceptance fraction for production: ", sampler.acceptance_fraction
