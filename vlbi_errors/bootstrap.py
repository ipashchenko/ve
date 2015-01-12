__author__ = 'ilya'
import copy
import numpy as np
from uv_data import open_fits
from gains import Absorber


# TODO: detach format of data storage (use instances of UV_Data and Model
# classes as arguments in constructors. But for ``calibs`` argument of
# ``SelfCalBootstrap`` we need paths to files as gain info is in tables.
class Bootstrap(object):
    """
    Basic class for bootstrapping data using specified model.

    :param model:
        Instance of ``Model`` class that represent model used for bootstrapping.
    :param data:
        Path to FITS-file with uv-data (self-calibrated or not).
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.model_data = copy.deepcopy(self.data)
        self.model_data.substitute([model])
        self.residuals = self.get_residuals()

    def get_residuals(self):
        """
        Implements different residuals calculation.
        :return:
            Residuals between model and data.
        """
        raise NotImplementedError

    def resample(self, outname, nonparametric=True, **kwargs):
        """
        Sample from residuals with replacement or sample from normal random
        noise fitted to residuals and add samples to model to form n bootstrap
        samples of data.

        :param outname:
            Output file name to save bootstrapped data.
        :param nonparametric (optional):
            If ``True`` then use actual residuals between model and data. If
            ``False`` then use gaussian noise fitted to actual residuals for
            parametric bootstrapping. (default: ``False``)
        :return:
            Just save bootstrapped data to file with specified ``outname``.
        """
        raise NotImplementedError

    def run(self, n, outname=['bootstrapped_data', '.FITS'], nonparametric=True,
            **kwargs):
        """
        Generate ``n`` data sets.

        """
        for i in range(n):
            outname_ = outname[0] + '_' + str(i + 1) + outname[1]
            self.resample(outname=outname_, nonparametric=nonparametric,
                          **kwargs)


class CleanBootstrap(Bootstrap):
    """
    Class that implements bootstrapping of uv-data using model and residuals
    between data and model. Data are self-calibrated visibilities.

    :param model:
        Instance of ``Model`` class that represent model used for bootstrapping.
    :param data:
        Path to FITS-file with uv-data (self-calibrated or not).
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.model_data = copy.deepcopy(self.data)
        self.model_data.substitute([model])
        self.residuals = self.get_residuals()

    def get_residuals(self):
        return self.data - self.model_data

    def resample(self, outname, nonparametric=True, split_scans=False,
                 use_V=True):
        """
        Sample from residuals with replacement or sample from normal random
        noise and adds samples to model to form n bootstrap samples.

        :param outname:
            Output file name to save bootstrapped data.
        :param outnam:
            Output file name.
        :param nonparametric (optional):
            If ``True`` then use actual residuals between model and data. If
            ``False`` then use gaussian noise fitted to actual residuals for
            parametric bootstrapping. (default: ``True``)
        :params split_scans (optional):
            Calculate noise for each scan individually? Not implemented yet.
        :param use_V (optional):
            Use stokes V for calculation of noise? (default: ``True``)
        :return:
            Just save bootstrapped data to file with specified ``outname``.
        """

        if split_scans:
            raise NotImplementedError('Implement split_scans=True option!')

        noise_residuals = self.residuals.noise(split_scans=split_scans,
                                               use_V=use_V)
        # To make ``noise_residuals`` shape (nstokes, nif,)
        if use_V:
            nstokes = self.residuals.nstokes
            nif = self.residuals.nif
            for key, value in noise_residuals.items():
                noise_residuals[key] = \
                    (value.repeat(nstokes).reshape((len(value), nstokes)))

        # Do resampling
        if not nonparametric:
            # Use noise_residuals as std of gaussian noise to add.
            # shape(noise_residuals[key]) = (8,4,)
            # shape(self.residuals._data[i]['hands']) = (8,4,)
            # for each baseline create (8,4,) normal random variables with
            # specified by noise_residuals[baseline] std
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
                indxs = np.where(self.residuals._data['baseline'] ==
                                 baseline)[0]
                data_to_resample = self.residuals._data[indxs]
                # Resample it
                resampled_data = np.random.choice(data_to_resample,
                                                  len(data_to_resample))

                # Add to residuals.substitute(model)
                self.model_data._data['hands'][indxs] = \
                    self.model_data._data['hands'][indxs] + \
                    resampled_data['hands']

        self.data.save(self.model_data._data, outname)

    def run(self, n, outname=['bootstrapped_data', '.FITS'], nonparametric=True,
            split_scans=False, use_V=True):
        super(CleanBootstrap, self).run(n, outname, nonparametric,
                                        split_scans=split_scans, use_V=use_V)


class SelfCalBootstrap(object):
    """
    Class that implements bootstrapping of uv-data using model and residuals
    between data and model. Residuals are difference between un-selfcalibrated
    uv-data and model visibilities multiplied by gains.

    :param model:
        Instance of ``Model`` class that represent model used for bootstrapping.

    :param data:
        Path to FITS-file with uv-data (self-calibrated or not).

    :param calibs:
        Iterable of paths to self-calibration sequence of FITS-files. That is
        used for constructing gain curves for each antenna. AIPS keep antenna
        gains solutions in each iteration of self-calibration circle in
        FITS-files that are calibrated. So in sequence of 1st, 2nd, ..., nth
        (final) files gain curve info lives in 1nd, ..., (n-1)th FITS-file.
        Sequence must be in order of self-calibration (longer solution times
        go first).

    :note:
        data argument is always data that is subject of substraction. So, it
        could be that first element of calibs argument is the same data.
    """
    # TODO: use super
    def __init__(self, model, data, calibs):
        self.model = model
        self.data = data
        self.calibs = calibs
        # Last self-calibrated data
        last_calib = open_fits(calibs[-1])
        self.last_calib = last_calib

        model_data = copy.deepcopy(self.data)
        model_data.substitute(model)
        self.model_data = model_data
        self.residuals = self.get_residuals()

    def get_residuals(self):
        absorber = Absorber()
        absorber.absorb(self.calibs)
        return self.data - absorber * self.model_data


if __name__ == "__main__":
    # Clean bootstrap
    uv_data = open_fits("1633+382.l22.2010_05_21.uvf")
    from model import CCModel
    ccmodel = CCModel(stokes='I')
    ccmodel.add_cc_from_fits("1633+382.l22.2010_05_21.icn.fits")
    cbootstrap = CleanBootstrap(ccmodel, uv_data)
    cbootstrap.run(100, outname=['1633', '.FITS'])

    # # Self-calibration bootstrap
    # sc_sequence_files = ["sc_1.fits", "sc_2.fits", "sc_final.fits"]
    # uv_data = open_fits("sc_1.fits")
    # scbootstrap = SelfCalBootstrap(ccmodel, uv_data, sc_sequence_files)
    # scbootstrap.run(100)
