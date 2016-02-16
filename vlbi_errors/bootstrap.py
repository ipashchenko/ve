import copy
import numpy as np
from gains import Absorber


# TODO: detach format of data storage (use instances of UV_Data and Model
# classes as arguments in constructors. But for ``calibs`` argument of
# ``SelfCalBootstrap`` we need paths to files as gain info is in tables.
class Bootstrap(object):
    """
    Basic class for bootstrapping data using specified model.

    :param models:
        Iterable of ``Model`` subclass instances that represent model used for
        bootstrapping.. There should be only one (or zero) model for each stokes
        parameter. If there are two, say I-stokes models, then sum them firstly
        using ``Model.__add__``.
    :param uvdata:
        Instance of ``UVData`` class.
    """
    def __init__(self, models, uvdata):
        self.models = models
        self.data = uvdata
        self.model_data = copy.deepcopy(uvdata)
        self.model_data.substitute(models)
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

    # FIXME: Implement arbitrary output directory for bootstrapped data
    def run(self, n, outname=['bootstrapped_data', '.FITS'], nonparametric=True,
            **kwargs):
        """
        Generate ``n`` data sets.

        """
        for i in range(n):
            outname_ = outname[0] + '_' + str(i + 1).zfill(3) + outname[1]
            self.resample(outname=outname_, nonparametric=nonparametric,
                          **kwargs)


class CleanBootstrap(Bootstrap):
    """
    Class that implements bootstrapping of uv-data using model and residuals
    between data and model. Data are self-calibrated visibilities.

    :param models:
        Iterable of ``Model`` subclass instances that represent model used for
        bootstrapping.. There should be only one (or zero) model for each stokes
        parameter. If there are two, say I-stokes models, then sum them firstly
        using ``Model.__add__``.
    :param data:
        Path to FITS-file with uv-data (self-calibrated or not).
    """

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
            copy_of_model_data = copy.deepcopy(self.model_data)
            for baseline in self.residuals.baselines:
                # Find data from one baseline
                indxs = self.residuals._choose_uvdata(baselines=[baseline],
                                                      indx_only=True)[1]
                # Generate (len(indxs),8,4,) array of random variables
                # ``anormvars`` to add:
                lnormvars = list()
                for std in noise_residuals[baseline].flatten():
                    lnormvars.append(np.random.normal(std, size=len(indxs)))
                anormvars = np.dstack(lnormvars).reshape((len(indxs), nif,
                                                          nstokes,))
                # Add normal random variables to data on current baseline
                copy_of_model_data.uvdata[indxs] += anormvars
                copy_of_model_data.sync()
        else:
            # TODO: should i resample all stokes and IFs together? Yes
            # Bootstrap from self.residuals._data. For each baseline.
            copy_of_model_data = copy.deepcopy(self.model_data)
            for baseline in self.residuals.baselines:
                # Find indexes of data from current baseline
                indx = self.residuals._choose_uvdata(baselines=[baseline],
                                                     indx_only=True)[1]
                indx = np.where(indx)[0]
                # Resample them
                indx_ = np.random.choice(indx, len(indx))

                # Add to residuals.substitute(model)
                copy_of_model_data.uvdata[indx] =\
                    copy_of_model_data.uvdata[indx] +\
                    self.residuals.uvdata[indx_]
                copy_of_model_data.sync()

        self.model_data.save(data=copy_of_model_data.hdu.data, fname=outname)

    def run(self, n, outname=['bootstrapped_data', '.fits'], nonparametric=True,
            split_scans=False, use_V=True):
        super(CleanBootstrap, self).run(n, outname, nonparametric,
                                        split_scans=split_scans, use_V=use_V)


class SelfCalBootstrap(object):
    """
    Class that implements bootstrapping of uv-data using model and residuals
    between data and model. Residuals are difference between un-selfcalibrated
    uv-data and model visibilities multiplied by gains.

    :param models:
        Iterable of ``Model`` subclass instances that represent model used for
        bootstrapping.. There should be only one (or zero) model for each stokes
        parameter. If there are two, say I-stokes models, then sum them firstly
        using ``Model.__add__``.

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
    def __init__(self, models, data, calibs):
        self.models = models
        self.data = data
        self.calibs = calibs
        # Last self-calibrated data
        last_calib = UVData(calibs[-1])
        self.last_calib = last_calib

        model_data = copy.deepcopy(self.data)
        model_data.substitute(models)
        self.model_data = model_data
        self.residuals = self.get_residuals()

    def get_residuals(self):
        absorber = Absorber()
        absorber.absorb(self.calibs)
        return self.data - absorber * self.model_data


if __name__ == "__main__":
    # Clean bootstrap
    import os
    import copy
    curdir = os.getcwd()
    base_path = "/home/ilya/code/vlbi_errors/examples/"
    from uv_data import UVData
    uvdata = UVData(os.path.join(base_path, "2230+114.x.2006_02_12.uvf"))
    from from_fits import create_model_from_fits_file
    ccmodeli = create_model_from_fits_file(os.path.join(base_path, 'cc.fits'))
    model_data = copy.deepcopy(uvdata)
    model_data.substitute([ccmodeli])
    residuals = uvdata - model_data
    indx = uvdata._choose_uvdata(baselines=[258], indx_only=True)[1]
    indx = np.where(indx)[0]
    uvdata.uvdata[indx] = residuals.uvdata[indx]
    uvdata.sync()
    fname='/home/ilya/code/vlbi_errors/examples/s7.fits'
    if os.path.exists(fname):
        os.remove(fname)
    uvdata.save(fname=fname)
    uvdata_ = UVData(os.path.join(base_path, "s7.fits"))
    print uvdata_.hdu.data[indx]


    # # Self-calibration bootstrap
    # sc_sequence_files = ["sc_1.fits", "sc_2.fits", "sc_final.fits"]
    # uv_data = create_uvdata_from_fits_file("sc_1.fits")
    # scbootstrap = SelfCalBootstrap(ccmodel, uv_data, sc_sequence_files)
    # scbootstrap.run(100)