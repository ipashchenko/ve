import copy
import numpy as np
from scipy.stats import t
from gains import Absorber
from utils import fit_2d_gmm, vcomplex, nested_ddict
import matplotlib


# TODO: Add 0.632-estimate of extra-sample error.
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
        # Dictionary with keys - baselines/IFs/Stokes & values - instances of
        # ``sklearn.mixture.GMM`` class fitted on residuals (Re&Im) of key
        # baselines
        self._residuals_fits = nested_ddict()
        # Dictionary with keys - baselines & values - boolean numpy arrays with
        # indexes of that baseline in ``UVData.uvdata`` array
        self._indxs_visibilities = dict()
        # Dictionary with keys - baselines & values - shapes of part for that
        # baseline in ``UVData.uvdata`` array
        self._shapes_visibilities = dict()
        # Dictionary with keys - baselines & values - tuples with centers of
        # real & imag residuals for that baseline
        self._residuals_centers = nested_ddict()
        self.get_baselines_info()

    def get_residuals(self):
        """
        Implements different residuals calculation.
        :return:
            Residuals between model and data.
        """
        raise NotImplementedError

    def get_baselines_info(self):
        """
        Count indexes of visibilities on each single baseline (for single IF &
        Stokes) in ``uvdata`` array.
        """
        for baseline in self.residuals.baselines:
            bl_data, indxs = self.residuals._choose_uvdata(baselines=[baseline])
            self._indxs_visibilities[baseline] = indxs
            self._shapes_visibilities[baseline] = np.shape(bl_data)

    def fit_residuals(self):
        """
        Fit residuals with Gaussian Mixture Model.

        :note:
            At each baseline residuals are fitted with Gaussian Mixture Model
            where number of mixture components is chosen based on BIC.
        """
        for baseline in self.residuals.baselines:
            baseline_data, _ =\
                self.residuals._choose_uvdata(baselines=[baseline])
            for if_ in range(baseline_data.shape[1]):
                for stokes in range(baseline_data.shape[2]):
                    data = baseline_data[:, if_, stokes]

                    # If data are zeros
                    if not np.any(data):
                        self._residuals_fits[baseline][if_][stokes] = None
                        continue

                    print "Baseline {}, IF {}, Stokes {}".format(baseline, if_,
                                                                 stokes)
                    clf = fit_2d_gmm(data)
                    self._residuals_fits[baseline][if_][stokes] = clf
                    x_c = np.sum(data.real) / len(data)
                    y_c = np.sum(data.imag) / len(data)
                    self._residuals_centers[baseline][if_][stokes] = (x_c, y_c)

    def plot_residuals(self, save_file, vis_range=None, ticks=None,
                       stokes='I'):
        """
        Plot histograms of the residuals.

        :param save_file:
            File to save plot.
        :param vis_range: (optional)
            Iterable of min & max range for plotting residuals Re & Im.
            Eg. ``[-0.15, 0.15]``. If ``None`` then choose one from data.
            (default: ``None``)
        :param ticks: (optional)
            Iterable of X-axis ticks to plot. Eg. ``[-0.1, 0.1]``. If ``None``
            then choose one from data. (default: ``None``)
        :param stokes:
            Stokes parameter to plot. (default: ``I``)
        """
        uvdata_r = self.residuals
        label_size = 6
        matplotlib.rcParams['xtick.labelsize'] = label_size
        matplotlib.rcParams['ytick.labelsize'] = label_size
        nrows = int(np.sqrt(2. * len(uvdata_r.baselines)))

        # Optionally choose range & ticks
        if vis_range is None:
            res = uvdata_r._choose_uvdata(stokes=stokes, freq_average=True)[0]
            range_ = min(abs(np.array([max(res.real), max(res.imag),
                                       min(res.real), min(res.imag)])))
            vis_range = [-range_, range_]
        if ticks is None:
            tick = min(abs(np.array(vis_range)))
            tick = float("{:.2f}".format(tick / 2.))
            ticks = [-tick, tick]

        fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=nrows,
                                               sharex=True, sharey=True)
        fig.set_size_inches(18.5, 18.5)
        matplotlib.pyplot.rcParams.update({'axes.titlesize': 'small'})
        i, j = 0, 0

        for baseline in uvdata_r.baselines:
            try:
                res = uvdata_r._choose_uvdata(baselines=[baseline],
                                              freq_average=True,
                                              stokes=stokes)[0]
                bins = min([10, np.sqrt(len(res.imag))])
                axes[i, j].hist(res.real, range=vis_range, color="#4682b4")
                axes[i, j].axvline(0.0, lw=1, color='r')
                axes[i, j].set_xticks(ticks)
                j += 1
                # Plot first row first
                if j // nrows > 0:
                    # Then second row, etc...
                    i += 1
                    j = 0
                bins = min([10, np.sqrt(len(res.imag))])
                axes[i, j].hist(res.imag, range=vis_range, color="#4682b4")
                axes[i, j].axvline(0.0, lw=1, color='r')
                axes[i, j].set_xticks(ticks)
                j += 1
                # Plot first row first
                if j // nrows > 0:
                    # Then second row, etc...
                    i += 1
                    j = 0
            except IndexError:
                break
        fig.savefig("{}".format(save_file), bbox_inches='tight', dpi=400)
        matplotlib.pyplot.close()

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
        if not nonparametric:
            print "Using parametric bootstrap"
            if not self._residuals_fits:
                print "Fitting residuals with GMM for each" \
                      " baseline/IF/Stokes..."
                self.fit_residuals()
            else:
                print "Residuals were already fitted with GMM on each" \
                      " baseline/IF/Stokes"
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
                 use_V=True, recenter=False):
        """
        Sample from residuals with replacement or sample from normal random
        noise and adds samples to model to form n bootstrap samples.

        :param outname:
            Output file name to save bootstrapped data.
        :param nonparametric: (optional)
            If ``True`` then use actual residuals between model and data. If
            ``False`` then use gaussian noise fitted to actual residuals for
            parametric bootstrapping. (default: ``True``)
        :params split_scans: (optional)
            Calculate noise for each scan individually? Not implemented yet.
        :param use_V: (optional)
            Use stokes V for calculation of noise? (default: ``True``)
        :param recenter: (optional)
            Boolean. Re-center sampled from fitted GMM residuals? (default:
            False)

        :return:
            Just save bootstrapped data to file with specified ``outname``.
        """

        if split_scans:
            raise NotImplementedError('Implement split_scans=True option!')

        # TODO: Put this to method
        noise_residuals = self.data.noise(split_scans=split_scans, use_V=use_V)
        # To make ``noise_residuals`` shape (nstokes, nif,)
        if use_V:
            nstokes = self.residuals.nstokes
            for key, value in noise_residuals.items():
                noise_residuals[key] = \
                    (value.repeat(nstokes).reshape((len(value), nstokes)))

        # Do resampling
        if not nonparametric:
            copy_of_model_data = copy.deepcopy(self.model_data)
            for baseline in self.residuals.baselines:
                indxs = self._indxs_visibilities[baseline]
                shape = self._shapes_visibilities[baseline]
                to_add = np.zeros(shape, complex)
                for if_ in range(shape[1]):
                    for stokes in range(shape[2]):
                        clf = self._residuals_fits[baseline][if_][stokes]

                        # If zeros on that baseline/IF/Stokes => just add zeros
                        if clf is None:
                            to_add[:, if_, stokes] += np.zeros(shape[0],
                                                               complex)
                            continue

                        clf_sample = clf.sample(shape[0])

                        if recenter:
                            # Center residuals
                            centers =\
                                self._residuals_centers[baseline][if_][stokes]
                            clf_sample[:, 0] -= centers[0]
                            clf_sample[:, 1] -= centers[1]

                        to_add[:, if_, stokes] = vcomplex(clf_sample[:, 0],
                                                          clf_sample[:, 1])

                # Add random variables to data on current baseline
                copy_of_model_data.uvdata[indxs] += to_add
                copy_of_model_data.sync()
        else:
            # TODO: should i resample all stokes and IFs together? Yes
            # Bootstrap from self.residuals._data. For each baseline.
            copy_of_model_data = copy.deepcopy(self.model_data)
            for baseline in self.residuals.baselines:
                # Find indexes of data from current baseline
                indxs = self._indxs_visibilities[baseline]
                indxs = np.where(indxs)[0]
                # Resample them
                indxs_ = np.random.choice(indxs, len(indxs))

                # Add to residuals.substitute(model)
                copy_of_model_data.uvdata[indxs] =\
                    copy_of_model_data.uvdata[indxs] +\
                    self.residuals.uvdata[indxs_]
                copy_of_model_data.sync()

        self.model_data.save(data=copy_of_model_data.hdu.data, fname=outname)

    def run(self, n, outname=['bootstrapped_data', '.fits'], nonparametric=True,
            split_scans=False, use_V=True, recenter=False):
        super(CleanBootstrap, self).run(n, outname, nonparametric,
                                        split_scans=split_scans, use_V=use_V,
                                        recenter=recenter)


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