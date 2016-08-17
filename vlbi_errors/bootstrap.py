import copy
import numpy as np
from gains import Absorber
from utils import (fit_2d_gmm, vcomplex, nested_ddict, make_ellipses,
                   baselines_2_ants, find_outliers_2d_mincov,
                   find_outliers_2d_dbscan, find_outliers_dbscan, fit_kde)
import matplotlib
matplotlib.use('Agg')
label_size = 6
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size


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
        self.noise_residuals = None
        # Dictionary with keys - baseline, #IF, #Stokes and values - instances
        # of ``sklearn.neighbors.KernelDensity`` class fitted on the residuals
        # (Re&Im) of key baselines
        self._residuals_fits = nested_ddict()
        # Dictionary with keys - baseline, #scan, #IF, #Stokes and values -
        # instances of ``sklearn.neighbors.KernelDensity`` class fitted on the
        # residuals (Re&Im)
        self._residuals_fits_scans = nested_ddict()
        # Dictionary with keys - baselines & values - tuples with centers of
        # real & imag residuals for that baseline
        self._residuals_centers = nested_ddict()
        self._residuals_centers_scans = nested_ddict()
        # Dictionary with keys - baseline, #IF, #Stokes and value - boolean
        # numpy array with outliers
        self._residuals_outliers = nested_ddict()
        # Dictionary with keys - baseline, #scan, #IF, #Stokes and value -
        # boolean numpy array with outliers
        self._residuals_outliers_scans = nested_ddict()

    def get_residuals(self):
        """
        Implements different residuals calculation.
        :return:
            Residuals between model and data.
        """
        raise NotImplementedError

    def plot_residuals_trio(self, outname, split_scans=True, freq_average=False,
                            IF=None, stokes=['RR']):
        if IF is None:
            IF = range(self.residuals.nif)
        if stokes is None:
            stokes = range(self.residuals.nstokes)
        else:
            stokes_list = list()
            for stoke in stokes:
                print "Parsing {}".format(stoke)
                print self.residuals.stokes
                stokes_list.append(self.residuals.stokes.index(stoke))
            stokes = stokes_list

        print "Plotting IFs {}".format(IF)
        print "Plotting Stokes {}".format(stokes)

        for baseline in self.residuals.baselines:
            print baseline
            ant1, ant2 = baselines_2_ants([baseline])

            if split_scans:
                try:
                    for i, indxs in enumerate(self.residuals._indxs_baselines_scans[baseline]):
                        # Complex (#, #IF, #stokes)
                        data = self.residuals.uvdata[indxs]
                        # weights = self.residuals.weights[indxs]

                        if freq_average:
                            raise NotImplementedError
                            # # FIXME: Aberage w/o outliers
                            # # Complex (#, #stokes)
                            # data = np.mean(data, axis=1)
                            # for stoke in stokes:
                            #     # Complex 1D array to plot
                            #     data_ = data[:, stoke]
                            #     fig, axes = matplotlib.pyplot.subplots(nrows=2,
                            #                                            ncols=2)
                            #     matplotlib.pyplot.rcParams.update({'axes.titlesize':
                            #                                         'small'})
                            #     axes[1, 0].plot(data_.real, data_.imag, '.k')
                            #     axes[1, 0].axvline(0.0, lw=0.2, color='g')
                            #     axes[1, 0].axhline(0.0, lw=0.2, color='g')
                            #     axes[0, 0].hist(data_.real, bins=10,
                            #                     label="Re {}-{}".format(ant1, ant2),
                            #                     color="#4682b4")
                            #     legend = axes[0, 0].legend(fontsize='small')
                            #     axes[0, 0].axvline(0.0, lw=1, color='g')
                            #     axes[1, 1].hist(data_.imag, bins=10, color="#4682b4",
                            #                     orientation='horizontal',
                            #                     label="Im {}-{}".format(ant1, ant2))
                            #     legend = axes[1, 1].legend(fontsize='small')
                            #     axes[1, 1].axhline(0.0, lw=1, color='g')
                            #     fig.savefig("res_2d_bl{}_st{}_scan_{}".format(baseline, stoke, i),
                            #                 bbox_inches='tight', dpi=400)
                            #     matplotlib.pyplot.close()
                        else:
                            for IF_ in IF:
                                for stoke in stokes:
                                    # Complex 1D array to plot
                                    data_ = data[:, IF_, stoke]
                                    # weigths_ = weights[:, IF_, stoke]
                                    # data_pw = data_[weigths_ > 0]
                                    data_pw = data_[self.residuals._pw_indxs[indxs, IF_, stokes]]
                                    data_nw = data_[self.residuals._nw_indxs[indxs, IF_, stokes]]
                                    data_out = data_pw[self._residuals_outliers_scans[baseline][i][IF_][stoke]]
                                    # data_nw = data_[weigths_ <= 0]
                                    fig, axes = matplotlib.pyplot.subplots(nrows=2,
                                                                           ncols=2)
                                    matplotlib.pyplot.rcParams.update({'axes.titlesize':
                                                                           'small'})
                                    axes[1, 0].plot(data_.real, data_.imag, '.k')
                                    axes[1, 0].plot(data_nw.real, data_nw.imag, '.', color='orange')
                                    axes[1, 0].plot(data_out.real, data_out.imag, '.r')
                                    try:
                                        x_c, y_c = self._residuals_centers_scans[baseline][i][IF_][stoke]
                                        axes[1, 0].plot(x_c, y_c, '.y')
                                    except ValueError:
                                        x_c, y_c = 0., 0.
                                    axes[1, 0].axvline(0.0, lw=0.2, color='g')
                                    axes[1, 0].axhline(0.0, lw=0.2, color='g')
                                    axes[0, 0].hist(data_.real, bins=10,
                                                    label="Re "
                                                          "{}-{}".format(ant1,
                                                                         ant2),
                                                    color="#4682b4",
                                                    histtype='stepfilled',
                                                    alpha=0.3,
                                                    normed=True)
                                    try:
                                        clf_re = self._residuals_fits_scans[baseline][i][IF_][stoke][0]
                                        sample = np.linspace(np.min(data_.real) - x_c,
                                                             np.max(data_.real) - x_c,
                                                             1000)
                                        pdf = np.exp(clf_re.score_samples(sample[:, np.newaxis]))
                                        axes[0, 0].plot(sample + x_c, pdf, color='blue',
                                                        alpha=0.5, lw=2, label='kde')
                                    # ``AttributeError`` when no ``clf`` for that
                                    # baseline, IF, Stokes
                                    except (ValueError, AttributeError):
                                        pass
                                    legend = axes[0, 0].legend(fontsize='small')
                                    axes[0, 0].axvline(0.0, lw=1, color='g')
                                    axes[1, 1].hist(data_.imag, bins=10,
                                                    color="#4682b4",
                                                    orientation='horizontal',
                                                    histtype='stepfilled',
                                                    alpha=0.3, normed=True,
                                                    label="Im "
                                                          "{}-{}".format(ant1,
                                                                         ant2))
                                    try:
                                        clf_im = self._residuals_fits_scans[baseline][i][IF_][stoke][1]
                                        sample = np.linspace(np.min(data_.imag) + y_c,
                                                             np.max(data_.imag) + y_c,
                                                             1000)
                                        pdf = np.exp(clf_im.score_samples(sample[:, np.newaxis]))
                                        axes[1, 1].plot(pdf, sample - y_c, color='blue',
                                                        alpha=0.5, lw=2, label='kde')
                                    # ``AttributeError`` when no ``clf`` for that
                                    # baseline, IF, Stokes
                                    except (ValueError, AttributeError):
                                        pass
                                    legend = axes[1, 1].legend(fontsize='small')
                                    axes[1, 1].axhline(0.0, lw=1, color='g')
                                    fig.savefig("{}_ant1_{}_ant2_{}_stokes_{}_IF_{}_scan_{}".format(outname,
                                        ant1, ant2, self.residuals.stokes[stoke],
                                        IF_, i), bbox_inches='tight', dpi=400)
                                    matplotlib.pyplot.close()
                # If ``self.residuals._indxs_baselines_scans[baseline] = None``
                except TypeError:
                    continue
            else:
                indxs = self.residuals._indxs_baselines[baseline]
                # Complex (#, #IF, #stokes)
                data = self.residuals.uvdata[indxs]
                # weights = self.residuals.weights[indxs]

                if freq_average:
                    raise NotImplementedError
                else:
                    for IF_ in IF:
                        for stoke in stokes:
                            print "Stokes {}".format(stoke)
                            # Complex 1D array to plot
                            data_ = data[:, IF_, stoke]
                            # weigths_ = weights[:, IF_, stoke]
                            # data_pw = data_[weigths_ > 0]
                            data_pw = data_[self.residuals._pw_indxs[indxs, IF_, stoke]]
                            data_nw = data_[self.residuals._nw_indxs[indxs, IF_, stoke]]
                            data_out = data_pw[self._residuals_outliers[baseline][IF_][stoke]]
                            # data_nw = data_[weigths_ <= 0]
                            fig, axes = matplotlib.pyplot.subplots(nrows=2,
                                                                   ncols=2)
                            matplotlib.pyplot.rcParams.update({'axes.titlesize':
                                                                'small'})
                            axes[1, 0].plot(data_.real, data_.imag, '.k')
                            axes[1, 0].plot(data_out.real, data_out.imag, '.r')
                            axes[1, 0].plot(data_nw.real, data_nw.imag, '.', color='orange')
                            try:
                                x_c, y_c = self._residuals_centers[baseline][IF_][stoke]
                                axes[1, 0].plot(x_c, y_c, '.y')
                            except ValueError:
                                x_c, y_c = 0., 0.
                            axes[1, 0].axvline(0.0, lw=0.2, color='g')
                            axes[1, 0].axhline(0.0, lw=0.2, color='g')
                            axes[0, 0].hist(data_.real, bins=20,
                                            label="Re {}-{}".format(ant1, ant2),
                                            color="#4682b4",
                                            histtype='stepfilled', alpha=0.3,
                                            normed=True)
                            try:
                                clf_re = self._residuals_fits[baseline][IF_][stoke][0]
                                sample = np.linspace(np.min(data_.real) - x_c,
                                                     np.max(data_.real) - x_c,
                                                     1000)
                                pdf = np.exp(clf_re.score_samples(sample[:, np.newaxis]))
                                axes[0, 0].plot(sample + x_c, pdf, color='blue',
                                                alpha=0.5, lw=2, label='kde')
                            # ``AttributeError`` when no ``clf`` for that
                            # baseline, IF, Stokes
                            except (ValueError, AttributeError):
                                pass
                            legend = axes[0, 0].legend(fontsize='small')
                            axes[0, 0].axvline(0.0, lw=1, color='g')
                            axes[1, 1].hist(data_.imag, bins=20,
                                            color="#4682b4",
                                            orientation='horizontal',
                                            histtype='stepfilled', alpha=0.3,
                                            normed=True,
                                            label="Im {}-{}".format(ant1, ant2))
                            try:
                                clf_im = self._residuals_fits[baseline][IF_][stoke][1]
                                sample = np.linspace(np.min(data_.imag) + y_c,
                                                     np.max(data_.imag) + y_c,
                                                     1000)
                                pdf = np.exp(clf_im.score_samples(sample[:, np.newaxis]))
                                axes[1, 1].plot(pdf, sample - y_c, color='blue',
                                                alpha=0.5, lw=2, label='kde')
                            # ``AttributeError`` when no ``clf`` for that
                            # baseline, IF, Stokes
                            except (ValueError, AttributeError):
                                pass
                            legend = axes[1, 1].legend(fontsize='small')
                            axes[1, 1].axhline(0.0, lw=1, color='g')
                            fig.savefig("{}_ant1_{}_ant2_{}_stokes_{}_IF_{}_second".format(outname,
                                ant1, ant2, self.residuals.stokes[stoke], IF_),
                                bbox_inches='tight', dpi=400)
                            matplotlib.pyplot.close()

    def find_outliers_in_residuals(self, split_scans=False):
        """
        Method that search outliers in residuals

        :param split_scans:
            Boolean. Find outliers on each scan separately?
        """
        print "Searching for outliers in residuals..."
        for baseline in self.residuals.baselines:
            indxs = self.residuals._indxs_baselines[baseline]
            baseline_data = self.residuals.uvdata[indxs]
            # If searching outliers in baseline data
            if not split_scans:
                for if_ in range(baseline_data.shape[1]):
                    for stokes in range(baseline_data.shape[2]):
                        # Complex array with visibilities for given baseline,
                        # #IF, Stokes
                        data = baseline_data[:, if_, stokes]
                        # weigths = self.residuals.weights[indxs, if_, stokes]

                        # Use only valid data with positive weight
                        data_pw = data[self.residuals._pw_indxs[indxs, if_, stokes]]
                        data_nw = data[self.residuals._nw_indxs[indxs, if_, stokes]]
                        print "NW {}".format(np.count_nonzero(data_nw))

                        # If data are zeros
                        if not np.any(data_pw):
                            continue

                        print "Baseline {}, IF {}, Stokes {}".format(baseline,
                                                                     if_,
                                                                     stokes)
                        outliers_re = find_outliers_dbscan(data_pw.real, 1., 5)
                        outliers_im = find_outliers_dbscan(data_pw.imag, 1., 5)
                        outliers_1d = np.logical_or(outliers_re, outliers_im)
                        outliers_2d = find_outliers_2d_dbscan(data_pw, 1.5, 5)
                        self._residuals_outliers[baseline][if_][stokes] =\
                            np.logical_or(outliers_1d, outliers_2d)

            # If searching outliers on each scan
            else:
                # Searching each scan on current baseline
                # FIXME: Use zero centers for shitty scans?
                if self.residuals.scans_bl[baseline] is None:
                    continue
                for i, scan_indxs in enumerate(self.residuals.scans_bl[baseline]):
                    scan_uvdata = self.residuals.uvdata[scan_indxs]
                    for if_ in range(scan_uvdata.shape[1]):
                        for stokes in range(scan_uvdata.shape[2]):
                            # Complex array with visibilities for given
                            # baseline, #scan, #IF, Stokes
                            data = scan_uvdata[:, if_, stokes]
                            # weigths = self.residuals.weights[scan_indxs, if_,
                            #                                  stokes]

                            # Use only valid data with positive weight
                            data_pw = data[self.residuals._pw_indxs[scan_indxs, if_, stokes]]
                            data_nw = data[self.residuals._nw_indxs[scan_indxs, if_, stokes]]
                            print "NW {}".format(np.count_nonzero(data_nw))

                            # If data are zeros
                            if not np.any(data_pw):
                                continue

                            print "Baseline {}, scan {}, IF {}," \
                                  " Stokes {}".format(baseline, i, if_, stokes)
                            outliers_re = find_outliers_dbscan(data_pw.real, 1., 5)
                            outliers_im = find_outliers_dbscan(data_pw.imag, 1., 5)
                            outliers_1d = np.logical_or(outliers_re, outliers_im)
                            outliers_2d = find_outliers_2d_dbscan(data_pw, 1.5, 5)
                            self._residuals_outliers_scans[baseline][i][if_][stokes] = \
                                np.logical_or(outliers_1d, outliers_2d)

    # TODO: Use only data without outliers
    def find_residuals_centers(self, split_scans):
        """
        Calculate centers of residuals for each baseline[/scan]/IF/stokes.
        """
        print "Finding centers"
        for baseline in self.residuals.baselines:
            # Find centers for baselines only
            if not split_scans:
                indxs = self.residuals._indxs_baselines[baseline]
                baseline_data = self.residuals.uvdata[indxs]
                for if_ in range(baseline_data.shape[1]):
                    for stokes in range(baseline_data.shape[2]):
                        data = baseline_data[:, if_, stokes]
                        # weigths = self.residuals.weights[indxs, if_, stokes]

                        # Use only valid data with positive weight
                        # data_pw = data[weigths > 0]
                        data_pw = data[self.residuals._pw_indxs[indxs, if_, stokes]]
                        # data_nw = data[self.residuals._nw_indxs[indxs, if_, stokes]]

                        # If data are zeros
                        if not np.any(data_pw):
                            continue

                        print "Baseline {}, IF {}, Stokes {}".format(baseline, if_,
                                                                     stokes)
                        outliers = self._residuals_outliers[baseline][if_][stokes]
                        x_c = np.sum(data_pw.real[~outliers]) / np.count_nonzero(~outliers)
                        y_c = np.sum(data_pw.imag[~outliers]) / np.count_nonzero(~outliers)
                        print "Center: ({:.4f}, {:.4f})".format(x_c, y_c)
                        self._residuals_centers[baseline][if_][stokes] = (x_c, y_c)
            # Find residuals centers on each scan
            else:
                # Searching each scan on current baseline
                # FIXME: Use zero centers for shitty scans?
                if self.residuals.scans_bl[baseline] is None:
                    continue
                for i, scan_indxs in enumerate(self.residuals.scans_bl[baseline]):
                    scan_uvdata = self.residuals.uvdata[scan_indxs]
                    for if_ in range(scan_uvdata.shape[1]):
                        for stokes in range(scan_uvdata.shape[2]):
                            data = scan_uvdata[:, if_, stokes]
                            # weigths = self.residuals.weights[scan_indxs, if_,
                            #                                  stokes]

                            # Use only valid data with positive weight
                            # data_pw = data[weigths > 0]
                            data_pw = data[self.residuals._pw_indxs[scan_indxs, if_, stokes]]

                            # If data are zeros
                            if not np.any(data_pw):
                                continue

                            print "Baseline {}, #scan {}, IF {}," \
                                  " Stokes {}".format(baseline, i, if_, stokes)
                            outliers = self._residuals_outliers_scans[baseline][i][if_][stokes]
                            x_c = np.sum(data_pw.real[~outliers]) / np.count_nonzero(~outliers)
                            y_c = np.sum(data_pw.imag[~outliers]) / np.count_nonzero(~outliers)
                            print "Center: ({:.4f}, {:.4f})".format(x_c, y_c)
                            self._residuals_centers_scans[baseline][i][if_][stokes] = (x_c, y_c)

    # FIXME: Use real Stokes parameters as keys.
    def fit_residuals_gmm(self):
        """
        Fit residuals with Gaussian Mixture Model.

        :note:
            At each baseline residuals are fitted with Gaussian Mixture Model
            where number of mixture components is chosen based on BIC.
        """
        for baseline in self.residuals.baselines:
            baseline_data, _ = \
                self.residuals._choose_uvdata(baselines=[baseline])
            for if_ in range(baseline_data.shape[1]):
                for stokes in range(baseline_data.shape[2]):
                    data = baseline_data[:, if_, stokes]

                    # If data are zeros
                    if not np.any(data):
                        continue

                    print "Baseline {}, IF {}, Stokes {}".format(baseline, if_,
                                                                 stokes)
                    print "Shape: {}".format(baseline_data.shape)
                    try:
                        clf = fit_2d_gmm(data)
                    # This occurs when baseline has 1 point only
                    except ValueError:
                        continue
                    self._residuals_fits[baseline][if_][stokes] = clf

    # FIXME: Use real Stokes parameters as keys.
    def fit_residuals_kde(self, split_scans, combine_scans, recenter):
        """
        Fit residuals with Gaussian Kernel Density.

        :param split_scans:
            Boolean. Fit to each scan of baseline independently?
        :param combine_scans:
            Boolean. Combine re-centered scans on each baseline before fit?
        :param recenter:
            Boolean. Recenter residuals before fit?

        :note:
            At each baseline/scan residuals are fitted with Kernel Density
            Model.
        """
        print "Fitting residuals"
        if combine_scans:
            raise NotImplementedError

        for baseline in self.residuals.baselines:
            # If fitting baseline data
            if not split_scans:
                indxs = self.residuals._indxs_baselines[baseline]
                baseline_data = self.residuals.uvdata[indxs]
                for if_ in range(baseline_data.shape[1]):
                    for stokes in range(baseline_data.shape[2]):
                        data = baseline_data[:, if_, stokes]
                        # weigths = self.residuals.weights[indxs, if_, stokes]

                        # Use only valid data with positive weight
                        # data_pw = data[weigths > 0]
                        data_pw = data[self.residuals._pw_indxs[indxs, if_, stokes]]
                        # If data are zeros
                        if not np.any(data_pw):
                            continue

                        # Don't count outliers
                        data_pw = data_pw[~self._residuals_outliers[baseline][if_][stokes]]

                        print "Baseline {}, IF {}, Stokes {}".format(baseline, if_,
                                                                     stokes)
                        if recenter:
                            x_c, y_c = self._residuals_centers[baseline][if_][stokes]
                            data_pw -= x_c - 1j * y_c
                        try:
                            clf_re = fit_kde(data_pw.real)
                            clf_im = fit_kde(data_pw.imag)
                        # This occurs when baseline has 1 point only
                        except ValueError:
                            continue
                        self._residuals_fits[baseline][if_][stokes] = (clf_re,
                                                                       clf_im)
            # If fitting each scan independently
            else:
                if self.residuals.scans_bl[baseline] is None:
                    continue
                for i, scan_indxs in enumerate(self.residuals.scans_bl[baseline]):
                    scan_uvdata = self.residuals.uvdata[scan_indxs]
                    for if_ in range(scan_uvdata.shape[1]):
                        for stokes in range(scan_uvdata.shape[2]):
                            data = scan_uvdata[:, if_, stokes]
                            # weigths = self.residuals.weights[scan_indxs, if_, stokes]

                            # Use only valid data with positive weight
                            # data_pw = data[weigths > 0]
                            data_pw = data[self.residuals._pw_indxs[scan_indxs, if_, stokes]]

                            # If data are zeros
                            if not np.any(data_pw):
                                continue

                            # Don't count outliers
                            data_pw = data_pw[~self._residuals_outliers_scans[baseline][i][if_][stokes]]

                            print "Baseline {}, Scan {}, IF {}, Stokes" \
                                  " {}".format(baseline, i, if_, stokes)
                            if recenter:
                                x_c, y_c = self._residuals_centers_scans[baseline][i][if_][stokes]
                                data_pw -= x_c - 1j * y_c
                            try:
                                clf_re = fit_kde(data_pw.real)
                                clf_im = fit_kde(data_pw.imag)
                            # This occurs when scan has 1 point only
                            except ValueError:
                                continue
                            self._residuals_fits_scans[baseline][i][if_][stokes] = (clf_re, clf_im)

    def get_residuals_noise(self, split_scans, use_V):
        """
        Estimate noise of the residuals using stokes V or successive
        differences approach. For each baseline or even scan.

        :param split_scans:
            Boolean. Estimate noise std for each baseline scan individually?
        :param use_V:
            Boolean. Use Stokes V visibilities to estimate noise std?

        :return:
            Dictionary with keys - baseline numbers & values - arrays of shape
            ([#scans], #IF, #stokes). First dimension is #scans if option
            ``split_scans=True`` is used.
        """
        # Dictionary with keys - baseline numbers & values - arrays of shape
        # ([#scans], #IF, [#stokes]). It means (#scans, #IF) if
        # ``split_scans=True`` & ``use_V=True``, (#IF, #stokes) if
        # ``split_scans=False`` & ``use_V=False`` etc.
        noise_residuals = self.residuals.noise(split_scans=split_scans,
                                               use_V=use_V)
        print "Getting noise residuals ", noise_residuals
        # To make ``noise_residuals`` shape ([#scans], #IF, #stokes) for
        # ``use_V=True`` option.
        if use_V:
            nstokes = self.residuals.nstokes
            for key, value in noise_residuals.items():
                print "key", key
                print "value", np.shape(value)
                shape = list(np.shape(value))
                shape.extend([nstokes])
                value = np.tile(value, nstokes)
                value = value.reshape(shape)
                noise_residuals[key] = value

        return noise_residuals

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
        nrows = int(np.ceil(np.sqrt(2. * len(uvdata_r.baselines))))

        # Optionally choose range & ticks
        if vis_range is None:
            res = uvdata_r._choose_uvdata(stokes=stokes, freq_average=True)
            range_ = min(abs(np.array([max(res.real), max(res.imag),
                                       min(res.real), min(res.imag)])))
            range_ = float("{:.3f}".format(range_))
            vis_range = [-range_, range_]
        print("vis_range", vis_range)
        if ticks is None:
            tick = min(abs(np.array(vis_range)))
            tick = float("{:.3f}".format(tick / 2.))
            ticks = [-tick, tick]
        print("ticks", ticks)

        fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=nrows,
                                               sharex=True, sharey=True)
        fig.set_size_inches(18.5, 18.5)
        matplotlib.pyplot.rcParams.update({'axes.titlesize': 'small'})
        i, j = 0, 0

        for baseline in uvdata_r.baselines:
            try:
                res = uvdata_r._choose_uvdata(baselines=[baseline],
                                              freq_average=True,
                                              stokes=stokes)
                bins = min([10, np.sqrt(len(res.imag))])
                ant1, ant2 = baselines_2_ants([baseline])
                axes[i, j].hist(res.real, range=vis_range, color="#4682b4",
                                label="Re {}-{}".format(ant1, ant2))
                axes[i, j].axvline(0.0, lw=1, color='r')
                axes[i, j].set_xticks(ticks)
                legend = axes[i, j].legend(fontsize='small')
                j += 1
                # Plot first row first
                if j // nrows > 0:
                    # Then second row, etc...
                    i += 1
                    j = 0
                bins = min([10, np.sqrt(len(res.imag))])
                axes[i, j].hist(res.imag, range=vis_range, color="#4682b4",
                                label="Im {}-{}".format(ant1, ant2))
                legend = axes[i, j].legend(fontsize='small')
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

    def plot_residuals_2d(self, vis_range=None, ticks=None):
        """
        Plot 2D distribution of complex residuals.

        :param vis_range: (optional)
            Iterable of min & max range for plotting residuals Re & Im.
            Eg. ``[-0.15, 0.15]``. If ``None`` then choose one from data.
            (default: ``None``)
        :param ticks: (optional)
            Iterable of X-axis ticks to plot. Eg. ``[-0.1, 0.1]``. If ``None``
            then choose one from data. (default: ``None``)
        """
        uvdata_r = self.residuals

        for baseline in uvdata_r.baselines:
            # n_if = self._residuals_fits[baseline]
            # n_stokes = self._residuals_fits[baseline]
            nrows = 4
            fig, axes = matplotlib.pyplot.subplots(nrows=4,
                                                   ncols=4,
                                                   sharex=True,
                                                   sharey=True)
            i, j = 0, 0
            fig.set_size_inches(18.5, 18.5)
            matplotlib.pyplot.rcParams.update({'axes.titlesize':
                                                   'small'})
            n_if = len(self._residuals_fits[baseline].keys())
            for if_ in self._residuals_fits[baseline].keys():
                n_stokes = len([val for val in
                                self._residuals_fits[baseline][if_].values() if
                                val is not None])
                for stoke in self._residuals_fits[baseline][if_].keys():
                    stoke_par = uvdata_r.stokes[stoke]

                    try:
                        clf = self._residuals_fits[baseline][if_][stoke]
                        if clf is None:
                            # No fitted residuals for this IF/Stokes
                            continue
                        res = uvdata_r._choose_uvdata(baselines=[baseline],
                                                      IF=if_+1,
                                                      stokes=stoke_par)[0][:, 0]
                        print "Baseline {}, IF {}, Stokes {}".format(baseline,
                                                                     if_,
                                                                     stoke)
                        print "Shape: {}".format(res.shape)
                        re = res.real
                        im = res.imag
                        reim = np.vstack((re, im)).T
                        y = clf.predict(reim)
                        for i_mix in range(clf.n_components):
                            color = "rgbyk"[i_mix]
                            re_ = re[np.where(y == i_mix)]
                            im_ = im[np.where(y == i_mix)]
                            axes[i, j].scatter(re_, im_, color=color)
                            make_ellipses(clf, axes[i, j])
                        # axes[i, j].set_xticks(ticks)
                        # axes[i, j].set_xlim(vis_range)
                        # axes[i, j].set_ylim(vis_range)
                        # axes[i, j].set_xticks(ticks)
                        # axes[i, j].set_yticks(ticks)
                        j += 1
                        # Plot first row first
                        if j // nrows > 0:
                            # Then second row, etc...
                            i += 1
                            j = 0
                    except IndexError:
                        break
            fig.savefig("res_2d_{}_{}_{}".format(baseline, if_, stoke),
                        bbox_inches='tight', dpi=400)
            matplotlib.pyplot.close()

    def resample(self, outname, nonparametric, split_scans, recenter, use_kde,
                 use_v, combine_scans):
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
    def run(self, n, nonparametric, split_scans, recenter, use_kde, use_v,
            combine_scans, outname=['bootstrapped_data', '.FITS']):
        """
        Generate ``n`` data sets.

        :note:
            Several steps are made before re-sampling ``n`` times:

            * First, outliers are found for each baseline or even scan (using
            DBSCAN clustering algorithm).
            * Centers of the residuals for each baselines or optionally
            scans (when ``split_scans=True``) are found excluding outliers.
            * In parametric bootstrap (when ``nonparameteric=False``) noise
            density estimates for each baseline/scan are maid using
            ``sklearn.neighbors.KernelDensity`` fits to Re & Im re-centered
            visibility data with gaussian kernel and bandwidth optimized by
            ``sklearn.grid_search.GridSearchCV`` with 5-fold CV.
            This is when ``use_kde=True``. Otherwise residuals are supposed to
            be distributed with gaussian density and it's std is estimated
            directly.

            Then, in parametric bootstrap re-sampling is maid by adding samples
            from fitted KDE (for ``use_kde=True``) or zero-mean Gaussian
            distribution with std of the residuals to model visibility data
            ``n`` times. In non-parametric case re-sampling is maid by sampling
            with replacement from re-centered residuals (with outliers
            excluded).

        """
        # Find outliers in baseline/scan data
        if not split_scans:
            if not self._residuals_outliers:
                print "Finding outliers in baseline's data..."
                self.find_outliers_in_residuals(split_scans=False)
            else:
                print "Already found outliers in baseline's data..."
        else:
            if not self._residuals_centers_scans:
                print "Finding outliers in scan's data..."
                self.find_outliers_in_residuals(split_scans=True)
            else:
                print "Already found outliers in scan's data..."

        # Find residuals centers
        if recenter:
            self.find_residuals_centers(split_scans=split_scans)

        # Fit residuals for parametric case
        if not nonparametric:
            # Using KDE estimate of residuals density
            if use_kde:
                print "Using parametric bootstrap"
                if not split_scans and not self._residuals_fits:
                    print "Fitting residuals with KDE for each" \
                          " baseline/IF/Stokes..."
                    self.fit_residuals_kde(split_scans=split_scans,
                                           combine_scans=combine_scans,
                                           recenter=recenter)
                if split_scans and not self._residuals_fits_scans:
                    print "Fitting residuals with KDE for each" \
                          " baseline/scan/IF/Stokes..."
                    self.fit_residuals_kde(split_scans=split_scans,
                                           combine_scans=combine_scans,
                                           recenter=recenter)
                if not split_scans and self._residuals_fits:
                    print "Residuals were already fitted with KDE on each" \
                          " baseline/IF/Stokes"
                if split_scans and self._residuals_fits_scans:
                    print "Residuals were already fitted with KDE on each" \
                          " baseline/scan/IF/Stokes"
            # Use parametric gaussian estimate of residuals density
            else:
                if not self.noise_residuals:
                    print "Estimating gaussian STDs on each baseline[/scan]..."
                    self.noise_residuals = self.get_residuals_noise(split_scans,
                                                                    use_v)
                else:
                    print "Gaussian STDs for each baseline[/scan] are already" \
                          " estimated"

        # Resampling is done in subclasses
        for i in range(n):
            outname_ = outname[0] + '_' + str(i + 1).zfill(3) + outname[1]
            self.resample(outname=outname_, nonparametric=nonparametric,
                          split_scans=split_scans, recenter=recenter,
                          use_kde=use_kde, use_v=use_v,
                          combine_scans=combine_scans)


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

    def resample_baseline_nonparametric(self, baseline, copy_of_model_data,
                                        recenter):
        print "Doing nonparametric baseline - for baseline {}".format(baseline)

        # Boolean array that defines indexes of current baseline data
        baseline_indxs = self.residuals._indxs_baselines[baseline]
        # FIXME: Here iterate over keys with not None values
        for if_ in range(self.residuals.nif):
            for stokes in range(self.residuals.nstokes):
                baseline_indxs_ = baseline_indxs.copy()
                # Boolean array that defines indexes of outliers in indexes of
                # current baseline data
                outliers = self._residuals_outliers[baseline][if_][stokes]
                pw_indxs = self.residuals._pw_indxs[baseline_indxs, if_, stokes]
                # If some Stokes parameter has no outliers calculation - pass it
                if isinstance(outliers, dict):
                    continue
                # Baseline indexes of inliers
                indxs = np.where(baseline_indxs_)[0][pw_indxs][~outliers]
                # If for some combinations baseline/IF/Stokes no data to
                # resample - pass it
                if not indxs.size:
                    continue
                # Resample them
                indxs_ = np.random.choice(indxs,
                                          np.count_nonzero(baseline_indxs))

                # Add to residuals.substitute(model)
                copy_of_model_data.uvdata[baseline_indxs, if_, stokes] = \
                    copy_of_model_data.uvdata[baseline_indxs, if_, stokes] + \
                    self.residuals.uvdata[indxs_, if_, stokes]
                if recenter:
                    x_c, y_c = self._residuals_centers[baseline][if_][stokes]
                    copy_of_model_data.uvdata[baseline_indxs, if_, stokes] -=\
                        x_c + 1j*y_c

                copy_of_model_data.sync()

    def resample_baseline_nonparametric_splitting_scans(self, baseline,
                                                        copy_of_model_data,
                                                        recenter):
        # Find indexes of data from current baseline
        scan_indxs = self.residuals._indxs_baselines_scans[baseline]
        for i, scan_indx in enumerate(scan_indxs):
            for if_ in range(self.residuals.nif):
                for stokes in range(self.residuals.nstokes):
                    outliers = self._residuals_outliers_scans[baseline][i][if_][stokes]
                    pw_indxs = self.residuals._pw_indxs[scan_indx, if_, stokes]
                    # Find what int indexes corresponds to `inliers`
                    indxs = np.where(scan_indx)[0][pw_indxs][~outliers]
                    # Resample them
                    indxs_ = np.random.choice(indxs, np.count_nonzero(scan_indx))

                    # Add to residuals.substitute(model)
                    copy_of_model_data.uvdata[scan_indx, if_, stokes] = \
                        copy_of_model_data.uvdata[scan_indx, if_, stokes] + \
                        self.residuals.uvdata[indxs_, if_, stokes]
                    if recenter:
                        x_c, y_c = self._residuals_centers_scans[baseline][i][if_][stokes]
                        copy_of_model_data.uvdata[scan_indx, if_, stokes] -= \
                            x_c + 1j*y_c
            copy_of_model_data.sync()

    def resample_baseline_parametric(self, baseline, copy_of_model_data,
                                     recenter, use_kde):
        indxs = self.residuals._indxs_baselines[baseline]
        shape = self.residuals._shapes_baselines[baseline]
        to_add = np.zeros(shape, complex)
        # FIXME: Here iterate over keys with not None values
        for if_ in self._residuals_fits[baseline]:
            for stokes in self._residuals_fits[baseline][if_]:
                if not use_kde:
                    std = self.noise_residuals[baseline][if_, stokes]

                    # FIXME: For zero scans std - use IF's averages!
                    try:
                        sample = np.random.normal(loc=0., scale=std,
                                                  size=2 * len(to_add))
                    except ValueError:
                        continue
                    # Add center coordinates if not re-centering
                    if not recenter:
                        # Center residuals
                        centers = \
                            self._residuals_centers[baseline][if_][stokes]
                        # FIXME: Handle shitty scans
                        try:
                            sample[: len(to_add)] += centers[0]
                            sample[len(to_add):] += centers[1]
                        except TypeError:
                            pass

                    sample = vcomplex(sample[: len(to_add)],
                                      sample[len(to_add):])
                else:
                    kde_re, kde_im = self._residuals_fits[baseline][if_][stokes]
                    sample_re = kde_re.sample(len(to_add))
                    sample_im = kde_im.sample(len(to_add))
                    sample = sample_re[:, 0] + 1j * sample_im[:, 0]

                to_add[:, if_, stokes] = sample

        # Add random variables to data on current baseline
        copy_of_model_data.uvdata[indxs] += to_add
        copy_of_model_data.sync()

    def resample_baseline_parametric_splitting_scans(self, baseline,
                                                     copy_of_model_data,
                                                     recenter, use_kde):
        for baseline in self.residuals.baselines:
            scan_indxs = self.residuals._indxs_baselines_scans[baseline]
            # FIXME: Use baseline's noise when scan is shitty!
            if scan_indxs is None:
                continue
            scan_shapes = self.residuals._shapes_baselines_scans[baseline]
            for i, scan_indx in enumerate(scan_indxs):
                to_add = np.zeros(scan_shapes[i], complex)
                for if_ in range(self.residuals.nif):
                    for stokes in range(self.residuals.nstokes):
                        if not use_kde:
                            std = self.noise_residuals[baseline][i, if_, stokes]

                            # FIXME: For zero scans std - use IF's averages!
                            try:
                                sample = np.random.normal(loc=0., scale=std,
                                                          size=2 * len(to_add))
                            except ValueError:
                                continue
                            # Add center coordinates if not re-centering
                            if not recenter:
                                # Center residuals
                                centers = \
                                    self._residuals_centers_scans[baseline][i][if_][stokes]
                                # FIXME: Handle shitty scans
                                try:
                                    sample[: len(to_add)] += centers[0]
                                    sample[len(to_add):] += centers[1]
                                except TypeError:
                                    pass

                            sample = vcomplex(sample[: len(to_add)],
                                              sample[len(to_add):])
                        else:
                            kde_re, kde_im =\
                                self._residuals_fits_scans[baseline][i][if_][stokes]
                            # If no KDE was fitted for some IF/Stokes - pass it
                            try:
                                sample_re = kde_re.sample(len(to_add))
                                sample_im = kde_im.sample(len(to_add))
                            except AttributeError:
                                continue
                            sample = sample_re[:, 0] + 1j * sample_im[:, 0]

                        to_add[:, if_, stokes] = sample

                # Add random variables to data on current scan
                copy_of_model_data.uvdata[scan_indx] += to_add
                copy_of_model_data.sync()

    def resample(self, outname, nonparametric, split_scans, recenter,
                 use_kde, use_v, combine_scans=False):
        """
        Sample from residuals with replacement or sample from normal random
        noise and adds samples to model to form n bootstrap samples.

        :return:
            Just save bootstrapped data to file with specified ``outname``.
        """

        # Model to add resamples
        copy_of_model_data = copy.deepcopy(self.model_data)

        # If do resampling for different scans independently
        if split_scans:
            print "Splitting scans"
            # Do parametric bootstrap
            if not nonparametric:
                print "doing parametric"
                for baseline in self.residuals.baselines:
                    self.resample_baseline_parametric_splitting_scans(baseline,
                                                                      copy_of_model_data,
                                                                      recenter,
                                                                      use_kde)
            # Do nonparametric bootstrap
            else:
                print "doing nonparametric"
                # Bootstrap from self.residuals._data. For each baseline.
                for baseline in self.residuals.baselines:
                    self.resample_baseline_nonparametric_splitting_scans(baseline,
                                                                         copy_of_model_data)

        # If do resampling for baselines
        else:
            # Do parametric bootstrap
            if not nonparametric:
                print "Doing parametric"
                for baseline in self.residuals.baselines:
                    self.resample_baseline_parametric(baseline,
                                                      copy_of_model_data,
                                                      recenter, use_kde)
            # Do nonparametric bootstrap
            else:
                print "doing nonparametric"
                # TODO: should i resample all stokes and IFs together? Yes
                # Bootstrap from self.residuals._data. For each baseline.
                for baseline in self.residuals.baselines:
                    self.resample_baseline_nonparametric(baseline,
                                                         copy_of_model_data,
                                                         recenter)

        self.model_data.save(data=copy_of_model_data.hdu.data, fname=outname)

    def run(self, n, nonparametric, split_scans=False, recenter=True,
            use_kde=True, use_v=True, combine_scans=False,
            outname=['bootstrapped_data', '.fits']):
        super(CleanBootstrap, self).run(n, nonparametric,
                                        split_scans=split_scans,
                                        recenter=recenter, use_kde=use_kde,
                                        use_v=use_v,
                                        combine_scans=combine_scans,
                                        outname=outname)


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
    from spydiff import import_difmap_model
    # data_dir = '/home/ilya/code/vlbi_errors/bin'
    # mdl_fname = '0125+487_L.mod_cir'
    # uv_fname = '0125+487_L.uvf_difmap'
    data_dir = '/home/ilya/vlbi_errors/test_boot'
    mdl_fname = '1253-055.q1.2010_01_26.mdl'
    uv_fname = '1253-055.Q1.2010_01_26.UV_CAL'
    from uv_data import UVData
    uvdata = UVData(os.path.join(data_dir, uv_fname))
    comps = import_difmap_model(mdl_fname, data_dir)
    from model import Model
    model = Model(stokes='I')
    model.add_components(*comps)
    boot = CleanBootstrap([model], uvdata)
    boot.find_outliers_in_residuals(split_scans=False)
    boot.find_residuals_centers(split_scans=False)
    boot.fit_residuals_kde(split_scans=False, combine_scans=False,
                           recenter=True)
    boot.plot_residuals_trio('test_name', freq_average=False, split_scans=True)

    # boot.run(10, nonparametric=False, split_scans=False,
    #          recenter=True, use_kde=True, use_v=True, combine_scans=False,
    #          outname=['boot_out', '.FITS'])


    # # Self-calibration bootstrap
    # sc_sequence_files = ["sc_1.fits", "sc_2.fits", "sc_final.fits"]
    # uv_data = create_uvdata_from_fits_file("sc_1.fits")
    # scbootstrap = SelfCalBootstrap(ccmodel, uv_data, sc_sequence_files)
    # scbootstrap.run(100)
