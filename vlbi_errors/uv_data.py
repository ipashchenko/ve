import math
import copy
import numpy as np
import pyfits as pf
from collections import OrderedDict
from numpy.lib.recfunctions import stack_arrays
from utils import baselines_2_ants, index_of, get_uv_correlations

try:
    import pylab
except ImportError:
    pylab = None

vec_complex = np.vectorize(np.complex)
vec_int = np.vectorize(np.int)


class UVData(object):

    def __init__(self, fname):
        self.fname = fname
        self.hdulist = pf.open(fname)
        self.hdu = self.hdulist[0]
        self._stokes_dict = {'RR': 0, 'LL': 1, 'RL': 2, 'LR': 3}
        self.learn_data_structure(self.hdu)
        self._error = None

    def save(self, data=None, fname=None):
        fname = fname or self.fname
        # Save data
        if data is None:
            self.hdulist.writeto(fname)
        else:
            # TODO: save data: #groups has changed
            new_hdu = pf.GroupsHDU(data)
            # PyFits updates header using given data (``GCOUNT``) anyway
            new_hdu.header = self.hdu.header

            hdulist = pf.HDUList([new_hdu])
            for hdu in self.hdulist[1:]:
                hdulist.append(hdu)
            hdulist.writeto(fname)

    # TODO: for IDI extend this method
    def learn_data_structure(self, hdu):
        # Learn parameters
        par_dict = OrderedDict()
        for i, par in enumerate(hdu.data.names):
            par_dict.update({par: i})
        self.par_dict = par_dict
        # Create mapping of FITS CTYPEi ``i`` number to dimensions of PyFits
        # hdu.data[`DATA`] (hdu.data.data) numpy.ndarray.
        data_dict = OrderedDict()
        data_dict.update({'GROUP': (0, hdu.header['GCOUNT'])})
        for i in range(hdu.header['NAXIS'], 1, -1):
            data_dict.update({hdu.header['CTYPE' + str(i)]:
                                     (hdu.header['NAXIS'] - i + 1,
                                      hdu.header['NAXIS' + str(i)])})
        # Save shape and dimensions of data recarray
        self.data_dict = data_dict
        self.nif = data_dict['IF'][1]
        self.nstokes = data_dict['STOKES'][1]

        # Create dictionary with necessary slices
        slices_dict = OrderedDict()
        for key, value in data_dict.items():
            if value[1] == 1 and key not in ['IF', 'STOKES']:
                slices_dict.update({key: 0})
            else:
                slices_dict.update({key: slice(None, None)})
        self.slices_dict = slices_dict

    @property
    def uvdata(self):
        """
        Returns (#groups, #if, #stokes) complex numpy.ndarray.
        """
        result = self.hdu.data.data[tuple(self.slices_dict.values())]
        return result[..., 0] + 1j*result[..., 1]

    @uvdata.setter
    def uvdata(self, uvdata):
        self.hdu.data.data[tuple(self.slices_dict.values())][..., 0] = uvdata.real
        self.hdu.data.data[tuple(self.slices_dict.values())][..., 1] = uvdata.imag

    @property
    def uvdata_freq_averaged(self):
        """
        Shortcut for ``self.data['hands']`` averaged in IFs.

        :returns:
            ``self.data['hands']`` averaged in IFs, with shape (#vis,
            #stokes).
        """
        if self.nif > 1:
            result = np.mean(self.uvdata, axis=1)
        # FIXME: if self.nif=1 then np.mean for axis=1 will remove this
        # dimension. So don't need this else
        else:
            result = self.uvdata[:, 0, :]
        return result

    @property
    def baselines(self):
        """
        Returns list of baselines numbers.

        :return:
            List of baselines numbers.
        """
        result = list(set(self.hdu.columns[self.par_dict['BASELINE']].array))
        return vec_int(sorted((result)))

    @property
    def antennas(self):
        """
        Returns list of antennas numbers.

        :return:
            List of antennas numbers.
        """
        return baselines_2_ants(self.baselines)

    @property
    def scans(self):
        """
        Returns list of times that separates different scans. If NX table is
        present in the original

        :return:
            np.ndarray with shape (#scans, 2,) with start & stop time for each
                of #scans scans.
        """
        try:
            indx = self.hdulist.index_of('AIPS NX')
            print "Found AIPS NX table!"
        except KeyError:
            indx = None
            print "No AIPS NX table are found!"

        if indx is not None:
            nx_hdu = self.hdulist[indx]
            scans = (np.vstack((nx_hdu.data['TIME'], nx_hdu.data['TIME'] +
                                nx_hdu.data['TIME INTERVAL']))).T
        else:
            scans = None

        return scans

    @property
    def scans_bl(self):
        """
        Calculate scans for each baseline separately.

        It won't coincide with UVData.scans because different baselines have
        different number of scans.

        :return:
            Dictionary with scans borders for each baseline.
        """
        scans_dict = dict()
        all_times = self.hdu.columns[self.par_dict['DATE']].array
        all_a, all_b = np.histogram(all_times[1:] - all_times[:-1])
        for bl in self.baselines:
            print "Processing baseline ", bl
            bl_indxs = self._choose_uvdata(baselines=bl)[1]
            bl_times = self.hdu.columns[self.par_dict['DATE']].array[bl_indxs]
            a, b = np.histogram(bl_times[1:] - bl_times[:-1])
            # If baseline consists only of 1 scan
            if b[-1] < all_b[1]:
                scans_dict.update({bl: np.atleast_2d([bl_times[0],
                                                      bl_times[-1]])})
            # If baseline has > 1 scan
            else:
                scan_borders = bl_times[(np.where((bl_times[1:] -
                                                   bl_times[:-1]) > b[1])[0])]
                scans_list = [[bl_times[0], scan_borders[0]]]
                for i in range(len(scan_borders) - 1):
                    scans_list.append([float(bl_times[np.where(bl_times == scan_borders[i])[0] + 1]),
                                       scan_borders[i + 1]])
                scans_list.append([float(bl_times[np.where(bl_times == scan_borders[i + 1])[0] + 1]),
                                   bl_times[-1]])
                scans_dict.update({bl: np.asarray(scans_list)})

        return scans_dict

    @property
    def uvw(self):
        """
        Shortcut for all (u, v, w)-elements of self.

        :return:
            Numpy.ndarray with shape (N, 3,), where N is the number of (u, v, w)
            points.
        """
        u = self.hdu.columns[self.par_dict['UU--']].array
        v = self.hdu.columns[self.par_dict['VV--']].array
        w = self.hdu.columns[self.par_dict['WW--']].array
        return np.vstack((u, v, w)).T

    @property
    def uv(self):
        """
        Shortcut for (u, v) -coordinates of self.

        :return:
            Numpy.ndarray with shape (N, 2,), where N is the number of (u, v, w)
            points.
        """
        return self.uvw[:, :2]

    def _choose_uvdata(self, times=None, baselines=None, IF=None, stokes=None,
                       freq_average=False):
        """
        Method that returns chosen data from ``_data`` numpy structured array
        based on user specified parameters.

        All checks on IF and stokes are made here in one place. This method is
        used by methods that retrieve data, plotting methods.

        :param times (optional):
            Container with start & stop time or ``None``. If ``None`` then use
            all time. (default: ``None``)

        :param baselines (optional):
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes (optional):
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use all available stokes.
            (default: ``None``)

        :return:
            Two numpy.ndarrays, where first array is array of data with shape
            (#N, #IF, #STOKES) and second array is boolean numpy array of
            indexes of data in original PyFits record array.
        """
        uvdata = self.uvdata

        # TODO: create general method for retrieving indexes of structured array
        # where the specified fields are equal to specified values. Also
        # consider using different types of fields: interval values (time),
        # equality to some values (baseline, IF), etc.
        if baselines is None:
            baselines = self.baselines
            indxs = np.ones(len(uvdata), dtype=bool)
        else:
            indxs = np.zeros(len(uvdata), dtype=bool)
            baselines_list = list()
            # If ``baselines`` is iterable
            try:
                baselines_list.extend(baselines)
            # If ``baselines`` is not iterable (int)
            except TypeError:
                baselines_list.append(baselines)

            # Check that given baseline numbers are among existing ones
            assert(set(baselines_list).issubset(self.baselines))

            # Find indexes of structured array (in zero dim) with given
            # baselines
            for baseline in baselines_list:
                # TODO: Vectorize that shit
                indxs = np.logical_or(indxs, self.hdu.columns[self.par_dict['BASELINE']].array == baseline)

        # If we are given some time interval then find among ``indxs`` only
        # those from given time interval
        if times is not None:
            # Assert that ``times`` consists of start and stop
            assert(len(times) == 2)
            lower_indxs = self.hdu.columns[self.par_dict['DATE']].array <\
                          times[1]
            high_indxs = self.hdu.columns[self.par_dict['DATE']].array >\
                         times[0]
            time_indxs = np.logical_and(lower_indxs, high_indxs)
            indxs = np.logical_and(indxs, time_indxs)

        if IF is None:
            IF = np.arange(self.nif) + 1
        else:
            IF_list = list()
            # If ``IF`` is iterable
            try:
                IF_list.extend(IF)
            # If ``IF`` is not iterable (int)
            except TypeError:
                IF_list.append(IF)
            IF = np.array(IF_list)

        if not set(IF).issubset(np.arange(1, self.nif + 1)):
            raise Exception('Choose IF numbers from ' + str(1) + ' to ' +
                            str(self.nif))
        IF -= 1

        if stokes == 'I':
            # I = 0.5 * (RR + LL)
            result = 0.5 * (uvdata[indxs][:, IF, 0] + uvdata[indxs][:, IF, 1])

        elif stokes == 'V':
            # V = 0.5 * (RR - LL)
            result = 0.5 * (uvdata[indxs][:, IF, 0] - uvdata[indxs][:, IF, 1])

        elif stokes == 'Q':
            # V = 0.5 * (LR + RL)
            result = 0.5 * (uvdata[indxs][:, IF, 3] + uvdata[indxs][:, IF, 2])

        elif stokes == 'U':
            # V = 0.5 * 1j * (LR - RL)
            result = 0.5 * 1j * (uvdata[indxs][:, IF, 3] - uvdata[indxs][:, IF, 2])

        elif stokes in self._stokes_dict.keys():
            result = uvdata[indxs][:, IF, self._stokes_dict[stokes]]

        elif stokes is None:
            # When fetching all stokes then need [:, None] tips
            result = uvdata[indxs][:, IF[:, None], np.arange(self.nstokes)]

        else:
            raise Exception('Allowed stokes parameters: I, Q, U, V, RR, LL, RL,'
                            'LR or (default) all [RR, LL, RL, LR].')

        if freq_average:
            result = np.mean(result, axis=1)

        return result, indxs

    # TODO: use qq = scipy.stats.probplot((v-mean(v))/std(v), fit=0) then
    # plot(qq[0], qq[1]) - how to check normality
    # TODO: should i fit gaussians? - np.std <=> scipy.stats.norm.fit()! NO FIT!
    def noise(self, split_scans=False, use_V=True, average_freq=False):
        """
        Calculate noise for each baseline. If ``split_scans`` is True then
        calculate noise for each scan too. If ``use_V`` is True then use stokes
        V data (`RR`` - ``LL``) for computation assuming no signal in V. Else
        use successive differences approach (Brigg's dissertation).

        :param split_scans (optional):
            Should we calculate noise for each scan? (default: ``False``)

        :param use_V (optional):
            Use stokes V data (``RR`` - ``LL``) to calculate noise assuming no
            signal in stokes V? If ``False`` then use successive differences
            approach (see Brigg's dissertation). (default: ``True``)

        :param average_freq (optional):
            Use IF-averaged data for calculating noise? (default: ``False``)

        :return:
            Dictionary with keys that are baseline numbers and values are
            arrays of noise std for each IF (if ``use_V==True``), or array with
            shape (#stokes, #if) with noise std values for each IF for each
            stokes parameter (eg. RR, LL, ...).
        """
        baselines_noises = dict()
        if use_V:
            # Calculate dictionary {baseline: noise} (if split_scans is False)
            # or {baseline: [noises]} if split_scans is True.
            if not split_scans:
                for baseline in self.baselines:
                    baseline_uvdata = self._choose_uvdata(baselines=baseline)[0]
                    if average_freq:
                        baseline_uvdata = np.mean(baseline_uvdata, axis=1)
                    v = (baseline_uvdata[..., 0] - baseline_uvdata[..., 1]).real
                    mask = ~np.isnan(v)
                    baselines_noises[baseline] =\
                        np.asarray(np.std(np.ma.array(v, mask=np.invert(mask)).data,
                                          axis=0))
            else:
                # Use each scan
                for baseline in self.baselines:
                    baseline_noise = list()
                    for scan in self.scans_bl[baseline]:
                        # (#obs in scan, #nif, #nstokes,)
                        scan_baseline_uvdata = self._choose_uvdata(baselines=baseline,
                                                                   times=(scan[0],
                                                                          scan[1],))[0]
                        if average_freq:
                            # (#obs in scan, #nstokes,)
                            scan_baseline_uvdata = np.mean(scan_baseline_uvdata,
                                                           axis=1)
                        v = (scan_baseline_uvdata[..., 0] -
                             scan_baseline_uvdata[..., 1]).real
                        mask = ~np.isnan(v)
                        scan_noise = np.asarray(np.std(np.ma.array(v,
                                                                   mask=np.invert(mask)).data,
                                                       axis=0))
                        baseline_noise.append(scan_noise)
                    baselines_noises[baseline] = np.asarray(baseline_noise)

        else:
            if not split_scans:
                for baseline in self.baselines:
                    baseline_uvdata = self._choose_uvdata(baselines=baseline)[0]
                    if average_freq:
                        baseline_uvdata = np.mean(baseline_uvdata, axis=1)
                    differences = (baseline_uvdata[:-1, ...] -
                                   baseline_uvdata[1:, ...])
                    mask = ~np.isnan(differences)
                    baselines_noises[baseline] = \
                        np.asarray([np.std(np.ma.array(differences,
                                                       mask=np.invert(mask)).real[..., i], axis=0) for i
                                    in range(self.nstokes)]).T
            else:
                # Use each scan
                for baseline in self.baselines:
                    baseline_noise = list()
                    for scan in self.scans_bl[baseline]:
                        # shape = (#obs in scan, #nif, #nstokes,)
                        scan_baseline_uvdata = self._choose_uvdata(baselines=baseline,
                                                                   times=(scan[0],
                                                                          scan[1],))[0]
                        if average_freq:
                            # shape = (#obs in scan, #nstokes,)
                            scan_baseline_uvdata = np.mean(scan_baseline_uvdata,
                                                           axis=1)
                        # (#obs in scan, #nif, #nstokes,)
                        differences = (scan_baseline_uvdata[:-1, ...] -
                                       scan_baseline_uvdata[1:, ...])
                        mask = ~np.isnan(differences)
                        # (nif, nstokes,)
                        scan_noise = np.asarray([np.std(np.ma.array(differences,
                                                                    mask=np.invert(mask)).real[..., i],
                                                        axis=0) for i in
                                                 range(self.nstokes)]).T
                        baseline_noise.append(scan_noise)
                    baselines_noises[baseline] = np.asarray(baseline_noise)

        return baselines_noises

    def noise_add(self, noise=None, df=None, split_scans=False):
        """
        Add noise to visibilities. Here std - standard deviation of
        real/imaginary component.

        :param noise:
            Mapping from baseline number to:

            1) std of noise. Will use one value of std for all stokes and IFs.
            2) iterable of stds. Will use different values of std for different
            scans. Will use first #scans values from iterable and ignore others.

        :param df (optional):
            Number of d.o.f. for standard Student t-distribution used as noise
            model.  If set to ``None`` then use gaussian noise model. (default:
            ``None``)

        :param split_scans (optional):
            Is parameter ``noise`` is mapping from baseline numbers to
            iterables of std of noise for each scan on baseline? (default:
            ``False``)
        """

        # TODO: if on df before generating noise values
        for baseline, stds in noise.items():
            nscans = len(self.scans_bl[baseline])
            try:
                assert len(stds) >= nscans, "Give >= " + str(nscans) + \
                                            " stds for baseline " + \
                                            str(baseline)
                for i, std in enumerate(stds):
                    try:
                        scan = self.scans_bl[baseline][i]
                    except IndexError:
                        break
                    scan_baseline_uvdata, sc_bl_indxs = \
                        self._choose_uvdata(baselines=baseline,
                                            times=(scan[0], scan[1],))
                    n = np.prod(np.shape(scan_baseline_uvdata))
                    noise_to_add = vec_complex(np.random.normal(scale=std,
                                                                size=n),
                                               np.random.normal(scale=std,
                                                                size=n))
                    noise_to_add = np.reshape(noise_to_add,
                                              np.shape(scan_baseline_uvdata))
                    scan_baseline_uvdata += noise_to_add
                    self.uvdata[sc_bl_indxs] = scan_baseline_uvdata
            except TypeError:
                baseline_uvdata, bl_indxs = \
                    self._choose_uvdata(baselines=baseline)
                n = np.prod(np.shape(baseline_uvdata))
                noise_to_add = vec_complex(np.random.normal(scale=stds,
                                                            size=n),
                                           np.random.normal(scale=stds,
                                                            size=n))
                noise_to_add = np.reshape(noise_to_add,
                                          np.shape(baseline_uvdata))
                baseline_uvdata += noise_to_add
                self.uvdata[bl_indxs] = baseline_uvdata

    def error(self, average_freq=False):
        """
        Shortcut for error associated with each visibility.

        It uses noise calculations based on zero V stokes or successive
        differences implemented in ``noise()`` method to infer sigma of
        gaussian noise.  Later it is supposed to add more functionality (see
        Issue #8).

        :return:
            Numpy.ndarray with shape (#N, #IF, #stokes,) that repeats the shape
            of self.data['hands'] array.
        """
        if self._error is None:
            noise_dict = self.noise(use_V=False, split_scans=False,
                                    average_freq=average_freq)
            if not average_freq:
                self._error = np.empty((len(self.uvdata), self.nif,
                                        self.nstokes,), dtype=float)
            else:
                self._error = np.empty((len(self.uvdata), self.nstokes,),
                                       dtype=float)

            for i, baseline in enumerate(self.data['baseline']):
                self._error[i] = noise_dict[baseline]

        return self._error

    # TODO: use different stokes and symmetry!
    def uv_coverage(self, antennas=None, baselines=None, sym='.k', xinc=1,
                    times=None, x_range=None, y_range=None):
        """
        Make plots of uv-coverage for selected baselines/antennas.

        If ``antenna`` is not None, then plot tracs for all baselines of
        selected antenna with antennas specified in ``baselines``. It is like
        AIPS task UVPLOT with bparm=6,7,0.

        :param times (optional):
            Container with start & stop time or ``None``. If ``None`` then use
            all time. (default: ``None``)
        """
        if antennas is None:
            antennas = self.antennas

        if baselines is None:
            raise Exception("Provide some antenna num. for baselines!")
        else:
            baselines_list = list()
            # If ``baselines`` is iterable
            try:
                baselines_list.extend(baselines)
            # If ``baselines`` is not iterable (int)
            except TypeError:
                baselines_list.append(baselines)
            baselines = set(baselines_list)

        # Check that given baseline numbers are among existing ones
        assert(baselines.issubset(self.antennas))
        # Assert that we don't have one and the same antenna and baseline
        if len(baselines) == len(antennas) == 1:
            assert not baselines.issubset(antennas), "Zero spacing baseline!"

        # Find what baselines to display
        baselines_to_display = list()
        antennas_list = list()
        # If ``antennas`` is iterable
        try:
            antennas_list.extend(antennas)
        # If ``antennas`` is not iterable (int)
        except TypeError:
            antennas_list.append(antennas)
        for ant1 in antennas_list:
            for ant2 in baselines:
                if ant2 > ant1:
                    baselines_to_display.append(ant2 + 256 * ant1)
                elif ant2 < ant1:
                    baselines_to_display.append(ant1 + 256 * ant2)

        baselines_to_display = set(baselines_to_display)

        uvdata, indxs = self._choose_uvdata(times=times,
                                            baselines=baselines_to_display,
                                            freq_average=True)

        u = self.hdu.columns[self.par_dict['UU--']].array[indxs]
        v = self.hdu.columns[self.par_dict['VV--']].array[indxs]
        uv = np.vstack((u, v)).T
        pylab.subplot(1, 1, 1)
        pylab.plot(uv[:, 0], uv[:, 1], sym)
        # FIXME: This is right only for RR/LL!
        pylab.plot(-uv[:, 0], -uv[:, 1], sym)
        # Find max(u & v)
        umax = max(abs(u))
        vmax = max(abs(v))
        uvmax = max(umax, vmax)
        uv_range = [-1.1 * uvmax, 1.1 * uvmax]
        pylab.xlim(uv_range)
        pylab.ylim(uv_range)
        pylab.show()

    def __add__(self, other):
        """
        Add to self another instance of UVData.

        :param other:
            Instance of ``UVData`` class. Or object that has ``uvdata``
            attribute that is numpy structured array with the same ``dtype`` as
            ``self``.

        :return:
            Instance od ``UVData`` class with uv-data in ``uvdata`` attribute
            that is sum of ``self`` and other.
        """

        assert(self.uvdata.shape == other.uvdata.shape)

        self_copy = copy.deepcopy(self)
        self_copy.uvdata = self.uvdata + other.uvdata

        return self_copy

    def __sub__(self, other):
        """
        Substruct from self another instance of UVData.

        :param other:
            Instance of ``UVData`` class. Or object that has ``uvdata``
            attribute that is numpy structured array with the same ``dtype`` as
            ``self``.

        :return:
            Instance od ``UVData`` class with uv-data in ``uvdata`` attribute
            that is difference of ``self`` and other.
        """

        assert(self.uvdata.shape == other.uvdata.shape)

        self_copy = copy.deepcopy(self)
        self_copy.uvdata = self.uvdata - other.uvdata

        return self_copy

    # TODO: TEST ME!!!
    # TODO: Do i need the possibility of multiplying on any complex number?
    # FIXME: After absorbing gains and multiplying on UVData instance some
    # entries do contain NaN. Is that because of some data is flagged and no
    # gains solution are available for that data?
    def __mul__(self, gains):
        """
        Applies complex antenna gains to the visibilities of ``self``.

        :param gains:
            Instance of ``Gains`` class. Or object with ``data`` attribute
            that is structured numpy array and has ``dtype``:
            dtype=[('start', '<f8'),
                   ('stop', '<f8'),
                   ('antenna', 'int'),
                   ('gains', 'complex', (nif, npol,)),
                   ('weights', '<f8', (nif, npol,))]

        :return:
            Instance of ``UVData`` class with visibilities multiplyied by
            complex antenna gains.
        """

        self_copy = copy.deepcopy(self)

        assert(self.nif == np.shape(gains.nif))
        # TODO: Now we need this to calculating gain * gains*. But try to
        # exclude this assertion
        assert(self.nstokes == 4)

        for t in set(self.hdu.columns[self.par_dict['DATE']].array):

            # Find all uv-data entries with time t:
            indxs = np.where(self.hdu.columns[self.par_dict['DATE']].array
                             == t)[0]
            # Loop through uv_indxs (different baselines with the same ``t``)
            # and multiply visibility with baseline ant1-ant2 to
            # gain(ant1)*gain(ant2)^*.
            for indx in indxs:
                bl = self.hdu.columns[self.par_dict['BASELINE']].array[indx]
                try:
                    gains12 = gains.find_gains_for_baseline(t, bl)
                # If gains is the instance of ``Absorber`` class
                except AttributeError:
                    gains12 = gains.absorbed_gains.find_gains_for_baseline(t,
                                                                           bl)
                # FIXME: In substitute() ['hands'] then [indxs] does return
                # view.
                # print "gains12 :"
                # print gains12
                # Doesn't it change copying? Order of indexing [][] has changed
                self_copy.uvdata[indx] *= gains12.T

        return self_copy

    def cv(self, q, fname):
        """
        Method that prepares training and testing samples for q-fold
        cross-validation.

        Inputs:

        :param q:
            Number of folds for cross-validation.

        :param fname:
            Base of file names for output the results.

        :return:
            ``q`` pairs of files (format that of ``IO`` subclass that loaded
            current instance of ``UVData``) with training and testing samples
            prepaired in a such way that 1/``q``- part of visibilities from
            each baseline falls in testing sample and other part falls in
            training sample.
        """

        # List of lists of ``q`` blocks of each baseline
        baselines_chunks = list()

        # Split data of each baseline to ``q`` blocks
        for baseline in self.baselines:
            baseline_indxs = np.where(self.hdu.columns[self.par_dict['BASELINE']].array ==
                                     baseline)[0]
            # Shuffle indexes
            np.random.shuffle(baseline_indxs)
            # Indexes of ``q`` nearly equal chunks. That is list of ``q`` index
            # arrays
            q_indxs = np.array_split(baseline_indxs, q)
            # ``q`` blocks for current baseline
            baseline_chunks = [list(indx) for indx in q_indxs]
            baselines_chunks.append(baseline_chunks)

        # Combine ``q`` chunks to ``q`` pairs of training & testing datasets
        for i in range(q):
            print i
            # List of i-th chunk for testing dataset for each baseline
            testing_indxs = [baseline_chunks[i] for baseline_chunks in
                            baselines_chunks]
            # List of "all - i-th" chunk as training dataset for each baseline
            training_indxs = [baseline_chunks[:i] + baseline_chunks[i + 1:] for
                             baseline_chunks in baselines_chunks]

            print testing_indxs
            print training_indxs
            # Combain testing & training samples of each baseline in one
            testing_indxs = np.sort([item for sublist in testing_indxs for item
                                     in sublist])
            training_indxs = [item for sublist in training_indxs for item in
                              sublist]
            training_indxs = [item for sublist in training_indxs for item in
                              sublist]
            # Save each pair of datasets to files
            # NAXIS changed!!!
            training_data=self.hdu.data[training_indxs]
            testing_data =self.hdu.data[testing_indxs]
            self.save(data=training_data,
                      fname='train' + '_' + str(i + 1).zfill(2) + 'of' + str(q))
            self.save(data=testing_data,
                      fname='test' + '_' + str(i + 1).zfill(2) + 'of' + str(q))

    def cv_score(self, model, stokes='I', average_freq=True):
        """
        Method that returns cross-validation score for ``self`` (as testing
        cv-sample) and model (trained on training cv-sample).

        :param model:
            Model to cross-validate. Instance of ``Model`` class.

        :param stokes (optional):
            Stokes parameter: ``I``, ``Q``, ``U``, ``V``. (default: ``I``)

        :return:
            Cross-validation score between uv-data of current instance and
            model for stokes ``I``.
        """

        baselines_cv_scores = list()

        # Calculate noise on each baseline
        # ``noise`` is dictionary with keys - baseline numbers and values -
        # numpy arrays of noise std for each IF
        noise = self.noise(average_freq=average_freq)
        print "Calculating noise..."
        print noise

        data_copied = copy.deepcopy(self)
        data_copied.substitute([model])
        data_copied.uvdata = self.uvdata - data_copied.uvdata

        if average_freq:
            uvdata = data_copied.uvdata_freq_averaged
        else:
            uvdata = data_copied.uvdata

        for baseline in self.baselines:
            # square difference for each baseline, divide by baseline noise
            # and then sum for current baseline
            indxs = data_copied.hdu.columns[data_copied.par_dict['BASELINE']].array == baseline
            if average_freq:
                hands_diff = uvdata[indxs] / noise[baseline]
            else:
                hands_diff = uvdata[indxs] / noise[baseline][None, :, None]
            # Construct difference for all Stokes parameters
            diffs = dict()
            diffs.update({'I': 0.5 * (hands_diff[..., 0] + hands_diff[..., 1])})
            diffs.update({'V': 0.5 * (hands_diff[..., 0] - hands_diff[..., 1])})
            diffs.update({'Q': 0.5 * (hands_diff[..., 2] + hands_diff[..., 3])})
            diffs.update({'U': 0.5 * 1j * (hands_diff[..., 3] - hands_diff[...,
                                                                           2])})
            for stoke in stokes:
                if stoke not in 'IQUV':
                    raise Exception('Stokes parameter must be in ``IQUV``!')
                result = 0
                diff = diffs[stoke].flatten()
                diff = diff * np.conjugate(diff)
                result += float(diff.sum())
                print "cv score for stoke " + stoke + " is : " + str(result)
            baselines_cv_scores.append(result)

        return sum(baselines_cv_scores)

    def substitute(self, models, baselines=None):
        """
        Method that substitutes visibilities of ``self`` with model values.

        :param models:
            Iterable of ``Model`` instances that substitute visibilities of
            ``self`` with it's own. There should be only one (or zero) model for
            each stokes parameter. If there are two, say I-stokes models, then
            sum them firstly using ``Model.__add__``.

        :param baseline (optional):
            Iterable of baselines on which to substitute visibilities. If
            ``None`` then substitute on all baselines.
            (default: ``None``)
        """

        if baselines is None:
            baselines = self.baselines
        # Indexes of hdu.data with chosen baselines
        indxs = np.hstack(index_of(baselines, self.hdu.columns[self.par_dict['BASELINE']].array))
        n = len(indxs)
        uv = self.uvw[indxs, :2]

        uv_correlations = get_uv_correlations(uv, models)
        for i, hand in enumerate(['RR', 'LL', 'RL', 'LR']):
            try:
                self.uvdata[indxs, :, i] = \
                    uv_correlations[hand].repeat(self.nif).reshape((n, self.nif))
            # If model doesn't have some hands => pass it
            except KeyError:
                pass

    # TODO: convert time to datetime format and use date2num for plotting
    # TODO: make a kwarg argument - to plot in different symbols/colors
    def tplot(self, baselines=None, IF=None, stokes=None, style='a&p',
              freq_average=False, sym=None):
        """
        Method that plots uv-data vs. time.

        :param baselines (optional):
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes (optional):
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)

        :param style (optional):
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)

        .. note:: All checks are in ``_choose_uvdata`` method.
        """

        if not pylab:
            raise Exception('Install ``pylab`` for plotting!')

        if not stokes:
            stokes = 'I'

        uvdata, indxs = self._choose_uvdata(baselines=baselines, IF=IF,
                                            stokes=stokes,
                                            freq_average=freq_average)
        # TODO: i need function to choose parameters
        times = self.hdu.columns[self.par_dict['DATE']].array[indxs]

        if style == 'a&p':
            a1 = np.angle(uvdata)
            a2 = np.real(np.sqrt(uvdata * np.conj(uvdata)))
        elif style == 're&im':
            a1 = uvdata.real
            a2 = uvdata.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        if not freq_average:

            # # of chosen IFs
            n_if = np.shape(uvdata)[1]

            # TODO: define colors
            try:
                syms = self.__color_list[:n_if]
            except AttributeError:
                print "Define self.__color_list to show in different colors!"
                syms = ['.k'] * n_if

            pylab.subplot(2, 1, 1)
            for _if in range(n_if):
                # TODO: plot in different colors and make a legend
                pylab.plot(times, a1[:, _if], syms[_if])
            pylab.subplot(2, 1, 2)
            for _if in range(n_if):
                pylab.plot(times, a2[:, _if], syms[_if])
                if style == 'a&p':
                    pylab.ylim([-math.pi, math.pi])
            pylab.show()

        else:
            if not sym:
                sym = '.k'
            pylab.subplot(2, 1, 1)
            pylab.plot(times, a1, sym)
            pylab.subplot(2, 1, 2)
            pylab.plot(times, a2, sym)
            if style == 'a&p':
                pylab.ylim([-math.pi, math.pi])
            pylab.show()

    # TODO: Implement PA[deg] slicing of uv-plane with keyword argument ``PA``.
    # TODO: Add ``model`` kwarg for plotting image plane model with data
    # together.
    # TODO: Add ``plot_noise`` boolean kwarg for plotting error bars also. (Use
    # ``UVData.noise()`` method for finding noise values.)
    # TODO: implement antennas/baselines arguments as in ``uv_coverage``.
    def uvplot(self, baselines=None, IF=None, stokes=None, style='a&p',
               freq_average=False, sym=None):
        """
        Method that plots uv-data for given baseline vs. uv-radius.

        :param baselines (optional):
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes (optional):
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)

        :param style (optional):
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)

        .. note:: All checks are in ``_choose_uvdata`` method.
        """

        if not pylab:
            raise Exception('Install ``pylab`` for plotting!')

        if not stokes:
            stokes = 'I'

        uvdata, indxs = self._choose_uvdata(baselines=baselines, IF=IF,
                                            stokes=stokes,
                                            freq_average=freq_average)

        # TODO: i need function choose parameters
        uvw_data = self.uvw
        uv_radius = np.sqrt(uvw_data[:, 0] ** 2 + uvw_data[:, 1] ** 2)

        if style == 'a&p':
            a1 = np.angle(uvdata)
            a2 = np.real(np.sqrt(uvdata * np.conj(uvdata)))
        elif style == 're&im':
            a1 = uvdata.real
            a2 = uvdata.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        if not freq_average:
            # # of chosen IFs
            # TODO: Better use len(IF) if ``data`` shape will change sometimes.
            try:
                # If ``IF`` is at least iterable
                n_if = len(IF)
            except TypeError:
                # If ``IF`` is just a number
                n_if = 1

            # TODO: define colors
            try:
                syms = self.__color_list[:n_if]
            except AttributeError:
                print "Define self.__color_list to show in different colors!"
                syms = ['.k'] * n_if

            pylab.subplot(2, 1, 1)
            for _if in range(n_if):
                # TODO: plot in different colors and make a legend
                pylab.plot(uv_radius, a2[:, _if], syms[_if])
            pylab.subplot(2, 1, 2)
            for _if in range(n_if):
                pylab.plot(uv_radius, a1[:, _if], syms[_if])
                if style == 'a&p':
                    pylab.ylim([-math.pi, math.pi])
            pylab.show()
        else:
            if not sym:
                sym = '.k'
            pylab.subplot(2, 1, 1)
            pylab.plot(uv_radius, a2, sym)
            pylab.subplot(2, 1, 2)
            pylab.plot(uv_radius, a1, sym)
            if style == 'a&p':
                pylab.ylim([-math.pi, math.pi])
            pylab.show()

    def uvplot_model(self, model, baselines=None, stokes=None, style='a&p'):
        """
        Plot given image plain model.

        :param model:
            Instance of ``Model`` class.

        :param baselines (optional):
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes (optional):
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)

        :param style (optional):
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)
        """
        # Copy ``model``, choose ``uvws`` given ``baselines`` and set ``_uvws``
        # atribute of ``model``'s copy to calculated ``uvws``. Use
        # ``model.uvplot()`` method to plot model.
        raise NotImplementedError


if __name__ == '__main__':
    import os
    os.chdir('/home/ilya/code/vlbi_errors/data/misha')
    uvdata = UVData('1308+326.U1.2009_08_28.UV_CAL')
    uvdata.cv(10, 'cv_test')
    from model import Model
    uvdata.cv_score(model)
    os.chdir('/home/ilya/code/vlbi_errors/vlbi_errors')
