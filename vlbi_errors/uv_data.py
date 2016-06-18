import math
import copy
import numpy as np
import astropy.io.fits as pf
from collections import OrderedDict
from utils import (baselines_2_ants, index_of, get_uv_correlations,
                   find_card_from_header, get_key)

try:
    import pylab
except ImportError:
    pylab = None

import matplotlib

vec_complex = np.vectorize(np.complex)
vec_int = np.vectorize(np.int)
stokes_dict = {-4: 'LR', -3: 'RL', -2: 'LL', -1: 'RR', 1: 'I', 2: 'Q', 3: 'U',
               4: 'V'}


# TODO: Implement iterator over IF/Stokes
class UVData(object):

    def __init__(self, fname):
        self.fname = fname
        self.hdulist = pf.open(fname)
        self.hdu = self.hdulist[0]
        self._stokes_dict = {'RR': 0, 'LL': 1, 'RL': 2, 'LR': 3}
        self.learn_data_structure(self.hdu)
        self._uvdata = self.view_uvdata({'COMPLEX': 0}) +\
            1j * self.view_uvdata({'COMPLEX': 1})
        self._error = None

    @property
    def sample_size(self):
        raise NotImplementedError

    def sync(self):
        """
        Sync internal representation with complex representation and update
        complex representation ``self._uvdata``. I need this because i don't
        know how to make a complex view to real numpy.ndarray
        """
        slices_dict = self.slices_dict.copy()
        slices_dict.update({'COMPLEX': 0})
        self.hdu.data.data[slices_dict.values()] = self.uvdata.real
        slices_dict.update({'COMPLEX': 1})
        self.hdu.data.data[slices_dict.values()] = self.uvdata.imag

    def save(self, fname=None, data=None):
        """
        Save uv-data to FITS-file.

        :param data: (optional)
            Numpy record array with uv-data & parameters info. If ``None`` then
            save current instance's uv-data. (default: ``None``)
        :param fname: (optional)
            Name of FITS-file to save. If ``None`` then use current instance's
            original file. (default: ``None``)
        """
        fname = fname or self.fname
        if data is None:
            self.hdulist.writeto(fname)
        else:
            new_hdu = pf.GroupsHDU(data)
            # PyFits updates header using given data (``GCOUNT`` key) anyway
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
            # FIXME: Generally we should avoid removing dims
            if value[1] == 1 and key not in ['IF', 'STOKES']:
                slices_dict.update({key: 0})
            else:
                slices_dict.update({key: slice(None, None)})
        self.slices_dict = slices_dict
        uvdata_slices_dict = OrderedDict()
        for key, value in slices_dict.items():
            if value is not 0:
                uvdata_slices_dict.update({key: value})
        self.uvdata_slices_dict = uvdata_slices_dict

    def new_slices(self, key, key_slice):
        """
        Return VIEW of internal ``hdu.data.data`` numpy.ndarray with given
        slice.
        """
        slices_dict = self.slices_dict.copy()
        slices_dict.update({key: key_slice})
        return slices_dict

    def view_uvdata(self, new_slices_dict):
        """
        Return VIEW of internal ``hdu.data.data`` numpy.ndarray with given
        slices.

        :param new_slices_dict:
            Ex. {'COMPLEX': slice(0, 1), 'IF': slice(0, 2)}
        """
        slices_dict = self.slices_dict.copy()
        for key, key_slice in new_slices_dict.items():
            slices_dict.update({key: key_slice})
        return self.hdu.data.data[slices_dict.values()]

    @property
    def stokes(self):
        """
        Shortcut to correlations present (or Stokes parameters).
        """
        ref_val = get_key(self.hdu.header, 'STOKES', 'CRVAL')
        ref_pix = get_key(self.hdu.header, 'STOKES', 'CRPIX')
        delta = get_key(self.hdu.header, 'STOKES', 'CDELT')
        n_stokes = get_key(self.hdu.header, 'STOKES', 'NAXIS')
        return [stokes_dict[ref_val + (i - ref_pix) * delta] for i in
                range(1, n_stokes + 1)]

    @property
    def uvdata(self):
        """
        Returns (#groups, #if, #stokes,) complex numpy.ndarray with last
        dimension - real&imag part of visibilities. It is A COPY of
        ``hdu.data.data`` numpy.ndarray.
        """
        # Always return complex representation of internal ``hdu.data.data``
        return self._uvdata

    @uvdata.setter
    def uvdata(self, other):
        # Updates A COPY of ``hdu.data.data`` numpy.ndarray (complex repr.)
        self._uvdata = other
        # Sync internal representation with changed complex representation.
        self.sync()

    @property
    def uvdata_freq_averaged(self):
        """
        Returns ``self.uvdata`` averaged in IFs, that is complex numpy.ndarray
        with shape (#N, #stokes).
        """
        if self.nif > 1:
            result = np.mean(self.uvdata, axis=1)
        # FIXME: if self.nif=1 then np.mean for axis=1 will remove this
        # dimension. So don't need this if-else
        else:
            result = self.uvdata[:, 0, :]
        return result

    @property
    def baselines(self):
        """
        Returns list of baselines numbers.
        """
        result = list(set(self.hdu.columns[self.par_dict['BASELINE']].array))
        return vec_int(sorted(result))

    @property
    def antennas(self):
        """
        Returns list of antennas numbers.
        """
        return baselines_2_ants(self.baselines)

    @property
    def frequency(self):
        """
        Returns sky frequency in Hz.
        """
        freq_card = find_card_from_header(self.hdu.header, value='FREQ')[0]
        return self.hdu.header['CRVAL{}'.format(freq_card[0][-1])]

    @property
    def freq_width_if(self):
        """
        Returns width of IF in Hz.
        """
        freq_card = find_card_from_header(self.hdu.header, value='FREQ')[0]
        return self.hdu.header['CDELT{}'.format(freq_card[0][-1])]

    @property
    def freq_width(self):
        """
        Returns width of all IFs in Hz.
        """
        freq_card = find_card_from_header(self.hdu.header, value='FREQ')[0]
        return self.nif * self.hdu.header['CDELT{}'.format(freq_card[0][-1])]

    @property
    def band_center(self):
        """
        Returns center of frequency bandwidth in Hz.
        """
        return self.frequency + self.freq_width_if * (self.nif / 2. - 0.5)

    @property
    def scans(self):
        """
        Returns list of times that separates different scans. If NX table is
        present in the original

        :return:
            numpy.ndarray with shape (#scans, 2,) with start & stop time for each
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

    # FIXME: doesn't work for ``J0005+3820_X_1998_06_24_fey_vis.fits``
    # FIXME: Sometimes only 1 measurement in `scan`. It results in noise =
    # ``nan`` for that scan
    # FIXME: It would be better to output indexes of different scans for each
    # baselines
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
            # print "Processing baseline ", bl
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
        suffix = '--'
        try:
            u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
            v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
            w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        except KeyError:
            try:
                suffix = '---SIN'
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
            except KeyError:
                suffix = ''
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        if abs(np.mean(u)) < 1.:
            u *= self.frequency
            v *= self.frequency
            w *= self.frequency
        return np.vstack((u, v, w)).T

    @property
    def uv(self):
        """
        Shortcut for (u, v) -coordinates of visibility values.

        :return:
            Numpy.ndarray with shape (N, 2,), where N is the number of (u, v, w)
            points.
        """
        return self.uvw[:, :2]

    # FIXME: shape depends on ``stokes``! Must be ``(N, #IF, #stokes)``
    def _choose_uvdata(self, times=None, baselines=None, IF=None, stokes=None,
                       freq_average=False, indx_only=False):
        """
        Method that returns chosen data from ``_data`` numpy structured array
        based on user specified parameters.

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

        :param indx_only: (optional)
            Boolean - return only indexes? (default: ``False``)

        :return:
            Two numpy.ndarrays, where first array is array of data with shape
            (#N, #IF, #STOKES) and second array is boolean numpy array of
            indexes of data in original PyFits record array.

        :note:
            All checks on IF and stokes are made here in one place. This method
            is heavily used by all methods that retrieve data and/or plot it.

        """
        # Copy with shape (#N, #IF, #STOKES, 2,)
        uvdata = self.uvdata

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
                # TODO: Vectorize that.
                indxs = np.logical_or(indxs,
                                      self.hdu.columns[self.par_dict['BASELINE']].array == baseline)

        # If we are given some time interval then find among ``indxs`` only
        # those from given time interval
        if times is not None:
            # Assert that ``times`` consists of start and stop
            assert(len(times) == 2)
            lower_indxs = self.hdu.columns[self.par_dict['DATE']].array < times[1]
            high_indxs = self.hdu.columns[self.par_dict['DATE']].array > times[0]
            time_indxs = np.logical_and(lower_indxs, high_indxs)
            indxs = np.logical_and(indxs, time_indxs)

        if indx_only:
            return None, indxs

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
    # FIXME: ``use_V = True`` gives lower more realistic noise estimates?
    # FIXME: Generally it should return (#sc, #band, #freq, #stok) array of
    # stds. Possible, last dimension only for ``use_V=False`` and first dim only
    # for ``split_scans=True``.
    def noise(self, split_scans=False, use_V=True, average_freq=False):
        """
        Calculate noise for each baseline. If ``split_scans`` is True then
        calculate noise for each scan too. If ``use_V`` is True then use stokes
        V data (`RR`` - ``LL``) for computation assuming no signal in V. Else
        use successive differences approach (Brigg's dissertation).

        :param split_scans: (optional)
            Should we calculate noise for each scan? (default: ``False``)

        :param use_V: (optional)
            Use stokes V data (``RR`` - ``LL``) to calculate noise assuming no
            signal in stokes V? If ``False`` then use successive differences
            approach (see Brigg's dissertation). (default: ``True``)

        :param average_freq: (optional)
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
            IFs.

        :param df: (optional)
            Number of d.o.f. for standard Student t-distribution used as noise
            model.  If set to ``None`` then use gaussian noise model. (default:
            ``None``)

        :param split_scans: (optional)
            Is parameter ``noise`` is mapping from baseline numbers to
            iterables of std of noise for each scan on baseline? (default:
            ``False``)
        """

        # TODO: if on df before generating noise values
        for baseline, baseline_stds in noise.items():
            for i, std in enumerate(baseline_stds):
                baseline_uvdata, bl_indxs = \
                    self._choose_uvdata(baselines=baseline, IF=i + 1)
                # (#, #IF, #CH, #stokes)
                n = np.shape(baseline_uvdata)[0]
                n_to_add = n * self.nstokes
                noise_to_add = vec_complex(np.random.normal(scale=std,
                                                            size=n_to_add),
                                           np.random.normal(scale=std,
                                                            size=n_to_add))
                noise_to_add = np.reshape(noise_to_add,
                                          (n, 1, self.nstokes))
                baseline_uvdata += noise_to_add
                self.uvdata[bl_indxs] = baseline_uvdata
        self.sync()

    # TODO: Calculate noise also by differences optionally of if only one
    # parallel hand is available and V can't be calculated
    def error(self, average_freq=False):
        """
        Shortcut for error associated with each visibility.

        It uses noise calculations based on zero V stokes or successive
        differences implemented in ``noise()`` method to infer sigma of
        gaussian noise.  Later it is supposed to add more functionality (see
        Issue #8).

        :param average_freq: (optional)
            Use IF-averaged data for calculating errors? (default: ``False``)

        :return:
            Numpy.ndarray with shape (#N, #IF, #stokes,) where #N - number of
            groups.
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

            for i, baseline in enumerate(self.hdu.data['BASELINE']):
                self._error[i] = noise_dict[baseline]

        return self._error

    # TODO: use different stokes and symmetry!
    # FIXME:
    def uv_coverage(self, antennas=None, baselines=None, sym='.k', xinc=1,
                    times=None, x_range=None, y_range=None):
        """
        Make plots of uv-coverage for selected baselines/antennas.

        If ``antenna`` is not None, then plot tracs for all baselines of
        selected antenna with antennas specified in ``baselines``. It is like
        AIPS task UVPLOT with bparm=6,7,0.

        :param antennas: (optional)
            AIPS-like ``uvplt`` parameter.
        :param baselines: (optional)
            AIPS-like ``uvplt`` parameter.
        :param sym: (optional)
            Matplotlib symbols to plot. (default: ``.k``)
        :param xinc: (optional)
            AIPS-like ``uvplt`` parameter.
        :param times: (optional)
            Container with start & stop time or ``None``. If ``None`` then use
            all time. (default: ``None``)
        :param x_range: (optional)
            U-distance range. If ``None`` then use all available. (default:
            ``None``)
        :param y_range: (optional)
            V-distance range. If ``None`` then use all available. (default:
            ``None``)
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

        # FIXME: Use properties for u, v
        u = self.hdu.columns[self.par_dict['UU--']].array[indxs]
        v = self.hdu.columns[self.par_dict['VV--']].array[indxs]
        uv = np.vstack((u, v)).T
        matplotlib.pyplot.subplot(1, 1, 1)
        matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], sym)
        # FIXME: This is right only for RR/LL!
        matplotlib.pyplot.plot(-uv[:, 0], -uv[:, 1], sym)
        # Find max(u & v)
        umax = max(abs(u))
        vmax = max(abs(v))
        uvmax = max(umax, vmax)
        uv_range = [-1.1 * uvmax, 1.1 * uvmax]
        matplotlib.pyplot.xlim(uv_range)
        matplotlib.pyplot.ylim(uv_range)
        matplotlib.pyplot.axes().set_aspect('equal')
        matplotlib.pyplot.xlabel('U, wavelengths')
        matplotlib.pyplot.ylabel('V, wavelengths')
        matplotlib.pyplot.show()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return UVData(self.hdulist.filename())

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
        assert(len(self.uvdata) == len(other.uvdata))

        self_copy = copy.deepcopy(self)
        self_copy.uvdata = self.uvdata + other.uvdata
        self_copy.sync()

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
        assert(len(self.uvdata) == len(other.uvdata))

        self_copy = copy.deepcopy(self)
        self_copy.uvdata = self.uvdata - other.uvdata
        self_copy.sync()

        return self_copy

    def multiply(self, x):
        """
        Multiply visibilities on number.
        :param x:
        :return:
        """
        self_copy = copy.deepcopy(self)
        self_copy.uvdata = x * self.uvdata
        self_copy.sync()

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
        self_copy.sync()

        return self_copy

    def zero_data(self):
        """
        Method that zeros all visibilities.
        """
        self.uvdata = np.zeros(np.shape(self.uvdata), dtype=self.uvdata.dtype)

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
                      fname=fname + '_train' + '_' + str(i + 1).zfill(2) + 'of'
                            + str(q) + '.FITS')
            self.save(data=testing_data,
                      fname=fname + '_test' + '_' + str(i + 1).zfill(2) + 'of' +
                            str(q) + '.FITS')

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
        # FIXME: This will cause `IndexError: index 1 is out of bounds for axis
        # 2 with size 1` if uv-data has just one parallel stokes (et. `LL`).
        # I should check that my uv-data has Stokes that model provides
        # otherwise ignore that Stokes. Implement ``stokes`` property!
        for i, hand in enumerate(['RR', 'LL', 'RL', 'LR']):
            try:
                self.uvdata[indxs, :, i] = \
                    uv_correlations[hand].repeat(self.nif).reshape((n, self.nif))
                self.sync()
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
               freq_average=False, sym=None, phase_range=None, amp_range=None,
               re_range=None, im_range=None, colors=None, color='#4682b4'):
        """
        Method that plots uv-data for given baseline vs. uv-radius.

        :param baselines: (optional)
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)
        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)
        :param stokes: (optional)
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)
        :param style: (optional)
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)
        :param color: (optional)
            Default color.
        :param colors: (optional)
            Default colors for multi IF plotting.

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
        uvw_data = self.uvw[indxs]
        uv_radius = np.sqrt(uvw_data[:, 0] ** 2 + uvw_data[:, 1] ** 2)

        if style == 'a&p':
            a1 = np.angle(uvdata)
            a2 = np.real(np.sqrt(uvdata * np.conj(uvdata)))
        elif style == 're&im':
            a1 = uvdata.real
            a2 = uvdata.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        fig, axes = matplotlib.pyplot.subplots(nrows=2, ncols=1, sharex=True,
                                               sharey=False)
        if not freq_average:
            # # of chosen IFs
            # TODO: Better use len(IF) if ``data`` shape will change sometimes.
            if IF is None:
                n_if = self.nif
            else:
                try:
                    # If ``IF`` is at least iterable
                    n_if = len(IF)
                except TypeError:
                    # If ``IF`` is just a number
                    n_if = 1

            # TODO: define colors
            if colors is not None:
                assert n_if <= len(colors)
                syms = colors[:n_if]
                syms = [".{}".format(color) for color in syms]
            else:
                print "Provide ``colors`` argument to show in different" \
                      " colors!"
                syms = ['.'] * n_if

            for _if in range(n_if):
                # TODO: plot in different colors and make a legend
                axes[0].plot(uv_radius, a2[:, _if], syms[_if], color=color)
            for _if in range(n_if):
                axes[1].plot(uv_radius, a1[:, _if], syms[_if], color=color)
            if style == 'a&p':
                axes[1].set_ylim([-math.pi, math.pi])
                axes[0].set_ylabel('Amplitude, [Jy]')
                axes[1].set_ylabel('Phase, [rad]')
                if amp_range is not None:
                    axes[0].set_ylim(amp_range)
                if phase_range is not None:
                    axes[1].set_ylim(phase_range)
            elif style == 're&im':
                axes[0].set_ylabel('Re, [Jy]')
                axes[1].set_ylabel('Im, [Jy]')
                if re_range is not None:
                    axes[0].set_ylim(re_range)
                if im_range is not None:
                    axes[1].set_ylim(im_range)
            matplotlib.pyplot.xlabel('UV-radius, wavelengths')
            fig.show()
        else:
            if not sym:
                sym = '.'
            axes[0].plot(uv_radius, a2, sym, color=color)
            axes[1].plot(uv_radius, a1, sym, color=color)
            matplotlib.pyplot.xlabel('UV-radius, wavelengths')
            if style == 'a&p':
                axes[1].set_ylim([-math.pi, math.pi])
                axes[0].set_ylabel('Amplitude, [Jy]')
                axes[1].set_ylabel('Phase, [rad]')
                if amp_range is not None:
                    axes[0].set_ylim(amp_range)
                if phase_range is not None:
                    axes[1].set_ylim(phase_range)

            elif style == 're&im':
                axes[0].set_ylabel('Re, [Jy]')
                axes[1].set_ylabel('Im, [Jy]')
                if re_range is not None:
                    axes[0].set_ylim(re_range)
                if im_range is not None:
                    axes[1].set_ylim(im_range)

            fig.show()

    def uvplot_model(self, model, baselines=None, stokes=None, style='a&p'):
        """
        Plot given image plain model.

        :param model:
            Instance of ``Model`` class.

        :param baselines: (optional)
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes: (optional)
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)

        :param style: (optional)
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)
        """
        # Copy ``model``, choose ``uvws`` given ``baselines`` and set ``_uvws``
        # atribute of ``model``'s copy to calculated ``uvws``. Use
        # ``model.uvplot()`` method to plot model.
        raise NotImplementedError

    def ft_to_image(self, image_params, baselines=None, IF=None, times=None,
                    freq_average=True):
        """
        FT uv-data to dirty map with specified parameters.

        :param image_params:
            Dictionary with image parameters.
        :param baselines: (optional)
            Baselines to use. If ``None`` then use all. (default: ``None``)
        :param IF: (optional)
            IFs to use. If ``None`` then use all. (default: ``None``)
        :param times: (optional)
            Time range to use. If ``None`` then use all. (default: ``None``)
        :param freq_average: (optional)
            Average IFs? (default: ``True``)

        :return:
            ``Image`` instance with dirty map.
        """
        # im(x, y) = vis(u, v) * np.exp(2. * math.pi * 1j * (u * x + v * y))
        # where x, y - distances from pase center [rad]
        pass


if __name__ == '__main__':
    fname = '/home/ilya/sandbox/heteroboot/0945+408.u.2007_04_18.uvf'
    # uvdata = UVData(fname)
    # uvdata.uvplot(style='re&im', freq_average=True)
    # uvdata.uvplot(style='re&im')
    # uvdata.uvplot(style='a&p', freq_average=True)
    # uvdata.uvplot(style='a&p', IF=[1, 2], freq_average=True)
    # uvdata.uvplot(style='a&p', amp_range=[0, 2], phase_range=[-0.5, 0.5],
    #               colors=['b', 'g', 'r', 'y'], baselines=uvdata.baselines[:4])
    # uvdata.uvplot(style='a&p', amp_range=[0, 2], phase_range=[-0.5, 0.5],
    #               colors=['b', 'g', 'r', 'y'], baselines=uvdata.baselines[:4],
    #               IF=[1, 2])
