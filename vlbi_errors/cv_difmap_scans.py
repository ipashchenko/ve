import os
import numpy as np
from astropy.time import Time
from astropy import units as u
from string import Template
import pickle
from uv_data import UVData
# from sklearn.model_selection import KFold
from model import Model
from spydiff import (clean_n, clean_difmap, flag_baseline, flag_baseline_scan,
                     import_difmap_model, modelfit_difmap)
from from_fits import create_model_from_fits_file
from utils import to_boolean_array, baselines_2_ants
import calendar

months_dict = {v: k for k, v in enumerate(calendar.month_abbr)}
months_dict_inv = {k: v for k, v in enumerate(calendar.month_abbr)}
mas_to_rad = u.mas.to(u.rad)


def score(uv_fits_path, mdl_path, stokes='I', bmaj=None):
    """
    Returns rms of the trained model (CLEAN or difmap) on a given test UVFITS
    data set.

    :param uv_fits_path:
        Path to uv-fits file (test data).
    :param mdl_path:
        Path to difmap model text file or FITS-file with CLEAN model (trained
        model).
    :param stokes: (optional)
        Stokes parameter string. ``I``, ``RR`` or ``LL`` currently supported.
        (default: ``I``)
    :param bmaj: (optional)
        FWHM of the circular beam to account for. If ``None`` than do not
        account for the beam. (default: ``None``)
    :return:
        Per-point rms between given test data and trained model evaluated at a
        given test data points.
    """
    stokes = stokes.upper()
    if stokes not in ('I', 'RR', 'LL'):
        raise Exception("Only stokes I, RR or LL are supported!")

    if bmaj is not None:
        c = (np.pi*bmaj*mas_to_rad)**2/(4*np.log(2))
    else:
        c = 1.0

    # Loading test data
    uvdata = UVData(uv_fits_path)
    uvdata_model = UVData(uv_fits_path)

    # Loading trained model
    # CC-model
    try:
        model = create_model_from_fits_file(mdl_path)
    # Difmap model
    except IOError:
        dfm_mdl_dir, dfm_mdl_fname = os.path.split(mdl_path)
        comps = import_difmap_model(dfm_mdl_fname, dfm_mdl_dir)
        model = Model(stokes=stokes)
        model.add_components(*comps)

    # Computing difference and score
    uvdata_model.substitute([model])
    uvdata_diff = uvdata-uvdata_model
    if stokes == 'I':
        i_diff = 0.5*(uvdata_diff.uvdata_weight_masked[..., 0] +
                      uvdata_diff.uvdata_weight_masked[..., 1])
    elif stokes == 'RR':
        i_diff = uvdata_diff.uvdata_weight_masked[..., 0]
    elif stokes == 'LL':
        i_diff = uvdata_diff.uvdata_weight_masked[..., 1]
    else:
        raise Exception("Only stokes (I, RR, LL) supported!")

    # Account for beam
    u = uvdata_diff.uv[:, 0]
    v = uvdata_diff.uv[:, 1]
    taper = np.exp(-c*(u*u + v*v))
    i_diff = i_diff*taper

    # Number of unmasked visibilities (accounting each IF)
    if stokes == "I":
        # 2 means that Re & Im are counted independently
        factor = 2*np.count_nonzero(~i_diff.mask)
    else:
        factor = np.count_nonzero(~i_diff.mask)

    print("Number of independent test data points = ", factor)
    # factor = np.count_nonzero(~uvdata_diff.uvdata_weight_masked.mask[:, :, :2])
    # squared_diff = uvdata_diff.uvdata_weight_masked[:, :, :2] * \
    #                uvdata_diff.uvdata_weight_masked[:, :, :2].conj()
    squared_diff = i_diff*i_diff.conj()
    return np.sqrt(float(np.sum(squared_diff))/factor)


class ScansCV(object):
    def __init__(self, original_uvfits, outdir):
        self.original_uvfits = original_uvfits
        self.uvdata = UVData(original_uvfits)
        self.outdir = outdir
        self.train_uvfits = "cv_train.uvf"
        self.test_uvfits = "cv_test.uvf"
        self.cur_bl = None
        self.cur_scan = None

    def __iter__(self):
    # def create_train_test_uvfits(self):
        all_baselines = self.uvdata.baselines
        for bl, scans_times in self.uvdata.baselines_scans_times.items():
            print("BASELINE = ", bl)
            self.cur_bl = bl
            ta, tb = baselines_2_ants([bl])
            ta = self.uvdata.antenna_mapping[ta]
            tb = self.uvdata.antenna_mapping[tb]
            n_scans = len(scans_times)
            rnd_scan_number = np.random.randint(n_scans)
            # half_scan_number = int(n_scans/2)
            # if half_scan_number == 0:
            #     half_scan_number = 1
            for scan_num, scan_times in enumerate(scans_times):
                self.cur_scan = scan_num
                # if scan_num != half_scan_number:
                #     continue
                print("SCAN # ", scan_num)
                start_time = Time(scan_times[0], format="jd")
                stop_time = Time(scan_times[-1], format="jd")
                # Train data
                # flag_baseline_scan(self.fname, os.path.join(self.outdir, "cv_bl_{}_scan_{}_train.uvf".format(int(bl), scan_num)), ta, tb,
                #                    start_time=start_time, stop_time=stop_time, except_time_range=False)
                flag_baseline_scan(self.original_uvfits, os.path.join(self.outdir, self.train_uvfits), ta, tb,
                                   start_time=start_time, stop_time=stop_time, except_time_range=False)
                # Test data
                # Manage target baseline
                # flag_baseline_scan(self.fname, os.path.join(self.outdir, "cv_bl_{}_scan_{}_test.uvf".format(int(bl), scan_num)), ta, tb,
                #                    start_time=start_time, stop_time=stop_time, except_time_range=True)
                flag_baseline_scan(self.original_uvfits, os.path.join(self.outdir, self.test_uvfits), ta, tb,
                                   start_time=start_time, stop_time=stop_time, except_time_range=True)
                # Manage all others baselines
                for other_bl in all_baselines:
                    ta_other, tb_other = baselines_2_ants([other_bl])
                    ta_other = self.uvdata.antenna_mapping[ta_other]
                    tb_other = self.uvdata.antenna_mapping[tb_other]
                    if ta in (ta_other, tb_other) and tb in (ta_other, tb_other):
                        continue
                    else:
                        # flag_baseline(os.path.join(self.outdir, "cv_bl_{}_scan_{}_test.uvf".format(int(bl), scan_num)),
                        #               os.path.join(self.outdir, "cv_bl_{}_scan_{}_test.uvf".format(int(bl), scan_num)),
                        #               ta_other, tb_other)
                        flag_baseline(os.path.join(self.outdir, self.test_uvfits),
                                      os.path.join(self.outdir, self.test_uvfits),
                                      ta_other, tb_other)

                yield self.train_uvfits, self.test_uvfits


    # def __iter__(self):
    #     for i in range(self.k):
    #         train_indxs = np.zeros(len(self.uvdata.hdu.data))
    #         test_indxs = np.zeros(len(self.uvdata.hdu.data))
    #         for bl, kfolds in self.baseline_folds.items():
    #             itrain, itest = kfolds[i]
    #             # itrain = to_boolean_array(itrain)
    #             train_indxs = np.logical_or(train_indxs, itrain)
    #             test_indxs = np.logical_or(test_indxs, itest)
    #         train_data = self.uvdata.hdu.data[train_indxs]
    #         test_data = self.uvdata.hdu.data[test_indxs]
    #         print("saving data to {} and {}".format(self.test_fname,
    #                                                 self.train_fname))
    #         self.uvdata.save(self.test_fname, test_data, rewrite=True)
    #         self.uvdata.save(self.train_fname, train_data, rewrite=True)
    #
    #         yield self.train_fname, self.test_fname


if __name__ == '__main__':
    data_dir = "/home/ilya/data/cv"
    original_uvfits = "0212+735.u.2010_11_20.uvf"

    scans_cv = ScansCV(os.path.join(data_dir, original_uvfits), outdir=data_dir)
    cv_scores = dict()

    for n_comp in np.arange(1, 10):
        cv_scores[n_comp] = dict()
        for i, (train_uvfits_fname, test_uvfits_fname) in enumerate(scans_cv):
            print("BASELINE = ", scans_cv.cur_bl, ", SCAN# = ", scans_cv.cur_scan, " ====================")

            # Optionally create entry in dict
            if scans_cv.cur_bl not in cv_scores[n_comp]:
                cv_scores[n_comp][scans_cv.cur_bl] = list()

            print(train_uvfits_fname, test_uvfits_fname)
            # Modelfit train data set
            modelfit_difmap(fname=train_uvfits_fname,
                            mdl_fname="{}cg.mdl".format(n_comp),
                            out_fname="trained.mdl", niter=100, stokes='i',
                            path=data_dir, mdl_path=data_dir, out_path=data_dir,
                            show_difmap_output=False)
            # Score trained model on test data set
            cv_score = score(os.path.join(data_dir, test_uvfits_fname),
                             os.path.join(data_dir, "trained.mdl"))
            print("CV score = ", cv_score)
            cv_scores[n_comp][scans_cv.cur_bl].append(cv_score)
            print("Result for current k = {} is {}".format(n_comp, cv_scores[n_comp]))

    with open("cv_scores_dump.pkl", "wb") as fo:
        pickle.dump(cv_scores, fo)