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


def score(uv_fits_path, mdl_path, stokes='I', bmaj=None, score="l2"):
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
    if bmaj is not None:
        u = uvdata_diff.uv[:, 0]
        v = uvdata_diff.uv[:, 1]
        taper = np.exp(-c*(u*u + v*v))
        i_diff = i_diff*taper[:, np.newaxis]

    # Number of unmasked visibilities (accounting each IF)
    if stokes == "I":
        # 2 means that Re & Im are counted independently
        factor = 2*np.count_nonzero(~i_diff.mask)
    else:
        factor = np.count_nonzero(~i_diff.mask)

    print("Number of independent test data points = ", factor)
    if score == "l2":
        result = np.sqrt(float(np.sum(i_diff*i_diff.conj())))/factor
    elif score == "l1":
        result = float(np.sum(np.abs(i_diff)))/factor
    else:
        raise Exception("score must be in (l1, l2)!")
    return result


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
    original_uvfits = "0212+735.u.2020_11_29_edt.uvf"
    # Beam mas
    beam = 0.66

    template_script = os.path.join(data_dir, "script_clean_rms_template")
    path_to_script = os.path.join(data_dir, "script_clean_rms")

    # path_to_scripts = {# 0.1: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.1',
    #                    # 0.5: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.5',
    #                    0.55: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.55',
    #                    0.6: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.6',
    #                    0.65: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.65',
    #                    0.7: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.7',
    #                    # 0.75: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.75',
    #                    # 0.875: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k0.875',
    #                    # 1: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k1',
    #                    # 1.125: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k1.125',
    #                    # 1.25: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k1.25',
    #                    # 1.5: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k1.5',
    #                    # 2: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k2',
    #                    # 3: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k3',
    #                    # 5: '/home/ilya/github/stackemall/external_scripts/script_clean_rms_k5'}
    # }
    scans_cv = ScansCV(os.path.join(data_dir, original_uvfits), outdir=data_dir)
    # scans_cv.create_train_test_uvfits(data_dir)

    cv_scores = dict()

    for k in np.linspace(0.5, 1.0, 5):

        filein = open(template_script)
        src = Template(filein.read())
        result = src.substitute({"overclean_coef": k})
        with open(path_to_script, "w") as fo:
            print(result, file=fo)

        cv_scores[k] = dict()
        for i, (train_uvfits_fname, test_uvfits_fname) in enumerate(scans_cv):
            print("BASELINE = ", scans_cv.cur_bl, ", SCAN# = ", scans_cv.cur_scan, " ====================")

            # Optionally create entry in dict
            if scans_cv.cur_bl not in cv_scores[k]:
                cv_scores[k][scans_cv.cur_bl] = list()

            print(train_uvfits_fname, test_uvfits_fname)
            # CLEAN train data set
            clean_difmap(fname=train_uvfits_fname,
                         outfname="trained_cc.fits",
                         stokes='i', mapsize_clean=(512, 0.1),
                         path=data_dir, outpath=data_dir,
                         path_to_script=path_to_script,
                         show_difmap_output=False)
            # Score trained model on test data set
            cv_score = score(os.path.join(data_dir, test_uvfits_fname),
                             os.path.join(data_dir, "trained_cc.fits"),
                             bmaj=beam, score="l1")
            print("CV score = ", cv_score)
            cv_scores[k][scans_cv.cur_bl].append(cv_score)
            print("Result for current k = {} is {}".format(k, cv_scores[k]))

    with open("cv_scores_dump_beam_l1_edt.pkl", "wb") as fo:
        pickle.dump(cv_scores, fo)