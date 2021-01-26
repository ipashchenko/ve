import pickle
import sys
import os
import glob
import numpy as np
from image import plot as iplot
from uv_data import UVData
from spydiff import find_size_errors_using_chi2


def convert_mojave_epoch(epoch):
    year = epoch.split('_')[0]
    month = epoch.split('_')[1]
    day = epoch.split('_')[2]
    return "{}-{}-{}".format(year, month, day)

pixsize_mas = 0.1
data_dir = "/home/ilya/github/difmap_addons/M87models"
save_dir = os.path.join(data_dir, "test")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
uvf_files = sorted(glob.glob(os.path.join(data_dir, "*uvf")))
uvf_files = [os.path.split(path)[-1] for path in uvf_files]
epochs = [fn.split(".")[2] for fn in uvf_files]
mdl_files = ["1228+126_{}.mod".format(convert_mojave_epoch(epoch)) for epoch in epochs]

for uvf_file, mdl_file, epoch in zip(uvf_files, mdl_files, epochs):
    # Problematic epochs
    # if epoch in ["1997_03_13", "2001_12_30", "2003_08_23", "2004_05_29"]:
    # if mdl_file != "1997_03_13.mod":
    #     continue


    print(mdl_file, uvf_file)


    # Find errors if they are not calculated
    if not os.path.exists(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch))):
        errors = find_size_errors_using_chi2(os.path.join(data_dir, mdl_file),
                                             os.path.join(data_dir, uvf_file),
                                             show_difmap_output=False, nmodelfit=50, use_selfcal=True)
        with open(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(errors, fo)
    # Or just load already calculated
    else:
        with open(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch)), "rb") as fo:
            errors = pickle.load(fo)