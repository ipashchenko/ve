import glob
import os
import numpy as np
from power_analysis import bootstrap_uv_fits
from spydiff import clean_difmap
from from_fits import (create_clean_image_from_fits_file,
                       create_image_from_fits_file)


def average_maps(cc_fits):
    n = len(cc_fits)
    image = create_clean_image_from_fits_file(cc_fits[0])
    imsize = image.imsize[0]
    data = np.zeros((imsize, imsize), dtype=float)
    for fn in cc_fits:
        print("Loading {}".format(fn))
        image = create_image_from_fits_file(fn)
        data = data + image.image
    return data/n


def average_ccmodels(cc_fits):
    n = len(cc_fits)
    image = create_clean_image_from_fits_file(cc_fits[0])
    imsize = image.imsize[0]
    data = np.zeros((imsize, imsize), dtype=float)
    for fn in cc_fits:
        print("Loading {}".format(fn))
        image = create_clean_image_from_fits_file(fn)
        data = data + image.cc
    return data/n


ccfits_fn = "/home/ilya/github/ve/vlbi_errors/bk_jet_cc_u.fits"
uvdata_fn = "/home/ilya/github/ve/vlbi_errors/bk_jet_u.uvf"
boot_dir = "/home/ilya/data/bagging_bkjet/"
path_to_script = '/home/ilya/github/ve/difmap/final_clean_nw'

bootstrap_uv_fits(uvdata_fn, [ccfits_fn], 100,
                  outpath=boot_dir,
                  outname=['bootstrapped_data', '.fits'])

boot_uvfs = glob.glob(os.path.join(boot_dir, "bootstrapped_data*"))
for boot_uvf in boot_uvfs:
    _, boot_fn = os.path.split(boot_uvf)
    istr = boot_fn.split("_")[2].split(".")[0]
    clean_difmap(boot_fn, 'cc_{}.fits'.format(istr), "I", (2048, 0.05),
                 path=boot_dir,
                 path_to_script=path_to_script, outpath=boot_dir)


cc_fits = glob.glob(os.path.join(boot_dir, "cc_*.fits"))