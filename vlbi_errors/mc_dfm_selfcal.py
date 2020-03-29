import glob
import os
import numpy as np
from uv_data import UVData
from model import Model
from spydiff import clean_difmap, selfcal_difmap, import_difmap_model, export_difmap_model, modelfit_difmap
from from_fits import create_model_from_fits_file
import sys
sys.path.insert(0, '/home/ilya/github/dterms')
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


data_dir = "/home/ilya/data/selfcal"
mapsize_clean = 512, 0.1
n_mc = 100
epochs = glob.glob(os.path.join(data_dir, "*_*_*.mod"))
epochs = [os.path.split(epoch)[-1] for epoch in epochs]
epochs = sorted([epoch.split(".")[0] for epoch in epochs])
# epochs = ["2019_08_23", "2019_10_11"]
epochs = ["2015_02_20"]

for epoch in epochs:

    # Obtain CLEAN model
    clean_difmap(fname="2200+420.u.{}.uvf".format(epoch),
                 outfname="cc_I.fits", path=data_dir,
                 stokes="I", outpath=data_dir, mapsize_clean=(512, 0.1),
                 path_to_script="/home/ilya/github/ve/difmap/final_clean_nw",
                 show_difmap_output=False)
    ccmodel = create_model_from_fits_file(os.path.join(data_dir, "cc_I.fits"))


    # Self-cal with script
    original_raw_uvf = "2200+420.u.{}.uvf_raw_edt".format(epoch)
    selfcal_difmap(fname=original_raw_uvf, outfname="myselfcaled.uvf",
                   path=data_dir, path_to_script="/home/ilya/github/ve/difmap/auto_selfcal", outpath=data_dir,
                   show_difmap_output=True)

    selfcaled_uvf = "myselfcaled.uvf"

    uvdata_sc = UVData(os.path.join(data_dir, selfcaled_uvf))

    if uvdata_sc._check_stokes_present("I"):
        stokes = "I"
    elif uvdata_sc._check_stokes_present("RR"):
        stokes = "RR"
    elif uvdata_sc._check_stokes_present("LL"):
        stokes = "LL"
    else:
        raise Exception

    uvdata_raw = UVData(os.path.join(data_dir, original_raw_uvf))
    uvdata_template = UVData(os.path.join(data_dir, original_raw_uvf))

    sc_data = uvdata_sc.hdu.data
    raw_data = uvdata_raw.hdu.data

    # Find gains products
    corrections = uvdata_raw.uvdata/uvdata_sc.uvdata

    # Create artificial raw data with known sky model and given corrections
    original_dfm_model = import_difmap_model(os.path.join(data_dir, "{}.mod".format(epoch)))

    # modelfit_difmap("myselfcaled.uvf", "{}.mod".format(epoch), "artificial.mdl", niter=200, stokes=stokes,
    #                 path=data_dir, mdl_path=data_dir, out_path=data_dir, show_difmap_output=True)
    # new_dfm_model = import_difmap_model("artificial.mdl", data_dir)
    # print([cg.p for cg in new_dfm_model])
    # print([cg.p for cg in original_dfm_model])
    # Create template model file
    # export_difmap_model([cg], os.path.join(data_dir, "template.mdl"), uvdata_template.frequency/10**9)
    # Modelfit artificial self-calibrated data

    # cg = CGComponent(0.5, 0, 0, 0.5)
    # model = Model(stokes=stokes)
    # model.add_components(*new_dfm_model)
    # model.add_components(ccmodel)

    if stokes == "I":
        use_V = True
    else:
        use_V = False
    noise = uvdata_template.noise(use_V=use_V)

    params = list()

    for i in range(n_mc):
        uvdata_template.substitute([ccmodel])
        uvdata_template.uvdata = uvdata_template.uvdata*corrections
        uvdata_template.noise_add(noise)
        uvdata_template.save(os.path.join(data_dir, "artificial.uvf"), rewrite=True, downscale_by_freq=True)

        # Self-calibrate
        selfcal_difmap(fname="artificial.uvf", outfname="artificial.uvf",
                       path=data_dir, path_to_script="/home/ilya/github/ve/difmap/auto_selfcal", outpath=data_dir,
                       show_difmap_output=True)

        modelfit_difmap("artificial.uvf", "{}.mod".format(epoch), "boot_artificial.mdl", niter=300, stokes=stokes,
                        path=data_dir, mdl_path=data_dir, out_path=data_dir, show_difmap_output=True)
        new_dfm_model = import_difmap_model("boot_artificial.mdl", data_dir)
        print([cg.p for cg in new_dfm_model])
        params.append([cg.p for cg in new_dfm_model])

    n_comps = len(original_dfm_model)
    param_name_dict = {0: "flux", 1: "dx", 2: "dy", 3: "fwhm"}

    with open(os.path.join(data_dir, "{}_errors_100.txt".format(epoch)), "w") as fo:

        for n_comp in range(n_comps):
            fo.write("component # {}\n".format(n_comp))
            n_params = original_dfm_model[n_comp].size
            for n_param in range(n_params):
                low, up = scoreatpercentile([mod[n_comp][n_param] for mod in params], [16, 84])
                sigma = 0.5*(up - low)
                if n_param in (1, 2):
                    add = "   "
                    unit = "mas"
                else:
                    add = " "
                    if n_param == 0:
                        unit = "Jy"
                    else:
                        unit = "mas"
                fo.write("{}{}{} {}\n".format(param_name_dict[n_param], add, format(sigma, '.5f'), unit))
            fo.write("\n")

