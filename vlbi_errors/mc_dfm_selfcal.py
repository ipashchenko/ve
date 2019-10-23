import glob
import os
import numpy as np
from uv_data import UVData
from model import Model
from spydiff import clean_difmap, selfcal_difmap, import_difmap_model, export_difmap_model, modelfit_difmap
import sys
sys.path.insert(0, '/home/ilya/github/dterms')
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


data_dir = "/home/ilya/data/silke"
# epoch = "1997_08_18"
epochs = glob.glob(os.path.join(data_dir, "*_*_*.mod"))
epochs = [os.path.split(epoch)[-1] for epoch in epochs]
epochs = sorted([epoch.split(".")[0] for epoch in epochs])


for epoch in epochs[1:2]:
    # Self-cal with script
    original_raw_uvf = "1502+106.u.{}.uvf_raw_edt".format(epoch)
    selfcal_difmap(fname=original_raw_uvf, outfname="myselfcaled.uvf",
                   path=data_dir, path_to_script="/home/ilya/github/ve/difmap/auto_selfcal", outpath=data_dir,
                   show_difmap_output=True)

    selfcaled_uvf = "myselfcaled.uvf"
    mapsize_clean = 512, 0.1
    n_mc = 100


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

    modelfit_difmap("myselfcaled.uvf", "{}.mod".format(epoch), "artificial.mdl", niter=200, stokes=stokes,
                    path=data_dir, mdl_path=data_dir, out_path=data_dir, show_difmap_output=True)
    new_dfm_model = import_difmap_model("artificial.mdl", data_dir)
    print([cg.p for cg in new_dfm_model])
    print([cg.p for cg in original_dfm_model])
    # Create template model file
    # export_difmap_model([cg], os.path.join(data_dir, "template.mdl"), uvdata_template.frequency/10**9)
    # Modelfit artificial self-calibrated data

    # cg = CGComponent(0.5, 0, 0, 0.5)
    model = Model(stokes=stokes)
    model.add_components(*new_dfm_model)

    if stokes == "I":
        use_V = True
    else:
        use_V = False
    noise = uvdata_template.noise(use_V=use_V)

    params = list()

    for i in range(n_mc):
        uvdata_template.substitute([model])
        uvdata_template.uvdata = uvdata_template.uvdata*corrections
        uvdata_template.noise_add(noise)
        uvdata_template.save(os.path.join(data_dir, "artificial.uvf"), rewrite=True, downscale_by_freq=True)

        # Self-calibrate
        selfcal_difmap(fname="artificial.uvf", outfname="artificial.uvf",
                       path=data_dir, path_to_script="/home/ilya/github/ve/difmap/auto_selfcal", outpath=data_dir,
                       show_difmap_output=True)

        modelfit_difmap("artificial.uvf", "artificial.mdl", "boot_artificial.mdl", niter=100, stokes=stokes,
                        path=data_dir, mdl_path=data_dir, out_path=data_dir, show_difmap_output=True)
        new_dfm_model = import_difmap_model("boot_artificial.mdl", data_dir)
        print([cg.p for cg in new_dfm_model])
        params.append([cg.p for cg in new_dfm_model])

    n_comps = len(new_dfm_model)
    param_name_dict = {0: "flux", 1: "dx", 2: "dy", 3: "fwhm"}

    with open(os.path.join(data_dir, "{}_errors.txt".format(epoch)), "w") as fo:

        for n_comp in range(n_comps):
            fo.write("component # {}\n".format(n_comp))
            n_params = new_dfm_model[n_comp].size
            for n_param in range(n_params):
                low, up = scoreatpercentile([mod[n_comp][n_param] for mod in params], [16, 86])
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

