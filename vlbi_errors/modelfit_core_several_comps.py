import sys
import os
import glob
import shutil
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from spydiff import modelfit_difmap, import_difmap_model


out_dir = "/home/ilya/github/bk_transfer/pics/flares/test"

# source = "J1310+3220"
source = "J0006-0623"
n_components = 3
data_path = f"/home/ilya/data/rfc/{source}"
uvfits_files = sorted(glob.glob(os.path.join(data_path, f"{source}_X_*_vis.fits")))
uvfits_file = [os.path.split(uvfits_file)[-1] for uvfits_file in uvfits_files]
epochs = list()
for uvfits_file in uvfits_files:
    splitted = uvfits_file.split("_")
    year = splitted[2]
    month = splitted[3]
    day = splitted[4]
    t = Time(f"{year}-{month}-{day}")
    epochs.append(t)

epochs = sorted(epochs)
epochs = Time(epochs)
epochs_dt = [t.datetime for t in epochs]
epochs = epochs - epochs[0]
epochs_days = np.round(epochs.value, 2)

################################################################################

path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

mapsize_clean = (1024, 0.1)
fluxes = list()

for uvfits in uvfits_files[:]:

    modelfit_difmap(uvfits,
                    mdl_fname=f"in{n_components}.mdl", out_fname="out.mdl", niter=200, stokes='i',
                    path=data_path, mdl_path=data_path, out_path=out_dir,
                    show_difmap_output=True,
                    save_dirty_residuals_map=False,
                    dmap_name=None, dmap_size=(1024, 0.1))

    components = import_difmap_model("out.mdl", out_dir)
    # Find closest to phase center component
    comps = sorted(components, key=lambda x: np.hypot(x.p[1], x.p[2]))
    core = comps[0]
    # Flux of the core
    flux = core.p[0]
    fluxes.append(flux)
    # Position of the core
    r = np.hypot(core.p[1], core.p[2])


    # results = modelfit_core_wo_extending(uvfits,
    #                                      beam_fractions, path=data_path,
    #                                      mapsize_clean=mapsize_clean,
    #                                      path_to_script=path_to_script,
    #                                      niter=500,
    #                                      out_path=out_dir,
    #                                      use_brightest_pixel_as_initial_guess=True,
    #                                      estimate_rms=True,
    #                                      stokes="i",
    #                                      use_ell=False,
    #                                      two_stage=False)
    # results_1.append(results)

# for result_1, result_2 in zip(results_1, results_2):
#     print("Two stages")
#     print(result_2)
#     print("One stage")
#     print(result_1)
#     print("==================")

fig, axes = plt.subplots(1, 1)
axes.plot(epochs_dt, fluxes)
axes.scatter(epochs_dt, fluxes)
axes.set_xlabel("Epoch, year")
axes.set_ylabel("Core flux density, Jy")
fig.savefig(os.path.join(out_dir, f"{source}_core_lightcurve_ncomps{n_components}.png"), bbox_inches="tight")
