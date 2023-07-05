import sys
import os
import glob
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from spydiff import modelfit_core_wo_extending


out_dir = "/home/ilya/github/bk_transfer/pics/flares/test"

# # Simulated files ##############################################################
# source = "simulated"
# data_path = "/home/ilya/github/bk_transfer/pics/flares/test"
# uvfits_files = ["template_S_1936.6.uvf", "template_S_2085.5.uvf",
#                 "template_S_2234.5.uvf", "template_S_2383.4.uvf"]
# uvfits_files = sorted(uvfits_files)
# epochs_days = [float(fn.split("_")[-1][:-4]) for fn in uvfits_files]
# epochs_days = np.array(epochs_days)
# epochs_days = epochs_days - epochs_days[0]


# Real rfc files ###############################################################
# source = "J1310+3220"
source = "J0006-0623"
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

beam_fractions = [1.0]
mapsize_clean = (1024, 0.1)
results_1 = list()
results_2 = list()

for uvfits in uvfits_files[:]:
    results = modelfit_core_wo_extending(uvfits,
                                         beam_fractions, path=data_path,
                                         mapsize_clean=mapsize_clean,
                                         path_to_script=path_to_script,
                                         niter=500,
                                         out_path=out_dir,
                                         use_brightest_pixel_as_initial_guess=True,
                                         estimate_rms=True,
                                         stokes="i",
                                         use_ell=False)
    results_2.append(results)

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

fluxes = [result[1.0]["flux"] for result in results_2]

fig, axes = plt.subplots(1, 1)
axes.plot(epochs_dt, fluxes)
axes.scatter(epochs_dt, fluxes)
axes.set_xlabel("Epoch, year")
axes.set_ylabel("Core flux density, Jy")
fig.savefig(os.path.join(out_dir, f"{source}_core_lightcurve_woextending.png"), bbox_inches="tight")
