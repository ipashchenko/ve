import sys
import os
import glob
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from spydiff import modelfit_core_wo_extending, modelfit_difmap, import_difmap_model


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
source = "J0102+5824"
band = "X"
last_year = 2017
n_components = 4
data_path = f"/home/ilya/data/rfc/{source}"
uvfits_files = sorted(glob.glob(os.path.join(data_path, f"{source}_{band}_*_vis.fits")))
uvfits_file = [os.path.split(uvfits_file)[-1] for uvfits_file in uvfits_files]
epochs = list()
n_data = 0
for uvfits_file in uvfits_files:
    splitted = uvfits_file.split("_")
    year = splitted[2]
    if float(year) > last_year:
        break
    n_data += 1
    month = splitted[3]
    day = splitted[4]
    t = Time(f"{year}-{month}-{day}")
    epochs.append(t)

print(f"Number epochs < {last_year} : {n_data}")
# sys.exit(0)
epochs = sorted(epochs)
epochs = Time(epochs)
epochs_dt = [t.datetime for t in epochs]
epochs = epochs - epochs[0]
epochs_days = np.round(epochs.value, 2)

################################################################################

path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

beam_fractions = [1.0]
# mapsize_clean = (512, 0.2)
mapsize_clean = (1024, 0.1)

fluxes_wo_ext = list()
for uvfits in uvfits_files[:n_data]:
    print(f"Doing {uvfits}")
    try:
        result = modelfit_core_wo_extending(uvfits,
                                            beam_fractions, path=data_path,
                                            mapsize_clean=mapsize_clean,
                                            path_to_script=path_to_script,
                                            niter=500,
                                            out_path=out_dir,
                                            use_brightest_pixel_as_initial_guess=True,
                                            estimate_rms=True,
                                            stokes="i",
                                            use_ell=False, two_stage=False,
                                            dump_json_result=False)
        flux = result[1.0]["flux"]
    except:
        flux = None
    fluxes_wo_ext.append(flux)



fluxes_ncomps = list()
for uvfits in uvfits_files[:n_data]:
    print(f"Doing {uvfits}")
    modelfit_difmap(uvfits,
                    mdl_fname=f"in{n_components}_{band}.mdl", out_fname="out.mdl", niter=200, stokes='i',
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
    fluxes_ncomps.append(flux)
    # Position of the core
    r = np.hypot(core.p[1], core.p[2])



n = n_data
fig, axes = plt.subplots(1, 1, figsize=(7.5, 5))
# axes.plot(epochs_dt[:n], fluxes_wo_ext[:n], color="C0")
axes.scatter(epochs_dt[:n], fluxes_wo_ext[:n], color="C0", label="extract", s=40, alpha=0.7)
# axes.plot(epochs_dt[:n], fluxes_ncomps[:n], color="C1")
axes.scatter(epochs_dt[:n], fluxes_ncomps[:n], color="C1", label="Gaussians", s=40, alpha=0.7)
axes.set_xlabel("Epoch, year")
axes.set_ylabel("Core flux density, Jy")
axes.set_ylim([0, 5.])
plt.legend(loc="upper left")
plt.show()
fig.savefig(os.path.join(out_dir, f"{source}_core_lightcurve_{band}_two_methods.png"), bbox_inches="tight")
