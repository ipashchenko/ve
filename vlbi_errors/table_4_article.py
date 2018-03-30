import glob
import os
import numpy as np
from spydiff import import_difmap_model, clean_difmap
from from_fits import create_clean_image_from_fits_file
from utils import hdi_of_mcmc
import matplotlib
label_size = 16
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
# matplotlib.rcParams['text.latex.preview'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'cm'

import matplotlib.pyplot as plt


base_dir = "/home/ilya/Dropbox/papers/boot/new_pics/corner/new/parametric/1807+698/"
mcmc_samples = os.path.join(base_dir, "samples_of_mcmc.txt")
# (60000, 50)
mcmc = np.loadtxt(mcmc_samples)
mcmc = mcmc[::10, :]

booted_mdl_paths = glob.glob(os.path.join(base_dir, "mdl_booted_*"))
boot_samples = list()
for booted_mdl in booted_mdl_paths:
    comps = import_difmap_model(booted_mdl)
    comps = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))
    params = list()
    for comp in comps:
        params.extend(list(comp.p))
    boot_samples.append(params)

boot = np.atleast_2d(boot_samples)

cred_mass = 0.68
count = 0
param_n = 2
ratios = list()
distances = list()
fluxes = list()
boot_stds = list()
mcmc_stds = list()
comps = import_difmap_model(os.path.join(base_dir, "new2.mdl"))
comps = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))
length = sum([comp.size for comp in comps])
for j, comp in enumerate(comps):
    hdi_min, hdi_max = hdi_of_mcmc(boot[:, count + param_n], cred_mass=cred_mass)
    boot_std = hdi_max - hdi_min
    hdi_min, hdi_max = hdi_of_mcmc(mcmc[:, count + param_n], cred_mass=cred_mass)
    mcmc_std = hdi_max - hdi_min
    # boot_std = np.std(boot[:, count + param_n])
    # mcmc_std = np.std(mcmc[:, count + param_n])
    count += len(comp)
    ratios.append(boot_std/mcmc_std)
    boot_stds.append(boot_std)
    mcmc_stds.append(mcmc_std)
    distances.append(np.hypot(comp.p[1], comp.p[2]))
    fluxes.append(comp.p[0])


# boot_stds = np.hypot(boot_std_1, boot_std_2)
# mcmc_stds = np.hypot(mcmc_std_1, mcmc_std_2)
# position_ratios = np.array(boot_stds)/np.array(mcmc_stds)
# np.savetxt(os.path.join(base_dir, "position_ratios.txt"), position_ratios)
# np.savetxt(os.path.join(base_dir, "flux_ratios.txt"), ratios)
# np.savetxt(os.path.join(base_dir, "size_ratios.txt"), ratios)
# np.savetxt(os.path.join(base_dir, "distances.txt"), distances)
# np.savetxt(os.path.join(base_dir, "fluxes.txt"), fluxes)


position_ratios = np.loadtxt(os.path.join(base_dir, "position_ratios.txt"))
flux_ratios = np.loadtxt(os.path.join(base_dir, "flux_ratios.txt"))
size_ratios = np.loadtxt(os.path.join(base_dir, "size_ratios.txt"))
distances = np.loadtxt(os.path.join(base_dir, "distances.txt"))
fluxes = np.loadtxt(os.path.join(base_dir, "fluxes.txt"))
# Get beam
uv_fits = "1807+698.u.2007_07_03.uvf"
path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
# clean_difmap(uv_fits, "cc.fits", "I", (1024, 0.1), path=base_dir,
#              path_to_script=path_to_script, outpath=base_dir)
ccimage = create_clean_image_from_fits_file(os.path.join(base_dir, "cc.fits"))
beam = ccimage.beam
beam = np.sqrt(beam[0]*beam[1])

# First row [:, 0] - for distance dependence
# Second row [:, 1] - for flux dependence
# fig, axes = plt.subplots(3, 2, sharex=True)
fig = plt.figure()

axes00 = fig.add_subplot(3, 2, 1)
axes01 = fig.add_subplot(3, 2, 2, sharey=axes00)
axes10 = fig.add_subplot(3, 2, 3, sharex=axes00)
axes11 = fig.add_subplot(3, 2, 4, sharey=axes10, sharex=axes01)
axes20 = fig.add_subplot(3, 2, 5, sharex=axes00)
axes21 = fig.add_subplot(3, 2, 6, sharex=axes01, sharey=axes20)

axes00.plot(np.array(distances)/beam, flux_ratios, 'o')
# axes.legend(loc='upper right')
# axes.set_xlabel("Distance from phase center, [beam widths]")
axes00.set_ylabel(r"$\sigma_{boot}^{flux}$ / $\sigma_{mcmc}^{flux}$", size=20)
# fig.savefig(os.path.join(base_dir, "flux_std_ratio_vs_distance.pdf"), format="pdf",
#             bbox_inches='tight', dpi=600)

# fig, axes = plt.subplots()
axes01.plot(np.array(fluxes), flux_ratios, 'o')
# axes.legend(loc='upper right')
# axes.set_xlabel("Flux of component, [Jy]")
# axes.set_ylabel(r"$\sigma_{boot}^{flux}$ / $\sigma_{mcmc}^{flux}$", size=20)
# fig.savefig(os.path.join(base_dir, "flux_std_ratio_vs_flux.pdf"), format="pdf",
#             bbox_inches='tight', dpi=600)


axes10.plot(np.array(distances)/beam, position_ratios, 'o')
axes10.set_ylabel(r"$\sigma_{boot}^{position}$ / $\sigma_{mcmc}^{position}$", size=20)
axes11.plot(np.array(fluxes), position_ratios, 'o')

axes20.plot(np.array(distances)/beam, size_ratios, 'o')
axes20.set_ylabel(r"$\sigma_{boot}^{size}$ / $\sigma_{mcmc}^{size}$", size=20)
axes20.set_xlabel("Distance from phase center, [beam widths]")
axes21.plot(np.array(fluxes), size_ratios, 'o')
axes21.set_xlabel("Flux of component, [Jy]")
fig.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
# fig.subplots_adjust(wspace=0)
fig.savefig(os.path.join(base_dir, "boot_to_mcmc.pdf"), format="pdf",
            bbox_inches='tight', dpi=600)