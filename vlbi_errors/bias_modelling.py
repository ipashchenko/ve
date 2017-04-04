import os
import numpy as np
import corner
from components import CGComponent
from model import Model
from uv_data import UVData
from mojave import mojave_uv_fits_fname, download_mojave_uv_fits
from spydiff import modelfit_difmap, import_difmap_model
from bootstrap import bootstrap_uvfits_with_difmap_model


source = '1514-241'
epoch = '2006_04_28'
data_dir = '/home/ilya/Dropbox/papers/boot/bias/new'
download_mojave_uv_fits(source, epochs=[epoch], bands=['u'],
                        download_dir=data_dir)

uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
uv_fits_path = os.path.join(data_dir, uv_fits_fname)
cg1 = CGComponent(2., 0., 0., 0.2)
cg2 = CGComponent(0.25, 0., 0.2, 0.2)
mdl = Model(stokes='I')
mdl.add_components(cg1, cg2)
uvdata = UVData(uv_fits_path)
noise = uvdata.noise()
for i in range(1, 11):
    uvdata = UVData(uv_fits_path)
    uvdata.substitute([mdl])
    uvdata.noise_add(noise)
    art_fits_fname = 'art_{}.fits'.format(i)
    art_fits_path = os.path.join(data_dir, art_fits_fname)
    uvdata.save(art_fits_path)
    modelfit_difmap(art_fits_fname, 'initial.mdl', 'out_{}.mdl'.format(i),
                    niter=100, path=data_dir, mdl_path=data_dir,
                    out_path=data_dir)

params = list()
for i in range(1, 11):
    comps = import_difmap_model('out_{}.mdl'.format(i), data_dir)
    params.append([comps[0].p[0], comps[0].p[1], comps[1].p[0], comps[1].p[2]])
params = np.array(params)


label_size = 16
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size

fig = corner.corner(params, labels=[r'$flux_1$', r'$r_1$', r'$flux_2$', r'$r_2$'],
                    truths=[2, 0, 0.25, 0.2])
fig.tight_layout()
fig.savefig(os.path.join(data_dir, 'bias_corner.eps'), dpi=1200, format='eps')


# Estimate bias using bootstrap
boot_dir = '/home/ilya/Dropbox/papers/boot/bias/new/boot'
for i in range(1, 11, 1):
    art_fits_fname = 'art_{}.fits'.format(i)
    art_fits_path = os.path.join(data_dir, art_fits_fname)
    dfm_model_path = os.path.join(data_dir, 'out_{}.mdl'.format(i))
    bootstrap_uvfits_with_difmap_model(art_fits_path, dfm_model_path,
                                       boot_dir=boot_dir,
                                       out_txt_file='bias_{}.txt'.format(i),
                                       out_plot_file='plot_{}.txt'.format(i))

biases = list()
for i in range(1, 11, 1):
    bias_file = os.path.join(boot_dir, 'bias_{}.txt'.format(i))
    with open(bias_file, 'r') as fo:
        lines = fo.readlines()
    orig_flux1 = float(lines[1].rstrip().split()[1])
    bmean_flux1 = float(lines[1].rstrip().split()[4])
    orig_flux2 = float(lines[5].rstrip().split()[1])
    bmean_flux2 = float(lines[5].rstrip().split()[4])
    orig_r1 = float(lines[3].rstrip().split()[1])
    bmean_r1 = float(lines[3].rstrip().split()[4])
    orig_r2 = float(lines[7].rstrip().split()[1])
    bmean_r2 = float(lines[7].rstrip().split()[4])
    biases.append([bmean_flux1-orig_flux1, bmean_r1-orig_r1,
                   bmean_flux2-orig_flux2, bmean_r2-orig_r2])

biases = np.array(biases)


originals = list()
corrected = list()
for i in range(1, 11, 1):
    bias_file = os.path.join(boot_dir, 'bias_{}.txt'.format(i))
    with open(bias_file, 'r') as fo:
        lines = fo.readlines()
    b_flux1 = float(lines[1].rstrip().split()[1])
    b_flux2 = float(lines[5].rstrip().split()[1])
    b_r1 = float(lines[3].rstrip().split()[1])
    b_r2 = float(lines[7].rstrip().split()[1])
    comps = import_difmap_model('out_{}.mdl'.format(i), data_dir)
    original_flux1 = comps[0].p[0]
    original_flux2 = comps[1].p[0]
    original_r1 = comps[0].p[2]
    original_r2 = comps[1].p[2]
    originals.append([original_flux1, original_r1, original_flux2, original_r2])
    corrected.append([original_flux1-b_flux1, original_r1-b_r1,
                      original_flux2-b_flux2, original_r2-b_r2])
originals = np.array(originals)
corrected = np.array(corrected)


# # f1
# fig, ax = plt.subplots(1, 1)
# fig.tight_layout()
# ax.hist(originals[:, 0], alpha=0.3, range=[0.5, 1.4], label=r'original')
# ax.axvline(1.)
# ax.hist(corrected[:, 0], alpha=0.3, range=[0.5, 1.4],
#         label=r'bias corrected')
# ax.legend()
# ax.set_xlabel(r'$Flux_1$')
# ax.set_ylabel(r'$N$')
# fig.savefig(os.path.join(data_dir, 'bias_flux1.pdf'), dpi=1200, format='pdf')
#
# # r1
# fig, ax = plt.subplots(1, 1)
# fig.tight_layout()
# ax.hist(originals[:, 1], alpha=0.3, range=[-0.15, 0.05], label=r'original')
# ax.axvline(0.)
# ax.hist(corrected[:, 1], alpha=0.3, range=[-0.15, 0.05],
#         label=r'bias corrected')
# ax.legend()
# ax.set_xlabel(r'$r_1$')
# ax.set_ylabel(r'$N$')
# fig.savefig(os.path.join(data_dir, 'bias_r1.pdf'), dpi=1200, format='pdf')
#
# # f2
# fig, ax = plt.subplots(1, 1)
# fig.tight_layout()
# ax.hist(originals[:, 2], alpha=0.3, range=[0.0, 1.2], label=r'original')
# ax.axvline(0.25)
# ax.hist(corrected[:, 2], alpha=0.3, range=[0.0, 1.2],
#         label=r'bias corrected')
# ax.legend()
# ax.set_xlabel(r'$Flux_2$')
# ax.set_ylabel(r'$N$')
# fig.savefig(os.path.join(data_dir, 'bias_flux2.pdf'), dpi=1200, format='pdf')
#
# # r2
# fig, ax = plt.subplots(1, 1)
# fig.tight_layout()
# ax.hist(originals[:, 3], alpha=0.3, range=[0.0, 0.5], label=r'original')
# ax.axvline(0.2)
# ax.hist(corrected[:, 3], alpha=0.3, range=[0.0, 0.5],
#         label=r'bias corrected')
# ax.legend()
# ax.set_xlabel(r'$r_2$')
# ax.set_ylabel(r'$N$')
# fig.savefig(os.path.join(data_dir, 'bias_r2.pdf'), dpi=1200, format='pdf')
