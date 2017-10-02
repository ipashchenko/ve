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
data_dir = '/home/ilya/Dropbox/papers/boot/bias/new/stationary'
# download_mojave_uv_fits(source, epochs=[epoch], bands=['u'],
#                         download_dir=data_dir)

uv_fits_fnames = {freq: mojave_uv_fits_fname(source, freq, epoch) for freq in
                  ('x', 'j', 'u')}
for freq, uv_fits_fname in uv_fits_fnames.items():
    uv_fits_path = os.path.join(data_dir, uv_fits_fname)
    cg1 = CGComponent(2.0, 0., 0., 0.2)
    cg2 = CGComponent(1.0, 0., 0.3, 0.3)
    cg3 = CGComponent(0.5, 0., 1.5, 0.4)
    mdl = Model(stokes='I')
    mdl.add_components(cg1, cg2, cg3)
    uvdata = UVData(uv_fits_path)
    noise = uvdata.noise()
    for i in range(1, 101):
        uvdata = UVData(uv_fits_path)
        uvdata.substitute([mdl])
        uvdata.noise_add(noise)
        art_fits_fname = 'art_{}_{}.fits'.format(freq, i)
        art_fits_path = os.path.join(data_dir, art_fits_fname)
        uvdata.save(art_fits_path)

        # Here we should MCMC posterior
        modelfit_difmap(art_fits_fname, 'initial.mdl', 'out_{}_{}.mdl'.format(freq, i),
                        niter=100, path=data_dir, mdl_path=data_dir,
                        out_path=data_dir)

    params = list()
    for i in range(1, 101):
        comps = import_difmap_model('out_{}_{}.mdl'.format(freq, i), data_dir)
        params.append([comps[0].p[0], comps[0].p[2],
                       comps[1].p[0], comps[1].p[2],
                       comps[2].p[0], comps[2].p[2]])
    params = np.array(params)


    label_size = 16
    import matplotlib
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size

    # fig = corner.corner(params, labels=[r'$flux_1$', r'$r_1$',
    #                                     r'$flux_2$', r'$r_2$',
    #                                     r'$flux_3$', r'$r_3$'],
    #                     truths=[2, 0, 1.0, 0.3, 0.5, 1.5])
    # fig.tight_layout()
    # fig.savefig(os.path.join(data_dir, 'bias_corner_{}.eps'.format(freq)),
    #             dpi=1200, format='eps')

    # Estimate bias using bootstrap
    boot_dir = '/home/ilya/Dropbox/papers/boot/bias/new/stationary/boot'
    for i in range(1, 101, 1):
        art_fits_fname = 'art_{}_{}.fits'.format(freq, i)
        art_fits_path = os.path.join(data_dir, art_fits_fname)
        dfm_model_path = os.path.join(data_dir, 'out_{}_{}.mdl'.format(freq, i))
        bootstrap_uvfits_with_difmap_model(art_fits_path, dfm_model_path,
                                           boot_dir=boot_dir,
                                           out_txt_file=os.path.join(boot_dir, 'bias_{}_{}.txt'.format(freq, i)),
                                           out_plot_file=None,
                                           n_boot=100, niter=100)

originals = dict()
corrected = dict()
for freq in ('x', 'j', 'u'):
    originals[freq] = list()
    corrected[freq] = list()
    biases = list()
    r1 = list()
    for i in range(1, 101, 1):
        bias_file = os.path.join(boot_dir, 'bias_{}_{}.txt'.format(freq, i))
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
        orig_flux3 = float(lines[9].rstrip().split()[1])
        bmean_flux3 = float(lines[9].rstrip().split()[4])
        orig_r3 = float(lines[11].rstrip().split()[1])
        bmean_r3 = float(lines[11].rstrip().split()[4])
        delta_r1_obs = float(lines[7].rstrip().split()[1]) - float(lines[3].rstrip().split()[1])
        delta_r1_bmean = float(lines[7].rstrip().split()[4]) - float(lines[3].rstrip().split()[4])
        delta_r1_bias = delta_r1_bmean - delta_r1_obs
        delta_r1_bias_corrected = delta_r1_obs - delta_r1_bias
        biases.append([bmean_flux1-orig_flux1, bmean_r1-orig_r1,
                       bmean_flux2-orig_flux2, bmean_r2-orig_r2,
                       bmean_flux3-orig_flux3, bmean_r3-orig_r3])
        r1.append([delta_r1_obs, delta_r1_bias_corrected])

    biases = np.array(biases)


    for i in range(1, 101, 1):
        # bias_file = os.path.join(boot_dir, 'bias_{}.txt'.format(i))
        # with open(bias_file, 'r') as fo:
        #     lines = fo.readlines()
        # b_flux1 = float(lines[1].rstrip().split()[1])
        # b_flux2 = float(lines[5].rstrip().split()[1])
        # b_r1 = float(lines[3].rstrip().split()[1])
        # b_r2 = float(lines[7].rstrip().split()[1])
        b_flux1 = biases[i-1, 0]
        b_flux2 = biases[i-1, 2]
        b_flux3 = biases[i-1, 4]
        b_r1 = biases[i-1, 1]
        b_r2 = biases[i-1, 3]
        b_r3 = biases[i-1, 5]
        comps = import_difmap_model('out_{}_{}.mdl'.format(freq, i), data_dir)
        original_flux1 = comps[0].p[0]
        original_flux2 = comps[1].p[0]
        original_flux3 = comps[2].p[0]
        original_r1 = comps[0].p[2]
        original_r2 = comps[1].p[2]
        original_r3 = comps[2].p[2]
        originals[freq].append([original_flux1, original_r1,
                                original_flux2, original_r2,
                                original_flux3, original_r3])
        corrected[freq].append([original_flux1-b_flux1, original_r1-b_r1,
                                original_flux2-b_flux2, original_r2-b_r2,
                                original_flux3-b_flux3, original_r3-b_r3])
    originals[freq] = np.array(originals[freq])
    corrected[freq] = np.array(corrected[freq])

import matplotlib.pyplot as plt
# # f1
# fig, ax = plt.subplots(1, 1)
# ax.hist(originals[:, 0], alpha=0.3, color='#1f77b4', label=r'estimates',
#         range=[0.25, 2.5], bins=10)
# ax.axvline(np.mean(originals[:, 0]), color='#1f77b4', label='mean of estimates', ls='solid')
# ax.axvline(2., color='k', label='true value', ls='solid')
# ax.hist(corrected[:, 0], alpha=0.3, label=r'BC estimates', color='#ff7f0e',
#         range=[0.25, 2.5], bins=10)
# ax.axvline(np.mean(corrected[:, 0]), color='#ff7f0e', label='mean of BC estimates',
#            ls='dashed')
# ax.legend()
# ax.set_xlabel(r'$Flux_1$', fontsize=16)
# ax.set_ylabel(r'$N$', fontsize=16)
# fig.tight_layout()
# # fig.savefig(os.path.join(data_dir, 'bias_flux1.pdf'), dpi=1200, format='pdf')
# #
# r1


biases_freqs = dict()
for freq in ('x', 'j', 'u'):
    biases = list()
    for i in range(1, 101, 1):
        bias_file = os.path.join(boot_dir, 'bias_{}_{}.txt'.format(freq, i))
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
        orig_flux3 = float(lines[9].rstrip().split()[1])
        bmean_flux3 = float(lines[9].rstrip().split()[4])
        orig_r3 = float(lines[11].rstrip().split()[1])
        bmean_r3 = float(lines[11].rstrip().split()[4])
        biases.append([bmean_flux1-orig_flux1, bmean_r1-orig_r1,
                       bmean_flux2-orig_flux2, bmean_r2-orig_r2,
                       bmean_flux3-orig_flux3, bmean_r3-orig_r3])

    biases = np.array(biases)
    biases_freqs[freq] = biases

fig, ax = plt.subplots(1, 1)
ax.hist(biases_freqs['x'][:, 1], range=[-0.13, 0.05], normed=True,
        histtype='step', stacked=True, ls='solid', label='X-band', lw=2)
ax.hist(biases_freqs['j'][:, 1], range=[-0.13, 0.05], normed=True,
        histtype='step', stacked=True, ls='dashed', label='J-band', lw=2)
ax.hist(biases_freqs['u'][:, 1], range=[-0.13, 0.05], normed=True,
        histtype='step', stacked=True, ls='dotted', label='U-band', lw=2)
ax.legend(loc='upper left')
ax.set_xlabel(r'bias of $r_{1}$, mas', fontsize=16)
ax.set_ylabel(r'$P$', fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(data_dir, 'bias_r1_hist.pdf'), dpi=1200, format='pdf')




freq = 'u'
fig, ax = plt.subplots(1, 1)
fig.tight_layout()
ax.hist(originals[freq][:, 5]-originals[freq][:, 1], color='#1f77b4',
        label=r'estimates', range=[1.48, 1.54], bins=10, ls='solid',
        histtype='step', stacked=True, lw=2)
ax.axvline(np.mean(originals[freq][:, 5]-originals[freq][:, 1]),
           color='#1f77b4', label='mean of estimates', ls='solid', lw=2)
ax.axvline(1.5, color='k', label='true value', ls='dotted', lw=2)
ax.hist(originals[freq][:, 5]-corrected[freq][:, 1], label=r'BC-estimates',
        color='#ff7f0e', range=[1.48, 1.54], bins=10, ls='dashed',
        histtype='step', stacked=True, lw=2)
ax.axvline(np.mean(originals[freq][:, 5]-corrected[freq][:, 1]),
           color='#ff7f0e', label='mean of BC-estimates', ls='dashed', lw=2)
ax.legend(loc='upper right')
ax.set_xlabel(r'$\triangle r, mas$', fontsize=16)
ax.set_ylabel(r'$N$', fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(data_dir, 'bias_delta_r.pdf'), dpi=1200, format='pdf')
# # #
# # f2
# fig, ax = plt.subplots(1, 1)
# ax.hist(originals[:, 2], alpha=0.3, color='#1f77b4', label=r'estimates',
#         range=[-1.0, 3.20], bins=15)
# ax.axvline(np.mean(originals[:, 2]), color='#1f77b4', label='mean of estimates', ls='solid')
# ax.axvline(0.2, color='k', label='true value', ls='solid')
# ax.hist(corrected[:, 2], alpha=0.3, label=r'BC estimates', color='#ff7f0e',
#         range=[-1.0, 3.20], bins=15)
# ax.axvline(np.mean(corrected[:, 2]), color='#ff7f0e', label='mean of BC estimates',
#            ls='dashed')
# ax.legend()
# ax.set_xlabel(r'$Flux_2$')
# ax.set_ylabel(r'$N$')
# fig.tight_layout()
# # fig.savefig(os.path.join(data_dir, 'bias_flux2.pdf'), dpi=1200, format='pdf')
# # # #
# # # r2
# fig, ax = plt.subplots(1, 1)
# ax.hist(originals[:, 3], alpha=0.3, color='#1f77b4', label=r'estimates',
#         range=[0.0, 0.70], bins=15)
# ax.axvline(np.mean(originals[:, 3]), color='#1f77b4', label='mean of estimates', ls='solid')
# ax.axvline(0.2, color='k', label='true value', ls='solid')
# ax.hist(corrected[:, 3], alpha=0.3, label=r'BC estimates', color='#ff7f0e',
#         range=[0.0, 0.70], bins=15)
# ax.axvline(np.mean(corrected[:, 3]), color='#ff7f0e', label='mean of BC estimates',
#            ls='dashed')
# ax.legend()
# ax.set_xlabel(r'$r_2$')
# ax.set_ylabel(r'$N$')
# fig.tight_layout()
# # fig.savefig(os.path.join(data_dir, 'bias_r2.pdf'), dpi=1200, format='pdf')
