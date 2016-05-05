import matplotlib
label_size = 6
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
import os
import numpy as np
from uv_data import UVData
from model import Model
from from_fits import create_model_from_fits_file
from bootstrap import CleanBootstrap
from spydiff import import_difmap_model
from utils import baselines_2_ants


# Path to project's root directory
base_path = '/home/ilya/sandbox/heteroboot/'
path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'

# Colors used
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Workflow for one source
source = '0945+408'
epoch = '2007_04_18'
band = 'u'
# TODO: Standard it
image_fname = 'original_cc.fits'
uv_fname_cc = '0945+408.u.2007_04_18.uvf'
uv_fname_uv = '0945+408.u.2007_04_18.uvf'
dfm_model_fname = 'dfmp_original_model.mdl'

comps = import_difmap_model(dfm_model_fname, base_path)
model_uv = Model(stokes='I')
model_uv.add_components(*comps)
uvdata = UVData(os.path.join(base_path, uv_fname_uv))
uvdata_m = UVData(os.path.join(base_path, uv_fname_uv))
uvdata_m.substitute([model_uv])
uvdata_r = uvdata - uvdata_m

# Plot uv-data
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
uvdata.uvplot(style='re&im', freq_average=True)
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/uvdata_original.png',
                          bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()

# # Plot residuals in radplot
# label_size = 12
# matplotlib.rcParams['xtick.labelsize'] = label_size
# matplotlib.rcParams['ytick.labelsize'] = label_size
# for i, color in enumerate(colors):
#     uvdata_r.uvplot(baselines=uvdata_r.baselines[i], freq_average=True,
#                     color=color, style='re&im')
# matplotlib.pyplot.show()
# matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/radplot_res_gauss.png',
#                           bbox_inches='tight', dpi=400)
# matplotlib.pyplot.close()

# Plot model in radplot
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
uvdata_m.uvplot(style='re&im', freq_average=True)
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/radplot_mod_gauss.png',
                          bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()

# Plot residuals
label_size = 6
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
nrows = int(np.sqrt(2. * len(uvdata_r.baselines)))
# nrows = int(np.sqrt(2. * len(uvdata_r.baselines))) + 1
fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=nrows, sharex=True,
                                       sharey=True)
fig.set_size_inches(18.5, 18.5)
matplotlib.pyplot.rcParams.update({'axes.titlesize': 'small'})
i, j = 0, 0

for baseline in uvdata_r.baselines:
    try:
        print "baseline {}".format(baseline)
        res = uvdata_r._choose_uvdata(baselines=[baseline], freq_average=True,
                                      stokes='I')[0]
        print "Plotting to {}-{}".format(i, j)
        axes[i, j].hist(res.real, range=[-0.15, 0.15], color="#4682b4")
        axes[i, j].axvline(0.0, lw=1, color='r')
        # axes[i, j].set_title("{}-{}".format(baseline, 'real'), fontsize=4)
        axes[i, j].set_xticks([-0.1, 0.1])
        j += 1
        # Plot first row first
        if j // nrows > 0:
            # Then second row, etc...
            i += 1
            j = 0
        print "Plotting to {}-{}".format(i, j)
        axes[i, j].hist(res.imag, range=[-0.15, 0.15], color="#4682b4")
        axes[i, j].axvline(0.0, lw=1, color='r')
        # axes[i, j].set_title("{}-{}".format(baseline, 'imag'), fontsize=4)
        axes[i, j].set_xticks([-0.1, 0.1])
        j += 1
        # Plot first row first
        if j // nrows > 0:
            # Then second row, etc...
            i += 1
            j = 0
    except IndexError:
        break
fig.show()
fig.savefig("{}.{}".format(os.path.join(base_path, 'res_gauss'), 'png'),
            bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()

ccmodel = create_model_from_fits_file(os.path.join(base_path, image_fname))
uvdata = UVData(os.path.join(base_path, uv_fname_cc))
uvdata_m = UVData(os.path.join(base_path, uv_fname_cc))
uvdata_m.substitute([ccmodel])
uvdata_r = uvdata - uvdata_m

# FIXME: Plot different colors for different baselines!
# Plot residuals in radplot
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
uvdata_r.uvplot(style='re&im', freq_average=True, re_range=[-0.3, 0.3],
                im_range=[-0.3, 0.3])
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/radplot_res_cc.png',
                          bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()
# for i, color in enumerate(colors):
#     uvdata_r.uvplot(baselines=uvdata_r.baselines[i], freq_average=True,
#                     color=color, style='re&im')
# matplotlib.pyplot.show()
# matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/radplot_res_cc.png',
#                           bbox_inches='tight', dpi=400)
# matplotlib.pyplot.close()

# Plot model in radplot
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
uvdata_m.uvplot(style='re&im', freq_average=True)
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/radplot_mod_cc.png',
                          bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()

# Plot residuals
label_size = 6
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
# nrows = int(np.sqrt(2. * len(uvdata_r.baselines))) + 1
nrows = int(np.sqrt(2. * len(uvdata_r.baselines)))
fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=nrows, sharex=True,
                                       sharey=True)
fig.set_size_inches(18.5, 18.5)
matplotlib.pyplot.rcParams.update({'axes.titlesize': 'small'})
i, j = 0, 0

for baseline in uvdata_r.baselines:
    try:
        print "baseline {}".format(baseline)
        res = uvdata_r._choose_uvdata(baselines=[baseline], freq_average=True,
                                      stokes='I')[0]
        print "Plotting to {}-{}".format(i, j)
        axes[i, j].hist(res.real, color="#4682b4", range=[-0.15, 0.15])
        axes[i, j].axvline(0.0, lw=1, color='r')
        # axes[i, j].set_title("{}-{}".format(baseline, 'real'), fontsize=4)
        axes[i, j].set_xticks([-0.1, 0.1])
        j += 1
        # Plot first row first
        if j // nrows > 0:
            # Then second row, etc...
            i += 1
            j = 0
        print "Plotting to {}-{}".format(i, j)
        axes[i, j].hist(res.imag, color="#4682b4", range=[-0.15, 0.15])
        axes[i, j].axvline(0.0, lw=1, color='r')
        # axes[i, j].set_title("{}-{}".format(baseline, 'imag'), fontsize=4)
        axes[i, j].set_xticks([-0.1, 0.1])
        j += 1
        # Plot first row first
        if j // nrows > 0:
            # Then second row, etc...
            i += 1
            j = 0
    except IndexError:
        break
fig.show()
fig.savefig("{}.{}".format(os.path.join(base_path, 'res_cc'), 'png'),
            bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()

# Plot uv-coverage
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
for i, color in enumerate(colors):
    uvdata_r.uv_coverage(baselines=baselines_2_ants(uvdata_r.baselines)[i],
                         sym='.{}'.format(color))
matplotlib.pyplot.show()
matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/uv.png',
                          bbox_inches='tight', dpi=400)
matplotlib.pyplot.close()

# Plot several histograms
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
for i, color in enumerate(colors):
    baseline = uvdata_r.baselines[i]
    res = uvdata_r._choose_uvdata(baselines=[baseline], freq_average=True,
                                  stokes='I')[0]
    matplotlib.pyplot.hist(res.real, color="#4682b4", range=[-0.16, 0.16],
                           bins=20)
    matplotlib.pyplot.axvline(0.0, lw=2, color='r')
    matplotlib.pyplot.xlabel("residuals, Jy")
    matplotlib.pyplot.ylabel("N")
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('/home/ilya/sandbox/heteroboot/res_b{}.png'.format(i+1),
                              bbox_inches='tight', dpi=400)
    matplotlib.pyplot.close()
