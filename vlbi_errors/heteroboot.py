import os
import matplotlib.pyplot as plt
from zhenya import (put_im_files_to_dirs, put_uv_files_to_dirs, im_fits_path,
                    uv_fits_path, create_dirtree)
from from_fits import (create_ccmodel_from_fits_file,
                       create_uvdata_from_fits_file)
from bootstrap import CleanBootstrap


# Directories that contain data for loading in project
uv_data_dir = '/home/ilya/Dropbox/Zhenya/to_ilya/uv/'
im_data_dir = '/home/ilya/Dropbox/Zhenya/to_ilya/clean_images/'
# Path to project's root directory
base_path = '/home/ilya/sandbox/heteroboot/'
path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'

# Workflow for one source
source = '0952+179'
epoch = '2007_04_30'
band = 'c1'
n_boot = 10
stoke = 'i'
image_fname = '0952+179.c1.2007_04_30.i.fits'
uv_fname = '0952+179.C1.2007_04_30.PINAL'

ccmodel = create_ccmodel_from_fits_file(os.path.join(im_data_dir, image_fname))
uvdata = create_uvdata_from_fits_file(os.path.join(uv_data_dir, uv_fname))
uvdata_m = create_uvdata_from_fits_file(os.path.join(uv_data_dir, uv_fname))
uvdata_m.substitute([ccmodel])
uvdata_r = uvdata - uvdata_m
bootstrap = CleanBootstrap([ccmodel], uvdata)

curdir = os.getcwd()
os.chdir(base_path)
bootstrap.run(n=n_boot, outname=['boot', '.fits'])
os.chdir(curdir)

nrows = int(np.sqrt(2. * len(uvdata_r.baselines))) + 1
fig, axes = plt.subplots(nrows=nrows, ncols=nrows, sharex=True,
                        sharey=True)
fig.set_size_inches(18.5, 18.5)
plt.rcParams.update({'axes.titlesize': 'small'})
i, j = 0, 0

for baseline in uvdata_r.baselines:
    print "baseline {}".format(baseline)
    res = uvdata_r._choose_uvdata(baselines=[baseline], freq_average=True,
                                  stokes='I')[0][:, 0]
    print "Plotting to {}-{}".format(i, j)
    axes[i, j].hist(res.real, normed=True)
    axes[i, j].axvline(0.0, lw=2, color='r')
    axes[i, j].set_title("{}-{}".format(baseline, 'real'))
    axes[i, j].set_xticks([-0.05, 0.05])
    j += 1
    # Plot first row first
    if j // nrows > 0:
        # Then second row, etc...
        i += 1
        j = 0
    print "Plotting to {}-{}".format(i, j)
    axes[i, j].hist(res.imag, normed=True)
    axes[i, j].axvline(0.0, lw=2, color='r')
    axes[i, j].set_title("{}-{}".format(baseline, 'imag'))
    axes[i, j].set_xticks([-0.05, 0.05])
    j += 1
    # Plot first row first
    if j // nrows > 0:
        # Then second row, etc...
        i += 1
        j = 0
fig.show()
fig.savefig("{}.{}".format(os.path.join(base_path, 'res'), 'png'),
                           bbox_inches='tight', dpi=200)

