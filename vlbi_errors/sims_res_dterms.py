import os
import numpy as np
from from_fits import (create_ccmodel_from_fits_file,
                       create_uvdata_from_fits_file)
from PA import PA
from utils import baselines_2_ants

# Coordinates N, W (deg)
br_coords = (48.13117, -119.68325)
fd_coords = (30.635214, -103.944826)
br_lat = br_coords[0]
br_long = br_coords[1]
fd_lat = fd_coords[0]
fd_long = fd_coords[1]

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

baseline = uvdata.baselines[0]
print "baseline {}".format(baseline)
i, indxs_i = uvdata._choose_uvdata(baselines=[baseline], IF=1, stokes='I')
rl, indxs_rl = uvdata._choose_uvdata(baselines=[baseline], IF=1, stokes='RL')
lr, indxs_lr = uvdata._choose_uvdata(baselines=[baseline], IF=1, stokes='LR')
i = i[:, 0, 0]
rl = rl[:, 0, 0]
lr = lr[:, 0, 0]

import pyfits as pf
hdus = pf.open(os.path.join(uv_data_dir, uv_fname))
hdu = hdus[3]
ant1, ant2 = baselines_2_ants([baseline])
ant1_name = hdu.data['ANNAME'][ant1 - 1]
ant2_name = hdu.data['ANNAME'][ant2 - 1]
ra = float(hdus[0].header['OBSRA'])
dec = float(hdus[0].header['OBSDEC'])
tzero = hdu.header['PZERO{}'.format(find_card_from_header(hdu.header,
                                                          value='DATE')[0][0][-1])]
times = uvdata.data['time'][indxs_i] + tzero

pa_br = np.array(PA(times, ra, dec, br_lat, br_long))
pa_fd = np.array(PA(times, ra, dec, fd_lat, fd_long))

delta_d = 0.05

d_br = np.random.normal(0, delta_d) + 1j * np.random.normal(0, delta_d)
d_fd = np.random.normal(0, delta_d) + 1j * np.random.normal(0, delta_d)

d_noise_rl = (d_br * np.exp(2 * 1j * pa_br) +
              d_fd.conjugate() * np.exp(2 * 1j * pa_fd)) * i
d_noise_lr = (d_br * np.exp(-2 * 1j * pa_br) +
              d_fd.conjugate() * np.exp(-2 * 1j * pa_fd)) * i
th_noise_std = uvdata.noise()[baseline][0]

noise = d_noise_rl + np.random.normal(0, th_noise_std, size=len(d_noise_rl)) +\
    1j * np.random.normal(0, th_noise_std, size=len(d_noise_rl))
