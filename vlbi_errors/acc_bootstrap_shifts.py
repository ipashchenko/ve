import os
import glob
import math
import numpy as np
from uv_data import UVData
from from_fits import (create_model_from_fits_file,
                       create_clean_image_from_fits_file,
                       create_image_from_fits_file)
from bootstrap import CleanBootstrap
from spydiff import clean_difmap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.labelsize'] = 14


path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
data_dir = '/home/ilya/Dropbox/ACC/3c120/boot'
uvdata_dir = '/home/ilya/Dropbox/ACC/3c120/uvdata'
uv_fits_u = '0430+052.u.2006_05_24.uvf'
uv_fits_x = '0430+052.x.2006_05_24.uvf'

# Clean original uv-data with native beam
clean_difmap(uv_fits_u, 'u_cc.fits', 'I', (1024, 0.1), path=uvdata_dir,
             path_to_script=path_to_script, show_difmap_output=True,
             outpath=data_dir)
clean_difmap(uv_fits_x, 'x_cc.fits', 'I', (1024, 0.1), path=uvdata_dir,
             path_to_script=path_to_script, show_difmap_output=True,
             outpath=data_dir)

# Clean original uv-data with common beam
clean_difmap(uv_fits_x, 'x_cc_same.fits', 'I', (1024, 0.1), path=uvdata_dir,
             path_to_script=path_to_script, show_difmap_output=True,
             outpath=data_dir)
ccimage_x = create_clean_image_from_fits_file(os.path.join(data_dir, 'x_cc.fits'))
clean_difmap(uv_fits_u, 'u_cc_same.fits', 'I', (1024, 0.1), path=uvdata_dir,
             path_to_script=path_to_script, show_difmap_output=True,
             outpath=data_dir, beam_restore=ccimage_x.beam)

u_model = create_model_from_fits_file(os.path.join(data_dir, 'u_cc.fits'))
x_model = create_model_from_fits_file(os.path.join(data_dir, 'x_cc.fits'))
u_uvdata = UVData(os.path.join(uvdata_dir, uv_fits_u))
x_uvdata = UVData(os.path.join(uvdata_dir, uv_fits_x))

# Bootstrap uv-data with original CLEAN models
xboot = CleanBootstrap([x_model], x_uvdata)
xboot.run(100, nonparametric=True, use_v=False, outname=['boot_x', '.fits'])
uboot = CleanBootstrap([u_model], u_uvdata)
uboot.run(100, nonparametric=True, use_v=False, outname=['boot_u', '.fits'])

# Clean bootstrapped uv-data with common parameters
x_boot_uvfits = sorted(glob.glob('boot_x_*.fits'))
u_boot_uvfits = sorted(glob.glob('boot_u_*.fits'))
for i, x_boot_uv in enumerate(x_boot_uvfits):
    clean_difmap(x_boot_uv, 'x_cc_same_{}.fits'.format(str(i+1).zfill(3)), 'I',
                 (1024, 0.1),
                 path_to_script=path_to_script, show_difmap_output=True,
                 outpath=data_dir, beam_restore=ccimage_x.beam)
for i, u_boot_uv in enumerate(u_boot_uvfits):
    clean_difmap(u_boot_uv, 'u_cc_same_{}.fits'.format(str(i+1).zfill(3)), 'I',
                 (1024, 0.1),
                 path_to_script=path_to_script, show_difmap_output=True,
                 outpath=data_dir, beam_restore=ccimage_x.beam)

# For mask size 0 to 80 pix find shifts for all bootstrapped images
boot_sh_values = list()
for i in range(1, 101):
    files = glob.glob(os.path.join(data_dir, '*_cc_same_{}.fits'.format(str(i).zfill(3))))
    im1, im2 = files
    im1 = create_image_from_fits_file(im1)
    im2 = create_image_from_fits_file(im2)
    sh_values = list()
    for r in range(0, 80):
        sh = im1.cross_correlate(im2, region1=(im1.x_c, im1.y_c, r, None),
                                 region2=(im2.x_c, im2.y_c, r, None),
                                 upsample_factor=100)
        sh_value = math.sqrt(sh[0]**2 + sh[1]**2)
        sh_values.append(sh_value)
    boot_sh_values.append(sh_values)

for i, sh_values in enumerate(boot_sh_values):
    plt.plot(np.arange(0, 80)/17.12, sh_values, color="#4682b4", alpha=0.25)
plt.xlabel("Mask Radius, beam")
plt.ylabel("Shift Value, pxl")
