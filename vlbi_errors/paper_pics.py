import os
import matplotlib.pyplot as plt
import matplotlib
import glob
from from_fits import (create_model_from_fits_file,
                       create_clean_image_from_fits_file)
from uv_data import UVData
from spydiff import clean_difmap
from bootstrap import CleanBootstrap
from images import Images
from image_ops import rms_image, rms_image_shifted
from image import plot as iplot
from image import find_bbox


# Plotting bootstrap error of I Stokes
path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
# data_dir = '/home/ilya/Dropbox/papers/boot/new_pics/uvdata'
data_dir = '/home/ilya/vlbi_errors/examples/coverage/1749+701/errors'
# uv_fits = '1633+382.l22.2010_05_21.uvf'
# im_fits = '1633+382.l22.2010_05_21.icn.fits'
# uv_fits = '1226+023.u.2006_03_09.uvf'
uv_fits = '1749+701.x.2006_04_05.uvf'
# im_fits = '3c273_orig_cc.fits'
im_fits = '1749+701.x.2006_04_05.cc.fits'
clean_difmap(uv_fits, im_fits, 'I', (512, 0.1), path=data_dir,
             path_to_script=path_to_script, outpath=data_dir)
model = create_model_from_fits_file(os.path.join(data_dir, im_fits))
uvdata = UVData(os.path.join(data_dir, uv_fits))
boot = CleanBootstrap([model], uvdata)
os.chdir(data_dir)
boot.run(100, nonparametric=False, use_v=False)
booted_uv_fits = glob.glob(os.path.join(data_dir, 'bootstrapped_data*.fits'))
for i, boot_uv in enumerate(sorted(booted_uv_fits)):
    clean_difmap(boot_uv, 'booted_cc_{}.fits'.format(i), 'I', (512, 0.1),
                 path=data_dir, path_to_script=path_to_script, outpath=data_dir)
booted_im_fits = glob.glob(os.path.join(data_dir, 'booted_cc*.fits'))

images = Images()
images.add_from_fits(booted_im_fits)
error_image = images.create_error_image()

orig_image = create_clean_image_from_fits_file(os.path.join(data_dir, im_fits))
rms = rms_image_shifted(os.path.join(data_dir, uv_fits),
                        image_fits=os.path.join(data_dir, im_fits),
                        path_to_script=path_to_script)
blc, trc = find_bbox(orig_image.image, level=1.5*rms)
image_mask = orig_image.image < 2 * rms
label_size = 14
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
# 3c273
# fig = iplot(orig_image.image, error_image.image/rms, x=orig_image.x,
#             y=orig_image.y, show_beam=True, min_abs_level=2*rms, cmap='hsv',
#             beam=orig_image.beam, colors_mask=image_mask,
#             beam_face_color='black', blc=(495, 385), trc=(680, 540))
# 1633
fig = iplot(orig_image.image, error_image.image/(2. * rms), x=orig_image.x,
            y=orig_image.y, show_beam=True, min_abs_level=2*rms, cmap='hsv',
            beam=orig_image.beam, colors_mask=image_mask,
            beam_face_color='black', blc=blc, trc=trc)
fig.savefig('/home/ilya/Dropbox/papers/boot/new_pics/1749_x_2006_04_05_boot.eps',
            bbox_inches='tight', format='eps', dpi=1200)
fig.savefig('/home/ilya/Dropbox/papers/boot/new_pics/1749_x_2006_04_05_boot.svg',
            bbox_inches='tight', format='svg', dpi=1200)
fig.savefig('/home/ilya/Dropbox/papers/boot/new_pics/1749_x_2006_04_05_boot.pdf',
            bbox_inches='tight', format='pdf', dpi=1200)
