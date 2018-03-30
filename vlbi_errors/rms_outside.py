import os
from spydiff import clean_n
import matplotlib.pyplot as plt
from from_fits import create_image_from_fits_file, create_clean_image_from_fits_file
from variogram import variogram
from image_ops import rms_image


# TODO: Add labels to colorbar()
data_dir = '/home/ilya/Dropbox/papers/boot/new_pics/cv_cc/'
niter = 1000
path_to_script = '/home/ilya/github/vlbi_errors/difmap/clean_n'
uv_fits_fname = '0055+300.u.2006_02_12.uvf'
uv_fits_path = os.path.join(data_dir, uv_fits_fname)
out_fits_fname = 'cv_1000_cc_1000mas_shifter.fits'
out_fits_path = os.path.join(data_dir, out_fits_fname)
beam_fits_fname = 'cv_1000_cc.fits'
beam_fits_path = os.path.join(data_dir, beam_fits_fname)

clean_n(uv_fits_path, out_fits_fname, 'I',
        (512, 0.1), niter=niter, path_to_script=path_to_script,
        outpath=data_dir, show_difmap_output=True, shift=(1000, 0))

image = create_image_from_fits_file(out_fits_path)
plt.matshow(image.image)
patch = image.image[250:300, 250:300]
_, _, _, C, A = variogram(patch)
# Image already shifted so don't need rms_image_shifted here
rms = rms_image(image)

from image import plot as iplot
ccimage = create_clean_image_from_fits_file(beam_fits_path)
fig = iplot(image.image, image.image, x=image.x, y=image.y, show_beam=True,
            min_abs_level=rms, cmap='viridis', beam=ccimage.beam,
            beam_face_color='black', blc=(250, 250), trc=(300, 300))
# fig.savefig(os.path.join(data_dir, 'ngc315_niter1000_shifted1000_patch.pdf'),
#             bbox_inches='tight', format='pdf', dpi=1200)

label_size = 14
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 20
fig, axes = plt.subplots(1, 1)
import numpy as np
ms = axes.matshow(np.sqrt(C[:50, :50])/rms)
# ms = axes.matshow(C[:250, :250]/C[0, 0])
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="10%", pad=0.00)
cb = fig.colorbar(ms, cax=cax)
axes.set_xlabel(r'Pixels', fontsize=18)
axes.set_ylabel(r'Pixels', fontsize=18)
axes.xaxis.set_ticks_position('bottom')
cb.set_label("Correlation coefficient")
fig.savefig(os.path.join(data_dir, 'ngc315_niter1000_corrmatrix_rev.pdf'),
            bbox_inches='tight', format='pdf', dpi=1200)


