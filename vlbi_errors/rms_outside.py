from spydiff import clean_difmap
import matplotlib.pyplot as plt


# TODO: Add labels to colorbar()
uv_fits_file = '/home/ilya/code/vlbi_errors/data/account_spix/0055+300.x.2006_02_12.uvf'
path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
clean_difmap(uv_fits_file, 'ngc315_cc.fits', 'I', (1024, 0.1),
             path_to_script=path_to_script, shift=(0, 1000),
             show_difmap_output=True)

from from_fits import create_image_from_fits_file
image = create_image_from_fits_file('ngc315_cc.fits')
plt.matshow(image.image)
patch = image.image[400:440, 400:440]
from variogram import variogram
from image_ops import rms_image
_, _, _, C, A = variogram(patch)
plt.matshow(C[:40, :40]/rms_image(image)); plt.colobar()
