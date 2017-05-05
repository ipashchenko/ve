import numpy as np
from utils import create_grid
from utils import gaussian


width = 10
j_length = 200
cj_length = 10
max_flux = 0.25
imsize = (256, 256)
center = (128, 128)
gauss_peak=0
dist_from_core=24
gauss_bmaj=10
gauss_e=1.
gauss_bpa=0.
gauss_peak_jet=0.0
dist_from_core_jet=24
gauss_bmaj_jet=10
gauss_e_jet=1.
gauss_bpa_jet=0.
cut=0.000001
transverse='gauss'


x, y = create_grid(imsize)
x -= center[0]
y -= center[1]

max_flux = float(max_flux)
along = np.where(x > 0,
                 0,
                 0)
if transverse == 'linear':
    perp = -(2 * max_flux / width) * abs(y)
elif transverse == 'quadratic':
    perp = -(max_flux / (width / 2) ** 2.) * y ** 2.
elif transverse == 'sqrt':
    perp = -(max_flux / np.sqrt(width / 2.)) * np.sqrt(abs(y))
elif transverse == 'gauss':
    gauss_width = 0.3 * x
    perp = np.where(x > 0.1, max_flux * np.exp(-y**2/(2. * gauss_width**2)) / x, 0)

else:
    raise Exception("transverse must be `linear`, `quadratic` or `sqrt`")
image = along + perp
image[image < 0] = 0

# # Jet feature
# if gauss_peak:
#     gaussian_ = gaussian(gauss_peak, dist_from_core, 0, gauss_bmaj, gauss_e,
#                          gauss_bpa)
#     image += gaussian_(x, y)
#     image[image < cut] = 0.
#
# if gauss_peak_jet:
#     gaussian_ = gaussian(gauss_peak_jet, dist_from_core_jet, 0,
#                          gauss_bmaj_jet, gauss_e_jet, gauss_bpa_jet)
#     image += gaussian_(x, y)
#     image[image < cut] = 0.

import matplotlib.pyplot as plt
# plt.matshow(np.log(image))
# plt.show()

from model import Model
from components import ImageComponent, CGComponent
from from_fits import create_image_from_fits_file
import os
from uv_data import UVData
original_uvdata = UVData(os.path.join('/home/ilya/Dropbox/0235/VLBA/',
                                      '0235+164.q1.2008_09_02.uvf_difmap'))
noise = original_uvdata.noise()
noise = {key: 0.25 * value for key, value in noise.items()}
image_stack = create_image_from_fits_file(os.path.join('/home/ilya/Dropbox/0235/VLBA/',
                                                       '0235+164.q1.stack.fits'))
pixsize = abs(image_stack.pixsize[0])
imsize = image_stack.imsize
model = Model(stokes='I')
model.from_2darray(image, (0.01, 0.01))
# model.add_component(ImageComponent(image, x[0,:],
#                                    y[:,0]))
# model.add_component(CGComponent(2.25, 0., 0., 0.01))
import copy
model_uvdata = copy.deepcopy(original_uvdata)
model_uvdata.substitute([model])
model_uvdata.noise_add(noise)
#fig = original_uvdata.uvplot()
model_uvdata.uvplot()
model_uvdata.save('opening.fits', rewrite=True)
# import astropy.io.fits as pf
# original_hdu = pf.open(os.path.join('/home/ilya/Dropbox/0235/VLBA/',
#                                       '0235+164.q1.2008_09_02.uvf_difmap'))[0]
# hdu = pf.open('opening.fits')[0]
# print original_hdu.data[0]
# print hdu.data[0]
from spydiff import clean_difmap
clean_difmap('opening.fits', 'opening_cc.fits', 'I', (512, 0.03),
             path_to_script='/home/ilya/code/vlbi_errors/difmap/final_clean_nw',
             show_difmap_output=True)
image = create_image_from_fits_file('opening_cc.fits')
plt.matshow(image.image)
plt.show()

