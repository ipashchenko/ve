import os
import numpy as np
from components import CGComponent, EGComponent, DeltaComponent
from from_fits import (create_uvdata_from_fits_file,
                       create_image_from_fits_file,
                       create_clean_image_from_fits_file,
                       create_ccmodel_from_fits_file)
from model import Model
from spydiff import clean_difmap
from data_io import get_fits_image_info
from utils import mas_to_rad, degree_to_rad, create_mask
from image import plot


shift = (0.50, 0.50)
noise_factor = 1.
base_path = '/home/ilya/sandbox/coresims/'
path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
im_path_c1 = os.path.join(base_path, '1458+718.c1.2007_03_01.i.fits')
im_path_x1 = os.path.join(base_path, '1458+718.x1.2007_03_01.i.fits')
map_info_c1 = get_fits_image_info(im_path_c1)
map_info_x1 = get_fits_image_info(im_path_x1)
imsize_c1, pixref_c1, pixrefval_c1, (bmaj_c1, bmin_c1, bpa_c1,), pixsize_c1, _, _ = map_info_c1
imsize_x1, pixref_x1, pixrefval_x1, (bmaj_x1, bmin_x1, bpa_x1,), pixsize_x1, _, _ = map_info_x1
image_c1 = create_clean_image_from_fits_file(im_path_c1)
# Plot real image
plot(contours=image_c1.image, min_rel_level=0.125, x=image_c1.x[0],
     y=image_c1.y[:, 0])

mask = create_mask(imsize_c1, (pixref_c1[0], pixref_c1[1], 5, None))
image_c1._image[mask] = 0.
# Plot core-cutted image
plot(contours=image_c1.image, min_rel_level=0.125, x=image_c1.x[0],
     y=image_c1.y[:, 0])

ccmodel_c1 = create_ccmodel_from_fits_file(im_path_c1)
for comp in ccmodel_c1._components:
    r = np.sqrt(comp.p[1] ** 2. + comp.p[2] ** 2.)
    print r
    if r < 5. * (abs(pixsize_c1[0]) / mas_to_rad):
        print "removing component ", comp.p
        ccmodel_c1.remove_component(comp)

x1_model = Model(stokes='I')
c1_model = Model(stokes='I')
# Add core
sc1 = (-0.03, -0.03)
sx1 = (-0.02, -0.02)
# CC to beam factor
# f = (2 * np.pi * (bmaj / pixsize[0]) ** 2 / 6.) ** (-1)
f = 1
c1_model.add_component(CGComponent(f * 0.35, 0. + sc1[0], 0. + sc1[1], 0.2))
x1_model.add_component(CGComponent(f * 0.35, 0. + sx1[0], 0. + sx1[1], 0.15))

def sp_steeper(r):
    if r < 20:
        return 1.3 + 0.4 * (r) / 20 + np.random.normal(0, 0.01)
    else:
        return 1.68 + np.random.normal(0., 0.01)

for comp in ccmodel_c1._components:
    c1_model.add_component(comp)
    r = np.sqrt(comp.p[1] ** 2. + comp.p[2] ** 2.)
    x1_model.add_component(DeltaComponent(comp.p[0] / sp_steeper(r),
                                          comp.p[1] + shift[0],
                                          comp.p[2] + shift[1]))


# Display model
image = create_clean_image_from_fits_file(im_path_c1)
image._image = np.zeros(image._image.shape, dtype=float)
image.add_model(c1_model)

# Move model to UV-plane
c1_uvdata = create_uvdata_from_fits_file(os.path.join(base_path,
                                                      '1458+718.C1.2007_03_01.PINAL'))
noise = c1_uvdata.noise(average_freq=True)
for baseline, std in noise.items():
    noise[baseline] = noise_factor * std
c1_uvdata.substitute([c1_model])
c1_uvdata.noise_add(noise)
c1_uvdata.save(c1_uvdata.data, os.path.join(base_path, 'c1_uv.fits'))

x1_uvdata = create_uvdata_from_fits_file(os.path.join(base_path,
                                                      '1458+718.X1.2007_03_01.PINAL'))
noise = x1_uvdata.noise(average_freq=True)
for baseline, std in noise.items():
    noise[baseline] = noise_factor * std
x1_uvdata.substitute([x1_model])
# FIXME: !!!
noise[264] = 0.04
x1_uvdata.noise_add(noise)
x1_uvdata.save(x1_uvdata.data, os.path.join(base_path, 'x1_uv.fits'))

# Clean parameters
mapsize_clean = (1024, abs(pixsize_x1[0]) / mas_to_rad)
beam_restore = (map_info_c1[3][0] / mas_to_rad,
                map_info_c1[3][1] / mas_to_rad,
                map_info_c1[3][2] / degree_to_rad)
# Clean
clean_difmap('c1_uv.fits', 'c1_cc.fits', 'i', mapsize_clean, path=base_path,
             path_to_script=path_to_script, beam_restore=beam_restore,
             show_difmap_output=True, outpath=base_path)
# Clean
clean_difmap('x1_uv.fits', 'x1_cc.fits', 'i', mapsize_clean, path=base_path,
             path_to_script=path_to_script, beam_restore=beam_restore,
             show_difmap_output=True, outpath=base_path)
