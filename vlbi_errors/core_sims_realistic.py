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


def sp_steeper(r):
    """
    Steepening of spectrum
    :param r:
        Radial distance from center [mas]
    :return:
    """
    if r < 20:
        return 1.6 + 0.1 * r / 20 + np.random.normal(0, 0.01)
    else:
        return 1.68 + np.random.normal(0., 0.01)


shift = (0.50, 0.50)
noise_factor = 2.
base_path = '/home/ilya/sandbox/coresims/'
path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
im_path_c1 = os.path.join(base_path, '1458+718.c1.2007_03_01.i.fits')
im_path_x1 = os.path.join(base_path, '1458+718.x1.2007_03_01.i.fits')
map_info_c1 = get_fits_image_info(im_path_c1)
map_info_x1 = get_fits_image_info(im_path_x1)
imsize_c1, pixref_c1, pixrefval_c1, (bmaj_c1, bmin_c1, bpa_c1,), pixsize_c1, _, _ = map_info_c1
imsize_x1, pixref_x1, pixrefval_x1, (bmaj_x1, bmin_x1, bpa_x1,), pixsize_x1, _, _ = map_info_x1
image_c1 = create_clean_image_from_fits_file(im_path_c1)
# # Plot real image
# plot(contours=image_c1.image, min_rel_level=0.125, x=image_c1.x[0],
#      y=image_c1.y[:, 0])
#
# mask = create_mask(imsize_c1, (pixref_c1[0], pixref_c1[1], 5, None))
# image_c1._image[mask] = 0.
# # Plot core-cutted image
# plot(contours=image_c1.image, min_rel_level=0.125, x=image_c1.x[0],
#      y=image_c1.y[:, 0])

# Add core shifts from map center
sc1 = (0.03, -0.03)
sx1 = (0.02, -0.02)

ccmodel = create_ccmodel_from_fits_file(im_path_c1)
ccmodel_x1 = Model(stokes='I')
for comp in ccmodel._components:
    r = np.sqrt(comp.p[1] ** 2. + comp.p[2] ** 2.)
    if r < 2. * (abs(pixsize_c1[0]) / mas_to_rad):
        print "removing component position ", comp.p
        continue
    if comp.p[0] > 4. * 10 ** (-4):
        print "removing component flux ", comp.p
        continue
    if comp.p[0] > 0:
        comp._p[0] = comp._p[0] * 30
    ccmodel_x1.add_component(comp)

# image_c1._image = np.zeros((512, 512))
ccmodel_x1.add_component(CGComponent(0.35, 0. + sx1[0], 0. + sx1[1], 0.10))
# ccmodel_c1_.add_to_image(image_c1)

# Really we did X-band model first
ccmodel_c1 = Model(stokes='I')
for comp in ccmodel_x1._components:
    r = np.sqrt(comp.p[1] ** 2. + comp.p[2] ** 2.)
    if r < 3. * (abs(pixsize_c1[0]) / mas_to_rad):
        print "removing component position ", comp.p
        continue
    comp._p[0] = comp._p[0] * sp_steeper(r)
    # Some shift
    comp._p[1] = comp._p[1] + 0.9
    comp._p[2] = comp._p[2] - 0.1
    ccmodel_c1.add_component(comp)
ccmodel_c1.add_component(CGComponent(0.35, 0.0 + sc1[0], 0.0 + sc1[1], 0.15))

# Overal shift = sqrt((0.9 + 0.1)**2 + (0.1 + 0.1)**2) = 1.02 mas
# Move model to UV-plane
c1_uvdata = create_uvdata_from_fits_file(os.path.join(base_path,
                                                      '1458+718.C1.2007_03_01.PINAL'))
noise = c1_uvdata.noise(average_freq=True)
for baseline, std in noise.items():
    noise[baseline] = noise_factor * std
c1_uvdata.substitute([ccmodel_c1])
c1_uvdata.noise_add(noise)
c1_uvdata.save(c1_uvdata.data, os.path.join(base_path, 'c1_uv_real.fits'))

x1_uvdata = create_uvdata_from_fits_file(os.path.join(base_path,
                                                      '1458+718.X1.2007_03_01.PINAL'))
noise = x1_uvdata.noise(average_freq=True)
for baseline, std in noise.items():
    noise[baseline] = 0.05 * std
x1_uvdata.substitute([ccmodel_x1])
# FIXME: !!!
noise[264] = 0.05 * noise[noise.keys()[0]]
x1_uvdata.noise_add(noise)
x1_uvdata.save(x1_uvdata.data, os.path.join(base_path, 'x1_uv_real.fits'))

# Clean parameters
mapsize_clean = (1024, abs(pixsize_x1[0]) / mas_to_rad)
beam_restore = (map_info_c1[3][0] / mas_to_rad,
                map_info_c1[3][1] / mas_to_rad,
                map_info_c1[3][2] / degree_to_rad)
# Clean
clean_difmap('c1_uv_real.fits', 'c1_cc_real.fits', 'i', mapsize_clean,
             path=base_path, path_to_script=path_to_script,
             beam_restore=beam_restore, show_difmap_output=True,
             outpath=base_path)
# Clean
clean_difmap('x1_uv_real.fits', 'x1_cc_real.fits', 'i', mapsize_clean,
             path=base_path, path_to_script=path_to_script,
             beam_restore=beam_restore, show_difmap_output=True,
             outpath=base_path)
