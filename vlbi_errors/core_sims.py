import os
import numpy as np
from components import CGComponent, EGComponent, DeltaComponent
from from_fits import (create_uvdata_from_fits_file,
                       create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from model import Model
from spydiff import clean_difmap
from data_io import get_fits_image_info
from utils import mas_to_rad, degree_to_rad


shift = (0.10, 0.10)
noise_factor = 10.
path = '/home/ilya/Dropbox/Ilya'
k1_uvfile = '1226+023.K1.2009_08_28.UV_CAL'
q1_uvfile = '1226+023.Q1.2009_08_28.UV_CAL'
# 23 GHz
k1_image = '1226+023.K1.2009_08_28_pair_1226+023.Q1.2009_08_28_vs_1226+023.K1.2009_08_28.fits'
# 43 GHZ
q1_image = '1226+023.Q1.2009_08_28_pair_1226+023.Q1.2009_08_28_vs_1226+023.K1.2009_08_28.fits'
map_info = get_fits_image_info(os.path.join(path, k1_image))
imsize, pixref, pixrefval, (bmaj, bmin, bpa,), pixsize, _, _ = map_info

path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'

k1_model = Model(stokes='I')
q1_model = Model(stokes='I')
# Add core
sk1 = (-0.03, -0.03)
sq1 = (-0.02, -0.02)
# CC to beam factor
# f = (2 * np.pi * (bmaj / pixsize[0]) ** 2 / 6.) ** (-1)
f = 1
spf = 0.5
k1_model.add_component(CGComponent(f * 0.40, 0. + sk1[0], 0. + sk1[1], 0.02))
q1_model.add_component(CGComponent(f * 0.40, 0. + sq1[0], 0. + sq1[1], 0.015))

k1_model.add_component(CGComponent(f * 0.15, 0.1 + sk1[0], 0.12 + sk1[1], 0.05))
q1_model.add_component(CGComponent(f * 0.19, 0.1 + shift[0] + sq1[0],
                                   0.12 + shift[1] + sq1[1], 0.05))
k1_model.add_component(CGComponent(f * 0.12, 0.15 + sk1[0], 0.14 + sk1[1], 0.05))
q1_model.add_component(CGComponent(f * 0.16, 0.15 + shift[0] + sq1[0],
                                   0.14 + shift[1] + sq1[1], 0.05))
# 1
k1_model.add_component(CGComponent(f * 0.05, 0.2 + sk1[0], 0.2 + sk1[1], 0.05))
q1_model.add_component(CGComponent(f * 0.09, 0.2 + shift[0] + sq1[0],
                                   0.2 + shift[1] + sq1[1], 0.05))
# 2
k1_model.add_component(CGComponent(f * 0.09, 0.45 + sk1[0], 0.5 + sk1[1], 0.1))
q1_model.add_component(CGComponent(f * 0.05, 0.45 + shift[0] + sq1[0],
                                   0.5 + shift[1] + sq1[1], 0.09))
# 9
k1_model.add_component(CGComponent(f * 0.09, 0.6 + sk1[0], 0.7 + sk1[1], 0.2))
q1_model.add_component(CGComponent(f * 0.05, 0.6 + shift[0] + sq1[0],
                                   0.7 + shift[1] + sq1[1], 0.3))
# 3
k1_model.add_component(CGComponent(f * 0.05, 1.1 + sk1[0], 1.3 + sk1[1], 0.2))
q1_model.add_component(CGComponent(f * 0.03, 1.1 + shift[0] + sq1[0],
                                   1.3 + shift[1] + sq1[1], 0.2))
# 8
k1_model.add_component(CGComponent(f * 0.10, 1.6 + sk1[0], 1.4 + sk1[1], 0.2))
q1_model.add_component(CGComponent(f * 0.06, 1.6 + shift[0] + sq1[0],
                                   1.4 + shift[1] + sq1[1], 0.3))
# 4
k1_model.add_component(CGComponent(f * 0.09, 2.3 + sk1[0], 2.2 + sk1[1], 0.4))
q1_model.add_component(CGComponent(f * 0.05, 2.3 + shift[0] + sq1[0],
                                   2.2 + shift[1] + sq1[1], 0.3))
# 5
k1_model.add_component(CGComponent(f * 0.03, 3.3 + sk1[0], 3.4 + sk1[1], 0.5))
q1_model.add_component(CGComponent(f * 0.015, 3.3 + shift[0] + sq1[0],
                                   3.4 + shift[1] + sq1[1], 0.5))
# 6
k1_model.add_component(CGComponent(f * 0.03, 3.5 + sk1[0], 3.6 + sk1[1], 0.6))
q1_model.add_component(CGComponent(f * 0.0165, 3.5 + shift[0] + sq1[0],
                                   3.6 + shift[1] + sq1[1], 0.45))
# 7
k1_model.add_component(CGComponent(f * 0.035, 4.3 + sk1[0], 4.4 + sk1[1], 0.9))
q1_model.add_component(CGComponent(f * 0.020, 4.3 + shift[0] + sq1[0],
                                   4.4 + shift[1] + sq1[1], 0.7))
# 10
k1_model.add_component(CGComponent(f * 0.04, 4.0 + sk1[0], 4.3 + sk1[1], 0.4))
q1_model.add_component(CGComponent(f * 0.02, 4.0 + shift[0] + sq1[0],
                                   4.3 + shift[1] + sq1[1], 0.3))
fs = np.random.uniform(0.003, 0.05, 2000)
xs = np.random.uniform(0., 4., 2000)
ys = np.random.uniform(0., 4., 2000)
ws = np.random.uniform(0.1, 0.3, 2000)
mask = np.logical_and((ys / xs) < 1.2, (ys / xs) > 0.8)
# xs = np.ma.array(xs, mask=mask)
# ys = np.ma.array(ys, mask=mask)
# fs = np.ma.array(fs, mask=mask)

for i, (fl, x, y, w) in enumerate(zip(fs, xs, ys, ws)):
    if mask[i]:
        k1_model.add_component(CGComponent(fl, x + sk1[0], y + sk1[1], w))
        q1_model.add_component(CGComponent(fl * 0.5, x + shift[0] + sq1[0],
                                           y + shift[1] + sq1[1], w))
        print i, fl, x, y, w

print "Adding to core region..."

fs = np.random.uniform(0.05, 0.1, 400)
xs = np.random.uniform(0., 1.7, 400)
ys = np.random.uniform(0., 1.7, 400)
ws = np.random.uniform(.01, 0.1, 400)
mask = np.logical_and((ys / xs) < 1.1, (ys / xs) > 0.9)
# xs = np.ma.array(xs, mask=mask)
# ys = np.ma.array(ys, mask=mask)
# fs = np.ma.array(fs, mask=mask)

for i, (fl, x, y, w) in enumerate(zip(fs, xs, ys, ws)):
    if mask[i]:
        k1_model.add_component(CGComponent(fl * 0.8, x + sk1[0], y + sk1[1], w))
        q1_model.add_component(CGComponent(fl, x + shift[0] + sq1[0],
                                           y + shift[1] + sq1[1], w))
        print i, fl, x, y, w

# Display model
# image = create_clean_image_from_fits_file(os.path.join(path, k1_image))
# image._image = np.zeros(image._image.shape, dtype=float)
# image.add_model(k1_model)

# Move model to UV-plane
k1_uvdata = create_uvdata_from_fits_file(os.path.join(path, k1_uvfile))
noise = k1_uvdata.noise(average_freq=True)
for baseline, std in noise.items():
    noise[baseline] = noise_factor * std
k1_uvdata.substitute([k1_model])
k1_uvdata.noise_add(noise)
k1_uvdata.save(k1_uvdata.data, os.path.join(path, 'k1_uv.fits'))

q1_uvdata = create_uvdata_from_fits_file(os.path.join(path, q1_uvfile))
noise = q1_uvdata.noise(average_freq=True)
for baseline, std in noise.items():
    noise[baseline] = noise_factor * std
q1_uvdata.substitute([q1_model])
q1_uvdata.noise_add(noise)
q1_uvdata.save(q1_uvdata.data, os.path.join(path, 'q1_uv.fits'))

# Clean parameters
mapsize_clean = (imsize[0], abs(pixsize[0]) / mas_to_rad)
beam_restore = (map_info[3][0] / mas_to_rad,
                map_info[3][1] / mas_to_rad,
                map_info[3][2] / degree_to_rad)
# Clean
clean_difmap('k1_uv.fits', 'k1_cc.fits', 'i', mapsize_clean, path=path,
             path_to_script=path_to_script, beam_restore=beam_restore,
             show_difmap_output=True, outpath=path)
# Clean
clean_difmap('q1_uv.fits', 'q1_cc.fits', 'i', mapsize_clean, path=path,
             path_to_script=path_to_script, beam_restore=beam_restore,
             show_difmap_output=True, outpath=path)
