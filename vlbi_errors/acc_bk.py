import os
import numpy as np
import pickle
from components import ImageComponent
from from_fits import create_clean_image_from_fits_file
from utils import mas_to_rad
from uv_data import UVData
from model import Model
from image import plot as iplot, find_bbox
from image_ops import rms_image



data_dir = '/home/ilya/Dropbox/ACC/BK/simulations/0952'
os.chdir('/home/ilya/code/jetsim/jetsim')
from transfer import Transfer
os.chdir('/home/ilya/code/vlbi_errors/vlbi_errors')
with open(os.path.join(data_dir, '15GHz_transfer_0d005.pkl'), 'rb') as fo:
    transfer = pickle.load(fo)

# 2D coordinates (mas)
x = transfer.pixsize[0]*transfer.image_coordinates[..., 0]
y = transfer.pixsize[1]*transfer.image_coordinates[..., 1]
x *= mas_to_rad
y *= mas_to_rad
image = transfer.image()

icomp = ImageComponent(image, x[0], y[..., 0])
uvdata = UVData(os.path.join(data_dir, '0952+179.U1.2007_04_30.PINAL'))
# uvdata = UVData('/home/ilya/Dropbox/ACC/3c120/uvdata/0430+052.u.2006_05_24.uvf')
# uvdata = UVData('/home/ilya/Dropbox/ACC/3c120/uvdata/0430+052.u.2006_05_24.uvf')
model = Model(stokes='I')
model.add_component(icomp)
noise = uvdata.noise(use_V=True)
uvdata.substitute([model])
for bl in noise:
    noise[bl] *= 10
uvdata.noise_add(noise)
uvdata.save(os.path.join(data_dir, '0952_15GHz_BK.fits'))

# clean_difmap('15GHz_BK.fits', 'u_BK_cc.fits', 'I', (1024, 0.1), path=data_dir,
#              path_to_script=path_to_script, show_difmap_output=True,
#              outpath=data_dir)

ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, '0952_15GHz_BK_cc.fits'))
beam = ccimage.beam
rms = rms_image(ccimage)
blc, trc = find_bbox(ccimage.image, rms, 10)
iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*rms, beam=beam,
      show_beam=True, blc=blc, trc=trc, core=tuple(p_map))

r = np.array([9.86391978e-01, 6.43996321e-01, 3.53391595e-01])
r_15 = np.array(r) - r[-1]
nu = np.array([5., 8., 15.])
w = np.array([1.38312167e+00, 9.95582942e-01, 5.60287022e-01])

rt = np.array([110, 68, 36])
rt = rt * 0.005
rt_15 = rt - rt[-1]

def wsize(nu, a):
    return a * nu ** (-1)

from scipy.optimize import curve_fit
res = curve_fit(wsize, nu, w, p0=0.1)
from matplotlib import pyplot as plt
plt.plot(nu, w, 'ro', markersize=10)
nunu = np.linspace(4.8, 15.2, 1000)
plt.plot(nunu, wsize(nunu, 7.3), label='k = -1')
plt.legend()
plt.xlabel("frequency, ghz")
plt.ylabel("Major axis, mas")
plt.savefig(os.path.join(data_dir, 'w_nu.png'), bbox_inches="tight", dpi=200)

def shift(nu, a):
    return a * (nu**(-1.) - 15.**(-1))

from scipy.optimize import curve_fit
res = curve_fit(shift, nu, r_15, p0=1.)

def r_(nu, b):
    return 4.8 * nu ** (-1) + b

def rt_(nu, b):
    return 2.8 * nu ** (-1) + b

from scipy.optimize import curve_fit
res = curve_fit(r_, nu, r, p0=0.1)

from matplotlib import pyplot as plt
plt.plot(nu, r, 'ro', markersize=10, label='observed')
plt.plot(nu, rt, 'go', markersize=10, label='true')
nunu = np.linspace(4.8, 15.2, 1000)
plt.plot(nunu, r_(nunu, 0.035), label='k = -1')
plt.plot(nunu, rt_(nunu, -0.009), label='k = -1')
plt.legend()
plt.xlabel("frequency, ghz")
plt.ylabel("distance from origin, mas")
plt.savefig(os.path.join(data_dir, 'r_nu.png'), bbox_inches="tight", dpi=200)
