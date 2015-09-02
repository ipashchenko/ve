import numpy as np
from data_io import get_fits_image_info
from image import BasicImage, CleanImage
from utils import mask_region, mas_to_rad
from model import Model
from components import DeltaComponent


highest_freq_ccimage =\
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/im/I/cc.fits'
lowest_freq_ccimage =\
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/im/I/cc.fits'
# Calculate common image parameters
# Parameters of the original lowest frequency map. For constructing simulated
# data we need models with that size (in rad) and increased number of pixels
# (wrt highest frequency map)
(imsize_h, pixref_h, pixrefval_h, (bmaj_h, bmin_h, bpa_h,), pixsize_h, stokes_h,
 freq_h) = get_fits_image_info(highest_freq_ccimage)
(imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l, stokes_l,
 freq_l) = get_fits_image_info(lowest_freq_ccimage)

# Set parameters of simulated jet
# How many pixels of model image will lie in one highest frequency pixel
k = 3.
# Jet width to lowest frequency bmaj ratio
w = 0.75
# Jet length to lowest frequency bmaj ratio
l = 3.
# Parameters of beam in rad
bmaj = bmaj_l
bmin = bmaj
bpa = 0

# new pixsize
pixsize = (abs(pixsize_h[0]) / k, abs(pixsize_h[1]) / k)
# new imsize
x1 = imsize_l[0] * abs(pixsize_l[0]) / abs(pixsize[0])
x2 = imsize_l[1] * abs(pixsize_l[1]) / abs(pixsize[1])
imsize = (int(x1 - x1 % 2),
          int(x2 - x2 % 2))
# new pixref
pixref = (imsize[0]/2, imsize[1]/2)
# Beam width in new pixels
beam_width = bmaj / abs(pixsize[0])

# Jet's parameters in new pixels
jet_width = w * bmaj_l / abs(pixsize[0])
jet_length = l * bmaj_l / abs(pixsize[0])

# Construct image with new parameters
image = BasicImage(imsize=imsize, pixsize=pixsize, pixref=pixref)

# Construct region with emission
# TODO: Construct cone region
jet_region = mask_region(image._image, region=(pixref[0] - int(beam_width // 2),
                                               pixref[1],
                                               pixref[0] + int(beam_width // 2),
                                               pixref[1] + jet_length))
jet_region = np.ma.array(image._image, mask=~jet_region.mask)

# Create model instance and fill it with components
model = Model(stokes='Q')


# Flux should decline with x (or y) by linear law
def flux(x, y, max_flux, length):
    return max_flux - (max_flux/length) * (x - pixref[0])


# Zero Rm in center and constant gradient ``2 * max_rm/width``.
def rm(x, y, max_rm, width):
    k = max_rm / (width / 2.)
    return k * (y - pixref[1])


# Could i use full polarization flux from original CC model, but split it to
# Q & U components to create uniform angles?
comps = [DeltaComponent(flux(x, y, 0.1, jet_length), image.x[x, y]/mas_to_rad,
                        image.y[x, y]/mas_to_rad) for (x, y), value in
         np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
model.add_components(*comps)
# Actually - i don't need CleanImage - just model to FT
image = CleanImage(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='Q',
                   freq=freq_l, bmaj=bmaj, bmin=bmin, bpa=0)
image.add_model(model)

