import numpy as np
from data_io import get_fits_image_info
from image import BasicImage, CleanImage, plot
from images import Images
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


# Flux should decline with x (or y) by linear law
def flux(x, y, max_flux, length):
    return max_flux - (max_flux/length) * (x - pixref[0])


# Zero Rm in center and constant gradient ``2 * max_rm/width``.
def rm(x, y, max_rm, width):
    k = max_rm / (width / 2.)
    return k * abs(x)


# Create map of ROTM
max_rm = 200.
image_rm = BasicImage(imsize=imsize, pixsize=pixsize, pixref=pixref)
image_rm._image = rm(image.x/abs(pixsize[0]), image.y/abs(pixsize[1]), max_rm,
                     jet_width)


# Create model instance and fill it with components
model_q = Model(stokes='Q')
model_u = Model(stokes='U')

# Could i use full polarization flux from original CC model, but split it to
# Q & U components to create uniform angles? I can create image with new pixsize
# & imsize of Q, U (PANG) at highest frequency. Use CCs in pixels as model.
# Then Q = P * cos(2*(chi_0 + RM * lambda^2)), U = P * sin(...). But straight
# jets are easy to analyze.
max_flux = 0.1 / (np.pi * beam_width ** 2)
comps_q = [DeltaComponent(flux(x, y, max_flux, jet_length),
                          image.x[x, y]/mas_to_rad,
                          image.y[x, y]/mas_to_rad) for (x, y), value in
           np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
comps_u = [DeltaComponent(flux(x, y, max_flux, jet_length),
                          image.x[x, y]/mas_to_rad,
                          image.y[x, y]/mas_to_rad) for (x, y), value in
           np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
model_q.add_components(*comps_q)
model_u.add_components(*comps_u)
# Actually - i don't need CleanImage - just model to FT
image_q = CleanImage(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='Q',
                     freq=freq_l, bmaj=bmaj, bmin=bmin, bpa=0)
image_u = CleanImage(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='U',
                     freq=freq_l, bmaj=bmaj, bmin=bmin, bpa=0)
image_q.add_model(model_q)
image_u.add_model(model_u)

# Create PANG map & plot it
images = Images()
images.add_images([image_q, image_u])
pang_image = images.create_pang_images()[0]
ppol_image = images.create_pol_images()[0]
mask = ppol_image.image < 0.01
# PANG = 0.5 * pi/4 = 0.39
plot(contours=ppol_image.image, colors=ppol_image.image,
     vectors=pang_image.image, vectors_values=ppol_image.image,
     x=image_u.x[0, :], y=image_u.y[:, 0], min_rel_level=0.01,
     vectors_mask=mask, contours_mask=mask, colors_mask=mask, vinc=20)

