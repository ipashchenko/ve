from matplotlib import pyplot as plt
from skimage.transform import rotate
import numpy as np


original_shift = -1.
map_rotation = 40


# First, create image at some high frequency
from from_fits import create_clean_image_from_fits_file
# fits_file = '/home/ilya/code/vlbi_errors/data/account_spix/0055+300.u.2012_12_10.icn.fits'
fits_file = '/home/ilya/vlbi_errors/account_spix/0055+300.u.2012_12_10.icn.fits'
image = create_clean_image_from_fits_file(fits_file)
beam = image.beam_image
beam = rotate(beam, map_rotation)

# Only CCs convolved with beam
image_15 = image.cc_image * 10.

rot_image = rotate(image_15, map_rotation)
mask = rot_image < 0.01
rot_image[mask] = 0
i15_orig = rot_image

# Create image of spix
from simulations import alpha, alpha_linear
imsize = image.imsize
center = (imsize[0]/2, imsize[1]/2)
# spix = alpha(imsize, center, 0, k=0.1)
spix = alpha_linear(imsize, center, -0.075)
spix[mask] = 0
fig, axes = plt.subplots(1, 1)
axes.matshow(spix)
fig.show()


# Create image of low-frequency
factor = (8.4/15.)**spix
i8_orig = i15_orig * factor


# Create 2 realizations of noise and convolve with beam
n1 = np.random.normal(0, 0.0003, size=imsize[0]*imsize[1]).reshape(imsize)
n2 = np.random.normal(0, 0.0003, size=imsize[0]*imsize[1]).reshape(imsize)
n3 = np.random.normal(0, 0.0003, size=imsize[0]*imsize[1]).reshape(imsize)
n4 = np.random.normal(0, 0.0003, size=imsize[0]*imsize[1]).reshape(imsize)
from scipy.signal import fftconvolve
n1 = fftconvolve(n1, beam, mode='same')
n2 = fftconvolve(n2, beam, mode='same')
n3 = fftconvolve(n3, beam, mode='same')
n4 = fftconvolve(n4, beam, mode='same')
# fig, axes = plt.subplots(1, 1)
# axes.matshow(n1)
# fig.show()
# fig, axes = plt.subplots(1, 1)
# axes.matshow(n2)
# fig.show()

# Add noise to images
i8 = i8_orig + n1
i15 = i15_orig + n2
i15_test = i15_orig + n3
i8_test = i8_orig + n4
fig, axes = plt.subplots(1, 1)
axes.matshow(i8)
fig.show()
fig, axes = plt.subplots(1, 1)
axes.matshow(i15)
fig.show()

# Mask images before any shifts!
from utils import create_mask
mask_ = create_mask(i15.shape, region=(center[0], center[1], 15, None))
i15_ = i15.copy()
i15_[mask_] = 0
i15_test_ = i15_test.copy()
i15_test_[mask_] = 0
i8_ = i8.copy()
i8_[mask_] = 0
i8_test_ = i8_test.copy()
i8_test_[mask_] = 0


# Now shift one image
from skimage.transform import warp, AffineTransform
tform = AffineTransform(translation=(original_shift, 0))
i8_shifted_ = warp(i8_, tform)
i8_test_shifted = warp(i8_test, tform)
i15_test_shifted = warp(i15_test, tform)
i15_test_shifted_ = i15_test_shifted.copy()
i15_test_shifted_[mask_] = 0
i8_test_shifted_ = i8_test_shifted.copy()
i8_test_shifted_[mask_] = 0


# Cross correlate original and images
from skimage.feature import register_translation
shift, error, diffphase = register_translation(i15_, i8_, 100)
print("Shift with spix added but no real shift added: {}".format(shift))

found_shift, error, diffphase = register_translation(i15_, i8_shifted_, 1000)
print("Shift with spix added and real shift added: {}".format(found_shift))

shift, error, diffphase = register_translation(i15_, i15_test_, 100)
print("Shift with no spix and no real shift: {}".format(shift))
shift, error, diffphase = register_translation(i15_, i15_test_shifted_, 100)
print("Shift with no spix and real shift: {}".format(shift))

print("True shift : {}".format([0., original_shift]))


# Now use shift value found and de-shift low frequnecy on that value
tform = AffineTransform(translation=(-found_shift[1], -found_shift[0]))
i8_deshifted = warp(i8_shifted_, tform)

# Calculate spix image from deshifted low freq & original high freq
from image_ops import spix_map
spix_estimated, _, _ = spix_map([8.4 * 10**9, 15. * 10**9], [i8_deshifted, i15],
                                mask=mask)
fig, axes = plt.subplots(1, 1)
axes.matshow(spix_estimated)
fig.show()

# TODO: check other option
# Multiply low freq map to high freq
factor = (8.4/15.)**spix_estimated
i15_8 = i15 * factor
i15_8[mask] = i15[mask]

# # Shift altered low freq image back to original position
# tform = AffineTransform(translation=(found_shift[1], found_shift[0]))
# i8_shifted_15 = warp(i8_deshifted, tform)
i15_8_ = i15_8.copy()
i15_8_[mask_] = 0

# And check what shift became
found_shift_, error, diffphase = register_translation(i15_8_, i8_shifted_, 1000)
print("Shift with spix accounted for {}".format(found_shift_))
