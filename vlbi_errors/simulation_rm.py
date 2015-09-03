import os
import numpy as np
from data_io import get_fits_image_info
from image import BasicImage, CleanImage, plot
from images import Images
from utils import mask_region, mas_to_rad
from model import Model
from components import DeltaComponent
from from_fits import create_uvdata_from_fits_file
from spydiff import clean_difmap


print "Constructing model image parameters..."
highest_freq_ccimage =\
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/im/I/cc.fits'
lowest_freq_ccimage =\
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/im/I/cc.fits'
lowest_freq_uvdata = \
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/uv/sc_uv.fits'
highest_freq_uvdata = \
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/uv/sc_uv.fits'
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
print "Creating ROTM image with gradient..."
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
max_flux = 0.05 / (np.pi * beam_width ** 2)
comps_q = [DeltaComponent(flux(x, y, max_flux, jet_length),
                          image.x[x, y]/mas_to_rad,
                          image.y[x, y]/mas_to_rad) for (x, y), value in
           np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
comps_u = [DeltaComponent(flux(x, y, max_flux, jet_length),
                          image.x[x, y]/mas_to_rad,
                          image.y[x, y]/mas_to_rad) for (x, y), value in
           np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
print "Adding components to Q&U models..."
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
pang_image = images.create_pang_images(convolved=False)[0]
print "Creating PPOL image..."
ppol_image = images.create_pol_images(convolved=False)[0]
mask = ppol_image.image < 0.01
# # PANG = 0.5 * pi/4 = 0.39
# plot(contours=ppol_image.image, colors=ppol_image.image,
#      vectors=pang_image.image, vectors_values=ppol_image.image,
#      x=image_u.x[0, :], y=image_u.y[:, 0], min_rel_level=1.,
#      vectors_mask=mask, contours_mask=mask, colors_mask=mask, vinc=20)

data_dir = '/home/ilya/vlbi_errors/0952+179/2007_04_30/'
i_dir_c1 = data_dir + 'C1/im/I/'
i_dir_c2 = data_dir + 'C2/im/I/'
i_dir_x1 = data_dir + 'X1/im/I/'
i_dir_x2 = data_dir + 'X2/im/I/'
q_dir_c1 = data_dir + 'C1/im/Q/'
u_dir_c1 = data_dir + 'C1/im/U/'
q_dir_c2 = data_dir + 'C2/im/Q/'
u_dir_c2 = data_dir + 'C2/im/U/'
q_dir_x1 = data_dir + 'X1/im/Q/'
u_dir_x1 = data_dir + 'X1/im/U/'
q_dir_x2 = data_dir + 'X2/im/Q/'
u_dir_x2 = data_dir + 'X2/im/U/'
band_dir = {'c1': {'i': i_dir_c1, 'q': q_dir_c1, 'u': u_dir_c1},
            'c2': {'i': i_dir_c2, 'q': q_dir_c2, 'u': u_dir_c2},
            'x1': {'i': i_dir_x1, 'q': q_dir_x1, 'u': u_dir_x1},
            'x2': {'i': i_dir_x2, 'q': q_dir_x2, 'u': u_dir_x2}}

uv_files_dirs = {'x2': os.path.join(data_dir, 'X2/uv/'),
                 'x1': os.path.join(data_dir, 'X1/uv/'),
                 'c2': os.path.join(data_dir, 'C2/uv/'),
                 'c1': os.path.join(data_dir, 'C1/uv/')}
lambda_sq_bands = {'x2': 0.00126661, 'x1': 0.00136888, 'c2': 0.00359502,
                   'c1': 0.00423771}
freq_bands = {'x2': 8429458750.0, 'x1': 8108458750.0, 'c2': 5003458750.0,
              'c1': 4608458750.0}
bands = ('c1', 'c2', 'x1', 'x2')
# Now circle for bands with lower frequencies
for band in ('x2', 'x1', 'c2', 'c1'):
    # Rotate PANG by multiplying polarized intensity on cos/sin
    print "Creating arrays of Q&U for {}-band".format(band)
    q_array = ppol_image._image * np.cos(2. * (1. + 2. * image_rm._image *
                                               lambda_sq_bands[band]))
    u_array = ppol_image._image * np.sin(2. * (1. + 2. * image_rm._image *
                                               lambda_sq_bands[band]))
    image_q = CleanImage(imsize=imsize, pixsize=pixsize, pixref=pixref,
                         stokes='Q', freq=freq_bands[band], bmaj=bmaj,
                         bmin=bmin, bpa=0)
    image_q._image = q_array
    image_u = CleanImage(imsize=imsize, pixsize=pixsize, pixref=pixref,
                         stokes='U', freq=freq_bands[band], bmaj=bmaj,
                         bmin=bmin, bpa=0)
    image_u._image = u_array

    model_q = Model(stokes='Q')
    model_u = Model(stokes='U')
    print "Creating components of Q&U for {}-band".format(band)
    comps_q = [DeltaComponent(image_q._image[x, y],
                              image.x[x, y]/mas_to_rad,
                              image.y[x, y]/mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    comps_u = [DeltaComponent(image_u._image[x, y],
                              image.x[x, y]/mas_to_rad,
                              image.y[x, y]/mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    print "Adding components of Q&U for {}-band models".format(band)
    model_q.add_components(*comps_q)
    model_u.add_components(*comps_u)

    # Substitute Q&U models to uv-data and add noise
    print "Substituting Q&U models in uv-data, adding noise, saving for" \
          " {}-band".format(band)
    uv_file = os.path.join(uv_files_dirs[band], 'sc_uv.fits')
    uvdata = create_uvdata_from_fits_file(uv_file)
    noise = uvdata.noise(average_freq=True)
    uvdata.substitute([model_q, model_u])
    uvdata.noise_add(noise)
    # Save uv-data to file
    uvdata.save(uvdata.data, os.path.join(uv_files_dirs[band], 'simul_uv.fits'))

# Now clean simulated uv-data with parameters of lowest frequency map
path_to_script = '/home/ilya/code/vlbi_errors/data/zhenya/clean/final_clean_nw'
map_info = get_fits_image_info(band_dir['c1']['i'] + 'cc.fits')
beam_restore = map_info[3]
map_info = get_fits_image_info(band_dir['x2']['i'] + 'cc.fits')
mapsize_clean = (map_info[0][0], map_info[-3][0] / mas_to_rad)
for band in bands:
    for stoke in ('i', 'q', 'u'):
        clean_difmap(fname='simul_uv.fits', outfname='cc_sim.fits',
                     stokes=stoke, mapsize_clean=mapsize_clean,
                     path=uv_files_dirs[band], path_to_script=path_to_script,
                     mapsize_restore=None, beam_restore=beam_restore,
                     outpath=band_dir[band][stoke])

images = Images()
cc_fnames = list()
for band in bands:
    for stoke in ('i', 'q', 'u'):
        cc_fname = os.path.join(band_dir[band][stoke], 'cc_sim.fits')
        cc_fnames.append(cc_fname)
images.add_from_fits(cc_fnames)
