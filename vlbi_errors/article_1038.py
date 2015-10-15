import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from data_io import get_fits_image_info
from utils import (mas_to_rad, degree_to_rad)
from spydiff import clean_difmap
from convenience_funcs import (find_core_shift, boot_uv_fits_with_cc_fits)
from images import Images
from from_fits import (create_uvdata_from_fits_file,
                       create_ccmodel_from_fits_file,
                       create_clean_image_from_fits_file,
                       create_image_from_fits_file)


data_dir = '/home/ilya/Dropbox/4vlbi_errors/DENISE'
path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
source = '1038+064'
uv_ext = 'uvf'
uv_paths = glob.glob(os.path.join(data_dir, "{}*.{}".format(source, uv_ext)))
cc_paths = glob.glob(os.path.join(data_dir, "{}*.{}".format(source,
                                                            '*cn.fits')))

# Find parameters for maps
info_h = get_fits_image_info(os.path.join(data_dir,
                                          '1038+064.l22.2010_05_21.icn.fits'))
mapsize_clean = (info_h[0][0], abs(info_h[-3][0] / mas_to_rad))

# # Check that we got rigth model
# uvdata = create_uvdata_from_fits_file(os.path.join(data_dir,
#                                                    '1038+064.l22.2010_05_21.uvf'))
# ccmodel = create_ccmodel_from_fits_file(os.path.join(data_dir,
#                                                      '1038+064.l22.2010_05_21.icn.fits'))
# uvdata.uvplot()
# uvdata.substitute([ccmodel])
# uvdata.uvplot(sym='.r')

# Clean longest wavelength uv-data for current source and find common beam
clean_difmap('1038+064.l18.2010_05_21.uvf', '1038+064.l18.2010_05_21.icn.fits',
             'i', mapsize_clean, data_dir, path_to_script, outpath=data_dir,
             show_difmap_output=True)
info_l = get_fits_image_info(os.path.join(data_dir,
                                          '1038+064.l18.2010_05_21.icn.fits'))

beam_restore = (info_l[3][0] / mas_to_rad, info_l[3][1] / mas_to_rad,
                info_l[3][2] / degree_to_rad)
print "Longest wavelength beam : {}".format(beam_restore)

# Cleaning all uv-files with naitive resolution
for uv_path in uv_paths:
    uv_dir, uv_file = os.path.split(uv_path)
    l = uv_file.split('.')[1]
    epoch = uv_file.split('.')[2]
    for stoke in ('i', 'q', 'u'):
        outfname = "{}.{}.{}.{}cn.fits".format(source, l, epoch, stoke)
        clean_difmap(uv_file, outfname, stoke, mapsize_clean, path=data_dir,
                     path_to_script=path_to_script, outpath=data_dir,
                     show_difmap_output=False)

# Cleaning all uv-files with the same parameters for all stokes.
for uv_path in uv_paths:
    uv_dir, uv_file = os.path.split(uv_path)
    l = uv_file.split('.')[1]
    for stoke in ('i', 'q', 'u'):
        outfname = "{}.{}.common.{}cn.fits".format(source, l, stoke)
        clean_difmap(uv_file, outfname, stoke, mapsize_clean, path=data_dir,
                     path_to_script=path_to_script, beam_restore=beam_restore,
                     outpath=data_dir, show_difmap_output=False)

# Find shifts between longest and shortest wavelengths
uv_paths_hl = [uv for uv in uv_paths if 'l18' in uv or 'l22' in uv]
cc_paths_hl = [cc for cc in cc_paths if 'l18' in cc or 'l22' in cc]
shift, shifts, r = find_core_shift(uv_paths_hl, cc_paths_hl,
                                   path_to_script=path_to_script,
                                   data_dir=data_dir, do_bootstrap=True,
                                   upsample_factor=1000, n_boot=100)
print "Using circular mask with r = {}".format(r)
# Histogram shifts
plt.hist(np.sqrt(shifts[:, 0] ** 2. + shifts[:, 1] ** 2), bins=15)

# Create ROTM image for current source
images = Images()
images.add_from_fits(wildcard=os.path.join(data_dir,
                                           "{}.*.common.*cn.fits".format(source)))
# Find PPA uncertainty Hovatta's way
# Uncertainty in EVPA calibration [deg]
sigma_evpa = 2.
# D-terms spread [dimensionless] are equal for all frequencies
d_spread = 0.002
n_ant = 8
n_if = 2
n_scans = 10
# For all frequencies calculate D-term error image
sigma_d_images_dict = dict()
for l in ('l18', 'l20', 'l21', 'l22'):
    fname = "{}.{}.common.icn.fits".format(source, l)
    image = create_image_from_fits_file(os.path.join(data_dir, fname))
    i_peak = np.max(image.image.ravel())
    sigma_d_image = d_spread * np.sqrt(image.image ** 2. +
                                       (0.3 * i_peak) ** 2) / np.sqrt(n_ant *
                                                                      n_if *
                                                                      n_scans)
    sigma_d_images_dict.update({l: sigma_d_image})

# For all frequencies find rms for Q & U images
rms_dict = dict()
for l in ('l18', 'l20', 'l21', 'l22'):
    rms_dict[l] = dict()
    for stoke in ('q', 'u'):
        fname = "{}.{}.common.{}cn.fits".format(source, l, stoke)
        image = create_image_from_fits_file(os.path.join(data_dir, fname))
        rms = image.rms(region=(40, 40, 40, None))
        rms_dict[l][stoke] = rms

# For all frequencies find overall Q & U error
overall_errors_dict = dict()
for l in ('l18', 'l20', 'l21', 'l22'):
    overall_errors_dict[l] = dict()
    for stoke in ('q', 'u'):
        overall_sigma = np.sqrt(rms_dict[l][stoke] ** 2. +
                                sigma_d_images_dict[l] ** 2. +
                                (1.5 * rms_dict[l][stoke]) ** 2.)
        overall_errors_dict[l][stoke] = overall_sigma

# For all frequencies find EVPA & PPOL errors
evpa_error_images_dict = dict()
ppol_error_images_dict = dict()
for l in ('l18', 'l20', 'l21', 'l22'):
    qfname = "{}.{}.common.qcn.fits".format(source, l)
    ufname = "{}.{}.common.ucn.fits".format(source, l)
    qimage = create_image_from_fits_file(os.path.join(data_dir, qfname))
    uimage = create_image_from_fits_file(os.path.join(data_dir, ufname))
    evpa_error_image = np.sqrt((qimage.image *
                                overall_errors_dict[l]['u']) ** 2. +
                               (uimage.image *
                                overall_errors_dict[l]['q']) ** 2.) /\
                       (2. * (qimage.image ** 2. + uimage.image ** 2.))
    ppol_error_image = 0.5 * (overall_errors_dict[l]['u'] +
                              overall_errors_dict[l]['q'])
    evpa_error_images_dict[l] = evpa_error_image
    ppol_error_images_dict[l] = ppol_error_image

# For all frequencies add EVPA calibration uncertainty
for l in ('l18', 'l20', 'l21', 'l22'):
    evpa_error_images_dict[l] = np.sqrt(evpa_error_images_dict[l] ** 2. +
                                        np.deg2rad(sigma_evpa) ** 2.)

s_pang_arrays = list()
# Create list of EVPA error images sorted by frequency
for l in sorted(evpa_error_images_dict.keys(), reverse=True):
    s_pang_arrays.append(evpa_error_images_dict[l])

# For all frequencies create masks based on PPOL flux
ppol_images = dict()
ppol_masks = dict()
freqs_dict = {'l18': 1665458750.0, 'l20': 1493458750.0, 'l21': 1430458750.0,
              'l22': 1358458750.0}
for l, freq in freqs_dict.items():
    ppol_image = images.create_pol_images(freq=freq)[0]
    ppol_images.update({l: ppol_image})
    ppol_mask = ppol_image.image < 3. * ppol_error_images_dict[l]
    ppol_masks.update({l: ppol_mask})

# Create overall mask for PPOL flux
masks = [np.array(mask, dtype=int) for mask in ppol_masks.values()]
ppol_mask = np.zeros(masks[0].shape, dtype=int)
for mask in masks:
    ppol_mask += mask
ppol_mask[ppol_mask != 0] = 1

# FIXME: Alter algorithm for resolve n-pi only when fit has bad chi squared
rotm_image, s_rotm_image = images.create_rotm_image(s_pang_arrays=s_pang_arrays,
                                                    mask=ppol_mask)
# Plot intensity contours + ROTM map
fname = "{}.{}.common.{}cn.fits".format(source, 'l18', 'i')
image = create_image_from_fits_file(os.path.join(data_dir, fname))
from image import plot
# FIXME: pay attention to slice coordinates - they are unintuitive!
plot(contours=image.image, colors=rotm_image.image, min_rel_level=0.5,
     slice_points=((110, 115), (130, 108)))
     # x=image.x[0], y=image.y[:, 0])

# Plot ROTM slice
slice_length = len(rotm_image.slice((115, 110), (108, 130)))
plt.errorbar(np.arange(slice_length), rotm_image.slice((115, 110), (108, 130)),
             s_rotm_image.slice((115, 110), (108, 130)), fmt='.k')

##################################
# Now make bootstrap realizations#
##################################
# Create bootstrapped uv-data
for uv_path in uv_paths:
    uv_dir, uv_file = os.path.split(uv_path)
    l = uv_file.split('.')[1]
    epoch = uv_file.split('.')[2]
    ccfname = "{}.{}.{}.*cn.fits".format(source, l, epoch)
    ccfnames = glob.glob(os.path.join(data_dir, ccfname))
    ccfnames = [os.path.split(ccfname)[-1] for ccfname in ccfnames]
    boot_uv_fits_with_cc_fits(uv_file, ccfnames, n=50, uvpath=data_dir,
                              ccpath=len(ccfnames) * [data_dir],
                              outname="{}.{}.{}_boot".format(source, l, epoch),
                              outpath=data_dir)

# Clean bootstrapped uv-data with common parameters
for uv_path in glob.glob(os.path.join(data_dir,
                                      "{}.*_boot_*.fits".format(source))):
    uv_dir, uv_file = os.path.split(uv_path)
    l = uv_file.split('.')[1]
    num = uv_file.split('.')[2].split('_')[-1]
    for stoke in ('i', 'q', 'u'):
        outfname = "{}.{}.boot_{}.{}cn.fits".format(source, l, num, stoke)
        clean_difmap(uv_file, outfname, stoke, mapsize_clean, path=data_dir,
                     path_to_script=path_to_script, beam_restore=beam_restore,
                     outpath=data_dir, show_difmap_output=False)

# EVPA-calibration error should be added to images of PANG to all bootstrapped
# realization before computation ROTM
# TODO: add param ``sigma_EVPA`` to ``Images.create_rotm_image`` method
# D-terms error should be added to Q&U by creating random complex map
# D * (I**2 + (0.3*I_peak)**2)**(0.50/ (#ant*#if*#scan)**(0.5) where D - random
# complex number 9circular normal complex variable) with std fixed.