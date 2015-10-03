import os
import glob
import numpy as np
from from_fits import create_clean_image_from_fits_file,\
    create_uvdata_from_fits_file
from spydiff import clean_difmap
from data_io import get_fits_image_info
from utils import mas_to_rad, find_card_from_header
from sim_func import bootstrap_uv_fits

n_boot = 100
data_dir = '/home/ilya/sandbox/coreshift'
path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
h_uvdata_path = '/home/ilya/vlbi_errors/0952+179/2007_04_30/U1/uv/sc_uv.fits'
l_uvdata_path = '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/uv/sc_uv.fits'
high_freq_map = \
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/U1/im/I/cc.fits'
low_freq_map = \
    '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/im/I/cc.fits'
region = (512, 512, 50, None)

map_info_l = get_fits_image_info(low_freq_map)
map_info_h = get_fits_image_info(high_freq_map)
beam_restore = map_info_l[3]
beam_restore_ = (beam_restore[0] / mas_to_rad, beam_restore[1] / mas_to_rad,
                 beam_restore[2])

# Use 1024x1024 map size
mapsize_clean = (2 * map_info_h[0][0],
                 map_info_h[-3][0] / mas_to_rad)

for uvdata_path in [h_uvdata_path, l_uvdata_path]:
    uvdata_dir, uvdata_file = os.path.split(uvdata_path)
    print "Cleaning uv file {}".format(uvdata_path)
    uvdata = create_uvdata_from_fits_file(uvdata_path)
    freq_card = find_card_from_header(uvdata._io.hdu.header,
                                      value='FREQ')[0]
    # Frequency in Hz
    freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
    for stoke in 'i':
        print "Stokes {}".format(stoke)
        print "restoring with beam ", beam_restore_
        print "Cleaning with map parama ", mapsize_clean
        clean_difmap(uvdata_file, "shift_{}_{}_cc.fits".format(stoke, freq),
                     stoke, mapsize_clean, path=uvdata_dir,
                     path_to_script=path_to_script, beam_restore=beam_restore_,
                     outpath=data_dir)

images = glob.glob(os.path.join(data_dir, 'shift_i_*_cc.fits'))
images = sorted(images)
ccimage_l = create_clean_image_from_fits_file(images[0])
ccimage_h = create_clean_image_from_fits_file(images[1])
shift = ccimage_h.cross_correlate(ccimage_l, region1=region, region2=region)
print "Shift found: {}".format(shift)

# Bootstrap clean images
print "Bootstrapping uv-data and original clean models..."
for uvdata_path, cc_fits_path in zip([h_uvdata_path, l_uvdata_path],
                                     [high_freq_map, low_freq_map]):
    uvdata_dir, uvdata_file = os.path.split(uvdata_path)
    cc_dir, cc_file = os.path.split(cc_fits_path)
    print "Bootstrapping uv file {}".format(uvdata_path)
    print "With cc-model {}".format(cc_fits_path)
    uvdata = create_uvdata_from_fits_file(uvdata_path)
    freq_card = find_card_from_header(uvdata._io.hdu.header,
                                      value='FREQ')[0]
    # Frequency in Hz
    freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
    print "Bootstrapping frequency {}".format(freq)
    bootstrap_uv_fits(uvdata_file, [cc_file], n_boot, uvpath=uvdata_dir,
                      ccpath=[cc_dir], outpath=data_dir,
                      outname='boot_{}'.format(freq))

# Clean bootstrapped uv-data
for uvdata_path in glob.glob(os.path.join(data_dir, 'boot_*')):
    uvdata_dir, uvdata_file = os.path.split(uvdata_path)
    i_boot = uvdata_file.split('_')[-1].split('.')[0]
    uvdata = create_uvdata_from_fits_file(uvdata_path)
    freq_card = find_card_from_header(uvdata._io.hdu.header,
                                      value='FREQ')[0]
    freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
    clean_difmap(uvdata_path, "shift_{}_{}_cc.fits".format(freq, i_boot),
                 'i', mapsize_clean, path=uvdata_dir,
                 path_to_script=path_to_script,
                 beam_restore=beam_restore_, outpath=data_dir)

# Caclulate shifts for all pairs of bootstrapped clean maps
shifts = list()
for i_boot in range(1, n_boot + 1):
    cc_fnames_glob = 'shift_*_{}_cc.fits'.format(i_boot)
    cc_fnames = glob.glob(os.path.join(data_dir, cc_fnames_glob))
    print "Caculating shifts for bootstrap realization {}".format(i_boot)
    cc_fnames = sorted(cc_fnames)
    ccimage_l = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                               cc_fnames[0]))
    ccimage_h = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                               cc_fnames[1]))
    shift = ccimage_h.cross_correlate(ccimage_l, region1=region, region2=region)
    print "Found shift {}".format(shift)
    shifts.append(shift)

shifts = np.vstack(shifts)
