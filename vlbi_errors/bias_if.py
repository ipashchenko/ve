# FIXME: Cutting IFs results in different uvw
import os
import numpy as np
from mojave import mojave_uv_fits_fname
from spydiff import import_difmap_model, modelfit_difmap
from model import Model
from uv_data import UVData
import astropy.io.fits as pf

data_dir = '/home/ilya/silke/if'
boot_dir = '/home/ilya/silke/if/boot'
epoch = '2017_01_28'

uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
uv_fits_path = os.path.join(data_dir, uv_fits_fname)

# # Create sequence of FITS files with different number of bands - from 1 to 7
# for i in range(1, 8):
#     uvdata = UVData(uv_fits_path)
#     imdata = uvdata.hdu.data['DATA'][:, :, :, :i, :, :, :]
#     pardata = [uvdata.hdu.data[key] for key in uvdata.hdu.data.parnames]
#     pardata[5] = np.zeros(len(pardata[4]))
#     pardata[0] *= uvdata.frequency
#     pardata[1] *= uvdata.frequency
#     pardata[2] *= uvdata.frequency
#     pscales = [uvdata.hdu.header['PSCAL{}'.format(i)] for i in
#                range(1, len(uvdata.hdu.parnames)+1)]
#     pzeros = [uvdata.hdu.header['PZERO{}'.format(i)] for i in
#               range(1, len(uvdata.hdu.parnames)+1)]
#     x = pf.GroupData(imdata, parnames=uvdata.hdu.data.parnames, pardata=pardata,
#                      bitpix=-32, parbzeros=pzeros)
#     header = uvdata.hdu.header.copy()
#     header['NAXIS5'] = i
#     header['EXTEND'] = True
#     hdu = pf.GroupsHDU(data=x, header=header)
#     hdulist = uvdata.hdulist
#     hdulist[0] = hdu
#     hdulist.writeto(os.path.join(data_dir, 'oj287_{}_bands.fits'.format(i)),
#                     output_verify='warn', clobber=True)
#
#
# # That is test data set with last IF only
# imdata = uvdata.hdu.data['DATA'][:, :, :, 7, :, :, :]
# pardata = [uvdata.hdu.data[key] for key in uvdata.hdu.data.parnames]
# x = pf.GroupData(imdata, parnames=uvdata.hdu.data.parnames, pardata=pardata,
#                  bitpix=-32)
# header = uvdata.hdu.header.copy()
# header['NAXIS5'] = 1
# hdu = pf.GroupsHDU(data=x, header=header)
# hdu.writeto(os.path.join(data_dir, 'oj287_test.fits'), clobber=True)
# uvdata_test = UVData(os.path.join(data_dir, 'oj287_test.fits'))


original_model_fname = '2017_01_28us'
original_model_path = os.path.join(data_dir, original_model_fname)
comps = import_difmap_model(original_model_fname, data_dir)
model = Model(stokes='I')
model.add_components(*comps)


cv_scores = list()
train_scores = list()
for i, fname in enumerate(['1IF.fits', '12IF.fits', '123IF.fits', '1234IF.fits',
                          '12345IF.fits', '123456IF.fits', '1234567IF.fits']):
    current_fits = os.path.join(data_dir, fname)
    modelfit_difmap(current_fits,
                    original_model_fname, 'out_{}.mdl'.format(i),
                    path=data_dir, mdl_path=data_dir,
                    out_path=data_dir, niter=100)
    comps = import_difmap_model('out_{}.mdl'.format(i), data_dir)
    model = Model(stokes='I')
    model.add_components(*comps)

    # Calculate performance on training data
    uvdata_train_model = UVData(current_fits)
    uvdata_train = UVData(current_fits)
    uvdata_train_model.substitute([model])
    uvdata_diff_train = uvdata_train - uvdata_train_model
    factor = np.count_nonzero(~uvdata_diff_train.uvdata_weight_masked.mask[:, :, :2])
    squared_diff = uvdata_diff_train.uvdata_weight_masked[:, :, :2] *\
                   uvdata_diff_train.uvdata_weight_masked[:, :, :2].conj()
    score = float(np.sum(squared_diff)) / factor
    train_scores.append(score)


    # Calculate performance on test data
    uvdata_test_model = UVData(os.path.join(data_dir, '8IF.fits'))
    uvdata_test = UVData(os.path.join(data_dir, '8IF.fits'))
    uvdata_test_model.substitute([model])
    uvdata_diff_test = uvdata_test - uvdata_test_model
    factor = np.count_nonzero(~uvdata_diff_test.uvdata_weight_masked.mask[:, :, :2])
    squared_diff = uvdata_diff_test.uvdata_weight_masked[:, :, :2] * \
                   uvdata_diff_test.uvdata_weight_masked[:, :, :2].conj()
    score = float(np.sum(squared_diff)) / factor
    cv_scores.append(score)

cv_scores = np.array(cv_scores)
train_scores = np.array(train_scores)

import matplotlib.pyplot as plt
plt.plot(range(1, 8), np.sqrt(cv_scores))
plt.plot(range(1, 8), np.sqrt(train_scores), 'r')
plt.show()
