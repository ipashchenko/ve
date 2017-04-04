# DOESNT WORK - ERRORS STILL SMALL
import os
from mojave import mojave_uv_fits_fname
from uv_data import UVData
from bootstrap import bootstrap_uvfits_with_difmap_model


data_dir = '/home/ilya/silke'
epoch = '2017_01_28'
original_model_fname = '2017_01_28us'
original_model_path = os.path.join(data_dir, original_model_fname)
uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
uv_fits_path = os.path.join(data_dir, uv_fits_fname)
uvdata = UVData(uv_fits_path)
uvdata.noise_add({baseline: [0.137] for baseline in uvdata.baselines})
new_fits_path = os.path.join(data_dir, 'added_noise.fits')
uvdata.save(new_fits_path)
bootstrap_uvfits_with_difmap_model(new_fits_path, original_model_path,
                                   boot_dir=os.path.join(data_dir, 'boot'))
