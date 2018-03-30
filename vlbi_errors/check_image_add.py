import copy
import matplotlib.pyplot as plt
import os
import numpy as np
from spydiff import import_difmap_model, modelfit_difmap, clean_difmap
from uv_data import UVData
from model import Model
from from_fits import create_clean_image_from_fits_file
from mojave import (download_mojave_uv_fits, get_mojave_mdl_file,
                    mojave_uv_fits_fname)


data_dir = '/home/ilya/vlbi_errors/model_cov/check_image_add'
tsv_table = os.path.join(data_dir, 'asu.tsv')
source = '2230+114'
epoch = '2005-02-05'
uv_fits = mojave_uv_fits_fname(source, 'u', epoch.replace('-', '_'))
path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
mdl_fname = '{}_{}.mdl'.format(source, epoch)

# Fetch uv-fits
download_mojave_uv_fits(source, [epoch.replace('-', '_')], bands=['u'],
                        download_dir=data_dir)
# Fetch model file
get_mojave_mdl_file(tsv_table, source, epoch, outdir=data_dir)
# Clean uv-fits
clean_difmap(uv_fits, 'cc.fits', 'I', [1024, 0.1], path=data_dir,
             path_to_script=path_to_script, outpath=data_dir)

# Create clean image instance
cc_image = create_clean_image_from_fits_file(os.path.join(data_dir, 'cc.fits'))
comps = import_difmap_model(mdl_fname, data_dir)
model = Model(stokes='I')
model.add_components(*comps)

# Check that model fits UV-data well
uv_data = UVData(os.path.join(data_dir, uv_fits))
uv_data.uvplot()
mdl_data = copy.deepcopy(uv_data)
mdl_data.substitute([model])
mdl_data.uvplot(sym='.r')

cc_image_ = copy.deepcopy(cc_image)
cc_image_._image = np.zeros(cc_image._image.shape, dtype=float)
cc_image_.add_model(model)
plt.figure()
plt.matshow(cc_image_.cc_image-cc_image.cc_image)
plt.colorbar()
