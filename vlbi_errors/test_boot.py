import os
from uv_data import UVData
from model import Model
from spydiff import import_difmap_model
from bootstrap import CleanBootstrap


data_dir = '/home/ilya/code/vlbi_errors/tests/ft'
uv_fits = '1308+326.U1.2009_08_28.UV_CAL'
uvdata = UVData(os.path.join(data_dir, uv_fits))
model = Model(stokes='I')
comps = import_difmap_model('1308+326.U1.2009_08_28.mdl', data_dir)
model.add_components(*comps)
boot = CleanBootstrap([model], uvdata)
fig = boot.data.uvplot()
boot.model_data.uvplot(fig=fig, color='r')
# boot.find_outliers_in_residuals()
# boot.find_residuals_centers(split_scans=False)
# boot.fit_residuals_kde(split_scans=False, combine_scans=False,
#                        recenter=True)
