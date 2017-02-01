import os
from uv_data import UVData
from model import Model
from spydiff import import_difmap_model


data_dir = '/home/ilya/code/vlbi_errors/silke'
uv_fits = '0851+202.u.2004_11_05.uvf'
mdl_fname = '1.mod.2004_11_05'
uv_data = UVData(os.path.join(data_dir, uv_fits))
comps = import_difmap_model(mdl_fname, data_dir)
model = Model(stokes='I')
model.add_components(*comps)

fig = uv_data.uvplot(style='a&p')
uv_data.substitute([model])
uv_data.uvplot(color='r', fig=fig, phase_range=[-0.2, 0.2])
