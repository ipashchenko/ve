import os
import copy
from astropy.io import fits as pf
from uv_data import UVData
from model import Model
from components import EGComponent, CGComponent
from spydiff import modelfit_difmap, import_difmap_model


data_dir = '/home/ilya/code/vlbi_errors/tests/ft'
uv_fits = '1308+326.U1.2009_08_28.UV_CAL'
hdus_orig = pf.open(os.path.join(data_dir, uv_fits))
print "orig", hdus_orig[0].data[0][0]
uvdata = UVData(os.path.join(data_dir, uv_fits))
noise = uvdata.noise(use_V=False)
# noise = {bl: 0.1*noise_ for bl, noise_ in noise.items()}
eg1 = EGComponent(5., 0, 0, 0.15, 0.33, 0.2)
eg2 = EGComponent(2.5, 1, 1, 0.5, 0.5, 0.)
model = Model(stokes='I')
model.add_components(eg1, eg2)
# model.add_components(eg1, eg2)
uvdata_c = copy.deepcopy(uvdata)
uvdata_c.substitute([model])
uvdata_c.noise_add(noise)
uvdata_c.save(os.path.join(data_dir, 'fake.fits'), rewrite=True)
modelfit_difmap('fake.fits', 'mod_c2_2ee.mdl', 'out_2c.mdl', path=data_dir,
                mdl_path=data_dir, out_path=data_dir, niter=100)
comps = import_difmap_model('out_2c.mdl', data_dir)
print [comp.p for comp in comps]
model_fitted = Model(stokes='I')
model_fitted.add_components(*comps)
uvdata_mf = copy.deepcopy(uvdata)
uvdata_mf.substitute([model_fitted])


fig = uvdata_c.uvplot(color='g', phase_range=[-1, 1])
uvdata_mf.uvplot(fig=fig, color='r', phase_range=[-1, 1])

hdus_orig = pf.open(os.path.join(data_dir, uv_fits))
hdus_fake = pf.open(os.path.join(data_dir, 'fake.fits'))
print "orig", hdus_orig[0].data[0][0]
print "fake", hdus_fake[0].data[0][0]

