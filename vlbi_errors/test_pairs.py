import os
import glob
from mojave import mojave_uv_fits_fname
from uv_data import UVData
from bootstrap import CleanBootstrap, analyze_bootstrap_samples
from spydiff import import_difmap_model, modelfit_difmap
from model import Model


# data_dir = '/home/ilya/silke'
# epoch = '2017_01_28'
# original_model_fname = '2017_01_28us'
# original_model_path = os.path.join(data_dir, original_model_fname)
# comps = import_difmap_model(original_model_fname, data_dir)
# model = Model(stokes='I')
# model.add_components(*comps)
# uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
# uv_fits_path = os.path.join(data_dir, uv_fits_fname)
# uvdata = UVData(uv_fits_path)
#
# boot = CleanBootstrap([model], uvdata)
# os.chdir(data_dir)
# boot.run(n=100, nonparametric=False, pairs=True)
#
# bootstrapped_uv_fits = sorted(glob.glob(os.path.join(data_dir,
#                                                      'bootstrapped_data*.fits')))
# for j, bootstrapped_fits in enumerate(bootstrapped_uv_fits):
#     modelfit_difmap(bootstrapped_fits, original_model_fname,
#                     'mdl_booted_{}.mdl'.format(j),
#                     path=data_dir, mdl_path=data_dir,
#                     out_path=data_dir, niter=100)
# booted_mdl_paths = glob.glob(os.path.join(data_dir, 'mdl_booted*'))
# analyze_bootstrap_samples(original_model_fname, booted_mdl_paths, data_dir,
#                           plot_comps=range(len(comps)),
#                           plot_file='pairs_corner.png', txt_file='pairs.txt')


from components import CGComponent

source = '1514-241'
epoch = '2006_04_28'
data_dir = '/home/ilya/Dropbox/papers/boot/bias/new'
# download_mojave_uv_fits(source, epochs=[epoch], bands=['u'],
#                         download_dir=data_dir)

uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
uv_fits_path = os.path.join(data_dir, uv_fits_fname)
original_model_path = os.path.join(data_dir, 'initial.mdl')
cg1 = CGComponent(2., 0., 0., 0.2)
cg2 = CGComponent(0.25, 0., 0.2, 0.2)
mdl = Model(stokes='I')
mdl.add_components(cg1, cg2)
uvdata = UVData(uv_fits_path)
noise = uvdata.noise()
for i in range(1, 2):
    uvdata = UVData(uv_fits_path)
    uvdata.substitute([mdl])
    uvdata.noise_add(noise)
    art_fits_fname = 'art_{}.fits'.format(i)
    art_fits_path = os.path.join(data_dir, art_fits_fname)
    uvdata.save(art_fits_path)
    modelfit_difmap(art_fits_fname, 'initial.mdl', 'out_{}.mdl'.format(i),
                    niter=100, path=data_dir, mdl_path=data_dir,
                    out_path=data_dir)

from bootstrap import bootstrap_uvfits_with_difmap_model
i=1
art_fits_fname = 'art_{}.fits'.format(i)
art_fits_path = os.path.join(data_dir, art_fits_fname)
bootstrap_uvfits_with_difmap_model(art_fits_path, os.path.join(data_dir, 'out_{}.mdl'.format(i)),
                                   boot_dir=os.path.join(data_dir, 'boot'),
                                   pairs=False)
