import os
import pandas as pd
import numpy as np
from mojave import download_mojave_uv_fits, mojave_uv_fits_fname
from spydiff import import_difmap_model, modelfit_difmap
from model import Model
from uv_data import UVData
from bootstrap import bootstrap_uvfits_with_difmap_model


data_dir = '/home/ilya/Dropbox/papers/boot/new_pics/corner'
source = '0016+731'
epoch = '2005_01_06'
epoch_ = '2005-01-06'
# source = '1226+023'
# epoch = '2007_09_06'
# epoch_ = '2007-09-06'
source_dir = os.path.join(data_dir, source)
if not os.path.exists(source_dir):
    os.mkdir(source_dir)
uv_fits_fname = mojave_uv_fits_fname(source, 'u', epoch)
uv_fits_path = os.path.join(source_dir, uv_fits_fname)
if not os.path.exists(uv_fits_path):
    download_mojave_uv_fits(source, [epoch], download_dir=source_dir)

names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
         'bpa']
df = pd.read_table(os.path.join(data_dir, 'asu.tsv'), sep=';', header=None,
                   names=names, dtype={key: str for key in names},
                   index_col=False)

# Create instance of Model and bootstrap uv-data
dfm_model_fname = 'dfmp_original_model.mdl'
dfm_model_path = os.path.join(source_dir, dfm_model_fname)
fn = open(os.path.join(source_dir, dfm_model_fname), 'w')
model_df = df.loc[np.logical_and(df['source'] == source,
                                 df['epoch'] == epoch_)]
for (flux, r, pa, bmaj, e, bpa) in np.asarray(model_df[['flux', 'r', 'pa',
                                                        'bmaj', 'e',
                                                        'bpa']]):
    print flux, r, pa, bmaj, e, bpa
    if not r.strip(' '):
        r = '0.0'
    if not pa.strip(' '):
        pa = '0.0'

    if not bmaj.strip(' '):
        bmaj = '0.0'
    if not e.strip(' '):
        e = "1.0"

    if np.isnan(float(bpa)):
        bpa = "0.0"
    else:
        bpa = bpa + 'v'

    if bmaj == '0.0':
        type_ = 0
        bpa = "0.0"
    else:
        bmaj = bmaj + 'v'
        type_ = 1
    fn.write("{}v {}v {}v {} {} {} {} {} {}".format(flux, r, pa, bmaj, e,
                                                    bpa, type_, "0", "0\n"))
fn.close()

refitted_mdl_fname = 'dfm_original_model_refitted.mdl'
refitted_mdl_path = os.path.join(source_dir, refitted_mdl_fname)
modelfit_difmap(uv_fits_fname, dfm_model_fname, refitted_mdl_fname,
                niter=200, path=source_dir, mdl_path=source_dir,
                out_path=source_dir)

# comps = import_difmap_model(dfm_model_fname, source_dir)
# model = Model(stokes='I')
# model.add_components(*comps)
# uvdata = UVData(uv_fits_path)
# fig = uvdata.uvplot()
# uvdata.substitute([model])
# uvdata.uvplot(fig=fig, color='r')

fig = bootstrap_uvfits_with_difmap_model(uv_fits_path, refitted_mdl_path,
                                         boot_dir=source_dir, n_boot=500,
                                         clean_after=False,
                                         out_plot_file='plot.eps',
                                         niter=50)
