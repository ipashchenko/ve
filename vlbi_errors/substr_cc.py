import os
import glob
import numpy as np
from from_fits import create_model_from_fits_file
from spydiff import clean_difmap, modelfit_difmap, import_difmap_model
from uv_data import UVData
from model import Model


data_dir = '/home/ilya/Dropbox/0235/tmp/test_mdl_cc/'
uv_fits = '0235+164.q1.2008_06_12.uvp'
model_fname = '0235+164.q1.2008_06_12.mdl'
path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
uvdata = UVData(os.path.join(data_dir, uv_fits))


# First CLEAN data
clean_difmap(uv_fits, 'cc.fits', 'I', (1024, 0.03), path=data_dir,
             path_to_script=path_to_script, show_difmap_output=True,
             outpath=data_dir)

model = create_model_from_fits_file(os.path.join(data_dir, 'cc.fits'))
comps = model._components
comps = sorted(comps, key=lambda x: np.sqrt(x._p[1]**2 + x._p[2]**2),
               reverse=True)

for i, comp in enumerate(comps[350:]):
    print "Substracting {} components".format((i+350))
    uvdata_ = UVData(os.path.join(data_dir, uv_fits))
    mdl1 = Model(stokes='I')
    mdl1.add_component(comp)
    uvdata_.substitute([mdl1])
    uvdata_diff = uvdata - uvdata_
    uvdata_diff.save(os.path.join(data_dir, 'without_{}_ccs.uvp'.format(str(i+350).zfill(3))))

fits_files = sorted(glob.glob(os.path.join(data_dir, 'without*')))
for fits_file in fits_files:
    fits_fname = os.path.split(fits_file)[-1]
    i = fits_fname.split('_')[1]
    modelfit_difmap(fits_file, model_fname, 'without_{}_css.mdl'.format(i),
                    path=data_dir, mdl_path=data_dir, out_path=data_dir)

bmajs = list()
es = list()
mdl_files = sorted(glob.glob(os.path.join(data_dir, 'without*.mdl')))
for mdl_file in mdl_files:
    mdl_fname = os.path.split(mdl_file)[-1]
    i = mdl_fname.split('_')[1]
    comps = import_difmap_model(mdl_fname, data_dir)
    core_comp = comps[0]
    bmaj = core_comp._p[3]
    e = core_comp._p[4]
    bmajs.append(bmaj)
    es.append(e)

import matplotlib.pyplot as plt
plt.figure()
plt.plot((np.arange(351, 351 + len(bmajs))), bmajs)
plt.xlabel("Number of substracted components")
plt.ylabel("Bmaj, mas")
plt.savefig(os.path.join(data_dir, "bmaj_vs_Nsubst_zoom.png"), bbox_inches='tight',
            dpi=200)
plt.figure()
plt.plot((np.arange(351, 351 + len(bmajs))), es)
plt.xlabel("Number of substracted components")
plt.ylabel("e")
plt.savefig(os.path.join(data_dir, "e_vs_Nsubst_zoom.png"), bbox_inches='tight',
            dpi=200)





