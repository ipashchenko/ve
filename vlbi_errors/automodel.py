import os
import scipy as sp
from collections import OrderedDict
from uv_data import UVData
from model import Model
from cv_model import cv_difmap_models
from spydiff import export_difmap_model, modelfit_difmap, import_difmap_model
from components import CGComponent
from check_resolved import check_resolved


priors_max_flux = None
priors_xy_max = 5.0
priors_bmaj_max = 3.0
out_dir = ''
uv_fits_path = ''
uvdata = UVData(uv_fits_path)
uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
performance_dict = OrderedDict()


# 1. Modelfit in difmap with CG
cg = CGComponent(1.0, 0.0, 0.0, 0.5)
out_fname = os.path.join(out_dir, 'cg1_init.mdl')
export_difmap_model([cg], out_fname)
modelfit_difmap(uv_fits_fname, 'cg1_init.mdl', 'cg1_fitted.mdl',
                path=uv_fits_dir, mdl_path=out_dir, out_path=out_dir)
model_cg1 = Model()
comps = import_difmap_model('cg1_fitted_mdl', out_dir)
model_cg1.add_components(*comps)

# "Informational" criteria
bic = model_cg1.bic(uvdata)
aic = model_cg1.aic(uvdata)

# Cross-Validation
cv_scores = cv_difmap_models([out_fname], uv_fits_path, K=10, out_dir=out_dir)

# Evidence
cg_priors = list()
cg_priors.append({'flux': (sp.stats.uniform.ppf, [0, priors_max_flux], {}),
                  'x': (sp.stats.uniform.ppf,
                        [-0.5*priors_xy_max, priors_xy_max], {}),
                  'y': (sp.stats.uniform.ppf,
                        [-0.5*priors_xy_max, priors_xy_max], {}),
                  'bmaj': (sp.stats.uniform.ppf, [0, priors_bmaj_max], {})})
mld_dict = {"cg1": out_fname}
components_priors = {"cg1": cg_priors}
result = check_resolved(uv_fits_path, mld_dict, components_priors,
                        outdir=out_dir)
evidence = result["cg1"]["logz"]
evidence_error = result["cg1"]["logzerr"]

performance_dict["cg1"] = {"aic": aic, "bic": bic, "logz": evidence,
                           "logzerr": evidence_error}

# Now substruct component from UV-data
uvdata_ = UVData(uv_fits_path)
uvdata_.substitute([model_cg1])
uvdata_residual = uvdata - uvdata_

# Image residual UV-data
