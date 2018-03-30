import os
import scipy as sp
import numpy as np
from check_resolved import check_resolved
from uv_data import UVData


uv_fits_path = '/home/ilya/github/bck/jetshow/uvf/2200+420_K_SVLBI.uvf'
out_dir = '/home/ilya/github/vlbi_errors/vlbi_errors'
uvdata = UVData(uv_fits_path)
uvdata.save_uvrange('/home/ilya/github/bck/jetshow/uvf/test_uvmin.uvf', 8.0*10**8)
uv_fits_path = '/home/ilya/github/bck/jetshow/uvf/test_uvmin.uvf'


# Evidence
cg_priors = list()
out_fname = '/home/ilya/github/vlbi_errors/vlbi_errors/initial_eg.mdl'
cg_priors.append({'flux': (sp.stats.uniform.ppf, [0, 5], {}),
                  'x': (sp.stats.uniform.ppf,
                        [-0.2, 0.4], {}),
                  'y': (sp.stats.uniform.ppf,
                        [-0.2, 0.4], {}),
                  'bmaj': (sp.stats.uniform.ppf, [0, 0.5], {}),
                  'e': (sp.stats.uniform.ppf, [0, 1], {}),
                  'bpa': (sp.stats.uniform.ppf, [0, np.pi], {})})
mld_dict = {"el": out_fname}
components_priors = {"el": cg_priors}
result = check_resolved(uv_fits_path, mld_dict, components_priors,
                        outdir=out_dir)
evidence = result["el"]["logz"]
evidence_error = result["el"]["logzerr"]

# # Evidence
# out_fname = '/home/ilya/github/vlbi_errors/vlbi_errors/initial_cg.mdl'
# cg_priors = list()
# cg_priors.append({'flux': (sp.stats.uniform.ppf, [0, 5], {}),
#                   'x': (sp.stats.uniform.ppf,
#                         [-0.2, 0.4], {}),
#                   'y': (sp.stats.uniform.ppf,
#                         [-0.2, 0.4], {}),
#                   'bmaj': (sp.stats.uniform.ppf, [0, 0.5], {})})
# mld_dict = {"cg": out_fname}
# components_priors = {"cg": cg_priors}
# result = check_resolved(uv_fits_path, mld_dict, components_priors,
#                         outdir=out_dir)
# evidence = result["cg"]["logz"]
# evidence_error = result["cg"]["logzerr"]