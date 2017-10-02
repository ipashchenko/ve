import numpy as np
import os
import shutil
from uv_data import UVData
from model import Model
from cv_model import cv_difmap_models
from spydiff import (export_difmap_model, modelfit_difmap, import_difmap_model,
                     clean_difmap, append_component_to_difmap_model)
from components import CGComponent
from from_fits import create_image_from_fits_file
from utils import mas_to_rad, infer_gaussian


n_max_comps = 5
mapsize_clean = (1024, 0.1)
out_dir = '/home/ilya/github/vlbi_errors/vlbi_errors'
uv_fits_fname = '1807+698.u.2007_07_03.uvf'
uv_fits_path = os.path.join(out_dir, uv_fits_fname)
path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
uvdata = UVData(uv_fits_path)
freq_hz = uvdata.frequency


def suggest_cg_component(uv_fits_path, mapsize_clean, path_to_script,
                         outname='image_cc.fits', out_dir=None):
    """
    Suggest single circular gaussian component using self-calibrated uv-data
    FITS file.
    :param uv_fits_path:
        Path to uv-data FITS-file.
    :param mapsize_clean:
        Iterable of image size (# pixels) and pixel size (mas).
    :param path_to_script:
        Path to difmap CLEANing script.
    :param outname: (optional)
        Name of file to save CC FITS-file. (default: ``image_cc.fits``)
    :param out_dir: (optional)
        Optional directory to save CC FITS-file. If ``None`` use CWD. (default:
        ``None``)
    :return:
        Instance of ``CGComponent``.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    clean_difmap(uv_fits_fname, outname, 'I', mapsize_clean,
                 path=uv_fits_dir, path_to_script=path_to_script,
                 outpath=out_dir)

    image = create_image_from_fits_file(os.path.join(out_dir, outname))
    imsize = image.imsize[0]
    mas_in_pix = abs(image.pixsize[0] / mas_to_rad)
    amp, x, y, bmaj = infer_gaussian(image.image)
    x = mas_in_pix * (x - imsize / 2) * np.sign(image.dx)
    y = mas_in_pix * (y - imsize / 2) * np.sign(image.dy)
    bmaj *= mas_in_pix
    return CGComponent(amp, x, y, bmaj)


def create_residuals(uv_fits_path, model=None, out_fname='residuals.uvf',
                     out_dir=None):
    if model is None:
        return uv_fits_path
    if out_dir is None:
        out_dir = os.getcwd()
    out_fits_path = os.path.join(out_dir, out_fname)
    uvdata = UVData(uv_fits_path)
    uvdata_ = UVData(uv_fits_path)
    uvdata_.substitute([model])
    uvdata_residual = uvdata - uvdata_
    uvdata_residual.save(out_fits_path, rewrite=True)
    return out_fits_path


cv_scores = list()
for i in range(1, n_max_comps+1):
    model = None
    uv_fits_path_res = create_residuals(uv_fits_path, model=model,
                                        out_dir=out_dir)
    # 1. Modelfit in difmap with CG
    cg = suggest_cg_component(uv_fits_path_res, mapsize_clean, path_to_script,
                              out_dir=out_dir)
    try:
        shutil.copy(os.path.join(out_dir, 'cg_fitted_{}.mdl'.format(i-1)),
                    os.path.join(out_dir, 'cg_init_{}.mdl'.format(i)))
    except IOError:
        pass
    append_component_to_difmap_model(cg, os.path.join(out_dir, 'cg_init_{}.mdl'.format(i)),
                                     freq_hz)
    out_fname = os.path.join(out_dir, 'cg_init_{}.mdl'.format(i))
    export_difmap_model([cg], out_fname, freq_hz)
    modelfit_difmap(uv_fits_fname, 'cg_init_{}.mdl'.format(i),
                    'cg_fitted_{}.mdl'.format(i), path=out_dir,
                    mdl_path=out_dir, out_path=out_dir)
    model = Model(stokes='I')
    comps = import_difmap_model('cg_fitted_{}.mdl'.format(i), out_dir)
    model.add_components(*comps)

    # Cross-Validation
    cv_score = cv_difmap_models([os.path.join(out_dir,
                                               'cg_fitted_{}.mdl'.format(i))],
                                 uv_fits_path, K=10, out_dir=out_dir, n_rep=1)
    cv_scores.append((cv_score[0][0][0], cv_score[1][0][0]))

# # Evidence
# cg_priors = list()
# cg_priors.append({'flux': (sp.stats.uniform.ppf, [0, priors_max_flux], {}),
#                   'x': (sp.stats.uniform.ppf,
#                         [-0.5*priors_xy_max, priors_xy_max], {}),
#                   'y': (sp.stats.uniform.ppf,
#                         [-0.5*priors_xy_max, priors_xy_max], {}),
#                   'bmaj': (sp.stats.uniform.ppf, [0, priors_bmaj_max], {})})
# mld_dict = {"cg1": out_fname}
# components_priors = {"cg1": cg_priors}
# result = check_resolved(uv_fits_path, mld_dict, components_priors,
#                         outdir=out_dir)
# evidence = result["cg1"]["logz"]
# evidence_error = result["cg1"]["logzerr"]
#
# performance_dict["cg1"] = {"aic": aic, "bic": bic, "logz": evidence,
#                            "logzerr": evidence_error}

# Now substruct component from UV-data
