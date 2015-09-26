import os
import glob
import numpy as np
from from_fits import create_uvdata_from_fits_file
from components import CGComponent, DeltaComponent
from bootstrap import CleanBootstrap
from model import Model
from utils import mas_to_rad
from spydiff import modelfit_difmap, clean_difmap


path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
data_dir = '/home/ilya/vlbi_errors/difmap/'
uv_fname = '1226+023.q1.2008_12_21.uvp'
mdl_fname = '1226+023.q1.2008_12_21.mdl'
outname = 'boot_uv'
n = 100


def import_difmap_model(mdl_fname, data_dir):
    mdl = os.path.join(data_dir, mdl_fname)
    mdlo = open(mdl, 'r')
    lines = mdlo.readlines()
    comps = list()
    for line in lines:
        if line.startswith('!'):
            continue
        line = line.strip('\n ')
        flux, radius, theta, major, axial, phi, type_, freq, spec = line.split()
        x = -float(radius[:-1]) * np.sin(np.deg2rad(float(theta[:-1])))
        y = -float(radius[:-1]) * np.cos(np.deg2rad(float(theta[:-1])))
        flux = float(flux[:-1])
        if int(type_) == 0:
            comp = DeltaComponent(flux, x, y)
        if int(type_) == 1:
            bmaj = float(major[:-1])
            comp = CGComponent(flux, x, y, bmaj)
        comps.append(comp)
    return comps


if __name__ == '__main__':

    uvdata = create_uvdata_from_fits_file(os.path.join(data_dir, uv_fname))
    model = Model(stokes='I')
    comps = import_difmap_model(mdl_fname, data_dir)
    model.add_components(*comps)
    boot = CleanBootstrap([model], uvdata)
    curdir = os.getcwd()
    os.chdir(data_dir)
    boot.run(n=n, nonparametric=True, outname=[outname, '.fits'])
    os.chdir(curdir)

    booted_uv_paths = glob.glob(os.path.join(data_dir, outname + "*"))
    # Modelfit bootstrapped uvdata
    for booted_uv_path in booted_uv_paths:
        path, booted_uv_file = os.path.split(booted_uv_path)
        i = booted_uv_file.split('_')[-1].split('.')[0]
        modelfit_difmap(booted_uv_file, mdl_fname, mdl_fname + '_' + i, path=path,
                        mdl_path=data_dir, out_path=data_dir)

    # Load models and plot
    params = list()
    booted_mdl_paths = glob.glob(os.path.join(data_dir, mdl_fname + "_*"))
    for booted_mdl_path in booted_mdl_paths:
        path, booted_mdl_file = os.path.split(booted_mdl_path)
        comps = import_difmap_model(booted_mdl_file, path)
        params.append([comp.p if isinstance(comp, CGComponent) else
                       np.array(list(comp.p) + [0.]) for comp in comps])
    params = np.dstack(params)
    # params[:, 1:3, :] /= mas_to_rad

    # # Clean sc uvdata to see where flux is
    # clean_difmap(uv_fname, "clean_i.fits", stokes='i', mapsize_clean=(1024, 0.1),
    #              path=data_dir, path_to_script=path_to_script, outpath=data_dir)

