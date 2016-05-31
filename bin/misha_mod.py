#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import (print_function)
import os
import sys
import warnings
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
import glob
import numpy as np
from vlbi_errors.uv_data import UVData
from vlbi_errors.spydiff import (import_difmap_model, modelfit_difmap)
from vlbi_errors.bootstrap import CleanBootstrap
from vlbi_errors.model import Model
from vlbi_errors.utils import hdi_of_mcmc

outname = 'boot_uv'
errors_fname = 'bootstrap_errors.txt'


if 'DIFMAP_LOGIN' in os.environ:
    del os.environ['DIFMAP_LOGIN']


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        print("Usage: python misha_mod.py uv-fits-path dfm-model-path n_boot"
              " n_iter cred_value [data_dir]")
        sys.exit(1)

    if len(sys.argv) == 7:
        data_dir = sys.argv[6]
    else:
        data_dir = os.getcwd()
    print("Data directory: {}".format(data_dir))

    cred_value = float(sys.argv[5])

    uv_fits_path = sys.argv[1]
    uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    dfm_model_path = sys.argv[2]
    dfm_model_dir, dfm_model_fname = os.path.split(dfm_model_path)
    n_boot = int(sys.argv[3])
    niter = int(sys.argv[4])
    print("==================================")
    print("Bootstrap uv-data: {}".format(uv_fits_fname))
    print("With model: {}".format(dfm_model_fname))
    print("Using {} replications".format(n_boot))
    print("Using {} iterations".format(niter))
    print("Finding {}-confidence regions".format(cred_value))
    print("==================================")

    try:
        comps = import_difmap_model(dfm_model_fname, dfm_model_dir)
    except ValueError:
        print("Problem importing difmap model...")
        sys.exit(1)
    uvdata = UVData(uv_fits_path)
    model = Model(stokes='I')
    model.add_components(*comps)
    try:
        boot = CleanBootstrap([model], uvdata)
    # If uv-data contains only one Stokes parameter (e.g. `0838+133`)
    except IndexError:
        print("Problem in bootstrapping data...")
        sys.exit(1)
    curdir = os.getcwd()
    os.chdir(data_dir)
    boot.run(n=n_boot, nonparametric=True, outname=[outname, '.fits'])
    os.chdir(curdir)

    booted_uv_paths = sorted(glob.glob(os.path.join(data_dir, outname + "*")))
    # Modelfit bootstrapped uvdata
    for booted_uv_path in booted_uv_paths:
        path, booted_uv_file = os.path.split(booted_uv_path)
        i = booted_uv_file.split('_')[-1].split('.')[0]
        modelfit_difmap(booted_uv_file, dfm_model_fname,
                        dfm_model_fname + '_' + i,
                        path=path, mdl_path=dfm_model_dir, out_path=data_dir,
                        niter=niter)

    # Get params of initial model used for bootstrap
    comps = import_difmap_model(dfm_model_fname, dfm_model_dir)
    comps_params0 = {i: [] for i in range(len(comps))}
    for i, comp in enumerate(comps):
        comps_params0[i].extend(list(comp.p))

    # Load bootstrap models
    booted_mdl_paths = glob.glob(os.path.join(data_dir, dfm_model_fname + "_*"))
    comps_params = {i: [] for i in range(len(comps))}
    for booted_mdl_path in booted_mdl_paths:
        path, booted_mdl_file = os.path.split(booted_mdl_path)
        comps = import_difmap_model(booted_mdl_file, path)
        for i, comp in enumerate(comps):
            comps_params[i].extend(list(comp.p))

    # Print 65-% intervals (1 sigma)
    comps = import_difmap_model(dfm_model_fname, dfm_model_dir)
    fn = open(os.path.join(data_dir, errors_fname), 'w')
    for i, comp in enumerate(comps):
        print("Component #{}".format(i + 1))
        for j in range(len(comp)):
            low, high, mean, median = hdi_of_mcmc(np.array(comps_params[i]).reshape((n_boot,
                                                                                     len(comp))).T[j],
                                                  cred_mass=cred_value,
                                                  return_mean_median=True)
            print("par, low, high : {} {} {}".format(comp.p[j], low, high))
            fn.write("{} {} {} ".format(comp.p[j], abs(mean - low),
                                        abs(high - mean)))
        if j == 2:
            fn.write(" {} {} {} {} {} {} {} {} {}".format(0, 0, 0, 0, 0, 0, 0,
                                                          0, 0))
        elif j == 3:
            fn.write(" {} {} {} {} {} {}".format(0, 0, 0, 0, 0, 0))
        elif j == 5:
            pass
        else:
            raise Exception

        fn.write("\n")
    fn.close()
