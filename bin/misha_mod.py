#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import (print_function)
import os
import sys
import warnings
import argparse
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
    parser = \
        argparse.ArgumentParser(description='Bootstrap Difmap models')

    parser.add_argument('-parametric', action='store_true', dest='parametric',
                        default=False,
                        help='Use parametric bootstrap instead of'
                             ' nonparametric (nonparametric is the default)')
    parser.add_argument('-clean_after', action='store_true', dest='clean_after',
                        default=False,
                        help='Remove bootstrapped data & model files in the'
                             ' end')
    parser.add_argument('uv_fits_path', type=str, metavar='uv_fits_path',
                        help='- path to FITS-file with self-calibrated UV-data')
    parser.add_argument('dfm_model_path', type=str, metavar='dfm_model_path',
                        help='- path to Difmap-format file with model')
    parser.add_argument('-n_boot', action='store', nargs='?', default=100,
                        type=int, help='Number of bootstrap realizations')
    parser.add_argument('-n_iter', action='store', nargs='?', default=50,
                        type=int, help='Number of iterations in fitting')
    parser.add_argument('-cred_value', action='store', nargs='?', default=0.68,
                        type=float, help='Credible interval specification.'
                                         ' Float from (0, 1) interval')
    parser.add_argument('-out_dir', action='store', nargs='?',
                        default=None, type=str, help='Directory to store'
                                                     ' bootstrap files, models'
                                                     ' & results.')
    parser.add_argument('-errors_file', action='store', nargs='?',
                        default='bootstrap_errors.txt', type=str,
                        help='File name to store bootstrap errors')

    args = parser.parse_args()

    data_dir = args.out_dir
    if data_dir is None:
        data_dir = os.getcwd()
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print("Data directory: {}".format(data_dir))

    cred_value = args.cred_value
    uv_fits_path = args.uv_fits_path
    dfm_model_path = args.dfm_model_path
    n_boot = args.n_boot
    niter = args.n_iter
    nonparametric = not args.parametric
    errors_fname = args.errors_file

    uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    dfm_model_dir, dfm_model_fname = os.path.split(dfm_model_path)
    boot_type_dict = {True: "non-parametric", False: "parametric"}
    print("==================================")
    print("Bootstrap uv-data: {}".format(uv_fits_fname))
    print("With model: {}".format(dfm_model_fname))
    print("Using {} bootstrap".format(boot_type_dict[nonparametric]))
    print("Using {} bootstrap replications".format(n_boot))
    print("Using {} fitting iterations".format(niter))
    print("Finding {}-confidence regions".format(cred_value))
    print("Using directory {} for storing output".format(data_dir))
    print("Saving errors to file {}".format(errors_fname))
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
    boot.run(n=n_boot, nonparametric=nonparametric, outname=[outname, '.fits'])
    os.chdir(curdir)

    booted_uv_paths = sorted(glob.glob(os.path.join(data_dir, outname + "*")))
    booted_mdl_paths = list()
    # Modelfit bootstrapped uvdata
    for booted_uv_path in booted_uv_paths:
        path, booted_uv_file = os.path.split(booted_uv_path)
        i = booted_uv_file.split('_')[-1].split('.')[0]
        out_fname = dfm_model_fname + '_' + i
        modelfit_difmap(booted_uv_file, dfm_model_fname, out_fname, path=path,
                        mdl_path=dfm_model_dir, out_path=data_dir, niter=niter)
        booted_mdl_paths.append(os.path.join(data_dir, out_fname))

    # Get params of initial model used for bootstrap
    comps = import_difmap_model(dfm_model_fname, dfm_model_dir)
    comps_params0 = {i: [] for i in range(len(comps))}
    for i, comp in enumerate(comps):
        comps_params0[i].extend(list(comp.p))

    # Load bootstrap models
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
            print("par, low, high : {:.4f} {:.4f} {:.4f}".format(comp.p[j], low,
                                                                 high))
            fn.write("{:.4f} {:.4f} {:.4f} ".format(comp.p[j], abs(mean - low),
                                                    abs(high - mean)))
        if j == 2:
            fn.write(" {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}"
                     " {:.4f}".format(0, 0, 0, 0, 0, 0, 0, 0, 0))
        elif j == 3:
            fn.write(" {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}"
                     " {:.4f}".format(0, 0, 0, 0, 0, 0))
        elif j == 5:
            pass
        else:
            raise Exception

        fn.write("\n")
    fn.close()

    if args.clean_after:
        for rmfile in booted_uv_paths:
            os.unlink(rmfile)
        for rmfile in booted_mdl_paths:
            os.unlink(rmfile)
