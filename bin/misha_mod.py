#!/usr/bin python
# -*- coding: utf-8 -*-
# FIXME: -par_plot option fails:
# ValueError: range parameter must be finite
# ('vis_range', [nan, nan])
# ('ticks', [nan, nan])


from __future__ import (print_function)
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
label_size = 6
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
import glob
import numpy as np
from vlbi_errors.uv_data import UVData
from vlbi_errors.spydiff import (import_difmap_model, modelfit_difmap)
from vlbi_errors.bootstrap import CleanBootstrap
from vlbi_errors.model import Model
from vlbi_errors.utils import hdi_of_mcmc

try:
    import corner as triangle
except ImportError:
    triangle = None

outname = 'boot_uv'
errors_fname = 'bootstrap_errors.txt'


if 'DIFMAP_LOGIN' in os.environ:
    del os.environ['DIFMAP_LOGIN']


def xy_2_rtheta(params):
    flux, x, y = params[:3]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.rad2deg(np.arctan(x / y))
    result = [flux, r, theta]
    try:
        result.extend(params[3:])
    except IndexError:
        pass
    return result


def analyze_bootstrap_samples(dfm_model_fname, booted_mdl_paths,
                              dfm_model_dir=None, plot_comps=None,
                              plot_file=None, txt_file=None, cred_mass=0.68,
                              coordinates='xy'):
    """
    Plot bootstrap distribution of model component parameters.

    :param dfm_model_fname:
        File name of original difmap model.
    :param booted_mdl_paths:
        Iterable of paths to bootstrapped difmap models.
    :param dfm_model_dir: (optional)
        Directory with original difmap model. If ``None`` then CWD. (default:
        ``None``)
    :param plot_comps: (optional)
        Iterable of components number to plot on same plot. If ``None`` then
        plot parameter distributions of all components.
    :param plot_file: (optional)
        File to save picture. If ``None`` then don't save picture. (default:
        ``None``)
    :param txt_file: (optional)
        File to save credible intervals for parameters. If ``None`` then don't
        save credible intervals. (default: ``None``)
    :param cred_mass: (optional)
        Value of credible interval mass. Float in range (0., 1.). (default:
        ``0.68``)
    :param coordinates: (optional)
        Type of coordinates to use. ``xy`` or ``rtheta``. (default: ``xy``)
    """
    n_boot = len(booted_mdl_paths)
    # Get params of initial model used for bootstrap
    comps_orig = import_difmap_model(dfm_model_fname, dfm_model_dir)
    comps_params0 = {i: [] for i in range(len(comps_orig))}
    for i, comp in enumerate(comps_orig):
        # FIXME: Move (x, y) <-> (r, theta) mapping to ``Component``
        if coordinates == 'xy':
            params = comp.p
        elif coordinates == 'rtheta':
            params = xy_2_rtheta(comp.p)
        else:
            raise Exception
        comps_params0[i].extend(list(params))

    # Load bootstrap models
    comps_params = {i: [] for i in range(len(comps_orig))}
    for booted_mdl_path in booted_mdl_paths:
        path, booted_mdl_file = os.path.split(booted_mdl_path)
        comps = import_difmap_model(booted_mdl_file, path)
        for i, comp in enumerate(comps):
            # FIXME: Move (x, y) <-> (r, theta) mapping to ``Component``
            if coordinates == 'xy':
                params = comp.p
            elif coordinates == 'rtheta':
                params = xy_2_rtheta(comp.p)
            else:
                raise Exception
            comps_params[i].extend(list(params))

    comps_to_plot = [comps_orig[k] for k in plot_comps]
    # (#boot, #parameters)
    boot_data = np.hstack(np.array(comps_params[i]).reshape((n_boot,
                                                             len(comps_orig[i]))) for
                          i in plot_comps)

    # Optionally plot
    if plot_file:
        if triangle:
            lens = list(np.cumsum([len(comp) for comp in comps_orig]))
            lens.insert(0, 0)

            labels = list()
            for comp in comps_to_plot:
                for lab in comp._parnames:
                    # FIXME: Move (x, y) <-> (r, theta) mapping to ``Component``
                    if coordinates == 'rtheta':
                        if lab == 'x':
                            lab = 'r'
                        if lab == 'y':
                            lab = 'theta'
                    elif coordinates == 'xy':
                        pass
                    else:
                        raise Exception
                    labels.append(lab)

            try:
                n = sum([len(c) for c in comps_to_plot])
                figure, axes = matplotlib.pyplot.subplots(nrows=n, ncols=n)
                figure.set_size_inches(19.5, 19.5)
                triangle.corner(boot_data, labels=labels, plot_contours=False,
                                truths=np.hstack([comps_params0[i] for i in
                                                  plot_comps]),
                                title_kwargs={"fontsize": 6},
                                label_kwargs={"fontsize": 6},
                                quantiles=[0.16, 0.5, 0.84], fig=figure,
                                use_math_text=True, show_titles=True,
                                title_fmt=".3f")
                figure.gca().annotate("Components {}".format(plot_comps),
                                      xy=(0.5, 1.0),
                                      xycoords="figure fraction",
                                      xytext=(0, -5),
                                      textcoords="offset points", ha="center",
                                      va="top")
                figure.savefig(plot_file, bbox_inches='tight', dpi=300)
            except ValueError:
                print("Failed to plot... ValueError")
        else:
            print("Install ``corner`` for corner-plots")

    if txt_file:
        # Print credible intervals
        fn = open(txt_file, 'w')
        fn.write("# parameter original.value low.boot high.boot mean.boot"
                 " median.boot (mean-low).boot (high-mean).boot\n")
        recorded = 0
        for i in plot_comps:
            comp = comps_orig[i]
            for j in range(len(comp)):
                low, high, mean, median = hdi_of_mcmc(boot_data[:, recorded+j],
                                                      cred_mass=cred_mass,
                                                      return_mean_median=True)
                # FIXME: Move (x, y) <-> (r, theta) mapping to ``Component``
                parnames = comp._parnames
                if coordinates == 'xy':
                    params = comp.p
                elif coordinates == 'rtheta':
                    params = xy_2_rtheta(comp.p)
                    parnames[1] = 'r'
                    parnames[2] = 'theta'
                else:
                    raise Exception
                fn.write("{:<4} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}"
                         " {:.4f}".format(parnames[j], params[j], low,
                                          high, mean, median, abs(median - low),
                                          abs(high - median)))
                fn.write("\n")
            recorded += (j + 1)
        fn.close()


if __name__ == "__main__":
    parser = \
        argparse.ArgumentParser(description='Bootstrap Difmap models.\n'
                                            ' Required '
                                            'arguments are ``dfm_model_path`` &'
                                            ' one of the ``-uv_fits_path`` or'
                                            ' ``-booted_mdl_path``. If'
                                            ' ``-booted_mdl_path`` is used then'
                                            ' options ``-parametric``,'
                                            ' ``-n_boot``, ``-n_iter``,'
                                            ' ``-res_plot`` are not used.')

    parser.add_argument('dfm_model_path', type=str, metavar='dfm_model_path',
                        help='Path to Difmap-format file with model.')
    parser.add_argument('-uv_fits_path', action='store', nargs='?', type=str,
                        metavar='PATH TO UV-DATA FITS FILE', default=None,
                        help='Path to FITS-file with self-calibrated UV-data.')
    parser.add_argument('-booted_mdl_card', action='store', nargs='?', type=str,
                        default=None,
                        help='Wildcard to find bootstrapped model'
                             ' files', metavar='WILCARD WITH FULL PATH')
    parser.add_argument('-n_boot', action='store', nargs='?', default=100,
                        type=int, help='Number of bootstrap realizations.'
                                       ' Default value = 100',
                        metavar='INT')
    parser.add_argument('-n_iter', action='store', nargs='?', default=50,
                        type=int, help='Number of iterations in difmap internal'
                                       ' fitting. Default is 50.',
                        metavar='INT')
    parser.add_argument('-cred_value', action='store', nargs='?', default=0.68,
                        type=float, help='Credible interval specification.'
                                         ' Float from (0, 1) interval. Default'
                                         ' is 0.68.',
                        metavar='FLOAT FROM (0, 1)')
    parser.add_argument('-out_dir', action='store', nargs='?',
                        default=os.getcwd(), type=str, help='Directory to store'
                                                     ' bootstrap files, models'
                                                     ' & results.',
                        metavar='DIRECTORY')
    parser.add_argument('-errors_file', action='store', nargs='?',
                        default='bootstrap_errors.txt', type=str,
                        help='File name to store bootstrap errors. Default is'
                             '`bootstrap_errors.txt`.',
                        metavar='FILE NAME')
    parser.add_argument('-res_plot', action='store', nargs='?', default=None,
                        type=str, help='File name to store IF-averages'
                                       ' residuals of Stokes I real & imag part'
                                       ' plot in output directory.',
                        metavar='FILE NAME')
    parser.add_argument('-res_plot_full', action='store', nargs='?', default=None,
                        type=str, help='File name to store residuals of Stokes '
                                       'RR & LL real & imag part'
                                       ' plot in output directory.',
                        metavar='FILE NAME')
    parser.add_argument('-par_plot', action='store', nargs='?', default=None,
                        type=str, help='File name to store parameters plot in'
                                       ' output directory.',
                        metavar='FILE NAME')
    parser.add_argument('-plot_comps', action='store', nargs='+', default=None,
                        type=str, help='Components numbers to plot.',
                        metavar='COMPONENT #')
    parser.add_argument('-txt_comps', action='store', nargs='*', default=None,
                        type=str, help='Components numbers to output'
                                       ' parameters in a text file.',
                        metavar='COMPONENT #')
    parser.add_argument('-parametric', action='store_true', dest='parametric',
                        default=False,
                        help='Use parametric bootstrap instead of'
                             ' nonparametric (nonparametric is the default).')
    parser.add_argument('-recenter', action='store_true', dest='recenter',
                        default=False,
                        help='Recenter residuals on each baseline.')
    parser.add_argument('-clean_after', action='store_true', dest='clean_after',
                        default=False,
                        help='Remove bootstrapped data & model files in the'
                             ' end.')
    parser.add_argument('-bic', action='store_true', dest='bic',
                        default=False,
                        help='Calculate BIC criterion value for original model'
                             ' and bootstrapped samples.')
    parser.add_argument('-rtheta', action='store_true', dest='use_rtheta',
                        default=False,
                        help='Use `r-theta` coordinates instead of `xy`.')
    parser.add_argument('-split_scans', action='store_true', dest='split_scans',
                        default=False, help='Resample each scan individually?')

    args = parser.parse_args()

    data_dir = args.out_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print("Data directory: {}".format(data_dir))

    cred_value = args.cred_value
    uv_fits_path = args.uv_fits_path
    booted_mdl_card = args.booted_mdl_card
    dfm_model_path = args.dfm_model_path
    n_boot = args.n_boot
    niter = args.n_iter
    nonparametric = not args.parametric
    errors_fname = args.errors_file
    par_plot = args.par_plot
    recenter = args.recenter
    plot_comps = args.plot_comps
    txt_comps = args.txt_comps
    split_scans = args.split_scans

    bic = args.bic
    if args.use_rtheta:
        coordinates = 'rtheta'
    else:
        coordinates = 'xy'

    if par_plot and not plot_comps:
        raise Exception("Use -plot_comps argument to specify # of components"
                        " to plot")

    dfm_model_dir, dfm_model_fname = os.path.split(dfm_model_path)
    try:
        comps = import_difmap_model(dfm_model_fname, dfm_model_dir)
    except ValueError:
        print("Problem importing difmap model...")
        sys.exit(1)

    # Check that component numbers in input are among model components
    if plot_comps:
        for c in plot_comps:
            if int(c) not in range(len(comps)):
                raise Exception("No such component {} in current"
                                " model!".format(c))
    if not txt_comps:
        txt_comps = range(len(comps))
    else:
        txt_comps = [int(k) for k in txt_comps]
    for c in txt_comps:
        if int(c) not in range(len(comps)):
            raise Exception("No such component {} in current model!".format(c))

    if uv_fits_path:
        print("Bootstrapping uv-data")
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        boot_type_dict = {True: "non-parametric", False: "parametric"}
        print("==================================")
        print("Bootstrap uv-data: {}".format(uv_fits_fname))
        print("With model: {}".format(dfm_model_fname))
        print("Using {} bootstrap".format(boot_type_dict[nonparametric]))
        if not nonparametric:
            if recenter:
                print("Recentering KDE-fitted residuals")
            else:
                print("Using fitted KDE Model to generate resamples")
        print("Using {} bootstrap replications".format(n_boot))
        print("Using {} fitting iterations".format(niter))
        print("Finding {}-confidence regions".format(cred_value))
        print("Using directory {} for storing output".format(data_dir))
        txt_save_dict = {'None': "all"}
        try:
            print("Saving errors of {} components to file"
                  " {}".format(txt_save_dict[str(txt_comps)], errors_fname))
        except KeyError:
            print("Saving errors of {} components to file"
                  " {}".format(txt_comps, errors_fname))

        if split_scans:
            print("Resampling each scan individually")
        if par_plot:
            print("Saving components {} parameters distributions plot to file"
                  " {}".format(plot_comps, par_plot))
        if args.res_plot:
            print("Saving residuals I plot to file {}".format(args.res_plot))
        if args.res_plot_full:
            print("Saving residulas RR & LL plots to files"
                  " {}*".format(args.res_plot_full))
        print("==================================")

        uvdata = UVData(uv_fits_path)
        model = Model(stokes='I')
        model.add_components(*comps)

        if bic:
            bic_orig = model.bic(uvdata)
            bic_booted = list()

        try:
            boot = CleanBootstrap([model], uvdata)
        # If uv-data contains only one Stokes parameter (e.g. `0838+133`)
        except IndexError:
            print("Problem in bootstrapping data...")
            sys.exit(1)
        # FIXME: Broken - ValueError
        if args.res_plot:
            print("Plotting histograms of I residuals...")
            boot.plot_residuals(args.res_plot)

        curdir = os.getcwd()
        os.chdir(data_dir)
        boot.run(n=n_boot, nonparametric=nonparametric, outname=[outname,
                                                                 '.fits'],
                 recenter=recenter, use_kde=False, use_v=False)
        if args.res_plot_full:
            print("Plotting histograms of RR & LL residuals...")
            boot.plot_residuals_trio(args.res_plot_full, split_scans,
                                     stokes=['RR', 'LL'])

        os.chdir(curdir)

        booted_uv_paths = sorted(glob.glob(os.path.join(data_dir,
                                                        outname + "*")))
        booted_mdl_paths = list()
        # Modelfit bootstrapped uvdata
        for booted_uv_path in booted_uv_paths:
            path, booted_uv_file = os.path.split(booted_uv_path)
            i = booted_uv_file.split('_')[-1].split('.')[0]
            out_fname = dfm_model_fname + '_' + i
            modelfit_difmap(booted_uv_file, dfm_model_fname, out_fname,
                            path=path, mdl_path=dfm_model_dir,
                            out_path=data_dir, niter=niter)
            booted_mdl_paths.append(os.path.join(data_dir, out_fname))
            if bic:
                uvdata_ = UVData(booted_uv_path)
                model_ = Model(stokes='I')
                comps_ = import_difmap_model(out_fname, data_dir)
                model_.add_components(comps_)
                bic_booted.append(model_.bic(uvdata_))

        if bic:
            low_, high_, mean_, median_ = hdi_of_mcmc(bic_booted,
                                                      cred_mass=0.68,
                                                      return_mean_median=True)
            print("Model BIC with bootstrapped 68% interval = {:.2f}"
                  " -{:.2f} +{:.2f}".format(bic_orig, abs(mean_ - low_),
                                            abs(high_ - mean_)))

    elif booted_mdl_card:
        print("Using already bootstrapped uv-data")
        booted_mdl_paths = glob.glob(booted_mdl_card)
        n_boot = len(booted_mdl_paths)
        print("==================================")
        print("With {} bootstrap replications".format(n_boot))
        print("Finding {}-confidence regions".format(cred_value))
        print("Using directory {} for storing output".format(data_dir))
        txt_save_dict = {'None': "all"}
        try:
            print("Saving errors of {} components to file"
                  " {}".format(txt_save_dict[str(txt_comps)], errors_fname))
        except KeyError:
            print("Saving errors of {} components to file"
                  " {}".format(txt_comps, errors_fname))
        if par_plot:
            print("Saving components {} parameters distributions plot to file"
                  " {}".format(plot_comps, par_plot))
        print("==================================")

    else:
        raise Exception("Use -uv_fits_path or -booted_mdl_card to create/get"
                        " bootstrapped models.")

    # Optionally plot component parameters
    if par_plot:
        plot_comps = [int(k) for k in plot_comps]
        analyze_bootstrap_samples(dfm_model_fname, booted_mdl_paths,
                                  dfm_model_dir=dfm_model_dir,
                                  plot_comps=plot_comps,
                                  plot_file=os.path.join(data_dir, par_plot),
                                  cred_mass=cred_value,
                                  coordinates=coordinates)
    analyze_bootstrap_samples(dfm_model_fname, booted_mdl_paths,
                              dfm_model_dir=dfm_model_dir,
                              plot_comps=txt_comps,
                              txt_file=os.path.join(data_dir, errors_fname),
                              cred_mass=cred_value, coordinates=coordinates)

    if args.clean_after:
        for rmfile in booted_uv_paths:
            os.unlink(rmfile)
        for rmfile in booted_mdl_paths:
            os.unlink(rmfile)
