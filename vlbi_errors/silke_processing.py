import os
import glob
import numpy as np
from mojave import download_mojave_uv_fits
from spydiff import import_difmap_model, modelfit_difmap
from utils import hdi_of_mcmc
from uv_data import UVData
from model import Model
from mojave import mojave_uv_fits_fname
from bootstrap import CleanBootstrap
import corner as triangle
import matplotlib
label_size = 16
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size


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
                                                             comps_orig[i].size)) for
                          i in plot_comps)

    # Optionally plot
    if plot_file:
        if triangle:
            lens = list(np.cumsum([comp.size for comp in comps_orig]))
            lens.insert(0, 0)

            labels = list()
            for comp in comps_to_plot:
                for lab in np.array(comp._parnames)[~comp._fixed]:
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
                n = sum([c.size for c in comps_to_plot])
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
            for j in range(comp.size):
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
                fn.write("{:<4} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}"
                         " {:.6f}".format(parnames[j], params[j], low,
                                          high, mean, median, abs(median - low),
                                          abs(high - median)))
                fn.write("\n")
            recorded += (j + 1)
        fn.close()


data_dir = '/home/ilya/silke'
original_dfm_models = glob.glob(os.path.join(data_dir, '*us'))
for path in original_dfm_models:
    fname = os.path.split(path)[-1]
    epoch = fname[:-2]
    download_mojave_uv_fits('0851+202', epochs=[epoch], download_dir=data_dir)


# Test proccess single epoch
data_dir = '/home/ilya/silke'
boot_dir = '/home/ilya/silke/boot'
epoch = '2017_01_28'
original_model_fname = '2017_01_28us'
original_model_path = os.path.join(data_dir, original_model_fname)
uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
uv_fits_path = os.path.join(data_dir, uv_fits_fname)
comps = import_difmap_model(original_model_fname, data_dir)
model = Model(stokes='I')
model.add_components(*comps)
uvdata = UVData(uv_fits_path)

boot = CleanBootstrap([model], uvdata)
os.chdir(boot_dir)
boot.run(nonparametric=False, use_kde=True, recenter=True, use_v=False,
         n=100)
bootstrapped_uv_fits = sorted(glob.glob(os.path.join(boot_dir,
                                                     'bootstrapped_data*.fits')))
for j, bootstrapped_fits in enumerate(bootstrapped_uv_fits):
    modelfit_difmap(bootstrapped_fits, original_model_fname, 'boot_{}.mdl'.format(j),
                    path=boot_dir, mdl_path=data_dir,
                    out_path=boot_dir, niter=100)
booted_mdl_paths = glob.glob(os.path.join(boot_dir, 'boot_*.mdl'))
analyze_bootstrap_samples(original_model_fname, booted_mdl_paths, data_dir,
                          plot_comps=range(len(comps)),
                          plot_file='plot_{}.png'.format(epoch),
                          txt_file='txt_{}.txt'.format(epoch))
# Clean
for file_ in bootstrapped_uv_fits:
    os.unlink(file_)
for file_ in booted_mdl_paths:
    os.unlink(file_)
