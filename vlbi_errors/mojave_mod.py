import matplotlib
matplotlib.use('Agg')
label_size = 8
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
import glob
import os
import pandas as pd
import numpy as np
from utils import hdi_of_mcmc
from image_ops import rms_image
from mojave import download_mojave_uv_fits, mojave_uv_fits_fname
from spydiff import import_difmap_model, modelfit_difmap, clean_difmap
from uv_data import UVData
from model import Model
from bootstrap import CleanBootstrap
from components import DeltaComponent, CGComponent, EGComponent
from from_fits import create_image_from_fits_file

try:
    import corner as triangle
except ImportError:
    triangle = None

# base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mojave_mod_first'
# n_boot = 100
# outname = 'boot_uv'
# names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
#          'bpa']
# df = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
#                    names=names, dtype={key: str for key in names},
#                    index_col=False)
#
# # Mow for all sources get the latest epoch and create directory for analysis
# for source in df['source'].unique()[44:]:
#     epochs = df.loc[df['source'] == source]['epoch']
#     last_epoch_ = list(epochs)[0]
#     last_epoch = last_epoch_.replace('-', '_')
#     data_dir = os.path.join(base_dir, source, last_epoch)
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     try:
#         download_mojave_uv_fits(source, epochs=[last_epoch], bands=['u'],
#                                 download_dir=data_dir)
#     except:
#         open(os.path.join(data_dir,  'problem_download_from_mojave'), 'a').close()
#         continue
#     uv_fits_fname = mojave_uv_fits_fname(source, 'u', last_epoch)
#
#     # Clean uv-data
#     path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
#     clean_difmap(uv_fits_fname, 'original_cc.fits', 'I', (1024, 0.1),
#                  path=data_dir, path_to_script=path_to_script,
#                  outpath=data_dir)
#     image = create_image_from_fits_file(os.path.join(data_dir,
#                                                      'original_cc.fits'))
#     rms = rms_image(image)
#
#     # Create instance of Model and bootstrap uv-data
#     dfm_model_fname = 'dfmp_original_model.mdl'
#     fn = open(os.path.join(data_dir, dfm_model_fname), 'w')
#     model_df = df.loc[np.logical_and(df['source'] == source,
#                                      df['epoch'] == last_epoch_)]
#     for (flux, r, pa, bmaj, e, bpa) in np.asarray(model_df[['flux', 'r', 'pa',
#                                                             'bmaj', 'e',
#                                                             'bpa']]):
#         print flux, r, pa, bmaj, e, bpa
#         if not r.strip(' '):
#             r = '0.0'
#         if not pa.strip(' '):
#             pa = '0.0'
#
#         if not bmaj.strip(' '):
#             bmaj = '0.0'
#         if not e.strip(' '):
#             e = "1.0"
#
#         if np.isnan(float(bpa)):
#             bpa = "0.0"
#         else:
#             bpa = bpa + 'v'
#
#         if bmaj == '0.0':
#             type_ = 0
#             bpa = "0.0"
#         else:
#             bmaj = bmaj + 'v'
#             type_ = 1
#         fn.write("{}v {}v {}v {} {} {} {} {} {}".format(flux, r, pa, bmaj, e,
#                                                         bpa, type_, "0", "0\n"))
#     fn.close()
#
#     try:
#         comps = import_difmap_model(dfm_model_fname, data_dir)
#     except ValueError:
#         open('problem_import_difmap_model', 'a').close()
#         continue
#     uvdata = UVData(os.path.join(data_dir, uv_fits_fname))
#     model = Model(stokes='I')
#     model.add_components(*comps)
#     try:
#         boot = CleanBootstrap([model], uvdata)
#     # If uv-data contains only one Stokes parameter (e.g. `0838+133`)
#     except IndexError:
#         open('problem_bootstrapping', 'a').close()
#         continue
#     curdir = os.getcwd()
#     os.chdir(data_dir)
#     boot.run(n=n_boot, nonparametric=False, use_v=False,
#              outname=[outname, '.fits'])
#     os.chdir(curdir)
#
#     booted_uv_paths = sorted(glob.glob(os.path.join(data_dir, outname + "*")))
#     # Modelfit bootstrapped uvdata
#     for booted_uv_path in booted_uv_paths:
#         path, booted_uv_file = os.path.split(booted_uv_path)
#         i = booted_uv_file.split('_')[-1].split('.')[0]
#         modelfit_difmap(booted_uv_file, dfm_model_fname,
#                         dfm_model_fname + '_' + i, niter=100,
#                         path=path, mdl_path=data_dir, out_path=data_dir)
#
#     # Get params of initial model used for bootstrap
#     comps = import_difmap_model(dfm_model_fname, data_dir)
#     comps_params0 = {i: [] for i in range(len(comps))}
#     for i, comp in enumerate(comps):
#         comps_params0[i].extend(list(comp.p))
#
#     # Load models and plot
#     booted_mdl_paths = glob.glob(os.path.join(data_dir, dfm_model_fname + "_*"))
#     comps_params = {i: [] for i in range(len(comps))}
#     for booted_mdl_path in booted_mdl_paths:
#         path, booted_mdl_file = os.path.split(booted_mdl_path)
#         comps = import_difmap_model(booted_mdl_file, path)
#         for i, comp in enumerate(comps):
#             comps_params[i].extend(list(comp.p))
#
#     n_pars = sum([len(comp) for comp in comps])
#     labels = {0: "flux", 1: "x", 2: "y", 3: "FWHM"}
#
#     # Optionally plot
#     if triangle:
#         lens = list(np.cumsum([len(comp) for comp in comps]))
#         lens.insert(0, 0)
#         for i, comp in enumerate(comps):
#             # Show one component
#             try:
#                 figure, axes = matplotlib.pyplot.subplots(nrows=len(comp), ncols=len(comp))
#                 triangle.corner(np.array(comps_params[i]).reshape((n_boot,
#                                                                    len(comp))),
#                                 labels=[r"${}$".format(lab) for lab in
#                                         comp._parnames],
#                                 truths=comps_params0[i],
#                                 title_kwargs={"fontsize": 10},
#                                 quantiles=[0.16, 0.5, 0.84], fig=figure)
#                 figure.gca().annotate("Source {}, component {}".format(source, i),
#                                       xy=(0.5, 1.0), xycoords="figure fraction",
#                                       xytext=(0, -5), textcoords="offset points",
#                                       ha="center", va="top")
#                 figure.savefig(os.path.join(data_dir,
#                                             '{}_{}_comp{}.png'.format(source,
#                                                                       last_epoch,
#                                                                       i)),
#                                bbox_inches='tight', dpi=300)
#             except ValueError:
#                 print "Failed to plot... ValueError"
#     else:
#         print "Install ``corner`` for corner-plots"
#
#     # Print 65-% intervals (1 sigma)
#     for i, comp in enumerate(comps):
#         errors_fname = '68_{}_{}_comp{}_rms_{}.txt'.format(source, last_epoch, i,
#                                                        "{0:.5f}".format(rms))
#         fn = open(os.path.join(data_dir, errors_fname), 'w')
#         print "Component #{}".format(i + 1)
#         for j in range(len(comp)):
#             low, high, mean, median = hdi_of_mcmc(np.array(comps_params[i]).reshape((n_boot,
#                                                                                      len(comp))).T[j],
#                                                   cred_mass=0.68,
#                                                   return_mean_median=True)
#             fn.write("{} {} {} {} {}".format(comp.p[j], low, high, mean,
#                                              median))
#             fn.write("\n")
#         fn.close()
#     # Print 95-% intervals (2 sigma)
#     for i, comp in enumerate(comps):
#         errors_fname = '95_{}_{}_comp{}_rms_{}.txt'.format(source, last_epoch, i,
#                                                        "{0:.5f}".format(rms))
#         fn = open(os.path.join(data_dir, errors_fname), 'w')
#         print "Component #{}".format(i + 1)
#         for j in range(len(comp)):
#             low, high, mean, median = hdi_of_mcmc(np.array(comps_params[i]).reshape((n_boot,
#                                                                                      len(comp))).T[j],
#                                                   cred_mass=0.95,
#                                                   return_mean_median=True)
#             fn.write("{} {} {} {} {}".format(comp.p[j], low, high, mean,
#                                              median))
#             fn.write("\n")
#         fn.close()
#
#     # Cleaning up
#     for booted_uv_path in booted_uv_paths:
#         print("Removing file {}".format(booted_uv_path))
#         os.unlink(booted_uv_path)


def sort_comp_files(files):
    files_dict = dict()
    for fl in files:
        data = np.loadtxt(fl)
        r = np.sqrt(data[2, 0]**2 + data[1, 0]**2)
        files_dict[fl] = r
    files_sorted = [x[0] for x in sorted(files_dict.items(), key=lambda x: x[1])]
    return files_sorted


def process_flux_components(n):
    names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
             'bpa']
    # base_dir = '/home/ilya/vlbi_errors/mojave_mod/mojave_mod'
    base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mojave_mod_first'
    import pandas as pd
    df = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
                       names=names, dtype={key: str for key in names},
                       index_col=False)
    fluxes = list()
    for source in df['source'].unique():
        print("Checking {}".format(source))
        epochs = df.loc[df['source'] == source]['epoch']
        last_epoch_ = list(epochs)[-1]
        first_epoch_ = list(epochs)[0]
        last_epoch = last_epoch_.replace('-', '_')
        first_epoch = first_epoch_.replace('-', '_')
        data_dir = os.path.join(base_dir, source, last_epoch)
        data_dir_first = os.path.join(base_dir, source, first_epoch)
        import glob
        files = sorted(glob.glob(os.path.join(data_dir_first, "68*comp*txt")))
        if not files:
            print("No component files")
            continue
        files = sort_comp_files(files)
        # Now files - list with core - first file, first component - second
        # file, last component - last file in ``files`` list.
        try:
            files_set = [files[n]]
        except IndexError:
            continue
        for file_ in files_set:
            try:
                with open(file_) as fo:
                    pars = fo.readlines()
                    flux_line = pars[0].strip().split(' ')
                    x_line = pars[1].strip().split(' ')
                    y_line = pars[2].strip().split(' ')
                    x = float(x_line[0])
                    y = float(y_line[0])
                    r = np.sqrt(x**2 + y**2)
                    boot_sigma = float(flux_line[2]) - float(flux_line[1])
                    flux = float(flux_line[0])
                    if flux < 0:
                        print("Negative flux")
                        continue
                    if flux > 10:
                        print("Too big flux")
                        continue
                    if boot_sigma > 100.*flux:
                        print("Too big boot_sigma for source {}".format(source))
                        continue
                    rms = float(file_.split('_')[-1][:-4])
                    fluxes.append([flux, boot_sigma, rms, r])
            except IOError:
                print("Failed reading file {}".format(file_))
                continue
    fluxes = np.atleast_2d(fluxes)
    return fluxes


def process_position_components(n):
    names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
             'bpa']
    # base_dir = '/home/ilya/vlbi_errors/mojave_mod/mojave_mod'
    base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mojave_mod_first'
    import pandas as pd
    df = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
                       names=names, dtype={key: str for key in names},
                       index_col=False)
    from from_fits import create_clean_image_from_fits_file
    params = list()
    params_dict = dict()
    for source in df['source'].unique():
        params_dict[source] = dict()
        print "Checking {}".format(source)
        epochs = df.loc[df['source'] == source]['epoch']
        last_epoch_ = list(epochs)[0]
        last_epoch = last_epoch_.replace('-', '_')
        data_dir = os.path.join(base_dir, source, last_epoch)
        import glob
        try:
            ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                     'original_cc.fits'))
            beam = ccimage.beam[0] * ccimage.beam[1] / 2.
        except IOError:
            print "No image"
            continue
        files = sorted(glob.glob(os.path.join(data_dir, "68*comp*txt")))
        if not files:
            print "No component files"
            continue
        files = sort_comp_files(files)
        # Now files - list with core - first file, first component - second
        # file, last component - last file in ``files`` list.
        try:
            files_set = [files[n]]
        except IndexError:
            continue
        for file_ in files_set:
            print "Checking file {}".format(file_)
            try:
                with open(file_) as fo:
                    pars = fo.readlines()
                    flux_line = pars[0].strip().split(' ')
                    xs_line = pars[1].strip().split(' ')
                    ys_line = pars[2].strip().split(' ')
                    # Flux sigma
                    boot_sigma = float(flux_line[2]) - float(flux_line[1])
                    boot_x = 0.5 * abs(float(xs_line[2]) - float(xs_line[1]))
                    boot_y = 0.5 * abs(float(ys_line[2]) - float(ys_line[1]))
                    # Distance
                    radius = np.sqrt(float(xs_line[0]) ** 2. +
                                     float(ys_line[0]) ** 2.)
                    # Sigma position
                    boot_xy = np.sqrt(float(boot_x) ** 2. + float(boot_y) ** 2)
                    # Flux
                    flux = float(flux_line[0])
                    if flux < 0:
                        print "Negative flux"
                        continue
                    if flux > 10:
                        print "Too big flux"
                        continue
                    if boot_sigma > 100.*flux:
                        print "Too big boot_sigma for source {}".format(source)
                        continue
                    # RMS
                    rms = float(file_.split('_')[-1][:-4])
                    # params.append([flux, boot_sigma, radius, boot_xy, beam,
                    #                rms])
                    params_dict[source].update({n: [flux, boot_sigma,
                                                           radius, boot_xy,
                                                           beam, rms]})
            except IOError:
                print "Failed reading file {}".format(file_)
                continue
    return params_dict


ratios_n = dict()
for n in (1, 2, 3, 4):
    ratios_n[n] = list()
    params_dict = process_position_components(n)
    ratios_dict = dict()
    # base_dir = '/home/ilya/vlbi_errors/mojave_mod'
    base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mojave_mod_last'
    names = ['source', 'id', 's_ra', 's_dec']
    df = pd.read_table(os.path.join(base_dir, 'asu_chisq.tsv'), sep='\t',
                       header=None, names=names, dtype={key: str for key in names},
                       index_col=False)
    sources = params_dict.keys()
    for source in sources:
        print("Source {}".format(source))
        comps = params_dict[source].keys()
        ratios_dict[source] = dict()
        for comp in comps:
            print("Component {}".format(comp))
            a = df.loc[(df['source'] == source) & (df['id'] == ' {}'.format(comp))]
            if a.empty:
                continue
            sigma_chi = np.sqrt(float(a['s_ra'])**2 + float(a['s_dec'])**2)
            sigma_boot = params_dict[source][comp][3]
            ratios_dict[source][comp] = sigma_boot/sigma_chi

    for k, v in ratios_dict.items():
        if v:
            ratios_n[n].append(v.values()[0])


# ratio_1 = list()
# ratio_2 = list()
# ratio_3 = list()
# ratio_4 = list()
#
# for source in ratios_dict.keys():
#     for comp in ratios_dict[source].keys():
#         if comp=='1':
#             ratio_1.append(ratios_dict[source][comp])
#         elif comp=='2':
#             ratio_2.append(ratios_dict[source][comp])
#         elif comp=='3':
#             ratio_3.append(ratios_dict[source][comp])
#         elif comp=='4':
#             ratio_4.append(ratios_dict[source][comp])

# from knuth_hist import histogram
# hist_d, edges_d = histogram(ratios_n[1], normed=False)
# lower_d = np.resize(edges_d, len(edges_d) - 1)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)
# ax.bar(lower_d, hist_d, width=np.diff(lower_d)[0], linewidth=1, color='#4682b4')
# ax.set_xlabel(r"Bootstrap $\sigma_{position}$ to kinematic post-fit"
#               r" $\sigma_{position}$")
# ax.set_ylabel(r"N")
# fig.savefig("/home/ilya/Dropbox/article/evn2016/boot_chi2_position_sigma_histogram_3comp.png", bbox_inches='tight', dpi=200)
# fig.close()


label_size = 16
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
# -2 for last component at 9
fig, ax = plt.subplots(1, 1)
for i, lab, ls in zip((1, 2, 3, 4), ('1',
                    '2', '3', '4'), ('solid', 'dashed', 'dashdot', 'dotted')):
    ax.hist([a for a in ratios_n[i]], bins=15, histtype='step', stacked=True,
         fill=False, label=lab, lw=2, ls=ls, color='black', range=[0, 1])
# hist(fluxes[:,1]/(0.1*fluxes[:,0]), bins=15, range=[0, 10])
# hist(fluxes[:,1]/(0.1*fluxes[:,0]), bins=15, range=[0, 1.4])
ax.set_ylabel(r'N')
ax.set_xlabel(r"Bootstrap $\sigma_{position}$ to kinematic post-fit")
#               r" $\sigma_{position}$")
ax.legend(handlelength=2.225)
plt.show()
# plt.savefig('/home/ilya/Dropbox/papers/boot/new_pics/mod_new_pos_all_black.pdf',
#             bbox_inches='tight', format='pdf', dpi=1200)

