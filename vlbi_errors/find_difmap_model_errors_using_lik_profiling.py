import sys

import numpy as np
import os
import glob
from uv_data import UVData
from from_fits import create_clean_image_from_fits_file
from spydiff import (find_2D_position_errors_using_chi2, convert_2D_position_errors_to_ell_components,
                     import_difmap_model, find_image_std, find_bbox, find_size_errors_using_chi2,
                     find_flux_errors_using_chi2, CLEAN_difmap, export_difmap_model,
                     time_average)
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.use('Agg')
from image import plot as iplot
import astropy.units as u


average_time_sec = 60.
account_gains = False

rad2mas = u.rad.to(u.mas)
# data_dir = "/home/ilya/data/silke/0735/43GHz"
data_dir = "/home/ilya/Downloads/3C454.3"
models_dir = data_dir
freq = 43E+09

# data_dir = "/home/ilya/Downloads/TXS0506"
# models_dir = data_dir
# freq = 15.3E+09

save_dir = os.path.join(models_dir, "save")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# First, remove exp-d files (we will create them again)
old_mdl_files = sorted(glob.glob(os.path.join(models_dir, "*_exp.mod")))
for mdl_file in old_mdl_files:
    try:
        os.unlink(mdl_file)
    except:
        pass

mdl_files = sorted(glob.glob(os.path.join(models_dir, "*.mod")))
mdl_files = [os.path.split(path)[-1] for path in mdl_files]
epochs = [fn.split(".")[0] for fn in mdl_files]

for mdl_file in mdl_files:
    comps = import_difmap_model(os.path.join(models_dir, mdl_file))
    base = mdl_file.split(".")[0]
    export_difmap_model(comps, os.path.join(models_dir, f"{base}_exp.mod"), freq)
# Update model files with re-written
mdl_files = sorted(glob.glob(os.path.join(models_dir, "*_exp.mod")))
mdl_files = [os.path.split(path)[-1] for path in mdl_files]

# ccfits_files = ['J0738+1742_Q_{}_mar_map.fits'.format(epoch) for epoch in epochs]
ccfits_files = ['J2253+1608_Q_{}_mar_map.fits'.format(epoch) for epoch in epochs]
# ccfits_files = ['0506+056.u.{}.icn.fits.gz'.format(epoch) for epoch in epochs]

for ccfits_file, mdl_file, epoch in zip(ccfits_files, mdl_files, epochs):
    # Problematic epochs
    # if epoch not in ("2017_05_01",):
    # if epoch not in ("2011_08_23",):
    #     continue

    print(epoch, mdl_file, ccfits_file)
    # continue

    uvfits_file = 'J2253+1608_Q_{}_mar_vis.fits'.format(epoch)
    # uvfits_file = '0506+056.u.{}.uvf'.format(epoch)
    if average_time_sec is not None:
        time_average(os.path.join(data_dir, uvfits_file), os.path.join(data_dir, "tmp.uvf"), average_time_sec)
        uvfits_file = "tmp.uvf"
    uvdata = UVData(os.path.join(data_dir, uvfits_file), verify_option="warn")
    all_stokes = uvdata.stokes
    if "RR" in all_stokes and "LL" in all_stokes:
        stokes = "I"
    else:
        if "RR" in all_stokes:
            stokes = "RR"
        else:
            stokes = "LL"
    print("Stokes parameter: ", stokes)

    # Find errors if they are not calculated
    if not os.path.exists(os.path.join(save_dir, "errors_{}.pkl".format(epoch))):
        if average_time_sec is not None:
            delta_t_sec = average_time_sec
        else:
            delta_t_sec = 30
        errors = find_2D_position_errors_using_chi2(os.path.join(models_dir, mdl_file),
                                                    os.path.join(data_dir, uvfits_file),
                                                    stokes=stokes,
                                                    show_difmap_output=False,
                                                    delta_t_sec=delta_t_sec,
                                                    use_gain_dofs=account_gains, freq=freq,
                                                    nmodelfit_cycle=50)
        with open(os.path.join(save_dir, "errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(errors, fo)
    # Or just load already calculated
    else:
        with open(os.path.join(save_dir, "errors_{}.pkl".format(epoch)), "rb") as fo:
            errors = pickle.load(fo)

    # Make dummy elliptical components for plotting errors
    error_comps = convert_2D_position_errors_to_ell_components(os.path.join(models_dir, mdl_file),
                                                               errors, include_shfit=False, filter_by_r=True)

    comps = import_difmap_model(os.path.join(models_dir, mdl_file))


    # Original image
    try:
        ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, ccfits_file))
        pixsize_mas = np.round(abs(ccimage.pixsize[0])*rad2mas, 2)
        npixels = ccimage.imsize[0]
    # Use previous pixsize_mas and npixels values
    except TypeError:
        pass

    # CLEAN
    CLEAN_difmap(os.path.join(data_dir, uvfits_file), stokes, (npixels, pixsize_mas),
                 os.path.join(save_dir, "{}_cc.fits".format(epoch)), restore_beam=None,
                 boxfile=None, working_dir=save_dir, uvrange=None,
                 box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=1.0,
                 remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                 noise_to_use="F", shift=None)

    # New image
    ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, "{}_cc.fits".format(epoch)))
    # In rad
    beam = ccimage.beam
    beam_deg = (beam[0], beam[1], np.rad2deg(beam[2]))
    beam_size = np.sqrt(beam[0]*beam[1])
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*pixsize_mas**2)
    std = find_image_std(ccimage.image, beam_npixels=npixels_beam)
    blc, trc = find_bbox(ccimage.image, level=3*std, min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=4*npixels_beam, delta=10)
    fig, axes = plt.subplots(1, 1, figsize=(10, 15))
    fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*std,
                blc=blc, trc=trc, beam=beam_deg, show_beam=True, show=False,
                close=True, contour_color='black',
                plot_colorbar=False, components=comps, components_errors=error_comps,
                outfile="{}_pos_errors2D".format(epoch), outdir=save_dir, fig=fig)

    if not os.path.exists(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch))):
        if average_time_sec is not None:
            delta_t_sec = average_time_sec
        else:
            delta_t_sec = 30
        size_errors = find_size_errors_using_chi2(os.path.join(models_dir, mdl_file),
                                                  os.path.join(data_dir, uvfits_file),
                                                  delta_t_sec=delta_t_sec,
                                                  show_difmap_output=False,
                                                  use_selfcal=account_gains, freq=freq,
                                                  nmodelfit_cycle=50)
        with open(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(size_errors, fo)
    else:
        with open(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch)), "rb") as fo:
            size_errors = pickle.load(fo)

    if not os.path.exists(os.path.join(save_dir, "flux_errors_{}.pkl".format(epoch))):
        if average_time_sec is not None:
            delta_t_sec = average_time_sec
        else:
            delta_t_sec = 30
        flux_errors = find_flux_errors_using_chi2(os.path.join(models_dir, mdl_file),
                                                  os.path.join(data_dir, uvfits_file),
                                                  delta_t_sec=delta_t_sec,
                                                  show_difmap_output=False,
                                                  use_selfcal=account_gains, freq=freq,
                                                  nmodelfit_cycle=50)
        with open(os.path.join(save_dir, "flux_errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(flux_errors, fo)
    else:
        with open(os.path.join(save_dir, "flux_errors_{}.pkl".format(epoch)), "rb") as fo:
            flux_errors = pickle.load(fo)

    # Put uncertainties in a distinct files
    degree_to_rad = u.deg.to(u.rad)
    rad_to_deg = u.rad.to(u.deg)
    with open(os.path.join(save_dir, "{}_pos_errors.txt".format(epoch)), "w") as fo:
        fo.write("# r theta r_err\n")
        for i, comp in enumerate(comps):
            error_comp = error_comps[i]
            try:
                flux, x, y, bmaj = comp.p
            # Size is fixed
            except ValueError:
                flux, x, y, bmaj = comp._p
            # mas
            r = np.hypot(x, y)
            # rad
            theta = np.arctan2(-x, -y)
            theta *= rad_to_deg
            # Here BMAJ - full width, e.g. 2*a
            pos_error = 0.5*(1+error_comp.p[4])*error_comp.p[3]/2.
            # if pos_error < 0.1*beam_size:
            #     pos_error = 0.1*beam_size
            fo.write(f"{r} {theta} {pos_error}\n")

    with open(os.path.join(save_dir, "{}_size_errors.txt".format(epoch)), "w") as fo:
        fo.write("# bmaj bmaj_err\n")
        for i, comp in enumerate(comps):

            try:
                flux, x, y, bmaj = comp.p
                size_err_low = size_errors[i][0][0]
                if size_err_low < 0.0005:
                    size_err_low = size_err_up
                size_err_up = size_errors[i][1][0]
                if size_err_up < 0.0005:
                    size_err_up = size_err_low
                size_err = 0.5*(size_err_low+size_err_up)
                fo.write(f"{bmaj} {size_err}\n")
            # Size is fixed
            except ValueError:
                flux, x, y, bmaj = comp._p
                fo.write(f"{bmaj} {0}\n")

    with open(os.path.join(save_dir, "{}_flux_errors.txt".format(epoch)), "w") as fo:
        fo.write("# flux flux_err\n")
        for i, comp in enumerate(comps):
            try:
                flux, x, y, bmaj = comp.p
            # Size is fixed
            except ValueError:
                flux, x, y, bmaj = comp._p

            flux_err_low = flux_errors[i][0][0]
            flux_err_up = flux_errors[i][1][0]
            print(f"Flux = {flux}, flux error low = {flux_err_low}, flux error up = {flux_err_up}")
            flux_err = 0.5*(flux_err_low + flux_err_up)
            fo.write(f"{flux} {flux_err}\n")

    # sys.exit(0)





    #
    # stat_dict = find_stat_of_difmap_model(os.path.join(data_dir, mdl_file),
    #                                       os.path.join(data_dir, uvfits_file),
    #                                       stokes, data_dir, nmodelfit=100, use_pselfcal=True,
    #                                       out_dfm_model="selfcaled.mdl")
    # selfcaled_comps = import_difmap_model(os.path.join(data_dir, "selfcaled.mdl"))
    # fig, axes = plt.subplots(1, 1, figsize=(10, 15))
    # fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*std,
    #             blc=blc, trc=trc, beam=beam, show_beam=True, show=False,
    #             close=True, contour_color='black',
    #             plot_colorbar=False, components=selfcaled_comps,
    #             outfile="{}_selfcaled_model".format(epoch), outdir=data_dir, fig=fig)
    #
    # sys.exit(0)