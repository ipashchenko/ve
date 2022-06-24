import sys

import numpy as np
import os
import glob
from uv_data import UVData
from from_fits import create_clean_image_from_fits_file
from spydiff import (find_2D_position_errors_using_chi2, convert_2D_position_errors_to_ell_components,
                     import_difmap_model, find_image_std, find_bbox, find_size_errors_using_chi2,
                     find_flux_errors_using_chi2, CLEAN_difmap)
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.use('Agg')
from image import plot as iplot
import astropy.units as u

# import pickle
# data_dir = "/home/ilya/data/Mkn501/difmap_models"
# pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
# for pkl_file in pkl_files:
#     epoch = os.path.split(pkl_file)[-1][7:17]
#     print("Processing epoch ", epoch)
#     mdl_file = os.path.join(data_dir, "{}.mod".format(epoch))
#     with open(pkl_file, "rb") as fo:
#         errors = pickle.load(fo)
#     errors_comps = convert_2D_position_errors_to_ell_components(os.path.join(data_dir, mdl_file),
#                                                                 errors, include_shfit=False)
#     pos_errors = [0.5*errors_comps[i].p[3]*(1+errors_comps[i].p[4]) for i in range(len(errors_comps))]
#     bpas = [np.rad2ded(errors_comps[i].p[5]) for i in range(len(errors_comps))]
#     bmajs = [errors_comps[i].p[3] for i in range(len(errors_comps))]
#     es = [errors_comps[i].p[4] for i in range(len(errors_comps))]
#     with open(os.path.join(data_dir, "{}_posistion_ellipse_errors_chi2_errors.txt".format(epoch)), "w") as fo:
#         for err in pos_errors:
#             fo.write("{}\n".format(err))


# ============

rad2mas = u.rad.to(u.mas)
data_dir = "/home/ilya/Downloads/Mrk501_Q_uvfits"
save_dir = os.path.join(data_dir, "save")
mdl_files = sorted(glob.glob(os.path.join(data_dir, "*.mod")))
mdl_files = [os.path.split(path)[-1] for path in mdl_files]
epochs = [fn.split(".")[0] for fn in mdl_files]
ccfits_files = ['J1653+3945_Q_{}_mar_map.fits'.format(epoch) for epoch in epochs]

for ccfits_file, mdl_file, epoch in zip(ccfits_files, mdl_files, epochs):
    # Problematic epochs
    if epoch in ["2011_09_24"]:
        continue

    print(mdl_file, ccfits_file)

    uvfits_file = 'J1653+3945_Q_{}_mar_vis.fits'.format(epoch)
    uvdata = UVData(os.path.join(data_dir, uvfits_file))
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
        errors = find_2D_position_errors_using_chi2(os.path.join(data_dir, mdl_file),
                                                    os.path.join(data_dir, uvfits_file),
                                                    stokes=stokes,
                                                    show_difmap_output=False,
                                                    delta_t_sec=30,
                                                    use_gain_dofs=False, freq=43E+09,
                                                    nmodelfit_cycle=50)
        with open(os.path.join(save_dir, "errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(errors, fo)
    # Or just load already calculated
    else:
        with open(os.path.join(save_dir, "errors_{}.pkl".format(epoch)), "rb") as fo:
            errors = pickle.load(fo)

    # Make dummy elliptical components for plotting errors
    error_comps = convert_2D_position_errors_to_ell_components(os.path.join(data_dir, mdl_file),
                                                               errors, include_shfit=False)

    comps = import_difmap_model(os.path.join(data_dir, mdl_file))


    # Original image
    ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, ccfits_file))
    pixsize_mas = np.round(abs(ccimage.pixsize[0])*rad2mas, 2)
    npixels = ccimage.imsize[0]

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
        size_errors = find_size_errors_using_chi2(os.path.join(data_dir, mdl_file),
                                                  os.path.join(data_dir, uvfits_file),
                                                  show_difmap_output=False,
                                                  use_selfcal=False, freq=43E+09,
                                                  nmodelfit_cycle=50)
        with open(os.path.join(save_dir, "size_errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(size_errors, fo)

    if not os.path.exists(os.path.join(save_dir, "flux_errors_{}.pkl".format(epoch))):
        flux_errors = find_flux_errors_using_chi2(os.path.join(data_dir, mdl_file),
                                                  os.path.join(data_dir, uvfits_file),
                                                  show_difmap_output=False,
                                                  use_selfcal=False, freq=43.0E+09,
                                                  nmodelfit_cycle=50)
        with open(os.path.join(save_dir, "flux_errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(flux_errors, fo)

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

    break