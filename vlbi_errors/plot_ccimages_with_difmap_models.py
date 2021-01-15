import pickle
import sys
import os
import glob
import numpy as np
from image import plot as iplot
from uv_data import UVData
from spydiff import (find_image_std, import_difmap_model, find_bbox,
                     find_stat_of_difmap_model,
                     find_2D_position_errors_using_chi2,
                     convert_2D_position_errors_to_ell_components)
from from_fits import create_clean_image_from_fits_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


pixsize_mas = 0.1
data_dir = "/home/ilya/data/Mkn501/difmap_models"
ccfits_files = sorted(glob.glob(os.path.join(data_dir, "*.icn.fits.gz")))
ccfits_files = [os.path.split(path)[-1] for path in ccfits_files]
print(ccfits_files)
epochs = [fn.split(".")[2] for fn in ccfits_files]
mdl_files = ["{}.mod".format(epoch) for epoch in epochs]

for ccfits_file, mdl_file, epoch in zip(ccfits_files, mdl_files, epochs):
    # Problematic epochs
    if epoch in ["1997_03_13", "2001_12_30", "2003_08_23", "2004_05_29"]:
    # if mdl_file != "1997_03_13.mod":
        continue

    print(mdl_file, ccfits_file)


    uvfits_file = "1652+398.u.{}.uvf".format(epoch)
    uvdata = UVData(os.path.join(data_dir, uvfits_file))
    all_stokes = uvdata.stokes
    if "I" in all_stokes:
        stokes = "I"
    else:
        if "RR" in all_stokes:
            stokes = "RR"
        else:
            stokes = "LL"
    print("Stokes parameter: ", stokes)

    # Find errors if they are not calculated
    if not os.path.exists(os.path.join(data_dir, "errors_{}.pkl".format(epoch))):
        errors = find_2D_position_errors_using_chi2(os.path.join(data_dir, mdl_file),
                                                    os.path.join(data_dir, uvfits_file),
                                                    stokes=stokes,
                                                    show_difmap_output=False)
        with open(os.path.join(data_dir, "errors_{}.pkl".format(epoch)), "wb") as fo:
            pickle.dump(errors, fo)
    # Or just load already calculated
    else:
        with open(os.path.join(data_dir, "errors_{}.pkl".format(epoch)), "rb") as fo:
            errors = pickle.load(fo)
    # Make dummy elliptical components for plotting errors
    error_comps = convert_2D_position_errors_to_ell_components(os.path.join(data_dir, mdl_file),
                                                               errors, include_shfit=False)

    comps = import_difmap_model(os.path.join(data_dir, mdl_file))
    ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, ccfits_file))
    beam = ccimage.beam
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*pixsize_mas**2)
    std = find_image_std(ccimage.image, beam_npixels=npixels_beam)
    blc, trc = find_bbox(ccimage.image, level=4*std, min_maxintensity_mjyperbeam=6*std,
                         min_area_pix=4*npixels_beam, delta=10)
    fig, axes = plt.subplots(1, 1, figsize=(10, 15))
    fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*std,
                blc=blc, trc=trc, beam=beam, show_beam=True, show=False,
                close=True, contour_color='black',
                plot_colorbar=False, components=comps, components_errors=error_comps,
                outfile="{}_original_model_errors2D".format(epoch), outdir=data_dir, fig=fig)


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