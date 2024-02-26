import os
import shutil

import numpy as np
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.stats import percentileofscore
from spydiff import (residuals_from_model, export_difmap_model, append_component_to_difmap_model,
                     modelfit_difmap, clean_difmap, import_difmap_model, find_bbox,
                     find_image_std, find_far_noise, components_info, convert_pixel_from_Jy_to_K,
                     time_average, get_contour_mask, find_nw_beam)
from from_fits import create_image_from_fits_file, create_clean_image_from_fits_file
from utils import infer_gaussian
from components import CGComponent, EGComponent
from image import plot as iplot
from image_ops import rms_image
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import mad_std
from astropy.stats import gaussian_sigma_to_fwhm

vint = np.vectorize(int)


def divide_img_blocks(img, n_blocks=(7, 7)):
    """
    https://stackoverflow.com/a/67197232
    :param img:
    :param n_blocks:
    :return:

    :note:
    This also works but only for 8x8
    >>> from skimage.util.shape import view_as_blocks
    >>> a = view_as_blocks(image, block_shape=(8, 8))
    """
    horizontal = np.array_split(img, n_blocks[0])
    splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
    return np.asarray(splitted_img, dtype=np.ndarray).reshape(n_blocks)


def get_image_entropy(image):
    npixels = image.size
    rms = mad_std(image)
    flux_bins = np.linspace(-5, 5, 11)*rms
    blocks = divide_img_blocks(image)
    result = 0
    for block in blocks.flat:
        for flux_bin in range(10):
            flux_low = flux_bins[flux_bin]
            flux_high = flux_bins[flux_bin + 1]
            # print("Counting flux between {:.1f} and {:.1f} mJy".format(1000*flux_low, 1000*flux_high))
            npixels_in_current_bin = 1 + np.count_nonzero(np.logical_and(block < flux_high, block > flux_low))
            # print(f"#pixels in bin = {npixels_in_current_bin}")
            p = npixels_in_current_bin/npixels
            result -= p*np.log(p)
    return result


# TODO: Suggest components inside countours
def suggest_component_from_dirty_residuals(residual_image, comp_type="cg", max_width_mas=None,
                                           show_suggestion=False, save_suggestion_to=None,
                                           show_histo=False, save_histo_to=None,
                                           original_image=None, original_rms=None,
                                           prev_comps=None,
                                           blc_2plot=None, trc_2plot=None,
                                           blc_2search=None, trc_2search=None,
                                           beam_2plot=None,
                                           contour_mask=None):

    if blc_2search is not None:
        x_slice = slice(blc_2search[1]-1, trc_2search[1], None)
        y_slice = slice(blc_2search[0]-1, trc_2search[0], None)
        mask = np.ones(residual_image.imsize)
        mask[x_slice, y_slice] = 0
    else:
        mask = None

    if contour_mask is not None:
        mask = contour_mask

    imsize = residual_image.imsize[0]
    # rms_original = mad_std(residual_image.image)
    kernel = Gaussian2DKernel(3, 3)
    image_convolved = convolve(residual_image.image, kernel, normalize_kernel=False)
    mas_in_pix = abs(np.round(residual_image.dx*u.rad.to(u.mas), 4))
    # FWHM
    amp, y, x, bmaj = infer_gaussian(image_convolved, mask=mask)
    amp_original = residual_image.image[int(y), int(x)]

    if show_histo or save_histo_to is not None:
        fig, axes = plt.subplots(1, 1)
        axes.hist(residual_image.image.ravel(), bins=100, alpha=0.5,
                  range=[-10*original_rms, 1.2*amp_original], label="residual image")
        axes.hist(original_image.image.ravel(), bins=100, alpha=0.5,
                  range=[-10*original_rms, 1.2*amp_original], label="original image")
        perc = percentileofscore(residual_image.image.ravel(), amp_original)
        perc_orig = percentileofscore(original_image.image.ravel(), amp_original)

        axes.axvline(amp_original, lw=2, label="{:.3f}, orig : {:.3f}".format(perc, perc_orig))
        plt.legend()
        if save_histo_to is not None:
            fig.savefig(save_histo_to, bbox_inches="tight")
        if show_histo:
            plt.show()
        plt.close()

    bmaj = np.sqrt(bmaj**2 - (3*gaussian_sigma_to_fwhm)**2)
    bmaj *= mas_in_pix
    if max_width_mas is not None:
        if bmaj > max_width_mas:
            bmaj = max_width_mas
    rms = mad_std(image_convolved)
    if blc_2plot is None or trc_2plot is None:
        blc_2plot, trc_2plot = find_bbox(image_convolved, 3*rms, min_maxintensity_mjyperbeam=30*rms,
                                          min_area_pix=10*100, delta=10)

    print("Suggested amp = {:.3f} Jy".format(amp))
    print("Suggested bmaj = {:.2f} mas".format(bmaj))

    x = mas_in_pix*(x-imsize/2)*np.sign(residual_image.dx)
    y = mas_in_pix*(y-imsize/2)*np.sign(residual_image.dy)
    print("Suggested x = {:.2f} mas".format(x))
    print("Suggested y = {:.2f} mas".format(y))
    if comp_type == 'cg':
        comp = CGComponent(10*amp, x, y, bmaj)
    elif comp_type == 'eg':
        comp = EGComponent(10*amp, x, y, bmaj, 1.0, 0.0)
    else:
        raise Exception

    if show_suggestion or save_suggestion_to is not None:
        if prev_comps is None:
            first_show = True
        else:
            first_show = False
        if beam_2plot is not None:
            show_beam = True
        else:
            show_beam = False
        fig = iplot(residual_image.image, x=residual_image.x, y=residual_image.y,
                    min_abs_level=3*rms, plot_colorbar=False,
                    blc=blc_2plot, trc=trc_2plot, components=[comp],
                    close=False, show=first_show)
        fig = iplot(original_image.image, x=original_image.x, y=original_image.y,
                    abs_levels=[3*original_rms], contour_color="C1", contour_linewidth=1.0,
                    blc=blc_2plot, trc=trc_2plot, components=prev_comps, components_facecolor="gray",
                    close=False, plot_colorbar=False, show=True,
                    fig=fig, show_beam=show_beam, beam=beam_2plot)
        ax = fig.axes[0]
        ax.invert_xaxis()
        if save_suggestion_to is not None:
            fig.savefig(save_suggestion_to, bbox_inches="tight")
        if show_suggestion:
            plt.show()
        plt.close()

    return comp


def plot_clean_image_and_components(image, comps, outname=None, ra_range=None,
                                    dec_range=None, n_rms_level=3.0,
                                    n_rms_size=1.0, blc=None, trc=None,
                                    rms=None):
    """
    :param image:
        Instance of ``CleanImage`` class.
    :param comps:
        Iterable of ``Components`` instances.
    :return:
        ``Figure`` object.
    """
    beam = image.beam
    beam = (beam[0], beam[1], np.rad2deg(beam[2]))
    if rms is None:
        rms = rms_image(image)
    if blc is None or trc is None:
        blc, trc = find_bbox(image.image, n_rms_size*rms, min_maxintensity_mjyperbeam=30*rms,
                             min_area_pix=10*100, delta=10)
    try:
        fig = iplot(image.image, x=image.x, y=image.y,
                    min_abs_level=n_rms_level*rms,
                    beam=beam, show_beam=True, blc=blc, trc=trc, components=comps,
                    close=True, colorbar_label="Jy/beam", ra_range=ra_range,
                    dec_range=dec_range, show=False,
                    plot_colorbar=False)
        if outname is not None:
            fig.savefig(outname, bbox_inches='tight', dpi=300)
    except:
        fig = None
    return fig


def filter_difmap_models(difmap_model_files, core_type):
    if not core_type in ("cg", "eg"):
        raise Exception
    core_fluxes = list()
    core_sizes = list()
    core_elongation = list()
    for difmap_model_file in difmap_model_files:
        components = import_difmap_model(difmap_model_file)
        n_comp = len(components)


def get_Tb_diagnostics(difmap_model_file, uvfits_file, residual_file, dmap_size, freq_ghz, working_dir,
                       save_histo_file=None):
    df = components_info(uvfits_file, difmap_model_file, dmap_size, out_path=working_dir, freq_ghz=freq_ghz)
    residuals = pf.getdata(residual_file).squeeze()
    coeff = convert_pixel_from_Jy_to_K(dmap_size[1], freq_ghz)
    percentiles = [percentileofscore(residuals.ravel()*coeff, l) for l in df.Tb]
    print("Tb percentiles : ", percentiles)
    fig, axes = plt.subplots(1, 1)
    colors = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5']
    axes.hist(coeff*residuals.ravel(), alpha=0.3, bins=100)
    for i, Tb in enumerate(df.Tb):
        axes.axvline(Tb, lw=1, label=f"comp # {i+1}", color=colors[i])
    axes.set_xlabel("log Tb, K")
    axes.set_xscale("log")
    plt.legend()
    if save_histo_file is not None:
        fig.savefig(save_histo_file, bbox_inches="tight")
    plt.close()


def automodeller_v2(basename, uvfits_file, working_dir, mapsize, n_max, comp_type, freq_ghz, path_to_script,
                    show_difmap_output_modelfit=False, show_suggested_components=True, show_histo=True):
    freq_hz = 1e+09*freq_ghz
    residuals_file = os.path.join(working_dir, "residuals.fits")
    model_difmap_file = None

    # bmin, bmaj, bpa = find_nw_beam(uvfits_file, mapsize, working_dir=working_dir)

    clean_difmap(fname=uvfits_file, path=working_dir,
                 outfname="clean_image.fits", outpath=working_dir, stokes="i",
                 mapsize_clean=mapsize, path_to_script=path_to_script,
                 show_difmap_output=True, dmap="clean_residuals.fits")
    original_image = create_clean_image_from_fits_file(os.path.join(working_dir, "clean_image.fits"))
    beam = original_image.beam
    beam = (beam[0], beam[1], np.rad2deg(beam[2]))
    # original_rms = find_image_std(original_image.image, 50, 5*50)
    farnoise, wtnoise, vnoise = find_far_noise(uvfits_file, mapsize)
    # original_rms = wtnoise
    original_rms = mad_std(original_image.image)
    blc, trc = find_bbox(original_image.image, 3*original_rms,
                         min_maxintensity_jyperbeam=30*original_rms,
                         min_area_pix=10*100, delta=10)
    print("BLC, TRC = ", blc, trc)

    contour_mask = get_contour_mask(original_image.image, 3*original_rms,
                                    min_maxintensity_jyperbeam=30*original_rms,
                                    min_area_pix=10*100)

    entropies = list()
    rchisqs = list()
    counter = 0

    for i in range(n_max):
        if i > 0:
            uvtaper = (0.5, 600)
        else:
            uvtaper = None
        residuals_from_model(uvfits_file, model_difmap_file, residuals_file, mapsize, uvtaper=uvtaper)
        shutil.copy(residuals_file, os.path.join(working_dir, f"residuals_{counter}.fits"))
        residuals = create_image_from_fits_file(residuals_file)
        entropy = get_image_entropy(residuals.image)
        entropies.append(entropy)

        # Optionally previous components
        if i > 0:
            prev_comps = import_difmap_model(os.path.join(working_dir, 'fitted_{}.mdl'.format(counter)))
            used_component_type = "cg"
        else:
            prev_comps = None
            used_component_type = comp_type

        comp = suggest_component_from_dirty_residuals(residuals, max_width_mas=0.5,
                                                      comp_type=used_component_type,
                                                      show_suggestion=show_suggested_components,
                                                      save_suggestion_to=os.path.join(working_dir, f"{basename}_{counter+1}_suggestion.png"),
                                                      show_histo=show_histo,
                                                      save_histo_to=os.path.join(working_dir, f"{basename}_{counter+1}_histo.png"),
                                                      prev_comps=prev_comps,
                                                      blc_2plot=blc, trc_2plot=trc,
                                                      blc_2search=blc, trc_2search=trc,
                                                      original_image=original_image,
                                                      original_rms=original_rms,
                                                      beam_2plot=beam,
                                                      contour_mask=contour_mask)

        if counter == 0:
            export_difmap_model([comp],
                                os.path.join(working_dir, 'pre_fitted_{}.mdl'.format(counter+1)),
                                freq_hz)
        else:
            shutil.copy(os.path.join(working_dir, 'fitted_{}.mdl'.format(counter)),
                        os.path.join(working_dir, 'pre_fitted_{}.mdl'.format(counter+1)))
            append_component_to_difmap_model(comp, os.path.join(working_dir, 'pre_fitted_{}.mdl'.format(counter+1)),
                                             freq_hz)

        # FIXME: SOmetimes it returns nan (singular matrix)
        rchisq = modelfit_difmap(uvfits_file, 'pre_fitted_{}.mdl'.format(counter+1),
                                 'fitted_{}.mdl'.format(counter+1),
                                 path=working_dir, mdl_path=working_dir,
                                 out_path=working_dir, dmap_name=f"residuals_{counter+1}.fits", niter=100,
                                 stokes="i", save_dirty_residuals_map=True, dmap_size=mapsize,
                                 show_difmap_output=show_difmap_output_modelfit)
        rchisqs.append(rchisq)
        model_difmap_file = os.path.join(working_dir, 'fitted_{}.mdl'.format(counter+1))
        shutil.copy(os.path.join(working_dir, 'fitted_{}.mdl'.format(counter+1)),
                    os.path.join(working_dir, '{}_fitted_{}.mdl'.format(basename, counter+1)))
        comps = import_difmap_model(model_difmap_file)

        plot_clean_image_and_components(original_image, comps, os.path.join(working_dir, f"image_{basename}_{counter+1}.png"),
                                        n_rms_size=3, rms=original_rms, blc=blc, trc=trc)
        get_Tb_diagnostics(model_difmap_file, uvfits_file, os.path.join(working_dir, f"residuals_{counter+1}.fits"),
                           mapsize, freq_ghz, working_dir, save_histo_file=os.path.join(working_dir, f"Tb_{counter+1}.png"))
        counter += 1

    fig, axes = plt.subplots(1, 1)
    axes.plot(np.arange(1, n_max+1), entropies)
    axes.set_xlim([2, n_max])
    axes.set_yscale('log')
    axes.set_xlabel("# comp")
    axes.set_ylabel("log Entropy")
    fig.savefig(os.path.join(working_dir, f"entropy_vs_ncomponents_{basename}.png"), bbox_inches="tight")

    fig, axes = plt.subplots(1, 1)
    axes.plot(np.arange(2, n_max+1), -np.diff(rchisqs))
    axes.axhline(0, lw=1, color="k")
    axes.set_ylim([-0.1, 1])
    # axes.set_yscale('log')
    # Where rchisq decrease just before the next time it increases
    try:
        n_best = np.arange(2, n_max+1)[np.diff(rchisqs) >= 0][0] - 1
        axes.axvline(n_best, label="best")
        plt.legend()
    except IndexError:
        pass
    axes.set_xlabel("# comp")
    axes.set_ylabel("rchisq decrease")
    # plt.show()
    fig.savefig(os.path.join(working_dir, f"rchisq_vs_ncomponents_{basename}.png"), bbox_inches="tight")

    return entropies, rchisqs


if __name__ == "__main__":
    # uvfits_file = "/home/ilya/Downloads/0851+202/Q/OJ287AUG10.UVP"
    # uvfits_file = "/home/ilya/Downloads/0851+202/Q/OJ287APR18.UVP"
    # uvfits_file = "/home/ilya/Downloads/0851+202/Q/OJ287MAY18.UVP"
    # working_dir = "/home/ilya/Downloads/0851+202/Q/results/automodelling"

    import glob
    import sys
    import astropy.io.fits as pf
    from uv_data import UVData

    source = "0851+202"
    band = "Q"
    sort_uvfits_files_by_epoch = True
    check_frequency_in_files = False

    # path_to_script = "/home/ilya/github/boston_stacks/difmap_scripts/script_clean_rms"
    path_to_script = "/home/ilya/github/boston_stacks/difmap_scripts/script_clean_rms_nooverclean"
    freq_ghz = 43.
    t_ave_sec = 0
    n_max = 10
    n_first_uvfits_files = None
    comp_type = "eg"
    show_suggested_components = False
    show_histo = False
    show_difmap_output_modelfit = True
    # If directory for some epoch exists - skip this epoch
    skip_done_epochs = False
    # If ``skip_done_epochs = False``, then remove any files from epoch directory
    remove_old_model_stuff = True
    # Bad epochs list: skip then anyway
    # 2008-11-16 - CLEAN gathers thousands of Jys.
    # 2012-05-26 - Need > 10k iterations to make it good!!!
    # 2013-11-18 - Long time to clean (singular matrix)
    # 2014-02-24 - Core - not ellipse, EG becomes jet component
    # 2014-12-05 - modelfit w singular matrix
    # 2019-01-10 - infinite modelfit?
    bad_epochs_list = ("2008-11-16",)
    only_epochs_list = ("2018-10-15",)
    # only_epochs_list = ("2007-08-06",)

    if band == "Q":
        mapsize = (512, 0.03)
    elif band == "W":
        mapsize = (512, 0.01)
    else:
        raise Exception("Band must be Q or W!")

    uvfits_dir = f"/home/ilya/Downloads/{source}/{band}"

    save_dir = os.path.join(uvfits_dir, "modelfit_results")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(f"Will put results in {save_dir}")

    uvfits_files = glob.glob(os.path.join(uvfits_dir, "*.UVP"))+glob.glob(os.path.join(uvfits_dir, "*.UVPN"))
    if n_first_uvfits_files is not None:
        uvfits_files = uvfits_files[:n_first_uvfits_files]
    for uvfits_file in uvfits_files:
        if not os.path.isfile(uvfits_file):
            raise Exception(f"{uvfits_file} is not a file!")

    print("Creating epoch - UVFITS dictionary...")
    uvfits_epoch_dict = dict()
    for uvfits_file in uvfits_files:
        fn = os.path.split(uvfits_file)[-1]
        try:
            base, epoch, _ = fn.split(".")
        except:
            epoch = pf.getheader(uvfits_file)["DATE-OBS"]
            base = f"{source}{band}"
        uvfits_epoch_dict[epoch] = uvfits_file

    if sort_uvfits_files_by_epoch:
        uvfits_epoch_dict = dict(sorted(uvfits_epoch_dict.items()))
    uvfits_files = list()
    for epoch in uvfits_epoch_dict:
        uvfits_files.append(uvfits_epoch_dict[epoch])

    print(f"Found {len(uvfits_files)} in directory {uvfits_dir}: ")
    for uvfits_file in uvfits_files:
        _, fn = os.path.split(uvfits_file)
        print(fn)

    # Check frequency
    if check_frequency_in_files:
        for uvfits_file in uvfits_files:
            print("Finding frequency for ", uvfits_file)
            uvdata = UVData(uvfits_file, verify_option="ignore")
            freq_ghz = uvdata.frequency/1E+09
            print("Freq [GHz] = ", freq_ghz)
            if band == "Q":
                assert np.floor(freq_ghz) == 43.0 or np.floor(freq_ghz) == 42.0
            elif band == "W":
                assert np.floor(freq_ghz) == 86.0

    for uvfits_file in uvfits_files:
        fn = os.path.split(uvfits_file)[-1]
        try:
            base, epoch, _ = fn.split(".")
        except:
            epoch = pf.getheader(uvfits_file)["DATE-OBS"]
            base = f"{source}{band}"

        if epoch in bad_epochs_list:
            continue

        if only_epochs_list:
            if epoch not in only_epochs_list:
                continue

        print(f"Epoch {epoch} : ")

        working_dir = os.path.join(save_dir, f"{epoch}")
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
            print(f"Created working directory for epoch {epoch} : {working_dir}")
        else:
            if skip_done_epochs:
                continue
            if remove_old_model_stuff:
                files = list()
                for end in ("*.png", "*.mdl", "*.fits", "*.uvf"):
                    remove_those_files = glob.glob(os.path.join(working_dir, end))
                    for fn in remove_those_files:
                        print(f"Removing {fn}")
                        os.unlink(fn)

        if t_ave_sec > 0:
            local_uvfits_file = os.path.join(working_dir, "time_avg.uvf")
            time_average(uvfits_file, local_uvfits_file, time_sec=t_ave_sec)
        else:
            local_uvfits_file = uvfits_file

        basename = f"{base}_{epoch}"
        entropies, rchisqs = automodeller_v2(basename, local_uvfits_file, working_dir, mapsize, n_max, comp_type, freq_ghz, path_to_script,
                                             show_difmap_output_modelfit, show_suggested_components, show_histo)
