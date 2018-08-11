import os
import sys
import copy
import numpy as np
import shutil
import matplotlib.pyplot as plt

from utils import hdi_of_mcmc, bc_endpoint, hdi_of_arrays
from conf_bands import create_sim_conf_band

sys.path.insert(0, '/home/ilya/github/ve')
from uv_data import UVData
from spydiff import clean_difmap
from from_fits import (create_clean_image_from_fits_file,
                       create_model_from_fits_file)
from bootstrap import CleanBootstrap
from image_ops import (rms_image, spix_map, pol_mask, spix_mask,
                       rms_image_shifted, pang_map, rotm_map)
from image import find_bbox, Image
from image import plot as iplot


bands = ("x", "y", "j", "u")
freqs = np.array([8.104458750, 8.424458750, 12.111458750, 15.353458750])
freqs *= 10**9
path_to_script = "/home/ilya/github/ve/difmap/final_clean_nw"

# Directory with resulting data sets
data_dir = "/home/ilya/data/revision"


def clean_original_data(uvdata_dict, data_dir, beam=None, plot=False,
                        mapsize_clean=(512, 0.1), outfname_postfix=None):
    if not isinstance(mapsize_clean, dict):
        assert len(mapsize_clean) == 2
        mapsize_clean = {band: mapsize_clean for band in bands}
    for band in bands:
        print("Band - {}".format(band))
        for stokes in ('I', 'Q', 'U'):
            print("Stokes - {}".format(stokes))
            if outfname_postfix is None:
                outfname = "cc_{}_{}.fits".format(band, stokes)
            else:
                outfname = "cc_{}_{}_{}.fits".format(band, stokes, outfname_postfix)
            print("Cleaning {} to {}".format(uvdata_dict[band], outfname))
            clean_difmap(fname=uvdata_dict[band], outfname=outfname,
                         stokes=stokes.lower(), path=data_dir, outpath=data_dir,
                         mapsize_clean=mapsize_clean[band],
                         path_to_script=path_to_script,
                         show_difmap_output=False, beam_restore=beam)

        # Rarely need this one
        if plot:
            if outfname_postfix is None:
                outfname = "cc_{}_{}.fits".format(band, "I")
            else:
                outfname = "cc_{}_{}_{}.fits".format(band, "I", outfname_postfix)
            ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, outfname))

            beam = ccimage.beam
            rms = rms_image(ccimage)
            blc, trc = find_bbox(ccimage.image, 1.0*rms, 10)
            fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y,
                        min_abs_level=2.0*rms,
                        beam=beam, show_beam=True, blc=blc, trc=trc,
                        close=False, colorbar_label="Jy/beam", show=True)
            if outfname_postfix is None:
                outfname = "cc_{}.png".format(band)
            else:
                outfname = "cc_{}_{}.png".format(band, outfname_postfix)
            fig.savefig(os.path.join(data_dir, outfname))


def process_mf(uvdata_dict, beam, data_dir, path_to_script, clean_after=True,
               rms_cs_dict=None, mapsize_clean=(512, 0.1)):
    images_dict = dict()
    print(" === CLEANing each band and Stokes ===")
    clean_original_data(uvdata_dict, data_dir, beam,
                        mapsize_clean=mapsize_clean)

    for band in bands:
        images_dict[band] = dict()
        for stokes in ("I", "Q", "U"):
            ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                     "cc_{}_{}.fits".format(band, stokes)))
            images_dict[band].update({stokes: ccimage})

    # Cacluate RMS for each band and Stokes
    print(" === Cacluate RMS for each band and Stokes ===")
    if rms_cs_dict is None:
        rms_cs_dict = dict()
        for band in bands:
            rms_cs_dict[band] = {stokes: rms_image_shifted(os.path.join(data_dir, uvdata_dict[band]),
                                                                stokes=stokes,
                                                                image=images_dict[band][stokes],
                                                                path_to_script=path_to_script) for stokes in ("I", "Q", "U")}
    for band in bands:
        for stokes in ("I", "Q", "U"):
            print("rms = {}".format(rms_cs_dict[band][stokes]))

    # Find mask for "I" for each band and combine them into single mask
    print("Calculating masks for I at each band and combining them")
    spix_mask_image = spix_mask({band: images_dict[band]["I"] for band in bands},
                                {band: rms_cs_dict[band]["I"] for band in bands},
                                n_sigma=3, path_to_script=path_to_script)

    # Find mask for "PPOL" for each band and combine them into single mask
    print("Calculating masks for PPOL at each band and combining them")
    ppol_mask_image = dict()
    for band in bands:
        ppol_mask_image[band] = pol_mask({stokes: images_dict[band][stokes] for stokes in ("I", "Q", "U")},
                                         {stokes: rms_cs_dict[band][stokes] for stokes in ("I", "Q", "U")},
                                         n_sigma=2, path_to_script=path_to_script)
    ppol_mask_image = np.logical_or.reduce([ppol_mask_image[band] for band in bands])

    spix_image, sigma_spix_image, chisq_spix_image =\
        spix_map(freqs, [images_dict[band]["I"].image for band in bands],
                 mask=spix_mask_image)

    print("Calculating PANG and it's error for each band")
    pang_images = dict()
    sigma_pang_images = dict()
    for band in bands:
        pang_images[band] = pang_map(images_dict[band]["Q"].image,
                                     images_dict[band]["U"].image,
                                     mask=ppol_mask_image)
        sigma_pang_images[band] = np.hypot(images_dict[band]["Q"].image*1.8*rms_cs_dict[band]["U"],
                                           images_dict[band]["U"].image*1.8*rms_cs_dict[band]["Q"])
        sigma_pang_images[band] = sigma_pang_images[band]/(2.*(images_dict[band]["Q"].image**2.+
                                                               images_dict[band]["U"].image**2.))

    print("Calculating ROTM image")
    rotm_image, sigma_rotm_image, chisq_rotm_image = rotm_map(freqs,
                                                              [pang_images[band] for band in bands],
                                                              [sigma_pang_images[band] for band in bands],
                                                              mask=ppol_mask_image)

    if clean_after:
        print("Removing maps")
        for band in bands:
            for stokes in ("I", "Q", "U"):
                os.unlink(os.path.join(data_dir, "cc_{}_{}.fits".format(band, stokes)))

    result = {"ROTM": {"value": rotm_image, "sigma": sigma_rotm_image,
                       "chisq": chisq_rotm_image},
              "SPIX": {"value": spix_image, "sigma": sigma_spix_image,
                       "chisq": chisq_spix_image},
              "RMS": rms_cs_dict}

    return result


def create_bootstrap_sample(uvdata_dict, ccfits_dict, data_dir, n_boot=10):
    """
    Create ``n_boot`` bootstrap replications of the original UV-data with
    given several Stokes CC-models for each band.

    :param uvdata_dict:
        Dictionary with keys - bands and values - files with uv-data.
    :param ccfits_dict:
        Dictionary with keys - bands, stokes and values - files with CC-fits
        files with models for given band and Stokes.

    Creates ``n_boot`` UV-data files for each band with names
    ``boot_band_i.uvf`` in ``data_dir``.
    """
    print("Bootstrap uv-data with CLEAN-models...")
    for band, uv_fits in uvdata_dict.items():
        uvdata = UVData(os.path.join(data_dir, uv_fits))
        # print("Band = {}".format(band))
        models = list()
        for stokes, cc_fits in ccfits_dict[band].items():
            # print("Stokes = {}".format(stokes))
            ccmodel = create_model_from_fits_file(os.path.join(data_dir,
                                                               cc_fits))
            models.append(ccmodel)

        boot = CleanBootstrap(models, uvdata)
        curdir = os.getcwd()
        os.chdir(data_dir)
        boot.run(n=n_boot, nonparametric=False, use_v=False,
                 use_kde=True, outname=['boot_{}'.format(band), '.uvf'])
        os.chdir(curdir)


def boot_ci(boot_images, original_image, alpha=0.68):
    """
    Calculate bootstrap CI for images.

    :param boot_images:
        Iterable of 2D numpy arrays of the bootstrapped images.
    :param original_image:
        2D numpy array of the original image.
    :return:
        Two numpy arrays with low and high CI borders for each pixel.

    """
    images_cube = np.dstack(boot_images)
    boot_ci = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(boot_ci):
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=alpha)
        boot_ci[x, y] = hdi[1] - hdi[0]

    hdi_low = original_image - boot_ci / 2.
    hdi_high = original_image + boot_ci / 2.

    return hdi_low, hdi_high


def boot_ci_asymm(boot_images, original_image, alpha=0.68):
    """
    Calculate bootstrap CI for images.

    :param boot_images:
        Iterable of 2D numpy arrays of the bootstrapped images
    :param original_image:
        2D numpy array of the original image..
    :return:
        Two numpy arrays with low and high CI borders for each pixel.

    """
    images_cube = np.dstack(boot_images)
    boot_ci = np.zeros(np.shape(images_cube[:, :, 0]))
    hdi_low = np.zeros(np.shape(images_cube[:, :, 0]))
    hdi_high = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(boot_ci):
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=alpha)
        mean_boot = np.nanmean(images_cube[x, y, :])

        hdi_low[x, y] = original_image[x, y] - (mean_boot - hdi[0])
        hdi_high[x, y] = original_image[x, y] + hdi[1] - mean_boot

    return hdi_low, hdi_high


def boot_ci_bc(boot_images, original_image, alpha=0.68):
    """
    Calculate bootstrap CI for images.

    :param boot_images:
        Iterable of 2D numpy arrays of the bootstrapped images.
    :param original_image:
        2D numpy array of the original image.
    :return:
        Two numpy arrays with low and high CI borders for each pixel.

    """
    alpha = 0.5 * (1. - alpha)

    images_cube = np.dstack(boot_images)
    boot_ci_0 = np.zeros(np.shape(images_cube[:, :, 0]))
    boot_ci_1 = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(boot_ci_0):
        # if np.alltrue(~np.isnan(images_cube[x, y, :])):
        if True:
            boot_ci_0[x, y] = bc_endpoint(images_cube[x, y, :],
                                          original_image[x, y], alpha)
            boot_ci_1[x, y] = bc_endpoint(images_cube[x, y, :],
                                          original_image[x, y], 1. - alpha)
        else:
            boot_ci_0[x, y] = np.nan
            boot_ci_1[x, y] = np.nan

    return boot_ci_0, boot_ci_1


def create_scb(boot_slices, obs_slice, conf_band_alpha=0.68):
    slices_ = [arr.reshape((1, len(obs_slice))) for arr in boot_slices]
    sigmas = hdi_of_arrays(slices_).squeeze()
    means = np.mean(np.vstack(boot_slices), axis=0)
    # diff = obs_slice_notna - means
    diff = obs_slice-means
    # Move bootstrap curves to original simulated centers
    slices_ = [slice_+diff for slice_ in boot_slices]
    low, up = create_sim_conf_band(slices_, obs_slice, sigmas,
                                   alpha=conf_band_alpha)
    return low, up


def plot_slices(original_npz, boot_npz, data_dir, point1=(0, 1),
                point2=(0, -10), ylabel=r"RM, $[rad/m^2]$", beam_size_pxl=None):

    import matplotlib
    label_size = 14
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size

    ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                             "cc_x_I.fits"))
    if beam_size_pxl is None:
        beam = ccimage.beam
        beam_size_pxl = np.sqrt(beam[0]*beam[1])/0.1

    loaded = np.load(os.path.join(data_dir, original_npz))
    loaded_boot = np.load(os.path.join(data_dir, boot_npz))

    conv_value = loaded["value"]
    conv_sigma = loaded["sigma"]

    im = Image()
    im._construct(imsize=ccimage.imsize, pixsize=ccimage.pixsize,
                  pixref=ccimage.pixref, stokes='MF', freq=tuple(freqs),
                  pixrefval=ccimage.pixrefval)

    im.image = conv_value
    sigma_im = copy.deepcopy(im)
    sigma_im.image = conv_sigma

    # Values from conventional methods
    original_slice = im.slice(point1=point1, point2=point2)
    original_slice_sigma = sigma_im.slice(point1=point1, point2=point2)


    # aslice = rotm_im.slice(point1=(2.5, -2), point2=(-2.5, -2))

    original_image = conv_value
    boot_images = [loaded_boot[str(i)] for i in range(n_boot)]
    low_ci, high_ci = boot_ci(boot_images, original_image)

    ci_low_im = copy.deepcopy(im)
    ci_low_im.image = low_ci
    ci_high_im = copy.deepcopy(im)
    ci_high_im.image = high_ci

    ci_low_slice = ci_low_im.slice(point1=point1, point2=point2)
    ci_high_slice = ci_high_im.slice(point1=point1, point2=point2)

    boot_slices = list()
    for i in range(n_boot):
        im = Image()
        im._construct(imsize=ccimage.imsize, pixsize=ccimage.pixsize,
                      pixref=ccimage.pixref, stokes='MF',
                      freq=tuple(freqs), pixrefval=ccimage.pixrefval)
        im.image = loaded_boot[str(i)]

        aslice = im.slice(point1=point1, point2=point2)
        boot_slices.append(aslice)

    x = np.arange(len(original_slice))/beam_size_pxl
    fig, axes = plt.subplots(1, 1)
    axes.errorbar(x[::2], original_slice[::2], yerr=original_slice_sigma[::2],
                 fmt=".k")
    axes.set_ylabel(ylabel)
    axes.set_xlabel("Distance along jet, [beam]")
    low_scb, up_scb = create_scb(boot_slices, original_slice)
    axes.fill_between(x, low_scb, up_scb, alpha=0.35, label="SCB")
    axes.fill_between(x, ci_low_slice, ci_high_slice, alpha=0.5, label="CB")
    axes.legend(loc="upper left")

    return fig


def find_cross_coverage(boot_npzs, original_npzs, data_dir, n_boot=100):
    """
    For each sample for each pixel find fraction of times when CB contains other
    sample's values. Averaged.
    """
    cov_arrays = list()
    for i, (boot_npz, original_npz) in enumerate(zip(boot_npzs, original_npzs)):
        # Find CB for given sample
        loaded_boot = np.load(os.path.join(data_dir, boot_npz))
        boot_images = [loaded_boot[str(i)] for i in range(n_boot)]
        original_image = np.load(os.path.join(data_dir, original_npz))["value"]
        low_ci, high_ci = boot_ci(boot_images, original_image)
        # Count number of times when this CB contains other's sample values
        cov_array = np.zeros(boot_images[0].shape, dtype=float)

        original_npzs_ex = [original_npzs[j] for j in range(len(original_npzs))
                            if j != i]
        true_images = [np.load(os.path.join(data_dir, original_npz))["value"]
                       for original_npz in original_npzs_ex]

        for true_image in true_images:
            for (x, y), value in np.ndenumerate(cov_array):
                cov_array[x, y] += float(np.logical_and(low_ci[x, y] < true_image[x, y],
                                                        true_image[x, y] < high_ci[x, y]))

        cov_array = cov_array/len(true_images)
        cov_arrays.append(cov_array)

    return np.mean(cov_arrays, axis=0)


def find_coverage(boot_npzs, original_npzs, true_image, data_dir, n_boot=100):
    """
    For each sample for each pixel find fraction of times when CB contains true
    value.
    """
    loaded_boot_all = list()
    loaded_all = list()
    for boot_npz, original_npz in zip(boot_npzs, original_npzs):
        print(boot_npz, original_npz)
        loaded_boot_all.append(np.load(os.path.join(data_dir, boot_npz)))
        loaded_all.append(np.load(os.path.join(data_dir, original_npz))["value"])

    cov_array = np.zeros(true_image.shape, dtype=float)
    for loaded_boot, original_image in zip(loaded_boot_all, loaded_all):
        boot_images = [loaded_boot[str(i)] for i in range(n_boot)]
        low_ci, high_ci = boot_ci(boot_images, original_image)
        for (x, y), value in np.ndenumerate(cov_array):
            cov_array[x, y] += float(np.logical_and(low_ci[x, y] < true_image[x, y],
                                                    true_image[x, y] < high_ci[x, y]))

    return cov_array/len(loaded_boot_all)


def find_coverage_conv(original_npzs, true_image, data_dir):
    """
    For each sample for each pixel find fraction of times when CB contains true
    value.
    """
    loaded_values = list()
    loaded_sigmas = list()
    for original_npz in original_npzs:
        loaded_sigmas.append(np.load(os.path.join(data_dir, original_npz))["sigma"])
        loaded_values.append(np.load(os.path.join(data_dir, original_npz))["value"])

    cov_array = np.zeros(true_image.shape, dtype=float)
    for sigma_image, original_image in zip(loaded_sigmas, loaded_values):
        low_ci = original_image - sigma_image
        high_ci = original_image + sigma_image
        for (x, y), value in np.ndenumerate(cov_array):
            cov_array[x, y] += float(np.logical_and(low_ci[x, y] < true_image[x, y],
                                                    true_image[x, y] < high_ci[x, y]))

    return cov_array/len(loaded_values)


def plot_coverage_maps(cov_array, cc_fits, colorbar_label="coverage", cmap="hsv"):
    ccimage = create_clean_image_from_fits_file(cc_fits)

    beam = ccimage.beam
    rms = rms_image(ccimage)
    print(rms)
    cov_array[cov_array == 0] = np.nan
    blc, trc = find_bbox(ccimage.image, 1.0*rms, 10)
    fig = iplot(ccimage.image, colors=cov_array, x=ccimage.x, y=ccimage.y,
                min_abs_level=3.0*rms, beam=beam, show_beam=True, blc=blc,
                trc=trc, close=False, colorbar_label=colorbar_label,
                show=True, cmap=cmap)
    return fig


def plot_coverage_hist(cov_array, cov_array_conv):
    import matplotlib
    label_size = 14
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    cov_array[cov_array == 0] = np.nan
    cov_array_conv[cov_array_conv == 0] = np.nan
    conv = list(cov_array_conv.flatten())
    conv = [i for i in conv if not np.isnan(i)]
    boot = list(cov_array.flatten())
    boot = [i for i in boot if not np.isnan(i)]
    fig, axes = plt.subplots(1, 1)
    axes.hist(conv, bins=20, alpha=0.5, label="CONV")
    axes.hist(boot, bins=20, alpha=0.5, label="BOOT")
    axes.axvline(0.68, color="red")
    axes.set_xlabel("Coverage")
    axes.set_ylabel("N")
    axes.legend()
    return fig


if __name__ == "__main__":
    # Find beam
    # ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
    #                                                          "bk_cc_x_I_wo_noise.fits"))
    # beam = ccimage.beam
    # Common beam
    beam = (1.5074023139377921, 1.3342292885366427, 0.36637646630496357)
    beam_size = np.sqrt(beam[0]*beam[1])/0.1

    data_dir = "/home/ilya/data/revision_results"
    import glob
    boot_npzs = glob.glob(os.path.join(data_dir, "SPIX_[0-9]*_boot.npz"))
    boot_npzs = [os.path.split(path)[-1] for path in boot_npzs]
    original_npzs = ["SPIX_{}.npz".format(fn.split("_")[1]) for fn in boot_npzs]

    true_image = np.load(os.path.join(data_dir, "SPIX_true.npz"))["value"]
    cov_array = find_coverage(boot_npzs, original_npzs, true_image, data_dir)
    cov_array_conv = find_coverage_conv(original_npzs, true_image, data_dir)
    np.savetxt("cov_SPIX.txt", cov_array)
    np.savetxt("cov_SPIX_conv.txt", cov_array_conv)

    ccfits = os.path.join(data_dir, "cc_x_I.fits")
    fig = plot_coverage_maps(cov_array_conv, ccfits)

    # fig = plot_slices("ROTM_99.npz", "ROTM_99_boot.npz", data_dir,
    #                   point1=(2.5, -2), point2=(-2.5, -2),
    #                   ylabel=r"RM, $[rad/m^2]$", beam_size_pxl=beam_size)







    # # Common mapsize
    # mapsize_clean = (512, 0.1)
    #
    # data_dir = "/home/ilya/data/revision"
    # uvdata_dir = "/home/ilya/data/revision_results"
    # path_to_script = "/home/ilya/github/ve/difmap/final_clean_nw"
    #
    # n_boot = 100
    # n_sample = 100
    #
    # rms_cs_dict = None
    #
    # # Loop over artificial sample
    # for i_art in range(n_sample)[::-1]:
    #     for band in bands:
    #         shutil.copy(os.path.join(uvdata_dir, "{}_{}.uvf".format(band, i_art)),
    #                     os.path.join(data_dir, "{}.uvf".format(band)))
    #
    #
    #     print("=== Processing original data for sample #{} ===".format(i_art))
    #     uvdata_dict = {band: "{}.uvf".format(band) for band in bands}
    #     result = process_mf(uvdata_dict, beam, data_dir, path_to_script,
    #                         rms_cs_dict=rms_cs_dict, clean_after=False,
    #                         mapsize_clean=mapsize_clean)
    #
    #     rms_cs_dict = result["RMS"]
    #
    #     np.savez_compressed(os.path.join(data_dir, "ROTM_{}".format(i_art)),
    #                         **{value: result["ROTM"][value] for value in
    #                            ("value", "sigma", "chisq")})
    #     np.savez_compressed(os.path.join(data_dir, "SPIX_{}".format(i_art)),
    #                         **{value: result["SPIX"][value] for value in
    #                            ("value", "sigma", "chisq")})
    #     np.save(os.path.join(data_dir, "RMS_{}".format(i_art)), result["RMS"])
    #
    #
    #     print("=== For current sample #{} create {} bootstrapped"
    #           " samples ===".format(i_art, n_boot))
    #     ccfits_dict = {band: {stokes: "cc_{}_{}.fits".format(band, stokes)
    #                           for stokes in ("I", "Q", "U")}
    #                    for band in bands}
    #     create_bootstrap_sample(uvdata_dict, ccfits_dict, data_dir,
    #                             n_boot=n_boot)
    #
    #
    #     # CLEANing all ``n_boot`` bootstrapped uv-data and collecting SPIX/ROTM
    #     # images in ``results`` list
    #     results = list()
    #     for i in range(1, n_boot+1):
    #         print("=== Processing bootstrap sample #{} ===".format(i))
    #         uvdata_dict = {band: "boot_{}_{}.uvf".format(band, str(i).zfill(3))
    #                        for band in bands}
    #         result = process_mf(uvdata_dict, beam, data_dir, path_to_script,
    #                             rms_cs_dict=rms_cs_dict,
    #                             mapsize_clean=mapsize_clean)
    #         rms_cs_dict = result["RMS"]
    #         results.append(result)
    #
    #
    #     # Removing bootstrapped uv-data
    #     print("Removing bootstrapped uv-data")
    #     for band in bands:
    #         for i in range(1, n_boot+1):
    #             os.unlink(os.path.join(data_dir,
    #                                    "boot_{}_{}.uvf".format(band,
    #                                                            str(i).zfill(3))))
    #
    #
    #     # Save resulting maps
    #     np.savez_compressed(os.path.join(data_dir, "ROTM_{}_boot".format(i_art)),
    #                         **{str(i): results[i]["ROTM"]["value"] for i in
    #                            range(n_boot)})
    #     np.savez_compressed(os.path.join(data_dir, "ROTM_SIGMA_{}_boot".format(i_art)),
    #                         **{str(i): results[i]["ROTM"]["sigma"] for i in
    #                            range(n_boot)})
    #     np.savez_compressed(os.path.join(data_dir, "ROTM_CHISQ_{}_boot".format(i_art)),
    #                         **{str(i): results[i]["ROTM"]["chisq"] for i in
    #                            range(n_boot)})
    #     np.savez_compressed(os.path.join(data_dir, "SPIX_{}_boot".format(i_art)),
    #                         **{str(i): results[i]["SPIX"]["value"] for i in
    #                            range(n_boot)})
    #     np.savez_compressed(os.path.join(data_dir, "SPIX_SIGMA_{}_boot".format(i_art)),
    #                         **{str(i): results[i]["SPIX"]["sigma"] for i in
    #                            range(n_boot)})
    #     np.savez_compressed(os.path.join(data_dir, "SPIX_CHISQ_{}_boot".format(i_art)),
    #                         **{str(i): results[i]["SPIX"]["chisq"] for i in
    #                            range(n_boot)})


    # ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
    #                                                          "cc_x_I.fits"))
    # # # loaded_spix = np.load("SPIX.npz")
    # # # loaded_rotm = np.load("ROTM.npz")
    # #
    # loaded_spix = np.load(os.path.join(data_dir, "SPIX_{}.npz".format(i_art)))
    # loaded_rotm = np.load(os.path.join(data_dir, "ROTM_{}.npz".format(i_art)))
    # loaded_spix_boot = np.load(os.path.join(data_dir, "SPIX_{}_boot.npz".format(i_art)))
    # loaded_rotm_boot = np.load(os.path.join(data_dir, "ROTM_{}_boot.npz".format(i_art)))
    #
    # conv_rotm_value = loaded_rotm["value"]
    # conv_rotm_sigma = loaded_rotm["sigma"]
    #
    # conv_spix_value = loaded_spix["value"]
    # conv_spix_sigma = loaded_spix["sigma"]
    #
    # spix_im = Image()
    # spix_im._construct(imsize=ccimage.imsize, pixsize=ccimage.pixsize,
    #                    pixref=ccimage.pixref, stokes='SPIX', freq=tuple(freqs),
    #                    pixrefval=ccimage.pixrefval)
    # spix_im.image = conv_spix_value
    # spix_sigma_im = copy.deepcopy(spix_im)
    # spix_sigma_im.image = conv_spix_sigma
    #
    # rotm_im = copy.deepcopy(spix_im)
    # rotm_im.image = conv_rotm_value
    # rotm_sigma_im = copy.deepcopy(spix_im)
    # rotm_sigma_im.image = conv_rotm_sigma
    #
    # # Values from conventional methods
    # # aslice = spix_im.slice(point1=(0, 1), point2=(0, -10))
    # # aslice_sigma = spix_sigma_im.slice(point1=(0, 1), point2=(0, -10))
    # original_slice = rotm_im.slice(point1=(0, 1), point2=(0, -10))
    # original_slice_sigma = rotm_sigma_im.slice(point1=(0, 1), point2=(0, -10))
    #
    #
    # import matplotlib
    # label_size = 14
    # matplotlib.rcParams['xtick.labelsize'] = label_size
    # matplotlib.rcParams['ytick.labelsize'] = label_size
    # matplotlib.rcParams['axes.titlesize'] = label_size
    # matplotlib.rcParams['axes.labelsize'] = label_size
    # matplotlib.rcParams['font.size'] = label_size
    # matplotlib.rcParams['legend.fontsize'] = label_size
    #
    # x = np.arange(len(original_slice))/beam_size
    # plt.errorbar(x[::2], original_slice[::2], yerr=original_slice_sigma[::2],
    #              fmt=".k")
    # plt.axhline(0)
    # plt.ylabel(r"RM, $[rad/m^2]$")
    # # plt.ylabel(r"$\alpha$")
    # plt.xlabel("Distance along jet, [beam]")
    #
    # # aslice = rotm_im.slice(point1=(2.5, -2), point2=(-2.5, -2))
    #
    # original_image = conv_rotm_value
    # boot_images = [loaded_rotm_boot[str(i)] for i in range(n_boot)]
    # low_ci, high_ci = boot_ci(boot_images, original_image)
    # # low_ci_as, high_ci_as = boot_ci_asymm(boot_images, original_image)
    # # low_ci_bc, high_ci_bc = boot_ci_bc(boot_images, original_image)
    #
    # ci_low_im = copy.deepcopy(spix_im)
    # ci_low_im.image = low_ci
    # ci_high_im = copy.deepcopy(spix_im)
    # ci_high_im.image = high_ci
    #
    # # ci_low_as_im = copy.deepcopy(spix_im)
    # # ci_low_as_im.image = low_ci_as
    # # ci_high_as_im = copy.deepcopy(spix_im)
    # # ci_high_as_im.image = high_ci_as
    #
    # # ci_low_bc_im = copy.deepcopy(spix_im)
    # # ci_low_bc_im.image = low_ci_bc
    # # ci_high_bc_im = copy.deepcopy(spix_im)
    # # ci_high_bc_im.image = high_ci_bc
    #
    # ci_low_slice = ci_low_im.slice(point1=(0, 1), point2=(0, -10))
    # ci_high_slice = ci_high_im.slice(point1=(0, 1), point2=(0, -10))
    #
    # # ci_as_low_slice = ci_low_as_im.slice(point1=(0, 1), point2=(0, -10))
    # # ci_as_high_slice = ci_high_as_im.slice(point1=(0, 1), point2=(0, -10))
    # #
    # # ci_bc_low_slice = ci_low_bc_im.slice(point1=(0, 1), point2=(0, -10))
    # # ci_bc_high_slice = ci_high_bc_im.slice(point1=(0, 1), point2=(0, -10))
    #
    # # plt.fill_between(x, ci_as_low_slice, ci_as_high_slice, alpha=0.25, label="CI AS")
    # # plt.fill_between(x, ci_bc_low_slice, ci_bc_high_slice, alpha=0.25, label="CI BC")
    # # plt.legend()
    #
    # boot_slices = list()
    # for i in range(n_boot):
    #     spix_im = Image()
    #     spix_im._construct(imsize=ccimage.imsize, pixsize=ccimage.pixsize,
    #                        pixref=ccimage.pixref, stokes='SPIX',
    #                        freq=tuple(freqs), pixrefval=ccimage.pixrefval)
    #     spix_im.image = loaded_spix_boot[str(i)]
    #
    #     rotm_im = copy.deepcopy(spix_im)
    #     rotm_im.image = loaded_rotm_boot[str(i)]
    #
    #     # aslice = spix_im.slice(point1=(0, 1), point2=(0, -10))
    #     aslice = rotm_im.slice(point1=(0, 1), point2=(0, -10))
    #     boot_slices.append(aslice)
    #     # aslice = rotm_im.slice(point1=(2.5, -2), point2=(-2.5, -2))
    #
    #     # x = np.arange(len(aslice))/beam_size
    #     # plt.plot(x, aslice, alpha=0.2, color='r')
    #
    # low_scb, up_scb = create_scb(boot_slices, original_slice)
    # plt.fill_between(x, low_scb, up_scb, alpha=0.35, label="SCB")
    # plt.fill_between(x, ci_low_slice, ci_high_slice, alpha=0.5, label="CB")
    # plt.legend(loc="upper left")