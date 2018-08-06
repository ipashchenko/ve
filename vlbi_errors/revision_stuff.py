import os
import sys
import copy
import numpy as np
import shutil
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    # Find beam
    # ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
    #                                                          "bk_cc_x_I_wo_noise.fits"))
    # beam = ccimage.beam
    # Common beam
    beam = (1.5074023139377921, 1.3342292885366427, 0.36637646630496357)
    beam_size = np.sqrt(beam[0]*beam[1])/0.1
    # Common mapsize
    mapsize_clean = (512, 0.1)

    data_dir = "/home/ilya/data/revision"
    path_to_script = "/home/ilya/github/ve/difmap/final_clean_nw"

    n_boot = 3
    n_sample = 3

    rms_cs_dict = None


    # Loop over artificial sample
    for i_art in range(n_sample):
        for band in bands:
            shutil.copy(os.path.join(data_dir, "{}_{}.uvf".format(band, i_art)),
                        os.path.join(data_dir, "{}.uvf".format(band)))


        print("=== Processing original data for sample #{} ===".format(i_art))
        uvdata_dict = {band: "{}.uvf".format(band) for band in bands}
        result = process_mf(uvdata_dict, beam, data_dir, path_to_script,
                            rms_cs_dict=rms_cs_dict, clean_after=False,
                            mapsize_clean=mapsize_clean)

        rms_cs_dict = result["RMS"]

        np.savez_compressed(os.path.join(data_dir, "ROTM_{}".format(i_art)),
                            **{value: result["ROTM"][value] for value in
                               ("value", "sigma", "chisq")})
        np.savez_compressed(os.path.join(data_dir, "SPIX_{}".format(i_art)),
                            **{value: result["SPIX"][value] for value in
                               ("value", "sigma", "chisq")})
        np.save(os.path.join(data_dir, "RMS_{}".format(i_art)), result["RMS"])


        print("=== For current sample #{} create {} bootstrapped"
              " samples ===".format(i_art, n_boot))
        ccfits_dict = {band: {stokes: "cc_{}_{}.fits".format(band, stokes)
                              for stokes in ("I", "Q", "U")}
                       for band in bands}
        create_bootstrap_sample(uvdata_dict, ccfits_dict, data_dir,
                                n_boot=n_boot)


        # CLEANing all ``n_boot`` bootstrapped uv-data and collecting SPIX/ROTM
        # images in ``results`` list
        results = list()
        for i in range(1, n_boot+1):
            print("=== Processing bootstrap sample #{} ===".format(i))
            uvdata_dict = {band: "boot_{}_{}.uvf".format(band, str(i).zfill(3))
                           for band in bands}
            result = process_mf(uvdata_dict, beam, data_dir, path_to_script,
                                rms_cs_dict=rms_cs_dict,
                                mapsize_clean=mapsize_clean)
            rms_cs_dict = result["RMS"]
            results.append(result)


        # Removing bootstrapped uv-data
        print("Removing bootstrapped uv-data")
        for band in bands:
            for i in range(1, n_boot+1):
                os.unlink(os.path.join(data_dir,
                                       "boot_{}_{}.uvf".format(band,
                                                               str(i).zfill(3))))


        # Save resulting maps
        np.savez_compressed(os.path.join(data_dir, "ROTM_{}_boot".format(i_art)),
                            **{str(i): results[i]["ROTM"]["value"] for i in
                               range(n_boot)})
        np.savez_compressed(os.path.join(data_dir, "ROTM_SIGMA_{}_boot".format(i_art)),
                            **{str(i): results[i]["ROTM"]["sigma"] for i in
                               range(n_boot)})
        np.savez_compressed(os.path.join(data_dir, "ROTM_CHISQ_{}_boot".format(i_art)),
                            **{str(i): results[i]["ROTM"]["chisq"] for i in
                               range(n_boot)})
        np.savez_compressed(os.path.join(data_dir, "SPIX_{}_boot".format(i_art)),
                            **{str(i): results[i]["SPIX"]["value"] for i in
                               range(n_boot)})
        np.savez_compressed(os.path.join(data_dir, "SPIX_SIGMA_{}_boot".format(i_art)),
                            **{str(i): results[i]["SPIX"]["sigma"] for i in
                               range(n_boot)})
        np.savez_compressed(os.path.join(data_dir, "SPIX_CHISQ_{}_boot".format(i_art)),
                            **{str(i): results[i]["SPIX"]["chisq"] for i in
                               range(n_boot)})

    ccimage = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                             "cc_x_I.fits"))
    # loaded_spix = np.load("SPIX.npz")
    # loaded_rotm = np.load("ROTM.npz")

    loaded_spix = np.load(os.path.join(data_dir, "SPIX_{}.npz".format(i_art)))
    loaded_rotm = np.load(os.path.join(data_dir, "ROTM_{}.npz".format(i_art)))
    loaded_spix_boot = np.load(os.path.join(data_dir, "SPIX_{}_boot.npz".format(i_art)))
    loaded_rotm_boot = np.load(os.path.join(data_dir, "ROTM_{}_boot.npz".format(i_art)))

    conv_rotm_value = loaded_rotm["value"]
    conv_rotm_sigma = loaded_rotm["sigma"]

    conv_spix_value = loaded_spix["value"]
    conv_spix_sigma = loaded_spix["sigma"]

    spix_im = Image()
    spix_im._construct(imsize=ccimage.imsize, pixsize=ccimage.pixsize,
                       pixref=ccimage.pixref, stokes='SPIX', freq=tuple(freqs),
                       pixrefval=ccimage.pixrefval)
    spix_im.image = conv_spix_value
    spix_sigma_im = copy.deepcopy(spix_im)
    spix_sigma_im.image = conv_spix_sigma

    rotm_im = copy.deepcopy(spix_im)
    rotm_im.image = conv_rotm_value
    rotm_sigma_im = copy.deepcopy(spix_im)
    rotm_sigma_im.image = conv_rotm_sigma

    aslice = spix_im.slice(point1=(0, 1), point2=(0, -10))
    aslice_sigma = spix_sigma_im.slice(point1=(0, 1), point2=(0, -10))

    x = np.arange(len(aslice))/beam_size
    plt.errorbar(x, aslice, yerr=aslice_sigma, fmt=".k")
    # aslice = rotm_im.slice(point1=(2.5, -2), point2=(-2.5, -2))

    for i in range(n_boot):
        spix_im = Image()
        spix_im._construct(imsize=ccimage.imsize, pixsize=ccimage.pixsize,
                           pixref=ccimage.pixref, stokes='SPIX',
                           freq=tuple(freqs), pixrefval=ccimage.pixrefval)
        spix_im.image = loaded_spix_boot[str(i)]

        rotm_im = copy.deepcopy(spix_im)
        rotm_im.image = conv_rotm_value

        aslice = spix_im.slice(point1=(0, 1), point2=(0, -10))
        # aslice = rotm_im.slice(point1=(2.5, -2), point2=(-2.5, -2))

        x = np.arange(len(aslice))/beam_size
        plt.plot(x, aslice, alpha=0.2)
    #
    # # x = np.arange(len(observed_spix_slice))/beam_size
    # # plt.errorbar(x, observed_spix_slice, yerr=observed_sigma_spix_slice, fmt=".k")
    # # plt.plot(x, true_spix_slice)
    # # plt.xlabel("distance, Beam size")
    # # plt.ylabel(r"$\alpha$")
    # # plt.show()
    # # plt.savefig("alpha_slice.png")