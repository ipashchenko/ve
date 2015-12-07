import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from from_fits import (get_fits_image_info, create_uvdata_from_fits_file,
                       create_ccmodel_from_fits_file,
                       create_clean_image_from_fits_file)
from image import (BasicImage, Image, CleanImage)
from utils import (mask_region, mas_to_rad, find_card_from_header,
                   slice_2darray)
from model import Model
from components import DeltaComponent
from images import Images
from spydiff import clean_difmap
from images import Images
from bootstrap import CleanBootstrap


def simulate_grad(low_freq_map, high_freq_map, uvdata_files, cc_flux,
                  outpath, grad_value, width, length, k, noise_factor=1.,
                  rm_value_0=0.0):
    """
    Function that simulates ROTM gradients in uv-data.

    :param low_freq_map:
        Path to FITS-file with clean map for lowest frequency.
    :param high_freq_map:
        Path to FITS-file with clean map for highest frequency.
    :param uvdata_files:
        Iterable of paths to FITS-files with uv-data.
    :param cc_flux:
        Flux density of CC-components that will model the polarization
        [Jy/pixel]. That CCs will be used in Q & U clean models. Being convolved
        with beam it lowers the maximum flux density by factor
        ``pi * beam_width ** 2`` and converts to units [Jy/beam].
    :param outpath:
        Path where to save all resulting files.
    :param grad_value:
        Value of ROTM gradient [rad/m/m/beam], where ``beam`` - major axis of
        low-frequency beam.
    :param width:
        Width of model jet in units of the lowest frequency beam's major axis.
    :param length:
        Length of model jet in units of the lowest frequency beam's major axis.
    :param k:
        How many model image pixels should be in one highest frequency image
        pixel?
    :param noise_factor:
        This enhanced noise that is added to model uv-data from those that in
        original uv-data.
    :param rm_value_0: (optional)
        Value of ROTM at image center [rad/m/m]. (default: ``0.0``)

    :return:
        Creates FITS-files with uv-data where uv-data is substituted by data
        with ROTM gradient.

    """

    (imsize_h, pixref_h, pixrefval_h, (bmaj_h, bmin_h, bpa_h,), pixsize_h,
     stokes_h, freq_h) = get_fits_image_info(high_freq_map)
    (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
     stokes_l, freq_l) = get_fits_image_info(low_freq_map)

    # new pixsize [rad]
    pixsize = (abs(pixsize_h[0]) / k, abs(pixsize_h[1]) / k)
    # new imsize
    x1 = imsize_l[0] * abs(pixsize_l[0]) / abs(pixsize[0])
    x2 = imsize_l[1] * abs(pixsize_l[1]) / abs(pixsize[1])
    imsize = (int(x1 - x1 % 2),
              int(x2 - x2 % 2))
    # new pixref
    pixref = (imsize[0]/2, imsize[1]/2)
    # FIXME: Should i use ellipse beam for comparing model with results?
    # Beam width (of low frequency map) in new pixels
    beam_width = bmaj_l / abs(pixsize[0])

    # Jet's parameters in new pixels
    jet_width = width * bmaj_l / abs(pixsize[0])
    jet_length = length * bmaj_l / abs(pixsize[0])

    # Construct image with new parameters
    image = BasicImage(imsize=imsize, pixsize=pixsize, pixref=pixref)

    # Construct region with emission
    # TODO: Construct cone region
    jet_region = mask_region(image._image, region=(pixref[0] -
                                                   int(jet_width // 2),
                                                   pixref[1],
                                                   pixref[0] +
                                                   int(jet_width // 2),
                                                   pixref[1] + jet_length))
    jet_region = np.ma.array(image._image, mask=~jet_region.mask)

    # TODO: add decline of contrjet
    def flux(x, y, max_flux, length, width):
        """
        Function that defines model flux distribution that declines linearly
        from phase center (0, 0) along jet and parabolically across.

        :param x:
            x-coordinates on image [pixels].
        :param y:
            y-coordinates on image [pixels].
        :param max_flux:
            Flux density maximum [Jy/pixels].
        :param length:
            Length of jet [pixels].
        :param width:
            Width of jet [pixels].
        """
        return max_flux - (max_flux / length) * x -\
            (max_flux / (width / 2) ** 2.) * y ** 2.

    def rm(x, y, grad_value, rm_value_0=0.0):
        """
        Function that defines model of ROTM gradient distribution.

        :param x:
            x-coordinates on image [pixels].
        :param y:
            y-coordinates on image [pixels].
        :param grad_value:
            Value of gradient [rad/m/m/pixel].
        :param rm_value_0: (optional)
            Value of ROTM at center [rad/m/m]. (default: ``0.0``)
        """
        return grad_value * x + rm_value_0

    # Create map of ROTM
    print "Creating ROTM image with gradient..."
    image_rm = Image(imsize=imsize, pixsize=pixsize, pixref=pixref)
    # Use value for gradient ``k`` times less then original because of pixel
    # size
    image_rm._image = rm(image.x/abs(pixsize[0]), image.y/abs(pixsize[1]),
                         grad_value / beam_width, rm_value_0=rm_value_0)
    image_rm._image = np.ma.array(image_rm._image, mask=jet_region.mask)

    # Create ROTM image with size as for lowest freq. and pixel size - as for
    # highest freq. map
    save_imsize = (int(imsize_l[0] * pixsize_l[0] / pixsize_h[0]),
                   int(imsize_l[1] * pixsize_l[1] / pixsize_h[1]))
    save_pixref = (int(save_imsize[0]/2), int(save_imsize[0]/2))
    save_pixsize = pixsize_h
    save_rm = Image(imsize=save_imsize, pixsize=save_pixsize,
                    pixref=save_pixref)
    save_rm._image = rm(save_rm.x/abs(save_pixsize[0]),
                        save_rm.y/abs(save_pixsize[1]),
                        grad_value/(bmaj_l/abs(save_pixsize[0])),
                        rm_value_0=rm_value_0)
    half_width_l = int(width * bmaj_l/abs(save_pixsize[0])//2)
    jet_length_l = int(length * bmaj_l/ abs(save_pixsize[0]))
    jet_region_l = mask_region(save_rm._image,
                               region=(save_pixref[0] - half_width_l,
                                       save_pixref[1],
                                       save_pixref[0] + half_width_l,
                                       save_pixref[1] + jet_length_l))
    save_rm._image = np.ma.array(save_rm._image, mask=~jet_region_l.mask)
    print "Saving image of ROTM gradient..."
    np.savetxt(os.path.join(outpath, 'RM_grad_image.txt'), save_rm._image)


    # Create model instance and fill it with components
    model_i = Model(stokes='I')
    model_q = Model(stokes='Q')
    model_u = Model(stokes='U')

    max_flux = k * cc_flux / (np.pi * beam_width ** 2)
    # Use 10x total intensity
    comps_i = [DeltaComponent(flux(image.y[x, y] / abs(pixsize[0]),
                                   image.x[x, y] / abs(pixsize[0]),
                                   10. * max_flux,
                                   jet_length, jet_width),
                              image.x[x, y] / mas_to_rad,
                              image.y[x, y] / mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    comps_q = [DeltaComponent(flux(image.y[x, y] / abs(pixsize[0]),
                                   image.x[x, y] / abs(pixsize[0]),
                                   max_flux, jet_length, jet_width),
                              image.x[x, y] / mas_to_rad,
                              image.y[x, y] / mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    # FIXME: just ``comps_u = comps_q``
    comps_u = [DeltaComponent(flux(image.y[x, y] / abs(pixsize[0]),
                                   image.x[x, y] / abs(pixsize[0]),
                                   max_flux, jet_length, jet_width),
                              image.x[x, y] / mas_to_rad,
                              image.y[x, y] / mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]

    # FIXME: why for Q & U?
    # Keep only positive components
    comps_i = [comp for comp in comps_i if comp.p[0] > 0]
    # comps_q = [comp for comp in comps_q if comp.p[0] > 0]
    # comps_u = [comp for comp in comps_u if comp.p[0] > 0]

    print "Adding components to I,Q & U models..."
    model_i.add_components(*comps_i)
    model_q.add_components(*comps_q)
    model_u.add_components(*comps_u)

    image_i = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='I')
    image_q = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='Q')
    image_u = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='U')
    image_i.add_model(model_i)
    image_q.add_model(model_q)
    image_u.add_model(model_u)

    print "Creating PPOL image for constructing Q & U images on supplied" \
          " frequencies..."
    images = Images()
    images.add_images([image_q, image_u])
    ppol_image = images.create_pol_images(convolved=False)[0]

    # Equal Q & U results in chi_0 = pi / 4
    chi_0 = np.pi / 4
    # chi_0 = 0.

    # Loop over specified uv-data, substitute real data with fake and save to
    # specified location
    print "Now substituting ROTM gradient in real data and saving out..."
    for uvfile in uvdata_files:
        uvdata = create_uvdata_from_fits_file(uvfile)
        freq_card = find_card_from_header(uvdata._io.hdu.header,
                                          value='FREQ')[0]
        # Frequency in Hz
        # FIXME: Create property ``freq`` for ``UVData`` class
        freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        # Rotate PANG by multiplying polarized intensity on cos/sin
        lambda_sq = (3. * 10 ** 8 / freq) ** 2
        print "Creating Faraday Rotated arrays of Q & U for frequency {}" \
              " Hz".format(freq)
        # The same flux for all frequencies
        # image_i = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
        #                 stokes='I', freq=freq)
        # Stokes ``I`` image remains the same - only frequency changes
        image_i.stokes = 'I'
        image_i.freq = freq
        q_array = ppol_image._image * np.cos(2. * (chi_0 + image_rm._image *
                                                   lambda_sq))
        u_array = ppol_image._image * np.sin(2. * (chi_0 + image_rm._image *
                                                   lambda_sq))
        image_q = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
                        stokes='Q', freq=freq)
        image_q._image = q_array
        image_u = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
                        stokes='U', freq=freq)
        image_u._image = u_array

        model_i = Model(stokes='I')
        model_q = Model(stokes='Q')
        model_u = Model(stokes='U')
        print "Creating components of I,Q & U for frequency {} Hz".format(freq)
        comps_i = [DeltaComponent(image_i._image[x, y],
                                  image_i.x[x, y] / mas_to_rad,
                                  image_i.y[x, y] / mas_to_rad) for (x, y), value
                   in np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        comps_q = [DeltaComponent(image_q._image[x, y],
                                  image_q.x[x, y] / mas_to_rad,
                                  image_q.y[x, y] / mas_to_rad) for (x, y), value
                   in np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        comps_u = [DeltaComponent(image_u._image[x, y],
                                  image_u.x[x, y] / mas_to_rad,
                                  image_u.y[x, y] / mas_to_rad) for (x, y), value
                   in np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        print "Adding components of I, Q & U for frequency {} Hz" \
              " models".format(freq)
        model_i.add_components(*comps_i)
        model_q.add_components(*comps_q)
        model_u.add_components(*comps_u)

        # Substitute I, Q & U models to uv-data and add noise
        print "Substituting I, Q & U models in uv-data, adding noise, saving" \
              " frequency {} Hz".format(freq)
        noise = uvdata.noise(average_freq=True)
        for baseline, std in noise.items():
            noise[baseline] = noise_factor * std
        uvdata.substitute([model_i, model_q, model_u])
        uvdata.noise_add(noise)
        # Save uv-data to file
        uv_save_fname = os.path.join(outpath,
                                     'simul_uv_{}_Hz.fits'.format(freq))
        if os.path.exists(uv_save_fname):
            print "Deleting existing file: {}".format(uv_save_fname)
            os.remove(uv_save_fname)
        uvdata.save(uvdata.data, uv_save_fname)


def bootstrap_uv_fits(uv_fits_fname, cc_fits_fnames, n, uvpath=None,
                      ccpath=None, outpath=None, outname=None):
    """
    Function that bootstraps UV-data in user-specified UV-FITS files and FITS
    files with CC-models.
    :param uv_fits_fname:
    :param cc_fits_fnames:
        Iterable of file names with CC models.
    :param uvpath:
    :param ccpath:
    :param outpath:
    :param boot_kwargs:
    """
    if ccpath is not None:
        if len(ccpath) > 1:
            assert len(cc_fits_fnames) == len(ccpath)
    else:
        ccpath = [None] * len(cc_fits_fnames)

    if uvpath is not None:
        uv_fits_fname = os.path.join(uvpath, uv_fits_fname)
    uvdata = create_uvdata_from_fits_file(uv_fits_fname)

    models = list()
    for cc_fits_fname, ccpath_ in zip(cc_fits_fnames, ccpath):
        if ccpath_ is not None:
            cc_fits_fname = os.path.join(ccpath_, cc_fits_fname)
        # FIXME: I can infer ``stokes`` from FITS-file!
        stokes = get_fits_image_info(cc_fits_fname)[-2].upper()
        ccmodel = create_ccmodel_from_fits_file(cc_fits_fname, stokes=stokes)
        models.append(ccmodel)

    boot = CleanBootstrap(models, uvdata)
    if outpath is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    curdir = os.getcwd()
    os.chdir(outpath)
    boot.run(n=n, outname=[outname, '.fits'])
    os.chdir(curdir)


if __name__ == '__main__':
    n_boot = 5
    noise_factor = 1.
    # Gradient value (per beam)
    grad_value = 100.
    rm_value_0 = 200.
    high_freq_map =\
        '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/im/I/cc.fits'
    low_freq_map =\
        '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/im/I/cc.fits'
    uvdata_files =\
        ['/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/uv/sc_uv.fits',
         '/home/ilya/vlbi_errors/0952+179/2007_04_30/C2/uv/sc_uv.fits',
         '/home/ilya/vlbi_errors/0952+179/2007_04_30/X1/uv/sc_uv.fits',
         '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/uv/sc_uv.fits']
    cc_flux = 0.20
    outpath = '/home/ilya/vlbi_errors/simdata_{}/'.format(noise_factor)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    width = 2.0
    length = 10.
    k = 1
    simulate_grad(low_freq_map, high_freq_map, uvdata_files, cc_flux=cc_flux,
                  outpath=outpath, grad_value=grad_value, width=width,
                  length=length, k=k, noise_factor=noise_factor,
                  rm_value_0=rm_value_0)

    # Now calculate ROTM image using simulated data
    path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
    map_info_l = get_fits_image_info(low_freq_map)
    map_info_h = get_fits_image_info(high_freq_map)
    beam_restore = map_info_l[3]
    beam_restore_ = (beam_restore[0] / mas_to_rad, beam_restore[1] / mas_to_rad,
                     beam_restore[2])
    mapsize_clean = (map_info_h[0][0],
                     map_info_h[-3][0] / mas_to_rad)
    for uvpath in glob.glob(os.path.join(outpath, "simul_uv_*")):
        uvdir, uvfile = os.path.split(uvpath)
        print "Cleaning uv file {}".format(uvpath)
        uvdata = create_uvdata_from_fits_file(uvpath)
        freq_card = find_card_from_header(uvdata._io.hdu.header,
                                          value='FREQ')[0]
        # Frequency in Hz
        freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        for stoke in ('i', 'q', 'u'):
            print "Stokes {}".format(stoke)
            clean_difmap(uvfile, "simul_{}_{}_cc.fits".format(stoke, freq),
                         stoke, mapsize_clean, path=uvdir,
                         path_to_script=path_to_script,
                         beam_restore=beam_restore_, outpath=uvdir)

    # Create image of ROTM
    print "Creating image of simulated ROTM"
    images = Images()
    images.add_from_fits(wildcard=os.path.join(outpath, "simul_*_cc.fits"))
    # Create mask for faster calculation
    mask = np.ones((512, 512), dtype=int)
    mask[50:300, 150:350] = 0
    rotm_image, s_rotm_image = images.create_rotm_image(mask=mask)
    # Plot slice
    plt.errorbar(np.arange(210, 302, 1),
                 rotm_image.slice((240, 210), (240, 302)),
                 s_rotm_image.slice((240, 210), (240, 302)), fmt='.k')
    # Plot real ROTM grad values
    (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
     stokes_l, freq_l) = get_fits_image_info(low_freq_map)
    # Jet width in pixels
    jet_width = width * bmaj_l / abs(rotm_image.pixsize[0])

    # Analytical gradient in real image (didn't convolved)
    def rm(x, y, grad_value, rm_value_0=0.0):
        k = grad_value / (bmaj_l/abs(rotm_image.pixsize[0]))
        return k * x + rm_value_0

    plt.plot(np.arange(210, 302, 1),
             rm(np.arange(210, 302, 1) - rotm_image.pixref[1], None,
                grad_value, rm_value_0=rm_value_0))
    plt.axvline(rotm_image.pixref[1] - jet_width / 2.)
    plt.axvline(rotm_image.pixref[1] + jet_width / 2.)


    # # Bootstrap simulated data
    # print "Bootstrapping simulated data..."
    # cc_fits_fnames_glob = 'simul_i_*_cc.fits'
    # ccpath = '/home/ilya/vlbi_errors/simdata_{}'.format(noise_factor)
    # uv_fits_fnames_glob = 'simul_uv_*_Hz.fits'
    # uvpath = '/home/ilya/vlbi_errors/simdata_{}'.format(noise_factor)
    # for uv_fits_fname in glob.glob(os.path.join(uvpath, uv_fits_fnames_glob)):
    #     uvdata = create_uvdata_from_fits_file(uv_fits_fname)
    #     freq_card = find_card_from_header(uvdata._io.hdu.header,
    #                                       value='FREQ')[0]
    #     # Frequency in Hz
    #     freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
    #     cc_fits_fnames_glob = 'simul_*_{}_cc.fits'.format(freq)
    #     cc_fits_fnames = glob.glob(os.path.join(ccpath, cc_fits_fnames_glob))
    #     print "Bootstrapping frequency {}".format(freq)
    #     bootstrap_uv_fits(uv_fits_fname, cc_fits_fnames, n_boot, uvpath=uvpath,
    #                       ccpath=None, outpath=uvpath,
    #                       outname='test_{}'.format(freq))

    # # Average bootstrapped data
    # freqs = list()
    # bootstrapped_files = glob.glob(os.path.join(uvpath, "test_*.fits"))
    # for fname in bootstrapped_files:
    #     freqs.append(os.path.split(fname)[-1].split('_')[1])
    # freqs = set(freqs)

    # import copy
    # import types


    # def multiply(self, x):
    #     """
    #     Multiply visibilities on number.
    #     :param x:
    #     :return:
    #     """
    #     self_copy = copy.deepcopy(self)
    #     self_copy.uvdata = x * self.uvdata

    #     return self_copy


    # for freq in freqs:
    #     print "==================================================="
    #     print "Averaging bootstrapped uv-data for frequency {}".format(freq)
    #     print "==================================================="
    #     uvdatas = list()
    #     for i, uv_path in enumerate(glob.glob(os.path.join(uvpath,
    #                                                        "test_{}_*.fits".format(freq)))):
    #         uvdir, uvfile = os.path.split(uv_path)
    #         uvdata = create_uvdata_from_fits_file(uv_path)
    #         uvdatas.append(uvdata)
    #     boot_averaged_uvdata = sum(uvdatas)
    #     boot_averaged_uvdata.multiply = types.MethodType(multiply,
    #                                                      boot_averaged_uvdata)
    #     boot_averaged_uvdata = boot_averaged_uvdata.multiply(1. / n_boot)
    #     uvdata.save(boot_averaged_uvdata._data,
    #                 "boot_averaged_{}.fits".format(freq))

    # # Clean averaged bootstrapped data
    # path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
    # for freq in freqs:
    #     print "==================================================="
    #     print "Cleaning frequency {}".format(freq)
    #     print "==================================================="
    #     for uv_path in glob.glob(os.path.join(uvpath,
    #                                           "boot_averaged_{}.fits".format(freq))):
    #         uvdir, uvfile = os.path.split(uv_path)
    #         print "Cleaning uv file {}".format(uv_path)
    #         uvdata = create_uvdata_from_fits_file(uv_path)
    #         freq_card = find_card_from_header(uvdata._io.hdu.header,
    #                                           value='FREQ')[0]
    #         # Frequency in Hz
    #         freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
    #         for stoke in ('i', 'q', 'u'):
    #             print "Stokes {}".format(stoke)
    #             clean_difmap(uvfile, "boot_averaged_{}_{}_cc.fits".format(stoke,
    #                                                                       freq),
    #                          stoke, mapsize_clean, path=uvdir,
    #                          path_to_script=path_to_script,
    #                          beam_restore=beam_restore_, outpath=uvdir)

    # # Create image of ROTM from bootstrapped averaged uv-data
    # print "Creating image of ROTM from bootstrapped averaged uv-data"
    # images = Images()
    # images.add_from_fits(wildcard=os.path.join(outpath,
    #                                            "boot_averaged_*_cc.fits"))
    # # Create mask for faster calculation
    # mask = np.ones((512, 512), dtype=int)
    # mask[200:400, 200:312] = 0
    # rotm_image, s_rotm_image = images.create_rotm_image(mask=mask)
    # # Plot slice
    # plt.errorbar(np.arange(240, 272, 1),
    #              rotm_image.slice((270, 240), (270,272)),
    #              s_rotm_image.slice((270,240), (270,272)), fmt='.k')
    # # Plot real ROTM grad values
    # (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
    #  stokes_l, freq_l) = get_fits_image_info(low_freq_map)
    # # Jet width in pixels
    # jet_width = width * bmaj_l / abs(rotm_image.pixsize[0])

    # # Analytical gradient in real image (didn't convolved)
    # def rm_(x, grad_value, imsize_x):
    #     return -grad_value * (x - imsize_x / 2)

    # plt.plot(np.arange(240, 272, 1), rm_(np.arange(240, 272, 1), grad_value,
    #                                      rotm_image.imsize[0]))
    # plt.axvline(rotm_image.pixref[1] - jet_width / 2.)
    # plt.axvline(rotm_image.pixref[1] + jet_width / 2.)


    # # Clean bootstrapped data
    # path_to_script = '/home/ilya/Dropbox/Zhenya/to_ilya/clean/final_clean_nw'
    # # Find all frequencies
    # freqs = list()
    # bootstrapped_files = glob.glob(os.path.join(uvpath, "simul_uv*.fits"))
    # for fname in bootstrapped_files:
    #     freqs.append(os.path.split(fname)[-1].split('_')[2])
    # freqs = set(freqs)

    # for freq in freqs:
    #     print "==================================================="
    #     print "Cleaning frequency {}".format(freq)
    #     print "==================================================="
    #     for i, uv_path in enumerate(glob.glob(os.path.join(uvpath,
    #                                                        "test_{}_*.fits".format(freq)))):
    #         uvdir, uvfile = os.path.split(uv_path)
    #         print "Cleaning uv file {}".format(uv_path)
    #         uvdata = create_uvdata_from_fits_file(uv_path)
    #         freq_card = find_card_from_header(uvdata._io.hdu.header,
    #                                           value='FREQ')[0]
    #         # Frequency in Hz
    #         freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
    #         for stoke in ('i', 'q', 'u'):
    #             print "Stokes {}".format(stoke)
    #             clean_difmap(uvfile, "test_{}_{}_{}_cc.fits".format(i, stoke,
    #                                                                 freq),
    #                          stoke, mapsize_clean, path=uvdir,
    #                          path_to_script=path_to_script,
    #                          beam_restore=beam_restore_, outpath=uvdir)

    # # Averaging Q & U bootstrapped images
    # avg_images = Images()
    # for freq in freqs:
    #     print "working with freq {}".format(freq)
    #     for stoke in ('q', 'u', 'i'):
    #         print "working with stokes {}".format(stoke)
    #         images = Images()
    #         wc = "test_*_{}_{}_cc.fits".format(stoke, freq)
    #         images.add_from_fits(wildcard=os.path.join(uvpath, wc))
    #         images._create_cube(stokes=stoke.upper(), freq=images.freqs[0])
    #         averaged_array = np.mean(images._images_cube, axis=2)
    #         img = images._images_dict[images.freqs[0]][stoke.upper()][0]
    #         image = Image(imsize=img.imsize, pixref=img.pixref,
    #                       pixrefval=img.pixrefval, pixsize=img.pixsize,
    #                       freq=img.freq, stokes=stoke.upper())
    #         image._image = averaged_array
    #         avg_images.add_image(image)

    # # Create ROTM image for each bootstrapped realization
    # rotm_images_dict = dict()
    # print "Creating images of bootstrapped data"
    # for i in range(n_boot):
    #     print "Creating ROTM image for bootstrap replica #{}".format(i)
    #     images = Images()
    #     images.add_from_fits(wildcard=os.path.join(uvpath,
    #                                                "test_{}_*_cc.fits".format(i)))
    #     # Create mask for faster calculation
    #     mask = np.ones((512, 512), dtype=int)
    #     mask[200:400, 200:312] = 0
    #     rotm_image, s_rotm_image = images.create_rotm_image(mask=mask)
    #     rotm_images_dict.update({i: (rotm_image, s_rotm_image)})
    #     # Plot slice
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.errorbar(np.arange(210, 302, 1),
    #                  rotm_image.slice((270, 210), (270, 302)),
    #                  s_rotm_image.slice((270, 210), (270, 302)), fmt='.k')
    #     # Plot real ROTM grad values
    #     (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
    #      stokes_l, freq_l) = get_fits_image_info(low_freq_map)
    #     # Jet width in pixels
    #     jet_width = width * bmaj_l / abs(rotm_image.pixsize[0])

    #     # Analytical gradient in real image (didn't convolved)
    #     def rm_(x, grad_value, imsize_x):
    #         return -grad_value * (x - imsize_x / 2)

    #     plt.plot(np.arange(210, 302, 1), rm_(np.arange(210, 302, 1), grad_value,
    #                                          rotm_image.imsize[0]))
    #     plt.axvline(rotm_image.pixref[1] - jet_width / 2.)
    #     plt.axvline(rotm_image.pixref[1] + jet_width / 2.)
    #     plt.title("{} of {}".format(i, n_boot))
    #     fig.show()
    #     fig.savefig('ROTM_slice_{}_{}.png'.format(i, n_boot),
    #                 bbox_inches='tight')
    #     plt.close()

    #     rotm_images = [im for im, sim in rotm_images_dict.values()]
    #     average_ROTM_array = np.mean(np.dstack(tuple(image.image for image in
    #                                                  rotm_images)), axis=2)
    #     median_ROTM_array = np.median(np.dstack(tuple(image.image for image in
    #                                                  rotm_images)), axis=2)

    #     # Plot slice of simulated map and average bootstrap
    #     images = Images()
    #     images.add_from_fits(wildcard=os.path.join(outpath, "simul_*_cc.fits"))
    #     # Create mask for faster calculation
    #     mask = np.ones((512, 512), dtype=int)
    #     mask[200:400, 200:312] = 0
    #     rotm_image, s_rotm_image = images.create_rotm_image(mask=mask)

    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.plot(np.arange(240, 272, 1),
    #              slice_2darray(average_ROTM_array, (270, 240), (270,272)), '.r')
    #     plt.plot(np.arange(240, 272, 1),
    #              slice_2darray(median_ROTM_array, (270, 240), (270,272)), '.g')
    #     # Plot slice
    #     plt.errorbar(np.arange(240, 272, 1),
    #                  rotm_image.slice((270, 240), (270,272)),
    #                  s_rotm_image.slice((270,240), (270,272)), fmt='.k')
    #     # Plot real ROTM grad values
    #     (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
    #      stokes_l, freq_l) = get_fits_image_info(low_freq_map)
    #     # Jet width in pixels
    #     jet_width = width * bmaj_l / abs(rotm_image.pixsize[0])

    #     # Analytical gradient in real image (didn't convolved)
    #     def rm_(x, grad_value, imsize_x):
    #         return -grad_value * (x - imsize_x / 2)

    #     plt.plot(np.arange(240, 272, 1), rm_(np.arange(240, 272, 1), grad_value,
    #                                          rotm_image.imsize[0]))
    #     plt.axvline(rotm_image.pixref[1] - jet_width / 2.)
    #     plt.axvline(rotm_image.pixref[1] + jet_width / 2.)
    #     plt.title("Average of bootstrap")
    #     fig.show()

    # # Plot spread of bootstrapped values
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # for i in range(n_boot):
    #     ax.plot(np.arange(210, 302, 1),
    #             rotm_images_dict[i][0].slice((270, 210), (270, 302)), '.k')
    # plt.plot(np.arange(240, 272, 1), rm_(np.arange(240, 272, 1), grad_value,
    #                                      rotm_image.imsize[0]))
    # plt.errorbar(np.arange(210, 302, 1),
    #              rotm_image.slice((270, 210), (270, 302)),
    #              s_rotm_image.slice((270, 210), (270, 302)), fmt='.r', lw=4)
    # plt.axvline(rotm_image.pixref[1] - jet_width / 2.)
    # plt.axvline(rotm_image.pixref[1] + jet_width / 2.)
    # fig.show()



