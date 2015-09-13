import os
import glob
import numpy as np
from from_fits import (get_fits_image_info, create_uvdata_from_fits_file)
from image import (BasicImage, Image, CleanImage)
from utils import (mask_region, mas_to_rad, find_card_from_header)
from model import Model
from components import DeltaComponent
from images import Images
from spydiff import clean_difmap


def simulate_grad(low_freq_map, high_freq_map, uvdata_files, cc_flux,
                  outpath, grad_value, width, length, k):
    """
    Function that simulates ROTM gradients in uv-data.

    :param low_freq_map:
        Path to FITS-file with clean map for lowest frequency.
    :param high_freq_map:
        Path to FITS-file with clean map for highest frequency.
    :param uvdata_files:
        Iterable of paths to FITS-files with uv-data.
    :param cc_flux:
        Flux of CC-components that will model the polarization [Jy]. That CCs
        will be used in Q & U clean models. Being convolved with beam it lowers
        the maximum flux by factor ``pi * beam_width ** 2``.
    :param outpath:
        Path where to save all resulting files.
    :param grad_value:
        Value of ROTM gradient [rad/m/m/width], where ``width`` - width of the
        model jet.
    :param width:
        Width of model jet in units of the lowest frequency beam's major axis.
    :param length:
        Length of model jet in units of the lowest frequency beam's major axis.
    :param k:
        How many model image pixels should be in one highest frequency image
        pixel?

    :return:
        Creates FITS-files with uv-data where cross-polarization data is
        substituted by data with ROTM gradient. Also creates FITS file with
        model image of ROTM gradient - that is ``CleanImage`` instance with
        ``_image`` attribute - unconvolved model of gradient (``true image``)
        and ``image`` - model, convolved with beam of lowest frequency original
        map.

    """

    (imsize_h, pixref_h, pixrefval_h, (bmaj_h, bmin_h, bpa_h,), pixsize_h,
     stokes_h, freq_h) = get_fits_image_info(high_freq_map)
    (imsize_l, pixref_l, pixrefval_l, (bmaj_l, bmin_l, bpa_l,), pixsize_l,
     stokes_l, freq_l) = get_fits_image_info(low_freq_map)

    # new pixsize
    pixsize = (abs(pixsize_h[0]) / k, abs(pixsize_h[1]) / k)
    # new imsize
    x1 = imsize_l[0] * abs(pixsize_l[0]) / abs(pixsize[0])
    x2 = imsize_l[1] * abs(pixsize_l[1]) / abs(pixsize[1])
    imsize = (int(x1 - x1 % 2),
              int(x2 - x2 % 2))
    # new pixref
    pixref = (imsize[0]/2, imsize[1]/2)
    # FIXME: Should i use ellipse beam for comparing model with results?
    # Beam width in new pixels
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

    # Flux should decline with x (or y) by linear law
    def flux(x, y, max_flux, length):
        return max_flux - (max_flux/length) * (x - pixref[0])


    # Zero Rm in center and constant gradient ``2 * max_rm/width``.
    def rm(x, y, max_rm, width):
        k = max_rm / (width / 2.)
        return k * (x - imsize[0] / 2) + k * imsize[0] / 2

    # Create map of ROTM
    print "Creating ROTM image with gradient..."
    max_rm = 200.
    image_rm = Image(imsize=imsize, pixsize=pixsize, pixref=pixref)
    image_rm._image = rm(image.x/abs(pixsize[0]), image.y/abs(pixsize[1]),
                         max_rm, jet_width)
    # Create ROTM image with size as for lowest freq. and pixel size - as for
    # highest freq. map
    # FIXME: Convolution enhances ROTM values
    save_rm = CleanImage(imsize=imsize_l, pixsize=pixsize_h, pixref=pixref_l,
                         bmaj=bmaj_l, bmin=bmin_l, bpa=bpa_l)
    save_rm._image = rm(save_rm.x/abs(pixsize_h[0]),
                        save_rm.y/abs(pixsize_h[1]), max_rm,
                        width * bmaj_l / abs(pixsize_h[0]))
    print "Saving image of ROTM gradient..."
    np.savetxt('RM_grad_image.txt', save_rm.image)


    # Create model instance and fill it with components
    model_q = Model(stokes='Q')
    model_u = Model(stokes='U')

    max_flux = cc_flux / (np.pi * beam_width ** 2)
    comps_q = [DeltaComponent(flux(x, y, max_flux, jet_length),
                              image.x[x, y]/mas_to_rad,
                              image.y[x, y]/mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    comps_u = [DeltaComponent(flux(x, y, max_flux, jet_length),
                              image.x[x, y]/mas_to_rad,
                              image.y[x, y]/mas_to_rad) for (x, y), value in
               np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
    print "Adding components to Q&U models..."
    model_q.add_components(*comps_q)
    model_u.add_components(*comps_u)

    image_q = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='Q',
                         freq=freq_l)
    image_u = Image(imsize=imsize, pixsize=pixsize, pixref=pixref, stokes='U',
                         freq=freq_l)
    image_q.add_model(model_q)
    image_u.add_model(model_u)

    # Create PANG map & plot it
    images = Images()
    images.add_images([image_q, image_u])
    print "Creating PPOL image..."
    ppol_image = images.create_pol_images(convolved=False)[0]

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
        print "Creating arrays of Q & U for frequency {} Hz".format(freq)
        q_array = ppol_image._image * np.cos(2. * (1. + 2. * image_rm._image *
                                                   lambda_sq))
        u_array = ppol_image._image * np.sin(2. * (1. + 2. * image_rm._image *
                                                   lambda_sq))
        image_q = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
                             stokes='Q', freq=freq)
        image_q._image = q_array
        image_u = Image(imsize=imsize, pixsize=pixsize, pixref=pixref,
                             stokes='U', freq=freq)
        image_u._image = u_array

        model_q = Model(stokes='Q')
        model_u = Model(stokes='U')
        print "Creating components of Q&U for frequency {} Hz".format(freq)
        comps_q = [DeltaComponent(image_q._image[x, y],
                                  image.x[x, y]/mas_to_rad,
                                  image.y[x, y]/mas_to_rad) for (x, y), value in
                   np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        comps_u = [DeltaComponent(image_u._image[x, y],
                                  image.x[x, y]/mas_to_rad,
                                  image.y[x, y]/mas_to_rad) for (x, y), value in
                   np.ndenumerate(jet_region) if not jet_region.mask[x, y]]
        print "Adding components of Q&U for frequency {} Hz models".format(freq)
        model_q.add_components(*comps_q)
        model_u.add_components(*comps_u)

        # Substitute Q&U models to uv-data and add noise
        print "Substituting Q&U models in uv-data, adding noise, saving for" \
              " frequency {} Hz".format(freq)
        noise = uvdata.noise(average_freq=True)
        uvdata.substitute([model_q, model_u])
        uvdata.noise_add(noise)
        # Save uv-data to file
        uv_save_fname = os.path.join(outpath,
                                     'simul_uv_{}_Hz.fits'.format(freq))
        if os.path.exists(uv_save_fname):
            print "Deleting existing file: {}".format(uv_save_fname)
            os.remove(uv_save_fname)
        uvdata.save(uvdata.data, uv_save_fname)


if __name__ == '__main__':
    high_freq_map =\
        '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/im/I/cc.fits'
    low_freq_map =\
        '/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/im/I/cc.fits'
    uvdata_files =\
        ['/home/ilya/vlbi_errors/0952+179/2007_04_30/C1/uv/sc_uv.fits',
         '/home/ilya/vlbi_errors/0952+179/2007_04_30/C2/uv/sc_uv.fits',
         '/home/ilya/vlbi_errors/0952+179/2007_04_30/X1/uv/sc_uv.fits',
         '/home/ilya/vlbi_errors/0952+179/2007_04_30/X2/uv/sc_uv.fits']
    cc_flux = 0.05
    outpath = '/home/ilya/vlbi_errors/simdata/'
    width = 0.5
    length = 3.
    k = 3
    simulate_grad(low_freq_map, high_freq_map, uvdata_files, cc_flux, outpath,
                  100., width, length, k)

    # Now calculate ROTM image using simulated data
    path_to_script = '/home/ilya/code/vlbi_errors/data/zhenya/clean/final_clean_nw'
    map_info_l = get_fits_image_info(low_freq_map)
    map_info_h = get_fits_image_info(high_freq_map)
    beam_restore = map_info_l[3]
    mapsize_clean = (map_info_h[0][0],
                     map_info_h[-3][0] / mas_to_rad)
    uvfiles_wc = os.path.join(outpath, "simul_uv_*")
    for uvfile in glob.glob(uvfiles_wc):
        print "Cleaning uv file {}".format(uvfile)
        uvdata = create_uvdata_from_fits_file(uvfile)
        freq_card = find_card_from_header(uvdata._io.hdu.header,
                                          value='FREQ')[0]
        # Frequency in Hz
        freq = uvdata._io.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        for stoke in ('i', 'q', 'u'):
            print "Stokes {}".format(stoke)
            clean_difmap(uvfile, "simul_{}_{}_cc.fits".format(stoke, freq),
                         stoke, mapsize_clean, path=outpath,
                         path_to_script=path_to_script,
                         beam_restore=beam_restore, outpath=outpath)
