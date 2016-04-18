import os
import numpy as np
from simulations import simulate
from mojave import download_mojave_uv_fits, mojave_uv_fits_fname
from spydiff import clean_difmap
from from_fits import create_clean_image_from_fits_file
from utils import mas_to_rad

# TODO: Get typical MOJAVE fluxes
# TODO: Get typical MOJAVE jet widths
if __name__ == '__main__':

    ############################################################################
    # Test simulate
    from mojave import get_epochs_for_source
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    base_dir = '/home/ilya/vlbi_errors/mojave_rm'
    sources = ['1514-241', '1302-102', '0754+100', '0055+300', '0804+499',
               '1749+701', '0454+844']
    bands = ['x', 'y', 'j', 'u']
    mapsize_dict = {'x': (512, 0.1), 'y': (512, 0.1), 'j': (512, 0.1),
                    'u': (512, 0.1)}
    mapsize_common = (512, 0.1)
    source_epoch_dict = dict()
    for source in sources:
        epochs = get_epochs_for_source(source, use_db='multifreq')
        print "Found epochs for source {}".format(source)
        print epochs
        source_epoch_dict.update({source: epochs[-1]})
    # for source in sources:
    #     print "Simulating source {}".format(source)
    #     simulate(source, source_epoch_dict[source], ['x', 'y', 'j', 'u'],
    #              n_sample=3, rotm_clim=[-200, 200],
    #              path_to_script=path_to_script, mapsize_dict=mapsize_dict,
    #              mapsize_common=mapsize_common, base_dir=base_dir,
    #              rotm_value_0=0., max_jet_flux=0.005)

    for source, epoch in source_epoch_dict.items():
        data_dir = os.path.join(base_dir, source)
        # First, download lowest frequency band and find beam width
        download_mojave_uv_fits(source, epochs=[epoch], bands=[bands[0]],
                                download_dir=data_dir)
        uv_fits_fname = mojave_uv_fits_fname(source, bands[0], epoch)
        cc_fits_fname = "{}_{}_{}_{}_naitive_cc.fits".format(source, epoch,
                                                             bands[0], 'I')
        clean_difmap(uv_fits_fname, cc_fits_fname, 'I',
                     mapsize_dict[bands[0]], path=data_dir,
                     path_to_script=path_to_script, outpath=data_dir)
        cc_image = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                  cc_fits_fname))
        # Beam in mas, mas, deg
        beam = cc_image.beam
        print "BW [mas, mas, deg] : ", beam
        # pixel size in mas
        pixsize = abs(cc_image.pixsize[0]) / mas_to_rad
        # This is characteristic jet width in pixels (half of that)
        width = 0.5 * (beam[0] + beam[1]) / pixsize
        print "BW scale [pixels] : ", width

        # Now find flux at distance > 1 BW from core & maximum flux at core
        len_y = cc_image.imsize[0]
        len_x = cc_image.imsize[1]
        y, x = np.mgrid[-len_y/2: len_y/2, -len_x/2: len_x/2]
        r = beam[0] / pixsize
        dr = 1
        mask = np.logical_and(r**2 <= x*x+y*y, x*x+y*y <= (r+dr)**2)
        image_circle = cc_image.image[mask]
        max_circle_flux = np.max(image_circle)
        print "Maximum flux at 1 BW from core : ", max_circle_flux
        max_flux = np.max(cc_image.image)
        print "Maximum flux : ", max_flux



        simulate(source, epoch, bands,
                 n_sample=100, max_jet_flux=0.003, rotm_clim_sym=[-300, 300],
                 rotm_clim_model=[-900, 900],
                 path_to_script=path_to_script, mapsize_dict=mapsize_dict,
                 mapsize_common=mapsize_common, base_dir=data_dir,
                 rotm_value_0=0., rotm_grad_value=0., n_rms=3.,
                 download_mojave=True, spix_clim_sym=[-1, 1])

    # ############################################################################
    # # Test for ModelGenerator
    # # Create jet model, ROTM & alpha images
    # imsize = (512, 512)
    # center = (256, 256)
    # # from `y`  band
    # pixsize = 4.848136191959676e-10
    # x, y = create_grid(imsize)
    # x -= center[0]
    # y -= center[1]
    # x *= pixsize
    # y *= pixsize
    # max_jet_flux = 0.01
    # qu_fraction = 0.1
    # rotm_grad_value = 40.
    # rotm_value_0 = 0.
    # model_freq = 20. * 10. ** 9.
    # jet_image = create_jet_model_image(30, 60, 10, max_jet_flux,
    #                                    (imsize[0], imsize[0]),
    #                                    (imsize[0] / 2, imsize[0] / 2),
    #                                    gauss_peak=0.001, dist_from_core=20,
    #                                    cut=0.0002)
    # rotm_image = rotm((imsize[0], imsize[0]),
    #                   (imsize[0] / 2, imsize[0] / 2),
    #                   grad_value=rotm_grad_value, rm_value_0=rotm_value_0)
    # alpha_image = alpha((imsize[0], imsize[0]),
    #                     (imsize[0] / 2, imsize[0] / 2), 0.)
    # stokes_models = {'I': jet_image, 'Q': qu_fraction * jet_image,
    #                  'U': qu_fraction * jet_image}
    # mod_generator = ModelGenerator(stokes_models, x, y, rotm=rotm_image,
    #                                alpha=alpha_image, freq=model_freq)
    # images = mod_generator.get_stokes_images(frequency=8.*10**9.,
    #                                          i_cut_frac=0.01)
    # pol_images = mod_generator.get_pol_maps(frequency=8.*10**9.,
    #                                         i_cut_frac=0.01)
    # frequencies = [5. * 10 ** 9, 8. * 10 ** 9, 12. * 10 ** 9, 15. * 10 ** 9]
    # mask = mod_generator.create_i_mask(frequencies[-1], i_cut_frac=0.01)
    # rm_map, s_rm_map, chsq_map = mod_generator.get_rotm_map(frequencies,
    #                                                         rotm_mask=mask)
    # sp_map, s_sp_map, chsq_map = mod_generator.get_spix_map(frequencies,
    #                                                         spix_mask=mask)
