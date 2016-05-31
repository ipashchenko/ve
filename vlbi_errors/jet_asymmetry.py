import numpy as np
import os
import matplotlib.pyplot as plt
from spydiff import clean_difmap
from mojave import (get_all_mojave_sources, download_mojave_uv_fits,
                    get_epochs_for_source, mojave_uv_fits_fname)
from from_fits import create_clean_image_from_fits_file, \
    create_image_from_fits_file
from beam import CleanBeam
from utils import mas_to_rad
from image_ops import (pol_map, jet_direction, pol_mask, jet_ridge_line)


def map_fname(source, epoch, stokes, suffix=None):
    if suffix is None:
        result = "{}_{}_{}.fits".format(source, epoch, stokes)
    else:
        result = "{}_{}_{}_{}.fits".format(source, epoch, stokes, suffix)
    return result


# def jet_medial_axis(image):
#     from image_ops import rms_image
#     rms = rms_image(image)
#     mask = image.image < 5. * rms
#     data = image.image.copy()
#     data[mask] = 0
#     data[~mask] = 1
#     from scipy import ndimage as ndi
#     from skimage.morphology import medial_axis
#     import matplotlib.pyplot as plt
#     # Compute the medial axis (skeleton) and the distance transform
#     skel, distance = medial_axis(data, return_distance=True)
#     # Distance to the background for pixels of the skeleton
#     dist_on_skel = distance * skel
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,
#                                subplot_kw={'adjustable': 'box-forced'})
#     ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
#     ax1.axis('off')
#     ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
#     ax2.contour(data, [0.5], colors='w')
#     ax2.axis('off')
#
#     fig.tight_layout()
#     plt.show()


if __name__ == '__main__':
    n_rms_max = 10
    n_sigma_pol = 1
    base_dir = '/home/ilya/code/vlbi_errors/examples/mojave/asymmetry/test'
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    mapsize = (512, 0.1)
    source_epoch_dict = dict()
    import json
    fp = open('/home/ilya/code/vlbi_errors/vlbi_errors/source_epoch_dict.json',
              'r')
    source_epoch_dict = json.load(fp)
    source_epoch_dict = {str(source): str(epoch) for source, epoch in
                         source_epoch_dict.items()}
    sources = sorted(source_epoch_dict.keys())
    # for source in sources:
    #     print("Querying source {}".format(source))
    #     epochs = get_epochs_for_source(source, use_db='u')
    #     source_epoch_dict.update({source: epochs})

    for source in sources:
        print "Working with source {}".format(source)
        data_dir = os.path.join(base_dir, source)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        source_images = dict()
        download_mojave_uv_fits(source, epochs=[source_epoch_dict[source]],
                                bands=['u'], download_dir=data_dir)
        for epoch in [source_epoch_dict[source]]:
            print "Working with epoch {}".format(epoch)
            fname = mojave_uv_fits_fname(source, 'u', epoch)
            for stokes in ('I', 'Q', 'U'):
                # First restore with naitive beam to get it parameters
                print "Cleaning stokes {} with naitive restoring" \
                      " beam".format(stokes)
                clean_difmap(fname, map_fname(source, epoch, stokes), stokes,
                             mapsize, path=data_dir,
                             path_to_script=path_to_script, outpath=data_dir)
            i_image = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                     map_fname(source, epoch, 'I')))
            imsize = i_image.imsize
            # Get beam parameters
            beam = i_image.beam
            print "Beam : {} [mas, mas, deg]".format(beam)
            circ_beam = (beam[0], beam[0], 0)
            print "Using circular beam {} [mas, mas, deg]".format(circ_beam)

            # Now clean and restore with circular beam
            for stokes in ('I', 'Q', 'U'):
                print "Cleaning stokes {} with circular restoring" \
                      " beam".format(stokes)
                # First restore with naitive beam to get it parameters
                clean_difmap(fname, map_fname(source, epoch, stokes, 'circ'),
                             stokes, mapsize, path=data_dir,
                             path_to_script=path_to_script, outpath=data_dir,
                             beam_restore=circ_beam)
            i_image_circ =\
                create_image_from_fits_file(os.path.join(data_dir,
                                                         map_fname(source,
                                                                   epoch, 'I',
                                                                   'circ')))

            i_rms = i_image_circ.rms(region=(imsize[0] / 10., imsize[0] / 10.,
                                             imsize[0] / 10., None))
            print "I r.m.s. of circular-convolved map : {}".format(i_rms)

            # Calculate distance to most distant pixel with rms > 10 * rms
            mask = i_image_circ.image < n_rms_max * i_rms
            i_image_zeroed = i_image_circ.image.copy()
            i_image_zeroed[mask] = 0.
            y, x = np.nonzero(i_image_zeroed)
            y -= i_image.pixref[0]
            x -= i_image.pixref[1]
            distances = np.sqrt(x ** 2. + y ** 2.)
            max_dist = int(sorted(distances)[-1])
            print "Max. distance to n_rms = {} is {} pix.".format(n_rms_max,
                                                                  max_dist)

            # Creating PPOL image with circular beam
            print "Creating polarization maps"
            q_image = create_image_from_fits_file(os.path.join(data_dir,
                                                               map_fname(source,
                                                                         epoch,
                                                                         'Q',
                                                                         'circ')))
            u_image = create_image_from_fits_file(os.path.join(data_dir,
                                                               map_fname(source,
                                                                         epoch,
                                                                         'U',
                                                                         'circ')))
            stokes_image_dict = {'I': i_image_circ, 'Q': q_image, 'U': u_image}
            print "Masking polarization flux map at" \
                  " n_sigma = {}".format(n_sigma_pol)
            mask = pol_mask(stokes_image_dict, n_sigma=n_sigma_pol)
            p_image_circ = pol_map(q_image.image, u_image.image)
            p_image_circ[mask] = 0.

            # Calculate ridge line for I and PPOL
            print "Calcualting ridge-line for I"
            i_coords = np.atleast_2d(jet_ridge_line(i_image_circ.image,
                                                    max_dist))
            print "Calcualting ridge-line for P"
            p_coords = np.atleast_2d(jet_ridge_line(p_image_circ, max_dist))
            deviances = list()
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(i_image_circ.image)
            ax.scatter(i_coords[:, 1], i_coords[:, 0])
            ax.scatter(p_coords[:, 1], p_coords[:, 0], color='r')
            fig.show()
            fig.savefig(os.path.join(data_dir, 'ridges_i.png'))
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(p_image_circ)
            ax.scatter(i_coords[:, 1], i_coords[:, 0])
            ax.scatter(p_coords[:, 1], p_coords[:, 0], color='r')
            fig.show()
            fig.savefig(os.path.join(data_dir, 'ridges_p.png'))
            plt.close()
            print "Asymetry statistics :"
            astat_none = np.linalg.norm(i_coords - p_coords, ord=None)
            astat_0 = np.linalg.norm(i_coords - p_coords, ord=0)
            astat_pinf = np.linalg.norm(i_coords - p_coords, ord=np.inf)
            astat_minf = np.linalg.norm(i_coords - p_coords, ord=-np.inf)
