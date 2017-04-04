import os
import numpy as np
import image_ops
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
from skel_utils import (isolateregions, pix_identify, init_lengths, pre_graph,
                        longest_path, prune_graph, extremum_pts, main_length,
                        make_final_skeletons, recombine_skeletons)
from mojave import (get_all_mojave_sources, download_mojave_uv_fits,
                    mojave_uv_fits_fname)
from spydiff import clean_difmap
from utils import mas_to_rad


def map_fname(source, epoch, stokes):
    return "{}_{}_{}.fits".format(source, epoch, stokes)


base_dir = '/home/ilya/vlbi_errors/asymmetry'
path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
mapsize = (512, 0.1)
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
    data_dir = os.path.join(base_dir, source)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    source_images = dict()
    download_mojave_uv_fits(source, epochs=[source_epoch_dict[source]],
                            bands=['u'], download_dir=data_dir)
    epoch = source_epoch_dict[source]
    fname = mojave_uv_fits_fname(source, 'u', epoch)
    stokes = 'I'
    cc_fits = map_fname(source, epoch, stokes)
    clean_difmap(fname, cc_fits, stokes, mapsize, path=data_dir,
                 path_to_script=path_to_script, outpath=data_dir)
    i_image = create_clean_image_from_fits_file(os.path.join(data_dir, cc_fits))
    # imsize = i_image.imsize
    # i_rms = i_image.rms(region=(imsize[0] / 10., imsize[0] / 10., imsize[0] / 10., None))

    # # Calculate distance to most distant pixel with rms > 7 * rms
    # mask = i_image.image < 10. * i_rms
    # i_image_zeroed = i_image.image.copy()
    # i_image_zeroed[mask] = 0.
    # y, x = np.nonzero(i_image_zeroed)
    # y -= i_image.pixref[0]
    # x -= i_image.pixref[1]
    # distances = np.sqrt(x ** 2. + y ** 2.)
    # max_dist = int(sorted(distances)[-1])

    beam = i_image.beam
    pixsize = abs(i_image.pixsize[0]) / mas_to_rad
    beam = (beam[0] / pixsize, beam[1] / pixsize, beam[2])

# cc_fits = '/home/ilya/vlbi_errors/examples/X/1226+023/I/boot/68/original_cc.fits'
# cc_fits = '/home/ilya/vlbi_errors/examples/L/1038+064/rms/68/original_cc.fits'
# cc_fits = '/home/ilya/vlbi_errors/examples/L/1633+382/rms/68/original_cc.fits'
    image = create_image_from_fits_file(os.path.join(data_dir, cc_fits))
    rms = image_ops.rms_image(image)
    data = image.image.copy()
    from scipy.ndimage.filters import gaussian_filter
    data = gaussian_filter(data, 5)
    mask = data < 3. * rms
    data[mask] = 0
    data[~mask] = 1

    skel, distance = medial_axis(data, return_distance=True)
    dist_on_skel = distance * skel

    # Plot area and skeleton
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})
    ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.contour(data, [0.5], colors='w')
    ax2.axis('off')
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(data_dir, 'skeleton_orig.png'))
    plt.close()

    isolated_filaments, num, offsets = isolateregions(skel)

    interpts, hubs, ends, filbranches, labeled_fil_arrays =\
        pix_identify(isolated_filaments, num)

    branch_properties = init_lengths(labeled_fil_arrays, filbranches, offsets, data)
    branch_properties["number"] = filbranches

    edge_list, nodes = pre_graph(labeled_fil_arrays, branch_properties, interpts,
                                 ends)

    max_path, extremum, G = longest_path(edge_list, nodes, verbose=True,
                                         save_png=False,
                                         skeleton_arrays=labeled_fil_arrays)

    updated_lists = prune_graph(G, nodes, edge_list, max_path, labeled_fil_arrays,
                                branch_properties, length_thresh=20,
                                relintens_thresh=0.1)
    labeled_fil_arrays, edge_list, nodes,  branch_properties = updated_lists

    filament_extents = extremum_pts(labeled_fil_arrays, extremum, ends)

    length_output = main_length(max_path, edge_list, labeled_fil_arrays, interpts,
                                branch_properties["length"], 1, verbose=True)
    filament_arrays = {}
    lengths, filament_arrays["long path"] = length_output
    lengths = np.asarray(lengths)

    filament_arrays["final"] = make_final_skeletons(labeled_fil_arrays, interpts,
                                                    verbose=True)

    skeleton = recombine_skeletons(filament_arrays["final"], offsets, data.shape,
                                   0, verbose=True)
    skeleton_longpath = recombine_skeletons(filament_arrays["long path"], offsets,
                                            data.shape, 1)
    skeleton_longpath_dist = skeleton_longpath * distance

    # Plot area and skeleton
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})
    ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(skeleton_longpath_dist, cmap=plt.cm.spectral,
               interpolation='nearest')
    ax2.contour(data, [0.5], colors='w')
    ax2.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(data_dir, 'skeleton.png'))
    plt.show()
    plt.close()
