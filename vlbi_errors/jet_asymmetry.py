import numpy as np
import os
import matplotlib.pyplot as plt
from spydiff import clean_difmap
from mojave import (get_all_mojave_sources, download_mojave_uv_fits,
                    get_epochs_for_source, mojave_uv_fits_fname)
from from_fits import create_clean_image_from_fits_file
from beam import CleanBeam
from utils import mas_to_rad
from image_ops import (pol_map, jet_direction, pol_mask)


def map_fname(source, epoch, stokes):
    return "{}_{}_{}.fits".format(source, epoch, stokes)


if __name__ == '__main__':
    base_dir = '/home/ilya/code/vlbi_errors/examples/mojave/asymmetry'
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    mapsize = (512, 0.1)
    sources = get_all_mojave_sources()
    source_epoch_dict = dict()
    import json
    fp = open('/home/ilya/code/vlbi_errors/vlbi_errors/source_epoch_dict.json',
              'r')
    source_epoch_dict = json.load(fp)
    source_epoch_dict = {str(source): str(epoch) for source, epoch in source_epoch_dict.items()}
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
        for epoch in [source_epoch_dict[source]]:
            fname = mojave_uv_fits_fname(source, 'u', epoch)
            for stokes in ('I', 'Q', 'U'):
                clean_difmap(fname, map_fname(source, epoch, stokes), stokes,
                             mapsize, path=data_dir,
                             path_to_script=path_to_script, outpath=data_dir)
            i_image = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                     map_fname(source, epoch, 'I')))
            imsize = i_image.imsize
            i_rms = i_image.rms(region=(imsize[0] / 10., imsize[0] / 10., imsize[0] / 10., None))

            # Calculate distance to most distant pixel with rms > 7 * rms
            mask = i_image.image < 10. * i_rms
            i_image_zeroed = i_image.image.copy()
            i_image_zeroed[mask] = 0.
            y, x = np.nonzero(i_image_zeroed)
            y -= i_image.pixref[0]
            x -= i_image.pixref[1]
            distances = np.sqrt(x ** 2. + y ** 2.)
            max_dist = int(sorted(distances)[-1])

            beam = i_image.beam
            pixsize = abs(i_image.pixsize[0]) / mas_to_rad
            beam = (beam[0] / pixsize, beam[1] / pixsize, beam[2])
            print beam
            circ_beam = CleanBeam()
            circ_beam._construct(bmaj=beam[0], bmin=beam[0], bpa=0.,
                                 imsize=imsize)
            i_image_circ = i_image.convolve(circ_beam.image)
            i_image_circ[mask] = 0.
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(i_image_circ)
            fig.show()

            # Creating PPOL image with circular beam
            q_image = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                     map_fname(source,
                                                                               epoch,
                                                                               'Q')))
            u_image = create_clean_image_from_fits_file(os.path.join(data_dir,
                                                                     map_fname(source,
                                                                               epoch,
                                                                               'U')))
            stokes_image_dict = {'I': i_image, 'Q': q_image, 'U': u_image}
            mask = pol_mask(stokes_image_dict, n_sigma=3)
            q_image_circ = q_image.convolve(circ_beam.image)
            u_image_circ = u_image.convolve(circ_beam.image)
            p_image_circ = pol_map(q_image_circ, u_image_circ)
            p_image_circ[mask] = 0.
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(p_image_circ)
            fig.show()

            # Calculate ridge line for I and PPOL
            i_rs, i_phis, i_fluxes = jet_direction(i_image_circ, rmin=beam[0],
                                                   rmax=max_dist, dr=2)
            p_rs, p_phis, p_fluxes = jet_direction(p_image_circ, rmin=beam[0],
                                                   rmax=max_dist, dr=2)
            print i_rs, p_rs
            print i_phis, p_phis
            print i_fluxes, p_fluxes
            deviances = list()
            for i, r in enumerate(i_rs):
                dr = r * np.sqrt(np.cos(i_phis[i]) * np.cos(p_phis[i]) +
                                 np.sin(i_phis[i]) * np.sin(p_phis[i]))
                deviances.append(dr)
            np.savetxt(os.path.join(data_dir, "{}_{}.txt".format(source,
                                                                 epoch)),
                       np.dstack((i_rs, np.array(deviances))))
