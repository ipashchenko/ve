import os
import sys
import numpy as np
from spydiff import deep_clean_difmap
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
sys.path.insert(0, '/home/ilya/github/dterms')
from my_utils import find_image_std, find_bbox


data_dir = "/home/ilya/data/deep_clean"
uvfits = "2200+420.u.2019_01_19.uvf"


deep_clean_difmap(fname=uvfits, outfname="cc.fits", stokes="I", path=data_dir, outpath=data_dir,
                  mapsize_clean=(1024, 0.1), path_to_script="/home/ilya/github/ve/difmap/final_clean_rms",
                  show_difmap_output=False)

ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, "cc.fits"))
dimage = create_image_from_fits_file(os.path.join(data_dir, "dmap_cc.fits"))
beam = ccimage.beam
npixels_beam = np.pi*beam[0]*beam[1]/0.1**2
std = find_image_std(ccimage.image, beam_npixels=npixels_beam)
# blc, trc = find_bbox(ccimage.image, level=4*std, min_maxintensity_mjyperbeam=4*std,
#                      min_area_pix=2*npixels_beam, delta=10)
fig = iplot(ccimage.image, dimage.image, x=ccimage.x, y=ccimage.y,
            abs_levels=[3 * std], colors_mask=None, color_clim=None, #blc=blc, trc=trc,
            beam=beam, close=False, colorbar_label="Jy/beam", show_beam=True, show=True,
            cmap='jet', contour_color='black')

fig = iplot(dimage.image, x=dimage.x, y=dimage.y,
            min_abs_level=2*std, colors_mask=None, color_clim=None, #blc=blc, trc=trc,
            beam=beam, close=False, show_beam=True, show=True,
            contour_color='white', fig=fig, plot_colorbar=False)
# fig.savefig(os.path.join(data_dir, "deep_more.png"))