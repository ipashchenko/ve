import os
import numpy as np
from uv_data import UVData
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file
from model import Model
from components import CGComponent
from spydiff import clean_difmap, selfcal_difmap, import_difmap_model, export_difmap_model, modelfit_difmap
import sys
sys.path.insert(0, '/home/ilya/github/dterms')
from my_utils import find_image_std, find_bbox
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


# def find_corrections(uvdata_raw, uvdata):
#     for data in uvdata.hdu.data:


original_selfcaled_uvf = "0125+487_S.uvf_difmap"
original_raw_uvf = "0125+487_S.uvf_raw_edt"
data_dir = "/home/ilya/data/selfcal"
mapsize_clean = 256, 0.5


uvdata_sc = UVData(os.path.join(data_dir, original_selfcaled_uvf))
uvdata_raw = UVData(os.path.join(data_dir, original_raw_uvf))
uvdata_template = UVData(os.path.join(data_dir, original_raw_uvf))


clean_difmap(fname=original_selfcaled_uvf, outfname="cc_original_selfcaled.fits",
             stokes="I", path=data_dir, outpath=data_dir, mapsize_clean=mapsize_clean,
             path_to_script="/home/ilya/github/ve/difmap/final_clean_nw",
             show_difmap_output=True)

original_selfcaled_image = create_clean_image_from_fits_file(os.path.join(data_dir, "cc_original_selfcaled.fits"))
beam = original_selfcaled_image.beam
npixels_beam = np.pi*beam[0]*beam[1]/mapsize_clean[1]**2
std = find_image_std(original_selfcaled_image.image, beam_npixels=npixels_beam)
blc, trc = find_bbox(original_selfcaled_image.image, level=4*std, min_maxintensity_mjyperbeam=4*std,
                     min_area_pix=2*npixels_beam, delta=10)
fig = iplot(original_selfcaled_image.image, original_selfcaled_image.image,
            x=original_selfcaled_image.x, y=original_selfcaled_image.y,
            min_abs_level=2 * std, colors_mask=None, color_clim=None, blc=blc, trc=trc,
            beam=beam, close=False, colorbar_label="Jy/beam", show_beam=True, show=True,
            cmap='viridis', contour_color='white')

fig = iplot(original_selfcaled_image.image,
            x=original_selfcaled_image.x, y=original_selfcaled_image.y,
            abs_levels=[2*std], colors_mask=None, color_clim=None, blc=blc, trc=trc,
            beam=beam, close=False, show_beam=True, show=True,
            cmap='viridis', contour_color='red', fig=fig)


fig.savefig(os.path.join(data_dir, "original_selfcaled.png"), dpi=300, bbox_inches="tight")
plt.close()


# Find gains products
corrections = uvdata_raw.uvdata/uvdata_sc.uvdata

# Create artificial raw data with known sky model and given corrections
cg = CGComponent(0.5, 0, 0, 0.5)
model = Model(stokes="RR")
model.add_component(cg)
noise = uvdata_template.noise(use_V=False)
uvdata_template.substitute([model])
uvdata_template.uvdata = uvdata_template.uvdata*corrections
uvdata_template.noise_add(noise)
uvdata_template.save(os.path.join(data_dir, "artificial.uvf"), rewrite=True, downscale_by_freq=True)
uvdata_artificial_raw = UVData(os.path.join(data_dir, "artificial.uvf"))


# # Self-calibrate
# selfcal_difmap(fname="artificial.uvf", outfname="artificial_selfcaled.uvf", mapsize_clean=mapsize_clean, solint=30,
#                path=data_dir, path_to_script="/home/ilya/github/ve/difmap/auto_selfcal", outpath=data_dir,
#                show_difmap_output=True, clean_box=None)
# uvdata_artificial_selfcaled = UVData(os.path.join(data_dir, "artificial_selfcaled.uvf"))
#
# # Image
# clean_difmap(fname="artificial_selfcaled.uvf", outfname="cc_artificial_selfcaled.fits",
#              stokes="I", path=data_dir, outpath=data_dir, mapsize_clean=mapsize_clean,
#              path_to_script="/home/ilya/github/ve/difmap/final_clean_nw",
#              show_difmap_output=True)
#
# artificial_selfcaled_image = create_clean_image_from_fits_file(os.path.join(data_dir, "cc_artificial_selfcaled.fits"))
# beam = original_selfcaled_image.beam
# npixels_beam = np.pi*beam[0]*beam[1]/mapsize_clean[1]**2
# std = find_image_std(original_selfcaled_image.image, beam_npixels=npixels_beam)
# blc, trc = find_bbox(original_selfcaled_image.image, level=4*std, min_maxintensity_mjyperbeam=4*std,
#                      min_area_pix=2*npixels_beam, delta=10)
# fig = iplot(artificial_selfcaled_image.image, artificial_selfcaled_image.image,
#             x=artificial_selfcaled_image.x, y=artificial_selfcaled_image.y,
#             min_abs_level=2 * std, colors_mask=None, color_clim=None, blc=blc, trc=trc,
#             beam=beam, close=False, colorbar_label="Jy/beam", show_beam=True, show=True,
#             cmap='viridis', contour_color='white')
# fig.savefig(os.path.join(data_dir, "artificial_selfcaled.png"), dpi=300, bbox_inches="tight")
# plt.close()
#
# # Create template model file
# export_difmap_model([cg], os.path.join(data_dir, "template.mdl"), uvdata_template.frequency/10**9)
# # Modelfit artificial self-calibrated data
# modelfit_difmap("artificial_selfcaled.uvf", "template.mdl", "artificial.mdl", niter=50, stokes='rr',
#                 path=data_dir, mdl_path=data_dir, out_path=data_dir, show_difmap_output=True)
# cg_new = import_difmap_model("artificial.mdl", data_dir)[0]
# print(cg_new.p)
#
# u = uvdata_template.uv[:, 0]
# v = uvdata_template.uv[:, 1]
#
# plt.scatter(u, v, c=np.angle(uvdata_artificial_raw.uvdata/uvdata_artificial_selfcaled.uvdata)[:, 0, 0])
# plt.colorbar()
# plt.show()