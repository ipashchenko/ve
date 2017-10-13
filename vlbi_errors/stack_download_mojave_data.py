from mojave import download_mojave_uv_fits
from automodel import automodel_uv_fits


# save_dir = '/home/ilya/fs/sshfs/frb/data'
# with open('/home/ilya/Dropbox/stack/sources', 'r') as fo:
#     lines = fo.readlines()
#
# lines = lines[1:]
# sources = list()
# for line in lines:
#     sources.append(line.strip('\n').split(" ")[0])
#
#
# for source in sources:
#     download_mojave_uv_fits(source, bands=['u'], download_dir=save_dir)


uv_fits_path = "/home/ilya/fs/sshfs/frb/data/0235+164.u.2004_08_28.uvf"
best_model_file = automodel_uv_fits(uv_fits_path, "/home/ilya/STACK",
                                    n_max_comps=40, mapsize_clean=(512, 0.1))