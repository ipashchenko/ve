import os
import math
import numpy as np
import glob
from from_fits import create_clean_image_from_fits_file
from image import find_shift
from knuth_hist import histogram
from utils import percent
import matplotlib
import matplotlib.pyplot as plt


data_dir = '/home/ilya/vlbi_errors/article/0952+179'
shifts = list()
shifts_values = list()
shifts_mask_size = dict()
for i in range(1, 2):
    files = sorted(glob.glob(os.path.join(data_dir,
                                          "cs_boot_*_I_cc_{}.fits".format(str(i).zfill(3)))))
    f15 = files[0]
    f5 = files[1]
    i15 = create_clean_image_from_fits_file(f15)
    i5 = create_clean_image_from_fits_file(f5)
    print("Finding shift between {} & {}".format(os.path.split(f15)[-1], os.path.split(f5)[-1]))
    beam_pxl = int(i15._beam.bmaj)

    shift = find_shift(i15, i5, max_shift=2 * beam_pxl, shift_step=1,
                       max_mask_r=5 * beam_pxl, mask_step=1,
                       upsample_factor=1000)
    shift_value = math.sqrt(shift[0]**2 + shift[1]**2)
    shifts.append(shift)
    shifts_values.append(shift_value)
    print("Found shift : {} with value {}".format(shift, shift_value))
    shifts_mask_size[i] = list()
    print("Found shift dependence on mask size for current bootstrap sample")
    for r in range(0, 100, 5):
        sh = i15.cross_correlate(i5, region1=(i15.x_c, i15.y_c, r, None),
                                 region2=(i5.x_c, i5.y_c, r, None),
                                 upsample_factor=1000)
        sh_value = math.sqrt(sh[0]**2 + sh[1]**2)
        print("r = {}, shift = {}".format(r, sh_value))
        shifts_mask_size[i].append(sh_value)


# np.savetxt('2230_shifts_300.txt', shifts)
hist_d, edges_d = histogram(shifts_values, normed=False)
lower_d = np.resize(edges_d, len(edges_d) - 1)
fig, ax = plt.subplots(1, 1)
ax.bar(lower_d, hist_d, width=np.diff(lower_d)[0], linewidth=1, color='#4682b4')
ax.axvline(x=percent(shifts_values, perc=16), color='r')
ax.axvline(x=percent(shifts_values, perc=84), color='r')
font = {'family': 'Droid Sans', 'weight': 'normal', 'size': 18}
matplotlib.rc('font', **font)
# matplotlib.rcParams.update({'font.size': 22})
ax.set_xlabel(ur"Shift values, [pixels]")
ax.set_ylabel(ur"Number of replications")
fig.savefig("shifts_histogram.png", bbox_inches='tight', dpi=200)
