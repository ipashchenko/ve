import os
import math
import glob
from from_fits import create_clean_image_from_fits_file
from image import find_shift


data_dir = '/home/ilya/vlbi_errors/article/2230+114'
shifts = list()
shifts_values = list()
for i in range(1, 100):
    files = sorted(glob.glob(os.path.join(data_dir,
                                          "cs_boot_*_I_cc_{}.fits".format(str(i).zfill(3)))))
    f15 = files[1]
    f8 = files[2]
    i15 = create_clean_image_from_fits_file(f15)
    i8 = create_clean_image_from_fits_file(f8)
    print("Finding shift between {} & {}".format(os.path.split(f15)[-1], os.path.split(f8)[-1]))
    beam_pxl = int(i15._beam.bmaj)

    shift = find_shift(i15, i8, max_shift=beam_pxl, shift_step=1,
                       max_mask_r=beam_pxl, mask_step=1, uspsample_factor=1000)
    shift_value = math.sqrt(shift[0]**2 + shift[1]**2)
    shifts.append(shift)
    shifts_values.append(shift_value)
    print("Found shift : {} with value {}".format(shift, shift_value))
