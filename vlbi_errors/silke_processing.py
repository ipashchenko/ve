import os
import glob
from astropy.io.fits import VerifyError
from mojave import download_mojave_uv_fits
from mojave import mojave_uv_fits_fname
from bootstrap import bootstrap_uvfits_with_difmap_model
import matplotlib
label_size = 16
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size


def convert_to_single_number(file_path, data_dir='/home/ilya/Dropbox/silke'):
    file_dir, fname = os.path.split(file_path)
    year = fname[:-4].split('_')[1]
    month = fname[:-4].split('_')[2]
    day = fname[:-4].split('_')[3]
    outfname = '{}_{}_{}_errors.txt'.format(year, month, day)
    with open(os.path.join(data_dir, fname), 'r') as fo:
        lines = fo.readlines()
        params = list()
        with open(os.path.join(data_dir, outfname), 'w') as of:
            for line in lines:
                if not line.startswith('#'):
                    param = line.rstrip().split()[0]
                    orig_value = line.rstrip().split()[1]
                    std = 0.5 * (float(line.rstrip().split()[-2]) +
                                 float(line.rstrip().split()[-1]))
                    of.write("{} {} {}".format(param, orig_value, std))
                    of.write("\n")

txt_file_dir = '/home/ilya/Dropbox/silke'
out_files = glob.glob(os.path.join(txt_file_dir, 'errors_*.dml'))

# Mass download uv-data for our epochs
# original_dfm_models = glob.glob(os.path.join(data_dir, '*us'))
# for path in original_dfm_models:
#     fname = os.path.split(path)[-1]
#     epoch = fname[:-2]
#     download_mojave_uv_fits('0851+202', epochs=[epoch], download_dir=data_dir)


data_dir = '/home/ilya/silke'
boot_dir = '/home/ilya/silke/boot'
txt_file_dir = '/home/ilya/Dropbox/silke'
original_dfm_models = glob.glob(os.path.join(data_dir, '*us'))
epochs_ready = glob.glob(os.path.join(txt_file_dir, 'errors_*.png'))
epochs_ready_ = list()
for epoch_ready in epochs_ready:
    dir, fname = os.path.split(epoch_ready)
    epochs_ready_.append(fname[7:-4])
for path in original_dfm_models:
    fname = os.path.split(path)[-1]
    epoch = fname[:-2]
    print "Processing epoch : {}".format(epoch)
    if epoch in epochs_ready_:
        print "Skipping epoch {}".format(epoch)
        continue
    original_model_fname = fname
    original_model_path = os.path.join(data_dir, original_model_fname)
    uv_fits_fname = mojave_uv_fits_fname('0851+202', 'u', epoch)
    uv_fits_path = os.path.join(data_dir, uv_fits_fname)

    out_txt_file = os.path.join(txt_file_dir, 'errors_{}.dml'.format(epoch))
    out_png_file = os.path.join(txt_file_dir, 'errors_{}.png'.format(epoch))
    try:
        bootstrap_uvfits_with_difmap_model(uv_fits_path, original_model_path,
                                           n_boot=300, boot_dir=boot_dir,
                                           out_txt_file=out_txt_file,
                                           out_plot_file=out_png_file,
                                           clean_after=True)
    except IOError:
        with open(os.path.join(txt_file_dir, '{}_io_error.txt'.format(epoch)), 'w'):
            print "IO Error"
        continue
    except VerifyError:
        with open(os.path.join(txt_file_dir, '{}_verify_error.txt'.format(epoch)), 'w'):
            print "Verify Error"
        bootstrapped_uv_fits = sorted(glob.glob(os.path.join(boot_dir,
                                                             'bootstrapped_data*.fits')))
        for file_ in bootstrapped_uv_fits:
            os.unlink(file_)
        continue
