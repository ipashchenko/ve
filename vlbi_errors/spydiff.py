import glob
import os
import sys
import datetime
import numpy as np
import copy
from utils import degree_to_rad, baselines_2_ants
from components import DeltaComponent, CGComponent, EGComponent
from astropy.stats import mad_std
from astropy.wcs import WCS
from astropy.io.fits import getheader, getdata
import astropy.io.fits as pf
import astropy.units as u
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy.stats import normaltest, anderson
from skimage.measure import regionprops
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
import calendar
from uv_data import UVData
from astropy.time import Time

months_dict = {v: k for k, v in enumerate(calendar.month_abbr)}
months_dict_inv = {k: v for k, v in enumerate(calendar.month_abbr)}


def check_bbox(blc, trc, image_size):
    """
    :note:
        This can make quadratic image rectangular.
    """
    # If some bottom corner coordinate become negative
    blc = list(blc)
    trc = list(trc)
    if blc[0] < 0:
        blc[0] = 0
    if blc[1] < 0:
        blc[1] = 0
    # If some top corner coordinate become large than image size
    if trc[0] > image_size:
        delta = abs(trc[0]-image_size)
        blc[0] -= delta
        # Check if shift have not made it negative
        if blc[0] < 0 and trc[0] > image_size:
            blc[0] = 0
        trc[0] -= delta
    if trc[1] > image_size:
        delta = abs(trc[1]-image_size)
        blc[1] -= delta
        # Check if shift have not made it negative
        if blc[1] < 0 and trc[1] > image_size:
            blc[1] = 0
        trc[1] -= delta
    return tuple(blc), tuple(trc)


def find_bbox(array, level, min_maxintensity_mjyperbeam, min_area_pix,
              delta=0.):
    """
    Find bounding box for part of image containing source.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_maxintensity_mjyperbeam:
        Minimum of the maximum intensity in the region to include.
    :param min_area_pix:
        Minimum area for region to include.
    :param delta: (optional)
        Extra space to add symmetrically [pixels]. (default: ``0``)
    :return:
        Tuples of BLC & TRC.

    :note:
        This is BLC, TRC for numpy array (i.e. transposed source map as it
        conventionally seen on VLBI maps).
    """
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    for prop in props:
        if prop.max_intensity > min_maxintensity_mjyperbeam/1000 and prop.area > min_area_pix:
            signal_props.append(prop)

    # Sometimes no regions are found. In that case return full image
    if not signal_props:
        return (0, 0,), (array.shape[1], array.shape[1],)

    blcs = list()
    trcs = list()

    for prop in signal_props:
        bbox = prop.bbox
        blc = (int(bbox[1]), int(bbox[0]))
        trc = (int(bbox[3]), int(bbox[2]))
        blcs.append(blc)
        trcs.append(trc)

    min_blc_0 = min([blc[0] for blc in blcs])
    min_blc_1 = min([blc[1] for blc in blcs])
    max_trc_0 = max([trc[0] for trc in trcs])
    max_trc_1 = max([trc[1] for trc in trcs])
    blc_rec = (min_blc_0-delta, min_blc_1-delta,)
    trc_rec = (max_trc_0+delta, max_trc_1+delta,)

    blc_rec_ = blc_rec
    trc_rec_ = trc_rec
    blc_rec_, trc_rec_ = check_bbox(blc_rec_, trc_rec_, array.shape[0])

    # Enlarge 10% each side
    delta_ra = abs(trc_rec[0]-blc_rec[0])
    delta_dec = abs(trc_rec[1]-blc_rec[1])
    blc_rec = (blc_rec[0] - int(0.1*delta_ra), blc_rec[1] - int(0.1*delta_dec))
    trc_rec = (trc_rec[0] + int(0.1*delta_ra), trc_rec[1] + int(0.1*delta_dec))

    blc_rec, trc_rec = check_bbox(blc_rec, trc_rec, array.shape[0])

    return blc_rec, trc_rec


def find_image_std(image_array, beam_npixels):
    # Robustly estimate image pixels std
    std = mad_std(image_array)

    # Find preliminary bounding box
    blc, trc = find_bbox(image_array, level=4*std,
                         min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*beam_npixels,
                         delta=0)

    # Now mask out source emission using found bounding box and estimate std
    # more accurately
    mask = np.zeros(image_array.shape)
    mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
    outside_icn = np.ma.array(image_array, mask=mask)
    return mad_std(outside_icn)


def is_prop_normally_distributed(prop, array):
    res = normaltest(array[prop.slice].ravel())
    if res.pvalue < 0.05:
        return False
    return True


def is_prop_inside_area(prop, array_shape, blc_target, trc_target):

    bbox_target = [blc_target[1], blc_target[0], trc_target[1], trc_target[0]]
    # print("bbox_target {}".format(bbox_target))
    slice_target = (slice(bbox_target[0], bbox_target[2], None), slice(bbox_target[1], bbox_target[3], None))
    # print("slice target {}".format(slice_target))

    sum = np.zeros(array_shape)
    sum[slice_target] += 1
    sum[prop.slice] += 1

    twos = np.count_nonzero(sum == 2)
    # print("Number of intersections = ", twos)
    # print("Prop.area = ", prop.area)
    if twos > 0.25*prop.area:
        return True
    return False


def find_patches(array, level, min_area_pix):
    """
    Find bounding box for part of image containing uncleaned flux.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_area_pix:
        Minimum area for region to include.
    """
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    # TODO: Simulate what threshold value for solidity is/
    # TODO: Using 10 beams as min area pix
    # TODO: Normally distribution check does not play much role here
    for prop in props:
        if prop.area > min_area_pix and prop.solidity < 0.75 and not is_prop_normally_distributed(prop, array):
            signal_props.append(prop)

    return signal_props


def check_dmap(darray, array, npixels_beam):
    std = find_image_std(array, beam_npixels=npixels_beam)
    patches = find_patches(darray, level=1*std, min_area_pix=10*npixels_beam)
    blc, trc = find_bbox(array, level=4*std, min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*npixels_beam, delta=0)
    uncleaned = list()
    for patch in patches:
        if is_prop_inside_area(patch, array.shape, blc, trc):
            uncleaned.append(patch)

    uncleaned_flux = list()
    for patch in uncleaned:
        uncleaned_flux.append(np.sum(darray[patch.slice].ravel())/npixels_beam)
    uncleaned_flux = np.sum(uncleaned_flux)
    print("Estimated uncleaned flux = {} Jy/beam".format(uncleaned_flux))
    return uncleaned_flux, uncleaned


def convert_bbox_to_radec_ranges(wcs, blc, trc):
    """
    Given BLC, TRC in coordinates of numpy array return RA, DEC ranges.
    :param wcs:
        Instance of ``astropy.wcs.wcs.WCS``.
    :return:
        RA_min, RA_max, DEC_min, DEC_max
    """
    blc_deg = wcs.all_pix2world(blc[0], blc[1], 1)
    trc_deg = wcs.all_pix2world(trc[0], trc[1], 1)

    ra_max = blc_deg[0]
    ra_min = trc_deg[0]
    dec_min = blc_deg[1]
    dec_max = trc_deg[1]

    return ra_min, ra_max, dec_min, dec_max


def clean_n(fname, outfname, stokes, mapsize_clean, niter=100,
            path_to_script=None, mapsize_restore=None, beam_restore=None,
            outpath=None, shift=None, show_difmap_output=False,
            txt_windows=None, clean_box=None):
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    difmapout = open("difmap_commands", "w")
    difmapout.write("observe " + fname + "\n")
    if txt_windows is not None:
        difmapout.write("rwin " + str(txt_windows) + "\n")

    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")

    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + ", " + str(niter) + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
                    str(mapsize_restore[1]) + "\n")
    if outpath is None:
        outpath = os.getcwd()
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < difmap_commands 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


def time_average(uvfits, outfname, time_sec=120, show_difmap_output=True,
                 reweight=True):
    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())

    difmapout = open(command_file, "w")
    if reweight:
        difmapout.write("observe " + uvfits + ", {}, true\n".format(time_sec))
    else:
        difmapout.write("observe " + uvfits + ", {}, false\n".format(time_sec))
    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    os.unlink(command_file)


def convolve_difmap_model_with_nonuniform_beam(difmap_model_file, stokes, mapsize, uvfits_template,
                                               original_beam, image_std_jyperbeam, out_ccfits="tmp_cc.fits",
                                               show_difmap_output=False):
    npixels_beam = np.pi*original_beam[0]*original_beam[1]/(4*np.log(2)*mapsize[1]**2)
    image_std_jyperpixel = image_std_jyperbeam/npixels_beam
    # Flux, r, theta
    cc_comps = np.loadtxt(difmap_model_file, comments="!")
    # Container for convolved images
    result_image = np.zeros((mapsize[0], mapsize[0]))
    for i in range(len(cc_comps)):
        # Create difmap model file for current component
        np.savetxt("difmap_tmp.mdl", cc_comps[i].reshape(1, -1))

        # Find SNR of current CC
        snr = abs(cc_comps[i][0])/image_std_jyperpixel
        print("CC Flux[Jy] = ", cc_comps[i][0], "SNR = ", snr)

        # Find beam to convolve with
        bmaj = original_beam[0]/np.log10(snr)
        local_beam = (bmaj, bmaj, 0)

        stamp = datetime.datetime.now()
        command_file = "difmap_commands_{}".format(stamp.isoformat())
        difmapout = open(command_file, "w")
        difmapout.write("observe " + uvfits_template + "\n")
        difmapout.write("select " + stokes + "\n")
        difmapout.write("rmodel " + difmap_model_file + "\n")
        difmapout.write("mapsize " + str(int(2*mapsize[0])) + ","+str(mapsize[1])+"\n")
        print("Restoring difmap model with BEAM : bmin = "+str(local_beam[1])+", bmaj = "+str(local_beam[0])+", "+str(local_beam[2])+" deg")
        # default dimfap: false,true (parameters: omit_residuals, do_smooth)
        difmapout.write("restore "+str(local_beam[1])+","+str(local_beam[0])+","+str(local_beam[2])+
                        ","+"true,false"+"\n")
        difmapout.write("wmap " + out_ccfits + "\n")
        difmapout.write("exit\n")
        difmapout.close()

        shell_command = "difmap < "+command_file+" 2>&1"
        if not show_difmap_output:
            shell_command += " >/dev/null"
        os.system(shell_command)
        # Remove command file
        os.unlink(command_file)

        # Load convolved image
        image = getdata(out_ccfits).squeeze()
        result_image += image

    hdus = pf.open(out_ccfits, mode="update")
    hdus[0].data[0, 0, ...] = result_image
    hdus.flush()


def clean_difmap_and_convolve_nubeam(uvfits, stokes, mapsize, outfile, original_beam, path_to_script):
    clean_difmap(fname=uvfits,
                 outfname=outfile, stokes=stokes.lower(),
                 mapsize_clean=mapsize,
                 path_to_script=path_to_script,
                 # With residuals
                 # show_difmap_output=True, beam_restore=None, omit_residuals=True, do_smooth=False)
                 beam_restore=None, omit_residuals=False, do_smooth=False,
                 dfm_model="dfm_cc_{}.mdl".format(stokes), show_difmap_output=True,
                 dmap="dmap_{}.fits".format(stokes.lower()))

    # Filter CCs - keep only those within the source
    pass

    # Find noise
    from uv_data import UVData
    uvdata = UVData(uvfits)
    noise = uvdata.noise(average_freq=True)
    noise = np.mean(list(noise.values()))
    std = noise/np.sqrt(uvdata.n_usable_visibilities_difmap(freq_average=True))
    # dimage = pf.getdata("dmap_{}.fits".format(stokes.lower()))[0, 0, ...]
    # std = mad_std(dimage)
    # print("dimage size = ", dimage.shape)

    print("Image std [Jy/beam] = ", std)

    convolve_difmap_model_with_nonuniform_beam("dfm_cc_{}.mdl".format(stokes), stokes, mapsize, uvfits,
                                               original_beam, std, out_ccfits=outfile)


def convert_difmap_model_file_to_CCFITS(difmap_model_file, stokes, mapsize, restore_beam, uvfits_template, out_ccfits,
                                        show_difmap_output=True):
    """
    Using difmap-formated model file (e.g. flux, r, theta) obtain convolution of your model with the specified beam.

    :param difmap_model_file:
        Difmap-formated model file. Use ``JetImage.save_image_to_difmap_format`` to obtain it.
    :param stokes:
        Stokes parameter.
    :param mapsize:
        Iterable of image size and pixel size (mas).
    :param restore_beam:
        Beam to restore: bmaj(mas), bmin(mas), bpa(deg).
    :param uvfits_template:
        Template uvfits observation to use. Difmap can't read model without having observation at hand.
    :param out_ccfits:
        File name to save resulting convolved map.
    :param show_difmap_output: (optional)
        Boolean. Show Difmap output? (default: ``True``)
    """
    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfits_template + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + difmap_model_file + "\n")
    difmapout.write("mapsize " + str(mapsize[0]) + "," + str(mapsize[1]) + "\n")
    print("Restoring difmap model with BEAM : bmin = " + str(restore_beam[1]) + ", bmaj = " + str(restore_beam[0]) + ", " + str(restore_beam[2]) + " deg")
    # default dimfap: false,true (parameters: omit_residuals, do_smooth)
    difmapout.write("restore " + str(restore_beam[1]) + "," + str(restore_beam[0]) + "," + str(restore_beam[2]) +
                    "," + "true,false" + "\n")
    difmapout.write("wmap " + out_ccfits + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)
    # Remove command file
    os.unlink(command_file)


def flag_baseline(uvfits, outfname, ta, tb, show_difmap_output=False):
    """
    Flag specified baseline.
    """
    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfits + "\n")
    difmapout.write("select i\n")

    difmapout.write("flag 1:{}-{}, true\n".format(ta, tb))

    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)
    # Remove command file
    os.unlink(command_file)


def flag_baseline_scan(uvfits, outfname, ta, tb=None, start_time=None, stop_time=None, except_time_range=False,
                       show_difmap_output=False):
    """
    Flag visibilities on baselines/antenna in specified time range (or all except specified time range).

    :param uvfits:
        Path to UVFITS file.
    :param outfname:
        Path to UVFITS save file.
    :param ta:
        First telescope to flag.
    :param tb: (optional)
        Second telescope to flag. If ``None``, then flag all with ``ta``. (default: ``None``)
    :param start_time: (optional)
        Astropy Time object. Start of the flagging interval. If ``None`` then flag from the beginning.
        (default: ``None``)
    :param stop_time: (optional)
        Astropy Time object. Stop of the flagging interval. If ``None`` then flag till the end.
        (default: ``None``)
    :param except_time_range: (optional)
        Flag outside of the time interval? (default: ``False``)
    """
    if start_time is not None:
        y, month, d, h, m, s = start_time.ymdhms
        # dd-mmm-yyyy-mm:hh:mm:ss
        t_start_difmap = "{}-{}-{}:{}:{}:{}".format(str(d).zfill(2), months_dict_inv[month].lower(), y, str(h).zfill(2),
                                                    str(m).zfill(2), str(int(round(s-1, 0))).zfill(2))
    else:
        t_start_difmap = ""

    if stop_time is not None:
        y, month, d, h, m, s = stop_time.ymdhms
        # dd-mmm-yyyy-mm:hh:mm:ss
        t_stop_difmap = "{}-{}-{}:{}:{}:{}".format(str(d).zfill(2), months_dict_inv[month].lower(), y, str(h).zfill(2),
                                                   str(m).zfill(2), str(int(round(s+1, 0))).zfill(2))
    else:
        t_stop_difmap = ""

    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfits + "\n")
    difmapout.write("select i\n")

    if not except_time_range:
        if tb is not None:
            difmapout.write("flag 1:{}-{}, true, {}, {}\n".format(ta, tb, t_start_difmap, t_stop_difmap))
        else:
            difmapout.write("flag 1:{}, true, {}, {}\n".format(ta, t_start_difmap, t_stop_difmap))
    else:
        if tb is not None:
            difmapout.write("flag 1:{}-{}, true, {}, {}\n".format(ta, tb, "", t_start_difmap))
            difmapout.write("flag 1:{}-{}, true, {}, {}\n".format(ta, tb, t_stop_difmap, ""))
        else:
            difmapout.write("flag 1:{}, true, {}, {}\n".format(ta, "", t_start_difmap))
            difmapout.write("flag 1:{}, true, {}, {}\n".format(ta, t_stop_difmap, ""))

    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)
    # Remove command file
    os.unlink(command_file)


# TODO: add ``shift`` argument, that shifts image before cleaning. It must be
# more accurate to do this in difmap. Or add such method in ``UVData`` that
# multiplies uv-data on exp(-1j * (u*x_shift + v*y_shift)).
def clean_difmap(fname, outfname, stokes, mapsize_clean, path=None,
                 path_to_script=None, mapsize_restore=None, beam_restore=None,
                 outpath=None, shift=None, show_difmap_output=False,
                 command_file=None, clean_box=None, dfm_model=None, omit_residuals=False,
                 do_smooth=True, dmap=None, text_box=None):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size [mas]).
    :param path: (optional)
        Path to uv-data to clean. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use naitive. (default: ``None``)
    :param beam_restore: (optional)
        Beam parameter for restore map (bmaj, bmin, bpa). If ``None`` then use
        the same beam as in cleaning. (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param shift: (optional)
        Iterable of 2 values - shifts in both directions - East & North [mas].
        If ``None`` then don't shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)
    :param command_file: (optional)
        Script file name to store `difmap` commands. If ``None`` then use
        ``difmap_commands``. (default: ``None``)
    :param clean_box: (optional)
         xa  -   The relative Right-Ascension of either edge of the new window.
         xb  -   The relative Right-Ascension of the opposite edge to 'xa', of
         the new window.
         ya  -   The relative Declination of either edge of the new window.
         yb  -   The relative Declination of the opposite edge to 'ya', of the
         new window.
         If ``None`` than do not use CLEAN windows. (default: ``None``)
    :param text_box: (optional)
        Path to text file with clean box (difmap output) to use. If ``None``
        then do not use text box. (default: ``None``)
    :param dfm_model: (optional)
        File name to save difmap-format model with CCs. If ``None`` then do
        not save model. (default: ``None``)
    :param dmap: (optional)
        FIle name to save the residual map. If ``None`` then do not save
        the residual map. (default: ``None``)
    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    if command_file is None:
        # command_file = "difmap_commands"
        stamp = datetime.datetime.now()
        command_file = os.path.join(outpath, "difmap_commands_{}".format(stamp.isoformat()))

    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")

    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")
    if text_box is not None:
        difmapout.write("rwins " + str(text_box) + "\n")

    difmapout.write("@" + path_to_script + " " + stokes + "\n")

    # FIXME: Do I need this?
    # difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
    #                 str(mapsize_clean[1]) + "\n")

    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")

    if beam_restore is not None:
        if omit_residuals:
            omit_residuals = "true"
        else:
            omit_residuals = "false"
        if do_smooth:
            do_smooth = "true"
        else:
            do_smooth = "false"
        difmapout.write("restore " + str(beam_restore[1]) + ', ' +
                        str(beam_restore[0]) + ', ' + str(beam_restore[2]) + ", " + omit_residuals + ", " + do_smooth +
                        "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"

    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    if dmap is not None:
        difmapout.write("wdmap " + os.path.join(outpath, dmap) + "\n")
    if dfm_model is not None:
        # FIXME: Difmap doesn't apply shift to model components!
        difmapout.write("wmodel " + os.path.join(outpath, dfm_model) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    os.unlink(command_file)


def deep_clean_difmap(fname, outfname, stokes, mapsize_clean, path=None,
                      path_to_script=None, mapsize_restore=None, beam_restore=None,
                      outpath=None, shift=None, show_difmap_output=False,
                      command_file=None, clean_box=None, prefix="tmp"):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size [mas]).
    :param path: (optional)
        Path to uv-data to clean. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use naitive. (default: ``None``)
    :param beam_restore: (optional)
        Beam parameter for restore map (bmaj, bmin, bpa). If ``None`` then use
        the same beam as in cleaning. (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param shift: (optional)
        Iterable of 2 values - shifts in both directions - East & North [mas].
        If ``None`` then don't shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)
    :param command_file: (optional)
        Script file name to store `difmap` commands. If ``None`` then use
        ``difmap_commands``. (default: ``None``)
    :param clean_box: (optional)
         xa  -   The relative Right-Ascension of either edge of the new window.
         xb  -   The relative Right-Ascension of the opposite edge to 'xa', of
         the new window.
         ya  -   The relative Declination of either edge of the new window.
         yb  -   The relative Declination of the opposite edge to 'ya', of the
         new window.
         If ``None`` than do not use CLEAN windows. (default: ``None``)


    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    if command_file is None:
        # command_file = "difmap_commands"
        stamp = datetime.datetime.now()
        command_file = os.path.join(outpath, "difmap_commands_{}".format(stamp.isoformat()))

    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    # if shift is not None:
    #     difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    # difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
    #                 str(mapsize_restore[1]) + "\n")
    # if outpath is None:
    #     outpath = path
    # elif not outpath.endswith("/"):
    #     outpath = outpath + "/"
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("wdmap " + os.path.join(outpath, "dmap_" + outfname) + "\n")

    if os.path.exists("{}.fits".format(prefix)):
        for ext in ("fits", "mod", "par", "uvf"):
            os.unlink("{}.{}".format(prefix, ext))

    difmapout.write("save " + prefix + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


    header = getheader(os.path.join(outpath, outfname))
    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial
    # Make offset coordinates
    wcs.wcs.crval = 0., 0.
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'mas', 'mas'
    wcs.wcs.cdelt = (wcs.wcs.cdelt * u.deg).to(u.mas)
    print(wcs)

    ccimage = create_clean_image_from_fits_file(os.path.join(outpath, outfname))
    dimage = create_image_from_fits_file(os.path.join(outpath, "dmap_" + outfname))
    beam = ccimage.beam
    npixels_beam = np.pi * beam[0] * beam[1] / mapsize_clean[1] ** 2
    print("Checking uncleaned patches...")
    uncleaned_flux, uncleaned_patches = check_dmap(dimage.image, ccimage.image, npixels_beam)

    # uncleaned_patches = list()

    if uncleaned_patches:
        # Remove old command file
        os.unlink(command_file)

        # TODO: Create CLEAN boxes from patches
        boxes = list()
        for patch in uncleaned_patches:
            bbox = patch.bbox
            blc = (int(bbox[1]), int(bbox[0]))
            trc = (int(bbox[3]), int(bbox[2]))
            print("patch wiht blc,trc ", blc, trc)
            ra_min, ra_max, dec_min, dec_max = convert_bbox_to_radec_ranges(wcs, blc, trc)
            boxes.append((ra_min, ra_max, dec_min, dec_max))

        print(boxes)

        difmapout = open(command_file, "w")
        print("Loading saved state...")
        # difmapout.write("get " + prefix + "\n")
        difmapout.write("@{}.par\n".format(prefix))
        difmapout.write("uvw 0,-2\n")

        for box in boxes:
            difmapout.write("addwin {},{},{},{}\n".format(*box))

        # difmapout.write("select i\n")
        # difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
        #                 str(mapsize_clean[1]) + "\n")
        print("NW CLEAN with 500, 0.03")
        difmapout.write("clean 10000,0.03\n")
        if beam_restore:
            difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                            str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                            "\n")
        difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
        difmapout.write("wdmap " + os.path.join(outpath, "dmap_" + outfname) + "\n")
        difmapout.write("exit\n")
        difmapout.close()

        # TODO: Use subprocess for silent cleaning?
        shell_command = "difmap < " + command_file + " 2>&1"
        os.system(shell_command)

    # Remove command file
    # os.unlink(command_file)
    # Remove saved state if any
    if os.path.exists(os.path.join(outpath, "{}.fits".format(prefix))):
        for ext in ("fits", "mod", "par", "uvf"):
            os.unlink(os.path.join(outpath, "{}.{}".format(prefix, ext)))


def selfcal_difmap(fname, outfname, path=None, path_to_script=None, outpath=None, show_difmap_output=False):
    """
    Self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param path: (optional)
        Path to uv-data to self-calibrate. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``selfcal`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)

    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    stamp = datetime.datetime.now()
    command_file = os.path.join(outpath, "difmap_commands_{}".format(stamp.isoformat()))

    difmapout = open(command_file, "w")
    difmapout.write("@" + path_to_script + " " + os.path.join(path, fname) + "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"
    difmapout.write("wobs " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    os.unlink(command_file)


def import_difmap_model(mdl_fname, mdl_dir=None, remove_last_char=True):
    """
    Function that reads difmap-format model and returns list of ``Components``
    instances.

    :param mdl_fname:
        File name with difmap model.
    :param mdl_dir: (optional)
        Directory with difmap model. If ``None`` then use CWD. (default:
        ``None``)
    :param remove_last_char: (optional)
        Remove last character (v ~ variable) from value? It works for difmap
        modelfit files, but for difmap CC files should be ``False``. (default:
        ``True``)
    :return:
        List of ``Components`` instances.
    """
    if mdl_dir is None:
        mdl_dir = os.getcwd()
    mdl = os.path.join(mdl_dir, mdl_fname)
    mdlo = open(mdl)
    lines = mdlo.readlines()
    comps = list()
    for line in lines:
        if line.startswith('!'):
            continue
        line = line.strip('\n ')
        try:
            flux, radius, theta, major, axial, phi, type_, freq, spec = line.split()
        except ValueError:
            try:
                flux, radius, theta, major, axial, phi, type_ = line.split()
            except ValueError:
                try:
                    flux, radius, theta = line.split()
                except ValueError:
                    print("Problem parsing line :\n")
                    print(line)
                    raise ValueError
                axial = 1.0
                major = 0.0
                type_ = 0

        list_fixed = list()
        if flux[-1] != 'v':
            list_fixed.append('flux')

        if remove_last_char:
            x = -float(radius[:-1]) * np.sin(np.deg2rad(float(theta[:-1])))
            y = -float(radius[:-1]) * np.cos(np.deg2rad(float(theta[:-1])))
            flux = float(flux[:-1])
        else:
            x = -float(radius) * np.sin(np.deg2rad(float(theta)))
            y = -float(radius) * np.cos(np.deg2rad(float(theta)))
            flux = float(flux)

        if int(type_) == 0:
            comp = DeltaComponent(flux, x, y)
        elif int(type_) == 1:

            try:
                bmaj = float(major)
                list_fixed.append('bmaj')
            except ValueError:
                bmaj = float(major[:-1])
            if float(axial[:-1]) == 1:
                comp = CGComponent(flux, x, y, bmaj, fixed=list_fixed)
            else:
                try:
                    e = float(axial)
                except ValueError:
                    e = float(axial[:-1])
                try:
                    bpa = np.deg2rad(float(phi)) + np.pi / 2.
                except ValueError:
                    bpa = np.deg2rad(float(phi[:-1])) + np.pi / 2.
                comp = EGComponent(flux, x, y, bmaj, e, bpa, fixed=list_fixed)
        else:
            raise NotImplementedError("Only CC, CG & EG are implemented")
        comps.append(comp)
    return comps


def export_difmap_model(comps, out_fname, freq_hz):
    """
    Export iterable of ``Component`` instances to the Difmap-format model file.

    :param comps:
        Iterable of ``Component`` instances.
    :param out_fname:
        Path for saving file.
    """
    with open(out_fname, "w") as fo:
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        for comp in comps:
            if isinstance(comp, EGComponent):
                if len(comp.p_all) == 4:
                    # Jy, mas, mas, mas
                    flux, x, y, bmaj = comp.p_all
                    e = "1.00000"
                    bpa = "000.000"
                    type = "1"
                    if not comp._fixed[3]:
                        bmaj = "{}v".format(bmaj)
                    else:
                        bmaj = "{}".format(bmaj)
                elif len(comp.p_all) == 6:
                    # Jy, mas, mas, mas, -, deg
                    flux, x, y, bmaj, e, bpa = comp.p_all
                    if not comp._fixed[4]:
                        e = "{}v".format(e)
                    else:
                        e = "{}".format(e)
                    if not comp._fixed[5]:
                        bpa = "{}v".format((bpa-np.pi/2)/degree_to_rad)
                    else:
                        bpa = "{}".format((bpa-np.pi/2)/degree_to_rad)
                    if not comp._fixed[3]:
                        bmaj = "{}v".format(bmaj)
                    else:
                        bmaj = "{}".format(bmaj)
                    type = "1"
                else:
                    raise Exception
            elif isinstance(comp, DeltaComponent):
                flux, x, y = comp.p_all
                e = "1.00000"
                bmaj = "0.0000"
                bpa = "000.000"
                type = "0"
            else:
                raise Exception
            # mas
            r = np.hypot(x, y)
            # rad
            theta = np.arctan2(-x, -y)
            theta /= degree_to_rad

            if not comp._fixed[0]:
                flux = "{}v".format(flux)
            else:
                flux = "{}".format(flux)

            if comp._fixed[1]:
                r = "{}".format(r)
            else:
                r = "{}v".format(r)

            if comp._fixed[2]:
                theta = "{}".format(theta)
            else:
                theta = "{}v".format(theta)

            fo.write("{} {} {} {} {} {} {} {} 0\n".format(flux, r, theta,
                                                          bmaj, e, bpa, type,
                                                          freq_hz))


def append_component_to_difmap_model(comp, out_fname, freq_hz):
    """
    :param comp:
        Instance of ``Component``.
    :param fname:
        File with difmap model.
    """
    with open(out_fname, "a") as fo:
        if isinstance(comp, EGComponent):
            if comp.size == 4:
                # Jy, mas, mas, mas
                flux, x, y, bmaj = comp.p
                e = "1.00000"
                bpa = "000.000"
                type = "1"
                bmaj = "{}v".format(bmaj)
            elif comp.size == 6:
                # Jy, mas, mas, mas, -, deg
                flux, x, y, bmaj, e, bpa = comp.p
                e = "{}v".format(e)
                bpa = "{}v".format((bpa - np.pi / 2) / degree_to_rad)
                bmaj = "{}v".format(bmaj)
                type = "1"
            else:
                raise Exception
        elif isinstance(comp, DeltaComponent):
            flux, x, y = comp.p
            e = "1.00000"
            bmaj = "0.0000"
            bpa = "000.000"
            type = "0"
        else:
            raise Exception
        # mas
        r = np.hypot(x, y)
        # rad
        theta = np.arctan2(-x, -y)
        theta /= degree_to_rad
        fo.write("{}v {}v {}v {} {} {} {} {} 0\n".format(flux, r, theta,
                                                         bmaj, e, bpa, type,
                                                         freq_hz))


def difmap_model_flux(mdl_file):
    """
    Returns flux of the difmap model.

    :param mdl_file:
        Path to difmap model file.
    :return:
        Sum of all component fluxes [Jy].
    """
    comps = import_difmap_model(mdl_file)
    fluxes = [comp.p[0] for comp in comps]
    return sum(fluxes)


def find_ids(comps0, comps1):
    assert len(comps0) == len(comps1)
    ids = list()
    for comp in comps0:
        ids.append(comps1.index(comp))
    return ids


def sort_components_by_distance_from_cj(mdl_path, freq_hz, n_check_for_core=2,
                                        outpath=None, perc_distant=75,
                                        only_indicate=False):
    """
    Function that re-arrange components in a such a way that closest to "counter
    jet" components goes first. If components were already sorted by distance
    from "counter-jet" then does nothing and return ``False``.

    :param mdl_path:
        Path to difmap-format model file.
    :param freq_hz:
        Frequency in Hz.
    :param n_check_for_core: (optional)
        Number of closest to phase center components to check for being on "cj"
        side. (default: ``2``)
    :param outpath: (optional)
        Path to save re-arranged model. If ``None`` then re-write ``mdl_path``.
        (default: ``None``)
    :param perc_distant: (optional)
        Percentile of the component's distance distribution to use as border
        line that defines "distant" component. Position of such components is
        then used to compare polar angles of "cj-candidates". (default: ``75``)
    :param only_indicate: (optional)
        Boolean - ust indicate by the returned value or also do re-arrangement
        of components? (default: ``False``)
    :return:
        Boolean if there any "cj"-components that weren't sorted by distance
        from "CJ".
    """
    mdl_dir, mdl_fname = os.path.split(mdl_path)
    comps = import_difmap_model(mdl_fname, mdl_dir)
    comps = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))
    comps_c = import_difmap_model(mdl_fname, mdl_dir)
    r = [np.hypot(comp.p[1], comp.p[2]) for comp in comps]
    dist = np.percentile(r, perc_distant)
    # Components that are more distant from phase center than ``perc_distant``
    # of components
    remote_comps = [comp for r_, comp in zip(r, comps) if r_ > dist]
    x_mean = np.mean([-comp.p[1] for comp in remote_comps])
    y_mean = np.mean([-comp.p[2] for comp in remote_comps])
    theta_remote = np.arctan2(x_mean, y_mean)/degree_to_rad

    found_cj_comps = list()
    for i in range(0, n_check_for_core):
        comp = comps[i]
        # Don't count EG component
        if len(comp) == 6:
            continue
        # This checks that cj component have different PA
        if abs(np.arctan2(-comp.p[1], -comp.p[2])/degree_to_rad -
               theta_remote) > 90.:
            print("Found cj components - {}".format(comp))
            found_cj_comps.append(comp)

    result_comps = list()
    if found_cj_comps:
        for comp in found_cj_comps:
            comps.remove(comp)

        for comp in sorted(found_cj_comps,
                           key=lambda x: np.hypot(x.p[1], x.p[2])):
            result_comps.append(comp)

        # Now check if cj-components were not first ones
        is_changed = False
        sorted_found_cj_comps = sorted(found_cj_comps,
                                       key=lambda x: np.hypot(x.p[1], x.p[2]))
        for comp1, comp2 in zip(comps_c[:len(sorted_found_cj_comps)],
                                sorted_found_cj_comps[::-1]):
            if comp1 != comp2:
                is_changed = True
        # If all job was already done just return
        if not is_changed:
            print("Components are already sorted")
            return False

    for comp in comps:
        result_comps.append(comp)

    if outpath is None:
        outpath = mdl_path
    if not only_indicate:
        export_difmap_model(result_comps, outpath, freq_hz)
    return bool(found_cj_comps)


def sum_components(comp0, comp1, type="eg"):
    print("Summing components {} and {}".format(comp0, comp1))
    flux = comp0.p[0] + comp1.p[0]
    x = (comp0.p[0] * comp0.p[1] + comp1.p[0] * comp1.p[1]) / flux
    y = (comp0.p[0] * comp0.p[2] + comp1.p[0] * comp1.p[2]) / flux
    bmaj = (comp0.p[0] * comp0.p[3] + comp1.p[0] * comp1.p[3]) / flux
    if type == "cg":
        component = CGComponent(flux, x, y, bmaj)
    elif type == "eg":
        dx = comp0.p[1]-comp1.p[1]
        dy = comp0.p[2]-comp1.p[2]
        bpa = 180*np.arctan(dx/dy)/np.pi
        e = 0.5*(comp0.p[3]+comp1.p[3])/np.hypot(comp0.p[1]-comp1.p[1], comp0.p[2]-comp1.p[2])
        component = EGComponent(flux, x, y, bmaj, e, bpa)
    else:
        raise Exception
    print("Sum = {}".format(component))
    return component


def component_joiner_serial(difmap_model_file, beam_size, freq_hz,
                            distance_to_join_max=1.0, outname=None, new_type="eg"):
    joined = False
    comps = import_difmap_model(difmap_model_file)
    print("Len = {}".format(len(comps)))
    new_comps = list()
    skip_next = False
    for i, comp0 in enumerate(comps):
        print("{} and {}".format(i, i+1))
        if skip_next:
            skip_next = False
            print ("Skipping {}th component".format(i))
            continue
        try:
            comp1 = comps[i+1]
        except IndexError:
            new_comps.append(comp0)
            print("Writing component {} and exiting".format(i))
            break
        print("Fluxes: {} and {}".format(comp0.p[0], comp1.p[0]))

        if i != 0:
            distance_before = max(np.hypot(comp0.p[1]-comps[i-1].p[1], comp0.p[2]-comps[i-1].p[2]),
                                  np.hypot(comp1.p[1]-comps[i-1].p[1], comp1.p[2]-comps[i-1].p[2]))
        else:
            distance_before = 10**10
        print("Distance before = {}".format(distance_before))

        distance_between = np.hypot(comp0.p[1]-comp1.p[1],
                                    comp0.p[2]-comp1.p[2])
        print("Distance between = {}".format(distance_between))

        if i < len(comps)-2:
            distance_after = max(np.hypot(comp0.p[1] - comps[i + 2].p[1],
                                          comp0.p[2] - comps[i + 2].p[2]),
                                 np.hypot(comp1.p[1] - comps[i + 2].p[1],
                                          comp1.p[2] - comps[i + 2].p[2]))
        else:
            distance_after = 10000.0
        print("Distance after = {}".format(distance_after))

        if distance_before > beam_size and\
                        distance_after > beam_size and\
                        distance_between < distance_to_join_max*beam_size and\
                len(comp0) == len(comp1):
            joined = True
            new_comps.append(sum_components(comp0, comp1, type=new_type))
            skip_next = True
        else:
            new_comps.append(comp0)
            skip_next = False

    if outname is None:
            outname = difmap_model_file

    export_difmap_model(new_comps, outname, freq_hz)
    return joined


def remove_furthest_component(difmap_model_file, freq_hz, outname=None):
    comps = import_difmap_model(difmap_model_file)
    # Sort by distance
    furthest_comp = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))[-1]

    new_comps = [comp for comp in comps if comp != furthest_comp]
    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)
    return True


def remove_small_component(difmap_model_file, freq_hz, size_limit=0.01,
                           outname=None):
    comps = import_difmap_model(difmap_model_file)
    small_comps = [comp for comp in comps if (len(comp.p) ==4 and
                                              comp.p[3] < size_limit)]
    if not small_comps:
        return False
    new_comps = [comp for comp in comps if comp not in small_comps]
    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)
    return True

# FIXME: Check if it works to ``DeltaComponent``!
def transform_component(difmap_model_file, new_type, freq_hz, comp_id=0,
                        outname=None, **kwargs):
    """
    Change component type in difmap model file.

    :param difmap_model_file:
        Path to difmap file with model.
    :param new_type:
        String that defines what is the new component type (delta, cg, eg).
    :param freq_hz:
        Frequency in Hz.
    :param comp_id: (optional)
        Number (ID) of component to transform. ``0`` means transform first
        component (core). (default: ``0``)
    :param outname: (optional)
        Where to save the result. If ``None`` then rewrite
        ``difmap_model_file``.
    :param kwargs: (optional)
        Arguments needed for transforming simple component to more complex.
    """
    if new_type not in ("delta", "cg", "eg"):
        raise Exception
    comps = import_difmap_model(difmap_model_file)
    comp_to_transform = comps[comp_id]
    new_comps = list()
    for comp in comps[:comp_id]:
        new_comps.append(comp)
    if new_type == "delta":
        new_comps.append(comp_to_transform.to_delta())
    elif new_type == "cg":
        new_comps.append(comp_to_transform.to_cirular(**kwargs))
    else:
        new_comps.append(comp_to_transform.to_elliptic(**kwargs))
    for comp in comps[comp_id+1:]:
        new_comps.append(comp)

    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)



def tb_gaussian(flux_jy, bmaj_mas, freq_ghz, z=0, bmin_mas=None, D=1.0):
    """
    Brightness temperature (optionally corrected for redshift and Doppler
    factor) in K.
    """
    k = 1.38*10**(-16)
    c = 29979245800.0
    mas_to_rad = 4.8481368*1E-09

    bmaj_mas *= mas_to_rad
    if bmin_mas is None:
        bmin_mas = bmaj_mas
    else:
        bmin_mas *= mas_to_rad
    freq_ghz *= 10**9
    flux_jy *= 10**(-23)
    return 2.*np.log(2)*(1.+z)*flux_jy*c**2/(freq_ghz**2*np.pi*k*bmaj_mas*bmin_mas*D)


def sigma_tb_gaussian(sigma_flux_jy, sigma_d_mas, flux_jy, d_mas, freq_ghz, z=0, D=1.0):
    """
    Brightness temperature (optionally corrected for redshift and Doppler
    factor) in K.
    """
    k = 1.38*10**(-16)
    c = 29979245800.0
    mas_to_rad = 4.8481368*1E-09

    d_mas *= mas_to_rad
    sigma_d_mas *= mas_to_rad
    freq_ghz *= 10**9
    flux_jy *= 10**(-23)
    sigma_flux_jy *= 10**(-23)
    return 2.*np.log(2)*(1.+z)*c**2/(freq_ghz**2*np.pi*k*d_mas**2*D)*np.sqrt(sigma_flux_jy**2 + (2*flux_jy/d_mas)**2*sigma_d_mas**2)


def convert_bbox_to_radec_ranges(wcs, blc, trc):
    """
    Given BLC, TRC in coordinates of numpy array return RA, DEC ranges.
    :param wcs:
        Instance of ``astropy.wcs.wcs.WCS``.
    :return:
        RA_min, RA_max, DEC_min, DEC_max
    """
    blc_deg = wcs.all_pix2world(blc[0], blc[1], 1)
    trc_deg = wcs.all_pix2world(trc[0], trc[1], 1)

    ra_max = blc_deg[0]
    ra_min = trc_deg[0]
    dec_min = blc_deg[1]
    dec_max = trc_deg[1]

    return ra_min, ra_max, dec_min, dec_max


def components_info(fname, mdl_fname, dmap_size, PA=None, dmap_name=None,
                    out_path=None, stokes="I", show_difmap_output=True):
    import glob
    import pandas as pd
    import astropy.io.fits as pf
    # Conversion coefficient from degrees to mas
    deg2mas = 3.6*10**6

    columns = ["flux", "ra", "dec", "major", "minor", "posangle", "type", "snr", "rms", "flux_err",
               "r_err", "major_unresolved", "minor_unresolved", "major_err", "minor_err",
               # "PA_core", "size_across_PA_core", "resolved_across_PA_core", "size_across_PA_core_err",
               "Tb", "Tb_err", "upper_limit_Tb"]
    df_dict = dict()
    for column in columns:
        df_dict[column] = list()

    if out_path is None:
        out_path = os.getcwd()

    # Remove old log-file if any
    difmap_log_files = glob.glob("difmap.log*")
    for difmap_log_file in difmap_log_files:
        try:
            os.unlink(difmap_log_file)
        except OSError:
            pass

    stamp = datetime.datetime.now()
    command_file = os.path.join(out_path, "difmap_commands_{}".format(stamp.isoformat()))
    difmapout = open(command_file, "w")
    difmapout.write("observe " + fname + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + mdl_fname + "\n")
    difmapout.write("mapsize " + str(dmap_size[0]) + "," + str(dmap_size[1]) + "\n")
    difmapout.write("invert\n")
    # Obtain dirty image
    if dmap_name is None:
        dmap_name = "dirty_residuals_map.fits"
    difmapout.write("wdmap " + os.path.join(out_path, dmap_name) + "\n")

    # Get beam info
    difmapout.write("restore\n")

    difmapout.write("exit\n")
    difmapout.close()

    # TODO: Use subprocess?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


    # Get final reduced chi_squared
    with open("difmap.log", "r") as fo:
        lines = fo.readlines()
    line = [line for line in lines if "Restoring with beam:" in line][-1]
    bmaj = float(line.split()[4])
    bmin = float(line.split()[6])
    if bmaj < bmin:
        bmaj, bmin = bmin, bmaj
    bpa = np.deg2rad(float(line.split()[8]))
    beam_area_mas = np.pi*bmaj*bmin/(4*np.log(2))

    if PA is None:
        theta_beam = np.sqrt(bmaj*bmin)
    else:
        theta_beam = 2*bmaj*bmin*np.sqrt((1+np.tan(PA-bpa)**2)/(bmin**2+bmaj**2*np.tan(PA-bpa)**2))

    # Remove Difmap command and log-file
    os.unlink(command_file)
    os.unlink("difmap.log")

    header = pf.getheader(os.path.join(out_path, dmap_name))

    # Build coordinate system
    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial
    # Make offset coordinates
    wcs.wcs.crval = 0., 0.
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'mas', 'mas'
    wcs.wcs.cdelt = (wcs.wcs.cdelt * u.deg).to(u.mas)

    dimage = pf.getdata(os.path.join(out_path, dmap_name)).squeeze()
    # Characteristic size of the beam in pixels
    beam_pxl = int(np.sqrt(bmaj * bmin) / dmap_size[1])

    # Read components from difmap dirty image CC-table
    comps = pf.getdata(os.path.join(out_path, dmap_name), ext=("AIPS CC ", 1))
    sizes_lim = list()
    unresolved_maj = None
    unresolved_min = None
    upper_limit_Tb = None

    x = np.arange(header["NAXIS1"])
    y = np.arange(header["NAXIS2"])
    X, Y = np.meshgrid(x, y)
    # Arrays of RA, DEC for all pixels (just in case)
    # ra_pix_all, dec_pix_all = wcs.wcs_pix2world(X, Y, 0)

    for comp in comps:
        # RA, DEC of component center
        ra = comp[1]*deg2mas
        dec = comp[2]*deg2mas
        # TODO: Currently it is from phase center - not core!
        PA_core = np.arctan2(ra, dec)
        ra_pix, dec_pix = wcs.wcs_world2pix(ra.reshape(-1, 1), dec.reshape(-1, 1), 0)
        ra_pix = ra_pix[0][0]
        dec_pix = dec_pix[0][0]
        # Create mask of size N*beam_pxl with center in (ra_pix, dec_pix)
        mask = np.sqrt((X-ra_pix)**2+(Y-dec_pix)**2) < 3*beam_pxl
        # Need to add to map center (512, 512) add DEC in pixels (with sign)
        rms = np.std(dimage[mask])
        print("RMS = ", rms)

        # For CG
        if comp["TYPE OBJ"] == 1.0:
            size = comp["MAJOR AX"]*deg2mas
            print("comp size = ", size)
            # Calculate first time for SNR
            peak_flux = 4*np.log(2)*comp["FLUX"] * (np.pi*size**2/(4*np.log(2)*beam_area_mas)) / (np.pi*size**2)
            print("peak flux = ", peak_flux)
            SNR = peak_flux / rms
            size_lim = theta_beam*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            print("Size lim = ", size_lim)
            # If it is a limit - re-calculate peak flux
            unresolved_min = None
            if size < size_lim:
                unresolved_maj = True
                size = size_lim
                upper_limit_Tb = True
            else:
                unresolved_maj = False
                upper_limit_Tb = False
            # Re-calculate if sizes are limits
            peak_flux = 4*np.log(2)*comp["FLUX"] * (np.pi*size**2/(4*np.log(2)*beam_area_mas)) / (np.pi*size**2)
            print("new peak flux = ", peak_flux)
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            sigma_tot_flux = sigma_peak*np.sqrt(1 + comp["FLUX"]**2/peak_flux**2)
            sigma_size = size*sigma_peak/peak_flux
            tb = tb_gaussian(comp["FLUX"], size, 15.4)
            sigma_tb = sigma_tb_gaussian(sigma_tot_flux, sigma_size, comp["FLUX"], size, 15.4)

            # size_across_PA_core = size
            # resolved_across_PA_core = unresolved_maj
            # size_across_PA_core_err = size_across_PA_core*sigma_peak/peak_flux

            df_dict["major"].append(size)
            df_dict["minor"].append(size)
            df_dict["major_err"].append(sigma_size)
            df_dict["minor_err"].append(sigma_size)
            df_dict["posangle"].append(None)
            df_dict["type"].append(1)
            df_dict["r_err"].append(0.5*sigma_size)

        # For EG
        elif comp["TYPE OBJ"] == 2.0:
            upper_limit_Tb = False
            size_major = comp["MAJOR AX"]*deg2mas
            size_minor = comp["MINOR AX"]*deg2mas
            # size_across_PA_core = 2*size_minor*size_major*np.sqrt((1+np.tan(PA_core+np.pi/2-np.deg2rad(comp["POSANGLE"]))**2)/(bmin**2+bmaj**2*np.tan(PA_core+np.pi/2-np.deg2rad(comp["POSANGLE"]))**2))

            # Calculate first time for SNR
            peak_flux = 4*np.log(2)*comp["FLUX"] * (np.pi*size_major*size_minor/(4*np.log(2)*beam_area_mas)) / (np.pi*size_major*size_minor)
            SNR = peak_flux / rms

            theta_beam_along_major = 2*bmaj*bmin*np.sqrt((1+np.tan(np.deg2rad(comp["POSANGLE"])-bpa)**2)/(bmin**2 + bmaj**2*np.tan(np.deg2rad(comp["POSANGLE"])-bpa)**2))
            theta_beam_along_minor = 2*bmaj*bmin*np.sqrt((1+np.tan(np.deg2rad(comp["POSANGLE"]+90)-bpa)**2)/(bmin**2 + bmaj**2*np.tan(np.deg2rad(comp["POSANGLE"]+90)-bpa)**2))

            size_lim_major = theta_beam_along_major*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            size_lim_minor = theta_beam_along_minor*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)

            # If it is a limit - re-calculate peak flux
            if size_major < size_lim_major:
                size_major = size_lim_major
                unresolved_maj = True
                upper_limit_Tb = True
            else:
                unresolved_maj = False

            if size_minor < size_lim_minor:
                size_minor = size_lim_minor
                unresolved_min = True
                upper_limit_Tb = True
            else:
                unresolved_min = False

            # Re-calculate if sizes are limits
            peak_flux = 4*np.log(2)*comp["FLUX"] * (np.pi*size_major*size_minor/(4*np.log(2)*beam_area_mas)) / (np.pi*size_minor*size_major)
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            sigma_tot_flux = sigma_peak*np.sqrt(1 + comp["FLUX"]**2/peak_flux**2)
            sigma_size_major = size_major*sigma_peak/peak_flux
            sigma_size_minor = size_minor*sigma_peak/peak_flux
            tb = tb_gaussian(comp["FLUX"], size_major, 15.4, bmin_mas=size_minor)
            sigma_tb = sigma_tb_gaussian(sigma_tot_flux, 0.5*(sigma_size_minor+sigma_size_major), comp["FLUX"], 0.5*(size_minor+size_major), 15.4)

            # theta_beam_across_PA_core = 2*bmaj*bmin*np.sqrt((1+np.tan(PA_core-bpa)**2)/(bmin**2 + bmaj**2*np.tan(PA_core-bpa)**2))
            # size_lim_across_PA_core = theta_beam_across_PA_core*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            # if size_across_PA_core > size_lim_across_PA_core:
            #     resolved_across_PA_core = True
            # else:
            #     resolved_across_PA_core = False
            #     size_across_PA_core = size_lim_across_PA_core
            # size_across_PA_core_err = size_across_PA_core*sigma_peak/peak_flux


            df_dict["major"].append(size_major)
            df_dict["minor"].append(size_minor)
            df_dict["posangle"].append(np.deg2rad(comp["POSANGLE"]))
            df_dict["type"].append(2)
            df_dict["r_err"].append(0.5*0.5*(sigma_size_minor+sigma_size_major))
            df_dict["major_err"].append(sigma_size_major)
            df_dict["minor_err"].append(sigma_size_minor)

        # For delta
        # FIXME: Must be different size limits along beam bmaj & bmin
        elif comp["TYPE OBJ"] == 0.0:
            unresolved_maj = None
            unresolved_min = None
            upper_limit_Tb = True
            peak_flux = comp["FLUX"]
            SNR = peak_flux/rms
            size_lim = theta_beam*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            size = size_lim
            peak_flux = 4*np.log(2)*comp["FLUX"] * (np.pi*size**2/(4*np.log(2)*beam_area_mas)) / (np.pi*size**2)
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            sigma_tot_flux = sigma_peak*np.sqrt(1 + comp["FLUX"]**2/peak_flux**2)
            sigma_size = size*sigma_peak/peak_flux
            tb = tb_gaussian(comp["FLUX"], size, 15.4)
            sigma_tb = sigma_tb_gaussian(sigma_tot_flux, sigma_size, comp["FLUX"], size, 15.4)
            # size_across_PA_core = size
            # resolved_across_PA_core = None
            # size_across_PA_core_err = size_across_PA_core*sigma_peak/peak_flux

            df_dict["major"].append(size)
            df_dict["minor"].append(size)
            df_dict["posangle"].append(None)
            df_dict["type"].append(0)
            df_dict["r_err"].append(0.5*sigma_size)
            df_dict["major_err"].append(sigma_size)
            df_dict["minor_err"].append(sigma_size)
        else:
            raise Exception

        df_dict["snr"].append(SNR)
        df_dict["rms"].append(rms)
        df_dict["flux"].append(comp["FLUX"])
        df_dict["ra"].append(ra)
        df_dict["dec"].append(dec)
        df_dict["flux_err"].append(sigma_tot_flux)
        df_dict["major_unresolved"].append(unresolved_maj)
        df_dict["minor_unresolved"].append(unresolved_min)
        # df_dict["PA_core"].append(PA_core)
        # df_dict["resolved_across_PA_core"].append(resolved_across_PA_core)
        # df_dict["size_across_PA_core"].append(size_across_PA_core)
        # df_dict["size_across_PA_core_err"].append(size_across_PA_core_err)
        df_dict["Tb"].append(tb)
        df_dict["Tb_err"].append(sigma_tb)
        df_dict["upper_limit_Tb"].append(upper_limit_Tb)

    try:
        os.unlink(os.path.join(out_path, "dirty_residuals_map.fits"))
    except FileNotFoundError:
        pass

    return pd.DataFrame.from_dict(df_dict)


# TODO: Add iteration till chi-squared is increasing for some number of
# iterations
def modelfit_difmap(fname, mdl_fname, out_fname, niter=50, stokes='i',
                    path=None, mdl_path=None, out_path=None,
                    show_difmap_output=False,
                    save_dirty_residuals_map=False,
                    dmap_name=None, dmap_size=(1024, 0.1)):
    """
    Modelfit self-calibrated uv-data in difmap.

    :param fname:
        Filename of uv-data to modelfit.
    :param mdl_fname:
        Filename with model.
    :param out_fname:
        Filename with output file with model.
    :param stokes: (optional)
        Stokes parameter 'i', 'q', 'u' or 'v'. (default: ``i``)
    :param path: (optional)
        Path to uv-data to modelfit. If ``None`` then use current directory.
        (default: ``None``)
    :param mdl_path: (optional)
        Path file with model. If ``None`` then use current directory.
        (default: ``None``)
    :param out_path: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param save_dirty_residuals_map: (optional)
        Boolean. Whever to save dirty residuals map? (default: ``False``)
    :param dmap_name: (optional)
        Name of the FITS file (in ``out_path``) where to save dirty residual
        map. If ``None`` then dirty_residuals_map.fits``. (default: ``None``)
    :param dmap_size: (optional)
        Size of the dirty residuals map. (default: ``(1024, 0.1)``
    """
    if path is None:
        path = os.getcwd()
    if mdl_path is None:
        mdl_path = os.getcwd()
    if out_path is None:
        out_path = os.getcwd()

    # Remove old log-file if any
    try:
        os.unlink(os.path.join(os.getcwd(), "difmap.log"))
    except OSError:
        pass

    stamp = datetime.datetime.now()
    command_file = os.path.join(out_path, "difmap_commands_{}".format(stamp.isoformat()))
    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + os.path.join(mdl_path, mdl_fname) + "\n")
    difmapout.write("modelfit " + str(niter) + "\n")
    difmapout.write("wmodel " + os.path.join(out_path, out_fname) + "\n")

    if save_dirty_residuals_map:
        difmapout.write("mapsize "+ str(2*dmap_size[0]) + "," + str(dmap_size[1])
                        + "\n")
        if dmap_name is None:
            dmap_name = "dirty_residuals_map.fits"
        difmapout.write("wdmap "+ os.path.join(out_path, dmap_name)+"\n")

    difmapout.write("exit\n")
    difmapout.close()

    # TODO: Use subprocess?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Get final reduced chi_squared
    log = os.path.join(os.getcwd(), "difmap.log")
    with open(log, "r") as fo:
        lines = fo.readlines()
    line = [line for line in lines if "Reduced Chi-squared=" in line][-1]
    rchisq = float(line.split(" ")[4].split("=")[1])

    # Remove command file
    os.unlink(command_file)

    return rchisq


def find_stat_of_difmap_model(dfm_model_file, uvfits, stokes="I", working_dir=None,
                              show_difmap_output=False, unselfcalibrated_uvfits=None,
                              nmodelfit=20, use_pselfcal=True, out_dfm_model=None):

    if unselfcalibrated_uvfits is None:
        uvfile = uvfits
    else:
        uvfile = unselfcalibrated_uvfits

    original_dir = os.getcwd()
    if working_dir is None:
        working_dir = os.getcwd()
    os.chdir(working_dir)

    # Find and remove all log-files
    previous_logs = glob.glob("difmap.log*")
    for log in previous_logs:
        os.unlink(log)

    stamp = datetime.datetime.now()
    command_file = os.path.join(working_dir, "difmap_commands_{}".format(stamp.isoformat()))
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfile + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + dfm_model_file + "\n")
    if use_pselfcal:
        difmapout.write("selfcal\n")
        difmapout.write("modelfit {}\n".format(nmodelfit))
        difmapout.write("selfcal\n")
    difmapout.write("modelfit {}\n".format(nmodelfit))
    if out_dfm_model:
        difmapout.write("wmod {}\n".format(out_dfm_model))
    difmapout.write("quit\n")
    difmapout.close()

    # TODO: Use subprocess?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Get final reduced chi_squared
    log = os.path.join(working_dir, "difmap.log")
    with open(log, "r") as fo:
        lines = fo.readlines()
    line = [line for line in lines if "Reduced Chi-squared=" in line][-1]
    rchisq = float(line.split(" ")[4].split("=")[1])

    line = [line for line in lines if "degrees of freedom" in line][-1]
    dof = int(float(line.split(" ")[-4]))

    # Remove command file
    os.unlink(command_file)
    # os.unlink("difmap.log")
    os.chdir(original_dir)

    return {"rchisq": rchisq, "dof": dof}


def find_position_errors_using_chi2(dfm_model_file, uvfits, stokes="I", working_dir=None,
                                    delta_t_sec=10.0, gain_phase_tcoh=30.0,
                                    show_difmap_output=False, nmodelfit=20,
                                    unselfcalibrated_uvfits=None):

    # Nuber of time stamps for gain phases to "forget" their values
    t_eff = gain_phase_tcoh/delta_t_sec
    rho = np.exp(-(delta_t_sec/gain_phase_tcoh)**2)

    # Find gains DoF
    uvdata = UVData(uvfits)
    n_use_vis = uvdata.n_usable_visibilities_difmap(stokes=stokes)
    n_IF = uvdata.nif
    n_ant = len(uvdata.antennas)

    # Coefficient of correlation between gains in close IFs
    rho_IF = 0.75
    n_IF_eff = n_IF/(1+(n_IF-1)*rho_IF)


    baseline_scan_times = uvdata.baselines_scans_times
    n_scans = np.argmax(np.bincount([len(a) for a in baseline_scan_times.values()]))

    n_measurements = 0
    time_of_scan = list()
    for bl, scans in baseline_scan_times.items():
        for scan in scans:
            delta_t_scan_sec = (Time(scan[-1], format="jd") - Time(scan[0], format="jd")).sec
            time_of_scan.append(delta_t_scan_sec)
            n_measurements += delta_t_scan_sec/10.
    time_of_scan = np.median(time_of_scan)

    n_eff_gain_phases = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff
    if stokes == "I":
        n_eff_gain_phases *= 2

    import dlib
    # Just best rchisq (w/o any fit or self-cal)
    stat_dict = find_stat_of_difmap_model(dfm_model_file, uvfits, stokes, working_dir, nmodelfit=100, use_pselfcal=True,
                                          out_dfm_model="selfcaled.mdl")
    rchisq0 = stat_dict["rchisq"]
    dof = stat_dict["dof"] - n_eff_gain_phases
    delta_rchisq = 1.0/dof
    required_chisq = rchisq0 + delta_rchisq
    # original_comps = import_difmap_model(dfm_model_file)
    original_comps = import_difmap_model("selfcaled.mdl")
    original_comps = sorted(original_comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    # comps = import_difmap_model(dfm_model_file)
    comps = import_difmap_model("selfcaled.mdl")

    errors = list()

    for i, comp in enumerate(comps):
        # x, y
        for index in (1, 2):

            par0 = comp.p[index]

            def min_func_lower(delta):
                ccomp = copy.copy(comp)
                ccomp.set_value(par0 - delta, index)
                # Hack to fix r & theta
                ccomp._fixed[1] = True
                ccomp._fixed[2] = True
                new_comps = comps.copy()
                new_comps.insert(i, ccomp)
                new_comps.pop(i+1)
                export_difmap_model(new_comps, "dfm.mdl", 15.4E+09)
                sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, show_difmap_output,
                                                  nmodelfit=nmodelfit, use_pselfcal=True)
                print("For delta = {} differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
                return (required_chisq - sdict["rchisq"])**2

            def min_func_upper(delta):
                ccomp = copy.copy(comp)
                ccomp.set_value(par0 + delta, index)
                # Hack to fix r & theta
                ccomp._fixed[1] = True
                ccomp._fixed[2] = True
                new_comps = comps.copy()
                new_comps.insert(i, ccomp)
                new_comps.pop(i+1)
                export_difmap_model(new_comps, "dfm.mdl", 15.4E+09)
                sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir,
                                                  nmodelfit=nmodelfit, use_pselfcal=True)
                return (required_chisq - sdict["rchisq"])**2

            # First find upper bound
            lower_bounds = [1E-7]
            upper_bounds = [1.0]
            n_fun_eval = 50
            delta_lower, _ = dlib.find_min_global(min_func_lower, lower_bounds, upper_bounds, n_fun_eval)
            delta_upper, _ = dlib.find_min_global(min_func_upper, lower_bounds, upper_bounds, n_fun_eval)

            print("Parameters = {} - {} + {}".format(par0, delta_lower, delta_upper))
            errors.append((delta_lower, delta_upper))
            # sys.exit(0)
    return errors


def find_2D_position_errors_using_chi2(dfm_model_file, uvfits, stokes="I", working_dir=None,
                                       delta_t_sec=10.0, gain_phase_tcoh=30.0,
                                       show_difmap_output=False, nmodelfit=50,
                                       unselfcalibrated_uvfits=None):

    # Nuber of time stamps for gain phases to "forget" their values
    t_eff = gain_phase_tcoh/delta_t_sec
    rho = np.exp(-(delta_t_sec/gain_phase_tcoh)**2)

    # Find gains DoF
    uvdata = UVData(uvfits)
    n_use_vis = uvdata.n_usable_visibilities_difmap(stokes=stokes)
    n_IF = uvdata.nif
    n_ant = len(uvdata.antennas)

    # Coefficient of correlation between gains in close IFs
    rho_IF = 0.75
    n_IF_eff = n_IF/(1+(n_IF-1)*rho_IF)


    baseline_scan_times = uvdata.baselines_scans_times
    n_scans = np.argmax(np.bincount([len(a) for a in baseline_scan_times.values()]))

    n_measurements = 0
    time_of_scan = list()
    for bl, scans in baseline_scan_times.items():
        for scan in scans:
            delta_t_scan_sec = (Time(scan[-1], format="jd") - Time(scan[0], format="jd")).sec
            time_of_scan.append(delta_t_scan_sec)
            n_measurements += delta_t_scan_sec/delta_t_sec
    time_of_scan = np.median(time_of_scan)

    n_eff_gain_phases = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff
    if stokes == "I":
        n_eff_gain_phases *= 2

    print("N_eff phases = ", n_eff_gain_phases)

    import dlib
    stat_dict = find_stat_of_difmap_model(dfm_model_file, uvfits, stokes, working_dir, nmodelfit=200, use_pselfcal=True,
                                          out_dfm_model="selfcaled.mdl")
    rchisq0 = stat_dict["rchisq"]
    dof = stat_dict["dof"] - n_eff_gain_phases
    print("DoF = ", dof)
    delta_rchisq = 2.0/dof
    required_chisq = rchisq0 + delta_rchisq
    # original_comps = import_difmap_model(dfm_model_file)
    original_comps = import_difmap_model("selfcaled.mdl")
    original_comps = sorted(original_comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    # comps = import_difmap_model(dfm_model_file)
    comps = import_difmap_model("selfcaled.mdl")
    comps = sorted(comps, key=lambda c: np.hypot(c.p[1], c.p[2]))

    errors = dict()

    for i, comp in enumerate(comps):
        errors[i] = list()
        # Cycle through the polar angle in coordinate system with origin at the best position
        for PA_cur in np.arange(0, np.deg2rad(360), np.deg2rad(15)):

            x0, y0 = comp.p[1], comp.p[2]
            # Convert to RA, DEC
            dec0 = -y0
            ra0 = -x0

            print("Searching along PA = {:.1f} deg around RA = {:.3f}, DEC = {:.3f}".format(np.rad2deg(PA_cur), ra0, dec0))

            def rot_shift(ra, dec, PA, delta):
                """
                RA&DEC after rotating on PA (rad) counter-clockwise around (ra, dec) and shifting on delta (mas)
                """
                return ra + delta*np.sin(PA), dec + delta*np.cos(PA)

            def min_func(delta):
                ra_new, dec_new = rot_shift(ra0, dec0, PA_cur, delta)
                ccomp = copy.copy(comp)
                # Set x (-RA)
                ccomp.set_value(-ra_new, 1)
                # Set y (-DEC)
                ccomp.set_value(-dec_new, 2)
                # Hack to fix r & theta
                ccomp._fixed[1] = True
                ccomp._fixed[2] = True
                new_comps = comps.copy()
                new_comps.insert(i, ccomp)
                new_comps.pop(i+1)
                export_difmap_model(new_comps, "dfm.mdl", 15.4E+09)
                sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, show_difmap_output,
                                                  nmodelfit=nmodelfit, use_pselfcal=True)
                print("For delta = {} differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
                return (required_chisq - sdict["rchisq"])**2

            # First find upper bound
            lower_bounds = [1E-7]
            upper_bounds = [1.0]
            n_fun_eval = 50
            delta_bound, _ = dlib.find_min_global(min_func, lower_bounds, upper_bounds, n_fun_eval)
            # delta_upper, _ = dlib.find_min_global(min_func_upper, lower_bounds, upper_bounds, n_fun_eval)

            print("Parameters: PA = {:.1f} deg, delta = {:.7f}".format(np.rad2deg(PA_cur), delta_bound[0]))
            errors[i].append((PA_cur, delta_bound[0]))
        # sys.exit(0)
    return errors


def convert_2D_position_errors_to_ell_components(dfm_model_file, errors, include_shfit=True):
    """
    :param dfm_model_file:
    :param errors:
        Dictionary with keys - component numbers and values lists of tuples with
        first element - theta [rad], second element - distance in this direction
        [mas]. Angle ``theta`` counts from North to East (as usually),
    :return:
        Iterable of elliptical components representing 2D positional errors.
    """
    from components import EGComponent
    comps = import_difmap_model(dfm_model_file)
    error_comps = list()
    for i, comp in enumerate(comps):
        error_list = errors[i]
        theta = list()
        r = list()
        for error_entry in error_list:
            theta.append(error_entry[0])
            r.append(error_entry[1])
        fit = fit_ellipse_to_2D_position_errors(theta, r)
        dRA = fit["xc"]
        dDEC = fit["yc"]
        a = fit["a"]
        b = fit["b"]
        bmaj = max(a, b)
        bmin = min(a, b)
        e = bmin/bmaj
        bpa = fit["theta"]
        # FIXME: Hack to fix skimage result
        if bpa < np.pi/2:
            bpa += np.pi/2
        print("Component #{}, BPA = {}".format(i, np.rad2deg(bpa)))
        comp_ra = -comp.p[1]
        comp_dec = -comp.p[2]
        if include_shfit:
            error_comp = EGComponent(0.0, -comp_ra-dRA, -comp_dec-dDEC, 2*bmaj, e,
                                     bpa + np.pi/2)
        else:
            error_comp = EGComponent(0.0, -comp_ra, -comp_dec, 2*bmaj, e,
                                     bpa + np.pi/2)
        error_comps.append(error_comp)
    return error_comps

# This return semi-axes lengths
def fit_ellipse_to_2D_position_errors(PA, dr, use="skimage"):

    t = np.array(PA)
    dr = np.array(dr)

    # Filter small values
    indx = dr > 1E-04
    t = t[indx]
    dr = dr[indx]

    x = dr*np.cos(t)
    y = dr*np.sin(t)

    if use == "skimage":
        from skimage.measure import EllipseModel
        ell = EllipseModel()
        ell.estimate(np.dstack((x, y))[0])
        xc, yc, a, b, theta = ell.params
        # xc - adds to DEC, yc - adds to RA, PA = theta + 90 deg
        return {"xc": xc, "yc": yc, "a": a, "b": b, "theta": theta}

    if use == "dlib":
        import dlib

        def min_func(a, b, theta):
            xt = a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t)
            yt = a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)
            return np.sum((x - xt)**2 + (y - yt)**2)

        lower_bounds = [1E-07, 1E-07, -np.pi]
        upper_bounds = [1.0, 1.0, np.pi]
        n_fun_eval = 1000
        res, _ = dlib.find_min_global(min_func, lower_bounds, upper_bounds, n_fun_eval)
        return {"a": res[0], "b": res[1], "theta": res[2]}

    else:
        raise Exception("Methods: skimage or dlib")


def modelfit_core_wo_extending(fname, beam_fractions, r_c=None,
                               mapsize_clean=None, path_to_script=None,
                               niter=50, stokes='i', path=None, out_path=None,
                               use_brightest_pixel_as_initial_guess=True,
                               flux_0=None, size_0=None, e_0=None, bpa_0=None,
                               show_difmap_output=False,
                               estimate_rms=False, use_ell=False):
    """
    Modelfit core after excluding extended emission around.

    :param fname:
        Filename of uv-data to modelfit.
    :param out_fname:
        Filename with output file with model.
    :param beam_fractions:
        Iterable of beam size fractions to consider while excluding extended
        emission.
    :param r_c: (optional)
        RA and DEC of the center of the circualr area. If ``None`` than
        use brightest pixel of phase center of the map depending on the
        ``use_brightest_pixel_as_initial_guess``. (default: ``None``)
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size).
    :param path_to_script:
        Path to ``clean`` difmap script.
    :param stokes: (optional)
        Stokes parameter 'i', 'q', 'u' or 'v'. (default: ``i``)
    :param path: (optional)
        Path to uv-data to modelfit. If ``None`` then use current directory.
        (default: ``None``)
    :param mdl_path: (optional)
        Path file with model. If ``None`` then use current directory.
        (default: ``None``)
    :param out_path: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param use_brightest_pixel_as_initial_guess: (optional)
        Boolean. Should use brightness pixel as initial guess for center of the
        core area? If ``False`` that use phase center. (default: ``True``)
    :param flux_0: (optional)
        Initial guess for core flux [Jy]. If ``None`` than use 1 Jy. (default:
        ``None``)
    :param size_0: (optional)
        Initial guess for core size [mas]. If ``None`` than use 1/10-th of the
        beam. (default: ``None``)
    :param e_0: (optional)
        Initial guess for eccentricity in case of the elliptic component. If
        ``None`` then use ``1``. (default: ``None``)
    :param bpa_0: (optional)
        Initial guess for BPA in case of the elliptic component. If
        ``None`` then use ``0``. (default: ``None``)
    :param show_difmap_output: (optional)
        Boolean. Show the output of difmap CLEAN and modelfit? (default:
        ``False``)
    :param estimate_rms: (optional)
        Boolean. Use dirty residual image after fitting the substracted data
        with a component to estimate rms? (default: ``False``)
    :param use_ell: (optional)
        Boolean. Use elliptic component? (default: ``False``)

    :return:
        Dictionary with keys - beam fractions used to exclude extended emission
        and values - dictionaries with core parameters obtained using this
        fractions.
    """
    from uv_data import UVData
    from components import (CGComponent, EGComponent)
    uvdata = UVData(os.path.join(path, fname))
    uvdata_tmp = UVData(os.path.join(path, fname))
    freq_hz = uvdata.frequency
    noise = uvdata.noise(use_V=False)
    # First CLEAN to obtain clean components
    clean_difmap(fname, os.path.join(out_path, "cc.fits"), stokes,
                 mapsize_clean, path=path, path_to_script=path_to_script,
                 show_difmap_output=show_difmap_output)
    from from_fits import create_clean_image_from_fits_file
    ccimage = create_clean_image_from_fits_file(os.path.join(out_path,
                                                             "cc.fits"))
    if r_c is None:
        # Find brightest pixel
        if use_brightest_pixel_as_initial_guess:
            im = np.unravel_index(np.argmax(ccimage.image), ccimage.image.shape)
            print("indexes of max intensity ", im)
            # - to RA cause dx_RA < 0
            r_c = (-(im[1]-mapsize_clean[0]/2)*mapsize_clean[1],
                   (im[0]-mapsize_clean[0]/2)*mapsize_clean[1])
        else:
            r_c = (0, 0)

    if flux_0 is None:
        flux_0 = 1.0
    if e_0 is None:
        e_0 = 1.0
    if bpa_0 is None:
        bpa_0 = 0.0

    beam = ccimage.beam
    beam = np.sqrt(beam[0]*beam[1])
    if size_0 is None:
        size_0 = 0.1*beam
    print("Using beam size = {} mas".format(beam))
    from from_fits import (create_model_from_fits_file,
                           create_image_from_fits_file)

    result = dict()

    for beam_fraction in sorted(beam_fractions):
        print("Estimating core using beam_fraction = {}".format(beam_fraction))
        print("1st iteration. Keeping core emission around RA={}, DEC={}".format(r_c[0], r_c[1]))
        # Need in RA, DEC
        model = create_model_from_fits_file(os.path.join(out_path, "cc.fits"))
        model.filter_components_by_r(beam_fraction*beam/2., r_c=r_c)
        uvdata.substitute([model])
        uvdata = uvdata_tmp - uvdata
        uvdata.save(os.path.join(out_path, "uv_diff.uvf"), rewrite=True)

        # Modelfit this uvdata with single component
        if not use_ell:
            comp0 = CGComponent(flux_0, -r_c[0], -r_c[1], size_0)
        else:
            comp0 = EGComponent(flux_0, -r_c[0], -r_c[1], size_0, e_0, bpa_0)

        export_difmap_model([comp0], os.path.join(out_path, "init.mdl"),
                            freq_hz)
        modelfit_difmap(fname=os.path.join(out_path, "uv_diff.uvf"),
                        mdl_fname=os.path.join(out_path, "init.mdl"),
                        out_fname=os.path.join(out_path, "it1.mdl"),
                        niter=niter, stokes=stokes.lower(), path=out_path,
                        mdl_path=out_path, out_path=out_path,
                        show_difmap_output=show_difmap_output)

        # Now use found component position to make circular filter around it
        comp = import_difmap_model("it1.mdl", out_path)[0]
        model_local = create_model_from_fits_file(os.path.join(out_path, "cc.fits"))
        # Need in RA, DEC
        r_c = (-comp.p[1], -comp.p[2])
        print("2nd iteration. Keeping core emission around RA={}, DEC={}".format(r_c[0], r_c[1]))
        model_local.filter_components_by_r(beam_fraction*beam/2., r_c=r_c)

        uvdata.substitute([model_local])
        uvdata = uvdata_tmp - uvdata
        uvdata.save(os.path.join(out_path, "uv_diff.uvf"), rewrite=True)

        # Modelfit this uvdata with single component
        print("Using 1st iteration component {} as initial guess".format(comp.p))
        if not use_ell:
            comp0 = CGComponent(comp.p[0], comp.p[1], comp.p[2], comp.p[3])
        else:
            comp0 = EGComponent(comp.p[0], comp.p[1], comp.p[2], comp.p[3],
                                comp.p[4], comp.p[5])

        export_difmap_model([comp0], os.path.join(out_path, "init.mdl"),
                            freq_hz)
        modelfit_difmap(fname=os.path.join(out_path, "uv_diff.uvf"),
                        mdl_fname=os.path.join(out_path, "init.mdl"),
                        out_fname=os.path.join(out_path, "it2.mdl"),
                        niter=niter, stokes=stokes.lower(), path=out_path,
                        mdl_path=out_path, out_path=out_path,
                        show_difmap_output=show_difmap_output,
                        save_dirty_residuals_map=estimate_rms,
                        dmap_size=mapsize_clean)
        comp = import_difmap_model("it2.mdl", out_path)[0]

        if estimate_rms:
            dimage = create_image_from_fits_file(os.path.join(out_path, "dirty_residuals_map.fits"))
            beam_pxl = int(beam/mapsize_clean[1])
            print("Beam = ", beam)
            # RA, DEC of component center
            r_center = (-int(comp.p[1]/mapsize_clean[1]),
                        -int(comp.p[2]/mapsize_clean[1]))
            # Need to add to map center (512, 512) add DEC in pixels (with sign)
            rms = mad_std(dimage.image[int(mapsize_clean[0]/2+r_center[1]-3*beam_pxl): int(mapsize_clean[0]/2+r_center[1]+3*beam_pxl),
                                       int(mapsize_clean[0]/2-3*beam_pxl): int(mapsize_clean[0]/2+3*beam_pxl)])
            # os.unlink(os.path.join(out_path, "dirty_residuals_map.fits"))

        else:
            rms = None

        if not use_ell:
            result.update({beam_fraction: {"flux": comp.p[0], "ra": -comp.p[1],
                                           "dec": -comp.p[2], "size": comp.p[3],
                                           "rms": rms}})
        else:
            result.update({beam_fraction: {"flux": comp.p[0], "ra": -comp.p[1],
                                           "dec": -comp.p[2], "size": comp.p[3],
                                           "e": comp.p[4], "bpa": comp.p[5],
                                           "rms": rms}})

    return result


def modelfit_core_wo_extending_single(fname, beam_fraction, r_c=None,
                                      mapsize_clean=None, path_to_script=None,
                                      niter=50, stokes='i', path=None, out_path=None,
                                      use_brightest_pixel_as_initial_guess=True,
                                      flux_0=None, size_0=None, e_0=None, bpa_0=None,
                                      show_difmap_output=False,
                                      estimate_rms=False, use_ell=False):
    """
    Modelfit core after excluding extended emission around.

    :param fname:
        Filename of uv-data to modelfit.
    :param out_fname:
        Filename with output file with model.
    :param beam_fraction:
        Beam size fraction to use while excluding extended emission.
    :param r_c: (optional)
        RA and DEC of the center of the circualr area. If ``None`` than
        use brightest pixel of phase center of the map depending on the
        ``use_brightest_pixel_as_initial_guess``. (default: ``None``)
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size).
    :param path_to_script:
        Path to ``clean`` difmap script.
    :param stokes: (optional)
        Stokes parameter 'i', 'q', 'u' or 'v'. (default: ``i``)
    :param path: (optional)
        Path to uv-data to modelfit. If ``None`` then use current directory.
        (default: ``None``)
    :param mdl_path: (optional)
        Path file with model. If ``None`` then use current directory.
        (default: ``None``)
    :param out_path: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param use_brightest_pixel_as_initial_guess: (optional)
        Boolean. Should use brightness pixel as initial guess for center of the
        core area? If ``False`` that use phase center. (default: ``True``)
    :param flux_0: (optional)
        Initial guess for core flux [Jy]. If ``None`` than use 1 Jy. (default:
        ``None``)
    :param size_0: (optional)
        Initial guess for core size [mas]. If ``None`` than use 1/10-th of the
        beam. (default: ``None``)
    :param e_0: (optional)
        Initial guess for eccentricity in case of the elliptic component. If
        ``None`` then use ``1``. (default: ``None``)
    :param bpa_0: (optional)
        Initial guess for BPA in case of the elliptic component. If
        ``None`` then use ``0``. (default: ``None``)
    :param show_difmap_output: (optional)
        Boolean. Show the output of difmap CLEAN and modelfit? (default:
        ``False``)
    :param estimate_rms: (optional)
        Boolean. Use dirty residual image after fitting the substracted data
        with a component to estimate rms? (default: ``False``)
    :param use_ell: (optional)
        Boolean. Use elliptic component? (default: ``False``)

    :return:
        Dictionary with core parameters and optionally image rms at the core
        position.
    """
    from uv_data import UVData
    from components import (CGComponent, EGComponent)
    uvdata = UVData(os.path.join(path, fname))
    uvdata_tmp = UVData(os.path.join(path, fname))
    freq_hz = uvdata.frequency
    noise = uvdata.noise(use_V=False)
    # First CLEAN to obtain clean components
    clean_difmap(fname, os.path.join(out_path, "cc.fits"), stokes,
                 mapsize_clean, path=path, path_to_script=path_to_script,
                 show_difmap_output=show_difmap_output)
    from from_fits import create_clean_image_from_fits_file
    ccimage = create_clean_image_from_fits_file(os.path.join(out_path,
                                                             "cc.fits"))
    if r_c is None:
        # Find brightest pixel
        if use_brightest_pixel_as_initial_guess:
            im = np.unravel_index(np.argmax(ccimage.image), ccimage.image.shape)
            print("indexes of max intensity ", im)
            # - to RA cause dx_RA < 0
            r_c = (-(im[1]-mapsize_clean[0]/2)*mapsize_clean[1],
                   (im[0]-mapsize_clean[0]/2)*mapsize_clean[1])
        else:
            r_c = (0, 0)

    if flux_0 is None:
        flux_0 = 1.0
    if e_0 is None:
        e_0 = 1.0
    if bpa_0 is None:
        bpa_0 = 0.0

    beam = ccimage.beam
    beam = np.sqrt(beam[0]*beam[1])
    if size_0 is None:
        size_0 = 0.1*beam
    print("Using beam size = {} mas".format(beam))
    from from_fits import (create_model_from_fits_file,
                           create_image_from_fits_file)

    print("Estimating core using beam_fraction = {}".format(beam_fraction))
    print("1st iteration. Keeping core emission around RA={}, DEC={}".format(r_c[0], r_c[1]))
    # Need in RA, DEC
    model = create_model_from_fits_file(os.path.join(out_path, "cc.fits"))
    model.filter_components_by_r(beam_fraction*beam/2., r_c=r_c)
    # extended structure visibilities
    uvdata.substitute([model])
    fig = uvdata.uvplot()
    fig.savefig(os.path.join(out_path, "extended_radplot.png"))
    uvdata = uvdata_tmp - uvdata
    uvdata.save(os.path.join(out_path, "uv_diff.uvf"), rewrite=True)

    # Modelfit this uvdata with single component
    if not use_ell:
        comp0 = CGComponent(flux_0, -r_c[0], -r_c[1], size_0)
    else:
        comp0 = EGComponent(flux_0, -r_c[0], -r_c[1], size_0, e_0, bpa_0)

    export_difmap_model([comp0], os.path.join(out_path, "init.mdl"),
                        freq_hz)
    modelfit_difmap(fname=os.path.join(out_path, "uv_diff.uvf"),
                    mdl_fname=os.path.join(out_path, "init.mdl"),
                    out_fname=os.path.join(out_path, "it1.mdl"),
                    niter=niter, stokes=stokes.lower(), path=out_path,
                    mdl_path=out_path, out_path=out_path,
                    show_difmap_output=show_difmap_output)

    # Now use found component position to make circular filter around it
    comp = import_difmap_model("it1.mdl", out_path)[0]
    model_local = create_model_from_fits_file(os.path.join(out_path, "cc.fits"))
    # Need in RA, DEC
    r_c = (-comp.p[1], -comp.p[2])
    print("2nd iteration. Keeping core emission around RA={}, DEC={}".format(r_c[0], r_c[1]))
    model_local.filter_components_by_r(beam_fraction*beam/2., r_c=r_c)

    uvdata.substitute([model_local])
    uvdata = uvdata_tmp - uvdata
    uvdata.save(os.path.join(out_path, "uv_diff.uvf"), rewrite=True)

    # Modelfit this uvdata with single component
    print("Using 1st iteration component {} as initial guess".format(comp.p))
    if not use_ell:
        comp0 = CGComponent(comp.p[0], comp.p[1], comp.p[2], comp.p[3])
    else:
        comp0 = EGComponent(comp.p[0], comp.p[1], comp.p[2], comp.p[3],
                            comp.p[4], comp.p[5])

    export_difmap_model([comp0], os.path.join(out_path, "init.mdl"),
                        freq_hz)
    modelfit_difmap(fname=os.path.join(out_path, "uv_diff.uvf"),
                    mdl_fname=os.path.join(out_path, "init.mdl"),
                    out_fname=os.path.join(out_path, "it2.mdl"),
                    niter=niter, stokes=stokes.lower(), path=out_path,
                    mdl_path=out_path, out_path=out_path,
                    show_difmap_output=show_difmap_output,
                    save_dirty_residuals_map=estimate_rms,
                    dmap_size=mapsize_clean)
    comp = import_difmap_model("it2.mdl", out_path)[0]

    if estimate_rms:
        dimage = create_image_from_fits_file(os.path.join(out_path, "dirty_residuals_map.fits"))
        beam_pxl = int(beam/mapsize_clean[1])
        print("Beam = ", beam)
        # RA, DEC of component center
        r_center = (-int(comp.p[1]/mapsize_clean[1]),
                    -int(comp.p[2]/mapsize_clean[1]))
        # Need to add to map center (512, 512) add DEC in pixels (with sign)
        rms = mad_std(dimage.image[int(mapsize_clean[0]/2+r_center[1]-3*beam_pxl): int(mapsize_clean[0]/2+r_center[1]+3*beam_pxl),
                                   int(mapsize_clean[0]/2-3*beam_pxl): int(mapsize_clean[0]/2+3*beam_pxl)])
        # os.unlink(os.path.join(out_path, "dirty_residuals_map.fits"))

    else:
        rms = None

    if not use_ell:
        result = {"flux": comp.p[0], "ra": -comp.p[1], "dec": -comp.p[2],
                  "size": comp.p[3], "rms": rms}
    else:
        result = {"flux": comp.p[0], "ra": -comp.p[1], "dec": -comp.p[2],
                  "size": comp.p[3], "e": comp.p[4], "bpa": comp.p[5],
                  "rms": rms}

    return result


def modelfit_core_wo_extending_it(fname, beam_fractions, r_c=None,
                                  mapsize_clean=None, path_to_script=None,
                                  niter=50, stokes='i', path=None, out_path=None,
                                  use_brightest_pixel_as_initial_guess=True,
                                  flux_0=None, size_0=None, e_0=None, bpa_0=None,
                                  show_difmap_output=False,
                                  estimate_rms=False, use_ell=False):
    """
    Iterative Modelfit core after excluding extended emission around.

    :param fname:
        Filename of uv-data to modelfit.
    :param out_fname:
        Filename with output file with model.
    :param beam_fractions:
        Iterable of beam size fractions to consider while excluding extended
        emission.
    :param r_c: (optional)
        RA and DEC of the center of the circualr area. If ``None`` than
        use brightest pixel of phase center of the map depending on the
        ``use_brightest_pixel_as_initial_guess``. (default: ``None``)
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size).
    :param path_to_script:
        Path to ``clean`` difmap script.
    :param stokes: (optional)
        Stokes parameter 'i', 'q', 'u' or 'v'. (default: ``i``)
    :param path: (optional)
        Path to uv-data to modelfit. If ``None`` then use current directory.
        (default: ``None``)
    :param mdl_path: (optional)
        Path file with model. If ``None`` then use current directory.
        (default: ``None``)
    :param out_path: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param use_brightest_pixel_as_initial_guess: (optional)
        Boolean. Should use brightness pixel as initial guess for center of the
        core area? If ``False`` that use phase center. (default: ``True``)
    :param flux_0: (optional)
        Initial guess for core flux [Jy]. If ``None`` than use 1 Jy. (default:
        ``None``)
    :param size_0: (optional)
        Initial guess for core size [mas]. If ``None`` than use 1/10-th of the
        beam. (default: ``None``)
    :param e_0: (optional)
        Initial guess for eccentricity in case of the elliptic component. If
        ``None`` then use ``1``. (default: ``None``)
    :param bpa_0: (optional)
        Initial guess for BPA in case of the elliptic component. If
        ``None`` then use ``0``. (default: ``None``)
    :param show_difmap_output: (optional)
        Boolean. Show the output of difmap CLEAN and modelfit? (default:
        ``False``)
    :param estimate_rms: (optional)
        Boolean. Use dirty residual image after fitting the substracted data
        with a component to estimate rms? (default: ``False``)
    :param use_ell: (optional)
        Boolean. Use elliptic component? (default: ``False``)

    :return:
        Dictionary with keys - beam fractions used to exclude extended emission
        and values - dictionaries with core parameters obtained using this
        fractions.
    """
    from uv_data import UVData
    from components import (CGComponent, EGComponent)
    uvdata = UVData(os.path.join(path, fname))
    uvdata_tmp = UVData(os.path.join(path, fname))
    freq_hz = uvdata.frequency
    noise = uvdata.noise(use_V=False)
    # First CLEAN to obtain clean components
    clean_difmap(fname, os.path.join(out_path, "cc.fits"), stokes,
                 mapsize_clean, path=path, path_to_script=path_to_script,
                 show_difmap_output=show_difmap_output)
    from from_fits import create_clean_image_from_fits_file
    ccimage = create_clean_image_from_fits_file(os.path.join(out_path,
                                                             "cc.fits"))
    if r_c is None:
        # Find brightest pixel
        if use_brightest_pixel_as_initial_guess:
            im = np.unravel_index(np.argmax(ccimage.image), ccimage.image.shape)
            print("indexes of max intensity ", im)
            # - to RA cause dx_RA < 0
            r_c = (-(im[1]-mapsize_clean[0]/2)*mapsize_clean[1],
                   (im[0]-mapsize_clean[0]/2)*mapsize_clean[1])
        else:
            r_c = (0, 0)

    if flux_0 is None:
        flux_0 = 1.0
    if e_0 is None:
        e_0 = 1.0
    if bpa_0 is None:
        bpa_0 = 0.0

    beam = ccimage.beam
    beam = np.sqrt(beam[0]*beam[1])
    if size_0 is None:
        size_0 = 0.1*beam
    print("Using beam size = {} mas".format(beam))
    from from_fits import (create_model_from_fits_file,
                           create_image_from_fits_file)

    result = dict()

    for beam_fraction in sorted(beam_fractions):
        print("Estimating core using beam_fraction = {}".format(beam_fraction))
        print("1st iteration. Keeping core emission around RA={}, DEC={}".format(r_c[0], r_c[1]))
        # Need in RA, DEC
        model = create_model_from_fits_file(os.path.join(out_path, "cc.fits"))
        model.filter_components_by_r(beam_fraction*beam/2., r_c=r_c)
        uvdata.substitute([model])
        uvdata = uvdata_tmp - uvdata
        uvdata.save(os.path.join(out_path, "uv_diff.uvf"), rewrite=True)

        # Modelfit this uvdata with single component
        if not use_ell:
            comp0 = CGComponent(flux_0, -r_c[0], -r_c[1], size_0)
        else:
            comp0 = EGComponent(flux_0, -r_c[0], -r_c[1], size_0, e_0, bpa_0)

        export_difmap_model([comp0], os.path.join(out_path, "init.mdl"),
                            freq_hz)
        modelfit_difmap(fname=os.path.join(out_path, "uv_diff.uvf"),
                        mdl_fname=os.path.join(out_path, "init.mdl"),
                        out_fname=os.path.join(out_path, "it1.mdl"),
                        niter=niter, stokes=stokes.lower(), path=out_path,
                        mdl_path=out_path, out_path=out_path,
                        show_difmap_output=show_difmap_output)

        # Now use found component position to make circular filter around it
        comp = import_difmap_model("it1.mdl", out_path)[0]
        model_local = create_model_from_fits_file(os.path.join(out_path, "cc.fits"))
        # Need in RA, DEC
        r_c = (-comp.p[1], -comp.p[2])
        print("2nd iteration. Keeping core emission around RA={}, DEC={}".format(r_c[0], r_c[1]))
        model_local.filter_components_by_r(beam_fraction*beam/2., r_c=r_c)

        uvdata.substitute([model_local])
        uvdata = uvdata_tmp - uvdata
        uvdata.save(os.path.join(out_path, "uv_diff.uvf"), rewrite=True)

        # Modelfit this uvdata with single component
        print("Using 1st iteration component {} as initial guess".format(comp.p))
        if not use_ell:
            comp0 = CGComponent(comp.p[0], comp.p[1], comp.p[2], comp.p[3])
        else:
            comp0 = EGComponent(comp.p[0], comp.p[1], comp.p[2], comp.p[3],
                                comp.p[4], comp.p[5])

        export_difmap_model([comp0], os.path.join(out_path, "init.mdl"),
                            freq_hz)
        modelfit_difmap(fname=os.path.join(out_path, "uv_diff.uvf"),
                        mdl_fname=os.path.join(out_path, "init.mdl"),
                        out_fname=os.path.join(out_path, "it2.mdl"),
                        niter=niter, stokes=stokes.lower(), path=out_path,
                        mdl_path=out_path, out_path=out_path,
                        show_difmap_output=show_difmap_output,
                        save_dirty_residuals_map=estimate_rms,
                        dmap_size=mapsize_clean)
        comp = import_difmap_model("it2.mdl", out_path)[0]

        if estimate_rms:
            dimage = create_image_from_fits_file(os.path.join(out_path, "dirty_residuals_map.fits"))
            beam_pxl = int(beam/mapsize_clean[1])
            print("Beam = ", beam)
            # RA, DEC of component center
            r_center = (-int(comp.p[1]/mapsize_clean[1]),
                        -int(comp.p[2]/mapsize_clean[1]))
            # Need to add to map center (512, 512) add DEC in pixels (with sign)
            rms = mad_std(dimage.image[int(mapsize_clean[0]/2+r_center[1]-3*beam_pxl): int(mapsize_clean[0]/2+r_center[1]+3*beam_pxl),
                                       int(mapsize_clean[0]/2-3*beam_pxl): int(mapsize_clean[0]/2+3*beam_pxl)])
            # os.unlink(os.path.join(out_path, "dirty_residuals_map.fits"))

        else:
            rms = None

        if not use_ell:
            result.update({beam_fraction: {"flux": comp.p[0], "ra": -comp.p[1],
                                           "dec": -comp.p[2], "size": comp.p[3],
                                           "rms": rms}})
        else:
            result.update({beam_fraction: {"flux": comp.p[0], "ra": -comp.p[1],
                                           "dec": -comp.p[2], "size": comp.p[3],
                                           "e": comp.p[4], "bpa": comp.p[5],
                                           "rms": rms}})

    return result


def make_map_with_core_at_zero(mdl_file, uv_fits_fname, mapsize_clean,
                               path_to_script, outfname="shifted_cc.fits",
                               stokes="I"):
    """
    Function that shifts map in a way that core in new map will have zero
    coordinates.

    :param mdl_file:
        Path to difmap model file with the core being the first component.
    :param uv_fits_fname:
        Path to UV FITS data.
    :param mapsize_clean:
        Iterable of number of pixels and pixel size [mas].
    :param path_to_script:
        Path to D.Homan difmap final CLEAN script.
    :param outfname:
        Where to save new FITS file with map.
    :param stokes: (optional)
        Stokes parameter. (default: ``I``)
    """
    mdl_dir, mdl_fn = os.path.split(mdl_file)
    uv_dir, uv_fn = os.path.split(uv_fits_fname)
    core = import_difmap_model(mdl_file, mdl_dir)[0]
    ra_mas = -core.p[1]
    dec_mas = -core.p[2]
    shift = (-ra_mas, -dec_mas)
    clean_difmap(uv_fn, outfname, stokes, mapsize_clean, uv_dir, path_to_script,
                 shift=shift)


if __name__ == "__main__":
    uvfits = "/home/ilya/github/jetpol/scripts/2200+420.u.2017_01_28.uvf"
    dfm_model = "/home/ilya/Documents/Tb/MOJAVE_difmap_models/modfits/2200+420_2017-01-28.mod"
    df = components_info(uvfits, dfm_model, dmap_size=(1024, 0.1), PA=None)