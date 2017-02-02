import os
import numpy as np
from components import DeltaComponent, CGComponent, EGComponent


def clean_n(fname, outfname, stokes, mapsize_clean, niter=100,
            path_to_script=None, mapsize_restore=None, beam_restore=None,
            outpath=None, shift=None, show_difmap_output=False,
            windows=None):
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    difmapout = open("difmap_commands", "w")
    difmapout.write("observe " + fname + "\n")
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    if windows is not None:
        difmapout.write("rwin " + str(windows) + "\n")
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
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < difmap_commands 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


# TODO: add ``shift`` argument, that shifts image before cleaning. It must be
# more accurate to do this in difmap. Or add such method in ``UVData`` that
# multiplies uv-data on exp(-1j * (u*x_shift + v*y_shift)).
def clean_difmap(fname, outfname, stokes, mapsize_clean, path=None,
                 path_to_script=None, mapsize_restore=None, beam_restore=None,
                 outpath=None, shift=None, show_difmap_output=False,
                 command_file=None):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean: (optional)
        Parameters of map for cleaning (map size, pixel size). If ``None``
        then use those of map in map directory (not bootstrapped).
        (default: ``None``)
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

    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    if command_file is None:
        command_file = "difmap_commands"

    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    # if shift is not None:
    #     difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
                    str(mapsize_restore[1]) + "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


def import_difmap_model(mdl_fname, mdl_dir=None):
    """
    Function that reads difmap-format model and returns list of ``Components``
    instances.

    :param mdl_fname:
        File name with difmap model.
    :param mdl_dir: (optional)
        Directory with difmap model. If ``None`` then use CWD. (default:
        ``None``)
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
                flux, radius, theta = line.split()
                axial = 1.0
                major = 0.0
                type_ = 0

        list_fixed = list()
        if flux[-1] != 'v':
            list_fixed.append('flux')

        x = -float(radius[:-1]) * np.sin(np.deg2rad(float(theta[:-1])))
        y = -float(radius[:-1]) * np.cos(np.deg2rad(float(theta[:-1])))
        flux = float(flux[:-1])

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


def modelfit_difmap(fname, mdl_fname, out_fname, niter=50, stokes='i',
                    path=None, mdl_path=None, out_path=None):
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
    """
    if path is None:
        path = os.getcwd()
    if mdl_path is None:
        mdl_path = os.getcwd()
    if out_path is None:
        out_path = os.getcwd()

    difmapout = open("difmap_commands", "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + os.path.join(mdl_path, mdl_fname) + "\n")
    difmapout.write("modelfit " + str(niter) + "\n")
    difmapout.write("wmodel " + os.path.join(out_path, out_fname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    os.system("difmap < difmap_commands")

# # DIFMAP_MAPPSR
# def difmap_mappsr(source, isll, centre_ra_deg, centre_dec_deg, uvweightstr,
#                   experiment, difmappath, uvprefix, uvsuffix, jmsuffix, \
#                   saveagain):
#     difmapout = open("difmap_commands", "w")
#     difmapout.write("float pkflux\n")
#     difmapout.write("float peakx\n")
#     difmapout.write("float peaky\n")
#     difmapout.write("float finepeakx\n")
#     difmapout.write("float finepeaky\n")
#     difmapout.write("float rmsflux\n")
#     difmapout.write("integer ilevs\n")
#     difmapout.write("float lowlev\n")
#     difmapout.write("obs " + sourceuvfile + "\n")
#     difmapout.write("mapsize 1024," + str(pixsize) + "\n")
#     difmapout.write("select ll,1,2,3,4\n")
#     difmapout.write("invert\n");
#     difmapout.write("wtscale " + str(threshold) + "\n")
#     difmapout.write("obs " + sourceuvfile + "\n")
#     if reweight:
#         difmapout.write("uvaver 180,true\n")
#     else:
#         difmapout.write("uvaver 20\n")
#     difmapout.write("mapcolor none\n")
#     difmapout.write("uvweight " + uvweightstr + "\n")
#
#     #Find the peak from the combined first
#     if isll:
#         difmapout.write("select ll,1,2,3,4\n")
#     else:
#         difmapout.write("select i,1,2\n")
#     if experiment == 'v190k':
#         difmapout.write("select rr,1,2,3,4\n")
#     difmapout.write("peakx = peak(x,max)\n")
#     difmapout.write("peaky = peak(y,max)\n")
#     difmapout.write("shift -peakx,-peaky\n")
#     difmapout.write("mapsize 1024,0.1\n")
#     difmapout.write("finepeakx = peak(x,max)\n")
#     difmapout.write("finepeaky = peak(y,max)\n")
#
#     #Do each individual band, one at a time
#     maxif = 4
#     pols = ['rr','ll']
#     for i in range(maxif):
#         for p in pols:
#             difmapout.write("select " + p + "," + str(i+1) + "\n")
#             write_difmappsrscript(source, p + "." + str(i+1), difmapout, \
#                                   pixsize, jmsuffix)
#     write_difmappsrscript(source, 'combined', difmapout, \
#                           pixsize, jmsuffix)
#     if saveagain:
#         difmapout.write("wobs " + rootdir + "/" + experiment + "/noise" + \
#                         source + ".uvf\n")
#     difmapout.write("exit\n")
#     difmapout.close()
#     os.system(difmappath + " < difmap_commands")
#
# # WRITE_DIFMAPPSRSCRIPT
# def write_difmappsrscript(source, bands, difmapout, pixsize, jmsuffix):
#     difmapout.write("clrmod true\n")
#     difmapout.write("unshift\n")
#     difmapout.write("shift -peakx,-peaky\n")
#     difmapout.write("mapsize 1024,0.1\n")
#     difmapout.write("pkflux = peak(flux,max)\n")
#     difmapout.write("addcmp pkflux, true, finepeakx, finepeaky, true, 0, " + \
#                     "false, 1, false, 0, false, 0, 0, 0\n")
#     difmapout.write("mapsize 1024," + str(pixsize) + "\n")
#     difmapout.write("modelfit 50\n")
#     difmapout.write("rmsflux = imstat(rms)\n")
#     difmapout.write("restore\n")
#     difmapout.write("pkflux = peak(flux,max)\n")
#     difmapout.write("ilevs = pkflux/rmsflux\n")
#     difmapout.write("lowlev = 300.0/ilevs\n")
#     difmapout.write("loglevs lowlev\n")
#     imagefile = rootdir + '/' + experiment + '/' + source + '.' + bands + \
#                 '.image.fits' + jmsuffix
#     if os.path.exists(imagefile):
#         os.system("rm -f " + imagefile)
#     difmapout.write("wmap " + imagefile + "\n")
#     difmapout.write("unshift\n")
#     difmapout.write("wmod\n")
#     difmapout.write("print rmsflux\n")
#    difmapout.write("print imstat(bmin)\n")
#    difmapout.write("print imstat(bmaj)\n")
#    difmapout.write("print imstat(bpa)\n")
