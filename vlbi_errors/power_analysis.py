import os
from spydiff import clean_difmap
from uv_data import UVData
from_from_fits import create_clean_model_from_fits_file
from bootstrap import CleanBootstrap


def cov_analysis_image(uv_fits_path, n_boot, cc_fits_path=None, imsize=None,
    path_to_script=None, stokes='I', outdir=None):
        """
    Function that runs coverage analysis of bootstrap CIs using
    user-specified FITS-file with uv-data and optional CLEAN model.
    
    :param uv_fits_path:
        Path to FITS-file with uv-data.
    :param n_boot:
        Number of bootstrap replications to use when calculating CIs.
    :cc_fits_path: (optional)
        Path to FITS-file with CLEAN models. This models will
        be used as model for power analysis. If ``None`` then CLEAN uv-data
        first and use result as real model for calculating coverage.
        (default: ``None``)
    :imsize: (optional)
        Image parameters (image size [pix], pixel size [mas]) to use
        when doing first CLEAN with ``cc_fits_path = None``.
    :stokes: (optional)
        Stokes parameter to deal with. (default: ``I``)
    :outdir: (optional)
        Directory to store files. If ``None`` then use CWD. (default:
        ``None``)
    """
    if cc_fits_paths is None:
        if imsize is None:
            raise Exception("Specify ``imszie``")
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        clean_difmap(uv_fits_fname, 'cc.fits', stokes, imsize, path=uv_fits_dir,
            path_to_script=path_to_script, outpath=outdir)
        cc_fits_path = os.path.join(outdir, 'cc.fits')

    uvdata = UVData(uv_fits_path)
    ccmodel = create_clean_model_from_fits_file(cc_fits_path)
    bt = CleanBootstrap(uvdata, [ccmodel])
    cwd = os.getcwd()
    os.chdir(outdir)
    bt.run(outname=['uv_boot', 'uvf'], n=n_boot)
    os.chdir(cwd)
    
    uv_fits_paths = glob.glob(os.path.join(outdir, 'uv_boot*.uvf'))
    for i, uv_fits_path in enumerate(uv_fits_paths):
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        clean_difmap(uv_fits_fname, 'cc_{}.fits'.format(i), stokes,
            imsize, path=uv_fits_dir, path_to_script=path_to_script,
            outpath=outdir)

        



def image_analysis(uv_fits_path, n_p, n_boot, cc_fits_paths=None, imsize=None,
    path_to_script=None, stokes='I', outdir=None):
    """
    Function that runs coverage analysis of bootstrap CIs using
    user-specified FITS-files with uv-data.
    
    :param uv_fits_path:
        Path to FITS-file with uv-data.
    :param n_p:
        Number of times to add noise to real model and calculate
        bootstrap error.
    :param n_boot:
        Number of bootstrap replications to use when calculating CIs.
    :cc_fits_paths: (optional)
        List of path to FITS-files with CLEAN models. This models will
        be used as model for power analysis. If ``None`` then CLEAN uv-data
        first and use result as real model for calculating coverage.
        (default: ``None``)
    :imsize: (optional)
        Image parameters (image size [pix], pixel size [mas]) to use
        when doing first CLEAN with ``cc_fits_path = None``.
    :stokes: (optional)
        Stokes parameter to deal with. (default: ``I``)
    :outdir: (optional)
        Directory to store files. If ``None`` then use CWD. (default:
        ``None``)
    """
    if cc_fits_paths is None:
        if imsize is None:
            raise Exception("Specify ``imszie``")
        uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
        clean_difmap(uv_fits_fname, 'cc.fits', stokes, imsize, path=uv_fits_dir,
            path_to_script=path_to_script, outpath=outdir)
        cc_fits_paths = [os.path.join(outdir, 'cc.fits')]
    if len(cc_fits_paths) == 1:
        cc_fits_paths *= n_p
        
    uvdata = UVData(uv_fits_path)
    noise = uvdata.noise()
    # Circle through ``cc_fits_paths``, add noise, create uv-data and run bootstrap
    # analysis of this uv-data with CLEAN model to get bootstrap CIs. Count CIs
    # containing TRUE values.
    for i, cc_fits_path in enumerate(cc_fits_paths):
        ccmodel = create_clean_model_from_fits_file(cc_fits_path)
        cc_fits_dir, cc_fits_fname = os.path.split(cc_fits_path)
        uvdata.substitute([ccmodel])
        uvdata.save("uv_{}.fits".format(i))
        
        
    
    
def gradient_analysis():
    pass