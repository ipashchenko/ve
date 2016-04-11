import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from utils import gen_rand_vecs, hdi_of_arrays, unwrap_phases, create_mask
from pixel import resolver_chisq
from conf_bands import create_sim_conf_band


def rotm_model(p, freqs):
    lambdasq = (3. * 10 ** 8 / freqs) ** 2
    return p[0] * lambdasq + p[1]

def weighted_residuals(p, freqs, chis, s_chis):
    return (chis - rotm_model(p, freqs)) / s_chis

def residuals(p, freqs, chis):
    return chis - rotm_model(p, freqs)


def add_dterm_evpa(q, u, i, evpa, d_term, n_ant, n_if, n_scans):
    """
    Function that adds D-term contribution to Q & U images.

    Parameter ``d_term`` supposed to be an estimate of residual D-term. It adds
    to ``q`` & ``u`` images with random phase thus this functions only usefull
    in Monter Carlo simulations of residual D-term contribution to errors in eg.
    ROTM.

    :param q:
        2D numpy array of stokes Q.
    :param u:
        2D numpy array of stokes U.
    :param i:
        2D numpy array of stokes I.
    :param evpa:
        Value of EVPA rotation.
    :param d_term:
        D-term value.
    :param n_ant:
        Number of antennas.
    :param n_if:
        Number of IF channels.
    :param n_scans:
        Number of scans with independent parallactic angels.
    :return:
        Two 2D numpy arrays - Q & U stokes images with added random phase D-term
        contribution.

    """
    assert q.shape == u.shape == i.shape

    # Create image of D-term contribution.
    i_peak = np.max(i.ravel())
    image = d_term * np.sqrt(i ** 2. + (0.3 * i_peak) ** 2) / np.sqrt(n_ant *
                                                                      n_if *
                                                                      n_scans)
    # Calculated image of D-term contribution adds to PPOL so to add it to Q & U
    # images one should add it as random vector with the same length.
    vec = gen_rand_vecs(2, 1)[0]
    q_ = q + image * vec[0]
    u_ = u + image * vec[1]

    p_new = (q_ + 1j * u_) * np.exp(-1j * 2. * np.deg2rad(evpa))
    q_ = np.real(p_new)
    u_ = np.imag(p_new)

    return q_, u_


def hovatta_find_sigma_pang(q, u, i, sigma_evpa, d_term, n_ant, n_if, n_scan,
                            rms_region):
    """
    Function that calculates uncertainty images of PANG & PPOL using (Hovatta et
    al. 2012) approach.

    :param q:
        2D numpy array of stokes Q.
    :param u:
        2D numpy array of stokes U.
    :param i:
        2D numpy array of stokes I.
    :param sigma_evpa:
        EVPA calibration error [deg].
    :param d_term:
        D-term calibration error.
    :param n_ant:
        Number of antennas. All antennas are supposed to have alt-azimuthal
        mounting.
    :param n_if:
        Number of Intermediate Frequency channels.
    :param n_scan:
        Number of scans with independent parallactic angles.
    :param rms_region:
        Region to include in rms calculation. Or (blc[0], blc[1], trc[0],
        trc[1],) or (center[0], center[1], r, None,). If ``None`` then use
        all image in rms calculation. Default ``None``.
    :return:
        Two 2D numpy arrays with the same shape as input images. First - with
        uncertainties of PANG and second - with uncertainties of PPOL.

    """
    assert q.shape == u.shape == i.shape

    # Create images of D-terms uncertainty
    i_peak = np.max(i.ravel())
    sigma_d_image = d_term * np.sqrt(i ** 2. + (0.3 * i_peak) ** 2) / \
        np.sqrt(n_ant * n_if * n_scan)

    # For each stokes Q & U find rms error
    rms_dict = dict()
    for stokes, image in zip(('q', 'u'), (q, u)):
        mask = create_mask(image.shape, rms_region)
        image = np.ma.array(image, mask=~mask)
        rms_dict[stokes] = np.ma.std(image.ravel())

    # Find overall Q & U error that is sum of rms and D-terms uncertainty
    overall_errors_dict = dict()
    for stoke in ('q', 'u'):
        overall_sigma = np.sqrt(rms_dict[stoke] ** 2. + sigma_d_image ** 2. +
                                (1.5 * rms_dict[stoke]) ** 2.)
        overall_errors_dict[stoke] = overall_sigma

    # Find EVPA & PPOL errors
    evpa_error_image = np.sqrt((q * overall_errors_dict['u']) ** 2. +
                               (u * overall_errors_dict['q']) ** 2.) /\
        (2. * (q ** 2. + u ** 2.))
    ppol_error_image = 0.5 * (overall_errors_dict['u'] +
                              overall_errors_dict['q'])

    # Add EVPA calibration uncertainty
    evpa_error_image = np.sqrt(evpa_error_image ** 2. +
                               np.deg2rad(sigma_evpa) ** 2.)

    return evpa_error_image, ppol_error_image


# FIXME: This functions just use ``rotm`` function for each pixel.
def rotm_map(freqs, chis, s_chis=None, mask=None, outfile=None, outdir=None,
             ext='png', mask_on_chisq=True):
    """
    Function that calculates Rotation Measure map.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of 2D numpy arrays with polarization positional angles [rad].
    :param s_chis: (optional)
        Iterable of 2D numpy arrays with polarization positional angles
        uncertainties estimates [rad].
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask. Note that ``mask`` must have dimensions of only one image,
        that is it should be 2D array.
    :param mask_on_chisq: (optional)
        Mask chi squared values that are larger then critical value? (default:
        ``True``)

    :return:
        Tuple of 2D numpy array with values of Rotation Measure [rad/m**2], 2D
        numpy array with uncertainties map [rad/m**2] and 2D numpy array with
        chi squared values of linear fit.

    """
    # Critical values (alpha=0.05) of chi-squared distribution for different]
    # dofs.
    chisq_crit_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
                         6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307}
    freqs = np.array(freqs)

    # Asserts on data consistency
    if s_chis is not None:
        assert len(freqs) == len(chis) == len(s_chis)
    else:
        assert len(freqs) == len(chis)

    chi_cube = np.dstack(chis)
    if s_chis is not None:
        s_chi_cube = np.dstack(s_chis)

    # Initialize arrays for storing results and fill them with NaNs
    rotm_array = np.empty(np.shape(chi_cube[:, :, 0]))
    s_rotm_array = np.empty(np.shape(chi_cube[:, :, 0]))
    chisq_array = np.empty(np.shape(chi_cube[:, :, 0]))
    rotm_array[:] = np.nan
    s_rotm_array[:] = np.nan
    chisq_array[:] = np.nan

    if mask is None:
        mask = np.zeros(rotm_array.shape, dtype=int)

    # If saving output thern create figure with desired number of cells.
    if outfile:
        if outdir is None:
            outdir = '.'
        # If the directory does not exist, create it
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Calculate how many pixels there should be
        npixels = len(np.where(mask.ravel() == 0)[0])
        print "{} pixels with fit will be plotted".format(npixels)
        nrows = int(np.sqrt(npixels) + 1)
        print "Plot will have dims: {} by {}".format(nrows, nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, sharex=True,
                                 sharey=True)
        fig.set_size_inches(18.5, 18.5)
        plt.rcParams.update({'axes.titlesize': 'small'})
        i, j = 0, 0

    # Cycle ROTM calculation for each unmasked pixel
    for (x, y), value in np.ndenumerate(rotm_array):
        # If pixel should be masked then just pass by and leave NaN as value
        if mask[x, y]:
            continue

        if s_chis is not None:
            p, pcov, s_sq = rotm(freqs, chi_cube[x, y, :], s_chi_cube[x, y, :])
        else:
            p, pcov, s_sq = rotm(freqs, chi_cube[x, y, :])

        if mask_on_chisq and s_sq > chisq_crit_values[len(chis) - 2]:
            chisq_array[x, y] = s_sq
            rotm_array[x, y] = np.nan
            s_rotm_array[x, y] = np.nan
        if pcov is not np.nan:
            chisq_array[x, y] = s_sq
            rotm_array[x, y] = p[0]
            s_rotm_array[x, y] = np.sqrt(pcov[0, 0])
        else:
            chisq_array[x, y] = s_sq
            rotm_array[x, y] = p[0]
            s_rotm_array[x, y] = np.nan

        # Plot to file
        if outfile:
            lambdasq = (3. * 10 ** 8 / freqs) ** 2
            if s_chis is not None:
                axes[i, j].errorbar(lambdasq, chi_cube[x, y, :],
                                    s_chi_cube[x, y, :], fmt='.k')
            else:
                axes[i, j].plot(lambdasq, chi_cube[x, y, :], '.k')
            lambdasq_ = np.linspace(lambdasq[0], lambdasq[-1], 10)
            axes[i, j].plot(lambdasq_,
                            rotm_model(p, 3. * 10 ** 8 / np.sqrt(lambdasq_)),
                            'r', lw=2, label="RM={0:.1f}".format(p[0]))
            axes[i, j].set_title("{}-{}".format(x, y))
            axes[i, j].legend(prop={'size': 6}, loc='best', fancybox=True,
                              framealpha=0.5)
            # Check this text box
            # ax.hist(x, 50)
            # # these are matplotlib.patch.Patch properties
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # # place a text box in upper left in axes coords
            # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            # verticalalignment='top', bbox=props)

            axes[i, j].set_xticks([lambdasq[0], lambdasq[-1]])
            axes[i, j].set_ylim(-np.pi, np.pi)
            j += 1
            # Plot first row first
            if j // nrows > 0:
                # Then second row, etc...
                i += 1
                j = 0

    # Save to file plotted figure
    if outfile:
        path = os.path.join(outdir, outfile)
        print "Saving linear fits to {}.{}".format(path, ext)
        fig.show()
        fig.savefig("{}.{}".format(path, ext), bbox_inches='tight', dpi=200)

    return rotm_array, s_rotm_array, chisq_array


# D-term contribution and EVPA-calibration uncertainty.
def pang_map(q_array, u_array, mask=None):
    """
    Function that calculates Polarization Angle map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Polarization Angle values [rad].

    :note:
        ``q_array`` & ``u_array`` must have the same units (e.g. [Jy/beam])

    """
    q_array = np.atleast_2d(q_array)
    u_array = np.atleast_2d(u_array)
    assert q_array.shape == u_array.shape

    if mask is not None:
        q_array = np.ma.array(q_array, mask=mask, fill_value=np.nan)
        u_array = np.ma.array(u_array, mask=mask, fill_value=np.nan)

    return 0.5 * np.arctan2(u_array, q_array)


def cpol_map(q_array, u_array, mask=None):
    """
    Function that calculates Complex Polarization map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Complex Polarization values.

    :note:
        ``q_array`` & ``u_array`` must have the same units (e.g. [Jy/beam]),
        then output array will have the same units.

    """
    q_array = np.atleast_2d(q_array)
    u_array = np.atleast_2d(u_array)
    assert q_array.shape == u_array.shape

    if mask is not None:
        q_array = np.ma.array(q_array, mask=mask, fill_value=np.nan)
        u_array = np.ma.array(u_array, mask=mask, fill_value=np.nan)

    return q_array + 1j * u_array


def pol_map(q_array, u_array, mask=None):
    """
    Function that calculates Polarization Flux map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Polarization Flux values.

    :note:
        ``q_array`` & ``u_array`` must have the same units (e.g. [Jy/beam])

    """
    cpol_array = cpol_map(q_array, u_array, mask=mask)
    return np.sqrt(cpol_array * cpol_array.conj()).real


def fpol_map(q_array, u_array, i_array, mask=None):
    """
    Function that calculates Fractional Polarization map.

    :param q_array:
        Numpy 2D array of Stokes Q values.
    :param u_array:
        Numpy 2D array of Stokes U values.
    :param i_array:
        Numpy 2D array of Stokes I values.
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask.

    :return:
        Numpy 2D array of Fractional Polarization values.

    :note:
        ``q_array``, ``u_array`` & ``i_array`` must have the same units (e.g.
        [Jy/beam])

    """
    cpol_array = cpol_map(q_array, u_array, mask=mask)
    return np.sqrt(cpol_array * cpol_array.conj()).real / i_array


def rotm(freqs, chis, s_chis=None, p0=None):
    """
    Function that calculates Rotation Measure.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of polarization positional angles [rad].
    :param s_chis: (optional)
        Iterable of polarization positional angles uncertainties estimates
        [rad].
    :param p0:
        Starting value for minimization (RM [rad/m**2], PA_zero_lambda [rad]).

    :return:
        Tuple of numpy array of (RM [rad/m**2], PA_zero_lambda [rad]) and 2D
        numpy array of covariance matrix.

    """

    if p0 is None:
        p0 = [0., 0.]

    if s_chis is not None:
        assert len(freqs) == len(chis) == len(s_chis)
    else:
        assert len(freqs) == len(chis)

    p0 = np.array(p0)
    freqs = np.array(freqs)
    chis = np.array(chis)
    if s_chis is not None:
        s_chis = np.array(s_chis)

    # Try to unwrap angles
    chis = unwrap_phases(chis)
    # Resolve ``n pi`` ambiguity resolved
    lambdasq = (3. * 10 ** 8 / freqs) ** 2
    chis = resolver_chisq(lambdasq, chis, s_chi=s_chis, p0=p0)

    if s_chis is None:
        func, args = residuals, (freqs, chis,)
    else:
        func, args = weighted_residuals, (freqs, chis, s_chis,)
    fit = leastsq(func, p0, args=args, full_output=True)
    (p, pcov, infodict, errmsg, ier) = fit

    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)

    s_sq = (func(p, *args) ** 2.).sum() / (len(chis) - len(p0))
    if (len(chis) > len(p0)) and pcov is not None:
        # Residual variance
        pcov *= s_sq
    else:
        pcov = np.nan

    return p, pcov, s_sq


def jet_direction(image, rmin=0, rmax=200, dr=4, plots=False):
    """ Find jet direction. Return array [radius, polar angle]"""

    phis = []
    fluxes = []
    leny = np.shape(image)[0]
    lenx = np.shape(image)[1]
    y, x = np.mgrid[-leny/2: leny/2, -lenx/2: lenx/2]
    rads = np.arange(rmin, rmax, dr)
    for r in rads:
        mask = np.logical_and(r**2 <= x*x+y*y, x*x+y*y <= (r+dr)**2)
        x1 = x[mask]
        y1 = y[mask]
        img1 = image[mask]
        angles = np.arctan2(y1, x1)
        phi = np.arctan2((img1 * np.sin(angles)).sum() / img1.sum(),
                         (img1 * np.cos(angles)).sum() / img1.sum())
        phis.append(phi)
        fluxes.append(np.ma.mean(img1))
    f = np.array(phis)
    if plots:
        plt.figure()
        plt.imshow(image)
        plt.plot(rads*np.cos(f)+leny/2, rads*np.sin(f)+lenx/2, '.k')
        plt.xlim(0, lenx)
        plt.ylim(0, leny)

    return rads, f, fluxes


# TODO: Add as method to ``images.Images``
def rms_image(image):
    """
    Calculate rms of Image.

    :param image:
        Instance of ``Image`` class.
    :return:
        Value of rms.
    """
    r_rms = image.imsize[0] / 10
    rms = 0.25 * image.rms(region=(r_rms, r_rms, r_rms, None))
    rms += 0.25 * image.rms(region=(image.imsize[0] - r_rms, r_rms, r_rms,
                                    None))
    rms += 0.25 * image.rms(region=(r_rms, image.imsize[0] - r_rms, r_rms,
                                    None))
    rms += 0.25 * image.rms(region=(image.imsize[0] - r_rms,
                                    image.imsize[0] - r_rms, r_rms, None))
    return rms


# TODO: Add as method to ``images.Images``
def pol_mask(stokes_image_dict, n_sigma=2.):
    """
    Find mask using stokes 'I' map and 'PPOL' map using specified number of
    sigma.
    :param stokes_image_dict:
        Dictionary with keys - stokes, values - instances of ``Image``.
    :param n_sigma:
        Number of sigma to consider for stokes 'I' and 'PPOL'. 1, 2 or 3.
    :return:
        Logical array of mask.
    """
    quantile_dict = {1: 0.68, 2: 0.95, 3: 0.99}
    rms_cs_dict = {stokes: rms_image(stokes_image_dict[stokes]) for stokes in
                   ('I', 'Q', 'U')}
    qu_rms = np.mean([rms_cs_dict[stoke] for stoke in ('Q', 'U')])
    ppol_quantile = qu_rms * np.sqrt(-np.log((1. -
                                              quantile_dict[n_sigma]) ** 2.))
    i_cs_mask = stokes_image_dict['I'].image < n_sigma * rms_cs_dict['I']
    ppol_cs_image = pol_map(stokes_image_dict['Q'].image,
                            stokes_image_dict['U'].image)
    ppol_cs_mask = ppol_cs_image < ppol_quantile
    return np.logical_or(i_cs_mask, ppol_cs_mask)


def analyze_rotm_slice(slice_coords, rotm_image, rotm_images,
                       conf_band_alpha=0.95, outdir=None,
                       outfname='rotm_slice_spread.png'):
    """
    Analyze ROTM slice.

    :param slice_coords:
        Iterable of to points (x1, y1) in pixels that are coordinates of slice.
    :param rotm_image:
        Instance of ``Image`` class with original ROTM map.
    :param rotm_images:
        Instance of ``Images`` class with bootstrapped ROTM maps.
    :param conf_band_alpha: (optional)
        Confidence to use (0-1). (default: ``0.95``)
    :param outdir:

    :return:
    """
    # Setting up the output directory
    if outdir is None:
        outdir = os.getcwd()
    print("Using output directory {}".format(outdir))

    # Calculate simultaneous confidence bands
    # Bootstrap slices
    slices = list()
    for image in rotm_images.images:
        slice_ = image.slice(slice_coords)
        slices.append(slice_[~np.isnan(slice_)])

    # Find means
    obs_slice = rotm_image.slice(slice_coords)
    length = int(round(np.hypot(slice_coords[1][0] - slice_coords[0][0],
                                slice_coords[1][1] - slice_coords[0][1])))
    x = np.arange(length)
    x = x[~np.isnan(obs_slice)]
    obs_slice = obs_slice[~np.isnan(obs_slice)]
    # Find sigmas
    slices_ = [arr.reshape((1, len(obs_slice))) for arr in slices]
    sigmas = hdi_of_arrays(slices_).squeeze()
    means = np.mean(np.vstack(slices), axis=0)
    diff = obs_slice - means
    # Move bootstrap curves to original simulated centers
    slices_ = [slice_ + diff for slice_ in slices]
    # Find low and upper confidence band
    low, up = create_sim_conf_band(slices_, obs_slice, sigmas,
                                   alpha=conf_band_alpha)

    # Plot confidence bands and model values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, low[::-1], 'g')
    ax.plot(x, up[::-1], 'g')
    [ax.plot(x, slice_[::-1], 'r', lw=0.15) for slice_ in slices_]
    ax.plot(x, obs_slice[::-1], '.k')
    fig.savefig(os.path.join(outdir, outfname), bbox_inches='tight', dpi=200)
    plt.close()


def plot_image_correlation(image, fname='correlation.png', show=True,
                           close=False, savefig=True, outdir=None):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('Pixles')
    ax.set_ylabel('Pixles')
    R = np.corrcoef(image)
    im = ax.matshow(R, cmap='hot')
    colorbar_ax = fig.add_axes([0.80, 0.10, 0.05, 0.80])
    cb = fig.colorbar(im, cax=colorbar_ax)
    cb = fig.colorbar(im, cax=colorbar_ax)
    cb.set_label('Correlation')
    if savefig:
        if outdir is None:
            outdir = os.getcwd()
        fig.savefig(os.path.join(outdir, fname), bbox_inches='tight', dpi=200)
    if show:
        fig.show()
    if close:
        plt.close()


