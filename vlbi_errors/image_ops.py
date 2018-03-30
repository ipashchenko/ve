import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from skimage.morphology import medial_axis
from scipy.ndimage.filters import gaussian_filter

from utils import (gen_rand_vecs, hdi_of_arrays, unwrap_phases, create_mask,
                   v_round)
from pixel import resolver_chisq
from conf_bands import create_sim_conf_band
from skel_utils import (isolateregions, pix_identify, init_lengths, pre_graph,
                        longest_path, prune_graph, extremum_pts, main_length,
                        make_final_skeletons, recombine_skeletons)


def rotm_model(p, freqs):
    lambdasq = (3. * 10 ** 8 / freqs) ** 2
    return p[0] * lambdasq + p[1]


def spix_model(p, freqs):
    return p[0] * np.log(freqs) + p[1]


def spix_weighted_residuals(p, freqs, fluxes, s_fluxes):
    return (np.log(fluxes) - spix_model(p, freqs)) / (s_fluxes / fluxes)


def spix_residuals(p, freqs, fluxes):
    return np.log(fluxes) - spix_model(p, freqs)


def rotm_weighted_residuals(p, freqs, chis, s_chis):
    return (chis - rotm_model(p, freqs)) / s_chis


def rotm_residuals(p, freqs, chis):
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


def hovatta_find_sigma_pang(q, u, i, sigma_evpa, d_term, n_ant, n_if, n_scan):
    """
    Function that calculates uncertainty images of PANG & PPOL using (Hovatta et
    al. 2012) approach.

    :param q:
        Instance of ``Image`` class for stokes Q.
    :param u:
        Instance of ``Image`` class for stokes U.
    :param i:
        Instance of ``Image`` class for stokes I.
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
    :return:
        Two 2D numpy arrays with the same shape as input images. First - with
        uncertainties of PANG and second - with uncertainties of PPOL.

    """
    assert q == u == i

    # Create images of D-terms uncertainty
    i_peak = np.max(i.image.ravel())
    sigma_d_image = d_term * np.sqrt(i.image ** 2. + (0.3 * i_peak) ** 2) / \
        np.sqrt(n_ant * n_if * n_scan)

    # For each stokes Q & U find rms error w/o Hovatta's correction
    rms_dict = dict()
    for stokes, image in zip(('q', 'u'), (q, u)):
        rms_dict[stokes] = rms_image(image, hovatta_factor=False)

    # Find overall Q & U error that is sum of rms and D-terms uncertainty plus
    # Hovatta factor of CLEAN errors.
    overall_errors_dict = dict()
    for stoke in ('q', 'u'):
        overall_sigma = np.sqrt(rms_dict[stoke] ** 2. + sigma_d_image ** 2. +
                                (1.5 * rms_dict[stoke]) ** 2.)
        overall_errors_dict[stoke] = overall_sigma

    # Find EVPA & PPOL errors
    evpa_error_image = np.sqrt((q.image * overall_errors_dict['u']) ** 2. +
                               (u.image * overall_errors_dict['q']) ** 2.) /\
        (2. * (q.image ** 2. + u.image ** 2.))
    ppol_error_image = 0.5 * (overall_errors_dict['u'] +
                              overall_errors_dict['q'])

    # Add EVPA calibration uncertainty
    evpa_error_image = np.sqrt(evpa_error_image ** 2. +
                               np.deg2rad(sigma_evpa) ** 2.)

    return evpa_error_image, ppol_error_image


# FIXME: This functions just use ``rotm`` function for each pixel.
# TODO: Add outputting PA at zero frequency
def rotm_map(freqs, chis, s_chis=None, mask=None, outfile=None, outdir=None,
             mask_on_chisq=True, plot_pxls=None, outfile_pxls=None):
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
    :param plot_pxls: (optional)
        Iterable of pixel coordinates to plot fit. If ``None`` then don't plot
        single pixel fits. (default: ``None``)
    :param outfile_pxls: (optional)
        Optional outfile postfix if plotting single pixel fits. If ``None`` then
        don't use any postfix. (default: ``None``)

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

    # If saving output then create figure with desired number of cells.
    if outfile or outfile_pxls:
        if outdir is None:
            outdir = os.getcwd()
        # If the directory does not exist, create it
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # Cycle ROTM calculation for each unmasked pixel
    for (x, y), value in np.ndenumerate(rotm_array):
        if plot_pxls is not None and (x, y) in plot_pxls:
            plot = True
            outfile = 'ROTM_fit_{}_{}'.format(x, y)
        else:
            plot = False
        # If pixel should be masked then just pass by and leave NaN as value
        if mask[x, y]:
            if plot_pxls is not None:
                if (x, y) in plot_pxls:
                    print("But Masking out")
            continue

        if s_chis is not None:
            p, pcov, s_sq = rotm(freqs, chi_cube[x, y, :], s_chi_cube[x, y, :],
                                 plot=plot, outdir=outdir, outfname=outfile)
        else:
            p, pcov, s_sq = rotm(freqs, chi_cube[x, y, :], plot=plot,
                                 outdir=outdir, outfname=outfile)

        if mask_on_chisq and s_sq * (len(chis) - 2) > chisq_crit_values[len(chis) - 2]:
            chisq_array[x, y] = s_sq * (len(chis) - 2)
            rotm_array[x, y] = np.nan
            s_rotm_array[x, y] = np.nan
        elif pcov is not np.nan:
            chisq_array[x, y] = s_sq * (len(chis) - 2)
            rotm_array[x, y] = p[0]
            s_rotm_array[x, y] = np.sqrt(pcov[0, 0])
        else:
            chisq_array[x, y] = s_sq * (len(chis) - 2)
            rotm_array[x, y] = p[0]
            s_rotm_array[x, y] = np.nan

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


def rotm(freqs, chis, s_chis=None, p0=None, plot=False, plot_title=None,
         outfname=None, outdir=None):
    """
    Function that calculates Rotation Measure.

    :param freqs:
        Iterable of frequencies [Hz].
    :param chis:
        Iterable of polarization positional angles [rad].
    :param s_chis: (optional)
        Iterable of polarization positional angles uncertainties estimates
        [rad]. If ``None`` then don't use weights in fitting. (default:
        ``None``)
    :param p0: (optional)
        Starting value for minimization (RM [rad/m**2], PA_zero_lambda [rad]).
        If ``None`` then use ``(0., 0.)``. (default: ``None``)
    :param plot: (optional)
        Boolean. Plot linear fit on single figure? (default: ``False``)
    :param plot_title: (optional)
        Optional title on plot. If ``None`` then don't show any title.
        (default: ``None``)
    :param outfname: (optional)
        File name for saving figure. If ``None`` then use ``ROTM_fit.png``
        (default: ``None``)
    :param outdir: (optional)
        Directory to save figure. If ``None`` then use CWD. (default: ``None``)

    :return:
        Tuple of numpy array of (RM [rad/m**2], PA_zero_lambda [rad]), 2D
        numpy array of covariance matrix & reduced chi-squared value.

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
        func, args = rotm_residuals, (freqs, chis,)
    else:
        func, args = rotm_weighted_residuals, (freqs, chis, s_chis,)
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

    if plot:
        fig, axes = plt.subplots()
        lambdasq = (3. * 10 ** 8 / freqs) ** 2
        print("P: {}, cov: {}".format(p, pcov))
        if s_chis is not None:
            axes.errorbar(lambdasq, np.rad2deg(chis), np.rad2deg(s_chis),
                          fmt='.k')
        else:
            axes.plot(lambdasq, np.rad2deg(chis), '.k')
        lambdasq_ = np.linspace(lambdasq[0], lambdasq[-1], 10)
        axes.plot(lambdasq_,
                  np.rad2deg(rotm_model(p, 3. * 10 ** 8 / np.sqrt(lambdasq_))), 'r',
                  lw=2, label="RM={0:.1f} +/- {1:.1f} rad/m/m".format(p[0], np.sqrt(pcov[0, 0])))
        if plot_title is not None:
            axes.set_title(plot_title)
        axes.legend(prop={'size': 6}, loc='best', fancybox=True,
                    framealpha=0.5)
        axes.set_xticks([lambdasq[0], lambdasq[-1]])
        # axes_.set_ylim(-np.pi, np.pi)
        axes.set_ylabel("PA, [deg]")

        if outfname is None:
            outfname = "ROTM_fit"
        if outdir is None:
            outdir = os.getcwd()
        path = os.path.join(outdir, outfname)
        fig.savefig("{}.png".format(path), bbox_inches='tight',
                    dpi=200)

    return p, pcov, s_sq


def spix(freqs, fluxes, s_fluxes=None, p0=None):
    """
    Function that calculates SPectral IndeX.

    :param freqs:
        Iterable of frequencies [Hz].
    :param fluxes:
        Iterable of flux values [Jy].
    :param s_fluxes: (optional)
        Iterable of fluxes uncertainties estimates [Jy].
    :param p0:
        Starting value for minimization (SPIX []).

    :return:
        Tuple of value of SPIX [] and numpy array of covariance matrix.

    """

    if p0 is None:
        p0 = [0., 0.]

    if s_fluxes is not None:
        assert len(freqs) == len(fluxes) == len(s_fluxes)
    else:
        assert len(freqs) == len(fluxes)

    p0 = np.array(p0)
    freqs = np.array(freqs)
    chis = np.array(fluxes)
    if s_fluxes is not None:
        s_fluxes = np.array(s_fluxes)

    if s_fluxes is None:
        func, args = spix_residuals, (freqs, fluxes,)
    else:
        func, args = spix_weighted_residuals, (freqs, fluxes, s_fluxes,)
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


def spix_map(freqs, flux_maps, s_flux_maps=None, mask=None, outfile=None,
             outdir=None, ext='png', mask_on_chisq=False):
    """
    Function that calculates SPectral IndeX map.

    :param freqs:
        Iterable of frequencies [Hz].
    :param flux_maps:
        Iterable of 2D numpy arrays with fluxes [Jy].
    :param s_flux_maps: (optional)
        Iterable of 2D numpy arrays with flux uncertainties estimates [Jy].
    :param mask: (optional)
        Mask to be applied to arrays before calculation. If ``None`` then don't
        apply mask. Note that ``mask`` must have dimensions of only one image,
        that is it should be 2D array.
    :param mask_on_chisq: (optional)
        Mask chi squared values that are larger then critical value? (default:
        ``False``)

    :return:
        Tuple of 2D numpy array with values of Spectral Index [], 2D numpy array
        with uncertainties map [] and 2D numpy array with chi squared values of
        log-log linear fit.

    """
    # Critical values (alpha=0.05) of chi-squared distribution for different]
    # dofs.
    chisq_crit_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
                         6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307}
    freqs = np.array(freqs)

    # Asserts on data consistency
    if s_flux_maps is not None:
        assert len(freqs) == len(flux_maps) == len(s_flux_maps)
    else:
        assert len(freqs) == len(flux_maps)

    flux_cube = np.dstack(flux_maps)
    if s_flux_maps is not None:
        s_flux_cube = np.dstack(s_flux_maps)

    # Initialize arrays for storing results and fill them with NaNs
    spix_array = np.empty(np.shape(flux_cube[:, :, 0]))
    s_spix_array = np.empty(np.shape(flux_cube[:, :, 0]))
    chisq_array = np.empty(np.shape(flux_cube[:, :, 0]))
    spix_array[:] = np.nan
    s_spix_array[:] = np.nan
    chisq_array[:] = np.nan

    if mask is None:
        mask = np.zeros(spix_array.shape, dtype=int)

    # If saving output then create figure with desired number of cells.
    if outfile:
        if outdir is None:
            outdir = '.'
        # If the directory does not exist, create it
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Calculate how many pixels there should be
        npixels = len(np.where(mask.ravel() == 0)[0])
        print("{} pixels with fit will be plotted".format(npixels))
        nrows = int(np.sqrt(npixels) + 1)
        print("Plot will have dims: {} by {}".format(nrows, nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, sharex=True,
                                 sharey=True)
        fig.set_size_inches(18.5, 18.5)
        plt.rcParams.update({'axes.titlesize': 'small'})
        i, j = 0, 0

    # Cycle SPIX calculation for each unmasked pixel
    for (x, y), value in np.ndenumerate(spix_array):
        # If pixel should be masked then just pass by and leave NaN as value
        if mask[x, y]:
            continue

        if s_flux_maps is not None:
            p, pcov, s_sq = spix(freqs, flux_cube[x, y, :], s_flux_cube[x, y, :])
        else:
            p, pcov, s_sq = spix(freqs, flux_cube[x, y, :])

        if mask_on_chisq and s_sq > chisq_crit_values[len(flux_maps) - 2]:
            chisq_array[x, y] = s_sq
            spix_array[x, y] = np.nan
            s_spix_array[x, y] = np.nan
        if pcov is not np.nan:
            chisq_array[x, y] = s_sq
            spix_array[x, y] = p[0]
            s_spix_array[x, y] = np.sqrt(pcov[0, 0])
        else:
            chisq_array[x, y] = s_sq
            spix_array[x, y] = p[0]
            s_spix_array[x, y] = np.nan

        # Plot to file
        if outfile:
            if s_flux_maps is not None:
                axes[i, j].errorbar(freqs, flux_cube[x, y, :],
                                    s_flux_cube[x, y, :], fmt='.k')
            else:
                axes[i, j].plot(freqs, flux_cube[x, y, :], '.k')
            freqs_ = np.linspace(freqs[0], freqs[-1], 10)
            axes[i, j].plot(freqs_,
                            np.exp(spix_model(p, np.log(freqs_))),
                            'r', lw=2, label="SPIX={0:.1f}".format(p[0]))
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

            axes[i, j].set_xticks([freqs_[0], freqs_[-1]])
            # axes[i, j].set_ylim(-np.pi, np.pi)
            j += 1
            # Plot first row first
            if j // nrows > 0:
                # Then second row, etc...
                i += 1
                j = 0

    # Save to file plotted figure
    if outfile:
        path = os.path.join(outdir, outfile)
        print("Saving linear fits to {}.{}".format(path, ext))
        fig.show()
        fig.savefig("{}.{}".format(path, ext), bbox_inches='tight', dpi=200)

    return spix_array, s_spix_array, chisq_array


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
        plt.imshow(np.log(image))
        plt.plot(rads*np.cos(f)+leny/2, rads*np.sin(f)+lenx/2, '.k')
        plt.xlim(0, lenx)
        plt.ylim(0, leny)

    return rads, f, fluxes


def jet_ridge_line(image, r_max, beam=None, dr=1, n=1.):
    """

    :param image:
    :param r_max:
    :param beam:
        Iterable of ``bmaj``, ``bmin``, ``bpa`` [pxl, pxl, rad]. If ``(r, None,
        None)`` then use ``circular_mean``.
    :param dr:
    :param n:
        Scale factor to decrease beam size to average before ridge line
        construction.
    :return:
    """
    from utils import circular_mean, elliptical_mean
    try:
        if beam[1] is None:
            mean_image = circular_mean(image, beam[0])
            print("Using circular filter with r = {}".format(beam[0]))
        else:
            mean_image = elliptical_mean(image, beam[0] / n, beam[1] / n, beam[2])
            print("Using elliptical filter with bmaj = {}, bmin = {}, bpa ="
                  " {}".format(beam[0]/n, beam[1]/n, beam[2]))
    except TypeError:
        print("No filter")
        mean_image = image
    leny, lenx = np.shape(mean_image)
    y, x = np.mgrid[-leny/2: leny/2, -lenx/2: lenx/2]
    coords = list()
    xy_max = np.unravel_index(np.argmax(mean_image), np.shape(mean_image))
    coords.append(xy_max)
    for r in range(1, r_max):
        mask = np.logical_and(r**2 < x * x + y * y, x * x + y * y <=
                              (r + dr) ** 2)
        img1 = np.ma.array(mean_image, mask=~mask)
        xy_max = np.unravel_index(np.ma.argmax(img1), np.shape(mean_image))
        coords.append(xy_max)
    plt.figure()
    plt.imshow(np.log(image))
    coords_ = np.atleast_2d(coords)
    plt.scatter(coords_[:, 1], coords_[:, 0])

    return coords


def jet_skeleton(image, data_dir=None, n_rms=3, hovatta_factor=False,
                 do_filter_before_rms_cut=True, filter_width=5):
    if data_dir is None:
        data_dir = os.getcwd()
    rms = rms_image(image, hovatta_factor=hovatta_factor)
    data = image.image.copy()
    if do_filter_before_rms_cut:
        data = gaussian_filter(data, filter_width)
    mask = data < n_rms * rms
    data[mask] = 0
    data[~mask] = 1

    skel, distance = medial_axis(data, return_distance=True)
    dist_on_skel = distance * skel

    # Plot area and skeleton
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})
    ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.contour(data, [0.5], colors='w')
    ax2.axis('off')
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(data_dir, 'skeleton_orig.png'))
    plt.close()

    isolated_filaments, num, offsets = isolateregions(skel)

    interpts, hubs, ends, filbranches, labeled_fil_arrays = \
        pix_identify(isolated_filaments, num)

    branch_properties = init_lengths(labeled_fil_arrays, filbranches, offsets, data)
    branch_properties["number"] = filbranches

    edge_list, nodes = pre_graph(labeled_fil_arrays, branch_properties, interpts,
                                 ends)

    max_path, extremum, G = longest_path(edge_list, nodes, verbose=True,
                                         save_png=False,
                                         skeleton_arrays=labeled_fil_arrays)

    updated_lists = prune_graph(G, nodes, edge_list, max_path, labeled_fil_arrays,
                                branch_properties, length_thresh=20,
                                relintens_thresh=0.1)
    labeled_fil_arrays, edge_list, nodes,  branch_properties = updated_lists

    filament_extents = extremum_pts(labeled_fil_arrays, extremum, ends)

    length_output = main_length(max_path, edge_list, labeled_fil_arrays, interpts,
                                branch_properties["length"], 1, verbose=True)
    filament_arrays = {}
    lengths, filament_arrays["long path"] = length_output
    lengths = np.asarray(lengths)

    filament_arrays["final"] = make_final_skeletons(labeled_fil_arrays, interpts,
                                                    verbose=True)

    skeleton = recombine_skeletons(filament_arrays["final"], offsets, data.shape,
                                   0, verbose=True)
    skeleton_longpath = recombine_skeletons(filament_arrays["long path"], offsets,
                                            data.shape, 1)
    skeleton_longpath_dist = skeleton_longpath * distance

    # Plot area and skeleton
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})
    ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(skeleton_longpath_dist, cmap=plt.cm.spectral,
               interpolation='nearest')
    ax2.contour(data, [0.5], colors='w')
    ax2.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(data_dir, 'skeleton.png'))
    plt.show()


def image_ridge_line(image):
    rms = rms_image(image)
    im = image.image.copy()
    im_shape = np.shape(im)
    im[im < 5. * rms] = 0
    # (2, imsize, imsize) array with gradient (in x & y -directions)
    grad = np.gradient(im, edge_order=2)
    from utils import hessian
    # (2, 2, imsize, imsize) array with second derivates
    hess = hessian(im)
    det_ar = np.zeros(im_shape)
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            det_ar[i, j] = np.linalg.det(hess[:, :, i, j])


# TODO: Add as method to ``images.Images``
def rms_image(image, hovatta_factor=True):
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
    if hovatta_factor:
        rms *= 1.8

    return rms


def rms_image_shifted(uv_fits_path, hovatta_factor=True, shift=(1000, 1000),
                      tmp_name='shifted_clean_map.fits', tmp_dir=None,
                      stokes='I', image=None, image_fits=None,
                      mapsize_clean=None, path_to_script=None,
                      niter=None):
    """
    Estimate image per-pixel rms using shifted image.
    """
    path, uv_fits_fname = os.path.split(uv_fits_path)
    if tmp_dir is None:
        tmp_dir = os.getcwd()
    if path_to_script is None:
        raise Exception("Provide location of difmap final CLEAN script!")
    if mapsize_clean is None and image is not None:
        import utils
        pixsize = abs(image.pixsize[0]) / utils.mas_to_rad
        mapsize_clean = (image.imsize[0], pixsize)
    if mapsize_clean is None and image_fits is not None:
        import from_fits
        image = from_fits.create_image_from_fits_file(image_fits)
        import utils
        pixsize = abs(image.pixsize[0]) / utils.mas_to_rad
        mapsize_clean = (image.imsize[0], pixsize)

    import spydiff
    if niter is None:
        spydiff.clean_difmap(uv_fits_fname, tmp_name, stokes=stokes,
                             mapsize_clean=mapsize_clean, path=path,
                             path_to_script=path_to_script, outpath=tmp_dir,
                             shift=shift)
    else:
        spydiff.clean_n(uv_fits_path, tmp_name, stokes=stokes,
                        mapsize_clean=mapsize_clean, niter=niter,
                        path_to_script=path_to_script, outpath=tmp_dir,
                        shift=shift)

    import from_fits
    image = from_fits.create_image_from_fits_file(os.path.join(tmp_dir,
                                                               tmp_name))
    return rms_image(image, hovatta_factor=hovatta_factor)


# TODO: Add as method to ``images.Images``
def pol_mask(stokes_image_dict, uv_fits_path=None, n_sigma=2.,
             path_to_script=None):
    """
    Find mask using stokes 'I' map and 'PPOL' map using specified number of
    sigma.
    :param stokes_image_dict:
        Dictionary with keys - stokes, values - instances of ``Image``.
    :param uv_fits_path: (optional)
        Path to uv-fits files.
    :param n_sigma: (optional)
        Number of sigma to consider for stokes 'I' and 'PPOL'. 1, 2 or 3.
        (default: ``2``)
    :param path_to_script:
        Path to difmap final CLEAN script.
    :return:
        Logical array of mask.
    """
    if path_to_script is None:
        raise Exception("Provide location of difmap final CLEAN script!")

    quantile_dict = {1: 0.6827, 2: 0.9545, 3: 0.9973, 4: 0.99994}
    # If no UV-data then calculate rms using outer parts of images
    if uv_fits_path is None:
        rms_cs_dict = {stokes: rms_image(stokes_image_dict[stokes]) for stokes
                       in ('I', 'Q', 'U')}
    else:
        rms_cs_dict = {stokes:
                           rms_image_shifted(uv_fits_path, stokes=stokes,
                                             image=stokes_image_dict[stokes],
                                             path_to_script=path_to_script)
                       for stokes in ('I', 'Q', 'U')}

    qu_rms = np.mean([rms_cs_dict[stoke] for stoke in ('Q', 'U')])
    ppol_quantile = qu_rms * np.sqrt(-np.log((1. -
                                              quantile_dict[n_sigma]) ** 2.))
    i_cs_mask = stokes_image_dict['I'].image < n_sigma * rms_cs_dict['I']
    ppol_cs_image = pol_map(stokes_image_dict['Q'].image,
                            stokes_image_dict['U'].image)
    ppol_cs_mask = ppol_cs_image < ppol_quantile
    return np.logical_or(i_cs_mask, ppol_cs_mask)


def analyze_rotm_slice(slice_coords, rotm_image, sigma_rotm_image=None,
                       rotm_images=None, conf_band_alpha=0.95, outdir=None,
                       outfname='rotm_slice_spread', beam_width=None,
                       model_rotm_image=None, ylim=None, fig=None,
                       external_sigma_slice=None, show_dots_boot=True):
    """
    Analyze ROTM slice.

    :param slice_coords:
        Iterable of points (x, y) in pixels that are coordinates of slice.
    :param rotm_image:
        Instance of ``Image`` class with original ROTM map.
    :param sigma_rotm_image: (optional)
        Instance of ``Image`` class with error ROTM map. If ``None`` then use
        ``rotm_images`` argument and build simultaneous confidence band.
        (default: ``None``)
    :param rotm_images: (optional)
        Instance of ``Images`` class with bootstrapped ROTM maps. If ``None``
        then use ``sigma_rotm_image`` argument and plot pointwise error bars.
        (default: ``None``)
    :param conf_band_alpha: (optional)
        Confidence to use when calculating simultaneous confidence band (0-1).
        (default: ``0.95``)
    :param outdir: (optional)
        Directory to save figure. If ``None`` then use CWD. (default: ``None``)
    :param outfname: (optional)
        File name for saved figure. (default: ``rotm_slice_spread.png``)
    :param beam_width: (optional)
        Beam width in pixels to plot. If ``None`` then don't plot. (default:
        ``None``)
    :param rotm_image: (optional)
        Instance of ``Image`` class with model ROTM map beam-convolved.
        Actually, model Q & U maps are beam convolved and then ROTM map is
        found. If ``None`` then don't use model values. (default: ``None``)
    :param external_sigma_slice: (optional)
        Iterable of \sigma for additional errorbar in bootstrap replications
        slices plot.
    """
    label_size = 14
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    # Setting up the output directory
    if outdir is None:
        outdir = os.getcwd()
    print("Using output directory {}".format(outdir))

    # Find means
    obs_slice = rotm_image.slice(pix1=slice_coords[0], pix2=slice_coords[1])
    length = int(round(np.hypot(slice_coords[1][0] - slice_coords[0][0],
                                slice_coords[1][1] - slice_coords[0][1])))
    x = np.arange(length)
    # x = x[~np.isnan(obs_slice)]
    # obs_slice_notna = obs_slice[~np.isnan(obs_slice)]

    if sigma_rotm_image is not None:
        sigma_slice = sigma_rotm_image.slice(pix1=slice_coords[0],
                                             pix2=slice_coords[1])
        sigma_slice_notna = sigma_slice[~np.isnan(obs_slice)]
        # Plot errorbars
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            # Suppose that figure has only one axes
            ax = fig.axes[0]
        fig.tight_layout()
        # ax.errorbar(x, obs_slice_notna[::-1], sigma_slice_notna[::-1], fmt='.k')
        ax.errorbar(x, obs_slice[::], sigma_slice[::], fmt='.k')
        if model_rotm_image is not None:
            model_slice = model_rotm_image.slice(pix1=slice_coords[0],
                                                 pix2=slice_coords[1])
            ax.plot(x, model_slice[::], 'b')
        ax.set_xlim([0, len(obs_slice)])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(r'Distance along slice, (pixels)')
        ax.set_ylabel(r'RM, (rad $\cdot$ m$^{-2}$)')
        if beam_width:
            # min_point = np.min(obs_slice_notna - sigma_slice_notna)
            min_point = np.nanmin(obs_slice - sigma_slice) - 25.
            ax.plot((x[1], x[1] + beam_width), (min_point, min_point), 'k',
                    lw=2)
        fig.savefig(os.path.join(outdir, outfname) + '.pdf', bbox_inches='tight',
                    dpi=1200, format='pdf')
        fig.savefig(os.path.join(outdir, outfname) + '.eps', bbox_inches='tight',
                    dpi=1200, format='eps')
        plt.close()

    elif rotm_images is not None:
        # Calculate simultaneous confidence bands
        # Bootstrap slices
        slices = list()
        for image in rotm_images.images:
            slice_ = image.slice(pix1=slice_coords[0], pix2=slice_coords[1])
            # slices.append(slice_[~np.isnan(slice_)])
            slices.append(slice_)

        # Find sigmas
        # slices_ = [arr.reshape((1, len(obs_slice_notna))) for arr in slices]
        slices_ = [arr.reshape((1, len(obs_slice))) for arr in slices]
        sigmas = hdi_of_arrays(slices_).squeeze()
        means = np.mean(np.vstack(slices), axis=0)
        # diff = obs_slice_notna - means
        diff = obs_slice - means
        # Move bootstrap curves to original simulated centers
        slices_ = [slice_ + diff for slice_ in slices]
        # Find low and upper confidence band
        # low, up = create_sim_conf_band(slices_, obs_slice_notna, sigmas,
        #                                alpha=conf_band_alpha)
        low, up = create_sim_conf_band(slices_, obs_slice, sigmas,
                                       alpha=conf_band_alpha)

        # Plot confidence band
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            # Suppose that figure has only one axes
            ax = fig.axes[0]
        fig.tight_layout()
        ax.set_xlim([0, len(obs_slice)])
        d = 0.125 * ((up - low)[~np.isnan(up - low)][0] +
                     (up - low)[~np.isnan(up - low)][-1])
        if ylim is None:
            ax.set_ylim([np.nanmin(low) - d, np.nanmax(up) + d])
        else:
            ax.set_ylim(ylim)
        ax.plot(x, low[::], 'k', lw=2)
        ax.plot(x, up[::], 'k', lw=2)
        [ax.plot(x, slice_[::], lw=0.1, color="#4682b4") for slice_ in slices_]
        # ax.plot(x, obs_slice_notna[::-1], '.k')
        if show_dots_boot:
            ax.plot(x, obs_slice[::], '.k')
        # Plot errorbar aditionally to bootstrap slices
        if external_sigma_slice is not None:
            assert len(obs_slice) == len(external_sigma_slice)
            ax.errorbar(x, obs_slice[::], external_sigma_slice[::], fmt='.k')
        if model_rotm_image is not None:
            model_slice = model_rotm_image.slice(pix1=slice_coords[0],
                                                 pix2=slice_coords[1])
            ax.plot(x, model_slice[::], 'k', ls='dashed')
        ax.set_xlabel("Position along slice, (pixels)")
        ax.set_ylabel(r'RM, (rad $\cdot$ m$^{-2}$)')
        if beam_width:
            # min_point = np.min(obs_slice_notna) -\
            #             sigmas[np.argmin(obs_slice_notna)]
            min_point = np.nanmin(low) - 0.5 * d
            ax.plot((x[1], x[1] + beam_width), (min_point, min_point), 'k',
                    lw=2)
        fig.savefig(os.path.join(outdir, outfname) + '.pdf',
                    bbox_inches='tight', dpi=1200, format='pdf')
        fig.savefig(os.path.join(outdir, outfname) + '.eps',
                    bbox_inches='tight', dpi=1200, format='eps')
        plt.close()
    else:
        raise Exception("Specify ROTM error map or bootstrapped ROTM images")
    return fig


def plot_image_correlation(image, fname='correlation.png', show=True,
                           close=False, savefig=True, outdir=None):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('Pixels')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Pixels')
    R = np.corrcoef(image)
    im = ax.matshow(R)
    colorbar_ax = fig.add_axes([0.80, 0.10, 0.05, 0.80])
    # cb = fig.colorbar(im, cax=colorbar_ax)
    cb = fig.colorbar(im, cax=colorbar_ax)
    cb.set_label('Correlation Coeff.')
    if savefig:
        if outdir is None:
            outdir = os.getcwd()
        fig.savefig(os.path.join(outdir, fname), bbox_inches='tight', dpi=200)
    if show:
        fig.show()
    if close:
        plt.close()
    return fig


def image_slice(image, pix1, pix2):
    """
    Returns slice of image along line.

    :param image:
        Numpy 2D array of image.
    :param pix1:
        Iterable of coordinates of first pixel.
    :param pix2:
        Iterable of coordinates of second pixel.
    :return:
        Numpy array of image values for given slice.
    """
    length = int(round(np.hypot(pix2[0] - pix1[0], pix2[1] - pix1[1])))
    if pix2[0] < pix1[0]:
        x = np.linspace(pix2[0], pix1[0], length)[::-1]
    else:
        x = np.linspace(pix1[0], pix2[0], length)
    if pix2[1] < pix1[1]:
        y = np.linspace(pix2[1], pix1[1], length)[::-1]
    else:
        y = np.linspace(pix1[1], pix2[1], length)

    return image[v_round(x).astype(np.int), v_round(y).astype(np.int)]


def bias(image, images):
    pass


def variance(image, images):
    pass