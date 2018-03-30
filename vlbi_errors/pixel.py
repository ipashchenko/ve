try:
    import corner as triangle
except ImportError:
    triangle = None
import numpy as np
from scipy.optimize import leastsq
from itertools import combinations
#from pymc3 import (Model, Normal, Categorical, Dirichlet, Metropolis,
#                   HalfCauchy, sample, constant, ElemwiseCategoricalStep, NUTS)
import matplotlib.pyplot as plt


def rotm_leastsq(lambda_sq, chi, s_chi=None, p0=None):
    """
    Fit ROTM using least squares.

    :param lambda_sq:
        Iterable of wavelengths squared [m**2]
    :param chi:
        Iterable of polarization positional angles [rad].
    :param s_chi: (optional)
        Iterable of uncertainties of polarization positional angles [rad].
        If ``None`` then model uncertainties. (default: ``None``)
    :param p0: (optional)
        Starting value for minimization (RM [rad/m**2], PA_zero_lambda [rad]).
        If ``None`` then use ``[0, 0]``. (default: ``None``)
    :return:
    """
    if p0 is None:
        p0 = [0., 0.]

    def rotm_model(p, lambda_sq):
        return p[0] * lambda_sq + p[1]

    def weighted_residuals(p, lambda_sq, chi, s_chi):
        return (chi - rotm_model(p, lambda_sq)) / s_chi

    def residuals(p, lambda_sq, chi):
        return chi - rotm_model(p, lambda_sq)

    if s_chi is None:
        func, args = residuals, (lambda_sq, chi,)
    else:
        func, args = weighted_residuals, (lambda_sq, chi, s_chi,)

    fit = leastsq(func, p0, args=args, full_output=True)
    (p, pcov, infodict, errmsg, ier) = fit

    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)

    if (len(chi) > len(p0)) and pcov is not None:
        # Residual variance
        s_sq = (func(p, *args) ** 2.).sum() / (len(chi) - len(p0))
        pcov *= s_sq
    else:
        pcov = np.nan
        s_sq = np.nan

    return p, pcov, s_sq * (len(chi) - len(p0))


def resolver_chisq(lambda_sq, chi, s_chi=None, p0=None):
    """
    Function that
    :param lambda_sq:
        Iterable of wavelengths squared [m**2]
    :param chi:
        Iterable of polarization positional angles [rad].
    :param s_chi: (optional)
        Iterable of uncertainties of polarization positional angles [rad].
        If ``None`` then model uncertainties. (default: ``None``)
    :param plot_fit: (optional)
        Plot fit values using resolved ambiguity? (default: ``False``)
    :param p0: (optional)
        Starting value for minimization (RM [rad/m**2], PA_zero_lambda [rad]).
        If ``None`` then use ``[0, 0]``. (default: ``None``)

    :return:
        Numpy array of polarization positional angles with +/-n*pi-ambiguity
        resolved.
    """
    n_data = len(lambda_sq)
    chi_sq = dict()
    # First check cases when only one frequency is affected
    for i in range(n_data):
        chi_ = list(chi)[:]
        chi_[i] = chi[i] + np.pi
        p, pcov, s_sq = rotm_leastsq(lambda_sq, chi_, s_chi=s_chi, p0=p0)
        chi_sq.update({"+{}".format(i): s_sq})
        chi_[i] = chi[i] - np.pi
        p, pcov, s_sq = rotm_leastsq(lambda_sq, chi_, s_chi=s_chi, p0=p0)
        chi_sq.update({"-{}".format(i): s_sq})

    # Now check cases when two frequencies are affected
    for comb in combinations(range(n_data), 2):
        chi_ = list(chi)[:]
        # Both frequencies + pi
        comb1 = "+{}+{}".format(comb[0], comb[1])
        chi_[comb[0]] = chi[comb[0]] + np.pi
        chi_[comb[1]] = chi[comb[1]] + np.pi
        p, pcov, s_sq = rotm_leastsq(lambda_sq, chi_, s_chi=s_chi, p0=p0)
        chi_sq.update({comb1: s_sq})

        # Both frequencies - pi
        comb2 = "-{}-{}".format(comb[0], comb[1])
        chi_[comb[0]] = chi[comb[0]] - np.pi
        chi_[comb[1]] = chi[comb[1]] - np.pi
        p, pcov, s_sq = rotm_leastsq(lambda_sq, chi_, s_chi=s_chi, p0=p0)
        chi_sq.update({comb2: s_sq})

        # + pi - pi
        comb3 = "+{}-{}".format(comb[0], comb[1])
        chi_[comb[0]] = chi[comb[0]] + np.pi
        chi_[comb[1]] = chi[comb[1]] - np.pi
        p, pcov, s_sq = rotm_leastsq(lambda_sq, chi_, s_chi=s_chi, p0=p0)
        chi_sq.update({comb3: s_sq})

        # - pi + pi
        comb4 = "-{}+{}".format(comb[0], comb[1])
        chi_[comb[0]] = chi[comb[0]] - np.pi
        chi_[comb[1]] = chi[comb[1]] + np.pi
        p, pcov, s_sq = rotm_leastsq(lambda_sq, chi_, s_chi=s_chi, p0=p0)
        chi_sq.update({comb4: s_sq})

    # Finally, original fit
    p, pcov, s_sq = rotm_leastsq(lambda_sq, chi, s_chi=s_chi, p0=p0)
    chi_sq.update({'0': s_sq})

    chi_ = list(chi)[:]
    best = min(chi_sq.keys(), key=lambda k: chi_sq[k])
    if len(best) == 1:
        # print "No correction"
        result = chi_
    elif len(best) == 2:
        # print "Corecting point #{} on {} pi".format(best[1], best[0])
        if best[0] == '+':
            chi_[int(best[1])] += np.pi
        elif best[0] == '-':
            chi_[int(best[1])] -= np.pi
        else:
            raise Exception()
    elif len(best) == 4:
        # print "Corecting point #{} on {} pi".format(best[1], best[0])
        # print "Corecting point #{} on {} pi".format(best[3], best[2])
        if best[0] == '+':
            chi_[int(best[1])] += np.pi
        elif best[0] == '-':
            chi_[int(best[1])] -= np.pi
        else:
            raise Exception()
        if best[2] == '+':
            chi_[int(best[3])] += np.pi
        elif best[2] == '-':
            chi_[int(best[3])] -= np.pi
        else:
            raise Exception()
    else:
        raise Exception()

    return chi_


def resolver_bayesian(lamba_sq, chi, s_chi=None, nsamples=10000, plot_fit=False):
    """
    Function that
    :param lambda_sq:
        Iterable of wavelengths squared [m**2]
    :param chi:
        Iterable of polarization positional angles [rad].
    :param s_chi: (optional)
        Iterable of uncertainties of polarization positional angles [rad].
        If ``None`` then model uncertainties. (default: ``None``)
    :param nsamples: (optional)
        Number of samples to sample. (default: ``10000``)
    :param plot_fit: (optional)
        Plot fit values using resolved ambiguity? (default: ``False``)

    :return:
        Numpy array of polarization positional angles with +/-n*pi-ambiguity
        resolved.
    """
    if s_chi is not None:
        with Model() as model:
            a = constant(np.array([1., 1., 1.]))
            alpha = Normal('alpha', mu=0., sd=np.pi)
            beta = Normal('beta', mu=0., sd=500.)
            dd = Dirichlet('dd', a=a, shape=3)
            j = Categorical('j', p=dd, shape=len(chi))

            # j=0 means data point should be lowered by pi
            # j=1 means data point is ok
            # j=2 means data point should be upped by pi
            mu = alpha + beta * lambda_sq - np.pi * (j - 1)

            Y_obs = Normal('Y_obs', mu=mu, sd=s_chi, observed=chi)

        with model:
            length = nsamples
            step1 = Metropolis(vars=[alpha, beta])
            step2 = ElemwiseCategoricalStep(var=j, values=[0, 1, 2])
            tr = sample(length, step=[step1, step2])
    else:
        with Model() as model:
            a = constant(np.array([1., 1., 1.]))
            alpha = Normal('alpha', mu=0., sd=np.pi)
            beta = Normal('beta', mu=0., sd=500.)
            std = HalfCauchy('std', beta=0.25, testval=0.1)
            dd = Dirichlet('dd', a=a, shape=3)
            j = Categorical('j', p=dd, shape=len(chi))

            # j=0 means data point should be lowered by pi
            # j=1 means data point is ok
            # j=2 means data point should be upped by pi
            mu = alpha + beta * lambda_sq - np.pi * (j - 1)

            Y_obs = Normal('Y_obs', mu=mu, sd=std, observed=chi)

        with model:
            length = nsamples
            step1 = Metropolis(vars=[alpha, beta, std])
            step2 = ElemwiseCategoricalStep(var=j, values=[0, 1, 2])
            tr = sample(length, step=[step1, step2])
        plt.hist(tr.get_values("j")[nsamples/5:, 3], normed=True)
        plt.show()

    # Find what points should be moved if any and move them
    points = chi[:]
    for n, point in enumerate(points):
        indxs = np.zeros(3)
        for i in range(3):
            indxs[i] = np.count_nonzero(tr.get_values('j')[:, n] == i)
        move_indx = np.argmax(indxs)
        if move_indx != 1:
            print("Moving point #{} on {} pi".format(n + 1, move_indx - 1))
        points[n] += np.pi * (move_indx - 1)

    if plot_fit:
        if s_chi is not None:
            with Model() as model:
                alpha = Normal('alpha', mu=0., sd=np.pi)
                beta = Normal('beta', mu=0., sd=500.)

                mu = alpha + beta * lambda_sq

                Y_obs = Normal('Y_obs', mu=mu, sd=s_chi, observed=points)

            with model:
                length = nsamples
                step = Metropolis(vars=[alpha, beta])
                tr = sample(length, step=[step])

            # Plot corner-plot of samples
            ndim = 2
            fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
            fig.set_size_inches(25.5, 25.5)
            # plt.rcParams.update({'axes.titlesize': 'small'})
            triangle.corner(np.vstack((tr.get_values('alpha')[nsamples/5:],
                                       tr.get_values('beta')[nsamples/5:])).T,
                            labels=["PA at zero wavelength, [rad]",
                                    "ROTM, [rad/m/m]"], fig=fig)
            fig.show()
            # fig.savefig('corner_plot.png', bbox_inches='tight', dpi=300)

        else:
            with Model() as model:
                alpha = Normal('alpha', mu=0., sd=np.pi)
                beta = Normal('beta', mu=0., sd=500.)
                std = HalfCauchy('std', beta=0.25, testval=0.1)

                mu = alpha + beta * lambda_sq

                Y_obs = Normal('Y_obs', mu=mu, sd=std, observed=points)

            with model:
                length = nsamples
                step = Metropolis(vars=[alpha, beta, std])
                tr = sample(length, step=[step])

            # Plot corner-plot of samples
            ndim = 3
            fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
            fig.set_size_inches(25.5, 25.5)
            # plt.rcParams.update({'axes.titlesize': 'small'})
            triangle.corner(np.vstack((tr.get_values('alpha')[nsamples/5:],
                                       tr.get_values('beta')[nsamples/5:],
                                       tr.get_values('std')[nsamples/5:])).T,
                            labels=["PA at zero wavelength, [rad]",
                                    "ROTM, [rad/m/m]",
                                    "STD ROTM, [rad/m/m]"], fig=fig)
            fig.show()
            # fig.savefig('corner_plot.png', bbox_inches='tight', dpi=300)

    return points

if __name__ == '__main__':

    # Create data
    # lambda squared [m^2]
    lambda_sq = np.array([0.00126661, 0.00136888, 0.00359502, 0.00423771])
    # PANG [rad]
    chi = np.array([-0.28306073, -0.21232782, -0.77439868,  0.75342187])
    # Uncertainties of PANG values [rad]
    s_chi = np.array([ 0.26500595,  0.29110131,  0.17655808,  0.44442663])

    # resolved_chi = resolver_bayesian(lambda_sq, chi, s_chi=None, nsamples=10000,
    #                         plot_fit=True)
    resolved_chi = resolver_chisq(lambda_sq, chi, s_chi=s_chi)

