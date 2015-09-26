import triangle
import numpy as np
from pymc3 import (Model, Normal, Categorical, Dirichlet, Metropolis,
                   HalfCauchy, sample, constant, ElemwiseCategoricalStep, NUTS)
import matplotlib.pyplot as plt


def resolver(lamba_sq, chi, s_chi=None, nsamples=10000, plot_fit=False):
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
            print "Moving point #{} on {} pi".format(n + 1, move_indx - 1)
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

    resolved_chi = resolver(lambda_sq, chi, s_chi=None, nsamples=10000,
                            plot_fit=True)

