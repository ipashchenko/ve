import triangle
import numpy as np
from pymc3 import (Model, Normal, Categorical, Dirichlet, Metropolis, sample,
                   constant, ElemwiseCategoricalStep, NUTS)


# Create data
# lambda squared [m^2]
x = np.array([0.00126661, 0.00136888, 0.00359502, 0.00423771])
# PANG [rad]
y = np.array([-0.28306073, -0.21232782, -0.77439868,  0.75342187])
# Uncertainties of PANG values [rad]
sy = np.array([ 0.26500595,  0.29110131,  0.17655808,  0.44442663])

with Model() as model:
    a = constant(np.array([1., 1., 1.]))
    alpha = Normal('alpha', mu=0., sd=np.pi)
    beta = Normal('beta', mu=0., sd=500.)
    dd = Dirichlet('dd', a=a, shape=3)
    j = Categorical('j', p=dd, shape=len(y))

    # j=0 means data point should be lowered by pi
    # j=1 means data point is ok
    # j=2 means data point should be upped by pi
    mu = alpha + beta * x - np.pi * (j - 1)

    Y_obs = Normal('Y_obs', mu=mu, sd=sy, observed=y)

# Fitting model
with model:
    length = 10000
    step1 = Metropolis(vars=[alpha, beta])
    step2 = ElemwiseCategoricalStep(var=j, values=[0, 1, 2])
    tr = sample(length, step=[step1, step2])

# Find what points should be moved if any and move them
points = y.copy()
for n, point in enumerate(points):
    indxs = np.zeros(3)
    for i in range(3):
        indxs[i] = np.count_nonzero(tr.get_values('j')[:, n] == i)
    move_indx = np.argmax(indxs)
    if move_indx != 1:
        print "Moving point #{} on {} pi".format(n + 1, move_indx - 1)
    points[n] += np.pi * (move_indx - 1)

triangle.corner(np.vstack((tr.get_values('alpha'), tr.get_values('beta'))).T,
                labels=["PA at zero wavelength, [rad]", "ROTM, [rad/m/m]"])


with Model() as model:
    alpha = Normal('alpha', mu=0., sd=np.pi)
    beta = Normal('beta', mu=0., sd=500.)

    mu = alpha + beta * x

    Y_obs = Normal('Y_obs', mu=mu, sd=sy, observed=points)

# Fitting model
with model:
    length = 10000
    step = Metropolis(vars=[alpha, beta])
    tr = sample(length, step=[step])


triangle.corner(np.vstack((tr.get_values('alpha'), tr.get_values('beta'))).T,
                labels=["PA at zero wavelength, [rad]", "ROTM, [rad/m/m]"])
