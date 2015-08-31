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
    step1 = Metropolis(vars=[alpha, beta])
    step2 = ElemwiseCategoricalStep(var=j, values=[0, 1, 2])
    tr = sample(10000, step=[step1, step2])
