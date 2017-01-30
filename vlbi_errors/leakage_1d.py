import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting


import copy
def fit_func(x, y, yerr, gg):
    gg_ = copy.deepcopy(gg)
    fitter = fitting.LevMarLSQFitter()
    gg_fit = fitter(gg_, x, y, weights=1/yerr)
    return gg_fit.mean_1.value




g1 = models.Gaussian1D(amplitude=10., mean=0., stddev=1.)
g2 = models.Gaussian1D(amplitude=4., mean=1., stddev=2.)
gg = g1 + g2
x = np.linspace(-5, 10, 50)
noise_std = 0.5
noise = np.random.normal(0, noise_std, size=len(x))
y = gg(x) + noise
yerr = noise_std * np.ones(len(x))
plt.figure()
plt.errorbar(x, y, yerr, fmt='.k', label='Obs. data')
plt.plot(x, g2(x), label='True model')
plt.legend()


fitter = fitting.LevMarLSQFitter()
gg_fit = fitter(gg, x, y)
g2_ = models.Gaussian1D(amplitude=gg_fit.amplitude_1, mean=gg_fit.mean_1,
                        stddev=gg_fit.stddev_1)
plt.plot(x, g2_(x), color='r', label='Fitted model')
plt.legend()
plt.show()



p_0 = [gg_fit.amplitude_0.value, gg_fit.mean_0.value, gg_fit.stddev_0.value,
       gg_fit.amplitude_1.value, gg_fit.mean_1.value, gg_fit.stddev_1.value]
p_0_true = [10., 0., 1., 4., 1., 2.]
ps = [p_0[:] for i in range(100)]
for i, mean in enumerate(np.linspace(0, 2, 100)):
    ps[i][4] = mean


def lnprob(p, x, y, yerr):
    amplitude_0, mean_0, stddev_0, amplitude_1, mean_1, stddev_1 = p
    g0 = models.Gaussian1D(amplitude=amplitude_0, mean=mean_0,
                           stddev=stddev_0)
    g1 = models.Gaussian1D(amplitude=amplitude_1, mean=mean_1,
                           stddev=stddev_1)
    model = g0 + g1
    return (-0.5 * np.log(2. * np.pi * yerr ** 2.) -
            (y - model(x)) ** 2. / (2. * yerr ** 2.)).sum()

plt.figure()
plt.plot([p[4] for p in ps], [lnprob(p, x, y, yerr) for p in ps])
ind = np.argmax([lnprob(p, x, y, yerr) for p in ps])
p_max = [p[4] for p in ps][ind]
plt.axvline(1, label='true value', color='k', lw=2)
plt.axvline(p_max, label='original fit', color='r', lw=2)
plt.xlabel("Location")
plt.ylabel("Log. of likelihood")
plt.legend()
plt.show()

# # Bootstrapping data (case)
# from astropy.stats import bootstrap
# boot_xys = bootstrap(np.vstack((x, y)).T, bootnum=1000)
# means = list()
# for boot_xy in boot_xys:
#     x_ = boot_xy[:, 0]
#     y_ = boot_xy[:, 1]
#     means.append(fit_func(x_, y_, yerr, gg_fit))

# Bootstrapping data (residuals)
from astropy.stats import bootstrap
boot_res = bootstrap(y-gg_fit(x), bootnum=1000)
means = list()
for boot_r in boot_res:
    gg_fit_ = copy.deepcopy(gg_fit)
    y_ = gg_fit_(x) + boot_r
    means.append(fit_func(x, y_, yerr, gg_fit_))

bc = np.mean(means) - gg_fit.mean_1.value
mean_1_bc = gg_fit.mean_1.value + bc
plt.figure()
plt.hist(means, bins=30, range=[0, 4])
plt.axvline(gg_fit.mean_1.value, label='original fit', color='r', lw=2)
plt.axvline(mean_1_bc, label='bias-corrected value', color='g', lw=2)
plt.axvline(1, label='true value', color='k', lw=2)
plt.xlabel("Location")
plt.legend()
plt.show()

# gg = fit_g(g, x, y, data)
# print gg.x_stddev, gg.y_stddev
