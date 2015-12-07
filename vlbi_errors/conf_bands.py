import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def model(x, p1, p2):
    return p1 * x + p2


n_boot = 500
alpha = 0.05
x = np.arange(20, dtype=float)
p0 = (1., -2.)
y = model(x, p0[0], p0[1])
noise = np.random.normal(0, 3., len(x))
y += noise

popt, pcov = curve_fit(model, x, y, p0=[0, 0])
print popt, pcov

ym = model(x, popt[0], popt[1])
yres = y - ym

p_boot = list()
y_boot = np.empty((len(y), n_boot))
for i in range(n_boot):
    yres_i = np.random.choice(yres, len(yres))
    y_i = ym + yres_i
    p_i, pcov_i = curve_fit(model, x, y_i, p0=[0, 0])
    ym_i = model(x, p_i[0], p_i[1])
    y_boot[:, i] = ym_i
    p_boot.append(p_i)
    print i, p_i

p_boot = np.vstack(p_boot)

# 1. Choose upper and lower percentile and find such that only \alpha of points
# lie outside
counts = dict()
N = len(x) * n_boot
for i in range(1, n_boot/2 + 1):
    print i
    low_perc = (n_boot/2. - i) / n_boot
    high_perc = (n_boot/2. + i) / n_boot
    print "low perc {}".format(low_perc)
    print "high perc {}".format(high_perc)
    nn = 0
    for j in range(len(x)):
        data = y_boot[j, :]
        l, h = np.percentile(data, (100 * low_perc, 100 * high_perc))
        n = np.count_nonzero(np.logical_and(data > l, data < h))
        nn += n
    print "updating counts with {}, {}".format(i, float(nn) / N)
    counts.update({i: float(nn) / N})

# 1. Choose upper and lower percentile and find such that only \alpha of curves
# lie outside
sim_counts = dict()
sim_N = n_boot
for i in range(1, n_boot/2 + 1):
    print i
    low_perc = (n_boot/2. - i) / n_boot
    high_perc = (n_boot/2. + i) / n_boot
    print "low perc {}".format(low_perc)
    print "high perc {}".format(high_perc)
    nn = 0
    for j in range(n_boot):
        data = y_boot[:, j]
        l, h = np.percentile(data, (100 * low_perc, 100 * high_perc))
        n = np.count_nonzero(np.logical_and(data > l, data < h))
        nn += n
    print "updating counts with {}, {}".format(i, float(nn) / N)
    counts.update({i: float(nn) / N})


# Find percentiles that embrace 95% of data
for key, value in counts.items():
    if value > 1 - alpha:
        res = key
        break
low_perc = 100 * (n_boot/2. - res) / n_boot
high_perc = 100 * (n_boot/2. + res) / n_boot

# Plot 95% CB
# Plot pointwise
pointwise = list()
simultenious = list()
for i in range(len(x)):
    data = y_boot[i, :]
    pointwise.append(np.percentile(data, ((100 * alpha / 2),
                                   100 * (1 - alpha / 2))))
    simultenious.append(np.percentile(data, (low_perc, high_perc)))
pointwise = np.array(pointwise)
simultenious = np.array(simultenious)


fig = plt.figure()
plt.plot(x, y, '.k', lw=5)
plt.plot(x, pointwise[:, 0], 'b', lw=2)
plt.plot(x, pointwise[:, 1], 'b', lw=2)
plt.plot(x, simultenious[:, 0], 'r', lw=2)
plt.plot(x, simultenious[:, 1], 'r', lw=2)
for i in range(n_boot):
    plt.scatter(x + np.random.normal(0, 0.03), y_boot[:, i], s=0.2)
fig.show()




