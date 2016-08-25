import numpy as np
from itertools import combinations
from sklearn import gaussian_process
from from_fits import create_image_from_fits_file

# Estimate correlation in image pixel values
# FIXME: Better use residuals image from difmap or AIPS
image_fits = '/home/ilya/code/vlbi_errors/vlbi_errors/residuals_15000.FITS'
image = create_image_from_fits_file(image_fits)

slices = [slice(50 * i, 50 * (i+1)) for i in range(20)]
sigma2_list = list()
for slice1, slice2 in list(combinations(slices, 2))[:51]:
    print "slices {} {}".format(slice1, slice2)
    data = image.image[slice1, slice2]

    X = list()
    y = list()
    for (i, j), val in np.ndenumerate(data):
        X.append([i, j])
        y.append(val)
    Y = np.array(y).reshape(2500, 1)

    gp = gaussian_process.GaussianProcess(thetaL=(0.01, 0.01),
                                          thetaU=(100., 100.),
                                          theta0=(1., 1.), nugget=0.0003**2,
                                          storage_mode='full')
    gpf = gp.fit(X, Y)
    Y_pred = gpf.predict(X)
    y_pred = Y_pred.reshape((50, 50))

    fwhm = 2.355 * gpf.theta_
    print "FWHM {}".format(fwhm)
    # GP variance
    sigma2 = gpf.sigma2
    print "GP std {}".format(np.sqrt(sigma2))

    sigma2_list.append((slice1, slice2, gpf.theta_))


