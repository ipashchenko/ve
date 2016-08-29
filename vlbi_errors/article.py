import numpy as np
from itertools import combinations
from sklearn import gaussian_process
from from_fits import create_image_from_fits_file
from simulations import simulate


# First find best NCLEAN using cv_cc.py


# Plot covariance matrix of the residuals (not difmap, but, probably, AIPS?)
# Plot covariogramm, GP fit?


if False:
    # Estimate correlation in image pixel values
    # FIXME: Better use residuals image from difmap or AIPS
    image_fits = '/home/ilya/code/vlbi_errors/vlbi_errors/residuals_3c273_15000.fits'
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


# Simulate gradient of RM on MOJAVE frequencies. Get "observed" data & model
# images & model data (w/o noise)
from mojave import get_epochs_for_source
path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
base_dir = '/home/ilya/vlbi_errors/article'
# sources = ['1514-241', '1302-102', '0754+100', '0055+300', '0804+499',
#            '1749+701', '0454+844']
mapsize_dict = {'x': (512, 0.1), 'y': (512, 0.1), 'j': (512, 0.1),
                'u': (512, 0.1)}
mapsize_common = (512, 0.1)
source = '0454+844'
epoch = '2006_03_09'
max_jet_flux = 0.0015
epochs = get_epochs_for_source(source, use_db='multifreq')
simulate(source, epoch, ['x', 'y', 'j', 'u'],
         n_sample=3, max_jet_flux=max_jet_flux, rotm_clim_sym=[-300, 300],
         rotm_clim_model=[-300, 300],
         path_to_script=path_to_script, mapsize_dict=mapsize_dict,
         mapsize_common=mapsize_common, base_dir=base_dir,
         rotm_value_0=0., rotm_grad_value=0., n_rms=2.,
         download_mojave=False, spix_clim_sym=[-1.5, 1],
         spix_clim_model=[-1.5, 1], qu_fraction=0.3)
