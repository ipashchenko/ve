import emcee
import triangle
import scipy as sp
import numpy as np
from from_fits import create_uvdata_from_fits_file
from components import CGComponent
from model import Model
from stats import LnPost


if __name__ == '__main__':
    uvdata = create_uvdata_from_fits_file('J0005+3820_X_1998_06_24_fey_vis.fits')
    # Estimate noise
    noise = uvdata.noise(average_freq=True, use_V=False)
    # Create some model of couple components 10 mas away
    cg1 = CGComponent(1.0, 0.0, 0.0, 1.5)
    cg2 = CGComponent(0.3, 3.0, 10.0, 3.)
    # Create model
    mdl = Model(stokes='RR')
    # Add components to model
    mdl.add_components(cg1, cg2)
    uvdata.substitute([mdl])
    uvdata.noise_add(noise)
    # radplot
    uvdata.uvplot(stokes='RR')
    # plot image
    mdl.make_image('J0005+3820_X_1998_06_24_fey_map.fits')
