import os
import sys
import copy
import shutil
import numpy as np
# agn_abc stuff
sys.path.insert(0, '/home/ilya/github/agn_abc')
from data import Data
from astropy import units as u
from astropy.cosmology import WMAP9
from jetmodel_analytic import JetModelAnalytic2M
from fourier import NFFT
import matplotlib.pyplot as plt


def create_data(jm_dict, noise_scale=1.0, qu_fraction=0.2):
    # Substitute original uv-data with model uv-data.
    for band, data in zip(bands, (data_x, data_y, data_j, data_u)):
        print("Band - {}".format(band))
        data.substitute_with_models([jm_dict[band]["I"],
                                     jm_dict[band]["Q"],
                                     jm_dict[band]["U"]])
        data.add_original_noise(scale=noise_scale)
        outname = "{}.uvf".format(band)
        data.save(os.path.join(data_dir, outname))


if __name__ == "__main__":

    n_sample = 3
    qu_fraction = 0.2
    bands = ("x", "y", "j", "u")
    freqs = np.array([8.104458750, 8.424458750, 12.111458750, 15.353458750])
    freqs *= 10**9
    path_to_script = "/home/ilya/github/ve/difmap/final_clean_nw"
    # Directory with resulting data sets
    data_dir = "/home/ilya/data/revision"
    # Directory with multifrequency data
    templates_dir = "/home/ilya/github/jetshow_utils/uvf_templates"

    observed_fits_x = os.path.join(templates_dir, "1458+718.x.2006_09_06.uvf")
    observed_fits_y = os.path.join(templates_dir, "1458+718.y.2006_09_06.uvf")
    observed_fits_j = os.path.join(templates_dir, "1458+718.j.2006_09_06.uvf")
    observed_fits_u = os.path.join(templates_dir, "1458+718.u.2006_09_06.uvf")

    data_x = Data(observed_fits_x)
    data_y = Data(observed_fits_y)
    data_j = Data(observed_fits_j)
    data_u = Data(observed_fits_u)

    jm_x = JetModelAnalytic2M(8.104458750*u.GHz, 0.1, 0.01*u.mas, [2048, 1024],
                              stokes='I', ft_class=NFFT)
    jm_y = JetModelAnalytic2M(8.424458750*u.GHz, 0.1, 0.01*u.mas*8.10/8.42,
                              [2048, 1024], stokes='I', ft_class=NFFT)
    jm_j = JetModelAnalytic2M(12.111458750*u.GHz, 0.1, 0.01*u.mas*8.10/12.11,
                              [2048, 1024], stokes='I', ft_class=NFFT)
    jm_u = JetModelAnalytic2M(15.353458750*u.GHz, 0.1, 0.01*u.mas*8.10/15.35,
                              [2048, 1024], stokes='I', ft_class=NFFT)

    # # Jet radius (mas) vs distance along jet (mas)
    # pix_to_pc = (0.1*u.mas*WMAP9.kpc_comoving_per_arcmin(0.1)).to(u.pc).value
    # ang_to_lin = WMAP9.kpc_comoving_per_arcmin(0.1).to(u.pc/u.mas)/(1+0.1)
    # dist = np.linspace(0, 15)*u.mas
    # rad = 2*np.tan(10*np.pi/180)*dist
    # beam = 1.42*u.mas
    # plt.plot(dist, rad/beam)
    # plt.xlabel("Distance along jet, [mas]")
    # plt.ylabel("Jet transverse size, [beam]")
    # plt.show()

    jm_dict = dict()
    # for band in bands:
    #     jm_dict.update({band: {}})

    for jm, band in zip((jm_x, jm_y, jm_j, jm_u), bands):
        # dx, dy, rot, phi_app, logA1, logA2, logm
        jm.set_params_vec([0, 0, 0, 10*u.deg.to(u.rad), 1.5, 2, np.log(1)])
        image = jm.image()
        print("Flux = {}".format(np.sum(image)))
        jm_Q = copy.copy(jm)
        jm_U = copy.copy(jm)
        jm_Q.stokes = "Q"
        jm_U.stokes = "U"
        jm_Q.fraction = qu_fraction
        jm_U.fraction = qu_fraction
        jm_dict[band] = {"I": jm, "Q": jm_Q, "U": jm_U}

    for i in range(n_sample):
        print("=== Creating artificial source #{} ===".format(i+1))
        create_data(jm_dict)
        for band in bands:
            shutil.move(os.path.join(data_dir, "{}.uvf".format(band)),
                        os.path.join(data_dir, "{}_{}.uvf".format(band, i)))