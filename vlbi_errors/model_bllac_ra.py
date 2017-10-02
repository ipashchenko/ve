import numpy as np
from uv_data import UVData
from components import ModelImageComponent
from model import Model
from from_fits import create_model_from_fits_file
from utils import mas_to_rad
from stats import LnLikelihood
from spydiff import import_difmap_model
from scipy.optimize import minimize, fmin


# uv_file = '/home/ilya/github/bck/jetshow/uvf/0716+714_raks01xg_C_LL_0060s_uva.fits'
uv_file = '/home/ilya/github/bck/jetshow/uvf/2200+420_K_SVLBI.uvf'
uvdata_ext = UVData(uv_file)
uvdata_orig = UVData(uv_file)
# clean_difmap('2200+420_K_SVLBI.uvf', 'bllac_cc.fits', 'I', (8192, 0.0035),
#              path='/home/ilya/github/bck/jetshow/uvf/',
#              path_to_script='/home/ilya/github/vlbi_errors/difmap/final_clean_nw',
#              show_difmap_output=True)
comps = import_difmap_model('/home/ilya/github/bck/jetshow/uvf/ell_c_ell.mdl')
ext_model = Model(stokes='I')
ext_model.add_component(comps[-1])
# cc_fits = '/home/ilya/github/vlbi_errors/vlbi_errors/bllac_cc.fits'
# fig = uvdata_ext.uvplot()
# ccmodel = create_model_from_fits_file(cc_fits)
# ccmodel.filter_components_by_r(r_max_mas=0.15)
uvdata_ext.substitute([ext_model])
uvdata_core = uvdata_orig - uvdata_ext
# uvdata_core.save('/home/ilya/github/vlbi_errors/vlbi_errors/bllac_core.uvf')

# Set up ModelImage component
image = '/home/ilya/github/bck/jetshow/cmake-build-debug/map_i.txt'
image = np.loadtxt(image)
imsize = 1734
imsize = (imsize, imsize)
mas_in_pix = 0.00253
y, z = np.meshgrid(np.arange(imsize[0]), np.arange(imsize[1]))
y = y - imsize[0] / 2. + 0.5
z = z - imsize[0] / 2. + 0.5
y_mas = y * mas_in_pix
z_mas = z * mas_in_pix
y_rad = mas_to_rad * y_mas
z_rad = mas_to_rad * z_mas
image[image < 0] = 0
image[image > 10.0] = 0
image[image < np.percentile(image[image > 0].ravel(), 90)] = 0
icomp = ModelImageComponent(image, y_rad[0, :], z_rad[:, 0])
model = Model(stokes='I')
model.add_component(icomp)
uv = uvdata_core.uv

lnlik = LnLikelihood(uvdata_core, model, average_freq=True, amp_only=False)

import emcee


def lnprior(p):
    if not 1.0 < p[0] < 5.0:
        return -np.inf
    if not -20 < p[1] < 20:
        return -np.inf
    if not -20 < p[2] < 20:
        return -np.inf
    if not 0.1 < p[3] < 2.0:
        return -np.inf
    if not 0.0 < p[4] < 2*np.pi:
        return -np.inf
    return 0.0


def lnpost(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lnlik(p) + lp


p0 = [1.0, 0.0, 0.0, 1.0, 3.14]
from emcee.utils import sample_ball
ndim = 5
nwalkers = 24
p = sample_ball(p0, [0.2, 3, 3, 0.2, 0.5], nwalkers)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, threads=4)
pos, prob, state = sampler.run_mcmc(p, 20)
print("Reseting sampler")
sampler.reset()
pos, lnp, _ = sampler.run_mcmc(pos, 50)

        # for angle in np.linspace(0, 2*np.pi, 12):
#     print(angle, lnlik(np.array([1., 0, 0, 1., angle])))
#
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# import hyperopt
#
#
# def objective(space):
#     neglnlik = -lnlik(np.array([space['flux'], space['x'], space['y'], space['scale'], space['angle']]))
#     print("Negative lnlike: {}".format(neglnlik))
#     return {'loss': neglnlik, 'status': STATUS_OK}
#
#
# space = {'flux': hp.loguniform('flux', -0.69, 2.0),
#          'x': hp.uniform('x', -20, 20),
#          'y': hp.uniform('y', -20, 20),
#          'scale': hp.loguniform('scale', -2.3, 0.69),
#          'angle': hp.uniform('angle', 0, 2*np.pi)}
#
# trials = Trials()
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=300,
#             trials=trials)
#
# print(hyperopt.space_eval(space, best))


# p_ml = fmin(lambda p: -lnlik(p), model.p)

# # TODO: Implement analitical grad of likelihood (it's gaussian)
# fit = minimize(lambda p: -lnlik(p), model.p, method='L-BFGS-B',
#                options={'factr': 10**12, 'eps': 0.2, 'disp': True},
#                bounds=[(0.5, 2), (-20, 20), (-20, 20), (0.5, 2),
#                        (2.4, 2.9)])
# if fit['success']:
#     print("Succesful fit!")
#     p_ml = fit['x']
#     print(p_ml)
# fig.savefig('/home/ilya/github/bck/jetshow/uvf_mf_adds/ra.png',
#             bbox_inches='tight', dpi=300)