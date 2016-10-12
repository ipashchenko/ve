# Test LS_estimates
import os
import sys
import numpy as np
from components import CGComponent, EGComponent
from uv_data import UVData
from model import Model
from spydiff import clean_difmap
from from_fits import create_clean_image_from_fits_file
try:
    from scipy.optimize import minimize, fmin
except ImportError:
    sys.exit("install scipy for ml estimation")
import scipy as sp
from stats import LnLikelihood, LnPost
from image import find_bbox
from image import plot as iplot
from image_ops import rms_image
import emcee
import corner


data_dir = '/home/ilya/Dropbox/Ilya/0235_bk150_uvf_data/'
uv_fname = '1803+784.u.2012_11_28.uvf'

uvdata = UVData(os.path.join(data_dir, uv_fname))
noise = uvdata.noise()
cg1 = EGComponent(2., 0., 0., 0.15, 0.33, 0.3)
cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 5.], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),),
              e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
              bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
cg2 = CGComponent(0.5, 0., -2., 0.55)
cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 5.], dict(),),
              bmaj=(sp.stats.uniform.logpdf, [0, 20.], dict(),))
model = Model(stokes='I')
model.add_components(cg1, cg2)
uvdata.substitute([model])
uvdata.noise_add(noise)
uvdata.save(os.path.join(data_dir, 'fake.uvf'))
# Clean uv-data
clean_difmap('fake.uvf', 'fake_cc.fits', 'I', (1024, 0.1), path=data_dir,
             path_to_script='/home/ilya/code/vlbi_errors/difmap/final_clean_nw',
             outpath=data_dir, show_difmap_output=True)
image = create_clean_image_from_fits_file(os.path.join(data_dir, 'fake_cc.fits'))
rms = rms_image(image)
blc, trc = find_bbox(image.image, 2.*rms, delta=int(image._beam.beam[0]))
# Plot image
iplot(image.image, x=image.x, y=image.y, min_abs_level=3. * rms,
      outfile='fake_image', outdir=data_dir, blc=blc, trc=trc, beam=image.beam,
      show_beam=True)

# Create posterior for data & model
lnpost = LnPost(uvdata, model)
ndim = model.size
nwalkers = 50
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
p_std1 = [0.1, 0.1, 0.1, 0.1]
p_std2 = [0.1, 0.1, 0.1, 0.1]
p0 = emcee.utils.sample_ball(model.p, p_std1 + p_std2, size=nwalkers)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, 300)

# Overplot data and model
p_map = list()
for i in range(ndim):
    counts, bin_values = np.histogram(sampler.flatchain[::10, i], bins=30)
    p_map.append(bin_values[counts == counts.max()][0])
mdl = Model(stokes='I')
cg1 = CGComponent(*(p_map[:4]))
cg2 = CGComponent(*(p_map[4:]))
mdl.add_components(cg1, cg2)
uvdata.uvplot(stokes='I')
mdl.uvplot(uv=uvdata.uv)
fig = corner.corner(sampler.flatchain[::10, :],
                    labels=["$flux$", "$y$", "$x$", "$maj$", "$flux$", "$y$",
                            "$x$", "$maj$"])
