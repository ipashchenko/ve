import os
import numpy as np
import scipy as sp
from uv_data import UVData
from spydiff import import_difmap_model
from components import EGComponent, CGComponent, DeltaComponent
from model import Model
from stats import LnLikelihood
import nestle


data_dir = '/home/ilya/code/vlbi_errors/bin_c1/'
uv_fits = os.path.join(data_dir, '0235+164.c1.2008_09_02.uvf_difmap')
mdl_file = '0235+164.c1.2008_09_02.mdl'
stokes = 'I'

uv_data = UVData(uv_fits)
mdl_dir, mdl_fname = os.path.split(mdl_file)
comps = import_difmap_model(mdl_file, data_dir)
# Sort components by distance from phase center
comps = sorted(comps, key=lambda x: np.sqrt(x.p[1]**2 + x.p[2]**2))

# Prior = cube0 * x + cube1, where x from [0, 1]
cube0 = list()
cube1 = list()
for comp in comps:
    print comp
    if isinstance(comp, EGComponent):
        flux_high = 2 * comp.p[0]
        bmaj_high = 4 * comp.p[3]
        if comp.size == 6:
            comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),),
                           x=(sp.stats.uniform.logpdf, [-10., 10.], dict(),),
                           y=(sp.stats.uniform.logpdf, [-10., 10.], dict(),),
                           bmaj=(sp.stats.uniform.logpdf, [0, bmaj_high], dict(),),
                           e=(sp.stats.uniform.logpdf, [0, 1.], dict(),),
                           bpa=(sp.stats.uniform.logpdf, [0, np.pi], dict(),))
            cube0.extend([flux_high, 20., 20., bmaj_high, 1., np.pi])
            cube1.extend([0., -10., -10., 0., 0., 0.])
        elif comp.size == 4:
            flux_high = 2 * comp.p[0]
            bmaj_high = 4 * comp.p[3]
            comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),),
                           x=(sp.stats.uniform.logpdf, [-10., 10.], dict(),),
                           y=(sp.stats.uniform.logpdf, [-10., 10.], dict(),),
                           bmaj=(sp.stats.uniform.logpdf, [0, bmaj_high], dict(),))
            cube0.extend([flux_high, 20., 20., bmaj_high])
            cube1.extend([0., -10., -10., 0.])
        else:
            raise Exception("Gauss component should have size 4 or 6!")
    elif isinstance(comp, DeltaComponent):
        flux_high = 5 * comp.p[0]
        comp.add_prior(flux=(sp.stats.uniform.logpdf, [0., flux_high], dict(),),
                       x=(sp.stats.uniform.logpdf, [-10., 10.], dict(),),
                       y=(sp.stats.uniform.logpdf, [-10., 10.], dict(),))
        cube0.extend([flux_high, 20., 20.])
        cube1.extend([0., -10., -10.])
    else:
        raise Exception("Unknown type of component!")

cube0 = np.array(cube0)
cube1 = np.array(cube1)


def hypercube(u):
    return cube0 * u + cube1


# Create model
mdl = Model(stokes=stokes)
# Add components to model
mdl.add_components(*comps)
loglike = LnLikelihood(uv_data, mdl)
result = nestle.sample(loglikelihood=loglike, prior_transform=hypercube,
                       ndim=mdl.size, npoints=50, method='classic',
                       callback=nestle.print_progress)
