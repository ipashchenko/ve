from __future__ import unicode_literals, print_function
import pymultinest
import math
import os
import threading, subprocess
from sys import platform
if not os.path.exists("chains"): os.mkdir("chains")
import os
import numpy as np
import scipy as sp
from uv_data import UVData
from spydiff import import_difmap_model
from components import EGComponent, CGComponent, DeltaComponent
from model import Model
from stats import LnLikelihood


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
    print(comp)
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


def hypercube(cube, ndim, nparams):
    for i in range(ndim):
        cube[i] = cube0[i] * cube[i] + cube1[i]


# Create model
mdl = Model(stokes=stokes)
# Add components to model
mdl.add_components(*comps)
loglike = LnLikelihood(uv_data, mdl)


def show(filepath):
    if os.name == 'mac' or platform == 'darwin':
        subprocess.call(('open', filepath))
    elif os.name == 'nt' or platform == 'win32':
        os.startfile(filepath)
    elif platform.startswith('linux'):
        subprocess.call(('xdg-open', filepath))


def myloglike(cube, ndim, nparams):
    return loglike(cube)


# number of dimensions our problem has
parameters = ["flux1", "x1", "y1", "bmaj1", "e1", "bpa1", "flux2", "x2", "y2",
              "bmaj2"]
n_params = len(parameters)

# we want to see some output while it is running
progress = pymultinest.ProgressPlotter(n_params=n_params,
                                       outputfiles_basename='chains/2-')
progress.start()
threading.Timer(2, show, ["chains/2-phys_live.points.pdf"]).start()
# run MultiNest
pymultinest.run(myloglike, hypercube, n_params,
                importance_nested_sampling=False, resume=False,
                verbose=True, sampling_efficiency='model',
                outputfiles_basename='chains/2-')
# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params=n_params, outputfiles_basename='chains/2-')
s = a.get_stats()

import json
# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
    json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
    json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
    plt.subplot(n_params, n_params, n_params * i + i + 1)
    p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])

    for j in range(i):
        plt.subplot(n_params, n_params, n_params * j + i + 1)
        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
        plt.xlabel(parameters[i])
        plt.ylabel(parameters[j])

plt.savefig("chains/marginals_multinest.pdf") #, bbox_inches='tight')
show("chains/marginals_multinest.pdf")

for i in range(n_params):
    outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()

    outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    plt.ylabel("Cumulative probability")
    plt.xlabel(parameters[i])
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()

print("Take a look at the pdf files in chains/")