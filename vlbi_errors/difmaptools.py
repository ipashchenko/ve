#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:51:58 2016

@author: osh
"""
import os
import sys
HOME = os.path.expanduser("~")
DROPBOX = HOME + "/Dropbox"
if DROPBOX + "/pro/py" not in sys.path:
    sys.path.append(DROPBOX + "/pro/py")
if __name__ == "__main__":
    import loggingtools as lt
    logger = lt.start_logging(log_level='debug', logfile=None)
else:
    import logging
    logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse


def rmodel(infile, xy='polar', sort=None):
    """
    read DIFMAP model (input a file)
    returning all components
    kwarg xy = 'polar' polar coordinates returned:
    [Flux, R, Theta, rmaj, rmin, phi]
    else [Flux, x, y, rmaj, rmin, phi]
    kwarg sort='dist' => sorted by distance from (0,0)
    kwarg sort='flux' => sorted by flux (max --> min)
    """
    fid = open(infile)
    lines = fid.readlines()
    fid.close()
    res = np.array([l.replace('v','').split() for l in lines
                    if not l.startswith('!')], dtype=float)
    if sort == 'flux':
        res = res[np.argsort(res[:,0]), :][::-1]
    elif sort == 'dist':
        res = res[np.argsort(res[:,1]), :]
    else:
        pass
    if xy != 'polar':
        x = res[:,1]*np.cos(res[:,2]*np.pi/180.0 + np.pi/2)
        y = res[:,1]*np.sin(res[:,2]*np.pi/180.0 + np.pi/2)
        res[:,1] = x
        res[:,2] = y

    return res


def wmodel(comps, fname):
    """ write model components to fname """
    with open(fname, 'w') as out:
        for comp in comps:
            if len(comp) == 6:
                comp = np.hstack((comp, [1, -1.0, -1.0])) # add fake data
            for p in comp:
                out.write('{} '.format(p))
            out.write('\n')


def find_cores_offset(mod1, mod2):
    """
    Finding [x,y]_core2 - [x,y]_core1. Assuming cores are [0,0] components.
    Beware of y-sign when perform correlation
    coreshift = imshift - cores_offset
    """
    x1, y1 = rmodel(mod1)[0,1:3]
    x2, y2 = rmodel(mod2)[0,1:3]
#    print x1, y1
#    print x2, y2
    return np.array([x2-x1, y2-y1])


def plot_model(mfile, ax=None, center_core=True):
    """
    Plot the model.
    The core is assumed to be the first component
    """

    mxy = rmodel(mfile, xy='xy')
    mpol = rmodel(mfile, xy='polar')
    x0, y0 = mxy[0][1], mxy[0][2]
    fluxes = np.array([_[0] for _ in mpol])
    rads = np.array([_[1] for _ in mpol])
    bmajes = np.array([_[3] for _ in mpol])
    fmax = fluxes.max()
    rmax = rads.max() + bmajes.max() + 0.1
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(rmax, -rmax)
        ax.set_ylim(-rmax, rmax)
    ax.axvline(0, ls=':', lw=0.1)
    ax.axhline(0, ls=':', lw=0.1)

    for a, b in zip(mxy, mpol):
#        print a[1], a[2], np.hypot(a[1], a[2]), b[1]
        flux = a[0]
        if center_core:
            x, y, = a[1] - x0, a[2] - y0
        else:
            x, y = a[1], a[2]

        x = -x # R.A. axis
        alpha = (flux / fmax + 0.3) / 2.0
        if a[3] < 1e-5:
            ax.plot(x, y, '+k', alpha=alpha)
        elif a[4] == 1:
            art = Circle((x, y), a[3]/2., fill=True, alpha=alpha, ec='k', fc='k')
            ax.add_artist(art)
        elif a[4] < 1:
            if a[4] == 0:
                a[4] = 0.01
            angle=b[5] + 90
            art = Ellipse(xy=(x, y),
                          width=a[3], height=a[3]*a[4], angle=angle,
                          alpha=alpha, fc='k', ec='k')
            ax.add_artist(art)
#        ax.annotate(s = str(ind), xy=(data[ind,1] + ind1*3 + data[ind,3], \
#                    data[ind,2]), color = col, fontsize=10)
#        colors.append(col)
#        plt.pause(0.1)
#    plt.legend(tuple(bands))

# TODO:
#def fomalont():


def print_model(mfile, latex=True):
    """ print model parameters """
    m = rmodel(mfile, xy='polar')
    if latex:
        s1 = '$'
        s2 = '&'
        s3 = '\\\\'
    else:
        s1 = ''
        s2 = ''
        s3 = ''
    print("flux   rad   theta  bmaj    axr   phi")
    for a in m:
        flux, rad, theta, bmaj, axr, phi = a
        if bmaj < 0.001:
            print("{a}{:.3f}{a} {b} {a}{:.3f}{a} {b} {a}{:4.0f}{a} {b} {a}<0.001{a}{b}{b} {c}".\
                  format(flux, rad, theta, a=s1, b=s2, c=s3))
            continue
        if axr == 1:
            print("{a}{:.3f}{a} {b} {a}{:.3f}{a} {b} {a}{:4.0f}{a} {b} {a}{:.3f}{a}{b}{b} {c}".\
                  format(flux, rad, theta, bmaj, a=s1, b=s2, c=s3))
            continue
        print("{a}{:.3f}{a} {b} {a}{:.3f}{a} {b} {a}{:4.0f}{a} {b} {a}{:6.3f}{a} {b} {a}{:.2f}{a} {b} {a}{:.1f}{a} {c}".\
                  format(flux, rad, theta, bmaj, axr, phi, a=s1, b=s2, c=s3))


def comp_analysis(models, comps=[]):
    """
    Core and component parmeters
    """
    if not comps or not any(np.asarray(comps)+1):
        print ("No components specified. Returning 0-th parameters...")
        comps = np.zeros(len(models))
    spec0 = [] # spectrum
    spec2 = []
# TODO: calculate size properly
    size0 = [] # transverse size
    size2 = []
    cs = [] # shifts

    for ind in range(len(models)):
        if comps[ind] == -1:
            continue
        d = rmodel(models[ind]) # current model data
        d0 = d[0,:]
        spec0.append(d0[0])
        size0.append(d0[4])
#        print comp[ind]
        d2 = d[comps[ind],:]
        spec2.append(d2[0])
        size2.append(d2[4])
        cs.append((d2[1] - d0[1])**2 + (d2[2] - d0[2])**2)
    res = np.vstack((spec0,size0,spec2,size2,cs))
    return res


def core_first(mfile, outfile):
    """
    if core is not at first line, rewrite model to outfile
    Maybe a bad idea for two sided sources
    """
    from itertools import combinations
#    comps = rmodel(mfile)
    comps = rmodel(mfile, sort='flux')
    fmax = comps[0][0]
    if len(comps) < 3:
        wmodel(comps, outfile)
        return

    cxy = rmodel(mfile, xy='xy', sort='flux')
    cmbs = combinations(cxy, 2)
    dists = []
    for c1, c2 in cmbs:
        dists.append(np.hypot(c1[1] - c2[1], c1[2] - c2[2]))
    dists = np.array(dists)
    dists.sort()

# the two most distant components:
    cmbs = combinations(cxy, 2)
    core_flux = None
    for c1, c2 in cmbs:
        if np.hypot(c1[1] - c2[1], c1[2] - c2[2]) == dists[-1]:
            if c1[0] > 0.5*fmax or c2[0] > 0.5*fmax:
                core_flux = max(c1[0], c2[0])
                break

    if core_flux is None:
        cmbs = combinations(cxy, 2)
        for c1, c2 in cmbs:
            if np.hypot(c1[1] - c2[1], c1[2] - c2[2]) == dists[-2]:
                if c1[0] > 0.5*fmax or c2[0] > 0.5*fmax:
                    core_flux = max(c1[0], c2[0])
                    break
    if core_flux is None:
        logger.error('Can not find the core. Try manually...')
        return 1
    logger.debug('Core flux is {}'.format(core_flux))

    for ind, comp in enumerate(comps):
        if comp[0] == core_flux:
            core_ind = ind

    comps = list(comps)
    cc = comps.pop(core_ind)
    comps.insert(0, cc)
    comps = np.asarray(comps)
    wmodel(comps, outfile)
    return 0


if __name__ == "__main__":
    print 'This is difmaptools module'
    m = '/home/osh/tmp/frb1/data/STACK/models/circ_core/0415+379/0415+379_q_2009_02_22_cg_fitted_6.mdl'
    core_first(m, outfile='/home/osh/test1.mdl')



