import os
import matplotlib.pyplot as plt
import datetime
import glob
import os
import pandas as pd
import numpy as np
from utils import hdi_of_mcmc
from image_ops import rms_image
from mojave import download_mojave_uv_fits, mojave_uv_fits_fname
from spydiff import import_difmap_model, modelfit_difmap, clean_difmap
from uv_data import UVData
from model import Model
from bootstrap import CleanBootstrap
from components import DeltaComponent, CGComponent, EGComponent
from from_fits import create_image_from_fits_file

try:
    import corner as triangle
except ImportError:
    triangle = None

base_dir = '/home/ilya/vlbi_errors/mojave_mod'
n_boot = 300
outname = 'boot_uv'
names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
         'bpa']
df = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
                   names=names, dtype={key: str for key in names},
                   index_col=False)

source_epochs_core = dict()

# Mow for all sources get the latest epoch and create directory for analysis
for source in df['source'].unique():
    print(source)
    source_epochs_core[source] = dict()
    epochs = df.loc[df['source'] == source]['epoch']
    for epoch_ in epochs:
        print(epoch_)
        epoch = epoch_.replace('-', '_')
        source_epochs_core[source][epoch_] = dict()

        model_df = df.loc[np.logical_and(df['source'] == source,
                                         df['epoch'] == epoch_)]
        for (flux, r, pa, bmaj, e, bpa) in np.asarray(model_df[['flux', 'r', 'pa',
                                                                'bmaj', 'e',
                                                                'bpa']])[:1]:
            # print flux, r, pa, bmaj, e, bpa
            if not r.strip(' '):
                r = '0.0'
            if not pa.strip(' '):
                pa = '0.0'

            if not bmaj.strip(' '):
                bmaj = '0.0'
            if not e.strip(' '):
                e = "1.0"

            if np.isnan(float(bpa)):
                bpa = "0.0"
            else:
                bpa = bpa + 'v'

            if bmaj == '0.0':
                type_ = 0
                bpa = "0.0"
            else:
                bmaj = bmaj + 'v'
                type_ = 1
            print flux, r, pa, bmaj, e, bpa
            source_epochs_core[source][epoch_] = {'flux': flux, 'r': r,
                                                  'pa': pa, 'e': e, 'bpa': bpa,
                                                  'bmaj': bmaj}


sources = source_epochs_core.keys()
outdir = '/home/ilya/code/vlbi_errors/pics/sanya'
for source in sources:
    bmajs = list()
    fluxes = list()
    epochs = list()
    for key, value in source_epochs_core[source].items():
        fluxes.append(value['flux'])
        bmajs.append(value['bmaj'])
        epochs.append(key)
    fluxes = np.array([float(flux) for flux in fluxes])
    bmajs = np.array([float(bmaj.rstrip('v')) for bmaj in bmajs])
    epochs = np.array([datetime.datetime.strptime(epoch, '%Y-%m-%d') for epoch
                       in epochs])
    zeros_indxs = bmajs == 0
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(epochs, fluxes, marker='o', c='red', label='flux', s=20)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Core flux, [Jy]", color='red')
    ax2 = ax1.twinx()
    ax2.scatter(epochs, bmajs, marker='o', c='g', label='size', s=20)
    ax2.set_ylabel("Core major axis, [mas]", color='g')
    plt.gcf().autofmt_xdate()
    plt.show()
    fig.savefig(os.path.join(outdir, "{}_core_flux_vs_size.png".format(source)),
                bbox_inches='tight', dpi=200)
    plt.close()

