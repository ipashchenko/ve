import os
import pandas as pd
import numpy as np
from mojave import get_mojave_mdl_file
from mojave import get_epochs_for_source
from mojave import download_mojave_uv_fits
from mojave import convert_mojave_epoch_to_float


base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mojave_chisq'

names = ['source', 'id', 'trash', 'epoch', 'flux', 'r', 'pa', 'bmaj', 'e',
         'bpa']
df1 = pd.read_table(os.path.join(base_dir, 'asu.tsv'), sep=';', header=None,
                    names=names, dtype={key: str for key in names},
                    index_col=False)
names = ['source', 'id', 'flux', 'r', 'pa', 'n_mu', 't_mid', 's_ra', 's_dec']
df2 = pd.read_table(os.path.join(base_dir, 'asu_chisq_all.tsv'), sep='\t',
                    header=None, names=names, dtype={key: str for key in names},
                    index_col=False, skiprows=1)

source_dict = dict()
for source in df2['source'].unique():
    # source_epochs = get_epochs_for_source(source)
    df = df1.loc[df1['source'] == source]
    source_epochs = df['epoch'].values
    source_epochs_ = list()
    for epoch in source_epochs:
        try:
            epoch_ = convert_mojave_epoch_to_float(epoch)
        except ValueError:
            print("Can't convert epoch : {} for source {}".format(epoch, source))
            continue
        source_epochs_.append(epoch_)
    source_dict[source] = list()
    df = df2.loc[df2['source'] == source]
    for index, row in df.iterrows():
        if row['n_mu'] != 'a':
            id = int(row['id'])
            flux = float(row['flux'])
            r = float(row['r'])
            pa = float(row['pa'])
            t_mid = float(row['t_mid'])
            closest_epoch = source_epochs[np.argmin(abs(np.array(source_epochs_)
                                                        - t_mid))]
            sigma_pos = np.hypot(float(row['s_ra']), float(row['s_dec']))
            source_dict[source].append([id, flux, r, pa, t_mid, closest_epoch,
                                        sigma_pos])





