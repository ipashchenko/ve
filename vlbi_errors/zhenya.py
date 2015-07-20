from from_fits import (create_uvdata_from_fits_file,
                       create_ccmodel_from_fits_file)
from bootstrap import CleanBootstrap

# TODO: Create directory structure for keeping bootstrapped data.
# TODO: We need to get RM map and it's uncertainty for each source.
# Input: calibrated visibilities, CLEAN models in "naitive" resolution.
# Maps on higher frequencies are made by:
#     1) convolving clean model with low-frequency beam
#     2) cleaning uv-data using low-frequency beam and parameters of
#         low-frequency CC-maps.
# I think i should use 2) because output of bootstrap - set of resampled
# uv-data - and i should use "naitive" CC-model for resampling.
# Then shift all low frequency CC-maps by specified shift.

# FIXME: Actually, this shift should be calculated between sets of resampled
# imaged data to obtain the distribution of shifts.


# C - 4.6&5GHz, X - 8.11&8.43GHz, U - 15.4GHz
bands = ['C1', 'C2', 'X1', 'X2', 'U1']
dates = ['2007_03_01', '2007_04_30', '2007_05_03', '2007_06_01']
sources = ['0148+274',
           '0342+147',
           '0425+048',
           '0507+179',
           '0610+260',
           '0839+187',
           '0923+392',
           '0952+179',
           '1004+141',
           '1011+250',
           '1049+215',
           '1219+285',
           '1226+023',
           '1406-076',
           '1458+718',
           '1642+690',
           '1655+077',
           '1803+784',
           '1830+285',
           '1845+797',
           '2201+315',
           '2320+506']

stokes = ['i', 'q', 'u']


def clean_fits_fname(source, band, date, stokes, ext='fits'):
    return source + '.' + band.lower() + '.' + date + '.' + stokes + '.' + ext


def uv_fits_fname(source, band, date, ext='PINAL'):
    return source + '.' + band + '.' + date + '.' + ext


if __name__ == '__main__':

    band = 'C1'
    source = '0425+048'
    date = '2007_04_30'
    uv_ext = 'PINAL'
    image_ext = 'fits'

    uv_fname = uv_fits_fname(source, band, date, ext=uv_ext)

    # Create instance of ``UVData``
    uvdata = create_uvdata_from_fits_file(uv_fname)
    for stoke in stokes:
        map_fname = clean_fits_fname(source, band, date, stoke, ext=image_ext)
        # Create instance of ``CCModel``
        ccmodel = create_ccmodel_from_fits_file(map_fname,
                                                stokes=stoke.upper())
        # I will resample the same uv-data set but different stokes
        boot = CleanBootstrap(ccmodel, uvdata)
        # 0342+147.x2.2007_06_01.i_13.BFITS where ``i`` - number 1-100.
        boot.run(n=100, outname=[uv_fname, ''])
