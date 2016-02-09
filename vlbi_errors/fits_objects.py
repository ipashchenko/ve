# Class that describes single VLBI observation and contains instances of such
# classes as UVData, Antenna, Frequency etc.
class Observation(object):
    pass


class UVData(object):
    pass


# pf.hdu.groups.GroupsHDU
class UVDataGroups(UVData):
    pass


# pf.hdu.BinTableHDU
class UVDataIDI(UVData):
    pass


# FREQUENCY table is mandatory if FREQID random parameter is used in UV_DATA
# tables
# Should calculate frequency of any channel that is specified
# by BAND and FREQ indexes (slice).
# pf.hdu.BinTableHDU
class Frequency(object):
    def __init__(self, hdu):
        self.hdu = hdu
        self.n_if = hdu.header['NO IF']
        self.band_offset = hdu.data['BANDFREQ']
        self.ch_width = hdu.data['CH_WIDTH']
        self.band_width = hdu.data['TOTAL BANDWIDTH']
        self.nchannels = self.band_width / self.ch_width

    def frequency_up(self, ch=None, band=None):
        result = self.band_offset[band] + (ch - pref) * self.ch_width[band]



# Get information on antenna position for PA calculation.
# pf.hdu.BinTableHDU
class Antenna(object):
    pass


# Contains instances of Gain classes.
# pf.hdu.BinTableHDU
class Gains(object):
    pass


class Gain(object):
    pass


class Image(object):
    pass


class CleanImage(Image):
    pass


# Info in AIPS CG table, or PrimaryHDU or history of Primary HDU.
class Beam(object):
    pass


class CleanBeam(Beam):
    pass


# Can fit central part and return CleanBeam instance
class DirtyBeam(Beam):
    pass


# Resolution that depends on pixel (position)
class VaryingResolutionBeam(Beam):
    pass


# Simplest image-plane model. Contains FT methods to get visibilities in
# uv-plane. Shortcut to parameters. Methods to adding to Image instances.
# Specifying prior distributions.
class Component(object):
    pass


# Image plane models. Contains instances of Component subclasses. Shortcut to
# parameters.
class Model(object):
    pass


if __name__ == '__main__':
    pass
