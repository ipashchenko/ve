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


# Should calculate frequency of any channel that is specified
# by BAND and FREQ indexes (slice).
# pf.hdu.BinTableHDU
class Frequency(object):
    pass


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
