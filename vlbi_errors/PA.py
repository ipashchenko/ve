import numpy as np


def JD_to_LST(JDs, longitude):
    """
    :param JDs:
        Iterable of Julian Dates.
    :param longitude:
        Value of local longitude [+/-degrees where sign "-" for West to
        Greenwitch sites].
    :return:
        Numpy array with Local Sidereal Time values.
    """
    JDs = np.array(JDs)
    longitude_h = longitude/15.

    D = JDs - 2451545.0
    GMST = 18.697374558 + 24.06570982441908 * D
    GMST = GMST % 24

    GMST[GMST < 0] += 24.0
    GMST[GMST >= 24.0] -= 24.0
    LST = GMST + longitude_h
    LST = LST / 24.
    return LST


def LST_to_HA(LST, ra):
    """
    :param LST:
        Iterable of Local Sidereal Time values [frac. of the day].
    :param ra:
        Right Ascension [deg].
    :return:
        Numpy array of hour angles [rad].
    """

    ra_rad = ra*np.pi/180.
    LST = np.array(LST)
    return 2.*np.pi*LST-ra_rad


def PA(JD, ra, dec, latitude, longitude):
    """
    :param JD:
        Iterable of Julian Dates of observation.
    :param ra:
        Right Ascension [deg] of the source.
    :param dec:
        Declination [deg] of the source.
    :param latitude:
        Geographical latitude [deg] of the site of observation.
    :param longitude:
        Geographical longitude [deg] of the site of observation. 'East' or
        'West' of Greenwich => +/- sign for longitude.

    """
    LSTs = JD_to_LST(JD, longitude)
    HA = LST_to_HA(LSTs, ra)
    return np.arctan2(np.sin(HA),
                      (np.tan(latitude) * np.cos(dec) -
                       np.sin(dec) * np.cos(HA)))