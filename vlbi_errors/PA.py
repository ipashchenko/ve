#!/usr/bin/env python
#-*- coding: utf-8 -*-
import math


def JD_to_LST(JDs, llong):
    """Returns LST [frac. of day] using:
        JD - [iterable] - values of Julian data
        llong - local longitude [+/-degrees].
    """

    LSTs = list()

    llong_h = llong / 15.

    for JD in list(JDs):

        D = JD - 2451545.0
        GMST = 18.697374558 + 24.06570982441908 * D
        GMST = GMST % 24

        if GMST < 0:
            GMST += 24.
        elif GMST >= 24.0:
            GMST -= 24.0

        LST = GMST + llong_h
        #convert to fraction of day
        LST = LST / 24.

        LSTs.append(LST)

    return LSTs


def LST_to_HA(LSTs, RA):
    """
    RA [degrees] - right ascenction
    LST [fr. of days] - local sidireal time
    Returns Hour Angle [rads]
    """

    HAs = list()

    RA_rad = RA * math.pi / 180.

    for LST in list(LSTs):
        HA = 2. * math.pi * LST - RA_rad
        HAs.append(HA)

    return HAs


def PA(JDs, ra, dec, latitude, longitude):
    """Function returns parallactic angles of source (ra, dec) observed at
    moments of Julian Date jds at geographic position (lat, llong) that is
    'east' or 'west' of Greenwitch.
    Parameters:
        jds - [iterable] - set of Julian Dates,
        ra, dec - [float] - right assention & declination of source,
        llong, lat - [float] - geographical longitude and latitude of
        the observer.
    """

    PAs = list()

    LSTs = JD_to_LST(JDs, longitude)

    HAs = LST_to_HA(LSTs, ra)

    for HA in list(HAs):

        PA = math.atan2(math.sin(HA), (math.tan(latitude) * math.cos(dec) -\
            math.sin(dec) * math.cos(HA)))
        PAs.append(PA)

    return PAs
