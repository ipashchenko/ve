import numpy as np


def is_embraced(curve, means, widths):
    """
    Function that checks if 1D curve is embraced by 1D band.

    :param curve:
        Iterable of curve's points.
    :param means:
        Iterable of band's means.
    :param widths:
        Iterable of widths (sigmas).
    :return:
        ``True`` if curve is embraced by band.
    """
    curve = np.array(curve)
    means = np.array(means)
    widths = np.array(widths)
    assert len(curve) == len(means) == len(widths)
    diff = np.abs(curve - means)

    return np.alltrue(diff < widths)


def count_contained(curves, means, widths):
    """
    Count haw many curves are contained inside given band.

    :param curves:
        Iterable of numpy 1D arrays with curves to count.
    :param means:
        Iterable of band's means.
    :param widths:
        Iterable of widths (sigmas).
    :return:
        Number of curves within band.
    """
    i = 0
    for curve in curves:
        if is_embraced(curve, means, widths):
            i += 1
    return i


def create_sim_conf_band(curves, means, widths, alpha=0.95, delta=0.01):
    """
    Function that builds simultaneous confidence band.

    :param curves:
        Iterable of numpy 1D arrays with curves to count.
    :param means:
        Iterable of band's means.
    :param widths:
        Iterable of widths (sigmas).
    :param alpha: (optional)
        Number in (0., 1.) - ``1 - significance`` of CB. (default: ``0.95``)
    :param delta: (optional)
        Step of widening of band. (default: ``0.01``)
    :return:
        Two 1D numpy arrays with low and upper alpha-level simultaneous
        confidence bands.
    """
    n = count_contained(curves, means, widths)
    f = 1
    while n < len(curves) * alpha:
        f += delta
        n = count_contained(curves, means, f * widths)
    return means - f * widths, means + f * widths
