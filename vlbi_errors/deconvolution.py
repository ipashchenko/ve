__author__ = 'ilya'


class Deconvolution(object):
    """
    Base class for deconvolution.
    """
    def __init__(self, uvdata, *args, **kwargs):
        self.uvdata = uvdata
