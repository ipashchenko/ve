__author__ = 'ilya'


class Deconvolution(object):
    """
    Base class for deconvolution.
    """
    def __init__(self, uvdata, *args, **kwargs):
        self.uvdata = uvdata


class MEMDeconvolution(Deconvolution):
    def __init_(self, uvdata, dirty_image, dirty_beam, *args, **kwargs):
        super(MEMDeconvolution, self).__init__(uvdata, *args, **kwargs)
        self.dirty_image = dirty_image
        self.dirty_beam = dirty_beam
