#!/usr/bin python
# -*- coding: utf-8 -*-


class Bootstrap(object):
    """
    Sample with replacement (if nonparametric=True) from residuals (data - model) or use normal zero
    mean random variable with std estimated from the residuals for each baseline (or even
    scan) and add this samples to the model to create n bootstrap FITS-files.
    """

    def __init__(self, model, data, nonparametric=False, split_scans=False):
        pass

    def sample(self, outname=None, n=100):
        pass
