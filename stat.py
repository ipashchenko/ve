#!/usr/bin python
# -*- coding: utf-8 -*-


class Bootstrap(object):
    """
    Sample with replacement (if nonparametric=True) from residuals or use
    normal zeromean random variable with std estimated from the residuals
    for each baseline (or even scan).
    
    Inputs:
    
        residuals - instance of Data class. Difference between unself-calibrated
        data and self-calibrated data with gains added.     
        
    """

    def __init__(self, residuals, nonparametric=False, split_scans=False):
        pass

    def sample(self, model, outname=None, nonparametric=None, n=100, sigma=None):   
        """
        Sample from residuals with replacement or sample from normal random noise
        and adds samples to model to form n bootstrap samples.
        """ 
        pass


class CrossValidation(object):
    """
    Class that implements cross-validation-related features.
    """
    
    def __init__(self, data):
        pass
        
    def generate_samples(self, frac=0.2, n=100, outname=None):
        """
        Generate training and testing samples as FITS-files thats inherit FITS-structure of initial data.
        Save training and testing samples to FITS-files.
        """
        pass
     
        
class LnLikelihood(object):
    """
    Class that implements likelihood calculation for given data.
    """
    
    def __init__(self, data, model):
        pass
        
    def __call__(self, p):
        pass