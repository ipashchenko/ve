#!/usr/bin python2
# -*- coding: utf-8 -*-

from fits_formats import UV_FITS


class Data(object):
    """
    Represent VLBI observational data.
    #FIXME Don't use u,v,w! Use records instead for iterations while ft
    """

    def __init__(self, fits_format=UV_FITS()):
        """
        format - instance of UV_FITS or FITS_IDI class used for io.
        """
        self._fits_format = fits_format
        self._recarray = None
        # Recarray - nice container of interferometric data
        #(u, v, w, t, bl, dt, wght, cmatrix(if, chan, stok))

    def load(self, fname):
        """
        Load data from FITS-file.
        """
        self._recarray = self._fits_format.load(fname)

    def save(self, fname):
        """
        Save data to FITS-file.
        """
        #TODO: put data from self.records (recarray) to HDU data
        # if recarray is not a view of HDU.data
        self._fits_format.save(fname)

    @property
    def uvw(self):
        """
        Returns recarray with (u, v, w) fileds.
        """
        return np.rec.fromarays(self._recarray.u, self._recarray.v, self._recarray.w)
        #return self._recarray[:3]
        
    @property
    def baselines(self):
        """
        Returns the list of baselines.
        """
        return sort(list(set(self._recarray['BASELINE'])))

    def uvplot(self, posangle=None):
        """
        Plot data vs. uv-plot distance.
        """
        pass
     
    def vplot(self):
        """
        Plot data vs. time.
        """
        pass

    def fit(self, model):
        """
        Fit visibility data with image plane model.
        """
        pass

    def __add__(self, data):
        """
        Add data to self.
        """
        pass

    def __multiply__(self, gains):
        """
        Applies complex antenna gains from instance of Gains class to
        the visibilities of self.
        """
        pass

    #TODO: how to substitute data to model only on one baseline?
    def substitute(self, model):
        """
        Substitue data of self with visibilities of the model.
        """
        uv_correlations = model.correlations(uvws=self.uvw)
        uv_correlations.broadcast(self._recarray)
        #self._recarray = uv_correlations
        
        return uv_correlations

    def noise_calculate(self, split_scans=False):
        """
        Calculate noise on all baselines. Used also for specifying priors
        on noise on each baseline.
        """
        for baseline in self.baselines:
            baseline_data = self.get_data(baseline=baseline)
            

    #TODO: implement the possibility to choose distribution for each baseline
    # and scan
    def noise_add(self, stds, split_scans=False):
        """
        Add gaussian noise to self using specified std for each baseline/
        scan.
        
        Inputs:
        
            stds - mapping from baseline number to std of noise or to
                iterables of stds (if split_scans=True).
        """
        pass

    def cv_score(self, model, stoke='I'):
        """
        Returns Cross-Validation score for self (as testing cv-sample) and model
        (trained on training cv-sample).
        """
        
        model_cmatrix = self.substitue(model, stoke=stoke)
        uv_difference = self._recarray.cmatrix - model_cmatrix
        
        for baseline in self.baselines:
            # square difference for each baseline, divide by baseline noise
            # and then sum for current baseline
            