import math
import numpy as np
import scipy as sp
from utils import degree_to_mas, mas_to_rad, gaussianBeam
from image import ImageModel
from scipy import signal
from data_io import BinTable, get_fits_image_info
from uv_data import open_fits
from stats import LnPost
from components import DeltaComponent, CGComponent

try:
    import pylab
except ImportError:
    pylab = None


# TODO: ``Model`` subclasses can't be convolved with anything! It is ``Image``
# that can be convolved.
# TODO: ``Model`` subclasses shouldn't have ``uv`` or ``image_grid`` attrs.
class Model(object):
    """
    Basic class that represents general functionality of models.
    """
    def __init__(self, stokes=None):
        self._components = list()
        self._uv = None
        self.stokes = stokes

    def __add__(self, other):
        self._components.extend(other._components)

    def add_component(self, component):
        self._components.append(component)

    def add_components(self, *components):
        for component in components:
            self._components.append(component)

    def remove_component(self, component):
        self._components.remove(component)

    def remove_components(self, *components):
        for component in components:
            self._components.remove(component)

    def clear_components(self):
        self._components = list()

    def clear_uv(self):
        self._uv = None

    def ft(self, uv=None):
        """
        Returns FT of model's components at specified points of uv-plane.
        """
        if uv is None:
            uv = self._uv
        ft = np.zeros(len(uv), dtype=complex)
        for component in self._components:
            ft += component.ft(uv)
        return ft

    def get_uv(self, uvdata):
        """
        Sets ``_uv`` attribute of self with values from UVData class instance
        ``uvdata``.

        :param uvdata:
            Instance of ``UVData`` class. Model visibilities will be calculated
            for (u,v)-points of this instance.
        """
        self._uv = uvdata.uvw[:, :2]

    def uvplot(self, uv=None, style='a&p', sym='.r'):
        """
        Plot FT of model vs uv-radius.
        """
        if uv is None:
            uv = self._uv
        ft = self.ft(uv=uv)

        if style == 'a&p':
            a1 = np.angle(ft)
            a2 = np.real(np.sqrt(ft * np.conj(ft)))
        elif style == 're&im':
            a1 = ft.real
            a2 = ft.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        uv_radius = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        pylab.subplot(2, 1, 1)
        pylab.plot(uv_radius, a2, sym)
        if style == 'a&p':
            pylab.ylim([0., 1.3 * max(a2)])
        pylab.subplot(2, 1, 2)
        pylab.plot(uv_radius, a1, sym)
        if style == 'a&p':
            pylab.ylim([-math.pi, math.pi])
        pylab.show()

    @property
    def p(self):
        """
        Shortcut for parameters of model.
        """
        p = list()
        for component in self._components:
            p.extend(component.p)
        return p

    @p.setter
    def p(self, p):
        for component in self._components:
            component.p = p[:component.size]
            p = p[component.size:]

    @property
    def size(self):
        return len(self.p)

    def add_to_image_grid(self, image_grid):
        """
        Add model to instances of ``ImagePlane`` subclasses.

        :param image_grid:
            Instance of ``ImagePlane`` subclass.
        """
        for component in self._components:
            image_grid.add_component(component)

    def make_image(self, fname=None, imsize=None, bmaj=None, bmin=None,
                   bpa=None, size_x=None, size_y=None):
        """
        Method that returns instance of Image class using model data (CCs, clean
        beam, map size & pixel size).

        :param fname: (optional)
            Fits file of image to get image parameters.
        :param bmaj: (optional)
            Beam major axis size [rad].
        :param bmin: (optional)
            Beam minor axis size [rad].
        :param bpa: (optional)
            Beam positional angle [deg]
        :param size_x (optional):
            Size of the first dimension [pixels]. Default is half of image size.
        :param size_y (optional):
            Size of the second dimension [pixels]. Default is half of image
            size.
        :return:
            Instance of ``Image`` class.
        """
        # If we got fits-file then get parameters of image from it
        if fname:
            imsize, pixref, (bmaj, bmin, bpa,), pixsize =\
                get_fits_image_info(fname)

        # First create image grid with components
        image_grid = ImageModel(imsize=imsize, pixref=pixref, pixsize=pixsize)
        # Putting model components to image grid
        self.add_to_image_grid(image_grid)

        if not size_x:
            size_x = int(self.imsize[0] / 2.)
        if not size_y:
            size_y = int(self.imsize[1] / 2.)
        # FIXME: this 3 line should be deleted. There're no beam attrs. in self!
        if not bmaj:
            bmaj = self.bmaj / abs(self.pixsize[0])
        if not bmin:
            bmin = self.bmin / abs(self.pixsize[1])
        if bpa is None:
            bpa = self.bpa
        gaussian_beam = gaussianBeam(size_x, bmaj, bmin, bpa)
        cc_convolved = signal.fftconvolve(image_grid.image_grid, gaussian_beam,
                                          mode='same')
        image = ImageModel()
        image.add_from_array(cc_convolved, pixsize=self.pixsize, bmaj=self.bmaj,
                             bmin=self.bmin, bpa=self.bpa)
        return image


# TODO: add ft_all method for speed up
class CCModel(Model):
    """
    Class that represents clean components model.
    """
    def add_cc_from_fits(self, fname, ver=1):
        """
        Adds CC components to model from FITS-file.

        :param fname:
            Path to FITS-file with model (Clean Components CC-table).
        """
        cc = BinTable(fname, extname='AIPS CC', ver=ver)
        adds = cc.load()
        for flux, x, y in zip(adds['FLUX'], adds['DELTAX'] * degree_to_mas,
                              adds['DELTAY'] * degree_to_mas):
            # We keep positions in mas
            component = DeltaComponent(flux, x, y)
            self.add_component(component)



if __name__ == "__main__":

    # TODO: test with new components.py
    ## TESTING plotting cc-models on uv-data
    # cc_fits_file = '1038+064.l22.2010_05_21.icn.fits'
    # uv_fits_file = '1038+064.l22.2010_05_21.uvf'
    # ccmodel = CCModel(stokes='I')
    # ccmodel.add_cc_from_fits(cc_fits_file)
    # uvdata = open_fits(uv_fits_file)
    # uv = uvdata.uvw[:, :2]
    # ft = ccmodel.ft(uv)
    # ccmodel.uvplot(uv)

    # TESTING fitting gaussian components to uv-data
    # # Load uv-data
    # uvdata = open_fits('1308+326.U1.2009_08_28.UV_CAL')
    # uv = uvdata.uvw[:, :2]
    # # Create several components
    # cg1 = CGComponent(2.44, 0.02, -0.02, 0.10)
    # cg2 = CGComponent(0.041, 0.71, -1.05, 1.18)
    # cg3 = CGComponent(0.044, 2.60, -3.20, 0.79)
    # cg4 = CGComponent(0.021, 1.50, -5.60, 2.08)
    # cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 3.], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 1.], dict(),))
    # cg2.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    # cg3.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 3.], dict(),))
    # cg4.add_prior(flux=(sp.stats.uniform.logpdf, [0., 0.1], dict(),),
    #               bmaj=(sp.stats.uniform.logpdf, [0, 5.], dict(),))
    # # Create model
    # mdl1 = Model(stokes='I')
    # # Add components to model
    # mdl1.add_component(cg1)
    # mdl1.add_component(cg2)
    # mdl1.add_component(cg3)
    # mdl1.add_component(cg4)
    # # Create posterior for data & model
    # lnpost = LnPost(uvdata, mdl1)
    # lnpr = LnPrior(mdl1)
    # lnlik = LnLikelihood(uvdata, mdl1)
    # # model.uvplot(uv = uv)
    # # model.ft(uv=uv)
    # import emcee
    # ndim = mdl1.size
    # nwalkers = 100
    # # p0 = mdl1.p
    # # cov = np.zeros(ndim * ndim).reshape((ndim, ndim,))
    # # cov[0, 0] = 0.1
    # # cov[1, 1] = 0.1
    # # cov[2, 2] = 0.1
    # # cov[3, 3] = 0.1
    # # sampler = emcee.MHSampler(cov, ndim, lnpost)
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    # p_std1 = [0.01, 0.01, 0.01, 0.01]
    # p_std2 = [0.003, 0.01, 0.01, 0.01]
    # p0 = emcee.utils.sample_ball(mdl1.p, p_std1 + p_std2 * 3, size=nwalkers)
    # pos, prob, state = sampler.run_mcmc(p0, 100)
    # sampler.reset()
    # sampler.run_mcmc(pos, 700)
    # # image_grid = ImageModel(fname='J0005+3820_S_1998_06_24_fey_map.fits')
    # # print image_grid.dx, image_grid.dy, image_grid.imsize, image_grid.x_c,\
    # #     image_grid.y_c
    # # print image_grid.image_grid
    # # eg = EGComponent(1., 5., 5., 3., 0.5, 1.)
    # # print eg.p
    # # print eg._p
    # # eg.add_to_image_grid(image_grid)
    # # print image_grid.image_grid



    # With sparse RA data
    # TESTING fitting gaussian components to uv-data
    # Load uv-data
    uvdata = open_fits('0716+714_raes03dp_C_LL_uva.fits')
    uvdata.data['uvw'] *= 10 ** 9
    uv = uvdata.uvw[:, :2]
    # Create several components
    cg1 = CGComponent(1., 0.0, 0.0, 0.05)
    cg1.add_prior(flux=(sp.stats.uniform.logpdf, [0., 2.], dict(),),
                  bmaj=(sp.stats.uniform.logpdf, [0, 1.], dict(),))
    # Create model
    mdl1 = Model(stokes='RR')
    # Add components to model
    mdl1.add_component(cg1)
    # Create posterior for data & model
    lnpost = LnPost(uvdata, mdl1)
    import emcee
    ndim = mdl1.size
    nwalkers = 50
    p0 = mdl1.p
    cov = np.zeros(ndim * ndim).reshape((ndim, ndim,))
    cov[0, 0] = 0.1
    cov[1, 1] = 0.01
    cov[2, 2] = 0.01
    cov[3, 3] = 0.01
    sampler = emcee.MHSampler(cov, ndim, lnpost)
    pos, prob, state = sampler.run_mcmc(p0, 1000)
    #sampler.reset()
    #sampler.run_mcmc(pos, 5000)
    # # image_grid = ImageModel(fname='J0005+3820_S_1998_06_24_fey_map.fits')
    # # print image_grid.dx, image_grid.dy, image_grid.imsize, image_grid.x_c,\
    # #     image_grid.y_c
    # # print image_grid.image_grid
    # # eg = EGComponent(1., 5., 5., 3., 0.5, 1.)
    # # print eg.p
    # # print eg._p
    # # eg.add_to_image_grid(image_grid)
    # # print image_grid.image_grid
