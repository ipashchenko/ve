import math
import numpy as np
from scipy import signal
from utils import create_grid, mask_region, fitgaussian, mas_to_rad
from beam import CleanBeam
from fft_routines import fft_convolve2d

try:
    import pylab
except ImportError:
    pylab = None

    # if plt is not None:
    #     plt.figure()
    #     plt.matshow(self.values, aspect='auto')
    #     plt.colorbar()
    #     if not plot_indexes:
    #         raise NotImplementedError("Ticks haven't implemented yet")
    #         # plt.xticks(np.linspace(0, 999, 10, dtype=int),
    #         # frame.t[np.linspace(0, 999, 10, dtype=int)])
    #     plt.xlabel("time steps")
    #     plt.ylabel("frequency ch. #")
    #     plt.title('Dynamical spectra')
    #     if savefig is not None:
    #         plt.savefig(savefig, bbox_inches='tight')
    #     plt.show()

# TODO: how plot coordinates in mas for -10, 0, 10 mas... if using matshow?
def plot(image, x=None, y=None, blc=None, trc=None, clim=None, cmap=None,
         abs_levels=None, rel_levels=None, min_abs_level=None,
         min_rel_level=None, factor=2., plot_color=False, show_beam=False):
    """
    Plot image.

    :param levels:
        Iterable of levels.

    :note:
        ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
        converted to python-like zero-indexing.
    """
    if not pylab:
        raise Exception("Install matplotlib for plotting!")
    if x is None:
        x = np.arange(image.imsize[0])
    if y is None:
        y = np.arange(image.imsize[1])
    if blc or trc:
        blc = blc or (1, 1,)
        trc = trc or image.imsize
        part_to_plot = image[blc[0] - 1: trc[0], blc[1]- 1: trc[1]]
        x = x[blc[0] - 1: trc[0], blc[0] - 1: trc[0]]
        y = y[blc[1] - 1: trc[1], blc[1] - 1: trc[1]]
    else:
        part_to_plot = image
        x = x
        y = y

    # Plot coordinates in milliarcseconds
    x = x / mas_to_rad
    y = y / mas_to_rad

    # Plotting using color if told
    if plot_color:
        imgplot = pylab.matshow(part_to_plot, origin='lower')
        #         # plt.xticks(np.linspace(0, 999, 10, dtype=int),
        #         # frame.t[np.linspace(0, 999, 10, dtype=int)])
        #         # plt.yticks(np.linspace(0, len(dm_grid) - 10, 5, dtype=int),
        #         #            vint(dm_grid[np.linspace(0, len(dm_grid) - 10, 5,
        #         #            dtype=int)]))
        if show_beam:
            raise NotImplementedError
        pylab.colorbar()

    # Or plot contours
    elif abs_levels or rel_levels or min_abs_level or min_rel_level:
        max_level = image.max()
        # Build levels (pylab.contour takes only absolute values)
        if abs_levels or rel_levels:
            # If given both then ``abs_levels`` has a priority
            if abs_levels:
                rel_levels = None
            else:
                abs_levels = [max_level * i for i in rel_levels]
        # If given only min_abs_level & increment factor (default is 2)
        elif min_abs_level or min_rel_level:
            if min_rel_level:
                min_abs_level = min_rel_level * max_level / 100.
            n_max = int(math.ceil(math.log(max_level / min_abs_level,
                                           factor)))
            abs_levels = [min_abs_level * factor ** k for k in range(n_max)]
        # Plot contours
        if abs_levels:
            if plot_color:
                # White levels on colored background
                colors = 'w'
            else:
                # Black levels on white background
                colors = 'b'
            print "Plotting contours with levels: " + str(abs_levels)
            imgplot = pylab.contour(x, y, part_to_plot, abs_levels,
                                    colors=colors)
    else:
        raise Exception("Specify ``plot_color=True`` or choose some "
                        "levels!")

    if cmap:
        try:
            imgplot.set_cmap(cmap)
        except:
            # Show wo ``cmap`` set, print availbale ``cmap``s.
            pass
    if clim:
        # TODO: Warn if ``clim`` is out of range for image.
        imgplot.set_clim(clim)

        # FIXME: use matplotlib.pyplot!
        # if plt is not None:
        #     plt.figure()
        #     plt.matshow(self.values, aspect='auto')
        #     plt.colorbar()
        #     if not plot_indexes:
        #         raise NotImplementedError("Ticks haven't implemented yet")
        #         # plt.xticks(np.linspace(0, 999, 10, dtype=int),
        #         # frame.t[np.linspace(0, 999, 10, dtype=int)])
        #         # plt.yticks(np.linspace(0, len(dm_grid) - 10, 5, dtype=int),
        #         #            vint(dm_grid[np.linspace(0, len(dm_grid) - 10, 5,
        #         #            dtype=int)]))
        # First find places where coordinates are ints and then label this
        # places
        # labelPositions = arange(len(D))
        # newLabels = ['z','y','x','w','v','u','t','s','q','r']
        # plt.xticks(labelPositions,newLabels)
        #     plt.xlabel("time steps")
        #     plt.ylabel("frequency ch. #")
        #     plt.title('Dynamical spectra')
        #     if savefig is not None:
        #         plt.savefig(savefig, bbox_inches='tight')
        #     plt.show()


# TODO: Option for saving ``Image`` instance
class BasicImage(object):
    """
    Class that represents images.
    """
    def __init__(self, imsize=None, pixref=None, pixrefval=None, pixsize=None):
        self.imsize = imsize
        self.pixsize = pixsize
        self.pixref = pixref
        self.dx, self.dy = pixsize
        self.x_c, self.y_c = pixref
        if pixrefval is None:
            pixrefval = (0., 0.,)
        self.x_c_val, self.y_c_val = pixrefval
        self.pixrefval = pixrefval
        # Create flux array
        self._image = np.zeros(self.imsize, dtype=float)
        # Create coordinate arrays
        x, y = create_grid(self.imsize)
        x = x - self.x_c
        y = y - self.y_c
        x = x * self.dx
        y = y * self.dy
        self.x = x
        self.y = y

    # TODO: Sometimes we need to add/substract convolved images. So subclasses
    # should implement property with convolution.
    @property
    def image(self):
        """
        Shorthand for image array.
        """
        return self._image

    # TODO: Am i need it? Should i compare instances before setting?
    @image.setter
    def image(self, image):
        if isinstance(image, BasicImage):
            if self == image:
                self._image = image.image.copy()
            else:
                raise Exception("Images have incompatible parameters!")
        # If ``image`` is array-like
        else:
            self._image = np.atleast_2d(image).copy()

    def __eq__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (self.imsize == other.imsize and self.pixsize == other.pixsize)

    def __ne__(self, image):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (self.imsize != image.imsize or self.pixsize != image.pixsize)

    def __add__(self, image):
        """
        Sums current instance of ``Image`` class with other instance.
        """
        self.image += image.image
        return self

    def __mul__(self, other):
        """
        Multiply current instance of ``Image`` class with other instance or
        some number.
        """
        if isinstance(other, BasicImage):
            self.image *= other.image
        else:
            self.image *= other
        return self

    def __sub__(self, other):
        """
        Substruct from current instance of ``Image`` class other instance or
        some number.
        """
        if isinstance(other, BasicImage):
            self._image -= other.image
        else:
            self._image -= other
        return self

    def __div__(self, other):
        """
        Divide current instance of ``Image`` class on other instance or some
        number.
        """
        if isinstance(other, BasicImage):
            self.image /= other.image
        else:
            self.image /= other
        return self

    # Convolve with any object that has ``image`` attribute
    def convolve(self, image_like):
        """
        Convolve ``Image`` array with image-like instance.
        """
        return signal.fftconvolve(self._image, image_like.image, mode='same')

    def add_component(self, component):
        component.add_to_image(self)

    # TODO: Should i compare stokes of ``Image`` and ``Model`` instances?
    def add_model(self, model):
        model.add_to_image(self)

    # TODO: Implement Rayleigh (Rice) distributed noise for stokes I
    def add_noise(self, std, df=None):
        size = self.imsize[0] * self.imsize[1]
        if df is None:
            rvs = np.random.normal(loc=0., scale=std, size=size)
        else:
            raise NotImplementedError
        rvs = rvs.reshape(self.imsize)
        self._image += rvs

    # TODO: Should i compare images before?
    # TODO: Implement several regions to include for each image
    def cross_correlate(self, image, region1=None, region2=None):
        """
        Cross-correlates current instance of ``Image`` with another instance.

        Computes normalized cross-correlation of images.

        :param image:
            Instance of image class.
        :param region1 (optional):
            Region to EXCLUDE in current instance of ``Image``.
            Or (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
            None,). Default ``None``.
        :param region2 (optional):
            Region to EXCLUDE in ``image``. Or (blc[0], blc[1], trc[0], trc[1],)
            or (center[0], center[1], r, None,). Default ``None``.
        :return:
            (dx, dy,) tuple of shifts (subpixeled) in each direction.
        """
        if region1 is not None:
            image1 = mask_region(self.image, region1)
        if region2 is not None:
            image2 = mask_region(image.image, region2)
        # Cross-correlate images
        shift_array = fft_convolve2d(image1, image2)
        params = fitgaussian(shift_array)
        return tuple(params[1: 3])

    def plot(self, blc=None, trc=None, clim=None, cmap=None, abs_levels=None,
             rel_levels=None, min_abs_level=None, min_rel_level=None, factor=2.,
             plot_color=False):
        """
        Plot image.

        :note:
            ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
            converted to python-like zero-indexing.

        """
        plot(self.image, x=self.x, y=self.y, blc=blc, trc=trc, clim=clim,
             cmap=cmap, abs_levels=abs_levels, rel_levels=rel_levels,
             min_abs_level=min_abs_level, min_rel_level=min_rel_level,
             factor=factor, plot_color=plot_color)
        # if not pylab:
        #     raise Exception("Install matplotlib for plotting!")
        # if blc or trc:
        #     blc = blc or (1, 1,)
        #     trc = trc or self.imsize
        #     part_to_plot = self.image[blc[0] - 1: trc[0], blc[1]- 1: trc[1]]
        #     x = self.x[blc[0] - 1: trc[0], blc[0] - 1: trc[0]]
        #     y = self.y[blc[1] - 1: trc[1], blc[1] - 1: trc[1]]
        # else:
        #     part_to_plot = self.image
        #     x = self.x
        #     y = self.y

        # # Plot coordinates in milliarcseconds
        # x = x / mas_to_rad
        # y = y / mas_to_rad

        # # Plotting using color if told
        # if plot_color:
        #     imgplot = pylab.matshow(part_to_plot, origin='lower')
        #     pylab.colorbar()

        # # Or plot contours
        # elif abs_levels or rel_levels or min_abs_level or min_rel_level:
        #     max_level = self.image.max()
        #     # Build levels (pylab.contour takes only absolute values)
        #     if abs_levels or rel_levels:
        #         # If given both then ``abs_levels`` has a priority
        #         if abs_levels:
        #             rel_levels = None
        #         else:
        #             abs_levels = [max_level * i for i in rel_levels]
        #     # If given only min_abs_level & increment factor (default is 2)
        #     elif min_abs_level or min_rel_level:
        #         if min_rel_level:
        #             min_abs_level = min_rel_level * max_level / 100.
        #         n_max = int(math.ceil(math.log(max_level / min_abs_level, factor)))
        #         abs_levels = [min_abs_level * factor ** k for k in range(n_max)]
        #     # Plot contours
        #     if abs_levels:
        #         if plot_color:
        #             # White levels on colored background
        #             colors='w'
        #         else:
        #             # Black levels on white background
        #             colors='b'
        #         print "Plotting contours with levels: " + str(abs_levels)
        #         imgplot = pylab.contour(x, y, part_to_plot, abs_levels,
        #                                 colors=colors)
        # else:
        #     raise Exception("Specify ``plot_color=True`` or choose some "
        #                     "levels!")

        # if cmap:
        #     try:
        #         imgplot.set_cmap(cmap)
        #     except:
        #         # Show wo ``cmap`` set, print availbale ``cmap``s.
        #         pass
        # if clim:
        #     # TODO: Warn if ``clim`` is out of range for image.
        #     imgplot.set_clim(clim)

        # # FIXME: use matplotlib.pyplot!
        # # if plt is not None:
        # #     plt.figure()
        # #     plt.matshow(self.values, aspect='auto')
        # #     plt.colorbar()
        # #     if not plot_indexes:
        # #         raise NotImplementedError("Ticks haven't implemented yet")
        # #         # plt.xticks(np.linspace(0, 999, 10, dtype=int),
        # #         # frame.t[np.linspace(0, 999, 10, dtype=int)])
        # #         # plt.yticks(np.linspace(0, len(dm_grid) - 10, 5, dtype=int),
        # #         #            vint(dm_grid[np.linspace(0, len(dm_grid) - 10, 5,
        # #         #            dtype=int)]))
        # #     plt.xlabel("time steps")
        #     plt.ylabel("frequency ch. #")
        #     plt.title('Dynamical spectra')
        #     if savefig is not None:
        #         plt.savefig(savefig, bbox_inches='tight')
        #     plt.show()


class Image(BasicImage):
    """
    Class that represents images obtained using radio interferometry.
    """
    def __init__(self, imsize=None, pixref=None, pixrefval=None, pixsize=None,
                 stokes=None, freq=None):
        super(Image, self).__init__(imsize=imsize, pixref=pixref,
                                         pixrefval=pixrefval, pixsize=pixsize)
        self.stokes = stokes
        self.freq = freq



# TODO: Add method ``shift`` that shifts image (CCs and residulas)
# FIXME: Better shift in uv-domain
# TODO: Should i extend ``__eq__`` by comparing beams too?
class CleanImage(Image):
    """
    Class that represents image made using CLEAN algorithm.
    """
    def __init__(self, imsize=None, pixref=None, pixrefval=None, pixsize=None,
                 stokes=None, freq=None, bmaj=None, bmin=None, bpa=None):
        super(CleanImage, self).__init__(imsize, pixref, pixrefval, pixsize,
                                         stokes, freq)
        # TODO: What if pixsize has different sizes???
        # FIXME: Beam has image twice the imsize. It's bad for plotting...
        self._beam = CleanBeam(bmaj / abs(pixsize[0]), bmin / abs(pixsize[0]),
                              bpa, imsize)
        self._residuals = BasicImage(imsize, pixref, pixrefval, pixsize)

    def __eq__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (super(CleanImage, self).__eq__(other) and
                self._beam.__eq__(other._beam))

    def __ne__(self, other):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (super(CleanImage, self).__ne__(other) or
                self._beam.__ne__(other._beam))

    @property
    def beam(self):
        """
        Shorthand for beam image.
        """
        return self._beam.image

    @beam.setter
    def beam(self, beam_pars):
        """
        Set beam parameters.

        :param beam_pars:
            Iterable of bmaj [pix], bmin [pix], bpa [deg].
        """
        self._beam = CleanBeam(beam_pars[0] / abs(self.pixsize[0]),
                               beam_pars[1] / abs(self.pixsize[0]),
                               beam_pars[2], self.imsize)

    @property
    def image(self):
        """
        Shorthand for CLEAN image.
        """
        return signal.fftconvolve(self._image, self.beam, mode='same')

    @property
    def cc(self):
        """
        Shorthand for image of clean components.
        """
        return self._image

    @property
    def image_w_residuals(self):
        """
        Shorthand for CLEAN image with residuals added.
        """
        return self.image + self.residuals

    # FIXME: Should be read-only as residuals have sense only for naitive clean
    @property
    def residuals(self):
        return self._residuals.image


    def plot(self, to_plot, blc=None, trc=None, clim=None, cmap=None,
             abs_levels=None, rel_levels=None, min_abs_level=None,
             min_rel_level=None, factor=2., plot_color=False):
        """
        Plot image.

        :param to_plot:
            "cc", "ccr", "ccrr", "r" or "beam" - to plot only CC, CC Restored
            with beam, CC Restored with Residuals added, Residuals only or Beam.

        :note:
            ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
            converted to python-like zero-indexing.

        """
        plot_dict = {"cc": self._image, "ccr": self.image, "ccrr":
            self.image_w_residuals, "r": self._residuals.image,
                     "beam": self.beam}
        plot(plot_dict[to_plot], x=self.x, y=self.y, blc=blc, trc=trc,
             clim=clim, cmap=cmap, abs_levels=abs_levels, rel_levels=rel_levels,
             min_abs_level=min_abs_level, min_rel_level=min_rel_level,
             factor=factor, plot_color=plot_color)


#class MemImage(BasicImage, Model):
#    """
#    Class that represents image made using MEM algorithm.
#    """
#    pass

if __name__ == '__main__':
    data_dir = '/home/ilya/vlbi_errors/0148+274/2007_03_01/'
    # Directory with fits-images of bootstrapped data
    i_dir_c1 = data_dir + 'C1/im/I/'
    i_dir_c2 = data_dir + 'C2/im/I/'
    i_dir_x1 = data_dir + 'X1/im/I/'
    i_dir_x2 = data_dir + 'X2/im/I/'
    q_dir_c1 = data_dir + 'C1/im/Q/'
    u_dir_c1 = data_dir + 'C1/im/U/'
    q_dir_c2 = data_dir + 'C2/im/Q/'
    u_dir_c2 = data_dir + 'C2/im/U/'
    q_dir_x1 = data_dir + 'X1/im/Q/'
    u_dir_x1 = data_dir + 'X1/im/U/'
    q_dir_x2 = data_dir + 'X2/im/Q/'
    u_dir_x2 = data_dir + 'X2/im/U/'
    i_cc_file = '/home/ilya/vlbi_errors/0148+274/2007_03_01/X2/im/I/orig_cc.fits'
    from from_fits import create_clean_image_from_fits_file
    ccimage = create_clean_image_from_fits_file(i_cc_file)

