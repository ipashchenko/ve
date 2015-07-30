import math
import numpy as np
from scipy import signal
from utils import create_grid, mask_region, fitgaussian, mas_to_rad
from beam import CleanBeam
from fft_routines import fft_convolve2d
# FIXME: w this import can't import anything from from_fits
from from_fits import create_image_from_fits_file

try:
    import pylab
except ImportError:
    pylab = None


class Image(object):
    """
    Class that represents images.
    """
    def __init__(self, imsize=None, pixref=None, pixrefval=None, pixsize=None):
        self.imsize = imsize
        self.dx, self.dy = pixsize
        self.x_c, self.y_c = pixref
        if pixrefval is None:
            pixrefval = (0., 0.,)
        self.x_c_val, self.y_c_val = pixrefval
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
        if self == image:
            self._image = image.image
        else:
            raise Exception("Images have incompatible parameters!")

    def __eq__(self, image):
        """
        Compares current instance of ``Image`` class with other instance.
        """
        return (self.imsize == image.imsize and self.pixsize == image.pixsize)

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
        if isinstance(other, Image):
            self.image *= other.image
        else:
            self.image *= other
        return self

    def __sub__(self, other):
        """
        Substruct from current instance of ``Image`` class other instance or
        some number.
        """
        if isinstance(other, Image):
            self.image -= other.image
        else:
            self.image -= other
        return self

    def __div__(self, other):
        """
        Divide current instance of ``Image`` class on other instance or some
        number.
        """
        if isinstance(other, Image):
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

    def add_model(self, model):
        model.add_to_image(self)

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

    # TODO: fix BLC,TRC to display expected behavior. Or use blc/trc-ing after
    # constructing x&y.
    # TODO: plot beam in corner if called inside ``CCImage``. But any image can
    # has beam... Check UML diagrams...
    # TODO: how plot coordinates in mas if using matshow?
    def plot(self, blc=None, trc=None, clim=None, cmap=None, abs_levels=None,
             rel_levels=None, min_abs_level=None, min_rel_level=None, factor=2.,
             plot_color=False):
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
        if blc or trc:
            blc = blc or (1, 1,)
            trc = trc or self.imsize
            part_to_plot = self.image[blc[0] - 1: trc[0], blc[1]- 1: trc[1]]
            x = self.x[blc[0] - 1: trc[0], blc[0] - 1: trc[0]]
            y = self.y[blc[1] - 1: trc[1], blc[1] - 1: trc[1]]
        else:
            part_to_plot = self.image
            x = self.x
            y = self.y

        # Plot coordinates in milliarcseconds
        x = x / mas_to_rad
        y = y / mas_to_rad

        # Plotting using color if told
        if plot_color:
            imgplot = pylab.matshow(part_to_plot, origin='lower')
            pylab.colorbar()

        # Or plot contours
        elif abs_levels or rel_levels or min_abs_level or min_rel_level:
            max_level = self.image.max()
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
                n_max = int(math.ceil(math.log(max_level / min_abs_level, factor)))
                abs_levels = [min_abs_level * factor ** k for k in range(n_max)]
            # Plot contours
            if abs_levels:
                if plot_color:
                    # White levels on colored background
                    colors='w'
                else:
                    # Black levels on white background
                    colors='b'
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
        #     plt.xlabel("time steps")
        #     plt.ylabel("frequency ch. #")
        #     plt.title('Dynamical spectra')
        #     if savefig is not None:
        #         plt.savefig(savefig, bbox_inches='tight')
        #     plt.show()


# FIXME: THIS DOESN't KEEP RESIDUALS! ONLY CCs! Do i need them here? Usecases?
# It is just visualization of CC model - don't need residuals here.
# Residuals give insight on fidelity of the CC model.
# Residuals should be distinct instance of ``Image`` class that can be added to
# any other instance
class CleanImage(Image):
    """
    Class that represents image made using CLEAN algorithm.
    """
    def __init__(self, imsize=None, pixref=None, pixrefval=None, pixsize=None,
                 bmaj=None, bmin=None, bpa=None):
        super(CleanImage, self).__init__(imsize, pixref, pixrefval, pixsize)
        # TODO: What if pixsize has different sizes???
        # FIXME: Beam has image twice the imsize. It's bad for plotting...
        self._beam = CleanBeam(bmaj / abs(pixsize[0]), bmin / abs(pixsize[0]),
                              bpa, imsize)
        self._residuals = Image(imsize, pixref, pixrefval, pixsize)

    @property
    def beam(self):
        """
        Shorthand for beam image.
        """
        return self._beam.image

    @beam.setter
    def beam(self, bmaj, bmin, bpa):
        self._beam = CleanBeam(bmaj / abs(self.pixsize[0]),
                               bmin / abs(self.pixsize[0]), bpa, self.imsize)

    @property
    def image(self):
        """
        Shorthand for CLEAN image.
        """
        return signal.fftconvolve(self._image, self.beam.image, mode='same')

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


#class MemImage(Image, Model):
#    """
#    Class that represents image made using MEM algorithm.
#    """
#    pass