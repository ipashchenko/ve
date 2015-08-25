import math
import numpy as np
from scipy import signal
from utils import (create_grid, mask_region, fitgaussian, mas_to_rad, v_round)
from beam import CleanBeam
from fft_routines import fft_convolve2d

try:
    import pylab
except ImportError:
    pylab = None

    # # Plot every k-th pol.vector
    # k = 2
    # blc = (240, 230)
    # trc = (300, 370)
    # # Pixels
    # x_center = blc[1] + (trc[1] - blc[1]) / 2. - 256
    # y_center = blc[0] + (trc[0] - blc[0]) / 2. - 256
    # # Pixsels
    # x_slice = slice(blc[1], trc[1], None)
    # y_slice = slice(blc[0], trc[0],  None)
    # # Create picture with contours - I, color - fpol and vectors - direction and
    # # value of Linear Polarization
    # import os
    # from images import Images
    # data_dir = '/home/ilya/vlbi_errors/0148+274/2007_03_01/'
    # i_dir_c1 = data_dir + 'C1/im/I/'
    # q_dir_c1 = data_dir + 'C1/im/Q/'
    # u_dir_c1 = data_dir + 'C1/im/U/'
    # print "Creating PANG image..."
    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
    #                              os.path.join(u_dir_c1, 'cc.fits')])
    # pang_image = images.create_pang_images()[0]

    # print "Creating PPOL image..."
    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
    #                              os.path.join(u_dir_c1, 'cc.fits')])
    # ppol_image = images.create_pol_images()[0]

    # print "Creating I image..."
    # from from_fits import create_clean_image_from_fits_file
    # i_image = create_clean_image_from_fits_file(os.path.join(i_dir_c1,
    #                                                          'cc.fits'))
    # print "Creating FPOL image..."
    # images = Images()
    # images.add_from_fits(fnames=[os.path.join(i_dir_c1, 'cc.fits'),
    #                              os.path.join(q_dir_c1, 'cc.fits'),
    #                              os.path.join(u_dir_c1, 'cc.fits')])
    # fpol_image = images.create_fpol_images()[0]

    # beam_place = 'ul'
    # pixsize = abs(i_image.pixsize[0])
    # imsize_x = x_slice.stop - x_slice.start
    # imsize_y = y_slice.stop - y_slice.start
    # factor = 206264806.719150
    # # mas
    # arc_length_x = pixsize * imsize_x * factor
    # arc_length_y = pixsize * imsize_y * factor
    # # mas
    # x_ = i_image.x[0, :][x_slice] * factor
    # y_ = i_image.y[:, 0][y_slice] * factor
    # # TODO: Does "-" sign because of RA increases to the left actually? VLBIers
    # # do count angles from North to negative RA.
    # u = -ppol_image.image[x_slice, y_slice] * np.sin(pang_image.image[x_slice,
    #                                                                   y_slice])
    # v = ppol_image.image[x_slice, y_slice] * np.cos(pang_image.image[x_slice,
    #                                                                  y_slice])
    # # arc_length = pixsize * imsize * factor
    # # x = y = np.linspace(-arc_length/2, arc_length/2, imsize)
    # # FIXME: wrong zero location
    # x = np.linspace(x_[0], x_[-1], imsize_x)
    # y = np.linspace(y_[0], y_[-1], imsize_y)
    # i_array = i_image.image_w_residuals[x_slice, y_slice]
    # ppol_array = ppol_image.image[x_slice, y_slice]
    # fpol_array = fpol_image.image[x_slice, y_slice]
    # # Creating masks
    # i_mask = np.zeros((imsize_x, imsize_y))
    # ppol_mask = np.zeros((imsize_x, imsize_y))
    # # i_mask[abs(i_array) < 0.0001] = 1
    # ppol_mask[ppol_array < 0.00125] = 1
    # fpol_mask = np.logical_or(i_mask, ppol_mask)
    # # Masking data
    # i_array_masked = np.ma.array(i_array, mask=i_mask)
    # fpol_array_masked = np.ma.array(fpol_array, mask=ppol_mask)
    # ppol_array_masked = np.ma.array(ppol_array, mask=ppol_mask)
    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    # # FIXME: wrong zero location
    # # aspect='auto' is bad for VLBI images
    # i = ax.imshow(fpol_array_masked, interpolation='none', label='FPOL',
    #               extent=[y[0], y[-1], x[0], x[-1]], origin='lower',
    #               cmap=plt.get_cmap('hsv'))
    # co = ax.contour(y, x, i_array_masked, [-0.00018 * 2] + [2 ** (j) * 2 * 0.00018 for
    #                                                         j in range(12)],
    #                 colors='k', label='I')
    # m = np.zeros(u.shape)
    # u = np.ma.array(u, mask=ppol_mask)
    # v = np.ma.array(v, mask=ppol_mask)
    # vec = ax.quiver(y[::k], x[::k], u[::k, ::k], v[::k, ::k],
    #                 angles='uv', units='xy', headwidth=0, headlength=0,
    #                 headaxislength=0, scale=0.005, width=0.05)
    # # Doesn't show anything
    # ax.legend()
    # # c = Circle((5, 5), radius=4,
    # #            edgecolor='red', facecolor='blue', alpha=
    # e_height = 10 * pixsize * factor
    # e_width = 5 * pixsize * factor
    # r_min = e_height / 2
    # if beam_place == 'lr':
    #     y_c = y[0] + r_min
    #     x_c = x[-1] - r_min
    # elif beam_place == 'll':
    #     y_c = y[0] + r_min
    #     x_c = x[0] + r_min
    # elif beam_place == 'ul':
    #     y_c = y[-1] - r_min
    #     x_c = x[0] + r_min
    # elif beam_place == 'ur':
    #     y_c = y[-1] - r_min
    #     x_c = x[-1] - r_min
    # else:
    #     raise Exception

    # e = Ellipse((x_c, y_c), e_height, e_width, angle=-30, edgecolor='black',
    #             facecolor='none', alpha=1)
    # ax.add_patch(e)
    # title = ax.set_title("My plot", fontsize='large')
    # colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
    # fig.colorbar(i, cax=colorbar_ax)
    # fig.show()

# TODO: how plot coordinates in mas for -10, 0, 10 mas... if using matshow?
def plot(image, x=None, y=None, blc=None, trc=None, clim=None, cmap=None,
         abs_levels=None, rel_levels=None, min_abs_level=None,
         min_rel_level=None, factor=2., plot_color=False, show_beam=False,
         beam_corner='ll'):
    """
    Plot image.

    :param x: (optional)
        Iterable of x-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param y: (optional)
        Iterable of y-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param blc: (optional)
        Iterable of two values for Bottom Left Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(1, 1)``. (default:
        ``None``)
    :param trc: (optional)
        Iterable of two values for Top Right Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(1, 1)``. (default:
        ``None``)
    :param clim: (optional)
        Iterable of limits for image values to display. If ``None`` the display
        all values. (default: ``None``)
    :param cmap: (optional)
        Colormap to use for plotting colors. Available color maps could be
        printed using ``sorted(m for m in plt.cm.datad if not
        m.endswith("_r"))`` where ``plt`` is imported ``matplotlib.pyplot``.
        For further details on plotting available colormaps see
        http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``factor``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``factor``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param factor: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param show_beam: (optional)
        Convertable to boolean. Should we plot beam in corner? (default:
        ``False``)
    :param beam_corner: (optional)
        Place (corner) where to plot beam on map. One of ('ll', 'lr', 'ul',
        'ur') where first letter means lower/upper and second - left/right.
        (default: ``ll'')

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

    pylab.show()


# TODO: Implement plotting w/o coordinates - in pixels. Use pixel numbers as
# coordinates.
def plot(contours=None, colors=None, vectors=None, vectors_values=None, x=None,
         y=None, blc=None, trc=None, cmap='hsv', abs_levels=None,
         rel_levels=None, min_abs_level=None, min_rel_level=None, factor=2.,
         show_beam=False, beam_corner='ll', beam=None):
    """
    Plot image(s).

    :param contours: (optional)
        Numpy 2D array (possibly masked) that should be plotted using contours.
    :param colors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using colors.
    :param vectors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using vectors.
    :param vectors_values: (optional)
        Numpy 2D array (possibly masked) that should be used as vector's lengths
        when plotting ``vectors`` array.
    :param x: (optional)
        Iterable of x-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param y: (optional)
        Iterable of y-coordinates. It's length must be comparable to that part
        of image to display. If ``None`` then don't plot coordinates - just
        pixel numbers. (default=``None``)
    :param blc: (optional)
        Iterable of two values for Bottom Left Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(1, 1)``. (default:
        ``None``)
    :param trc: (optional)
        Iterable of two values for Top Right Corner (in pixels). Must be in
        range ``[1, image_size]``. If ``None`` then use ``(image_size,
        image_size)``. (default: ``None``)
    :param cmap: (optional)
        Colormap to use for plotting colors. Available color maps could be
        printed using ``sorted(m for m in plt.cm.datad if not
        m.endswith("_r"))`` where ``plt`` is imported ``matplotlib.pyplot``.
        For further details on plotting available colormaps see
        http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html.
        (default: ``hsv``)
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``factor``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``factor``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param factor: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param show_beam: (optional)
        Convertable to boolean. Should we plot beam in corner? (default:
        ``False``)
    :param beam_corner: (optional)
        Place (corner) where to plot beam on map. One of ('ll', 'lr', 'ul',
        'ur') where first letter means lower/upper and second - left/right.
        (default: ``ll'')
    :param beam: (optional)
        If ``show_beam`` is True then ``beam`` should be iterable of major axis,
        minor axis [pix] and beam positional angle [deg]. If no coordinats are
        supplied then beam parameters must be in pixels.

    :note:
        ``blc`` & ``trc`` are AIPS-like (from 1 to ``imsize``). Internally
        converted to python-like zero-indexing. If none are specified then use
        default values. In that case all images must have the same shape.
    """
    pass


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

    def slice(self, pix1, pix2):
        """
        Method that returns slice of image along line.

        :param x1:
            Iterable of cordinates of first pixel.
        :param x2:
            Iterable of cordinates of second pixel.
        :return:
            Numpy array of image values for given slice.
        """
        length = int(round(np.hypot(pix2[0] - pix1[0], pix2[1] - pix1[1])))
        if pix2[0] < pix1[0]:
            x = np.linspace(pix2[0], pix1[0], length)[::-1]
        else:
            x = np.linspace(pix1[0], pix2[0], length)
        if pix2[1] < pix1[1]:
            y = np.linspace(pix2[1], pix1[1], length)[::-1]
        else:
            y = np.linspace(pix1[1], pix2[1], length)

        return self.image[v_round(x).astype(np.int), v_round(y).astype(np.int)]

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

    # Importing stuff
    import os
    from images import Images
    from from_fits import create_clean_image_from_fits_file

    abs_levels = [-0.0004]+[0.0004 * 2**(j) for j in range(15)]

    data_dir = '/home/ilya/vlbi_errors/0148+274/2007_03_01/'
    i_dir_c1 = data_dir + 'C1/im/I/'
    q_dir_c1 = data_dir + 'C1/im/Q/'
    u_dir_c1 = data_dir + 'C1/im/U/'
    print "Creating PANG image..."
    images = Images()
    images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
                                 os.path.join(u_dir_c1, 'cc.fits')])
    pang_image = images.create_pang_images()[0]

    print "Creating PPOL image..."
    images = Images()
    images.add_from_fits(fnames=[os.path.join(q_dir_c1, 'cc.fits'),
                                 os.path.join(u_dir_c1, 'cc.fits')])
    ppol_image = images.create_pol_images()[0]

    print "Creating I image..."
    i_image = create_clean_image_from_fits_file(os.path.join(i_dir_c1,
                                                             'cc.fits'))
    print "Creating FPOL image..."
    images = Images()
    images.add_from_fits(fnames=[os.path.join(i_dir_c1, 'cc.fits'),
                                 os.path.join(q_dir_c1, 'cc.fits'),
                                 os.path.join(u_dir_c1, 'cc.fits')])
    fpol_image = images.create_fpol_images()[0]

    # plot(image, x=None, y=None, blc=None, trc=None, clim=None, cmap=None,
    #      abs_levels=None, rel_levels=None, min_abs_level=None,
    #      min_rel_level=None, factor=2., plot_color=False, show_beam=False):
    plot(i_image.image_w_residuals, x=i_image.x, y=i_image.y, blc=(230, 230),
         trc=(400, 400), abs_levels=abs_levels)
    # plot(contour_array, color_array, vec_array)
