import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def gaussian(height, x0, y0, bmaj, e=0.3, bpa=0, size_x=100):
    """
    Returns a gaussian function with the given parameters.

    :example:
    create grid:
        x, y = np.meshgrid(x, y)
        imshow(gaussian(x, y))

    """
    x_, y_ = np.mgrid[0: size_x, 0: size_x]
    bmin = bmaj * e
    # This brings PA to VLBI-convention (- = from NOrth counterclocwise)
    bpa = -bpa
    a = math.cos(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.sin(bpa) ** 2. / (2. * bmin ** 2.)
    b = math.sin(2. * bpa) / (2. * bmaj ** 2.) - \
        math.sin(2. * bpa) / (2. * bmin ** 2.)
    c = math.sin(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.cos(bpa) ** 2. / (2. * bmin ** 2.)
    func = lambda x, y: height * np.exp(-(a * (x - x0) ** 2 +
                                          b * (x - x0) * (y - y0) +
                                          c * (y - y0) ** 2))
    return func(x_, y_)


if __name__ == '__main__':

    # Plot every k-th pol.vector
    k = 3
    blc = (240, 200)
    trc = (300, 350)
    # Pixels
    x_center = blc[1] + (trc[1] - blc[1]) / 2. - 256
    y_center = blc[0] + (trc[0] - blc[0]) / 2. - 256
    # Pixsels
    x_slice = slice(blc[1], trc[1], None)
    y_slice = slice(blc[0], trc[0],  None)
    # Create picture with contours - I, color - fpol and vectors - direction and
    # value of Linear Polarization
    import os
    from images import Images
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
    from from_fits import create_clean_image_from_fits_file
    i_image = create_clean_image_from_fits_file(os.path.join(i_dir_c1,
                                                             'cc.fits'))
    print "Creating FPOL image..."
    images = Images()
    images.add_from_fits(fnames=[os.path.join(i_dir_c1, 'cc.fits'),
                                 os.path.join(q_dir_c1, 'cc.fits'),
                                 os.path.join(u_dir_c1, 'cc.fits')])
    fpol_image = images.create_fpol_images()[0]

    beam_place = 'ul'
    pixsize = abs(i_image.pixsize[0])
    imsize_x = x_slice.stop - x_slice.start
    imsize_y = y_slice.stop - y_slice.start
    factor = 206264806.719150
    # mas
    arc_length_x = pixsize * imsize_x * factor
    arc_length_y = pixsize * imsize_y * factor
    # mas
    x_ = i_image.x[0, :][x_slice] * factor
    y_ = i_image.y[:, 0][y_slice] * factor
    u = ppol_image.image[x_slice, y_slice] * np.cos(pang_image.image[x_slice,
                                                    y_slice])
    v = ppol_image.image[x_slice, y_slice] * np.sin(pang_image.image[x_slice,
                                                                     y_slice])
    # arc_length = pixsize * imsize * factor
    # x = y = np.linspace(-arc_length/2, arc_length/2, imsize)
    # FIXME: wrong zero location
    x = np.linspace(x_[0], x_[-1], imsize_x)
    y = np.linspace(y_[0], y_[-1], imsize_y)
    i_array = i_image.image_w_residuals[x_slice, y_slice]
    ppol_array = ppol_image.image[x_slice, y_slice]
    fpol_array = fpol_image.image[x_slice, y_slice]
    # Creating masks
    i_mask = np.zeros((imsize_x, imsize_y))
    ppol_mask = np.zeros((imsize_x, imsize_y))
    # i_mask[abs(i_array) < 0.0001] = 1
    ppol_mask[ppol_array < 0.003] = 1
    fpol_mask = np.logical_or(i_mask, ppol_mask)
    # Masking data
    i_array_masked = np.ma.array(i_array, mask=i_mask)
    fpol_array_masked = np.ma.array(fpol_array, mask=ppol_mask)
    ppol_array_masked = np.ma.array(ppol_array, mask=ppol_mask)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    # FIXME: wrong zero location
    i = ax.imshow(fpol_array_masked, interpolation='none', aspect='auto',
                  label='FPOL', extent=[y[0], y[-1], x[0], x[-1]], origin='lower')
    co = ax.contour(y, x, i_array_masked, colors='k', label='I')
    m = np.zeros(u.shape)
    u = np.ma.array(u, mask=ppol_mask)
    v = np.ma.array(v, mask=ppol_mask)
    vec = ax.quiver(y[::k], x[::k], u[::k, ::k], v[::k, ::k],
                    angles='uv', units='xy', headwidth=0, headlength=0,
                    headaxislength=0, scale=0.005, width=0.05)
    # Doesn't show anything
    ax.legend()
    # c = Circle((5, 5), radius=4,
    #            edgecolor='red', facecolor='blue', alpha=0.5)
    e_height = 10 * pixsize * factor
    e_width = 5 * pixsize * factor
    r_min = e_height / 2
    if beam_place == 'lr':
        y_c = y[0] + r_min
        x_c = x[-1] - r_min
    elif beam_place == 'll':
        y_c = y[0] + r_min
        x_c = x[0] + r_min
    elif beam_place == 'ul':
        y_c = y[-1] - r_min
        x_c = x[0] + r_min
    elif beam_place == 'ur':
        y_c = y[-1] - r_min
        x_c = x[-1] - r_min
    else:
        raise Exception

    e = Ellipse((x_c, y_c), e_height, e_width, angle=-30, edgecolor='black',
                facecolor='none', alpha=1)
    ax.add_patch(e)
    title = ax.set_title("My plot", fontsize='large')
    colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
    fig.colorbar(i, cax=colorbar_ax)
    fig.show()


