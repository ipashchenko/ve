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
    imsize = i_image.imsize[0]
    factor = 206264806.719150
    arc_length = pixsize * imsize * factor
    x_, y_ = np.mgrid[-imsize/2: imsize/2, -imsize/2: imsize/2]
    x_ *= pixsize * factor
    y_ *= pixsize * factor
    u = ppol_image.image * np.cos(pang_image.image)
    v = ppol_image.image * np.sin(pang_image.image)
    arc_length = pixsize * imsize * factor
    x = y = np.linspace(-arc_length/2, arc_length/2, imsize)
    i_array = i_image.image_w_residuals
    ppol_array = ppol_image.image
    fpol_array = fpol_image.image
    # Creating masks
    i_mask = np.zeros((imsize, imsize))
    ppol_mask = np.zeros((imsize, imsize))
    # i_mask[abs(i_array) < 0.0001] = 1
    ppol_mask[ppol_array < 0.003] = 1
    fpol_mask = np.logical_or(i_mask, ppol_mask)
    # Masking data
    i_array_masked = np.ma.array(i_array, mask=i_mask)
    fpol_array_masked = np.ma.array(fpol_array, mask=ppol_mask)
    ppol_array_masked = np.ma.array(ppol_array, mask=ppol_mask)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    i = ax.imshow(fpol_array_masked, interpolation='none', aspect='auto',
                  label='FPOL', extent=[-arc_length/2, arc_length/2,
                                        -arc_length/2, arc_length/2],
                  origin='lower')
    co = ax.contour(x, y, i_array_masked, colors='k', label='I')
    m = np.zeros(u.shape)
    u = np.ma.array(u, mask=ppol_mask)
    v = np.ma.array(v, mask=ppol_mask)
    vec = ax.quiver(x[::2], y[::2], u[::2, ::2], v[::2, ::2],
                    angles='uv', units='xy', headwidth=0, headlength=0,
                    headaxislength=0, scale=0.005, width=0.05)
    # Doesn't show anything
    ax.legend()
    # c = Circle((5, 5), radius=4,
    #            edgecolor='red', facecolor='blue', alpha=0.5)
    e_height = 10
    e_width = 5
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


