import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse


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

    beam_place = 'ul'
    pixsize = 2.908881604153578e-09
    imsize = 100
    factor = 206264806.719150
    x_, y_ = np.mgrid[-imsize/2: imsize/2, -imsize/2: imsize/2]
    x_ *= pixsize * factor
    y_ *= pixsize * factor
    u = np.ones((imsize, imsize)) * 3
    v = np.ones((imsize, imsize)) * 3
    arc_length = pixsize * imsize * factor
    x = y = np.linspace(-arc_length/2, arc_length/2, imsize)
    data = np.random.normal(0, 1, (100,100))
    data += gaussian(50, 60, 60, 40, 0.3, 0, size_x=imsize)
    data += gaussian(30, 40, 20, 10, 0.5, 0, size_x=imsize)
    mask = np.zeros((imsize, imsize))
    mask_ = np.zeros((imsize, imsize))
    mask[data < 10] = 1
    mask_[data < 3] = 1
    data_masked = np.ma.array(data, mask=mask)
    data_masked_ = np.ma.array(data, mask=mask_)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.6,0.8])
    i = ax.imshow(data_masked, interpolation='none', aspect='auto', label='RM',
                  extent=[-arc_length/2, arc_length/2, -arc_length/2,
                          arc_length/2], origin='lower')
    co = ax.contour(x, y, data_masked_, colors='k', label='I')
    m = np.zeros(u.shape)
    m[data < 10] = 1
    u = np.ma.array(u, mask=m)
    v = np.ma.array(v, mask=m)
    vec = ax.quiver(y_[::2], x_[::2], u[::2], v[::2],
                    angles='uv', units='xy', headwidth=0, headlength=0,
                    headaxislength=0)
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


