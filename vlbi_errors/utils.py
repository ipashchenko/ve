from itertools import combinations
import re
import math
import numpy as np
import string
from math import floor
from scipy import optimize


vcomplex = np.vectorize(complex)
v_int = np.vectorize(int)
v_round = np.vectorize(round)

mas_to_rad = 4.8481368 * 1E-09
degree_to_rad = 0.01745329
degree_to_mas = 36. * 10 ** 5


class AbsentHduExtensionError(Exception):
    pass


class AbsentVersionOfBinTableError(Exception):
    pass


class EmptyImageFtError(Exception):
    pass

# TODO: convert utils to using arrays instead of lists




# numpy.lib.recfunctions.append_fields
def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.
    http://stackoverflow.com/questions/1201817/adding-a-field-to-a-structured-numpy-array

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError, "`A' must be a structured numpy array"
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


def index_of(ar1, ar2, issubset=True):
    """
    Find indexes of elements of 1d-numpy arrays ar1 in ar2.

    Output:

        list (len = len(ar1)) of arrays with indexes of elements in ar2
        corresponding to current (list[i] -> ar1[i]) element of ar1. If no
        elements are found then i-th elementh of list is None.
    """

    if issubset:
        # assert that all elements of ar1 are in ar2
        assert np.all(np.intersect1d(ar2, ar1) == np.sort(ar1))
        # assert np.all(np.in1d(ar1, ar2))

    indxs_ar2_sorted = np.argsort(ar2)
    ar1_pos_left = np.searchsorted(ar2[indxs_ar2_sorted], ar1, side='left')
    ar1_pos_right = np.searchsorted(ar2[indxs_ar2_sorted], ar1, side='right')

    indxs = list()
    for i in range(len(ar1_pos_left)):
        indxs.append(range(ar1_pos_left[i], ar1_pos_right[i]))

    # indxs = sum(indxs, [])
    result = list()
    for indx in indxs:
        if indx:
            result.append(indxs_ar2_sorted[indx])
        else:
            result.append(None)

    # return indxs_ar2_sorted[indxs]
    return result


def _to_complex_array(struct_array, real_name, imag_name):
    """
    Method that takes structured array and names of 2 fields and returns
    complex numpy.ndarray.
    """

    assert(np.shape(struct_array[real_name]) ==
           np.shape(struct_array[imag_name]))

    return struct_array[real_name] + 1j * struct_array[imag_name]


def _to_one_ndarray(struct_array, *names):
    """
    Method that takes structured array and names of 2 (or more) fields and
    returns numpy.ndarray with expanded shape. Field can be 2-dim array.
    """

    # TODO: can i use struct_array[[name1, name2]] synthax? Yes but you'll get
    # structured array with this 2 fields.

    l = list()
    for name in names:
        if struct_array[name].ndim == 1:
            l.append(struct_array[name][:, None])
        elif struct_array[name].ndim == 2:
            l.extend(np.hsplit(struct_array[name], struct_array[name].shape[1]))

    return np.hstack(l)
    # return np.vstack([struct_array[name] for name in names]).T


def change_shape(_array, _dict1, _dict2):
    """
    Function that takes ndarray and 2 dictionaries with array's shape and
    permuted shape and returns array with permuted shape.

    Inputs:
        _array [numpy.ndarray] - array to change,
        dict1 - shape of array, that will be changed,
        dict2 - dictionary of new shape.
        dict2 can contain more keys then dict1. Only keys in dict2 that are in
            dict1 influence new shape. If dict1 contains some keys that are not
            in dict2, then position of such axes will be changed by other axis
            that contained in both dict1 and dict2.
    """

    dict1 = _dict1.copy()
    dict2 = _dict2.copy()
    array = _array.copy()

    for key in dict1:
        print "check " + str(key)
        if key not in dict2:
            # Don't alter position of this dimension directly (but it could
            # change it's position because of other dimensions).
            pass
        else:
            if not dict1[key] == dict2[key]:
                print "positin of axis " + str(key) + " has changed"
                print "from " + str(dict1[key]) + " to " + str(dict2[key])
                array = np.swapaxes(array, dict1[key], dict2[key])
                # Updated values for 2 changed keys in dict1
                dict1[key] = dict2[key]
                for item in dict1.items():
                    # If found other key in dict1 with the same value
                    if (item[1] == dict2[key]) and (item[0] != key):
                        print "Found item : " + str(item)
                        dict1[item[0]] = dict1[key]
                print "Updated dict1 is :"
                print dict1

    # Assert that altered dict1 (it's part with shapes from dict2) coincide
    # with dict2
    for key in dict2:
        if key in dict1:
            assert(dict1[key] == dict2[key])

    return array


# TODO: if ``min`` or ``max`` key is absent then only upper/lower bound does
# exist.
def get_indxs_from_struct_array(array, **kwargs):
    """
    Function that given structured array ``array`` and specified fields and
    conditions in ``kwargs`` argument returns corresponding indexes.

    Inputs:

        array - numpy structured array,

        kwargs - keyword arguments that specifies conditions:

            {field: value}

            If value is dictionary then use ``min`` & ``max`` keys to bound,
            if value is iterable then use its content.

    Output:

        numpy.array of indexes of ``array``.

    Example:
        >>>get_indxs_from_struct_array(array, time={'min': None, 'max': 0.5},
                                       baseline = [515, 517])
    """

    pass


def aips_bintable_fortran_fields_to_dtype_conversion(aips_type):
    """Given AIPS fortran format of binary table (BT) fields, returns
    corresponding numpy dtype format and shape. Examples:
    4J => array of 4 32bit integers,
    E(4,32) => two dimensional array with 4 columns and 32 rows.
    """

    intv = np.vectorize(int)
    aips_char = None
    dtype_char = None
    repeat = None
    _shape = None

    format_dict = {'L': 'bool', 'I': '>i2', 'J': '>i4', 'A': 'S',  'E': '>f4',
                   'D': '>f8'}

    for key in format_dict.keys():
        if key in aips_type:
            aips_char = key

    if not aips_char:
        raise Exception("aips data format reading problem " + str(aips_type))

    try:
        dtype_char = format_dict[aips_char]
    except KeyError:
        raise Exception("no dtype counterpart for aips data format" +
                        str(aips_char))

    try:
        repeat = int(re.search(r"^(\d+)" + aips_char,
                     aips_type).groups()[0])
        if aips_char is 'A':
            dtype_char = str(repeat) + dtype_char
            repeat = 1
    except AttributeError:
        repeat = None

    if repeat is None:
        _shape = tuple(intv(string.split(re.search(r"^" + aips_char +
                                                   "\((.+)\)$",
                                                   aips_type).groups()[0],
                                         sep=',')))
    else:
        _shape = repeat

    return dtype_char, _shape


def build_dtype_for_bintable_data(header):
    """Builds dtype for recarray from header.
    """

    # substitue = {'UV--SIN': 'u', 'VV--SIN': 'v', 'WW--SIN': 'w', 'BASELINE':
    # 'bl', 'DATE': 't'}
    # assert(header_dict['EXTNAME'] == 'UV_DATA')

    # # # of axis. 2 => matrix
    # naxis = int(header['NAXIS'])
    # # # of fields in a item
    tfields = int(header['TFIELDS'])
    # # # of Bytes in a item (sum of length of tfields elements)
    # naxis1 = int(header['NAXIS1'])
    # # # of items
    # naxis2 = int(header['NAXIS2'])
    # nrecords = naxis2

    # parameters of regular data matrix if in UV_DATA table
    try:
        maxis = int(header['MAXIS'])
    except KeyError:
        print "non UV_DATA"
        maxis = None

    # build np.dtype format
    names = []
    formats = []
    shapes = []
    tuple_shape = []
    array_names = []

    for i in range(1, tfields + 1):
        name = header['TTYPE' + str(i)]
        if name in names:
            name = name * 2
        names.append(name)
        _format, _shape = \
            aips_bintable_fortran_fields_to_dtype_conversion(header['TFORM' +
                                                                    str(i)])

        # building format & names for regular data matrix
        if name == 'FLUX' and maxis is not None:
            for i in range(1, maxis + 1):
                maxisi = int(header['MAXIS' + str(i)])
                if maxisi > 1:
                    tuple_shape.append(int(header['MAXIS' + str(i)]))
                    array_names.append(header['CTYPE' + str(i)])
            formats.append('>f4')
            shapes.append(tuple(tuple_shape))
            array_names = array_names
        else:
            formats.append(_format)
            shapes.append(_shape)

    print names, formats, shapes, array_names

    dtype_builder = zip(names, formats, shapes)
    dtype = [(name, _format, shape) for (name, _format, shape) in dtype_builder]

    return dtype, array_names


def baselines_2_ants(baselines):
    """Given list of baseline numbers (fits keyword) returns list of
    corresponding antennas.
    """
    # TODO: CHECK IF OUTPUT/INPUT IS OK!!!
    for baseline in baselines:
        baseline = abs(baseline)
        assert(baseline > 256)

    ants = list()
    for baseline in baselines:
        baseline = abs(baseline)
        ant1 = int(baseline // 256)
        ant2 = int(baseline - ant1 * 256)
        if ant1 * 256 + ant2 != baseline:
            continue
        ants.append(ant1)
        ants.append(ant2)
    ants = list(set(ants))
    ants.sort()

    return ants


# def ants_2_baselines(ants):
#     """Given several antennas returns corresponding baselines.
#     """
#
#     baselines = list()
#     ants_by2 = list(permutations(ants, 2))
#     for ant in ants_by2:
#         baseline = 256 * ant[0] + ant[1]
#         baselines.append(baseline)
#     return baselines


def ant_2_containing_baslines(ant, antennas):
    """
    Given antenna returns list of all baselines among given list with that
    antenna.
    """

    baselines = list()
    for antenna in antennas:
        if antenna < ant:
            baselines.append(256 * antenna + ant)
        elif antenna > ant:
            baselines.append(256 * ant + antenna)
        else:
            pass

    return baselines


def ants_2_baselines(ants):
    baselines = list()
    for ant in ants:
        baselines.extend(ant_2_containing_baslines(ant, ants))
    baselines = list(set(baselines))
    baselines = baselines.sort()

    return baselines

def get_triangles(antenna, antennas=None):
    """
    Find triangles of antennas.
    :param antenna:
    Number of antenna to build triangles with.
    :param antennas:
    Iterable of antenna numbers to build triangles with.
    :return:
    Dictionary with keys - ijk of antenna numbers and values - lists of
    3 baseline numbers.
    """
    if antennas is None:
        raise Exception("Provide some antenna num. for antennas!")
    else:
        baselines_list = list()
        assert (len(antennas) >= 2), "Need > 2 antennas for triangle!"
        # antennas must be iterable
        baselines_list.extend(list(antennas))

    # Assert that we don't have the same antennas in ``antennas`` and
    # ``antennas`` keywords
    if len(baselines_list) == 2:
        assert antenna not in baselines_list, "Need 3 diff. antennas!"
    else:
        if antenna in baselines_list:
            baselines_list.remove(antenna)

    # Find triangles (combinations of 3 antenna numbers)
    triangles = list()
    ant_numbers = [antenna] + baselines_list
    for comb in combinations(ant_numbers, 3):
        if comb[0] == antenna:
            triangles.append(comb)

    # Convert to baseline numbers
    triangle_baselines = dict()
    for triangle in triangles:
        i, j, k = sorted(triangle)
        triangle_baselines.update({str(i) + str(j) + str(k): [j + 256 * i,
                                                              k + 256 * i,
                                                              k + 256 * j]})

    return triangle_baselines

def time_frac_to_dhms(fractime):
    """Converts time in fraction of the day format to time in d:h:m:s
    format."""

    dhms = list()

    for time in fractime:
        day = int(floor(time))
        hour = int(floor(time * 24.))
        minute = int(floor(time * 1440. - hour * 60.))
        second = int(floor(time * 86400. - hour * 3600. - minute * 60.))
        dhms.append(tuple([day, hour, minute, second]))

    return dhms


def time_dhms_to_frac(dhmses):
    """Converts time in format d:h:m:s to time in parameters format =
    fraction of the day.
    """
    fractions = list()

    for dhms in dhmses:
        day, hour, minute, second = dhms
        fraction = day + hour / 24.0 + minute / (24.0 * 60.0) + \
            second / (24.0 * 60.0 * 60.0)
        fractions.append(fraction)

    return fractions


def gaussianBeam(size_x, bmaj, bmin, bpa, size_y=None):
    """
    Generate and return a 2D Gaussian function
    of dimensions (size_x,size_y).

    See Briggs PhD (Appendix B) for details.

    :param size_x:
        Size of first dimension [pixels].
    :param bmaj:
        Beam major axis size [pixels].
    :param bmin:
        Beam minor axis size [pixels].
    :param bpa:
        Beam positional angle [deg].
    :param size_y (optional):
        Size of second dimension. Default is ``size_x``.
    :return:
        Numpy.ndarray of size (``size_x``, ``size_y``,).
    """
    size_y = size_y or size_x
    x, y = np.mgrid[-size_x: size_x + 1, -size_y: size_y + 1]
    # Constructing parameters of gaussian from ``bmaj``, ``bmin``, ``bpa``.
    a0 = 1. / (0.5 * bmaj) ** 2.
    c0 = 1. / (0.5 * bmin) ** 2.
    theta = math.pi * (bpa + 90.) / 180.
    a = math.log(2) * (a0 * math.cos(theta) ** 2. +
                       c0 * math.sin(theta) ** 2.)
    b = (-(c0 - a0) * math.sin(2. * theta)) * math.log(2.)
    c = math.log(2) * (a0 * math.sin(theta) ** 2. +
                       c0 * math.cos(theta) ** 2.)

    g = np.exp(-a * x ** 2. - b * x * y - c * y ** 2.)
    # FIXME: It is already normalized?
    #return g/g.sum()
    return g


def infer_gaussian(data):
    """
    Return (amplitude, x_0, y_0, width), where width - rough estimate of
    gaussian width
    """
    amplitude = data.max()
    x_0, y_0 = np.where(data == amplitude)
    row = data[x_0, :]
    column = data[:, y_0]
    x_0 = float(x_0)
    y_0 = float(y_0)
    dx = len(np.where(row - amplitude/2 > 0)[0])
    dy = len(np.where(column - amplitude/2 > 0)[0])
    width = math.sqrt(dx ** 2. + dy ** 2.)

    return amplitude, x_0, y_0, width


def gaussian(height, x0, y0, bmaj, e, bpa):
    """
    Returns a gaussian function with the given parameters.

    :example:
    create grid:
        x, y = np.meshgrid(x, y)
        imshow(gaussian(x, y))

    """
    bmin = bmaj * e
    a = math.cos(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.sin(bpa) ** 2. / (2. * bmin ** 2.)
    b = math.sin(2. * bpa) / (2. * bmaj ** 2.) - \
        math.sin(2. * bpa) / (2. * bmin ** 2.)
    c = math.sin(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.cos(bpa) ** 2. / (2. * bmin ** 2.)
    return lambda x, y: height * np.exp(-(a * (x - x0) ** 2 +
                                          b * (x - x0) * (y - y0) +
                                          c * (y - y0) ** 2))


def fitgaussian(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit.
    """
    # Calculate initial values of circular gaussian + dummy params for
    # ellipticity
    params = list(infer_gaussian(data)) + [1., 0.]
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def mask_region(data, region):
    """
    Function that maskes 2D numpy array.
    :param data:
        2D numpy array.
    :param region:
        Tuple (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
        None,).
    :return:
        Masked 2D numpy array.
    """
    if region[3] is None:
        # Creating a disc shaped mask with radius r
        a, b = region[0], region[1]
        n = min(data.shape)
        r = region[2]
        y, x = np.ogrid[-a: n - a, -b: n - b]
        mask = x ** 2 + y ** 2 <= r ** 2
        masked_array = np.ma.array(data, mask=mask)

    else:
        # Creating rectangular mask
        y, x = np.ogrid[0: data.shape[0], 0: data.shape[1]]
        mask = (region[0] < x) & (x < region[2]) & (region[1] < y) & (y < region[3])
        masked_array = np.ma.array(data, mask=mask)

    return masked_array


def create_grid(imsize):
    """Create meshgrid of size ``imsize``.

        :param imsize:
            Container of image dimensions
        :return:
            Meshgrid of size (imsize[0], imsize[1])
    """
    xsize, ysize = imsize
    x = np.linspace(0, xsize - 1, xsize)
    y = np.linspace(0, ysize - 1, ysize)
    x, y = np.meshgrid(x, y)
    return (x, y,)


def find_close_regions(data, std_decrease_factor=1.1):
    """
    Function that finds entries of array with close elements (aka scans for time
    domain).
    :param data:
        1D numpy array with data.
    :return:
        list of lists with (first index, last index) of close regions.
    """
    maxs = np.diff(data)[np.argsort(np.diff(data))[::-1]]
    i = 0
    while np.std(maxs[i:])/np.std(maxs[i+1:]) > std_decrease_factor:
        i += 1
    threshold = maxs[i-1]
    borders = np.where((data[1:] - data[:-1]) > maxs[i-1])[0]
    print len(borders)
    regions_list = list()
    # Append first region
    regions_list.append([data[0], data[borders[0]]])
    # Append others
    for k in range(len(borders) - 1):
        regions_list.append([data[borders[k] + 1], data[borders[k+1]]])
    # Append last
    regions_list.append([data[borders[k+1] + 1], data[-1]])

    return regions_list


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    and ``kwargs``are also included.
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("vlbi_errors: Exception while calling your prior pdf:")
            print(" params:", x)
            print(" args:", self.args)
            print(" kwargs:", self.kwargs)
            print(" exception:")
            traceback.print_exc()
            raise


def ln_uniform(x, a, b):
    assert(a < b)
    if not a < x < b:
        return -np.inf
    return -math.log(b - a)


def is_sorted(lst):
    return (sorted(lst) == lst)


def get_uv_correlations(uv, models):
    """
    Function that accepts models of stokes parameters in image plane and returns
    cross-correlations (whatever possible) for given instance of ``UVData``
    class.

    :param uv:
        Numpy 2d-array of (u,v)-coordinates used for calculating correlations.
    :param models:
        Iterable of ``Model`` subclass instances. There should be only one (or
        zero) model for each stokes parameter. If there are two, say I-stokes
        models, then sum them firstly using ``Model.__add__``.
    :return:
        Dictionary with keys from 'RR', 'LL', 'RL', 'LR' and values - 1d numpy
        arrays with comlex values of visibilities. Length of array equals to
        number of (u,v)-points specified in argument (that is ``len(uv)``).
    """
    # Create dictionary of type {stokes/hands: model}
    model_dict = {'I': None, 'Q': None, 'U': None, 'V': None, 'RR': None,
                  'RL': None}
    model_dict.update({model.stokes: model for model in models})
    # Dictionary with keys - 'RR', 'LL', ... and values - correlations
    uv_correlations = dict()
    if model_dict['I'] or model_dict['V']:
        if model_dict['I'] and model_dict['V']:
            RR = model_dict['I'].ft(uv) + model_dict['V'].ft(uv)
            LL = model_dict['I'].ft(uv) - model_dict['V'].ft(uv)
        elif not model_dict['V'] and model_dict['I']:
            RR = model_dict['I'].ft(uv)
            LL = RR.copy()
        elif not model_dict['I'] and model_dict['V']:
            RR = model_dict['V'].ft(uv)
            LL = RR.copy()
        else:
            # Actually, we shouldn't get there
            raise EmptyImageFtError('Not enough data for RR&LL visibility'
                                    ' calculation')
        # Setting up parallel hands correlations
        uv_correlations.update({'RR': RR})
        uv_correlations.update({'LL': LL})

    else:
        if model_dict['RR'] or model_dict['LL']:
            RR = model_dict['RR'].ft(uv)
            LL = model_dict['LL'].ft(uv)
            # Setting up parallel hands correlations
            uv_correlations.update({'RR': RR})
            uv_correlations.update({'LL': LL})

    if model_dict['Q'] or model_dict['U']:
        if model_dict['Q'] and model_dict['U']:
            RL = model_dict['Q'].ft(uv) + 1j * model_dict['U'].ft(uv)
            LR = model_dict['Q'].ft(uv) - 1j * model_dict['U'].ft(uv)
            # RL = FT(Q + j*U)
            # LR = FT(Q - j*U)
            # Setting up cross hands correlations
            uv_correlations.update({'RL': RL})
            uv_correlations.update({'LR': LR})
        else:
            raise EmptyImageFtError('Not enough data for RL&LR visibility'
                                    ' calculation')

    return uv_correlations


if __name__ == '__main__':
    from uv_data import create_uvdata_from_fits_file
    from model import CCModel
    import os
    os.chdir('/home/ilya/code/vlbi_errors/data/Denise')
    # Load self-calinrated uv-data
    uvdata = create_uvdata_from_fits_file('1038+064.l22.2010_05_21.uvf')
    ccmodel = CCModel(stokes='I')
    # Load clean components model
    ccmodel.add_cc_from_fits('1038+064.l22.2010_05_21.icn.fits')
    uv = uvdata.uvw[:, :2]

