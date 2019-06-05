import numpy as np
import os
import tqdm
import glob
import shutil
from scipy import ndimage
from uv_data import UVData
from model import Model
# from cv_model import cv_difmap_models
from spydiff import (export_difmap_model, modelfit_difmap, import_difmap_model,
                     clean_difmap, append_component_to_difmap_model,
                     clean_n, difmap_model_flux,
                     sort_components_by_distance_from_cj,
                     component_joiner_serial)
from components import CGComponent, EGComponent
from from_fits import (create_clean_image_from_fits_file,
                       create_model_from_fits_file)
from utils import mas_to_rad, infer_gaussian
from image import plot as iplot
from image import find_bbox
from image_ops import rms_image
label_size = 14
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tarfile
from colorama import Fore, Back, Style


class FailedFindBestModelException(Exception):
    pass


class ChangeOfCoreModelException(Exception):
    pass


class StoppingIterationsCriterion(object):
    def __init__(self, mode="and"):
        self.files = list()
        self.mode = mode

    def check_criterion(self):
        """
        :return:
            Boolean - is criterion fulfilled?
        """
        raise NotImplementedError

    def is_applicable(self):
        raise NotImplementedError

    def do_stop(self, info):
        """
        :param info:
            Dictionary with results of current iteration.
        :return:
            Boolean - is criterion fulfilled?
        """
        self.files.append(info["model_file"])
        if self.is_applicable():
            return self.check_criterion()
        else:
            return False

    def clear(self):
        self.files = list()


class RChiSquaredStopping(StoppingIterationsCriterion):
    """
    Stop iterations when reduced chi-squared doesn't decreasing significantly
    anymore
    """
    def __init__(self, mode="or", delta_min=0.1):
        super(RChiSquaredStopping, self).__init__(mode=mode)
        self.delta_min = delta_min
        self.values = list()

    def is_applicable(self):
        return len(self.files) > 1

    def check_criterion(self):
        if self.values[-1] - self.values[-2] < self.delta_min:
            return True
        else:
            return False

    def do_stop(self, info):
        self.values.append(info["values"])
        print("ChiSquared = {}".format(info["values"]))
        super(RChiSquaredStopping, self).do_stop(info)


class AddedOverlappingComponentStopping(StoppingIterationsCriterion):
    """
    Added component overlaps with some other components.
    """
    def __init__(self, mode="or"):
        super(AddedOverlappingComponentStopping, self).__init__(mode=mode)

    def check_criterion(self):
        last_comps = import_difmap_model(self.files[-1])
        last_comp = last_comps[-1]
        distances = [np.hypot((comp.p[1]-last_comp.p[1]),
                              (comp.p[2]-last_comp.p[2])) for comp in
                     last_comps[:-1]]
        sizes = list()
        for comp in last_comps[:-1]:
            if len(comp) == 4:
                sizes.append(comp.p[3])
            elif len(comp) == 6:
                sizes.append(comp.p[3]*comp.p[4])
            else:
                raise Exception("Using only CG or EG components")
        ratios = [dist/(size/2 + last_comp.p[3]/2) for dist, size in
                  zip(distances, sizes)]
        return np.any(np.array(ratios) < 1.0)

    def is_applicable(self):
        return len(self.files) > 1


class AddedNegativeFluxComponentStopping(StoppingIterationsCriterion):
    """
    Added component overlaps with some other components.
    """
    def __init__(self, mode="or"):
        super(AddedNegativeFluxComponentStopping, self).__init__(mode=mode)

    def check_criterion(self):
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        comps = import_difmap_model(self.files[-1])
        return np.any([comp.p[0] < 0.0 for comp in comps])

    def is_applicable(self):
        return self.files


class AddedTheSameComponentStopping(StoppingIterationsCriterion):
    def __init__(self, mode="or"):
        super(AddedTheSameComponentStopping, self).__init__(mode=mode)

    def check_criterion(self):
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        comps = import_difmap_model(self.files[-1])
        return comps[-2] == comps[1]

    def is_applicable(self):
        return len(self.files) > 1


class ImageBasedStoppingCriterion(StoppingIterationsCriterion):
    def __init__(self, mode="and"):
        super(ImageBasedStoppingCriterion, self).__init__(mode=mode)
        self.ccimage = None

    def is_applicable(self):
        return self.files

    def set_ccimage(self, ccimage):
        self.ccimage = ccimage


class UVDataBasedStoppingCriterion(StoppingIterationsCriterion):
    def __init__(self, mode="and"):
        super(UVDataBasedStoppingCriterion, self).__init__(mode=mode)
        self.uvdata = None

    def is_applicable(self):
        return self.files

    def set_uvdata(self, uvdata):
        self.uvdata = uvdata


class TotalFluxStopping(ImageBasedStoppingCriterion):
    """
    Total flux of difmap model must be close to total flux of CC to stop.
    """
    def __init__(self, total_flux=None, abs_threshold=None,
                 rel_threshold=0.01, mode="and"):
        super(ImageBasedStoppingCriterion, self).__init__(mode=mode)
        self._total_flux = total_flux
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold

    @property
    def total_flux(self):
        if self._total_flux is None:
            self._total_flux = self.ccimage.total_flux
        return self._total_flux

    def check_criterion(self):
        threshold = self.abs_threshold or self.rel_threshold * self.total_flux
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        print(Style.DIM + "Last model has flux = {:.3f}"
                          " while CC total flux = {:.3f}".format(difmap_model_flux(self.files[-1]), self.total_flux) +
              Style.RESET_ALL)
        if difmap_model_flux(self.files[-1]) > self.total_flux:
            return True
        return abs(difmap_model_flux(self.files[-1]) -
                   self.total_flux) < threshold


class AddedComponentFluxLessRMSStopping(ImageBasedStoppingCriterion):
    """
    Last added component must have flux less the ``n_rms`` of image RMS to stop.
    """
    def __init__(self, n_rms=7.0, mode="and"):
        super(AddedComponentFluxLessRMSStopping, self).__init__(mode=mode)
        self.n_rms = n_rms
        self._threshold = None

    @property
    def threshold(self):
        if self._threshold is None:
            self._threshold = self.n_rms*rms_image(self.ccimage,
                                                   hovatta_factor=True)
        return self._threshold

    def check_criterion(self):
        _dir, _fn = os.path.split(self.files[-1])
        last_comp = import_difmap_model(_fn, _dir)[-1]
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        print(Style.DIM + "Last added component has flux = {:.4f}"
                          " while threshold = {:.4f}".format(last_comp.p[0], self.threshold) +
              Style.RESET_ALL)
        return last_comp.p[0] < self.threshold


class AddedComponentFluxLessRMSFluxStopping(ImageBasedStoppingCriterion):
    """
    Last added component must have flux less the it's area multiplyied by
    image RMS.
    """
    def __init__(self, mode="and"):
        super(AddedComponentFluxLessRMSFluxStopping, self).__init__(mode=mode)
        self._threshold = None
        self._beam_size = None

    @property
    def threshold(self):
        if self._threshold is None:
            self._threshold = rms_image(self.ccimage, hovatta_factor=True)
        return self._threshold

    @property
    def beam_size(self):
        """
        Beam area in pixels
        """
        if self._beam_size is None:
            self._beam_size = np.sqrt(self.ccimage.beam[0] * self.ccimage.beam[1])
        return self._beam_size

    def check_criterion(self):
        _dir, _fn = os.path.split(self.files[-1])
        last_comp = import_difmap_model(_fn, _dir)[-1]
        square_pixels = (np.hypot(self.beam_size, last_comp.p[3])/(2*self.ccimage.pixsize[0]/mas_to_rad))**2
        threshold = square_pixels*self.threshold
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        print(Style.DIM + "Last added component has flux = {:.4f}"
                          " while threshold = {:.4f}".format(last_comp.p[0], threshold) +
              Style.RESET_ALL)
        return last_comp.p[0] < threshold


class AddedTooDistantComponentStopping(ImageBasedStoppingCriterion):
    """
    Stopping when distant component is added. Distance is determined by
    specified RA & DEC ranges or by using area of image that is inside
    ``n_rms`` rectangular area. In last case image must be specified.
    """
    def __init__(self, n_rms=1.0, hovatta_factor=False, dec_range=None,
                 ra_range=None, mode="or"):
        super(AddedTooDistantComponentStopping, self).__init__(mode=mode)
        self.n_rms = n_rms
        self.hovatta_factor = hovatta_factor
        self._bbox = None
        self.dec_range = dec_range
        self.ra_range = ra_range

    def is_applicable(self):
        return self.files

    @property
    def bbox(self):
        if self._bbox is None:
            threshold = self.n_rms*rms_image(self.ccimage, self.hovatta_factor)
            blc, trc = find_bbox(self.ccimage.image, threshold)
            print(Style.DIM + "Calculating BLC, TRC in {}".format(self.__class__.__name__))
            print(blc, trc)
            print(Style.RESET_ALL)
            self._bbox = (blc, trc)
        return self._bbox

    def check_criterion(self):
        _dir, _fn = os.path.split(self.files[-1])
        last_comp = import_difmap_model(_fn, _dir)[-1]
        ra_mas, dec_mas = -last_comp.p[1], -last_comp.p[2]
        if self.ra_range is None and self.dec_range is None:
            blc, trc = self.bbox
            dec_range, ra_range = self.ccimage._convert_array_bbox(blc, trc)
        else:
            dec_range = self.dec_range
            ra_range = self.ra_range
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        print(Style.DIM + "Last added component located at "
                          "(dec,ra) = {:.2f}, {:.2f}"
                          " while BBOX DEC : {:.2f} to {:.2f},"
                          " RA : {:.2f} to {:.2f}".format(dec_mas, ra_mas,
                                                          dec_range[0],
                                                          dec_range[1],
                                                          ra_range[0],
                                                          ra_range[1]) +
              Style.RESET_ALL)
        return not last_comp.is_within_radec(ra_range, dec_range)


class AddedTooSmallComponentStopping(ImageBasedStoppingCriterion):
    """
    Last added component must be larger then specified threshold. Component
    must be dimmer then specified flux for this criterion to work.
    """
    def __init__(self, size_limit=0.001, flux_limit=0.1, mode="or"):
        super(AddedTooSmallComponentStopping, self).__init__(mode=mode)
        self.size_limit = size_limit
        self.flux_limit = flux_limit

    def is_applicable(self):
        return self.files

    def check_criterion(self):
        _dir, _fn = os.path.split(self.files[-1])
        last_comp = import_difmap_model(_fn, _dir)[-1]
        if last_comp.p[0] < self.flux_limit:
            return last_comp.p[3] < self.size_limit
        else:
            return False


class NLast(StoppingIterationsCriterion):
    """
    Abstract class defines criteria that need several iterations before starting
    to work.
    """
    def __init__(self, n_check, mode="and"):
        super(NLast, self).__init__(mode=mode)
        self.n_check = n_check

    def is_applicable(self):
        print("In NLast.is_applicable")
        if len(self.files) > self.n_check:
            print("Len files > n_check={} => applicable!".format(self.n_check))
            return True
        else:
            return False


class NLastDifferesFromLast(NLast):
    """
    Since last ``n_check`` iterations parameters of core component haven't
    changed.
    """
    def __init__(self, n_check=5, frac_flux_min=0.002, delta_flux_min=0.001,
                 delta_size_min=0.001, mode="and"):
        super(NLastDifferesFromLast, self).__init__(n_check, mode=mode)
        self.flux_min = None
        self.delta_flux_min = delta_flux_min
        self.frac_flux_min = frac_flux_min
        self.delta_size_min = delta_size_min

    def check_criterion(self):
        files = self.files[-self.n_check-1: -1]
        comps = list()
        for fn in files:
            core = import_difmap_model(fn)[0]
            comps.append(core)
        last_comp = comps[-1]
        last_flux = last_comp.p[0]
        last_size = last_comp.p[3]
        fluxes = np.array([comp.p[0] for comp in comps])
        sizes = np.array([comp.p[3] for comp in comps])
        self.flux_min = max(self.delta_flux_min, last_flux*self.frac_flux_min)
        delta_fluxes = abs(fluxes - last_flux)
        delta_sizes = abs(sizes - last_size)
        return np.alltrue(delta_fluxes < self.flux_min) or\
               np.alltrue(delta_sizes < self.delta_size_min)


class NLastDifferencesAreSmall(NLast):
    """
    Since last ``n_check`` iterations differences of core parameters are small
    enough.
    """
    def __init__(self, n_check=5, frac_flux_min=0.002, delta_flux_min=0.001,
                 delta_size_min=0.001, mode="and"):
        super(NLastDifferencesAreSmall, self).__init__(n_check, mode=mode)
        self.flux_min = None
        self.delta_flux_min = delta_flux_min
        self.frac_flux_min = frac_flux_min
        self.delta_size_min = delta_size_min

    def check_criterion(self):
        files = self.files[-self.n_check-1: -1]
        comps = list()
        for fn in files:
            core = import_difmap_model(fn)[0]
            comps.append(core)

        last_comp = comps[-1]
        last_flux = last_comp.p[0]

        fluxes = np.array([comp.p[0] for comp in comps])
        sizes = np.array([comp.p[3] for comp in comps])
        delta_fluxes = abs(fluxes[:-1]-fluxes[1:])
        delta_sizes = abs(sizes[:-1]-sizes[1:])
        self.flux_min = max(self.delta_flux_min, last_flux*self.frac_flux_min)
        return np.alltrue(delta_fluxes < self.flux_min) and\
               np.alltrue(delta_sizes < self.delta_size_min)


class NLastJustStop(NLast):
    """
    Just stop after specified number of iterations.
    """
    def __init__(self, n, mode="or"):
        super(NLastJustStop, self).__init__(n, mode=mode)
        self.n_stop = n+1
        print("NLastJustStop.init - n_stop = {}".format(n))

    def check_criterion(self):
        print("NLastJustStop.check_criterion => stopping now!")
        return True


class ModelSelector(object):
    """
    Basic class for selecting among several models.
    """

    def order_files(self, files):
        out_dir = os.path.split(files[0])[0]
        files = [os.path.split(file_path)[-1] for file_path in files]
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        files = [os.path.join(out_dir, file_) for file_ in files]
        return files

    def select(self, files):
        """
        Returns index (not number) of the best model in ``files`` list.
        """
        raise NotImplementedError


class FluxBasedModelSelector(ModelSelector):
    def __init__(self, frac_flux=0.01, delta_flux=0.001):
        self.frac_flux = frac_flux
        self.delta_flux = delta_flux

    def select(self, files):
        files = self.order_files(files)
        comps = list()
        for file_ in files:
            comps_ = import_difmap_model(file_)
            comps.append(comps_[0])
        fluxes = np.array([comp.p[0] for comp in comps])
        last_flux = fluxes[-1]
        fluxes_inv = fluxes[::-1]
        flux_min = min(self.delta_flux, self.frac_flux*last_flux)
        a = (abs(fluxes_inv - fluxes_inv[0]) < flux_min)[::-1]
        try:
            # This is index not number! Number is index + 1 (python 0-based
            # indexing)
            if np.count_nonzero(a) == 1:
                k = list(a.astype(np.int)).index(1)
            else:
                k = list(ndimage.binary_opening(a, structure=np.ones(2)).astype(np.int)).index(1)
        except ValueError:
            k = 0
        return k


class SizeBasedModelSelector(ModelSelector):
    def __init__(self, frac_size=0.01, delta_size=0.001,
                 small_size_of_the_core=0.001):
        self.frac_size = frac_size
        self.delta_size = delta_size
        self.small_size_of_the_core = small_size_of_the_core

    def select(self, files):
        files = self.order_files(files)
        comps = list()
        for file_ in files:
            comps_ = import_difmap_model(file_)
            comps.append(comps_[0])
        sizes = np.array([comp.p[3] for comp in comps])
        last_size = sizes[-1]
        sizes_inv = sizes[::-1]
        size_min = min(self.delta_size, self.frac_size * last_size)
        # If last model's core size is too small then not compare differences,
        # but compare only fraction changes
        if last_size < self.small_size_of_the_core:
            a = (abs((sizes_inv - sizes_inv[0]) / sizes_inv[0]) < self.frac_size)[::-1]
        else:
            a = (abs(sizes_inv - sizes_inv[0]) < size_min)[::-1]
        try:
            # This is index not number! Number is index + 1 (python 0-based
            # indexing)
            if np.count_nonzero(a) == 1:
                k = list(a.astype(np.int)).index(1)
            else:
                k = list(ndimage.binary_opening(a, structure=np.ones(2)).astype(np.int)).index(1)
        except ValueError:
            k = 0
        return k


class ModelFilter(object):
    """
    Basic class that filters models (e.g. discards models with very small
    components or components that are far away from source.
    """
    def order_files(self, files):
        out_dir = os.path.split(files[0])[0]
        files = [os.path.split(file_path)[-1] for file_path in files]
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        files = [os.path.join(out_dir, file_) for file_ in files]
        return files

    def do_filter(self, model_file):
        """Returns ``True`` if model specified in file ``model_file`` should be
        filtered out.
        """
        raise NotImplementedError


class SmallSizedComponentsModelFilter(ModelFilter):
    """
    Filter out models with small components that are dimmer then specified
    flux.
    """
    def __init__(self, small_size=10**(-3),
                 threshold_flux_small_sized_component=0.1):
        self.small_size = small_size
        self.threshold_flux_small_sized_component = threshold_flux_small_sized_component

    def do_filter(self, model_file):
        print(Style.DIM + "Checking {} in {}".format(os.path.basename(model_file),
                                                     self.__class__.__name__) +
              Style.RESET_ALL)
        comps = import_difmap_model(model_file)
        small_sizes = [comp.p[3] > self.small_size for comp in comps[1:]]
        fluxes_of_small_sized_components = [comp.p[0] for comp in comps[1:]
                                            if comp.p[3] < self.small_size]
        fluxes_of_small_sized_components =\
            [flux > self.threshold_flux_small_sized_component for flux in
             fluxes_of_small_sized_components]
        if not np.alltrue(small_sizes) and\
                not np.alltrue(fluxes_of_small_sized_components):
            print(Fore.RED + "Decreasing complexity because of too small"
                             " component(s) present" + Style.RESET_ALL)
            return True
        else:
            return False


class NegativeFluxComponentModelFilter(ModelFilter):
    def do_filter(self, model_file):
        print(Style.DIM + "Checking {} in {}".format(os.path.basename(model_file),
                                                     self.__class__.__name__) +
              Style.RESET_ALL)
        comps = import_difmap_model(model_file)
        negative_fluxes = [comp.p[0] < 0 for comp in comps]
        if np.any(negative_fluxes):
            print(Fore.RED + "Decreasing complexity because of too negative flux"
                             " component(s) present" + Style.RESET_ALL)
            return True
        else:
            return False


class ToElongatedCoreModelFilter(ModelFilter):
    def __init__(self, small_e=10**(-3)):
        self.small_e = small_e

    def do_filter(self, model_file):
        print(Style.DIM + "Checking {} in {}".format(os.path.basename(model_file),
                                                     self.__class__.__name__) +
              Style.RESET_ALL)
        core = import_difmap_model(model_file)[0]
        try:
            e = core.p[4]
        except IndexError:
            return False
        if e < self.small_e:
            print(Fore.RED +
                  "Decreasing complexity because of too elongated core" +
                  Style.RESET_ALL)
            return True
        else:
            return False


class ComponentAwayFromSourceModelFilter(ModelFilter):
    """
    Filters out models with distant component. Distance is determined by
    specified RA & DEC ranges or by using area of image that is inside
    ``n_rms`` rectangular area. In last case image must be specified.
    """
    def __init__(self, ccimage=None, cc_image_fits=None, n_rms=1,
                 hovatta_factor=False, ra_range=None, dec_range=None):
        if ccimage is None:
            if cc_image_fits is not None:
                ccimage = create_clean_image_from_fits_file(cc_image_fits)
        if ccimage is not None:
            threshold = n_rms*rms_image(ccimage, hovatta_factor)
            blc, trc = find_bbox(ccimage.image, threshold)
            dec_range, ra_range = ccimage._convert_array_bbox(blc, trc)

        self.ra_range = ra_range
        self.dec_range = dec_range

    def do_filter(self, model_file):
        print(Style.DIM + "Checking {} in {}".format(os.path.basename(model_file),
                                                     self.__class__.__name__) +
              Style.RESET_ALL)
        comps = import_difmap_model(model_file)
        do_comps_in_bbox = [comp.is_within_radec(self.ra_range, self.dec_range)
                            for comp in comps]
        if not np.alltrue(do_comps_in_bbox):
            print(Fore.RED +
                  "Decreasing complexity because of too distant component(s)"
                  " present" + Style.RESET_ALL)
            return True
        else:
            return False


class OverlappingComponentsModelFilter(ModelFilter):

    def do_filter(self, model_file):
        comps = import_difmap_model(model_file)
        do_any_overlap = list()
        for last_comp in comps:
            others = comps[:]
            others.remove(last_comp)
            distances = [np.hypot((comp.p[1]-last_comp.p[1]),
                                  (comp.p[2]-last_comp.p[2])) for comp in
                         others]
            sizes = list()
            for comp in others:
                if len(comp) == 4:
                    sizes.append(comp.p[3])
                elif len(comp) == 6:
                    sizes.append(comp.p[3]*comp.p[4])
                else:
                    raise Exception("Using only CG or EG components")
            ratios = [dist/(size/2 + last_comp.p[3]/2) for dist, size in
                      zip(distances, sizes)]
            do_any_overlap.append(np.any(np.array(ratios) < 1.0))
        if np.any(do_any_overlap):
            print(Fore.RED + "Decreasing complexity because of overlapping"
                             " component(s) present" + Style.RESET_ALL)
            return True
        else:
            return False


# TODO: Suggesting component by fitting residuals in uv-plane with difmap
# TODO: Determine image region with image convolved with some gaussian - to
# catch extended regions
# TODO: Find core using Sasha's algorithm when core is not first
class AutoModeler(object):
    def __init__(self, uv_fits_path, out_dir, path_to_script,
                 mapsize_clean=None, core_elliptic=False,
                 compute_CV=False, n_CV=5, n_rep_CV=1, n_comps_terminate=20,
                 niter_difmap=100, show_difmap_output_clean=False,
                 show_difmap_output_modelfit=False,
                 ra_range_plot=None, dec_range_plot=None,
                 merge_close_components=False):
        self.uv_fits_path = uv_fits_path
        self.uv_fits_dir, self.uv_fits_fname = os.path.split(uv_fits_path)
        if out_dir is None:
            self.out_dir = os.getcwd()
        else:
            self.out_dir = out_dir
        self.path_to_script = path_to_script
        self.compute_CV = compute_CV
        self.n_CV = n_CV
        self.n_rep_CV = n_rep_CV
        self.n_comps_terminate = n_comps_terminate
        print("Automodeler.init n_comps_terminate = {}".format(n_comps_terminate))
        if core_elliptic:
            self.core_type = 'eg'
        else:
            self.core_type = 'cg'

        if mapsize_clean is None:
            raise Exception("Indicate mapsize_clean!")
        else:
            self.mapsize_clean = mapsize_clean

        self.uvdata = UVData(uv_fits_path)
        self.uvdata_residuals = UVData(uv_fits_path)

        self.choose_stokes()

        self.freq_hz = self.uvdata.frequency

        self.niter_difmap = niter_difmap
        self.show_difmap_output_clean = show_difmap_output_clean
        self.show_difmap_output_modelfit = show_difmap_output_modelfit

        self.cv_scores = list()
        # ``CleanImage`` instance for CLEANed original uv data set
        self._ccimage = None
        self._beam = None
        # Path to original CLEAN image
        self._ccimage_path = os.path.join(self.out_dir, 'image_cc_orig.fits')
        # Path to image with residuals. It will be overrided each iteration
        self._ccimage_residuals_path = os.path.join(self.out_dir, 'image_cc_residuals.fits')
        # Total flux of all CC components in CLEAN model of the original uv data
        # set
        self._total_flux = None
        self._uv_residuals_fits_path = os.path.join(self.out_dir, "residuals.uvf")
        # Number of iterations passed
        self.counter = 0
        # Instance of ``Model`` class that represents current model
        self.model = None
        self._mdl_prefix = '{}_fitted'.format(self.core_type)
        self.fitted_model_paths = list()
        self.dec_range_plot = dec_range_plot
        self.ra_range_plot = ra_range_plot
        self.merge_close_components = merge_close_components
        self._init_single_comp_cg = os.path.join(self.out_dir, 'init_single_cg.mdl')
        cg = CGComponent(1, 0, 0, 0.1)
        export_difmap_model([cg], self._init_single_comp_cg, self.freq_hz)
        self._init_single_comp_eg = os.path.join(self.out_dir, 'init_single_eg.mdl')
        eg = EGComponent(1, 0, 0, 0.1, 1.0, 0.0)
        export_difmap_model([eg], self._init_single_comp_eg, self.freq_hz)

        # RMS near core before suggesting components
        self.rms_residuals = list()

        # ID of the best model
        self.best_id = None

    # @property
    # def files(self):
    #     return list(set(self.fitted_model_paths))

    @property
    def ccimage(self):
        if self._ccimage is None:
            print(Style.DIM + "CLEANing original uv data set" + Style.RESET_ALL)
            clean_difmap(self.uv_fits_fname, self._ccimage_path, self.stokes,
                         self.mapsize_clean, path=self.uv_fits_dir,
                         path_to_script=self.path_to_script,
                         outpath=self.out_dir,
                         show_difmap_output=self.show_difmap_output_clean)
            self._ccimage = create_clean_image_from_fits_file(self._ccimage_path)
        return self._ccimage

    @property
    def beam(self):
        if self._beam is None:
            self._beam = np.sqrt(self.ccimage.beam[0] * self.ccimage.beam[1])
        return self._beam

    @property
    def total_flux(self):
        if self._total_flux is None:
            self._total_flux = self.ccimage.total_flux
        return self._total_flux

    def create_residuals(self, model):
        """
        :param model: (optional)
            Instance of ``Model`` class. If ``None`` then treat original uv data
            set as residuals.
        """
        if model is not None:
            print(Style.DIM + "Creating residuals using " + Style.RESET_ALL +
                  "fitted model :")
            print(model)
            uvdata_ = UVData(self.uv_fits_path)
            uvdata_.substitute([model])
            uvdata_residual = self.uvdata - uvdata_
            self.uvdata_residuals = uvdata_residual
        else:
            print(Style.DIM + "Creating \"residuals\" from original data alone" +
                  Style.RESET_ALL)
            uvdata_residual = self.uvdata_residuals
        uvdata_residual.save(self._uv_residuals_fits_path, rewrite=True)

    def suggest_component(self, type='cg', bmaj_nan=0.1):
        """
        Suggest single circular gaussian component using self-calibrated uv-data
        FITS file.
        :param type: (optional)
            Type of component to suggest. Circular ("cg") or Elliptical ("eg")
             gaussian. (default: "cg")
        :param bmaj_nan: (optional)
            When estimated ``bmaj`` is NaN this value is used as ``bmaj`` [mas].
            (default: ``0.1``)

        :return:
            Instance of ``CGComponent``.
        """
        print(Style.DIM + "Suggesting component..." + Style.RESET_ALL)
        clean_difmap(self._uv_residuals_fits_path, self._ccimage_residuals_path,
                     self.stokes, self.mapsize_clean, path=self.out_dir,
                     path_to_script=self.path_to_script, outpath=self.out_dir)

        image = create_clean_image_from_fits_file(self._ccimage_residuals_path)

        imsize = image.imsize[0]
        mas_in_pix = abs(image.pixsize[0] / mas_to_rad)

        # RMS in core region
        beam = int(round(self.beam/mas_in_pix))
        print("imsize = {}, beam={}".format(imsize, beam))
        self.rms_residuals.append(np.std(image.image[int(imsize/2)-3*beam: int(imsize/2)+3*beam,
                                         int(imsize/2)-3*beam: int(imsize/2)+3*beam]))
        amp, y, x, bmaj = infer_gaussian(image.image)
        print("Suggested bmaj = {} pixels".format(bmaj))
        x = mas_in_pix * (x - imsize / 2) * np.sign(image.dx)
        y = mas_in_pix * (y - imsize / 2) * np.sign(image.dy)
        if np.isnan(bmaj):
            bmaj = bmaj_nan
        else:
            bmaj *= mas_in_pix
            bmaj = np.sqrt(bmaj ** 2 - beam ** 2)
        # If ``bmaj`` is less then beam than use ``bmaj_nan`` instead
        if np.isnan(bmaj):
            bmaj = bmaj_nan
        if type == 'cg':
            comp = CGComponent(amp, x, y, bmaj)
        elif type == 'eg':
            comp = EGComponent(amp, x, y, bmaj, 1.0, 0.0)
        else:
            raise Exception

        print(Style.DIM + "Suggested: {}".format(comp) + Style.RESET_ALL)
        return comp

    def suggest_component_uv(self, type='cg'):
        """
        Suggest single circular gaussian component using self-calibrated uv-data
        FITS file.

        This fits residuals between UV-data and current model in uv-plane using
        ``difmap`` with 1 component.

        :param type: (optional)
            Type of component to suggest. Circular ("cg") or Elliptical ("eg")
             gaussian. (default: "cg")

        :return:
            Instance of ``CGComponent``.
        """
        print(Style.DIM + "Suggesting component by fitting in UV-plane..." +
              Style.RESET_ALL)

        # Make last component as new suggested
        # If not first iteration
        if self.fitted_model_paths:
            last_comp = import_difmap_model(self.fitted_model_paths[-1])[-1]
            print("Using last component as single component for residuals")
            print("Parameters :", last_comp.p)
            uvw = self.uvdata_residuals.uvw
            uv_radius = np.sqrt(uvw[:, 0]**2+uvw[:, 1]**2)
            amp = np.hypot(self.uvdata_residuals.uvdata.real,
                           self.uvdata_residuals.uvdata.imag)
            amp = np.mean(amp[:, :, 0], axis=1)
            uv_dist = 0.2*np.max(uv_radius)
            amp = np.median(amp[uv_radius < uv_dist])
            last_comp.p[0] = amp
            last_comp.p[1] = 0
            last_comp.p[2] = 0
            print("Using amplitude : ", amp)
            export_difmap_model([last_comp], self._init_single_comp_cg,
                                self.freq_hz)
            init_model_file = self._init_single_comp_cg
        # If first iteration
        else:
            if self.core_type == "cg":
                init_model_file = self._init_single_comp_cg
            elif self.core_type == "eg":
                init_model_file = self._init_single_comp_eg
            else:
                raise Exception

        modelfit_difmap(self._uv_residuals_fits_path, init_model_file,
                        init_model_file,
                        path=self.uv_fits_dir, mdl_path=self.out_dir,
                        out_path=self.out_dir, niter=self.niter_difmap,
                        stokes=self.stokes,
                        show_difmap_output=self.show_difmap_output_modelfit)

        comp = import_difmap_model(init_model_file)[0]

        return comp

    def check_first_elliptic(self):
        # Check that first component is elliptic one and adjust model in case it
        # is not
        if self.counter > 1 and self.core_type == "eg":
            print(Fore.GREEN + "Checking if elliptic core goes first!" + Style.RESET_ALL)
            model_2check = os.path.join(self.out_dir,
                                        '{}_{}.mdl'.format(self._mdl_prefix,
                                                           self.counter))
            cj_sorted = sort_components_by_distance_from_cj(model_2check,
                                                            self.freq_hz,
                                                            n_check_for_core=1,
                                                            perc_distant=75)
            comps = import_difmap_model(model_2check, self.out_dir)
            ell_first = len(comps[0]) == 6
            if not ell_first:
                print(Back.RED + "Core has changed position!" + Style.RESET_ALL)
                comps = import_difmap_model(model_2check, self.out_dir)
                first_comp = comps[0]

                # Create new model with EG component first and CG all others
                new_comps = list()
                new_comps.append(first_comp.to_elliptic())
                for comp in comps[1:]:
                    new_comps.append(comp.to_circular())

                export_difmap_model(new_comps, model_2check,
                                    self.freq_hz)
                modelfit_difmap(self.uv_fits_fname,
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter),
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter),
                                path=self.uv_fits_dir, mdl_path=self.out_dir,
                                out_path=self.out_dir,
                                niter=self.niter_difmap, stokes=self.stokes,
                                show_difmap_output=self.show_difmap_output_modelfit)

    def check_merging(self):
        # Check if there any close circular gaussians that can be merged in one
        # elliptical component
        if self.counter > 1:
            print(Fore.GREEN + "Checking if components could be merged!" + Style.RESET_ALL)
            model_2check = os.path.join(self.out_dir,
                                        '{}_{}.mdl'.format(self._mdl_prefix,
                                                           self.counter))
            joined = component_joiner_serial(model_2check, self.beam, self.freq_hz)
            if joined:
                print(Fore.RED + "Merged components" + Style.RESET_ALL)
                modelfit_difmap(self.uv_fits_fname,
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter),
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter-1),
                                path=self.uv_fits_dir, mdl_path=self.out_dir,
                                out_path=self.out_dir,
                                niter=self.niter_difmap, stokes=self.stokes,
                                show_difmap_output=self.show_difmap_output_modelfit)
                self.counter -= 1

    def do_iteration(self):
        self.counter += 1
        self.create_residuals(self.model)
        if self.counter == 1:
            core_type = self.core_type
        else:
            core_type = 'cg'
            # core_type = self.core_type
        comp = self.suggest_component(core_type)
        # comp = self.suggest_component_uv(core_type)

        if self.counter > 1:
            # If this is not first iteration then append component to existing
            # file
            shutil.copy(os.path.join(self.out_dir, '{}_{}.mdl'.format(self._mdl_prefix, self.counter-1)),
                        os.path.join(self.out_dir, 'init_{}.mdl'.format(self.counter)))
            append_component_to_difmap_model(comp, os.path.join(self.out_dir, 'init_{}.mdl'.format(self.counter)),
                                             self.freq_hz)
        else:
            # If this is first iteration then create model file
            export_difmap_model([comp],
                                os.path.join(self.out_dir, 'init_{}.mdl'.format(self.counter)),
                                self.freq_hz)

        rchisq = modelfit_difmap(self.uv_fits_fname, 'init_{}.mdl'.format(self.counter),
                        '{}_{}.mdl'.format(self._mdl_prefix, self.counter),
                        path=self.uv_fits_dir, mdl_path=self.out_dir,
                        out_path=self.out_dir, niter=self.niter_difmap,
                        stokes=self.stokes,
                        show_difmap_output=self.show_difmap_output_modelfit)

        # Checks that alter model files
        self.check_first_elliptic()
        if self.merge_close_components:
            self.check_merging()

        # Update model and plot results of current iteration
        model = Model(stokes='I')
        comps = import_difmap_model('{}_{}.mdl'.format(self._mdl_prefix, self.counter), self.out_dir)
        try:
            plot_clean_image_and_components(self.ccimage, comps, n_rms_size=2.,
                                            outname=os.path.join(self.out_dir, "{}_image_{}.png".format(self._mdl_prefix, self.counter)),
                                            ra_range=self.ra_range_plot,
                                            dec_range=self.dec_range_plot)
        except TypeError:
            with open(os.path.join(self.out_dir, "{}_image_{}_NOPNG_TypeError.txt".format(self._mdl_prefix, self.counter)), "w") as fo:
                pass

        model.add_components(*comps)
        self.model = model

        result = {"values": rchisq, "model_file": os.path.join(self.out_dir, '{}_{}.mdl'.format(self._mdl_prefix, self.counter))}

        return result

    def clear(self):
        self.counter = 0
        self.fitted_model_paths = list()

    def run(self, stoppers, start_model_fname=None, refit_start_model=True):
        stoppers = list(stoppers)
        stoppers.append(NLastJustStop(self.n_comps_terminate))
        for stopper in stoppers:
            if isinstance(stopper, ImageBasedStoppingCriterion):
                stopper.set_ccimage(self.ccimage)
            elif isinstance(stopper, UVDataBasedStoppingCriterion):
                stopper.set_uvdata(self.uvdata)

        if start_model_fname is not None:
            print(Style.DIM + "Using model from {} as starting point".format(start_model_fname) +
                  Style.RESET_ALL)

            if refit_start_model:
                comps = import_difmap_model(start_model_fname)
                self.counter += len(comps)
                modelfit_difmap(self.uv_fits_fname, start_model_fname,
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter),
                                path=self.uv_fits_dir, mdl_path=self.out_dir,
                                out_path=self.out_dir, niter=self.niter_difmap,
                                stokes=self.stokes,
                                show_difmap_output=self.show_difmap_output_modelfit)
                # Update model and plot results of current iteration
                model = Model(stokes='I')
                comps = import_difmap_model(
                    '{}_{}.mdl'.format(self._mdl_prefix, self.counter),
                    self.out_dir)
                model.add_components(*comps)
                self.model = model

            else:
                comps = import_difmap_model(start_model_fname)
                model = Model(stokes=self.stokes)
                model.add_components(*comps)
                self.model = model

        while True:
            result = self.do_iteration()
            new_mdl_file = result["model_file"]
            # info = {"values": result["values"]}
            if new_mdl_file not in self.fitted_model_paths:
                self.fitted_model_paths.append(new_mdl_file)
            stoppers_and = [stopper for stopper in stoppers if
                            stopper.mode == "and"]
            stoppers_or = [stopper for stopper in stoppers if
                           stopper.mode == "or"]
            stoppers_while = [stopper for stopper in stoppers if
                              stopper.mode == "while"]
            decisions_and = [stopper.do_stop(result) for stopper in
                             stoppers_and]
            decisions_or = [stopper.do_stop(result) for stopper in
                            stoppers_or]
            decisions_while = [not stopper.do_stop(result) for stopper in
                               stoppers_while]
            # Thus go at least until e.g. model flux will be close to total flux
            if decisions_while:
                if np.any(~np.array(decisions_while)):
                    continue
            # decision = decisions_and + decisions_or
            # This fixes absence of decisions of one type (``np.alltrue([]) =
            # True``).
            if not decisions_and:
                decisions_and = [False]
            if not decisions_or:
                decisions_or = [False]
            do_stop = np.alltrue(decisions_and) or np.any(decisions_or)
            print(Back.GREEN + "Stopping criteria (AND):" +
                  Style.RESET_ALL)
            for stopper, decision in zip(stoppers_and, decisions_and):
                if decision:
                    print(Fore.RED + "{}".format(stopper.__class__.__name__) +
                          Style.RESET_ALL)
                else:
                    print(Fore.GREEN + "{}".format(stopper.__class__.__name__) +
                          Style.RESET_ALL)
            print(Back.GREEN + "Stopping criteria (OR):" + Style.RESET_ALL)
            for stopper, decision in zip(stoppers_or, decisions_or):
                if decision:
                    print(Fore.RED + "{}".format(stopper.__class__.__name__) +
                          Style.RESET_ALL)
                else:
                    print(Fore.GREEN + "{}".format(stopper.__class__.__name__) +
                          Style.RESET_ALL)

            if do_stop:
                break

        # FIXME: Finish this part
        # best_model_file = self.select_best()
        # self.archive_images()
        # self.archive_models()
        # self.clean()

    def select_best(self, selectors, filters):
        # Select best model using custom selectors
        files = self.fitted_model_paths
        files_toremove = self.files[:]
        if selectors:
            id_best = max(selector.select(files) for selector in selectors)
        else:
            id_best = len(files)-1
        files = files[:id_best+1]

        # Additionally filter
        for fn in files[::-1]:
            if np.any([flt.do_filter(fn) for flt in filters]):
                id_best -= 1
            else:
                break
        print("Best model is {}".format(files[id_best]))
        best_model = files[id_best]

        self.best_id = id_best

        return best_model

    def plot_results(self, id_best=None, best_model=None, stoppers_dict=None):
        """
        Plot picture with values of core parameter and other statistics
        collected in specified stoppers versus number of components in model.

        :param id_best: (optional)
            ID of the best model. If ``None`` then use ``best_model`` to specify
            it. (default: ``None``)
        :param best_model: (optional)
            Path to the best model. If ``None`` then use ``id_best`` to specify
            it. (default: ``None``)
        :param stoppers_dict: (optional)
            Dictionary with keys - names (e.g. "ChiSquared") and values -
            stopper that collect some statistics during modelling (e.g.
            ``RChiSquaredStopping``) - that has ``values`` container attribute.
        """

        if id_best is None:
            if best_model is None:
                raise Exception("Specify id_best or best_model argument")
            else:
                id_best = self.fitted_model_paths.index(best_model)

        if stoppers_dict is not None:
            n_additional = len(stoppers_dict)
        else:
            n_additional = 0

        cores = list()
        for file_ in self.fitted_model_paths:
            core = import_difmap_model(file_)[0]
            cores.append(core)
        if len(core) == 4:
            fig, axes = plt.subplots(2+n_additional, 1, sharex=True)
            axes[0].plot(range(1, len(cores) + 1),
                         [comp.p[0] for comp in cores])
            axes[0].plot(range(1, len(cores) + 1),
                         [comp.p[0] for comp in cores], '.k')
            axes[1].plot(range(1, len(cores) + 1),
                         [comp.p[3] for comp in cores])
            axes[1].plot(range(1, len(cores) + 1),
                         [comp.p[3] for comp in cores],
                         '.k')
            if stoppers_dict is not None:
                for i, (name, stopper) in zip(range(n_additional),
                                              stoppers_dict.items()):
                    axes[i+2].plot(range(2, len(cores)+1), stopper.values[1:])
                    axes[i+2].plot(range(2, len(cores)+1), stopper.values[1:], '.k')
                    axes[i+2].set_ylabel(name)
                    axes[i+2].axvline(id_best+1, color='r')

            # On the last axis
            axes[1+n_additional].set_xlabel("Number of components")
            axes[1+n_additional].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].set_ylabel("Flux, [Jy]")
            axes[1].set_ylabel("Size, [mas]")
            axes[0].axvline(id_best+1, color='r')
            axes[1].axvline(id_best+1, color='r')
        elif len(core) == 6:
            fig, axes = plt.subplots(3+n_additional, 1, sharex=True)
            axes[0].plot(range(1, len(cores) + 1),
                         [comp.p[0] for comp in cores])
            axes[0].plot(range(1, len(cores) + 1),
                         [comp.p[0] for comp in cores], '.k')
            axes[0].set_ylabel("Flux, [Jy]")
            axes[1].plot(range(1, len(cores) + 1),
                         [comp.p[3] for comp in cores])
            axes[1].plot(range(1, len(cores) + 1),
                         [comp.p[3] for comp in cores],
                         '.k')
            axes[1].set_ylabel("Size, [mas]")
            axes[2].plot(range(1, len(cores) + 1),
                         [comp.p[4] for comp in cores])
            axes[2].plot(range(1, len(cores) + 1),
                         [comp.p[4] for comp in cores], '.k')
            axes[2].set_ylabel("e")

            if stoppers_dict is not None:
                for i, (name, stopper) in zip(range(n_additional),
                                              stoppers_dict.items()):
                    axes[i+3].plot(range(2, len(cores)+1), stopper.values[1:])
                    axes[i+3].plot(range(2, len(cores)+1), stopper.values[1:], '.k')
                    axes[i+3].set_ylabel(name)
                    axes[i+3].axvline(id_best+1, color='r')

            axes[2+n_additional].set_xlabel("Number of components")
            axes[2+n_additional].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].axvline(id_best+1, color='r')
            axes[1].axvline(id_best+1, color='r')
            axes[2].axvline(id_best+1, color='r')

        fig.savefig(os.path.join(self.out_dir, '{}_core_parameters_vs_ncomps.png'.format(self._mdl_prefix)),
                    bbox_inches='tight', dpi=200)

        # fig, axes = plt.subplots(1, 1, sharex=True)
        # axes.plot(range(1, len(cores) + 1), [difmap_model_flux(fn) for
        #                                      fn in self.fitted_model_paths])
        # axes.plot(range(1, len(cores) + 1), [difmap_model_flux(fn) for
        #                                      fn in self.fitted_model_paths],
        #           '.k')
        # axes.set_ylabel("Total Flux, [Jy]")
        # axes.set_xlabel("Number of components")
        # axes.axvline(id_best+1, color='r')
        # axes.axhline(self.total_flux)
        # fig.savefig(os.path.join(self.out_dir, '{}_total_flux_vs_ncomps.png'.format(self._mdl_prefix)),
        #             bbox_inches='tight', dpi=200)

    def archive_models(self):
        with tarfile.open(os.path.join(self.out_dir, "{}_models.tar.gz".format(self._mdl_prefix)),
                          "w:gz") as tar:
            for fn in self.fitted_model_paths:
                tar.add(fn, arcname=os.path.split(fn)[-1])

    def archive_images(self):
        files = glob.glob(os.path.join(self.out_dir, "{}_image_*.png".format(self._mdl_prefix)))
        with tarfile.open(os.path.join(self.out_dir, "{}_images.tar.gz".format(self._mdl_prefix)),
                          "w:gz") as tar:
            for fn in files:
                tar.add(fn, arcname=os.path.split(fn)[-1])

    def clean(self):
        # Clean model files (we have copies in archive)
        for fn in self.fitted_model_paths:
            os.unlink(fn)
        # Clean images with components superimposed (we have copies in archive)
        files = glob.glob(os.path.join(self.out_dir, "{}_image_*.png".format(self._mdl_prefix)))
        for fn in files:
            os.unlink(fn)

    def choose_stokes(self):
        if self.uvdata._check_stokes_present('I'):
            self.stokes = 'I'
        elif self.uvdata._check_stokes_present('RR'):
            self.stokes = 'RR'
        elif self.uvdata._check_stokes_present('LL'):
            self.stokes = 'LL'
        else:
            raise Exception("No Stokes I, RR or LL in {}".format(self.uv_fits_fname))


def create_cc_model_uvf(uv_fits_path, mapsize_clean, path_to_script,
                        outname='image_cc_model.uvf', out_dir=None):
    """
    Function that creates uv-data set from CC-model.

    The rational is that FT of CC-model (that were derived using CLEAN-boxes) is
    free of off-source noise.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    uvdata = UVData(uv_fits_path)
    noise = uvdata.noise(use_V=True)
    # uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    clean_n(uv_fits_path, 'cc.fits', 'I', mapsize_clean,
            path_to_script=path_to_script, niter=700,
            outpath=out_dir, clean_box=(1.5, -3, -2, 2.5),
            show_difmap_output=True)
    # ccimage = create_clean_image_from_fits_file(os.path.join(out_dir, 'cc.fits'))
    # rms = rms_image(ccimage)
    # blc, trc = find_bbox(ccimage.image, rms)
    # mask = create_mask(ccimage.image.shape, (blc[0], blc[1], trc[0], trc[1]))

    ccmodel = create_model_from_fits_file(os.path.join(out_dir, 'cc.fits'))
    # Here optionally filter CC

    uvdata.substitute([ccmodel])
    uvdata.noise_add(noise)
    uvdata.save(os.path.join(out_dir, outname))


def plot_clean_image_and_components(image, comps, outname=None, ra_range=None,
                                    dec_range=None, n_rms_level=3.0,
                                    n_rms_size=1.0):
    """
    :param image:
        Instance of ``CleanImage`` class.
    :param comps:
        Iterable of ``Components`` instances.
    :return:
        ``Figure`` object.
    """
    beam = image.beam
    rms = rms_image(image)
    if ra_range is None or dec_range is None:
        blc, trc = find_bbox(image.image, n_rms_size*rms, 10)
    else:
        blc, trc = None, None
    fig = iplot(image.image, x=image.x, y=image.y,
                min_abs_level=n_rms_level*rms,
                beam=beam, show_beam=True, blc=blc, trc=trc, components=comps,
                close=True, colorbar_label="Jy/beam", ra_range=ra_range,
                dec_range=dec_range, show=False)
    if outname is not None:
        fig.savefig(outname, bbox_inches='tight', dpi=300)
    return fig


if __name__ == '__main__':
    import glob
    data_dir = "/home/ilya/data/sashaplavin"
    # uv_fits_paths = glob.glob(os.path.join(data_dir, "*_S_*"))
    uv_fits_paths = [os.path.join(data_dir, "J0510+1800_S_2007_05_03_sok_vis.fits")]

    for uv_fits_path in uv_fits_paths:
        # Create directory for current source
        source = os.path.split(uv_fits_path)[-1].split("_")[0]
        out_dir = "/home/ilya/data/sashaplavin/test_uv/{}".format(source)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'

        automodeler = AutoModeler(uv_fits_path, out_dir, path_to_script,
                                  n_comps_terminate=20,
                                  core_elliptic=False,
                                  mapsize_clean=(1024, 0.75),
                                  ra_range_plot=None,
                                  dec_range_plot=None,
                                  show_difmap_output_modelfit=True)
        # Stoppers define when to stop adding components to model
        rchsq_stopping = RChiSquaredStopping(mode="or")
        stoppers = [# AddedComponentFluxLessRMSStopping(mode="or"),
                    # AddedComponentFluxLessRMSFluxStopping(mode="or"),
                    # AddedTooDistantComponentStopping(mode="or"),
                    # AddedTooSmallComponentStopping(mode="and"),
                    # AddedNegativeFluxComponentStopping(mode="or"),
                    # for 0430 exclude it
                    # AddedOverlappingComponentStopping(),
                    # NLastDifferesFromLast(mode="or"),
                    # NLastDifferencesAreSmall(mode="or"),
                    rchsq_stopping]
                    # Keep iterating while this stopper fires
                    # TotalFluxStopping(rel_threshold=0.2, mode="while")]
        # Selectors choose best model using different heuristics
        selectors = [FluxBasedModelSelector(delta_flux=0.001),
                     SizeBasedModelSelector(delta_size=0.001)]

        # Filters additionally remove complex models with non-physical
        # components (e.g. too small faint component or component
        # located far away from source.)
        filters = [# SmallSizedComponentsModelFilter(),
                   # ComponentAwayFromSourceModelFilter(ccimage=automodeler.ccimage),
                   # NegativeFluxComponentModelFilter()]
                   # ToElongatedCoreModelFilter()]
                   # OverlappingComponentsModelFilter()]
]
        automodeler.run(stoppers)
        best_model = automodeler.select_best(selectors, filters)
        automodeler.plot_results(best_model=best_model,
                                 stoppers_dict={"RChiSQ": rchsq_stopping})
        # automodeler.archive_images()
        # automodeler.archive_models()
        # automodeler.clean()