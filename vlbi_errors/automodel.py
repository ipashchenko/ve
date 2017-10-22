import numpy as np
import os
import tqdm
import glob
import shutil
from scipy import ndimage
from uv_data import UVData
from model import Model
from cv_model import cv_difmap_models
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

    def do_stop(self, new_file):
        """
        :param new_file:
            Path to current difmap model file.
        :return:
            Boolean - is criterion fulfilled?
        """
        self.files.append(new_file)
        if self.is_applicable():
            return self.check_criterion()
        else:
            return False

    def clear(self):
        self.files = list()


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
                                                   hovatta_factor=False)
        return self._threshold

    def check_criterion(self):
        _dir, _fn = os.path.split(self.files[-1])
        last_comp = import_difmap_model(_fn, _dir)[-1]
        print(Style.DIM + "{} message:".format(self.__class__.__name__))
        print(Style.DIM + "Last added component has flux = {:.4f}"
                          " while threshold = {:.4f}".format(last_comp.p[0], self.threshold) +
              Style.RESET_ALL)
        return last_comp.p[0] < self.threshold


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
        if len(self.files) > self.n_check:
            return True
        else:
            return False


class NLastDifferesFromLast(NLast):
    """
    Since last ``n_check`` iterations parameters of core component haven't
    changed.
    """
    def __init__(self, n_check=5, frac_flux_min=0.002, delta_flux_min=0.001,
                 delta_size_min=0.001):
        super(NLastDifferesFromLast, self).__init__(n_check)
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
                 delta_size_min=0.001):
        super(NLastDifferencesAreSmall, self).__init__(n_check)
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
        self.n_stop = n

    def check_criterion(self):
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


class AutoModeler(object):
    def __init__(self, uv_fits_path, out_dir, path_to_script,
                 mapsize_clean=None, core_elliptic=False,
                 compute_CV=False, n_CV=5, n_rep_CV=1, n_comps_terminate=50,
                 niter_difmap=100, show_difmap_output_clean=False,
                 show_difmap_output_modelfit=False,
                 ra_range_plot=None, dec_range_plot=None):
        self.uv_fits_path = uv_fits_path
        self.uv_fits_dir, self.uv_fits_fname = os.path.split(uv_fits_path)
        self.out_dir = out_dir
        self.path_to_script = path_to_script
        self.compute_CV = compute_CV
        self.n_CV = n_CV
        self.n_rep_CV = n_rep_CV
        self.n_comps_terminate = n_comps_terminate
        if core_elliptic:
            self.core_type = 'eg'
        else:
            self.core_type = 'cg'

        self.source = self.uv_fits_fname.split(".")[0]
        self.freq = self.uv_fits_fname.split(".")[1]

        if mapsize_clean is None:
            if self.freq == 'u':
                self.mapsize_clean = (512, 0.1)
            elif self.freq == 'q':
                self.mapsize_clean = (512, 0.03)
            else:
                raise Exception("Indicate mapsize_clean!")
        else:
            self.mapsize_clean = mapsize_clean

        self.epoch = self.uv_fits_fname.split(".")[2]
        self.uvdata = UVData(uv_fits_path)

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
        self._ccimage_path = os.path.join(self.out_dir, 'image_cc_orig_{}_{}_{}.fits'.format(self.source, self.epoch, self.freq))
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
        self._mdl_prefix = '{}_{}_{}_{}_fitted'.format(self.source, self.freq,
                                                       self.epoch, self.core_type)
        self.fitted_model_paths = list()
        self.dec_range_plot = dec_range_plot
        self.ra_range_plot = ra_range_plot

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
        else:
            print(Style.DIM + "Creating \"residuals\" from original data alone" +
                  Style.RESET_ALL)
            uvdata_residual = self.uvdata
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
        beam = np.sqrt(image.beam[0] * image.beam[1])
        imsize = image.imsize[0]
        mas_in_pix = abs(image.pixsize[0] / mas_to_rad)
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

    def do_iteration(self):
        self.counter += 1
        self.create_residuals(self.model)
        if self.counter == 1:
            core_type = self.core_type
        else:
            core_type = 'cg'
        comp = self.suggest_component(core_type)

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

        modelfit_difmap(self.uv_fits_fname, 'init_{}.mdl'.format(self.counter),
                        '{}_{}.mdl'.format(self._mdl_prefix, self.counter),
                        path=self.uv_fits_dir, mdl_path=out_dir, out_path=out_dir,
                        niter=self.niter_difmap, stokes=self.stokes,
                        show_difmap_output=self.show_difmap_output_modelfit)



        # Check that first component is elliptic one and adjust model in case it
        # is not
        if self.counter > 1 and self.core_type == "eg":
            print(Fore.GREEN + "Checking if elliptic core goes first!" + Style.RESET_ALL)
            model_2check = os.path.join(self.out_dir,
                                        '{}_{}.mdl'.format(self._mdl_prefix,
                                                           self.counter))
            cj_sorted = sort_components_by_distance_from_cj(model_2check,
                                                            automodeler.freq_hz,
                                                            n_check_for_core=1,
                                                            perc_distant=60)
            comps = import_difmap_model(model_2check, out_dir)
            ell_first = len(comps[0]) == 6
            if not ell_first:
                print(Back.RED + "Core has changed position!" + Style.RESET_ALL)
                comps = import_difmap_model(model_2check, out_dir)
                first_comp = comps[0]

                # Create new model with EG component first and CG all others
                new_comps = list()
                new_comps.append(first_comp.to_elliptic())
                for comp in comps[1:]:
                    new_comps.append(comp.to_circular())

                export_difmap_model(new_comps, model_2check,
                                    automodeler.freq_hz)
                modelfit_difmap(self.uv_fits_fname,
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter),
                                '{}_{}.mdl'.format(self._mdl_prefix,
                                                   self.counter),
                                path=self.uv_fits_dir, mdl_path=out_dir,
                                out_path=out_dir,
                                niter=self.niter_difmap, stokes=self.stokes,
                                show_difmap_output=self.show_difmap_output_modelfit)



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
                                path=self.uv_fits_dir, mdl_path=out_dir,
                                out_path=out_dir,
                                niter=self.niter_difmap, stokes=self.stokes,
                                show_difmap_output=self.show_difmap_output_modelfit)
                self.counter -= 1




        # Update model and plot results of current iteration
        model = Model(stokes='I')
        comps = import_difmap_model('{}_{}.mdl'.format(self._mdl_prefix, self.counter), self.out_dir)
        plot_clean_image_and_components(self.ccimage, comps,
                                        outname=os.path.join(out_dir, "{}_image_{}.png".format(self._mdl_prefix, self.counter)),
                                        ra_range=self.ra_range_plot,
                                        dec_range=self.dec_range_plot)
        model.add_components(*comps)
        self.model = model

        return os.path.join(self.out_dir,
                            '{}_{}.mdl'.format(self._mdl_prefix, self.counter))

    def clear(self):
        self.counter = 0
        self.fitted_model_paths = list()

    def run(self, stoppers, start_model_fname=None):
        stoppers = list(stoppers)
        stoppers.append(NLastJustStop(self.n_comps_terminate))
        for stopper in stoppers:
            if isinstance(stopper, ImageBasedStoppingCriterion):
                stopper.set_ccimage(self.ccimage)
            elif isinstance(stopper, UVDataBasedStoppingCriterion):
                stopper.set_uvdata(self.uvdata)

        if start_model_fname is not None:
            mdl_dir, mdl_fname = os.path.split(start_model_fname)
            print(Style.DIM + "Using model from {} as starting point".format(mdl_fname) +
                  Style.RESET_ALL)
            comps = import_difmap_model(mdl_fname, mdl_dir)
            model = Model(stokes=self.stokes)
            model.add_components(*comps)
            self.model = model

        while True:
            new_mdl_file = self.do_iteration()
            if new_mdl_file not in self.fitted_model_paths:
                self.fitted_model_paths.append(new_mdl_file)
            stoppers_and = [stopper for stopper in stoppers if
                            stopper.mode == "and"]
            stoppers_or = [stopper for stopper in stoppers if
                           stopper.mode == "or"]
            stoppers_while = [stopper for stopper in stoppers if
                              stopper.mode == "while"]
            decisions_and = [stopper.do_stop(new_mdl_file) for stopper in
                             stoppers_and]
            decisions_or = [stopper.do_stop(new_mdl_file) for stopper in
                            stoppers_or]
            decisions_while = [not stopper.do_stop(new_mdl_file) for stopper in
                               stoppers_while]
            if np.any(decisions_while):
                continue
            # decision = decisions_and + decisions_or
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

        # best_model_file = self.select_best()
        # self.archive_images()
        # self.archive_models()
        # self.clean()

    def plot_results(self, id_best):
        cores = list()
        for file_ in self.fitted_model_paths:
            core = import_difmap_model(file_)[0]
            cores.append(core)
        if len(core) == 4:
            fig, axes = plt.subplots(2, 1, sharex=True)
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
            axes[1].set_xlabel("Number of components")
            axes[1].set_ylabel("Size, [mas]")
            axes[0].axvline(id_best+1)
            axes[1].axvline(id_best+1)
        elif len(core) == 6:
            fig, axes = plt.subplots(3, 1, sharex=True)
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
            axes[2].set_xlabel("Number of components")
            axes[0].axvline(id_best+1)
            axes[1].axvline(id_best+1)
            axes[2].axvline(id_best+1)

        fig.savefig(os.path.join(out_dir, '{}_core_parameters_vs_ncomps.png'.format(self._mdl_prefix)),
                    bbox_inches='tight', dpi=200)

        fig, axes = plt.subplots(1, 1, sharex=True)
        axes.plot(range(1, len(cores) + 1), [difmap_model_flux(fn) for
                                             fn in self.fitted_model_paths])
        axes.plot(range(1, len(cores) + 1), [difmap_model_flux(fn) for
                                             fn in self.fitted_model_paths],
                  '.k')
        axes.set_ylabel("Total Flux, [Jy]")
        axes.set_xlabel("Number of components")
        axes.axvline(id_best+1)
        axes.axhline(self.total_flux)
        fig.savefig(os.path.join(out_dir, '{}_total_flux_vs_ncomps.png'.format(self._mdl_prefix)),
                    bbox_inches='tight', dpi=200)

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


def redo_automodel(automodeler, stoppers, selectors, filters, id_best):
    ##### EXPERIMENTAL #########################################################
    if sort_components_by_distance_from_cj(best_model, automodeler.freq_hz,
                                           outpath=os.path.join(out_dir,
                                                                "best_cjsorted.mdl")):
        print(Back.RED + "Core has changed position!" + Style.RESET_ALL)
        print(Back.RED + "Starting new round" + Style.RESET_ALL)
        comps = import_difmap_model("best_cjsorted.mdl", out_dir)
        first_comp = comps[0]

        # Create new model with EG component first and CG all others
        new_comps = list()
        new_comps.append(first_comp.to_elliptic())
        for comp in comps[1:]:
            new_comps.append(comp.to_circular())

        automodeler.clean()
        export_difmap_model(new_comps, os.path.join(out_dir, "{}_{}.mdl".format(
            automodeler._mdl_prefix, id_best + 1)),
                            automodeler.freq_hz)

        # Clear all
        automodeler.clear()
        for stopper in stoppers:
            stopper.clear()

        automodeler.fitted_model_paths.append(os.path.join(out_dir,
                                                           "{}_{}.mdl".format(
                                                               automodeler._mdl_prefix,
                                                               id_best + 1)))
        automodeler.counter = id_best + 1
        # Run number of iterations that is defined by stoppers
        automodeler.run(stoppers, start_model_fname=os.path.join(out_dir,
                                                                 "{}_{}.mdl".format(
                                                                     automodeler._mdl_prefix,
                                                                     id_best + 1)))

        files = automodeler.fitted_model_paths
        id_best = max(selector.select(files) for selector in selectors)
        files = files[:id_best + 1]

        filters = [SmallSizedComponentsModelFilter(),
                   ComponentAwayFromSourceModelFilter(ccimage=automodeler.ccimage),
                   ToElongatedCoreModelFilter()]

        for fn in files[::-1]:
            if np.any([flt.do_filter(fn) for flt in filters]):
                id_best -= 1
            else:
                break

        best_model = files[id_best]
        print("Best model is {}".format(best_model))
        return best_model
        ##### EXPERIMENTAL #########################################################


def create_cc_model_uvf(uv_fits_path, mapsize_clean, path_to_script,
                        outname='image_cc_model.uvf', out_dir=None):
    """
    Function that creates uv-data set from CC-model.

    The rational is that FT of CC-model (that were derived using CLEAN-boxes) is
    free of off-source nosie.
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
                dec_range=dec_range)
    if outname is not None:
        fig.savefig(outname, bbox_inches='tight', dpi=300)
    return fig


# TODO: Remove beam from ``bmaj``
def suggest_cg_component(uv_fits_path, mapsize_clean, path_to_script,
                         outname='image_cc.fits', out_dir=None, bmaj_nan=0.1):
    """
    Suggest single circular gaussian component using self-calibrated uv-data
    FITS file.
    :param uv_fits_path:
        Path to uv-data FITS-file.
    :param mapsize_clean:
        Iterable of image size (# pixels) and pixel size (mas).
    :param path_to_script:
        Path to difmap CLEANing script.
    :param outname: (optional)
        Name of file to save CC FITS-file. (default: ``image_cc.fits``)
    :param out_dir: (optional)
        Optional directory to save CC FITS-file. If ``None`` use CWD. (default:
        ``None``)
    :param bmaj_nan: (optional)
        When estimated ``bmaj`` is NaN this value is used as ``bmaj`` [mas].
        (default: ``0.1``)
    :return:
        Instance of ``CGComponent``.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    clean_difmap(uv_fits_fname, outname, 'I', mapsize_clean,
                 path=uv_fits_dir, path_to_script=path_to_script,
                 outpath=out_dir)

    image = create_clean_image_from_fits_file(os.path.join(out_dir, outname))
    beam = np.sqrt(image.beam[0] * image.beam[1])
    imsize = image.imsize[0]
    mas_in_pix = abs(image.pixsize[0] / mas_to_rad)
    amp, y, x, bmaj = infer_gaussian(image.image)
    print("Estimated bmaj = {}".format(bmaj))
    x = mas_in_pix * (x - imsize / 2) * np.sign(image.dx)
    y = mas_in_pix * (y - imsize / 2) * np.sign(image.dy)
    if np.isnan(bmaj):
        bmaj = bmaj_nan
    else:
        bmaj *= mas_in_pix
        bmaj = np.sqrt(bmaj**2 - beam**2)
    # If ``bmaj`` is less then beam than use ``bmaj_nan`` instead
    if np.isnan(bmaj):
        bmaj = bmaj_nan
    return CGComponent(amp, x, y, bmaj), os.path.join(out_dir, outname)


def suggest_eg_component(uv_fits_path, mapsize_clean, path_to_script,
                         outname='image_cc.fits', out_dir=None,
                         bmaj_nan=0.1):
    """
    Suggest single circular gaussian component using self-calibrated uv-data
    FITS file.
    :param uv_fits_path:
        Path to uv-data FITS-file.
    :param mapsize_clean:
        Iterable of image size (# pixels) and pixel size (mas).
    :param path_to_script:
        Path to difmap CLEANing script.
    :param outname: (optional)
        Name of file to save CC FITS-file. (default: ``image_cc.fits``)
    :param out_dir: (optional)
        Optional directory to save CC FITS-file. If ``None`` use CWD. (default:
        ``None``)
    :param bmaj_nan: (optional)
        When estimated ``bmaj`` is NaN this value is used as ``bmaj`` [mas].
        (default: ``0.1``)
    :return:
        Instance of ``CGComponent``.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    clean_difmap(uv_fits_fname, outname, 'I', mapsize_clean,
                 path=uv_fits_dir, path_to_script=path_to_script,
                 outpath=out_dir)

    image = create_clean_image_from_fits_file(os.path.join(out_dir, outname))
    beam = np.sqrt(image.beam[0] * image.beam[1])
    imsize = mapsize_clean[0]
    mas_in_pix = mapsize_clean[1]
    amp, y, x, bmaj = infer_gaussian(image.image)
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
    return EGComponent(amp, x, y, bmaj, 1.0, 0.0), os.path.join(out_dir, outname)


def create_residuals(uv_fits_path, model=None, out_fname='residuals.uvf',
                     out_dir=None):
    if model is None:
        return uv_fits_path
    if out_dir is None:
        out_dir = os.getcwd()
    out_fits_path = os.path.join(out_dir, out_fname)
    uvdata = UVData(uv_fits_path)
    uvdata_ = UVData(uv_fits_path)
    uvdata_.substitute([model])
    uvdata_residual = uvdata - uvdata_
    uvdata_residual.save(out_fits_path, rewrite=True)
    return out_fits_path


def find_best(files, frac_flux=0.01, delta_flux=0.001, frac_size=0.01,
              delta_size=0.001, small_size=10**(-5),
              threshold_flux_small_sized_component=0.1,
              small_size_of_the_core=0.001):
    """
    Select best model from given difmap model files.

    :param files:
        Iterable of paths difmap models.
    :return:
        Path to difmap model file with best model.
    """
    out_dir = os.path.split(files[0])[0]
    files = [os.path.split(file_path)[-1] for file_path in files]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files = [os.path.join(out_dir, file_) for file_ in files]
    comps = list()
    for file_ in files:
        comps_ = import_difmap_model(file_)
        comps.append(comps_[0])

    fluxes = np.array([comp.p[0] for comp in comps])
    last_flux = fluxes[-1]
    fluxes_inv = fluxes[::-1]
    flux_min = min(delta_flux, frac_flux*last_flux)
    a = (abs(fluxes_inv - fluxes_inv[0]) < flux_min)[::-1]
    try:
        # This is index not number! Number is index + 1 (python 0-based
        # indexing)
        n_flux = list(ndimage.binary_opening(a, structure=np.ones(2)).astype(np.int)).index(1)
    except IndexError:
        n_flux = 0

    sizes = np.array([comp.p[3] for comp in comps])
    last_size = sizes[-1]
    sizes_inv = sizes[::-1]
    size_min = min(delta_size, frac_size * last_size)
    # If last model's core size is too small then not compare differences, but
    # compare fraction changes
    if last_size < small_size_of_the_core:
        a = (abs((sizes_inv-sizes_inv[0])/sizes_inv[0]) < frac_size)[::-1]
    else:
        a = (abs(sizes_inv - sizes_inv[0]) < size_min)[::-1]
    try:
        # This is index not number! Number is index + 1 (python 0-based
        # indexing)
        n_size = list(ndimage.binary_opening(a, structure=np.ones(2)).astype(np.int)).index(1)
    except IndexError:
        n_size = 0

    # Now go from largest model to simpler ones and excluding models with small
    # components
    n = max(n_flux, n_size)
    print("Flux+Size==>{}th model".format(n+1))
    if n == 0:
        raise FailedFindBestModelException
    n_best = n
    for model_file in files[:n+1][::-1]:
        comps = import_difmap_model(model_file)
        small_sizes = [comp.p[3] > small_size for comp in comps[1:]]
        fluxes_of_small_sized_components = [comp.p[0] for comp in comps[1:] if comp.p[3] < small_size]
        fluxes_of_small_sized_components = [flux > threshold_flux_small_sized_component for flux in fluxes_of_small_sized_components]
        # print(fluxes_of_small_sized_components)
        # print(small_sizes)
        if not np.alltrue(small_sizes) and not np.alltrue(fluxes_of_small_sized_components):
            print("Decreasing complexity because of small component present")
            n_best = n_best - 1
        else:
            break

    return files[n_best]


def stop_adding_models(files, n_check=5, frac_flux_min=0.002,
                       delta_flux_min=0.001, delta_size_min=0.001):
    """
    Since last ``n_check`` models parameters of core haven't changed.
    """
    out_dir = os.path.split(files[0])[0]
    files = [os.path.split(file_path)[-1] for file_path in files]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files = [os.path.join(out_dir, file_) for file_ in files]
    last_file = files[-1]
    files = files[-n_check-1: -1]
    comps = list()
    last_comp = import_difmap_model(last_file)[0]
    for file_ in files:
        comps_ = import_difmap_model(file_)
        comps.append(comps_[0])
    last_flux = last_comp.p[0]
    last_size = last_comp.p[3]
    fluxes = np.array([comp.p[0] for comp in comps])
    sizes = np.array([comp.p[3] for comp in comps])
    delta_fluxes = abs(fluxes - last_flux)
    delta_sizes = abs(sizes - last_size)
    flux_min = max(delta_flux_min, last_flux*frac_flux_min)
    return np.alltrue(delta_fluxes < flux_min) or np.alltrue(delta_sizes < delta_size_min)


def stop_adding_models_by_total_flux(last_difmap_model, total_flux,
                                     abs_threshold=None,
                                     rel_threshold=0.001):
    """
    Total flux of last difmap model doesn't differ by more then specified
    values.

    :param last_difmap_model:
        Path to last difmap model file.
    :param total_flux:
        Total flux to compare with [Jy].
    :param abs_threshold: (optional)
        Absolute deviations must be less then this value [Jy]. If ``None`` then
        don't use absolute deviations. (default: ``None``)
    :param rel_threshold: (optional)
        Fractional deviations must be less then this value. (default: ``0.001``)
    :return:
        Boolean - stop iterations?
    """
    threshold = abs_threshold or total_flux*rel_threshold
    return abs(difmap_model_flux(last_difmap_model) - total_flux) < threshold


def stop_adding_models_(files, n_check=5, frac_flux_min=0.002,
                        delta_flux_min=0.001,
                        delta_size_min=0.001):
    """
    Each of the last ``n_check`` models differs from previous one by less then
    specified values.
    """
    out_dir = os.path.split(files[0])[0]
    files = [os.path.split(file_path)[-1] for file_path in files]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files = [os.path.join(out_dir, file_) for file_ in files]
    files = files[-n_check-1:]
    comps = list()
    for file_ in files:
        comps_ = import_difmap_model(file_)
        comps.append(comps_[0])
    fluxes = np.array([comp.p[0] for comp in comps])
    last_flux = fluxes[-1]
    sizes = np.array([comp.p[3] for comp in comps])
    delta_fluxes = abs(fluxes[:-1]-fluxes[1:])
    delta_sizes = abs(sizes[:-1]-sizes[1:])
    flux_min = max(delta_flux_min, last_flux * frac_flux_min)
    return np.alltrue(delta_fluxes < flux_min) and np.alltrue(delta_sizes < delta_size_min)


# TODO: Add option to begin with some specified model
def automodel_uv_fits(uv_fits_path, out_dir, path_to_script, start_model_file=None,
                      mapsize_clean=None,
                      core_elliptic=False, compute_CV=False, n_CV=5, n_rep_CV=1,
                      n_max_comps=30, frac_flux=0.01, delta_flux=0.001,
                      delta_size=0.001, small_size=10**(-5),
                      threshold_flux_small_sized_component=0.1,
                      small_size_of_the_core=0.001,
                      n_check=5,
                      check_frac_flux_min=0.01,
                      check_delta_size_min=0.001,
                      use_total_flux_as_criterion=False,
                      abs_diff_total_flux=None,
                      rel_diff_total_flux=0.001):
    """
    Function that automatically models uv-data in difmap.

    It's just like CLEAN but using gaussians. Function uses ``difmap`` for
    CLEAN and modelling.

    :param uv_fits_path:
        Path to FITS uv-data file to model.
    :param out_dir:
        Directory to keep results.
    :param path_to_script:
        Path to D.Homan difmap script for automatical cleaning (final_clean_nw).
    :param start_model_file: (optional)
        Path to difmap file with model to start with. If ``None`` than start
        from scratch (estimate one-component model using uv-data specified by
        ``uv_fits_path``). (default: ``None``)
    :param mapsize_clean: (optional)
        Tuple of number of pixels and pixel size [mas]. If ``None`` then use
        default values for Q & U bands (512, 0.03) & (512, 0.1). (default:
        ``None``)
    :param core_elliptic:
        Boolean - use elliptical core? If ``False`` then use circular gaussian.
        (default: ``False``)
    :param compute_CV: (optional)
        Boolean - compute CV-score for each model? (default: ``False``)
    :param n_CV: (optional)
        Number of CV folds to use when computing CV-score. (default: ``5``)
    :param n_rep_CV: (optional)
        Number of CV repetitions with different seed to use. Used to estimate
        the error of CV-score more precisely. (default: ``1``)
    :param n_max_comps:
        Maximum number of components to try. Try models up to ``n_max_comps``
        components while searching. (default: ``30``)
    :param frac_flux: (optional)
        The best model is the simplest one that has core flux that differs by
        less than ``frac_flux`` of last models core flux [Jy] from all more
        complex models. (default: ``0.002``)
    :note:
        Used together with ``delta_flux`` and max of two is used.
    :param delta_flux: (optional)
        The best model is the simplest one that has core flux that differs by
        less than ``delta_flux`` of last models core flux [Jy] from all more
        complex models. (default: ``0.001``)
    :note:
        Used together with ``frac_flux`` and max of two is used.
    :param delta_size: (optional)
        The best model is the simplest one that has core size that differs by
        less than ``delta_size`` [mas] from all more complex models. (default:
        ``0.001``)
    :param small_size: (optional)
        The smallest size [mas] of single component allowed to present in best
        model. (default: ``0.00001``)
    :param threshold_flux_small_sized_component: (optional)
        Current best model is changed to more simple one if it contains small
        component with flux less then ``hreshold_flux_small_sized_component``
        [Jy]. (default: ``0.1``)
    :param small_size_of_the_core: (optional)
        When core size of the last component is less then
        ``small_size_of_the_core`` [mas] then use fractions (not difference) to
        select the best model. (default: ``0.001``)
    :param n_check: (optional)
        Number of last consequence models to check while checking stopping
        criteria. (default: ``5``)
    :param check_frac_flux_min: (optional)
        All last ``n_check`` models must have core fluxes that differs not more
        than ``check_frac_flux_min`` of last model core [Jy] to stop adding
        components. (default: ``0.002``)
    :param check_delta_size_min: (optional)
        All last ``n_check`` models must have core sizes that differs not more
        than ``check_delta_size_min`` [mas] to stop adding components.
        (default: ``0.001``)
    :param use_total_flux_as_criterion: (optional)
        Boolean - use criterion of total flux? (default: ``False``)
    :param abs_diff_total_flux: (optional)
        See ``stop_adding_models_by_total_flux`` docs. (default: ``None``)
    :param rel_diff_total_flux: (optional)
        See ``stop_adding_models_by_total_flux`` docs. (default: ``0.001``)
    :return:
        Path to best difmap model.

    :notes:
        When working with strong complex structured sources (like 3C273) one
        should use lower ``frac_*`` parameters, as gaussian components model
        has high bias for such sources thus less variance in component's
        parameters (and we use this variance as stopping and best-model
        criteria).
    """
    if core_elliptic:
        core_type = 'eg'
    else:
        core_type = 'cg'

    # mapsize_clean = (1024, 0.1)
    # out_dir = '/home/ilya/github/vlbi_errors/0552'
    # uv_fits_fname = '0552+398.u.2006_07_07.uvf'
    # uv_fits_path = os.path.join(out_dir, uv_fits_fname)
    # path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
    uv_fits_dir, uv_fits_fname = os.path.split(uv_fits_path)
    source = uv_fits_fname.split(".")[0]
    freq = uv_fits_fname.split(".")[1]

    if mapsize_clean is None:
        if freq == 'u':
            mapsize_clean = (512, 0.1)
        elif freq == 'q':
            mapsize_clean = (512, 0.03)
        else:
            raise Exception("Indicate mapsize_clean!")

    epoch = uv_fits_fname.split(".")[2]
    uvdata = UVData(uv_fits_path)
    freq_hz = uvdata.frequency
    if start_model_file is None:
        model = None
    else:
        mdl_dir, mdl_fname = os.path.split(start_model_file)
        print("Using model from {} as starting point".format(mdl_fname))
        comps = import_difmap_model(mdl_fname, mdl_dir)
        model = Model(stokes="I")
        model.add_components(*comps)
    cv_scores = list()
    # ``CleanImage`` instance for CLEANed original uv data set.
    ccimage_orig = None
    # Total flux of all CC components in CLEAN model of the original uv data
    # set.
    total_flux = None

    for i in tqdm.tqdm(range(1, n_max_comps+1), initial=1,
                       desc="# of components", unit_scale=1,
                       dynamic_ncols=True,
                       bar_format='{desc}: {n}|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]'):
        # print("{}-th iteration begins".format(i))
        uv_fits_path_res = create_residuals(uv_fits_path, model=model,
                                            out_dir=out_dir)
        # 1. Modelfit in difmap with CG
        if i == 1 and core_elliptic and not model:
            print("Suggesting EG component to add...")
            cg, image_cc_fits = suggest_eg_component(uv_fits_path_res, mapsize_clean,
                                                     path_to_script, out_dir=out_dir)
        else:
            print("Suggesting CG component to add...")
            cg, image_cc_fits = suggest_cg_component(uv_fits_path_res, mapsize_clean,
                                                     path_to_script, out_dir=out_dir)

        # Saving original CLEAN image to plot pictures with models
        if ccimage_orig is None:
            # If starting from scratch than CC FITS-file is ``image_cc_fits``
            if model is None:
                shutil.copy(image_cc_fits,
                            os.path.join(out_dir, 'image_cc_orig.fits'))
                ccimage_orig = create_clean_image_from_fits_file(os.path.join(out_dir, 'image_cc_orig.fits'))
            else:
                # If first iteration with some model that we need to CLEAN
                # original UV-data
                clean_difmap(uv_fits_fname,
                             os.path.join(out_dir, 'image_cc_orig.fits'), 'I',
                             mapsize_clean, path=uv_fits_dir,
                             path_to_script=path_to_script,
                             outpath=out_dir)
                ccimage_orig = create_clean_image_from_fits_file(os.path.join(out_dir, 'image_cc_orig.fits'))
            total_flux = ccimage_orig.total_flux
        print("Suggested: {}".format(cg))

        if i > 1:
            # If this is not first iteration then append component to existing
            # file
            shutil.copy(os.path.join(out_dir, '{}_{}_{}_{}_fitted_{}.mdl'.format(source, freq, epoch, core_type, i-1)),
                        os.path.join(out_dir, 'init_{}.mdl'.format(i)))
            append_component_to_difmap_model(cg, os.path.join(out_dir, 'init_{}.mdl'.format(i)),
                                             freq_hz)
        else:
            # If this is first iteration then create model file
            export_difmap_model([cg],
                                os.path.join(out_dir, 'init_{}.mdl'.format(i)),
                                freq_hz)

        modelfit_difmap(uv_fits_fname, 'init_{}.mdl'.format(i),
                        '{}_{}_{}_{}_fitted_{}.mdl'.format(source, freq, epoch, core_type, i), path=uv_fits_dir,
                        mdl_path=out_dir, out_path=out_dir, niter=100,
                        show_difmap_output=False)
        model = Model(stokes='I')
        comps = import_difmap_model('{}_{}_{}_{}_fitted_{}.mdl'.format(source, freq, epoch, core_type, i), out_dir)
        plot_clean_image_and_components(ccimage_orig, comps,
                                        outname=os.path.join(out_dir, "{}_{}_{}_{}_image_{}.png".format(source, freq, epoch, core_type, i)))
        model.add_components(*comps)

        # Cross-Validation
        if compute_CV:
            cv_score = cv_difmap_models([os.path.join(out_dir,
                                                      '{}_{}_{}_{}_fitted_{}.mdl'.format(source, freq, epoch, core_type, i))],
                                        uv_fits_path, K=n_CV, out_dir=out_dir,
                                        n_rep=n_rep_CV)
            cv_scores.append((cv_score[0][0][0], cv_score[1][0][0]))

        # Check if we need go further
        if i > n_check:
            fitted_model_files = glob.glob(os.path.join(out_dir, "{}_{}_{}_{}_fitted*.mdl".format(source, freq, epoch, core_type)))
            if stop_adding_models(fitted_model_files, n_check=n_check,
                                  delta_flux_min=delta_flux,
                                  frac_flux_min=check_frac_flux_min,
                                  delta_size_min=check_delta_size_min):
                break

    # # Optionally plot CV-scores
    # import matplotlib.pyplot as plt
    # plt.errorbar(range(1, len(cv_scores)+1), np.atleast_2d(cv_scores)[:, 0],
    #              yerr=np.atleast_2d(cv_scores)[:, 1], fmt='.k')
    # plt.plot(range(1, len(cv_scores)+1), np.atleast_2d(cv_scores)[:, 0])
    # plt.show()

    # Choose best model
    files = glob.glob(os.path.join(out_dir, "{}_{}_{}_{}_fitted*.mdl".format(source, freq, epoch, core_type)))
    # Save files in archive
    with tarfile.open(os.path.join(out_dir, "{}_{}_{}_{}_fitted_models.tar.gz".format(source, freq, epoch, core_type)), "w:gz") as tar:
        for fn in files:
            tar.add(fn, arcname=os.path.split(fn)[-1])

    files = [os.path.split(file_path)[-1] for file_path in files]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files = [os.path.join(out_dir, file_) for file_ in files]
    comps = list()
    for file_ in files:
        comps_ = import_difmap_model(file_)
        comps.append(comps_[0])

    try:
        best_model_file = find_best(files, frac_flux=frac_flux,
                                    delta_flux=delta_flux,
                                    delta_size=delta_size,
                                    small_size=small_size,
                                    threshold_flux_small_sized_component=threshold_flux_small_sized_component,
                                    small_size_of_the_core=small_size_of_the_core)
        k = files.index(best_model_file) + 1
    except FailedFindBestModelException:
        return None

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(range(1, len(comps)+1), [comp.p[0] for comp in comps])
    axes[0].plot(range(1, len(comps)+1), [comp.p[0] for comp in comps], '.k')
    axes[0].set_ylabel("Core Flux, [Jy]")
    axes[1].plot(range(1, len(comps)+1), [comp.p[3] for comp in comps])
    axes[1].plot(range(1, len(comps)+1), [comp.p[3] for comp in comps], '.k')
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Core Size, [mas]")
    axes[0].axvline(k)
    axes[1].axvline(k)
    fig.savefig(os.path.join(out_dir, '{}_{}_{}_{}_core_parameters_vs_ncomps.png'.format(source, freq, epoch, core_type)),
                bbox_inches='tight', dpi=200)

    # Clean model files (we have copies in archive)
    files = glob.glob(os.path.join(out_dir, "{}_{}_{}_{}_fitted_*.mdl".format(source, freq, epoch, core_type)))
    for fn in files:
        os.unlink(fn)
    # Clean images with components superimposed (we have copies in archive)
    files = glob.glob(os.path.join(out_dir, "{}_{}_{}_{}_image_*.png".format(source, freq, epoch, core_type)))
    with tarfile.open(os.path.join(out_dir, "{}_{}_{}_{}_images.tar.gz".format(source, freq, epoch, core_type)), "w:gz") as tar:
        for fn in files:
            tar.add(fn, arcname=os.path.split(fn)[-1])
    for fn in files:
        os.unlink(fn)

    return best_model_file


if __name__ == '__main__':
    sources = ["2251+158", "2230+114", "2223-052", "2200+420"]
    path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
    for source in sources[:1]:
        out_dir = "/home/ilya/STACK/{}".format(source)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for freq in ("u", "q")[:1]:
            files = glob.glob("/home/ilya/STACK/uvf/to_process/{}.{}.*.uvf".format(source, freq))
            epochs = sorted([os.path.basename(fn).split(".")[-2] for fn in files])
            for epoch in epochs:
                # epoch = "2007_01_26"
                epoch = "2006_12_04"
                uv_fits_path = "/home/ilya/STACK/uvf/to_process/{}.{}.{}.uvf".format(source, freq, epoch)
                out_dir = "/home/ilya/STACK/{}/{}".format(source, epoch)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                automodeler = AutoModeler(uv_fits_path, out_dir, path_to_script,
                                          n_comps_terminate=20, core_elliptic=True)
                # Stoppers define when to stop adding components to model
                stoppers = [TotalFluxStopping(),
                            AddedComponentFluxLessRMSStopping(),
                            AddedTooDistantComponentStopping(),
                            AddedTooSmallComponentStopping(),
                            # for 0430 exclude it
                            # AddedOverlappingComponentStopping(),
                            NLastDifferesFromLast(),
                            NLastDifferencesAreSmall(),
                            # Keep iterating while this stopper fires
                            TotalFluxStopping(rel_threshold=0.2, mode="while")]
                # Selectors choose best model using different heuristics
                selectors = [FluxBasedModelSelector(delta_flux=0.001),
                             SizeBasedModelSelector(delta_size=0.001)]

                # Run number of iterations that is defined by stoppers
                automodeler.run(stoppers)

                # Select best model using custom selectors
                files = automodeler.fitted_model_paths
                id_best = max(selector.select(files) for selector in selectors)
                files = files[:id_best+1]

                # Filters additionally remove complex models with non-physical components
                # (e.g. too small faint component or component located far away from
                # source.)
                filters = [SmallSizedComponentsModelFilter(),
                           ComponentAwayFromSourceModelFilter(ccimage=automodeler.ccimage)]
                           # ToElongatedCoreModelFilter()]
                           # OverlappingComponentsModelFilter()]

                # Additionally filter too small, too distant components
                for fn in files[::-1]:
                    if np.any([flt.do_filter(fn) for flt in filters]):
                        id_best -= 1
                    else:
                        break
                print("Best model is {}".format(files[id_best]))

                best_model = files[id_best]

                # EXPERIMENTAL
                # In case of using elliptical model for core check that core is the first
                # one.
                # if automodeler.core_type == 'eg':
                #     core_changed = sort_components_by_distance_from_cj(best_model,
                #                                                        automodeler.freq_hz)
                #     if core_changed:
                #         best_model = redo_automodel(automodeler, stoppers, selectors,
                #                                     filters)
                # EXPERIMENTAL

                automodeler.plot_results(id_best)
                break
                automodeler.archive_images()
                automodeler.archive_models()
                automodeler.clean()