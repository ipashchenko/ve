import os
import pickle
import json
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mode
from astropy.time import Time
from uv_data import UVData
from spydiff import clean_difmap
from from_fits import (create_clean_image_from_fits_file,
                       create_image_from_fits_file, create_model_from_fits_file)
from image import find_shift, find_bbox
from images import Images
from image_ops import (pol_mask, analyze_rotm_slice, hovatta_find_sigma_pang,
                       rms_image, rms_image_shifted)
from bootstrap import CleanBootstrap
from image import plot as iplot
from utils import hdi_of_mcmc
from mojave import download_mojave_uv_fits, get_epochs_for_source


chisq_crit_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
                     6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307}


def boot_ci(boot_images, original_image, alpha=0.68, kind=None):
    """
    Calculate bootstrap CI.

    :param boot_images:
        Iterable of 2D numpy arrays with bootstrapped images.
    :param original_image:
        2D numpy array with original image.
    :param kind: (optional)
        Type of CI.
    :return:
        Two numpy arrays with low and high CI borders for each pixel.

    """

    images_cube = np.dstack(boot_images)
    boot_ci = np.zeros(np.shape(images_cube[:, :, 0]))
    mean_boot = np.zeros(np.shape(images_cube[:, :, 0]))
    hdi_0 = np.zeros(np.shape(images_cube[:, :, 0]))
    hdi_1 = np.zeros(np.shape(images_cube[:, :, 0]))
    print("calculating CI intervals")
    for (x, y), value in np.ndenumerate(boot_ci):
        hdi = hdi_of_mcmc(images_cube[x, y, :], cred_mass=alpha)
        boot_ci[x, y] = hdi[1] - hdi[0]
        hdi_0[x, y] = hdi[0]
        hdi_1[x, y] = hdi[1]
        mean_boot[x, y] = np.mean(images_cube[x, y, :])

    if kind == 'asym':
        hdi_low = original_image.image - (mean_boot - hdi_0)
        hdi_high = original_image.image + hdi_1 - mean_boot
    else:
        hdi_low = original_image.image - boot_ci / 2.
        hdi_high = original_image.image + boot_ci / 2.

    return hdi_low, hdi_high


class MFObservations(object):
    def __init__(self, fits_files, imsizes=None, n_boot=100, common_imsize=None,
                 common_beam=None, find_shifts=False, path_to_script=None,
                 data_dir=None, clear_difmap_logs=True, rotm_slices=None,
                 sigma_evpa=None, sigma_d_term=None, n_scans=None):
        """
        :param fits_files:
            Iterable of FITS files with self-calibrated simultaneous
            mutlifrequency UV-data.
        :param imsizes: (optional)
            Iterable of tuples (image size [pix], pixel size [mas]) for data
            sorted by frequency. If ``None`` then derive from data UV-plane
            coverages. (default: ``None``)
        :param common_imsize: (optional)
            Iterable (image size [pix], pixel size [mas]) for common image
            parameters. If ``None`` then use minimal pixel size but keep the
            same physical size. (default: ``None``)
        :param common_beam: (optional)
            Parameters of beam to convolve CCs to make matched resolution maps.
            If ``None`` then derive from available as beam of lowest frequency.
            (default: ``None``)
        :param find_shifts:
            Boolean. Find shifts between pairs of images at different bands?
            (default: ``False``)
        :param path_to_script:
            Path to `difmap` cleaning script `final_clean_nw`. If ``None`` then
            suppose it is in CWD. (default: ``None``)
        :param data_dir: (optional)
            Directory to store output. If ``None`` then use CWD. (default:
            ``None``)
        :param clear_difmap_logs: (optional)
            Boolean. Clear `difmap` log-files after CLEANing methods? (default:
            ``True``)
        :param rotm_slices: (optional)
            Iterable of start & stop point coordinates (in image coordinates,
            in mas) of ROTM slice to plot. If ``None`` then don't plot any
            slices. (default: ``None``)
        :param sigma_evpa: (optional)
            Iterable of uncertainies of absolute EVPA calibration for each band.
            If ``None`` then use zero values. (default: ``None``)
        :param sigma_d_term: (optional)
            Iterable of D-terms calibration uncertainties for each band. If
            ``None`` then use zero for each band. (default: ``None``)
        :param n_scans: (optional)
            Iterable of numbers of independend scans for each band. If ``None``
            then number of scans is determined from data (some of them may be
            dependent). (default: ``None``)

        """
        self.original_fits_files = fits_files
        self.uvdata_dict = dict()
        self.uvfits_dict = dict()
        self.uvfits_boot_dict = dict()
        self.load_uvdata()
        self.stokes = ('I', 'Q', 'U')
        self.n_boot = n_boot
        self._cs_mask = None
        self._cs_mask_n_sigma = None
        self.rotm_slices = rotm_slices
        self._chisq_crit = chisq_crit_values[len(self.uvdata_dict) - 2]
        self.figures = dict()

        if sigma_evpa is None:
            self.sigma_evpa = np.zeros(len(self.uvdata_dict), dtype=float)
        else:
            assert len(sigma_evpa) == len(self.uvdata_dict)
            self.sigma_evpa = np.array(sigma_evpa)
        if sigma_d_term is None:
            self.sigma_d_term = np.zeros(len(self.uvdata_dict), dtype=float)
        else:
            assert len(sigma_d_term) == len(self.uvdata_dict)
        self.sigma_d_term = np.array(sigma_d_term)
        self.n_scans = n_scans

        if imsizes is None:
            imsizes = [self.uvdata_dict[freq].imsize_by_uv_coverage for
                       freq in self.freqs]
        self.imsizes = imsizes
        self.imsizes_dict = {freq: imsize for freq, imsize in
                             zip(self.freqs, self.imsizes)}

        self._common_imsize = common_imsize

        self._common_beam = common_beam

        self.find_shifts = find_shifts
        if path_to_script is None:
            path_to_script = os.path.join(os.getcwd(), 'final_clean_nw')
        self.path_to_script = path_to_script
        if data_dir is None:
            data_dir = os.getcwd()
        self.data_dir = data_dir
        self.clear_difmap_logs = clear_difmap_logs

        # Container for original CLEAN-images of self-calibrated uv-data
        self.cc_image_dict = dict()
        # Container for paths to FITS-files with original CLEAN-images of
        # self-calibrated uv-data
        self.cc_fits_dict = dict()
        # Container for original CLEAN-image's beam parameters
        self.cc_beam_dict = dict()

        # Containers for images and paths to FITS files with common size images
        self.cc_cs_image_dict = dict()
        self.cc_cs_fits_dict = dict()

        # Container for rms of common sized images
        self._cs_images_rms = dict()

        # Container for rms of original images
        self._images_rms = dict()

        # Instance of ``Images`` class with original common sized images.
        self._original_cs_images = None

        # Containers for paths to FITS files with native parameters images made
        # from bootstraped data
        self.cc_boot_fits_dict = dict()
        # Containers for paths to FITS files with common size images made from
        # bootstraped data
        self.cc_boot_cs_fits_dict = dict()

        # Hovatta-type errors estimation. Values - 2D numpy arrays.
        self.evpa_sigma_dict = dict()
        self.ppol_sigma_dict = dict()

        self._boot_images = None

        # Instance of ``Images`` class with ROTM maps
        self._boot_rotm_images = None

        # Uncertainties found using bootstrapped data (accounting for D- and
        # EVPA-error). Values - instances of ``Image`` class.
        self.evpa_sigma_boot_dict = dict()

        # ROTM image made by conventional method (accounting for D- and
        # EVPA-error in PANG)
        self._rotm_image_conv = None
        # ROTM image made by conventional method (accounting for D-error in
        # PANG only)
        self._rotm_image_conv_grad = None

        # ROTM error map made by conventional method (accounting for D- and
        # EVPA-error in PANG)
        self._rotm_image_sigma_conv = None
        # ROTM error map made by conventional method (accounting for only
        # D-error in PANG)
        self._rotm_image_sigma_conv = None

        # Chi-squared image made by conventional method (must account for both
        # D- and EVPA-error in PANG)
        self._rotm_chisq_image_conv = None

        # ROTM image made using PANG errors obtained by bootstrapping PANG maps
        # with D-terms and EVPA calibrations error accounted for.
        self._rotm_image_boot = None

        # ROTM error map made by bootstrapping with D-terms and EVPA
        # calibration errors accounted for.
        self._rotm_image_sigma_boot = None
        # ROTM error map made by bootstrapping with only D-terms error accounted
        # for.
        self._rotm_image_sigma_boot_grad = None

        # Chi-squared image made by bootstrapped data with D-terms and EVPA
        # calibration errors accounted for.
        self._rotm_chisq_image_boot = None

    def run(self, sigma_evpa=None, sigma_d_term=None, colors_clim=None,
            n_sigma_mask=None, rotm_slices=None, pxls_plot=None,
            plot_points=None, model_generator=None, slice_ylim=None,
            freq_stokes_dict_native=None, freq_stokes_dict_common=None):
        self._t0 = Time.now()
        date, time = str(self._t0.utc.datetime).split()
        self._difmap_commands_file =\
            os.path.join(self.data_dir,
                         "difmap_commands_{}_{}".format(date, time))
        self.clean_original_native(freq_stokes_dict=freq_stokes_dict_native)
        self.clean_original_common(freq_stokes_dict=freq_stokes_dict_common)
        if self.find_shifts:
            self.get_shifts()
        self.bootstrap_uvdata()
        # self.clean_boot_native()
        self.clean_boot_common()
        self.set_common_mask(n_sigma=n_sigma_mask)
        self.analyze_rotm_conv(colors_clim=colors_clim, sigma_evpa=sigma_evpa,
                               sigma_d_term=sigma_d_term,
                               rotm_slices=rotm_slices, pxls_plot=pxls_plot,
                               plot_points=plot_points, slice_ylim=slice_ylim)
        self.analyze_rotm_boot(colors_clim=colors_clim, rotm_slices=rotm_slices,
                               slice_ylim=slice_ylim)

    def load_uvdata(self):
        self.uvdata_dict = dict()
        self.uvfits_dict = dict()
        for fits_file in self.original_fits_files:
            print("Loading UV-FITS file {}".format(os.path.split(fits_file)[-1]))
            uvdata = UVData(fits_file)
            self.uvdata_dict.update({uvdata.band_center: uvdata})
            self.uvfits_dict.update({uvdata.band_center: fits_file})

    @property
    def freqs(self):
        return sorted(self.uvdata_dict.keys())

    @property
    def common_imsize(self):
        if self._common_imsize is None:
            pixsizes = [imsize[1] for imsize in self.imsizes]
            phys_sizes = [imsize[0]*imsize[1] for imsize in self.imsizes]
            pixsize = min(pixsizes)
            phys_size = max(phys_sizes)
            powers = (phys_size / pixsize) // np.array([2**i + 1 for i in range(20)])
            powers = powers.tolist()
            imsize = 2 ** powers.index(0)
            self._common_imsize = imsize, pixsize
            print("Common image parameteres: {}, {}".format(imsize, pixsize))
        return self._common_imsize

    @property
    def common_beam(self):
        """
        :note:
            By default beam of lowest frequency is used.
        """
        if self._common_beam is None:
            self._common_beam = self.cc_beam_dict[self.freqs[0]]
            print("Common beam parameters: {}".format(self._common_beam))
        return self._common_beam

    def cs_images_rms(self):
        raise NotImplementedError
        self._cs_images_rms = None

    def images_rms(self):
        raise NotImplementedError
        self._images_rms = None

    def set_common_mask(self, n_sigma=3.):
        print("Finding rough mask for creating bootstrap images of RM, alpha,"
              " ...")
        cs_mask = pol_mask({stokes: self.cc_cs_image_dict[self.freqs[-1]][stokes] for
                            stokes in self.stokes},
                           rms_cs_dict=None,
                           uv_fits_path=self.uvfits_dict[self.freqs[-1]],
                           n_sigma=n_sigma,
                           path_to_script=self.path_to_script)
        self._cs_mask = cs_mask
        self._cs_mask_n_sigma = n_sigma

    def clean_original_native(self, freq_stokes_dict=None):
        """
        Clean original FITS-files with uv-data using native resolution.
        """
        for freq in self.freqs:
            self.cc_image_dict.update({freq: dict()})
            self.cc_fits_dict.update({freq: dict()})
            self.cc_beam_dict.update({freq: dict()})

        if freq_stokes_dict is not None:
            print("Found CLEANed images of original uvdata with naitive map &"
                  " beam parameters...")
            for freq in self.freqs:
                for stokes in self.stokes:
                    image = create_clean_image_from_fits_file(freq_stokes_dict[freq][stokes])
                    self.cc_image_dict[freq].update({stokes: image})
                    if stokes == 'I':
                        self.cc_beam_dict.update({freq: image.beam})

        else:
            print("Clean original uv-data with native map & beam parameters...")
            for freq in self.freqs:
                print("Cleaning frequency {} with image "
                      "parameters {}".format(freq, self.imsizes_dict[freq]))
                uv_fits_path = self.uvfits_dict[freq]
                uv_dir, uv_fname = os.path.split(uv_fits_path)
                for stokes in self.stokes:
                    outfname = '{}_{}_cc.fits'.format(freq, stokes)
                    outpath = os.path.join(self.data_dir, outfname)
                    # Check if it is already done
                    if not os.path.exists(outpath):
                        clean_difmap(uv_fname, outfname, stokes,
                                     self.imsizes_dict[freq], path=uv_dir,
                                     path_to_script=self.path_to_script,
                                     outpath=self.data_dir,
                                     command_file=self._difmap_commands_file)
                    else:
                        print("Found CLEAN model in file {}".format(outfname))
                    self.cc_fits_dict[freq].update({stokes: os.path.join(self.data_dir,
                                                                         outfname)})
                    image = create_clean_image_from_fits_file(outpath)
                    self.cc_image_dict[freq].update({stokes: image})
                    if stokes == 'I':
                        self.cc_beam_dict.update({freq: image.beam})

        if self.clear_difmap_logs:
            print("Removing difmap log-files...")
            difmap_logs = glob.glob(os.path.join(self.data_dir, "difmap.log*"))
            for difmap_log in difmap_logs:
                os.unlink(difmap_log)

    def clean_original_common(self, freq_stokes_dict=None):

        # if freq_stokes_dict is not None:
        #     print("Found CLEANed images of original uvdata with common map &"
        #           " beam parameters...")
        #     for freq in self.freqs:
        #         self.cc_cs_image_dict.update({freq: dict()})
        #         self.cc_cs_fits_dict.update({freq: dict()})
        #         for stokes in self.stokes:
        #             image = create_clean_image_from_fits_file(freq_stokes_dict[freq][stokes])
        #             self.cc_image_dict[freq].update({stokes: image})
        #             if stokes == 'I':
        #                 self.cc_beam_dict.update({freq: image.beam})

        print("Clean original uv-data with common map parameters "
              " {} and beam {}".format(self.common_imsize, self.common_beam))
        for freq in self.freqs:
            self.cc_cs_image_dict.update({freq: dict()})
            self.cc_cs_fits_dict.update({freq: dict()})

            uv_fits_path = self.uvfits_dict[freq]
            uv_dir, uv_fname = os.path.split(uv_fits_path)
            for stokes in self.stokes:
                outfname = 'cs_{}_{}_cc.fits'.format(freq, stokes)
                outpath = os.path.join(self.data_dir, outfname)
                # Check if it is already done
                if not os.path.exists(outpath):
                    clean_difmap(uv_fname, outfname, stokes, self.common_imsize,
                                 path=uv_dir,
                                 path_to_script=self.path_to_script,
                                 beam_restore=self.common_beam,
                                 outpath=self.data_dir,
                                 command_file=self._difmap_commands_file)
                else:
                    print("Found CLEAN model in file {}".format(outfname))
                self.cc_cs_fits_dict[freq].update({stokes: os.path.join(self.data_dir,
                                                                        outfname)})
                image = create_clean_image_from_fits_file(outpath)
                self.cc_cs_image_dict[freq].update({stokes: image})

        if self.clear_difmap_logs:
            print("Removing difmap log-files...")
            difmap_logs = glob.glob(os.path.join(self.data_dir, "difmap.log*"))
            for difmap_log in difmap_logs:
                os.unlink(difmap_log)

    def get_shifts(self):
        print("Optionally find shifts between original CLEAN-images...")
        print("Determining images shift...")
        beam_pxl = int(self.common_beam[0] / self.common_imsize[1])
        shift_dict = dict()
        freq_1 = self.freqs[0]
        image_1 = self.cc_cs_image_dict[freq_1]['I']

        for freq_2 in self.freqs[1:]:
            image_2 = self.cc_cs_image_dict[freq_2]['I']
            # Coarse grid of possible shifts
            shift = find_shift(image_1, image_2, beam_pxl, 1,
                               max_mask_r=beam_pxl, mask_step=2)
            # More accurate grid of possible shifts
            print("Using fine grid for accurate estimate")
            coarse_grid = range(0, 100, 5)
            idx = coarse_grid.index(shift)
            if idx > 0:
                min_shift = coarse_grid[idx - 1]
            else:
                min_shift = 0
            shift = find_shift(image_1, image_2, coarse_grid[idx + 1], 1,
                               min_shift=min_shift, max_mask_r=200,
                               mask_step=5)

            shift_dict.update({str((freq_1, freq_2,)): shift})

        # Dumping shifts to json file in target directory
        with open(os.path.join(self.data_dir, "shifts.json"), 'w') as fp:
            json.dump(shift_dict, fp)

    def bootstrap_uvdata(self):
        print("Bootstrap self-calibrated uv-data with CLEAN-models...")
        for freq, uv_fits_path in self.uvfits_dict.items():

            # Check if it is already done
            files = glob.glob(os.path.join(self.data_dir,
                                           'boot_{}*.uvf'.format(freq)))
            # If number of found files doesn't equal to ``n_boot`` - remove them
            if not len(files) == self.n_boot:
                for file in files:
                    os.unlink(file)

                # and bootstrap current frequency again
                cc_fits_paths = [self.cc_fits_dict[freq][stokes] for stokes in self.stokes]
                uvdata = self.uvdata_dict[freq]

                models = list()
                for cc_fits_path in cc_fits_paths:
                    ccmodel = create_model_from_fits_file(cc_fits_path)
                    models.append(ccmodel)

                # Position of current ``freq``: ``0`` means the lowest
                # frequency.
                i = self.freqs.index(freq)
                boot = CleanBootstrap(models, uvdata,
                                      sigma_dterms=self.sigma_d_term[i])
                curdir = os.getcwd()
                os.chdir(self.data_dir)
                boot.run(n=self.n_boot, nonparametric=False, use_v=False,
                         use_kde=True, outname=['boot_{}'.format(freq), '.uvf'])
                os.chdir(curdir)

                files = glob.glob(os.path.join(self.data_dir,
                                               'boot_{}*.uvf'.format(freq)))
            print("Found bootstraped uvdata files!")
            self.uvfits_boot_dict.update({freq: sorted(files)})

    def clean_boot_native(self):
        print("Clean bootstrap replications with native restoring beam and map"
              " size...")
        for freq in self.freqs:
            self.cc_boot_fits_dict.update({freq: dict()})
            uv_fits_paths = self.uvfits_boot_dict[freq]
            for stokes in self.stokes:
                for i, uv_fits_path in enumerate(uv_fits_paths):
                    uv_dir, uv_fname = os.path.split(uv_fits_path)
                    outfname = 'boot_{}_{}_cc_{}.fits'.format(freq, stokes,
                                                              str(i + 1).zfill(3))
                    # Check if it is already done
                    if not os.path.exists(os.path.join(self.data_dir,
                                                       outfname)):
                        clean_difmap(uv_fname, outfname, stokes,
                                     self.common_imsize, path=uv_dir,
                                     path_to_script=self.path_to_script,
                                     beam_restore=self.common_beam,
                                     outpath=self.data_dir,
                                     command_file=self._difmap_commands_file)
                    else:
                        print("Found CLEAN model in file {}".format(outfname))
                files = sorted(glob.glob(os.path.join(self.data_dir,
                                                      'boot_{}_{}_cc_*.fits'.format(freq, stokes))))
                self.cc_boot_fits_dict[freq].update({stokes: files})

        if self.clear_difmap_logs:
            print("Removing difmap log-files...")
            difmap_logs = glob.glob(os.path.join(self.data_dir, "difmap.log*"))
            for difmap_log in difmap_logs:
                os.unlink(difmap_log)

    def clean_boot_common(self):
        print("Clean bootstrap replications with common "
              "restoring beams and map sizes...")
        for freq in self.freqs:
            self.cc_boot_cs_fits_dict.update({freq: dict()})
            uv_fits_paths = self.uvfits_boot_dict[freq]
            for stokes in self.stokes:
                for i, uv_fits_path in enumerate(uv_fits_paths):
                    uv_dir, uv_fname = os.path.split(uv_fits_path)
                    outfname = 'cs_boot_{}_{}_cc_{}.fits'.format(freq, stokes,
                                                              str(i + 1).zfill(3))
                    # Check if it is already done
                    if not os.path.exists(os.path.join(self.data_dir,
                                                       outfname)):
                        clean_difmap(uv_fname, outfname, stokes,
                                     self.common_imsize, path=uv_dir,
                                     path_to_script=self.path_to_script,
                                     beam_restore=self.common_beam,
                                     outpath=self.data_dir,
                                     command_file=self._difmap_commands_file)
                    else:
                        print("Found CLEAN model in file {}".format(outfname))
                files = sorted(glob.glob(os.path.join(self.data_dir,
                                                      'cs_boot_{}_{}_cc_*.fits'.format(freq, stokes))))
                self.cc_boot_cs_fits_dict[freq].update({stokes: files})

        if self.clear_difmap_logs:
            print("Removing difmap log-files...")
            difmap_logs = glob.glob(os.path.join(self.data_dir, "difmap.log*"))
            for difmap_log in difmap_logs:
                os.unlink(difmap_log)

    @property
    def original_cs_images(self):
        if self._original_cs_images is None:
            self._original_cs_images = Images()
            for freq in self.freqs:
                for stokes in self.stokes:
                    self.original_cs_images.add_image(self.cc_cs_image_dict[freq][stokes])
        return self._original_cs_images

    def analyze_rotm_conv(self, sigma_evpa=None, sigma_d_term=None,
                          rotm_slices=None, colors_clim=None, n_sigma=None,
                          pxls_plot=None, plot_points=None, slice_ylim=None):
        print("Estimate RM map and it's error using conventional method...")

        if sigma_evpa is None:
            sigma_evpa = self.sigma_evpa
        if sigma_d_term is None:
            sigma_d_term = self.sigma_d_term
        if rotm_slices is None:
            rotm_slices = self.rotm_slices
        if n_sigma is not None:
            self.set_common_mask(n_sigma)

        # Find EVPA error for each frequency
        print("Calculating maps of PANG errors for each band using Hovatta's"
              " approach...")
        # Fetch common size `I` map on highest frequency for plotting PANG error
        # maps
        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']

        if pxls_plot is not None:
            pxls_plot = [i_image._convert_coordinate(pxl) for pxl in pxls_plot]

        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))

        for i, freq in enumerate(self.freqs):
            n_ant = len(self.uvdata_dict[freq].antennas)
            n_if = self.uvdata_dict[freq].nif
            d_term = sigma_d_term[i]
            s_evpa = sigma_evpa[i]

            # Get number of scans
            if self.n_scans is not None:
                try:
                    # If `NX` table is present
                    n_scans = len(self.uvdata_dict[freq].scans)
                except TypeError:
                    scans_dict = self.uvdata_dict[freq].scans_bl
                    scan_lengths = list()
                    for scans in scans_dict.values():
                        if scans is not None:
                            scan_lengths.append(len(scans))
                    n_scans = mode(scan_lengths)[0][0]
            else:
                n_scans = self.n_scans[i]

            q = self.cc_cs_image_dict[freq]['Q']
            u = self.cc_cs_image_dict[freq]['U']
            i = self.cc_cs_image_dict[freq]['I']

            # We need ``s_evpa = 0`` for testing RM gradient significance but
            # ``s_evpa != 0`` for calculating RM errors
            pang_std, ppol_std = hovatta_find_sigma_pang(q, u, i, s_evpa,
                                                         d_term, n_ant, n_if,
                                                         n_scans)
            self.evpa_sigma_dict[freq] = pang_std
            fig = iplot(i_image.image, pang_std, x=i_image.x, y=i_image.y,
                        min_abs_level=3. * rms, colors_mask=self._cs_mask,
                        color_clim=[0, 1], blc=blc, trc=trc,
                        beam=self.common_beam, colorbar_label='sigma EVPA, [rad]',
                        show_beam=True, show=False)
            self.figures['EVPA_sigma_{}'.format(freq)] = fig

        sigma_pang_arrays = [self.evpa_sigma_dict[freq] for freq in self.freqs]
        sigma_pang_arrays_grad = [np.sqrt(self.evpa_sigma_dict[freq]**2 -
                                          np.deg2rad(self.sigma_evpa[i])**2) for
                                  i, freq in enumerate(self.freqs)]
        rotm_image, sigma_rotm_image, chisq_image =\
            self.original_cs_images.create_rotm_image(sigma_pang_arrays,
                                                      mask=self._cs_mask,
                                                      return_chisq=True,
                                                      plot_pxls=pxls_plot,
                                                      outdir=self.data_dir,
                                                      mask_on_chisq=False)
        rotm_image_grad, sigma_rotm_image_grad, _ =\
            self.original_cs_images.create_rotm_image(sigma_pang_arrays_grad,
                                                      mask=self._cs_mask,
                                                      return_chisq=True,
                                                      plot_pxls=pxls_plot,
                                                      outdir=self.data_dir,
                                                      mask_on_chisq=False)
        self._rotm_image_conv = rotm_image
        self._rotm_image_conv_grad = rotm_image_grad
        self._rotm_image_sigma_conv = sigma_rotm_image
        self._rotm_image_sigma_conv_grad = sigma_rotm_image_grad
        self._rotm_chisq_image_conv = chisq_image

        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        uv_fits_path = self.uvfits_dict[self.freqs[-1]]
        image_fits_path = self.cc_cs_fits_dict[self.freqs[-1]]['I']
        # RMS using Hovatta-style
        rms = rms_image_shifted(uv_fits_path, image_fits=image_fits_path,
                                path_to_script=self.path_to_script)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))
        fig = iplot(i_image.image, rotm_image.image, x=i_image.x, y=i_image.y,
                    min_abs_level=3. * rms, colors_mask=self._cs_mask,
                    color_clim=colors_clim, blc=blc, trc=trc,
                    beam=self.common_beam, slice_points=rotm_slices,
                    show_beam=True, show=False, show_points=plot_points,
                    cmap='viridis')
        self.figures['rotm_image_conv'] = fig
        fig = iplot(i_image.image, sigma_rotm_image.image, x=i_image.x,
                    y=i_image.y, min_abs_level=3. * rms,
                    colors_mask=self._cs_mask, color_clim=[0, 200], blc=blc,
                    trc=trc, beam=self.common_beam, slice_points=rotm_slices,
                    show_beam=True, show=False, cmap='viridis', beam_place='ul')
        self.figures['rotm_image_conv_sigma'] = fig
        fig = iplot(i_image.image, sigma_rotm_image_grad.image, x=i_image.x,
                    y=i_image.y, min_abs_level=3. * rms,
                    colors_mask=self._cs_mask, color_clim=[0, 200], blc=blc,
                    trc=trc, beam=self.common_beam, slice_points=rotm_slices,
                    show_beam=True, show=False, cmap='viridis', beam_place='ul')
        self.figures['rotm_image_conv_sigma_grad'] = fig
        fig = iplot(i_image.image, chisq_image.image, x=i_image.x, y=i_image.y,
                    min_abs_level=3. * rms, colors_mask=self._cs_mask,
                    outfile='rotm_chisq_image_conv', outdir=self.data_dir,
                    color_clim=[0., self._chisq_crit], blc=blc, trc=trc,
                    beam=self.common_beam,
                    colorbar_label='Chi-squared', slice_points=rotm_slices,
                    show_beam=True, show=False, cmap='viridis')
        self.figures['rotm_chisq_image_conv'] = fig

        if rotm_slices is not None:
            self.figures['slices_conv'] = dict()
            for rotm_slice in rotm_slices:
                rotm_slice_ = i_image._convert_coordinates(rotm_slice[0],
                                                           rotm_slice[1])
                # Here we use RM image error calculated using PANG errors
                # without contribution of the EVPA calibration errors.
                fig = analyze_rotm_slice(rotm_slice_, rotm_image,
                                         sigma_rotm_image=sigma_rotm_image_grad,
                                         outdir=self.data_dir,
                                         beam_width=int(i_image._beam.beam[0]),
                                         outfname="ROTM_{}_slice".format(rotm_slice),
                                         ylim=slice_ylim)
                self.figures['slices_conv'][str(rotm_slice)] = fig

        return rotm_image, sigma_rotm_image

    @property
    def boot_images(self):
        if self._boot_images is None:
                self._boot_images = Images()
                for freq in self.freqs:
                    for stokes in self.stokes:
                        # FIXME: to low memory usage for now
                        if stokes not in ('Q', 'U'):
                            continue
                        self._boot_images.add_from_fits(self.cc_boot_cs_fits_dict[freq][stokes])
        return self._boot_images

    def create_boot_pang_errors(self, cred_mass=0.68, n_sigma=None):
        """
        Create dictionary with images of PANG errors calculated from bootstrap
        PANG maps.

        :param cred_mass: (optional)
            Credibility mass. (default: ``0.68``)
        :param n_sigma: (optional)
            Sigma clipping for mask. If ``None`` then use instance's value.
            (default: ``None``)
        :return:
            Dictionary with keys - frequencies & values - instances of ``Image``
            class with error maps.
        """
        print("Calculating maps of PANG errors for each band using bootstrapped"
              " PANG maps...")
        result = dict()
        if n_sigma is not None:
            self.set_common_mask(n_sigma)

        # Fetch common size `I` map on highest frequency for plotting PANG error
        # maps
        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))
        for i, freq in enumerate(self.freqs):
            images = self.boot_images.create_pang_images(freq=freq,
                                                         mask=self._cs_mask)
            pang_images = Images()
            pang_images.add_images(images)
            error_image = pang_images.create_error_image(cred_mass=cred_mass)
            # As this errors are used for linear fit judgement only - add EVPA
            # absolute calibration error in quadrature
            evpa_error = np.deg2rad(self.sigma_evpa[i]) * np.ones(error_image.image.shape)
            error_image.image = np.sqrt((error_image.image)**2. + evpa_error**2.)
            result[freq] = error_image
            fig = iplot(i_image.image, error_image.image, x=i_image.x, y=i_image.y,
                        min_abs_level=3. * rms, colors_mask=self._cs_mask,
                        color_clim=[0, 1], blc=blc, trc=trc, beam=self.common_beam,
                        colorbar_label='sigma EVPA, [rad]', slice_points=None,
                        show_beam=True, show=False, cmap='viridis')
            self.figures['EVPA_sigma_boot_{}'.format(freq)] = fig
        self.evpa_sigma_boot_dict = result
        return result
        # FIXME: Beam slice length must be projection - not major axis
        # FIXME: Point fit file name must include coordinates in mas, not pixels
        # FIXME: Figure out how to work with low memory usage!

    def analyze_rotm_boot(self, n_sigma=None, cred_mass=0.68, rotm_slices=None,
                          colors_clim=None, slice_ylim=None,
                          use_conv_image_in_boot_slice=True):
        print("Estimate RM map and it's error using bootstrap...")

        if rotm_slices is None:
            rotm_slices = self.rotm_slices
        if n_sigma is not None:
            self.set_common_mask(n_sigma)
        # This accounts for D-terms and EVPA calibration errors.
        sigma_pangs_dict = self.create_boot_pang_errors()
        sigma_pang_arrays = [sigma_pangs_dict[freq].image for freq in
                             self.freqs]
        # This is RM image made using bootstrap-based PANG errors that
        # originally account for D-terms and EVPA calibration errors.
        rotm_image, _, chisq_image = \
            self.original_cs_images.create_rotm_image(sigma_pang_arrays,
                                                      mask=self._cs_mask,
                                                      return_chisq=True,
                                                      mask_on_chisq=True)
        self._rotm_image_boot = rotm_image
        self._rotm_chisq_image_boot = chisq_image

        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]/2))
        fig = iplot(i_image.image, rotm_image.image, x=i_image.x, y=i_image.y,
                    min_abs_level=3. * rms, colors_mask=self._cs_mask,
                    color_clim=colors_clim, blc=blc, trc=trc, beam=self.common_beam,
                    slice_points=rotm_slices, cmap='viridis',
                    show_beam=True, show=False, beam_place="ul")
        self.figures['rotm_image_boot'] = fig
        fig = iplot(i_image.image, chisq_image.image, x=i_image.x, y=i_image.y,
                    min_abs_level=3. * rms, colors_mask=self._cs_mask,
                    color_clim=[0., self._chisq_crit], blc=blc, trc=trc, beam=self.common_beam,
                    colorbar_label='Chi-squared', slice_points=rotm_slices,
                    show_beam=True, show=False, cmap='viridis')
        self.figures['rotm_chisq_image_boot'] = fig

        self._boot_rotm_images =\
            self.boot_images.create_rotm_images(mask=self._cs_mask,
                                                mask_on_chisq=False,
                                                sigma_evpa=self.sigma_evpa)
        sigma_rotm_image =\
            self._boot_rotm_images.create_error_image(cred_mass=cred_mass)
        self._rotm_image_sigma_boot = sigma_rotm_image

        # Now ``self._boot_rotm_images`` doesn't contain contribution from
        # EVPA calibration error
        self._boot_rotm_images =\
            self.boot_images.create_rotm_images(mask=self._cs_mask,
                                                mask_on_chisq=False,
                                                sigma_evpa=None)
        sigma_rotm_image_grad =\
            self._boot_rotm_images.create_error_image(cred_mass=cred_mass)
        self._rotm_image_sigma_boot_grad = sigma_rotm_image_grad


        # This sigma take absolute EVPA calibration uncertainty into
        # account
        fig = iplot(i_image.image, sigma_rotm_image.image, x=i_image.x, y=i_image.y,
                    min_abs_level=3. * rms, colors_mask=self._cs_mask,
                    outfile='rotm_image_boot_sigma', outdir=self.data_dir,
                    color_clim=[0, 200], blc=blc, trc=trc, beam=self.common_beam,
                    slice_points=rotm_slices, cmap='viridis', beam_place='ul',
                    show_beam=True, show=False)
        self.figures['rotm_image_boot_sigma'] = fig
        # This sigma RM doesn't include EVPA
        fig = iplot(i_image.image, sigma_rotm_image_grad.image, x=i_image.x, y=i_image.y,
                    min_abs_level=3. * rms, colors_mask=self._cs_mask,
                    outfile='rotm_image_boot_sigma', outdir=self.data_dir,
                    color_clim=[0, 200], blc=blc, trc=trc, beam=self.common_beam,
                    slice_points=rotm_slices, cmap='viridis', beam_place='ul',
                    show_beam=True, show=False)
        self.figures['rotm_image_boot_sigma_grad'] = fig

        if rotm_slices is not None:
            self.figures['slices_boot'] = dict()
            self.figures['slices_boot_conv'] = dict()
            for rotm_slice in rotm_slices:
                rotm_slice_ = i_image._convert_coordinates(rotm_slice[0],
                                                           rotm_slice[1])
                if use_conv_image_in_boot_slice:
                    rotm_image_ = self._rotm_image_conv
                else:
                    rotm_image_ = rotm_image
                # ``self._boot_rotm_images`` doesn't contain contribution from
                # EVPA calibration error - so used for RM gradient searches

                fig = analyze_rotm_slice(rotm_slice_, rotm_image_,
                                         rotm_images=self._boot_rotm_images,
                                         outdir=self.data_dir,
                                         beam_width=int(i_image._beam.beam[0]),
                                         outfname="ROTM_{}_slice_boot".format(rotm_slice),
                                         ylim=slice_ylim, show_dots_boot=True,
                                         # fig=self.figures['slices_conv'][str(rotm_slice)],
                                         fig=None)
                self.figures['slices_boot'][str(rotm_slice)] = fig

                fig = analyze_rotm_slice(rotm_slice_, rotm_image_,
                                         rotm_images=self._boot_rotm_images,
                                         outdir=self.data_dir,
                                         beam_width=int(i_image._beam.beam[0]),
                                         outfname="ROTM_{}_slice_boot".format(rotm_slice),
                                         ylim=slice_ylim, show_dots_boot=True,
                                         fig=self.figures['slices_conv'][str(rotm_slice)],
                                         # fig=None,
                                         )
                self.figures['slices_boot_conv'][str(rotm_slice)] = fig

        return rotm_image, sigma_rotm_image


if __name__ == '__main__':
    import glob
    # 0923+392
    source = '0923+392'
    rotm_slices = [((0.3, 2.6), (-0.6, -2.75))]
    colors_clim = [-800, 1500]
    epoch = '2006_07_07'

    # 2230+114
    # source = '2230+114'
    # rotm_slices = [((4, -5), (1, -7))]
    # colors_clim = [-600, 250]
    # epoch = '2006_02_12'
    # slice_ylim = [-200, 600]

    # 0945+408
    # source = '0945+408'
    # epoch = '2006_06_15'
    # rotm_slices = [((2.5, 1), (0.5, -3)), ((1.5, 1.5), (0, -2))]
    # colors_clim = [-120, 440]

    # 1641+399
    # source = '1641+399'
    # epoch = '2006_06_15'
    # rotm_slices = [((-2, -3), (-2, 3))]
    # colors_clim = [-550, 650]

    path_to_script = '/home/ilya/github/ve/difmap/final_clean_nw'
    # epochs = get_epochs_for_source(source, use_db='multifreq')
    # print(epochs)
    # print("Found epochs for source {}:".format(source))
    # for epoch in epochs:
    #     print(epoch)
    # epoch = epochs[-1]
    # base_dir = '/home/ilya/vlbi_errors/article'
    # base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/mf'
    # base_dir = '/home/ilya/Dropbox/papers/boot/new_pics/revision_pics'
    base_dir = '/home/ilya/data/boot'
    data_dir = os.path.join(base_dir, source)

    # Download uv-data from MOJAVE web DB optionally
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        download_mojave_uv_fits(source, epochs=[epoch],
                                download_dir=data_dir)
    fits_files = glob.glob(os.path.join(data_dir, "{}*.uvf".format(source)))
    imsizes = [(512, 0.1), (512, 0.1), (512, 0.1), (512, 0.1)]
    n_boot = 100
    mfo = MFObservations(fits_files, imsizes, n_boot, data_dir=data_dir,
                         path_to_script=path_to_script,
                         n_scans=[4., 4., 4., 4.],
                         sigma_d_term=[0.002, 0.002, 0.002, 0.002],
                         sigma_evpa=[4., 4., 2., 3.])
    mfo.run(n_sigma_mask=3.0, colors_clim=colors_clim,
            rotm_slices=rotm_slices, slice_ylim=None)
            # pxls_plot=[(0, 0), (1, 0), (-3, 0), (-4, 0)],
            # plot_points=[(0, 0), (-2, 0), (-3, 0), (-4, 0)])

    # Plot RM
    fig = mfo.figures['rotm_image_boot']
    fig.set_size_inches(4.5, 3.5, forward=True)
    ax = fig.gca()
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    # Set the tick labels font
    for label in (ax.get_xticklabels()+ax.get_yticklabels()):
        label.set_fontsize(14)
    fig.savefig(os.path.join(data_dir, "{}_rotm_image_boot.pdf".format(source)),
                dpi=600, bbox_inches="tight")
    fig.show()
    plt.close()

    # Plot slices
    for slice_ in mfo.figures['slices_boot']:
        fig = mfo.figures['slices_boot'][slice_]
        fig.set_size_inches(4.5, 3.5, forward=True)
        ax = fig.gca()
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        # Set the tick labels font
        for label in (ax.get_xticklabels()+ax.get_yticklabels()):
            label.set_fontsize(14)
        fig.savefig(os.path.join(data_dir, "{}_slice_{}.pdf".format(source, slice_)),
                    dpi=600, bbox_inches="tight")
        fig.show()
        plt.close()

    # # For 2230 only
    # # Plot RM
    # fig = mfo.figures['rotm_image_conv_sigma_grad']
    # fig.set_size_inches(4.5, 3.5, forward=True)
    # ax = fig.gca()
    # ax.xaxis.label.set_fontsize(14)
    # ax.yaxis.label.set_fontsize(14)
    # # Set the tick labels font
    # for label in (ax.get_xticklabels()+ax.get_yticklabels()):
    #     label.set_fontsize(14)
    # fig.savefig(os.path.join(data_dir, "{}_rotm_image_conv_sigma_grad.pdf".format(source)),
    #             dpi=600, bbox_inches="tight")
    # fig.show()
    # plt.close()
    #
    # fig = mfo.figures['rotm_image_boot_sigma_grad']
    # fig.set_size_inches(4.5, 3.5, forward=True)
    # ax = fig.gca()
    # ax.xaxis.label.set_fontsize(14)
    # ax.yaxis.label.set_fontsize(14)
    # # Set the tick labels font
    # for label in (ax.get_xticklabels()+ax.get_yticklabels()):
    #     label.set_fontsize(14)
    # fig.savefig(os.path.join(data_dir, "{}_rotm_image_boot_sigma.pdf".format(source)),
    #             dpi=600, bbox_inches="tight")
    # fig.show()
    # plt.close()