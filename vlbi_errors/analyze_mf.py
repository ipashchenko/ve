import os
import pickle
import json
import glob
import numpy as np
from scipy.stats import mode
from uv_data import UVData
from spydiff import clean_difmap
from from_fits import (create_clean_image_from_fits_file,
                       create_image_from_fits_file, create_model_from_fits_file)
from image import find_shift, find_bbox
from images import Images
from image_ops import (pol_mask, analyze_rotm_slice, hovatta_find_sigma_pang,
                       rms_image)
from bootstrap import CleanBootstrap
from image import plot as iplot
from utils import hdi_of_mcmc
from mojave import download_mojave_uv_fits, get_epochs_for_source


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
                 sigma_evpa=None, sigma_d_term=None):
        """
        :param fits_files:
        :param imsizes: (optional)
            Iterable of tuples (image size [pix], pixel size [mas]) for data
            sorted by frequency. If ``None`` then derive from data UV-plane
            coverages. (default: ``None``)
        :param common_imsize: (optional)
            Iterable (image size [pix], pixel size [mas]) for common image
            parameters. If ``None`` then use minimal pixel size but keep the
            same physical size. (default: ``None``)
        :param common_beam:
        :param find_shifts:
        :param path_to_script:
        :param data_dir:
        :param clear_difmap_logs:
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

        if imsizes is None:
            imsizes = [self.uvdata_dict[freq].imsize_by_uv_coverage for
                       freq in self.freqs]
        self.imsizes = imsizes
        self.imsizes_dict = {freq: imsize for freq, imsize in
                             zip(self.freqs, self.imsizes)}

        self._common_imsize = common_imsize

        self._common_beam = common_beam

        self.find_shifts = find_shifts
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

        # Instance of ``Images`` class with original common sized images.
        self._original_cs_images = None

        # Containers for paths to FITS files with native parameters images made
        # from bootstraped data
        self.cc_boot_fits_dict = dict()
        # Containers for paths to FITS files with common size images made from
        # bootstraped data
        self.cc_boot_cs_fits_dict = dict()

        # Hovatta-type errors estimation
        self.evpa_sigma_dict = dict()
        self.ppol_sigma_dict = dict()

        self._boot_images = None

        # Instance of ``Images`` class with ROTM maps
        self._boot_rotm_images = None

        # ROTM image made by conventional method
        self._rotm_image_conv = None
        # ROTM error map made by conventional method
        self._rotm_image_sigma_conv = None
        # ROTM image made by bootstrapping
        self._rotm_image_boot = None
        # ROTM error map made by bootstrapping
        self._rotm_image_sigma_boot = None

    def run(self, sigma_evpa=None, sigma_d_term=None, colors_clim=None,
            n_sigma_mask=None, rotm_slices=None):
        self.clean_original_native()
        self.clean_original_common()
        if self.find_shifts:
            self.get_shifts()
        self.bootstrap_uvdata()
        self.clean_boot_native()
        self.clean_boot_common()
        self.set_common_mask(n_sigma=n_sigma_mask)
        self.analyze_rotm_conv(colors_clim=colors_clim, sigma_evpa=sigma_evpa,
                               sigma_d_term=sigma_d_term,
                               rotm_slices=rotm_slices)
        self.analyze_rotm_boot(colors_clim=colors_clim, rotm_slices=rotm_slices)

    def load_uvdata(self):
        self.uvdata_dict = dict()
        self.uvfits_dict = dict()
        for fits_file in self.original_fits_files:
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
        return self._common_imsize

    @property
    def common_beam(self):
        """
        :note:
            By default beam of lowest frequency is used.
        """
        if self._common_beam is None:
            self._common_beam = self.cc_beam_dict[self.freqs[0]]
        return self._common_beam

    def set_common_mask(self, n_sigma=3.):
        print("Finding rough mask for creating bootstrap images of RM, alpha,"
              " ...")
        cs_mask = pol_mask({stokes: self.cc_cs_image_dict[self.freqs[-1]][stokes] for
                            stokes in self.stokes}, n_sigma=n_sigma)
        self._cs_mask = cs_mask
        self._cs_mask_n_sigma = n_sigma

    def clean_original_native(self):
        """
        Clean original FITS-files with uv-data using native resolution.
        """
        for freq in self.freqs:
            self.cc_image_dict.update({freq: dict()})
            self.cc_fits_dict.update({freq: dict()})
            self.cc_beam_dict.update({freq: dict()})

        print("Clean original uv-data with native map parameters...")
        for freq in self.freqs:
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
                                 outpath=self.data_dir, show_difmap_output=False)
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

    def clean_original_common(self):
        print("Clean original uv-data with common map parameters...")
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
                                 path=uv_dir, path_to_script=self.path_to_script,
                                 beam_restore=self.common_beam,
                                 outpath=self.data_dir, show_difmap_output=False)
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
        shift_dict = dict()
        freq_1 = self.freqs[0]
        image_1 = self.cc_cs_image_dict[freq_1]['I']

        for freq_2 in self.freqs[1:]:
            image_2 = self.cc_cs_image_dict[freq_2]['I']
            # Coarse grid of possible shifts
            shift = find_shift(image_1, image_2, 100, 5, max_mask_r=200,
                               mask_step=5)
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

                boot = CleanBootstrap(models, uvdata)
                curdir = os.getcwd()
                os.chdir(self.data_dir)
                boot.run(n=self.n_boot, nonparametric=True,
                         outname=['boot_{}'.format(freq), '.uvf'])
                os.chdir(curdir)

                files = glob.glob(os.path.join(self.data_dir,
                                               'boot_{}*.uvf'.format(freq)))
            self.uvfits_boot_dict.update({freq: sorted(files)})

    def clean_boot_native(self):
        print("Clean bootstrap replications with common restoring beam and map"
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
                        clean_difmap(uv_fname, outfname, stokes, self.common_imsize,
                                     path=uv_dir,
                                     path_to_script=self.path_to_script,
                                     beam_restore=self.common_beam,
                                     outpath=self.data_dir,
                                     show_difmap_output=False)
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
        print("Optionally clean bootstrap replications with common "
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
                        clean_difmap(uv_fname, outfname, stokes, self.common_imsize,
                                     path=uv_dir,
                                     path_to_script=self.path_to_script,
                                     beam_restore=self.common_beam,
                                     outpath=self.data_dir,
                                     show_difmap_output=False)
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
                          rotm_slices=None, colors_clim=None, n_sigma=None):
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
        # Fetch common size `I` map on highest frequency for plotting PNAG error
        # maps
        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))
        for i, freq in enumerate(self.freqs):
            n_ant = len(self.uvdata_dict[freq].antennas)
            n_if = self.uvdata_dict[freq].nif
            d_term = sigma_d_term[i]
            s_evpa = sigma_evpa[i]

            # Get number of scans
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

            q = self.cc_cs_image_dict[freq]['Q']
            u = self.cc_cs_image_dict[freq]['U']
            i = self.cc_cs_image_dict[freq]['I']
            rms_region = rms_image(i)
            pang_std, ppol_std = hovatta_find_sigma_pang(q, u, i, s_evpa,
                                                         d_term, n_ant, n_if,
                                                         n_scans, rms_region)
            self.evpa_sigma_dict[freq] = pang_std
            iplot(i_image.image, pang_std, x=i_image.x, y=i_image.y,
                  min_abs_level=3. * rms, colors_mask=self._cs_mask,
                  outfile='EVPA_sigma_{}'.format(freq),
                  outdir=self.data_dir, color_clim=[0, 1], blc=blc, trc=trc,
                  beam=self.common_beam, colorbar_label='sigma EVPA, [rad]',
                  show_beam=True, show=False)

        sigma_pang_arrays = [self.evpa_sigma_dict[freq] for freq in self.freqs]
        rotm_image, sigma_rotm_image =\
            self.original_cs_images.create_rotm_image(sigma_pang_arrays,
                                                      mask=self._cs_mask)
        self._rotm_image_conv = rotm_image
        self._rotm_image_sigma_conv = sigma_rotm_image

        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))
        iplot(i_image.image, rotm_image.image, x=i_image.x, y=i_image.y,
              min_abs_level=3. * rms, colors_mask=self._cs_mask,
              outfile='rotm_image_conv', outdir=self.data_dir,
              color_clim=colors_clim, blc=blc, trc=trc, beam=self.common_beam,
              colorbar_label='RM, [rad/m/m]', slice_points=None, show_beam=True,
              show=False)
        iplot(i_image.image, sigma_rotm_image.image, x=i_image.x, y=i_image.y,
              min_abs_level=3. * rms, colors_mask=self._cs_mask,
              outfile='rotm_image_conv_sigma', outdir=self.data_dir,
              color_clim=[0, 200], blc=blc, trc=trc, beam=self.common_beam,
              colorbar_label='RM, [rad/m/m]', slice_points=None, show_beam=True,
              show=False)

        if rotm_slices is not None:
            for rotm_slice in rotm_slices:
                rotm_slice_ = i_image._convert_coordinates(rotm_slice[0],
                                                           rotm_slice[1])
                analyze_rotm_slice(rotm_slice_, rotm_image,
                                   sigma_rotm_image=sigma_rotm_image,
                                   outdir=self.data_dir,
                                   outfname="ROTM_{}_slice.png".format(rotm_slice))

        return rotm_image, sigma_rotm_image

    @property
    def boot_images(self):
        if self._boot_images is None:
                self._boot_images = Images()
                for freq in self.freqs:
                    for stokes in self.stokes:
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

        # Fetch common size `I` map on highest frequency for plotting PNAG error
        # maps
        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))
        for freq in self.freqs:
            images = self.boot_images.create_pang_images(freq=freq,
                                                         mask=self._cs_mask)
            pang_images = Images()
            pang_images.add_images(images)
            error_image = pang_images.create_error_image(cred_mass=cred_mass)
            result[freq] = error_image
            iplot(i_image.image, error_image.image, x=i_image.x, y=i_image.y,
                  min_abs_level=3. * rms, colors_mask=self._cs_mask,
                  outfile='EVPA_sigma_68_boot_{}'.format(freq),
                  outdir=self.data_dir,
                  color_clim=[0, 1], blc=blc, trc=trc, beam=self.common_beam,
                  colorbar_label='sigma EVPA, [rad]', slice_points=None,
                  show_beam=True, show=False)
        return result

    def analyze_rotm_boot(self, n_sigma=None, cred_mass=0.68, rotm_slices=None,
                          colors_clim=None):
        print("Estimate RM map and it's error using bootstrap...")

        if rotm_slices is None:
            rotm_slices = self.rotm_slices
        if n_sigma is not None:
            self.set_common_mask(n_sigma)
        sigma_pangs_dict = self.create_boot_pang_errors()
        sigma_pang_arrays = [sigma_pangs_dict[freq].image for freq in
                             self.freqs]
        rotm_image, _ = \
            self.original_cs_images.create_rotm_image(sigma_pang_arrays,
                                                      mask=self._cs_mask)
        self._rotm_image_boot = rotm_image

        i_image = self.cc_cs_image_dict[self.freqs[-1]]['I']
        rms = rms_image(i_image)
        blc, trc = find_bbox(i_image.image, 2.*rms,
                             delta=int(i_image._beam.beam[0]))
        iplot(i_image.image, rotm_image.image, x=i_image.x, y=i_image.y,
              min_abs_level=3. * rms, colors_mask=self._cs_mask,
              outfile='rotm_image_boot', outdir=self.data_dir,
              color_clim=colors_clim, blc=blc, trc=trc, beam=self.common_beam,
              colorbar_label='RM, [rad/m/m]', slice_points=None, show_beam=True,
              show=False)

        self._boot_rotm_images =\
            self.boot_images.create_rotm_images(mask=self._cs_mask)
        sigma_rotm_image =\
            self._boot_rotm_images.create_error_image(cred_mass=cred_mass)
        self._rotm_image_sigma_boot = sigma_rotm_image

        iplot(i_image.image, sigma_rotm_image.image, x=i_image.x, y=i_image.y,
              min_abs_level=3. * rms, colors_mask=self._cs_mask,
              outfile='rotm_image_boot_sigma', outdir=self.data_dir,
              color_clim=[0, 200], blc=blc, trc=trc, beam=self.common_beam,
              colorbar_label='RM, [rad/m/m]', slice_points=None, show_beam=True,
              show=False)

        if rotm_slices is not None:
            for rotm_slice in rotm_slices:
                rotm_slice_ = i_image._convert_coordinates(rotm_slice[0],
                                                           rotm_slice[1])
                analyze_rotm_slice(rotm_slice_, rotm_image,
                                   rotm_images=self._boot_rotm_images,
                                   outdir=self.data_dir,
                                   outfname="ROTM_{}_slice_boot.png".format(rotm_slice))

        return rotm_image, sigma_rotm_image


if __name__ == '__main__':
    import glob
    source = '0454+844'
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    epochs = get_epochs_for_source(source, use_db='multifreq')
    print(epochs)
    print("Found epochs for source {}:".format(source))
    for epoch in epochs:
        print(epoch)
    base_dir = '/home/ilya/vlbi_errors/article/'
    data_dir = os.path.join(base_dir, source)

    # Download uv-data from MOJAVE web DB optionally
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        download_mojave_uv_fits(source, epochs=[epochs[-1]],
                                download_dir=data_dir)
    fits_files = glob.glob(os.path.join(data_dir, "*.uvf"))
    imsizes = [(512, 0.1), (512, 0.1), (512, 0.1), (512, 0.1)]
    n_boot = 10
    mfo = MFObservations(fits_files, imsizes, n_boot, data_dir=data_dir,
                         path_to_script=path_to_script)
    mfo.run(sigma_evpa=[2., 2., 2., 2.], n_sigma_mask=3.0,
            sigma_d_term=[0.003, 0.003, 0.003, 0.003],
            rotm_slices=[((3, -0.5), (-3, -0.5)), ((3, 0), (-3, 0))])
