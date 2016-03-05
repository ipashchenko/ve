import os
import copy
import matplotlib.pyplot as plt
from vlbi_errors.simulations import (ModelGenerator, create_jet_model_image,
                                     rotm)
from vlbi_errors.utils import get_fits_image_info, mas_to_rad
from vlbi_errors.from_fits import (create_clean_image_from_fits_file,
                                   create_image_from_fits_file)
from vlbi_errors.uv_data import UVData
from vlbi_errors.spydiff import clean_difmap


if __name__ == '__main__':

    data_dir = '/home/ilya/code/vlbi_errors/examples/L'
    path_to_script = '/home/ilya/code/vlbi_errors/difmap/final_clean_nw'
    cc_fits = os.path.join(data_dir, 'original_cc.fits')
    image_info = get_fits_image_info(cc_fits)
    imsize = (image_info['imsize'][0], abs(image_info['pixsize'][0]) /
              mas_to_rad)
    image = create_clean_image_from_fits_file(cc_fits)
    x = image.x
    y = image.y
    print x
    print y
    # Iterable of ``UVData`` instances with simultaneous multifrequency uv-data
    # of the same source
    observed_uv_fits = os.path.join(data_dir, '1038+064.l18.2010_05_21.uvf')
    observed_uv = UVData(observed_uv_fits)
    image = create_jet_model_image(10, 50, 10, 1., (256, 256), (128, 128))
    plt.matshow(image)
    plt.show()
    rotm = rotm((256, 256), (128, 128))
    stokes_models = {'I': image}
    mod_generator = ModelGenerator(stokes_models, x, y, rotm=rotm)
    models = mod_generator.create_models_for_frequency(observed_uv.frequency)
    simulated_uv = copy.deepcopy(observed_uv)
    noise = observed_uv.noise()
    simulated_uv.substitute(models)
    simulated_uv.noise_add(noise)
    simulated_uv.save(os.path.join(data_dir, 'test.fits'))
    cc_fits_fname = 'test_I.fits'
    clean_difmap('test.fits', cc_fits_fname, 'I', imsize,
                 path=data_dir, path_to_script=path_to_script,
                 outpath=data_dir)
    image = create_image_from_fits_file(os.path.join(data_dir, 'test_I.fits'))
    plt.matshow(image.image)
    plt.show()
