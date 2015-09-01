import numpy as np
from from_fits import create_clean_image_from_fits_file
from image import BasicImage
from utils import mask_region, mas_to_rad
from model import Model
from components import DeltaComponent

# Create image for simulating
# Clean image of lowest frequency data
ccimage = create_clean_image_from_fits_file('/home/ilya/code/vlbi_errors/vlbi_errors/cc.fits')
# pix, pix, deg
beam = (ccimage._beam.bmaj, ccimage._beam.bmin, ccimage._beam.bpa)
# (NxN)
imsize = ccimage.imsize
# dx, dy [rads]
pixsize = ccimage.pixsize


# Beam to jet width ratio
ratio_w = 0.5
# Jet length to beam ratio
ratio_l = 20
# Maximum I, P flux
I_max = 0.5
P_max = 0.02
Q_max = U_max = np.sqrt(P_max / 2.)

# Create image with pixsize 10x smaller then original
factor = 3
# In new pix
beam_width = beam[0] * ratio_w * factor
# In new pix
jet_length = ratio_l * beam[0] * factor

# Construct image
new_imsize = (factor*imsize[0], factor*imsize[1])
new_pixref = (factor*imsize[0]/2, factor*imsize[1]/2)
image = BasicImage(imsize=new_imsize,
                   pixsize=(abs(pixsize[0])/factor, abs(pixsize[1])/factor),
                   pixref=new_pixref)

# Construct region with emission
jet_region = mask_region(image._image, region=(new_pixref[0]-int(beam_width//2),
                                               new_pixref[1],
                                               new_pixref[0]+int(beam_width//2),
                                               new_pixref[1]+jet_length))
jet_mask = ~jet_region.mask
jet_region = np.ma.array(image._image, mask=jet_mask)

# Create model instance and fill it with components
model = Model()
comps = [DeltaComponent(0.1, image.x[x, y]/mas_to_rad, image.y[x, y]/mas_to_rad)
         for (x, y), value in np.ndenumerate(jet_region) if not
         jet_region.mask[x, y]]
model.add_components(*comps)
image = BasicImage(imsize=new_imsize,
                   pixsize=(abs(pixsize[0])/factor, abs(pixsize[1])/factor),
                   pixref=new_pixref)
image.add_model(model)











