import numpy as np

def scale_and_shift_pixels(image_in):
   image_out = np.array(2.*image_in/255. - 1, dtype=np.float32)
   return image_out
