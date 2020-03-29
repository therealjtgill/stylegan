import argparse
import numpy as np
import os
from stylegan import stylegan
import sys
import time
from utils import scale_and_shift_pixels
np.set_printoptions(threshold=sys.maxsize)

from keras.preprocessing.image import ImageDataGenerator

def main(argv):
   parser = argparse.ArgumentParser(
      description="Training script for stylegan++."
   )

   parser.add_argument("-o", "--outdir",
      help     = "Location where training output (e.g. checkpoints and loss " +
                 "files) should be stored.",
      required = False,
      default  = "."
   )

   parser.add_argument("-c", "--config",
      help     = "Location where a model config file can be found. A config " +
                 "file contains layer dimensions, training specs, etc., in " +
                 "dictionary format.",
      required = False,
      default  = None
   )

   parser.add_argument("-t", "--traindatadir",
      help     = "Location of training data for the GAN (single folder with " +
                 "all images inside of it.",
      required = False,
      default  = "/home/jg/Documents/stylegan/ffhq-dataset/thisfolderisjustforkeras"
   )

   args = parser.parse_args()

   batch_size = 8
   sess = tf.Session()
   model = stylegan(sess, gamma=0.5, batch_size=batch_size, use_r1_reg=True)
   sess.run(tf.global_variables_initializer())

   gan_data_generator = ImageDataGenerator(
      rescale=1,
      preprocessing_function=scale_and_shift_pixels,
      horizontal_flip=True,
   )

   data_flow = gan_data_generator.flow_from_directory(
      args.traindatadir,
      target_size=(256, 256),
      batch_size=batch_size,
      shuffle=True
   )

   iterations = 0
   for x, _ in data_flow:


if __name__ == "__main__":
   main(sys.argv)