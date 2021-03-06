import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from stylegan import stylegan
import sys
import tensorflow as tf
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

   parser.add_argument("-t", "--train_data_dir",
      help     = "Location of training data for the GAN (single folder with " +
                 "all images inside of it.",
      required = False,
      default  = "/home/jg/Documents/stylegan/ffhq-dataset/thisfolderisjustforkeras"
   )

   parser.add_argument("--n_critic",
      help     = "The number of times to run the critic per minibatch.",
      required = False,
      default  = 3,
      type     = int
   )

   parser.add_argument("--save_frequency",
      help     = "The number of iterations between saving checkpoints.",
      required = False,
      default  = 360, # Roughly every 6 hours on GTX 1080 TI
      type     = int
   )

   args = parser.parse_args()

   batch_size = 8
   sess = tf.Session()
   model = stylegan(sess, gamma=0.5, batch_size=batch_size, use_r1_reg=True)

   gan_data_generator = ImageDataGenerator(
      rescale=1,
      preprocessing_function=scale_and_shift_pixels,
      horizontal_flip=True,
   )

   data_flow = gan_data_generator.flow_from_directory(
      args.train_data_dir,
      target_size=(256, 256),
      batch_size=batch_size,
      shuffle=True
   )

   disc_losses    = []
   gen_losses     = []
   num_epochs     = 0
   num_iterations = 0
   for x, _ in data_flow:
      train_start_time = time.time()
      if x.shape[0] != batch_size:
         num_epochs += 1
         continue

      disc_loss, fake_pred, real_pred = model.trainDiscriminatorBatch(x)
      disc_losses.append(disc_loss)
      print("Discriminator loss:", disc_loss)
      print("Discriminator train time:", time.time() - train_start_time)
      print("Real prediction:\n", real_pred, "\nFake prediction:\n", fake_pred)

      num_iterations += 1

      if (num_iterations % args.n_critic) == 0:
         gen_start_time = time.time()
         gen_loss = model.trainGeneratorBatch()
         print("Generator loss:", gen_loss)
         print("Generator train time:", time.time() - gen_start_time)

      if (num_iterations % 10*args.n_critic) == 0:
         gen_start_time = time.time()
         gen_images = model.runGeneratorBatch()
         for i in range(min(gen_images.shape[0], 5)):
            plt.figure()
            plt.imshow(np.array((gen_images[i, :, :, :] + 1.)/2.))
            save_filename = os.path.join(
               args.outdir,
               'generated_image_' + str(num_iterations) + '_' + str(i) + '.png'
            )
            plt.savefig(save_filename)
            plt.close()
         print("\nGenerated some images! Took", time.time() - gen_start_time, "seconds.\n")

      if (num_iterations % args.save_frequency) == 0:
         model.saveParams(args.outdir, num_iterations)

      print("Iteration ", num_iterations, " took ", time.time() - train_start_time, " seconds.")

if __name__ == "__main__":
   main(sys.argv)