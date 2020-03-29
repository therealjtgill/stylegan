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

   disc_losses    = []
   gen_losses     = []
   num_epochs     = 0
   num_iterations = 0
   for x, _ in data_flow:
      train_start_time = time.time()
      if x.shape[0] != batch_size:
         num_epochs += 1
         continue

      loss, fake_pred, real_pred = model.trainDiscriminatorBatch(x)
      disc_losses.append(loss)
      print("discriminator run/loss time:", time.time() - disc_start_time)
      print("Real prediction:", real_pred, " Fake prediction: ", fake_pred)

      num_iterations += 1

      if iterations % 5 == 0:
         gen_loss, gen_images = model.trainGeneratorBatch()
         gen_losses.append(gen_loss)

         gen_images = model.runGeneratorBatch()

         for i in range(min(gen_images.shape[0], 5)):
            plt.figure()
            plt.imshow(np.array((gen_images[i, :, :, :] + 1.)/2.))
            save_filename = os.path.join(
               args.outdir,
               'generated_image_' + str(iterations) + '_' + str(i) + '.png'
            )
            plt.savefig(save_filename)
            plt.close()
      print("Iteration ", i, " took ", time.time() - train_start_time, " seconds.")

if __name__ == "__main__":
   main(sys.argv)