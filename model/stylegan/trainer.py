import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from stylegan import stylegan
import sys
import tensorflow as tf
import time
#from utils import scale_and_shift_pixels
import utils
np.set_printoptions(threshold=sys.maxsize)

#from keras.preprocessing.image import ImageDataGenerator

def main(argv):
   parser = argparse.ArgumentParser(
      description="Training script for stylegan++."
   )

   parser.add_argument("-o", "--outdir",
      help     = "Location where training output (e.g. checkpoints and loss " +
                 "files) should be stored. pwd by default.",
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
      help     = "The number of times to run the critic per minibatch. " +
                 "3 by default.",
      required = False,
      default  = 3,
      type     = int
   )

   parser.add_argument("--save_frequency",
      help     = "The number of seconds between saving checkpoints. 1 " +
                 "hour by default.",
      required = False,
      default  = 3600,
      type     = int
   )

   parser.add_argument("--save_after_delta_t",
      help     = "Run the saver once after this number of seconds have " +
                 "elapsed. This is intended for use in HPC environments " +
                 "where sessions have a fixed amount of time that they are " +
                 "allowed to run. 4 hours by default.",
      required = False,
      default  = 3600*4,
      type     = int
   )

   args = parser.parse_args()

   batch_size = 8
   sess = tf.Session()
   model = stylegan(sess, gamma=0.5, batch_size=batch_size, use_r1_reg=True, use_pl_reg=True)

   # gan_data_generator = ImageDataGenerator(
   #    rescale=1,
   #    preprocessing_function=utils.scale_and_shift_pixels,
   #    horizontal_flip=True,
   # )

   # data_flow = gan_data_generator.flow_from_directory(
   #    args.train_data_dir,
   #    target_size=(256, 256),
   #    batch_size=batch_size,
   #    shuffle=True
   # )

   # training_filenames = os.listdir(args.train_data_dir)
   # dataset = (tf.data.Dataset.from_tensor_slices(training_filenames)
   #    .map(utils.parse_image_tf, num_parallel_calls=1)
   #    .shuffle(buffer_size=50)
   #    .batch(batch_size)
   #    .prefetch(1)
   # )

   # iterator = dataset.make_one_shot_iterator()

   data_flow = utils.JankyImageLoader(
      dir_to_load=args.train_data_dir,
      batch_size=8,
      preprocess=utils.scale_and_shift_pixels
   )

   disc_losses    = [-1,]
   gen_losses     = [-1,]
   num_epochs     = 0
   num_iterations = 0
   next_save_time = time.time() + args.save_frequency
   final_save_time = time.time() + args.save_after_delta_t

   save_dir = os.path.join(args.outdir, utils.get_save_folder_name())
   losses_filename = os.path.join(save_dir, "losses.dat")
   os.makedirs(save_dir)

   losses_file = open(losses_filename, "w")

   #for x, _ in data_flow:
   for x in data_flow:
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
               save_dir,
               'generated_image_' + str(num_iterations) + '_' + str(i) + '.png'
            )
            plt.savefig(save_filename)
            plt.close()
         print("\nGenerated some images! Took", time.time() - gen_start_time, "seconds.\n")

      # if (num_iterations % args.save_frequency) == 0:
      #    model.saveParams(save_dir, num_iterations)
      if time.time() >= next_save_time:
         next_save_time = time.time() + args.save_frequency
         model.saveParams(os.path.join(save_dir, "stylegan_ckpt"), num_iterations)

      if time.time() >= final_save_time:
         final_save_time = np.inf
         model.saveParams(os.path.join(save_dir, "stylegan_ckpt"), num_iterations)

      losses_file.write(str(disc_losses[-1]) + " " + str(gen_losses[-1]) + "\n")
      print("Iteration ", num_iterations, " took ", time.time() - train_start_time, " seconds.")

if __name__ == "__main__":
   main(sys.argv)
