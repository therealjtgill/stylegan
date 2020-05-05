import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from stylegan import stylegan
import sys
import tensorflow as tf
import time
import utils
np.set_printoptions(threshold=sys.maxsize)

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

   parser.add_argument("--num_epochs",
      help     = "Number of training epochs.",
      required = False,
      default  = 28,
      type     = int
   )

   parser.add_argument("--load_checkpoint",
      help     = "The location of a checkpoint to load, including the name " +
                 "of the checkpoint, but not the file extension.",
      required = False,
      default  = None,
      type     = str
   )

   args = parser.parse_args()

   batch_size = 8
   sess = tf.Session()
   model = stylegan(
      sess,
      gamma=0.5,
      batch_size=batch_size,
      use_r1_reg=True,
      use_pl_reg=True
   )

   if args.load_checkpoint is not None:
      model.loadParams(args.load_checkpoint)

   num_epochs     = 0

   save_dir = os.path.join(args.outdir, utils.get_save_folder_name())
   os.makedirs(save_dir)

   for e in range(args.num_epochs):
      gen_images = model.runGeneratorBatch()
      for i in range(gen_images.shape[0]):
         plt.figure()
         plt.imshow(np.array((gen_images[i, :, :, :] + 1.)/2.))
         save_filename = os.path.join(
            save_dir,
            'generated_image_%09d_%02d.png' % (e, i)
         )
         plt.savefig(save_filename)
         plt.close()

if __name__ == "__main__":
   main(sys.argv)
