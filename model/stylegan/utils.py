import datetime
import numpy as np
import os
from PIL import Image
import tensorflow as tf

class JankyImageLoader(object):
   def __init__(
      self,
      dir_to_load=None,
      batch_size=8,
      preprocess=lambda x: x,
      max_num_epochs=1
   ):
      self.dir_to_load = dir_to_load
      self.batch_size = batch_size
      self.preprocess = preprocess
      self.max_num_epochs = max_num_epochs
      self.num_epochs_elapsed = 0
      self.image_names = [
         os.path.join(dir_to_load, f) for f in os.listdir(dir_to_load)
         if "png" in f
      ]
      self.resetImageNumbers()
      self.num_images_loaded = 0
      
   def resetImageNumbers(self):
      self.image_numbers = [
         i for i in range(len(self.image_names))
      ]

   def __iter__(self):
      self.resetImageNumbers()
      return self

   def __next__(self):
      images_to_use = []
      if (
            (len(self.image_numbers) < self.batch_size) \
            and (self.num_epochs_elapsed < self.max_num_epochs)
         ):
         self.resetImageNumbers()
         self.num_epochs_elapsed += 1

      if self.num_epochs_elapsed == self.max_num_epochs:
         self.resetImageNumbers()
         raise StopIteration

      for b in range(self.batch_size):
         image_number_index = np.random.randint(0, len(self.image_numbers))
         image_name_index = self.image_numbers.pop(image_number_index)
         images_to_use.append(self.image_names[image_name_index])

      pngs = []
      for image_name in images_to_use:
         pngs.append(self.preprocess(Image.open(image_name)))
      batch = np.stack(pngs)
      self.num_images_loaded += self.batch_size

      return batch

   @property
   def epochs(self):
      return self.num_epochs_elapsed

def scale_and_shift_pixels(image_in):
   image_out = 2.*np.array(image_in, dtype=np.float32)/255. - 1.
   return image_out

def parse_image_tf(filename):
   image_str = tf.read_file(filename)
   image_decoded = tf.image.decode_png(image_str)
   image = 2.0*tf.image.convert_image_dtype(image_decoded, tf.float32) - 1

   return image

def upsample_np(fm_in):
   # fm_in --> feature map input [batch_size, h, w, c]
   old_h = fm_in.shape[1]
   old_w = fm_in.shape[2]
   new_shape = np.array([1, 2, 2, 1])*fm_in.shape
   batch_size = fm_in.shape[0]
   num_channels = fm_in.shape[3]

   temp_fm = np.zeros(new_shape)
   print(new_shape, temp_fm.shape)
   for i in range(old_h):
      for j in range(old_w):
         for k in range(batch_size):
            for l in range(num_channels):
               temp_fm[k, (2*i):(2*i + 2), (2*j):(2*j + 2), l] = fm_in[k, i, j, l]

   return temp_fm

def get_save_folder_name():
   return str(datetime.datetime.today()).replace(":", "-").replace(" ", "-")

def upsample_tf(x):
   print("shape of x: ", x.shape)
   k = np.array([[1, 1], [1, 1]], dtype=np.float32)
   inH = x.shape[1]
   inW = x.shape[2]
   minorDim = x.shape[3]

   upx = 2
   upy = 2
   kernelH, kernelW = k.shape
   x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
   #x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
   x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
   x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

   # Convolve with filter.
   x = tf.transpose(x, [0, 3, 1, 2])
   x = tf.reshape(x, [-1, 1, inH * upy, inW * upx])
   w = tf.constant(k[:, :, np.newaxis, np.newaxis], dtype=x.dtype)
   x = tf.pad(x, [[0, 0], [0, 0], [1, 0], [1, 0]])
   y = tf.nn.conv2d(x, w, padding='VALID', data_format='NCHW', strides=1)
   #y = tf.nn.conv2d(tf.transpose(x, [0, 2, 3, 1]), w, padding='SAME')
   z = tf.reshape(y, [-1, minorDim, inH * upy, inW * upx])
   z = tf.transpose(z, [0, 2, 3, 1])
   print("shape of z:", z.shape)
   return z

if __name__ == "__main__":
   # a = tf.get_variable(name="thing", shape=[3,3,3,5], dtype=tf.float32, initializer=tf.initializers.random_normal())
   # ops, b = upsample_assignop_tf(a)
   # sess = tf.Session()
   # sess.run(tf.global_variables_initializer())
   # print("a")
   # print(sess.run(a[0, :, :, 0]))
   # print("b pre op")
   # print(sess.run(b[0, :, :, 0]))
   # sess.run(ops)
   # print("ops[0]:", ops[0])
   # print("b post op")
   # print(sess.run(b[0, :, :, 0]))
   # print("derivative")
   # print(sess.run(tf.gradients(0.5*tf.reduce_sum(a*a), [a])[0]))
   # print(sess.run(tf.gradients(0.5*tf.reduce_sum(b*b), [a])[0]))

   import matplotlib.pyplot as plt
   loader = JankyImageLoader(
      "/home/jg/Documents/stylegan/ffhq-dataset/thisfolderisjustforkeras/images256x256",
      batch_size=128,
      preprocess=scale_and_shift_pixels
   )

   for j in loader:
      print(j.shape)
      print(loader.num_images_loaded)
      print(loader.epochs)

      plt.figure()
      plt.imshow((np.array(j[0, :, :, :], dtype=np.float32) + 1.0)/2.0)
      plt.show()
      #plt.close()