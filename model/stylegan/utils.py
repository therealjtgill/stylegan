import numpy as np
import tensorflow as tf

def scale_and_shift_pixels(image_in):
   image_out = np.array(2.*image_in/255. - 1, dtype=np.float32)
   return image_out

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
