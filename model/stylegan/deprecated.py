def evalGenerator(self, W_in):
   with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as vs:
      self.constant_input = tf.get_variable(
            "c_1",
            [4, 4, 256],
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
      )
      
      batch_size = tf.shape(W_in)[0]
      
      tiled_constant_input = tf.tile(
            tf.expand_dims(
               self.constant_input, axis=0
            ),
            [batch_size, 1, 1, 1]
      )
      
      print("tiled constant input:", tiled_constant_input)
      
      block_4_2 = self.styleBlock(
            tiled_constant_input,
            W_in,
            num_input_channels=256,
            #1
            num_output_channels=256,
            fm_dimension=4
      )
      
      self.b4 = block_4_2

      to_rgb_1 = self.toRgb(block_4_2, 4, 4, 256)
      
      self.r1 = to_rgb_1

      block_8_1 = self.styleBlock(
            block_4_2,
            W_in,
            num_input_channels=256,
            #2
            num_output_channels=256,
            fm_dimension=8,
            upsample=True
      )
      
      block_8_2 = self.styleBlock(
            block_8_1,
            W_in,
            num_input_channels=256,
            #3
            num_output_channels=256,
            fm_dimension=8
      )
      
      self.b8 = block_8_2

      to_rgb_2 = self.toRgb(block_8_2, 8, 8, 256) + self.upsample(to_rgb_1)

      self.r2 = to_rgb_2
      
      block_16_1 = self.styleBlock(
            block_8_2,
            W_in,
            num_input_channels=256,
            #4
            num_output_channels=256,
            fm_dimension=16,
            upsample=True
      )
      
      block_16_2 = self.styleBlock(
            block_16_1,
            W_in,
            num_input_channels=256,
            #5
            num_output_channels=256,
            fm_dimension=16,
      )
      
      to_rgb_3 = self.toRgb(block_16_2, 16, 16, 256) + self.upsample(to_rgb_2)
      
      block_32_1 = self.styleBlock(
            block_16_2,
            W_in,
            num_input_channels=256,
            #6
            num_output_channels=256,
            fm_dimension=32,
            upsample=True
      )
      
      block_32_2 = self.styleBlock(
            block_32_1,
            W_in,
            num_input_channels=256,
            #7
            num_output_channels=256,
            fm_dimension=32,
      )
      
      to_rgb_4 = self.toRgb(block_32_2, 32, 32, 256) + self.upsample(to_rgb_3)
      
      block_64_1 = self.styleBlock(
            block_32_2,
            W_in,
            num_input_channels=256,
            #8
            num_output_channels=256,
            fm_dimension=64,
            upsample=True
      )
      
      block_64_2 = self.styleBlock(
            block_64_1,
            W_in,
            num_input_channels=256,
            num_output_channels=256,
            fm_dimension=64,
      )
      print("block_64_2:", block_64_2)
      to_rgb_5 = self.toRgb(block_64_2, 64, 64, 256) + self.upsample(to_rgb_4)
      
      block_128_1 = self.styleBlock(
            block_64_2,
            W_in,
            num_input_channels=256,
            num_output_channels=256,
            fm_dimension=128,
            upsample=True
      )
      
      block_128_2 = self.styleBlock(
            block_128_1,
            W_in,
            num_input_channels=256,
            num_output_channels=128,
            fm_dimension=128,
      )
      
      to_rgb_6 = self.toRgb(block_128_2, 128, 128, 128) + self.upsample(to_rgb_5)
      
      block_256_1 = self.styleBlock(
            block_128_2,
            W_in,
            num_input_channels=128,
            num_output_channels=128,
            fm_dimension=256,
            upsample=True
      )
      
      block_256_2 = self.styleBlock(
            block_256_1,
            W_in,
            num_input_channels=128,
            num_output_channels=64,
            fm_dimension=256
      )
      
      to_rgb_7 = tf.nn.tanh(self.toRgb(block_256_2, 256, 256, 64) + self.upsample(to_rgb_6))
      
      return to_rgb_7


def evalDiscriminator(self, rgb_in, reuse):
   with tf.variable_scope("discriminator", reuse=reuse) as vs:
      disc_layer_outputs = []
      from_rgb_1 = self.fromRgb(rgb_in, 256, 256, 16, id=1)
      disc_layer_outputs.append(rgb_in)
      print("from rgb:", from_rgb_1)
      disc_layer_outputs.append(from_rgb_1)
      
      downsample_1 = self.downsample(from_rgb_1, 16, id=1)
      block_128_1 = self.discriminatorBlock(
            from_rgb_1,
            16,
            32,
            id=1
      )
      disc_layer_outputs.append(block_128_1)

      res_1 = downsample_1 + block_128_1
      print("res 1:", res_1)
      downsample_2 = self.downsample(res_1, 32, id=2)
      block_64_2 = self.discriminatorBlock(
            res_1,
            32,
            64,
            id=2
      )
      disc_layer_outputs.append(block_64_2)

      res_2 = downsample_2 + block_64_2
      
      print("res 2:", res_2)
      
      downsample_3 = self.downsample(res_2, 64, id=3)
      block_32_3 = self.discriminatorBlock(
            res_2,
            64,
            128,
            id=3
      )
      disc_layer_outputs.append(block_32_3)
      
      res_3 = downsample_3 + block_32_3
      
      downsample_4 = self.downsample(res_3, 128, id=4)
      block_16_4 = self.discriminatorBlock(
            res_3,
            128,
            256,
            id=4
      )
      disc_layer_outputs.append(block_16_4)
      
      res_4 = downsample_4 + block_16_4
      
      downsample_5 = self.downsample(res_4, 256, 256, id=5)
      block_8_5 = self.discriminatorBlock(
            res_4,
            256,
            256,
            id=5
      )
      disc_layer_outputs.append(block_8_5)
      
      res_5 = downsample_5 + block_8_5
      
      downsample_6 = self.downsample(res_5, 256, 256, id=6)
      block_4_6 = self.discriminatorBlock(
            res_5,
            256,
            256,
            id=6
      )
      disc_layer_outputs.append(block_4_6)
      
      res_6 = downsample_6 + block_4_6

      conv_w_a = tf.get_variable(
            "conv_w_disc_end_3x3",
            [3, 3, 256, 256],
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
      )
      
      conv_out_1 = tf.nn.leaky_relu(
            tf.nn.conv2d(res_6, conv_w_a, padding="SAME"),
            alpha=0.2
      )
      disc_layer_outputs.append(conv_out_1)
      
      conv_w_b = tf.get_variable(
            "conv_w_disc_end_4x4",
            [4, 4, 256, 256],
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
      )
      
      conv_out_2 = tf.nn.leaky_relu(
            tf.nn.conv2d(conv_out_1, conv_w_b, padding="VALID"),
            alpha=0.2
      )
      disc_layer_outputs.append(conv_out_2)

      batch_size = tf.shape(conv_out_2)[0]
      
      conv_outputs_flat = tf.reshape(conv_out_2, shape=[batch_size, -1])
      
      W_fc = tf.get_variable(
            "fully_connected_W_disc_end",
            [256, 1],
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
      )

      b_fc = tf.get_variable(
            "fully_connected_b_disc_end",
            [1],
            initializer=tf.initializers.random_normal(0, 0.4)
      )
      disc_layer_outputs.append(tf.matmul(conv_outputs_flat, W_fc) + b_fc)

      disc_out = tf.nn.sigmoid(
            tf.matmul(conv_outputs_flat, W_fc) + b_fc
      )
      self.disc_layer_outputs.append(disc_layer_outputs)

      return disc_out

def upsample(V_in):
   fm_size = tf.shape(V_in)
   b = fm_size[0]
   h = fm_size[1]
   w = fm_size[2]
   c = fm_size[3]
   V_in_a = tf.concat([V_in, V_in,], axis=2)
   V_in_b = tf.reshape(V_in_a, [b, 2*h, w, c])

   V_in_c = tf.transpose(V_in_b, perm=[0, 2, 1, 3])
   V_in_d = tf.concat([V_in_c, V_in_c], axis=2)
   V_out = tf.transpose(
       tf.reshape(V_in_d, [b, 2*h, 2*w, c]),
       perm=[0, 2, 1, 3]
   )
   return V_out