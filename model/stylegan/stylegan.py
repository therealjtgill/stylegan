
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import time
#from utils import scale_and_shift_pixels
import utils

np.set_printoptions(threshold=sys.maxsize)

#from keras.preprocessing.image import ImageDataGenerator

# In[2]:

class stylegan(object):
    def __init__(
        self,
        session,
        batch_size=8,
        output_resolution=256,
        gamma=10,
        use_r1_reg=True,
        use_pl_reg=True,
        disc_config=None,
        gen_config=None
    ):
        self.sess = session
        self.output_resolution = output_resolution
        self.num_style_blocks = 0
        self.num_to_rgbs = 0
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.gen_config = [
            (256, 256, 4),
            (256, 256, 8),
            (256, 256, 16),
            (256, 128, 32),
            (128, 128, 64),
            (128, 128, 128),
            (128, 64, 256)
        ]
        self.disc_config = [
            (256, 16),
            (16, 32),
            (32, 64),
            (64, 128),
            (128, 128),
            (128, 256),
            (256, 256)
        ]

        if gen_config is not None:
            self.gen_config = gen_config
        if disc_config is not None:
            self.disc_config = disc_config

        self.disc_layer_outputs = []
        
        self.conv_weights = []
        self.w_transforms = []
        
        self.latent_z = tf.random.normal(
            shape=[self.batch_size,256],
            stddev=1.0
        )
        
        self.latent_z_ph = tf.placeholder(
            shape=[self.batch_size, 256],
            dtype=tf.float32
        )

        self.true_images_ph = tf.placeholder(
            shape=[None, 256, 256, 3],
            dtype=tf.float32
        )
        self.fake_images_ph = tf.placeholder(
            shape=[self.batch_size, 256, 256, 3],
            dtype=tf.float32
        )

        self.latent_w = self.latentZMapper(self.latent_z)
        
        self.generator_out = self.evalGeneratorLoop(self.latent_w, self.gen_config)
        
        print(self.generator_out)
        
        e = tf.random.uniform(
            shape=[self.batch_size], minval=0., maxval=1., dtype=tf.float32
        )

        self.interp_images = tf.add(
            tf.einsum("b,bhwc->bhwc", e, self.generator_out),
            tf.einsum("b,bhwc->bhwc", (1. - e), self.true_images_ph)
        )
        
        self.disc_of_gen_out = self.evalDiscriminatorLoop(
            self.generator_out,
            False,
            self.disc_config
        )

        self.disc_of_truth_out = self.evalDiscriminatorLoop(
            self.true_images_ph,
            True,
            self.disc_config
        )
        self.disc_of_interp_out = self.evalDiscriminatorLoop(
            self.interp_images,
            True,
            self.disc_config
        )
        
        disc_grad_interp = tf.gradients(
           self.disc_of_interp_out,
           [self.interp_images]
        )[0]

        self.lipschitz_penalty = 10.*tf.reduce_mean(
           tf.square(
               tf.sqrt(
                   tf.reduce_sum(
                       disc_grad_interp*disc_grad_interp, axis=[1, 2, 3]
                   )
               ) - 1.
           )
        )

        self.wgan_disc_loss = tf.reduce_mean(
            self.disc_of_gen_out - self.disc_of_truth_out
        )

        self.disc_loss = self.wgan_disc_loss + self.lipschitz_penalty
        
        # self.r1_reg = 0.0
        # if use_r1_reg:
        #     disc_grad_truth = tf.gradients(
        #         self.disc_of_truth_out,
        #         [self.true_images_ph]
        #     )[0]
            
        #     self.r1_reg = (self.gamma/2)*tf.reduce_sum(
        #         tf.square(disc_grad_truth),
        #         axis=[1, 2, 3]
        #     )

        # self.gan_disc_loss = tf.add(
        #     tf.reduce_mean(-tf.log(tf.maximum(self.disc_of_truth_out, 1e-6))),    
        #     tf.reduce_mean(-tf.log(tf.maximum(1. - self.disc_of_gen_out, 1e-6)))
        # )
        
        # self.disc_loss = self.gan_disc_loss + tf.reduce_mean(self.r1_reg)
        
        self.disc_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.00025, beta1=0.0, beta2=0.99, epsilon=1e-8
        )
        
        self.disc_minimize = self.disc_optimizer.minimize(
            self.disc_loss,
            var_list=tf.trainable_variables(scope="discriminator")
        )

        # disc_comp_gradients = self.disc_optimizer.compute_gradients(self.disc_loss)

        # self.disc_gradients = [
        #     grad for grad, var in disc_comp_gradients
        #     if "discriminator" in var.name
        # ]
        
        # disc_grads_and_vars = [
        #     (grad, var) for grad, var in disc_comp_gradients
        #     if "discriminator" in var.name
        # ]
        # capped_grads = [
        #   (grad if grad is None else tf.clip_by_norm(grad, 1.0), var)
        #   for grad, var in disc_grads_and_vars
        # ]
        # self.disc_minimize_clip = self.disc_optimizer.apply_gradients(capped_grads)

        # self.disc_gradient_mags = [
        #     tf.sqrt(tf.reduce_sum(tf.square(g))) for g, v in capped_grads
        # ]

        self.wgan_gen_loss = tf.reduce_mean(-self.disc_of_gen_out)

        # self.gan_gen_loss = tf.reduce_mean(
        #     -tf.log(tf.maximum(self.disc_of_gen_out, 1e-6))
        # )
        
        self.gen_path_len_reg = 0.0
        if use_pl_reg:
            gen_grad_w = tf.gradients(
                tf.reduce_sum(
                    self.generator_out*tf.random_normal(
                        shape=tf.shape(self.generator_out),
                        dtype=tf.float32
                    )
                ),
                [self.latent_w]
            )[0]

            self.gen_grad_w_mag = tf.sqrt(
                tf.reduce_sum(
                    tf.square(gen_grad_w)
                )
            )

            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            m = ema.apply([self.gen_grad_w_mag])
            self.gen_a = ema.average(self.gen_grad_w_mag)

            self.gen_path_len_reg = tf.reduce_mean(
                tf.square(
                    self.gen_grad_w_mag - self.gen_a
                )
            )

        self.gen_loss = self.wgan_gen_loss + self.gen_path_len_reg

        self.gen_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.0005, beta1=0.1, beta2=0.99, epsilon=1e-8
        )

        self.gen_minimize = self.gen_optimizer.minimize(
            self.gen_loss,
            var_list=tf.trainable_variables(scope="generator")
        )

        self.mapper_loss = self.gen_loss
        self.mapper_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.01*0.0005, beta1=0.0, beta2=0.9
        )
        self.mapper_minimize = self.mapper_optimizer.minimize(
            self.mapper_loss,
            var_list=tf.trainable_variables(scope="mapper")
        )

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)
        
    def latentZMapper(self, Z_in, depth=8, reuse=False):
        result = None
        with tf.variable_scope("mapper", reuse=reuse) as vs:
            for i in range(depth):
                W = tf.get_variable(
                    "W_mapper_" + str(i),
                    [256, 256],
                    initializer=tf.contrib.layers.variance_scaling_initializer(
                        dtype=tf.float32
                    )
                )
                b = tf.get_variable(
                    "b_mapper_" + str(i),
                    [256,],
                    initializer=tf.initializers.random_normal(0, 0.4)
                )
                
                if i == 0:
                    result = tf.nn.leaky_relu(
                        tf.matmul(Z_in, W) + b,
                        alpha=0.2
                    )
                else:
                    result = tf.nn.leaky_relu(
                        tf.matmul(result, W) + b,
                        alpha=0.2
                    )
                
        return result
        
    def evalGeneratorLoop(self, W_in, config):
        # config = [
        #   (channel_in_size_1, chanel_out_size_1, k_1),
        #   (channel_in_size_2, chanel_out_size_2, k_2),
        #   ...
        # ]
        # Where k_i is the size of one of the sides of the output feature map.
        batch_size = tf.shape(W_in)[0]
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as vs:
            self.constant_input = tf.get_variable(
                "c_1",
                [4, 4, 256],
                initializer=tf.contrib.layers.variance_scaling_initializer(
                    dtype=tf.float32
                )
            )
            
            tiled_constant_input = tf.tile(
                tf.expand_dims(
                    self.constant_input, axis=0
                ),
                [batch_size, 1, 1, 1]
            )
            
            print("tiled constant input:", tiled_constant_input)

            c_in, c_out, k = config[0]
            block_input = tiled_constant_input
            #block_1 = None
            block_2 = self.styleBlock(
                block_input,
                W_in,
                num_input_channels=c_in,
                num_output_channels=c_out,
                fm_dimension=k
            )
            to_rgb = self.toRgb(block_2, c_out)

            for c_in, c_out, k in config[1:]:

                block_1 = self.styleBlock(
                    block_2,
                    W_in,
                    num_input_channels=c_in,
                    num_output_channels=c_in,
                    fm_dimension=k,
                    upsample=True
                )
                
                block_2 = self.styleBlock(
                    block_1,
                    W_in,
                    num_input_channels=c_in,
                    num_output_channels=c_out,
                    fm_dimension=k
                )

                to_rgb = self.toRgb(block_2, c_out) + utils.upsample_tf(to_rgb)
                
            c_in, c_out, k = config[-1]
            #rgb_out = tf.nn.tanh(to_rgb)
            rgb_out = to_rgb

            if rgb_out is None:
                print("rgb_out is none!")
                sys.exit(-1)

            return rgb_out
            
    def evalDiscriminatorLoop(self, rgb_in, reuse, config):
        with tf.variable_scope("discriminator", reuse=reuse) as vs:
            from_rgb_k, from_rgb_c_out = config[0]

            from_rgb = self.fromRgb(
                rgb_in, from_rgb_c_out, id=1
            )

            res = from_rgb
            var_id = 1
            for c_in, c_out in config[1:]:
                downsample = self.downsample(res, c_in, c_out, id=var_id)
                disc_block = self.discriminatorBlock(
                    res,
                    c_in,
                    c_out,
                    id=var_id
                )
                res = downsample + disc_block
                var_id += 1

            c_in, c_out = config[-1]
            conv_w_a = tf.get_variable(
                "conv_w_disc_end_3x3",
                [3, 3, c_in, c_out],
                initializer=tf.contrib.layers.variance_scaling_initializer(
                    dtype=tf.float32
                )
            )
            
            conv_out_1 = tf.nn.leaky_relu(
                tf.nn.conv2d(res, conv_w_a, padding="SAME"),
                alpha=0.2
            )
            
            conv_w_b = tf.get_variable(
                "conv_w_disc_end_4x4",
                [4, 4, c_in, c_out],
                initializer=tf.contrib.layers.variance_scaling_initializer(
                    dtype=tf.float32
                )
            )
            
            conv_out_2 = tf.nn.leaky_relu(
                tf.nn.conv2d(conv_out_1, conv_w_b, padding="VALID"),
                alpha=0.2
            )

            batch_size = tf.shape(conv_out_2)[0]
            
            conv_outputs_flat = tf.reshape(conv_out_2, shape=[batch_size, -1])
            
            W_fc = tf.get_variable(
                "fully_connected_W_disc_end",
                [c_out, 1],
                initializer=tf.contrib.layers.variance_scaling_initializer(
                    dtype=tf.float32
                )
            )

            b_fc = tf.get_variable(
                "fully_connected_b_disc_end",
                [1],
                initializer=tf.initializers.random_normal(0, 0.4)
            )

            #disc_out = tf.nn.sigmoid(
            #    tf.matmul(conv_outputs_flat, W_fc) + b_fc
            #)
            disc_out = tf.matmul(conv_outputs_flat, W_fc) + b_fc

            return disc_out
            
    def discriminatorBlock(self, V_in, num_input_channels, num_output_channels, downsample=True, id=0):
        # V_in        --> [batch_size, height, width, c_in]
        # latent_w    --> [batch_size, 512]
        #    c_in  = number of input feature maps
        #    num_output_channels = number of output feature maps
        c_in = num_input_channels
        c_out = num_output_channels

        conv_weight_a = tf.get_variable(
            "conv_w_disc_a_" + str(id),
            [3, 3, c_in, c_in],
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        )
        
        V_out_a = tf.nn.leaky_relu(
            tf.nn.conv2d(V_in, conv_weight_a, padding="SAME"),
            alpha=0.2
        )
        
        conv_weight_b = tf.get_variable(
            "conv_w_disc_b_" + str(id),
            [3, 3, c_in, c_out],
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        )
        
        V_out_b = tf.nn.leaky_relu(
            tf.nn.conv2d(V_out_a, conv_weight_b, padding="SAME"),
            alpha=0.2
        )
        
        # Bad ID deconfliction - assumes that regular ID's will never exceed 100.
        V_out = self.downsample(V_out_b, c_out, c_out, id+100)
        
        return V_out
            
    def styleBlock(self, V_in, latent_w, num_input_channels, num_output_channels, fm_dimension, upsample=False):
        # V_in        --> [batch_size, height, width, num_input_channels]
        # latent_w    --> [batch_size, 512]
        #    num_input_channels  = number of input feature maps
        #    num_output_channels = number of output feature maps
        self.num_style_blocks += 1
        c_in = num_input_channels
        c_out = num_output_channels
        
        if upsample:
            V_in = utils.upsample_tf(V_in)
        
        A = tf.get_variable(
            "A_style" + str(self.num_style_blocks),
            [256, c_in],
            initializer=tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32
            )
        )
        
        conv_weight = tf.get_variable(
            "conv_w_style" + str(self.num_style_blocks),
            [3, 3, c_in, c_out],
            initializer=tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32
            )
        )
        
        conv_bias = tf.get_variable(
            "conv_b_style" + str(self.num_style_blocks),
            [1, fm_dimension, fm_dimension, c_out],
            initializer=tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32
            )
        )
        
        # Affine transformation of latent space vector.
        scale = tf.matmul(latent_w, A)
        
        # Scale input feature map acros input channels by the affine transformation
        V_in_scaled = tf.einsum("bi,bhwi->bhwi", scale, V_in)
        
        V_out = tf.nn.conv2d(V_in_scaled, conv_weight, padding="SAME")

        # This increases the number of weights by a factor of batch_size,
        # which is weird.
        modul_conv_weight = tf.einsum("bj,hwjc->bhwjc", scale, conv_weight)
        sigma_j = 1./tf.sqrt(
            tf.reduce_sum(
                tf.square(modul_conv_weight),
                axis=[1, 2, 3]
            ) + 1e-6
        )

        # Need to add biases and broadcast noise.
        #V_out_scaled = tf.nn.leaky_relu(
        #    tf.einsum("bhwj,bj->bhwj", V_out, sigma_j) + conv_bias,
        #    alpha=0.2
        #)
        V_out_scaled = tf.einsum("bhwj,bj->bhwj", V_out, sigma_j) + conv_bias

        b = tf.random.normal(shape=latent_w.shape, dtype=tf.float32)

        B = tf.get_variable(
            "B_style" + str(self.num_style_blocks),
            [256, c_out],
            initializer=tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32
            )
        )

        noise_input = tf.expand_dims(tf.expand_dims(tf.matmul(b, B), 1), 1)

        V_out_noised = tf.nn.leaky_relu(
            V_out_scaled + noise_input,
            alpha=0.2
        )

        return V_out_noised
    
    def upsample(self, V_in):
        # Tested with the channel dimension.
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

        #V_out = tf.image.resize_bilinear(V_in, (2*h, 2*w))
        return V_out
    
    def downsample(self, V_in, num_input_channels, num_output_channels=None, id=0):
        c_in = num_input_channels
        c_out = num_output_channels

        V_in_shape = tf.shape(V_in)
        h = V_in_shape[1]
        w = V_in_shape[2]
        V_larger = V_in
        if c_out is None:
            c_out = 2*c_in
    
        if c_out != c_in:
            channel_increase = tf.get_variable(
                "channel_increaser" + str(id),
                [1, 1, c_in, c_out]
            )
            
            V_larger = tf.nn.leaky_relu(
                tf.nn.conv2d(V_in, channel_increase, padding="SAME"),
                alpha=0.2
            )
        
        #V_out = tf.nn.max_pool2d(V_larger, ksize=2, strides=2, padding="VALID")
        V_out = tf.nn.avg_pool2d(V_larger, ksize=2, strides=2, padding="VALID")
        #V_out = tf.image.resize_bilinear(V_larger, (h//2, w//2))
        return V_out
    
    def toRgb(self, V_in, c_in):
        '''
        Convert an NxNxC output block to an RGB image with dimensions
        NxNx3.
        '''
        
        self.num_to_rgbs += 1

        to_rgb = tf.get_variable(
            "to_rgb" + str(self.num_to_rgbs),
            [1, 1, c_in, 3],
            initializer=tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32
            )
        )
        #print("###############")
        #print("V_in:", V_in)
        #print("to_rgb:", to_rgb)
        # rgb_out = tf.nn.tanh(
        #     tf.nn.conv2d(V_in, to_rgb, padding="SAME")
        # )
        rgb_out = tf.nn.conv2d(V_in, to_rgb, padding="SAME")
        
        return rgb_out
    
    def fromRgb(self, V_in, c, id=0):
        '''
        Convert an NxNx3 output block to an feature map with dimensions
        NxNxC.
        '''

        from_rgb = tf.get_variable(
            "from_rgb" + str(id),
            [1, 1, 3, c],
            initializer=tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32
            )
        )
        
        # feature_map_out = tf.nn.tanh(
        #     tf.nn.conv2d(V_in, from_rgb, padding="SAME")
        # )
        feature_map_out = tf.nn.conv2d(V_in, from_rgb, padding="SAME")

        return feature_map_out

    def trainDiscriminatorBatch(self, true_images):
        fetches = [
            self.disc_loss,
            self.disc_minimize,
            self.disc_of_gen_out,
            self.disc_of_truth_out,
        ]
        feeds = {
            self.true_images_ph: true_images
        }
        
        loss, _, fake_pred, real_pred = self.sess.run(fetches, feed_dict=feeds)
        return loss, fake_pred, real_pred

    def trainGeneratorBatch(self):
        fetches = [
            self.gen_loss,
            self.gen_minimize,
            self.mapper_minimize,
        ]

        gen_loss, _1, _2 = self.sess.run(fetches, feed_dict={})

        return gen_loss

    def runGeneratorBatch(self):
        gen_out = self.sess.run(self.generator_out)

        return gen_out

    def saveParams(self, save_dir, global_step):
        
        self.saver.save(self.sess, save_dir, global_step=global_step)

    def loadParams(self, load_dir):
        try:
            self.saver.restore(self.sess, load_dir)
        except Exception as e:
            print("Could not load checkpoint with name ", load_dir,
                  "received exception", str(e)
            )

# In[3]:

if __name__ == "__main__":
    batch_size = 8
    sess = tf.Session()

    s = stylegan(sess, gamma=5, batch_size=batch_size, use_r1_reg=True, use_pl_reg=False)
    sess.run(tf.global_variables_initializer())
    print("Initialized variables")
    saver = tf.train.Saver(max_to_keep=3)
    print("Initialized saver")

    # print("tensorflow variables:")
    # sorted_varnames = sorted([v.name for v in tf.global_variables()])
    # for var in sorted_varnames:
    #     print(var)
    # sys.exit(-1)

    # In[ ]:

    print("Defining generators")
    gan_data_generator = ImageDataGenerator(
        rescale=1,
        preprocessing_function=utils.scale_and_shift_pixels,
        horizontal_flip=True,
    )

    data_flow = gan_data_generator.flow_from_directory(
        '/home/jg/Documents/stylegan/ffhq-dataset/thisfolderisjustforkeras',
        target_size=(256, 256),
        batch_size=batch_size,
        shuffle=True
    )
    print("Initialized generators")
    train = True

    num_images = 0
    num_epochs = 0
    disc_losses = [-1,]
    gen_losses = [-1,]
    iterations = 0
    start = time.time()

    losses_filename = "losses.dat"
    losses_file = open(losses_filename, "w")

    #saver.save(sess, "stylegan2_ckpt", global_step=123456)
    summary_writer = tf.summary.FileWriter("logs", sess.graph)

    if train:
        for x, _ in data_flow:
            if iterations == 4:
                break
            print("min, max of training image:", np.min(x[0, :, :, :]), np.max(x[0, :, :, :]))

            disc_start_time = time.time()
            if num_epochs == 20:
                break
            if x.shape[0] != batch_size:
                num_epochs += 1
                continue

            loss, fake_pred, real_pred = s.trainDiscriminatorBatch(x)

            disc_losses.append(loss)
            print("discriminator run/loss time:", time.time() - disc_start_time)
            print("Real prediction:", real_pred, " Fake prediction: ", fake_pred)

            # disc_weight_names = [w.name for w in tf.trainable_variables(scope="discriminator")]
            # weights = sess.run(disc_weight_names)

            # with open("discriminator_layer_output_mags_" + str(iterations) + ".txt", "w") as f:
            #     for i, disc_call in enumerate(layer_outs):
            #         f.write("discriminator call #" + str(i) + "\n\n\n")
            #         for j, layer_out in enumerate(disc_call):
            #             f.write("layer " + str(j) + "\n")
            #             f.write(str(np.sqrt(np.sum(np.square(layer_out)))) + "\n")

            # with open("discriminator_layer_outputs_" + str(iterations) + ".txt", "w") as f:
            #     for i, disc_call in enumerate(layer_outs):
            #         f.write("discriminator call #" + str(i) + "\n\n\n")
            #         for j, layer_out in enumerate(disc_call):
            #             f.write("layer " + str(j) + "\n")
            #             f.write(str(layer_out) + "\n")

            # with open("discriminator_weights_" + str(iterations) + ".txt", "w") as f:
            #     for w, n in zip(weights, disc_weight_names):
            #         f.write(n + "\n\n" + str(w) + "\n")

            # img_gen_start_time = time.time()
            # raw_generated_images = s.runGeneratorBatch()

            # for i in range(min(raw_generated_images.shape[0], 5)):
            #     plt.figure()
            #     plt.imshow(np.array((raw_generated_images[i, :, :, :] + 1.)/2.))
            #     print((raw_generated_images[i, :, :, :] + 1.)/2.)
            #     plt.savefig('generated_image_' + str(iterations) + '_' + str(i) + '.png')
            #     plt.close()

            # print("Took ", (time.time() - img_gen_start_time), " seconds to generate/save those images")

            iterations += 1

            if iterations % 5 == 0:
                img_gen_start_time = time.time()
                gen_loss = s.trainGeneratorBatch()
                gen_losses.append(gen_loss)

                for i in range(min(raw_generated_images.shape[0], 5)):
                    plt.figure()
                    plt.imshow(np.array((raw_generated_images[i, :, :, :] + 1.)/2.))
                    #print((raw_generated_images[i, :, :, :] + 1.)/2.)
                    plt.savefig('generated_image_' + str(iterations) + '_' + str(i) + '.png')
                    plt.close()

                print("Took ", (time.time() - img_gen_start_time), " seconds to generate/save those images")

            #iterations += 1
            losses_file.write(str(iterations) + " " + str(disc_losses[-1]) + " " + str(gen_losses[-1]) + "\n")
            print("num iterations:", iterations, "disc loss:", disc_losses[-1], "gen loss:", gen_losses[-1], "time elapsed:", time.time() - start)
            start = time.time()
