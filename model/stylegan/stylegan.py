
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import time

from keras.preprocessing.image import ImageDataGenerator


# In[2]:


class stylegan(object):
    def __init__(self, session, batch_size=64, output_resolution=256, gamma=10):
        self.sess = session
        self.output_resolution = output_resolution
        self.num_style_blocks = 0
        self.num_to_rgbs = 0
        self.num_from_rgbs = 0
        self.num_downsamples = 0
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.conv_weights = []
        self.w_transforms = []
        
        self.latent_z = tf.random.normal(
            shape=[self.batch_size,512],
            stddev=1.0
        )

        self.latent_w = self.latentZMapper(self.latent_z)
        
        self.true_images_ph = tf.placeholder(
            shape=[None, 256, 256, 3],
            dtype=tf.float32
        )
        
        self.generator_out = self.evalGenerator(self.latent_w)
        
        e = tf.random.uniform(
            shape=[self.batch_size], minval=0., maxval=1., dtype=tf.float32
        )
        
        print(self.generator_out)
        
        self.interp_images = tf.add(
            tf.einsum("b,bhwc->bhwc", e, self.generator_out),
            tf.einsum("b,bhwc->bhwc", (1. - e), self.true_images_ph)
        )
        
        self.disc_of_gen_out = self.evalDiscriminator(self.generator_out)
        self.disc_of_truth_out = self.evalDiscriminator(self.true_images_ph)
        self.disc_of_interp_out = self.evalDiscriminator(self.interp_images)
        
        disc_grad_truth = tf.gradients(
            self.disc_of_truth_out,
            [self.true_images_ph]
        )[0]
        
        self.r1_reg = (self.gamma/2)*tf.reduce_sum(
            tf.square(disc_grad_truth),
            axis=[1, 2, 3]
        )
        
        #disc_grad_interp = tf.gradients(
        #    self.disc_of_interp_out,
        #    [self.interp_images]
        #)[0]

        #self.lipschitz_penalty = tf.reduce_mean(
        #    tf.square(
        #        tf.sqrt(
        #            tf.reduce_sum(
        #                disc_grad_interp*disc_grad_interp, axis=[1, 2, 3]
        #            )
        #        ) - 1.
        #    )
        #)
        
        self.wgan_disc_loss = tf.reduce_mean(
            self.disc_of_gen_out - self.disc_of_truth_out
        )
        
        self.gan_disc_loss = -tf.reduce_mean(
            tf.add(
                -tf.log(tf.maximum(self.disc_of_truth_out, 1e-6)),
                -tf.log(tf.maximum(1. - self.disc_of_gen_out, 1e-6))
            )
        )
        
        #self.disc_loss = self.wgan_disc_loss# + self.lipschitz_penalty
        self.disc_loss = self.gan_disc_loss + tf.reduce_mean(self.r1_reg)
        
        self.disc_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.002, beta1=0, beta2=0.9
        )
        self.disc_minimize = self.disc_optimizer.minimize(
            self.disc_loss,
            var_list=[v for v in tf.global_variables() if "discriminator" in v.name]
        )
        
        self.wgan_gen_loss = tf.reduce_mean(-self.disc_of_gen_out)

        self.gan_gen_loss = tf.reduce_mean(
            -tf.log(tf.maximum(self.disc_of_gen_out, 1e-6))
        )

        self.gen_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.002, beta1=0, beta2=0.9
        )
        self.gen_minimize = self.gen_optimizer.minimize(
            self.gan_gen_loss,
            var_list=[v for v in tf.global_variables() if "generator" in v.name]
        )

        self.mapper_loss = self.gan_gen_loss
        self.mapper_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.01*0.002, beta1=0, beta2=0.9
        )
        self.mapper_minimize = self.mapper_optimizer.minimize(
            self.mapper_loss
        )
        
    def latentZMapper(self, Z_in, depth=8):
        result = None
        with tf.variable_scope("mapper") as vs:
            for i in range(depth):
                W = tf.get_variable(
                    "W_mapper_" + str(i),
                    [512, 512],
                    initializer=tf.initializers.random_normal(stddev=0.3)
                )
                b = tf.get_variable(
                    "b_mapper_" + str(i),
                    [512,],
                    initializer=tf.initializers.random_normal(stddev=0.3)
                )
                
                if i == 0:
                    result = tf.nn.relu(tf.matmul(Z_in, W) + b)
                else:
                    result = tf.nn.relu(tf.matmul(result, W) + b)
                
        return result
        
    def evalGenerator(self, W_in):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as vs:
            self.constant_input = tf.get_variable(
                "c_1",
                [4, 4, 256],
                initializer=tf.initializers.random_normal
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
            
    def evalDiscriminator(self, rgb_in):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as vs:
            from_rgb_1 = self.fromRgb(rgb_in, 256, 256, 16)
            
            print("from rgb:", from_rgb_1)
            
            downsample_1 = self.downsample(from_rgb_1, 16)
            block_128_1 = self.discriminatorBlock(
                from_rgb_1,
                16,
                32,
                id=1
            )
            
            res_1 = downsample_1 + block_128_1
            print("res 1:", res_1)
            downsample_2 = self.downsample(res_1, 32)
            block_64_2 = self.discriminatorBlock(
                res_1,
                32,
                64,
                id=2
            )
            
            res_2 = downsample_2 + block_64_2
            
            print("res 2:", res_2)
            
            downsample_3 = self.downsample(res_2, 64)
            block_32_3 = self.discriminatorBlock(
                res_2,
                64,
                128,
                id=3
            )
            
            res_3 = downsample_3 + block_32_3
            
            downsample_4 = self.downsample(res_3, 128)
            block_16_4 = self.discriminatorBlock(
                res_3,
                128,
                256,
                id=4
            )
            
            res_4 = downsample_4 + block_16_4
            
            downsample_5 = self.downsample(res_4, 256, 256)
            block_8_5 = self.discriminatorBlock(
                res_4,
                256,
                256,
                id=5
            )
            
            res_5 = downsample_5 + block_8_5
            
            downsample_6 = self.downsample(res_5, 256, 256)
            block_4_6 = self.discriminatorBlock(
                res_5,
                256,
                256,
                id=6
            )
            
            res_6 = downsample_6 + block_4_6
            
            #downsample_7 = self.downsample(res_6, 512, 512)
            #block_4_7 = self.discriminatorBlock(
            #    res_6,
            #    512,
            #    512
            #)
            #
            #res_7 = downsample_7 + block_4_7
            
            #print("res 7:", res_7)
            
            conv_w_a = tf.get_variable(
                "conv_w_disc_end_3x3",
                [3, 3, 256, 256],
                initializer=tf.initializers.orthogonal
            )
            
            conv_out_1 = tf.nn.leaky_relu(
                tf.nn.conv2d(res_6, conv_w_a, padding="SAME"),
                alpha=0.2
            )
            
            conv_w_b = tf.get_variable(
                "conv_w_disc_end_4x4",
                [4, 4, 256, 256],
                initializer=tf.initializers.orthogonal
            )
            
            conv_out_2 = tf.nn.leaky_relu(
                tf.nn.conv2d(conv_out_1, conv_w_b, padding="VALID"),
                alpha=0.2
            )
            
            batch_size = tf.shape(conv_out_2)[0]
            
            conv_outputs_flat = tf.reshape(conv_out_2, shape=[batch_size, -1])
            
            W_fc = tf.get_variable(
                "fully_connected_W_disc_end",
                [256, 1],
                initializer=tf.initializers.orthogonal
            )
            
            b_fc = tf.get_variable(
                "fully_connected_b_disc_end",
                [1],
                initializer=tf.initializers.random_normal
            )
            
            #disc_out = tf.nn.leaky_relu(
            #    tf.matmul(conv_outputs_flat, W_fc) + b_fc
            #)
            
            disc_out = tf.nn.sigmoid(
                tf.matmul(conv_outputs_flat, W_fc) + b_fc
            )

            return disc_out
            
    def discriminatorBlock(self, V_in, num_input_channels, num_output_channels, downsample=True, id=0):
        # V_in        --> [batch_size, height, width, num_input_channels]
        # latent_w    --> [batch_size, 512]
        #    num_input_channels  = number of input feature maps
        #    num_output_channels = number of output feature maps
        
        conv_weight_a = tf.get_variable(
            "conv_w_disc_a_" + str(id),
            [3, 3, num_input_channels, num_input_channels],
            initializer=tf.initializers.orthogonal
        )
        
        V_out_a = tf.nn.leaky_relu(
            tf.nn.conv2d(V_in, conv_weight_a, padding="SAME"),
            alpha=0.2
        )
        
        conv_weight_b = tf.get_variable(
            "conv_w_disc_b_" + str(id),
            [3, 3, num_input_channels, num_output_channels],
            initializer=tf.initializers.orthogonal
        )
        
        V_out_b = tf.nn.leaky_relu(
            tf.nn.conv2d(V_out_a, conv_weight_b, padding="SAME"),
            alpha=0.2
        )
        
        V_out = self.downsample(V_out_b, num_output_channels, num_output_channels)
        
        return V_out
            
    def styleBlock(self, V_in, latent_w, num_input_channels, num_output_channels, fm_dimension, upsample=False):
        # V_in        --> [batch_size, height, width, num_input_channels]
        # latent_w    --> [batch_size, 512]
        #    num_input_channels  = number of input feature maps
        #    num_output_channels = number of output feature maps
        self.num_style_blocks += 1
        
        if upsample:
            V_in = self.upsample(V_in)
        
        A = tf.get_variable(
            "A_style" + str(self.num_style_blocks),
            [512, num_input_channels],
            #[512, num_output_channels],
            #initializer=tf.initializers.orthogonal
            initializer=tf.initializers.random_normal
        )
        
        conv_weight = tf.get_variable(
            "conv_w_style" + str(self.num_style_blocks),
            [3, 3, num_input_channels, num_output_channels],
            #initializer=tf.initializers.orthogonal
            initializer=tf.initializers.random_normal
        )
        
        conv_bias = tf.get_variable(
            "conv_b_style" + str(self.num_style_blocks),
            [1, fm_dimension, fm_dimension, num_output_channels],
            initializer=tf.initializers.random_normal
        )
        
        # Affine transformation of latent space vector.
        scale = tf.matmul(latent_w, A)
        
        # Scale input feature map acros input channels by the affine transformation
        # of the latent space input.
        #print("##########")
        #print("scale input feature map")
        #print("scale", scale)
        #print("V_in", V_in)
        V_in_scaled = tf.einsum("bi,bhwi->bhwi", scale, V_in)
        
        V_out = tf.nn.conv2d(V_in_scaled, conv_weight, padding="SAME")
        #print("V_out:", V_out)
        # This increases the number of weights by a factor of batch_size,
        # which is weird.
        #print("calculate sigma_j")
        #print("scale", scale)
        #print("conv_weight", conv_weight)
        #modul_conv_weight = tf.einsum("bc,hwjc->bhwjc", scale, conv_weight)
        modul_conv_weight = tf.einsum("bj,hwjc->bhwjc", scale, conv_weight)
        sigma_j = 1./tf.sqrt(tf.reduce_sum(tf.square(modul_conv_weight), axis=[1, 2, 3]) + 1e-6)

        #print("calculate output")
        #print("V_in_scaled", V_in_scaled)
        #print("sigma_j", sigma_j)
        # Need to add biases and broadcast noise.
        V_out_scaled = tf.nn.leaky_relu(
            tf.einsum("bhwj,bj->bhwj", V_out, sigma_j) + conv_bias,
            alpha=0.2
        )

        return V_out_scaled
    
    def upsample(self, V_in):
        # Tested with the channel dimension.
        fm_size = tf.shape(V_in)
        b = fm_size[0]
        h = fm_size[1]
        w = fm_size[2]
        c = fm_size[3]
        # V_in_a = tf.concat([V_in, V_in,], axis=2)
        # V_in_b = tf.reshape(V_in_a, [b, 2*h, w, c])

        # V_in_c = tf.transpose(V_in_b, perm=[0, 2, 1, 3])
        # V_in_d = tf.concat([V_in_c, V_in_c], axis=2)
        # V_out = tf.transpose(
        #     tf.reshape(V_in_d, [b, 2*h, 2*w, c]),
        #     perm=[0, 2, 1, 3]
        # )
        
        V_out = tf.image.resize_bilinear(V_in, (2*h, 2*w))
        return V_out
    
    def downsample(self, V_in, input_channels, output_channels=None):
        self.num_downsamples += 1
        V_in_shape = tf.shape(V_in)
        h = V_in_shape[1]
        w = V_in_shape[2]
        if output_channels is None:
            output_channels = 2*input_channels
        
        channel_increase = tf.get_variable(
            "channel_increaser" + str(self.num_downsamples),
            [1, 1, input_channels, output_channels]
        )
        
        V_larger = tf.nn.relu(
            tf.nn.conv2d(V_in, channel_increase, padding="SAME")
        )
        
        V_out = tf.nn.max_pool2d(V_larger, ksize=2, strides=2, padding="VALID")
        #V_out = tf.image.resize_bilinear(V_larger, (h//2, w//2))
        return V_out
    
    def toRgb(self, V_in, h, w, c):
        '''
        Convert an NxNxC output block to an RGB image with dimensions
        NxNx3.
        '''
        
        self.num_to_rgbs += 1

        to_rgb = tf.get_variable(
            "to_rgb" + str(self.num_to_rgbs),
            [h, w, c, 3],
            initializer=tf.initializers.random_normal
        )
        #print("###############")
        #print("V_in:", V_in)
        #print("to_rgb:", to_rgb)
        rgb_out = tf.nn.tanh(
            tf.nn.conv2d(V_in, to_rgb, padding="SAME")
        )
        
        return rgb_out
    
    def fromRgb(self, V_in, h, w, c):
        '''
        Convert an NxNx3 output block to an feature map with dimensions
        NxNxC.
        '''
        
        self.num_from_rgbs += 1
        
        from_rgb = tf.get_variable(
            "from_rgb" + str(self.num_from_rgbs),
            [h, w, 3, c],
            initializer=tf.initializers.random_normal
        )
        
        feature_map_out = tf.nn.relu(
            tf.nn.conv2d(V_in, from_rgb, padding="SAME")
        )
        
        return feature_map_out


# In[3]:

batch_size = 8
sess = tf.Session()


s = stylegan(sess, batch_size=batch_size)
sess.run(tf.global_variables_initializer())
#sys.exit()

# In[ ]:

def scale_and_shift_pixels(image_in):
    image_out = np.array(2.*image_in/255. - 1, dtype=np.float32)
    return image_out

gan_data_generator = ImageDataGenerator(
    rescale=1,
    preprocessing_function=scale_and_shift_pixels,
    horizontal_flip=True,
)

data_flow = gan_data_generator.flow_from_directory(
    '/home/jg/Documents/stylegan/ffhq-dataset/thisfolderisjustforkeras',
    target_size=(256, 256),
    batch_size=batch_size,
    shuffle=True
)

train = True

num_images = 0
num_epochs = 0
disc_losses = [-1,]
gen_losses = [-1,]
iterations = 0
start = time.time()

# raw_generated_images = sess.run(s.generator_out)

# weird_things = sess.run([s.b4, s.b8, s.r1, s.r2])
# print(weird_things)

# for i in range(min(raw_generated_images.shape[0], 5)):
#     plt.figure()
#     plt.imshow(np.array((raw_generated_images[i, :, :, :] + 1.)/2.))
#     plt.savefig('generated_image_test' + '.png')
#     plt.close()

if train:
    for x, y in data_flow:
        #num_images += x.shape[0]
        #print(num_images)
        if num_epochs == 20:
            break
        if x.shape[0] != batch_size:
            num_epochs += 1
            continue
        fetches = [s.disc_loss, s.disc_minimize]
        feeds = {s.true_images_ph: x}
        
        loss, _ = sess.run(fetches, feed_dict=feeds)
        disc_losses.append(loss)
        
        iterations += 1

        # img_gen_start_time = time.time()
        # raw_generated_images = sess.run(s.generator_out)

        # for i in range(min(raw_generated_images.shape[0], 5)):
        #     plt.figure()
        #     plt.imshow(np.array((raw_generated_images[i, :, :, :] + 1.)/2.))
        #     print((raw_generated_images[i, :, :, :] + 1.)/2.)
        #     plt.savefig('generated_image_' + str(iterations) + '_' + str(i) + '.png')
        #     plt.close()

        # print("Took ", (time.time() - img_gen_start_time), " seconds to generate/save those images")

        if iterations % 5 == 0:
            fetches = [s.gan_gen_loss, s.gen_minimize, s.mapper_minimize]
            feeds = {}

            gen_loss, _1, _2 = sess.run(fetches, feed_dict=feeds)
            gen_losses.append(gen_loss)

            #raw_generated_images = sess.run(s.generator_out)

            #for i in range(min(raw_generated_images.shape[0], 5)):
            #    plt.figure()
            #    plt.imshow(np.array(raw_generated_images[i, :, :, :], dtype=np.int32))
            #    plt.savefig('generated_image_' + str(iterations) + '_' + str(i) + '.png')
            #    plt.close()

        print("num iterations:", iterations, "disc loss:", disc_losses[-1], "gen loss:", gen_losses[-1], "time elapsed:", time.time() - start)
        start = time.time()

# In[ ]:


a = tf.constant(
    np.array(
        [
            [[1,2,3],
             [4,5,6],
             [7,8,9]
            ],
            [[-1,-2,-3],
             [-4,-5,-6],
             [-7,-8,-9]
            ],
            [[1.2,2.2,3.2],
             [4.2,5.2,6.2],
             [7.2,8.2,9.2]
            ],
        ]
    )
)


# In[ ]:


sess = tf.Session()


# In[ ]:

sizes = tf.shape(a)
d1 = sizes[0]
d2 = sizes[1]
d3 = sizes[2]

b = tf.concat([a, a,], axis=2)
c = tf.reshape(b, [d1,2*d2,d3])

d = tf.transpose(c, perm=[0, 2, 1])
e = tf.concat([d, d], axis=2)
f = tf.transpose(tf.reshape(e, [d1, 2*d2, 2*d3]), perm=[0, 2, 1])

g = tf.stack([a, 2*a, 3.4*a], axis=3)
h = tf.concat([g, g], axis=2)
i = tf.reshape(h, [3, 6, 3, 3])
j = tf.transpose(i, perm=[0, 2, 1, 3])
k = tf.concat([j, j], axis=2)
l = tf.transpose(tf.reshape(k, [3, 6, 6, 3]), perm=[0, 2, 1, 3])

for i in range(3):
    print(sess.run(l)[:, :, :, i])


# In[ ]:


sess.run(c)


# In[ ]:


sess.run(f)

