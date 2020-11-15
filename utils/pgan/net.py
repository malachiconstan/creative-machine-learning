import tensorflow as tf
import numpy as np

class EqualisedInitialiser(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        fan_in = shape[0]*shape[1]*shape[-1]
        self.he_constant = tf.constant(tf.math.sqrt(2./fan_in), dtype=tf.float32)
        return self.he_constant*tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=dtype)

    def get_config(self):
        return dict(he_constant = self.he_constant)

class EqualisedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                output_channels,
                kernel_size,
                strides=[1, 1, 1, 1],
                padding='same',
                activation=None,
                **kwargs
                ):
        super(EqualisedConv2D, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding

        self.kernel_init = tf.keras.initializers.HeNormal() #EqualisedInitialiser()
        self.bias_init = tf.keras.initializers.Zeros()

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.kernel_size = [self.kernel_size, self.kernel_size, input_channels, self.output_channels]
        self.bias_size = [self.output_channels]

        self.kernel = self.add_weight('kernel', shape = self.kernel_size, initializer=self.kernel_init, trainable=True)
        self.bias = self.add_weight('bias', shape = self.bias_size, initializer=self.bias_init, trainable=True)

    def call(self, X):
        X = tf.nn.conv2d(X, self.kernel, self.strides, self.padding.upper())
        return tf.nn.bias_add(X, self.bias)

    def get_config(self):
        return dict(kernel_size = self.kernel_size, output_channels = self.output_channels, strides = self.strides, padding = self.padding)

class PixelwiseNorm(tf.keras.layers.Layer):
    def __init__(self, name='pixel_norm', epsilon = 1e-8):
        super(PixelwiseNorm, self).__init__(name = name)
        self.epsilon = epsilon

    def call(self, X):
        return X / tf.math.sqrt(tf.math.reduce_mean(X * X, axis=3, keepdims=True) + self.epsilon)

class EqualisedConv2DModule(tf.keras.Model):
    def __init__(self,
                output_filters,
                leak,
                kernel_sizes=None,
                norms=None,
                padding='same',
                module_name='',
                ):
        super(EqualisedConv2DModule, self).__init__(name=module_name)

        if kernel_sizes is None:
            kernel_sizes = [3] * len(output_filters)
        if norms is None:
            norms = [None, None]

        self.body = tf.keras.Sequential()

        for i, (filters, kernel_size, norm) in enumerate(zip(output_filters, kernel_sizes, norms)):
            layer_name = module_name + f'_conv_layer_{i+1}'
            self.body.add(tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, name=layer_name, kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer='zeros'))
            # self.body.add(EqualisedConv2D(filters, kernel_size, padding=padding, name=layer_name))
            self.body.add(tf.keras.layers.LeakyReLU(alpha=leak))
            if norm is not None:
                if norm == 'batch_norm':
                    self.body.add(tf.keras.layers.BatchNormalization())
                elif norm == 'pixel_norm':
                    self.body.add(PixelwiseNorm(name=module_name + f'_conv_layer_{i+1}_px_norm'))
                elif norm == 'layer_norm':
                    self.body.add(tf.keras.layers.LayerNormalization())
                else:
                    raise NotImplementedError(f'{norm} not yet implemented')

    def call(self, X):
        return self.body(X)

class PGGenerator(tf.keras.Model):
    def __init__(self,
                cfg,
                name = 'PGGAN_Generator',
                **kwargs):
        super(PGGenerator, self).__init__(name = name, **kwargs)

        self.leak = cfg.leakyRelu_alpha
        input_size, _, nc = cfg.input_shape
        self.res = cfg.resolution
        self.min_res = cfg.min_resolution
        
        # number of times to upsample/downsample for full resolution:
        self.n_scalings = int(np.log2(input_size / self.min_res))
        # number of times to upsample/downsample for current resolution:
        self.n_layers = int(np.log2(self.res / self.min_res))
        self.nf_min = cfg.nf_min  # min feature depth
        self.nf_max = cfg.nf_max  # max feature depth
        self.batch_size = cfg.batch_size

        # Placeholders
        self.z_dim = cfg.z_dim
        self.transition = cfg.transition
        self.use_tanh = cfg.use_tanh

        # Fixed Layers
        self.upsampling_layer = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')

        # Define Convolution Layers
        feat_depth = min(self.nf_max, self.nf_min * 2 ** self.n_scalings)
        r = self.min_res
        self.first_conv = EqualisedConv2DModule(output_filters=[feat_depth, feat_depth],
                                                leak=self.leak,
                                                kernel_sizes=[4,3],
                                                norms=[None, cfg.norm_g],
                                                module_name='{0:}x{0:}'.format(r),
                                                )
        
        self.subsequent_conv = []
        
        for i in range(self.n_layers):
            n = self.nf_min * 2 ** (self.n_scalings - i - 1)
            feat_depth = min(self.nf_max, n)
            r *= 2
            self.subsequent_conv.append(EqualisedConv2DModule(
                                        output_filters=[feat_depth, feat_depth],
                                        leak=self.leak,
                                        norms=[None, cfg.norm_g],
                                        module_name='{0:}x{0:}'.format(r),
                                        ))
        assert r == self.res, '{:} not equal to {:}'.format(r, self.res)

        # To RGB Layers using 1x1 convolution
        self.to_image_conv = tf.keras.layers.Conv2D(cfg.input_shape[-1], 1, name='{0:}x{0:}_to_rgb'.format(r), kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer='zeros')
        # EqualisedConv2D(cfg.input_shape[-1], 1, name='{0:}x{0:}_to_rgb'.format(r))
        # self.to_image_conv_prime = EqualisedConv2D(cfg.input_shape[-1], 1, name='{0:}x{0:}_to_rgb_prime'.format(r//2))
        self.to_image_conv_prime = tf.keras.layers.Conv2D(cfg.input_shape[-1], 1, name='{0:}x{0:}_to_rgb_prime'.format(r//2), kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer='zeros')

    @property
    def alpha(self):
        """
        Get alpha value
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """
        if value < 0 or value > 1:
            raise ValueError("alpha must be in [0,1]")

        self._alpha = value

    def call(self, z, verbose=False):

        feat_size = self.min_res
        X = tf.reshape(z, (-1, 1, 1, self.z_dim))
        padding = int(feat_size / 2)
        X = tf.pad(X, [[0, 0], [padding - 1, padding],[padding - 1, padding], [0, 0]])
        X = self.first_conv(X)

        for i in range(self.n_layers):
            X = self.upsampling_layer(X)
            X = self.subsequent_conv[i](X)
            if i == self.n_layers-2 and self.transition:
                X_prime = X

        X = self.to_image_conv(X)
        if self.transition:
            print('Using Transition Route')
            X_prime = self.upsampling_layer(X_prime)
            X_prime = self.to_image_conv_prime(X_prime)
            X = self.alpha*X + (1.-self.alpha)*X_prime

        if self.use_tanh:
            X = tf.tanh(X)

        return X

class PGDiscriminator(tf.keras.Model):
    def __init__(self,
                cfg,
                name = 'PGGAN_Discriminator',
                **kwargs):
        super(PGDiscriminator, self).__init__(name = name, **kwargs)

        self.leak = cfg.leakyRelu_alpha
        input_size, _, nc = cfg.input_shape
        self.res = cfg.resolution
        self.min_res = cfg.min_resolution
        
        # number of times to upsample/downsample for full resolution:
        self.n_scalings = int(np.log2(input_size / self.min_res))
        # number of times to upsample/downsample for current resolution:
        self.n_layers = int(np.log2(self.res / self.min_res))
        self.nf_min = cfg.nf_min  # min feature depth
        self.nf_max = cfg.nf_max  # max feature depth
        self.batch_size = cfg.batch_size
        self.transition = cfg.transition

        # Fixed Layers
        self.downsample_layer = tf.keras.layers.AveragePooling2D(pool_size=(2,2))

        # Define Convolution Layers
        self.feat_depths = [min(self.nf_max, self.nf_min * 2 ** i)for i in range(self.n_scalings)]

        norm = cfg.norm_d
        if (cfg.loss_mode == 'wgan_gp') and (norm == 'batch_norm'):
            norm = None

        r = self.res
        self.from_image_conv = tf.keras.layers.Conv2D(self.feat_depths[-self.n_layers], 1, name='{0:}x{0:}_from_rgb'.format(r), kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer='zeros') 
        # EqualisedConv2D(self.feat_depths[-self.n_layers], 1, name='{0:}x{0:}_from_rgb'.format(r))

        self.subsequent_conv = []
        for i in range(self.n_layers):
            feat_depth_1 = self.feat_depths[-self.n_layers + i]
            feat_depth_2 = min(self.nf_max, 2 * feat_depth_1)
            self.subsequent_conv.append(EqualisedConv2DModule(
                                        output_filters=[feat_depth_1, feat_depth_2],
                                        leak=self.leak,
                                        norms=[norm, norm],
                                        module_name='{0:}x{0:}'.format(r)
                                        ))
            r /= 2
            if i == 0 and self.transition:
                self.from_image_conv_prime = EqualisedConv2D(self.feat_depths[-self.n_layers+1], 1, name='{0:}x{0:}_from_rgb_prime'.format(r//2))

        assert r == self.min_res, '{:} not equal to {:}'.format(r, self.min_res)

        self.final_feat_depth = min(self.nf_max, self.nf_min * 2 ** self.n_scalings)
        self.final_conv_module = EqualisedConv2DModule(output_filters=[self.final_feat_depth, self.final_feat_depth],
                                                        leak=self.leak,
                                                        kernel_sizes=[3,4],
                                                        norms=[norm,None],
                                                        module_name='{0:}x{0:}'.format(r)
                                                        )

        # Classification layer
        self.classifier = tf.keras.layers.Dense(1, name='classifier')

    @property
    def alpha(self):
        """
        Get alpha value
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """
        if value < 0 or value > 1:
            raise ValueError("alpha must be in [0,1]")

        self._alpha = value

    def minibatch_stddev(self, X):
        _, h, w, _ = X.get_shape().as_list()
        new_feat_shape = [self.batch_size, h, w, 1]

        _, var = tf.nn.moments(X, axes=[0], keepdims=True)
        stddev = tf.math.sqrt(tf.math.reduce_mean(var, keepdims=True))
        new_feat = tf.tile(stddev, multiples=new_feat_shape)
        return tf.concat([X, new_feat], axis=3)
        
    def call(self, input_, verbose=False):
    
        X = self.from_image_conv(input_)
        for i in range(self.n_layers):
            X = self.subsequent_conv[i](X)
            X = self.downsample_layer(X)
            if i == 0 and self.transition:
                print('Using Transition Route')
                X_prime = self.downsample_layer(input_)
                X_prime = self.from_image_conv_prime(X_prime)
                X = self.alpha * X + (1.-self.alpha) * X_prime

        X = self.minibatch_stddev(X)
        X = self.final_conv_module(X)
        X = tf.reduce_mean(X, axis=[1, 2])
        X = tf.reshape(X, [self.batch_size, self.final_feat_depth])
        X = self.classifier(X)

        return X

# class DCGAN(Model):
#     def __init__(self, cfg):
#         self.alpha = cfg.leakyRelu_alpha
#         input_size, _, nc = cfg.input_shape
#         self.res = cfg.resolution
#         self.min_res = cfg.min_resolution
#         # number of times to upsample/downsample for full resolution:
#         self.n_scalings = int(np.log2(input_size / self.min_res))
#         # number of times to upsample/downsample for current resolution:
#         self.n_layers = int(np.log2(self.res / self.min_res))
#         self.nf_min = cfg.nf_min  # min feature depth
#         self.nf_max = cfg.nf_max  # max feature depth
#         self.batch_size = cfg.batch_size
#         Model.__init__(self, cfg)

#     def leaky_relu(self, input_):
#         return tf.maximum(self.alpha * input_, input_)

#     def add_minibatch_stddev_feat(self, input_):
#         _, h, w, _ = input_.get_shape().as_list()
#         new_feat_shape = [self.cfg.batch_size, h, w, 1]

#         mean, var = tf.nn.moments(input_, axes=[0], keep_dims=True)
#         stddev = tf.sqrt(tf.reduce_mean(var, keep_dims=True))
#         new_feat = tf.tile(stddev, multiples=new_feat_shape)
#         return tf.concat([input_, new_feat], axis=3)

#     def pixelwise_norm(self, a):
#         return a / tf.sqrt(tf.reduce_mean(a * a, axis=3, keep_dims=True) + 1e-8)

#     def conv2d(self, input_, n_filters, k_size, padding='same'):
#         if not self.cfg.weight_scale:
#             return tf.layers.conv2d(input_, n_filters, k_size, padding=padding)

#         n_feats_in = input_.get_shape().as_list()[-1]
#         fan_in = k_size * k_size * n_feats_in
#         c = tf.constant(np.sqrt(2. / fan_in), dtype=tf.float32)
#         kernel_init = tf.random_normal_initializer(stddev=1.)
#         w_shape = [k_size, k_size, n_feats_in, n_filters]
#         w = tf.get_variable('kernel', shape=w_shape, initializer=kernel_init)
#         w = c * w
#         strides = [1, 1, 1, 1]
#         net = tf.nn.conv2d(input_, w, strides, padding=padding.upper())
#         b = tf.get_variable('bias', [n_filters],
#                             initializer=tf.constant_initializer(0.))
#         net = tf.nn.bias_add(net, b)
#         return net

#     def up_sample(self, input_):
#         _, h, w, _ = input_.get_shape().as_list()
#         new_size = [2 * h, 2 * w]
#         return tf.image.resize_nearest_neighbor(input_, size=new_size)

#     def down_sample(self, input_):
#         return tf.layers.average_pooling2d(input_, 2, 2)

#     def conv_module(self, input_, n_filters, training, k_sizes=None,
#                     norms=None, padding='same'):
#         conv = input_
#         if k_sizes is None:
#             k_sizes = [3] * len(n_filters)
#         if norms is None:
#             norms = [None, None]

#         # series of conv + lRelu + norm
#         for i, (nf, k_size, norm) in enumerate(zip(n_filters, k_sizes, norms)):
#             var_scope = 'conv_block_' + str(i+1)
#             with tf.variable_scope(var_scope):
#                 conv = self.conv2d(conv, nf, k_size, padding=padding)
#                 conv = self.leaky_relu(conv)
#                 if norm == 'batch_norm':
#                     conv = tf.layers.batch_normalization(conv, training=training)
#                 elif norm == 'pixel_norm':
#                     conv = self.pixelwise_norm(conv)
#                 elif norm == 'layer_norm':
#                     conv = tf.contrib.layers.layer_norm(conv)
#         return conv

#     def to_image(self, input_, resolution):
#         nc = self.cfg.input_shape[-1]
#         var_scope = '{0:}x{0:}'.format(resolution)
#         with tf.variable_scope(var_scope + '/to_image'):
#             out = self.conv2d(input_, nc, 1)
#             return out

#     def from_image(self, input_, n_filters, resolution):
#         var_scope = '{0:}x{0:}'.format(resolution)
#         with tf.variable_scope(var_scope + '/from_image'):
#             out = self.conv2d(input_, n_filters, 1)
#             return self.leaky_relu(out)

#     def build_generator(self, training):
#         z = self.tf_placeholders['z']
#         z_dim = self.cfg.z_dim
#         feat_size = self.min_res
#         norm = self.cfg.norm_g

#         with tf.variable_scope('generator', reuse=(not training)):
#             net = tf.reshape(z, (-1, 1, 1, z_dim))
#             padding = int(feat_size / 2)
#             net = tf.pad(net, [[0, 0], [padding - 1, padding],
#                                [padding - 1, padding], [0, 0]])
#             feat_depth = min(self.nf_max, self.nf_min * 2 ** self.n_scalings)
#             r = self.min_res
#             var_scope = '{0:}x{0:}'.format(r)
#             with tf.variable_scope(var_scope):
#                 net = self.conv_module(net, [feat_depth, feat_depth],
#                                        training, k_sizes=[4, 3],
#                                        norms=[None, norm])
#             layers = []
#             for i in range(self.n_layers):
#                 net = self.up_sample(net)
#                 n = self.nf_min * 2 ** (self.n_scalings - i - 1)
#                 feat_depth = min(self.nf_max, n)
#                 r *= 2
#                 var_scope = '{0:}x{0:}'.format(r)
#                 with tf.variable_scope(var_scope):
#                     net = self.conv_module(net, [feat_depth, feat_depth],
#                                            training, norms=[norm, norm])
#                 layers.append(net)

#             # final layer:
#             assert r == self.res, \
#                 '{:} not equal to {:}'.format(r, self.res)
#             net = self.to_image(net, self.res)
#             if self.cfg.transition:
#                 alpha = self.tf_placeholders['alpha']
#                 branch = layers[-2]
#                 branch = self.up_sample(branch)
#                 branch = self.to_image(branch, r / 2)
#                 net = alpha * net + (1. - alpha) * branch
#             if self.cfg.use_tanh:
#                 net = tf.tanh(net)
#             return net

#     def build_discriminator(self, input_, reuse, training):
#         norm = self.cfg.norm_d
#         if (self.cfg.loss_mode == 'wgan_gp') and (norm == 'batch_norm'):
#             norm = None
#         with tf.variable_scope('discriminator', reuse=reuse):
#             feat_depths = [min(self.nf_max, self.nf_min * 2 ** i)
#                            for i in range(self.n_scalings)]
#             r = self.res
#             net = self.from_image(input_, feat_depths[-self.n_layers], r)
#             for i in range(self.n_layers):
#                 feat_depth_1 = feat_depths[-self.n_layers + i]
#                 feat_depth_2 = min(self.nf_max, 2 * feat_depth_1)
#                 var_scope = '{0:}x{0:}'.format(r)
#                 with tf.variable_scope(var_scope):
#                     net = self.conv_module(net, [feat_depth_1, feat_depth_2],
#                                            training, norms=[norm, norm])
#                 net = self.down_sample(net)
#                 r /= 2
#                 # add a transition branch if required
#                 if i == 0 and self.cfg.transition:
#                     alpha = self.tf_placeholders['alpha']
#                     input_low = self.down_sample(input_)
#                     idx = -self.n_layers + 1
#                     branch = self.from_image(input_low, feat_depths[idx],
#                                              self.res / 2)
#                     net = alpha * net + (1. - alpha) * branch

#             # add final layer
#             assert r == self.min_res, \
#                 '{:} not equal to {:}'.format(r, self.min_res)
#             net = self.add_minibatch_stddev_feat(net)
#             feat_depth = min(self.nf_max, self.nf_min * 2 ** self.n_scalings)
#             var_scope = '{0:}x{0:}'.format(r)
#             with tf.variable_scope(var_scope):
#                 net = self.conv_module(net, [feat_depth, feat_depth],
#                                        training, k_sizes=[3, 4],
#                                        norms=[norm, None])
#                 net = tf.reduce_mean(net, axis=[1, 2])
#                 net = tf.reshape(net, [self.cfg.batch_size, feat_depth])
#                 net = tf.layers.dense(net, 1)
#             return net

