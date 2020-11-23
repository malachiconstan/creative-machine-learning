import tensorflow as tf
import numpy as np

'''
Code below adapted from:
'''

class EqualizeLearningRate(tf.keras.layers.Wrapper):
    """
    Reference from WeightNormalization implementation of TF Addons
    EqualizeLearningRate wrapper works for keras CNN and Dense (RNN not tested).
    ```python
      net = EqualizeLearningRate(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = EqualizeLearningRate(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = EqualizeLearningRate(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = EqualizeLearningRate(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
    Raises:
      ValueError: If `Layer` does not contain a `kernel` of weights
    """

    def __init__(self, layer, **kwargs):
        super(EqualizeLearningRate, self).__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`EqualizeLearningRate` must wrap a layer that contains a `kernel` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # He constant
        self.fan_in, self.fan_out= self._compute_fans(kernel.shape)
        self.he_constant = tf.Variable(1.0 / np.sqrt(self.fan_in), dtype=tf.float32, trainable=False)

        self.v = kernel
        self.built = True
    
    def call(self, inputs, training=True):
        """Call `Layer`"""

        with tf.name_scope('compute_weights'):
            # Multiply the kernel with the he constant.
            kernel = tf.identity(self.v * self.he_constant)
            
            if self.is_rnn:
                print(self.is_rnn)
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())
    
    def _compute_fans(self, shape, data_format='channels_last'):
        """
        From Official Keras implementation
        Computes the number of input and output units for a weight shape.
        # Arguments
            shape: Integer shape tuple.
            data_format: Image data format to use for convolution kernels.
                Note that all kernels in Keras are standardized on the
                `channels_last` ordering (even when inputs are set
                to `channels_first`).
        # Returns
            A tuple of scalars, `(fan_in, fan_out)`.
        # Raises
            ValueError: in case of invalid `data_format` argument.
        """
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) in {3, 4, 5}:
            # Assuming convolution kernels (1D, 2D or 3D).
            # TH kernel shape: (depth, input_depth, ...)
            # TF kernel shape: (..., input_depth, depth)
            if data_format == 'channels_first':
                receptive_field_size = np.prod(shape[2:])
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
            elif data_format == 'channels_last':
                receptive_field_size = np.prod(shape[:-2])
                fan_in = shape[-2] * receptive_field_size
                fan_out = shape[-1] * receptive_field_size
            else:
                raise ValueError('Invalid data_format: ' + data_format)
        else:
            # No specific assumptions.
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out

class PixelNormalization(tf.keras.layers.Layer):
    """
    Arguments:
      epsilon: a float-point number, the default is 1e-8
    """
    def __init__(self, epsilon=1e-8, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, X):
        return X * tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(X), axis=-1, keepdims=True) + self.epsilon)
    
    def compute_output_shape(self, input_shape):
        return input_shape

class MinibatchSTDDEV(tf.keras.layers.Layer):
    """
    Reference from official pggan implementation
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
    
    Arguments:
      group_size: a integer number, minibatch must be divisible by (or smaller than) group_size.
    """
    def __init__(self, group_size=4):
        super(MinibatchSTDDEV, self).__init__()
        self.group_size = group_size

    def call(self, X):
        group_size = tf.minimum(self.group_size, tf.shape(X)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = X.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(X, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, X.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [NHW1]  Replicate over group and pixels.
        return tf.concat([X, y], axis=-1)                        # [NHWC]  Append as new fmap.
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 1)

# TF Graph Approach      
def pg_upsample_block(input_,
                    input_filters,
                    output_filters,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    leaky_relu_alpha=0.2,
                    kernel_initializer='he_normal',
                    name=''):

    upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(input_)
    upsample_x = EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters,
                                                    kernel_size,
                                                    strides,
                                                    padding=padding,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'), name=name+'_conv2d_1')(upsample)
    X = PixelNormalization(name=name+'_norm_1')(upsample_x)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name=name+'_activation_1')(X)
    X = EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters,
                                                    kernel_size,
                                                    strides,
                                                    padding=padding,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'), name=name+'_conv2d_2')(X)
    X = PixelNormalization(name=name+'_norm_2')(X)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name=name+'_activation_2')(X)
    return X, upsample

def pg_downsample_block(X,
                        output_filters1,
                        output_filters2,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        leaky_relu_alpha=0.2,
                        kernel_initializer='he_normal',
                        name=''):
                        
    X = EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters1,
                                                    kernel_size,
                                                    strides,
                                                    padding=padding,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'),
                                                    name=name+'_conv2d_1')(X)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name=name+'_activation_1')(X)
    X = EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters2,
                                                    kernel_size,
                                                    strides,
                                                    padding=padding,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'),
                                                    name=name+'_conv2d_2')(X)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name=name+'_activation_2')(X)
    X = tf.keras.layers.AveragePooling2D(pool_size=2, name=name+'_avg_pool2D')(X)

    return X

def generator_input_block(X,
                        kernel_initializer='he_normal',
                        leaky_relu_alpha=0.2,
                        latent_dim=512):

    # Block 1
    X = EqualizeLearningRate(tf.keras.layers.Dense(4*4*latent_dim,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'),
                                                    name='g_input_dense')(X)

    X = PixelNormalization(name='g_input_norm_1')(X)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name='g_input_activation_1')(X)

    # Transition
    X = tf.keras.layers.Reshape((4, 4, latent_dim), name='g_input_reshape')(X)

    # Block 2
    X = EqualizeLearningRate(tf.keras.layers.Conv2D(latent_dim,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding='same',
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'),
                                                    name='g_input_conv2d')(X)
    X = PixelNormalization(name='g_input_norm_2')(X)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name='g_input_activation_2')(X)

    return X

def discriminator_output_block(X,
                            kernel_initializer='he_normal',
                            leaky_relu_alpha=0.2):

    X = MinibatchSTDDEV()(X)
    X = EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding='same',
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'),
                                                    name='d_output_conv2d_1')(X)

    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name='d_output_activation_1')(X)
    
    X = EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                    kernel_size=4,
                                                    strides=1,
                                                    padding='valid',
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'),
                                                    name='d_output_conv2d_2')(X)
    X = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha, name='d_output_activation_2')(X)
    
    X = tf.keras.layers.Flatten(name='d_output_flatten')(X)
    X = EqualizeLearningRate(tf.keras.layers.Dense(1,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer='zeros'),
                                            name='d_output_dense')(X)
    return X

class PGUpSampleBlock(tf.keras.Model):
    def __init__(self,
                input_filters,
                output_filters,
                kernel_size=3,
                strides=1,
                padding='valid',
                activation_layer=tf.keras.layers.LeakyReLU(alpha=0.2),
                kernel_initializer='he_normal',
                name=''
                ):
        super(PGUpSampleBlock, self).__init__(name=name)

        self.upsampling_layer = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')

        self.body = tf.keras.Sequential(name=name+'_body')
        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters,
                                                                kernel_size,
                                                                strides,
                                                                padding=padding,
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name=name+'_conv2d_1'))
        self.body.add(PixelNormalization())
        self.body.add(activation_layer)

        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters,
                                                                kernel_size,
                                                                strides,
                                                                padding=padding,
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name=name+'_conv2d_2'))
        self.body.add(PixelNormalization())
        self.body.add(activation_layer)

    def call(self, X):
        upsampled_X = self.upsampling_layer(X)

        return self.body(upsampled_X), upsampled_X

class PGDownSampleBlock(tf.keras.Model):
    def __init__(self,
                output_filters1,
                output_filters2,
                kernel_size=3,
                strides=1,
                padding='valid',
                activation_layer=tf.keras.layers.LeakyReLU(alpha=0.2),
                kernel_initializer='he_normal',
                name=''
                ):
        super(PGDownSampleBlock, self).__init__(name=name)

        self.body = tf.keras.Sequential(name=name+'_body')
        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters1,
                                                                kernel_size,
                                                                strides,
                                                                padding=padding,
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name=name+'_conv2d_1'))
        self.body.add(activation_layer)

        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(output_filters2,
                                                                kernel_size,
                                                                strides,
                                                                padding=padding,
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name=name+'_conv2d_2'))
        self.body.add(activation_layer)
        self.body.add(tf.keras.layers.AveragePooling2D(pool_size=2))

    def call(self, X):
        return self.body(X)

class GeneratorInputBlock(tf.keras.Model):
    def __init__(self,
                kernel_initializer='he_normal',
                leaky_relu_alpha=0.2,
                latent_dim=512,
                name=''
                ):
        super(GeneratorInputBlock, self).__init__(name=name)

        self.body = tf.keras.Sequential(name=name+'_body')
        
        self.body.add(EqualizeLearningRate(tf.keras.layers.Dense(4*4*latent_dim,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer='zeros'),
                                        name='g_input_dense'))
        self.body.add(PixelNormalization())
        self.body.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha))

        self.body.add(tf.keras.layers.Reshape((4, 4, latent_dim)))

        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(latent_dim,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name='g_input_conv2d'))
        self.body.add(PixelNormalization())
        self.body.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha))

    def call(self, X):
        return self.body(X)

class DiscriminatorOutputBlock(tf.keras.Model):
    def __init__(self,
                kernel_initializer='he_normal',
                leaky_relu_alpha=0.2,
                name=''
                ):
        super(DiscriminatorOutputBlock, self).__init__(name=name)

        self.body = tf.keras.Sequential(name=name+'_body')
        
        self.body.add(MinibatchSTDDEV())
        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name='d_output_conv2d_1'))

        self.body.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha))
        self.body.add(EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                                kernel_size=4,
                                                                strides=1,
                                                                padding='valid',
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer='zeros'),
                                                                name='d_output_conv2d_2'))
        self.body.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha))
        self.body.add(tf.keras.layers.Flatten())

        self.body.add(EqualizeLearningRate(tf.keras.layers.Dense(1,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer='zeros'),
                                        name='d_output_dense'))

    def call(self, X):
        return self.body(X)

# def Upscale2d(X, factor=2):
#     assert isinstance(factor, int) and factor >= 1
#     if factor == 1:
#         return X
#     s = X.shape
#     X = tf.reshape(X,(s[0], s[1], 1, s[2], 1, s[3]))
#     X = tf.broadcast_to(X, (s[0], s[1], factor, s[2], factor, s[3]))
#     X = tf.reshape(X,(s[0], s[1] * factor, s[2] * factor, s[3]))
#     return X

# def get_layer_normalisation_factor(weight_shape):
#     """
#     Get He's constant for the given layer
#     https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
#     """
#     return tf.math.sqrt(tf.constant(2.0)/tf.math.reduce_prod(tf.cast(weight_shape, tf.float32)[1:]))

# def mini_batch_sd(X, subGroupSize=4, epsilon=1e-8):
#     """
#     Add a minibatch standard deviation channel to the current layer.
#     In other words:
#         1) Compute the standard deviation of the feature map over the minibatch
#         2) Get the mean, over all pixels and all channels of thsi ValueError
#         3) expand the layer and cocatenate it with the input
#     Args:
#         - x (tensor): previous layer
#         - subGroupSize (int): size of the mini-batches on which the standard deviation
#         should be computed
#     """
#     shape = X.shape
#     subGroupSize = min(shape[0], subGroupSize)
#     if shape[0] % subGroupSize != 0:
#         subGroupSize = shape[0]
#     G = int(shape[0] / subGroupSize)
#     if subGroupSize > 1:
#         y = tf.reshape(X, (-1, subGroupSize, shape[1], shape[2], shape[3]))
#         y = tf.math.sqrt(tf.math.reduce_variance(y, axis = 3) + epsilon)
#         y = tf.reshape(y, (G, -1))
#         y = tf.reshape(tf.math.reduce_mean(y, axis=1), (G, 1))
#         y = tf.reshape(tf.broadcast_to(y, (G, shape[1]*shape[2])), (G, 1, shape[1], shape[2], 1))
#         y = tf.broadcast_to(y, (G, subGroupSize, shape[1], shape[2], 1))
#         y = tf.reshape(y, (-1, shape[1], shape[2], 1))
#     else:
#         y = tf.zeros((shape[0], shape[1], shape[2], 1))

#     return tf.concat([X, y], axis=3)

# def mini_batch_sd(X):
#     batch_size, h, w, _ = X.get_shape().as_list()
#     new_feat_shape = [batch_size, h, w, 1]

#     mean, var = tf.nn.moments(X, axes=[0], keepdims=True)
#     stddev = tf.math.sqrt(tf.math.reduce_mean(var, keepdims=True))
#     new_feat = tf.tile(stddev, multiples=new_feat_shape)
#     return tf.concat([X, new_feat], axis=3)

# class NormalizationLayer(tf.keras.layers.Layer):
#     '''
#     Pixel Wise Normalisation for PGGAN
#     '''
#     def __init__(self, **kwargs):
#         super(NormalizationLayer, self).__init__(**kwargs)
    
#     def call(self, X, epsilon=1e-8):
#         return X * tf.math.rsqrt(tf.math.reduce_mean(X**2, axis=3, keepdims=True) + epsilon)

# class EqualizerInitializer(tf.keras.initializers.Initializer):

#     def __init__(self, lrMul):
#         self.lrMul = lrMul

#     def __call__(self, shape, dtype=tf.float32):
#         W = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=dtype)
#         W /= self.lrMul
#         return W

#     def get_config(self):
#         return {"lrMul": self.lrMul}

# class EqualizedConv2D(tf.keras.Model):
#     '''
#     Special Convolution Layer that initialise bias to 0 and apply He's initialisation at runtime
#     '''
#     def __init__(self, channels_out, kernel_size, padding='same', use_bias=True, equalized=True, lrMul=1.0, init_bias_zero=True, **kwargs):
#         super(EqualizedConv2D, self).__init__(**kwargs)

#         self.module = tf.keras.layers.Conv2D(filters=channels_out, kernel_size=kernel_size, padding=padding,use_bias=use_bias)
#         self.equalized = equalized

#         if init_bias_zero:
#             self.module.bias_initializer = tf.keras.initializers.Zeros()
#         if self.equalized:
#             self.module.kernel_initializer = EqualizerInitializer(lrMul=lrMul)
        
#     def call(self, X):
#         X = self.module(X)
#         if self.equalized:
#             X *= get_layer_normalisation_factor(self.module.kernel.shape)
#         return X

# class EqualizedDense(tf.keras.Model):
#     '''
#     Special Dense Layer that initialise bias to 0 and apply He's initialisation at runtime
#     '''
#     def __init__(self, channels_out, use_bias=True, equalized=True, lrMul=1.0, init_bias_zero=True, **kwargs):
#         super(EqualizedDense, self).__init__(**kwargs)

#         self.module = tf.keras.layers.Dense(units=channels_out,use_bias=use_bias)
#         self.equalized = equalized

#         if init_bias_zero:
#             self.module.bias_initializer = tf.keras.initializers.Zeros()
#         if self.equalized:
#             self.module.kernel_initializer = EqualizerInitializer(lrMul=lrMul)
        
#     def call(self, X):
#         X = self.module(X)
#         if self.equalized:
#             X *= get_layer_normalisation_factor(self.module.kernel.shape)
#         return X
