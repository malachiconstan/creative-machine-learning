import tensorflow as tf
'''
Code below adapted from: https://github.com/facebookresearch/pytorch_GAN_zoo/ re-written for tensorflow 2.x
'''
# def Upscale2d(X, factor=2):
#     assert isinstance(factor, int) and factor >= 1
#     if factor == 1:
#         return X
#     s = X.shape
#     X = tf.reshape(X,(s[0], s[1], 1, s[2], 1, s[3]))
#     X = tf.broadcast_to(X, (s[0], s[1], factor, s[2], factor, s[3]))
#     X = tf.reshape(X,(s[0], s[1] * factor, s[2] * factor, s[3]))
#     return X

def get_layer_normalisation_factor(weight_shape):
    """
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    return tf.math.sqrt(tf.constant(2.0)/tf.math.reduce_prod(tf.cast(weight_shape, tf.float32)[1:]))

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

def mini_batch_sd(X):
    batch_size, h, w, _ = X.get_shape().as_list()
    new_feat_shape = [batch_size, h, w, 1]

    mean, var = tf.nn.moments(X, axes=[0], keepdims=True)
    stddev = tf.math.sqrt(tf.math.reduce_mean(var, keepdims=True))
    new_feat = tf.tile(stddev, multiples=new_feat_shape)
    return tf.concat([X, new_feat], axis=3)

class NormalizationLayer(tf.keras.layers.Layer):
    '''
    Pixel Wise Normalisation for PGGAN
    '''
    def __init__(self, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)
    
    def call(self, X, epsilon=1e-8):
        return X * tf.math.rsqrt(tf.math.reduce_mean(X**2, axis=3, keepdims=True) + epsilon)

class EqualizerInitializer(tf.keras.initializers.Initializer):

    def __init__(self, lrMul):
        self.lrMul = lrMul

    def __call__(self, shape, dtype=tf.float32):
        W = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=dtype)
        W /= self.lrMul
        return W

    def get_config(self):
        return {"lrMul": self.lrMul}

class EqualizedConv2D(tf.keras.Model):
    '''
    Special Convolution Layer that initialise bias to 0 and apply He's initialisation at runtime
    '''
    def __init__(self, channels_out, kernel_size, padding='same', use_bias=True, equalized=True, lrMul=1.0, init_bias_zero=True, **kwargs):
        super(EqualizedConv2D, self).__init__(**kwargs)

        self.module = tf.keras.layers.Conv2D(filters=channels_out, kernel_size=kernel_size, padding=padding,use_bias=use_bias)
        self.equalized = equalized

        if init_bias_zero:
            self.module.bias_initializer = tf.keras.initializers.Zeros()
        if self.equalized:
            self.module.kernel_initializer = EqualizerInitializer(lrMul=lrMul)
        
    def call(self, X):
        X = self.module(X)
        if self.equalized:
            X *= get_layer_normalisation_factor(self.module.kernel.shape)
        return X

class EqualizedDense(tf.keras.Model):
    '''
    Special Dense Layer that initialise bias to 0 and apply He's initialisation at runtime
    '''
    def __init__(self, channels_out, use_bias=True, equalized=True, lrMul=1.0, init_bias_zero=True, **kwargs):
        super(EqualizedDense, self).__init__(**kwargs)

        self.module = tf.keras.layers.Dense(units=channels_out,use_bias=use_bias)
        self.equalized = equalized

        if init_bias_zero:
            self.module.bias_initializer = tf.keras.initializers.Zeros()
        if self.equalized:
            self.module.kernel_initializer = EqualizerInitializer(lrMul=lrMul)
        
    def call(self, X):
        X = self.module(X)
        if self.equalized:
            X *= get_layer_normalisation_factor(self.module.kernel.shape)
        return X
