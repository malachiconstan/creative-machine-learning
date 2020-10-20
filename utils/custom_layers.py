import tensorflow as tf
'''
Code below adapted from: https://github.com/facebookresearch/pytorch_GAN_zoo/ re-written for tensorflow 2.x
'''
def Upscale2d(X, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return X
    s = X.shape
    X = tf.reshape(X,(s[0], s[1], 1, s[2], 1, s[3]))
    X = tf.broadcast_to(X, (s[0], s[1], factor, s[2], factor, s[3]))
    X = tf.reshape(X,(s[0], s[1] * factor, s[2] * factor, s[3]))
    return X

def get_layer_normalisation_factor(weight_shape):
    """
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    return tf.math.sqrt(tf.constant(2.0)/tf.math.reduce_prod(tf.cast(weight_shape, tf.float32)[1:]))

class NormalizationLayer(tf.keras.layers.Layer):
    '''
    Pixel Wise Normalisation for PGGAN
    '''
    def __init__(self, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)
    
    def call(self, X, epsilon=1e-8):
        return X * tf.math.rsqrt(tf.math.reduce_mean(X**2, axis=1, keepdims=True) + epsilon)

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