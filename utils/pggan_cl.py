from __future__ import division
import os
import time
import math
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Conv2D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Input, UpSampling2D, Dropout, Concatenate, Add, Dense, Multiply, LeakyReLU, Flatten, AveragePooling2D, Multiply
from tensorflow.keras import initializers, regularizers, constraints, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical, plot_model

AUTOTUNE = tf.data.experimental.AUTOTUNE

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
            raise ValueError('`EqualizeLearningRate` must wrap a layer that'
                             ' contains a `kernel` for weights')

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
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
    
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
        
kernel_initializer = 'he_normal'

class PixelNormalization(tf.keras.layers.Layer):
    """
    Arguments:
      epsilon: a float-point number, the default is 1e-8
    """
    def __init__(self, epsilon=1e-8):
        super(PixelNormalization, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
    
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

    def call(self, inputs):
        group_size = tf.minimum(self.group_size, tf.shape(inputs)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = inputs.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(inputs, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, inputs.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [NHW1]  Replicate over group and pixels.
        return tf.concat([inputs, y], axis=-1)                        # [NHWC]  Append as new fmap.
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 1)


def upsample_block(x, in_filters, filters, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu, name=''):
    '''
        Upsampling + 2 Convolution-Activation
    '''
    upsample = UpSampling2D(size=2, interpolation='nearest')(x)
    upsample_x = EqualizeLearningRate(Conv2D(filters, kernel_size, strides, padding=padding,
                   kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(upsample)
    x = PixelNormalization()(upsample_x)
    x = Activation(activation)(x)
    x = EqualizeLearningRate(Conv2D(filters, kernel_size, strides, padding=padding,
                                   kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
    x = PixelNormalization()(x)
    x = Activation(activation)(x)
    return x, upsample

def downsample_block(x, filters1, filters2, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu, name=''):
    '''
        2 Convolution-Activation + Downsampling
    '''
    x = EqualizeLearningRate(Conv2D(filters1, kernel_size, strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(x)
    x = Activation(activation)(x)
    x = EqualizeLearningRate(Conv2D(filters2, kernel_size, strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
    x = Activation(activation)(x)
    downsample = AveragePooling2D(pool_size=2)(x)

    return downsample
    
output_activation = tf.keras.activations.tanh

def generator_input_block(x):
    '''
        Generator input block
    '''
    x = EqualizeLearningRate(Dense(4*4*512, kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='g_input_dense')(x)
    x = PixelNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((4, 4, 512))(x)
    x = EqualizeLearningRate(Conv2D(512, 3, strides=1, padding='same',
                                          kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='g_input_conv2d')(x)
    x = PixelNormalization()(x)
    x = LeakyReLU()(x)
    return x

def build_4x4_generator(noise_dim=NOISE_DIM):
    '''
        4 * 4 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    # Not used in 4 * 4, put it here in order to keep the input here same as the other models
    alpha = Input((1), name='input_alpha')
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
    
    rgb_out = to_rgb(x)
    model = Model(inputs=[inputs, alpha], outputs=rgb_out)
    return model

def build_8x8_generator(noise_dim=NOISE_DIM):
    '''
        8 * 8 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_16x16_generator(noise_dim=NOISE_DIM):
    '''
        16 * 16 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(16, 16))

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_32x32_generator(noise_dim=NOISE_DIM):
    '''
        32 * 32 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(16, 16))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(32, 32))

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_64x64_generator(noise_dim=NOISE_DIM):
    '''
        64 * 64 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(32, 32))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(64, 64))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_128x128_generator(noise_dim=NOISE_DIM):
    '''
        128 * 128 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=256, filters=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(64, 64))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(128, 128))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_256x256_generator(noise_dim=NOISE_DIM):
    '''
        256 * 256 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    x, _ = upsample_block(x, in_filters=256, filters=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=128, filters=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(256, 256))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(128, 128))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(256, 256))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_512x512_generator(noise_dim=NOISE_DIM):
    '''
        512 * 512 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    x, _ = upsample_block(x, in_filters=256, filters=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    x, _ = upsample_block(x, in_filters=128, filters=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(256, 256))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, in_filters=64, filters=32, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(512, 512))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(256, 256))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(512, 512))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model
    
def discriminator_block(x):
    '''
        Discriminator output block
    '''
    x = MinibatchSTDDEV()(x)
    x = EqualizeLearningRate(Conv2D(512, 3, strides=1, padding='same',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_1')(x)
    x = LeakyReLU()(x)
    x = EqualizeLearningRate(Conv2D(512, 4, strides=1, padding='valid',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_2')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = EqualizeLearningRate(Dense(1, kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_dense')(x)
    return x

def build_4x4_discriminator():
    '''
        4 * 4 Discriminator
    '''
    inputs = Input((4,4,3))
    # Not used in 4 * 4
    alpha = Input((1), name='input_alpha')
    # From RGB
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
    x = from_rgb(inputs)
    x = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='conv2d_up_channel')(x)
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_8x8_discriminator():
    '''
        8 * 8 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((8,8,3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable block
    ########################
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_16x16_discriminator():
    '''
        16 * 16 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((16, 16, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(16, 16))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_32x32_discriminator():
    '''
        32 * 32 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((32, 32, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(16, 16))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(32, 32))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_64x64_discriminator():
    '''
        64 * 64 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((64, 64, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(32, 32))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(64, 64))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=256, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_128x128_discriminator():
    '''
        128 * 128 Discriminator
    '''
    fade_in_channel = 256
    inputs = Input((128, 128, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
   
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(64, 64))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(128, 128))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=128, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_256x256_discriminator():
    '''
        256 * 256 Discriminator
    '''
    fade_in_channel = 128
    inputs = Input((256, 256, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(128, 128))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(256, 256))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=64, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_512x512_discriminator():
    '''
        512 * 512 Discriminator
    '''
    fade_in_channel = 64
    inputs = Input((512, 512, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(256, 256))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(32, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(512, 512))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=32, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(512,512))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=64, filters2=128, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
    x = downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def model_builder(target_resolution):
    '''
        Helper function to build models
    '''
    generator = None
    discriminator = None
    if target_resolution == 4:
        generator = build_4x4_generator()
        discriminator = build_4x4_discriminator()
    elif target_resolution == 8:
        generator = build_8x8_generator()
        discriminator = build_8x8_discriminator()
    elif target_resolution == 16:
        generator = build_16x16_generator()
        discriminator = build_16x16_discriminator()
    elif target_resolution == 32:
        generator = build_32x32_generator()
        discriminator = build_32x32_discriminator()
    elif target_resolution == 64:
        generator = build_64x64_generator()
        discriminator = build_64x64_discriminator()
    elif target_resolution == 128:
        generator = build_128x128_generator()
        discriminator = build_128x128_discriminator()
    elif target_resolution == 256:
        generator = build_256x256_generator()
        discriminator = build_256x256_discriminator()
    elif target_resolution == 512:
        generator = build_512x512_generator()
        discriminator = build_512x512_discriminator()
    else:
        print("target resolution models are not defined yet")
    return generator, discriminator
