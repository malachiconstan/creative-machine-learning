import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from utils.custom_layers import EqualizedConv2D, EqualizedDense, NormalizationLayer, mini_batch_sd
from utils.config import BaseConfig
from utils.losses import wgan_loss
class Generator(tf.keras.Model):
    def __init__(self, latent_dim, name = 'Vanilla_GAN', upscale = False, **kwargs):
        super(Generator, self).__init__(name = name, **kwargs)
        self.latent_dim = latent_dim

        if upscale:
            self.body = tf.keras.Sequential([
                layers.Input(shape=(latent_dim,)),
                layers.Dense(8*8*1024),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Reshape((8,8,1024)),
                
                layers.Conv2DTranspose(512,(5,5),strides=(1,1),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(256,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(32,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh')
            ])
        else:
            self.body = tf.keras.Sequential([
                layers.Input(shape=(latent_dim,)),
                layers.Dense(8*8*256),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Reshape((8,8,256)),
                
                layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh')
            ])
    
    def call(self, X):
        X = self.body(X)

        return X

class Discriminator(tf.keras.Model):
    def __init__(self, name = 'Vanilla_GAN_Discriminator', upscale = False, **kwargs):
        super(Discriminator, self).__init__(name = name, **kwargs)

        if upscale:
            self.body = tf.keras.Sequential([
                layers.Input(shape=(256,256,3)),
                layers.Conv2D(32,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(64,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(128,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(256,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(512,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(1024,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Flatten(),
                layers.Dense(1)
            ])
        else:
            self.body = tf.keras.Sequential([
                layers.Input(shape=(32,32,3)),
                layers.Conv2D(32,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(64,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Conv2D(128,(5,5),strides=(2,2),padding='same'),
                layers.LeakyReLU(),
                layers.Dropout(0.3),

                layers.Flatten(),
                layers.Dense(1)
            ])
    
    def call(self, X):
        X = self.body(X)

        return X
class DownSampleBlock(tf.keras.Model):
    def __init__(self,
                filters,
                size,
                apply_instance_norm=True
                ):
        super(DownSampleBlock, self).__init__()
        self.apply_instance_norm = apply_instance_norm

        self.conv = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0.,0.02), use_bias=False)
        self.instance_norm = tfa.layers.InstanceNormalization()        
        self.relu = tf.keras.layers.LeakyReLU()

    def call(self, X):
        X = self.conv(X)
        if self.apply_instance_norm:
            X = self.instance_norm(X)
        X = self.relu(X)
        return X


class UpSampleBlock(tf.keras.Model):
    def __init__(self,
                filters,
                size,
                apply_dropout=False
                ):
        super(UpSampleBlock, self).__init__()
        self.apply_dropout = apply_dropout

        self.conv = tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0.,0.02), use_bias=False)
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.keras.layers.LeakyReLU()

    def call(self, X):
        X = self.conv(X)
        X = self.instance_norm(X)
        if self.apply_dropout:
            X = self.dropout(X)
        X = self.relu(X)
        return X

class CGGenerator(tf.keras.Model):
    def __init__(self, name = 'Cycle_GANG', output_channels=3, **kwargs):
        super(CGGenerator, self).__init__(name = name, **kwargs)

        # self.input_layer = tf.keras.layers.Input(shape=[256,256,3])

        self.down_stack = [
            DownSampleBlock(64,4,apply_instance_norm=False), # (?, 128, 128, 64)
            DownSampleBlock(128,4), # (?, 64, 64, 128)
            DownSampleBlock(256,4), # (?, 32, 32, 256)
            DownSampleBlock(512,4), # (?, 16, 16, 512)
            DownSampleBlock(512,4), # (?, 8, 8, 512)
            DownSampleBlock(512,4), # (?, 4, 4, 512)
            DownSampleBlock(512,4), # (?, 2, 2, 512)
            # DownSampleBlock(512,4), # (?, 1, 1, 512)
        ]

        self.up_stack = [
            UpSampleBlock(512,4,apply_dropout=True), # (?, 2, 2, 512)
            UpSampleBlock(512,4,apply_dropout=True), # (?, 4, 4, 512)
            UpSampleBlock(512,4,apply_dropout=True), # (?, 8, 8, 512)
            UpSampleBlock(256,4),                    # (?, 16, 16, 512)
            UpSampleBlock(128,4),                    # (?, 32, 32, 256)
            UpSampleBlock(64,4),                    # (?, 64, 64, 128)
            # UpSampleBlock(64,4),                     # (?, 128, 128, 64)
        ]

        self.last_layer = tf.keras.layers.Conv2DTranspose(output_channels,4,2,padding='same',kernel_initializer=tf.random_normal_initializer(0.,0.02),activation='tanh')
    
    def call(self, X):
        # X = self.input_layer(X)
        
        skips = []
        for dn in self.down_stack:
            X = dn(X)
            skips.append(X)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack,skips):
            X = up(X)
            X = tf.keras.layers.Concatenate()([X,skip]) 
        
        X = self.last_layer(X)

        return X

class CGDiscriminator(tf.keras.Model):
    def __init__(self, name = 'Cycle_GAND', **kwargs):
        super(CGDiscriminator, self).__init__(name = name, **kwargs)

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[128,128,3]),
            DownSampleBlock(64, 4, False),
            DownSampleBlock(128, 4),
            # DownSampleBlock(256, 4),
            tf.keras.layers.ZeroPadding2D(),
            tf.keras.layers.Conv2D(256,4,strides=1,kernel_initializer=tf.random_normal_initializer(0.,0.02),use_bias=False),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.ZeroPadding2D(),
            tf.keras.layers.Conv2D(1,4,strides=1,kernel_initializer=tf.random_normal_initializer(0.,0.02))
        ])
    
    def call(self, X):
        X = self.body(X)

        return X

class PGGenerator(tf.keras.Model):
    def __init__(self,
                latent_dim,
                level_0_channels,
                init_bias_zero=True,
                leaky_relu_leak=0.2,
                normalization=True,
                activation=None,
                output_dim=3,
                equalizedlR=True,
                name = 'PGGAN_Generator',
                **kwargs):
        """
        Build a generator for a progressive GAN model
        Args:
            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 ->
                               grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime
        """
        super(PGGenerator, self).__init__(name = name, **kwargs)

        self.equalizedlR = equalizedlR
        self.init_bias_zero = init_bias_zero

        # Initalize the scales
        self.scalesDepth = [level_0_channels]
        self.scaleLayers = list()
        self.toRGBLayers = list()

        # Initialize the scale 0
        self.initFormatLayer(latent_dim)
        self.dimOutput = output_dim
        self.groupScale0 = list()
        self.groupScale0.append(EqualizedConv2D(level_0_channels, 3, equalized=equalizedlR, init_bias_zero=init_bias_zero))

        # 1x1 Convolution to RGB
        self.toRGBLayers.append(EqualizedConv2D(self.dimOutput, 1, equalized=equalizedlR, init_bias_zero=init_bias_zero))

        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=leaky_relu_leak)

        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Upscaling
        self.upscale = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')

        # Last layer activation function
        self.generationActivation = activation
        self.level_0_channels = level_0_channels

    def initFormatLayer(self, latent_dim):
        """
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 xscalesDepth[0]
        layer.
        """

        self.latent_dim = latent_dim
        self.formatLayer = EqualizedDense(16 * self.scalesDepth[0], equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero)

    def getOutputSize(self):
        """
        Get the size of the generated image.
        """
        side = 4 * (2**(len(self.toRGBLayers) - 1))
        return (side, side)

    def addScale(self, depthNewScale):
        """
        Add a new scale to the model. Increasing the output resolution by
        a factor 2
        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """

        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(list())

        self.scaleLayers[-1].append(EqualizedConv2D(depthNewScale, 3, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero))
        self.scaleLayers[-1].append(EqualizedConv2D(depthNewScale, 3, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero))
        self.toRGBLayers.append(EqualizedConv2D(self.dimOutput, 1, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero))

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

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0""is defined")

        self._alpha = value

    def call(self, X, verbose=False):

        if verbose:
            print('')
            print('Calling Generator')

        X = tf.reshape(X,(X.shape[0], tf.math.reduce_prod(X.shape[1:])))
        # format layer
        X = self.leakyRelu(self.formatLayer(X))
        X = tf.reshape(X, (X.shape[0], 4, 4, self.level_0_channels))

        if self.normalizationLayer is not None:
            X = self.normalizationLayer(X)
        
        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            X = self.leakyRelu(convLayer(X))
            if self.normalizationLayer is not None:
                X = self.normalizationLayer(X)
        
        if verbose:
            print('Before Upsampling: ', X.shape)

        # Dirty, find a better way
        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](X)
            y = self.upscale(y)

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers):
            X = self.upscale(X)
            for convLayer in layerGroup:
                X = self.leakyRelu(convLayer(X))
                if self.normalizationLayer is not None:
                    X = self.normalizationLayer(X)

            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                # For the final loop only
                y = self.toRGBLayers[-2](X)
                y = self.upscale(y)(y)

            if verbose:
                print('Upscale: ',X.shape)

        # To RGB (no alpha parameter for now)
        X = self.toRGBLayers[-1](X)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            X = self.alpha * y + (1.0-self.alpha) * X

        if self.generationActivation is not None:
            X = self.generationActivation(X)

        if verbose:
            print('')

        return X

class PGDiscriminator(tf.keras.Model):
    def __init__(self,
                 level_0_channels,
                 init_bias_zero=True,
                 leaky_relu_leak=0.2,
                 decision_layer_size=1,
                 mini_batch_sd=True,
                 input_dim=3,
                 equalizedlR=True):
        """
        Build a discriminator for a progressive GAN model
        Args:
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(PGDiscriminator, self).__init__()

        # Initialization paramneters
        self.init_bias_zero = init_bias_zero
        self.equalizedlR = equalizedlR
        self.input_dim = input_dim

        # Initalize the scales
        self.scale_channels = [level_0_channels]
        self.scaleLayers = list()
        self.fromRGBLayers = list()

        self.mergeLayers = list()

        # Initialize the last layer
        self.init_decision_layer(decision_layer_size)

        # Layer 0
        self.groupScaleZero = list()
        self.fromRGBLayers.append(EqualizedConv2D(level_0_channels, 1, equalized=equalizedlR, init_bias_zero=init_bias_zero))

        # Minibatch standard deviation
        self.mini_batch_sd = mini_batch_sd
        self.groupScaleZero.append(EqualizedConv2D(level_0_channels, 3, equalized=equalizedlR, init_bias_zero=init_bias_zero))
        self.groupScaleZero.append(EqualizedDense(level_0_channels, equalized=equalizedlR, init_bias_zero=init_bias_zero))

        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=leaky_relu_leak)

        # Pooling layer
        self.avg_pool2d = tf.keras.layers.AveragePooling2D(pool_size=(2,2))

    def addScale(self, new_level_channels):

        last_level_channels = self.scale_channels[-1]
        self.scale_channels.append(new_level_channels)

        self.scaleLayers.append(list())

        self.scaleLayers[-1].append(EqualizedConv2D(new_level_channels, 3, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero))
        self.scaleLayers[-1].append(EqualizedConv2D(last_level_channels, 3, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero))

        self.fromRGBLayers.append(EqualizedConv2D(new_level_channels, 1, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero))

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

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0""is defined")

        self._alpha = value

    def init_decision_layer(self, decision_layer_size):

        self.decisionLayer = EqualizedDense(decision_layer_size, equalized=self.equalizedlR, init_bias_zero=self.init_bias_zero)

    def call(self, X, getFeature = False, verbose=False):
        
        if verbose:
            print('')
            print('Calling Discriminator')

        # Alpha blending
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = self.avg_pool2d(X)
            y = self.leakyRelu(self.fromRGBLayers[- 2](y))

        # From RGB layer
        X = self.leakyRelu(self.fromRGBLayers[-1](X))

        if verbose:
            print('Before Reduction: ', X.shape)

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        shift = len(self.fromRGBLayers) - 2
        for groupLayer in reversed(self.scaleLayers):

            for layer in groupLayer:
                X = self.leakyRelu(layer(X))

            X = self.avg_pool2d(X)

            if mergeLayer:
                mergeLayer = False
                X = self.alpha * y + (1-self.alpha) * X
            if verbose:
                print('Reduce: ',X.shape)
            shift -= 1

        # Now the scale 0

        # Minibatch standard deviation
        if self.mini_batch_sd:
            X = mini_batch_sd(X)

        X = self.leakyRelu(self.groupScaleZero[0](X))
        if verbose:
            print('Before linear reshape: ',X.shape)
        X = tf.reshape(X, (X.shape[0], tf.math.reduce_prod(X.shape[1:])))

        if verbose:
            print('After linear reshape: ',X.shape)
        X = self.leakyRelu(self.groupScaleZero[1](X))

        out = self.decisionLayer(X)

        if verbose:
            print('')

        if not getFeature:
            return out

        return out, X

class ProgressiveGAN(object):
    def __init__(self,
                latent_dim=512,
                level_0_channels=512,
                init_bias_zero=True,
                leaky_relu_leak=0.2,
                per_channel_normalisation=True,
                mini_batch_sd=False,
                equalizedlR=True,
                output_dim=3,
                GDPP=False,
                lambdaGP=0.,
                depthOtherScales = [],
                **kwargs):
    
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.level_0_channels = level_0_channels
        self.config.init_bias_zero = init_bias_zero
        self.config.leaky_relu_leak = leaky_relu_leak
        self.config.depthOtherScales = depthOtherScales
        self.config.per_channel_normalisation = per_channel_normalisation
        self.config.alpha = 0
        self.config.mini_batch_sd = mini_batch_sd
        self.config.equalizedlR = equalizedlR
        
        self.config.latent_dim = latent_dim
        self.config.output_dim = output_dim

        self.config.GDPP = GDPP

        # WGAN-GP
        self.loss_criterion = wgan_loss
        self.config.lambdaGP = lambdaGP

        self.netD = self.getNetD()
        self.netG = self.getNetG()

    def infer(self, input, toCPU=True):
        """
        Generate some data given the input latent vector.
        Args:
            input (torch.tensor): input latent vector
        """
        if toCPU:
            return tf.stop_gradient(self.netG(input)).cpu()
        else:
            return tf.stop_gradient(self.netG(input))

    def getNetG(self):
        gnet = PGGenerator(self.config.latent_dim,
                        self.config.level_0_channels,
                        init_bias_zero=self.config.init_bias_zero,
                        leaky_relu_leak=self.config.leaky_relu_leak,
                        normalization=self.config.per_channel_normalisation,
                        activation=None,
                        output_dim=self.config.output_dim,
                        equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.alpha = self.config.alpha

        return gnet

    def getNetD(self):
        dnet = PGDiscriminator(self.config.level_0_channels,
                            init_bias_zero=self.config.init_bias_zero,
                            leaky_relu_leak=self.config.leaky_relu_leak,
                            decision_layer_size=1,
                            mini_batch_sd=self.config.mini_batch_sd,
                            input_dim=self.config.output_dim,
                            equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.alpha = self.config.alpha

        return dnet

    def addScale(self, new_level_channels):
        """
        Add a new scale to the model. The output resolution becomes twice
        bigger.
        """
        self.netG = self.getNetG()
        self.netD = self.getNetD()

        self.netG.addScale(new_level_channels)
        self.netD.addScale(new_level_channels)

        self.config.depthOtherScales.append(new_level_channels)

    def updateAlpha(self, newAlpha):
        """
        Update the blending factor alpha.
        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        print("Changing alpha to %.3f" % newAlpha) 

        self.getNetG().alpha = newAlpha
        self.getNetD().alpha = newAlpha

        self.config.alpha = newAlpha

    def getSize(self):
        """
        Get output image size (W, H)
        """
        return self.getNetG().getOutputSize()
