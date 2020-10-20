import tensorflow as tf
import tensorflow.keras.layers as layers

from utils.custom_layers import EqualizedConv2D, EqualizedDense, NormalizationLayer, Upscale2d

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, name = 'Vanilla_GAN', **kwargs):
        super(Generator, self).__init__(name = name, **kwargs)
        self.latent_dim = latent_dim

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
    def __init__(self, name = 'Vanilla_GAN_Discriminator', **kwargs):
        super(Discriminator, self).__init__(name = name, **kwargs)

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

class PGGenerator(tf.keras.Model):
    def __init__(self,
                latent_dim,
                depth_scale_0,
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
        self.scalesDepth = [depth_scale_0]
        self.scaleLayers = list()
        self.toRGBLayers = list()

        # Initialize the scale 0
        self.initFormatLayer(latent_dim)
        self.dimOutput = output_dim
        self.groupScale0 = list()
        self.groupScale0.append(EqualizedConv2D(depth_scale_0, 3, equalized=equalizedlR, init_bias_zero=init_bias_zero))

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

        # Last layer activation function
        self.generationActivation = activation
        self.depth_scale_0 = depth_scale_0

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

    def call(self, X):
        ## Normalize the input
        if self.normalizationLayer is not None:
            X = self.normalizationLayer(X)

        X = tf.reshape(X,(X.shape[0], tf.math.reduce_prod(X.shape[1:])))
        # format layer
        X = self.leakyRelu(self.formatLayer(X))
        print(X.shape)
        X = tf.reshape(X, (X.shape[0], 4, 4, self.depth_scale_0))
        print(X.shape)

        if self.normalizationLayer is not None:
            X = self.normalizationLayer(X)
        
        print(X.shape)
        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            X = self.leakyRelu(convLayer(X))
            if self.normalizationLayer is not None:
                X = self.normalizationLayer(X)
        
        print(X.shape)
        # Dirty, find a better way
        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](X)
            y = Upscale2d(y)
            print('y',y.shape)

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers):
            X = Upscale2d(X)
            for convLayer in layerGroup:
                X = self.leakyRelu(convLayer(X))
                if self.normalizationLayer is not None:
                    X = self.normalizationLayer(X)

            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                # For the final loop only
                y = self.toRGBLayers[-2](X)
                y = Upscale2d(y)

        # To RGB (no alpha parameter for now)
        X = self.toRGBLayers[-1](X)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            X = self.alpha * y + (1.0-self.alpha) * X

        if self.generationActivation is not None:
            X = self.generationActivation(X)

        return X