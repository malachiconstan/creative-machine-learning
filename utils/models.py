import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from utils.custom_layers import EqualizeLearningRate, PGUpSampleBlock, PGDowmSampleBlock, GeneratorInputBlock, DiscriminatorOutputBlock #EqualizedConv2D, EqualizedDense, NormalizationLayer, mini_batch_sd
from utils.config import BaseConfig
from utils.losses import wgan_loss

def get_classifier(input_shape, num_classes=19):

    base_model = tf.keras.applications.EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet')
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

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
                output_resolution=4,
                latent_dim=512,
                leaky_relu_leak=0.2,
                kernel_initializer='he_normal',
                output_activation = tf.keras.activations.tanh,
                name = 'PGGAN_Generator',
                **kwargs):

        super(PGGenerator, self).__init__(name = name, **kwargs)

        self.leaky_relu_leak = leaky_relu_leak
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer

        # Define Model
        self.input_block = GeneratorInputBlock(kernel_initializer=kernel_initializer,leaky_relu_alpha=leaky_relu_leak,latent_dim=latent_dim)
        self.upsample_blocks = {}

        # 1x1 Convolution to RGB
        self.to_rgb = {4:EqualizeLearningRate(tf.keras.layers.Conv2D(3,
                                                                    kernel_size=1,
                                                                    strides=1,
                                                                    padding='same',
                                                                    activation=self.output_activation,
                                                                    kernel_initializer=self.kernel_initializer,
                                                                    bias_initializer='zeros'), 
                                                                    name=f'to_rgb_{4}x{4}')}
        self.alpha = 0

        # Add resolution layers
        if output_resolution not in [4,8,16,32,64,128,256,512]:
            raise ValueError("resolution must be in [4,8,16,32,64,128,256,512]")

        self.resolution = 4
        self.output_resolution = output_resolution
        while self.resolution < output_resolution:
            self.double_resolution()

    def get_output_resolution(self):
        """
        Get the size of the generated image.
        """
        return self.output_resolution

    def double_resolution(self):
        """
        Add a new scale to the model. Increasing the output resolution by
        a factor 2
        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        self.resolution *= 2
        self.upsample_blocks[self.resolution] = PGUpSampleBlock(input_filters=512,
                                                                output_filters=512,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='same',
                                                                activation_layer=tf.keras.layers.LeakyReLU(alpha=self.leaky_relu_leak),
                                                                kernel_initializer='he_normal',
                                                                name=f'Up_{self.resolution}x{self.resolution}')

        self.to_rgb[self.resolution] = EqualizeLearningRate(tf.keras.layers.Conv2D(3,
                                                                    kernel_size=1,
                                                                    strides=1,
                                                                    padding='same',
                                                                    activation=self.output_activation,
                                                                    kernel_initializer=self.kernel_initializer,
                                                                    bias_initializer='zeros'), 
                                                                    name=f'to_rgb_{self.resolution}x{self.resolution}')
        print('Resolution doubled. Current resolution: ', self.resolution)

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

    @property
    def resolution(self):
        """
        Get alpha value
        """
        return self._res

    @resolution.setter
    def resolution(self, value):
        """
        Update the value of the merging factor resolution
        Args:
            - resolution (int): merging factor, must be in [4,8,16,32,64,128,256,512]
        """
        if value not in [4,8,16,32,64,128,256,512]:
            raise ValueError("resolution must be in [4,8,16,32,64,128,256,512]")

        self._res = value

    def call(self, X, verbose=False):

        if verbose:
            print('')
            print('Calling Generator')

        X = self.input_block(X)

        if self.resolution == 4:
            X = self.to_rgb[self.resolution](X)
            return X

        # Fade In
        res = 8
        while res <= self.resolution:
            X, upsampled_X = self.upsample_blocks[res](X)
            res *= 2

        l_X = self.to_rgb[self.resolution](X)
        r_X = self.to_rgb[self.resolution//2](upsampled_X)

        # Left Branch
        l_X = (1-self.alpha)*l_X

        # Right branch
        r_X = self.alpha*r_X

        return l_X + r_X

class PGDiscriminator(tf.keras.Model):
    def __init__(self,
                input_resolution=4,
                leaky_relu_leak=0.2,
                kernel_initializer='he_normal',
                name = 'PGGAN_Discriminator',
                **kwargs):

        super(PGDiscriminator, self).__init__(name = name, **kwargs)

        self.leaky_relu_leak = leaky_relu_leak
        self.kernel_initializer = kernel_initializer

        # Define Model
        self.output_block = DiscriminatorOutputBlock()
        self.downsample_blocks = {}

        # 1x1 Convolution From RGB
        self.from_rgb = {4:EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                                    kernel_size=1,
                                                                    strides=1,
                                                                    padding='same',
                                                                    activation=tf.nn.leaky_relu,
                                                                    kernel_initializer=self.kernel_initializer,
                                                                    bias_initializer='zeros'),
                                                                    name=f'from_rgb_{4}x{4}')}
        self.alpha = 0

        # Special Channel for 4x4
        self.conv2d_up_channel = EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                                            kernel_size=1,
                                                                            strides=1,
                                                                            padding='same',
                                                                            activation=tf.nn.leaky_relu,
                                                                            kernel_initializer=self.kernel_initializer,
                                                                            bias_initializer='zeros'), name='conv2d_up_channel')

        # Upscaling
        self.downscale = tf.keras.layers.AveragePooling2D(pool_size=2)

        # Add resolution layers
        if input_resolution not in [4,8,16,32,64,128,256,512]:
            raise ValueError("resolution must be in [4,8,16,32,64,128,256,512]")

        self.resolution = 4
        self.input_resolution = input_resolution
        while self.resolution < input_resolution:
            self.double_resolution()

    def double_resolution(self):
        """
        Add a new scale to the model. Increasing the output resolution by
        a factor 2
        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        self.resolution *= 2
        self.downsample_blocks[self.resolution] = PGDowmSampleBlock(output_filters1=512,
                                                                output_filters2=512,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='same',
                                                                activation_layer=tf.keras.layers.LeakyReLU(alpha=self.leaky_relu_leak),
                                                                kernel_initializer='he_normal',
                                                                name=f'Down_{self.resolution}x{self.resolution}')

        self.from_rgb[self.resolution] = EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                                    kernel_size=1,
                                                                    strides=1,
                                                                    padding='same',
                                                                    activation=tf.nn.leaky_relu,
                                                                    kernel_initializer=self.kernel_initializer,
                                                                    bias_initializer='zeros'), 
                                                                    name=f'from_rgb_{self.resolution}x{self.resolution}')
        print('Resolution doubled. Current resolution: ', self.resolution)

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

    @property
    def resolution(self):
        """
        Get alpha value
        """
        return self._res

    @resolution.setter
    def resolution(self, value):
        """
        Update the value of the merging factor resolution
        Args:
            - resolution (int): merging factor, must be in [4,8,16,32,64,128,256,512]
        """
        if value not in [4,8,16,32,64,128,256,512]:
            raise ValueError("resolution must be in [4,8,16,32,64,128,256,512]")

        self._res = value

    def call(self, input, verbose=False):
        assert input.shape[1] == self.resolution, 'Input Shape must be equal to resolution'
        
        if verbose:
            print('')
            print('Calling Discriminator')

        if self.resolution == 4:
            X = self.from_rgb[4](input)
            X = self.conv2d_up_channel(X)
            return X

        # Left branch
        l_X = self.downscale(input)
        l_X = self.from_rgb[self.resolution//2](l_X)
        l_X = (1-self.alpha)*l_X

        # Right branch
        r_X = self.from_rgb[self.resolution](input)
        r_X = self.downsample_blocks[self.resolution](r_X)
        r_X = self.alpha*r_X

        X = l_X + r_X
        res = 8
        while res < self.resolution:
            X = self.downsample_blocks[res](X)
            res *= 2

        X = self.output_block(X)
        return X

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
