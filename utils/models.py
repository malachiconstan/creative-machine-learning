import tensorflow as tf
import tensorflow.keras.layers as layers
try:
    import tensorflow_addons as tfa
except:
    print('TFA cannot be imported')

from utils.custom_layers import EqualizeLearningRate, pg_upsample_block, pg_downsample_block, generator_input_block, discriminator_output_block
from utils.pggan_cl import model_builder

def get_classifier(input_shape, num_classes=19):
    '''
    Returns a pre-trained efficientnetb3 model
    
    :params:
        tuple input_shape: Input shape of model e.g. (128,128,3)
        int num_classes: Number of output classes for model
    
    :return:
        tf.keras.Model: Pre-trained imagenet model
    '''

    # Initialise base model with pre-trained weights
    base_model = tf.keras.applications.EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet')

    # Get preprocessing layer
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)
    
    # Build model using Keras functional API
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

class Generator(tf.keras.Model):
    '''
    Defines a DCGAN Generator
    '''
    def __init__(self, latent_dim, name = 'Vanilla_GAN', upscale = False, **kwargs):
        super(Generator, self).__init__(name = name, **kwargs)
        self.latent_dim = latent_dim

        if upscale:
            self.body = tf.keras.Sequential([
                layers.Input(shape=(latent_dim,)),
                layers.Dense(8*8*512),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Reshape((8,8,512)),
                
                layers.Conv2DTranspose(256,(5,5),strides=(1,1),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),

                # layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False),
                # layers.BatchNormalization(),
                # layers.LeakyReLU(),

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
    '''
    Defines a DCGAN Discriminator
    '''
    def __init__(self, name = 'Vanilla_GAN_Discriminator', upscale = False, **kwargs):
        super(Discriminator, self).__init__(name = name, **kwargs)

        if upscale:
            self.body = tf.keras.Sequential([
                layers.Input(shape=(128,128,3)),
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

                # layers.Conv2D(1024,(5,5),strides=(2,2),padding='same'),
                # layers.LeakyReLU(),
                # layers.Dropout(0.3),

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
    '''
    Defines a CGAN Downsample Block
    '''
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
    '''
    Defines a CGAN Upsample Block
    '''
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
    '''
    Defines a CGAN Generator
    '''
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
    '''
    Defines a CGAN Discriminator
    '''
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

def pg_generator(
        output_resolution=4,
        latent_dim=512,
        filters={4:512, 8:512, 16:512, 32:512, 64:256, 128:128, 256:64, 512:32},
        leaky_relu_leak=0.2,
        kernel_initializer='he_normal',
        output_activation = tf.keras.activations.tanh,
        name = 'PGGAN_Generator'
    ):
    '''
    Returns a function PGGAN Generator
    Fully adapted from https://github.com/henry32144/pggan-tensorflow
    Original Nvidia Model: https://github.com/tkarras/progressive_growing_of_gans
    
    :params:
        int output_resolution: Output resolution of generator
        int latent_dim: Latent Dimension of input noise
        dict filters: Number of input and output filters at all resolutions
        float leaky_relu_leak: Alpha for leaky ReLU leak
        str kernel_initializer: Type of kernel_initializer to use
        tf.keras.activations.Activation output_activation: Type of output activation for layers
        str name: Name of model
    
    :return:
        tf.keras.Model: PGGAN Generator at specified resolution
    '''
    # Check output_resolution
    if output_resolution not in [4,8,16,32,64,128,256,512]:
        raise ValueError("resolution must be in [4,8,16,32,64,128,256,512]")

    # Declare Inputs
    z = tf.keras.Input(latent_dim, name=name+'_input')
    alpha = tf.keras.Input((1), name='input_alpha')
    
    # Create input block
    X = generator_input_block(z, kernel_initializer=kernel_initializer, leaky_relu_alpha=leaky_relu_leak, latent_dim=latent_dim)

    # Output immediately if output resolution is 4
    if output_resolution == 4:
        to_rgb = EqualizeLearningRate(tf.keras.layers.Conv2D(3,
                                                        kernel_size=1,
                                                        strides=1,
                                                        padding='same',
                                                        activation=output_activation,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer='zeros'), 
                                                        name='to_rgb_4x4')
        X = to_rgb(X)

        return tf.keras.Model(inputs=[z, alpha], outputs=X)

    # Create additional blocks otherwise
    resolution = 8
    while resolution <= output_resolution:
        X, upsampled_X = pg_upsample_block(X,
                                            input_filters=filters[resolution//2],
                                            output_filters=filters[resolution],
                                            kernel_size=3,
                                            strides=1,
                                            padding='same',
                                            leaky_relu_alpha=leaky_relu_leak,
                                            kernel_initializer='he_normal',
                                            name=f'Up_{resolution}x{resolution}')
        resolution *= 2
    
    # Build left and right fade in branches
    to_rgb_current_resolution = EqualizeLearningRate(tf.keras.layers.Conv2D(3,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='same',
                                                    activation=output_activation,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'), 
                                                    name=f'to_rgb_{output_resolution}x{output_resolution}')
    
    to_rgb_previous_resolution = EqualizeLearningRate(tf.keras.layers.Conv2D(3,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='same',
                                                    activation=output_activation,
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer='zeros'), 
                                                    name=f'to_rgb_{output_resolution//2}x{output_resolution//2}')
    l_X = to_rgb_current_resolution(X)
    r_X = to_rgb_previous_resolution(upsampled_X)

    # Left Branch
    l_X = tf.keras.layers.Multiply()([alpha, l_X])

    # Right branch
    r_X = tf.keras.layers.Multiply()([1-alpha, r_X])

    output = tf.keras.layers.Add()([l_X, r_X])

    return tf.keras.Model(inputs=[z, alpha], outputs=output, name=name)

def pg_discriminator(
        input_resolution=4,
        filters={4:512, 8:512, 16:512, 32:512, 64:256, 128:128, 256:64, 512:32},
        leaky_relu_leak=0.2,
        kernel_initializer='he_normal',
        name = 'PGGAN_Discriminator'
    ):
    '''
    Returns a function PGGAN Discriminator.
    Fully adapted from https://github.com/henry32144/pggan-tensorflow
    Original Nvidia Model: https://github.com/tkarras/progressive_growing_of_gans 
    
    :params:
        int input_resolution: Input resolution of discriminator
        float leaky_relu_leak: Alpha for leaky ReLU leak
        str kernel_initializer: Type of kernel_initializer to use
        str name: Name of model
    
    :return:
        tf.keras.Model: PGGAN Generator at specified resolution
    '''

    # Define inputs
    input_images = tf.keras.Input((input_resolution, input_resolution, 3), name=name+'_input')
    alpha = tf.keras.Input((1), name='input_alpha')

    # Build upconv layers if input resolution is 4
    if input_resolution == 4:
        from_rgb = EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                            kernel_size=1,
                                                            strides=1,
                                                            padding='same',
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer='zeros'),
                                                            name=f'from_rgb_{4}x{4}')

        conv2d_up_channel = EqualizeLearningRate(tf.keras.layers.Conv2D(512,
                                                                        kernel_size=1,
                                                                        strides=1,
                                                                        padding='same',
                                                                        activation=tf.nn.leaky_relu,
                                                                        kernel_initializer=kernel_initializer,
                                                                        bias_initializer='zeros'), name='conv2d_up_channel')

        X = from_rgb(input_images)
        X = conv2d_up_channel(X)
        X = discriminator_output_block(X, kernel_initializer, leaky_relu_leak)
        return tf.keras.Model(inputs=[input_images, alpha], outputs=X)

    # Otherwise build left and right branches
    # Left branch
    from_rgb_current_resolution = EqualizeLearningRate(tf.keras.layers.Conv2D(filters[input_resolution],
                                                                            kernel_size=1,
                                                                            strides=1,
                                                                            padding='same',
                                                                            activation=tf.nn.leaky_relu,
                                                                            kernel_initializer=kernel_initializer,
                                                                            bias_initializer='zeros'), 
                                                                            name=f'from_rgb_{input_resolution}x{input_resolution}')


    from_rgb_previous_resolution = EqualizeLearningRate(tf.keras.layers.Conv2D(filters[input_resolution//2],
                                                                            kernel_size=1,
                                                                            strides=1,
                                                                            padding='same',
                                                                            activation=tf.nn.leaky_relu,
                                                                            kernel_initializer=kernel_initializer,
                                                                            bias_initializer='zeros'), 
                                                                            name=f'from_rgb_{input_resolution//2}x{input_resolution//2}')

    # Left branch
    l_X = tf.keras.layers.AveragePooling2D(pool_size=2)(input_images)
    l_X = from_rgb_previous_resolution(l_X)
    l_X = tf.keras.layers.Multiply()([1-alpha, l_X])

    # Right branch
    r_X = from_rgb_current_resolution(input_images)
    r_X = pg_downsample_block(r_X,
                            output_filters1=filters[input_resolution],
                            output_filters2=filters[input_resolution//2],
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            leaky_relu_alpha=leaky_relu_leak,
                            kernel_initializer='he_normal',
                            name=f'Down_{input_resolution}x{input_resolution}')
    r_X = tf.keras.layers.Multiply()([alpha, r_X])

    X = tf.keras.layers.Add()([l_X, r_X])
    resolution = input_resolution//2

    # Add additional blocks
    while resolution >= 8:
        X = pg_downsample_block(X,
                            output_filters1=filters[resolution],
                            output_filters2=filters[resolution//2],
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            leaky_relu_alpha=leaky_relu_leak,
                            kernel_initializer='he_normal',
                            name=f'Down_{resolution}x{resolution}')
        resolution //= 2

    X = discriminator_output_block(X, kernel_initializer, leaky_relu_leak)
    return tf.keras.Model(inputs=[input_images, alpha], outputs=X, name=name)

class ProgressiveGAN(object):
    '''
    PGGAN Class containing both generator and discriminator
    '''
    def __init__(self,
                resolution=4,
                latent_dim=512,
                filters={4:512, 8:512, 16:512, 32:512, 64:256, 128:128, 256:64, 512:32},
                leaky_relu_leak=0.2,
                kernel_initializer='he_normal',
                output_activation = tf.keras.activations.tanh,
                original=True,
                **kwargs):
        '''
        Constructor
        :params:
            int resolution: PGGAN Resolution
            int latent_dim: Latent Dimension of input noise
            dict filters: Number of input and output filters at all resolutions
            float leaky_relu_leak: Alpha for leaky ReLU leak
            str kernel_initializer: Type of kernel_initializer to use
            tf.keras.activations.Activation output_activation: Type of output activation for layers
        '''

        self.original = original
        
        # Store parameters
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.filters = filters
        self.leaky_relu_leak = leaky_relu_leak
        self.kernel_initializer = kernel_initializer
        self.output_activation = output_activation

        self.__initialise_models()

    def __initialise_models(self):
        '''
        Initialise generator and discriminator
        '''
        if self.original:
            self.Generator, self.Discriminator = model_builder(self.resolution)
        else:
            self.Discriminator = pg_discriminator(
                self.resolution,
                self.leaky_relu_leak,
                self.kernel_initializer
            )

            self.Generator = pg_generator(
                self.resolution,
                self.latent_dim,
                self.leaky_relu_leak,
                self.kernel_initializer,
                self.output_activation
            )

    def double_resolution(self):
        '''
        Doubles resolution of PGGAN. Note: Old weights have to be loaded manually. This does not load back the weights of the previous model
        '''

        self.resolution *= 2
        self.__initialise_models()
        
        print(f'Resolution Doubled to {self.resolution}. New Model Built. Load back weights')
