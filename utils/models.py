import tensorflow as tf
import tensorflow.keras.layers as layers

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