import tensorflow as tf

def cggan_discriminator_loss(real_images, generated_images):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_images), real_images, from_logits=True)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(generated_images), generated_images, from_logits=True)
    total_loss = real_loss + fake_loss
    return total_loss

def cggan_generator_loss(generated_images):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(generated_images), generated_images, from_logits=True)

def cycle_loss(real_image, cycled_image, l=10):
    return l*tf.math.reduce_mean(tf.abs(real_image-cycled_image))

def identity_loss(real_image, same_image, l=10):
    return 0.5*l*tf.math.reduce_mean(tf.abs(real_image-same_image))
