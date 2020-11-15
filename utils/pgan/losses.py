import tensorflow as tf
# Reference : https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py


# def js_loss(logits_real, logits_fake, smooth_factor=0.9):
#     # discriminator loss for real/fake classification
#     d_loss_real = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=logits_real, labels=tf.ones_like(logits_real) * smooth_factor))
#     d_loss_fake = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=logits_fake, labels=tf.zeros_like(logits_fake)))
#     d_loss = d_loss_real + d_loss_fake

#     # generator loss for fooling discriminator
#     g_loss = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=logits_fake, labels=tf.ones_like(logits_fake)))
#     return d_loss, g_loss


def wgan_loss(d_real, d_fake):
    # Standard WGAN loss
    g_loss = -tf.math.reduce_mean(d_fake)
    d_loss = tf.math.reduce_mean(d_fake) - tf.math.reduce_mean(d_real)
    return d_loss, g_loss

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([real_output.shape[0],1],0.7,1.2), real_output, from_logits=True) # set noise to 1
    fake_loss = tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0,0.3), fake_output, from_logits=True) # set noise to 0
    total_loss = tf.math.reduce_mean(real_loss) + tf.math.reduce_mean(fake_loss)
    return total_loss

def generator_loss(fake_output):
    return tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(tf.random.uniform([fake_output.shape[0],1],0.7,1.2), fake_output, from_logits=True)) # set noise to 1

