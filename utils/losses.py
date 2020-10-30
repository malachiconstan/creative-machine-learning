import tensorflow as tf

# class WGANGP(object):
#     """
#     Paper WGANGP loss : linear activation for the generator.
#     https://arxiv.org/pdf/1704.00028.pdf
#     """

#     def __init__(self):

#         self.activation = None
#         self.decision_layer_size = 1

#     def getCriterion(self, y_pred, status):
#         """
#         Given an input tensor and its targeted status (detected as real or
#         detected as fake) build the associated loss
#         Args:
#             - y_pred (Tensor): decision tensor build by the model's discrimator
#             - status (bool): if True -> this tensor should have been detected
#                              as a real input
#                              else -> it shouldn't have
#         """
#         assert len(y_pred.shape) == 2 and y_pred.shape[1] == 1, 'Shape of y pred should be of length 2, and axis-1 only has 1 value, i.e. [N, 1]'
#         if status:
#             # Wasserstein loss for real images is negative
#             return -1*tf.math.reduce_mean(y_pred)
#         # Wasserstein loss for fake images is positive
#         return tf.math.reduce_mean(y_pred)
    
def wgan_loss(y_pred, predict_real=True):
    if predict_real:
        return -tf.math.reduce_mean(y_pred)
    else:
        return tf.math.reduce_mean(y_pred)

def WGANGPGradientPenalty(real_images, fake_images, discriminator, weight):
    """
    Gradient penalty as described in
    "Improved Training of Wasserstein GANs"
    https://arxiv.org/pdf/1704.00028.pdf
    Args:
        - input (Tensor): batch of real data
        - fake (Tensor): batch of generated data. Must have the same size
          as the input
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """
    # 1. get the interplated image
    assert real_images.shape == fake_images.shape, 'Real and fake image batch are of different shapes'
    
    batch_size = real_images.shape[0]
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gradient_penalty_tape:
        gradient_penalty_tape.watch(interpolated)
        # 2. Get the discriminator output for this interpolated image.
        interpolated_prediction = discriminator(interpolated, training=True)

    # 3. Calculate the gradients w.r.t to this interpolated image.
    gradient = gradient_penalty_tape.gradient(interpolated_prediction, [interpolated])[0]
    # 4. Calcuate the norm of the gradients
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty

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
