import tensorflow as tf

class WGANGP(object):
    """
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self):

        self.activation = None
        self.decision_layer_size = 1

    def getCriterion(self, y_pred, status):
        """
        Given an input tensor and its targeted status (detected as real or
        detected as fake) build the associated loss
        Args:
            - y_pred (Tensor): decision tensor build by the model's discrimator
            - status (bool): if True -> this tensor should have been detected
                             as a real input
                             else -> it shouldn't have
        """
        if status:
            return -1*tf.math.reduce_sum(y_pred[:, 0])
        return tf.math.reduce_sum(y_pred[:, 0])

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

    batch_size = real_images.shape[0]
    # alpha = torch.rand(batchSize, 1)
    alpha = tf.random.uniform(shape=[batch_size, 1])
    alpha = tf.reshape(tf.broadcast_to(alpha, shape=(batch_size, int(tf.size(real_images)/batch_size))), shape=real_images.shape)
    # alpha = alpha.expand(batchSize, int(input.nelement() /batch_size)).contiguous().view(input.size())
    # alpha = alpha.to(input.device)
    interpolated_images = tf.Variable(alpha * real_images + ((1 - alpha) * fake_images))
    # interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    with tf.GradientTape() as tape:
        interpolated_output = tf.math.reduce_sum(discriminator(interpolated_images)[:, 0])

    gradients = tape.gradient(interpolated_output, interpolated_images)

    # gradients = torch.autograd.grad(outputs=decisionInterpolate,inputs=interpolates,create_graph=True, retain_graph=True)

    gradients = tf.reshape(gradients[0], shape=(batch_size, -1)) 

    # gradients = gradients[0].view(batchSize, -1)
    gradients = tf.math.sqrt(tf.math.reduce_sum(gradients * gradients, axis=1))

    # gradients = ().sum(dim=1).sqrt()
    # gradient_penalty = (((gradients - 1.0)**2)).sum() * weight
    gradient_penalty = weight * tf.math.reduce_sum((gradients - 1.0)**2)

    return gradient_penalty