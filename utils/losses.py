import tensorflow as tf

class WGANGP(object):
    """
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self):

        self.activation = None
        self.decision_layer_size = 1

    def getCriterion(self, input, status):
        """
        Given an input tensor and its targeted status (detected as real or
        detected as fake) build the associated loss
        Args:
            - input (Tensor): decision tensor build by the model's discrimator
            - status (bool): if True -> this tensor should have been detected
                             as a real input
                             else -> it shouldn't have
        """
        if status:
            return -input[:, 0].sum()
        return input[:, 0].sum()

def WGANGPGradientPenalty(input, fake, discriminator, weight, backward=True):
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

    batchSize = input.shape[0]
    alpha = torch.rand(batchSize, 1)
    alpha = alpha.expand(batchSize, int(input.nelement() /batchSize)).contiguous().view(input.size())
    alpha = alpha.to(input.device)
    interpolates = alpha * input + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(
        interpolates, requires_grad=True)

    decisionInterpolate = discriminator(interpolates, False)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()