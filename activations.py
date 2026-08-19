# activation functions and their derivatives. eevery activation g is applied elementwise
# to a pre-activation matrix z, where z = w . a_prev + b for some layer. we need both 
# g(z) (used in forward pass) and g'(z) (used in the backward pass, via the chain rule).

import numpy as np


def relu(z):
    """
    rectified linear unit.

    formula:
        relu(z) = max(0, z)

    elementwise: any negative entry gets clipped to 0, positive entries
    pass through unchanged. this is what lets the network learn
    non-linear decision boundaries. without a non-linearity here,
    stacking w1.x + b1 then w2.(...) + b2 would just collapse into one
    big linear function, no matter how many layers we stack.
    """
    return np.maximum(0.0, z)


def relu_prime(z):
    """
    derivative of relu, needed for backprop.

    formula:
        relu'(z) = 1 if z > 0
                   0 if z <= 0

    (technically relu is not differentiable at z = 0, but in practice we
    just define the derivative as 0 there cz it never matters numerically)

    during backprop this acts as a 0/1 mask: dz = da * relu'(z), ie the
    upstream gradient da is passed straight through wherever neuron was 
    "on" (z > 0) during the forward pass, and killed wherever it was "off"
    """
    return (z > 0).astype(z.dtype)


def softmax(z):
    """
    softmax turns a vector of raw scores ("logits") into a probability
    distribution over classes. applied per-column, where each column is
    one training example's vector of 10 class scores.

    formula, for class i out of k classes:
        softmax(z)_i = exp(z_i) / sum_{j=1..k} exp(z_j)

    properties: every output is in (0, 1), and the outputs for one
    example sum to exactly 1, so we can read them as p(class = i | x).

    numerical stability trick:
        softmax(z)_i = exp(z_i - m) / sum_j exp(z_j - m)
    for any constant m, since it cancels top and bottom:
        exp(z_i - m) / sum_j exp(z_j - m)
      = exp(z_i)*exp(-m) / (exp(-m) * sum_j exp(z_j))
      = exp(z_i) / sum_j exp(z_j)
    we pick m = max(z) per example so the largest exponent is exp(0) = 1,
    which avoids overflow from exponentiating large positive logits.
    """
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)