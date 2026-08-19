# weight initialization: why not just start every weight at 0 or at some fixed value? 
# (1) symmetry breaking: if every weight in a layer starts identical, every neuron in 
# that layer computes the exact same function of the input and receives the exact same 
# gradient, forever, theyd never ifferentiate from each other. random init breaks that, 
# and (2) controlling the variance of activations as they pass through many layers, 
# because if weights are drawn with too large a variance, preactivations z = w.a + b 
# explode layer over layer; too small, and they shrink toward 0 (vanishing signal, 
# vanishing gradients). we want var(z) to stay roughly constant from layer to layer

import numpy as np


def he_init(layer_dims, seed=None):
    """
    "he" initialization (kaiming), tuned for relu activated layers.

    for a layer with n_in inputs and n_out outputs, draw each weight as
        w_ij ~ normal(mean=0, std = sqrt(2 / n_in))
    i.e. w = randn(n_out, n_in) * sqrt(2 / n_in)

    where this sqrt(2/n_in) comes from: for a preactivation
        z_k = sum_{j=1..n_in} w_kj * a_j + b_k
    if the a_j are independent with variance v_a and the w_kj are drawn
    i.i.d. with mean 0 and variance v_w, then
        var(z_k) = n_in * v_w * v_a
    to keep var(z) roughly equal to v_a as it flows through the layer we
    want n_in * v_w ~ 1, ie v_w = 1 / n_in; that's the classic
    "xavier" result for a linear/tanh network. relu additionally zeroes
    out about half its inputs (everything z <= 0), which roughly halves
    the variance downstream, so he et al. compensate by doubling the
    target variance: v_w = 2 / n_in, giving std = sqrt(2 / n_in).

    biases are initialized to 0; theres no symmetry problem with
    biases (the weight randomness already breaks symmetry between
    neurons), and starting them at 0 is a safe, common default

    layer_dims: list like [784, 128, 64, 10] meaning
        input layer: 784 units (28x28 pixels)
        hidden layer 1: 128 units
        hidden layer 2: 64 units
        output layer: 10 units (digits 0-9)

    returns a dict of parameters: w1, b1, w2, b2, ..., wL, bL
    where wl has shape (layer_dims[l], layer_dims[l-1])
    and   bl has shape (layer_dims[l], 1)
    """
    if seed is not None:
        np.random.seed(seed)

    params = {}
    num_layers = len(layer_dims) - 1  #number of weight matrices

    for l in range(1, num_layers + 1):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]
        params[f"w{l}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        params[f"b{l}"] = np.zeros((n_out, 1))

    return params