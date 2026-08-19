"""
the network, an arbitrary-depth multilayer perceptron, with forward pass,
backward pass (backpropagation), and parameter update rule (mini-batch 
radient descent with momentum and l2 regularization) all implemented in numpy.

layer convention: layers 1..L-1 use relu, layer L (the output layer)
uses softmax, everything is stored as matrices shaped (features, m)
where m is the number of examples in current batch, so a whole
batch is processed with a handful of matrix multiplies rather than a
python loop over examples.
"""

import numpy as np

from activations import relu, relu_prime, softmax
from init import he_init
from losses import cross_entropy_loss, one_hot


class NeuralNetwork:
    def __init__(self, layer_dims, lr=0.1, momentum=0.9, l2_lambda=0.0, seed=None):
        """
        layer_dims: such as [784, 128, 64, 10]
        lr: learning rate, alpha in the update rule below
        momentum: beta in the momentum update below (0 disables momentum)
        l2_lambda: l2 regularization strength (0 disables it)
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.lr = lr
        self.momentum = momentum
        self.l2_lambda = l2_lambda

        self.params = he_init(layer_dims, seed=seed)

        #velocity terms for momentum, one per parameter, initialized to 0
        self.velocity = {
            key: np.zeros_like(val) for key, val in self.params.items()
        }

    # forward pass
    def forward(self, x):
        """
        propagate a batch of inputs x (shape: 784 x m) through every
        layer. for layer l:
            z[l] = w[l] . a[l-1] + b[l]
            a[l] = g_l(z[l])

        where a[0] = x (the input itself), g_l = relu for hidden layers
        and g_l = softmax for the final layer. "." is matrix multiply.
        we cache every z[l] and a[l] because backprop needs them: the
        derivative of relu needs z[l], and the weight-gradient formula
        needs a[l-1].
        returns a_last (the network's predicted probabilities, shape
        10 x m) and a cache dict used by backward().
        """
        cache = {"a0": x}
        a = x

        for l in range(1, self.num_layers + 1):
            w = self.params[f"w{l}"]
            b = self.params[f"b{l}"]
            z = w.dot(a) + b

            if l == self.num_layers:
                a = softmax(z)
            else:
                a = relu(z)

            cache[f"z{l}"] = z
            cache[f"a{l}"] = a

        return a, cache

    # backward pass
    def backward(self, y, cache):
        """
        backpropagation: apply the chain rule layer by layer, starting
        from the output and working back toward the input, to get
        dl/dw[l] and dl/db[l] for every layer.
        output layer (l = L), using the softmax + cross-entropy
        shortcut derived in losses.py:
            dz[L] = a[L] - y <- y is one-hot, shape 10 x m

        every other layer (l = L-1, ..., 1), given dz[l+1] and w[l+1]
        from the layer above:
            da[l] = w[l+1]^t . dz[l+1]  <- how much layer l contributed to loss
            dz[l] = da[l] * relu'(z[l]) <- elementwise; relu' is a 0/1 mask

        once we have dz[l] for a layer, the parameter gradients are:
            dw[l] = (1/m) * dz[l] . a[l-1]^t   + (lambda/m) * w[l]
            db[l] = (1/m) * sum(dz[l], axis=1, keepdims=true)

        the extra (lambda/m) * w[l] term is the gradient of the l2
        penalty (lambda/2m) * sum(w[l]^2) added to the loss -- it pulls
        weights gently toward 0 every step, which discourages any single
        weight from growing huge and helps reduce overfitting
        m is the batch size, used to average gradients over the batch
        rather than summing them, so the effective step size doesn't
        depend on how big a batch happens to be
        """
        m = y.shape[1]
        grads = {}

        #output layer: softmax + cross-entropy gradient shortcut
        dz = cache[f"a{self.num_layers}"] - y

        for l in range(self.num_layers, 0, -1):
            a_prev = cache[f"a{l - 1}"]
            w = self.params[f"w{l}"]

            grads[f"dw{l}"] = (dz.dot(a_prev.T) / m) + (self.l2_lambda / m) * w
            grads[f"db{l}"] = np.sum(dz, axis=1, keepdims=True) / m

            if l > 1:
                da_prev = w.T.dot(dz)
                dz = da_prev * relu_prime(cache[f"z{l - 1}"])

        return grads

    # parameter update: gradient descent with momentum
    def update_params(self, grads):
        """
        plain gradient descent just does:
            w := w - alpha * dw
        momentum instead keeps a running, exponentially-decayed average
        of past gradients (a "velocity" v), and steps in that direction:
            v := beta * v + (1 - beta) * dw
            w := w - alpha * v
        beta (self.momentum) is typically ~0.9, meaning v mostly
        remembers the recent gradient history and only partially
        reacts to the newest gradient. intuitively this is a ball
        rolling downhill picking up momentum: it smooths out
        oscillations in noisy/steep directions and accelerates movement
        in directions where the gradient consistently points the same
        way across steps
        with self.momentum = 0 this degenerates back to vanilla
        gradient descent, since v just becomes dw every step.
        """
        beta = self.momentum
        for l in range(1, self.num_layers + 1):
            for p in ("w", "b"):
                key = f"{p}{l}"
                dkey = f"d{p}{l}"

                self.velocity[key] = (
                    beta * self.velocity[key] + (1 - beta) * grads[dkey]
                )
                self.params[key] -= self.lr * self.velocity[key]

    #prediction/evaluation helpers
    def predict(self, x):
        """
        argmax over the 10 softmax outputs gives the predicted digit:
            prediction = argmax_k  a_last[k]
        (the class the network assigns the highest probability to)
        """
        a_last, _ = self.forward(x)
        return np.argmax(a_last, axis=0)

    def accuracy(self, x, y):
        """fraction of examples where prediction == true label."""
        preds = self.predict(x)
        return np.mean(preds == y)

    def loss(self, x, y_int):
        """cross-entropy loss on integer labels y_int (not one-hot)."""
        a_last, _ = self.forward(x)
        y_oh = one_hot(y_int, num_classes=self.layer_dims[-1])
        return cross_entropy_loss(a_last, y_oh)

    # one full training step on a single batch
    def train_step(self, x_batch, y_batch_int):
        """
        one iteration = forward -> backward -> update, this is called
        once per mini-batch inside the training loop in train.py
        """
        y_oh = one_hot(y_batch_int, num_classes=self.layer_dims[-1])
        a_last, cache = self.forward(x_batch)
        grads = self.backward(y_oh, cache)
        self.update_params(grads)
        return cross_entropy_loss(a_last, y_oh)

    #save/load
    def save(self, path):
        np.savez(path, layer_dims=np.array(self.layer_dims), **self.params)

    @classmethod
    def load(cls, path, lr=0.1, momentum=0.9, l2_lambda=0.0):
        data = np.load(path)
        layer_dims = list(data["layer_dims"])
        net = cls(layer_dims, lr=lr, momentum=momentum, l2_lambda=l2_lambda)
        for key in net.params:
            net.params[key] = data[key]
        return net