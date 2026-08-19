#load and preprocess mnist.

import os
import numpy as np

def _load_raw():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

    if os.path.exists(csv_path):
        print(f"loading data from {csv_path} ...")
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        labels = raw[:, 0].astype(int)
        pixels = raw[:, 1:]
    else:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        pixels = mnist.data.astype(float)
        labels = mnist.target.astype(int)

    return pixels, labels


def load_mnist(dev_size=1000, test_size=2000, seed=42):
    """
    returns (x_train, y_train, x_dev, y_dev, x_test, y_test).

    preprocessing:
      1. normalize pixel values from [0, 255] to [0, 1] by dividing by
         255. this keeps activations/gradients in a well-scaled range
         for the network to learn from -- unnormalized inputs in the
         hundreds would produce huge, unstable pre-activations
         z = w.x + b at the very first layer.
      2. shuffle, so train/dev/test splits aren't biased by whatever
         order the original file happened to be in.
      3. split into three sets:
           - train: what the network actually learns from (gradients
             computed on this set drive every parameter update)
           - dev (validation): held out, used only to monitor accuracy
             during training as a check for overfitting
           - test: held out until the very end, for a final, unbiased
             read on generalization performance

    arrays follow the (features, examples) convention used throughout
    this project, so x has shape (784, m) and y has shape (m,)
    """
    pixels, labels = _load_raw()

    x = pixels.T / 255.0
    y = labels

    m = x.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    x, y = x[:, perm], y[perm]

    x_dev, y_dev = x[:, :dev_size], y[:dev_size]
    x_test, y_test = x[:, dev_size:dev_size + test_size], y[dev_size:dev_size + test_size]
    x_train, y_train = x[:, dev_size + test_size:], y[dev_size + test_size:]

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def iterate_minibatches(x, y, batch_size, seed=None):
    """
    yields shuffled mini-batches (x_batch, y_batch) covering the full
    dataset once (one "epoch")

    why mini-batches instead of the whole dataset at once (batch
    gradient descent) or one example at a time (stochastic gradient
    descent)? mini-batches are the usual middle ground:
      - full-batch gradients are smooth/accurate but slow (one update
        per full pass over potentially tens of thousands of examples)
      - single-example gradients are noisy but let you update far more
        often
      - a batch of, say, 64-256 examples approximates the true gradient
        well enough while still allowing many updates per epoch, and it
        maps efficiently onto vectorized matrix operations.
    """
    m = x.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    x_shuffled, y_shuffled = x[:, perm], y[perm]

    for start in range(0, m, batch_size):
        end = start + batch_size
        yield x_shuffled[:, start:end], y_shuffled[start:end]