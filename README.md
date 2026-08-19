# MNIST neural network using numpy

A multilayer perceptron for classifying mnist handwritten digits, implemented with only numpy. Every piece of math (forward pass, backpropagation, gradient descent) is written out explicitly, generalized to an arbitrary number of layers, with mini-batches, momentum, l2 regularization, and a train/dev/test split.

## Project structure

```
nn_numpy/
├── activations.py #relu, softmax, and their derivatives
├── losses.py #cross-entropy loss, softmax cross-entropy, gradient derivation
├── init.py #he/kaiming weight initialization
├── model.py #NeuralNetwork class: forward, backward, momentum update, save/load
├── data.py #loading + preprocessing mnist, mini-batch iterator
├── train.py #training script to run
├── MATH.md #mathematical derivation
└── README.md
```

Every file's comments are written to double as a math reference. Readthem in this order (activations -> losses -> init -> model) for full derivation from "what is a neuron" up to "here is the update rule". For a full rigorous, first principles derivation of each equation, check `MATH.md`. For a simple overview, read below.

## The math, briefly

**Notation:** a batch of `m` examples is stored as a `(features, m)` matrix, so one column is one example. `x` is the input batch (784 × m, one row per pixel). `w[l]` and `b[l]` are the weights and bias of layer `l`. `z[l]` is the pre-activation, `a[l]` the post-activation output of layer `l`, with `a[0] = x`.

**Forward pass:** for each layer `l = 1 .. L`:

```
z[l] = w[l] . a[l-1] + b[l]
a[l] = g(z[l])
```

where `g` = relu for every hidden layer and `g` = softmax for the final layer `L`. relu is `max(0, z)`; softmax turns the last layer's raw scores into a probability distribution over the 10 digit classes.

**Loss:** categorical cross-entropy between the predicted probabilities `a[L]` and the true one-hot label `y`:

```
loss = -(1/m) * sum over examples and classes of  y * log(a[L])
```

**Backward pass (backpropagation):** the chain rule applied layer by layer, from output back to input. Output layer has a closed form, such that when you pair softmax with cross-entropy, the gradient wrt the logits collapses to just:

```
dz[L] = a[L] - y
```

(the full derivation of *why* that's true, via the softmax jacobian, is in `losses.py`, and the rigourus one in `MATH.md`) every earlier layer then follows:

```
da[l]   = w[l+1]^T . dz[l+1]
dz[l]   = da[l] * relu'(z[l]) (elementwise; relu' is a 0/1 mask)
dw[l]   = (1/m) * dz[l] . a[l-1]^T  +  (lambda/m) * w[l]
db[l]   = (1/m) * sum(dz[l], axis=1)
```

the extra `(lambda/m) * w[l]` term is l2 regularizationl; the gradient of an added `(lambda / 2m) * sum(w^2)` penalty on the loss, which gently shrinks weights toward zero each step and helps reduce
overfitting.

**Parameter update (gradient descent with momentum):** instead of stepping directly along the gradient, we keep a running exponentially-decayed average of it (a "velocity" `v`) and step along that instead:

```
v := beta * v + (1 - beta) * dw
w := w - alpha * v
```

`alpha` is the learning rate, `beta` (typically 0.9) controls how much history the velocity remembers. this smooths out noisy steps and tends to speed up convergence versus plain gradient descent (`beta = 0` recovers plain gradient descent exactly).

**Initialization:** weights are drawn as `w ~ normal(0, sqrt(2 / n_in))` ("he" initialization), which keeps the variance of activations roughly constant as they flow through many relu layers. Biases start at 0.

**Mini-batches:** rather than computing the gradient on the entire training set (slow) or one example at a time (noisy), each epoch is split into shuffled batches of `batch_size` examples, and one gradient step is taken per batch.

## Usage

```bash
pip install numpy scikit-learn matplotlib
python train.py
```

Running `train.py` will:
1. split the data into train/dev/test sets
2. train for `epochs` epochs of mini-batch gradient descent + momentum
3. print train/dev loss and accuracy every epoch
4. save the trained parameters to `mnist_params.npz`
5. save `training_curves.png` (loss/accuracy vs. epoch) and `sample_predictions.png` (a few example digits with predictions)

Expect roughly 97-98% test accuracy with the default `[784, 128, 64, 10]` architecture after ~20 epochs/

## Tweaking it

All the interesting knobs are keyword arguments to `train()` in `train.py`:

```python
train(
    layer_dims=(784, 128, 64, 10), #add/remove/resize layers freely
    lr=0.1, #learning rate (alpha)
    momentum=0.9, #beta; set to 0 for plain gradient descent
    l2_lambda=1e-4, #regularization strength; 0 disables it
    batch_size=128,
    epochs=20,
)
```

Because `model.py` loops over `layer_dims` generically, you can make the network deeper or wider just by editing that tuple.

## Loading a trained model later

```python
from model import NeuralNetwork

net = NeuralNetwork.load("mnist_params.npz")
predictions = net.predict(some_batch_of_images)  # shape (784, m) -> (m,)
```