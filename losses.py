# loss function: categorical cross-entropy on top of a softmax output.
# we also derive, in comments, the gradient that falls out when you pair 
# softmax with cross-entropy. it's the reason `dz_last` in model.py is just 
# `a_last - y` instead of some messy jacobian product.

import numpy as np


def one_hot(y, num_classes=10):
    """
    turn integer labels into one-hot column vectors so they can be
    compared directly against the network's softmax output.

    formula: for label y_i in {0, ..., 9}, the one-hot vector e(y_i) is
        e(y_i)_k = 1 if k == y_i
                   0 otherwise

    shape convention used throughout this project: columns are examples,
    rows are classes, so one_hot(y) has shape (num_classes, m) to line
    up with a network output a_last of shape (num_classes, m).
    """
    m = y.size
    one_hot_y = np.zeros((num_classes, m))
    one_hot_y[y, np.arange(m)] = 1
    return one_hot_y


def cross_entropy_loss(a_last, y_one_hot, epsilon=1e-12):
    """
    categorical cross-entropy, averaged over the batch.

    for a single example with true one-hot label vector y and predicted
    probability vector a (from softmax), the per-example loss is
        l(a, y) = - sum_k y_k * log(a_k)

    since y is one-hot (all zeros except a 1 at the true class c), every
    term in that sum vanishes except k = c, so it reduces to
        l(a, y) = - log(a_c)
    i.e. "negative log of the probability the model assigned to the
    correct class". if the model is confident and correct, a_c is close
    to 1 and the loss is close to 0. if the model is confident and
    wrong, a_c is close to 0 and -log(a_c) blows up, so confident, wrong
    predictions are punished hard

    averaged over a batch of m examples:
        j = (1/m) * sum_{i=1..m} l(a^(i), y^(i))
          = -(1/m) * sum_i sum_k y_k^(i) * log(a_k^(i))

    epsilon is added purely for numerical safety, in case a predicted
    probability underflows to exactly 0.0 (log(0) = -inf).
    """
    m = y_one_hot.shape[1]
    a_clipped = np.clip(a_last, epsilon, 1.0 - epsilon)
    return -np.sum(y_one_hot * np.log(a_clipped)) / m

# derivation: gradient of cross-entropy w.r.t. the pre-softmax logits z
#
# this is the key simplification that makes backprop through the output
# layer cheap. we want dl/dz_j for logit j, where a = softmax(z) and
# l = -sum_k y_k log(a_k).
#
# first, the softmax jacobian:
#     da_i/dz_j = a_i * (1 - a_i)   if i == j
#     da_i/dz_j = -a_i * a_j        if i != j
#   (both cases are captured by:  da_i/dz_j = a_i * (delta_ij - a_j),
#    where delta_ij is 1 if i == j else 0)
#
# by the chain rule:
#     dl/dz_j = sum_i (dl/da_i) * (da_i/dz_j)
# and dl/da_i = -y_i / a_i (from differentiating -sum_k y_k log(a_k)).
#
# substituting and expanding:
#     dl/dz_j = sum_i [ -y_i / a_i ] * [ a_i * (delta_ij - a_j) ]
#             = sum_i -y_i * (delta_ij - a_j)
#             = -y_j + a_j * sum_i y_i
# since y is one-hot, sum_i y_i = 1, so this collapses to:
#     dl/dz_j = a_j - y_j
#
# in vector form, for the whole output layer at once:
#     dz_last = a_last - y_one_hot
#
# model.py -> backward() where this is used directly.