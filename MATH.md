# Full Derivation of MLP Math

This document derives, rigorously and from first principles, every
equation implemented in `nn_numpy/`. Nothing is asserted without
justification: activation derivatives, the softmax Jacobian, the
cross-entropy gradient, the general backpropagation recursion, the
He-initialization variance calculation, and the momentum update are
all derived in full.

Notation is fixed once in §1 and used consistently throughout. All
"formulas" sections in the code comments are special cases of what's
derived here.

---

## 1. Notation and Problem Setup

We are learning a function $f_\theta : \mathbb{R}^{784} \to \Delta^{9}$
that maps a flattened $28\times 28$ grayscale image to a probability
distribution over 10 digit classes ($\Delta^9$ denotes the 9-simplex:
vectors of 10 nonnegative numbers summing to 1).

**Architecture.** The network has $L$ layers (not counting the input),
with layer sizes $n_0, n_1, \dots, n_L$, where $n_0 = 784$ and
$n_L = 10$. Layers $1, \dots, L-1$ are hidden layers using ReLU;
layer $L$ is the output layer using softmax.

**A single training example.** An input $x \in \mathbb{R}^{n_0}$ and
a label $y \in \{0, \dots, 9\}$, represented as a one-hot vector
$y \in \{0,1\}^{10}$ with $y_c = 1$ for the true class $c$ and
$y_k = 0$ otherwise:

$$
y_k = \mathbb{1}[k = c], \qquad k = 0, \dots, 9.
$$

**A batch of $m$ examples.** Stacked column-wise into a matrix
$X \in \mathbb{R}^{784 \times m}$, with labels
$Y \in \{0,1\}^{10 \times m}$ (each column one-hot). This
column-major convention (*features along rows, examples along
columns*) is used for every quantity below ($Z^{[l]}, A^{[l]}$,
etc. are all $(\text{features}) \times (\text{examples})$ matrices).

**Parameters.** Layer $l$ has a weight matrix
$W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ and a bias vector
$b^{[l]} \in \mathbb{R}^{n_l \times 1}$, for $l = 1, \dots, L$.
$b^{[l]}$ broadcasts (repeats) across all $m$ columns when added to a
$n_l \times m$ matrix. NumPy does this automatically; mathematically,
think of it as $b^{[l]} \mathbf{1}_m^\top$ where $\mathbf{1}_m$ is the
all-ones $m$-vector.

---

## 2. Forward Propagation

### 2.1 The affine map

For layer $l = 1, \dots, L$, define the pre-activation matrix

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[0]} := X.
$$

**Dimension check.** $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$
times $A^{[l-1]} \in \mathbb{R}^{n_{l-1}\times m}$ gives an
$n_l \times m$ matrix, matching $b^{[l]} \in \mathbb{R}^{n_l \times 1}$
broadcast across $m$ columns. So $Z^{[l]} \in \mathbb{R}^{n_l \times m}$,
as required.

In index notation, for the $k$-th feature of the $i$-th example:

$$
z^{[l]}_{k,i} = \sum_{j=1}^{n_{l-1}} W^{[l]}_{kj}\, a^{[l-1]}_{j,i} \;+\; b^{[l]}_k.
$$

### 2.2 The nonlinearity

$$
A^{[l]} = g^{[l]}\!\left(Z^{[l]}\right), \qquad
g^{[l]} =
\begin{cases}
\text{ReLU}, & l = 1, \dots, L-1 \\[2pt]
\text{softmax}, & l = L
\end{cases}
$$

Applied elementwise for ReLU, and per-*column* for softmax (each
column (one example's 10 logits) is normalized independently into a
probability vector; see §3.2).

### 2.3 Why nonlinearity is necessary

If every $g^{[l]}$ were the identity, then

$$
A^{[L]} = W^{[L]}\big(W^{[L-1]}\big(\cdots (W^{[1]}X + b^{[1]})\cdots\big) + b^{[L-1]}\big) + b^{[L]}
$$

is an affine function of $X$: expanding the products, there exist
$\widetilde W = W^{[L]}W^{[L-1]}\cdots W^{[1]}$ and some vector
$\widetilde b$ such that $A^{[L]} = \widetilde W X + \widetilde b$
identically, regardless of $L$. An $L$-layer linear network is
therefore no more expressive than a single affine layer, hence the
necessity of a nonlinear $g$ between layers.

---

## 3. Activation Functions

### 3.1 ReLU

$$
\text{ReLU}(z) = \max(0, z), \qquad
\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0\end{cases}
$$

(Undefined at $z=0$; by convention we set $\text{ReLU}'(0) := 0$,
a measure-zero choice that never affects a floating-point computation
in practice.) Applied elementwise, so for a matrix $Z$,
$[\text{ReLU}(Z)]_{k,i} = \max(0, Z_{k,i})$.

### 3.2 Softmax

For a logit vector $z \in \mathbb{R}^{10}$ (one column of $Z^{[L]}$),

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}, \qquad k = 0,\dots,9.
$$

**Claim: this is well-defined as a probability distribution.**
Each $\text{softmax}(z)_k > 0$ since $e^{(\cdot)} > 0$ everywhere, and

$$
\sum_{k=0}^9 \text{softmax}(z)_k
= \sum_k \frac{e^{z_k}}{\sum_j e^{z_j}}
= \frac{\sum_k e^{z_k}}{\sum_j e^{z_j}} = 1.
$$

So $\text{softmax}(z) \in \Delta^9$ for any $z \in \mathbb{R}^{10}$. $\blacksquare$

**Numerical stability (shift invariance).** For any constant
$m \in \mathbb{R}$,

$$
\frac{e^{z_k - m}}{\sum_j e^{z_j - m}}
= \frac{e^{z_k}e^{-m}}{e^{-m}\sum_j e^{z_j}}
= \frac{e^{z_k}}{\sum_j e^{z_j}} = \text{softmax}(z)_k.
$$

So softmax is invariant to subtracting any constant from every logit.
Choosing $m = \max_k z_k$ makes the largest exponent $e^0 = 1$,
preventing overflow from exponentiating large positive logits. This
is exactly the `z - np.max(z, axis=0)` step in `activations.py`.

### 3.3 The softmax Jacobian

Softmax is a $\mathbb{R}^{10}\to\mathbb{R}^{10}$ map, so its derivative
is a $10\times 10$ Jacobian matrix, $J_{ik} = \partial a_i/\partial z_k$
where $a = \text{softmax}(z)$.

**Case $i = k$:** Using the quotient rule on
$a_i = e^{z_i} / S$ where $S = \sum_j e^{z_j}$,

$$
\frac{\partial a_i}{\partial z_i}
= \frac{e^{z_i}\cdot S - e^{z_i}\cdot e^{z_i}}{S^2}
= \frac{e^{z_i}}{S}\left(1 - \frac{e^{z_i}}{S}\right)
= a_i(1 - a_i).
$$

**Case $i \ne k$:** Here $e^{z_i}$ does not depend on $z_k$, but
$S$ does ($\partial S/\partial z_k = e^{z_k}$):

$$
\frac{\partial a_i}{\partial z_k}
= e^{z_i}\cdot\frac{\partial}{\partial z_k}\left[S^{-1}\right]
= e^{z_i}\cdot\left(-S^{-2}e^{z_k}\right)
= -\,\frac{e^{z_i}}{S}\cdot\frac{e^{z_k}}{S}
= -a_i a_k.
$$

Combining both cases with the Kronecker delta $\delta_{ik}$
($=1$ if $i=k$, else $0$):

$$
\frac{\partial a_i}{\partial z_k} = a_i(\delta_{ik} - a_k)
$$

This single formula reproduces both cases above and is used directly
in §4.2.

---

## 4. Loss Function and Its Gradient

### 4.1 Categorical cross-entropy

For one example with prediction $a = \text{softmax}(z) \in \Delta^9$
and one-hot label $y$,

$$
\ell(a, y) = -\sum_{k=0}^{9} y_k \log a_k.
$$

Since $y$ is one-hot with the $1$ at index $c$, every term vanishes
except $k = c$:

$$
\ell(a,y) = -\log a_c,
$$

i.e. the negative log-probability the model assigned to the true
class. As $a_c \to 1$ (confident, correct), $\ell \to 0$; as
$a_c \to 0$ (confident, wrong), $\ell \to \infty$.

**Batch loss** is the mean over $m$ examples:

$$
J = \frac{1}{m}\sum_{i=1}^m \ell\big(a^{(i)}, y^{(i)}\big)
= -\frac{1}{m}\sum_{i=1}^m\sum_{k=0}^9 y^{(i)}_k \log a^{(i)}_k.
$$

**Information-theoretic interpretation.** For a single example,
$\ell(a,y) = H(y, a)$, the cross-entropy between the true (one-hot)
distribution and the predicted distribution. Since
$H(y,a) = H(y) + D_{KL}(y \,\|\, a)$ and $H(y) = 0$ for a one-hot
(zero-entropy) $y$, minimizing $\ell$ is exactly minimizing the
KL-divergence from the predicted distribution to the (degenerate)
true one, i.e. pushing all predicted mass onto the correct class.

### 4.2 Gradient of the loss w.r.t. the logits (the key shortcut)

We want $\partial \ell / \partial z_k$ for the *pre-softmax* logits,
since that is what feeds directly into the chain rule for earlier
layers. Write it via the chain rule through $a$:

$$
\frac{\partial \ell}{\partial z_k}
= \sum_{i=0}^9 \frac{\partial \ell}{\partial a_i}\cdot\frac{\partial a_i}{\partial z_k}.
$$

**First factor.** From $\ell = -\sum_i y_i \log a_i$:

$$
\frac{\partial \ell}{\partial a_i} = -\frac{y_i}{a_i}.
$$

**Second factor.** The Jacobian entry derived in §3.3:
$\partial a_i/\partial z_k = a_i(\delta_{ik}-a_k)$.

**Substitute and expand:**

$$
\frac{\partial \ell}{\partial z_k}
= \sum_{i} \left(-\frac{y_i}{a_i}\right) a_i(\delta_{ik}-a_k)
= -\sum_i y_i(\delta_{ik} - a_k)
= -\sum_i y_i\delta_{ik} \;+\; a_k\sum_i y_i.
$$

The first sum picks out $i=k$: $\sum_i y_i \delta_{ik} = y_k$. The
second sum is $\sum_i y_i = 1$ because $y$ is a one-hot (probability)
vector. So:

$$
\frac{\partial \ell}{\partial z_k} = -y_k + a_k = a_k - y_k.
$$

**Result**, in vector/matrix form for the whole batch:

$$
dZ^{[L]} := \frac{\partial J}{\partial Z^{[L]}} = A^{[L]} - Y \tag{4.1}
$$

This is a remarkably simple result given how involved the softmax
Jacobian (§3.3) and the $1/a_i$ derivative are individually: the two
non-linearities of softmax and log-loss are, in a precise sense,
inverse to each other, and the composition's derivative collapses to
a plain residual. This is why softmax and cross-entropy are almost
always implemented as a fused pair rather than composed generically.

---

## 5. Backpropagation: The General Recursion

### 5.1 Setup

Define, for every layer $l$, the *error signal*

$$
dZ^{[l]} := \frac{\partial J}{\partial Z^{[l]}} \in \mathbb{R}^{n_l\times m}.
$$

We already have $dZ^{[L]}$ from (4.1). The goal is a recursion
expressing $dZ^{[l]}$ in terms of $dZ^{[l+1]}$, so we can walk
backward from $l=L$ to $l=1$ computing every layer's error signal (and
from it, every parameter gradient) in one backward pass. This is
precisely what makes backprop $O(\text{network size})$ per example
rather than requiring a separate forward pass per parameter.

### 5.2 Deriving $dA^{[l]}$

$Z^{[l+1]} = W^{[l+1]} A^{[l]} + b^{[l+1]}$, so by the chain rule,
for a single example (column) with $a^{[l]}\in\mathbb{R}^{n_l}$ and
$z^{[l+1]}\in\mathbb{R}^{n_{l+1}}$:

$$
\frac{\partial \ell}{\partial a^{[l]}_j}
= \sum_{k=1}^{n_{l+1}} \frac{\partial \ell}{\partial z^{[l+1]}_k}\cdot\frac{\partial z^{[l+1]}_k}{\partial a^{[l]}_j}.
$$

Since $z^{[l+1]}_k = \sum_j W^{[l+1]}_{kj} a^{[l]}_j + b^{[l+1]}_k$, we
have $\partial z^{[l+1]}_k/\partial a^{[l]}_j = W^{[l+1]}_{kj}$, so

$$
\frac{\partial \ell}{\partial a^{[l]}_j}
= \sum_k dz^{[l+1]}_k\, W^{[l+1]}_{kj}
= \sum_k \big(W^{[l+1]}\big)^\top_{jk}\, dz^{[l+1]}_k.
$$

In matrix form, for the whole batch:

$$
dA^{[l]} = \big(W^{[l+1]}\big)^\top dZ^{[l+1]}. \tag{5.1}
$$

**Dimension check.** $(W^{[l+1]})^\top \in \mathbb{R}^{n_l \times n_{l+1}}$
times $dZ^{[l+1]} \in \mathbb{R}^{n_{l+1}\times m}$ gives
$dA^{[l]} \in \mathbb{R}^{n_l \times m}$, matching $A^{[l]}$'s shape,
as it must (a gradient has the same shape as the variable it's taken
with respect to).

### 5.3 Deriving $dZ^{[l]}$ from $dA^{[l]}$

For $l < L$, $A^{[l]} = \text{ReLU}(Z^{[l]})$ elementwise, so by the
(elementwise, hence diagonal-Jacobian) chain rule,

$$
dZ^{[l]} = dA^{[l]} \odot \text{ReLU}'(Z^{[l]}), \tag{5.2}
$$

where $\odot$ is the elementwise (Hadamard) product. This holds
elementwise because for an elementwise function $a_j = g(z_j)$ (no
cross terms between different $j$, unlike softmax), the Jacobian
$\partial a_j/\partial z_k$ is diagonal: $g'(z_j)$ if $j=k$, else $0$.
So the chain-rule sum in (5.1)'s derivation collapses to a single
term, and $\partial \ell/\partial z_j = (\partial\ell/\partial a_j)\cdot g'(z_j)$
which is exactly the elementwise product in (5.2).

### 5.4 The full recursion

Combining (5.1) and (5.2), for $l = L-1, \dots, 1$:

$$
dZ^{[l]} = \Big(\big(W^{[l+1]}\big)^\top dZ^{[l+1]}\Big) \odot \text{ReLU}'\!\big(Z^{[l]}\big) \tag{5.3}
$$

Seeded at the top by (4.1): $dZ^{[L]} = A^{[L]} - Y$. This is a
backward recursion: computing $dZ^{[l]}$ requires $dZ^{[l+1]}$, which
is why the layers must be visited in order $L, L-1, \dots, 1$
("back"-propagation).

### 5.5 Parameter gradients

Given $dZ^{[l]}$, the weight and bias gradients follow from
$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$. For a single example,
$z^{[l]}_k = \sum_j W^{[l]}_{kj}a^{[l-1]}_j + b^{[l]}_k$, so

$$
\frac{\partial z^{[l]}_k}{\partial W^{[l]}_{kj}} = a^{[l-1]}_j,
\qquad
\frac{\partial \ell}{\partial W^{[l]}_{kj}}
= \frac{\partial \ell}{\partial z^{[l]}_k}\cdot a^{[l-1]}_j
= dz^{[l]}_k\, a^{[l-1]}_j.
$$

Summing over the $m$ examples in a batch and averaging (since
$J = \frac1m\sum_i \ell^{(i)}$):

$$
\frac{\partial J}{\partial W^{[l]}_{kj}}
= \frac{1}{m}\sum_{i=1}^m dz^{[l]}_{k,i}\, a^{[l-1]}_{j,i}
= \frac{1}{m}\Big[dZ^{[l]} \big(A^{[l-1]}\big)^\top\Big]_{kj}.
$$

So, in matrix form:

$$
dW^{[l]} = \frac{1}{m}\, dZ^{[l]} \big(A^{[l-1]}\big)^\top \tag{5.4}
$$

And, since $\partial z^{[l]}_k/\partial b^{[l]}_k = 1$ (and $b^{[l]}$ is
shared/broadcast across all $m$ columns, so its total gradient sums
the contribution from every example):

$$
db^{[l]} = \frac{1}{m}\sum_{i=1}^m dZ^{[l]}_{:,i}
= \frac{1}{m}\, dZ^{[l]}\mathbf{1}_m \tag{5.5}
$$

**Dimension check.** $dZ^{[l]}\in\mathbb{R}^{n_l\times m}$ times
$(A^{[l-1]})^\top \in \mathbb{R}^{m\times n_{l-1}}$ gives
$dW^{[l]} \in \mathbb{R}^{n_l\times n_{l-1}}$, matching $W^{[l]}$
exactly. Summing $dZ^{[l]}$ over its $m$ columns gives
$db^{[l]}\in\mathbb{R}^{n_l\times 1}$, matching $b^{[l]}$.

### 5.6 The backward pass, assembled

Putting §5.4-§5.5 together with the recursion (5.3), one backward
pass computes, for $l = L, L-1, \dots, 1$ in order:

$$
dZ^{[l]} =
\begin{cases}
A^{[L]} - Y, & l = L \\[4pt]
\big(W^{[l+1]\top} dZ^{[l+1]}\big) \odot \text{ReLU}'(Z^{[l]}), & l < L
\end{cases}
$$
$$
dW^{[l]} = \tfrac{1}{m}\, dZ^{[l]} A^{[l-1]\top}, \qquad
db^{[l]} = \tfrac{1}{m}\, dZ^{[l]}\mathbf{1}_m.
$$

Each layer's $(dZ, dW, db)$ requires only $dZ^{[l+1]}$ (already
computed, one step earlier in the loop) and cached forward-pass values
$A^{[l-1]}, Z^{[l]}$, hence the need to store every $Z^{[l]}, A^{[l]}$
during the forward pass (`cache` in `model.py`).

---

## 6. $\ell_2$ Regularization

To discourage individual weights from growing arbitrarily large (which
tends to correlate with overfitting the training set), augment the
loss with a penalty on the squared weights:

$$
J_{\text{reg}} = J + \frac{\lambda}{2m}\sum_{l=1}^L \sum_{k,j} \big(W^{[l]}_{kj}\big)^2,
$$

where $\lambda \ge 0$ controls the penalty's strength (biases are
conventionally left unregularized, as here). Differentiating the
added term w.r.t. a single weight,

$$
\frac{\partial}{\partial W^{[l]}_{kj}}\left[\frac{\lambda}{2m}\sum (W^{[l]}_{kj})^2\right]
= \frac{\lambda}{m}W^{[l]}_{kj},
$$

so the full regularized gradient is (5.4) plus this term:

$$
dW^{[l]}_{\text{reg}} = \frac{1}{m}\,dZ^{[l]}A^{[l-1]\top} + \frac{\lambda}{m}W^{[l]} \tag{6.1}
$$

Each update step therefore also shrinks every weight geometrically
toward $0$ by a factor depending on $\lambda$, sometimes called
"weight decay" for this reason (see §8.1, where the two are shown to
coincide exactly for plain gradient descent).

---

## 7. Weight Initialization: The He/Kaiming Variance Calculation

### 7.1 The goal

We want the variance of pre-activations $Z^{[l]}$ to stay roughly
constant across layers at the start of training. If it grows
layer-over-layer, deep networks saturate/explode; if it shrinks,
signal (and later, gradient) vanishes.

### 7.2 Assumptions

- Weights $W^{[l]}_{kj}$ are drawn i.i.d. with mean $0$ and variance
  $\sigma_W^2$, independent of the inputs $a^{[l-1]}_j$.
- Inputs $a^{[l-1]}_j$ (across the $n_{l-1}$ units feeding into this
  layer) are independent with common variance $v$, mean $0$ (a
  reasonable approximation early in training, before any bias develops).
- Biases are initialized to $0$, so they don't contribute variance.

### 7.3 Variance of a pre-activation

For a fixed output unit $k$,

$$
z_k = \sum_{j=1}^{n_{l-1}} W_{kj}\, a_j.
$$

Since the $W_{kj}$ are independent of the $a_j$, and both have mean
$0$: for a product of two independent, zero-mean random variables $U,V$,
$\mathrm{Var}(UV) = E[U^2V^2] - (E[UV])^2 = E[U^2]E[V^2] - 0 = \mathrm{Var}(U)\mathrm{Var}(V)$.
So each term $W_{kj}a_j$ has variance $\sigma_W^2 \cdot v$. Summing
$n_{l-1}$ *independent* such terms (variances add for independent
sums):

$$
\mathrm{Var}(z_k) = n_{l-1}\,\sigma_W^2\, v. \tag{7.1}
$$

### 7.4 The ReLU correction

We additionally want to relate $v = \mathrm{Var}(a^{[l-1]})$, the
variance *entering* this layer, to $\mathrm{Var}(z^{[l-1]})$, the
variance of the *previous* layer's pre-activation, since
$a^{[l-1]} = \text{ReLU}(z^{[l-1]})$.

For $z$ symmetric about $0$ (a reasonable approximation for
pre-activations with zero-mean weights), ReLU passes through the
positive half exactly and zeroes the negative half. So, informally:

$$
E[a^2] = E[\text{ReLU}(z)^2] = E[z^2 \mid z>0]\cdot P(z>0)
\approx \tfrac{1}{2}E[z^2] = \tfrac{1}{2}\mathrm{Var}(z)
$$

(using $P(z>0)=\tfrac12$ by symmetry, and $E[z^2\mid z>0] \approx 2\cdot\tfrac12 E[z^2]$
for a symmetric distribution split at 0; more carefully, for
$z\sim\mathcal N(0,\sigma^2)$, $E[\text{ReLU}(z)^2] = \tfrac12\sigma^2$
exactly, since half the Gaussian's second moment lies in each half by
symmetry of $z^2$ about $0$). So each ReLU layer roughly *halves* the
variance passed forward compared to a linear layer.

### 7.5 The initialization rule

We want the variance to be preserved end-to-end:
$\mathrm{Var}(z^{[l]}) \approx \mathrm{Var}(z^{[l-1]})$. Combining
(7.1) with the ReLU factor of $\tfrac12$ from §7.4:

$$
\mathrm{Var}(z^{[l]}) = n_{l-1}\,\sigma_W^2\cdot\underbrace{\tfrac12\mathrm{Var}(z^{[l-1]})}_{v}.
$$

Setting this equal to $\mathrm{Var}(z^{[l-1]})$ and solving for
$\sigma_W^2$:

$$
n_{l-1}\,\sigma_W^2\cdot\tfrac12 = 1 \quad\Longrightarrow\quad
\sigma_W^2 = \frac{2}{n_{l-1}}.
$$

$$
W^{[l]}_{kj} \sim \mathcal{N}\!\left(0,\ \frac{2}{n_{l-1}}\right),
\quad\text{i.e.}\quad
W^{[l]} = \texttt{randn}(n_l, n_{l-1}) \cdot \sqrt{2/n_{l-1}} \tag{7.2}
$$

This is He/Kaiming initialization (He et al., 2015), matched to ReLU's
variance-halving effect (the analogous derivation *without* the
factor-of-2 ReLU correction, i.e. assuming a linear/tanh-like
activation, gives $\sigma_W^2 = 1/n_{l-1}$, the earlier
"Xavier/Glorot" result).

---

## 8. Parameter Updates: Gradient Descent with Momentum

### 8.1 Plain gradient descent

The loss $J(\theta)$, viewed as a function of all parameters $\theta$
jointly, decreases fastest locally in the direction of $-\nabla_\theta J$
(steepest descent direction, by the Cauchy-Schwarz inequality applied
to the first-order Taylor expansion
$J(\theta + \epsilon d) \approx J(\theta) + \epsilon\, \nabla J^\top d$,
which is minimized over unit vectors $d$ by $d = -\nabla J/\|\nabla J\|$).
The update rule takes a small step in that direction:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J, \tag{8.1}
$$

with $\alpha > 0$ the learning rate: small enough that the linear
approximation above remains locally valid, large enough to make
tractable progress.

### 8.2 Momentum

Plain gradient descent can oscillate badly in narrow "ravines" of the
loss surface (large curvature in one direction, small in another) and
converges slowly along shallow, consistent-gradient directions.
Momentum maintains an exponentially-decayed running average $v$ of
past gradients ("velocity") and steps along $v$ instead of the raw
gradient:

$$
v_t = \beta\, v_{t-1} + (1-\beta)\, \nabla_\theta J(\theta_t), \qquad
\theta_{t+1} = \theta_t - \alpha\, v_t, \tag{8.2}
$$

with $\beta \in [0,1)$ (typically $0.9$). Unrolling the recursion,

$$
v_t = (1-\beta)\sum_{s=0}^{t} \beta^{\,t-s}\, \nabla_\theta J(\theta_s),
$$

an exponentially-weighted moving average of *all* past gradients, with
weight decaying geometrically into the past. Two consequences:

- **Oscillation cancellation.** If the gradient alternates in sign
  along some direction from step to step (a symptom of a narrow
  ravine), those contributions partially cancel in the sum, damping
  the oscillation.
- **Acceleration.** If the gradient has a roughly consistent sign
  along some direction across steps, the weighted sum reinforces it,
  and the effective step size in that direction grows relative to
  plain gradient descent (up to a factor of $\approx 1/(1-\beta)$ at
  steady state, e.g. $10\times$ for $\beta=0.9$).

Setting $\beta = 0$ in (8.2) recovers $v_t = \nabla_\theta J(\theta_t)$
exactly, i.e. plain gradient descent (8.1) as a special case.

### 8.3 Mini-batch stochastic gradient descent

Computing $\nabla_\theta J$ exactly requires a forward+backward pass
over *all* $N$ training examples. Instead, at each step we draw a
random mini-batch $B \subset \{1,\dots,N\}$ of size $m \ll N$ and
compute the gradient of the batch-average loss (5.4)-(5.5), using
$B$ in place of the full dataset.

**Why this is a valid (unbiased) estimator.** The full-dataset
gradient is $\nabla_\theta J = \frac1N\sum_{i=1}^N \nabla_\theta \ell^{(i)}$.
If $B$ is drawn uniformly at random without replacement (or, for this
argument, with replacement, the expectation is the same to leading
order), then for each $i \in B$, $E[\nabla_\theta \ell^{(i)}]$ over the
randomness of the draw equals the same fixed quantity as if it were
drawn from the full set, so

$$
E_B\!\left[\frac{1}{m}\sum_{i\in B}\nabla_\theta \ell^{(i)}\right]
= \frac{1}{N}\sum_{i=1}^N \nabla_\theta \ell^{(i)} = \nabla_\theta J,
$$

i.e. the mini-batch gradient is an unbiased estimator of the true
gradient, just a noisier one (variance $O(1/m)$: smaller batches
mean a noisier estimate, but cheaper and more frequent updates). One
epoch partitions the (shuffled) training set into such batches so
that, across an epoch, every example contributes exactly once.

---

## 9. Putting It All Together: One Full Training Step

For one mini-batch $(X,Y)$ of size $m$:

**Forward** ($l = 1,\dots,L$):
$$Z^{[l]} = W^{[l]}A^{[l-1]}+b^{[l]}, \qquad A^{[l]} = g^{[l]}(Z^{[l]})$$

**Loss:**
$$J = -\frac1m\sum_{i,k} Y_{k,i}\log A^{[L]}_{k,i} + \frac{\lambda}{2m}\sum_l \|W^{[l]}\|_F^2$$

**Backward** ($l = L,\dots,1$):
$$
dZ^{[l]} =
\begin{cases}
A^{[L]}-Y & l=L\\
(W^{[l+1]\top}dZ^{[l+1]})\odot\text{ReLU}'(Z^{[l]}) & l<L
\end{cases}
$$
$$
dW^{[l]} = \tfrac1m dZ^{[l]}A^{[l-1]\top} + \tfrac{\lambda}{m}W^{[l]}, \qquad
db^{[l]} = \tfrac1m dZ^{[l]}\mathbf 1_m
$$

**Update** ($l=1,\dots,L$, for each parameter $\theta\in\{W^{[l]},b^{[l]}\}$):
$$
v_\theta \leftarrow \beta v_\theta + (1-\beta)\, d\theta, \qquad
\theta \leftarrow \theta - \alpha\, v_\theta
$$

This exact sequence is what `NeuralNetwork.train_step()` executes: one
call to `forward`, one to `backward`, one to `update_params`, per
mini-batch, repeated for `epochs` full passes over the shuffled
training set.

---

## 10. Computational Complexity (Why This Is Fast)

For a batch of $m$ examples through a layer of size $n_{l-1}\to n_l$:

- Forward: $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ costs
  $O(n_l\, n_{l-1}\, m)$ multiply-adds (a single dense matmul).
- Backward: computing $dA^{[l-1]}$, $dW^{[l]}$ both involve a matmul
  of the same two matrix dimensions (transposed), so backward is also
  $O(n_l\,n_{l-1}\,m)$ — **backprop costs the same asymptotic order as
  the forward pass**, roughly a constant factor (≈2-3×) more, not
  exponentially more, which is what makes training deep networks
  tractable at all.

Summed over all $L$ layers, one forward+backward pass over a batch is
$O\!\left(m\sum_{l=1}^L n_l n_{l-1}\right)$ — linear in batch size and
in the total number of weights in the network.

---

## Summary Table

| Quantity | Formula | Shape |
|---|---|---|
| Pre-activation | $Z^{[l]} = W^{[l]}A^{[l-1]}+b^{[l]}$ | $n_l\times m$ |
| ReLU | $A^{[l]}=\max(0,Z^{[l]})$ | $n_l\times m$ |
| Softmax | $A^{[L]}_{k,i}=e^{Z^{[L]}_{k,i}}/\sum_j e^{Z^{[L]}_{j,i}}$ | $10\times m$ |
| Loss | $J=-\frac1m\sum Y\odot\log A^{[L]}$ | scalar |
| Output error | $dZ^{[L]}=A^{[L]}-Y$ | $10\times m$ |
| Hidden error | $dZ^{[l]}=(W^{[l+1]\top}dZ^{[l+1]})\odot\text{ReLU}'(Z^{[l]})$ | $n_l\times m$ |
| Weight grad | $dW^{[l]}=\tfrac1m dZ^{[l]}A^{[l-1]\top}+\tfrac{\lambda}{m}W^{[l]}$ | $n_l\times n_{l-1}$ |
| Bias grad | $db^{[l]}=\tfrac1m dZ^{[l]}\mathbf1_m$ | $n_l\times 1$ |
| He init | $W^{[l]}\sim\mathcal N(0,\,2/n_{l-1})$ | $n_l\times n_{l-1}$ |
| Momentum | $v\leftarrow\beta v+(1-\beta)d\theta,\ \ \theta\leftarrow\theta-\alpha v$ | — |

Every row above corresponds one-to-one with a line of code in
`model.py`, `init.py`, `activations.py`, and `losses.py`.