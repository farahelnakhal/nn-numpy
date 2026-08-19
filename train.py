# main training script, trains the network with mini-batch gradient descent 
# + momentum, tracks loss/accuracy over epochs, plots the curves, evaluates 
# on a held-out test set, and saves the learned parameters to disk.

import numpy as np

from data import load_mnist, iterate_minibatches
from model import NeuralNetwork


def train(
    layer_dims=(784, 128, 64, 10),
    lr=0.1,
    momentum=0.9,
    l2_lambda=1e-4,
    batch_size=128,
    epochs=20,
    seed=42,
):
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_mnist(seed=seed)
    print(
        f"train: {x_train.shape[1]}  dev: {x_dev.shape[1]}  "
        f"test: {x_test.shape[1]}  examples"
    )

    net = NeuralNetwork(
        layer_dims=list(layer_dims),
        lr=lr,
        momentum=momentum,
        l2_lambda=l2_lambda,
        seed=seed,
    )

    history = {"train_loss": [], "dev_loss": [], "train_acc": [], "dev_acc": []}

    for epoch in range(1, epochs + 1):
        #one epoch = one full pass over the training set, in shuffled mini-batches
        epoch_losses = []
        for x_batch, y_batch in iterate_minibatches(
            x_train, y_train, batch_size, seed=seed + epoch
        ):
            batch_loss = net.train_step(x_batch, y_batch)
            epoch_losses.append(batch_loss)

        train_loss = float(np.mean(epoch_losses))
        dev_loss = net.loss(x_dev, y_dev)
        train_acc = net.accuracy(x_train, y_train)
        dev_acc = net.accuracy(x_dev, y_dev)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["train_acc"].append(train_acc)
        history["dev_acc"].append(dev_acc)

        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"dev loss {dev_loss:.4f} acc {dev_acc:.4f}"
        )

    test_acc = net.accuracy(x_test, y_test)
    print(f"\nfinal test accuracy: {test_acc:.4f}")

    net.save("mnist_params.npz")
    print("saved trained parameters to mnist_params.npz")

    _plot_history(history)
    _show_sample_predictions(net, x_test, y_test)

    return net, history


def _plot_history(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping training curves plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["dev_loss"], label="dev")
    axes[0].set_title("cross-entropy loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["dev_acc"], label="dev")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("saved training curves to training_curves.png")


def _show_sample_predictions(net, x, y, n=8):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    idx = np.random.default_rng(0).choice(x.shape[1], size=n, replace=False)
    preds = net.predict(x[:, idx])

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for i, ax in enumerate(axes):
        img = x[:, idx[i]].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"pred: {preds[i]}\ntrue: {y[idx[i]]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("saved sample predictions to sample_predictions.png")


if __name__ == "__main__":
    train()