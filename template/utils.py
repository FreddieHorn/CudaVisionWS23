import numpy as np
import matplotlib.pyplot as plt

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def plot_loss(train_loss_list = [], val_loss_list = []):
    """Plots train and validation loss.

    Args:
        train_loss_list (list, optional): Training loss. Defaults to [].
        val_loss_list (list, optional): Validation loss. Defaults to [].
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    smooth_train_loss = smooth(train_loss_list, 31)
    smooth_val_loss = smooth(val_loss_list, 31)

    ax[0].plot(train_loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_train_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[0].plot(val_loss_list, c="green", label="Test Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_val_loss, c="yellow", label="Smoothed Test Loss", linewidth=3)


    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress (linearscale)")

    ax[1].plot(train_loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[1].plot(val_loss_list, c="green", label="Test Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_val_loss, c="yellow", label="Smoothed Test Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (logscale)")
    plt.show()

def plot_accuracy(train_accuracy = [], val_accuracy = []):
    """Plots accuracy for training and validation data

    Args:
        train_accuracy (list, optional): Accuracy for training dataset. Defaults to [].
        val_accuracy (list, optional): Accuarcy for validation dataset. Defaults to [].
    """
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    smooth_val_accuracy = smooth(val_accuracy, 31)
    smooth_train_accuracy = smooth(train_accuracy, 31)

    ax[0].plot(val_accuracy, c="blue", label="Testing Accuracy", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_val_accuracy, c="red", label="Smoothed Testing Accuracy", linewidth=3, alpha=0.5)



    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Testing Accuracy")
    ax[0].set_title("Testing Accuracy Change")


    ax[1].plot(train_accuracy, c="blue", label="Training Accuracy", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train_accuracy, c="red", label="Smoothed Training Accuracy", linewidth=3, alpha=0.5)

    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Training Accuracy")
    ax[1].set_title("Training Accuracy Change")

    plt.show()

def plot_gradients(norms, max_values, min_values):
    """PLots related to gradient

    Args:
        norms (list): Gradient norms
        max_values (list): Max Gradient Values
        min_values (list): Min Gradient values
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(18, 5)

    ax[0].plot(norms, c="blue", label="Gradient Norms", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Gradient Norms")
    ax[0].set_title("Gradient Norms Change")

    ax[1].plot(max_values, c="red", label="Max Gradient Values", linewidth=3, alpha=0.5)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Max Gradient Values")
    ax[1].set_title("Max Gradient Values Change")

    ax[2].plot(min_values, c="green", label="Min Gradient Values", linewidth=3, alpha=0.5)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Min Gradient Values")
    ax[2].set_title("Min Gradient Values Change")

    plt.show()
